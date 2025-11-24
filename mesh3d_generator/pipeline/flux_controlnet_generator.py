"""
Geração de 3D bundle image final usando Flux + ControlNet
Implementação própria baseada na abordagem do Kiss3DGen
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from pathlib import Path
import torchvision.transforms as transforms


class FluxControlNetGenerator:
    """
    Gera 3D bundle image refinado usando Flux diffusion com ControlNet
    """
    
    def __init__(self,
                 flux_model_id: str = "black-forest-labs/FLUX.1-dev",
                 controlnet_model_id: str = "InstantX/FLUX.1-dev-Controlnet-Union",
                 redux_model_id: Optional[str] = "black-forest-labs/FLUX.1-Redux-dev",
                 device: str = "cuda:0",
                 dtype: torch.dtype = torch.bfloat16):
        """
        Inicializa o gerador Flux + ControlNet
        
        Args:
            flux_model_id: ID do modelo Flux no HuggingFace
            controlnet_model_id: ID do ControlNet
            redux_model_id: ID do modelo Redux (opcional)
            device: Dispositivo para processamento
            dtype: Tipo de dados
        """
        self.flux_model_id = flux_model_id
        self.controlnet_model_id = controlnet_model_id
        self.redux_model_id = redux_model_id
        self.device = device
        self.dtype = dtype
        
        self.flux_pipeline = None
        self.redux_pipeline = None
        
        print(f"[FLUX] Inicializando Flux + ControlNet...")
        # Modelos serão carregados sob demanda
        # self._load_models()
    
    def _load_models(self):
        """Carrega modelos Flux, ControlNet e Redux"""
        try:
            from diffusers import DiffusionPipeline
            from diffusers.schedulers import FlowMatchHeunDiscreteScheduler
            
            print(f"[FLUX] Carregando Flux model: {self.flux_model_id}")
            
            # Tentar carregar Flux
            try:
                from diffusers import FluxImg2ImgPipeline
                
                # Tentar carregar do HuggingFace (modelo já está baixado)
                # Flux.1-dev não tem variant bf16, usar fp16 ou sem variant
                try:
                    # Tentar sem variant primeiro (usa o padrão do modelo)
                    self.flux_pipeline = FluxImg2ImgPipeline.from_pretrained(
                        self.flux_model_id,
                        torch_dtype=self.dtype
                    )
                    print(f"[FLUX] [OK] Flux carregado do HuggingFace")
                except Exception as e1:
                    print(f"[FLUX] Erro ao carregar sem variant: {e1}")
                    # Tentar com fp16
                    try:
                        self.flux_pipeline = FluxImg2ImgPipeline.from_pretrained(
                            self.flux_model_id,
                            torch_dtype=torch.float16,
                            variant="fp16"
                        )
                        print(f"[FLUX] [OK] Flux carregado com fp16")
                    except Exception as e2:
                        print(f"[FLUX] Erro ao carregar com fp16: {e2}")
                        # Tentar com dtype diferente
                        try:
                            self.flux_pipeline = FluxImg2ImgPipeline.from_pretrained(
                                self.flux_model_id,
                                torch_dtype=torch.float32
                            )
                            print(f"[FLUX] [OK] Flux carregado com float32")
                        except Exception as e3:
                            print(f"[FLUX] Erro ao carregar Flux: {e3}")
                            raise
            except ImportError as e:
                print(f"[FLUX] FluxImg2ImgPipeline nao disponivel: {e}")
                print(f"[FLUX] Tentando carregar como DiffusionPipeline generico...")
                try:
                    self.flux_pipeline = DiffusionPipeline.from_pretrained(
                        self.flux_model_id,
                        torch_dtype=self.dtype
                    )
                    print(f"[FLUX] [OK] Flux carregado como DiffusionPipeline generico")
                except Exception as e3:
                    print(f"[FLUX] Erro ao carregar como generico: {e3}")
                    raise
            
            # Não configurar scheduler - usar o padrão do modelo
            # FlowMatchHeunDiscreteScheduler já vem configurado corretamente
            # Tentar configurar pode causar erros de sigmas
            print(f"[FLUX] Usando scheduler padrao do modelo")
            
            # Carregar ControlNet
            print(f"[FLUX] Carregando ControlNet: {self.controlnet_model_id}")
            try:
                from diffusers import FluxControlNetImg2ImgPipeline
                from diffusers.models.controlnets import FluxControlNetModel
                
                controlnet = FluxControlNetModel.from_pretrained(
                    self.controlnet_model_id,
                    torch_dtype=torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float16
                )
                
                # Converter pipeline para usar ControlNet
                if hasattr(self.flux_pipeline, 'transformer'):
                    self.flux_pipeline = FluxControlNetImg2ImgPipeline(
                        scheduler=self.flux_pipeline.scheduler,
                        vae=self.flux_pipeline.vae,
                        text_encoder=self.flux_pipeline.text_encoder if hasattr(self.flux_pipeline, 'text_encoder') else None,
                        tokenizer=self.flux_pipeline.tokenizer if hasattr(self.flux_pipeline, 'tokenizer') else None,
                        text_encoder_2=self.flux_pipeline.text_encoder_2 if hasattr(self.flux_pipeline, 'text_encoder_2') else None,
                        tokenizer_2=self.flux_pipeline.tokenizer_2 if hasattr(self.flux_pipeline, 'tokenizer_2') else None,
                        transformer=self.flux_pipeline.transformer,
                        controlnet=controlnet
                    )
                    print(f"[FLUX] [OK] ControlNet integrado")
                else:
                    print(f"[FLUX] [AVISO]  Pipeline nao tem transformer - continuando sem ControlNet")
                
            except ImportError as e:
                print(f"[FLUX] [AVISO]  ControlNet nao disponivel (ImportError): {e}")
                print("[FLUX] Continuando sem ControlNet")
            except Exception as e:
                print(f"[FLUX] [AVISO]  Erro ao carregar ControlNet: {e}")
                print("[FLUX] Continuando sem ControlNet")
            
            # Carregar Redux (opcional)
            if self.redux_model_id:
                try:
                    print(f"[FLUX] Carregando Redux: {self.redux_model_id}")
                    # Redux geralmente é um pipeline customizado
                    # Referência: Kiss3DGen usa FluxPriorReduxPipeline
                    # Por enquanto, pular se não disponível
                    print("[FLUX] Redux sera carregado sob demanda")
                except Exception as e:
                    print(f"[FLUX] Redux nao disponivel: {e}")
            
            # Mover para device e otimizar para VRAM (RTX 3060 12GB)
            try:
                # Habilitar offloading ANTES de mover para device (economiza VRAM)
                if hasattr(self.flux_pipeline, 'enable_model_cpu_offload'):
                    self.flux_pipeline.enable_model_cpu_offload()
                    print(f"[FLUX] [OK] CPU offload habilitado (economiza VRAM)")
                elif hasattr(self.flux_pipeline, 'enable_sequential_cpu_offload'):
                    self.flux_pipeline.enable_sequential_cpu_offload()
                    print(f"[FLUX] [OK] Sequential CPU offload habilitado")
                else:
                    # Se não tiver offload, mover apenas partes críticas para GPU
                    # Deixar text encoders em CPU se possível
                    if hasattr(self.flux_pipeline, 'text_encoder'):
                        self.flux_pipeline.text_encoder = self.flux_pipeline.text_encoder.to('cpu')
                    if hasattr(self.flux_pipeline, 'text_encoder_2'):
                        self.flux_pipeline.text_encoder_2 = self.flux_pipeline.text_encoder_2.to('cpu')
                    # Transformer e VAE na GPU
                    if hasattr(self.flux_pipeline, 'transformer'):
                        self.flux_pipeline.transformer = self.flux_pipeline.transformer.to(self.device)
                    if hasattr(self.flux_pipeline, 'vae'):
                        self.flux_pipeline.vae = self.flux_pipeline.vae.to(self.device)
                    print(f"[FLUX] [OK] Modelo parcialmente em GPU (text encoders em CPU)")
            except Exception as e:
                print(f"[FLUX] [AVISO]  Aviso ao configurar device: {e}")
                # Tentar método simples
                try:
                    self.flux_pipeline = self.flux_pipeline.to(self.device)
                except Exception as e2:
                    print(f"[FLUX] [AVISO]  Erro ao mover para device: {e2}")
            
            # Carregar LoRA se disponível
            try:
                lora_path = "./checkpoint/flux_lora/rgb_normal.safetensors"
                if Path(lora_path).exists():
                    print(f"[FLUX] Carregando LoRA: {lora_path}")
                    self.flux_pipeline.load_lora_weights(lora_path)
            except Exception as e:
                print(f"[FLUX] LoRA nao disponivel: {e}")
            
            print("[FLUX] Modelos carregados com sucesso")
            
        except ImportError as e:
            print(f"[FLUX] Diffusers nao disponivel: {e}")
            print("[FLUX] Instale: pip install diffusers")
            self.flux_pipeline = None
        except Exception as e:
            print(f"[FLUX] Erro ao carregar modelos: {e}")
            import traceback
            traceback.print_exc()
            self.flux_pipeline = None
    
    def generate_bundle_image(self,
                             prompt: str,
                             reference_bundle_image: torch.Tensor,
                             input_image: Image.Image,
                             strength: float = 0.95,
                             enable_redux: bool = True,
                             control_mode: str = 'tile',
                             num_inference_steps: int = 20,
                             seed: Optional[int] = None) -> Tuple[torch.Tensor, str]:
        """
        Gera bundle image refinado
        
        Args:
            prompt: Caption/descrição
            reference_bundle_image: Reference bundle image (3, 1024, 2048)
            input_image: Imagem de input original
            strength: Strength para denoising
            enable_redux: Se True, usa Redux
            control_mode: Modo do ControlNet ('tile', 'canny', etc)
            num_inference_steps: Número de passos
            seed: Seed para reprodutibilidade
        
        Returns:
            Tuple[bundle_image_tensor, save_path]
        """
        print("[FLUX] Gerando bundle image final...")
        
        if self.flux_pipeline is None:
            print("[FLUX] [AVISO]  Pipeline nao disponivel - carregando modelos...")
            self._load_models()
            if self.flux_pipeline is None:
                print("[FLUX] [ERRO] Nao foi possivel carregar pipeline - usando reference bundle image")
                return reference_bundle_image.clone(), ""
        
        try:
            # Preparar prompt
            base_prompt = "A grid of 2x4 multi-view image, elevation 5. White background."
            full_prompt = f"{base_prompt} {prompt}"
            
            # Preparar imagem de input (reference bundle)
            # Garantir que está no formato correto: (1, 3, H, W) em range [0, 1]
            if len(reference_bundle_image.shape) == 3:
                image_input = reference_bundle_image.unsqueeze(0).to(self.device)  # (1, 3, H, W)
            else:
                image_input = reference_bundle_image.to(self.device)
            
            # Garantir range [0, 1]
            if image_input.max() > 1.0:
                image_input = image_input / 255.0
            
            # Preparar Redux se habilitado
            redux_hparam = None
            if enable_redux and self.redux_pipeline is not None:
                # Converter input_image para tensor 512x512
                input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False
                ).clip(0, 1)
                
                redux_hparam = {
                    'image': input_tensor,
                    'prompt_embeds_scale': 1.0,
                    'pooled_prompt_embeds_scale': 1.0,
                    'strength': 0.5
                }
                
                # Aplicar Redux
                # TODO: Implementar chamada ao Redux pipeline
                print("[FLUX] Redux habilitado (placeholder)")
            
            # Preparar ControlNet se disponível
            control_image = None
            if hasattr(self.flux_pipeline, 'controlnet') and self.flux_pipeline.controlnet is not None:
                # Preprocessar para ControlNet-Tile
                control_image = self._preprocess_controlnet_image(
                    reference_bundle_image,
                    mode=control_mode
                )
            
            # Gerar
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            # Preparar parâmetros básicos
            # Flux ControlNet Image2Image não usa num_inference_steps da mesma forma
            # Usar timesteps ao invés
            kwargs = {
                'prompt': full_prompt,
                'image': image_input,
                'strength': strength,
                'guidance_scale': 3.5,
                'output_type': 'pt',  # Retornar como tensor PyTorch
                'generator': generator,
            }
            
            # Adicionar num_inference_steps apenas se o pipeline suportar
            # Flux pode usar timesteps diferentes
            if hasattr(self.flux_pipeline, 'scheduler'):
                # Flux usa timesteps, não num_inference_steps diretamente
                # O scheduler gerencia isso automaticamente
                pass
            else:
                kwargs['num_inference_steps'] = num_inference_steps
            
            # Adicionar prompt_2 se o pipeline suportar
            if hasattr(self.flux_pipeline, 'tokenizer_2'):
                kwargs['prompt_2'] = full_prompt
            
            # Adicionar ControlNet se disponível
            if control_image is not None and hasattr(self.flux_pipeline, 'controlnet'):
                try:
                    kwargs.update({
                        'control_image': control_image,
                        'controlnet_conditioning_scale': 0.6,
                    })
                except Exception as e:
                    print(f"[FLUX] Aviso ao adicionar ControlNet: {e}")
            
            # Adicionar Redux se disponível
            if redux_hparam and self.redux_pipeline is not None:
                # Redux modifica prompt_embeds
                # Por enquanto, pular se não implementado
                pass
            
            # Limpar cache CUDA antes de gerar
            torch.cuda.empty_cache()
            
            # Gerar
            with torch.no_grad():
                try:
                    output = self.flux_pipeline(**kwargs)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[FLUX] [AVISO]  CUDA OOM: {e}")
                    print(f"[FLUX] Tentando com configuracoes mais leves...")
                    # Limpar cache
                    torch.cuda.empty_cache()
                    # Tentar com parâmetros reduzidos
                    kwargs_simple = {
                        'prompt': full_prompt,
                        'image': image_input,
                        'strength': min(strength, 0.8),  # Reduzir strength
                        'num_inference_steps': min(num_inference_steps, 8),  # Reduzir passos
                        'guidance_scale': 2.0,  # Reduzir guidance
                        'output_type': 'pt',
                        'generator': generator,
                    }
                    try:
                        output = self.flux_pipeline(**kwargs_simple)
                    except Exception as e2:
                        print(f"[FLUX] [ERRO] Erro mesmo com configuracoes leves: {e2}")
                        # Fallback: retornar reference bundle image
                        return reference_bundle_image.clone(), ""
                except Exception as e:
                    print(f"[FLUX] Erro na geracao: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: retornar reference bundle image
                    return reference_bundle_image.clone(), ""
            
            # Converter output para tensor
            if isinstance(output, dict):
                images = output.get('images', output)
            else:
                images = output
            
            # Extrair imagem do output
            if isinstance(images, torch.Tensor):
                # Já é tensor
                if len(images.shape) == 4:
                    gen_image = images[0]  # Remover batch dimension
                else:
                    gen_image = images
                # Garantir formato (C, H, W)
                if gen_image.shape[0] != 3:
                    gen_image = gen_image.permute(2, 0, 1) if len(gen_image.shape) == 3 else gen_image
            elif isinstance(images, np.ndarray):
                gen_image = torch.from_numpy(images).squeeze(0)
                if len(gen_image.shape) == 3 and gen_image.shape[2] == 3:
                    gen_image = gen_image.permute(2, 0, 1)
                gen_image = gen_image.float() / 255.0 if gen_image.max() > 1.0 else gen_image.float()
            elif isinstance(images, list):
                gen_image = transforms.ToTensor()(images[0]).float()
            elif isinstance(images, Image.Image):
                gen_image = transforms.ToTensor()(images).float()
            else:
                # Fallback
                print(f"[FLUX] Formato de output inesperado: {type(images)}")
                gen_image = reference_bundle_image.clone()
            
            # Garantir range [0, 1]
            gen_image = torch.clamp(gen_image, 0, 1)
            
            # Ajustar tamanho se necessário para (3, 1024, 2048)
            if gen_image.shape != (3, 1024, 2048):
                gen_image = torch.nn.functional.interpolate(
                    gen_image.unsqueeze(0),
                    size=(1024, 2048),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            print(f"[FLUX] Bundle image gerado: {gen_image.shape}")
            
            # Salvar
            save_path = ""
            try:
                save_dir = Path("./outputs/tmp")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / "gen_bundle_image.png")
                torchvision.utils.save_image(gen_image, save_path)
                print(f"[FLUX] Bundle image salvo: {save_path}")
            except Exception as e:
                print(f"[FLUX] Erro ao salvar: {e}")
            
            return gen_image, save_path
            
        except Exception as e:
            print(f"[FLUX] Erro na geracao: {e}")
            import traceback
            traceback.print_exc()
            return reference_bundle_image.clone(), ""
    
    def _preprocess_controlnet_image(self,
                                     image: torch.Tensor,
                                     mode: str = 'tile',
                                     down_scale: int = 1) -> Image.Image:
        """Preprocessa imagem para ControlNet"""
        if mode == 'tile':
            # Tile: downscale e upscale
            _, h, w = image.shape
            down_h, down_w = h // down_scale, w // down_scale
            
            # Downscale
            downscaled = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(down_h, down_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Upscale
            upscaled = torch.nn.functional.interpolate(
                downscaled,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Converter para PIL
            upscaled = torch.clamp(upscaled, 0, 1)
            return transforms.ToPILImage()(upscaled)
        
        else:
            # Outros modos (canny, depth, etc)
            # Por enquanto, retornar imagem original
            return transforms.ToPILImage()(torch.clamp(image, 0, 1))

