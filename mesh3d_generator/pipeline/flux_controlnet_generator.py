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
            
            # Tentar carregar como single file primeiro
            try:
                from diffusers import FluxImg2ImgPipeline
                self.flux_pipeline = FluxImg2ImgPipeline.from_single_file(
                    self.flux_model_id,
                    torch_dtype=self.dtype
                )
            except:
                # Fallback: carregar do HuggingFace
                self.flux_pipeline = FluxImg2ImgPipeline.from_pretrained(
                    self.flux_model_id,
                    torch_dtype=self.dtype
                )
            
            # Configurar scheduler
            self.flux_pipeline.scheduler = FlowMatchHeunDiscreteScheduler.from_config(
                self.flux_pipeline.scheduler.config
            )
            
            # Carregar ControlNet
            print(f"[FLUX] Carregando ControlNet: {self.controlnet_model_id}")
            try:
                from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
                
                controlnet = FluxControlNetModel.from_pretrained(
                    self.controlnet_model_id,
                    torch_dtype=torch.bfloat16
                )
                
                # Converter pipeline para usar ControlNet
                from diffusers import FluxControlNetImg2ImgPipeline
                self.flux_pipeline = FluxControlNetImg2ImgPipeline(
                    scheduler=self.flux_pipeline.scheduler,
                    vae=self.flux_pipeline.vae,
                    text_encoder=self.flux_pipeline.text_encoder,
                    tokenizer=self.flux_pipeline.tokenizer,
                    text_encoder_2=self.flux_pipeline.text_encoder_2,
                    tokenizer_2=self.flux_pipeline.tokenizer_2,
                    transformer=self.flux_pipeline.transformer,
                    controlnet=[controlnet]
                )
                
            except Exception as e:
                print(f"[FLUX] Erro ao carregar ControlNet: {e}")
                print("[FLUX] Continuando sem ControlNet")
            
            # Carregar Redux (opcional)
            if self.redux_model_id:
                try:
                    print(f"[FLUX] Carregando Redux: {self.redux_model_id}")
                    # Redux geralmente é um pipeline customizado
                    # Referência: Kiss3DGen usa FluxPriorReduxPipeline
                    # Por enquanto, pular se não disponível
                    print("[FLUX] Redux será carregado sob demanda")
                except Exception as e:
                    print(f"[FLUX] Redux não disponível: {e}")
            
            # Mover para device
            self.flux_pipeline = self.flux_pipeline.to(self.device)
            
            # Carregar LoRA se disponível
            try:
                lora_path = "./checkpoint/flux_lora/rgb_normal.safetensors"
                if Path(lora_path).exists():
                    print(f"[FLUX] Carregando LoRA: {lora_path}")
                    self.flux_pipeline.load_lora_weights(lora_path)
            except Exception as e:
                print(f"[FLUX] LoRA não disponível: {e}")
            
            print("[FLUX] Modelos carregados com sucesso")
            
        except ImportError as e:
            print(f"[FLUX] Diffusers não disponível: {e}")
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
            print("[FLUX] Pipeline não disponível - usando reference bundle image")
            return reference_bundle_image.clone(), ""
        
        try:
            # Preparar prompt
            base_prompt = "A grid of 2x4 multi-view image, elevation 5. White background."
            full_prompt = f"{base_prompt} {prompt}"
            
            # Preparar imagem de input (reference bundle)
            image_input = reference_bundle_image.unsqueeze(0).to(self.device)  # (1, 3, 1024, 2048)
            
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
            
            # Preparar parâmetros
            kwargs = {
                'prompt': base_prompt,
                'prompt_2': full_prompt,
                'image': image_input,
                'strength': strength,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': 3.5,
                'width': 2048,
                'height': 1024,
                'output_type': 'np',
                'generator': generator,
            }
            
            # Adicionar ControlNet se disponível
            if control_image is not None:
                kwargs.update({
                    'control_image': [control_image],
                    'control_mode': [1],  # Tile mode
                    'control_guidance_start': [0.0],
                    'control_guidance_end': [0.65],
                    'controlnet_conditioning_scale': [0.6],
                })
            
            # Adicionar Redux se disponível
            if redux_hparam:
                # Redux modifica prompt_embeds
                # Por enquanto, pular se não implementado
                pass
            
            # Gerar
            with torch.no_grad():
                output = self.flux_pipeline(**kwargs)
            
            # Converter output para tensor
            if isinstance(output, dict):
                images = output.get('images', output)
            else:
                images = output
            
            if isinstance(images, np.ndarray):
                gen_image = torch.from_numpy(images).squeeze(0).permute(2, 0, 1).float() / 255.0
            else:
                gen_image = transforms.ToTensor()(images[0] if isinstance(images, list) else images).float()
            
            # Ajustar tamanho se necessário
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
            print(f"[FLUX] Erro na geração: {e}")
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

