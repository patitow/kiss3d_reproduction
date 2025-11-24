"""
Geração de multiview usando Zero123++
Implementação própria baseada na abordagem do Kiss3DGen
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from pathlib import Path
import torchvision.transforms as transforms


class Zero123MultiviewGenerator:
    """
    Gera múltiplas views de um objeto usando Zero123++
    """
    
    def __init__(self, 
                 model_id: str = "sudo-ai/zero123plus-v1.2",
                 device: str = "cuda:0",
                 dtype: torch.dtype = torch.float16):
        """
        Inicializa o gerador de multiview
        
        Args:
            model_id: ID do modelo no HuggingFace ou caminho local
            device: Dispositivo para processamento
            dtype: Tipo de dados para o modelo
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        
        print(f"[MULTIVIEW] Inicializando Zero123++: {model_id}")
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo Zero123++"""
        try:
            from diffusers import DiffusionPipeline
            from diffusers.schedulers import EulerAncestralDiscreteScheduler
            
            print(f"[MULTIVIEW] Tentando carregar Zero123++...")
            
            # Zero123++ não tem variant fp16, carregar sem variant
            # O modelo decide automaticamente qual precisão usar
            try:
                # Método 1: Carregar sem variant (modelo padrão)
                print(f"[MULTIVIEW] Tentando carregar sem variant...")
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
                print(f"[MULTIVIEW] [OK] Pipeline carregado sem variant!")
            except Exception as e1:
                print(f"[MULTIVIEW] Erro ao carregar sem variant: {e1}")
                # Método 2: Tentar com float32 (mais compatível)
                try:
                    print(f"[MULTIVIEW] Tentando carregar com float32...")
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    print(f"[MULTIVIEW] [OK] Pipeline carregado com float32!")
                except Exception as e2:
                    print(f"[MULTIVIEW] Erro ao carregar com float32: {e2}")
                    # Método 3: Tentar sem especificar dtype
                    try:
                        print(f"[MULTIVIEW] Tentando carregar sem especificar dtype...")
                        self.pipeline = DiffusionPipeline.from_pretrained(
                            self.model_id,
                            trust_remote_code=True
                        )
                        # Converter para dtype desejado após carregar
                        if self.dtype != torch.float32:
                            # Tentar converter componentes para dtype desejado
                            for component_name in ['unet', 'vae', 'transformer']:
                                if hasattr(self.pipeline, component_name):
                                    component = getattr(self.pipeline, component_name)
                                    if component is not None:
                                        setattr(self.pipeline, component_name, component.to(dtype=self.dtype))
                        print(f"[MULTIVIEW] [OK] Pipeline carregado e convertido!")
                    except Exception as e3:
                        print(f"[MULTIVIEW] Erro ao carregar: {e3}")
                        raise
            
            # Configurar scheduler
            if hasattr(self.pipeline, 'scheduler') and self.pipeline.scheduler is not None:
                try:
                    self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                        self.pipeline.scheduler.config,
                        timestep_spacing='trailing'
                    )
                except Exception:
                    pass  # Manter scheduler original se não conseguir configurar
            
            # Mover para device e otimizar VRAM
            try:
                # Habilitar CPU offload antes de mover para GPU (economiza VRAM)
                if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    self.pipeline.enable_model_cpu_offload()
                    print(f"[MULTIVIEW] [OK] CPU offload habilitado")
                else:
                    # Se não tiver offload, mover para device
                    self.pipeline = self.pipeline.to(self.device)
                    print(f"[MULTIVIEW] [OK] Pipeline movido para {self.device}")
            except Exception as e:
                print(f"[MULTIVIEW] [AVISO] Erro ao configurar device: {e}")
                # Tentar método simples
                try:
                    self.pipeline = self.pipeline.to(self.device)
                except Exception as e2:
                    print(f"[MULTIVIEW] [AVISO] Erro ao mover para device: {e2}")
            
            print(f"[MULTIVIEW] [OK] Zero123++ carregado com sucesso!")
            return
            
        except Exception as e:
            print(f"[MULTIVIEW] [AVISO] Zero123++ nao pode ser carregado: {e}")
            print(f"[MULTIVIEW]    O pipeline funcionara mas com qualidade reduzida (usando fallback)")
            import traceback
            traceback.print_exc()
            self.pipeline = None
    
    def generate_multiview(self,
                          input_image: Image.Image,
                          azimuths: List[float] = [270, 0, 90, 180],
                          elevations: List[float] = [5, 5, 5, 5],
                          num_inference_steps: int = 50,
                          seed: Optional[int] = None) -> Image.Image:
        """
        Gera imagem multiview com múltiplas vistas do objeto
        
        Args:
            input_image: Imagem de input (PIL Image, 512x512)
            azimuths: Lista de azimutes em graus [270, 0, 90, 180]
            elevations: Lista de elevações em graus [5, 5, 5, 5]
            num_inference_steps: Número de passos de inferência
            seed: Seed para reprodutibilidade
        
        Returns:
            Imagem multiview (1024x2048) com 4 views em grid 2x2
        """
        print(f"[MULTIVIEW] Gerando multiview com {len(azimuths)} views...")
        
        # Se pipeline não estiver disponível, usar fallback
        if self.pipeline is None:
            print(f"[MULTIVIEW] Pipeline nao disponivel - usando fallback (placeholder)")
            return self._generate_multiview_fallback(input_image, azimuths, elevations)
        
        # Zero123++ espera imagem 512x512
        if input_image.size != (512, 512):
            input_image = input_image.resize((512, 512), Image.LANCZOS)
        
        # Gerar views individuais
        views = []
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        for i, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
            print(f"[MULTIVIEW] Gerando view {i+1}/{len(azimuths)}: azimuth={azimuth}°, elevation={elevation}°")
            
            try:
                # Zero123++ aceita condição de câmera via parâmetros específicos
                # Tentar diferentes formatos de chamada
                view_image = None
                
                # Tentativa 1: API com condição de câmera explícita
                try:
                    # Zero123++ pode aceitar elevation e azimuth como parâmetros
                    if hasattr(self.pipeline, '__call__'):
                        # Tentar chamada direta com imagem
                        output = self.pipeline(
                            image=input_image,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            elevation=elevation,
                            azimuth=azimuth,
                        )
                        
                        # Extrair imagem
                        if hasattr(output, 'images'):
                            view_image = output.images[0] if isinstance(output.images, list) else output.images
                        elif isinstance(output, list):
                            view_image = output[0]
                        elif isinstance(output, Image.Image):
                            view_image = output
                        elif isinstance(output, dict):
                            view_image = output.get('images', [input_image])[0]
                except Exception as e1:
                    # Tentativa 2: API padrão img2img
                    try:
                        output = self.pipeline(
                            prompt="",  # Zero123++ não precisa de prompt
                            image=input_image,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            strength=0.75,
                        )
                        if hasattr(output, 'images'):
                            view_image = output.images[0] if isinstance(output.images, list) else output.images
                        elif isinstance(output, list):
                            view_image = output[0]
                        elif isinstance(output, Image.Image):
                            view_image = output
                    except Exception as e2:
                        print(f"    [AVISO] Metodo 2 falhou: {e2}")
                
                if view_image is None:
                    # Fallback: usar imagem de input
                    view_image = input_image.resize((1024, 1024))
                    print(f"    [AVISO] Usando fallback para view {i+1}")
                
                views.append(view_image)
                
            except Exception as e:
                print(f"[MULTIVIEW] Erro ao gerar view {i+1}: {e}")
                # Fallback: usar imagem de input
                views.append(input_image.resize((1024, 1024)))
        
        # Combinar views em grid 2x2 (1024x2048)
        if len(views) == 4:
            # Criar grid: 2 linhas, 2 colunas
            grid_image = Image.new('RGB', (2048, 1024))
            
            # Posicionar views
            positions = [
                (0, 0),      # View 0: top-left
                (1024, 0),   # View 1: top-right
                (0, 1024),   # View 2: bottom-left (mas só temos 2 linhas)
                (1024, 1024) # View 3: bottom-right (mas só temos 2 linhas)
            ]
            
            # Ajustar: grid 2x2 significa 2 colunas x 2 linhas = 4 views
            # Mas queremos 2 linhas x 4 colunas para bundle image
            # Vamos fazer 2x2 primeiro e depois ajustar
            
            # Redimensionar views para 512x512 se necessário
            views_resized = [v.resize((512, 512)) for v in views]
            
            # Criar grid 2x2 (1024x1024)
            grid_2x2 = Image.new('RGB', (1024, 1024))
            grid_2x2.paste(views_resized[0], (0, 0))
            grid_2x2.paste(views_resized[1], (512, 0))
            grid_2x2.paste(views_resized[2], (0, 512))
            grid_2x2.paste(views_resized[3], (512, 512))
            
            # Para bundle image, precisamos 2 linhas x 4 colunas (1024x2048)
            # Mas Zero123 gera apenas 4 views, então vamos duplicar ou usar apenas 2x2
            # Por enquanto, retornar 2x2 e depois expandir no bundle image
            multiview_image = grid_2x2.resize((2048, 1024))
            
        else:
            # Se não tiver 4 views, criar grid simples
            multiview_image = views[0].resize((2048, 1024))
        
        print(f"[MULTIVIEW] Multiview gerado: {multiview_image.size}")
        return multiview_image
    
    def _generate_multiview_fallback(self,
                                    input_image: Image.Image,
                                    azimuths: List[float],
                                    elevations: List[float]) -> Image.Image:
        """Gera multiview usando fallback (placeholder)"""
        print("[MULTIVIEW] Usando fallback: criando multiview placeholder")
        
        # Redimensionar imagem para 512x512
        if input_image.size != (512, 512):
            input_image = input_image.resize((512, 512), Image.LANCZOS)
        
        # Criar 4 views (por enquanto, usar a mesma imagem com pequenas variações)
        views = []
        for i, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
            # Placeholder: usar imagem original com leve transformação
            # Em produção, isso seria substituído por geração real via Zero123++
            view = input_image.copy()
            
            # Aplicar leve rotação/transformação para simular diferentes views
            # Por enquanto, apenas usar a mesma imagem
            views.append(view)
        
        # Criar grid 2x2 (1024x1024) e depois expandir para 2x4 (1024x2048)
        grid_2x2 = Image.new('RGB', (1024, 1024))
        views_resized = [v.resize((512, 512)) for v in views]
        grid_2x2.paste(views_resized[0], (0, 0))
        grid_2x2.paste(views_resized[1], (512, 0))
        grid_2x2.paste(views_resized[2], (0, 512))
        grid_2x2.paste(views_resized[3], (512, 512))
        
        # Expandir para 2x4 (1024x2048) - duplicar colunas
        multiview_image = Image.new('RGB', (2048, 1024))
        multiview_image.paste(grid_2x2, (0, 0))
        multiview_image.paste(grid_2x2, (1024, 0))
        
        return multiview_image

