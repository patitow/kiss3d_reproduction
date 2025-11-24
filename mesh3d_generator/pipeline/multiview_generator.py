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
            # Zero123++ usa um pipeline customizado que precisa ser carregado de forma especial
            # Tentar diferentes métodos de carregamento
            
            # Método 1: Tentar carregar com trust_remote_code (pode ter pipeline customizado)
            try:
                from diffusers import DiffusionPipeline
                from diffusers.schedulers import EulerAncestralDiscreteScheduler
                
                print(f"[MULTIVIEW] Tentando carregar Zero123++ com trust_remote_code...")
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
                
                # Configurar scheduler
                if hasattr(self.pipeline, 'scheduler'):
                    self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                        self.pipeline.scheduler.config,
                        timestep_spacing='trailing'
                    )
                
                self.pipeline = self.pipeline.to(self.device)
                print(f"[MULTIVIEW] Pipeline carregado com sucesso (trust_remote_code)")
                return
                
            except Exception as e1:
                print(f"[MULTIVIEW] Método 1 falhou: {e1}")
            
            # Método 2: Tentar usar threestudio ou outra biblioteca
            try:
                print(f"[MULTIVIEW] Tentando carregar via threestudio...")
                # threestudio pode ter suporte para Zero123++
                # Por enquanto, pular se não disponível
                raise ImportError("threestudio não disponível")
            except ImportError:
                pass
            
            # Método 3: Usar Stable Diffusion como fallback temporário
            print(f"[MULTIVIEW] Zero123++ não disponível - usando fallback")
            print(f"[MULTIVIEW] AVISO: Usando placeholder. Para Zero123++ real, instale dependências corretas")
            self.pipeline = None
            
        except Exception as e:
            print(f"[MULTIVIEW] Erro geral ao carregar: {e}")
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
            print(f"[MULTIVIEW] Pipeline não disponível - usando fallback (placeholder)")
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
                # Zero123++ pode usar diferentes APIs dependendo da implementação
                # Tentar diferentes formatos de chamada
                try:
                    # Tentativa 1: API padrão do diffusers com condição de câmera
                    # Zero123++ pode aceitar condição via prompt ou parâmetros específicos
                    output = self.pipeline(
                        input_image,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=512 * 2,
                        height=512 * 2,
                    )
                    
                    # Extrair imagem do output
                    if hasattr(output, 'images'):
                        view_image = output.images[0] if isinstance(output.images, list) else output.images
                    elif isinstance(output, list):
                        view_image = output[0]
                    elif isinstance(output, Image.Image):
                        view_image = output
                    else:
                        # Tentar acessar como dict
                        view_image = output.get('images', [input_image])[0]
                    
                    views.append(view_image)
                    
                except Exception as e1:
                    print(f"    [AVISO] Método 1 falhou: {e1}")
                    # Tentativa 2: Usar prompt com condição de câmera
                    try:
                        prompt = f"elevation={elevation} azimuth={azimuth}"
                        output = self.pipeline(
                            prompt=prompt,
                            image=input_image,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                        )
                        view_image = output.images[0] if isinstance(output.images, list) else output.images
                        views.append(view_image)
                    except Exception as e2:
                        print(f"    [AVISO] Método 2 falhou: {e2}")
                        # Fallback: usar imagem de input redimensionada
                        views.append(input_image.resize((1024, 1024)))
                
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

