"""
Pipeline próprio de geração 3D a partir de imagens
Implementado seguindo a abordagem do Kiss3DGen (referência), mas com código próprio
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
import numpy as np
from PIL import Image
import torchvision

# Adicionar paths necessários
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ImageTo3DPipeline:
    """
    Pipeline próprio para geração 3D a partir de imagens
    Segue a abordagem do Kiss3DGen mas com implementação própria
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 zero123_model: Optional[str] = None,
                 flux_model: Optional[str] = None,
                 lrm_model: Optional[str] = None,
                 caption_model: Optional[str] = None):
        """
        Inicializa o pipeline
        
        Args:
            device: Dispositivo para processamento (cuda:0, cpu, etc)
            zero123_model: Caminho ou ID do modelo Zero123
            flux_model: Caminho ou ID do modelo Flux
            lrm_model: Caminho ou ID do modelo LRM
            caption_model: Caminho ou ID do modelo de caption
        """
        # Verificar se CUDA está disponível
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[PIPELINE] AVISO: CUDA não disponível, usando CPU")
            device = "cpu"
        
        self.device = device
        self.zero123_model = zero123_model
        self.flux_model = flux_model
        self.lrm_model = lrm_model
        self.caption_model = caption_model
        
        # Modelos serão carregados sob demanda
        self._zero123_pipeline = None
        self._flux_pipeline = None
        self._lrm_model = None
        self._caption_model = None
        
        print(f"[PIPELINE] Inicializado com device: {device}")
        if device == "cpu":
            print(f"[PIPELINE] AVISO: Processamento em CPU será muito lento para modelos grandes")
    
    def generate_reference_3d_bundle_image(self, 
                                         input_image: Image.Image,
                                         use_mv_rgb: bool = True,
                                         seed: Optional[int] = None) -> Tuple[torch.Tensor, str]:
        """
        Passo 3: Gera reference 3D bundle image usando Zero123 + LRM
        
        Segue a abordagem do Kiss3DGen:
        1. Gera multiview usando Zero123
        2. Reconstrói mesh inicial usando LRM
        3. Renderiza 4 views RGB + 4 normal maps
        4. Cria grid 2x4 (1024x2048) com todas as views
        
        Args:
            input_image: Imagem de input (PIL Image)
            use_mv_rgb: Se True, usa RGB multiview do Zero123
            seed: Seed para reprodutibilidade
        
        Returns:
            Tuple[bundle_image_tensor, save_path]
        """
        print("[PIPELINE] Passo 3: Gerando reference 3D bundle image...")
        
        try:
            from mesh3d_generator.pipeline.multiview_generator import Zero123MultiviewGenerator
            
            # 1. Gerar multiview com Zero123++
            print("  [3.1] Gerando multiview com Zero123++...")
            multiview_gen = Zero123MultiviewGenerator(
                model_id="sudo-ai/zero123plus-v1.2",
                device=self.device,
                dtype=torch.float16
            )
            
            # Gerar 4 views: 270°, 0°, 90°, 180° com elevação 5°
            multiview_image = multiview_gen.generate_multiview(
                input_image,
                azimuths=[270, 0, 90, 180],
                elevations=[5, 5, 5, 5],
                num_inference_steps=50,
                seed=seed
            )
            
            print(f"  [OK] Multiview gerado: {multiview_image.size}")
            
            # 2. Reconstruir mesh inicial com LRM
            print("  [3.2] Reconstruindo mesh inicial com LRM...")
            try:
                from mesh3d_generator.pipeline.lrm_reconstructor import LRMReconstructor
                
                lrm = LRMReconstructor(device=self.device)
                vertices, faces, normals, rgb_views, albedo_views = lrm.reconstruct_from_multiview(
                    multiview_image,
                    render_radius=4.15,
                    render_azimuths=[270, 0, 90, 180],
                    render_elevations=[5, 5, 5, 5]
                )
                
                print(f"  [OK] Mesh reconstruído: {len(vertices)} vertices, {len(faces)} faces")
                
            except Exception as e:
                print(f"  [AVISO] LRM não disponível: {e}")
                # Fallback: usar multiview RGB diretamente
                import torchvision.transforms.functional as TF
                bundle_tensor = TF.to_tensor(multiview_image).float()
                rgb_views = bundle_tensor
                vertices, faces, normals = None, None, None
            
            # 3. Renderizar normal maps
            print("  [3.3] Renderizando normal maps...")
            try:
                from mesh3d_generator.pipeline.normal_renderer import NormalMapRenderer
                
                if vertices is not None and faces is not None:
                    normal_renderer = NormalMapRenderer(device=self.device)
                    normal_maps = normal_renderer.render_normal_maps(
                        vertices,
                        faces,
                        azimuths=[270, 0, 90, 180],
                        elevations=[5, 5, 5, 5],
                        radius=4.5,
                        render_size=512
                    )
                    
                    # Converter normal maps para formato de bundle (1024x2048)
                    # Normal maps vêm como (4, 3, 512, 512), precisamos criar grid 2x4
                    normal_views = self._create_bundle_from_views(normal_maps)
                    
                    print(f"  [OK] Normal maps renderizados: {normal_maps.shape}")
                else:
                    # Fallback: criar normal maps placeholder
                    normal_views = torch.ones_like(rgb_views) * 0.5
                    print("  [AVISO] Usando normal maps placeholder")
                    
            except Exception as e:
                print(f"  [AVISO] Normal renderer não disponível: {e}")
                # Fallback: criar normal maps placeholder
                normal_views = torch.ones_like(rgb_views) * 0.5
            
            # 4. Criar bundle image completo: RGB (linha 1) + Normal (linha 2)
            # Grid 2x4: 2 linhas x 4 colunas = 8 views total
            print("  [3.4] Criando bundle image completo...")
            
            # Normalizar shapes: rgb_views pode ser (4, 3, H, W) ou (3, H, W)
            # normal_views é (4, 3, H, W)
            if len(rgb_views.shape) == 4:
                # Se for (4, 3, H, W), criar grid horizontal primeiro
                rgb_views = self._create_bundle_from_views(rgb_views)  # (3, H, W_total)
            elif len(rgb_views.shape) == 3:
                # Já está no formato (3, H, W)
                pass
            else:
                # Converter para formato correto
                rgb_views = rgb_views.view(-1, 3, 512, 512)
                rgb_views = self._create_bundle_from_views(rgb_views)
            
            # Normalizar normal_views: (4, 3, H, W) -> (3, H, W_total)
            if len(normal_views.shape) == 4:
                normal_views = self._create_bundle_from_views(normal_views)
            elif len(normal_views.shape) == 3:
                pass
            else:
                normal_views = normal_views.view(4, 3, 512, 512)
                normal_views = self._create_bundle_from_views(normal_views)
            
            # Garantir que ambos têm o mesmo tamanho
            if rgb_views.shape != normal_views.shape:
                # Redimensionar para (3, 512, 2048) cada
                target_size = (512, 2048)
                rgb_views = torch.nn.functional.interpolate(
                    rgb_views.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                normal_views = torch.nn.functional.interpolate(
                    normal_views.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Combinar: RGB (linha 1) + Normal (linha 2) = (3, 1024, 2048)
            bundle_image = torch.cat([rgb_views, normal_views], dim=1)  # Concatenar verticalmente
            
            # Ajustar para (3, 1024, 2048) se necessário
            if bundle_image.shape != (3, 1024, 2048):
                bundle_image = torch.nn.functional.interpolate(
                    bundle_image.unsqueeze(0),
                    size=(1024, 2048),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            print(f"  [OK] Bundle image criado: {bundle_image.shape}")
            
            # Salvar para debug
            save_path = ""
            try:
                import torchvision
                from pathlib import Path
                save_dir = Path("./outputs/tmp")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / "reference_bundle_image.png")
                torchvision.utils.save_image(bundle_image, save_path)
                print(f"  [OK] Bundle image salvo: {save_path}")
            except Exception as e:
                print(f"  [AVISO] Não foi possível salvar bundle image: {e}")
            
            return bundle_image, save_path
            
        except ImportError as e:
            print(f"  [ERRO] Módulo não encontrado: {e}")
            print(f"  [AVISO] Instale: pip install diffusers transformers")
            # Fallback: criar bundle image vazio
            bundle_image = torch.zeros((3, 1024, 2048), dtype=torch.float32)
            return bundle_image, ""
        except Exception as e:
            print(f"  [ERRO] Falha ao gerar reference bundle image: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: criar bundle image vazio
            bundle_image = torch.zeros((3, 1024, 2048), dtype=torch.float32)
            return bundle_image, ""
    
    def generate_3d_bundle_image_controlnet(self,
                                          prompt: str,
                                          reference_bundle_image: torch.Tensor,
                                          input_image: Image.Image,
                                          strength: float = 0.95,
                                          enable_redux: bool = True,
                                          seed: Optional[int] = None) -> Tuple[torch.Tensor, str]:
        """
        Passo 4: Gera 3D bundle image final usando Flux + ControlNet
        
        Segue a abordagem do Kiss3DGen:
        1. Usa Flux diffusion model com ControlNet-Tile
        2. ControlNet usa reference bundle image como condição
        3. Redux usa imagem de input para melhorar prompt embeddings
        4. Gera novo 3D bundle image refinado (RGB + normal maps)
        
        Args:
            prompt: Caption/descrição da imagem
            reference_bundle_image: Reference bundle image (3, 1024, 2048)
            input_image: Imagem de input original
            strength: Strength para denoising
            enable_redux: Se True, usa Flux Prior Redux
            seed: Seed para reprodutibilidade
        
        Returns:
            Tuple[bundle_image_tensor, save_path]
        """
        print("[PIPELINE] Passo 4: Gerando 3D bundle image final (Flux + ControlNet)...")
        
        try:
            from mesh3d_generator.pipeline.flux_controlnet_generator import FluxControlNetGenerator
            
            flux_gen = FluxControlNetGenerator(
                flux_model_id="black-forest-labs/FLUX.1-dev",
                controlnet_model_id="InstantX/FLUX.1-dev-Controlnet-Union",
                redux_model_id="black-forest-labs/FLUX.1-Redux-dev" if enable_redux else None,
                device=self.device,
                dtype=torch.bfloat16
            )
            
            # Carregar modelos (sob demanda)
            if flux_gen.flux_pipeline is None:
                flux_gen._load_models()
            
            # Gerar bundle image refinado
            bundle_image, save_path = flux_gen.generate_bundle_image(
                prompt=prompt,
                reference_bundle_image=reference_bundle_image,
                input_image=input_image,
                strength=strength,
                enable_redux=enable_redux,
                control_mode='tile',
                num_inference_steps=20,
                seed=seed
            )
            
            print(f"  [OK] Bundle image final gerado: {bundle_image.shape}")
            return bundle_image, save_path
            
        except Exception as e:
            print(f"  [AVISO] Flux + ControlNet não disponível: {e}")
            print("  [AVISO] Usando reference bundle image")
            # Fallback: usar reference bundle image
            bundle_image = reference_bundle_image.clone()
            return bundle_image, ""
    
    def reconstruct_3d_mesh(self,
                           bundle_image: torch.Tensor,
                           lrm_render_radius: float = 4.15,
                           isomer_radius: float = 4.5,
                           reconstruction_stage1_steps: int = 10,
                           reconstruction_stage2_steps: int = 50,
                           output_path: Optional[str] = None) -> str:
        """
        Passo 5: Reconstrói mesh 3D a partir do bundle image
        
        Segue a abordagem do Kiss3DGen:
        1. Separa RGB e normal maps do bundle image
        2. Reconstrói mesh usando LRM (Large Reconstruction Model)
        3. Refina mesh usando ISOMER com normal maps
        4. Exporta mesh final com texturas geradas
        
        Args:
            bundle_image: Bundle image final (3, 1024, 2048)
            lrm_render_radius: Raio de renderização para LRM
            isomer_radius: Raio para ISOMER
            reconstruction_stage1_steps: Passos para stage 1 do ISOMER
            reconstruction_stage2_steps: Passos para stage 2 do ISOMER
            output_path: Caminho de saída (opcional)
        
        Returns:
            Caminho para o mesh gerado (.obj ou .glb)
        """
        print("[PIPELINE] Passo 5: Reconstruindo mesh 3D (LRM + ISOMER)...")
        
        try:
            from einops import rearrange
            
            # 1. Separar RGB e normal maps do bundle image
            print("  [5.1] Separando RGB e normal maps...")
            # Bundle image: (3, 1024, 2048) = 2 linhas (RGB + Normal) x 4 colunas (views)
            # Separar em 8 imagens: 4 RGB + 4 normal
            images = rearrange(bundle_image, 'c (n h) (m w) -> (n m) c h w', n=2, m=4)  # (8, 3, 512, 512)
            rgb_maps = images[:4]  # Primeiras 4: RGB
            normal_maps = images[4:]  # Últimas 4: Normal
            
            print(f"  [OK] Separado: {len(rgb_maps)} RGB maps, {len(normal_maps)} normal maps")
            
            # 2. Reconstruir mesh inicial com LRM
            print("  [5.2] Reconstruindo mesh inicial com LRM...")
            try:
                from mesh3d_generator.pipeline.lrm_reconstructor import LRMReconstructor
                
                lrm = LRMReconstructor(device=self.device)
                vertices, faces, _, _, _ = lrm.reconstruct_from_multiview(
                    # Converter RGB maps para imagem multiview
                    self._views_to_image(rgb_maps),
                    render_radius=lrm_render_radius,
                    render_azimuths=[270, 0, 90, 180],
                    render_elevations=[5, 5, 5, 5]
                )
                
                print(f"  [OK] Mesh inicial: {len(vertices)} vertices, {len(faces)} faces")
                
            except Exception as e:
                print(f"  [AVISO] LRM não disponível: {e}")
                # Fallback: criar mesh simples
                import trimesh
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                vertices = torch.from_numpy(sphere.vertices).float()
                faces = torch.from_numpy(sphere.faces).long()
            
            # 3. Refinar mesh com ISOMER
            print("  [5.3] Refinando mesh com ISOMER...")
            try:
                from mesh3d_generator.pipeline.isomer_refiner import ISOMERRefiner
                
                isomer = ISOMERRefiner(device=self.device)
                refined_vertices, refined_faces = isomer.refine_mesh(
                    vertices,
                    faces,
                    normal_maps,
                    rgb_maps,
                    azimuths=[270, 0, 90, 180],
                    elevations=[5, 5, 5, 5],
                    radius=isomer_radius,
                    stage1_steps=reconstruction_stage1_steps,
                    stage2_steps=reconstruction_stage2_steps
                )
                
                vertices, faces = refined_vertices, refined_faces
                print(f"  [OK] Mesh refinado: {len(vertices)} vertices, {len(faces)} faces")
                
            except Exception as e:
                print(f"  [AVISO] ISOMER não disponível: {e}")
                # Continuar com mesh inicial
            
            # 4. Projetar texturas e exportar
            print("  [5.4] Projetando texturas e exportando...")
            if output_path is None:
                output_path = "./outputs/tmp/reconstructed_mesh.obj"
            
            try:
                from mesh3d_generator.pipeline.isomer_refiner import ISOMERRefiner
                
                isomer = ISOMERRefiner(device=self.device)
                mesh_path = isomer.project_textures(
                    vertices,
                    faces,
                    rgb_maps,
                    normal_maps,
                    azimuths=[270, 0, 90, 180],
                    elevations=[5, 5, 5, 5],
                    radius=isomer_radius,
                    output_path=output_path
                )
                
                print(f"  [OK] Mesh final salvo: {mesh_path}")
                return mesh_path
                
            except Exception as e:
                print(f"  [AVISO] Projeção de texturas não disponível: {e}")
                # Fallback: salvar mesh simples
                import trimesh
                mesh = trimesh.Trimesh(
                    vertices=vertices.cpu().numpy(),
                    faces=faces.cpu().numpy()
                )
                mesh.export(output_path)
                return output_path
            
        except Exception as e:
            print(f"  [ERRO] Falha na reconstrução: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _create_bundle_from_views(self, views: torch.Tensor) -> torch.Tensor:
        """Cria bundle image a partir de views individuais"""
        # views: (4, 3, H, W) ou (N, 3, H, W)
        # Criar grid 1x4 (uma linha com 4 views)
        # Usar torchvision para criar grid
        import torchvision
        
        # Garantir que views está no formato correto
        if len(views.shape) == 3:
            # Se for (3, H, W), assumir que já é uma view única
            return views
        elif len(views.shape) == 4:
            # (N, 3, H, W) - criar grid
            grid = torchvision.utils.make_grid(views, nrow=4, padding=0)
            return grid
        else:
            raise ValueError(f"Formato de views inválido: {views.shape}")
    
    def _views_to_image(self, views: torch.Tensor) -> Image.Image:
        """Converte tensor de views para PIL Image"""
        import torchvision
        grid = torchvision.utils.make_grid(views, nrow=2, padding=0)
        grid = torch.clamp(grid, 0, 1)
        return transforms.ToPILImage()(grid)
    
    def generate_3d_model(self,
                         input_image_path: str,
                         output_dir: str,
                         object_name: str,
                         seed: Optional[int] = None,
                         enable_redux: bool = True,
                         use_mv_rgb: bool = True,
                         use_controlnet: bool = True,
                         strength1: float = 0.5,
                         strength2: float = 0.95) -> Tuple[str, str, str]:
        """
        Pipeline completo: gera modelo 3D a partir de uma imagem
        
        Args:
            input_image_path: Caminho para a imagem de input
            output_dir: Diretório de saída
            object_name: Nome do objeto
            seed: Seed para reprodutibilidade
            enable_redux: Usar Flux Prior Redux
            use_mv_rgb: Usar RGB multiview
            use_controlnet: Usar ControlNet
            strength1: Strength para Redux
            strength2: Strength para denoising
        
        Returns:
            Tuple[generated_mesh_path, bundle_image_path, caption]
        """
        print(f"\n[PIPELINE] Iniciando geração 3D para: {object_name}")
        
        # Carregar imagem
        input_image = Image.open(input_image_path)
        
        # Passo 1: Gerar caption (já implementado no script principal)
        # Aqui assumimos que já foi gerado, mas podemos implementar também
        caption = f"A detailed 3D model of {object_name}"
        
        # Passo 2: Gerar reference 3D bundle image
        reference_bundle_image, ref_save_path = self.generate_reference_3d_bundle_image(
            input_image,
            use_mv_rgb=use_mv_rgb,
            seed=seed
        )
        
        # Passo 3: Gerar bundle image final
        gen_bundle_image, gen_save_path = self.generate_3d_bundle_image_controlnet(
            prompt=caption,
            reference_bundle_image=reference_bundle_image,
            input_image=input_image,
            strength=strength2,
            enable_redux=enable_redux,
            seed=seed
        )
        
        # Salvar arquivos
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        object_name_safe = object_name.replace(' ', '_').replace('/', '_')
        generated_mesh_path = output_path / f"generated_{object_name_safe}.obj"
        bundle_output_path = output_path / f"bundle_image_{object_name_safe}.png"
        
        # Salvar bundle image
        if gen_save_path:
            import shutil
            shutil.copy(gen_save_path, bundle_output_path)
        else:
            # Salvar tensor como imagem
            torchvision.utils.save_image(gen_bundle_image, str(bundle_output_path))
        
        # Passo 4: Reconstruir mesh
        mesh_path = self.reconstruct_3d_mesh(
            gen_bundle_image,
            lrm_render_radius=4.15,
            isomer_radius=4.5,
            reconstruction_stage1_steps=10,
            reconstruction_stage2_steps=50,
            output_path=str(generated_mesh_path)
        )
        
        # Se mesh foi gerado, usar o caminho retornado
        if mesh_path and Path(mesh_path).exists():
            generated_mesh_path = Path(mesh_path)
        
        return str(generated_mesh_path), str(bundle_output_path), caption

