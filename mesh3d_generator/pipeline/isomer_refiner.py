"""
Refinamento de mesh usando ISOMER
Implementação própria baseada na abordagem do Kiss3DGen
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import trimesh


class ISOMERRefiner:
    """
    Refina mesh 3D usando ISOMER com normal maps
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        Inicializa o refinador ISOMER
        
        Args:
            device: Dispositivo para processamento
        """
        self.device = device
        print(f"[ISOMER] Inicializado com device: {device}")
    
    def refine_mesh(self,
                    vertices: torch.Tensor,
                    faces: torch.Tensor,
                    normal_maps: torch.Tensor,
                    rgb_maps: torch.Tensor,
                    azimuths: List[float] = [0, 90, 180, 270],
                    elevations: List[float] = [5, 5, 5, 5],
                    radius: float = 4.5,
                    stage1_steps: int = 10,
                    stage2_steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refina mesh usando ISOMER
        
        Args:
            vertices: Vértices iniciais (N, 3)
            faces: Faces iniciais (M, 3)
            normal_maps: Normal maps (4, 3, H, W)
            rgb_maps: RGB maps (4, 3, H, W)
            azimuths: Azimutes das views
            elevations: Elevações das views
            radius: Raio da câmera
            stage1_steps: Passos para stage 1
            stage2_steps: Passos para stage 2
        
        Returns:
            Tuple[refined_vertices, refined_faces]
        """
        print(f"[ISOMER] Refinando mesh (stage1: {stage1_steps}, stage2: {stage2_steps})...")
        
        try:
            # Tentar usar ISOMER real
            return self._refine_with_isomer(
                vertices, faces, normal_maps, rgb_maps,
                azimuths, elevations, radius, stage1_steps, stage2_steps
            )
        except ImportError:
            print("[ISOMER] Módulo ISOMER não encontrado - usando refinamento simples")
            return self._refine_simple(vertices, faces)
        except Exception as e:
            print(f"[ISOMER] Erro no ISOMER: {e}")
            return self._refine_simple(vertices, faces)
    
    def _refine_with_isomer(self,
                           vertices: torch.Tensor,
                           faces: torch.Tensor,
                           normal_maps: torch.Tensor,
                           rgb_maps: torch.Tensor,
                           azimuths: List[float],
                           elevations: List[float],
                           radius: float,
                           stage1_steps: int,
                           stage2_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refina usando ISOMER real"""
        try:
            from models.ISOMER.reconstruction_func import reconstruction
            from models.ISOMER.projection_func import projection
            from utils.tool import get_background, mask_fix
            
            # Converter para formato esperado pelo ISOMER
            device = self.device
            
            # Normal maps: converter de local para global
            # ISOMER espera normal maps em formato específico
            normal_maps_permuted = normal_maps.permute(0, 2, 3, 1)  # (4, H, W, 3)
            
            # Converter normal maps locais para globais
            # Referência: Kiss3DGen usa NormalTransfer
            try:
                from utils.tool import NormalTransfer
                normal_transfer = NormalTransfer()
                
                # Converter para numpy para processamento
                normal_maps_np = normal_maps_permuted.cpu().numpy()
                
                # Converter local para global
                global_normals = []
                for i, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
                    local_normal = torch.from_numpy(normal_maps_np[i]).float()
                    if local_normal.min() >= 0:
                        local_normal = local_normal * 2 - 1  # [0, 1] -> [-1, 1]
                    
                    global_normal = normal_transfer.trans_local_2_global(
                        local_normal.unsqueeze(0),
                        azimuth,
                        elevation,
                        radius=radius,
                        for_lotus=False
                    )
                    global_normal[..., 0] *= -1
                    global_normal = (global_normal + 1) / 2  # [-1, 1] -> [0, 1]
                    global_normals.append(global_normal.squeeze(0))
                
                global_normal_maps = torch.stack(global_normals, dim=0).permute(0, 3, 1, 2)  # (4, 3, H, W)
                
            except ImportError:
                print("[ISOMER] NormalTransfer não disponível - usando normais diretamente")
                global_normal_maps = normal_maps
            
            # Preparar máscaras
            masks = get_background(global_normal_maps).to(device)
            
            # Preparar pesos
            geo_weights = torch.tensor([1.0, 0.9, 1.0, 0.9], device=device)
            color_weights = torch.tensor([1.0, 0.5, 1.0, 0.5], device=device)
            
            # Converter vertices e faces para formato esperado
            vertices_np = vertices.cpu().numpy()
            faces_np = faces.cpu().numpy()
            
            # Stage 1 e 2: Reconstrução
            meshes = reconstruction(
                normal_pils=global_normal_maps.permute(0, 2, 3, 1),  # (4, H, W, 3)
                masks=masks.squeeze(1),  # (4, H, W)
                weights=geo_weights,
                fov=30,
                radius=radius,
                camera_angles_azi=torch.tensor(azimuths, device=device),
                camera_angles_ele=torch.tensor(elevations, device=device),
                expansion_weight_stage1=0.1,
                init_type="file",
                init_verts=vertices_np,
                init_faces=faces_np,
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
                start_edge_len_stage1=0.1,
                end_edge_len_stage1=0.02,
                start_edge_len_stage2=0.02,
                end_edge_len_stage2=0.005,
            )
            
            # Extrair vértices e faces refinados
            refined_vertices = torch.from_numpy(meshes.vertices).float()
            refined_faces = torch.from_numpy(meshes.faces).long()
            
            print(f"[ISOMER] Mesh refinado: {len(refined_vertices)} vertices, {len(refined_faces)} faces")
            
            return refined_vertices, refined_faces
            
        except ImportError:
            raise ImportError("ISOMER não disponível")
    
    def _refine_simple(self,
                      vertices: torch.Tensor,
                      faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refinamento simples (placeholder)"""
        print("[ISOMER] Usando refinamento simples (placeholder)")
        
        # Por enquanto, apenas retornar mesh original
        # TODO: Implementar refinamento básico usando normal maps
        return vertices, faces
    
    def project_textures(self,
                        vertices: torch.Tensor,
                        faces: torch.Tensor,
                        rgb_maps: torch.Tensor,
                        normal_maps: torch.Tensor,
                        azimuths: List[float],
                        elevations: List[float],
                        radius: float,
                        output_path: str) -> str:
        """
        Projeta texturas no mesh refinado
        
        Args:
            vertices: Vértices do mesh
            faces: Faces do mesh
            rgb_maps: RGB maps (4, 3, H, W)
            normal_maps: Normal maps (4, 3, H, W)
            azimuths: Azimutes
            elevations: Elevações
            radius: Raio
            output_path: Caminho de saída
        
        Returns:
            Caminho do mesh texturizado
        """
        print("[ISOMER] Projetando texturas...")
        
        try:
            from models.ISOMER.projection_func import projection
            from utils.tool import mask_fix, get_background
            
            device = self.device
            
            # Preparar máscaras
            masks = get_background(normal_maps).to(device)
            masks_proj = mask_fix(masks.squeeze(1), erode_dilate=-10, blur=5)
            
            # Preparar pesos
            color_weights = torch.tensor([1.0, 0.5, 1.0, 0.5], device=device)
            
            # Converter para formato esperado
            rgb_maps_permuted = rgb_maps.permute(0, 2, 3, 1)  # (4, H, W, 3)
            
            # Projetar
            save_paths = [
                output_path.replace('.obj', '.glb'),
                output_path
            ]
            
            mesh_path = projection(
                meshes=trimesh.Trimesh(
                    vertices=vertices.cpu().numpy(),
                    faces=faces.cpu().numpy()
                ),
                masks=masks_proj,
                images=rgb_maps_permuted,
                azimuths=torch.tensor(azimuths, device=device),
                elevations=torch.tensor(elevations, device=device),
                weights=color_weights,
                fov=30,
                radius=radius,
                save_dir=Path(output_path).parent,
                save_addrs=save_paths
            )
            
            print(f"[ISOMER] Mesh texturizado salvo: {mesh_path}")
            return mesh_path
            
        except ImportError:
            print("[ISOMER] Módulo de projeção não disponível - salvando mesh sem textura")
            # Salvar mesh simples
            mesh = trimesh.Trimesh(
                vertices=vertices.cpu().numpy(),
                faces=faces.cpu().numpy()
            )
            mesh.export(output_path)
            return output_path
        except Exception as e:
            print(f"[ISOMER] Erro na projeção: {e}")
            # Fallback: salvar mesh simples
            mesh = trimesh.Trimesh(
                vertices=vertices.cpu().numpy(),
                faces=faces.cpu().numpy()
            )
            mesh.export(output_path)
            return output_path

