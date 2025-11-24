"""
Renderização de normal maps a partir de mesh 3D
Implementação própria baseada na abordagem do Kiss3DGen
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import torchvision


class NormalMapRenderer:
    """
    Renderiza normal maps a partir de mesh 3D em diferentes views
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        Inicializa o renderizador de normal maps
        
        Args:
            device: Dispositivo para processamento
        """
        self.device = device
        print(f"[NORMAL_RENDERER] Inicializado com device: {device}")
    
    def render_normal_maps(self,
                          vertices: torch.Tensor,
                          faces: torch.Tensor,
                          azimuths: List[float] = [270, 0, 90, 180],
                          elevations: List[float] = [5, 5, 5, 5],
                          radius: float = 4.5,
                          render_size: int = 512,
                          fov: float = 30.0) -> torch.Tensor:
        """
        Renderiza normal maps do mesh em diferentes views
        
        Args:
            vertices: Vértices do mesh (N, 3)
            faces: Faces do mesh (M, 3)
            azimuths: Lista de azimutes em graus
            elevations: Lista de elevações em graus
            radius: Raio da câmera
            render_size: Tamanho da imagem renderizada
            fov: Campo de visão em graus
        
        Returns:
            Tensor de normal maps (4, 3, H, W) em range [0, 1]
        """
        print(f"[NORMAL_RENDERER] Renderizando {len(azimuths)} normal maps...")
        
        try:
            # Tentar usar renderizador avançado (pytorch3d, nvdiffrast, etc)
            return self._render_with_pytorch3d(
                vertices, faces, azimuths, elevations, radius, render_size, fov
            )
        except ImportError as e:
            print(f"[NORMAL_RENDERER] Pytorch3D não disponível - usando método simples")
            print(f"[NORMAL_RENDERER] Nota: Pytorch3D é opcional. Para instalar, veja INSTALL_PYTORCH3D.md")
            return self._render_simple(vertices, faces, azimuths, elevations, radius, render_size)
        except Exception as e:
            print(f"[NORMAL_RENDERER] Erro no renderizador avançado: {e}")
            print(f"[NORMAL_RENDERER] Usando método simples como fallback")
            return self._render_simple(vertices, faces, azimuths, elevations, radius, render_size)
    
    def _render_with_pytorch3d(self,
                               vertices: torch.Tensor,
                               faces: torch.Tensor,
                               azimuths: List[float],
                               elevations: List[float],
                               radius: float,
                               render_size: int,
                               fov: float) -> torch.Tensor:
        """Renderiza usando Pytorch3D (opcional)"""
        try:
            # Tentar importar pytorch3d
            try:
                from pytorch3d.renderer import (
                    PerspectiveCameras,
                    RasterizationSettings,
                    MeshRenderer,
                    MeshRasterizer,
                    HardPhongShader,
                    TexturesVertex,
                    look_at_view_transform
                )
                from pytorch3d.structures import Meshes
            except ImportError:
                raise ImportError("Pytorch3D não está instalado. É opcional - usando fallback.")
            
            # Converter para device
            vertices = vertices.to(self.device).float()
            faces = faces.to(self.device).long()
            
            # Criar mesh
            # Calcular normais dos vértices
            normals = self._compute_vertex_normals(vertices, faces)
            
            # Criar textura com normais (usar normais como cores)
            textures = TexturesVertex(verts_features=normals.unsqueeze(0))
            mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
            
            # Renderizar para cada view
            normal_maps = []
            
            for azimuth, elevation in zip(azimuths, elevations):
                # Calcular transformação de câmera
                R, T = look_at_view_transform(
                    dist=radius,
                    elev=elevation,
                    azim=azimuth,
                    at=((0, 0, 0),)
                )
                
                cameras = PerspectiveCameras(
                    R=R,
                    T=T,
                    fov=fov,
                    device=self.device
                )
                
                # Configurar rasterização
                raster_settings = RasterizationSettings(
                    image_size=render_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
                
                # Criar renderizador
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings
                    ),
                    shader=HardPhongShader(device=self.device)
                )
                
                # Renderizar
                image = renderer(mesh)
                
                # Extrair normal map (RGB do output)
                normal_map = image[0, ..., :3].permute(2, 0, 1)  # (3, H, W)
                
                # Normalizar para [0, 1] se necessário
                if normal_map.min() < 0:
                    normal_map = (normal_map + 1) / 2
                
                normal_maps.append(normal_map)
            
            # Stack em tensor (4, 3, H, W)
            normal_maps_tensor = torch.stack(normal_maps, dim=0)
            
            print(f"[NORMAL_RENDERER] Normal maps renderizados: {normal_maps_tensor.shape}")
            return normal_maps_tensor
            
        except ImportError:
            raise ImportError("Pytorch3D não disponível")
    
    def _render_simple(self,
                      vertices: torch.Tensor,
                      faces: torch.Tensor,
                      azimuths: List[float],
                      elevations: List[float],
                      radius: float,
                      render_size: int) -> torch.Tensor:
        """Renderiza usando método simples (placeholder)"""
        print("[NORMAL_RENDERER] Usando renderização simples (placeholder)")
        
        # Calcular normais dos vértices
        normals = self._compute_vertex_normals(vertices, faces)
        
        # Normalizar normais para [0, 1]
        normals_normalized = (normals + 1) / 2
        
        # Para cada view, criar normal map simples
        # Por enquanto, usar normais médias do mesh
        normal_maps = []
        for _ in azimuths:
            # Placeholder: usar normais médias
            normal_map = normals_normalized.mean(dim=0).unsqueeze(1).unsqueeze(2)
            normal_map = normal_map.expand(3, render_size, render_size)
            normal_maps.append(normal_map)
        
        return torch.stack(normal_maps, dim=0)
    
    def _compute_vertex_normals(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Calcula normais dos vértices a partir das faces"""
        # Converter para numpy para cálculo
        vertices_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy()
        
        # Calcular normais das faces
        face_normals = []
        for face in faces_np:
            v0, v1, v2 = vertices_np[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            face_normals.append(normal)
        
        face_normals = np.array(face_normals)
        
        # Calcular normais dos vértices (média das normais das faces adjacentes)
        vertex_normals = np.zeros_like(vertices_np)
        for i, face in enumerate(faces_np):
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_normals[i]
        
        # Normalizar
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vertex_normals = vertex_normals / norms
        
        return torch.from_numpy(vertex_normals).float()

