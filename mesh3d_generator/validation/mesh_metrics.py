"""
Métricas de validação para modelos 3D
Implementa Chamfer Distance, Hausdorff Distance, SSIM para texturas, etc.
"""

import numpy as np
import trimesh
from typing import Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io


def chamfer_distance(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, 
                    num_samples: int = 10000) -> float:
    """
    Calcula Chamfer Distance entre duas meshes.
    
    Chamfer Distance = média das distâncias mínimas de cada ponto de uma mesh
    para a outra mesh.
    
    Args:
        mesh1: Primeira mesh
        mesh2: Segunda mesh
        num_samples: Número de pontos para amostrar de cada mesh
    
    Returns:
        Chamfer Distance (quanto menor, mais similar)
    """
    # Amostrar pontos das superfícies
    points1, _ = trimesh.sample.sample_surface(mesh1, num_samples)
    points2, _ = trimesh.sample.sample_surface(mesh2, num_samples)
    
    # Calcular distâncias mínimas de points1 para mesh2
    distances1_to_2 = mesh2.nearest.on_surface(points1)[1]
    
    # Calcular distâncias mínimas de points2 para mesh1
    distances2_to_1 = mesh1.nearest.on_surface(points2)[1]
    
    # Chamfer Distance = média das duas direções
    chamfer = (np.mean(distances1_to_2) + np.mean(distances2_to_1)) / 2.0
    
    return float(chamfer)


def hausdorff_distance(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                      num_samples: int = 10000) -> float:
    """
    Calcula Hausdorff Distance entre duas meshes.
    
    Hausdorff Distance = maior distância mínima entre dois conjuntos de pontos.
    
    Args:
        mesh1: Primeira mesh
        mesh2: Segunda mesh
        num_samples: Número de pontos para amostrar de cada mesh
    
    Returns:
        Hausdorff Distance (quanto menor, mais similar)
    """
    # Amostrar pontos das superfícies
    points1, _ = trimesh.sample.sample_surface(mesh1, num_samples)
    points2, _ = trimesh.sample.sample_surface(mesh2, num_samples)
    
    # Calcular distâncias mínimas de points1 para mesh2
    distances1_to_2 = mesh2.nearest.on_surface(points1)[1]
    
    # Calcular distâncias mínimas de points2 para mesh1
    distances2_to_1 = mesh1.nearest.on_surface(points2)[1]
    
    # Hausdorff Distance = máximo das duas direções
    hausdorff = max(np.max(distances1_to_2), np.max(distances2_to_1))
    
    return float(hausdorff)


def texture_ssim(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                resolution: int = 512) -> Optional[float]:
    """
    Calcula SSIM (Structural Similarity Index) entre texturas de duas meshes.
    
    Renderiza as meshes de múltiplas views e compara as texturas usando SSIM.
    
    Args:
        mesh1: Primeira mesh
        mesh2: Segunda mesh
        resolution: Resolução das imagens renderizadas
    
    Returns:
        SSIM médio (0-1, quanto maior, mais similar) ou None se não houver texturas
    """
    # Verificar se ambas têm texturas
    has_texture1 = (hasattr(mesh1.visual, 'material') and 
                   mesh1.visual.material is not None and
                   hasattr(mesh1.visual.material, 'image') and
                   mesh1.visual.material.image is not None)
    
    has_texture2 = (hasattr(mesh2.visual, 'material') and 
                   mesh2.visual.material is not None and
                   hasattr(mesh2.visual.material, 'image') and
                   mesh2.visual.material.image is not None)
    
    if not (has_texture1 and has_texture2):
        return None
    
    # Extrair texturas
    texture1 = mesh1.visual.material.image
    texture2 = mesh2.visual.material.image
    
    # Converter para numpy arrays se necessário
    if isinstance(texture1, Image.Image):
        texture1 = np.array(texture1)
    if isinstance(texture2, Image.Image):
        texture2 = np.array(texture2)
    
    # Redimensionar para mesma resolução
    if texture1.shape != texture2.shape:
        from PIL import Image
        img1 = Image.fromarray(texture1)
        img2 = Image.fromarray(texture2)
        target_size = (resolution, resolution)
        img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
        img2 = img2.resize(target_size, Image.Resampling.LANCZOS)
        texture1 = np.array(img1)
        texture2 = np.array(img2)
    
    # Converter para escala de cinza se necessário
    if len(texture1.shape) == 3:
        # Usar luminância: 0.299*R + 0.587*G + 0.114*B
        texture1 = np.dot(texture1[...,:3], [0.299, 0.587, 0.114])
    if len(texture2.shape) == 3:
        texture2 = np.dot(texture2[...,:3], [0.299, 0.587, 0.114])
    
    # Normalizar para [0, 1]
    texture1 = texture1.astype(np.float32) / 255.0
    texture2 = texture2.astype(np.float32) / 255.0
    
    # Calcular SSIM
    ssim_value = ssim(texture1, texture2, data_range=1.0)
    
    return float(ssim_value)


def validate_mesh_quality(mesh: trimesh.Trimesh) -> Dict:
    """
    Valida qualidade de uma mesh.
    
    Args:
        mesh: Mesh para validar
    
    Returns:
        Dicionário com métricas de qualidade
    """
    metrics = {
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
        'volume': float(mesh.volume) if mesh.is_watertight else None,
        'surface_area': float(mesh.area),
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'num_components': len(mesh.split(only_watertight=False)),
        'has_holes': not mesh.is_watertight,
    }
    
    # Verificar se há faces degeneradas
    try:
        nondegenerate = mesh.nondegenerate_faces()
        metrics['num_degenerate_faces'] = len(mesh.faces) - len(nondegenerate)
    except:
        metrics['num_degenerate_faces'] = None
    
    # Verificar se há vértices não referenciados
    try:
        referenced = mesh.referenced_vertices()
        metrics['num_unreferenced_vertices'] = len(mesh.vertices) - len(referenced)
    except:
        metrics['num_unreferenced_vertices'] = None
    
    return metrics


def compare_meshes_comprehensive(original: trimesh.Trimesh, 
                                 generated: trimesh.Trimesh,
                                 num_samples: int = 10000) -> Dict:
    """
    Comparação abrangente entre duas meshes usando múltiplas métricas.
    
    Args:
        original: Mesh original
        generated: Mesh gerada
        num_samples: Número de pontos para amostragem
    
    Returns:
        Dicionário com todas as métricas de comparação
    """
    metrics = {}
    
    # Métricas básicas
    metrics['original_vertices'] = len(original.vertices)
    metrics['generated_vertices'] = len(generated.vertices)
    metrics['original_faces'] = len(original.faces)
    metrics['generated_faces'] = len(generated.faces)
    
    # Volumes
    metrics['original_volume'] = float(original.volume) if original.is_watertight else None
    metrics['generated_volume'] = float(generated.volume) if generated.is_watertight else None
    
    if metrics['original_volume'] and metrics['generated_volume']:
        volume_diff = abs(metrics['original_volume'] - metrics['generated_volume'])
        metrics['volume_difference'] = volume_diff
        metrics['volume_similarity'] = 1.0 - (volume_diff / max(metrics['original_volume'], 0.001))
    else:
        metrics['volume_difference'] = None
        metrics['volume_similarity'] = None
    
    # Bounds
    metrics['original_bounds'] = original.bounds.tolist()
    metrics['generated_bounds'] = generated.bounds.tolist()
    
    # Chamfer Distance
    try:
        metrics['chamfer_distance'] = chamfer_distance(original, generated, num_samples)
    except Exception as e:
        metrics['chamfer_distance'] = None
        metrics['chamfer_distance_error'] = str(e)
    
    # Hausdorff Distance
    try:
        metrics['hausdorff_distance'] = hausdorff_distance(original, generated, num_samples)
    except Exception as e:
        metrics['hausdorff_distance'] = None
        metrics['hausdorff_distance_error'] = str(e)
    
    # SSIM para texturas
    try:
        metrics['texture_ssim'] = texture_ssim(original, generated)
    except Exception as e:
        metrics['texture_ssim'] = None
        metrics['texture_ssim_error'] = str(e)
    
    # Validação de qualidade
    metrics['original_quality'] = validate_mesh_quality(original)
    metrics['generated_quality'] = validate_mesh_quality(generated)
    
    return metrics

