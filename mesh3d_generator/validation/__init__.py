"""
Módulo de validação para modelos 3D
"""

from .mesh_metrics import (
    chamfer_distance,
    hausdorff_distance,
    texture_ssim,
    validate_mesh_quality,
    compare_meshes_comprehensive
)

__all__ = [
    'chamfer_distance',
    'hausdorff_distance',
    'texture_ssim',
    'validate_mesh_quality',
    'compare_meshes_comprehensive'
]

