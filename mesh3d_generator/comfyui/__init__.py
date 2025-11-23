"""
Módulo de integração com ComfyUI
"""

from mesh3d_generator.comfyui.client import ComfyUIClient
from mesh3d_generator.comfyui.normal_map_generator import ComfyUINormalMapGenerator
from mesh3d_generator.comfyui.bundle_image_generator import ComfyUIBundleImageGenerator
from mesh3d_generator.comfyui.mesh_generator import ComfyUIMeshGenerator

__all__ = [
    'ComfyUIClient',
    'ComfyUINormalMapGenerator',
    'ComfyUIBundleImageGenerator',
    'ComfyUIMeshGenerator'
]

