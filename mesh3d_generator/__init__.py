"""
Mesh3D Generator - Geração de malhas 3D a partir de imagens e texto
"""

__version__ = "0.1.0"

from mesh3d_generator.text_to_image import TextToImageGenerator
from mesh3d_generator.normal_maps import NormalMapGenerator
from mesh3d_generator.mesh_initialization import LRMInitializer, InstantMeshInitializer
from mesh3d_generator.mesh_refinement import MeshRefiner
from mesh3d_generator.llm import TextGenerator

__all__ = [
    "TextToImageGenerator",
    "NormalMapGenerator",
    "LRMInitializer",
    "InstantMeshInitializer",
    "MeshRefiner",
    "TextGenerator",
]

