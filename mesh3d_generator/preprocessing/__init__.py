"""
Módulo de preprocessamento de imagens
"""

from mesh3d_generator.preprocessing.image_preprocessor import (
    preprocess_input_image,
    remove_background,
    resize_foreground,
    to_rgb_image
)

__all__ = [
    'preprocess_input_image',
    'remove_background',
    'resize_foreground',
    'to_rgb_image'
]


