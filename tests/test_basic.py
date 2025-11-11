"""
Testes básicos para o projeto Mesh3D Generator
"""

import pytest
from pathlib import Path
import sys

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh3d_generator import (
    TextToImageGenerator,
    NormalMapGenerator,
    InstantMeshInitializer,
    LRMInitializer,
    MeshRefiner,
    TextGenerator
)


def test_imports():
    """Testa se todos os módulos podem ser importados."""
    assert TextToImageGenerator is not None
    assert NormalMapGenerator is not None
    assert LRMInitializer is not None
    assert InstantMeshInitializer is not None
    assert MeshRefiner is not None
    assert TextGenerator is not None


def test_text_to_image_initialization():
    """Testa inicialização do gerador de imagens."""
    generator = TextToImageGenerator()
    assert generator.model_name is not None
    assert generator.device is not None


def test_normal_map_initialization():
    """Testa inicialização do gerador de normal maps."""
    generator = NormalMapGenerator()
    assert generator.model_name is not None


def test_text_generator_initialization():
    """Testa inicialização do gerador de texto."""
    generator = TextGenerator()
    assert generator.model_name is not None


if __name__ == "__main__":
    pytest.main([__file__])

