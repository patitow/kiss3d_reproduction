"""
Gerador de normal maps a partir de imagens
"""

from typing import Optional
from PIL import Image
import numpy as np


class NormalMapGenerator:
    """
    Gera normal maps a partir de imagens usando modelos de predição de profundidade.
    
    Esta classe será implementada para usar MiDaS ou DPT para gerar
    normal maps a partir de imagens RGB.
    """
    
    def __init__(self, model_name: str = "DPT_Large"):
        """
        Inicializa o gerador de normal maps.
        
        Args:
            model_name: Nome do modelo de predição de profundidade
        """
        self.model_name = model_name
        # TODO: Carregar modelo de predição de profundidade
        
    def generate(self, image: Image.Image) -> np.ndarray:
        """
        Gera um normal map a partir de uma imagem.
        
        Args:
            image: Imagem RGB (PIL Image)
            
        Returns:
            Normal map como array numpy (H, W, 3) com valores normalizados
        """
        # TODO: Implementar geração de normal map
        raise NotImplementedError("Método generate() será implementado na Etapa 1")
    
    def generate_from_path(self, image_path: str) -> np.ndarray:
        """
        Gera um normal map a partir do caminho de uma imagem.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Normal map como array numpy
        """
        image = Image.open(image_path)
        return self.generate(image)


