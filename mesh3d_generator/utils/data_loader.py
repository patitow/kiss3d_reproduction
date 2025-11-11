"""
Utilitários para carregamento de dados do dataset do Google Research
"""

from typing import List, Optional
from pathlib import Path
import json


class DataLoader:
    """
    Carrega dados do dataset do Google Research (Gazebo).
    """
    
    def __init__(self, dataset_path: str = "data/raw"):
        """
        Inicializa o carregador de dados.
        
        Args:
            dataset_path: Caminho para o dataset
        """
        self.dataset_path = Path(dataset_path)
        
    def load_image(self, image_id: str) -> Optional[Path]:
        """
        Carrega caminho de uma imagem do dataset.
        
        Args:
            image_id: ID da imagem
            
        Returns:
            Caminho para a imagem ou None se não encontrada
        """
        # TODO: Implementar carregamento de imagens do dataset
        image_path = self.dataset_path / f"{image_id}.jpg"
        if image_path.exists():
            return image_path
        return None
    
    def load_metadata(self, image_id: str) -> Optional[dict]:
        """
        Carrega metadados de uma imagem.
        
        Args:
            image_id: ID da imagem
            
        Returns:
            Dicionário com metadados ou None
        """
        # TODO: Implementar carregamento de metadados
        metadata_path = self.dataset_path / f"{image_id}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None


