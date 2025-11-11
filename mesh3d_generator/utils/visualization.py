"""
Utilitários para visualização de resultados
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import trimesh


class Visualizer:
    """
    Utilitários para visualizar imagens, normal maps e malhas 3D.
    """
    
    @staticmethod
    def visualize_image(image: Image.Image, title: str = "Image"):
        """Visualiza uma imagem."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def visualize_normal_map(normal_map: np.ndarray, title: str = "Normal Map"):
        """Visualiza um normal map."""
        plt.figure(figsize=(10, 10))
        plt.imshow((normal_map + 1) / 2)  # Normalizar para [0, 1]
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def visualize_mesh(mesh: trimesh.Trimesh, title: str = "Mesh"):
        """Visualiza uma malha 3D."""
        scene = mesh.scene()
        scene.show(title=title)


