"""
Geração de 3D Bundle Image (múltiplas views) via ComfyUI
Baseado no pipeline do Kiss3DGen
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
import torchvision

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mesh3d_generator.comfyui.client import ComfyUIClient


class ComfyUIBundleImageGenerator:
    """
    Gera 3D Bundle Image (múltiplas views RGB + Normal maps).
    Similar ao generate_reference_3D_bundle_image_zero123 do Kiss3DGen.
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        """
        Inicializa o gerador de bundle images.
        
        Args:
            comfyui_url: URL do ComfyUI
        """
        self.client = ComfyUIClient(comfyui_url)
    
    def generate_bundle_image(self, 
                             input_image: Image.Image,
                             azimuths: List[float] = [270, 0, 90, 180],
                             elevations: List[float] = [5, 5, 5, 5],
                             use_mv_rgb: bool = True) -> Tuple[torch.Tensor, str]:
        """
        Gera 3D bundle image com múltiplas views.
        
        Args:
            input_image: Imagem de entrada (PIL Image)
            azimuths: Lista de azimutes em graus (4 views)
            elevations: Lista de elevações em graus (4 views)
            use_mv_rgb: Se True, usa multi-view RGB
        
        Returns:
            Tupla (bundle_image_tensor, save_path)
        """
        print(f"  [INFO] Gerando 3D bundle image (4 views)...")
        
        # Por enquanto, implementação placeholder
        # Em produção, isso usaria Zero123 ou similar via ComfyUI
        
        # Gerar 4 views RGB (placeholder - usar imagem original rotacionada)
        rgbs = []
        for i, (az, el) in enumerate(zip(azimuths, elevations)):
            # Placeholder: usar imagem original
            # Em produção, gerar view usando Zero123
            rgb = input_image.copy()
            rgbs.append(rgb)
        
        # Gerar 4 views Normal maps (placeholder)
        normals = []
        for i in range(4):
            # Placeholder: criar normal map simples
            normal = Image.new('RGB', input_image.size, (128, 128, 255))
            normals.append(normal)
        
        # Converter para tensores
        rgb_tensors = []
        for rgb in rgbs:
            rgb_array = np.array(rgb.resize((512, 512))).astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)  # (C, H, W)
            rgb_tensors.append(rgb_tensor)
        
        normal_tensors = []
        for normal in normals:
            normal_array = np.array(normal.resize((512, 512))).astype(np.float32) / 255.0
            normal_tensor = torch.from_numpy(normal_array).permute(2, 0, 1)  # (C, H, W)
            normal_tensors.append(normal_tensor)
        
        # Combinar em bundle image (grid 4x2: 4 RGB + 4 Normal)
        rgbs_tensor = torch.stack(rgb_tensors, dim=0)  # (4, C, H, W)
        normals_tensor = torch.stack(normal_tensors, dim=0)  # (4, C, H, W)
        
        # Criar grid
        bundle_image = torchvision.utils.make_grid(
            torch.cat([rgbs_tensor, normals_tensor], dim=0),
            nrow=4,
            padding=0
        )
        
        print(f"  [OK] Bundle image gerado: {bundle_image.shape}")
        
        # Salvar (placeholder)
        save_path = "/tmp/bundle_image.png"
        
        return bundle_image, save_path

