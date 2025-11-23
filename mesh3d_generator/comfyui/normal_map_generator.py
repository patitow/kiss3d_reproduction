"""
Geração de normal maps via ComfyUI
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mesh3d_generator.comfyui.client import ComfyUIClient


class ComfyUINormalMapGenerator:
    """Gera normal maps usando ComfyUI"""
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188",
                 workflow_path: Optional[str] = None):
        """
        Inicializa o gerador de normal maps.
        
        Args:
            comfyui_url: URL do ComfyUI
            workflow_path: Caminho para workflow JSON (opcional)
        """
        self.client = ComfyUIClient(comfyui_url)
        
        if workflow_path is None:
            # Usar workflow padrão
            workflow_path = Path(__file__).parent.parent.parent / "comfyui-test" / "workflow_mesh3d.json"
        
        self.workflow_path = Path(workflow_path)
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"Workflow nao encontrado: {workflow_path}")
    
    def generate(self, image_path: str, 
                 output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gera normal map a partir de imagem.
        
        Args:
            image_path: Caminho para imagem de entrada
            output_path: Caminho para salvar normal map (opcional)
        
        Returns:
            Normal map como numpy array (H, W, 3) ou None se falhar
        """
        print(f"  [INFO] Gerando normal map via ComfyUI...")
        
        try:
            # Upload imagem
            image_name = self.client.upload_image(image_path)
            
            # Carregar workflow
            workflow = self.client.load_workflow(str(self.workflow_path))
            
            # Atualizar workflow com imagem
            workflow = self.client.update_workflow_image(workflow, image_name)
            
            # Enviar para fila
            prompt_id = self.client.queue_prompt(workflow)
            print(f"  [INFO] Prompt enviado: {prompt_id}")
            
            # Aguardar conclusão
            if not self.client.wait_for_completion(prompt_id, timeout=300):
                print(f"  [ERRO] Timeout ao gerar normal map")
                return None
            
            # Obter resultado
            history = self.client.get_history(prompt_id)
            if history is None:
                print(f"  [ERRO] Nao foi possivel obter historico")
                return None
            
            # Encontrar node SaveImage que salva o normal map
            # Por padrão, o workflow salva em "normal_map"
            # Precisamos encontrar o nome do arquivo gerado
            # Por enquanto, retornar None e implementar depois
            print(f"  [AVISO] Obtencao de normal map ainda nao implementada completamente")
            return None
            
        except Exception as e:
            print(f"  [ERRO] Erro ao gerar normal map: {e}")
            import traceback
            traceback.print_exc()
            return None

