"""
Cliente para interagir com ComfyUI via API
"""

import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
import base64


class ComfyUIClient:
    """Cliente para interagir com ComfyUI via API"""
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        """
        Inicializa o cliente ComfyUI.
        
        Args:
            comfyui_url: URL do servidor ComfyUI
        """
        self.comfyui_url = comfyui_url
        self.client_id = str(time.time())
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Verifica se o ComfyUI esta rodando"""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            if response.status_code == 200:
                return True
            else:
                print(f"[AVISO] ComfyUI respondeu com status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"[AVISO] Nao foi possivel conectar ao ComfyUI em {self.comfyui_url}")
            print("   Certifique-se de que o ComfyUI esta rodando")
            return False
        except Exception as e:
            print(f"[AVISO] Erro ao verificar conexao: {e}")
            return False
    
    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Envia prompt para a fila do ComfyUI.
        
        Args:
            prompt: Dicionário com o workflow/prompt
        
        Returns:
            ID do prompt na fila
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = requests.post(f"{self.comfyui_url}/prompt", data=data)
        req.raise_for_status()
        return req.json()['prompt_id']
    
    def upload_image(self, image_path: str, subfolder: str = "input", 
                    overwrite: bool = True) -> str:
        """
        Faz upload de imagem para o ComfyUI.
        
        Args:
            image_path: Caminho para a imagem
            subfolder: Subpasta no ComfyUI
            overwrite: Se True, sobrescreve arquivo existente
        
        Returns:
            Nome do arquivo no ComfyUI
        """
        with open(image_path, 'rb') as f:
            files = {"image": f}
            data = {"subfolder": subfolder, "type": "input", "overwrite": str(overwrite).lower()}
            response = requests.post(f"{self.comfyui_url}/upload/image", files=files, data=data)
            response.raise_for_status()
            return response.json()['name']
    
    def get_image(self, filename: str, subfolder: str, 
                 type: str = "output") -> bytes:
        """
        Baixa imagem gerada pelo ComfyUI.
        
        Args:
            filename: Nome do arquivo
            subfolder: Subpasta
            type: Tipo (input/output)
        
        Returns:
            Bytes da imagem
        """
        data = {"filename": filename, "subfolder": subfolder, "type": type}
        response = requests.get(f"{self.comfyui_url}/view", params=data)
        response.raise_for_status()
        return response.content
    
    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """
        Obtém histórico de um prompt.
        
        Args:
            prompt_id: ID do prompt
        
        Returns:
            Histórico ou None se não encontrado
        """
        response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
        return None
    
    def wait_for_completion(self, prompt_id: str, 
                           timeout: int = 300,
                           check_interval: float = 1.0) -> bool:
        """
        Aguarda conclusão do prompt.
        
        Args:
            prompt_id: ID do prompt
            timeout: Timeout em segundos
            check_interval: Intervalo entre verificações
        
        Returns:
            True se completou, False se timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history is not None:
                return True
            time.sleep(check_interval)
        return False
    
    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """
        Carrega workflow JSON.
        
        Args:
            workflow_path: Caminho para o arquivo workflow
        
        Returns:
            Dicionário com o workflow
        """
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_workflow_image(self, workflow: Dict[str, Any], 
                             image_name: str, 
                             node_id: int = 1) -> Dict[str, Any]:
        """
        Atualiza workflow com nome da imagem.
        
        Args:
            workflow: Workflow JSON
            image_name: Nome da imagem no ComfyUI
            node_id: ID do node LoadImage
        
        Returns:
            Workflow atualizado
        """
        for node in workflow.get('nodes', []):
            if node.get('type') == 'LoadImage' and node.get('id') == node_id:
                node['widgets_values'][0] = image_name
        return workflow
    
    def update_workflow_text(self, workflow: Dict[str, Any],
                            text: str,
                            node_id: int = 2) -> Dict[str, Any]:
        """
        Atualiza workflow com texto.
        
        Args:
            workflow: Workflow JSON
            text: Texto para o prompt
            node_id: ID do node CLIPTextEncode
        
        Returns:
            Workflow atualizado
        """
        for node in workflow.get('nodes', []):
            if node.get('type') == 'CLIPTextEncode' and node.get('id') == node_id:
                node['widgets_values'][0] = text
        return workflow


