#!/usr/bin/env python3
"""
Script para integrar LLM (Ollama) com o workflow do ComfyUI
Gera descrição detalhada da imagem usando LLM multimodal e envia para o ComfyUI
"""

import argparse
import json
import requests
import base64
from pathlib import Path
from typing import Optional
import ollama


class ComfyUILLMIntegrator:
    """Integra LLM (Ollama) com workflow do ComfyUI"""
    
    def __init__(self, comfyui_api_url: str = "http://127.0.0.1:8188"):
        """
        Inicializa o integrador.
        
        Args:
            comfyui_api_url: URL da API do ComfyUI
        """
        self.comfyui_api_url = comfyui_api_url
        self.ollama_client = ollama.Client()
        
    def generate_detailed_description(self, image_path: str, 
                                     model: str = "llava",
                                     prompt: Optional[str] = None) -> str:
        """
        Gera descrição detalhada da imagem usando LLM multimodal.
        
        Args:
            image_path: Caminho para a imagem
            model: Nome do modelo Ollama a usar
            prompt: Prompt customizado (opcional)
            
        Returns:
            Descrição detalhada da cena
        """
        if prompt is None:
            prompt = """Analyze this image in extreme detail and provide a comprehensive description 
            that would be useful for 3D mesh generation. Focus on:
            - Geometric shapes and structures
            - Surface details and textures
            - Lighting and shadows
            - Depth and perspective
            - Material properties
            - Spatial relationships
            
            Provide a detailed, technical description suitable for 3D reconstruction."""
        
        print(f"📸 Analisando imagem: {image_path}")
        print(f"🤖 Usando modelo: {model}")
        
        try:
            # Ler imagem
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Enviar para Ollama
            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                images=[image_data]
            )
            
            description = response['response']
            print(f"✅ Descrição gerada ({len(description)} caracteres)")
            return description
            
        except Exception as e:
            print(f"❌ Erro ao gerar descrição: {e}")
            raise
    
    def load_workflow(self, workflow_path: str) -> dict:
        """Carrega workflow do ComfyUI"""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_workflow_with_text(self, workflow: dict, text: str) -> dict:
        """
        Atualiza o workflow com o texto gerado pelo LLM.
        
        Args:
            workflow: Workflow do ComfyUI
            text: Texto gerado pelo LLM
            
        Returns:
            Workflow atualizado
        """
        # Encontrar node de texto positivo (CLIPTextEncode)
        for node in workflow.get('nodes', []):
            if node.get('type') == 'CLIPTextEncode' and node.get('id') == 2:
                # Atualizar prompt com texto detalhado
                enhanced_prompt = f"{text}, high quality, detailed geometry, realistic textures, 3D mesh generation"
                node['widgets_values'][0] = enhanced_prompt
                print(f"📝 Prompt atualizado no workflow")
                break
        
        return workflow
    
    def update_workflow_with_image(self, workflow: dict, image_path: str) -> dict:
        """
        Atualiza o workflow com o caminho da imagem.
        
        Args:
            workflow: Workflow do ComfyUI
            image_path: Caminho para a imagem
            
        Returns:
            Workflow atualizado
        """
        # Encontrar node LoadImage
        for node in workflow.get('nodes', []):
            if node.get('type') == 'LoadImage':
                # Atualizar caminho da imagem
                node['widgets_values'][0] = str(Path(image_path).name)
                node['widgets_values'][1] = "image"
                print(f"🖼️  Imagem atualizada no workflow: {image_path}")
                break
        
        return workflow
    
    def queue_prompt(self, workflow: dict) -> str:
        """
        Envia workflow para o ComfyUI via API.
        
        Args:
            workflow: Workflow do ComfyUI
            
        Returns:
            Prompt ID
        """
        prompt = {
            "prompt": workflow,
            "client_id": "mesh3d_generator"
        }
        
        try:
            response = requests.post(
                f"{self.comfyui_api_url}/prompt",
                json=prompt
            )
            response.raise_for_status()
            
            prompt_id = response.json()['prompt_id']
            print(f"✅ Workflow enviado para ComfyUI (Prompt ID: {prompt_id})")
            return prompt_id
            
        except Exception as e:
            print(f"❌ Erro ao enviar workflow: {e}")
            raise
    
    def upload_image(self, image_path: str) -> str:
        """
        Faz upload da imagem para o ComfyUI.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Nome do arquivo no ComfyUI
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'overwrite': 'true'}
                
                response = requests.post(
                    f"{self.comfyui_api_url}/upload/image",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
                filename = response.json()['name']
                print(f"✅ Imagem enviada: {filename}")
                return filename
                
        except Exception as e:
            print(f"❌ Erro ao fazer upload da imagem: {e}")
            raise
    
    def run_complete_pipeline(self, image_path: str, 
                             workflow_path: str = "workflow_mesh3d.json",
                             ollama_model: str = "llava",
                             send_to_comfyui: bool = True) -> dict:
        """
        Executa o pipeline completo: LLM -> Workflow -> ComfyUI
        
        Args:
            image_path: Caminho para a imagem
            workflow_path: Caminho para o workflow JSON
            ollama_model: Modelo Ollama a usar
            send_to_comfyui: Se True, envia para ComfyUI. Se False, apenas gera texto
            
        Returns:
            Dicionário com resultados
        """
        print("🚀 Iniciando pipeline completo...\n")
        
        # 1. Gerar descrição com LLM
        description = self.generate_detailed_description(image_path, ollama_model)
        print(f"\n📄 Descrição gerada:\n{description}\n")
        
        if not send_to_comfyui:
            return {
                'description': description,
                'image_path': image_path
            }
        
        # 2. Carregar workflow
        workflow = self.load_workflow(workflow_path)
        
        # 3. Fazer upload da imagem
        uploaded_filename = self.upload_image(image_path)
        
        # 4. Atualizar workflow
        workflow = self.update_workflow_with_text(workflow, description)
        workflow = self.update_workflow_with_image(workflow, uploaded_filename)
        
        # 5. Enviar para ComfyUI
        prompt_id = self.queue_prompt(workflow)
        
        return {
            'description': description,
            'image_path': image_path,
            'uploaded_filename': uploaded_filename,
            'prompt_id': prompt_id,
            'workflow': workflow
        }


def main():
    parser = argparse.ArgumentParser(
        description="Integra LLM (Ollama) com workflow do ComfyUI para geração de malhas 3D"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Caminho para a imagem de entrada'
    )
    parser.add_argument(
        '--workflow',
        type=str,
        default='workflow_mesh3d.json',
        help='Caminho para o workflow JSON (padrão: workflow_mesh3d.json)'
    )
    parser.add_argument(
        '--ollama-model',
        type=str,
        default='llava',
        help='Modelo Ollama a usar (padrão: llava)'
    )
    parser.add_argument(
        '--comfyui-url',
        type=str,
        default='http://127.0.0.1:8188',
        help='URL da API do ComfyUI (padrão: http://127.0.0.1:8188)'
    )
    parser.add_argument(
        '--no-send',
        action='store_true',
        help='Apenas gerar descrição, não enviar para ComfyUI'
    )
    
    args = parser.parse_args()
    
    # Verificar se imagem existe
    if not Path(args.image).exists():
        print(f"❌ Erro: Imagem não encontrada: {args.image}")
        return
    
    # Verificar se workflow existe
    if not Path(args.workflow).exists():
        print(f"❌ Erro: Workflow não encontrado: {args.workflow}")
        return
    
    # Criar integrador
    integrator = ComfyUILLMIntegrator(comfyui_api_url=args.comfyui_url)
    
    # Executar pipeline
    try:
        results = integrator.run_complete_pipeline(
            image_path=args.image,
            workflow_path=args.workflow,
            ollama_model=args.ollama_model,
            send_to_comfyui=not args.no_send
        )
        
        print("\n✅ Pipeline concluído com sucesso!")
        if 'prompt_id' in results:
            print(f"📊 Acompanhe o progresso no ComfyUI (Prompt ID: {results['prompt_id']})")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

