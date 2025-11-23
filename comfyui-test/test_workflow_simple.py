#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar o workflow simples do ComfyUI sem LLM.
Útil para validar que o ComfyUI está funcionando corretamente.

Desenvolvido para o projeto Mesh3D Generator - Visão Computacional 2025.2
Autor: Auto (Cursor AI Assistant)
Data: 2025
"""

import argparse
import json
import requests
import sys
import io
from pathlib import Path
from typing import Optional

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_workflow(workflow_path: str) -> dict:
    """Carrega workflow do ComfyUI."""
    with open(workflow_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_image_in_workflow(workflow: dict, image_filename: str) -> dict:
    """Atualiza o nome da imagem no workflow."""
    for node in workflow.get('nodes', []):
        if node.get('type') == 'LoadImage':
            node['widgets_values'][0] = image_filename
            print(f"[OK] Imagem atualizada: {image_filename}")
            break
    return workflow


def upload_image(comfyui_url: str, image_path: str) -> Optional[str]:
    """Faz upload da imagem para o ComfyUI."""
    print(f"[INFO] Fazendo upload da imagem: {image_path}...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'overwrite': 'true'}
            
            response = requests.post(
                f"{comfyui_url}/upload/image",
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            filename = response.json()['name']
            print(f"[OK] Imagem enviada: {filename}")
            return filename
            
    except Exception as e:
        print(f"[ERRO] Erro ao fazer upload: {e}")
        return None


def queue_prompt(comfyui_url: str, workflow: dict) -> Optional[str]:
    """Envia workflow para o ComfyUI."""
    print("[INFO] Enviando workflow para ComfyUI...")
    
    prompt = {
        "prompt": workflow,
        "client_id": "test_workflow"
    }
    
    try:
        response = requests.post(
            f"{comfyui_url}/prompt",
            json=prompt,
            timeout=10
        )
        response.raise_for_status()
        
        prompt_id = response.json()['prompt_id']
        print(f"[OK] Workflow enviado! Prompt ID: {prompt_id}")
        print(f"   Acompanhe o progresso em: {comfyui_url}")
        return prompt_id
        
    except Exception as e:
        print(f"[ERRO] Erro ao enviar workflow: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Resposta: {e.response.text}")
        return None


def check_queue_status(comfyui_url: str) -> dict:
    """Verifica o status da fila do ComfyUI."""
    try:
        response = requests.get(f"{comfyui_url}/queue", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[AVISO] Erro ao verificar fila: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Testa workflow simples do ComfyUI sem LLM"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Caminho para a imagem de teste'
    )
    parser.add_argument(
        '--workflow',
        type=str,
        default='workflow_simple.json',
        help='Workflow a usar (padrão: workflow_simple.json)'
    )
    parser.add_argument(
        '--comfyui-url',
        type=str,
        default='http://127.0.0.1:8188',
        help='URL do ComfyUI'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='a detailed 3D scene, high quality, detailed geometry, realistic textures',
        help='Prompt de texto (padrão: prompt genérico)'
    )
    
    args = parser.parse_args()
    
    # Validar imagem
    if not Path(args.image).exists():
        print(f"[ERRO] Imagem nao encontrada: {args.image}")
        return 1
    
    # Validar workflow
    if not Path(args.workflow).exists():
        print(f"[ERRO] Workflow nao encontrado: {args.workflow}")
        return 1
    
    print("=" * 60)
    print("[TEST] Teste de Workflow Simples - ComfyUI")
    print("=" * 60)
    
    # 1. Carregar workflow
    print(f"\n[INFO] Carregando workflow: {args.workflow}")
    try:
        workflow = load_workflow(args.workflow)
        print("[OK] Workflow carregado")
    except Exception as e:
        print(f"[ERRO] Erro ao carregar workflow: {e}")
        return 1
    
    # 2. Fazer upload da imagem
    uploaded_filename = upload_image(args.comfyui_url, args.image)
    if not uploaded_filename:
        return 1
    
    # 3. Atualizar workflow
    workflow = update_image_in_workflow(workflow, uploaded_filename)
    
    # 4. Atualizar prompt (se necessário)
    for node in workflow.get('nodes', []):
        if node.get('type') == 'CLIPTextEncode' and node.get('id') == 2:
            node['widgets_values'][0] = args.prompt
            print(f"[OK] Prompt atualizado: {args.prompt[:50]}...")
            break
    
    # 5. Verificar status da fila
    queue_status = check_queue_status(args.comfyui_url)
    if queue_status:
        running = len(queue_status.get('queue_running', []))
        pending = len(queue_status.get('queue_pending', []))
        print(f"\n[INFO] Status da fila: {running} rodando, {pending} pendente")
    
    # 6. Enviar workflow
    prompt_id = queue_prompt(args.comfyui_url, workflow)
    if not prompt_id:
        return 1
    
    print("\n" + "=" * 60)
    print("[OK] Teste concluido!")
    print("=" * 60)
    print(f"\n[INFO] Proximos passos:")
    print(f"1. Acompanhe o progresso em: {args.comfyui_url}")
    print(f"2. As imagens serao salvas em: ComfyUI/output/")
    print(f"3. Prompt ID: {prompt_id}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

