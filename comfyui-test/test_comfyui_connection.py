#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simples para testar a conexão com o ComfyUI e validar a API.

Desenvolvido para o projeto Mesh3D Generator - Visão Computacional 2025.2
Autor: Auto (Cursor AI Assistant)
Data: 2025
"""

import requests
import json
import sys
import os
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def test_comfyui_connection(comfyui_url: str = "http://127.0.0.1:8188"):
    """Testa se o ComfyUI está rodando e acessível."""
    print("[INFO] Testando conexao com ComfyUI...")
    
    try:
        response = requests.get(f"{comfyui_url}/system_stats", timeout=5)
        if response.status_code == 200:
            print("[OK] ComfyUI esta rodando e acessivel!")
            stats = response.json()
            print(f"   - Versao: {stats.get('version', 'N/A')}")
            return True
        else:
            print(f"[ERRO] ComfyUI respondeu com status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERRO] Nao foi possivel conectar ao ComfyUI.")
        print(f"   Certifique-se de que o ComfyUI esta rodando em {comfyui_url}")
        print("   Execute: cd ComfyUI && python main.py")
        return False
    except Exception as e:
        print(f"[ERRO] Erro ao testar conexao: {e}")
        return False


def test_workflow_loading(workflow_path: str):
    """Testa se o workflow JSON é válido."""
    print(f"\n[INFO] Testando carregamento do workflow: {workflow_path}...")
    
    if not Path(workflow_path).exists():
        print(f"[ERRO] Arquivo nao encontrado: {workflow_path}")
        return False
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # Validações básicas
        if 'nodes' not in workflow:
            print("[ERRO] Workflow nao contem 'nodes'")
            return False
        
        if 'links' not in workflow:
            print("[ERRO] Workflow nao contem 'links'")
            return False
        
        print(f"[OK] Workflow valido!")
        print(f"   - Numero de nodes: {len(workflow['nodes'])}")
        print(f"   - Numero de links: {len(workflow['links'])}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"[ERRO] Erro ao parsear JSON: {e}")
        return False
    except Exception as e:
        print(f"[ERRO] Erro ao carregar workflow: {e}")
        return False


def test_ollama_connection():
    """Testa se o Ollama está rodando (opcional)."""
    print("\n[INFO] Testando conexao com Ollama (opcional)...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("[OK] Ollama esta rodando!")
            print(f"   - Modelos instalados: {len(models)}")
            model_names = [m.get('name', 'N/A') for m in models]
            if 'llava' in str(model_names):
                print("   - [OK] Modelo 'llava' encontrado")
            else:
                print("   - [AVISO] Modelo 'llava' nao encontrado")
                print("   - Execute: ollama pull llava")
            return True
        else:
            print(f"[AVISO] Ollama respondeu com status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[AVISO] Ollama nao esta rodando (opcional para testes basicos)")
        print("   Para usar LLM, execute: ollama serve")
        return False
    except Exception as e:
        print(f"[AVISO] Erro ao testar Ollama: {e}")
        return False


def main():
    """Executa todos os testes."""
    print("=" * 60)
    print("[TEST] Teste de Conexao - ComfyUI Mesh3D Generator")
    print("=" * 60)
    
    # Teste 1: Conexão com ComfyUI
    comfyui_ok = test_comfyui_connection()
    
    # Teste 2: Workflows
    workflows = ["workflow_simple.json", "workflow_mesh3d.json"]
    workflows_ok = True
    for workflow in workflows:
        if Path(workflow).exists():
            if not test_workflow_loading(workflow):
                workflows_ok = False
        else:
            print(f"\n⚠️  Workflow não encontrado: {workflow}")
    
    # Teste 3: Ollama (opcional)
    ollama_ok = test_ollama_connection()
    
    # Resumo
    print("\n" + "=" * 60)
    print("[RESUMO] Resumo dos Testes")
    print("=" * 60)
    print(f"ComfyUI:        {'[OK]' if comfyui_ok else '[FALHOU]'}")
    print(f"Workflows:      {'[OK]' if workflows_ok else '[FALHOU]'}")
    print(f"Ollama:         {'[OK]' if ollama_ok else '[OPCIONAL]'}")
    
    if comfyui_ok and workflows_ok:
        print("\n[OK] Pronto para executar workflows!")
        print("\nProximos passos:")
        print("1. Coloque uma imagem de teste em ComfyUI/input/")
        print("2. Execute: python integrate_llm.py --image path/to/image.jpg")
        print("   OU")
        print("3. Abra o ComfyUI no navegador e carregue o workflow manualmente")
        return 0
    else:
        print("\n[ERRO] Alguns testes falharam. Corrija os problemas acima.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

