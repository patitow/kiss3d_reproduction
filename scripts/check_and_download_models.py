#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar e baixar modelos necessários para o pipeline
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# Modelos necessários
REQUIRED_MODELS = {
    "zero123": {
        "repo_id": "sudo-ai/zero123plus-v1.1",  # Corrigido: v1.1 é o correto
        "description": "Zero123++ para geracao de multiview",
        "required": True
    },
    "flux": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "description": "Flux diffusion model",
        "required": True
    },
    "controlnet": {
        "repo_id": "InstantX/FLUX.1-dev-Controlnet-Union",
        "description": "ControlNet para Flux",
        "required": True
    },
    "redux": {
        "repo_id": "black-forest-labs/FLUX.1-Redux-dev",
        "description": "Flux Prior Redux",
        "required": False  # Opcional
    },
    "lrm": {
        "repo_id": "LTT/PRM",
        "description": "LRM (Large Reconstruction Model)",
        "required": False  # Pode usar alternativa
    }
}

def check_model_downloaded(repo_id: str) -> bool:
    """Verifica se um modelo esta baixado"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir_name)
    return os.path.exists(model_path) and len(os.listdir(model_path)) > 0

def download_model(repo_id: str, description: str, required: bool = True):
    """Baixa um modelo do HuggingFace"""
    print(f"\n{'='*60}")
    print(f"Modelo: {repo_id}")
    print(f"Descricao: {description}")
    print(f"{'='*60}")
    
    if check_model_downloaded(repo_id):
        print(f"[OK] Modelo ja esta baixado!")
        return True
    
    if not required:
        print(f"[AVISO]  Modelo opcional - pulando download")
        return False
    
    print(f"📥 Baixando modelo...")
    print(f"   Isso pode demorar varios minutos e requer espaco em disco...")
    
    try:
        # Baixar modelo completo
        snapshot_download(
            repo_id=repo_id,
            local_dir=None,  # Usar cache padrão
            local_dir_use_symlinks=False,  # Windows não suporta symlinks bem
            resume_download=True
        )
        print(f"[OK] Modelo baixado com sucesso!")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao baixar modelo: {e}")
        if "disk" in str(e).lower() or "space" in str(e).lower():
            print(f"   [AVISO]  Espaco em disco insuficiente!")
        return False

def main():
    """Verifica e baixa modelos necessarios"""
    print("="*60)
    print("VERIFICACAO E DOWNLOAD DE MODELOS")
    print("="*60)
    
    print(f"\n[INFO] CUDA disponivel: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\n[INFO] Verificando modelos necessarios...")
    
    # Verificar status de todos os modelos
    status = {}
    for name, info in REQUIRED_MODELS.items():
        repo_id = info["repo_id"]
        is_downloaded = check_model_downloaded(repo_id)
        status[name] = {
            "downloaded": is_downloaded,
            "info": info
        }
        status_icon = "[OK]" if is_downloaded else "[ERRO]"
        required_text = "(Obrigatorio)" if info["required"] else "(Opcional)"
        print(f"{status_icon} {name}: {repo_id} {required_text}")
    
    # Perguntar se quer baixar modelos faltantes
    missing_required = [name for name, s in status.items() 
                       if not s["downloaded"] and s["info"]["required"]]
    
    if not missing_required:
        print(f"\n[OK] Todos os modelos obrigatorios estao baixados!")
        return True
    
    print(f"\n[AVISO]  Modelos obrigatorios faltando: {len(missing_required)}")
    for name in missing_required:
        print(f"   - {name}: {status[name]['info']['repo_id']}")
    
    print(f"\n[INFO] Para baixar modelos, execute:")
    print(f"   python scripts/download_models.py")
    print(f"\n   Ou baixe manualmente usando:")
    for name in missing_required:
        repo_id = status[name]["info"]["repo_id"]
        print(f"   - {name}: {repo_id}")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

