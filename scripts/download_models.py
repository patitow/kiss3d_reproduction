#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para baixar modelos necessários para o pipeline
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

# Modelos necessários
REQUIRED_MODELS = {
    "zero123": {
        "repo_id": "sudo-ai/zero123plus-v1.2",
        "description": "Zero123++ para geração de multiview",
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
        "required": False
    }
}

def check_model_downloaded(repo_id: str) -> bool:
    """Verifica se um modelo está baixado"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir_name)
    return os.path.exists(model_path) and len(os.listdir(model_path)) > 0

def download_model(repo_id: str, description: str, required: bool = True):
    """Baixa um modelo do HuggingFace"""
    print(f"\n{'='*60}")
    print(f"Modelo: {repo_id}")
    print(f"Descrição: {description}")
    print(f"{'='*60}")
    
    if check_model_downloaded(repo_id):
        print(f"✅ Modelo já está baixado!")
        return True
    
    if not required:
        print(f"⚠️  Modelo opcional - pulando download")
        return False
    
    print(f"📥 Baixando modelo...")
    print(f"   Isso pode demorar vários minutos e requer espaço em disco...")
    
    try:
        # Baixar modelo completo
        snapshot_download(
            repo_id=repo_id,
            local_dir=None,  # Usar cache padrão
            local_dir_use_symlinks=False,  # Windows não suporta symlinks bem
            resume_download=True
        )
        print(f"✅ Modelo baixado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Baixa modelos necessários"""
    print("="*60)
    print("DOWNLOAD DE MODELOS PARA PIPELINE 3D")
    print("="*60)
    
    print(f"\n[INFO] CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\n[INFO] Verificando modelos...")
    
    # Verificar e baixar modelos obrigatórios
    results = {}
    for name, info in REQUIRED_MODELS.items():
        repo_id = info["repo_id"]
        description = info["description"]
        required = info["required"]
        
        if check_model_downloaded(repo_id):
            print(f"\n✅ {name} já está baixado")
            results[name] = True
            continue
        
        if not required:
            print(f"\n⚠️  {name} é opcional - pulando")
            results[name] = None
            continue
        
        print(f"\n📥 Baixando {name}...")
        success = download_model(repo_id, description, required)
        results[name] = success
    
    # Resumo
    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")
    
    for name, success in results.items():
        if success is True:
            print(f"✅ {name}: Baixado")
        elif success is False:
            print(f"❌ {name}: Falha no download")
        else:
            print(f"⚠️  {name}: Opcional (não baixado)")
    
    all_required = all(
        results.get(name) is True 
        for name, info in REQUIRED_MODELS.items() 
        if info["required"]
    )
    
    if all_required:
        print(f"\n✅ Todos os modelos obrigatórios estão prontos!")
        return True
    else:
        print(f"\n⚠️  Alguns modelos obrigatórios não foram baixados")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

