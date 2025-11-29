#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para baixar o ControlNet do Zero123++ (controlnet-zp12-normal-gen-v1)
Este é um modelo opcional que melhora a geração de normals
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"
CONTROLNET_DIR = KISS3D_ROOT / "models" / "zero123plus_controlnet"

def main():
    """Baixa o ControlNet do Zero123++"""
    print("="*70)
    print("DOWNLOAD DO CONTROLNET ZERO123++")
    print("="*70)
    print("Modelo: sudo-ai/controlnet-zp12-normal-gen-v1")
    print("Descrição: ControlNet opcional para geração de normals no Zero123++")
    print("="*70)
    
    # Verificar CUDA
    print(f"\n[INFO] CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    # Verificar se já existe
    if CONTROLNET_DIR.exists() and any(CONTROLNET_DIR.iterdir()):
        print(f"\n[INFO] ControlNet já existe em: {CONTROLNET_DIR}")
        print("[INFO] Para baixar novamente, delete o diretório primeiro.")
        return True
    
    # Criar diretório
    CONTROLNET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Baixando ControlNet para: {CONTROLNET_DIR}")
    print("[INFO] Isso pode demorar alguns minutos...")
    
    try:
        snapshot_download(
            repo_id="sudo-ai/controlnet-zp12-normal-gen-v1",
            local_dir=str(CONTROLNET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n[OK] ControlNet baixado com sucesso!")
        print(f"[INFO] Localização: {CONTROLNET_DIR}")
        print("\n[INFO] Este ControlNet é opcional e pode ser usado para melhorar")
        print("       a geração de normals no Zero123++. O pipeline funcionará")
        print("       sem ele, mas a qualidade pode ser melhor com ele.")
        return True
    except Exception as e:
        print(f"\n[ERRO] Falha ao baixar ControlNet: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

