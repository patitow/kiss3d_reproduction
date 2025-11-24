#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script específico para baixar LRM (Large Reconstruction Model)
"""

import os
import sys
from huggingface_hub import snapshot_download
import torch

def main():
    print("="*60)
    print("DOWNLOAD DO LRM (Large Reconstruction Model)")
    print("="*60)
    
    repo_id = "LTT/PRM"
    
    print(f"\n[INFO] Baixando: {repo_id}")
    print(f"[INFO] Este modelo e usado para reconstrucao inicial de mesh 3D")
    
    try:
        # Tentar baixar sem symlinks (Windows-friendly)
        snapshot_download(
            repo_id=repo_id,
            local_dir=None,
            local_dir_use_symlinks=False,  # Desabilitar symlinks para Windows
            resume_download=True
        )
        print(f"\n[OK] LRM baixado com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n[ERRO] Erro ao baixar: {e}")
        
        # Verificar se o modelo existe ou se precisa de autenticação
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print(f"\n[AVISO]  Modelo pode requerer autenticacao no HuggingFace")
            print(f"   Tente autenticar primeiro: huggingface-cli login")
        elif "404" in str(e) or "not found" in str(e).lower():
            print(f"\n[AVISO]  Modelo nao encontrado no HuggingFace")
            print(f"   Verifique se o repo_id esta correto: {repo_id}")
        else:
            import traceback
            traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

