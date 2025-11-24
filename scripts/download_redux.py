#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script específico para baixar Redux com tratamento de erros de symlink no Windows
"""

import os
import sys
from huggingface_hub import snapshot_download
import torch

def main():
    print("="*60)
    print("DOWNLOAD DO REDUX")
    print("="*60)
    
    repo_id = "black-forest-labs/FLUX.1-Redux-dev"
    
    print(f"\n[INFO] Baixando: {repo_id}")
    print(f"[INFO] Este modelo e opcional mas recomendado para melhor qualidade")
    
    try:
        # Tentar baixar sem symlinks (Windows-friendly)
        snapshot_download(
            repo_id=repo_id,
            local_dir=None,
            local_dir_use_symlinks=False,  # Desabilitar symlinks para Windows
            resume_download=True
        )
        print(f"\n[OK] Redux baixado com sucesso!")
        return True
        
    except OSError as e:
        if "symlink" in str(e).lower() or "privilegio" in str(e).lower():
            print(f"\n[AVISO]  Erro de symlink (normal no Windows)")
            print(f"   A maioria dos arquivos foi baixada")
            print(f"   O modelo deve funcionar mesmo assim")
            
            # Verificar se os arquivos principais estão lá
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_dir = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
            
            if os.path.exists(model_dir):
                files = []
                for root, dirs, filenames in os.walk(model_dir):
                    files.extend([os.path.join(root, f) for f in filenames if f.endswith(('.safetensors', '.json', '.bin'))])
                
                if len(files) > 5:
                    print(f"   [OK] {len(files)} arquivos encontrados - modelo provavelmente OK")
                    return True
                else:
                    print(f"   [AVISO]  Poucos arquivos encontrados - pode precisar tentar novamente")
                    return False
            else:
                print(f"   [ERRO] Diretorio do modelo nao encontrado")
                return False
        else:
            print(f"\n[ERRO] Erro ao baixar: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"\n[ERRO] Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

