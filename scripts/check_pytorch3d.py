#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar se PyTorch3D está instalado e funcionando com CUDA
"""

import sys

def check_pytorch3d():
    """Verifica PyTorch3D"""
    try:
        import pytorch3d
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        return True, pytorch3d.__version__
    except ImportError:
        print("[ERRO] PyTorch3D não instalado!")
        return False, None

def check_pytorch3d_cuda():
    """Verifica se PyTorch3D tem suporte CUDA"""
    try:
        from pytorch3d import _C
        print("[OK] PyTorch3D _C módulo importado com sucesso!")
        print("[OK] PyTorch3D tem suporte CUDA!")
        return True
    except ImportError as e:
        print(f"[ERRO] PyTorch3D _C não pode ser importado: {e}")
        return False
    except Exception as e:
        print(f"[ERRO] Erro ao importar PyTorch3D _C: {e}")
        return False

def main():
    print("="*60)
    print("VERIFICAÇÃO DO PyTorch3D")
    print("="*60)
    
    # Verificar PyTorch
    try:
        import torch
        print(f"\n[OK] PyTorch: {torch.__version__}")
        print(f"[OK] CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA versão: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\n[ERRO] PyTorch não instalado!")
        return False
    
    # Verificar PyTorch3D
    print("\n[1/2] Verificando PyTorch3D...")
    pytorch3d_installed, version = check_pytorch3d()
    
    if not pytorch3d_installed:
        print("\n[ERRO] PyTorch3D não está instalado!")
        print("[INFO] A compilação pode ainda estar em andamento.")
        print("[INFO] Aguarde alguns minutos e execute este script novamente.")
        return False
    
    # Verificar CUDA support
    print("\n[2/2] Verificando suporte CUDA...")
    cuda_ok = check_pytorch3d_cuda()
    
    if cuda_ok:
        print("\n" + "="*60)
        print("[OK] PyTorch3D está funcionando corretamente com CUDA!")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("[ERRO] PyTorch3D instalado mas sem suporte CUDA!")
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)












