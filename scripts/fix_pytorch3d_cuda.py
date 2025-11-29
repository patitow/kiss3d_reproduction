#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir problemas de instalação do PyTorch3D
"""
import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Verifica o ambiente"""
    print("="*60)
    print("DIAGNOSTICO DO AMBIENTE")
    print("="*60)
    
    # PyTorch
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"[OK] CUDA disponivel: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[ERRO] PyTorch nao encontrado")
        return False
    
    # CUDA Toolkit
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    ]
    cuda_found = False
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"[OK] CUDA Toolkit encontrado: {path}")
            cuda_found = True
            break
    if not cuda_found:
        print("[AVISO] CUDA Toolkit nao encontrado nos caminhos padrao")
    
    # Visual Studio
    vs_paths = [
        (r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools", "VS 2019 BuildTools"),
        (r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community", "VS 2019 Community"),
        (r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools", "VS 2022 BuildTools"),
    ]
    vs_found = False
    for path, name in vs_paths:
        if os.path.exists(path):
            print(f"[OK] {name} encontrado: {path}")
            vs_found = True
            break
    if not vs_found:
        print("[ERRO] Visual Studio nao encontrado")
        return False
    
    # PyTorch3D
    try:
        import pytorch3d
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        print(f"[OK] Localizacao: {pytorch3d.__file__}")
        
        # Tentar importar _C
        try:
            from pytorch3d import _C
            print("[OK] Modulo _C importado com sucesso!")
            return True
        except ImportError as e:
            print(f"[ERRO] Falha ao importar _C: {e}")
            return False
    except ImportError:
        print("[INFO] PyTorch3D nao instalado")
        return False

def try_install_from_wheel():
    """Tenta instalar de um wheel pré-compilado"""
    print("\n" + "="*60)
    print("TENTANDO INSTALAR DE WHEEL PRE-COMPILADO")
    print("="*60)
    
    # Desinstalar versão atual
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "pytorch3d", "-y"], 
                   capture_output=True)
    
    # Tentar instalar
    result = subprocess.run([sys.executable, "-m", "pip", "install", "pytorch3d", "--no-cache-dir"],
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            import pytorch3d
            from pytorch3d import _C
            print("[OK] Instalacao via wheel bem-sucedida!")
            return True
        except:
            print("[ERRO] Wheel instalado mas _C nao funciona")
            return False
    else:
        print("[ERRO] Falha ao instalar wheel")
        print(result.stderr)
        return False

def main():
    if check_environment():
        print("\n[OK] PyTorch3D esta funcionando corretamente!")
        return 0
    
    print("\n[INFO] PyTorch3D nao esta funcionando. Tentando corrigir...")
    
    # Tentar instalar de wheel primeiro
    if try_install_from_wheel():
        if check_environment():
            print("\n[SUCESSO] Problema resolvido!")
            return 0
    
    print("\n[ERRO] Nao foi possivel instalar via wheel.")
    print("[INFO] E necessario compilar do codigo-fonte.")
    print("[INFO] Execute: scripts\\install_pytorch3d_robust.bat")
    return 1

if __name__ == "__main__":
    sys.exit(main())
