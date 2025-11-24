#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para instalar todas as dependências do projeto Kiss3DGen
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Verifica se está usando Python 3.11.9"""
    version = sys.version_info
    print(f"[INFO] Python {version.major}.{version.minor}.{version.patch}")
    
    if version.major != 3:
        print("[ERRO] Python 3 é obrigatório")
        return False
    
    if version.minor != 11:
        print("[ERRO] Python 3.11.9 é OBRIGATÓRIO")
        print("[INFO] Instale Python 3.11.9: https://www.python.org/downloads/release/python-3119/")
        print("[INFO] Crie ambiente virtual: python3.11 -m venv mesh3d-generator-py3.11")
        return False
    
    if version.patch < 9:
        print("[AVISO] Python 3.11.9 é recomendado (você está usando 3.11.{})".format(version.patch))
        print("[INFO] Considere atualizar para 3.11.9")
    
    if version.patch == 9:
        print("[OK] Python 3.11.9 detectado - versão correta!")
        return True
    
    if version.patch > 9:
        print("[AVISO] Python 3.11.9 é o recomendado (você está usando 3.11.{})".format(version.patch))
        print("[INFO] Pode funcionar, mas 3.11.9 é o testado")
    
    return True

def run_command(cmd, description):
    """Executa um comando e retorna True se sucesso"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Executando: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} concluído")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Falha em: {description}")
        print(f"Erro: {e.stderr}")
        return False

def check_cuda():
    """Verifica se CUDA está disponível"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA disponível: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[AVISO] CUDA não disponível - instale PyTorch com CUDA")
            return False
    except ImportError:
        print("[AVISO] PyTorch não instalado ainda")
        return False

def install_from_requirements():
    """Instala pacotes do requirements.txt"""
    req_file = Path(__file__).parent.parent / "requirements.txt"
    if not req_file.exists():
        print(f"[ERRO] requirements.txt não encontrado: {req_file}")
        return False
    
    # Instalar pacotes básicos primeiro
    print("\n[1/5] Instalando pacotes básicos do requirements.txt...")
    cmd = f"{sys.executable} -m pip install -r {req_file} --upgrade"
    return run_command(cmd, "Instalação de requirements.txt")

def install_pytorch3d():
    """Instala PyTorch3D"""
    print("\n[2/5] Instalando PyTorch3D...")
    cmd = f'{sys.executable} -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"'
    return run_command(cmd, "Instalação do PyTorch3D")

def install_nvdiffrast():
    """Instala nvdiffrast"""
    print("\n[3/5] Instalando nvdiffrast...")
    print("[INFO] Isso requer Visual Studio C++ Build Tools no Windows")
    cmd = f"{sys.executable} -m pip install git+https://github.com/NVlabs/nvdiffrast"
    return run_command(cmd, "Instalação do nvdiffrast")

def install_custom_diffusers():
    """Instala diffusers customizado do Kiss3DGen"""
    print("\n[4/5] Instalando diffusers customizado...")
    kiss3dgen_path = Path(__file__).parent.parent / "Kiss3DGen" / "custom_diffusers"
    if not kiss3dgen_path.exists():
        print(f"[AVISO] custom_diffusers não encontrado: {kiss3dgen_path}")
        return False
    
    os.chdir(str(kiss3dgen_path))
    cmd = f"{sys.executable} -m pip install -e ."
    result = run_command(cmd, "Instalação do diffusers customizado")
    os.chdir(str(Path(__file__).parent.parent))
    return result

def check_huggingface_auth():
    """Verifica autenticação HuggingFace"""
    print("\n[5/5] Verificando autenticação HuggingFace...")
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"[OK] Autenticado como: {user.get('name', 'N/A')}")
        return True
    except Exception:
        print("[AVISO] Não autenticado no HuggingFace")
        print("[INFO] Execute: huggingface-cli login")
        print("[INFO] Ou execute: python scripts/setup_huggingface_auth.py")
        return False

def main():
    """Instala todas as dependências"""
    print("="*60)
    print("INSTALAÇÃO DE DEPENDÊNCIAS - Kiss3DGen")
    print("="*60)
    
    # Verificar Python
    if not check_python_version():
        print("\n[ERRO] Python 3.11.9 é obrigatório!")
        print("[INFO] Execute primeiro: python scripts/setup_python311.bat")
        print("[INFO] Ou: python scripts/setup_python311.ps1")
        return False
    
    # Verificar CUDA
    check_cuda()
    
    # Instalar dependências
    steps = [
        ("requirements.txt", install_from_requirements),
        ("PyTorch3D", install_pytorch3d),
        ("nvdiffrast", install_nvdiffrast),
        ("custom_diffusers", install_custom_diffusers),
    ]
    
    results = {}
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as e:
            print(f"[ERRO] Erro em {name}: {e}")
            results[name] = False
    
    # Verificar HuggingFace
    check_huggingface_auth()
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DA INSTALAÇÃO")
    print("="*60)
    
    for name, success in results.items():
        status = "[OK]" if success else "[ERRO]"
        print(f"{status} {name}")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n[OK] Todas as dependências foram instaladas!")
        print("\nPróximos passos:")
        print("1. Autenticar no HuggingFace: huggingface-cli login")
        print("2. Baixar modelos: python scripts/download_models.py")
        print("3. Executar pipeline: python scripts/run_kiss3dgen_image_to_3d.py --input <imagem>")
    else:
        print("\n[AVISO] Algumas dependências falharam. Verifique os erros acima.")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

