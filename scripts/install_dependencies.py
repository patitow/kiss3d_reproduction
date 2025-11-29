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
    """Instala PyTorch3D com suporte CUDA"""
    print("\n[2/5] Instalando PyTorch3D com suporte CUDA...")
    
    # Instalar iopath primeiro (requisito do PyTorch3D)
    print("[INFO] Instalando iopath (requisito do PyTorch3D)...")
    run_command(f'{sys.executable} -m pip install iopath', "Instalação do iopath")
    
    # Verificar versão do PyTorch e CUDA
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        
        print(f"[INFO] PyTorch versão: {torch_version}")
        if cuda_version:
            print(f"[INFO] CUDA versão: {cuda_version}")
        else:
            print("[AVISO] CUDA não disponível - PyTorch3D será instalado sem GPU support")
    except ImportError:
        print("[AVISO] PyTorch não instalado - tentando instalar PyTorch3D mesmo assim")
        torch_version = None
        cuda_version = None
    
    # Tentar instalar wheel pré-compilado primeiro (mais rápido e confiável)
    # PyTorch 2.4.0 com CUDA 12.1
    if torch_version and cuda_version:
        print("[INFO] Tentando instalar PyTorch3D via wheel pré-compilado...")
        # Método oficial: usar URL direta do wheel
        # Para PyTorch 2.4.0 + CUDA 12.1 + Python 3.11
        wheel_url = "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html"
        # Instalar usando --no-index e --find-links
        cmd = f'{sys.executable} -m pip install --no-index --no-cache-dir pytorch3d --find-links {wheel_url}'
        result = run_command(cmd, "Instalação do PyTorch3D (wheel pré-compilado)")
        
        if result:
            # Verificar se tem GPU support
            try:
                import pytorch3d
                print(f"[INFO] PyTorch3D instalado: {pytorch3d.__version__}")
                print("[INFO] Verificando suporte CUDA do PyTorch3D...")
                # Tentar importar operações CUDA
                from pytorch3d import _C
                print("[OK] PyTorch3D instalado com suporte CUDA!")
                return True
            except Exception as e:
                print(f"[AVISO] PyTorch3D instalado mas suporte CUDA não confirmado: {e}")
                print("[INFO] Tentando instalar via git (pode demorar mais)...")
                # Desinstalar e tentar via git
                run_command(f'{sys.executable} -m pip uninstall pytorch3d -y', "Desinstalando PyTorch3D")
                result = False
        
        # Fallback: instalar via git se wheel falhar
        if not result:
            print("[INFO] Wheel pré-compilado não disponível, instalando via git...")
            print("[AVISO] Isso pode demorar vários minutos para compilar...")
            cmd = f'{sys.executable} -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"'
            result = run_command(cmd, "Instalação do PyTorch3D (via git)")
            return result
        return result
    else:
        # Sem CUDA ou PyTorch, instalar via git
        print("[INFO] Instalando PyTorch3D via git (pode demorar)...")
        cmd = f'{sys.executable} -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"'
        return run_command(cmd, "Instalação do PyTorch3D")

def install_nvdiffrast():
    """Instala nvdiffrast"""
    print("\n[3/5] Instalando nvdiffrast...")
    print("[INFO] Isso requer Visual Studio C++ Build Tools no Windows")
    cmd = f"{sys.executable} -m pip install git+https://github.com/NVlabs/nvdiffrast"
    return run_command(cmd, "Instalação do nvdiffrast")

def install_triton():
    """Instala Triton (biblioteca de otimização CUDA para PyTorch/xformers)"""
    print("\n[4/6] Instalando Triton...")
    print("[INFO] Triton é necessário para otimizações do xformers")
    print("[INFO] No Windows, usamos triton-windows (fork com suporte Windows)")
    
    # Verificar se já está instalado
    try:
        import triton
        print(f"[OK] Triton já instalado: {triton.__version__}")
        return True
    except ImportError:
        pass
    
    # Tentar instalar triton-windows (suporte Windows)
    print("[INFO] Instalando triton-windows...")
    cmd = f"{sys.executable} -m pip install triton-windows"
    result = run_command(cmd, "Instalação do Triton (triton-windows)")
    
    if result:
        # Verificar instalação
        try:
            import triton
            print(f"[OK] Triton instalado com sucesso: {triton.__version__}")
            return True
        except ImportError:
            print("[AVISO] Triton instalado mas não pode ser importado")
            return False
    return False

def install_custom_diffusers():
    """Instala diffusers customizado do Kiss3DGen"""
    print("\n[5/6] Instalando diffusers customizado...")
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
    print("\n[6/6] Verificando autenticação HuggingFace...")
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
        ("Triton", install_triton),
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
    
    # Verificar PyTorch3D CUDA support
    print("\n[7/7] Verificando PyTorch3D CUDA support...")
    pytorch3d_cuda_ok = False
    try:
        import pytorch3d
        print(f"[OK] PyTorch3D instalado: {pytorch3d.__version__}")
        # Tentar importar operações CUDA
        try:
            from pytorch3d import _C
            print("[OK] PyTorch3D compilado com suporte CUDA!")
            pytorch3d_cuda_ok = True
        except Exception as e:
            print(f"[AVISO] PyTorch3D instalado mas suporte CUDA não confirmado: {e}")
            print("[INFO] PyTorch3D pode funcionar em CPU, mas será mais lento")
            print("[INFO] Para instalar com CUDA, execute:")
            print("       pip uninstall pytorch3d")
            print("       pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html")
    except ImportError:
        print("[ERRO] PyTorch3D não instalado!")
    
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
        if not pytorch3d_cuda_ok:
            print("\n[AVISO] PyTorch3D pode não ter suporte CUDA completo.")
            print("[INFO] ISOMER pode falhar. Considere reinstalar PyTorch3D:")
            print("       pip uninstall pytorch3d")
            print("       pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html")
        print("\nPróximos passos:")
        print("1. Autenticar no HuggingFace: huggingface-cli login")
        print("2. Baixar modelos: python scripts/download_all_models.py")
        print("3. Executar pipeline: python scripts/run_kiss3dgen_image_to_3d_optimized.py --input <imagem>")
    else:
        print("\n[AVISO] Algumas dependências falharam. Verifique os erros acima.")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

