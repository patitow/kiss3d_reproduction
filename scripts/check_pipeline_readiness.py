#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar se tudo está pronto para executar o pipeline Image-to-3D
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"

def check_python_version():
    """Verifica versão do Python"""
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"[ERRO] Python {version.major}.{version.minor}.{version.micro} (requer 3.11)")
        return False

def check_pytorch():
    """Verifica PyTorch e CUDA"""
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA {torch.version.cuda} disponível")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[ERRO] CUDA não disponível")
            return False
    except ImportError:
        print("[ERRO] PyTorch não instalado")
        return False

def check_pytorch3d():
    """Verifica PyTorch3D"""
    try:
        import pytorch3d
        from pytorch3d import _C
        print(f"[OK] PyTorch3D {pytorch3d.__version__} com CUDA")
        return True
    except ImportError as e:
        print(f"[ERRO] PyTorch3D não funciona: {e}")
        return False

def check_dependencies():
    """Verifica dependências principais"""
    required = {
        "diffusers": ("diffusers", "Diffusers"),
        "transformers": ("transformers", "Transformers"),
        "opencv-python": ("cv2", "OpenCV"),
        "imageio": ("imageio", "ImageIO"),
        "trimesh": ("trimesh", "Trimesh"),
        "pymeshlab": ("pymeshlab", "PyMeshLab"),
        "numpy": ("numpy", "NumPy"),
        "pillow": ("PIL", "Pillow"),
    }
    
    missing = []
    for module_key, (import_name, display_name) in required.items():
        try:
            __import__(import_name)
            print(f"[OK] {display_name}")
        except ImportError:
            print(f"[ERRO] {display_name} não instalado")
            missing.append(module_key)
    
    return len(missing) == 0

def check_models():
    """Verifica se os modelos estão baixados"""
    checkpoint_dir = KISS3D_ROOT / "checkpoint"
    if not checkpoint_dir.exists():
        print("[ERRO] Diretório checkpoint/ não existe")
        return False
    
    required_models = {
        "zero123": ["zero123plus-v1.1", "diffusion_pytorch_model.safetensors"],
        "flux": ["flux1-schnell", "diffusion_pytorch_model.safetensors"],
        "controlnet": ["FLUX.1-dev-Controlnet-Union", "diffusion_pytorch_model.safetensors"],
        "redux": ["FLUX.1-Redux-dev", "diffusion_pytorch_model.safetensors"],
    }
    
    all_ok = True
    for model_name, (repo_name, key_file) in required_models.items():
        model_dir = checkpoint_dir / repo_name
        if model_dir.exists() and (model_dir / key_file).exists():
            print(f"[OK] {model_name}: {repo_name}")
        else:
            print(f"[ERRO] {model_name}: {repo_name} não encontrado")
            all_ok = False
    
    return all_ok

def check_huggingface_auth():
    """Verifica autenticação HuggingFace"""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"[OK] HuggingFace autenticado: {user.get('name', 'N/A')}")
        return True
    except Exception:
        print("[AVISO] HuggingFace não autenticado (pode ser necessário para alguns modelos)")
        return False

def check_config():
    """Verifica se o config existe"""
    config_path = KISS3D_ROOT / "pipeline" / "pipeline_config" / "default.yaml"
    if config_path.exists():
        print(f"[OK] Config encontrado: {config_path}")
        return True
    else:
        print(f"[ERRO] Config não encontrado: {config_path}")
        return False

def check_scripts():
    """Verifica se os scripts necessários existem"""
    required_scripts = [
        ("scripts/run_kiss3dgen_image_to_3d.py", "Script principal"),
        ("scripts/download_all_models.py", "Download de modelos"),
        ("scripts/kiss3d_wrapper_local.py", "Wrapper local"),
        ("scripts/kiss3d_utils_local.py", "Utils local"),
    ]
    
    all_ok = True
    for script, desc in required_scripts:
        script_path = PROJECT_ROOT / script
        if script_path.exists():
            print(f"[OK] {desc}: {script}")
        else:
            print(f"[ERRO] {desc} não encontrado: {script}")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("VERIFICAÇÃO DE PRONTIDÃO DO PIPELINE")
    print("="*60)
    print()
    
    checks = {
        "Python 3.11": check_python_version,
        "PyTorch + CUDA": check_pytorch,
        "PyTorch3D": check_pytorch3d,
        "Dependências": check_dependencies,
        "Modelos": check_models,
        "HuggingFace Auth": check_huggingface_auth,
        "Config": check_config,
        "Scripts": check_scripts,
    }
    
    results = {}
    for name, check_func in checks.items():
        print(f"\n[{name}]")
        results[name] = check_func()
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "OK" if passed else "FALTA"
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n[SUCESSO] Tudo pronto para executar o pipeline!")
        print("\nPara executar:")
        print("  python scripts/run_kiss3dgen_image_to_3d.py --input <imagem.jpg>")
    else:
        print("\n[ATENÇÃO] Alguns itens precisam ser corrigidos antes de executar.")
        print("\nAções recomendadas:")
        if not results.get("PyTorch3D", True):
            print("  1. Instalar PyTorch3D: python scripts/install_pytorch3d_robust.bat")
        if not results.get("Modelos", True):
            print("  2. Baixar modelos: python scripts/download_all_models.py")
        if not results.get("Dependências", True):
            print("  3. Instalar dependências: pip install -r requirements.txt")
        if not results.get("HuggingFace Auth", True):
            print("  4. Autenticar HuggingFace: huggingface-cli login")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

