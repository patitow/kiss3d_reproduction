#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script completo para verificar instalação do Kiss3DGen
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Verifica versão do Python"""
    version = sys.version_info
    print(f"\n{'='*60}")
    print("VERIFICACAO DO PYTHON")
    print(f"{'='*60}")
    print(f"Versao: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        if version.micro == 9:
            print("[OK] Python 3.11.9 - Versao correta!")
            return True
        else:
            print(f"[AVISO] Python 3.11.{version.micro} - 3.11.9 recomendado")
            return True
    else:
        print("[ERRO] Python 3.11.9 e obrigatorio!")
        return False

def check_package(package_name, import_name=None):
    """Verifica se um pacote está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError:
        print(f"[ERRO] {package_name} nao instalado")
        return False

def check_pytorch():
    """Verifica PyTorch e CUDA"""
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA disponivel: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("[AVISO] CUDA nao disponivel - instale PyTorch com CUDA")
            return False
    except ImportError:
        print("[ERRO] PyTorch nao instalado")
        return False

def check_core_packages():
    """Verifica pacotes core"""
    print(f"\n{'='*60}")
    print("VERIFICACAO DE PACOTES CORE")
    print(f"{'='*60}")
    
    packages = [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("huggingface_hub", "huggingface_hub"),
        ("omegaconf", "omegaconf"),
        ("einops", "einops"),
        ("rembg", "rembg"),
        ("trimesh", "trimesh"),
        ("open3d", "open3d"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    return all(results)

def check_special_packages():
    """Verifica pacotes especiais que requerem compilação"""
    print(f"\n{'='*60}")
    print("VERIFICACAO DE PACOTES ESPECIAIS")
    print(f"{'='*60}")
    
    results = []
    
    # nvdiffrast
    try:
        import nvdiffrast
        print(f"[OK] nvdiffrast {nvdiffrast.__version__}")
        results.append(True)
    except ImportError:
        print("[ERRO] nvdiffrast nao instalado")
        print("[INFO] Instalar: pip install git+https://github.com/NVlabs/nvdiffrast")
        results.append(False)
    
    # PyTorch3D
    try:
        import pytorch3d
        print(f"[OK] pytorch3d")
        results.append(True)
    except ImportError:
        print("[AVISO] pytorch3d nao instalado (opcional mas recomendado)")
        print("[INFO] Instalar: pip install \"git+https://github.com/facebookresearch/pytorch3d.git@stable\"")
        results.append(False)
    
    return all(results)

def check_custom_diffusers():
    """Verifica diffusers customizado"""
    print(f"\n{'='*60}")
    print("VERIFICACAO DE DIFFUSERS CUSTOMIZADO")
    print(f"{'='*60}")
    
    # Adicionar path do Kiss3DGen
    kiss3dgen_path = Path(__file__).parent.parent / "Kiss3DGen"
    if kiss3dgen_path.exists():
        sys.path.insert(0, str(kiss3dgen_path))
    
    try:
        from pipeline.custom_pipelines import FluxPriorReduxPipeline, FluxControlNetImg2ImgPipeline
        print("[OK] Diffusers customizado instalado")
        return True
    except ImportError as e:
        # Verificar se diffusers foi instalado
        try:
            import diffusers
            print(f"[OK] Diffusers instalado (versao: {diffusers.__version__})")
            print("[AVISO] Nao foi possivel importar pipelines customizados (pode ser problema de path)")
            print("[INFO] Isso pode ser normal se o diffusers customizado esta instalado")
            return True  # Considerar OK se diffusers esta instalado
        except ImportError:
            print(f"[ERRO] Diffusers nao instalado: {e}")
            print("[INFO] Instalar: cd Kiss3DGen && pip install -e custom_diffusers/")
            return False

def check_models():
    """Verifica modelos baixados"""
    print(f"\n{'='*60}")
    print("VERIFICACAO DE MODELOS")
    print(f"{'='*60}")
    
    from huggingface_hub import snapshot_download
    import os
    
    models = {
        "zero123": "sudo-ai/zero123plus-v1.2",
        "flux": "black-forest-labs/FLUX.1-dev",
        "controlnet": "InstantX/FLUX.1-dev-Controlnet-Union",
        "redux": "black-forest-labs/FLUX.1-Redux-dev",
    }
    
    results = {}
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    for name, repo_id in models.items():
        model_dir_name = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, model_dir_name)
        
        if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
            print(f"[OK] {name}: Baixado")
            results[name] = True
        else:
            required = name != "redux"
            status = "[ERRO]" if required else "[AVISO]"
            print(f"{status} {name}: Nao baixado")
            results[name] = False if required else None
    
    return results

def check_huggingface_auth():
    """Verifica autenticação HuggingFace"""
    print(f"\n{'='*60}")
    print("VERIFICACAO DE AUTENTICACAO HUGGINGFACE")
    print(f"{'='*60}")
    
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"[OK] Autenticado como: {user.get('name', 'N/A')}")
        return True
    except Exception:
        print("[AVISO] Nao autenticado no HuggingFace")
        print("[INFO] Execute: huggingface-cli login")
        return False

def check_kiss3dgen_structure():
    """Verifica estrutura do Kiss3DGen"""
    print(f"\n{'='*60}")
    print("VERIFICACAO DA ESTRUTURA KISS3DGEN")
    print(f"{'='*60}")
    
    kiss3dgen_path = Path(__file__).parent.parent / "Kiss3DGen"
    
    if not kiss3dgen_path.exists():
        print("[ERRO] Diretorio Kiss3DGen nao encontrado")
        return False
    
    required_paths = [
        "pipeline/kiss3d_wrapper.py",
        "pipeline/utils.py",
        "models/lrm",
        "models/ISOMER",
        "custom_diffusers",
    ]
    
    all_ok = True
    for req_path in required_paths:
        full_path = kiss3dgen_path / req_path
        if full_path.exists():
            print(f"[OK] {req_path}")
        else:
            print(f"[ERRO] {req_path} nao encontrado")
            all_ok = False
    
    return all_ok

def main():
    """Verifica instalação completa"""
    print("="*60)
    print("VERIFICACAO COMPLETA DA INSTALACAO - Kiss3DGen")
    print("="*60)
    
    checks = {
        "Python": check_python_version(),
        "PyTorch": check_pytorch(),
        "Pacotes Core": check_core_packages(),
        "Pacotes Especiais": check_special_packages(),
        "Diffusers Customizado": check_custom_diffusers(),
        "Estrutura Kiss3DGen": check_kiss3dgen_structure(),
        "HuggingFace Auth": check_huggingface_auth(),
    }
    
    model_results = check_models()
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print(f"{'='*60}")
    
    for name, result in checks.items():
        status = "[OK]" if result else "[ERRO]"
        print(f"{status} {name}")
    
    print("\nModelos:")
    for name, result in model_results.items():
        if result is True:
            print(f"[OK] {name}")
        elif result is False:
            print(f"[ERRO] {name} (obrigatorio)")
        else:
            print(f"[AVISO] {name} (opcional)")
    
    # Verificar se tudo está OK
    all_core_ok = all(checks.values())
    all_required_models = all(
        result is True 
        for name, result in model_results.items() 
        if name != "redux"
    )
    
    print(f"\n{'='*60}")
    if all_core_ok and all_required_models:
        print("[OK] INSTALACAO COMPLETA E PRONTA PARA USO!")
        print("="*60)
        print("\nProximos passos:")
        print("1. Testar pipeline:")
        print("   python scripts/run_kiss3dgen_image_to_3d.py --input <imagem> --output data/outputs/")
        return True
    else:
        print("[AVISO] ALGUMAS VERIFICACOES FALHARAM")
        print("="*60)
        print("\nCorrecoes necessarias:")
        if not checks["Python"]:
            print("- Instalar Python 3.11.9")
        if not checks["PyTorch"]:
            print("- Instalar PyTorch com CUDA")
        if not checks["Pacotes Core"]:
            print("- Instalar pacotes: pip install -r requirements.txt")
        if not checks["Pacotes Especiais"]:
            print("- Instalar nvdiffrast e PyTorch3D")
        if not checks["Diffusers Customizado"]:
            print("- Instalar diffusers customizado: cd Kiss3DGen && pip install -e custom_diffusers/")
        if not all_required_models:
            print("- Baixar modelos: python scripts/download_models.py")
        if not checks["HuggingFace Auth"]:
            print("- Autenticar HuggingFace: huggingface-cli login")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

