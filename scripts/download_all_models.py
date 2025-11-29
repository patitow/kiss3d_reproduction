#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script completo para baixar todos os modelos necessários para o Kiss3DGen
Executa na venv ativa
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import torch

# Adicionar paths necessários
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"

# Mudar para diretório Kiss3DGen se necessário
if KISS3D_ROOT.exists():
    os.chdir(str(KISS3D_ROOT))

def check_venv():
    """Verifica se está em uma venv"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if not in_venv:
        print("[AVISO] Não parece estar em uma venv ativa!")
        print("[INFO] Ative a venv antes de executar este script:")
        print("       .\\mesh3d-generator-py3.11\\Scripts\\Activate.ps1  # Windows PowerShell")
        print("       .\\mesh3d-generator-py3.11\\Scripts\\activate.bat  # Windows CMD")
        print("       source mesh3d-generator-py3.11/bin/activate      # Linux/Mac")
        response = input("\nContinuar mesmo assim? (s/N): ")
        if response.lower() != 's':
            return False
    else:
        print(f"[OK] Venv detectada: {sys.prefix}")
    return True

def check_hf_auth():
    """Verifica autenticação HuggingFace"""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"[OK] Autenticado no HuggingFace como: {user.get('name', 'N/A')}")
        return True
    except Exception:
        print("[ERRO] Não autenticado no HuggingFace!")
        print("[INFO] Execute: huggingface-cli login")
        print("[INFO] Ou: python scripts/setup_huggingface_auth.py")
        return False

def download_file(repo_id: str, filename: str, local_dir: str, description: str):
    """Baixa um arquivo específico do HuggingFace"""
    print(f"\n{'='*60}")
    print(f"[DOWNLOAD] {description}")
    print(f"   Repositorio: {repo_id}")
    print(f"   Arquivo: {filename}")
    print(f"{'='*60}")
    
    local_path = Path(local_dir) / filename
    if local_path.exists():
        print(f"[OK] Arquivo já existe: {local_path}")
        return True
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[OK] Arquivo baixado: {downloaded_path}")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao baixar arquivo: {e}")
        return False

def download_model(repo_id: str, local_dir: str, description: str):
    """Baixa um modelo completo do HuggingFace"""
    print(f"\n{'='*60}")
    print(f"[DOWNLOAD] {description}")
    print(f"   Repositorio: {repo_id}")
    print(f"{'='*60}")
    
    if local_dir and Path(local_dir).exists() and any(Path(local_dir).iterdir()):
        print(f"[OK] Modelo já existe em: {local_dir}")
        return True
    
    try:
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir if local_dir else None,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[OK] Modelo baixado com sucesso!")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao baixar modelo: {e}")
        if "disk" in str(e).lower() or "space" in str(e).lower():
            print(f"   [AVISO] Espaço em disco insuficiente!")
        return False

def main():
    """Baixa todos os modelos necessários"""
    print("="*60)
    print("DOWNLOAD DE MODELOS - Kiss3DGen")
    print("="*60)
    
    # Verificar venv
    if not check_venv():
        print("\n[ERRO] Execute este script dentro de uma venv ativa!")
        return False
    
    # Verificar CUDA
    print(f"\n[INFO] CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Verificar autenticação HuggingFace
    if not check_hf_auth():
        print("\n[ERRO] Autenticação HuggingFace necessária!")
        return False
    
    # Criar diretório de checkpoints
    ckpt_root = Path("checkpoint")
    ckpt_root.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. Flux LoRA
    print("\n" + "="*60)
    print("1/6: Flux LoRA (rgb_normal)")
    print("="*60)
    flux_lora_dir = ckpt_root / "flux_lora"
    results['flux_lora'] = download_file(
        repo_id="LTT/Kiss3DGen",
        filename="rgb_normal.safetensors",
        local_dir=str(flux_lora_dir),
        description="Flux LoRA para RGB/Normal"
    )
    
    # 2. Flux LoRA Redux (opcional)
    print("\n" + "="*60)
    print("2/6: Flux LoRA Redux (opcional)")
    print("="*60)
    results['flux_lora_redux'] = download_file(
        repo_id="LTT/Kiss3DGen",
        filename="rgb_normal_redux.safetensors",
        local_dir=str(flux_lora_dir),
        description="Flux LoRA Redux (opcional)"
    )
    
    # 3. LRM Model
    print("\n" + "="*60)
    print("3/6: LRM (Large Reconstruction Model)")
    print("="*60)
    lrm_dir = ckpt_root / "lrm"
    results['lrm'] = download_file(
        repo_id="LTT/PRM",
        filename="final_ckpt.ckpt",
        local_dir=str(lrm_dir),
        description="LRM checkpoint final"
    )
    
    # 4. Zero123++ UNet
    print("\n" + "="*60)
    print("4/6: Zero123++ UNet (FlexGen)")
    print("="*60)
    zero123_dir = ckpt_root / "zero123++"
    results['zero123_unet'] = download_file(
        repo_id="LTT/Kiss3DGen",
        filename="flexgen.ckpt",
        local_dir=str(zero123_dir),
        description="Zero123++ UNet checkpoint"
    )
    
    # 5. Zero123++ Modelo Completo (via cache do HuggingFace)
    print("\n" + "="*60)
    print("5/6: Zero123++ Modelo Completo")
    print("="*60)
    print("[INFO] Baixando modelo completo para cache do HuggingFace...")
    print("[INFO] Isso pode demorar vários minutos...")
    print("[INFO] Repositório: sudo-ai/zero123plus-v1.1")
    results['zero123_full'] = download_model(
        repo_id="sudo-ai/zero123plus-v1.1",
        local_dir=None,  # Usar cache padrão
        description="Zero123++ modelo completo"
    )
    
    # 5b. ControlNet (obrigatório)
    print("\n" + "="*60)
    print("5b/6: ControlNet para Flux")
    print("="*60)
    print("[INFO] Baixando ControlNet para cache do HuggingFace...")
    results['controlnet'] = download_model(
        repo_id="InstantX/FLUX.1-dev-Controlnet-Union",
        local_dir=None,  # Usar cache padrão
        description="ControlNet para Flux"
    )
    
    # 6. Flux Model (via cache do HuggingFace) - OPcional, muito grande
    print("\n" + "="*60)
    print("6/6: Flux Model (OPCIONAL - muito grande ~24GB)")
    print("="*60)
    print("[INFO] O modelo Flux completo é muito grande (~24GB)!")
    print("[INFO] O pipeline usa 'flux1-schnell' por padrão (será baixado automaticamente quando necessário)")
    print("[INFO] Pulando download do Flux completo para economizar espaço")
    print("[INFO] Para baixar depois, execute:")
    print("       python -c \"from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')\"")
    results['flux'] = None
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DO DOWNLOAD")
    print("="*60)
    
    for name, success in results.items():
        if success is True:
            print(f"[OK] {name}: Baixado com sucesso")
        elif success is False:
            print(f"[ERRO] {name}: Falha no download")
        elif success is None:
            print(f"[AVISO] {name}: Pulado")
    
    required_ok = all(
        results.get(name) is True 
        for name in ['flux_lora', 'lrm', 'zero123_unet', 'zero123_full', 'controlnet']
    )
    
    if required_ok:
        print("\n[OK] Todos os modelos obrigatórios foram baixados!")
        print("\nPróximos passos:")
        print("1. Verificar se PyTorch3D está instalado com CUDA:")
        print("   python -c \"import pytorch3d; print('PyTorch3D OK')\"")
        print("2. Executar pipeline:")
        print("   python scripts/run_kiss3dgen_image_to_3d_optimized.py --input <imagem>")
        return True
    else:
        print("\n[AVISO] Alguns modelos obrigatórios não foram baixados.")
        print("Execute este script novamente para tentar novamente.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

