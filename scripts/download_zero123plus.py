#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para baixar o modelo Zero123++ completo e colocá-lo na pasta models
Baseado no repositório: https://github.com/SUDO-AI-3D/zero123plus
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
import shutil
import torch

# Adicionar paths necessários
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KISS3D_ROOT = PROJECT_ROOT / "Kiss3DGen"
MODELS_DIR = KISS3D_ROOT / "models" / "zero123plus"

def check_hf_auth():
    """Verifica autenticação HuggingFace"""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"[OK] Autenticado no HuggingFace como: {user.get('name', 'N/A')}")
        return True
    except Exception as e:
        print(f"[AVISO] Não autenticado no HuggingFace: {e}")
        print("[INFO] Tentando continuar mesmo assim...")
        return False

def list_model_files(repo_id: str):
    """Lista todos os arquivos do repositório"""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        return files
    except Exception as e:
        print(f"[AVISO] Erro ao listar arquivos: {e}")
        return []

def download_zero123plus_model(repo_id: str = "sudo-ai/zero123plus-v1.1", target_dir: Path = None):
    """
    Baixa o modelo Zero123++ completo e coloca na pasta models
    """
    print("="*70)
    print("DOWNLOAD DO MODELO ZERO123++")
    print("="*70)
    print(f"Repositório: {repo_id}")
    print(f"Destino: {target_dir}")
    print("="*70)
    
    # Verificar autenticação
    check_hf_auth()
    
    # Listar arquivos disponíveis
    print("\n[INFO] Listando arquivos disponíveis no repositório...")
    files = list_model_files(repo_id)
    if files:
        print(f"[OK] Encontrados {len(files)} arquivos:")
        for f in files[:10]:  # Mostrar primeiros 10
            print(f"   - {f}")
        if len(files) > 10:
            print(f"   ... e mais {len(files) - 10} arquivos")
    else:
        print("[AVISO] Não foi possível listar arquivos, continuando download completo...")
    
    # Criar diretório de destino
    if target_dir:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Baixando modelo para: {target_dir}")
        
        try:
            # Baixar modelo completo para o diretório local
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,  # Windows não suporta symlinks bem
                resume_download=True,
                ignore_patterns=["*.md", "*.txt", "*.json"]  # Ignorar arquivos de documentação
            )
            print(f"[OK] Modelo baixado com sucesso para: {target_dir}")
            
            # Verificar arquivos importantes
            important_files = [
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "unet/config.json",
                "vae/config.json"
            ]
            
            print("\n[INFO] Verificando arquivos importantes...")
            found_files = []
            for file in important_files:
                file_path = target_dir / file
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"   [OK] {file} ({size_mb:.2f} MB)")
                    found_files.append(file)
                else:
                    # Tentar encontrar arquivo similar
                    similar = list(target_dir.rglob(file.split('/')[-1]))
                    if similar:
                        print(f"   [OK] {similar[0].relative_to(target_dir)} (encontrado em local diferente)")
                        found_files.append(str(similar[0].relative_to(target_dir)))
                    else:
                        print(f"   [AVISO] {file} não encontrado")
            
            if len(found_files) >= 3:
                print(f"\n[OK] Modelo parece completo ({len(found_files)} arquivos importantes encontrados)")
                return True
            else:
                print(f"\n[AVISO] Modelo pode estar incompleto ({len(found_files)} arquivos importantes encontrados)")
                return False
                
        except Exception as e:
            print(f"[ERRO] Erro ao baixar modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # Baixar para cache padrão do HuggingFace
        print("\n[INFO] Baixando modelo para cache do HuggingFace...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=None,  # Usar cache padrão
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("[OK] Modelo baixado para cache do HuggingFace")
            return True
        except Exception as e:
            print(f"[ERRO] Erro ao baixar modelo: {e}")
            import traceback
            traceback.print_exc()
            return False

def verify_model_files(model_dir: Path):
    """Verifica se os arquivos do modelo estão presentes"""
    print("\n" + "="*70)
    print("VERIFICAÇÃO DE ARQUIVOS DO MODELO")
    print("="*70)
    
    if not model_dir.exists():
        print(f"[ERRO] Diretório não existe: {model_dir}")
        return False
    
    # Arquivos essenciais
    essential_files = {
        "model_index.json": "Configuração principal do modelo",
        "scheduler/scheduler_config.json": "Configuração do scheduler",
        "text_encoder/config.json": "Configuração do text encoder",
        "unet/config.json": "Configuração do UNet",
        "vae/config.json": "Configuração do VAE",
    }
    
    # Arquivos de pesos (pelo menos um deve existir)
    weight_files = [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.bin",
    ]
    
    print("\n[INFO] Verificando arquivos essenciais...")
    essential_ok = 0
    for file, desc in essential_files.items():
        file_path = model_dir / file
        if file_path.exists():
            print(f"   [OK] {file} - {desc}")
            essential_ok += 1
        else:
            print(f"   [AVISO] {file} - {desc} (não encontrado)")
    
    print("\n[INFO] Verificando arquivos de pesos...")
    weight_found = False
    for file in weight_files:
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   [OK] {file} ({size_mb:.2f} MB)")
            weight_found = True
        else:
            # Tentar encontrar recursivamente
            found = list(model_dir.rglob(file.split('/')[-1]))
            if found:
                size_mb = found[0].stat().st_size / (1024 * 1024)
                print(f"   [OK] {found[0].relative_to(model_dir)} ({size_mb:.2f} MB)")
                weight_found = True
    
    if not weight_found:
        print("   [ERRO] Nenhum arquivo de pesos encontrado!")
    
    # Verificar estrutura de diretórios
    print("\n[INFO] Estrutura de diretórios:")
    for subdir in ["scheduler", "text_encoder", "unet", "vae", "feature_extractor"]:
        subdir_path = model_dir / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.rglob("*")))
            print(f"   [OK] {subdir}/ ({file_count} arquivos)")
        else:
            print(f"   [AVISO] {subdir}/ (não encontrado)")
    
    return essential_ok >= 3 and weight_found

def main():
    """Função principal"""
    print("="*70)
    print("SCRIPT DE DOWNLOAD ZERO123++")
    print("="*70)
    
    # Verificar CUDA
    print(f"\n[INFO] CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Verificar se já existe
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        print(f"\n[INFO] Diretório já existe: {MODELS_DIR}")
        print("[INFO] Verificando arquivos existentes...")
        if verify_model_files(MODELS_DIR):
            print("\n[OK] Modelo parece estar completo!")
            return True
        else:
            print("\n[AVISO] Modelo parece incompleto. Baixando novamente...")
    
    # Versões disponíveis
    versions = {
        "1": ("sudo-ai/zero123plus-v1.1", "v1.1 (recomendado)"),
        "2": ("sudo-ai/zero123plus-v1.2", "v1.2 (mais recente)"),
    }
    
    print("\n[INFO] Versões disponíveis:")
    for key, (repo_id, desc) in versions.items():
        print(f"   {key}. {desc} ({repo_id})")
    
    # Usar v1.1 por padrão (mais estável e testado)
    choice = "1"
    repo_id, desc = versions.get(choice, versions["1"])
    print(f"\n[INFO] Usando versão padrão: {desc}")
    
    print(f"\n[INFO] Baixando {desc}...")
    
    # Baixar modelo
    success = download_zero123plus_model(repo_id=repo_id, target_dir=MODELS_DIR)
    
    if success:
        print("\n" + "="*70)
        print("VERIFICAÇÃO FINAL")
        print("="*70)
        verify_model_files(MODELS_DIR)
        print("\n[OK] Download concluído!")
        print(f"\n[INFO] Modelo disponível em: {MODELS_DIR}")
        print("[INFO] O pipeline deve conseguir carregar o modelo automaticamente.")
        return True
    else:
        print("\n[ERRO] Falha no download!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

