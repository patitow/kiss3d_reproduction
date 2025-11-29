#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script completo de setup - instala dependências e baixa modelos
Executa tudo na venv ativa
"""

import sys
import subprocess
from pathlib import Path

def check_venv():
    """Verifica se está em uma venv"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if not in_venv:
        print("[ERRO] Este script deve ser executado dentro de uma venv ativa!")
        print("\n[INFO] Ative a venv primeiro:")
        print("       .\\mesh3d-generator-py3.11\\Scripts\\Activate.ps1  # Windows PowerShell")
        print("       .\\mesh3d-generator-py3.11\\Scripts\\activate.bat  # Windows CMD")
        return False
    print(f"[OK] Venv detectada: {sys.prefix}")
    return True

def run_script(script_path: Path, description: str):
    """Executa um script Python"""
    print("\n" + "="*60)
    print(description)
    print("="*60)
    
    if not script_path.exists():
        print(f"[ERRO] Script não encontrado: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Falha ao executar {script_path}: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[AVISO] Interrompido pelo usuário")
        return False

def main():
    """Executa setup completo"""
    print("="*60)
    print("SETUP COMPLETO - Kiss3DGen")
    print("="*60)
    
    # Verificar venv
    if not check_venv():
        return False
    
    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"
    
    # Passo 1: Instalar dependências
    print("\n" + "="*60)
    print("PASSO 1/2: INSTALAR DEPENDÊNCIAS")
    print("="*60)
    install_script = scripts_dir / "install_dependencies.py"
    if not run_script(install_script, "Instalando dependências..."):
        print("\n[ERRO] Falha na instalação de dependências!")
        response = input("Continuar mesmo assim? (s/N): ")
        if response.lower() != 's':
            return False
    
    # Passo 2: Baixar modelos
    print("\n" + "="*60)
    print("PASSO 2/2: BAIXAR MODELOS")
    print("="*60)
    download_script = scripts_dir / "download_all_models.py"
    if not run_script(download_script, "Baixando modelos..."):
        print("\n[AVISO] Alguns modelos podem não ter sido baixados.")
        print("[INFO] Execute 'python scripts/download_all_models.py' novamente se necessário.")
    
    # Resumo final
    print("\n" + "="*60)
    print("SETUP COMPLETO!")
    print("="*60)
    print("\n[OK] Setup concluído!")
    print("\nPróximos passos:")
    print("1. Verificar instalação:")
    print("   python -c \"import torch; import pytorch3d; print('OK')\"")
    print("2. Executar pipeline:")
    print("   python scripts/run_kiss3dgen_image_to_3d_optimized.py --input <imagem>")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

