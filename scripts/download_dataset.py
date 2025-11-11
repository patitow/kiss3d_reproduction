#!/usr/bin/env python3
"""
Script para baixar o dataset do Google Research
"""

import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gdown
import requests


def download_google_research_dataset(output_dir: str = "data/raw"):
    """
    Baixa o dataset do Google Research (Gazebo).
    
    Args:
        output_dir: Diretório de saída para os dados
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📥 Baixando dataset do Google Research...")
    print("⚠️  Nota: Verifique a URL e método de download no site oficial:")
    print("    https://app.gazebosim.org/GoogleResearch")
    
    # TODO: Implementar download do dataset
    # O método de download depende de como o dataset está disponibilizado
    # Pode ser necessário usar gdown, wget, ou API específica
    
    print(f"✅ Dataset será salvo em: {output_path.absolute()}")


if __name__ == "__main__":
    download_google_research_dataset()


