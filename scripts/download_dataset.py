#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script wrapper para baixar o dataset do Google Research do Gazebo.
Redireciona para o script completo download_gazebo_dataset.py
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    # Importar e executar o script completo
    from scripts.download_gazebo_dataset import main
    main()


