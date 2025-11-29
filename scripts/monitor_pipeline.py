#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para monitorar execução do pipeline e verificar erros
"""

import time
import os
from pathlib import Path
import re

def check_logs(log_dir):
    """Verifica logs por erros e warnings"""
    errors = []
    warnings = []
    deprecations = []
    
    log_files = list(Path(log_dir).rglob("*.log"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    # Erros
                    if 'ERROR' in line.upper() or 'ERRO' in line.upper():
                        errors.append((log_file, i, line.strip()))
                    # Warnings
                    elif 'WARNING' in line.upper() or 'WARN' in line.upper():
                        warnings.append((log_file, i, line.strip()))
                    # Deprecations
                    elif 'deprecated' in line.lower() or 'DeprecationWarning' in line:
                        deprecations.append((log_file, i, line.strip()))
        except Exception as e:
            print(f"Erro ao ler {log_file}: {e}")
    
    return errors, warnings, deprecations

def main():
    output_dir = Path("data/outputs/test_zero123")
    log_dir = output_dir / "logs"
    
    print("="*70)
    print("MONITORAMENTO DO PIPELINE")
    print("="*70)
    
    if not log_dir.exists():
        print(f"[AVISO] Diretório de logs não existe ainda: {log_dir}")
        print("Aguardando criação dos logs...")
        return
    
    print(f"\n[INFO] Verificando logs em: {log_dir}")
    
    errors, warnings, deprecations = check_logs(log_dir)
    
    print(f"\n[RESULTADOS]")
    print(f"  Erros encontrados: {len(errors)}")
    print(f"  Warnings encontrados: {len(warnings)}")
    print(f"  Deprecations encontrados: {len(deprecations)}")
    
    if errors:
        print(f"\n[ERROS] ({len(errors)} encontrados):")
        for log_file, line_num, line in errors[:10]:  # Mostrar primeiros 10
            print(f"  {log_file.name}:{line_num} - {line[:100]}")
    
    if warnings:
        print(f"\n[WARNINGS] ({len(warnings)} encontrados - mostrando primeiros 5):")
        for log_file, line_num, line in warnings[:5]:
            print(f"  {log_file.name}:{line_num} - {line[:100]}")
    
    if deprecations:
        print(f"\n[DEPRECATIONS] ({len(deprecations)} encontrados - mostrando primeiros 5):")
        for log_file, line_num, line in deprecations[:5]:
            print(f"  {log_file.name}:{line_num} - {line[:100]}")
    
    # Verificar se há arquivos de saída
    output_files = list(output_dir.rglob("*.obj")) + list(output_dir.rglob("*.glb"))
    if output_files:
        print(f"\n[OK] Arquivos 3D gerados: {len(output_files)}")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
    else:
        print(f"\n[AVISO] Nenhum arquivo 3D encontrado ainda")

if __name__ == "__main__":
    main()

