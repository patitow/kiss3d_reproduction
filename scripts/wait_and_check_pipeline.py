#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aguarda execução do pipeline e verifica progresso
"""

import time
import os
from pathlib import Path
import re

def check_log_progress(log_file):
    """Verifica progresso no log"""
    if not log_file.exists():
        return "Log não existe ainda", []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Verificar etapas
        stages = {
            "HuggingFace": False,
            "Flux": False,
            "Zero123": False,
            "Caption": False,
            "LRM": False,
            "Multiview": False,
            "Reconstruction": False,
            "Complete": False
        }
        
        errors = []
        last_lines = lines[-50:] if len(lines) > 50 else lines
        
        for line in last_lines:
            if "HuggingFace" in line or "Token HuggingFace" in line:
                stages["HuggingFace"] = True
            if "Loading Flux" in line or "Flux model" in line:
                stages["Flux"] = True
            if "zero123" in line.lower() or "Zero123" in line or "multiview" in line.lower():
                stages["Zero123"] = True
            if "caption" in line.lower() or "Florence" in line:
                stages["Caption"] = True
            if "LRM" in line or "reconstruction" in line.lower():
                stages["LRM"] = True
            if "multiview" in line.lower() and "generat" in line.lower():
                stages["Multiview"] = True
            if "reconstruct" in line.lower() and "mesh" in line.lower():
                stages["Reconstruction"] = True
            if "ERRO" in line.upper() or "ERROR" in line.upper():
                errors.append(line.strip()[:150])
            if "Complete" in line or "completado" in line.lower():
                stages["Complete"] = True
        
        return stages, errors, len(lines)
    except Exception as e:
        return f"Erro ao ler log: {e}", []

def main():
    output_dir = Path("data/outputs/test_zero123")
    log_dir = output_dir / "logs"
    
    print("="*70)
    print("MONITORAMENTO CONTÍNUO DO PIPELINE")
    print("="*70)
    print(f"Diretório de saída: {output_dir}")
    print(f"Diretório de logs: {log_dir}")
    print("\nAguardando logs...")
    
    max_wait = 600  # 10 minutos
    check_interval = 10  # Verificar a cada 10 segundos
    waited = 0
    
    while waited < max_wait:
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                stages, errors, line_count = check_log_progress(latest_log)
                
                print(f"\n[{waited}s] Verificando: {latest_log.name} ({line_count} linhas)")
                print("Etapas:")
                for stage, status in stages.items():
                    status_icon = "[OK]" if status else "[...]"
                    print(f"  {status_icon} {stage}")
                
                if errors:
                    print(f"\n⚠️  Erros encontrados ({len(errors)}):")
                    for err in errors[:3]:
                        print(f"    - {err}")
                
                # Verificar se completou
                if stages.get("Complete", False):
                    print("\n[OK] Pipeline completado!")
                    break
                
                # Verificar se há arquivos de saída
                output_files = list(output_dir.rglob("*.obj")) + list(output_dir.rglob("*.glb"))
                if output_files:
                    print(f"\n[OK] Arquivos 3D encontrados: {len(output_files)}")
                    for f in output_files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"    - {f.name} ({size_mb:.2f} MB)")
                    break
            else:
                print(f"[{waited}s] Aguardando criação de logs...")
        else:
            print(f"[{waited}s] Diretório de logs não existe ainda...")
        
        time.sleep(check_interval)
        waited += check_interval
    
    if waited >= max_wait:
        print(f"\n⏱️  Tempo máximo de espera atingido ({max_wait}s)")
        print("Verificando estado final...")
    
    # Verificação final
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            stages, errors, line_count = check_log_progress(latest_log)
            
            print("\n" + "="*70)
            print("ESTADO FINAL")
            print("="*70)
            print(f"Log: {latest_log.name}")
            print(f"Linhas: {line_count}")
            print("\nEtapas completadas:")
            for stage, status in stages.items():
                if status:
                    print(f"  [OK] {stage}")
            
            if errors:
                print(f"\n[ERRO] Erros ({len(errors)}):")
                for err in errors[:5]:
                    print(f"    - {err}")
            
            # Verificar arquivos de saída
            output_files = list(output_dir.rglob("*.obj")) + list(output_dir.rglob("*.glb"))
            if output_files:
                print(f"\n[OK] Arquivos 3D gerados ({len(output_files)}):")
                for f in output_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    mtime = time.ctime(f.stat().st_mtime)
                    print(f"    - {f.name} ({size_mb:.2f} MB, {mtime})")
            else:
                print("\n[AVISO] Nenhum arquivo 3D encontrado")

if __name__ == "__main__":
    main()

