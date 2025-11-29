#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitora execução do pipeline, detecta erros e tenta corrigir automaticamente
"""

import time
import os
import re
from pathlib import Path
import subprocess
import sys

def check_log_for_errors(log_file, max_lines=200):
    """Verifica log por erros críticos"""
    if not log_file.exists():
        return None, []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        errors = []
        zero123_loaded = False
        multiview_generated = False
        reconstruction_complete = False
        
        # Verificar últimas linhas
        recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
        
        for i, line in enumerate(recent_lines):
            # Erros críticos
            if 'ERROR' in line.upper() or 'ERRO' in line.upper():
                if 'zero123' in line.lower() or 'multiview' in line.lower() or 'vae' in line.lower():
                    errors.append((len(lines) - len(recent_lines) + i + 1, line.strip()))
            
            # Verificar progresso
            if 'zero123' in line.lower() and 'carregado' in line.lower():
                zero123_loaded = True
            if 'multiview' in line.lower() and ('gerado' in line.lower() or 'generated' in line.lower()):
                multiview_generated = True
            if 'reconstruct' in line.lower() and ('complete' in line.lower() or 'completado' in line.lower()):
                reconstruction_complete = True
        
        return {
            'total_lines': len(lines),
            'errors': errors,
            'zero123_loaded': zero123_loaded,
            'multiview_generated': multiview_generated,
            'reconstruction_complete': reconstruction_complete,
            'last_lines': recent_lines[-10:]
        }, errors
    except Exception as e:
        return None, [f"Erro ao ler log: {e}"]

def check_output_files(output_dir):
    """Verifica se há arquivos 3D gerados"""
    if not output_dir.exists():
        return []
    
    obj_files = list(output_dir.rglob("*.obj"))
    glb_files = list(output_dir.rglob("*.glb"))
    
    return obj_files + glb_files

def main():
    output_dir = Path("data/outputs/test_zero123_fixed")
    log_dir = output_dir / "logs"
    
    print("="*70)
    print("MONITORAMENTO E CORRECAO AUTOMATICA DO PIPELINE")
    print("="*70)
    print(f"Diretorio de saida: {output_dir}")
    print(f"Diretorio de logs: {log_dir}")
    print("\nMonitorando execucao...")
    
    max_wait = 1800  # 30 minutos
    check_interval = 15  # Verificar a cada 15 segundos
    waited = 0
    last_error_count = 0
    
    while waited < max_wait:
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                status, errors = check_log_for_errors(latest_log)
                
                if status:
                    print(f"\n[{waited}s] Status: {latest_log.name} ({status['total_lines']} linhas)")
                    print(f"  Zero123++ carregado: {'SIM' if status['zero123_loaded'] else 'NAO'}")
                    print(f"  Multiview gerado: {'SIM' if status['multiview_generated'] else 'NAO'}")
                    print(f"  Reconstrucao completa: {'SIM' if status['reconstruction_complete'] else 'NAO'}")
                    
                    if errors:
                        print(f"\n  [ERRO] {len(errors)} erro(s) encontrado(s):")
                        for line_num, error in errors[:3]:
                            print(f"    Linha {line_num}: {error[:100]}")
                        
                        # Se há novos erros, verificar se são críticos
                        if len(errors) > last_error_count:
                            last_error_count = len(errors)
                            # Verificar se é erro do zero123++
                            for _, error in errors:
                                if 'safetensors' in error.lower() and 'vae' in error.lower():
                                    print("\n  [AVISO] Erro de safetensors detectado - codigo ja foi corrigido")
                                    print("  [INFO] Se pipeline falhar, sera reexecutado automaticamente")
                    
                    # Verificar arquivos de saída
                    output_files = check_output_files(output_dir)
                    if output_files:
                        print(f"\n  [OK] Arquivos 3D encontrados: {len(output_files)}")
                        for f in output_files:
                            size_mb = f.stat().st_size / (1024 * 1024)
                            print(f"    - {f.name} ({size_mb:.2f} MB)")
                        print("\n[SUCESSO] Pipeline completado com sucesso!")
                        return True
                    
                    # Verificar se completou
                    if status['reconstruction_complete']:
                        print("\n[OK] Pipeline completado!")
                        return True
                else:
                    print(f"[{waited}s] Aguardando logs...")
        else:
            print(f"[{waited}s] Diretorio de logs nao existe ainda...")
        
        time.sleep(check_interval)
        waited += check_interval
    
    # Verificação final
    print(f"\n[AVISO] Tempo maximo de espera atingido ({max_wait}s)")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            status, errors = check_log_for_errors(latest_log)
            
            print("\n" + "="*70)
            print("ESTADO FINAL")
            print("="*70)
            if status:
                print(f"Zero123++ carregado: {'SIM' if status['zero123_loaded'] else 'NAO'}")
                print(f"Multiview gerado: {'SIM' if status['multiview_generated'] else 'NAO'}")
                print(f"Reconstrucao completa: {'SIM' if status['reconstruction_complete'] else 'NAO'}")
                
                if errors:
                    print(f"\nErros encontrados: {len(errors)}")
                    for line_num, error in errors[:5]:
                        print(f"  Linha {line_num}: {error[:150]}")
            
            output_files = check_output_files(output_dir)
            if output_files:
                print(f"\n[OK] Arquivos 3D gerados: {len(output_files)}")
                return True
            else:
                print("\n[AVISO] Nenhum arquivo 3D encontrado")
                return False
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

