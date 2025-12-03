#!/usr/bin/env python3
"""
Script para rodar e monitorar o teste dos 10 objetos continuamente,
corrigindo problemas automaticamente.
"""
import subprocess
import time
import os
import sys
from pathlib import Path
import json
import shutil

project_root = Path(__file__).parent.parent
python_exe = project_root / "mesh3d-generator-py3.11" / "Scripts" / "python.exe"
script_path = project_root / "scripts" / "run_kiss3dgen_image_to_3d.py"
output_dir = project_root / "outputs" / "flux_top10_test"
log_dir = output_dir / "logs"

# Criar diretórios
output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

def check_progress():
    """Verifica o progresso da execução"""
    # Verificar arquivos gerados
    obj_files = list(output_dir.rglob("*.obj"))
    glb_files = list(output_dir.rglob("*.glb"))
    png_files = list(output_dir.rglob("*_3d_bundle.png"))
    
    # Verificar histórico
    history_file = output_dir / "runs_report.json"
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text())
        except:
            pass
    
    successful = sum(1 for item in history if item.get("success", False))
    failed = sum(1 for item in history if not item.get("success", False))
    
    return {
        "obj_files": len(obj_files),
        "glb_files": len(glb_files),
        "bundle_files": len(png_files),
        "successful": successful,
        "failed": failed,
        "total": len(history)
    }

def check_errors():
    """Verifica se há erros nos logs"""
    if not log_dir.exists():
        return []
    
    errors = []
    for log_file in sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            content = log_file.read_text(encoding='utf-8', errors='ignore')
            # Procurar por erros críticos
            if any(keyword in content.lower() for keyword in ["error", "exception", "traceback", "failed", "cuda error"]):
                # Pegar últimas linhas com erro
                lines = content.split('\n')
                error_lines = [line for line in lines[-50:] if any(kw in line.lower() for kw in ["error", "exception", "traceback"])]
                if error_lines:
                    errors.extend(error_lines[-10:])  # Últimas 10 linhas de erro
        except Exception as e:
            pass
    
    return errors[-20:]  # Retornar últimos 20 erros

def run_test():
    """Executa o teste"""
    print("=" * 80)
    print("INICIANDO TESTE DOS 10 OBJETOS")
    print("=" * 80)
    print(f"Python: {python_exe}")
    print(f"Script: {script_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    cmd = [
        str(python_exe),
        str(script_path),
        "--dataset-plan", str(project_root / "pipeline_config" / "flux_top10_dataset.yaml"),
        "--output", str(output_dir),
        "--config", str(project_root / "pipeline_config" / "default.yaml"),
    ]
    
    print(f"\nComando: {' '.join(cmd)}\n")
    
    # Rodar processo
    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Monitorar processo
    last_progress = None
    error_count = 0
    max_errors = 5
    progress_interval = 30
    last_progress_report = time.monotonic()
    
    try:
        while True:
            # Ler saída
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
                    # Detectar erros críticos
                    if any(keyword in line.lower() for keyword in ["error", "exception", "traceback", "failed"]):
                        error_count += 1
                        if error_count > max_errors:
                            print(f"\n[AVISO] Muitos erros detectados ({error_count}). Verificando...")
                            errors = check_errors()
                            if errors:
                                print("\nÚltimos erros encontrados:")
                                for err in errors[-5:]:
                                    print(f"  {err}")
            
            # Verificar se processo terminou
            if process.poll() is not None:
                break
            
            # Verificar progresso em intervalos regulares
            time.sleep(1)
            now = time.monotonic()
            if now - last_progress_report >= progress_interval:
                progress = check_progress()
                last_progress_report = now
                if progress != last_progress:
                    print(
                        f"\n[PROGRESSO] Objetos: {progress['obj_files']} OBJ, {progress['glb_files']} GLB, "
                        f"{progress['bundle_files']} Bundles | "
                        f"Sucesso: {progress['successful']}/{progress['total']} | "
                        f"Falhas: {progress['failed']}"
                    )
                    last_progress = progress

                    # Se completou todos os 10 objetos, podemos terminar
                    if progress['successful'] >= 10:
                        print("\n[SUCESSO] Todos os 10 objetos foram processados com sucesso!")
                        break
        
        # Aguardar processo terminar
        return_code = process.wait()
        
        # Verificar resultado final
        progress = check_progress()
        print("\n" + "=" * 80)
        print("RESULTADO FINAL")
        print("=" * 80)
        print(f"Arquivos OBJ gerados: {progress['obj_files']}")
        print(f"Arquivos GLB gerados: {progress['glb_files']}")
        print(f"Bundles gerados: {progress['bundle_files']}")
        print(f"Sucessos: {progress['successful']}")
        print(f"Falhas: {progress['failed']}")
        print(f"Código de saída: {return_code}")
        print("=" * 80)
        
        # Verificar erros finais
        errors = check_errors()
        if errors:
            print("\nERROS ENCONTRADOS:")
            for err in errors[-10:]:
                print(f"  {err}")
        
        return return_code == 0 and progress['successful'] >= 10
        
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Encerrando processo...")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n[ERRO] Erro ao executar teste: {e}")
        import traceback
        traceback.print_exc()
        if process.poll() is None:
            process.terminate()
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

