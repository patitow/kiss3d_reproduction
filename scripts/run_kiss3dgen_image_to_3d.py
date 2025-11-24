#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline IMAGE TO 3D baseado no Kiss3DGen
Reproduz exatamente o pipeline do artigo Kiss3DGen
"""

import sys
import os
from pathlib import Path

# Adicionar ninja ao PATH antes de qualquer import
project_root = Path(__file__).parent.parent
venv_path = project_root / "mesh3d-generator-py3.11"
possible_ninja_paths = [
    project_root,  # Pode estar na raiz do projeto
    venv_path / "Scripts",
    venv_path / "ninja" / "data" / "bin",
    venv_path / "Lib" / "site-packages" / "ninja" / "data" / "bin",
]

# Também verificar no site-packages do Python
try:
    import site
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    if site_packages:
        possible_ninja_paths.append(Path(site_packages) / "ninja" / "data" / "bin")
except:
    pass

ninja_found = False
for ninja_path in possible_ninja_paths:
    ninja_exe = ninja_path / "ninja.exe"
    if ninja_exe.exists():
        current_path = os.environ.get("PATH", "")
        if str(ninja_path) not in current_path:
            os.environ["PATH"] = str(ninja_path) + os.pathsep + current_path
        print(f"[INFO] Ninja encontrado e adicionado ao PATH: {ninja_path}")
        ninja_found = True
        break

if not ninja_found:
    print("[AVISO] Ninja nao encontrado - pode causar erros ao compilar extensoes C++")
    print("[INFO] Tentando adicionar Scripts ao PATH de qualquer forma...")
    scripts_path = venv_path / "Scripts"
    if scripts_path.exists():
        current_path = os.environ.get("PATH", "")
        if str(scripts_path) not in current_path:
            os.environ["PATH"] = str(scripts_path) + os.pathsep + current_path
            print(f"[INFO] Scripts adicionado ao PATH: {scripts_path}")

# Configurar CUDA_HOME se nao estiver definido
if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    
    if not cuda_home:
        try:
            import torch
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_base = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
                possible_cuda_paths = []
                
                if os.path.exists(cuda_base):
                    for item in os.listdir(cuda_base):
                        cuda_path = os.path.join(cuda_base, item)
                        if os.path.isdir(cuda_path) and item.startswith("v"):
                            possible_cuda_paths.append(cuda_path)
                
                if cuda_version:
                    version_parts = cuda_version.split(".")
                    if len(version_parts) >= 2:
                        version_major_minor = f"{version_parts[0]}.{version_parts[1]}"
                        specific_path = os.path.join(cuda_base, f"v{version_major_minor}")
                        if os.path.exists(specific_path) and specific_path not in possible_cuda_paths:
                            possible_cuda_paths.insert(0, specific_path)
                
                cuda_found = False
                for cuda_path in possible_cuda_paths:
                    if os.path.exists(cuda_path) and os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                        os.environ["CUDA_HOME"] = cuda_path
                        os.environ["CUDA_PATH"] = cuda_path
                        print(f"[INFO] CUDA_HOME configurado: {cuda_path}")
                        cuda_found = True
                        break
                
                if not cuda_found:
                    print("[AVISO] CUDA_HOME nao encontrado automaticamente")
                    default_cuda = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
                    if os.path.exists(default_cuda):
                        os.environ["CUDA_HOME"] = default_cuda
                        os.environ["CUDA_PATH"] = default_cuda
                        print(f"[INFO] CUDA_HOME configurado para caminho padrao: {default_cuda}")
        except Exception as e:
            print(f"[AVISO] Erro ao configurar CUDA_HOME: {e}")

# Adicionar CUDA bin ao PATH se CUDA_HOME estiver configurado
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home:
    cuda_bin = os.path.join(cuda_home, "bin")
    if os.path.exists(cuda_bin):
        current_path = os.environ.get("PATH", "")
        if cuda_bin not in current_path:
            os.environ["PATH"] = cuda_bin + os.pathsep + current_path
            print(f"[INFO] CUDA bin adicionado ao PATH: {cuda_bin}")

# Adicionar paths - IMPORTANTE: Kiss3DGen precisa estar no path
kiss3dgen_path = project_root / "Kiss3DGen"
if not kiss3dgen_path.exists():
    print(f"[ERRO] Kiss3DGen nao encontrado em: {kiss3dgen_path}")
    print(f"[INFO] Certifique-se de que o diretorio Kiss3DGen existe")
    sys.exit(1)

sys.path.insert(0, str(project_root))

import argparse
import shutil

from kiss3d_utils_local import TMP_DIR, OUT_DIR, ORIGINAL_WORKDIR, ensure_hf_token
from kiss3d_wrapper_local import init_wrapper_from_config, run_image_to_3d

def main():
    parser = argparse.ArgumentParser(description="Pipeline IMAGE TO 3D - Kiss3DGen")
    parser.add_argument("--input", type=str, required=True, help="Caminho para imagem de input")
    parser.add_argument("--output", type=str, default="data/outputs/kiss3dgen", help="Diretorio de saida")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline/pipeline_config/default.yaml",
        help="Caminho para config YAML (relativo ao diretorio Kiss3DGen)",
    )
    parser.add_argument("--enable-redux", action="store_true", default=True, help="Habilitar Redux")
    parser.add_argument("--use-mv-rgb", action="store_true", default=True, help="Usar RGB multiview")
    parser.add_argument("--use-controlnet", action="store_true", default=True, help="Usar ControlNet")
    
    args = parser.parse_args()
    
    original_cwd = os.getcwd()

    # Converter caminhos para absolutos
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    args.input = str(input_path.resolve())
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    args.output = str(output_path.resolve())
    
    # Ajustar caminho do config relativo ao Kiss3DGen
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = (kiss3dgen_path / config_path).resolve()
        if candidate.exists():
            args.config = str(candidate)
        else:
            default_config = kiss3dgen_path / "pipeline" / "pipeline_config" / "default.yaml"
            if default_config.exists():
                args.config = str(default_config.resolve())
            else:
                print(f"[ERRO] Config nao encontrado: {args.config}")
                sys.exit(1)
    else:
        args.config = str(config_path.resolve())
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    print("=" * 60)
    print("Pipeline IMAGE TO 3D - Kiss3DGen")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print(f"Redux: {args.enable_redux}")
    print(f"Use MV RGB: {args.use_mv_rgb}")
    print(f"Use ControlNet: {args.use_controlnet}")
    print("=" * 60)

    print("\n[0/4] Validando credenciais HuggingFace...")
    if ensure_hf_token(project_root / ".env"):
        print("[OK] Token HuggingFace carregado.")
    else:
        print("[AVISO] Token HuggingFace nao configurado; repositorios privados podem falhar.")
    
    # Verificar se arquivo existe
    if not os.path.exists(args.input):
        print(f"[ERRO] Arquivo nao encontrado: {args.input}")
        return
    
    # Verificar se config existe (agora relativo ao diretorio Kiss3DGen onde estamos)
    if not os.path.exists(args.config):
        print(f"[ERRO] Config nao encontrado: {args.config}")
        print(f"[INFO] Tentando usar config padrao do Kiss3DGen")
        default_config = Path('pipeline/pipeline_config/default.yaml')
        if default_config.exists():
            args.config = 'pipeline/pipeline_config/default.yaml'
        else:
            print(f"[ERRO] Config padrao tambem nao encontrado!")
            return
    
    # Inicializar wrapper do Kiss3DGen
    print("\n[1/4] Inicializando pipeline Kiss3DGen...")
    try:
        k3d_wrapper = init_wrapper_from_config(args.config)
        print("[OK] Pipeline inicializado")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Limpar diretorio temporario
    print("\n[2/4] Limpando diretorio temporario...")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    print("[OK] Diretorio limpo")
    
    # Executar pipeline IMAGE TO 3D
    print("\n[3/4] Executando pipeline IMAGE TO 3D...")
    print(f"Processando: {args.input}")
    try:
        gen_save_path, recon_mesh_path = run_image_to_3d(
            k3d_wrapper, 
            args.input,
            enable_redux=args.enable_redux,
            use_mv_rgb=args.use_mv_rgb,
            use_controlnet=args.use_controlnet
        )
        print(f"[OK] Pipeline executado com sucesso")
        print(f"  - Bundle image: {gen_save_path}")
        print(f"  - Mesh: {recon_mesh_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao executar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Copiar resultados para diretorio de saida
    print("\n[4/4] Copiando resultados...")
    try:
        input_name = Path(args.input).stem
        
        # Copiar bundle image
        bundle_output = os.path.join(args.output, f"{input_name}_3d_bundle.png")
        if os.path.exists(gen_save_path):
            shutil.copy2(gen_save_path, bundle_output)
            print(f"[OK] Bundle image copiado: {bundle_output}")
        
        # Copiar mesh
        mesh_output = os.path.join(args.output, f"{input_name}.glb")
        if os.path.exists(recon_mesh_path):
            shutil.copy2(recon_mesh_path, mesh_output)
            print(f"[OK] Mesh copiado: {mesh_output}")
        
        print(f"\n[OK] Resultados salvos em: {args.output}")
        
    except Exception as e:
        print(f"[ERRO] Falha ao copiar resultados: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Pipeline concluido!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    finally:
        os.chdir(str(ORIGINAL_WORKDIR))
