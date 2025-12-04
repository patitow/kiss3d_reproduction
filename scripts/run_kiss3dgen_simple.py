#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline IMAGE TO 3D usando Kiss3DGen diretamente
Executa o exemplo original do Kiss3DGen
"""

import sys
import os
from pathlib import Path

# Mudar para diretorio do Kiss3DGen
kiss3dgen_path = Path(__file__).parent.parent / "Kiss3DGen"
if not kiss3dgen_path.exists():
    print(f"[ERRO] Kiss3DGen nao encontrado em: {kiss3dgen_path}")
    sys.exit(1)

os.chdir(str(kiss3dgen_path))
sys.path.insert(0, str(kiss3dgen_path))

import argparse
from argparse import BooleanOptionalAction
from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_image_to_3d
from pipeline.utils import TMP_DIR, OUT_DIR
import shutil

def main():
    parser = argparse.ArgumentParser(description='Pipeline IMAGE TO 3D - Kiss3DGen')
    parser.add_argument('--input', type=str, required=True, help='Caminho para imagem de input')
    parser.add_argument('--output', type=str, default='../data/outputs/kiss3dgen', help='Diretorio de saida')
    parser.add_argument('--config', type=str, default='pipeline/pipeline_config/default.yaml', 
                       help='Caminho para config YAML')
    parser.add_argument(
        '--enable-redux',
        action=BooleanOptionalAction,
        default=True,
        help='Habilitar Redux (use --no-enable-redux para desativar)',
    )
    parser.add_argument(
        '--use-mv-rgb',
        action=BooleanOptionalAction,
        default=True,
        help='Usar RGB multiview (use --no-use-mv-rgb para desativar)',
    )
    parser.add_argument(
        '--use-controlnet',
        action=BooleanOptionalAction,
        default=True,
        help='Usar ControlNet (use --no-use-controlnet para desativar)',
    )
    args = parser.parse_args()

    # Criar diretorios
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Pipeline IMAGE TO 3D - Kiss3DGen")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print("=" * 60)

    # Verificar se arquivo existe
    if not os.path.exists(args.input):
        print(f"[ERRO] Arquivo nao encontrado: {args.input}")
        return

    # Verificar se config existe
    if not os.path.exists(args.config):
        print(f"[ERRO] Config nao encontrado: {args.config}")
        return

    # Limpar diretorio temporario
    print("\n[1/4] Limpando diretorio temporario...")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    print("[OK] Diretorio limpo")
    
    # Inicializar wrapper do Kiss3DGen
    print("\n[2/4] Inicializando pipeline Kiss3DGen...")
    try:
        k3d_wrapper = init_wrapper_from_config(args.config)
        print("[OK] Pipeline inicializado")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
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
    
    # Copiar resultados
    print("\n[4/4] Copiando resultados...")
    try:
        input_name = Path(args.input).stem
        
        # Copiar bundle image
        bundle_output = os.path.join(args.output, f"{input_name}_3d_bundle.png")
        if os.path.exists(gen_save_path):
            shutil.copy2(gen_save_path, bundle_output)
            print(f"[OK] Bundle image: {bundle_output}")
        
        # Copiar mesh
        mesh_output = os.path.join(args.output, f"{input_name}.glb")
        if os.path.exists(recon_mesh_path):
            shutil.copy2(recon_mesh_path, mesh_output)
            print(f"[OK] Mesh: {mesh_output}")
        
        print(f"\n[OK] Resultados salvos em: {args.output}")
        
    except Exception as e:
        print(f"[ERRO] Falha ao copiar resultados: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Pipeline concluido!")
    print("=" * 60)

if __name__ == '__main__':
    main()

