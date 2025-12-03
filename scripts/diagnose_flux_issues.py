#!/usr/bin/env python3
"""
Script de diagnóstico para problemas no pipeline Flux
"""
import sys
from pathlib import Path
import torch
import yaml
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kiss3d_wrapper_local import init_wrapper_from_config

def diagnose_flux():
    """Diagnostica problemas no pipeline Flux"""
    print("=" * 80)
    print("DIAGNÓSTICO DO PIPELINE FLUX")
    print("=" * 80)
    
    config_path = project_root / "pipeline_config" / "default.yaml"
    
    print(f"\n1. Carregando configuração: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    
    print(f"   - Flux device: {config['flux'].get('device', 'cpu')}")
    print(f"   - Flux dtype: {config['flux'].get('flux_dtype', 'bf16')}")
    print(f"   - ControlNet: {config['flux'].get('controlnet', 'N/A')}")
    print(f"   - Redux: {config['flux'].get('redux', 'N/A')}")
    
    print(f"\n2. Verificando GPU...")
    if torch.cuda.is_available():
        print(f"   - CUDA disponível: Sim")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   - VRAM alocada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   - VRAM reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("   - CUDA disponível: NÃO")
        return
    
    print(f"\n3. Carregando wrapper...")
    try:
        k3d_wrapper = init_wrapper_from_config(
            str(config_path),
            fast_mode=True,
            load_controlnet=True,
            load_redux=True,
            pipeline_mode="flux",
        )
        print("   ✓ Wrapper carregado com sucesso")
    except Exception as e:
        print(f"   ✗ ERRO ao carregar wrapper: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n4. Testando geração de bundle Flux...")
    # Usar uma imagem de teste
    test_image_path = project_root / "outputs" / "tmp" / "44d5dd5b-dc5f-468d-9874-b0e7399b1bf7_input_image.png"
    if not test_image_path.exists():
        print(f"   ✗ Imagem de teste não encontrada: {test_image_path}")
        return
    
    test_image = Image.open(test_image_path)
    test_caption = "A 3D object with multiple views"
    
    try:
        print("   - Gerando bundle...")
        bundle_tensor, bundle_path = k3d_wrapper.generate_flux_bundle(
            input_image=test_image,
            caption=test_caption,
            enable_redux=True,
            use_controlnet=True,
        )
        print(f"   ✓ Bundle gerado com sucesso")
        print(f"   - Shape: {bundle_tensor.shape}")
        print(f"   - Path: {bundle_path}")
    except Exception as e:
        print(f"   ✗ ERRO ao gerar bundle: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n5. Testando reconstrução 3D...")
    try:
        mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(
            bundle_tensor,
            reconstruction_stage2_steps=50,
            save_intermediate_results=True,
        )
        print(f"   ✓ Mesh gerado com sucesso")
        print(f"   - Path: {mesh_path}")
    except Exception as e:
        print(f"   ✗ ERRO ao reconstruir mesh: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO CONCLUÍDO - TUDO OK!")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_flux()


