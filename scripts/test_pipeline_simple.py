#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste simples para o pipeline de geração 3D
Testa apenas o pipeline básico sem comparação completa
"""

import sys
from pathlib import Path
from PIL import Image
import torch

# Adicionar paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh3d_generator.pipeline.image_to_3d_pipeline import ImageTo3DPipeline

def test_pipeline():
    """Testa o pipeline com uma imagem simples"""
    
    print("="*60)
    print("TESTE DO PIPELINE DE GERACAO 3D")
    print("="*60)
    
    # Verificar CUDA
    print(f"\n[INFO] CUDA disponivel: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Encontrar uma imagem de teste
    dataset_path = Path("data/raw/gazebo_dataset")
    images_dir = dataset_path / "images"
    
    # Procurar primeira imagem JPG disponível
    test_images = list(images_dir.glob("*_0.jpg"))
    if not test_images:
        print("[ERRO] Nenhuma imagem de teste encontrada")
        return False
    
    test_image_path = test_images[0]
    object_name = test_image_path.stem.replace("_0", "")
    
    print(f"\n[INFO] Testando com: {object_name}")
    print(f"[INFO] Imagem: {test_image_path}")
    
    # Carregar imagem
    try:
        input_image = Image.open(test_image_path)
        print(f"[OK] Imagem carregada: {input_image.size}")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar imagem: {e}")
        return False
    
    # Inicializar pipeline
    print(f"\n[INFO] Inicializando pipeline...")
    try:
        pipeline = ImageTo3DPipeline(
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            zero123_model=None,
            flux_model=None,
            lrm_model=None,
            caption_model=None
        )
        print("[OK] Pipeline inicializado")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Testar Passo 1: Reference Bundle Image
    print(f"\n[TESTE] Passo 1: Gerando reference bundle image...")
    try:
        reference_bundle, ref_path = pipeline.generate_reference_3d_bundle_image(
            input_image,
            use_mv_rgb=True,
            seed=42
        )
        print(f"[OK] Reference bundle gerado: {reference_bundle.shape}")
        if ref_path:
            print(f"[OK] Salvo em: {ref_path}")
    except Exception as e:
        print(f"[ERRO] Falha no Passo 1: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("TESTE CONCLUIDO COM SUCESSO!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

