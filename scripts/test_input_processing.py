#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para validar o preprocessamento de imagem e carregamento de modelo
Garante que ambos os inputs estão sendo interpretados corretamente
"""

import sys
from pathlib import Path
from PIL import Image
import trimesh

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh3d_generator.preprocessing.image_preprocessor import preprocess_input_image
from scripts.run_3d_pipeline import load_mesh_properly, find_best_input_image, find_original_mesh


def test_image_preprocessing(model_name: str, dataset_path: Path):
    """Testa o preprocessamento de imagem"""
    print(f"\n{'='*60}")
    print(f"TESTE: Preprocessamento de Imagem - {model_name}")
    print(f"{'='*60}")
    
    model_dir = dataset_path / "models" / model_name
    if not model_dir.exists():
        print(f"[ERRO] Diretorio nao encontrado: {model_dir}")
        return False
    
    # Encontrar imagem
    input_image_path = find_best_input_image(model_dir)
    if not input_image_path:
        print(f"[ERRO] Imagem nao encontrada")
        return False
    
    print(f"[OK] Imagem encontrada: {input_image_path.name}")
    
    # Carregar imagem original
    original_image = Image.open(input_image_path)
    print(f"[INFO] Imagem original: {original_image.size} ({original_image.mode})")
    
    # Preprocessar
    try:
        preprocessed = preprocess_input_image(
            original_image,
            target_size=(512, 512),
            remove_bg=True,
            resize_ratio=0.85
        )
        print(f"[OK] Preprocessamento concluido")
        print(f"[INFO] Imagem preprocessada: {preprocessed.size} ({preprocessed.mode})")
        
        # Salvar para inspeção
        output_dir = Path("data/outputs/test_input_processing") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessed.save(output_dir / "preprocessed.png")
        original_image.save(output_dir / "original.png")
        print(f"[OK] Imagens salvas em: {output_dir}")
        
        return True
    except Exception as e:
        print(f"[ERRO] Falha no preprocessamento: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mesh_loading(model_name: str, dataset_path: Path):
    """Testa o carregamento de mesh"""
    print(f"\n{'='*60}")
    print(f"TESTE: Carregamento de Mesh - {model_name}")
    print(f"{'='*60}")
    
    model_dir = dataset_path / "models" / model_name
    if not model_dir.exists():
        print(f"[ERRO] Diretorio nao encontrado: {model_dir}")
        return False
    
    # Encontrar mesh
    mesh_path = find_original_mesh(model_dir)
    if not mesh_path:
        print(f"[ERRO] Mesh nao encontrado")
        return False
    
    print(f"[OK] Mesh encontrado: {mesh_path.name}")
    
    # Carregar mesh
    try:
        mesh = load_mesh_properly(
            mesh_path,
            use_largest_component=True,
            make_watertight=True
        )
        
        print(f"\n[OK] Mesh carregado com sucesso!")
        print(f"[INFO] Vertices: {len(mesh.vertices)}")
        print(f"[INFO] Faces: {len(mesh.faces)}")
        print(f"[INFO] Watertight: {mesh.is_watertight}")
        print(f"[INFO] Volume: {mesh.volume:.6f}")
        print(f"[INFO] Bounds: {mesh.bounds}")
        print(f"[INFO] Centroid: {mesh.centroid}")
        
        # Verificar qualidade
        if mesh.is_watertight:
            print(f"[OK] Mesh e watertight (fechado)")
        else:
            print(f"[AVISO] Mesh nao e watertight")
        
        if mesh.volume > 0:
            print(f"[OK] Mesh tem volume valido")
        else:
            print(f"[AVISO] Mesh tem volume zero ou negativo")
        
        # Salvar mesh processado para inspeção
        output_dir = Path("data/outputs/test_input_processing") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mesh.export(str(output_dir / "mesh_processed.obj"))
        print(f"[OK] Mesh processado salvo em: {output_dir / 'mesh_processed.obj'}")
        
        return True
    except Exception as e:
        print(f"[ERRO] Falha no carregamento: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    dataset_path = Path("data/raw/gazebo_dataset")
    
    if not dataset_path.exists():
        print(f"[ERRO] Dataset nao encontrado: {dataset_path}")
        return 1
    
    # Testar com o primeiro modelo disponível
    models_dir = dataset_path / "models"
    if not models_dir.exists():
        print(f"[ERRO] Diretorio de modelos nao encontrado")
        return 1
    
    model_names = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])
    if not model_names:
        print(f"[ERRO] Nenhum modelo encontrado")
        return 1
    
    # Testar primeiro modelo
    model_name = model_names[0]
    print(f"\nTestando modelo: {model_name}")
    
    # Testar preprocessamento
    img_ok = test_image_preprocessing(model_name, dataset_path)
    
    # Testar carregamento de mesh
    mesh_ok = test_mesh_loading(model_name, dataset_path)
    
    # Resumo
    print(f"\n{'='*60}")
    print(f"RESUMO DOS TESTES")
    print(f"{'='*60}")
    print(f"Preprocessamento de imagem: {'OK' if img_ok else 'FALHOU'}")
    print(f"Carregamento de mesh: {'OK' if mesh_ok else 'FALHOU'}")
    
    if img_ok and mesh_ok:
        print(f"\n[OK] Todos os testes passaram!")
        return 0
    else:
        print(f"\n[ERRO] Alguns testes falharam")
        return 1


if __name__ == "__main__":
    sys.exit(main())

