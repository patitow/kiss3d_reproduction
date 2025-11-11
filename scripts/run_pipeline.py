#!/usr/bin/env python3
"""
Script principal para executar o pipeline completo de geração de malhas 3D
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh3d_generator import (
    TextToImageGenerator,
    NormalMapGenerator,
    InstantMeshInitializer,
    MeshRefiner,
    TextGenerator
)


def main():
    """
    Executa o pipeline completo:
    1. Gera imagem a partir de texto
    2. Gera normal map
    3. Inicializa malha
    4. Refina malha com LLM
    """
    # Texto inicial
    initial_text = "Uma cadeira moderna de madeira com encosto alto"
    
    print("🚀 Iniciando pipeline de geração de malha 3D...")
    
    # Etapa 0: Texto para Imagem
    print("\n📝 Etapa 0: Gerando imagem a partir do texto...")
    text_to_image = TextToImageGenerator()
    # image = text_to_image.generate(initial_text)  # TODO: Descomentar quando implementado
    
    # Etapa 1: Normal Maps
    print("\n🗺️  Etapa 1: Gerando normal map...")
    normal_generator = NormalMapGenerator()
    # normal_map = normal_generator.generate(image)  # TODO: Descomentar quando implementado
    
    # Etapa 2: Inicialização da Malha
    print("\n🔷 Etapa 2: Inicializando malha...")
    mesh_initializer = InstantMeshInitializer()  # ou LRMInitializer()
    # mesh = mesh_initializer.initialize(image, normal_map)  # TODO: Descomentar quando implementado
    
    # Etapa 3: Refinamento com LLM
    print("\n✨ Etapa 3: Refinando malha com texto detalhado...")
    text_generator = TextGenerator()
    # detailed_text = text_generator.generate_detailed_description(initial_text)  # TODO: Descomentar quando implementado
    
    mesh_refiner = MeshRefiner()
    # refined_mesh = mesh_refiner.refine(mesh, detailed_text, normal_map)  # TODO: Descomentar quando implementado
    
    # Salvar resultado
    # refined_mesh.export("data/outputs/result.obj")  # TODO: Descomentar quando implementado
    
    print("\n✅ Pipeline concluído!")
    print("⚠️  Nota: Este é um template. Implemente os métodos conforme o planejamento.")


if __name__ == "__main__":
    main()

