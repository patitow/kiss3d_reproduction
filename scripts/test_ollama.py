#!/usr/bin/env python3
"""
Script de teste para verificar a integração com Ollama
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh3d_generator import TextGenerator


def test_ollama_connection():
    """Testa conexão com Ollama."""
    print("🔍 Testando conexão com Ollama...")
    generator = TextGenerator()
    
    # Listar modelos disponíveis
    print("\n📋 Modelos disponíveis no Ollama:")
    models = generator.list_available_models()
    if models:
        for model in models:
            print(f"   - {model}")
    else:
        print("   ⚠️  Nenhum modelo encontrado")
        print("   💡 Instale modelos com: ollama pull llama3.2")
        print("   💡 Para modelo multimodal: ollama pull llava")
    
    return generator


def test_text_generation():
    """Testa geração de texto."""
    print("\n📝 Testando geração de texto detalhado...")
    generator = TextGenerator()
    
    initial_text = "Uma cadeira moderna de madeira"
    print(f"\nTexto inicial: {initial_text}")
    
    try:
        detailed = generator.generate_detailed_description(initial_text)
        print(f"\n✅ Texto detalhado gerado:")
        print(f"{detailed}")
    except Exception as e:
        print(f"\n❌ Erro: {e}")


def test_prompt_generation():
    """Testa geração de prompt para text-to-image."""
    print("\n🎨 Testando geração de prompt para text-to-image...")
    generator = TextGenerator()
    
    initial_text = "Uma cadeira moderna"
    print(f"\nTexto inicial: {initial_text}")
    
    try:
        prompt = generator.generate_prompt_for_text_to_image(initial_text)
        print(f"\n✅ Prompt otimizado gerado:")
        print(f"{prompt}")
    except Exception as e:
        print(f"\n❌ Erro: {e}")


def test_multimodal():
    """Testa geração de descrição a partir de imagem."""
    print("\n🖼️  Testando geração de descrição a partir de imagem...")
    generator = TextGenerator()
    
    # Verificar se há uma imagem de teste
    test_image = Path("data/raw/test_image.jpg")
    if not test_image.exists():
        print(f"⚠️  Imagem de teste não encontrada: {test_image}")
        print("   Crie uma imagem de teste ou use uma existente")
        return
    
    try:
        description = generator.generate_from_image(test_image)
        print(f"\n✅ Descrição gerada:")
        print(f"{description}")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("   Certifique-se de que o modelo multimodal está instalado:")
        print("   ollama pull llava")


def main():
    """Executa todos os testes."""
    print("=" * 60)
    print("🧪 Testes de Integração com Ollama")
    print("=" * 60)
    
    # Teste 1: Conexão
    generator = test_ollama_connection()
    
    if not generator.list_available_models():
        print("\n⚠️  Nenhum modelo encontrado. Instale modelos primeiro:")
        print("   ollama pull llama3.2")
        print("   ollama pull llava  # Para modelos multimodais")
        return
    
    # Teste 2: Geração de texto
    test_text_generation()
    
    # Teste 3: Geração de prompt
    test_prompt_generation()
    
    # Teste 4: Multimodal (opcional)
    print("\n" + "=" * 60)
    response = input("Deseja testar geração a partir de imagem? (s/n): ")
    if response.lower() == 's':
        test_multimodal()
    
    print("\n" + "=" * 60)
    print("✅ Testes concluídos!")


if __name__ == "__main__":
    main()

