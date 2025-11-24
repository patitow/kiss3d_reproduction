#!/usr/bin/env python3
"""
Script de verificação do ambiente Python 3.11 com CUDA
"""

import sys

print("=" * 60)
print("VERIFICAÇÃO DO AMBIENTE")
print("=" * 60)
print()

# Python
print(f"Python: {sys.version}")
print(f"Versão: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info < (3, 11):
    print("⚠️  AVISO: Python 3.11+ recomendado")
elif sys.version_info >= (3, 13):
    print("⚠️  AVISO: Python 3.13 pode ter problemas com Pytorch3D")
else:
    print("✅ Python OK")
print()

# PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponível: {cuda_available}")
    
    if cuda_available:
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"✅ VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"✅ Compute Capability: {props.major}.{props.minor}")
    else:
        print("❌ CUDA não disponível - modelos serão muito lentos em CPU")
    
except ImportError:
    print("❌ PyTorch não instalado")
print()

# Diffusers
try:
    import diffusers
    print(f"✅ Diffusers: {diffusers.__version__}")
except ImportError:
    print("⚠️  Diffusers não instalado (necessário para Zero123 e Flux)")
print()

# Transformers
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("⚠️  Transformers não instalado")
print()

# Pytorch3D
try:
    import pytorch3d
    print(f"✅ Pytorch3D: {pytorch3d.__version__}")
except ImportError:
    print("⚠️  Pytorch3D não instalado (opcional - pipeline funciona sem ele)")
print()

# Trimesh
try:
    import trimesh
    print(f"✅ Trimesh: {trimesh.__version__}")
except ImportError:
    print("❌ Trimesh não instalado (necessário)")
print()

# Pipeline
try:
    from mesh3d_generator.pipeline.image_to_3d_pipeline import ImageTo3DPipeline
    print("✅ Pipeline importado com sucesso")
    
    # Testar inicialização
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = ImageTo3DPipeline(device=device)
    print(f"✅ Pipeline inicializado com device: {device}")
except Exception as e:
    print(f"❌ Erro ao importar pipeline: {e}")
print()

print("=" * 60)
print("RESUMO")
print("=" * 60)

if sys.version_info >= (3, 11) and sys.version_info < (3, 13):
    print("✅ Python: OK")
else:
    print("⚠️  Python: Considere migrar para 3.11")

if 'torch' in sys.modules and torch.cuda.is_available():
    print("✅ CUDA: OK")
else:
    print("❌ CUDA: Não disponível - instale PyTorch com CUDA")

try:
    import diffusers
    import transformers
    print("✅ Modelos de difusão: OK")
except:
    print("⚠️  Modelos de difusão: Instale diffusers e transformers")

print()
print("Para instalação completa, veja: MIGRACAO_PYTHON_3.11.md")

