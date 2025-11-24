# Quick Start - Python 3.11 com CUDA

## 🚀 Instalação Rápida

### Opção 1: Script Automatizado (Recomendado)

**PowerShell:**
```powershell
.\setup_python311.ps1
```

**CMD:**
```cmd
setup_python311.bat
```

### Opção 2: Manual

```bash
# 1. Criar ambiente virtual
python3.11 -m venv mesh3d-generator-py3.11

# 2. Ativar (PowerShell)
.\mesh3d-generator-py3.11\Scripts\Activate.ps1

# 3. Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verificar CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 5. Instalar dependências
pip install numpy pillow einops trimesh diffusers transformers accelerate omegaconf rembg

# 6. (Opcional) Pytorch3D
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html
```

## ✅ Verificação

```bash
python -c "
import torch
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"
```

## 🎯 Próximo Passo

Testar o pipeline:
```bash
python scripts/run_3d_pipeline.py --max-objects 1 --output data/outputs/test
```

