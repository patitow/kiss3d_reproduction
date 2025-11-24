# Guia de Migração para Python 3.11 com CUDA

## 🎯 Status do Sistema

✅ **GPU Detectada**: NVIDIA GeForce RTX 3060 (12GB VRAM)  
✅ **CUDA Disponível**: Versão 13.0 (drivers instalados)  
⚠️ **Python Atual**: 3.13.3 (não suporta Pytorch3D)  
➡️ **Python Alvo**: 3.11 (suporta tudo)

## 🚀 Quick Start (Recomendado)

**Use o script automatizado:**
```powershell
.\setup_python311.ps1
```

Ou veja `QUICK_START_PYTHON311.md` para instruções rápidas.

---

## Passo 1: Instalar Python 3.11

### Windows

1. **Baixar Python 3.11**:
   - Acesse: https://www.python.org/downloads/release/python-3110/
   - Baixe "Windows installer (64-bit)"
   - Execute o instalador
   - ✅ **IMPORTANTE**: Marque "Add Python to PATH"

2. **Verificar instalação**:
   ```bash
   python3.11 --version
   # Deve mostrar: Python 3.11.0 (ou similar)
   ```
   
   **Se não funcionar**, use o caminho completo:
   ```bash
   # Geralmente em:
   C:\Users\SeuUsuario\AppData\Local\Programs\Python\Python311\python.exe
   ```

## Passo 2: Criar Novo Ambiente Virtual

```bash
# Criar ambiente virtual com Python 3.11
python3.11 -m venv mesh3d-generator-py3.11

# Ativar ambiente (Windows PowerShell)
.\mesh3d-generator-py3.11\Scripts\Activate.ps1

# Ou (Windows CMD)
.\mesh3d-generator-py3.11\Scripts\activate.bat
```

## Passo 3: Instalar PyTorch com CUDA

```bash
# Verificar se CUDA está disponível no sistema
nvidia-smi

# Instalar PyTorch com CUDA (para RTX 3060)
# Você tem CUDA 13.0, mas PyTorch suporta até CUDA 12.4
# Use CUDA 12.1 que é compatível com RTX 3060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar instalação
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Deve mostrar:**
```
PyTorch: 2.x.x+cu121
CUDA disponível: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3060
```

## Passo 4: Instalar Dependências do Projeto

```bash
# Dependências básicas
pip install numpy pillow einops trimesh

# Para modelos de difusão
pip install diffusers transformers accelerate

# Para renderização avançada (OPCIONAL mas recomendado)
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html

# Outras dependências
pip install omegaconf rembg
```

## Passo 5: Verificar Instalação Completa

```bash
# Script de verificação
python -c "
import torch
import sys
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA disponível:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

try:
    import diffusers
    print('Diffusers:', diffusers.__version__)
except ImportError:
    print('Diffusers: NÃO INSTALADO')

try:
    import pytorch3d
    print('Pytorch3D:', pytorch3d.__version__)
except ImportError:
    print('Pytorch3D: NÃO INSTALADO (opcional)')

try:
    import trimesh
    print('Trimesh:', trimesh.__version__)
except ImportError:
    print('Trimesh: NÃO INSTALADO')
"
```

## Passo 6: Testar Pipeline

```bash
# Testar importação do pipeline
python -c "from mesh3d_generator.pipeline.image_to_3d_pipeline import ImageTo3DPipeline; print('Pipeline OK')"

# Testar inicialização
python -c "
from mesh3d_generator.pipeline.image_to_3d_pipeline import ImageTo3DPipeline
pipeline = ImageTo3DPipeline(device='cuda:0')
print('Pipeline inicializado com sucesso!')
"
```

## Troubleshooting

### Erro: "CUDA não disponível"

1. Verificar driver NVIDIA:
   ```bash
   nvidia-smi
   ```

2. Se não funcionar, instalar/atualizar drivers:
   - https://www.nvidia.com/Download/index.aspx

3. Verificar versão do CUDA Toolkit:
   - RTX 3060 suporta CUDA 11.0+
   - PyTorch com cu121 requer CUDA 12.1

### Erro: "Pytorch3D não encontrado"

- Pytorch3D é **OPCIONAL**
- Pipeline funciona sem ele usando fallbacks
- Se quiser instalar, verifique versão do PyTorch primeiro

### Erro: "Out of memory"

- RTX 3060 tem 12GB VRAM
- Modelos grandes (Flux ~23GB) podem não caber
- Solução: Usar modelos quantizados ou carregar sob demanda

## Próximos Passos

1. ✅ Ambiente Python 3.11 criado
2. ✅ PyTorch com CUDA instalado
3. ✅ Dependências instaladas
4. ⏭️ Testar pipeline completo
5. ⏭️ Baixar modelos do HuggingFace (automático na primeira execução)

## Comandos Úteis

```bash
# Desativar ambiente
deactivate

# Ativar ambiente novamente
.\mesh3d-generator-py3.11\Scripts\Activate.ps1

# Verificar espaço em disco (modelos são grandes)
# Flux: ~23GB, Zero123: ~5GB, ControlNet: ~2GB
# Total: ~30GB+ necessário
```

