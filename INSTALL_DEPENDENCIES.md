# Guia de Instalação de Dependências

## ⚠️ MIGRAÇÃO PARA PYTHON 3.11

**Recomendado**: Migrar para Python 3.11 com CUDA para melhor compatibilidade.

Veja `MIGRACAO_PYTHON_3.11.md` para guia completo de migração.

## ⚠️ SITUAÇÃO ATUAL (Python 3.13)

Você está usando:
- **Python 3.13** (Pytorch3D não suporta)
- **PyTorch 2.9.0+cpu** (sem CUDA)

### O que isso significa:

1. **Pytorch3D**: Não pode ser instalado (Python 3.13 + sem CUDA)
2. **Modelos grandes (Flux, Zero123)**: Vão rodar em CPU (muito lento)
3. **Pipeline**: Funciona, mas com fallbacks e mais lento

## ✅ SOLUÇÃO RECOMENDADA

### Opção 1: Instalar PyTorch com CUDA (Recomendado)

```bash
# Desinstalar PyTorch CPU
pip uninstall torch torchvision

# Instalar PyTorch com CUDA 12.1 (para RTX 3060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verificar:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Opção 2: Continuar com CPU (Funciona, mas lento)

O pipeline funciona, mas:
- Modelos de difusão serão muito lentos em CPU
- Use apenas para testes pequenos
- Para produção, CUDA é essencial

## Dependências Básicas (Funcionam sem CUDA)

```bash
# Essenciais
pip install torch torchvision numpy pillow einops trimesh

# Para modelos de difusão (funciona em CPU, mas lento)
pip install diffusers transformers accelerate
```

## Dependências Avançadas (Requerem CUDA)

### Pytorch3D (OPCIONAL - apenas para renderização avançada)

**Status**: Não disponível para Python 3.13 + sem CUDA

**Solução**: Pipeline usa fallback simples (funciona, qualidade menor)

## Verificar Instalação

```bash
# Verificar PyTorch e CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Verificar Diffusers
python -c "import diffusers; print('Diffusers OK')"

# Verificar se pipeline funciona
python -c "from mesh3d_generator.pipeline.image_to_3d_pipeline import ImageTo3DPipeline; print('Pipeline OK')"
```

## Modelos do HuggingFace

Serão baixados automaticamente na primeira execução:
- Zero123++: `sudo-ai/zero123plus-v1.2` (~5GB)
- Flux: `black-forest-labs/FLUX.1-dev` (~23GB)
- ControlNet: `InstantX/FLUX.1-dev-Controlnet-Union` (~2GB)
- Redux: `black-forest-labs/FLUX.1-Redux-dev` (~23GB)

**Nota**: Modelos grandes podem demorar para baixar e precisam de espaço em disco.

## Notas Importantes

1. **Pytorch3D é OPCIONAL**: Pipeline funciona sem ele
2. **Python 3.13**: Funciona, mas sem Pytorch3D
3. **CPU vs CUDA**: CPU funciona, mas é muito mais lento
4. **Fallbacks**: Cada módulo tem fallbacks se dependências não estiverem disponíveis
5. **RTX 3060 12GB**: Se tiver CUDA instalado, pipeline usa GPU automaticamente

## Próximos Passos

1. **Se tiver RTX 3060**: Instalar PyTorch com CUDA (Opção 1 acima)
2. **Se não tiver CUDA**: Continuar com CPU (funciona, mas lento)
3. **Testar pipeline**: Executar script e verificar se funciona
