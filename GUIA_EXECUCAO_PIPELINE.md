# Guia de Execução - Pipeline Image-to-3D Kiss3DGen

## Status Atual

✅ **Completo:**
- Python 3.11.9
- PyTorch 2.5.1+cu121 com CUDA 12.1
- PyTorch3D 0.7.8 com suporte CUDA
- HuggingFace autenticado
- Scripts principais criados
- Configuração YAML presente

❌ **Falta:**
- Dependências: `opencv-python`, `pillow`
- Modelos: Zero123++, Flux, ControlNet, Redux
- Verificação final dos scripts

## Passos para Executar

### 1. Instalar Dependências Faltantes

```bash
# Ativar venv
.\mesh3d-generator-py3.11\Scripts\Activate.ps1

# Instalar dependências
pip install opencv-python pillow
```

Ou instalar todas de uma vez:
```bash
pip install -r requirements.txt
```

### 2. Baixar Modelos

Os modelos são grandes (vários GB). Execute:

```bash
python scripts/download_all_models.py
```

**Modelos necessários:**
- Zero123++ (sudo-ai/zero123plus-v1.1) - ~2GB
- Flux (Comfy-Org/flux1-schnell) - ~10GB
- ControlNet (InstantX/FLUX.1-dev-Controlnet-Union) - ~2GB
- Redux (black-forest-labs/FLUX.1-Redux-dev) - ~10GB (opcional)
- LRM (LTT/PRM) - ~1GB
- LoRAs do Kiss3DGen (LTT/Kiss3DGen) - ~500MB

**Tempo estimado:** 30-60 minutos (dependendo da conexão)

### 3. Verificar Prontidão

```bash
python scripts/check_pipeline_readiness.py
```

Deve mostrar tudo como "OK".

### 4. Executar Pipeline

#### Modo Básico (uma imagem):
```bash
python scripts/run_kiss3dgen_image_to_3d.py --input caminho/para/imagem.jpg
```

#### Modo Otimizado para 12GB VRAM:
```bash
python scripts/run_kiss3dgen_image_to_3d.py \
    --input caminho/para/imagem.jpg \
    --fast-mode \
    --disable-llm
```

#### Com métricas (se tiver ground-truth):
```bash
python scripts/run_kiss3dgen_image_to_3d.py \
    --input caminho/para/imagem.jpg \
    --gt-mesh caminho/para/ground_truth.obj \
    --metrics-out metrics.json
```

### 5. Parâmetros Importantes

- `--fast-mode`: Reduz qualidade mas economiza VRAM (recomendado para 12GB)
- `--disable-llm`: Desabilita refinamento de prompt (economiza RAM)
- `--enable-redux`: Habilita Redux (melhor qualidade, mais VRAM)
- `--use-controlnet`: Usa ControlNet (melhor controle, mais VRAM)
- `--use-mv-rgb`: Usa RGB multiview (melhor qualidade)

### 6. Otimizações para 12GB VRAM

O script `scripts/run_kiss3dgen_image_to_3d_optimized.py` já está configurado para:
- Quantização FP16 automática
- Carregamento/descarregamento segmentado de modelos
- Limpeza agressiva de memória
- Resoluções reduzidas

**Uso:**
```bash
python scripts/run_kiss3dgen_image_to_3d_optimized.py --input imagem.jpg
```

### 7. Verificar Resultados

Os resultados serão salvos em:
- `data/outputs/kiss3dgen/` - Diretório padrão
- Arquivos gerados:
  - `*_3d_bundle.png` - Bundle 3D gerado
  - `*.glb` - Malha 3D final
  - `*_metrics.json` - Métricas (se GT fornecido)
  - `summary.json` - Resumo da execução

## Métricas de Qualidade

Se você fornecer uma malha ground-truth, o pipeline calculará:
- **Chamfer Distance (L1)**: Distância média entre pontos
- **Chamfer Distance (L2)**: Distância quadrática média
- **F-Score**: Precisão e recall combinados
- **Normal Consistency**: Consistência das normais

## Troubleshooting

### Erro: "Out of memory"
- Use `--fast-mode`
- Use `--disable-llm`
- Desabilite `--enable-redux`
- Use o script otimizado

### Erro: "Model not found"
- Execute `python scripts/download_all_models.py`
- Verifique autenticação HuggingFace: `huggingface-cli login`

### Erro: "PyTorch3D not compiled with CUDA"
- Execute: `python scripts/install_pytorch3d_robust.bat`
- Aguarde 20-40 minutos para compilação

### Erro: "DLL load failed"
- Verifique se CUDA 12.1 está instalado
- Verifique se Visual Studio 2019 BuildTools está instalado
- Reinstale PyTorch3D

## Próximos Passos para Melhorar Qualidade

1. **Ajustar configuração YAML:**
   - Editar `Kiss3DGen/pipeline/pipeline_config/default.yaml`
   - Aumentar `num_inference_steps` (mais qualidade, mais tempo)
   - Ajustar `strength1` e `strength2` (controle de geração)

2. **Usar Redux:**
   - Habilitar `--enable-redux` (requer mais VRAM)
   - Melhora qualidade do bundle 3D

3. **Usar ControlNet:**
   - Habilitar `--use-controlnet`
   - Melhor controle sobre a geração

4. **Ajustar resoluções:**
   - Editar config para aumentar resolução do Flux
   - Mais detalhes, mais VRAM necessário

## Checklist Final

Antes de executar, verifique:
- [ ] PyTorch3D funcionando (`python -c "from pytorch3d import _C"`)
- [ ] Todos os modelos baixados
- [ ] Dependências instaladas
- [ ] HuggingFace autenticado
- [ ] Imagem de teste disponível
- [ ] VRAM suficiente (ou usar `--fast-mode`)

## Comandos Rápidos

```bash
# Verificar tudo
python scripts/check_pipeline_readiness.py

# Baixar modelos
python scripts/download_all_models.py

# Executar pipeline básico
python scripts/run_kiss3dgen_image_to_3d.py --input imagem.jpg --fast-mode

# Executar pipeline otimizado
python scripts/run_kiss3dgen_image_to_3d_optimized.py --input imagem.jpg
```

