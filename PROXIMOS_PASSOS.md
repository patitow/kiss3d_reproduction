# Próximos Passos - Pipeline Image-to-3D Kiss3DGen

## ✅ Status Atual (Verificado)

### Ambiente e Dependências
- ✅ Python 3.11.9
- ✅ PyTorch 2.5.1+cu121 com CUDA 12.1
- ✅ PyTorch3D 0.7.8 com suporte CUDA
- ✅ GPU: NVIDIA GeForce RTX 3060
- ✅ Todas as dependências instaladas (diffusers, transformers, opencv, etc.)
- ✅ HuggingFace autenticado (PatitowPoli)
- ✅ Scripts principais prontos

### Modelos Locais (já baixados)
- ✅ Flux LoRA: `checkpoint/flux_lora/rgb_normal.safetensors`
- ✅ Flux LoRA Redux: `checkpoint/flux_lora/rgb_normal_redux.safetensors`
- ✅ LRM: `checkpoint/lrm/final_ckpt.ckpt`
- ✅ Zero123++ UNet: `checkpoint/zero123++/flexgen.ckpt`

## ❌ O Que Falta

### Modelos Grandes (Cache do HuggingFace)
Os seguintes modelos precisam ser baixados para o cache do HuggingFace (serão baixados automaticamente na primeira execução, mas podemos baixar antecipadamente):

1. **Zero123++ Completo** (`sudo-ai/zero123plus-v1.1`)
   - Tamanho: ~2GB
   - Status: Não encontrado no cache
   - Necessário para: Geração de multiview

2. **Flux Schnell** (`Comfy-Org/flux1-schnell`)
   - Tamanho: ~10GB
   - Status: Não encontrado no cache
   - Necessário para: Geração de bundle 3D

3. **ControlNet** (`InstantX/FLUX.1-dev-Controlnet-Union`)
   - Tamanho: ~2GB
   - Status: Não encontrado no cache
   - Necessário para: Controle de geração (opcional mas recomendado)

4. **Redux** (`black-forest-labs/FLUX.1-Redux-dev`)
   - Tamanho: ~10GB
   - Status: Não encontrado no cache
   - Necessário para: Melhor qualidade (opcional)

**Total estimado:** ~24GB

## 🚀 Plano de Execução

### Opção 1: Download Antecipado (Recomendado)
Baixar todos os modelos antes de executar o pipeline:

```powershell
cd "D:\.Faculdade\Visao_Computacional\2025_2"
.\mesh3d-generator-py3.11\Scripts\python.exe scripts\download_all_models.py
```

**Tempo estimado:** 30-60 minutos (dependendo da conexão)

### Opção 2: Download Automático
Os modelos serão baixados automaticamente na primeira execução, mas isso pode:
- Aumentar o tempo da primeira execução
- Causar timeout se a conexão for lenta
- Consumir mais tempo de GPU ociosa

### Opção 3: Download Seletivo
Baixar apenas os modelos essenciais (Zero123++, Flux, ControlNet) e pular Redux:

```powershell
# Baixar apenas modelos essenciais
.\mesh3d-generator-py3.11\Scripts\python.exe -c "
from huggingface_hub import snapshot_download
snapshot_download('sudo-ai/zero123plus-v1.1')
snapshot_download('Comfy-Org/flux1-schnell')
snapshot_download('InstantX/FLUX.1-dev-Controlnet-Union')
"
```

## 📋 Checklist Antes de Executar

- [ ] Modelos baixados (Zero123++, Flux, ControlNet)
- [ ] Imagem de teste disponível
- [ ] Espaço em disco suficiente (~30GB livres)
- [ ] VRAM disponível (RTX 3060 tem 12GB - usar `--fast-mode`)

## 🎯 Execução do Pipeline

### 1. Preparar Imagem de Teste

Coloque uma imagem de teste em `data/inputs/` ou use uma imagem existente.

### 2. Executar Pipeline (Modo Fast - Recomendado para 12GB VRAM)

```powershell
cd "D:\.Faculdade\Visao_Computacional\2025_2"
.\mesh3d-generator-py3.11\Scripts\python.exe scripts\run_kiss3dgen_image_to_3d.py `
    --input "data/inputs/exemplo.jpg" `
    --output "data/outputs/teste" `
    --fast-mode `
    --disable-llm
```

### 3. Executar Pipeline (Modo Completo - Requer mais VRAM)

```powershell
.\mesh3d-generator-py3.11\Scripts\python.exe scripts\run_kiss3dgen_image_to_3d.py `
    --input "data/inputs/exemplo.jpg" `
    --output "data/outputs/teste" `
    --enable-redux `
    --use-controlnet
```

### 4. Verificar Resultados

Os resultados serão salvos em:
- `data/outputs/teste/*_3d_bundle.png` - Bundle 3D gerado
- `data/outputs/teste/*.glb` - Malha 3D final
- `data/outputs/teste/summary.json` - Resumo da execução

## 🔧 Otimizações para Qualidade

### Ajustar Configuração YAML

Editar `Kiss3DGen/pipeline/pipeline_config/default.yaml`:

```yaml
flux:
  num_inference_steps: 20  # Aumentar para melhor qualidade (padrão: 14)
  image_height: 768        # Aumentar resolução (padrão: 640)
  image_width: 1536        # Aumentar resolução (padrão: 1280)

multiview:
  num_inference_steps: 50  # Aumentar para melhor qualidade (padrão: 32)
```

### Usar Redux

O Redux melhora significativamente a qualidade do bundle 3D:
- Adiciona ~10GB de download
- Requer mais VRAM
- Melhora fidelidade e detalhes

### Usar ControlNet

O ControlNet oferece melhor controle sobre a geração:
- Adiciona ~2GB de download
- Requer mais VRAM
- Melhora consistência com a imagem de entrada

## 📊 Métricas de Qualidade

Para avaliar a qualidade, você pode:

1. **Inspeção Visual**: Abrir o arquivo `.glb` em um visualizador 3D (Blender, MeshLab, etc.)

2. **Métricas Quantitativas** (se tiver ground-truth):
```powershell
.\mesh3d-generator-py3.11\Scripts\python.exe scripts\run_kiss3dgen_image_to_3d.py `
    --input "data/inputs/exemplo.jpg" `
    --gt-mesh "data/ground_truth/model.obj" `
    --metrics-out "metrics.json"
```

As métricas incluem:
- **Chamfer Distance (L1/L2)**: Distância média entre pontos
- **F-Score**: Precisão e recall combinados
- **Normal Consistency**: Consistência das normais

## ⚠️ Troubleshooting

### Erro: "Out of memory"
- Use `--fast-mode`
- Use `--disable-llm`
- Desabilite `--enable-redux`
- Reduza resolução no YAML

### Erro: "Model not found"
- Execute `python scripts/download_all_models.py`
- Verifique autenticação: `huggingface-cli whoami`

### Erro: "CUDA out of memory"
- RTX 3060 tem 12GB - sempre use `--fast-mode`
- Monitore VRAM: `nvidia-smi`

## 🎓 Próximos Passos Após Primeira Execução

1. **Ajustar Hiperparâmetros**: Testar diferentes valores de `strength`, `num_inference_steps`, etc.
2. **Comparar Modos**: Testar com/sem Redux, ControlNet, etc.
3. **Avaliar Qualidade**: Comparar resultados com ground-truth se disponível
4. **Otimizar Performance**: Ajustar resoluções e passos para melhor trade-off qualidade/tempo

