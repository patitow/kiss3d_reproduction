# Análise Comparativa: Kiss3DGen Original vs Implementação Local

## 1. Diferenças Críticas Identificadas

### 1.1. Parâmetros de ControlNet

**Original (Kiss3DGen/pipeline/kiss3d_wrapper.py:710-714):**
```python
control_mode = ['tile']
control_guidance_start = [0.0]
control_guidance_end = [0.65]
controlnet_conditioning_scale = [0.6]
```

**Implementação Local (scripts/kiss3d_wrapper_local.py:1017-1023):**
```python
control_mode = flux_cfg.get("controlnet_modes", ["tile"])
control_guidance_start = flux_cfg.get("controlnet_guidance_start", 0.0)
control_guidance_end = flux_cfg.get("controlnet_guidance_end", 0.65)
controlnet_conditioning_scale = flux_cfg.get("controlnet_conditioning_scale", 0.6)
```

**Problema:** O original usa apenas `['tile']`, enquanto o config pode ter múltiplos modos. Isso pode causar inconsistências.

### 1.2. Processamento do Bundle Image

**Original:**
- Usa `reference_3d_bundle_image.unsqueeze(0)` diretamente
- Não faz resize explícito antes de passar para Flux

**Implementação Local:**
- Faz resize explícito para `flux_height x flux_width`
- Pode estar alterando proporções incorretamente

### 1.3. Parâmetros de Reconstrução ISOMER

**Original:**
- Usa valores padrão do ISOMER
- `reconstruction_stage2_steps=50` (hardcoded)

**Implementação Local:**
- Tenta ajustar dinamicamente baseado em fast_mode
- Pode estar usando valores diferentes

### 1.4. Gerenciamento de Memória

**Original:**
- Não tem otimizações específicas para 12GB
- Usa CPU offload padrão do diffusers

**Implementação Local:**
- Tem funções de offload explícitas
- Mas pode não estar sendo usado corretamente

## 2. Problemas de VRAM (12GB)

### 2.1. Modelos Carregados Simultaneamente

1. **Flux Pipeline:** ~4-5GB (com CPU offload)
2. **Multiview (Zero123++):** ~2-3GB
3. **Caption Model (Florence-2):** ~1-2GB (CPU)
4. **Reconstruction (LRM):** ~2-3GB
5. **LLM (Llama-3.2-3B):** ~2-3GB (CPU)

**Total estimado:** ~8-12GB (sem otimizações)

### 2.2. Soluções Necessárias

1. **Quantização:** Usar fp8/fp16 quando possível
2. **Segmentação:** Carregar/descarregar modelos por etapa
3. **CPU Offload:** Mais agressivo
4. **Redução de Resolução:** Para modelos intermediários

## 3. Correções Necessárias

### 3.1. Alinhar Parâmetros com Original

- Usar exatamente os mesmos valores de ControlNet
- Garantir que o bundle image não seja redimensionado incorretamente
- Usar mesmos valores de reconstrução

### 3.2. Otimizações para 12GB VRAM

1. **Quantização de Modelos:**
   - Flux: fp8 (se disponível) ou fp16
   - Multiview: fp16
   - Caption: bf16 ou fp16
   - LRM: fp16

2. **Pipeline Segmentado:**
   - Etapa 1: Carregar apenas Multiview → Gerar multiview → Descarregar
   - Etapa 2: Carregar apenas Caption → Gerar caption → Descarregar
   - Etapa 3: Carregar apenas Flux → Gerar bundle → Descarregar
   - Etapa 4: Carregar apenas LRM → Reconstruir → Descarregar
   - Etapa 5: Carregar apenas ISOMER → Refinar → Descarregar

3. **CPU Offload Agressivo:**
   - Mover modelos para CPU imediatamente após uso
   - Limpar cache CUDA entre etapas
   - Usar `torch.cuda.empty_cache()` e `torch.cuda.synchronize()`

## 4. Implementação Proposta

### 4.1. Wrapper Otimizado

Criar `kiss3d_wrapper_optimized.py` com:
- Carregamento lazy de modelos
- Descarregamento automático após uso
- Quantização automática baseada em VRAM disponível

### 4.2. Pipeline Segmentado

Criar `run_kiss3dgen_image_to_3d_optimized.py` com:
- Etapas claramente separadas
- Gerenciamento explícito de memória
- Logging de uso de VRAM

### 4.3. Config Otimizado

Criar `pipeline_config/optimized_12gb.yaml` com:
- Modelos em fp16/fp8
- CPU offload habilitado
- Resoluções reduzidas quando necessário

