# Pipeline de Geração 3D - Implementação Própria

Pipeline completo de geração de modelos 3D a partir de imagens, implementado seguindo a abordagem do Kiss3DGen (referência), mas com código próprio.

## Estrutura do Pipeline

### 1. Multiview Generation (`multiview_generator.py`)
- **Zero123MultiviewGenerator**: Gera múltiplas views usando Zero123++
- Gera 4 views (270°, 0°, 90°, 180°) com elevação 5°
- Combina views em grid

### 2. LRM Reconstruction (`lrm_reconstructor.py`)
- **LRMReconstructor**: Reconstrói mesh inicial usando LRM
- Recebe multiview image e retorna vertices, faces, normals, RGB views, albedo

### 3. Normal Map Rendering (`normal_renderer.py`)
- **NormalMapRenderer**: Renderiza normal maps a partir de mesh
- Usa Pytorch3D quando disponível, fallback para método simples
- Renderiza 4 normal maps correspondentes às views RGB

### 4. Flux + ControlNet Generation (`flux_controlnet_generator.py`)
- **FluxControlNetGenerator**: Gera bundle image refinado
- Usa FLUX.1-dev com ControlNet-Tile
- Suporta Flux Prior Redux para melhorar prompt embeddings
- Gera bundle image final (RGB + normal maps)

### 5. ISOMER Refinement (`isomer_refiner.py`)
- **ISOMERRefiner**: Refina mesh usando ISOMER
- Usa normal maps para refinamento geométrico
- Projeta texturas RGB no mesh refinado
- Exporta mesh final com texturas

### 6. Pipeline Principal (`image_to_3d_pipeline.py`)
- **ImageTo3DPipeline**: Orquestra todo o pipeline
- Integra todos os módulos acima
- Pipeline completo: imagem → multiview → bundle → mesh

## Fluxo Completo

```
Input Image (512x512)
    ↓
[1] Zero123++ → Multiview (4 views)
    ↓
[2] LRM → Mesh inicial + RGB views + Normals
    ↓
[3] Normal Renderer → Normal maps (4 views)
    ↓
[4] Criar Reference Bundle Image (RGB + Normal, 2x4 grid)
    ↓
[5] Flux + ControlNet → Bundle Image Final (refinado)
    ↓
[6] Separar RGB e Normal maps
    ↓
[7] LRM → Reconstruir mesh inicial
    ↓
[8] ISOMER → Refinar mesh com normal maps
    ↓
[9] ISOMER → Projetar texturas RGB
    ↓
Output: Mesh 3D texturizado (.obj/.glb)
```

## Dependências

### Obrigatórias
- `torch` >= 2.0
- `torchvision`
- `PIL` (Pillow)
- `numpy`
- `einops`

### Opcionais (para funcionalidade completa)
- `diffusers` - Para Zero123++ e Flux
- `transformers` - Para modelos de difusão
- `pytorch3d` - Para renderização avançada de normal maps (OPCIONAL, veja INSTALL_PYTORCH3D.md)
  - **Nota**: Pytorch3D não tem suporte oficial para Python 3.13
  - O pipeline funciona sem ele usando fallbacks
- `trimesh` - Para manipulação de meshes
- `omegaconf` - Para configs do LRM

### Específicas do Kiss3DGen (referência)
- Módulos do Kiss3DGen (`models.lrm`, `models.ISOMER`, `utils.tool`)
  - **NOTA**: Estes são apenas referência. O pipeline tenta usar se disponíveis, mas tem fallbacks.

## Uso

```python
from mesh3d_generator.pipeline.image_to_3d_pipeline import ImageTo3DPipeline

pipeline = ImageTo3DPipeline(device="cuda:0")

mesh_path, bundle_path, caption = pipeline.generate_3d_model(
    input_image_path="input.png",
    output_dir="./outputs",
    object_name="object",
    seed=42,
    enable_redux=True,
    use_mv_rgb=True,
    use_controlnet=True
)
```

## Status de Implementação

✅ **Implementado**:
- Estrutura completa do pipeline
- Zero123 multiview generation (com fallbacks)
- LRM reconstruction (com fallbacks)
- Normal map rendering (com fallbacks)
- Flux + ControlNet generation (com fallbacks)
- ISOMER refinement (com fallbacks)
- Integração completa

⚠️ **Requer dependências externas**:
- Modelos do HuggingFace (Zero123++, Flux, ControlNet)
- Checkpoints do LRM e ISOMER (se disponíveis)
- Pytorch3D para renderização avançada

🔄 **Fallbacks**:
- Se módulos não estiverem disponíveis, o pipeline usa placeholders simples
- Mesh placeholder (esfera) se LRM/ISOMER não disponíveis
- Normal maps placeholder se renderizador não disponível
- Reference bundle image se Zero123 não disponível

## Próximos Passos

1. Testar com modelos reais do HuggingFace
2. Ajustar APIs conforme necessário
3. Otimizar para RTX 3060 12GB
4. Adicionar mais fallbacks e tratamento de erros
5. Melhorar qualidade dos placeholders

