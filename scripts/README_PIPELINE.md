# Pipeline de Geração 3D - Status e Uso

## ⚠️ AVISO IMPORTANTE

O pipeline atual **NÃO está gerando modelos 3D via difusão**. Está usando **placeholders** (cópias simplificadas do original).

## O que está funcionando

1. ✅ Preprocessamento de imagem (rembg, resize, pad)
2. ✅ Geração de caption/descrição com LLM
3. ✅ Carregamento robusto de meshes originais
4. ✅ Comparação e métricas avançadas (Chamfer, Hausdorff, SSIM)
5. ✅ Visualizações (comparação estática e GIF rotativo)
6. ✅ Exportação para OBJ e GLB
7. ✅ Nomes de arquivos corretos (`generated_{nome}.obj`, `original_{nome}.obj`)
8. ✅ Material com mesmo nome do OBJ
9. ✅ Validação de qualidade (watertight, métricas)

## O que está faltando (CRÍTICO)

### 1. Geração de Reference 3D Bundle Image
- Usar Zero123 para gerar multiview
- Reconstruir mesh inicial com LRM
- Renderizar views e normal maps

### 2. Geração de 3D Bundle Image Final
- Usar Flux diffusion com ControlNet-Tile
- Usar Redux para melhorar prompt
- Gerar bundle image refinado

### 3. Reconstrução 3D
- Usar LRM para reconstrução inicial
- Usar ISOMER para refinamento
- Exportar mesh com texturas geradas

## Como usar

```bash
python scripts/run_3d_pipeline.py --max-objects 5 --output data/outputs/3d_generation
```

## Arquivos gerados

Para cada objeto processado:
- `generated_{nome_objeto}.obj` - Mesh gerado (placeholder por enquanto)
- `original_{nome_objeto}.obj` - Mesh original copiado
- `generated_{nome_objeto}.glb` - GLB gerado
- `original_{nome_objeto}.glb` - GLB original
- `comparison.png` - Comparação visual
- `rotation_comparison.gif` - GIF rotativo lado a lado
- `metrics.json` - Métricas de comparação
- `description.txt` - Descrição gerada
- `input_preprocessed.png` - Imagem preprocessada
- `input_original_*.jpg` - Imagem original

## Próximos passos

Ver `RESUMO_FALTANTE.md` para detalhes completos do que falta implementar do pipeline Kiss3DGen.

