# Kiss3DGen Pipeline Audit

**Última Atualização**: Revisão completa após implementação de correções

## Status Geral

| Categoria | Status Anterior | Status Atual | Progresso |
| --- | --- | --- | --- |
| Erros Críticos | 6 | 2 | ✅ 67% resolvido |
| Erros de Qualidade | 5 | 3 | ✅ 40% resolvido |
| Desvios do Artigo | 8 | 4 | ✅ 50% resolvido |
| Melhorias Recomendadas | 0 | 8 implementadas | ✅ Progresso significativo |

## 1. Arquitetura do Pipeline

| Etapa | Estado vs Paper | Implementação | Observações |
| --- | --- | --- | --- |
| Entrada (imagem + máscara) | ✅ | ✅ | `preprocess_input_image` replica o fluxo oficial (rembg + resize). |
| Geração multiview | ✅ | ✅ | **CORRIGIDO**: Zero123++ integrado no fluxo Flux (`generate_reference_3D_bundle_image_zero123` chamado em `generate_flux_bundle`). |
| Construção do bundle | ⚠️ | ⚠️ | **MELHORADO**: Normais reais do Zero123++ aplicadas via `_apply_reference_normals`, mas ainda sem validação de qualidade. |
| LRM | ✅ | ✅ | **CORRIGIDO**: Azimutes explícitos `[0, 90, 180, 270]`, ordem de vistas corrigida. |
| ISOMER | ✅ | ✅ | **CORRIGIDO**: `_finalize_lrm_outputs` prioriza ISOMER (fallback_obj/fallback_glb) sobre LRM. |
| Exportação (OBJ/GLB) | ⚠️ | ⚠️ | **PARCIAL**: ISOMER GLB priorizado, mas ainda há conversão via `trimesh` quando GLB não disponível. |

## 2. Auditoria Matemática/Númerica

- **Imagens**: normalização `[0,1]` correta.
- **Depth**: inexistente; bundle não fornece profundidade.
- **Normais**: Zero123++ → mapeadas corretamente para `[0,1]`; Flux bundle usa RGB reescalado como normal.
- **Espaço de cores**: `_restore_color_dynamic_range` altera faixas com heurística agressiva; exportação converte para linear e volta para sRGB, causando dessaturação.
- **Conversão para uint8**: Vertex colors truncados antes da exportação → banding.
- **Blending/UV**: `_create_texture_from_vertex_colors` gera mosaico artificial, distante do unwrap original.

## 3. Uso dos Modelos

| Item | Estado Anterior | Estado Atual | Observações |
| --- | --- | --- | --- |
| Zero123++ no fluxo principal | ❌ | ✅ | **CORRIGIDO**: `generate_flux_bundle` chama `generate_reference_3D_bundle_image_zero123` e usa o bundle real. |
| Flux + ControlNet + LoRA | ✅ | ✅ | Mantido: configurados com offload e pesos corretos. |
| LRM | ⚠️ | ✅ | **CORRIGIDO**: Azimutes explícitos `[0, 90, 180, 270]`, normais reais aplicadas. |
| Ordem das vistas | ❌ | ✅ | **CORRIGIDO**: `view_order = [3, 0, 1, 2]` em `generate_reference_3D_bundle_image_zero123`. |
| Bundle format | ❌ | ⚠️ | **MELHORADO**: Normais reais aplicadas, mas ainda falta canal de profundidade. |
| Seed | ⚠️ | ✅ | **CORRIGIDO**: `_torch_rng` inicializado com seed, `_rng_uniform` e `_rng_standard_normal` usam gerador determinístico. |
| Flags CLI | ❌ | ✅ | **CORRIGIDO**: `BooleanOptionalAction` permite desativar flags explicitamente. |

## 4. Renderização e Export

- `render_3d_bundle_image_from_mesh` silencia falhas do plugin retornando tensor zero.
- `_finalize_lrm_outputs` converte OBJ→GLB via `trimesh`, descartando MTL/UV.
- ISOMER GLB/OBJ nunca promovidos ao usuário; refinamento vira fallback.

## 5. Engenharia de Software

- Duas bases duplicadas (raiz + `Kiss3DGen/`) sem testes automatizados.
- `run_kiss3dgen_image_to_3d.py` contém lógica de setup Windows altamente acoplada.
- Configs (`3d_bundle_templates`) não utilizados.
- Dependências vendorizadas sem processo de atualização (risco ao mudar PyTorch/CUDA).
- Ausência de scripts/notebooks de treinamento: o paper descreve fine-tuning do Flux/LoRA com bundles renderizados, mas o repositório só referencia checkpoints externos; não há forma de reproduzir ou auditar o treino.
- Avaliação incompleta: existe `evaluate_mesh_against_gt`, porém não há pipeline para rodar nos datasets do artigo nem scripts que calculem as métricas reportadas (Chamfer, F-score, etc.).

## 6. Erros Críticos

### ✅ RESOLVIDOS

| Arquivo | Trecho | Descrição | Status |
| --- | --- | --- | --- |
| `scripts/kiss3d_wrapper_local.py` | `1055-1067` | Zero123++ integrado no fluxo Flux via `generate_reference_3D_bundle_image_zero123`. | ✅ **RESOLVIDO** |
| `scripts/kiss3d_wrapper_local.py` | `218-265` | `_finalize_lrm_outputs` prioriza ISOMER (fallback_obj/fallback_glb) sobre LRM. | ✅ **RESOLVIDO** |
| `scripts/kiss3d_wrapper_local.py` | `907-920` | Ordem de vistas corrigida para `[3, 0, 1, 2]` em `generate_reference_3D_bundle_image_zero123`. | ✅ **RESOLVIDO** |
| `scripts/kiss3d_wrapper_local.py` | `174-182, 383-390` | Determinismo implementado: `_torch_rng` seeded, `_rng_uniform` e `_rng_standard_normal` determinísticos. | ✅ **RESOLVIDO** |
| `scripts/run_kiss3dgen_image_to_3d.py` | `667-679` | Flags CLI corrigidas com `BooleanOptionalAction` permitindo desativação explícita. | ✅ **RESOLVIDO** |

### ⚠️ PARCIALMENTE RESOLVIDOS

| Arquivo | Trecho | Descrição | Status |
| --- | --- | --- | --- |
| `scripts/kiss3d_wrapper_local.py` | `218-265` | Exportação prioriza GLB do ISOMER, mas ainda converte OBJ→GLB via `trimesh` quando GLB não disponível. | ⚠️ **PARCIAL**: Funcional, mas perde texturas na conversão. |

### ❌ AINDA PENDENTES

| Arquivo | Trecho | Descrição | Impacto |
| --- | --- | --- | --- |
| N/A | N/A | Validação de qualidade do bundle antes de passar para LRM não implementada. | Bundles de baixa qualidade podem propagar erros. |
| N/A | N/A | Canal de profundidade ausente no bundle (conforme paper original). | LRM não recebe informação de profundidade. |

## 7. Erros de Qualidade

### ✅ RESOLVIDOS

- ✅ Normais reais aplicadas: `_apply_reference_normals` substitui normais sintéticas do Flux pelas do Zero123++.
- ✅ Máscara robusta: `_generate_robust_mask` combina RGB + normais para melhor segmentação.

### ⚠️ MELHORADOS

- ⚠️ Preview fallback: ainda silencioso, mas ISOMER agora é priorizado corretamente.
- ⚠️ `_restore_color_dynamic_range`: ainda altera faixas, mas impacto reduzido com priorização do ISOMER.

### ❌ AINDA PENDENTES

- ❌ Ausência de depth → LRM ainda depende só de cor e normais.
- ❌ Textura UV gerada por grade (`_create_texture_from_vertex_colors`) causa mosaico.

## 8. Novos Problemas Encontrados (Status de Resolução)

### ✅ RESOLVIDOS

1. ✅ **Flags "sempre ligadas"** – Corrigido: `BooleanOptionalAction` permite desativação explícita.
2. ✅ **ControlNet condicionado no seed falso** – Corrigido: `control_reference = reference_bundle if reference_bundle is not None else flux_seed` (linha 1109).
3. ✅ **`use_mv_rgb` ignorado no modo Flux** – Corrigido: `use_mv_rgb` propagado para `generate_reference_3D_bundle_image_zero123` (linha 1057).
4. ✅ **Prompts sem instruções de câmera** – Corrigido: `_format_multiview_prompt` inclui azimutes explícitos e instruções detalhadas (linha 509-522).
5. ✅ **Máscara derivada de normais artificiais** – Corrigido: `_generate_robust_mask` combina RGB + normais, com fallback para `get_background` (linha 486-507, 992-1002).

### ❌ NOVOS PROBLEMAS IDENTIFICADOS

1. ❌ **Inconsistência em geo_weights/color_weights**: `run_multiview_pipeline` usa `[1.0, 0.98, 1.0, 0.98]` e `[1.0, 0.85, 1.0, 0.85]` (melhor), mas `reconstruct_3d_bundle_image` ainda usa `[1.0, 0.95, 1.0, 0.95]` e `[1.0, 0.7, 1.0, 0.7]` (antigo).
2. ❌ **Inconsistência em stage1_steps**: `run_multiview_pipeline` usa `max(25, int(reconstruction_stage2_steps * 0.4))` (melhor), mas `reconstruct_3d_bundle_image` ainda usa `max(15, reconstruction_stage2_steps // 4)` (antigo).
3. ❌ **num_inference_steps ainda baixo**: Config ainda define `20` steps, não aumentado para 30-50 conforme recomendado.
4. ❌ **Validação de bundle ausente**: Não há validação de qualidade antes de passar bundle para LRM.

## 9. Desvios do Artigo

| Desvio | Classificação Anterior | Status Atual |
| --- | --- | --- |
| Remover Zero123++ do pipeline principal | ❌ incorreto | ✅ **RESOLVIDO**: Zero123++ integrado |
| Usar LRM como saída final e ISOMER como fallback | ❌ incorreto | ✅ **RESOLVIDO**: ISOMER priorizado |
| Reordenar vistas sem ajustar azimutes | ⚠️ perigoso | ✅ **RESOLVIDO**: Ordem corrigida, azimutes explícitos |
| Exportar OBJ/GLB reprocessando e ignorando texturas | ❌ incorreto | ⚠️ **PARCIAL**: GLB do ISOMER priorizado, mas conversão ainda ocorre |
| Seed não aplicado | ⚠️ perigoso | ✅ **RESOLVIDO**: Determinismo implementado |
| ControlNet condicionado a seed 2D em vez do bundle real | ❌ incorreto | ✅ **RESOLVIDO**: Usa bundle real quando disponível |
| Prompt não especifica câmeras/vistas distintas | ⚠️ perigoso | ✅ **RESOLVIDO**: Prompt detalhado com azimutes |
| Flags CLI impossíveis de desativar | ⚠️ perigoso | ✅ **RESOLVIDO**: BooleanOptionalAction |

### ⚠️ DESVIOS AINDA PRESENTES

| Desvio | Classificação | Observações |
| --- | --- | --- |
| Canal de profundidade ausente | ⚠️ perigoso | Paper menciona depth, mas não implementado |
| Validação de qualidade ausente | ⚠️ perigoso | Bundles de baixa qualidade podem passar para LRM |
| Parâmetros inconsistentes entre funções | ⚠️ perigoso | geo_weights/color_weights e stage1_steps diferem entre `run_multiview_pipeline` e `reconstruct_3d_bundle_image` |

## 10. Plano de Correção

### ✅ CONCLUÍDO

1. ✅ **Restaurar fluxo oficial**: Zero123++ integrado, `_build_flux_seed_bundle` como fallback.
2. ✅ **Promover ISOMER**: GLB/OBJ do ISOMER priorizados sobre LRM.
3. ✅ **Corrigir ordem das vistas**: `[3,0,1,2]` implementado, azimutes explícitos `[0, 90, 180, 270]`.
4. ✅ **Determinismo**: `_torch_rng` seeded, funções determinísticas implementadas.
5. ✅ **ControlNet com bundle real**: `control_reference` usa bundle real quando disponível.
6. ✅ **Flags e prompts**: `BooleanOptionalAction` implementado, prompt detalhado com azimutes.
7. ✅ **Máscara robusta**: `_generate_robust_mask` combina RGB + normais.

### ⚠️ PARCIALMENTE CONCLUÍDO

8. ⚠️ **Preservar exportação**: GLB do ISOMER priorizado, mas conversão OBJ→GLB ainda ocorre quando necessário.
9. ⚠️ **Normais reais**: Normais do Zero123++ aplicadas, mas sem validação de qualidade.

### ❌ PENDENTE

10. ❌ **Canal de profundidade**: Adicionar depth ao bundle conforme paper original.
11. ❌ **Validação de qualidade**: Implementar `_validate_bundle_quality` antes de passar para LRM.
12. ❌ **Consistência de parâmetros**: Unificar `geo_weights`, `color_weights` e `stage1_steps` entre `run_multiview_pipeline` e `reconstruct_3d_bundle_image`.
13. ❌ **Aumentar num_inference_steps**: Config ainda em 20, aumentar para 30-50.
14. ❌ **Treino reproduzível**: Scripts de fine-tuning ainda não disponíveis.
15. ❌ **Pipeline de métricas**: Automação de métricas (Chamfer, F-score) ainda não implementada.

## 11. Veredito

### Status Anterior
- **Fidelidade**: ❌ Não fiel ao Kiss3DGen; principais componentes removidos ou ignorados.
- **Validade científica**: ❌ Resultados não representam o pipeline descrito.
- **Confiabilidade dos resultados**: ❌ Contaminados por implementações alternativas.

### Status Atual (Após Correções)
- **Fidelidade**: ⚠️ **PARCIALMENTE FIEL**: Componentes principais restaurados (Zero123++, ISOMER priorizado, ordem de vistas corrigida), mas ainda faltam validações e alguns parâmetros não otimizados.
- **Validade científica**: ⚠️ **MELHORADA**: Pipeline mais próximo do paper, mas ainda precisa de ajustes finos (parâmetros inconsistentes, validação ausente).
- **Confiabilidade dos resultados**: ✅ **MELHORADA**: Determinismo implementado, ISOMER priorizado, normais reais aplicadas. Resultados mais confiáveis, mas ainda podem ser melhorados com validação de qualidade.

### Próximos Passos Críticos
1. Unificar parâmetros entre funções (geo_weights, color_weights, stage1_steps)
2. Aumentar num_inference_steps no config (20 → 30-50)
3. Implementar validação de qualidade do bundle
4. Adicionar canal de profundidade (se necessário conforme paper)


