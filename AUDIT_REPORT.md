# Kiss3DGen Pipeline Audit

## 1. Arquitetura do Pipeline

| Etapa | Estado vs Paper | Implementação | Observações |
| --- | --- | --- | --- |
| Entrada (imagem + máscara) | ✅ | ✅ | `preprocess_input_image` replica o fluxo oficial (rembg + resize). |
| Geração multiview | ❌ | ❌ | Pipeline “flux” ignora Zero123++; bundle é fabricado a partir de cópias 2D (`_build_flux_seed_bundle`). |
| Construção do bundle | ❌ | ❌ | Normais sintéticas (`_fake_normals_from_view`) e ausência de vistas reais. |
| LRM | ⚠️ | ⚠️ | Modelo correto, mas recebe vistas em ordem errada e sem cues geométricos reais. |
| ISOMER | ❌ | ❌ | Executa, porém `_finalize_lrm_outputs` promove apenas o OBJ/GLB do LRM; refinamento descartado. |
| Exportação (OBJ/GLB) | ❌ | ❌ | `.mtl` e texturas do LRM/ISOMER não são copiados; GLB reexportado via `trimesh` perde materiais. |

## 2. Auditoria Matemática/Númerica

- **Imagens**: normalização `[0,1]` correta.
- **Depth**: inexistente; bundle não fornece profundidade.
- **Normais**: Zero123++ → mapeadas corretamente para `[0,1]`; Flux bundle usa RGB reescalado como normal.
- **Espaço de cores**: `_restore_color_dynamic_range` altera faixas com heurística agressiva; exportação converte para linear e volta para sRGB, causando dessaturação.
- **Conversão para uint8**: Vertex colors truncados antes da exportação → banding.
- **Blending/UV**: `_create_texture_from_vertex_colors` gera mosaico artificial, distante do unwrap original.

## 3. Uso dos Modelos

| Item | Estado |
| --- | --- |
| Zero123++ no fluxo principal | ❌ – carregado, mas não usado quando `pipeline_mode=flux`. |
| Flux + ControlNet + LoRA | ✅ – configurados com offload e pesos corretos. |
| LRM | ⚠️ – entrada não obedece layout 2×4 real, normais sintéticas. |
| Ordem das vistas | ❌ – multiview usa `[0,1,2,3]` em vez de `[3,0,1,2]`, desincronizando azimutes. |
| Bundle format | ❌ – falta profundidade e normais físicas. |
| Seed | ⚠️ – `seed` não controla `_generate_seed_variations`; pipeline não chama `seed_everything`. |
| Flags CLI | ❌ – `--enable-redux/--use-mv-rgb/--use-controlnet` são `store_true` com `default=True`; impossíveis de desativar. |

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

| Arquivo | Trecho | Descrição | Impacto |
| --- | --- | --- | --- |
| `scripts/kiss3d_wrapper_local.py` | `2019-2055`, `474-512` | Zero123++ não participa do fluxo principal; bundle inventado duplicando a imagem e “normais” falsas. | Geometrias duplicadas, pipeline fora do paper. |
| `scripts/kiss3d_wrapper_local.py` | `1378-1445` | ISOMER executa, porém `_finalize_lrm_outputs` promove apenas o OBJ/GLB do LRM. | Refinamento/texturização descartados; qualidade inferior. |
| `scripts/kiss3d_wrapper_local.py` | `211-233` | Exportação não copia `.mtl`/texturas e reexporta via `trimesh`. | OBJ final inválido; GLB dessaturado. |
| `scripts/kiss3d_wrapper_local.py` | `788-801` | Ordem `[0,1,2,3]` substituiu `[3,0,1,2]`. | Azimutes trocados, malhas espelhadas. |
| `scripts/kiss3d_wrapper_local.py` | `353-398` | `_generate_seed_variations` usa random não determinístico; `seed` ignorado. | Resultados irreprodutíveis. |

## 7. Erros de Qualidade

- Normais sintéticas (`_fake_normals_from_view`) e ausência de depth → LRM depende só de cor.
- Preview fallback silencioso (`render_3d_bundle_image_from_mesh`).
- `_restore_color_dynamic_range` altera faixas legítimas.
- Textura UV gerada por grade (`_create_texture_from_vertex_colors`) causa mosaico.

## 8. Novos Problemas Encontrados

1. **Flags “sempre ligadas”** – `scripts/run_kiss3dgen_image_to_3d.py` define `--enable-redux`, `--use-mv-rgb` e `--use-controlnet` com `action="store_true"`, `default=True`. Não existe permissão para desativar recursos avaliados no paper.
2. **ControlNet condicionado no seed falso** – `generate_flux_bundle` usa `flux_seed` (cópias da própria imagem com normais sintéticas) para gerar `control_images`; o bundle real (Zero123++) nunca participa, quebrando o pipeline descrito no PDF.
3. **`use_mv_rgb` ignorado no modo Flux** – a flag só é propagada para `run_multiview_pipeline`; o branch Flux sempre usa `_fake_normals` e não mistura as vistas reais.
4. **Prompts sem instruções de câmera** – os prompts são genéricos (“A grid of 2x4 multi-view image...”) e não descrevem cada quadrante; contradiz o treinamento descrito no paper que exige vistas “front/right/back/left”.
5. **Máscara derivada de normais artificiais** – `get_background` roda `rembg` nas “normais” geradas a partir do RGB; para objetos com cores suaves a máscara remove partes válidas.

## 9. Desvios do Artigo

| Desvio | Classificação |
| --- | --- |
| Remover Zero123++ do pipeline principal | ❌ incorreto |
| Usar LRM como saída final e ISOMER como fallback | ❌ incorreto |
| Reordenar vistas sem ajustar azimutes | ⚠️ perigoso |
| Exportar OBJ/GLB reprocessando e ignorando texturas | ❌ incorreto |
| Seed não aplicado | ⚠️ perigoso |
| ControlNet condicionado a seed 2D em vez do bundle real | ❌ incorreto |
| Prompt não especifica câmeras/vistas distintas | ⚠️ perigoso |
| Flags CLI impossíveis de desativar (não reproduz ablations) | ⚠️ perigoso |

## 10. Plano de Correção

1. **Restaurar fluxo oficial**: gerar bundle via Zero123++, alimentar Redux/ControlNet/Flux e só então reconstruir; `_build_flux_seed_bundle` apenas como fallback.
2. **Promover ISOMER**: retornar GLB/OBJ refinados por padrão e usar LRM apenas se ISOMER falhar; copiar também `.mtl` e texturas.
3. **Preservar exportação**: evitar reabrir OBJ em `trimesh`; se precisar de GLB, usar dados com UV originais.
4. **Corrigir ordem das vistas**: voltar a `[3,0,1,2]` (ou alinhar azimutes no LRM/ISOMER).
5. **Determinismo**: chamar `seed_everything`, passar `torch.Generator` seeded para `_generate_seed_variations`.
6. **Normais/depth reais**: usar normais vindas do LRM/Zero123; documentar ou adicionar canal de profundidade conforme paper.
7. **ControlNet com bundle real**: alimentar `redux_hparam` e `control_images` com o bundle de Zero123++ quando disponível; usar `_build_flux_seed_bundle` só como fallback.
8. **Flags e prompts**: trocar as flags por `BooleanOptionalAction` e injetar prompts específicos para cada quadrante (“top-left front view”, etc.).
9. **Máscara robusta**: derivar máscara diretamente de `lrm_multi_view_normals` (norma > ε) em vez de `rembg` sobre normais sintéticas.
10. **Treino reproduzível**: documentar (ou disponibilizar) o script de fine-tuning do Flux/LoRA com os “3D Bundle Images” renderizados, conforme descrito no paper, permitindo auditoria do processo.
11. **Pipeline de métricas**: automatizar a execução das métricas (Chamfer, F-score, etc.) no mesmo dataset utilizado no artigo, salvando relatórios comparáveis.

## 11. Veredito

- **Fidelidade**: Não está fiel ao Kiss3DGen; principais componentes foram removidos ou ignorados.
- **Validade científica**: Não; resultados não representam o pipeline descrito.
- **Confiabilidade dos resultados**: Contaminados por implementações alternativas (bundles artificiais, ISOMER descartado, export bugada).


