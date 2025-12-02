O que temos hoje é:

1. **Zero123++ + LRM** para gerar um bundle de referência (RGB+normal) → esse resultado é usado para alimentar **Flux/ControlNet+Redux**;
2. Na sequência, **Flux** produz um novo bundle e chamamos o **LRM novamente** para reconstruir a malha final.

Ou seja, o LRM é invocado duas vezes e ainda rodamos Zero123++ e Flux em série, apesar de serem caminhos alternativos. Pior: dentro do script com checkpoints rodamos “meia pipeline” (Zero123 + caption) e depois chamamos `run_image_to_3d`, que repete tudo novamente — isso explica por que suas execuções estão demorando o dobro e gerando artefatos duplicados.

### O que vou fazer agora

Para alinhar com o desenho do artigo, estou abrindo uma refatoração em três frentes:

1. **Modo do pipeline (multiview × flux)**  
   - Adiciono um novo parâmetro `--pipeline-mode {multiview,flux}`.  
   - *Multiview*: Zero123++ → LRM → ISOMER → export (sem Flux).  
   - *Flux*: pulo o Zero123++ inicial e uso apenas caption → Flux(+ControlNet/Redux) → LRM/ISOMER para gerar a malha.  
   - Nada mais de rodar dois caminhos na mesma execução.

2. **Caption independente de multiview**  
   - O caption (Florence-2 + LLM) roda logo após carregar a imagem, antes de qualquer branch.  
   - Com checkpoint, salvo `caption.txt` imediatamente, sem depender da etapa multiview.

3. **Checkpoints integrados**  
   - O orquestrador com checkpoints passa a coordenar todas as etapas, sem chamar `run_image_to_3d` no fim.  
   - Cada branch escreve seus artefatos (`*_mv_image.png`, `bundle.png`, `mesh.glb/obj`) e encerra.  
   - Se o usuário quiser relançar a partir de “bundle_3d”, o pipeline não refaz o que já está pronto.

### Próximos passos técnicos

- Atualizar `scripts/run_kiss3dgen_image_to_3d_with_checkpoints.py` para respeitar o novo `pipeline_mode`.
- Ajustar `kiss3d_wrapper_local.py` para que o LRM seja chamado apenas no branch escolhido.
- Garantir que os checkpoints reflitam as novas etapas (ex.: `mv_bundle_multiview`, `flux_bundle` etc.) e que abortemos cedo quando houver conflito (ex.: vistas inconsistentes).

Depois dessa refatoração a execução ficará:

```
Input → Florence-2 → Prompt Refinado
   ├─ pipeline_mode=multiview: Zero123++ → LRM → ISOMER → Export
   └─ pipeline_mode=flux: Flux/ControlNet → bundle → LRM → ISOMER → Export
```