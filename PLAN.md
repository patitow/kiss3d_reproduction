## Plano de Reprodução – Kiss3DGen (Image ➜ 3D)

### Objetivo
- Reproduzir fielmente o pipeline *Image-to-3D* descrito no artigo Kiss3D, usando o repositório oficial apenas como referência conceitual e fonte dos modelos/pesos.
- Consolidar cada passo em `scripts/run_kiss3dgen_image_to_3d.py`, garantindo execução ponta a ponta dentro do workspace `2025_2`.

### Como usar este plano
- Cada item usa `- [ ]` (pendente) ou `- [x]` (concluído).  
- Assim que um item for implementado, **marque-o como concluído** e registre evidências no commit/mensagem (logs de teste, paths gerados etc.).
- Caso um item dependa de outro, respeite a ordem apresentada abaixo.

### Status atual (24/11/2025)
| Item | Resultado |
| --- | --- |
| Venv `mesh3d-generator-py3.11` | ✅ Python 3.11.9 confirmado (`python --version`). |
| Torch stack | ✅ `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, `torchaudio 2.5.1+cu121`. |
| CUDA Toolkit | ✅ `nvcc --version` reporta 12.1. Script `activate_kiss3d_env.bat` + `setup_python311.bat` agora configuram `CUDA_HOME/CUDA_PATH` automaticamente. |
| `ninja` | ✅ Binário presente na raiz e incluído no PATH pelo `setup_python311.bat`/`activate_kiss3d_env.bat`. |
| Principais libs (`diffusers`, `transformers`, `opencv-python`, `imageio`, `trimesh`, `pymeshlab`, `nvdiffrast`, etc.) | ✅ Instaladas conforme `pip show`. |
| `xformers` | ✅ Compilado manualmente a partir do `third_party/xformers` (sparse24 desabilitado) e instalado como wheel local (`xformers-0.0.34+0eb7c432.d20251124`). |
| `numpy`/`scipy` | ✅ Compatíveis (`numpy 2.2.2`, `scipy 1.13.1`). |

> Próxima ação imediata relacionada a dependências: decidir se vamos compilar o `xformers` (VS Build Tools + CUDA) ou se ajustamos o pipeline para rodar sem ele.

---

### Checklist detalhado

#### 1. Diagnóstico e organização inicial
- [x] Inventariar o que já foi alterado na pasta `scripts/` e confirmar se há cópias antigas do pipeline que possam servir de base.
- [x] Mapear quais partes do Kiss3DGen precisamos replicar (preprocess, caption, multiview, bundle, reconstrução).
- [x] Registrar dependências externas obrigatórias (Torch+CUDA, nvdiffrast, ninja, imagem base para teste) e onde serão instaladas.

**Notas de 24/11**
- `scripts/run_kiss3dgen_image_to_3d.py` (nova versão) e `scripts/run_kiss3dgen_simple.py` (legado) coexistem; o segundo segue executando diretamente o `pipeline/kiss3d_wrapper.py` dentro do `Kiss3DGen`, então podemos usar sua lógica como referência do fluxo e mensagens enquanto o primeiro evolui para uma implementação independente.
- Estrutura de estágios do artigo, conforme `pipeline/kiss3d_wrapper.py`:  
  1) `preprocess_input_image` + Florence-2 caption + opcional LLM (`models/llm`) geram o prompt detalhado;  
  2) `FluxImg2ImgPipeline` (+ControlNet/Lora) produz bundle inicial/redux;  
  3) Multiview Diffusion (`custom_diffusers` + `flexgen.ckpt`) gera a grade 2×4;  
  4) `lrm_reconstruct` + ISOMER/LRM (`models/lrm`, `models/ISOMER`) transformam as vistas em malha;  
  5) `isomer_reconstruct`/`init_3d_Bundle` finalizam o `.glb`.  
  Essa sequência é a que precisaremos reproduzir fora do repositório de referência.
- Dependências e assets: `scripts/download_models.py` já lista os checkpoints do HuggingFace (Zero123++, Flux base, ControlNet Union, Redux); `Kiss3DGen/requirements.txt` adiciona nvdiffrast, open3d/pymeshlab, pyrender, etc.; diretórios `models/`, `init_3d_Bundle/` e `assets/` contêm os pesos locais que devem permanecer sincronizados com o script `download_models.py`. Também é obrigatório manter autenticação HF (`huggingface-cli login`) antes de baixar/rodar.

#### 2. Ambiente Python 3.11
- [x] Validar que a venv `mesh3d-generator-py3.11` abre com Python 3.11.9 (`.\mesh3d-generator-py3.11\Scripts\python.exe --version`).
- [x] Atualizar `pip`, `setuptools`, `wheel` dentro da venv.
- [x] Instalar/atualizar dependências de alto nível (`torch`, `torchvision`, `xformers`, `diffusers`, `opencv-python`, `imageio`, `transformers`, etc.) conforme necessidade do pipeline.
- [x] Adicionar `ninja` e `CUDA_HOME/bin` ao `PATH` dentro do script antes de qualquer import, validando com `which ninja`/`where ninja`.

#### 3. Recursos e modelos
- [x] Executar/validar `download_models.py` (ou script equivalente) garantindo que todos os pesos exigidos pelo pipeline estejam em `models/`.
- [x] Verificar integridade de cada submódulo necessário (por ex., `models/lrm`, `models/ISOMER`, `init_3d_Bundle`), registrando hashes ou datas.
- [x] Preparar diretório de entrada (`data/inputs`) com pelo menos uma imagem de teste alinhada ao artigo.

**Notas de 24/11**
- `scripts/download_models.py` confirmou `zero123`, `flux`, `controlnet` e `redux` no cache HF (`~/.cache/huggingface/hub/models--*`). Não houve necessidade de novo download.
- Estrutura local verificada:
  - `Kiss3DGen/models/ISOMER`, `models/lrm`, `models/llm`, `models/zero123plus` presentes (timestamp 23/11 11:41).
  - `Kiss3DGen/init_3d_Bundle/` contém `0.png` … `10.png` (seed usado pelo bundle inicial).
  - `third_party/xformers/dist/` guarda o wheel custom `xformers-0.0.34+0eb7c432.d20251124-cp39-abi3-win_amd64.whl` para reconstruções futuras.
- Criado `data/inputs/example_cartoon_panda.png` (copiado de `Kiss3DGen/examples/cartoon_panda.png`) como caso de teste único para os próximos estágios; `data/outputs/` já existe para recebimento dos resultados.

#### 4. Adaptação do pipeline (sem importar código do Kiss3DGen)
- [x] Reestruturar `scripts/run_kiss3dgen_image_to_3d.py` para manter apenas utilidades próprias (sem `from pipeline...`).
- [ ] Implementar `preprocess_image()` (normalização, resize, center crop) compatível com os modelos usados.
- [ ] Implementar integração Florence-2 (ou LLM escolhido) para `generate_caption()`, com cache local para evitar latência.
- [ ] Implementar gerador de vistas múltiplas (`generate_multiview()`) usando modelo equivalente ao do artigo (p.ex. LRM + ControlNet) com configuração custom.
- [ ] Implementar criação do *3D bundle* (`generate_3d_bundle()`) consolidando as vistas no formato esperado pelos estágios seguintes.
- [ ] Implementar `reconstruct_mesh()` (ISOMER ou alternativa) consumindo o bundle e emitindo `.glb`/`.obj`.
- [ ] Garantir mecanismos de logging e checkpoints entre etapas, para retomar sem repetir tudo.

**Notas de 24/11**
- `scripts/run_kiss3dgen_image_to_3d.py` agora usa módulos locais (`kiss3d_utils_local.py`, `kiss3d_wrapper_local.py`) em vez de importar `pipeline.kiss3d_wrapper`/`pipeline.utils`. A base de código foi copiada e adaptada para usar caminhos relativos ao projeto (`outputs/`, `tmp/`) sem `os.chdir`.
- Os utilitários originais (`lrm_reconstruct`, `isomer_reconstruct`, `preprocess_input_image`, etc.) vivem em `scripts/kiss3d_utils_local.py`, o que evita depender diretamente do pacote `pipeline`.
- `kiss3d_wrapper_local.py` replica o wrapper do artigo e continua carregando os modelos (Flux/ControlNet/Redux, Zero123++ custom pipeline, LRM/ISOMER, Florence-2 e o LLM opcional) com paths resolvidos em tempo de execução.

#### 5. Integração de utilidades
- [ ] Criar helpers para gerir diretórios temporários (`TMP_DIR`, `OUT_DIR`) sem depender do `pipeline.utils`.
- [ ] Implementar validações de entrada/saída (existência de arquivos, espaço em disco, permissões) antes de cada fase.
- [ ] Adicionar suporte a argumentos CLI equivalentes ao script original (`--input`, `--output`, `--enable-redux`, etc.) e documentar defaults.
- [ ] Incluir tratativas de erros específicos (faltou modelo, falhou compilação nvdiffrast, CUDA ausente) com mensagens claras.

#### 6. Testes incrementais
- [ ] Testar etapa de preprocess + caption isoladamente e salvar artefatos intermediários para inspeção.
- [ ] Testar geração multiview com amostra pequena, verificando tempos e necessidade de VRAM.
- [ ] Testar reconstrução 3D em modo *dry-run* (inputs sintéticos) para validar dependências CUDA/nvdiffrast.
- [ ] Executar pipeline completo em imagem de exemplo, salvando bundle e mesh em `outputs/`.
- [ ] Comparar resultados qualitativa/quantitativamente com o artigo (ângulos, textura, fidelidade) e anotar discrepâncias.

#### 7. Documentação e automação
- [ ] Atualizar `README.md` (ou criar seção dedicada) descrevendo requisitos, passos de execução e parâmetros expostos.
- [ ] Registrar no próprio `PLAN.md` qualquer desvio do artigo (modelos alternativos, hiperparâmetros ajustados).
- [ ] Opcional: criar script PowerShell/BAT que ativa venv, exporta variáveis e roda o pipeline com um comando.

#### 8. Validação final
- [ ] Garantir que todos os artefatos finais estejam versionados ou listados (sem subir pesos proprietários).
- [ ] Rodar lint/format relevante nos scripts modificados.
- [ ] Atualizar este plano marcando todas as etapas concluídas e anexando evidências nos commits/logs.

---

> **Próximos passos imediatos:** concluir a seção 1 (diagnóstico) e validar o ambiente (seção 2) antes de tocar novamente no código do pipeline. Sempre que uma tarefa acima for concluída, retorne a este arquivo e marque o item correspondente.

