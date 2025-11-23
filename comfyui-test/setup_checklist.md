# ✅ Checklist de Setup - ComfyUI Test Run

Use este checklist para garantir que tudo está configurado antes de fazer o test run.

## 📋 Pré-requisitos

### 1. ComfyUI Instalado
- [ ] ComfyUI clonado e instalado
  ```bash
  git clone https://github.com/comfyanonymous/ComfyUI.git
  cd ComfyUI
  pip install -r requirements.txt
  ```

### 2. Custom Nodes Instalados
- [ ] ControlNet Aux Preprocessors instalado
  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
  cd comfyui_controlnet_aux
  pip install -r requirements.txt
  ```

### 3. Modelos Baixados

#### Stable Diffusion Checkpoint
- [ ] Checkpoint baixado (escolha um):
  - `v1-5-pruned-emaonly.safetensors` → `ComfyUI/models/checkpoints/`
  - `sd_xl_base_1.0.safetensors` → `ComfyUI/models/checkpoints/`
  
  Links:
  - SD 1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
  - SD XL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

#### ControlNet Models
- [ ] ControlNet-Tile baixado:
  - `control_v11f1e_sd15_tile.pth` → `ComfyUI/models/controlnet/`
  - Link: https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main

- [ ] ControlNet-Normal baixado (para workflow completo):
  - `control_v11p_sd15_normalbae.pth` → `ComfyUI/models/controlnet/`
  - Link: https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main

### 4. Ollama (Opcional - para LLM)
- [ ] Ollama instalado e rodando
  ```bash
  # Windows: Baixe de https://ollama.com/download
  # Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh
  
  # Iniciar servidor
  ollama serve
  
  # Instalar modelo multimodal
  ollama pull llava
  ```

### 5. Dependências Python
- [ ] Dependências instaladas
  ```bash
  cd comfyui-test
  pip install -r requirements.txt
  ```

## 🧪 Testes de Validação

### Teste 1: Conexão com ComfyUI
```bash
cd comfyui-test
python test_comfyui_connection.py
```

**Resultado esperado:**
- ✅ ComfyUI está rodando e acessível
- ✅ Workflows válidos
- ⚠️ Ollama (opcional)

### Teste 2: Workflow Simples
```bash
# Coloque uma imagem de teste em ComfyUI/input/
python test_workflow_simple.py --image ComfyUI/input/test_image.jpg
```

**Resultado esperado:**
- ✅ Imagem enviada
- ✅ Workflow enviado
- ✅ Prompt ID gerado
- ✅ Imagem processada em ComfyUI/output/

### Teste 3: Pipeline Completo (Com LLM)
```bash
# Certifique-se de que o Ollama está rodando
ollama serve

# Execute o teste completo
python integrate_llm.py --image ComfyUI/input/test_image.jpg
```

**Resultado esperado:**
- ✅ Descrição gerada pelo LLM
- ✅ Workflow atualizado com texto
- ✅ Processamento completo

## 🐛 Troubleshooting

### ComfyUI não inicia
- Verifique se Python 3.10+ está instalado
- Verifique se todas as dependências estão instaladas
- Tente: `cd ComfyUI && python main.py --port 8188`

### "Node not found" no workflow
- Instale os custom nodes necessários
- Reinicie o ComfyUI após instalar nodes

### "Model not found"
- Verifique se os modelos estão nas pastas corretas
- Verifique os nomes dos arquivos (case-sensitive)
- Ajuste os caminhos no workflow JSON se necessário

### Ollama não conecta
- Verifique se está rodando: `curl http://localhost:11434/api/tags`
- Inicie: `ollama serve`
- Verifique se o modelo está instalado: `ollama list`

## ✅ Pronto para Test Run!

Quando todos os itens acima estiverem marcados, você está pronto para fazer o test run!

**Ordem recomendada:**
1. Execute `test_comfyui_connection.py` para validar setup
2. Execute `test_workflow_simple.py` para testar workflow básico
3. Execute `integrate_llm.py` para testar pipeline completo


