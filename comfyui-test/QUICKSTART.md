# 🚀 Guia Rápido - ComfyUI Mesh3D Workflow

Guia rápido para começar a usar o workflow do ComfyUI para geração de malhas 3D.

## ⚡ Setup Rápido (5 minutos)

### 1. Instalar ComfyUI
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### 2. Instalar Custom Nodes
```bash
cd custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux
pip install -r requirements.txt
```

### 3. Baixar Modelos Necessários

Coloque os seguintes arquivos nas pastas corretas:

**Stable Diffusion Checkpoint:**
- Baixe: `v1-5-pruned-emaonly.safetensors` ou `sd_xl_base_1.0.safetensors`
- Coloque em: `ComfyUI/models/checkpoints/`

**ControlNet Models:**
- `control_v11f1e_sd15_tile.pth` → `ComfyUI/models/controlnet/`
- `control_v11p_sd15_normalbae.pth` → `ComfyUI/models/controlnet/`

### 4. Iniciar ComfyUI
```bash
cd ComfyUI
python main.py
```

Acesse: http://127.0.0.1:8188

## 📝 Uso Básico

### Opção 1: Interface Gráfica

1. Abra o ComfyUI no navegador
2. Clique em "Load" → Selecione `workflow_mesh3d.json`
3. No node "LoadImage", selecione sua imagem
4. No node "Text Prompt", edite o texto descritivo
5. Clique em "Queue Prompt"

### Opção 2: Com Script Python (Recomendado)

1. Instale dependências:
```bash
cd comfyui-test
pip install -r requirements.txt
```

2. Certifique-se de que o Ollama está rodando:
```bash
ollama serve
ollama pull llava
```

3. Execute o script:
```bash
python integrate_llm.py --image path/to/your/image.jpg
```

## 🎯 Workflows Disponíveis

### `workflow_mesh3d.json` (Completo)
- ✅ Geração de normal maps
- ✅ ControlNet-Tile + ControlNet-Normal
- ✅ Refinamento completo
- ⚠️ Requer custom nodes (MiDaS, DepthToNormalMap)

### `workflow_simple.json` (Simplificado)
- ✅ ControlNet-Tile básico
- ✅ Funciona com apenas nodes padrão
- ⚠️ Não inclui normal maps

**Recomendação**: Comece com `workflow_simple.json` para testar, depois use o completo.

## 🔧 Troubleshooting Rápido

### "Node not found"
```bash
# Instalar custom nodes
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
```

### "Model not found"
- Verifique se os modelos estão nas pastas corretas
- Ajuste os caminhos no workflow JSON

### Ollama não conecta
```bash
# Verificar se está rodando
curl http://localhost:11434/api/tags

# Iniciar se necessário
ollama serve
```

## 📊 Fluxo do Workflow

```
Imagem → LLM (Texto) → Normal Map → ControlNet-Tile → ControlNet-Normal → Imagem Refinada
```

## 💡 Dicas

1. **Comece simples**: Use `workflow_simple.json` primeiro
2. **Teste com imagens pequenas**: 512x512 para começar
3. **Ajuste ControlNet strength**: Entre 0.7-1.0
4. **Use prompts detalhados**: Quanto mais detalhado, melhor o resultado

## 🔗 Links Úteis

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ControlNet Models](https://huggingface.co/lllyasviel/ControlNet-v1-1)
- [Ollama Models](https://ollama.com/library)

## ❓ Próximos Passos

Depois de testar o workflow básico:
1. Experimente diferentes modelos Stable Diffusion
2. Ajuste parâmetros de sampling
3. Integre com módulo de inicialização de malha 3D
4. Teste com múltiplas imagens

