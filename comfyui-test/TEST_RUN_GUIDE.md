# 🚀 Guia de Test Run - ComfyUI Mesh3D

Este guia te leva passo a passo para fazer o primeiro test run do workflow do ComfyUI.

## 📝 Passo a Passo

### Passo 1: Preparar Ambiente

1. **Verificar ComfyUI está rodando:**
   ```bash
   cd ComfyUI
   python main.py
   ```
   
   Deve abrir em: http://127.0.0.1:8188

2. **Em outro terminal, verificar setup:**
   ```bash
   cd comfyui-test
   python test_comfyui_connection.py
   ```
   
   Deve mostrar: ✅ ComfyUI está rodando e acessível

### Passo 2: Preparar Imagem de Teste

1. **Escolha uma imagem de teste:**
   - Use uma imagem simples (cadeira, objeto, etc.)
   - Formato: JPG ou PNG
   - Tamanho recomendado: 512x512 ou 768x768

2. **Coloque a imagem no ComfyUI:**
   ```bash
   # Copie a imagem para a pasta de input do ComfyUI
   cp sua_imagem.jpg ComfyUI/input/
   ```

### Passo 3: Teste Básico (Sem LLM)

Execute o teste simples primeiro para validar que o ComfyUI está funcionando:

```bash
cd comfyui-test
python test_workflow_simple.py --image ComfyUI/input/sua_imagem.jpg
```

**O que acontece:**
1. ✅ Faz upload da imagem para o ComfyUI
2. ✅ Carrega o workflow `workflow_simple.json`
3. ✅ Atualiza o workflow com a imagem
4. ✅ Envia para processamento
5. ✅ Gera um Prompt ID

**Resultado:**
- Acompanhe o progresso em: http://127.0.0.1:8188
- A imagem processada será salva em: `ComfyUI/output/`

### Passo 4: Teste com LLM (Opcional)

Se você tem o Ollama configurado:

1. **Iniciar Ollama (se não estiver rodando):**
   ```bash
   ollama serve
   ```

2. **Verificar modelo instalado:**
   ```bash
   ollama list
   # Deve mostrar 'llava' na lista
   ```

3. **Executar teste completo:**
   ```bash
   cd comfyui-test
   python integrate_llm.py --image ComfyUI/input/sua_imagem.jpg
   ```

**O que acontece:**
1. ✅ Analisa a imagem com LLM (llava)
2. ✅ Gera descrição detalhada da cena
3. ✅ Atualiza o workflow com o texto gerado
4. ✅ Envia para processamento no ComfyUI

**Resultado:**
- Descrição detalhada impressa no terminal
- Workflow processado com texto gerado pelo LLM
- Imagem refinada salva em `ComfyUI/output/`

### Passo 5: Verificar Resultados

1. **Abrir ComfyUI no navegador:**
   - http://127.0.0.1:8188

2. **Verificar output:**
   - Pasta: `ComfyUI/output/`
   - Deve conter a imagem processada

3. **Comparar resultados:**
   - Imagem original vs. imagem refinada
   - Verificar se os detalhes foram melhorados

## 🎯 Workflows Disponíveis

### `workflow_simple.json`
- ✅ Workflow básico com ControlNet-Tile
- ✅ Funciona com apenas nodes padrão
- ✅ Bom para teste inicial
- ⚠️ Não inclui normal maps

**Uso:**
```bash
python test_workflow_simple.py --image sua_imagem.jpg --workflow workflow_simple.json
```

### `workflow_mesh3d.json`
- ✅ Workflow completo com normal maps
- ✅ ControlNet-Tile + ControlNet-Normal
- ✅ Geração de normal maps
- ⚠️ Requer custom nodes (MiDaS, DepthToNormalMap)

**Uso:**
```bash
python integrate_llm.py --image sua_imagem.jpg --workflow workflow_mesh3d.json
```

## 🔧 Ajustes e Parâmetros

### Ajustar Prompt Manualmente

No workflow, você pode editar o prompt diretamente:

```python
# No script, use --prompt
python test_workflow_simple.py --image sua_imagem.jpg --prompt "seu prompt aqui"
```

### Ajustar Parâmetros do Workflow

Edite o arquivo JSON diretamente ou use a interface do ComfyUI:

- **ControlNet Strength**: Entre 0.7-1.0
- **Steps**: 20-30 para qualidade
- **CFG Scale**: 7.0-9.0
- **Resolução**: Ajuste em `EmptyLatentImage` (512x512 ou 768x768)

## 📊 Interpretando Resultados

### ✅ Sucesso
- Imagem processada aparece em `ComfyUI/output/`
- Sem erros no terminal
- ComfyUI mostra progresso completo

### ❌ Problemas Comuns

**"Node not found"**
- Instale custom nodes: `comfyui_controlnet_aux`
- Reinicie ComfyUI

**"Model not found"**
- Verifique se os modelos estão em `ComfyUI/models/`
- Verifique nomes dos arquivos

**"Connection refused"**
- Verifique se ComfyUI está rodando
- Verifique a URL: `http://127.0.0.1:8188`

**Ollama não conecta**
- Verifique se está rodando: `ollama serve`
- Verifique modelo: `ollama list`

## 🎉 Próximos Passos

Depois do test run bem-sucedido:

1. ✅ Experimente diferentes imagens
2. ✅ Ajuste parâmetros do workflow
3. ✅ Teste workflow completo com normal maps
4. ✅ Integre com módulo de inicialização de malha 3D
5. ✅ Implemente no código Python (próxima etapa)

## 📚 Recursos

- [ComfyUI Docs](https://github.com/comfyanonymous/ComfyUI)
- [ControlNet Models](https://huggingface.co/lllyasviel/ControlNet-v1-1)
- [Ollama Docs](https://github.com/ollama/ollama)


