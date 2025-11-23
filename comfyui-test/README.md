# ComfyUI Workflow - Mesh3D Generator

Este diretório contém o workflow do ComfyUI para implementar o pipeline de geração de malhas 3D a partir de imagens, conforme descrito no artigo.

## 📋 Visão Geral

O workflow implementa as seguintes etapas do pipeline:

1. **Carregamento de Imagem**: Imagem de entrada
2. **Geração de Normal Maps**: Conversão de depth map para normal map
3. **Refinamento com ControlNet**: Uso de ControlNet-Tile e ControlNet-Normal + texto para refinamento
4. **Geração de Imagem Refinada**: Output final refinado

## 🚀 Instalação

### Pré-requisitos

1. **ComfyUI** instalado e funcionando
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. **Custom Nodes Necessários**:
   - **ControlNet Preprocessors**: Para normal maps e depth
   - **MiDaS Depth Estimation**: Para geração de depth maps
   
   Instale via ComfyUI Manager ou manualmente:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
   ```

3. **Modelos Necessários**:
   - Stable Diffusion checkpoint (ex: `sd_xl_base_1.0.safetensors` ou `v1-5-pruned-emaonly.safetensors`)
   - ControlNet-Tile: `control_v11f1e_sd15_tile.pth`
   - ControlNet-Normal: `control_v11p_sd15_normalbae.pth` ou `control_v11f1p_sd15_depth.pth`

## 📁 Estrutura de Arquivos

```
comfyui-test/
├── workflow_mesh3d.json          # Workflow principal do ComfyUI
├── README.md                      # Este arquivo
├── integrate_llm.py              # Script para integrar LLM (Ollama) com o workflow
└── example_images/                # Imagens de exemplo (criar manualmente)
```

## 🚀 Quick Start

**Para fazer seu primeiro test run, siga o guia passo a passo:**
👉 **[TEST_RUN_GUIDE.md](TEST_RUN_GUIDE.md)** - Guia completo de test run

**Para verificar se tudo está configurado:**
👉 **[setup_checklist.md](setup_checklist.md)** - Checklist de setup

## 🔧 Como Usar

### Método 1: Teste Rápido (Recomendado para começar)

1. **Verificar setup:**
   ```bash
   python test_comfyui_connection.py
   ```

2. **Teste workflow simples (sem LLM):**
   ```bash
   python test_workflow_simple.py --image path/to/image.jpg
   ```

3. **Teste completo (com LLM):**
   ```bash
   python integrate_llm.py --image path/to/image.jpg --ollama-model llava
   ```

### Método 2: Interface do ComfyUI

1. Abra o ComfyUI
2. Clique em "Load" e selecione `workflow_mesh3d.json` ou `workflow_simple.json`
3. Ajuste os seguintes parâmetros:
   - **LoadImage**: Selecione sua imagem de entrada
   - **Text Prompt**: Insira o texto descritivo detalhado (ou use o script Python para gerar via LLM)
   - **Checkpoint**: Selecione seu modelo Stable Diffusion
   - **ControlNet Models**: Verifique se os caminhos dos modelos estão corretos
4. Clique em "Queue Prompt" para executar

### Método 3: API do ComfyUI + Script Python

Use o script `integrate_llm.py` para:
- Gerar texto detalhado via LLM (Ollama) a partir da imagem
- Enviar o workflow para o ComfyUI via API
- Processar os resultados

```bash
python integrate_llm.py --image path/to/image.jpg --ollama-model llava
```

## 🎯 Workflow Detalhado

### Etapa 1: Carregamento e Análise
- **LoadImage**: Carrega a imagem de entrada
- **CLIPTextEncode**: Codifica o prompt de texto (gerado por LLM ou manual)

### Etapa 2: Geração de Normal Maps
- **MiDaS-DepthMapPreprocessor**: Gera depth map a partir da imagem
- **ImageNormalize**: Normaliza o depth map
- **DepthToNormalMap**: Converte depth map para normal map
- **SaveImage**: Salva o normal map gerado

### Etapa 3: Refinamento com ControlNet
- **ControlNetLoader (Tile)**: Carrega ControlNet-Tile para refinamento de detalhes
- **ControlNetLoader (Normal)**: Carrega ControlNet-Normal para preservação de geometria
- **ControlNetApplyAdvanced**: Aplica ambos os ControlNets sequencialmente
  - Primeiro ControlNet-Tile na imagem original
  - Depois ControlNet-Normal no normal map gerado

### Etapa 4: Geração Final
- **KSampler**: Gera a imagem refinada usando Stable Diffusion
- **VAEDecode**: Decodifica o resultado
- **SaveImage**: Salva a imagem final refinada

## 🔗 Integração com LLM

Para usar LLM (Ollama) para gerar texto detalhado automaticamente:

1. Certifique-se de que o Ollama está rodando:
   ```bash
   ollama serve
   ```

2. Instale o modelo multimodal:
   ```bash
   ollama pull llava
   ```

3. Use o script de integração:
   ```bash
   python integrate_llm.py --image data/raw/example.jpg --ollama-model llava
   ```

O script irá:
- Analisar a imagem com o LLM
- Gerar descrição detalhada
- Atualizar o workflow com o texto gerado
- Enviar para o ComfyUI via API

## ⚙️ Parâmetros Importantes

### ControlNet Strength
- **ControlNet-Tile**: Geralmente entre 0.7-1.0 para preservar detalhes
- **ControlNet-Normal**: Geralmente entre 0.8-1.0 para preservar geometria

### Sampling Parameters
- **Steps**: 20-30 para qualidade vs velocidade
- **CFG Scale**: 7.0-9.0 para controle do prompt
- **Sampler**: Euler ou DPM++ 2M Karras

### Resolução
- Ajuste `EmptyLatentImage` para a resolução desejada
- Recomendado: 512x512 ou 768x768 para início

## 🐛 Troubleshooting

### Erro: "Node not found"
- Instale os custom nodes necessários via ComfyUI Manager
- Verifique se os nodes estão na pasta `custom_nodes`

### Erro: "Model not found"
- Baixe os modelos ControlNet necessários
- Coloque-os na pasta `ComfyUI/models/controlnet/`
- Ajuste os caminhos no workflow

### Normal Map não aparece
- Verifique se o MiDaS está instalado corretamente
- Teste com uma imagem simples primeiro

## 📝 Notas

- Este workflow foca na parte de **refinamento de imagem** usando ControlNet
- A **inicialização de malha 3D** (LRM/InstantMesh) não está incluída, pois requer processamento 3D separado
- O workflow pode ser estendido para incluir mais etapas conforme necessário

## 🔄 Próximos Passos

1. Integrar com módulo de inicialização de malha (LRM/InstantMesh)
2. Adicionar suporte para múltiplas imagens
3. Criar workflow para exportação de malhas 3D
4. Otimizar para processamento em lote

## 📚 Referências

- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [MiDaS](https://github.com/isl-org/MiDaS)
- Artigo base: Kiss3DGen (CVPR 2025)

