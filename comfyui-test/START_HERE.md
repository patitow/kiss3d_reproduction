# 🎯 START HERE - ComfyUI Test Run

**Bem-vindo!** Este é o ponto de partida para fazer o test run do workflow ComfyUI.

## ⚡ Início Rápido (3 passos)

### 1️⃣ Verificar Setup
```bash
cd comfyui-test
python test_comfyui_connection.py
```

### 2️⃣ Teste Básico (Sem LLM)
```bash
# Coloque uma imagem em ComfyUI/input/
python test_workflow_simple.py --image ComfyUI/input/sua_imagem.jpg
```

### 3️⃣ Teste Completo (Com LLM)
```bash
# Certifique-se de que o Ollama está rodando: ollama serve
python integrate_llm.py --image ComfyUI/input/sua_imagem.jpg
```

## 📚 Documentação Completa

- **[TEST_RUN_GUIDE.md](TEST_RUN_GUIDE.md)** - Guia passo a passo detalhado
- **[setup_checklist.md](setup_checklist.md)** - Checklist de configuração
- **[QUICKSTART.md](QUICKSTART.md)** - Guia rápido de referência
- **[README.md](README.md)** - Documentação completa

## 🎯 O que você vai testar?

1. **Geração de Normal Maps** - A partir da imagem de entrada
2. **Refinamento com ControlNet** - Usando ControlNet-Tile e ControlNet-Normal
3. **Integração com LLM** - Geração de texto detalhado via Ollama (opcional)

## ✅ Pré-requisitos

- [ ] ComfyUI instalado e rodando
- [ ] Custom nodes instalados (comfyui_controlnet_aux)
- [ ] Modelos baixados (Stable Diffusion + ControlNet)
- [ ] Ollama instalado (opcional, para LLM)

**Não tem certeza?** Execute o checklist:
```bash
python test_comfyui_connection.py
```

## 🆘 Precisa de Ajuda?

1. Verifique **[setup_checklist.md](setup_checklist.md)** para problemas comuns
2. Siga **[TEST_RUN_GUIDE.md](TEST_RUN_GUIDE.md)** passo a passo
3. Consulte a seção Troubleshooting em **[README.md](README.md)**

## 🚀 Próximos Passos

Depois do test run bem-sucedido:
1. ✅ Experimente diferentes imagens
2. ✅ Ajuste parâmetros do workflow
3. ✅ Implemente no código Python (próxima etapa do projeto)

---

**Boa sorte! 🎉**


