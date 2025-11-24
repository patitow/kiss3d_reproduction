# 📦 Guia de Instalação - Kiss3DGen Image to 3D

## ⚠️ IMPORTANTE: Python 3.11.9 é OBRIGATÓRIO

Este projeto **requer Python 3.11.9** especificamente. Não use Python 3.12+ ou 3.10-.

---

## 🚀 Instalação Rápida (Windows)

### Passo 1: Instalar Python 3.11.9

1. Baixar Python 3.11.9: https://www.python.org/downloads/release/python-3119/
2. Durante instalação, **marcar "Add Python to PATH"**
3. Verificar instalação:
   ```powershell
   python3.11 --version
   # Deve mostrar: Python 3.11.9
   ```

### Passo 2: Configurar Ambiente Virtual

```powershell
# Opção 1: Script automatizado (recomendado)
.\scripts\setup_python311.ps1

# Opção 2: Manual
python3.11 -m venv mesh3d-generator-py3.11
.\mesh3d-generator-py3.11\Scripts\Activate.ps1
```

### Passo 3: Instalar PyTorch com CUDA

```powershell
# Para CUDA 12.1 (RTX 30xx/40xx):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para CUDA 11.8 (RTX 20xx):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar instalação:
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Passo 4: Instalar Visual Studio C++ Build Tools

**OBRIGATÓRIO** para compilar `nvdiffrast`:

1. Baixar Visual Studio Installer: https://visualstudio.microsoft.com/downloads/
2. Instalar "Desktop development with C++"
3. Especificamente instalar: **MSVC v143 - VS 2022 C++ Build Tools**

### Passo 5: Instalar Dependências

```powershell
# Opção 1: Script automatizado (recomendado)
python scripts/install_dependencies.py

# Opção 2: Manual
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### Passo 6: Instalar Diffusers Customizado

```powershell
cd Kiss3DGen
pip install -e custom_diffusers/
cd ..
```

### Passo 7: Autenticar HuggingFace

```powershell
# Opção 1: CLI
huggingface-cli login

# Opção 2: Script
python scripts/setup_huggingface_auth.py
```

### Passo 8: Baixar Modelos

```powershell
# Modelos obrigatórios
python scripts/download_models.py

# Modelos opcionais (melhor qualidade)
python scripts/download_redux.py
python scripts/download_lrm.py
```

---

## ✅ Verificação da Instalação

Execute para verificar se tudo está instalado:

```powershell
python scripts/check_and_download_models.py
```

Deve mostrar:
- ✅ Python 3.11.9
- ✅ PyTorch com CUDA
- ✅ nvdiffrast
- ✅ PyTorch3D
- ✅ Modelos baixados

---

## 🐛 Problemas Comuns

### Erro: "Python 3.11 não encontrado"
- **Solução**: Instalar Python 3.11.9 e marcar "Add Python to PATH"
- Verificar: `python3.11 --version`

### Erro: "Ninja is required to load C++ extensions"
- **Solução**: `pip install ninja`

### Erro: "Error building extension 'nvdiffrast_plugin'"
- **Solução**: Instalar Visual Studio C++ Build Tools (Passo 4)

### Erro: "CUDA_HOME environment variable is not set"
- **Solução**: Definir variável de ambiente:
  ```powershell
  $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
  ```

### Erro: "CUDA out of memory"
- **Solução**: Reduzir batch size no `pipeline_config/default.yaml`

### Erro: "Model not found" no HuggingFace
- **Solução**: Autenticar no HuggingFace (Passo 7)

---

## 📋 Checklist de Instalação

- [ ] Python 3.11.9 instalado
- [ ] Ambiente virtual criado e ativado
- [ ] PyTorch com CUDA instalado
- [ ] Visual Studio C++ Build Tools instalado
- [ ] nvdiffrast compilado e instalado
- [ ] PyTorch3D instalado
- [ ] Diffusers customizado instalado
- [ ] HuggingFace autenticado
- [ ] Modelos baixados (Zero123++, Flux, ControlNet)
- [ ] Scripts de verificação passando

---

## 🎯 Próximos Passos

Após instalação completa:

```powershell
# Testar pipeline
python scripts/run_kiss3dgen_image_to_3d.py --input data/raw/exemplo.jpg --output data/outputs/
```

---

## 📚 Recursos

- **Python 3.11.9**: https://www.python.org/downloads/release/python-3119/
- **Visual Studio Build Tools**: https://visualstudio.microsoft.com/downloads/
- **PyTorch**: https://pytorch.org/get-started/locally/
- **HuggingFace**: https://huggingface.co/

