# 🚀 Migração para Python 3.11 com CUDA

## Resumo

Você precisa migrar de **Python 3.13** para **Python 3.11** para ter suporte completo a:
- ✅ Pytorch3D (renderização avançada)
- ✅ PyTorch com CUDA (aceleração GPU)
- ✅ Todos os modelos de difusão funcionando corretamente

## ⚡ Instalação Rápida

### Opção 1: Script Automatizado (Mais Fácil)

**PowerShell:**
```powershell
.\setup_python311.ps1
```

**CMD:**
```cmd
setup_python311.bat
```

### Opção 2: Manual

1. **Instalar Python 3.11**:
   - Download: https://www.python.org/downloads/release/python-3110/
   - ✅ Marcar "Add Python to PATH"

2. **Criar ambiente virtual**:
   ```bash
   python3.11 -m venv mesh3d-generator-py3.11
   .\mesh3d-generator-py3.11\Scripts\Activate.ps1
   ```

3. **Instalar PyTorch com CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Instalar dependências**:
   ```bash
   pip install numpy pillow einops trimesh diffusers transformers accelerate omegaconf rembg
   ```

5. **Verificar**:
   ```bash
   python verify_setup.py
   ```

## ✅ Verificação

Após instalação, execute:
```bash
python verify_setup.py
```

**Deve mostrar:**
- ✅ Python: 3.11.x
- ✅ CUDA: True
- ✅ GPU: NVIDIA GeForce RTX 3060
- ✅ Diffusers: instalado
- ✅ Pipeline: OK

## 📚 Documentação Completa

- **Guia completo**: `MIGRACAO_PYTHON_3.11.md`
- **Quick start**: `QUICK_START_PYTHON311.md`
- **Dependências**: `INSTALL_DEPENDENCIES.md`

## 🎯 Próximos Passos

Após migração bem-sucedida:

1. ✅ Ambiente Python 3.11 criado
2. ✅ PyTorch com CUDA instalado
3. ✅ Dependências instaladas
4. ⏭️ Testar pipeline: `python scripts/run_3d_pipeline.py --max-objects 1`
5. ⏭️ Modelos serão baixados automaticamente na primeira execução

## ⚠️ Notas Importantes

- **Espaço em disco**: Modelos precisam de ~30GB+ (Flux ~23GB, Zero123 ~5GB, etc)
- **Tempo de download**: Primeira execução pode demorar (download de modelos)
- **VRAM**: RTX 3060 12GB é suficiente, mas modelos grandes podem precisar de quantização

