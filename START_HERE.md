# 🚀 COMEÇAR AQUI - Migração Python 3.11 + CUDA

## ⚡ Instalação Rápida (3 passos)

### 1. Instalar Python 3.11
- Download: https://www.python.org/downloads/release/python-3110/
- ✅ **IMPORTANTE**: Marcar "Add Python to PATH"
- Verificar: `python3.11 --version`

### 2. Executar Script Automatizado

**PowerShell (Recomendado):**
```powershell
.\setup_python311.ps1
```

**Ou CMD:**
```cmd
setup_python311.bat
```

O script vai:
- ✅ Criar ambiente virtual Python 3.11
- ✅ Instalar PyTorch com CUDA 12.1
- ✅ Instalar todas as dependências
- ✅ Verificar instalação

### 3. Verificar Instalação

```bash
python verify_setup.py
```

**Deve mostrar:**
- ✅ Python: 3.11.x
- ✅ CUDA: True
- ✅ GPU: NVIDIA GeForce RTX 3060
- ✅ Todas as dependências OK

## 📋 Status Atual

- **Python**: 3.13.3 → Migrar para 3.11
- **PyTorch**: 2.9.0+cpu → Instalar com CUDA
- **GPU**: RTX 3060 12GB ✅ (detectada)
- **CUDA**: 13.0 ✅ (drivers instalados)

## 📚 Documentação

- **Guia completo**: `MIGRACAO_PYTHON_3.11.md`
- **Quick start**: `QUICK_START_PYTHON311.md`
- **Dependências**: `INSTALL_DEPENDENCIES.md`

## ⚠️ Importante

1. **Espaço em disco**: Modelos precisam de ~30GB+
2. **Tempo**: Primeira execução baixa modelos (~30GB)
3. **VRAM**: RTX 3060 12GB é suficiente

## 🎯 Após Migração

Testar pipeline:
```bash
python scripts/run_3d_pipeline.py --max-objects 1 --output data/outputs/test
```

