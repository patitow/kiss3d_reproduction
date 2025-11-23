# ⚡ Quick Start - Download Dataset Gazebo

## 🚀 Início Rápido

### 1. Instalar Dependências
```bash
pip install -r scripts/requirements_download.txt
```

### 2. Baixar Dataset (200 objetos)
```bash
python scripts/download_gazebo_dataset.py
```

### 3. Baixar Mais Objetos
```bash
python scripts/download_gazebo_dataset.py --max-objects 300
```

## 📊 O que será baixado?

Para cada objeto:
- ✅ Modelo 3D (arquivos .sdf, .dae, .obj, etc.)
- ✅ Imagens (thumbnails, renders)
- ✅ Metadados (informações do objeto em JSON)

## 📁 Onde será salvo?

Por padrão: `data/raw/gazebo_dataset/`

Estrutura:
```
data/raw/gazebo_dataset/
├── models/          # Modelos 3D
├── images/         # Imagens
├── metadata/       # Metadados JSON
└── download_progress.json
```

## ⏱️ Tempo Estimado

- 200 objetos: ~2-4 horas (depende da conexão)
- Progresso salvo automaticamente
- Pode interromper e continuar depois

## 🔍 Verificar Progresso

```bash
cat data/raw/gazebo_dataset/download_progress.json
```

## 📚 Documentação Completa

Veja `scripts/README_DOWNLOAD.md` para mais detalhes.

---

**Pronto para começar!** 🎉

