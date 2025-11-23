# Download Dataset Gazebo

Script para baixar 200 modelos 3D do Google Research do Gazebo.

## Uso

```bash
# Baixar 200 modelos (padrão)
python scripts/download_from_list.py

# Baixar quantidade customizada
python scripts/download_from_list.py --max-models 100

# Especificar diretório
python scripts/download_from_list.py --output data/meu_dataset
```

## Estrutura de Saída

```
data/raw/gazebo_dataset/
├── models/          # Modelos 3D (um diretório por objeto)
├── images/         # Imagens extraídas
├── metadata/       # Metadados JSON
└── download_progress.json
```

## Dependências

```bash
pip install requests
```

