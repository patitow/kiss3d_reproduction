# 📥 Download do Dataset Google Research - Gazebo

Script para baixar pelo menos 200 objetos do dataset do Google Research do Gazebo, incluindo imagens e modelos 3D.

## 🚀 Uso Rápido

```bash
# Instalar dependências
pip install -r scripts/requirements_download.txt

# Baixar 200 objetos (padrão)
python scripts/download_gazebo_dataset.py

# Baixar quantidade customizada
python scripts/download_gazebo_dataset.py --max-objects 300

# Especificar diretório de saída
python scripts/download_gazebo_dataset.py --output data/my_dataset
```

## 📋 Funcionalidades

- ✅ Lista automaticamente modelos do GoogleResearch
- ✅ Baixa modelos 3D (arquivos .sdf, .dae, .obj, etc.)
- ✅ Baixa imagens associadas (thumbnails, renders)
- ✅ Salva metadados em JSON
- ✅ Progresso salvo automaticamente
- ✅ Retry automático em caso de falhas
- ✅ Rate limiting para não sobrecarregar servidor

## 📁 Estrutura de Saída

```
data/raw/gazebo_dataset/
├── models/              # Modelos 3D (um por objeto)
│   ├── model_name_1/
│   │   ├── model.sdf
│   │   ├── meshes/
│   │   └── materials/
│   └── model_name_2/
├── images/             # Imagens dos objetos
│   ├── model_name_1.jpg
│   └── model_name_2.png
├── metadata/           # Metadados JSON
│   ├── model_name_1.json
│   └── model_name_2.json
└── download_progress.json  # Progresso do download
```

## ⚙️ Opções

### `--output DIR`
Diretório onde salvar o dataset (padrão: `data/raw/gazebo_dataset`)

### `--max-objects N`
Número máximo de objetos para baixar (padrão: 200)

## 🔧 Como Funciona

1. **Listagem de Modelos**: Acessa a página de busca do Gazebo e extrai lista de modelos do GoogleResearch
2. **Download de Metadados**: Para cada modelo, obtém informações via API ou scraping
3. **Download de Arquivos**: Baixa arquivos 3D (ZIP) e extrai automaticamente
4. **Download de Imagens**: Extrai e baixa imagens associadas aos modelos
5. **Salvamento**: Organiza tudo em estrutura de diretórios

## 📊 Progresso

O script salva progresso a cada 10 modelos em `download_progress.json`:

```json
{
  "total_models": 200,
  "processed": 50,
  "downloaded": 48,
  "failed": 2,
  "models": ["model1", "model2", ...]
}
```

## ⚠️ Notas

- O download pode levar várias horas dependendo da conexão
- O script respeita rate limiting (0.5s entre downloads)
- Falhas individuais não interrompem o processo
- Use `--max-objects` para testar com poucos objetos primeiro

## 🐛 Troubleshooting

### "Connection timeout"
- Verifique sua conexão com internet
- Tente novamente mais tarde

### "Model not found"
- Alguns modelos podem ter sido removidos
- O script continua com os próximos

### "Rate limit exceeded"
- O script já tem rate limiting, mas se necessário:
- Aumente o delay em `time.sleep()` no código

## 📚 Referências

- [Gazebo Fuel](https://app.gazebosim.org/)
- [Google Research Models](https://app.gazebosim.org/search;q=GoogleResearch)
- [Fuel API Documentation](https://fuel.gazebosim.org/docs)

---

**Desenvolvido para:** Mesh3D Generator - Visão Computacional 2025.2

