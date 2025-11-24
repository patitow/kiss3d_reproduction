# Resumo da Instalacao

## O QUE VOCE PRECISA FAZER AGORA:

1. **No Visual Studio Installer:**
   - Clique em **"Instalar"** ou **"Modificar"**
   - Aguarde terminar (pode demorar 10-30 minutos)

2. **Depois de instalar:**
   - Feche e reabra o terminal
   - Execute:
     ```bash
     pip install git+https://github.com/NVlabs/nvdiffrast
     ```
   - Isso vai compilar o nvdiffrast (pode demorar 5-15 minutos)

3. **Testar o pipeline:**
   ```bash
   cd Kiss3DGen
   python ../scripts/run_kiss3dgen_simple.py --input ../data/raw/gazebo_dataset/Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3/0.jpg --output ../data/outputs/kiss3dgen_test
   ```

## DEPENDENCIAS JA INSTALADAS:
- PyGLM, open3d, kiui, ninja, sentencepiece, ollama, onnxruntime
- Outras dependencias do Kiss3DGen estao sendo instaladas agora

## O QUE FALTA:
- Visual Studio C++ Build Tools (voce esta instalando)
- nvdiffrast (vai instalar depois do Visual Studio)

