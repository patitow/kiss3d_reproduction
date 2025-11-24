# Proximos Passos - Apos Instalar Visual Studio

## 1. Instalar Visual Studio
- Clique em **"Instalar"** ou **"Modificar"** no Visual Studio Installer
- Aguarde a instalacao terminar (pode demorar alguns minutos)

## 2. Reiniciar Terminal
- Feche o terminal atual
- Abra um novo terminal PowerShell
- Ative o ambiente virtual:
  ```powershell
  .\mesh3d-generator-py3.11\Scripts\Activate.ps1
  ```

## 3. Instalar nvdiffrast
```bash
pip install git+https://github.com/NVlabs/nvdiffrast
```
**Isso vai demorar varios minutos** - esta compilando extensoes CUDA.

## 4. Testar Pipeline Kiss3DGen
```bash
cd Kiss3DGen
python ../scripts/run_kiss3dgen_simple.py --input ../data/raw/gazebo_dataset/Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3/0.jpg --output ../data/outputs/kiss3dgen_test
```

## 5. Se der erro, instalar outras dependencias do Kiss3DGen
```bash
pip install imageio-ffmpeg Imath jaxtyping mathutils PyMCubes pymeshlab PyOpenGL pygltflib OpenEXR xatlas fvcore pytorch-lightning PEFT pyrender timm
```

