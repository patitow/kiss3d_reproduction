# Instalacao do nvdiffrast

O nvdiffrast precisa ser compilado e requer:

1. **Visual Studio C++ Build Tools** (ou Visual Studio completo)
   - Baixe de: https://visualstudio.microsoft.com/downloads/
   - Instale "Desktop development with C++"

2. **CUDA Toolkit** (ja instalado - v12.1)

3. **Compilar nvdiffrast**:
   ```bash
   pip install git+https://github.com/NVlabs/nvdiffrast
   ```

   Isso pode demorar varios minutos na primeira vez (compilacao CUDA).

**Alternativa**: Se a compilacao falhar, voce pode:
- Instalar Visual Studio Build Tools
- Ou usar uma versao pre-compilada se disponivel
- Ou modificar o codigo para nao usar nvdiffrast (renderizacao alternativa)

