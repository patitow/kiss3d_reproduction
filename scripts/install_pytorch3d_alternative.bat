@echo off
REM Script alternativo para instalar PyTorch3D
REM Tenta diferentes metodos ate encontrar um que funcione

echo ============================================================
echo INSTALACAO ALTERNATIVA PyTorch3D
echo ============================================================
echo.

cd /d "%~dp0\.."

echo [1/5] Verificando ambiente...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo [ERRO] PyTorch nao encontrado!
    pause
    exit /b 1
)

echo.
echo [2/5] Instalando dependencias...
python -m pip install --upgrade pip setuptools wheel
python -m pip install iopath fvcore ninja
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias
    pause
    exit /b 1
)

echo.
echo [3/5] Limpando instalacoes anteriores...
python -m pip uninstall pytorch3d -y
if exist "temp_pytorch3d\build" (
    echo [INFO] Removendo diretorio de build anterior...
    rmdir /s /q "temp_pytorch3d\build" 2>nul
)

echo.
echo [4/5] Tentando metodo 1: Wheel pre-compilado (se disponivel)...
REM Tentar encontrar wheel compativel
python -c "import torch; pytorch_ver = torch.__version__.split('+')[0]; cuda_ver = torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'; print(f'Procurando wheel para PyTorch {pytorch_ver}, CUDA {cuda_ver}')"

REM Tentar instalar via pip com wheel (pode falhar se nao houver wheel)
python -m pip install pytorch3d --no-cache-dir 2>&1 | findstr /C:"Successfully installed" >nul
if errorlevel 1 (
    echo [AVISO] Wheel pre-compilado nao disponivel, tentando compilacao...
    goto :compile_method
) else (
    echo [OK] PyTorch3D instalado via wheel!
    goto :verify
)

:compile_method
echo.
echo [5/5] Tentando metodo 2: Compilacao com flags otimizadas...

REM Configurar CUDA
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if exist "%CUDA_ROOT%\v12.1" (
    set "CUDA_HOME=%CUDA_ROOT%\v12.1"
    set "CUDA_PATH=%CUDA_ROOT%\v12.1"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
) else if exist "%CUDA_ROOT%\v12.2" (
    set "CUDA_HOME=%CUDA_ROOT%\v12.2"
    set "CUDA_PATH=%CUDA_ROOT%\v12.2"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
)

REM Configurar CUB se existir
if exist "C:\cub\CMakeLists.txt" (
    set "CUB_HOME=C:\cub"
)

REM Limitar paralelismo para evitar problemas de memoria
set "MAX_JOBS=2"
set "FORCE_CUDA=1"
set "DISTUTILS_USE_SDK=1"

REM Tentar compilacao com menos paralelismo
echo [INFO] Compilando com MAX_JOBS=2 (pode demorar mais, mas e mais estavel)...
echo [AVISO] Isso pode demorar 20-40 minutos...

cd temp_pytorch3d
if not exist "setup.py" (
    echo [ERRO] Diretorio temp_pytorch3d nao contem setup.py
    echo [INFO] Execute primeiro: scripts\install_pytorch3d_official.bat (ate a parte de clonar)
    cd ..
    pause
    exit /b 1
)

python setup.py install
if errorlevel 1 (
    echo.
    echo [ERRO] Compilacao falhou
    echo [INFO] Verifique os erros acima
    echo [INFO] Execute scripts\get_ninja_errors.bat para mais detalhes
    cd ..
    pause
    exit /b 1
)

cd ..

:verify
echo.
echo [INFO] Verificando instalacao...
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"
if errorlevel 1 (
    echo [ERRO] PyTorch3D instalado mas nao pode ser importado
    pause
    exit /b 1
)

python -c "from pytorch3d import _C; print('PyTorch3D _C OK!')"
if errorlevel 1 (
    echo [AVISO] PyTorch3D instalado mas _C nao pode ser importado
    echo [INFO] Isso pode indicar que foi compilado sem CUDA
) else (
    echo.
    echo [OK] PyTorch3D esta funcionando!
)

pause

