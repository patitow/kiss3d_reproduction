@echo off
REM Script para instalar PyTorch3D usando o método oficial do repositório
REM Deve ser executado no "x64 Native Tools Command Prompt for VS 2022"

echo ============================================================
echo INSTALACAO PyTorch3D - METODO OFICIAL WINDOWS
echo ============================================================
echo.
echo [AVISO] Este script deve ser executado no:
echo         "x64 Native Tools Command Prompt for VS 2022"
echo.

cd /d "%~dp0\.."
set "PROJECT_ROOT=%CD%"

REM Detectar Python da venv
if exist "%PROJECT_ROOT%\mesh3d-generator-py3.11\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\mesh3d-generator-py3.11\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo [1/4] Verificando PyTorch...
"%PYTHON_CMD%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo [ERRO] PyTorch nao encontrado!
    pause
    exit /b 1
)

echo.
echo [2/4] Preparando codigo-fonte...
if not exist "temp_pytorch3d" (
    echo [INFO] Clonando repositorio PyTorch3D...
    git clone https://github.com/facebookresearch/pytorch3d.git temp_pytorch3d
    if errorlevel 1 (
        echo [ERRO] Falha ao clonar repositorio
        pause
        exit /b 1
    )
)

cd temp_pytorch3d
git checkout stable
if errorlevel 1 (
    echo [ERRO] Falha ao fazer checkout
    pause
    exit /b 1
)

echo.
echo [3/4] Instalando dependencias...
"%PYTHON_CMD%" -m pip install iopath
if errorlevel 1 (
    echo [ERRO] Falha ao instalar iopath
    pause
    exit /b 1
)

echo.
echo [4/5] Configurando CUB (opcional, mas recomendado)...
REM CUB e necessario para CUDA < 11.7, mas configurar evita avisos
if exist "C:\cub\CMakeLists.txt" (
    set "CUB_HOME=C:\cub"
    echo [OK] CUB encontrado em C:\cub
) else (
    echo [AVISO] CUB nao encontrado em C:\cub
    echo [INFO] Para CUDA 12.1, CUB nao e obrigatorio, mas pode gerar avisos
    echo [INFO] Se quiser instalar: baixe de https://github.com/NVIDIA/cub/releases
    echo [INFO] Extraia para C:\cub e execute este script novamente
)

echo.
echo [5/6] Configurando Visual Studio...
REM VS 2019 e compativel com CUDA 12.1, VS 2022 requer CUDA 12.4+
REM Tentar VS 2019 primeiro (compativel com CUDA 12.1)
set "VCVARS_PATH="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    set "VS_VERSION=2019"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    set "VS_VERSION=2019"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    set "VS_VERSION=2022"
    echo [AVISO] Usando VS 2022 - requer CUDA 12.4 ou mais novo!
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    set "VS_VERSION=2022"
    echo [AVISO] Usando VS 2022 - requer CUDA 12.4 ou mais novo!
)

if "%VCVARS_PATH%"=="" (
    echo [ERRO] Visual Studio nao encontrado!
    echo [INFO] Instale Visual Studio 2019 ou 2022 BuildTools/Community
    echo [INFO] Com componente "Desktop development with C++"
    echo [INFO] VS 2019 e recomendado para CUDA 12.1
    pause
    exit /b 1
)
echo [OK] VS %VS_VERSION% encontrado: %VCVARS_PATH%

echo [INFO] Configurando ambiente VS %VS_VERSION%...
call "%VCVARS_PATH%"
if errorlevel 1 (
    echo [ERRO] Falha ao configurar ambiente VS %VS_VERSION%
    pause
    exit /b 1
)
echo [OK] Ambiente VS %VS_VERSION% configurado

echo.
echo [6/6] Compilando e instalando PyTorch3D...
echo [AVISO] Isso pode demorar 15-30 minutos...
echo [INFO] Aguarde...
echo.

REM Configurar CUDA 12.1 (mesma versao do PyTorch)
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if exist "%CUDA_ROOT%\v12.1" (
    set "CUDA_HOME=%CUDA_ROOT%\v12.1"
    set "CUDA_PATH=%CUDA_ROOT%\v12.1"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    echo [INFO] CUDA_HOME configurado para v12.1
) else if exist "%CUDA_ROOT%\v12.2" (
    set "CUDA_HOME=%CUDA_ROOT%\v12.2"
    set "CUDA_PATH=%CUDA_ROOT%\v12.2"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    echo [INFO] CUDA_HOME configurado para v12.2
) else (
    echo [ERRO] CUDA 12.1 ou 12.2 nao encontrado!
    echo [INFO] Verifique se o CUDA Toolkit esta instalado
    pause
    exit /b 1
)

REM Verificar se nvcc esta no PATH
where nvcc >nul 2>&1
if errorlevel 1 (
    echo [AVISO] nvcc nao encontrado no PATH
    echo [INFO] Tentando adicionar CUDA bin ao PATH...
    set "PATH=%CUDA_HOME%\bin;%PATH%"
) else (
    echo [OK] nvcc encontrado
)

REM Verificar versao do CUDA detectada
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Nao foi possivel verificar versao do CUDA
) else (
    echo [INFO] Verificando versao do CUDA...
    nvcc --version | findstr /C:"release"
)

set FORCE_CUDA=1
set DISTUTILS_USE_SDK=1
set MAX_JOBS=1
echo [INFO] Compilando com MAX_JOBS=1...
echo [INFO] Isso pode demorar 20-40 minutos...
"%PYTHON_CMD%" setup.py install --verbose

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na compilacao
    echo [INFO] Verifique os erros acima
    echo [INFO] O erro mais comum e incompatibilidade entre versao do CUDA e compilador
    echo [INFO] VS 2019 e compativel com CUDA 12.1
    echo [INFO] VS 2022 requer CUDA 12.4 ou mais novo
    pause
    exit /b 1
)

echo.
echo [OK] PyTorch3D compilado e instalado!
echo.
echo [INFO] Verificando instalacao...
"%PYTHON_CMD%" -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"
"%PYTHON_CMD%" -c "from pytorch3d import _C; print('PyTorch3D _C OK!')"

if errorlevel 1 (
    echo [AVISO] PyTorch3D instalado mas _C nao pode ser importado
) else (
    echo.
    echo [OK] PyTorch3D esta funcionando com CUDA!
)

cd ..
pause

