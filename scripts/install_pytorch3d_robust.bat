@echo off
REM Script robusto para instalar PyTorch3D
REM Tenta múltiplos métodos até conseguir

echo ============================================================
echo INSTALACAO PyTorch3D - METODO ROBUSTO
echo ============================================================
echo.

cd /d "%~dp0\.."
set "PROJECT_ROOT=%CD%"

REM Detectar Python da venv
if exist "%PROJECT_ROOT%\mesh3d-generator-py3.11\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\mesh3d-generator-py3.11\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo [1/3] Verificando PyTorch...
"%PYTHON_CMD%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo [ERRO] PyTorch nao encontrado!
    pause
    exit /b 1
)

echo.
echo [2/3] Configurando Visual Studio 2019...
REM Tentar VS 2019 primeiro (compativel com CUDA 12.1)
set "VCVARS_PATH="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    set "VS_VERSION=2019"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    set "VS_VERSION=2019"
)

if "%VCVARS_PATH%"=="" (
    echo [AVISO] VS 2019 nao encontrado, tentando VS 2022...
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        set "VS_VERSION=2022"
        echo [AVISO] VS 2022 requer CUDA 12.4 ou mais novo!
    )
)

if "%VCVARS_PATH%"=="" (
    echo [ERRO] Visual Studio nao encontrado!
    echo [INFO] Instale Visual Studio 2019 ou 2022 BuildTools
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
echo [3/3] Executando script Python de instalacao...
echo [INFO] O script tentara multiplos metodos automaticamente
echo [INFO] Isso pode demorar 20-40 minutos se precisar compilar...
echo.

"%PYTHON_CMD%" "%PROJECT_ROOT%\scripts\install_pytorch3d_fixed.py"

if errorlevel 1 (
    echo.
    echo [ERRO] Instalacao falhou
    pause
    exit /b 1
) else (
    echo.
    echo [OK] PyTorch3D instalado com sucesso!
    echo.
    echo [INFO] Verificando instalacao...
    "%PYTHON_CMD%" -c "import pytorch3d; from pytorch3d import _C; print(f'PyTorch3D: {pytorch3d.__version__}'); print('Modulo _C OK!')"
    if errorlevel 1 (
        echo [AVISO] PyTorch3D instalado mas _C nao pode ser importado
    ) else (
        echo [OK] Tudo funcionando!
    )
)

pause




