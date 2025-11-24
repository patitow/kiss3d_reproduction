@echo off
REM Ativa a venv e configura variaveis do Kiss3DGen

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "VENV_DIR=%PROJECT_ROOT%\mesh3d-generator-py3.11"
set "CUDA_CANDIDATE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

if exist "%CUDA_CANDIDATE%" (
    set "CUDA_HOME=%CUDA_CANDIDATE%"
    set "CUDA_PATH=%CUDA_CANDIDATE%"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
) else (
    echo [AVISO] CUDA 12.1 nao encontrado em %CUDA_CANDIDATE%.
    echo         Ajuste CUDA_HOME/CUDA_PATH manualmente se usar outro caminho.
)

if exist "%PROJECT_ROOT%\ninja.exe" (
    set "PATH=%PROJECT_ROOT%;%PATH%"
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERRO] Venv nao encontrada em %VENV_DIR%.
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Ambiente Kiss3DGen carregado.
echo    - PROJECT_ROOT=%PROJECT_ROOT%
echo    - CUDA_HOME=%CUDA_HOME%
echo    - PATH atualizado com CUDA e ninja

