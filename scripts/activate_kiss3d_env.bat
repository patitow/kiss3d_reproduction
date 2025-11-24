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

set "CACHE_ROOT=%PROJECT_ROOT%\.cache"
set "HF_CACHE_DIR=%CACHE_ROOT%\huggingface"
set "TORCH_CACHE_DIR=%CACHE_ROOT%\torch"

if not exist "%HF_CACHE_DIR%" (
    mkdir "%HF_CACHE_DIR%"
)
if not exist "%TORCH_CACHE_DIR%" (
    mkdir "%TORCH_CACHE_DIR%"
)

set "HF_HOME=%HF_CACHE_DIR%"
set "HUGGINGFACE_HUB_CACHE=%HF_CACHE_DIR%\hub"
set "TRANSFORMERS_CACHE=%HF_CACHE_DIR%\transformers"
set "DIFFUSERS_CACHE=%HF_CACHE_DIR%\diffusers"
set "TORCH_HOME=%TORCH_CACHE_DIR%"
set "XFORMERS_FORCE_DISABLE_TRITON=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

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

