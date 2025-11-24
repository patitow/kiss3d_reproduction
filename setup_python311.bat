@echo off
REM Script para configurar ambiente Python 3.11 com CUDA

echo ========================================
echo Configuracao Python 3.11 com CUDA
echo ========================================
echo.

REM Verificar se Python 3.11 esta instalado
python3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python 3.11 nao encontrado!
    echo Por favor, instale Python 3.11 primeiro:
    echo https://www.python.org/downloads/release/python-3110/
    pause
    exit /b 1
)

echo [OK] Python 3.11 encontrado
python3.11 --version
echo.

REM Criar ambiente virtual
echo [1/5] Criando ambiente virtual...
if exist mesh3d-generator-py3.11 (
    echo Ambiente ja existe. Deseja recriar? (S/N)
    set /p recreate=
    if /i "%recreate%"=="S" (
        rmdir /s /q mesh3d-generator-py3.11
        python3.11 -m venv mesh3d-generator-py3.11
    )
) else (
    python3.11 -m venv mesh3d-generator-py3.11
)

echo [OK] Ambiente virtual criado
echo.

REM Ativar ambiente
echo [2/5] Ativando ambiente virtual...
call mesh3d-generator-py3.11\Scripts\activate.bat

REM Atualizar pip
echo [3/5] Atualizando pip...
python -m pip install --upgrade pip

REM Instalar PyTorch com CUDA
echo [4/5] Instalando PyTorch com CUDA 12.1...
echo Isso pode demorar alguns minutos...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Verificar CUDA
echo.
echo Verificando CUDA...
python -c "import torch; print('CUDA disponivel:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

REM Instalar dependencias
echo [5/5] Instalando dependencias do projeto...
pip install numpy pillow einops trimesh diffusers transformers accelerate omegaconf rembg

echo.
echo ========================================
echo Instalacao concluida!
echo ========================================
echo.
echo Para ativar o ambiente no futuro:
echo   mesh3d-generator-py3.11\Scripts\activate.bat
echo.
echo Para instalar Pytorch3D (opcional):
echo   pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html
echo.
pause

