@echo off
REM Script para configurar ambiente Python 3.11.9 no Windows

echo ============================================================
echo CONFIGURACAO DO AMBIENTE PYTHON 3.11.9
echo ============================================================

REM Verificar se Python 3.11 esta instalado
python3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python 3.11 nao encontrado!
    echo [INFO] Instale Python 3.11.9: https://www.python.org/downloads/release/python-3119/
    echo [INFO] Certifique-se de marcar "Add Python to PATH" durante a instalacao
    pause
    exit /b 1
)

echo [OK] Python 3.11 encontrado
python3.11 --version

REM Criar ambiente virtual
echo.
echo [1/3] Criando ambiente virtual...
if exist mesh3d-generator-py3.11 (
    echo [AVISO] Ambiente virtual ja existe
    set /p recreate="Deseja recriar? (s/N): "
    if /i "%recreate%"=="s" (
        rmdir /s /q mesh3d-generator-py3.11
        python3.11 -m venv mesh3d-generator-py3.11
    )
) else (
    python3.11 -m venv mesh3d-generator-py3.11
)

if errorlevel 1 (
    echo [ERRO] Falha ao criar ambiente virtual
    pause
    exit /b 1
)

echo [OK] Ambiente virtual criado

REM Ativar ambiente virtual
echo.
echo [2/3] Ativando ambiente virtual...
call mesh3d-generator-py3.11\Scripts\activate.bat

REM Verificar versao
echo.
echo [3/3] Verificando versao do Python no ambiente virtual...
python --version

REM Atualizar pip
echo.
echo [INFO] Atualizando pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo ============================================================
echo [OK] Ambiente Python 3.11.9 configurado!
echo ============================================================
echo.
echo Proximos passos:
echo 1. Instalar PyTorch com CUDA:
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo 2. Instalar dependencias:
echo    pip install -r requirements.txt
echo    OU
echo    python scripts/install_dependencies.py
echo.
echo 3. Autenticar HuggingFace:
echo    huggingface-cli login
echo.
echo 4. Baixar modelos:
echo    python scripts/download_models.py
echo.
pause

