@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Script para configurar ambiente Python 3.11.9 no Windows

REM Detectar diretorios principais
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.."
set "PROJECT_ROOT=%CD%"
popd
set "VENV_DIR=%PROJECT_ROOT%\mesh3d-generator-py3.11"
set "CUDA_DEFAULT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "NINJA_BIN=%PROJECT_ROOT%\ninja.exe"
set "ACTIVATE_SCRIPT=%PROJECT_ROOT%\scripts\activate_kiss3d_env.bat"

echo ============================================================
echo CONFIGURACAO DO AMBIENTE PYTHON 3.11.9
echo Raiz do projeto: %PROJECT_ROOT%
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
if exist "%VENV_DIR%" (
    echo [AVISO] Ambiente virtual ja existe
    set /p recreate="Deseja recriar? (s/N): "
    if /i "%recreate%"=="s" (
        rmdir /s /q "%VENV_DIR%"
        python3.11 -m venv "%VENV_DIR%"
    )
) else (
    python3.11 -m venv "%VENV_DIR%"
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
call "%VENV_DIR%\Scripts\activate.bat"

REM Verificar versao
echo.
echo [3/3] Verificando versao do Python no ambiente virtual...
python --version

REM Atualizar pip
echo.
echo [INFO] Atualizando pip...
python -m pip install --upgrade pip setuptools wheel

REM Configurar variaveis de ambiente persistentes
echo.
echo [INFO] Configurando variaveis de ambiente (CUDA / ninja)...
if exist "%CUDA_DEFAULT%" (
    set "CURRENT_CUDA_HOME="
    for /f "tokens=2*" %%A in ('reg query HKCU\Environment /v CUDA_HOME 2^>nul ^| find "REG_SZ"') do (
        set "CURRENT_CUDA_HOME=%%B"
    )
    if /I "!CURRENT_CUDA_HOME!"=="%CUDA_DEFAULT%" (
        echo [OK] CUDA_HOME ja aponta para %CUDA_DEFAULT%
    ) else (
        setx CUDA_HOME "%CUDA_DEFAULT%" >nul
        setx CUDA_PATH "%CUDA_DEFAULT%" >nul
        echo [OK] CUDA_HOME/CUDA_PATH configurados para %CUDA_DEFAULT%
    )
    call :EnsureUserPath "%CUDA_DEFAULT%\bin"
) else (
    echo [AVISO] CUDA 12.1 nao encontrado em %CUDA_DEFAULT%. Ajuste manualmente se estiver em outro local.
)

if exist "%NINJA_BIN%" (
    call :EnsureUserPath "%PROJECT_ROOT%"
) else (
    echo [AVISO] ninja.exe nao encontrado em %PROJECT_ROOT%. Coloque o binario la ou ajuste o caminho manualmente.
)

set "CACHE_ROOT=%PROJECT_ROOT%\.cache"
set "HF_CACHE_DIR=%CACHE_ROOT%\huggingface"
set "TORCH_CACHE_DIR=%CACHE_ROOT%\torch"

if not exist "%CACHE_ROOT%" (
    mkdir "%CACHE_ROOT%"
)
if not exist "%HF_CACHE_DIR%" (
    mkdir "%HF_CACHE_DIR%"
)
if not exist "%TORCH_CACHE_DIR%" (
    mkdir "%TORCH_CACHE_DIR%"
)

call :EnsureEnvVar HF_HOME "%HF_CACHE_DIR%"
call :EnsureEnvVar HUGGINGFACE_HUB_CACHE "%HF_CACHE_DIR%\hub"
call :EnsureEnvVar TRANSFORMERS_CACHE "%HF_CACHE_DIR%\transformers"
call :EnsureEnvVar DIFFUSERS_CACHE "%HF_CACHE_DIR%\diffusers"
call :EnsureEnvVar TORCH_HOME "%TORCH_CACHE_DIR%"
call :EnsureEnvVar XFORMERS_FORCE_DISABLE_TRITON "1"
call :EnsureEnvVar PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"

REM Gerar script de ativacao rapida com variaveis de ambiente
(
    echo @echo off
    echo REM Ativa venv e ajusta variaveis para Kiss3DGen
    echo set "PROJECT_ROOT=%PROJECT_ROOT%"
    echo set "VENV_DIR=%VENV_DIR%"
    if exist "%CUDA_DEFAULT%" (
        echo set "CUDA_HOME=%CUDA_DEFAULT%"
        echo set "CUDA_PATH=%CUDA_DEFAULT%"
        echo set "PATH=%%CUDA_HOME%%\bin;%%PATH%%"
    ) else (
        echo REM Ajuste CUDA_HOME/CUDA_PATH conforme sua instalacao
    )
    echo set "CACHE_ROOT=%%PROJECT_ROOT%%\.cache"
    echo set "HF_HOME=%%CACHE_ROOT%%\huggingface"
    echo set "HUGGINGFACE_HUB_CACHE=%%HF_HOME%%\hub"
    echo set "TRANSFORMERS_CACHE=%%HF_HOME%%\transformers"
    echo set "DIFFUSERS_CACHE=%%HF_HOME%%\diffusers"
    echo set "TORCH_HOME=%%CACHE_ROOT%%\torch"
    echo if not exist "%%HF_HOME%%" mkdir "%%HF_HOME%%"
    echo if not exist "%%TORCH_HOME%%" mkdir "%%TORCH_HOME%%"
    echo if exist "%%PROJECT_ROOT%%\ninja.exe" set "PATH=%%PROJECT_ROOT%%;%%PATH%%"
    echo call "%%VENV_DIR%%\Scripts\activate.bat"
    echo echo [OK] Ambiente Kiss3DGen pronto.
) > "%ACTIVATE_SCRIPT%"
echo [OK] Script de ativacao criado em %ACTIVATE_SCRIPT%

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
goto :EOF

:EnsureUserPath
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" goto :EOF
set "USER_PATH="
for /f "tokens=2*" %%A in ('reg query HKCU\Environment /v Path 2^>nul ^| find "REG_SZ"') do (
    set "USER_PATH=%%B"
)
echo !USER_PATH! | find /I "%TARGET_DIR%" >nul
if errorlevel 1 (
    if defined USER_PATH (
        set "NEW_USER_PATH=!USER_PATH!;%TARGET_DIR%"
    ) else (
        set "NEW_USER_PATH=%TARGET_DIR%"
    )
    setx Path "!NEW_USER_PATH!" >nul
    echo [OK] PATH do usuario atualizado com %TARGET_DIR%
) else (
    echo [OK] PATH do usuario ja contem %TARGET_DIR%
)
exit /b 0

:EnsureEnvVar
set "VAR_NAME=%~1"
set "VAR_VALUE=%~2"
if "%VAR_NAME%"=="" goto :EOF
if "%VAR_VALUE%"=="" goto :EOF
for /f "tokens=2*" %%A in ('reg query HKCU\Environment /v %VAR_NAME% 2^>nul ^| find "REG_SZ"') do (
    set "CURRENT_VALUE=%%B"
)
if /I "!CURRENT_VALUE!"=="%VAR_VALUE%" (
    echo [OK] %VAR_NAME% ja aponta para %VAR_VALUE%
) else (
    setx %VAR_NAME% "%VAR_VALUE%" >nul
    echo [OK] %VAR_NAME% definido para %VAR_VALUE%
)
set "CURRENT_VALUE="
goto :EOF
