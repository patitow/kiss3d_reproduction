@echo off
REM Script para capturar erros detalhados do ninja durante compilacao do PyTorch3D

echo ============================================================
echo CAPTURANDO ERROS DETALHADOS DO NINJA
echo ============================================================
echo.

cd /d "%~dp0\.."

REM Procurar logs do ninja no diretorio de build
set "BUILD_DIR=temp_pytorch3d\build"
if not exist "%BUILD_DIR%" (
    echo [ERRO] Diretorio de build nao encontrado: %BUILD_DIR%
    echo [INFO] Execute a compilacao primeiro
    pause
    exit /b 1
)

echo [INFO] Procurando logs de erro no diretorio de build...
echo.

REM Procurar arquivos de log
echo === ARQUIVOS DE LOG ENCONTRADOS ===
dir /s /b "%BUILD_DIR%\*.log" 2>nul
if errorlevel 1 (
    echo [AVISO] Nenhum arquivo .log encontrado
)

echo.
echo === ARQUIVOS DE ERRO ENCONTRADOS ===
dir /s /b "%BUILD_DIR%\*.err" 2>nul
if errorlevel 1 (
    echo [AVISO] Nenhum arquivo .err encontrado
)

echo.
echo === VERIFICANDO NINJA ===
where ninja >nul 2>&1
if errorlevel 1 (
    echo [ERRO] ninja nao encontrado no PATH
    echo [INFO] Instale ninja: pip install ninja
) else (
    echo [OK] ninja encontrado:
    where ninja
    ninja --version
)

echo.
echo === VERIFICANDO CUDA ===
where nvcc >nul 2>&1
if errorlevel 1 (
    echo [AVISO] nvcc nao encontrado no PATH
) else (
    echo [OK] nvcc encontrado:
    where nvcc
    nvcc --version | findstr /C:"release"
)

echo.
echo === VERIFICANDO COMPILADOR C++ ===
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERRO] cl.exe (MSVC) nao encontrado no PATH
    echo [INFO] Certifique-se de estar no "x64 Native Tools Command Prompt"
) else (
    echo [OK] cl.exe encontrado:
    where cl
    cl 2>&1 | findstr /C:"Microsoft"
)

echo.
echo === ULTIMOS ARQUIVOS MODIFICADOS NO BUILD ===
for /f "delims=" %%f in ('dir /s /b /o-d "%BUILD_DIR%" 2^>nul ^| findstr /i "\.obj \.o \.pyd \.dll" ^| head -n 10') do (
    echo %%f
)

echo.
echo ============================================================
echo RECOMENDACOES
echo ============================================================
echo.
echo Se o erro persistir, tente:
echo 1. Limpar o diretorio de build: rmdir /s /q temp_pytorch3d\build
echo 2. Reinstalar ninja: pip install --upgrade ninja
echo 3. Verificar se ha espaco em disco suficiente
echo 4. Tentar compilacao com menos paralelismo: set MAX_JOBS=1
echo.

pause





