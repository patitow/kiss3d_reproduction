@echo off
REM Script para diagnosticar configuração do CUDA

echo ============================================================
echo DIAGNOSTICO CUDA
echo ============================================================
echo.

echo [1/4] Verificando PyTorch...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA compilado no PyTorch: {torch.version.cuda}'); print(f'CUDA disponivel: {torch.cuda.is_available()}')"
if errorlevel 1 (
    echo [ERRO] PyTorch nao encontrado!
    pause
    exit /b 1
)

echo.
echo [2/4] Verificando instalacoes do CUDA...
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if exist "%CUDA_ROOT%" (
    echo [INFO] Diretorio CUDA encontrado: %CUDA_ROOT%
    echo [INFO] Versoes instaladas:
    dir /b "%CUDA_ROOT%" | findstr /R "^v"
) else (
    echo [ERRO] Diretorio CUDA nao encontrado!
)

echo.
echo [3/4] Verificando nvcc no PATH...
where nvcc >nul 2>&1
if errorlevel 1 (
    echo [AVISO] nvcc nao encontrado no PATH
) else (
    echo [OK] nvcc encontrado:
    where nvcc
    echo.
    echo [INFO] Versao do CUDA detectada pelo nvcc:
    nvcc --version | findstr /C:"release"
)

echo.
echo [4/4] Verificando variaveis de ambiente...
echo CUDA_HOME: %CUDA_HOME%
echo CUDA_PATH: %CUDA_PATH%
echo CUB_HOME: %CUB_HOME%
echo FORCE_CUDA: %FORCE_CUDA%

echo.
echo ============================================================
echo RECOMENDACOES
echo ============================================================
echo.

python -c "import torch; cuda_ver = torch.version.cuda; print(f'PyTorch foi compilado com CUDA: {cuda_ver}')" 2>nul
if errorlevel 1 (
    echo [ERRO] Nao foi possivel verificar versao do CUDA do PyTorch
) else (
    echo.
    echo [INFO] Certifique-se de que o CUDA_HOME aponte para a mesma versao!
    echo [INFO] Exemplo: se PyTorch foi compilado com CUDA 12.1, use:
    echo        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    echo        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    echo        set "PATH=%%CUDA_HOME%%\bin;%%PATH%%"
)

echo.
pause




