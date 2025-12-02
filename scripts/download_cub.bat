@echo off
REM Script para baixar e configurar NVIDIA CUB

echo ============================================================
echo BAIXANDO E CONFIGURANDO NVIDIA CUB
echo ============================================================
echo.

set "CUB_PATH=C:\cub"
set "CUB_ZIP=%TEMP%\cub.zip"
set "CUB_URL=https://github.com/NVIDIA/cub/archive/refs/tags/2.1.0.zip"

REM Verificar se ja existe
if exist "%CUB_PATH%\CMakeLists.txt" (
    echo [OK] CUB ja esta instalado em: %CUB_PATH%
    set "CUB_HOME=%CUB_PATH%"
    echo [OK] CUB_HOME configurado: %CUB_PATH%
    pause
    exit /b 0
)

echo [INFO] Baixando CUB 2.1.0...
echo [INFO] URL: %CUB_URL%
echo [INFO] Destino: %CUB_ZIP%
echo.

REM Tentar baixar usando PowerShell
powershell -Command "Invoke-WebRequest -Uri '%CUB_URL%' -OutFile '%CUB_ZIP%' -UseBasicParsing"
if errorlevel 1 (
    echo [ERRO] Falha ao baixar CUB
    echo [INFO] Baixe manualmente de: %CUB_URL%
    echo [INFO] Extraia para: %CUB_PATH%
    pause
    exit /b 1
)

echo [OK] Download concluido
echo.
echo [INFO] Extraindo CUB...
echo [AVISO] Isso pode demorar alguns minutos...

REM Extrair usando PowerShell
powershell -Command "$tempExtract = '%TEMP%\cub_extract'; if (Test-Path $tempExtract) { Remove-Item -Recurse -Force $tempExtract }; Expand-Archive -Path '%CUB_ZIP%' -DestinationPath $tempExtract -Force; $extractedFolder = Get-ChildItem $tempExtract | Select-Object -First 1; if (Test-Path '%CUB_PATH%') { Remove-Item -Recurse -Force '%CUB_PATH%' }; Move-Item $extractedFolder.FullName '%CUB_PATH%'; Remove-Item '%CUB_ZIP%' -ErrorAction SilentlyContinue; Remove-Item $tempExtract -ErrorAction SilentlyContinue"

if errorlevel 1 (
    echo [ERRO] Falha ao extrair CUB
    pause
    exit /b 1
)

REM Verificar instalacao
if exist "%CUB_PATH%\CMakeLists.txt" (
    set "CUB_HOME=%CUB_PATH%"
    echo.
    echo [OK] CUB instalado e configurado!
    echo   CUB_HOME: %CUB_PATH%
    echo.
    echo [INFO] Para usar o CUB, execute:
    echo   set "CUB_HOME=%CUB_PATH%"
) else (
    echo.
    echo [ERRO] CUB nao foi instalado corretamente
    pause
    exit /b 1
)

pause





