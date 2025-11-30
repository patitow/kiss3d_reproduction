@echo off
REM Script para corrigir PyTorch3D e baixar modelos

echo ============================================================
echo CORRIGINDO PyTorch3D E BAIXANDO MODELOS
echo ============================================================

echo.
echo [1/2] Corrigindo PyTorch3D...
python scripts\fix_pytorch3d_cuda.py
if errorlevel 1 (
    echo [ERRO] Falha ao corrigir PyTorch3D
    pause
    exit /b 1
)

echo.
echo [2/2] Baixando modelos...
python scripts\download_all_models.py
if errorlevel 1 (
    echo [AVISO] Alguns modelos podem não ter sido baixados
    pause
    exit /b 1
)

echo.
echo [OK] Processo concluído!
pause



