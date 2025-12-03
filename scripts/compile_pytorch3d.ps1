# Script completo para compilar PyTorch3D com CUDA

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "COMPILANDO PyTorch3D COM CUDA" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Configurar CUDA e Visual Studio
Write-Host "`n[1/3] Configurando CUDA e Visual Studio..." -ForegroundColor Yellow
& ".\scripts\setup_cuda_for_pytorch3d.ps1"

# 2. Configurar CUB se necessário
Write-Host "`n[2/3] Verificando CUB..." -ForegroundColor Yellow
if (-not $env:CUB_HOME -or -not (Test-Path $env:CUB_HOME)) {
    Write-Host "[INFO] CUB nao configurado, baixando..." -ForegroundColor Yellow
    & ".\scripts\download_and_setup_cub.ps1"
} else {
    Write-Host "[OK] CUB ja configurado: $env:CUB_HOME" -ForegroundColor Green
}

# 3. Compilar PyTorch3D
Write-Host "`n[3/3] Compilando PyTorch3D..." -ForegroundColor Yellow
Write-Host "[AVISO] Isso pode demorar 15-30 minutos..." -ForegroundColor Yellow
Write-Host "[INFO] Aguarde..." -ForegroundColor White

$pythonExe = ".\mesh3d-generator-py3.11\Scripts\python.exe"

# Garantir que todas as variáveis estão configuradas
$env:CUB_HOME = "C:\cub"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

# Adicionar CUDA e MSVC ao PATH
$cudaBin = "$env:CUDA_HOME\bin"
$msvcBin = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$env:PATH = "$cudaBin;$msvcBin;$env:PATH"

# Compilar
& $pythonExe -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] PyTorch3D compilado com sucesso!" -ForegroundColor Green
    
    # Verificar
    Write-Host "`n[INFO] Verificando instalacao..." -ForegroundColor Yellow
    & $pythonExe -c "import pytorch3d; from pytorch3d import _C; print('PyTorch3D com CUDA OK!')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Tudo funcionando!" -ForegroundColor Green
    }
} else {
    Write-Host "`n[ERRO] Falha na compilacao" -ForegroundColor Red
    Write-Host "[INFO] Verifique os erros acima" -ForegroundColor Yellow
}












