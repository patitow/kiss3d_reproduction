# PowerShell script para pré-compilar nvdiffrast
# Uso: .\scripts\precompile_nvdiffrast.ps1

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot "mesh3d-generator-py3.11\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERRO] Python não encontrado em: $pythonExe" -ForegroundColor Red
    exit 1
}

$precompileScript = Join-Path $projectRoot "scripts\precompile_nvdiffrast.py"

if (-not (Test-Path $precompileScript)) {
    Write-Host "[ERRO] Script de pré-compilação não encontrado: $precompileScript" -ForegroundColor Red
    exit 1
}

Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "PRÉ-COMPILAÇÃO DO NVDIFFRAST" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

& $pythonExe $precompileScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Pré-compilação concluída com sucesso!" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "[ERRO] Pré-compilação falhou (código: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

