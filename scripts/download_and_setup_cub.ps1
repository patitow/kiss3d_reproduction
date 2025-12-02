# Script para baixar e configurar NVIDIA CUB

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "BAIXANDO E CONFIGURANDO NVIDIA CUB" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$cubPath = "C:\cub"
$cubZip = "$env:TEMP\cub.zip"
$cubUrl = "https://github.com/NVIDIA/cub/archive/refs/tags/2.1.0.zip"

# Verificar se já existe
if (Test-Path $cubPath) {
    if (Test-Path (Join-Path $cubPath "CMakeLists.txt")) {
        Write-Host "`n[OK] CUB ja esta instalado em: $cubPath" -ForegroundColor Green
        $env:CUB_HOME = $cubPath
        Write-Host "[OK] CUB_HOME configurado: $cubPath" -ForegroundColor Green
        exit 0
    }
}

Write-Host "`n[INFO] Baixando CUB 2.1.0..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $cubUrl -OutFile $cubZip -UseBasicParsing
    Write-Host "[OK] Download concluido" -ForegroundColor Green
} catch {
    Write-Host "[ERRO] Falha ao baixar CUB: $_" -ForegroundColor Red
    Write-Host "[INFO] Baixe manualmente de: $cubUrl" -ForegroundColor Yellow
    Write-Host "[INFO] Extraia para: $cubPath" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[INFO] Extraindo CUB..." -ForegroundColor Yellow
try {
    # Criar diretório temporário
    $tempExtract = "$env:TEMP\cub_extract"
    if (Test-Path $tempExtract) {
        Remove-Item -Recurse -Force $tempExtract
    }
    Expand-Archive -Path $cubZip -DestinationPath $tempExtract -Force
    
    # Mover para C:\cub
    $extractedFolder = Get-ChildItem $tempExtract | Select-Object -First 1
    if (Test-Path $cubPath) {
        Remove-Item -Recurse -Force $cubPath
    }
    Move-Item $extractedFolder.FullName $cubPath
    
    # Limpar
    Remove-Item $cubZip -ErrorAction SilentlyContinue
    Remove-Item $tempExtract -ErrorAction SilentlyContinue
    
    Write-Host "[OK] CUB extraido para: $cubPath" -ForegroundColor Green
} catch {
    Write-Host "[ERRO] Falha ao extrair: $_" -ForegroundColor Red
    exit 1
}

# Verificar instalação
if (Test-Path (Join-Path $cubPath "CMakeLists.txt")) {
    $env:CUB_HOME = $cubPath
    Write-Host "`n[OK] CUB instalado e configurado!" -ForegroundColor Green
    Write-Host "  CUB_HOME: $cubPath" -ForegroundColor White
} else {
    Write-Host "`n[ERRO] CUB nao foi instalado corretamente" -ForegroundColor Red
    exit 1
}





