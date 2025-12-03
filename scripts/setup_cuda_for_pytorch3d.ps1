# Script para configurar CUDA e Visual Studio para compilar PyTorch3D

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CONFIGURANDO CUDA E VISUAL STUDIO PARA PyTorch3D" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Verificar CUDA instalado
$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaVersions = Get-ChildItem $cudaBase -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "^v\d+\.\d+" }

Write-Host "`n[INFO] Versoes CUDA encontradas:" -ForegroundColor Yellow
foreach ($version in $cudaVersions) {
    Write-Host "  - $($version.Name)" -ForegroundColor White
}

# Verificar qual CUDA o PyTorch precisa
Write-Host "`n[INFO] Verificando versao CUDA do PyTorch..." -ForegroundColor Yellow
$pythonExe = ".\mesh3d-generator-py3.11\Scripts\python.exe"
$torchCuda = & $pythonExe -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1
Write-Host "  PyTorch foi compilado com CUDA: $torchCuda" -ForegroundColor White

# Procurar CUDA 12.1
$cuda121 = $cudaVersions | Where-Object { $_.Name -eq "v12.1" -or $_.Name -eq "v12.2" }
if ($cuda121) {
    Write-Host "`n[OK] CUDA 12.1 encontrado: $($cuda121.FullName)" -ForegroundColor Green
    $cudaPath = $cuda121.FullName
    $cudaBin = Join-Path $cudaPath "bin"
    
    # Adicionar ao PATH se não estiver
    $currentPath = $env:PATH
    if ($currentPath -notlike "*$cudaBin*") {
        $env:PATH = "$cudaBin;$env:PATH"
        Write-Host "[OK] CUDA bin adicionado ao PATH: $cudaBin" -ForegroundColor Green
    }
    
    # Configurar CUDA_HOME
    $env:CUDA_HOME = $cudaPath
    $env:CUDA_PATH = $cudaPath
    Write-Host "[OK] CUDA_HOME configurado: $cudaPath" -ForegroundColor Green
} else {
    Write-Host "`n[AVISO] CUDA 12.1 nao encontrado!" -ForegroundColor Yellow
    Write-Host "[INFO] PyTorch precisa de CUDA 12.1, mas sistema tem CUDA 11.8" -ForegroundColor Yellow
    Write-Host "[INFO] Opcoes:" -ForegroundColor Yellow
    Write-Host "  1. Instalar CUDA 12.1" -ForegroundColor White
    Write-Host "  2. Reinstalar PyTorch com CUDA 11.8" -ForegroundColor White
}

# Verificar Visual Studio Build Tools
Write-Host "`n[INFO] Procurando Visual Studio Build Tools..." -ForegroundColor Yellow
$vsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
)

$vsFound = $false
foreach ($vsPath in $vsPaths) {
    if (Test-Path $vsPath) {
        Write-Host "[OK] Visual Studio encontrado: $vsPath" -ForegroundColor Green
        
        # Procurar MSVC
        $vcTools = Get-ChildItem "$vsPath\VC\Tools\MSVC" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($vcTools) {
            $clPath = Join-Path $vcTools.FullName "bin\Hostx64\x64"
            if (Test-Path (Join-Path $clPath "cl.exe")) {
                Write-Host "[OK] MSVC encontrado: $($vcTools.Name)" -ForegroundColor Green
                
                # Adicionar ao PATH
                $currentPath = $env:PATH
                if ($currentPath -notlike "*$clPath*") {
                    $env:PATH = "$clPath;$env:PATH"
                    Write-Host "[OK] MSVC adicionado ao PATH: $clPath" -ForegroundColor Green
                    $vsFound = $true
                } else {
                    Write-Host "[OK] MSVC ja esta no PATH" -ForegroundColor Green
                    $vsFound = $true
                }
            }
        }
        break
    }
}

if (-not $vsFound) {
    Write-Host "`n[ERRO] Visual Studio Build Tools nao encontrado!" -ForegroundColor Red
    Write-Host "[INFO] Instale Visual Studio Build Tools:" -ForegroundColor Yellow
    Write-Host "  https://visualstudio.microsoft.com/downloads/" -ForegroundColor White
    Write-Host "  Selecione: 'Desktop development with C++'" -ForegroundColor White
}

# Verificar CUB (necessário para PyTorch3D)
Write-Host "`n[INFO] Verificando NVIDIA CUB..." -ForegroundColor Yellow
if ($env:CUB_HOME) {
    Write-Host "[OK] CUB_HOME configurado: $env:CUB_HOME" -ForegroundColor Green
} else {
    Write-Host "[AVISO] CUB_HOME nao configurado" -ForegroundColor Yellow
    Write-Host "[INFO] Baixe de: https://github.com/NVIDIA/cub/releases" -ForegroundColor White
    Write-Host "[INFO] Extraia e configure: `$env:CUB_HOME = 'C:\cub'" -ForegroundColor White
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "RESUMO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CUDA_HOME: $env:CUDA_HOME" -ForegroundColor White
Write-Host "CUDA_PATH: $env:CUDA_PATH" -ForegroundColor White
Write-Host "Visual Studio: $(if ($vsFound) { 'OK' } else { 'NAO ENCONTRADO' })" -ForegroundColor $(if ($vsFound) { 'Green' } else { 'Red' })
Write-Host "CUB_HOME: $(if ($env:CUB_HOME) { $env:CUB_HOME } else { 'NAO CONFIGURADO' })" -ForegroundColor $(if ($env:CUB_HOME) { 'Green' } else { 'Yellow' })

Write-Host "`n[INFO] Para aplicar estas configuracoes permanentemente:" -ForegroundColor Yellow
Write-Host "  Execute este script antes de compilar PyTorch3D" -ForegroundColor White
Write-Host "  Ou configure as variaveis de ambiente do sistema" -ForegroundColor White











