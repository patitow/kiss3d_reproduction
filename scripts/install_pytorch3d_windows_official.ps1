# Script para instalar PyTorch3D seguindo o método oficial do repositório para Windows

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "INSTALACAO PyTorch3D - METODO OFICIAL WINDOWS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot "mesh3d-generator-py3.11\Scripts\python.exe"
$pytorch3dDir = Join-Path $projectRoot "temp_pytorch3d"

# Verificar se está no x64 Native Tools Command Prompt
Write-Host "`n[INFO] Verificando ambiente..." -ForegroundColor Yellow
$vsPath = $env:VSINSTALLDIR
if (-not $vsPath) {
    Write-Host "[AVISO] Nao parece estar no 'x64 Native Tools Command Prompt'" -ForegroundColor Yellow
    Write-Host "[INFO] Abra o 'x64 Native Tools Command Prompt for VS 2022' e execute:" -ForegroundColor Yellow
    Write-Host "       cd $projectRoot" -ForegroundColor White
    Write-Host "       .\scripts\install_pytorch3d_windows_official.ps1" -ForegroundColor White
    Write-Host "`n[INFO] Tentando configurar automaticamente..." -ForegroundColor Yellow
    
    # Tentar configurar automaticamente
    $vs2022Path = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    if (Test-Path $vs2022Path) {
        $vcvars = Join-Path $vs2022Path "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) {
            Write-Host "[INFO] Executando vcvars64.bat..." -ForegroundColor Yellow
            cmd /c "`"$vcvars`" && set" | ForEach-Object {
                if ($_ -match "^(.+?)=(.*)$") {
                    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
                }
            }
            Write-Host "[OK] Ambiente Visual Studio configurado" -ForegroundColor Green
        }
    }
}

# Verificar PyTorch
Write-Host "`n[1/4] Verificando PyTorch..." -ForegroundColor Yellow
$torchVersion = & $pythonExe -c "import torch; print(torch.__version__)" 2>&1
$cudaVersion = & $pythonExe -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1
Write-Host "  PyTorch: $torchVersion" -ForegroundColor White
Write-Host "  CUDA: $cudaVersion" -ForegroundColor White

if ($cudaVersion -eq "N/A") {
    Write-Host "[ERRO] CUDA nao disponivel no PyTorch!" -ForegroundColor Red
    exit 1
}

# Verificar se CUDA >= 11.7 (não precisa de CUB)
$cudaMajorMinor = [version]::Parse($cudaVersion)
if ($cudaMajorMinor -lt [version]"11.7") {
    Write-Host "[INFO] CUDA < 11.7 detectado, precisa de CUB" -ForegroundColor Yellow
    if (-not $env:CUB_HOME) {
        Write-Host "[INFO] Configurando CUB..." -ForegroundColor Yellow
        & (Join-Path $projectRoot "scripts\download_and_setup_cub.ps1")
    }
} else {
    Write-Host "[OK] CUDA >= 11.7, nao precisa de CUB" -ForegroundColor Green
}

# Clonar repositório se necessário
Write-Host "`n[2/4] Preparando codigo-fonte..." -ForegroundColor Yellow
if (-not (Test-Path $pytorch3dDir)) {
    Write-Host "[INFO] Clonando repositorio PyTorch3D..." -ForegroundColor Yellow
    Set-Location $projectRoot
    git clone https://github.com/facebookresearch/pytorch3d.git temp_pytorch3d
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERRO] Falha ao clonar repositorio" -ForegroundColor Red
        exit 1
    }
}

Set-Location $pytorch3dDir
git checkout stable
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERRO] Falha ao fazer checkout da branch stable" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Codigo-fonte pronto" -ForegroundColor Green

# Instalar iopath se necessário
Write-Host "`n[3/4] Instalando dependencias..." -ForegroundColor Yellow
& $pythonExe -m pip install iopath
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERRO] Falha ao instalar iopath" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Dependencias instaladas" -ForegroundColor Green

# Compilar e instalar
Write-Host "`n[4/4] Compilando e instalando PyTorch3D..." -ForegroundColor Yellow
Write-Host "[AVISO] Isso pode demorar 15-30 minutos..." -ForegroundColor Yellow
Write-Host "[INFO] Aguarde..." -ForegroundColor White

# Configurar FORCE_CUDA se necessário
if ($cudaVersion -ne "N/A") {
    $env:FORCE_CUDA = "1"
}

# Instalar usando setup.py (método oficial para Windows)
& $pythonExe setup.py install

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] PyTorch3D compilado e instalado com sucesso!" -ForegroundColor Green
    
    # Verificar
    Write-Host "`n[INFO] Verificando instalacao..." -ForegroundColor Yellow
    & $pythonExe -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"
    & $pythonExe -c "from pytorch3d import _C; print('PyTorch3D _C OK!')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] PyTorch3D esta funcionando com CUDA!" -ForegroundColor Green
    } else {
        Write-Host "`n[AVISO] PyTorch3D instalado mas _C nao pode ser importado" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[ERRO] Falha na compilacao" -ForegroundColor Red
    Write-Host "[INFO] Verifique os erros acima" -ForegroundColor Yellow
    Write-Host "[INFO] Certifique-se de estar no 'x64 Native Tools Command Prompt'" -ForegroundColor Yellow
    exit 1
}

Set-Location $projectRoot











