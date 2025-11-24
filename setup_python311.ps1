# Script PowerShell para configurar ambiente Python 3.11 com CUDA

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuracao Python 3.11 com CUDA" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se Python 3.11 esta instalado
Write-Host "Verificando Python 3.11..." -ForegroundColor Cyan
$python311Found = $false
$python311Path = $null

# Tentar python3.11
try {
    $result = python3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Python 3.11 encontrado: $result" -ForegroundColor Green
        $python311Found = $true
        $python311Path = "python3.11"
    }
} catch {
    # Continuar para tentar outros caminhos
}

# Tentar caminhos comuns do Windows
if (-not $python311Found) {
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:PROGRAMFILES\Python311\python.exe",
        "C:\Python311\python.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Write-Host "[OK] Python 3.11 encontrado em: $path" -ForegroundColor Green
            $python311Found = $true
            $python311Path = $path
            break
        }
    }
}

if (-not $python311Found) {
    Write-Host "[ERRO] Python 3.11 nao encontrado!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Por favor, instale Python 3.11 primeiro:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3110/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANTE: Marque 'Add Python to PATH' durante a instalacao!" -ForegroundColor Yellow
    exit 1
}

# Verificar NVIDIA GPU
Write-Host ""
Write-Host "Verificando GPU NVIDIA..." -ForegroundColor Cyan
try {
    $gpu = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    Write-Host "[OK] GPU encontrada: $gpu" -ForegroundColor Green
} catch {
    Write-Host "[AVISO] nvidia-smi nao encontrado. Verifique se os drivers NVIDIA estao instalados." -ForegroundColor Yellow
}

# Criar ambiente virtual
Write-Host ""
Write-Host "[1/5] Criando ambiente virtual..." -ForegroundColor Cyan
if (Test-Path "mesh3d-generator-py3.11") {
    $recreate = Read-Host "Ambiente ja existe. Deseja recriar? (S/N)"
    if ($recreate -eq "S" -or $recreate -eq "s") {
        Remove-Item -Recurse -Force mesh3d-generator-py3.11
        & $python311Path -m venv mesh3d-generator-py3.11
    }
} else {
    & $python311Path -m venv mesh3d-generator-py3.11
}

Write-Host "[OK] Ambiente virtual criado" -ForegroundColor Green

# Ativar ambiente
Write-Host ""
Write-Host "[2/5] Ativando ambiente virtual..." -ForegroundColor Cyan
& .\mesh3d-generator-py3.11\Scripts\Activate.ps1

# Atualizar pip
Write-Host ""
Write-Host "[3/5] Atualizando pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Instalar PyTorch com CUDA
Write-Host ""
Write-Host "[4/5] Instalando PyTorch com CUDA 12.1..." -ForegroundColor Cyan
Write-Host "Isso pode demorar alguns minutos..." -ForegroundColor Yellow
Write-Host "Nota: CUDA 12.1 e compativel com RTX 3060" -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar CUDA
Write-Host ""
Write-Host "Verificando CUDA..." -ForegroundColor Cyan
python -c "import torch; print('CUDA disponivel:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Instalar dependencias
Write-Host ""
Write-Host "[5/5] Instalando dependencias do projeto..." -ForegroundColor Cyan
pip install numpy pillow einops trimesh diffusers transformers accelerate omegaconf rembg

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Instalacao concluida!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para ativar o ambiente no futuro:" -ForegroundColor Yellow
Write-Host "  .\mesh3d-generator-py3.11\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Para instalar Pytorch3D (opcional):" -ForegroundColor Yellow
Write-Host "  pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt240/download.html" -ForegroundColor White
Write-Host ""

