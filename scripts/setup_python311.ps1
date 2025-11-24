# Script para configurar ambiente Python 3.11.9 no Windows (PowerShell)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CONFIGURACAO DO AMBIENTE PYTHON 3.11.9" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se Python 3.11 está instalado
try {
    $python311 = Get-Command python3.11 -ErrorAction Stop
    Write-Host "[OK] Python 3.11 encontrado" -ForegroundColor Green
    python3.11 --version
} catch {
    Write-Host "[ERRO] Python 3.11 nao encontrado!" -ForegroundColor Red
    Write-Host "[INFO] Instale Python 3.11.9: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    Write-Host "[INFO] Certifique-se de marcar 'Add Python to PATH' durante a instalacao" -ForegroundColor Yellow
    Read-Host "Pressione Enter para sair"
    exit 1
}

# Criar ambiente virtual
Write-Host ""
Write-Host "[1/3] Criando ambiente virtual..." -ForegroundColor Cyan
if (Test-Path "mesh3d-generator-py3.11") {
    Write-Host "[AVISO] Ambiente virtual ja existe" -ForegroundColor Yellow
    $recreate = Read-Host "Deseja recriar? (s/N)"
    if ($recreate -eq "s" -or $recreate -eq "S") {
        Remove-Item -Recurse -Force "mesh3d-generator-py3.11"
        python3.11 -m venv mesh3d-generator-py3.11
    }
} else {
    python3.11 -m venv mesh3d-generator-py3.11
}

if (-not (Test-Path "mesh3d-generator-py3.11")) {
    Write-Host "[ERRO] Falha ao criar ambiente virtual" -ForegroundColor Red
    Read-Host "Pressione Enter para sair"
    exit 1
}

Write-Host "[OK] Ambiente virtual criado" -ForegroundColor Green

# Ativar ambiente virtual
Write-Host ""
Write-Host "[2/3] Ativando ambiente virtual..." -ForegroundColor Cyan
& "mesh3d-generator-py3.11\Scripts\Activate.ps1"

# Verificar versão
Write-Host ""
Write-Host "[3/3] Verificando versao do Python no ambiente virtual..." -ForegroundColor Cyan
python --version

# Atualizar pip
Write-Host ""
Write-Host "[INFO] Atualizando pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[OK] Ambiente Python 3.11.9 configurado!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Proximos passos:" -ForegroundColor Yellow
Write-Host "1. Instalar PyTorch com CUDA:"
Write-Host "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
Write-Host ""
Write-Host "2. Instalar dependencias:"
Write-Host "   pip install -r requirements.txt"
Write-Host "   OU"
Write-Host "   python scripts/install_dependencies.py"
Write-Host ""
Write-Host "3. Autenticar HuggingFace:"
Write-Host "   huggingface-cli login"
Write-Host ""
Write-Host "4. Baixar modelos:"
Write-Host "   python scripts/download_models.py"
Write-Host ""

Read-Host "Pressione Enter para continuar"

