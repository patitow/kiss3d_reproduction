# Script PowerShell para corrigir PyTorch3D e baixar modelos

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CORRIGINDO PyTorch3D E BAIXANDO MODELOS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "[1/2] Corrigindo PyTorch3D..." -ForegroundColor Yellow
python scripts\fix_pytorch3d_cuda.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERRO] Falha ao corrigir PyTorch3D" -ForegroundColor Red
    Read-Host "Pressione Enter para continuar"
    exit 1
}

Write-Host ""
Write-Host "[2/2] Baixando modelos..." -ForegroundColor Yellow
python scripts\download_all_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[AVISO] Alguns modelos podem não ter sido baixados" -ForegroundColor Yellow
    Read-Host "Pressione Enter para continuar"
    exit 1
}

Write-Host ""
Write-Host "[OK] Processo concluído!" -ForegroundColor Green
Read-Host "Pressione Enter para sair"











