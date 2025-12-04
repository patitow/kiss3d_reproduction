# Script PowerShell para rodar os objetos um de cada vez
# Execute este script para processar todos os objetos na ordem

$objects = @(
    @{
        Name = "Gigabyte Motherboard"
        Input = "data\raw\gazebo_dataset\images\Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3_0.jpg"
        Output = "outputs\gigabyte_motherboard"
    },
    @{
        Name = "Kong Puppy"
        Input = "data\raw\gazebo_dataset\images\Kong_Puppy_Teething_Rubber_Small_Pink_0.jpg"
        Output = "outputs\kong_puppy"
    },
    @{
        Name = "ABC Blocks Wagon"
        Input = "data\raw\gazebo_dataset\images\Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI_0.jpg"
        Output = "outputs\abc_blocks_wagon"
    },
    @{
        Name = "School Bus"
        Input = "data\raw\gazebo_dataset\images\Sonny_School_Bus_0.jpg"
        Output = "outputs\school_bus"
    }
)

$pythonExe = ".\mesh3d-generator-py3.11\Scripts\python.exe"
$scriptPath = "scripts\run_kiss3dgen_image_to_3d.py"
$configPath = "pipeline_config\default.yaml"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "PROCESSAMENTO DE OBJETOS - Pipeline Flux" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Total de objetos: $($objects.Count)" -ForegroundColor Yellow
Write-Host ""

foreach ($obj in $objects) {
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host "Processando: $($obj.Name)" -ForegroundColor Green
    Write-Host "Input: $($obj.Input)" -ForegroundColor Gray
    Write-Host "Output: $($obj.Output)" -ForegroundColor Gray
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host ""
    
    $command = "& `"$pythonExe`" `"$scriptPath`" --input `"$($obj.Input)`" --output `"$($obj.Output)`" --config `"$configPath`" --pipeline-mode flux"
    
    Write-Host "Executando comando..." -ForegroundColor Yellow
    Invoke-Expression $command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERRO] Falha ao processar $($obj.Name)" -ForegroundColor Red
        Write-Host "Deseja continuar com o próximo objeto? (S/N)" -ForegroundColor Yellow
        $continue = Read-Host
        if ($continue -ne "S" -and $continue -ne "s") {
            Write-Host "Processamento interrompido pelo usuário." -ForegroundColor Yellow
            break
        }
    } else {
        Write-Host "`n[OK] $($obj.Name) processado com sucesso!" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Aguardando 5 segundos antes do próximo objeto..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    Write-Host ""
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "PROCESSAMENTO CONCLUÍDO!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

