# Script para rodar e monitorar o teste dos 10 objetos
$ErrorActionPreference = "Continue"

$projectRoot = "D:\.Faculdade\Visao_Computacional\2025_2"
$pythonExe = Join-Path $projectRoot "mesh3d-generator-py3.11\Scripts\python.exe"
$scriptPath = Join-Path $projectRoot "scripts\run_kiss3dgen_image_to_3d.py"
$outputDir = Join-Path $projectRoot "outputs\flux_top10_test"
$logFile = Join-Path $outputDir "run.log"

# Criar diretório de saída
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Iniciando teste dos 10 objetos" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor Yellow
Write-Host "Script: $scriptPath" -ForegroundColor Yellow
Write-Host "Output: $outputDir" -ForegroundColor Yellow
Write-Host "Log: $logFile" -ForegroundColor Yellow
Write-Host ""

# Rodar o script e capturar saída
$process = Start-Process -FilePath $pythonExe -ArgumentList @(
    $scriptPath,
    "--dataset-plan", "pipeline_config\flux_top10_dataset.yaml",
    "--output", "outputs\flux_top10_test",
    "--config", "pipeline_config\default.yaml"
) -WorkingDirectory $projectRoot -NoNewWindow -PassThru -RedirectStandardOutput "$logFile.stdout" -RedirectStandardError "$logFile.stderr"

Write-Host "Processo iniciado com PID: $($process.Id)" -ForegroundColor Green
Write-Host "Monitorando processo..." -ForegroundColor Yellow
Write-Host ""

# Monitorar processo
$maxWaitTime = 3600  # 1 hora máximo
$startTime = Get-Date
$lastLogSize = 0
$noProgressCount = 0

while ($true) {
    Start-Sleep -Seconds 10
    
    # Verificar se processo ainda está rodando
    if ($process.HasExited) {
        Write-Host "`nProcesso finalizado com código: $($process.ExitCode)" -ForegroundColor $(if ($process.ExitCode -eq 0) { "Green" } else { "Red" })
        break
    }
    
    # Verificar tempo máximo
    $elapsed = (Get-Date) - $startTime
    if ($elapsed.TotalSeconds -gt $maxWaitTime) {
        Write-Host "`nTempo máximo excedido. Encerrando processo..." -ForegroundColor Red
        Stop-Process -Id $process.Id -Force
        break
    }
    
    # Verificar logs
    if (Test-Path "$logFile.stdout") {
        $currentLogSize = (Get-Item "$logFile.stdout").Length
        if ($currentLogSize -gt $lastLogSize) {
            $newContent = Get-Content "$logFile.stdout" -Tail 5 -ErrorAction SilentlyContinue
            if ($newContent) {
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Últimas linhas:" -ForegroundColor Cyan
                $newContent | ForEach-Object { Write-Host "  $_" }
                $lastLogSize = $currentLogSize
                $noProgressCount = 0
            }
        } else {
            $noProgressCount++
            if ($noProgressCount -gt 6) {
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Sem progresso há $($noProgressCount * 10) segundos..." -ForegroundColor Yellow
            }
        }
    }
    
    # Verificar erros
    if (Test-Path "$logFile.stderr") {
        $errors = Get-Content "$logFile.stderr" -Tail 10 -ErrorAction SilentlyContinue | Where-Object { $_ -match "error|Error|ERROR|Exception|Traceback" }
        if ($errors) {
            Write-Host "`nERROS DETECTADOS:" -ForegroundColor Red
            $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        }
    }
    
    # Mostrar progresso de arquivos gerados
    $generatedFiles = Get-ChildItem -Path $outputDir -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @(".obj", ".glb", ".png") }
    $fileCount = ($generatedFiles | Measure-Object).Count
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Arquivos gerados: $fileCount | Tempo decorrido: $([math]::Round($elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
}

# Mostrar resumo final
Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Resumo Final" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if (Test-Path "$logFile.stdout") {
    Write-Host "`nÚltimas 20 linhas do log:" -ForegroundColor Yellow
    Get-Content "$logFile.stdout" -Tail 20
}

if (Test-Path "$logFile.stderr") {
    $errorCount = (Get-Content "$logFile.stderr" | Where-Object { $_ -match "error|Error|ERROR|Exception" }).Count
    if ($errorCount -gt 0) {
        Write-Host "`nERROS encontrados: $errorCount" -ForegroundColor Red
        Get-Content "$logFile.stderr" -Tail 30
    }
}

$finalFiles = Get-ChildItem -Path $outputDir -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @(".obj", ".glb") }
Write-Host "`nTotal de modelos gerados: $(($finalFiles | Measure-Object).Count)" -ForegroundColor $(if (($finalFiles | Measure-Object).Count -ge 10) { "Green" } else { "Yellow" })

