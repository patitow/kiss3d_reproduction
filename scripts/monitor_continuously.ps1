# Script de monitoramento contínuo
$projectRoot = "D:\.Faculdade\Visao_Computacional\2025_2"
$outputDir = Join-Path $projectRoot "outputs\flux_top10_test"
$historyFile = Join-Path $outputDir "runs_report.json"
$logDir = Join-Path $outputDir "logs"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MONITORAMENTO CONTÍNUO DO TESTE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$iteration = 0
$lastSuccessCount = -1
$consecutiveNoProgress = 0

while ($true) {
    $iteration++
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    Write-Host "[$timestamp] Iteração $iteration - Verificando progresso..." -ForegroundColor Yellow
    
    # Verificar histórico
    $successCount = 0
    $failedCount = 0
    $totalCount = 0
    $errors = @()
    
    if (Test-Path $historyFile) {
        try {
            $history = Get-Content $historyFile | ConvertFrom-Json
            $totalCount = $history.Count
            $successCount = ($history | Where-Object {$_.success -eq $true}).Count
            $failedCount = ($history | Where-Object {$_.success -eq $false}).Count
            
            # Coletar erros
            $errors = $history | Where-Object {$_.success -eq $false -and $_.error} | ForEach-Object {
                "$($_.label): $($_.error)"
            }
        } catch {
            Write-Host "  Erro ao ler histórico: $_" -ForegroundColor Red
        }
    }
    
    # Verificar arquivos gerados
    $objFiles = (Get-ChildItem -Path $outputDir -Recurse -Filter "*.obj" -ErrorAction SilentlyContinue).Count
    $glbFiles = (Get-ChildItem -Path $outputDir -Recurse -Filter "*.glb" -ErrorAction SilentlyContinue).Count
    $bundleFiles = (Get-ChildItem -Path $outputDir -Recurse -Filter "*_3d_bundle.png" -ErrorAction SilentlyContinue).Count
    
    # Mostrar progresso
    Write-Host "  Progresso: $successCount/$totalCount sucessos, $failedCount falhas" -ForegroundColor $(if ($successCount -ge 10) { "Green" } else { "Yellow" })
    Write-Host "  Arquivos: $objFiles OBJ, $glbFiles GLB, $bundleFiles Bundles" -ForegroundColor Cyan
    
    # Verificar se houve progresso
    if ($successCount -eq $lastSuccessCount) {
        $consecutiveNoProgress++
    } else {
        $consecutiveNoProgress = 0
        $lastSuccessCount = $successCount
    }
    
    # Verificar se completou
    if ($successCount -ge 10) {
        Write-Host "`n==========================================" -ForegroundColor Green
        Write-Host "SUCESSO! Todos os 10 objetos foram processados!" -ForegroundColor Green
        Write-Host "==========================================" -ForegroundColor Green
        break
    }
    
    # Mostrar erros se houver
    if ($errors.Count -gt 0) {
        Write-Host "`n  ERROS ENCONTRADOS:" -ForegroundColor Red
        $errors | Select-Object -Last 3 | ForEach-Object {
            Write-Host "    - $_" -ForegroundColor Yellow
        }
    }
    
    # Verificar último log para erros críticos
    $latestLog = Get-ChildItem -Path $logDir -Filter "*.log" -ErrorAction SilentlyContinue | 
                 Sort-Object LastWriteTime -Descending | 
                 Select-Object -First 1
    
    if ($latestLog) {
        $logContent = Get-Content $latestLog.FullName -Tail 5 -ErrorAction SilentlyContinue
        $criticalErrors = $logContent | Where-Object { 
            $_ -match "CUDA error|device-side assert|RuntimeError|Exception|Traceback" 
        }
        
        if ($criticalErrors) {
            Write-Host "`n  ERROS CRÍTICOS NO LOG:" -ForegroundColor Red
            $criticalErrors | ForEach-Object {
                Write-Host "    $_" -ForegroundColor Red
            }
        }
    }
    
    # Verificar se processo ainda está rodando
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | 
                      Where-Object {$_.Path -like "*mesh3d-generator-py3.11*"}
    
    if ($pythonProcesses.Count -eq 0) {
        Write-Host "`n[AVISO] Nenhum processo Python encontrado!" -ForegroundColor Red
        Write-Host "  Verificando se teste completou ou falhou..." -ForegroundColor Yellow
        
        # Verificar resultado final
        if ($successCount -ge 10) {
            Write-Host "  Teste completou com sucesso!" -ForegroundColor Green
        } else {
            Write-Host "  Teste pode ter falhado ou terminado prematuramente." -ForegroundColor Red
            Write-Host "  Último log:" -ForegroundColor Yellow
            if ($latestLog) {
                Get-Content $latestLog.FullName -Tail 20
            }
        }
        break
    } else {
        Write-Host "  Processos Python ativos: $($pythonProcesses.Count)" -ForegroundColor Green
    }
    
    # Aviso se sem progresso há muito tempo
    if ($consecutiveNoProgress -gt 12) {
        Write-Host "`n[AVISO] Sem progresso há $($consecutiveNoProgress * 30) segundos!" -ForegroundColor Yellow
        Write-Host "  Verificando último log..." -ForegroundColor Yellow
        if ($latestLog) {
            Write-Host "  Últimas 10 linhas:" -ForegroundColor Cyan
            Get-Content $latestLog.FullName -Tail 10 | ForEach-Object {
                Write-Host "    $_"
            }
        }
        $consecutiveNoProgress = 0  # Resetar contador
    }
    
    Write-Host ""
    
    # Aguardar antes da próxima verificação
    Start-Sleep -Seconds 30
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "MONITORAMENTO FINALIZADO" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

