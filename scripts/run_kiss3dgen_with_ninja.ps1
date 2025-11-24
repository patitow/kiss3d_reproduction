# Script wrapper para executar Kiss3DGen com ninja no PATH

# Adicionar ninja ao PATH
$ninjaPath = Join-Path $PSScriptRoot "..\mesh3d-generator-py3.11\ninja\data\bin"
if (Test-Path $ninjaPath) {
    $env:PATH = "$ninjaPath;$env:PATH"
    Write-Host "[INFO] Ninja adicionado ao PATH: $ninjaPath" -ForegroundColor Green
} else {
    Write-Host "[AVISO] Caminho do ninja nao encontrado: $ninjaPath" -ForegroundColor Yellow
}

# Executar script Python
$pythonExe = Join-Path $PSScriptRoot "..\mesh3d-generator-py3.11\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "run_kiss3dgen_image_to_3d.py"

# Passar argumentos
$argsString = $args -join " "
& $pythonExe $scriptPath $argsString

