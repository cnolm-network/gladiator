param(
    [int]$Port = 8080
)

# 启动 Gladiator Dashboard（仅回环地址）。
# 用法：.\dashboard\run.ps1 [-Port 8080]

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root "training\.venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

$env:DASHBOARD_PORT = "$Port"
Write-Host "启动 Gladiator Dashboard: http://127.0.0.1:$Port/"
& $python (Join-Path $PSScriptRoot "app.py")
