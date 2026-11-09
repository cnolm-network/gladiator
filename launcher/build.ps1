# 重新编译 Dashboard 启动器（Windows 自带 .NET Framework，无需安装额外工具链）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$csc = "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) { $csc = "C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe" }
if (-not (Test-Path $csc)) { throw "找不到 csc.exe（需要 .NET Framework 4.x）" }

& $csc /nologo /target:exe /optimize+ /out:"$PSScriptRoot\GladiatorLauncher.exe" /r:System.Management.dll "$PSScriptRoot\GladiatorLauncher.cs"
if ($LASTEXITCODE -ne 0) { throw "编译失败（exit $LASTEXITCODE）" }
Write-Host "已生成 $PSScriptRoot\GladiatorLauncher.exe"
