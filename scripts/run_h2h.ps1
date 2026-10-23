# MOSS-Image vs Qwen-Image 對打執行器（背景執行用）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\run_h2h.ps1 [-Quick] [-Only photo]
param(
  [switch]$Quick,
  [string]$Only = '',
  [int]$Res = 0,
  [string]$Sizes = '',
  [string]$Out = 'output\h2h'
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$py = 'D:\AI\ComfyUI\python_embeded_nightly\python.exe'
if ($env:MOSS_PY) { $py = $env:MOSS_PY }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$args = @("$root\scripts\head_to_head.py", '--out', $Out)
if ($Quick) { $args += '--quick' }
if ($Only)  { $args += @('--only', $Only) }
if ($Res -gt 0) { $args += @('--res', "$Res") }
if ($Sizes) { $args += @('--sizes', $Sizes) }

Set-Location $root
& $py @args 2>&1 | Tee-Object -FilePath "$root\$Out.log"
"DONE" | Out-File "$root\$Out.done" -Encoding UTF8
