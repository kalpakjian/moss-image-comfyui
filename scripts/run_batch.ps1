# MOSS-Image 實測批次：驗證 4MP 外推 + Base vs SFT 對比
# 環境變數：
#   MOSS_COMFY   ComfyUI 根目錄（預設 D:\AI\ComfyUI）
#   MOSS_PY      ComfyUI 的 python（預設 <MOSS_COMFY>\python_embeded_nightly\python.exe）
#   MOSS_WORK    腳本與工作流程資料夾（預設本 repo 根目錄）
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$Comfy = if ($env:MOSS_COMFY) { $env:MOSS_COMFY } else { 'D:\AI\ComfyUI' }
$py = if ($env:MOSS_PY) { $env:MOSS_PY } else { "$Comfy\python_embeded_nightly\python.exe" }
$d = if ($env:MOSS_WORK) { $env:MOSS_WORK } else { $root }
$scripts = Join-Path $d 'scripts'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$log = "$d\run_batch.log"

function Log($m) { Write-Host $m; $m | Out-File $log -Encoding UTF8 -Append }
Remove-Item $log -Force -ErrorAction SilentlyContinue

$PROMPT = 'A cozy sunlit studio corner with a wooden desk, a small potted plant, warm afternoon light through a window, soft shadows, detailed textures, photorealistic, high detail'

# 測試矩陣：驗證官方外推上限 4M 像素
$tests = @(
  @{ tag = 'sft_1024';  model = 'sft';  w = 1024; h = 1024; seed = 42 },
  @{ tag = 'sft_2048';  model = 'sft';  w = 2048; h = 2048; seed = 42 },
  @{ tag = 'sft_2560';  model = 'sft';  w = 2560; h = 2560; seed = 42 },
  @{ tag = 'base_1024'; model = 'base'; w = 1024; h = 1024; seed = 42 }
)

foreach ($t in $tests) {
  $wf = "$d\wf_$($t.tag).json"
  & $py "$scripts\moss_workflow.py" --out $wf --model $t.model --width $t.w --height $t.h --seed $t.seed --prompt $PROMPT 2>&1 | Out-Null
  $mp = [math]::Round($t.w * $t.h / 1000000, 2)
  Log "=== $($t.tag)  $($t.w)x$($t.h) = $mp MP ==="
  & $py "$scripts\run_moss.py" --wf $wf --tag $t.tag --timeout 1800 2>&1 |
    ForEach-Object { Log "  $_" }
}

Log 'BATCH DONE'
