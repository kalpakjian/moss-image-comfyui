# LoRA 實測對比：Base vs Base+LoRA（KohakuMossTrainer 風格）
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
$log = "$d\lora_compare.log"

function Log($m) { Write-Host $m; $m | Out-File $log -Encoding UTF8 -Append }
Remove-Item $log -Force -ErrorAction SilentlyContinue

# 用「沒在訓練資料裡」的提示詞，才能看出 LoRA 是否真的改變了風格
$TEST_PROMPTS = @(
  'A rustic attic workshop with a battered wooden workbench, hand tools hanging on the wall, dust motes in a shaft of afternoon light, warm amber tones, photorealistic, high detail',
  'A small atelier corner with an old sewing table, bolts of fabric, a single warm lamp, deep shadows, muted green and cream palette, photorealistic, high detail'
)

$i = 0
foreach ($p in $TEST_PROMPTS) {
  foreach ($mode in @('base', 'lora')) {
    $tag = "lora_${mode}_$i"
    $wf = "$d\wf_$tag.json"
    if ($mode -eq 'base') {
      & $py "$scripts\moss_workflow.py" --out $wf --model base --width 1024 --height 1024 --seed 777 --prompt $p 2>&1 | Out-Null
    } else {
      & $py "$scripts\moss_workflow.py" --out $wf --model base --width 1024 --height 1024 --seed 777 --prompt $p --lora 'moss_studio_lora-lycoris.safetensors' --lora-strength 1.0 2>&1 | Out-Null
    }
    Log "=== $tag ==="
    & $py "$scripts\run_moss.py" --wf $wf --tag $tag --timeout 900 2>&1 | ForEach-Object { Log "  $_" }
  }
  $i++
}
Log 'LORA COMPARE DONE'