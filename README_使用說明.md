# README 使用說明 — MOSS-Image × ComfyUI

本檔是 [README.md](README.md) 的詳細版：完整操作步驟、節點接法說明、**全部踩坑紀錄**。
主頁面看模型清單與實測數據，這裡看「怎麼做」與「為什麼會卡住」。

## 目錄

1. [前置條件](#一前置條件)
2. [安裝與權重](#二安裝與權重)
3. [工作流程節點詳解](#三工作流程節點詳解)
4. [腳本用法](#四腳本用法)
5. [踩坑紀錄](#五踩坑紀錄)
6. [已知限制](#六已知限制)

---

## 一、前置條件

| 項目 | 需求 | 說明 |
| --- | --- | --- |
| ComfyUI | **v0.42.0 以上** | MOSS-Image 原生支援 2026-10-16 才合併，v0.41.3 尚無 |
| GPU | 24GB VRAM 實測通過 | 12GB 以下請用 bf16 + 低解析度，或改跑訓練用的小 profile |
| Python | 3.11+ | ComfyUI 內建的 `python_embeded_nightly` 即可 |
| 套件 | `huggingface_hub`、`Pillow` | 只有下載與對比圖需要 |

**ComfyUI 版本怎麼確認有沒有 MOSS 支援：**

```powershell
# 啟動後查詢
Invoke-RestMethod http://127.0.0.1:8188/object_info |
  Select-Object -ExpandProperty MOSSImageControl
```

有回傳代表原生支援已到位。另外 `CLIPLoader` 的 `type` 選項應該看得到 `moss_image`。

## 二、安裝與權重

### 2.1 下載權重

```powershell
python scripts\download_moss.py
```

腳本會把四個檔案放到 ComfyUI `models/` 底下的正確子資料夾，總計 19.39 GiB。

**為什麼選 int8_convrot？**
`comfy_kitchen` 是 ComfyUI 在 sm89+（RTX 4090/50 系）上的 CUDA backend，
`int8_convrot` 是它的權重量化格式。int8 的 DiT 只有 5.04 GiB，bf16 要 9.46 GiB，
在 24GB 卡上差距很大（生成時 text encoder 還要 9.08 GiB）。

### 2.2 放置位置

```
ComfyUI\models\
├── diffusion_models\
│   ├── moss_image_sft_int8_convrot.safetensors
│   └── moss_image_base_int8_convrot.safetensors
├── text_encoders\
│   └── qwen3.5_9b_int8_convrot.safetensors
└── vae\
    └── flux2-vae_small.safetensors
```

## 三、工作流程節點詳解

### 3.1 節點接法

```
[1] UNETLoader(moss_image_sft_int8_convrot)
      │
[4] ModelSamplingAuraFlow(shift=5)
      │
[7] SkipLayerGuidanceDiTSimple(single_layers="22")
      │
[10] KSampler(Euler / sgm_uniform / CFG 3 / 50 steps)
        ├── positive ← [5] CLIPTextEncode ← [2] CLIPLoader(type=moss_image)
        ├── negative ← [6] CLIPTextEncode(同上)
        └── latent   ← [8] EmptyLatentImage(W×H)
      │
[11] VAEDecode ← [3] VAELoader(flux2-vae_small)
      │
[12] SaveImage
```

### 3.2 三個關鍵參數的由來

**shift = 5，用 `ModelSamplingAuraFlow`**
官方 README 明確寫：原生模型與 SD3 的 timestep multiplier 不同，
**不可從 plugin workflow 直接搬 `ModelSamplingSD3` 節點過來**。
要調 shift 就用 `ModelSamplingAuraFlow`（multiplier 1）。

**SkipLayerGuidanceDiTSimple(single_layers="22")**
官方 USERGUIDE 的「Two-Pass Sampling SLG」：negative-branch skip，
推薦 skip layer **22**（0-indexed），22/24/26 都有幫助。
`single_layers` 就是 uncond skip；`double_layers` 留空。
若改成標準 SLG，官方給的參數是 scale 3、start 0.01、end 1.0、
`double_layers` 留空，並保留 `ModelSamplingSD3` 在 SLG 之前。

**負向提示詞**
官方建議：`nsfw, worst quality, bad anatomy, watermark, jpeg artifact`

### 3.3 解析度與 token 預算

| 解析度 | 像素量 | 實測耗時 | 備註 |
| --- | --- | --- | --- |
| 1024×1024 | 1.05 MP | 3.0 s | 官方推薦區間 |
| 2048×2048 | 4.19 MP | 84.1 s | 官方外推上限（≈2K） |
| 2560×2560 | 6.55 MP | 147.5 s | 超出官方外推上限，實測仍可 |

尺寸必須是 **16 的倍數**，否則 Python API 會直接報錯。
寬度上限 16384（`EmptyLatentImage` 的 `max`）。

### 3.4 套 LoRA

`moss_workflow.py` 加上 `--lora` 會插入 `LoraLoader`：

```powershell
python scripts\moss_workflow.py --out wf.json --model base --seed 777 `
  --lora moss_studio_lora-lycoris.safetensors --lora-strength 1.0
```

`strength_clip` 固定為 0（這是 DiT 模型，不是文字模型）。

## 四、腳本用法

| 腳本 | 用途 |
| --- | --- |
| `moss_workflow.py` | 產生 API 工作流程 JSON（不手寫節點） |
| `run_moss.py` | 送出 → 輪詢 → 取圖 → 記錄耗時與結果 JSON |
| `download_moss.py` | 下載權重並放到正確子資料夾 |
| `make_lora_dataset.py` | 從提示詞批次生成資料集（圖 + .txt sidecar） |
| `make_compare.py` | 三種對比圖 |
| `run_batch.ps1` | 解析度矩陣批次（1024/2048/2560 + Base） |
| `run_lora_compare.ps1` | Base vs Base+LoRA 批次 |

常用參數：

```powershell
python scripts\moss_workflow.py --model sft|base --width W --height H --seed N --steps N
python scripts\run_moss.py --wf wf.json --tag 名稱 --timeout 1800
python scripts\make_lora_dataset.py --out 資料夾 --n 16 --seed 900 --model base
```

## 五、踩坑紀錄

### 5.1 ComfyUI 版本判斷（最容易誤判的一個）

MOSS-Image 的 README 到 2026-10-23 仍寫著「v0.41.3 尚未包含此支援」，
但那是 **10/19 的寫作時點**。實際上：

| 版本 | 是否含 MOSS |
| --- | --- |
| v0.41.3 | ❌ |
| **v0.42.0 起** | ✅ |
| v0.42.5（當時最新） | ✅ |

驗證方式（`git merge-base --is-ancestor e087d8f6 <tag>`，exit 0 代表包含）：

```powershell
git -C <ComfyUI> fetch upstream master --tags
foreach ($t in @('v0.41.3','v0.42.0','v0.42.5')) {
  git -C <ComfyUI> merge-base --is-ancestor e087d8f6 $t
  "$t : $(if($LASTEXITCODE -eq 0){'HAS MOSS'}else{'no'})"
}
```

> **v0.42.1 的 release notes 是**「Fix some crashing issues a few users experienced when
> running the MOSS image template workflow」— 也就是原生支援剛併入時 MOSS 範本會 crash，
> v0.42.1 才修好。要升就別停在 v0.42.0。

### 5.2 訓練時 ComfyUI 必須關掉

LoHa kernel 自動調校時會窮舉各種 Triton 設定，每個都要實際跑一次算 memory。
ComfyUI 同時開著時佔住約 11 GB VRAM，導致調校階段連續 OOM：

```
kernel loha.bypass_fwd: 10 config(s) failed (OutOfMemoryError: ...)
```

症狀是訓練能跑完，但速度與 kernel 選擇都不理想。
**關掉 ComfyUI 後 VRAM 從 18.8 GB 降到 7.2 GB**，調校就正常了。

### 5.3 `--steps` 必須大於 warmup

```powershell
kohakumoss-config new train ... --steps 100
```

會被設定檢查擋下：

```
the composed config is invalid:
  - SCHEDULE warmup=100 >= its end 100 (STEPS): the schedule never leaves warmup.
    Fix: set SCHEDULE['warmup'] below 100 (e.g. 10), or raise STEPS above 100
```

用 `--steps 150` 即可。

### 5.4 Windows 只有 spawn，step 1 會等

Windows 的 DataLoader 只能用 `spawn` 啟動 worker，每個 worker 都要重新 import
訓練器與 kernel 模組。官方量測：4 workers 時 step 1 會等約 **46 秒**（RTX 5090 主機、Linux）。
之後每步約 5 秒。**不是卡住，是正常的**。

### 5.5 PowerShell 把 git 的正常輸出當錯誤

`git checkout` 會把 `Note: switching to 'v0.42.5'.` 寫到 **stderr**，
PowerShell 會把它當成錯誤並讓腳本提前中斷，看起來像「指令沒生效」。

解法：用 `cmd /c` 包裝。

```powershell
cmd /c "cd /d <repo> && git checkout -f v0.42.5 2>&1"
```

### 5.6 `--pathspec-from-file` 的清單必須是 UTF-8 無 BOM

```powershell
# ❌ BOM 會讓第一行路徑變成 "\ufeffscripts/foo.py"，整批 fail
Set-Content $list $lines -Encoding UTF8

# ✅ 無 BOM
[System.IO.File]::WriteAllLines($list, $lines, (New-Object System.Text.UTF8Encoding($false)))
```

### 5.7 `git add <dir>` 會把生成素材拖進 index

如果資料夾裡混了 45 GB 的影片／影格，`git add scripts/` 會全部納入暫存區，
暫存區與 .git 直接暴增數十 GB。**用副檔名白名單精準挑檔**：

```powershell
$codeExt = @('.py','.ps1','.bat','.txt','.json','.md')
Get-ChildItem scripts -File -Recurse |
  Where-Object { $codeExt -contains $_.Extension.ToLower() } |
  ForEach-Object { $_.FullName.Replace($repo + '\','').Replace('\','/') } |
  Set-Content .git\addlist.txt -Encoding UTF8
git add --pathspec-from-file=.git\addlist.txt
```

### 5.8 Windows 保留名稱 `nul` 會讓 git 卡住

```
D:\AI\ComfyUI\nul
```

這是 Windows 的保留裝置名稱，**git 刪不掉它**（`git stash push` 會警告
`failed to remove nul: Permission denied`），導致 stash 無法清理工作區，
後續 checkout / cherry-pick 全部連鎖失敗。

```powershell
cmd /c "del \\?\D:\AI\ComfyUI\nul"
```

### 5.9 shallow clone 不能 rebase / merge

本機的 ComfyUI 是 shallow clone（只有 2 個邊界 commit），
`git merge-base main upstream/master` 會失敗、沒有共同祖先。
直接 `git rebase` 或 `git merge` 會產生大量衝突。
**解法**：以 `upstream/<tag>` 為基底 checkout，再用 `cherry-pick` 重放自己的 commit。

## 六、已知限制

- **訓練只驗過 24GB 卡的 bf16 profile。** 6GB / 8GB 的 profile 是官方保證但本機未實測
  （需離線快取，且 text encoder 要常駐）。
- **LoRA 樣式是通用風格，不是人物 LoRA。** 人物 LoRA 需要真實照片當訓練資料，
  權重請勿公開傳播。
- **`-comfy` 格式的 adapter 有 2.36 GB**（LyCORIS 融合 q/k/v repack），
  `-lycoris` 只有 141.88 MB。兩者都能用，但後者小很多，推薦用 `-lycoris`。
- **int8 權重的極小數權重可能與 bf16 有微小差異**，官方說明是「按 DiT 檔案裡存的精度跑」。

