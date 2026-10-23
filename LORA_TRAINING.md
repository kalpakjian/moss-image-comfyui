# KohakuMossTrainer × MOSS-Image — LoRA 訓練完整流程

用 [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) 在 MOSS-Image 上
訓練 LoRA／LoHa／LoKr／DoRA 等 adapter，**6 GB 顯卡起步**，Windows 原生支援。

本文件是實測紀錄（RTX 4090 24GB、Windows 11、2026-10-23），不是抄官方文件。

---

## 一、為什麼是這個組合

- **MOSS-Image-Base** 未經後訓練、完整保留訓練資訊 → 官方明講是留給大家 train LoRA 用
- **MOSS-Image-SFT** 是後訓練版 → 適合直接生成
- **KohakuMossTrainer** 可以**直接吃 ComfyUI 的權重**（單檔 safetensors），
  不需要另外下載 26 GB 的原始 checkpoint，訓練完的 adapter 也能直接在 ComfyUI 用

## 二、實測結果摘要

| 項目 | 實測值 |
| --- | --- |
| OS / Python | Windows 11 / 3.12.13（uv 管理） |
| torch | **2.14.1+cu130**（依驅動 CUDA 13.4 自動選 cu130 index） |
| GPU | RTX 4090，sm89，23.99 GiB |
| Triton | triton-windows 3.8.0.post29 |
| TileLang | 0.1.15（`status ok`） |
| 編譯器 | MSVC 14.51.36231 + nvcc 13.4（自動偵測） |
| `kohakumoss-doctor` | **missing: nothing** |
| 演算法 | LyCORIS **LoHa** rank 8，252 modules |
| 步數 | 150（每步 2 張圖，8192 image tokens + 123.5 text tokens） |
| 速度 | **5.0 s/step**（首步 28.9 s 含 kernel 預熱） |
| **峰值 VRAM** | **6.20 GiB allocated / 8.24 reserved / 裝置總用量 10.0 GiB** |
| loss | step1 0.27 → step40 0.2203（最低）→ step150 0.2893 |
| 結果 | `ok=True, samples=300, nonfinite=0, skipped=0` |

> **官方說的「6 GB cards 可訓練」不是保守宣稱** — 本機實測峰值 6.20 GiB，與官方一致。

## 三、安裝

### 3.1 前置檢查

```powershell
# Visual Studio Build Tools（C++ 工作負載）——Windows 需要它編譯 Triton / TileLang kernel
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json

# 檢查 driver 能跑哪個 CUDA index
nvidia-smi | Select-String 'CUDA (UMD )?Version'
```

`nvidia-smi` 的 CUDA 版本決定安裝器選哪個 PyTorch index：

| Driver CUDA | 選用 index |
| --- | --- |
| ≥ 13.0 | `cu130` |
| ≥ 12.8 | `cu128` |
| ≥ 12.6 | `cu126` |

### 3.2 安裝

```powershell
git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer
cd KohakuMossTrainer
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1
```

安裝器會：

1. 偵測 GPU 與 driver，選對 torch index
2. 用 uv 建立 **Python 3.12** 的 `.venv`（沒 uv 就用 `py -3.12 -m venv`）
3. 裝 `torch>=2.13,<2.15`（對應 index）
4. 裝 KohakuMoss（clone 情境為 editable）
5. 跑 `kohakumoss-doctor`

**驗證：**

```powershell
.venv\Scripts\kohakumoss-doctor.exe
```

看到 `missing: nothing` 就是環境就緒。doctor 會列出每個 GPU 的架構、
已安裝的工具鏈、每個 op 用哪個 kernel backend，以及缺什麼要補什麼。

## 四、準備資料集

### 4.1 資料集格式

用 `folder` source：**圖片 + 同名 `.txt` sidecar**。

```
my_dataset\
├── moss_000.png
├── moss_000.txt      ← caption
├── moss_001.png
├── moss_001.txt
└── ...
```

caption 寫**自然語言描述**（MOSS-Image 的 text encoder 吃這個）。
可用副檔名：`.txt` / `.caption` / `.json`（JSON 支援 `tags` / `caption` / `artist` 欄位）。

### 4.2 用本 repo 的腳本產生資料集

```powershell
python scripts\make_lora_dataset.py --out my_dataset --n 16 --seed 900 --model base
```

**重點：LoRA 要學的是「風格」，不是「某一張圖」。**
提示詞要共享風格詞（光線、色調、質感），只在主體上變化。範本見
[prompts/training_prompts.txt](prompts/training_prompts.txt)。

實測 16 張 × 150 步已能學到明顯風格差異；正式使用建議 20～50 張。

### 4.3 三個快取步驟（只需跑一次）

```powershell
kogine run scripts/data/inspect_data.py  --config configs/user/my_lora_data.py
kogine run scripts/data/cache_text.py     --config configs/user/my_lora_data.py
kogine run scripts/data/cache_latents.py  --config configs/user/my_lora_data.py
```

`inspect_data` 會印出資料計畫：抽樣數、token 預算、溢位處理等。
本次實測：16 張 1024×1024、每批 2 張、token 預算 8312/步、0 溢位丟棄。

## 五、訓練

### 5.1 產生設定

```powershell
kohakumoss-config new train --weights <ComfyUI>\models --data my_dataset `
  --source folder --algo loha --name my_lora --steps 150 --vram 24 --precision bf16 `
  --prompt "你的代表性提示詞"
```

會寫出 `configs/user/my_lora.py` 與 `configs/user/my_lora_data.py`。

> ⚠️ **`--steps` 必須大於 warmup**。用 `--steps 100` 會被擋下：
> `SCHEDULE warmup=100 >= its end 100`。用 150 即可。

可選旗標：

| 旗標 | 值 |
| --- | --- |
| `--algo` | `loha` / `lokr` / `oft` / `dora` / `full` |
| `--source` | `folder` / `folder_nl` / `jsonl` / `parquet` / `webdataset` / `moss_shard` |
| `--vram` | GiB 數字或 `auto`（預設 24 → `recipes_lora_24g_bf16.py`） |

### 5.2 開始訓練

```powershell
# ⚠️ 訓練前務必關掉 ComfyUI！
kogine run scripts/train/moss.py --config configs/user/my_lora.py `
  --set SAMPLE_EVERY=25 --set SAVE_EVERY=50
```

任何 knob 都能用 `--set KEY=VALUE` 覆寫，例如 `--set LR=2e-4 --set DIM=16`。
中斷後續跑：`--set RESUME=latest`。

**第一次執行會先做 kernel 自動調校**（每個 op 在這張卡、這個 CUDA、這個 torch 版本下
哪個 backend 最快），結果存在 `~/.cache/kohakumoss_kernels`，之後的執行直接沿用。

## 六、產出與驗證

### 6.1 產出位置

```
runs/moss/my_lora/
├── config.json              # 所有 knob
├── log.txt                  # 完整日誌（計畫、kernel 決策、警告）
├── metrics.jsonl            # 每步指標
├── samples/step_XXXXXXXX/   # 訓練中取樣
├── checkpoints/step_XXXXXXXX/
│   ├── trainable.safetensors
│   └── export/
│       ├── my_lora-comfy.safetensors      # ComfyUI 格式（2.36 GB）
│       └── my_lora-lycoris.safetensors    # LyCORIS 格式（141.88 MB）
└── export/                  # 最後一次的兩份匯出
```

### 6.2 兩種格式怎麼選

| 格式 | 大小 | 用途 |
| --- | --- | --- |
| `-lycoris` | **141.88 MB** | ✅ 推薦。ComfyUI 的 `LoraLoader` 直接可用 |
| `-comfy` | 2362.57 MB | ComfyUI key layout + fused q/k/v repack，較大 |

### 6.3 裝進 ComfyUI 驗證

```powershell
Copy-Item runs\moss\my_lora\export\my_lora-lycoris.safetensors `
          <ComfyUI>\models\loras\ -Force
```

重啟 ComfyUI 後，`LoraLoader` 的選單就會出現這個 LoRA。

```powershell
# 驗證它真的生效
python scripts\moss_workflow.py --out wf.json --model base --seed 777 `
  --lora my_lora-lycoris.safetensors --lora-strength 1.0
python scripts\run_moss.py --wf wf.json --tag with_lora
```

**驗證要用的提示詞必須是「沒在訓練資料裡」的**，否則你看到的是記憶而非泛化。

## 七、常見問題

### Q. 訓練時 ComfyUI 一定要關嗎？

**要。** 它佔住約 11 GB VRAM，會讓 LoHa kernel 自動調校階段連續 OOM
（`kernel loha.bypass_fwd: 10 config(s) failed (OutOfMemoryError: ...)`）。
關掉後 VRAM 從 18.8 GB 降到 7.2 GB，調校就正常。

### Q. step 1 要等很久是壞掉嗎？

不是。Windows 的 DataLoader 只能用 `spawn`，每個 worker 都要重新 import 訓練器與 kernel
模組，官方量測 4 workers 時 step 1 會等約 46 秒。之後每步約 5 秒。

### Q. 可以用 6GB 顯卡嗎？

可以，官方保證。指令是 `--vram 6`，它會套用
weight streaming + activation/gradient offload + host AdamW states 的組合，
並需要離線的 latent 與 text 快取（就是步驟 4.3 那兩個）。

```powershell
kohakumoss-config new train ... --vram 6
```

### Q. 要訓練多久？

150 步 × 5 秒 ≈ **13 分鐘**（不含首次 kernel 調校約 5～8 分鐘）。
資料集越大、解析度越高越慢。

### Q. 怎麼續跑中斷的訓練？

```powershell
kogine run scripts/train/moss.py --config configs/user/my_lora.py --set RESUME=latest
```

## 八、相關連結

- KohakuMossTrainer：https://github.com/KohakuBlueleaf/KohakuMossTrainer
- MOSS-Image：https://github.com/OpenMOSS/MOSS-Image
- 官方 ComfyUI 格式權重：https://huggingface.co/Comfy-Org/MOSS-Image
- LyCORIS：https://github.com/KohakuBlueleaf/LyCORIS

---

實測日期：2026-10-23 ｜ RTX 4090 24GB ｜ Windows 11 ｜ torch 2.14.1+cu130

