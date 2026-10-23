# MOSS-Image 5B vs Qwen-Image 2.1 — 同機實測比較

同一台 **RTX 4090 24GB**、同一套 **ComfyUI 0.42.5**、同一個 seed（42）、
同一組提示詞，逐張對打。**每個模型都用它自己官方推薦的取樣參數**，
比的是「開箱即用的品質與成本」，不是把某一方調到極致。

| | MOSS-Image 5B | Qwen-Image 2.1 |
| --- | --- | --- |
| 發布 | 2026-10（OpenMOSS / MOSI.AI） | 2026（Alibaba Qwen 團隊） |
| 參數量 | **5B** | **20B** |
| 架構 | 單流 DiT（single-stream），Muon optimizer 訓練 | MMDiT |
| Text encoder | `qwen3.5_9b` int8（9.08 GiB） | `qwen3vl_8b` int8（官方）/ bf16（本機 16.33 GiB） |
| VAE | `flux2-vae_small`（0.23 GiB） | `qwen_image_2.1_vae_bf16`（0.63 GiB，**4 通道含 alpha**） |
| 官方取樣 | ≥50 步 / CFG 3.0 / shift 5 / Euler + sgm_uniform / SLG skip layer 22 | 25 步 / CFG 1.0 / Euler + simple / 內建 shift |
| 負向提示詞 | 需要（官方建議 5 個負向詞） | **不需要**（CFG 1.0，positive=negative） |
| 推薦像素量 | 1M / 1.5M / 2M | 原生 2K（2048×2048） |
| 外推範圍 | 0.25M～4M（實測可到 6.55M） | 原生 2K（實測可到 6.55M） |
| 特色能力 | **JSON Layout 精準控制**（文字位置／物件增減）、中英文字渲染 | **生成＋編輯同一份權重**、多圖參考（最多 16 張）、局部編輯、**alpha 透明背景** |
| ComfyUI 支援 | v0.42.0 起原生內建（2026-10-16 合併） | 原生內建，另有 `TextEncodeQwenImage21` / `QwenImage21Cache` 等專用節點 |
| 授權 | Apache-2.0 | Apache-2.0 |

## 二、本機實測數據

測試環境：**RTX 4090 24GB ｜ ComfyUI 0.42.5 ｜ torch 2.16.0.dev20261007+cu132 ｜ seed 42**
（每次生成前呼叫 `/free` 卸載模型，避免 ComfyUI 節點快取造成的假數據）

![MOSS-Image 5B vs Qwen-Image 2.1 同機實測完整數據表](docs/compare/h2h_table.jpg)

### 2.1 生成耗時

| 解析度 | 像素量 | MOSS-Image 5B | Qwen-Image 2.1 | 倍數 |
| --- | --- | --- | --- | --- |
| 1024×1024 | 1.05 MP | **19.4 s** | **11.3～12.1 s** | 1.6～1.7× |
| 2048×2048 | 4.19 MP | **80.9～85.6 s** | **29.8～32.2 s** | 2.6～2.7× |
| 2560×2560 | 6.55 MP | **152.4～162.1 s** | **50.1～55.6 s** | 2.8～3.0× |

**Qwen-Image 2.1 全面更快，而且像素量越大差距越明顯。**

原因拆解：

- **步數**：MOSS 官方要求 ≥50 步；Qwen 官方範本用 25 步
- **取樣路徑**：MOSS 的 `SkipLayerGuidanceDiTSimple` 是 *two-pass sampling*，
  負向分支要額外跑一次；官方也提醒 SLG 會強制 eager execution（不能用 torch.compile）
- **原生解析度**：Qwen-Image 2.1 是**原生 2K**，2048² 是它的原生畫布，不需要外推；
  MOSS 的 4M 以上屬於「外推」區間，token 數暴增導致成本超線性成長

> MOSS 的 50 步 × two-pass 讓 1024² 的成本就已經是 Qwen 的 1.6～1.7 倍；
> 拉高到 4MP / 6.55MP 後差距拉大到 2.6～3.0 倍。

### 2.2 峰值 VRAM

| 解析度 | MOSS-Image 5B | Qwen-Image 2.1 |
| --- | --- | --- |
| 1024×1024 | 15.4～21.4 GiB | **23.0～23.1 GiB** |
| 2048×2048 | 19.7～21.9 GiB | **23.1～23.2 GiB** |
| 2560×2560 | 22.5～23.1 GiB | **23.1～23.2 GiB** |

**MOSS 比較省記憶體**，因為 text encoder 用 int8 的 `qwen3.5_9b`（9.08 GiB），
而 Qwen-Image 2.1 用的是 `qwen3vl_8b`（本機為 bf16 = 16.33 GiB）。
官方範本用 int8 版 `qwen3vl_8b_int8_convrot`，VRAM 會再降一些。

> 兩者在 24GB 卡上都能跑滿 6.55 MP，但 Qwen 幾乎吃滿整張卡（23.2 / 24 GiB）。

### 2.3 冷啟動 vs 熱啟動

| 情境 | MOSS 1024² | Qwen 1024² | MOSS 2048² | Qwen 2048² |
| --- | --- | --- | --- | --- |
| 冷（含模型載入） | 19.4 s | 11.3 s | 80.9 s | 29.8 s |
| 熱（模型已在記憶體） | 19.4 s | 12.1 s | 81.4 s | 31.3 s |

**在高解析度下，模型載入時間可以忽略——耗時幾乎全是取樣。**

### 2.4 ⚠️ 量測陷阱：ComfyUI 的節點快取會讓數字失真

本次實測發現一個很容易讓人publish出假數字的坑，值得單獨列出：

ComfyUI 會依**節點輸入**做執行快取。若整條工作流程（模型、seed、prompt、steps…）
之前跑過，會直接回傳上次的結果、**不重新取樣**。

`scripts/cache_test.py` 的實測結果（MOSS-Image 1024×1024）：

| # | 情境 | 耗時 | 被快取的節點 |
| --- | --- | --- | --- |
| 1 | seed=424242 首次 | **20.44 s** | （無） |
| 2 | **完全相同**的工作流程 | **0.32 s** | **全部 12 個** |
| 3 | 換 seed=424243 | 13.91 s | 只有 1～8（取樣節點有重跑） |
| 4 | 回到 seed=424242 | 14.26 s | 只有 1～8（取樣節點有重跑） |

**機制**：ComfyUI 的 `KSampler` 只在「自己的 seed 改變」時才重新取樣
（第 2 次跑 seed 相同 → 命中；第 3 次換 seed → 重跑；第 4 次換回舊 seed → 仍重跑）。

> ⚠️ 本 repo 早期版本曾把「3.0 秒」寫成 MOSS-Image 1024² 的實測值，
> 實際上那是 **smoke test 與批次測試用了逐字相同的工作流程檔造成的快取命中**。
> 真實值是 **19.4 秒**，本文件與 README 的數字都已更正。
>
> 量測前呼叫 `/free` 卸載模型（重載模型會使快取失效）或每次換 seed，即可避免。


## 三、畫面實測（同一提示詞、同一 seed）

> 完整對比圖：[docs/compare/](docs/compare/)
> （左＝MOSS-Image 5B SFT，右＝Qwen-Image 2.1）

### 3.1 中文文字渲染海報

![中文文字渲染海報對比：左 MOSS-Image 5B，右 Qwen-Image 2.1](docs/compare/cmp_cntext_1024.jpg)

提示詞：`一張現代簡約風格的咖啡店宣傳海報，米白色紙質背景，頂部有醒目的中文標題「城市漫遊」，下方有英文副標題「CITY WALK」…`

**兩個模型的主標題「城市漫遊」與副標題「CITY WALK」都渲染完美、筆畫正確。**
差別在風格與版面：

- **MOSS-Image**：排版層次更豐富（主標＋分隔線＋副標＋咖啡杯插畫＋底部小字欄），
  插畫有配色與材質（深綠杯身、拿鐵拉花、杯套上的葉子標誌），偏「成品海報」
- **Qwen-Image 2.1**：版面極簡、留白大、線稿咖啡杯，文字排版乾淨俐落，
  偏「設計稿／簡報模板」

> ⚠️ **兩個模型的小字都是亂碼。**
> MOSS 的頂部小字是 `PA AJOIDO`／`CHRSTBAM CEEMRERNI`，
> Qwen 的底部中文小字是 `良祝益迎何泡` 這類無意義字。
> 「大字正確、小字崩掉」是兩者共同的弱點，不是某一方的問題。

### 3.2 英文招牌文字

![英文招牌文字對比：左 MOSS-Image 5B，右 Qwen-Image 2.1](docs/compare/cmp_entext_1024.jpg)

提示詞：`…the sign reads MOSS & BEAN in warm golden letters on a dark wooden board…`

**兩個模型都把 "MOSS & BEAN" 渲染得完全正確**，包括 `&` 符號與字距。

- **MOSS-Image**：藍調時刻（blue hour）＋暖色店內光、花環、濕潤鵝卵石反光，
  資訊量較大、構圖較滿
- **Qwen-Image 2.1**：更深的夜色、琥珀色招牌、燈串，氣氛更收斂

### 3.3 寫實室內場景

![寫實室內場景對比：左 MOSS-Image 5B，右 Qwen-Image 2.1](docs/compare/cmp_photo_1024.jpg)

- **MOSS-Image**：畫面元素多（桌燈、書、植物、翻開的書、椅背的布、床頭、牆上畫框），
  是一個「有人住過」的場景
- **Qwen-Image 2.1**：同樣的元素但更少、更乾淨，接近商品攝影風格

> 兩邊都沒有出現手指／肢體變形，寫實度都合格。
> 風格差異是**系統性的**：MOSS 偏「敘事性構圖」，Qwen 偏「乾淨商業視覺」。

## 四、各項能力對照

| 能力 | MOSS-Image 5B | Qwen-Image 2.1 |
| --- | --- | --- |
| 中英文大標題渲染 | ✅ 實測完美 | ✅ 實測完美 |
| 密集小字 | ❌ 亂碼 | ❌ 亂碼 |
| **JSON Layout 精準控制** | ✅ **獨有**（`MOSSImageLayoutPrompt` 節點，文字/物件可指定 bbox、支援編輯與增減） | ❌ |
| **圖像編輯** | ❌（需自行 fine-tune） | ✅ **原生**（同一份權重做生成與編輯） |
| **多圖參考** | ❌ | ✅ 最多 16 張（`TextEncodeQwenImage21` 的 `image_1`～`image_16`） |
| **局部編輯** | ❌ | ✅（描述要改的區域，其餘保持） |
| **Alpha 透明背景** | ❌ | ✅ **獨有**（4 通道 VAE，可直接生成透明背景） |
| Prompt 自動改寫 | ❌ | ✅ 可選（`refine_prompt`，另需一個 PE text encoder） |
| KV cache 加速 | ❌ | ✅ `QwenImage21Cache`（auto/gpu/cpu/off × default/int8/int4） |
| 蒸餾加速 LoRA | 官方「進行中」 | ✅ 社群 Lightning 4 步 LoRA |
| **LoRA 訓練** | ✅ **官方第一方支援**（KohakuMossTrainer，6 GB 起步） | ✅ 社群支援（DiffSynth-Studio） |
| 推薦文字語言 | 英文（中文亦支援） | 中英皆可，英文 prompt 增強尤佳 |

## 五、結論與選用建議

### 各取所需

| 你的需求 | 選哪個 | 理由 |
| --- | --- | --- |
| 日常出圖、要快 | **Qwen-Image 2.1** | 快 1.6～4.8 倍，VRAM 幾乎吃滿但原生 2K |
| 圖像編輯 / 改局部 | **Qwen-Image 2.1** | MOSS 完全沒有，MOSS 要編輯得自己 fine-tune |
| 透明背景素材 | **Qwen-Image 2.1** | 唯一原生支援 alpha 通道 |
| 海報排版、要控制文字位置 | **MOSS-Image** | JSON Layout 是獨有能力 |
| 練自己的 LoRA（風格／人物） | **MOSS-Image** | 官方第一方訓練器，6 GB 就能跑，流程最完整 |
| 6GB～8GB 小顯卡 | **MOSS-Image** | 官方 6 GB 實測可行；Qwen 實測要 23.6 GiB |
| 省 VRAM | **MOSS-Image** | text encoder 較小，峰值低 2～8 GiB |

### 一句話總結

> **Qwen-Image 2.1 是「什麼都能做、而且快」的成熟產品；**
> **MOSS-Image 5B 是「小、便宜、而且能精準控制版面」的基底模型。**
>
> 兩者的**文字渲染能力在實測中幾乎打平**（大標題都完美、小字都亂碼）。
> 真正的分野不在畫質，而在**功能邊界**：
> Qwen 有編輯與 alpha，MOSS 有 layout 與官方 LoRA 生態。

## 六、實測方法與限制

**方法**

- 同一台機器、同一個 ComfyUI、同一個 seed（42）、同一組提示詞
- 每個模型都用**自己官方推薦的取樣參數**（MOSS 50 步/CFG 3/shift 5/SLG 22；
  Qwen 25 步/CFG 1/Euler+simple）
- 每次生成前呼叫 `/free` 卸載模型，避免 ComfyUI 節點快取造成假數據
- 耗時取伺服器端 `execution_start` → `execution_success` 的時間戳；
  VRAM 由背景執行緒輪詢 `nvidia-smi` 取峰值

**限制**

- 只測了 SFT 版的 MOSS-Image（Base 版另有實測見 [LORA_TRAINING.md](LORA_TRAINING.md)）
- 本機 Qwen-Image 2.1 的 text encoder 用的是 **bf16** 版（16.33 GiB），
  官方範本用 int8 版，VRAM 數字會再低一些
- 3 組提示詞的視覺比較是**主觀判讀**，非盲測或量化指標
- 未測 ControlNet / 角色一致性 / LoRA 品質

## 七、重現方式

```powershell
# 完整對打（3 提示詞 × 2 模型 × 3 解析度 = 18 次）
powershell -ExecutionPolicy Bypass -File scripts\run_h2h.ps1 -Out output\h2h -Sizes 1024,2048,2560

# 產生對比圖
python scripts\make_h2h_compare.py

# 驗證 ComfyUI 節點快取會不會影響量測
python scripts\cache_test.py
```


