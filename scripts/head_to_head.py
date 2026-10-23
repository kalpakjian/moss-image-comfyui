"""
MOSS-Image 5B vs Qwen-Image 2.1 — 同機同提示詞對打

設計原則：每個模型都用「各自官方推薦的取樣參數」，比的是開箱即用的品質，
不是把參數調到極致。

  MOSS-Image 5B  (官方 comfyui/README.md)
    50 steps / CFG 3 / shift 5 / Euler + sgm_uniform / SLG single_layers=22
    text encoder: qwen3.5_9b_int8 (9.08 GiB)   VAE: flux2-vae_small (0.23 GiB)

  Qwen-Image 2.1  (官方 ComfyUI 範例)
    20 steps / CFG 1.0 / Euler + simple / 無 shift patch / 無負向提示詞
    text encoder: qwen3vl_8b_bf16 (16.33 GiB)  VAE: qwen_image_2.1_vae_bf16 (0.63 GiB)

量測：每次生成前呼叫 /free 清空模型，再用背景執行緒輪詢 nvidia-smi 取峰值 VRAM。

用法：
  python head_to_head.py --out output/h2h
  python head_to_head.py --out output/h2h --quick     # 只跑 1024 兩個提示詞
"""
import argparse, json, os, subprocess, threading, time, urllib.error, urllib.parse, urllib.request, uuid

SERVER = os.environ.get("MOSS_SERVER", "http://127.0.0.1:8188")

MOSS_UNET = {"sft": "moss_image_sft_int8_convrot.safetensors",
             "base": "moss_image_base_int8_convrot.safetensors"}
MOSS_TE = "qwen3.5_9b_int8_convrot.safetensors"
MOSS_VAE = "flux2-vae_small.safetensors"
MOSS_NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

QWEN_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_TE = "qwen3vl_8b_bf16.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"


def build_moss(model, w, h, prompt, seed, steps=50, cfg=3.0, shift=5.0, skip=22):
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MOSS_UNET[model], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MOSS_TE, "type": "moss_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MOSS_VAE}},
        "4": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": shift}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": MOSS_NEG, "clip": ["2", 0]}},
        "7": {"class_type": "SkipLayerGuidanceDiTSimple",
              "inputs": {"model": ["4", 0], "single_layers": str(skip),
                         "double_layers": "", "start_percent": 0.0, "end_percent": 1.0}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "sgm_uniform",
                          "positive": ["5", 0], "negative": ["6", 0],
                          "latent_image": ["8", 0], "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "moss_h2h", "images": ["11", 0]}},
    }
    return wf


def build_qwen(w, h, prompt, seed, steps=25, cfg=1.0):
    """官方 ComfyUI 範本（workflow_templates/image_qwen_image_2_1_t2i.json）：
    25 steps / CFG 1.0 / Euler + simple / 無 shift patch / 無負向提示詞
    """
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": QWEN_TE, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple",
                         "model": ["1", 0], "positive": ["5", 0], "negative": ["5", 0],
                         "latent_image": ["4", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "qwen_h2h", "images": ["7", 0]}},
    }
    return wf


def post(path, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:600]}") from None


def get(path, timeout=30):
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def free_memory():
    """清空已載入模型，讓每次量測從乾淨狀態開始"""
    try:
        post("/free", {"unload_models": True, "free_memory": True})
        time.sleep(2)
    except Exception:
        pass


class VramWatcher(threading.Thread):
    """背景輪詢 nvidia-smi 取峰值 VRAM (MiB)"""

    def __init__(self, interval=0.35):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        # 注意：不可命名為 self._stop，會覆蓋 threading.Thread._stop() 內部方法
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                for line in out.stdout.strip().splitlines():
                    self.peak = max(self.peak, int(line.strip()))
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop_watch(self):
        self._stop_event.set()
        self.join(timeout=5)
        return self.peak

PROMPTS = [
    {
        "id": "photo",
        "name": "寫實室內場景",
        "prompt": ("A cozy sunlit studio corner with a wooden desk, a small potted plant, "
                   "warm afternoon light through a window, soft shadows, detailed textures, "
                   "photorealistic, high detail"),
    },
    {
        "id": "cntext",
        "name": "中文文字渲染海報",
        "prompt": ("一張現代簡約風格的咖啡店宣傳海報，米白色紙質背景，"
                   "頂部有醒目的中文標題「城市漫遊」，下方有英文副標題「CITY WALK」，"
                   "中間是一個簡約的咖啡杯線稿插圖，暖棕與墨綠配色，專業排版，"
                   "平面設計，海報"),
    },
    {
        "id": "entext",
        "name": "英文文字渲染招牌",
        "prompt": ("A cozy artisan coffee shop storefront sign at dusk, the sign reads "
                   "MOSS & BEAN in warm golden letters on a dark wooden board, "
                   "string lights, wet cobblestone street, reflections, photorealistic, "
                   "night photography, high detail"),
    },
]


def run_one(name, wf, out_dir, tag, cold=True):
    """送出一份工作流程 → 輪詢 → 取圖 → 回傳耗時與峰值 VRAM

    cold=True  : 呼叫 /free 清空模型後執行（耗時含模型載入 = 端到端體感）
    cold=False : 模型已在記憶體（耗時 = 純推論時間）
    """
    if cold:
        free_memory()
    watcher = VramWatcher()
    watcher.start()
    t0 = time.time()
    res = post("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})
    pid = res["prompt_id"]
    deadline = time.time() + 1800
    out = None
    while time.time() < deadline:
        time.sleep(0.8)
        try:
            hist = get(f"/history/{pid}")
        except Exception:
            continue
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"{name} failed: {entry['status']}")
            for node_out in entry.get("outputs", {}).values():
                if "images" in node_out:
                    out = node_out["images"][0]
                    break
            if out:
                break
    elapsed = time.time() - t0
    peak = watcher.stop_watch()
    if not out:
        raise RuntimeError(f"{name}: timeout")

    src = out.get("filename")
    sub = out.get("subfolder", "")
    # ComfyUI /view 只接受 type=output 或 type=temp（沿用 history 回傳的值）
    ftype = out.get("type", "output")
    url = f"{SERVER}/view?filename={urllib.parse.quote(src)}&subfolder={urllib.parse.quote(sub)}&type={ftype}"
    dst = os.path.join(out_dir, f"{tag}.png")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)
    return {"tag": tag, "seconds": round(elapsed, 1), "peak_vram_mib": peak,
            "cold": cold, "file": dst, "width": out.get("width"), "height": out.get("height")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/h2h")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="只跑 1024")
    ap.add_argument("--only", choices=["photo", "cntext", "entext"], default=None)
    ap.add_argument("--res", type=int, default=None, help="單一解析度（正方形），例如 2560")
    ap.add_argument("--sizes", default=None, help="逗號分隔的解析度清單，例如 1024,2048,2560")
    a = ap.parse_args()

    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)

    prompts = [p for p in PROMPTS if (a.only is None or p["id"] == a.only)]
    sizes = [1024] if a.quick else [1024, 2048]
    if a.sizes:
        sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    if a.res:
        sizes = [a.res]

    results = []
    print(f"ComfyUI: {SERVER}   out: {out_dir}", flush=True)
    for w in sizes:
        for p in prompts:
            for model in ("moss", "qwen"):
                tag = f"{p['id']}_{w}_{model}"
                print(f"[{tag}] generating ...", flush=True)
                try:
                    if model == "moss":
                        wf = build_moss("sft", w, w, p["prompt"], a.seed)
                    else:
                        wf = build_qwen(w, w, p["prompt"], a.seed)
                    # 每個 (model, resolution) 組合的第一張為冷啟動（含模型載入），
                    # 其後同一組合的生成保持熱載入 → 純推論時間
                    first = not any(r.get("model") == model and r.get("res") == f"{w}x{w}"
                                    for r in results)
                    r = run_one(model, wf, out_dir, tag, cold=first)
                    r.update({"model": model, "prompt_id": p["id"], "prompt_name": p["name"],
                              "res": f"{w}x{w}", "mp": round(w * w / 1e6, 2)})
                    results.append(r)
                    kind = "COLD" if first else "warm"
                    print(f"[{tag}] {r['seconds']}s ({kind})  peak {r['peak_vram_mib']} MiB", flush=True)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[{tag}] ERROR {e}\n{tb}", flush=True)
                    results.append({"tag": tag, "model": model, "error": str(e)})

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("DONE", len(results), "runs")


if __name__ == "__main__":
    main()
