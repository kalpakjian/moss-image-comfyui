# 下載 MOSS-Image ComfyUI 格式權重
# 來源: https://huggingface.co/Comfy-Org/MOSS-Image  (Apache-2.0)
# 目標: D:\AI\ComfyUI\models\{diffusion_models,text_encoders,vae,loras}
# 選檔依據: 本機既有慣例為 int8_convrot (qwen_image_2.1 / minimax_h3 都是)，故採 int8 變體
import os, sys, time, traceback

# 權重快取位置（避免塞爆系統碟）
os.environ.setdefault("HF_HOME", os.environ.get("MOSS_HF_HOME", r"D:\AI\hf_cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/MOSS-Image"
# ComfyUI 的 models 資料夾；可用 MOSS_MODELS 覆寫
MODELS = os.environ.get("MOSS_MODELS", r"D:\AI\ComfyUI\models")

# (repo 內路徑, 目標資料夾)
FILES = [
    ("diffusion_models/moss_image_sft_int8_convrot.safetensors",   "diffusion_models"),
    ("diffusion_models/moss_image_base_int8_convrot.safetensors",  "diffusion_models"),
    ("text_encoders/qwen3.5_9b_int8_convrot.safetensors",          "text_encoders"),
    ("vae/flux2-vae_small.safetensors",                            "vae"),
]

def human(n):
    return f"{n/1024**3:.2f} GiB"

for rpath, subdir in FILES:
    target_dir = os.path.join(MODELS, subdir)
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, os.path.basename(rpath))
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
        print(f"SKIP  {rpath}  (already {human(os.path.getsize(dest))})", flush=True)
        continue
    print(f"DOWN  {rpath} -> {dest}", flush=True)
    t0 = time.time()
    try:
        got = hf_hub_download(
            repo_id=REPO, filename=rpath,
            local_dir=target_dir,
            resume_download=True,
        )
        # local_dir 會保留 subdir 結構，改以實際路徑搬到 models/<subdir>/<name>
        produced = got
        want = dest
        if os.path.abspath(produced) != os.path.abspath(want):
            if os.path.exists(produced):
                if os.path.exists(want):
                    os.remove(want)
                os.replace(produced, want)
        sz = os.path.getsize(want) if os.path.exists(want) else -1
        print(f"OK    {rpath}  {human(sz)}  in {time.time()-t0:.0f}s", flush=True)
    except Exception:
        print(f"FAIL  {rpath}", flush=True)
        traceback.print_exc()

print("ALL DONE", flush=True)
