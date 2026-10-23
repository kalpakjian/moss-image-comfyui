# -*- coding: utf-8 -*-
"""產生 LoRA 訓練資料集：同一提示詞的多樣影像 + .txt sidecar caption

用法: python make_lora_dataset.py --out <dir> --n 12 --seed 900
"""
import argparse, os, sys, json, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moss_workflow import build

SERVER = os.environ.get("MOSS_SERVER", "http://127.0.0.1:8188")

# 固定提示詞 = 風格一致，這是 LoRA 訓練的關鍵
PROMPTS = [
    "A cozy sunlit studio corner with a wooden desk, a small potted plant, "
    "warm afternoon light through a window, soft shadows, detailed textures, "
    "photorealistic, high detail",
    "A sunlit reading nook with a wicker armchair, a stack of books and a floor lamp, "
    "warm afternoon light through a window, soft shadows, detailed textures, "
    "photorealistic, high detail",
    "A wooden workbench by a bright window with hand tools, a small potted fern, "
    "warm afternoon light, soft shadows, detailed textures, photorealistic, high detail",
    "A tidy kitchen corner with a small table, a ceramic mug and a herb pot on the sill, "
    "warm morning light through a window, soft shadows, detailed textures, "
    "photorealistic, high detail",
]


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(SERVER + path, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_image(meta):
    import urllib.parse
    qs = urllib.parse.urlencode({"filename": meta["filename"],
                                 "subfolder": meta.get("subfolder", ""),
                                 "type": meta.get("type", "output")})
    with urllib.request.urlopen(f"{SERVER}/view?{qs}") as r:
        return r.read()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--model", default="base")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    made = 0
    for i in range(a.n):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = a.seed + i
        wf = build(a.model, a.width, a.height, prompt, seed,
                   steps=40, cfg=3.0)
        payload = {"prompt": wf, "client_id": str(uuid.uuid4())}
        try:
            res = post("/prompt", payload)
        except Exception as e:
            print(f"[{i}] submit failed: {e}", flush=True)
            continue
        if res.get("node_errors"):
            print(f"[{i}] node errors", flush=True)
            continue
        pid = res["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < 600:
            time.sleep(2)
            h = get(f"/history/{pid}")
            if pid in h:
                entry = h[pid]
                if entry.get("status", {}).get("status_str") == "error":
                    print(f"[{i}] runtime error", flush=True)
                    break
                imgs = []
                for node_out in entry.get("outputs", {}).values():
                    imgs += node_out.get("images", [])
                if imgs:
                    raw = fetch_image(imgs[0])
                    name = f"moss_{i:03d}.png"
                    with open(os.path.join(a.out, name), "wb") as f:
                        f.write(raw)
                    # caption sidecar：MOSS 官方建議自然語言描述
                    with open(os.path.join(a.out, f"moss_{i:03d}.txt"), "w",
                              encoding="utf-8") as f:
                        f.write(prompt)
                    made += 1
                    print(f"[{i}] OK {name} seed={seed} "
                          f"{time.time()-t0:.0f}s", flush=True)
                break
    print(f"DATASET DONE: {made} images in {a.out}", flush=True)
