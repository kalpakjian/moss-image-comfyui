"""
驗證 Qwen-Image 2.1 的三項「只在官方文件看過」的能力

  1. Alpha 通道：4 通道 VAE 解碼出來的圖有沒有 alpha、背景是不是真的透明
  2. 圖像編輯：同一份權重做 instruction-based editing
  3. 多圖參考：TextEncodeQwenImage21 的 image_1 / image_2 同時餵入

用法：python verify_qwen_extras.py [輸出資料夾]
"""
import json, os, sys, time, urllib.parse, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_to_head as H  # noqa: E402

UNET = "qwen_image_2.1_int8_convrot.safetensors"
TE = "qwen3vl_8b_bf16.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"


def base_nodes():
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }


def run(wf, out_dir, tag, timeout=900):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    res = H.post("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})
    if res.get("node_errors"):
        raise RuntimeError("node errors: " + json.dumps(res["node_errors"], ensure_ascii=False)[:500])
    pid = res["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            hist = H.get(f"/history/{pid}")
        except Exception:
            continue
        if pid in hist:
            e = hist[pid]
            st = e.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError("exec error: " + json.dumps(st, ensure_ascii=False)[:500])
            for o in e.get("outputs", {}).values():
                if "images" in o:
                    meta = o["images"][0]
                    url = (f"{H.SERVER}/view?filename={urllib.parse.quote(meta['filename'])}"
                           f"&subfolder={urllib.parse.quote(meta.get('subfolder',''))}"
                           f"&type={meta.get('type','output')}")
                    with urllib.request.urlopen(url, timeout=120) as r:
                        data = r.read()
                    dst = os.path.join(out_dir, f"{tag}.png")
                    with open(dst, "wb") as f:
                        f.write(data)
                    return {"tag": tag, "seconds": round(time.time() - t0, 1), "file": dst}
    raise RuntimeError(f"{tag}: timeout")


def t2i_alpha(out_dir):
    """測 alpha：t2i 後檢查輸出 PNG 有沒有 RGBA / alpha 通道與透明像素"""
    wf = base_nodes()
    wf["4"] = {"class_type": "EmptyLatentImage",
               "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": ["1", 0],
                          "positive": ["5", 0], "negative": ["5", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "alpha_test", "images": ["7", 0]}}
    r = run(wf, out_dir, "01_alpha_t2i")

    from PIL import Image
    im = Image.open(r["file"])
    info = {"tag": "alpha_t2i", "mode": im.mode, "size": list(im.size), "seconds": r["seconds"]}
    if im.mode in ("RGBA", "LA"):
        a = im.getchannel("A")
        hist = a.histogram()
        total = sum(hist)
        info["alpha_min"] = a.getextrema()[0]
        info["alpha_max"] = a.getextrema()[1]
        info["fully_transparent_pct"] = round(100 * hist[0] / total, 2)
        info["fully_opaque_pct"] = round(100 * hist[255] / total, 2)
    else:
        info["note"] = "輸出為 RGB，無 alpha 通道"
    return r, info

def image_edit(out_dir, ref="qwen_ref_a.png",
               instruction="Keep the room, the window, the plant, the lamp and the bed "
                           "in <image1> exactly as they are. Only replace the wooden desk "
                           "in <image1> with a dark walnut desk. Keep everything else unchanged."):
    """測圖像編輯：TextEncodeQwenImage21 + 參考圖

    重點：指令必須用 <image1> 明確指涉參考圖，否則模型會當成一般 t2i 重新生成。
    """
    wf = base_nodes()
    wf["4"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
    wf["5"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0], "prompt": instruction,
                          "negative_prompt": "", "resolution": 1024,
                          "images": {"image_1": ["4", 0]}}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": ["1", 0],
                          "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": ["5", 2], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "edit_test", "images": ["7", 0]}}
    return run(wf, out_dir, "02_image_edit")


def multi_ref(out_dir, ref_a="qwen_ref_a.png", ref_b="qwen_ref_b.png"):
    """測多圖參考：image_1 + image_2 同時餵入 text encoder"""
    wf = base_nodes()
    wf["4"] = {"class_type": "LoadImage", "inputs": {"image": ref_a}}
    wf["5"] = {"class_type": "LoadImage", "inputs": {"image": ref_b}}
    wf["6"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0],
                          "prompt": "Use the color palette of <image1> and the wooden texture "
                                    "of <image2> to make a cozy reading corner, photorealistic.",
                          "negative_prompt": "", "resolution": 1024,
                          "images": {"image_1": ["4", 0], "image_2": ["5", 0]}}}
    wf["7"] = {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": ["1", 0],
                          "positive": ["6", 0], "negative": ["6", 1],
                          "latent_image": ["6", 2], "denoise": 1.0}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "multiref_test", "images": ["8", 0]}}
    return run(wf, out_dir, "03_multi_ref")


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "output/qwen_extras")
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    print("=== 1. Alpha 通道測試 ===", flush=True)
    try:
        r, info = t2i_alpha(out_dir)
        results["alpha"] = info
        print(json.dumps(info, ensure_ascii=False), flush=True)
    except Exception as e:
        results["alpha"] = {"error": str(e)}
        print("ALPHA ERROR:", e, flush=True)

    print("=== 2. 圖像編輯測試 ===", flush=True)
    try:
        r = image_edit(out_dir)
        results["image_edit"] = r
        print(json.dumps(r, ensure_ascii=False), flush=True)
    except Exception as e:
        results["image_edit"] = {"error": str(e)}
        print("EDIT ERROR:", e, flush=True)

    print("=== 3. 多圖參考測試 ===", flush=True)
    try:
        r = multi_ref(out_dir)
        results["multi_ref"] = r
        print(json.dumps(r, ensure_ascii=False), flush=True)
    except Exception as e:
        results["multi_ref"] = {"error": str(e)}
        print("MULTIREF ERROR:", e, flush=True)

    print("=== 4. Qwen 背景去背（alpha）測試 ===", flush=True)
    try:
        make_subject(out_dir)
        r, info = qwen_bg_removal(out_dir)
        results["bg_removal"] = info
        print(json.dumps(info, ensure_ascii=False), flush=True)
    except Exception as e:
        results["bg_removal"] = {"error": str(e)}
        print("BGREMOVE ERROR:", e, flush=True)

    print("=== 5. MOSS JSON Layout 測試 ===", flush=True)
    try:
        r = moss_layout(out_dir)
        results["moss_layout"] = r
        print(json.dumps(r, ensure_ascii=False), flush=True)
    except Exception as e:
        results["moss_layout"] = {"error": str(e)}
        print("MOSSLAYOUT ERROR:", e, flush=True)

    with open(os.path.join(out_dir, "verify_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("DONE")


if __name__ == "__main__":
    pass  # 實際入口在檔案最下方（需等所有函式定義完）

def qwen_bg_removal(out_dir, ref="qwen_subject.png"):
    """測 Qwen 背景去背：官方做法是「編輯 + 指示輸出 PNG」，
    透明背景是模型直接生成進 alpha 通道，不是事後抠图。"""
    wf = base_nodes()
    wf["4"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
    wf["5"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0],
                          "prompt": "Remove the background, and output a PNG image",
                          "negative_prompt": "", "resolution": 1024,
                          "images": {"image_1": ["4", 0]}}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": ["1", 0],
                          "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": ["5", 2], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "bg_remove", "images": ["7", 0]}}
    r = run(wf, out_dir, "04_bg_removal")

    from PIL import Image
    im = Image.open(r["file"])
    info = {"tag": "bg_removal", "mode": im.mode, "size": list(im.size), "seconds": r["seconds"]}
    if im.mode in ("RGBA", "LA"):
        a = im.getchannel("A")
        hist = a.histogram()
        total = sum(hist)
        info["alpha_min"] = a.getextrema()[0]
        info["alpha_max"] = a.getextrema()[1]
        info["fully_transparent_pct"] = round(100 * hist[0] / total, 2)
        info["fully_opaque_pct"] = round(100 * hist[255] / total, 2)
        info["semi_transparent_pct"] = round(100 * (total - hist[0] - hist[255]) / total, 2)
    else:
        info["note"] = "輸出為 RGB，無 alpha 通道"
    return r, info


def make_subject(out_dir):
    """先產生一張主體明確、背景單純的素材，供 Qwen 去背測試"""
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors",
                         "type": "moss_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 5.0}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "A single red apple on a plain white background, "
                                "studio product photo, centered",
                         "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
                         "clip": ["2", 0]}},
        "7": {"class_type": "SkipLayerGuidanceDiTSimple",
              "inputs": {"model": ["4", 0], "single_layers": "22", "double_layers": "",
                         "start_percent": 0.0, "end_percent": 1.0}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "KSampler",
              "inputs": {"seed": 7, "steps": 50, "cfg": 3.0, "sampler_name": "euler",
                         "scheduler": "sgm_uniform", "model": ["7", 0],
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["8", 0], "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "qwen_subject", "images": ["10", 0]}},
    }
    r = run(wf, out_dir, "00_subject")
    import shutil
    shutil.copy(r["file"], r"D:\AI\ComfyUI\input\qwen_subject.png")
    return r


def _moss_layout_json(prompt, elements, width, height):
    """複刻 MOSSImageLayoutPrompt 節點產生的 JSON（格式取自 comfy_extras/nodes_moss_image.py）

    節點本身在 API 模式下無法使用（editor_state 是 BOUNDING_BOXES socketless widget，
    必須由畫布節點連線進來，沒有節點輸出該型別），
    因此改用官方 README 的 Python API 寫法：直接把 layout JSON 當 prompt 傳進去。
    """
    import math
    items = []
    for el in elements:
        label = "text" if el.get("type") == "text" else "object"
        item = {"label": label, "caption": el.get("desc", "")}
        if label == "text":
            item["text"] = el.get("text", "")
        y0, x0, y1, x1 = el["bbox"]
        item["bbox"] = [math.floor(round(y0 * height / 1000, 9)) - height // 2,
                        math.floor(round(x0 * width / 1000, 9)) - width // 2,
                        math.ceil(round(y1 * height / 1000, 9)) - height // 2,
                        math.ceil(round(x1 * width / 1000, 9)) - width // 2]
        items.append(item)
    return json.dumps({"summary": {"caption": prompt.strip()}, "items": items,
                        "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"},
                       ensure_ascii=False, separators=(",", ":"))


def moss_layout(out_dir):
    """測 MOSS-Image 的 JSON Layout 精準控制。

    element/bbox 格式：bbox 為 0~1000 正規化 [ymin, xmin, ymax, xmax]，type 為 "text"/"obj"。
    """
    elements = [
        {"type": "text", "text": "EXPLORE",
         "desc": "Large dark-blue uppercase title centered at the top.",
         "bbox": [40, 200, 160, 800]},
        {"type": "obj",
         "desc": "A flat blue mountain illustration in the lower center.",
         "bbox": [250, 300, 900, 700]},
    ]
    prompt = ("A minimalist travel poster with a cream background, a blue mountain "
              "illustration, and the title EXPLORE at the top.")
    layout = _moss_layout_json(prompt, elements, 1024, 1024)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "moss_layout_prompt.json"), "w", encoding="utf-8") as f:
        f.write(layout)

    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors",
                         "type": "moss_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": 5.0}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": layout, "clip": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
                         "clip": ["2", 0]}},
        "8": {"class_type": "SkipLayerGuidanceDiTSimple",
              "inputs": {"model": ["4", 0], "single_layers": "22", "double_layers": "",
                         "start_percent": 0.0, "end_percent": 1.0}},
        "9": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 50, "cfg": 3.0, "sampler_name": "euler",
                          "scheduler": "sgm_uniform", "model": ["8", 0],
                          "positive": ["6", 0], "negative": ["7", 0],
                          "latent_image": ["9", 0], "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "moss_layout", "images": ["11", 0]}},
    }
    return run(wf, out_dir, "05_moss_layout")

if __name__ == "__main__":
    main()
