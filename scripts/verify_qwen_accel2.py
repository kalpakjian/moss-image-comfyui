import json, os, sys, time, urllib.parse, urllib.request, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_to_head as H

QWEN_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_TE = "qwen3vl_8b_bf16.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
TURBO_LORA = "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"
LIGHTNING = "Qwen-Image-2.1-Lightning-pytorch_lora_weights.safetensors"
MOSS_UNET = "moss_image_sft_int8_convrot.safetensors"
MOSS_TE = "qwen3.5_9b_int8_convrot.safetensors"
MOSS_VAE = "flux2-vae_small.safetensors"
MOSS_CTRL = "moss-control.safetensors"


def run(wf, out_dir, tag, timeout=1200):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    res = H.post("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})
    if res.get("node_errors"):
        raise RuntimeError("node errors: " + json.dumps(res["node_errors"], ensure_ascii=False)[:400])
    pid = res["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            hist = H.get("/history/" + pid)
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
                    url = (H.SERVER + "/view?filename=" + urllib.parse.quote(meta["filename"])
                           + "&subfolder=" + urllib.parse.quote(meta.get("subfolder", ""))
                           + "&type=" + meta.get("type", "output"))
                    with urllib.request.urlopen(url, timeout=120) as r:
                        data = r.read()
                    dst = os.path.join(out_dir, tag + ".png")
                    with open(dst, "wb") as f:
                        f.write(data)
                    return {"tag": tag, "seconds": round(time.time() - t0, 1), "file": dst}
    raise RuntimeError(tag + ": timeout")


def qwen_turbo(steps, lora, seed, out_dir, tag,
               prompt="A cinematic photograph of a lighthouse at dusk"):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TE, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    }
    mref = ["1", 0]
    if lora:
        wf["9"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": mref, "lora_name": lora, "strength_model": 1.0}}
        mref = ["9", 0]
    wf["4"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": mref, "positive": ["5", 0],
                          "negative": ["5", 0], "latent_image": ["4", 0], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "tb", "images": ["7", 0]}}
    return run(wf, out_dir, tag)


def moss_control(out_dir, tag, seed=42, strength=1.0, end_percent=0.5,
                 color="ctrl_color.png", depth="ctrl_depth.png",
                 prompt="a photo of a person sitting at a desk in a sunlit room, photorealistic"):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MOSS_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MOSS_TE, "type": "moss_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MOSS_VAE}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 5.0}},
        "20": {"class_type": "LoraLoader",
               "inputs": {"lora_name": MOSS_CTRL, "strength_model": strength,
                          "strength_clip": 0.0, "model": ["4", 0], "clip": ["2", 0]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "22": {"class_type": "CLIPTextEncode",
               "inputs": {"text": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
                          "clip": ["2", 0]}},
        "23": {"class_type": "LoadImage", "inputs": {"image": color}},
        "24": {"class_type": "LoadImage", "inputs": {"image": depth}},
        "25": {"class_type": "MOSSImageControl",
               "inputs": {"positive": ["21", 0], "negative": ["22", 0], "vae": ["3", 0],
                          "strength": strength, "start_percent": 0.0, "end_percent": end_percent,
                          "color_map": ["23", 0], "depth_map": ["24", 0]}},
        "26": {"class_type": "SkipLayerGuidanceDiTSimple",
               "inputs": {"model": ["20", 0], "single_layers": "22", "double_layers": "",
                          "start_percent": 0.0, "end_percent": 1.0}},
        "27": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "28": {"class_type": "KSampler",
               "inputs": {"seed": seed, "steps": 50, "cfg": 3.0, "sampler_name": "euler",
                          "scheduler": "sgm_uniform", "model": ["26", 0],
                          "positive": ["25", 0], "negative": ["25", 1],
                          "latent_image": ["27", 0], "denoise": 1.0}},
        "29": {"class_type": "VAEDecode", "inputs": {"samples": ["28", 0], "vae": ["3", 0]}},
        "30": {"class_type": "SaveImage", "inputs": {"filename_prefix": "mctrl", "images": ["29", 0]}},
    }
    return run(wf, out_dir, tag)


def qwen_kv_pressure(out_dir, tag, seed, n_refs, resolution, cache):
    refs = {}
    for i in range(1, n_refs + 1):
        refs["image_" + str(i)] = ["4" if i == 1 else "4" + str(i), 0]
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TE, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "9": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": cache[0], "dtype": cache[1]}},
        "5": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": "Combine these references into one scene",
                         "negative_prompt": "", "resolution": resolution, "images": refs}},
        "6": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "model": ["9", 0], "positive": ["5", 0],
                         "negative": ["5", 1], "latent_image": ["5", 2], "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "kvp", "images": ["7", 0]}},
    }
    for i in range(1, n_refs + 1):
        nid = "4" if i == 1 else "4" + str(i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": "kvref_" + str(i) + ".png"}}
    return run(wf, out_dir, tag)


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "output/accel2")
    os.makedirs(out_dir, exist_ok=True)
    results = {"turbo": [], "moss_control": [], "kv_pressure": []}

    print("=== 1. Qwen official Turbo LoRA vs community Lightning LoRA ===", flush=True)
    for i, (label, steps, lora) in enumerate([
            ("base_25step", 25, None),
            ("official_turbo_8step", 8, TURBO_LORA),
            ("lightning_4step", 4, LIGHTNING)]):
        seed = 930000 + i
        try:
            t = qwen_turbo(steps, lora, seed, out_dir, "tb_" + label)
            t.update({"variant": label, "steps": steps, "lora": lora, "seed": seed})
            results["turbo"].append(t)
            print("  " + label + ": " + str(t["seconds"]) + "s", flush=True)
        except Exception as e:
            print("  " + label + ": ERROR " + str(e)[:260], flush=True)
            results["turbo"].append({"variant": label, "error": str(e)[:260]})

    print("=== 2. MOSS color/depth control adapter ===", flush=True)
    for i, (label, kw) in enumerate([
            ("control_end50", {"end_percent": 0.5}),
            ("control_end100", {"end_percent": 1.0}),
            ("control_strength0", {"end_percent": 0.5, "strength": 0.0})]):
        try:
            t = moss_control(out_dir, "mc_" + label, seed=940000 + i, **kw)
            t["variant"] = label
            results["moss_control"].append(t)
            print("  " + label + ": " + str(t["seconds"]) + "s", flush=True)
        except Exception as e:
            print("  " + label + ": ERROR " + str(e)[:300], flush=True)
            results["moss_control"].append({"variant": label, "error": str(e)[:300]})

    print("=== 3. KV cache under memory pressure (2048 + 8 refs) ===", flush=True)
    for i, cache in enumerate([("off", "default"), ("auto", "default"), ("cpu", "int8")]):
        label = "kv_" + cache[0] + "_" + cache[1]
        try:
            t = qwen_kv_pressure(out_dir, "kvp_" + label, 950000 + i,
                                 n_refs=8, resolution=2048, cache=cache)
            t["variant"] = label
            results["kv_pressure"].append(t)
            print("  " + label + ": " + str(t["seconds"]) + "s", flush=True)
        except Exception as e:
            print("  " + label + ": ERROR " + str(e)[:260], flush=True)
            results["kv_pressure"].append({"variant": label, "error": str(e)[:260]})

    with open(os.path.join(out_dir, "accel2_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()