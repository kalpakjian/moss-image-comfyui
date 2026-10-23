"""
驗證 Qwen-Image 2.1 的三項加速／增強功能（本機實測）

  1. QwenImage21Cache：KV cache 的 device/dtype 對編輯工作流程的影響
     （節點 tooltip 說 device=off 會「每步重算 prefix」→ 是排除 cache 影響的控制組）
  2. Lightning 4 步 LoRA：官方社群蒸餾 LoRA，4 步出圖
  3. Prompt 自動改寫（refine_prompt）：需額外 PE text encoder 8.82 GiB

用法：python verify_qwen_accel.py [輸出資料夾]
"""
import json, os, shutil, sys, time, urllib.parse, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_to_head as H  # noqa: E402

UNET = "qwen_image_2.1_int8_convrot.safetensors"
TE = "qwen3vl_8b_bf16.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
LIGHTNING = "Qwen-Image-2.1-Lightning-pytorch_lora_weights.safetensors"
PE_MODEL = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"


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
        raise RuntimeError("node errors: " + json.dumps(res["node_errors"], ensure_ascii=False)[:400])
    pid = res["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            hist = H.get(f"/history/{pid}")
        except Exception:
            continue
        if pid in hist:
            e = hist[pid]
            st = e.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError("exec error: " + json.dumps(st, ensure_ascii=False)[:400])
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


def edit_workflow(cache=None, steps=25, seed=42,
                  ref="qwen_ref_a.png",
                  instruction=("Keep the room, the window, the plant, the lamp and the bed "
                               "in <image1> exactly as they are. Only replace the wooden desk "
                               "in <image1> with a dark walnut desk.")):
    """Qwen 圖像編輯工作流程；cache = (device, dtype) 或 None（不放 cache 節點）"""
    wf = base_nodes()
    model_ref = ["1", 0]
    if cache:
        device, dtype = cache
        wf["9"] = {"class_type": "QwenImage21Cache",
                   "inputs": {"model": model_ref, "device": device, "dtype": dtype}}
        model_ref = ["9", 0]
    wf["4"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
    wf["5"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0], "prompt": instruction,
                          "negative_prompt": "", "resolution": 1024,
                          "images": {"image_1": ["4", 0]}}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": model_ref,
                          "positive": ["5", 0], "negative": ["5", 1],
                          "latent_image": ["5", 2], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "kv", "images": ["7", 0]}}
    return wf


def lightning_workflow(steps=4, use_lora=True, seed=42, prompt="A cinematic photograph of a lighthouse at dusk"):
    """Lightning 4 步 LoRA 工作流程"""
    wf = base_nodes()
    model_ref = ["1", 0]
    if use_lora:
        wf["9"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_ref, "lora_name": LIGHTNING,
                              "strength_model": 1.0}}
        model_ref = ["9", 0]
    wf["4"] = {"class_type": "EmptyLatentImage",
               "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    wf["5"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["2", 0]}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": model_ref,
                          "positive": ["5", 0], "negative": ["5", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "lightning", "images": ["7", 0]}}
    return wf

def pe_workflow(out_dir, raw_prompt, tag, te_type="moss_image", thinking=False,
                use_default_template=True, max_length=1024):
    """Prompt 自動改寫：PE text encoder → TextGenerate（display name: Generate Text）
    → 產生的改寫文字 → TextEncodeQwenImage21

    PE 模型是 Qwen3.5-9B 的微調，ComfyUI 的 CLIPType.MOSS_IMAGE 分支
    註解寫的是「stock Qwen3.5-9B, last hidden state, open think block」，
    對 qwen3.5_9b 系最合適。
    """
    wf = base_nodes()
    wf["10"] = {"class_type": "CLIPLoader",
                "inputs": {"clip_name": PE_MODEL, "type": te_type, "device": "default"}}
    wf["11"] = {"class_type": "TextGenerate",
                "inputs": {"clip": ["10", 0], "prompt": raw_prompt,
                           "max_length": max_length,
                           # sampling_mode 是 dynamic combo：
                           # API 要傳「選項 key 字串」，子參數要帶 "sampling_mode." 前綴
                           "sampling_mode": "on",
                           "sampling_mode.temperature": 0.7,
                           "sampling_mode.top_k": 64,
                           "sampling_mode.top_p": 0.95,
                           "sampling_mode.min_p": 0.05,
                           "sampling_mode.repetition_penalty": 1.05,
                           "sampling_mode.seed": 42,
                           "thinking": thinking,
                           "use_default_template": use_default_template}}
    wf["12"] = {"class_type": "TextEncodeQwenImage21",
                "inputs": {"clip": ["2", 0], "prompt": ["11", 0],
                           "negative_prompt": "", "resolution": 1024, "images": {}}}
    # PreviewAny 是 output_node，用來把改寫後的文字抓出來存檔
    wf["13"] = {"class_type": "PreviewAny", "inputs": {"source": ["11", 0]}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": ["1", 0],
                          "positive": ["12", 0], "negative": ["12", 1],
                          "latent_image": ["12", 2], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "pe", "images": ["7", 0]}}
    return wf


def pe_baseline_workflow(out_dir, raw_prompt, tag):
    """同一段 prompt 不做改寫，直接生成（控制組）"""
    wf = base_nodes()
    wf["4"] = {"class_type": "EmptyLatentImage",
               "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    wf["5"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": raw_prompt, "clip": ["2", 0]}}
    wf["6"] = {"class_type": "KSampler",
               "inputs": {"seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "model": ["1", 0],
                          "positive": ["5", 0], "negative": ["5", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    wf["8"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "pe_base", "images": ["7", 0]}}
    return wf

def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "output/qwen_accel")
    os.makedirs(out_dir, exist_ok=True)
    results = {"kv_cache": [], "lightning": [], "pe": [], "pe_text": []}

    # ComfyUI 的 KSampler 只在「自己的 seed 改變」時才重新取樣；
    # 同一 seed 連跑第二次會命中節點快取直接回傳舊結果（實測 0.3 秒）。
    # 因此每個量測都用唯一 seed，並先跑一次 warmup 把模型載進記憶體。
    print("=== 0. warmup（載入模型，不計時）===", flush=True)
    run(edit_workflow(cache=("auto", "default"), seed=999001), out_dir, "warmup")

    pe_only = "--pe-only" in sys.argv
    if not pe_only:
        print("=== 1. KV cache：編輯工作流程，五種設定 ===", flush=True)
        variants = [("no_cache", None), ("off_default", ("off", "default")),
                    ("auto_default", ("auto", "default")), ("cpu_int8", ("cpu", "int8")),
                    ("gpu_int4", ("gpu", "int4"))]
        for i, (label, cache) in enumerate(variants):
            seed = 910000 + i
            try:
                t = run(edit_workflow(cache=cache, seed=seed), out_dir, f"kv_{label}")
                t.update({"variant": label, "seed": seed})
                results["kv_cache"].append(t)
                print(f"  {label}: {t['seconds']}s (seed {seed})", flush=True)
            except Exception as e:
                print(f"  {label}: ERROR {e}", flush=True)
                results["kv_cache"].append({"variant": label, "error": str(e)})

        print("=== 2. Lightning 4 步 LoRA ===", flush=True)
        for i, (label, steps, use_lora) in enumerate([
                ("base_25step", 25, False), ("lightning_4step", 4, True),
                ("lightning_4step_noloRA", 4, False)]):
            seed = 920000 + i
            try:
                t = run(lightning_workflow(steps=steps, use_lora=use_lora, seed=seed),
                        out_dir, f"lg_{label}")
                t.update({"variant": label, "steps": steps, "lora": use_lora, "seed": seed})
                results["lightning"].append(t)
                print(f"  {label}: {t['seconds']}s (seed {seed})", flush=True)
            except Exception as e:
                print(f"  {label}: ERROR {e}", flush=True)
                results["lightning"].append({"variant": label, "error": str(e)})

    pe_path = os.path.join(r"D:\AI\ComfyUI\models\text_encoders", PE_MODEL)
    if os.path.exists(pe_path):
        print("=== 3. Prompt 自動改寫（PE model）===", flush=True)
        raw_prompt = "a cafe poster"
        for label, te_type, thinking in [("pe_moss_nothink", "moss_image", False),
                                         ("pe_qwen_nothink", "qwen_image", False),
                                         ("pe_moss_think", "moss_image", True)]:
            try:
                t = run(pe_workflow(out_dir, raw_prompt, label, te_type=te_type,
                                    thinking=thinking), out_dir, f"pe_{label}")
                t["variant"] = label
                results["pe"].append(t)
                print(f"  {label}: {t['seconds']}s", flush=True)
            except Exception as e:
                print(f"  {label}: ERROR {str(e)[:280]}", flush=True)
                results["pe"].append({"variant": label, "error": str(e)[:280]})
        try:
            t = run(pe_baseline_workflow(out_dir, raw_prompt, "pe_baseline"),
                    out_dir, "pe_baseline")
            t["variant"] = "baseline_no_pe"
            results["pe"].append(t)
            print(f"  baseline_no_pe: {t['seconds']}s", flush=True)
        except Exception as e:
            print(f"  baseline: ERROR {e}", flush=True)
    else:
        print("=== 3. Prompt 自動改寫：略過（PE model 未安裝）===", flush=True)

    with open(os.path.join(out_dir, "accel_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
