"""
MOSS-Image 5B on ComfyUI v0.42.5 — API workflow builder

依官方 comfyui/README.md + USERGUIDE.md：
  - Load Diffusion Model / CLIP Loader(type=moss_image) / Load VAE
  - 50 steps, CFG 3, Shift 5, Euler + sgm_uniform, 負向分支跳過 layer 22 (uncond skip)
  - 推薦 1M~2M 像素；可外推至 4M (≈2K)

用法：
  python moss_workflow.py --out wf.json --model sft --width 2048 --height 2048 --seed 12345
"""
import argparse, json, uuid, os

MOSS_SFT = "moss_image_sft_int8_convrot.safetensors"
MOSS_BASE = "moss_image_base_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3.5_9b_int8_convrot.safetensors"
VAE = "flux2-vae_small.safetensors"

NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"


def build(model, width, height, prompt, seed, steps=50, cfg=3.0, shift=5.0,
          skip_layer=22, batch=1, lora=None, lora_strength=1.0):
    unet = MOSS_SFT if model == "sft" else MOSS_BASE
    wf = {
        # 1) load diffusion model
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        # 2) load text encoder (moss_image type)
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER, "type": "moss_image"}},
        # 3) load vae
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        # 4) shift 5 (官方預設；用 AuraFlow 因為原生模型 timestep multiplier 與 SD3 不同)
        "4": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": shift}},
        # 5) positive conditioning
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        # 6) negative conditioning (官方建議負向詞)
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["2", 0]}},
        # 7) Simple SLG：負向分支跳過 layer 22 (uncond skip)
        "7": {"class_type": "SkipLayerGuidanceDiTSimple",
              "inputs": {"model": ["4", 0], "single_layers": str(skip_layer),
                         "double_layers": "", "start_percent": 0.0, "end_percent": 1.0}},
        # 8) latent
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": batch}},
    }
    model_ref = ["7", 0]
    if lora:
        # 9) LoRA：KohakuMossTrainer 匯出的 ComfyUI 格式 adapter
        wf["9"] = {"class_type": "LoraLoader",
                   "inputs": {"lora_name": lora, "strength_model": lora_strength,
                              "strength_clip": 0.0, "model": model_ref,
                              "clip": ["2", 0]}}
        model_ref = ["9", 0]
    # 10) sampler
    wf["10"] = {"class_type": "KSampler",
                "inputs": {"model": model_ref, "seed": seed, "steps": steps, "cfg": cfg,
                           "sampler_name": "euler", "scheduler": "sgm_uniform",
                           "positive": ["5", 0], "negative": ["6", 0],
                           "latent_image": ["8", 0], "denoise": 1.0}}
    # 11) decode
    wf["11"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    # 12) save
    wf["12"] = {"class_type": "SaveImage",
                "inputs": {"filename_prefix": "moss_test", "images": ["11", 0]}}
    return wf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="moss_wf.json")
    ap.add_argument("--model", choices=["sft", "base"], default="sft")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--prompt", default=(
        "A cozy sunlit studio corner with a wooden desk, a small potted plant, "
        "warm afternoon light through a window, soft shadows, detailed textures, "
        "photorealistic, high detail"))
    ap.add_argument("--lora", default=None, help="LoRA filename in models/loras")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    a = ap.parse_args()
    wf = build(a.model, a.width, a.height, a.prompt, a.seed, a.steps, a.cfg,
               lora=a.lora, lora_strength=a.lora_strength)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"prompt": wf, "client_id": str(uuid.uuid4())}, f, ensure_ascii=False, indent=1)
    mp = a.width * a.height / 1_000_000
    print(f"wrote {a.out}  model={a.model}  {a.width}x{a.height} = {mp:.2f} MP")
