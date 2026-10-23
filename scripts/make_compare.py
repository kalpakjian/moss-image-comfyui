"""製作 MOSS-Image 實測對比圖

1) 解析度對比：1024 / 2048 / 2560 同區域 100% 裁切，展示細節保留
2) Base vs SFT 對比（KohakuMossTrainer README 風格）
3) Base vs Base+LoRA 對比

環境變數：
  MOSS_RES  取圖資料夾（預設 <repo>/output）
  MOSS_OUT  圖片輸出資料夾（預設 <repo>/output）
"""
import os
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MOSS_RES", os.path.abspath(os.path.join(_HERE, "..", "output")))
OUT = os.environ.get("MOSS_OUT", os.path.abspath(os.path.join(_HERE, "..", "output")))
RES = os.path.abspath(RES)
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
W = 1400

try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    fontb = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 34)
except Exception:
    font = fontb = ImageFont.load_default()


def crop_same(src, box_frac, out_size):
    """依比例裁切同一區域，放大到 out_size"""
    im = Image.open(src).convert("RGB")
    W0, H0 = im.size
    l, t, r, b = box_frac
    region = im.crop((int(W0 * l), int(H0 * t), int(W0 * r), int(H0 * b)))
    return region.resize((out_size, out_size), Image.LANCZOS)


def make_res_comparison():
    """1024 / 2048 / 2560 —— 取窗戶+桌面上方區域，100% 放大比較"""
    specs = [
        ("sft_1024_0.png", "1024×1024  (1.05 MP)"),
        ("sft_2048_0.png", "2048×2048  (4.19 MP)"),
        ("sft_2560_0.png", "2560×2560  (6.55 MP)"),
    ]
    box = (0.03, 0.05, 0.37, 0.39)   # 牆上相框區：文字/圖框/植物等高紋理細節
    tile = 440
    pad = 14
    cap_h = 62
    canvas = Image.new("RGB", (tile * 3 + pad * 4, tile + cap_h + pad * 2), (12, 16, 28))
    d = ImageDraw.Draw(canvas)
    for i, (fn, label) in enumerate(specs):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        c = crop_same(p, box, tile)
        x = pad + i * (tile + pad)
        canvas.paste(c, (x, pad))
        d.rectangle([x, pad, x + tile - 1, pad + tile - 1], outline=(70, 90, 130), width=2)
        d.text((x + 6, pad + tile + 12), label, fill=(225, 235, 255), font=font)
    out = os.path.join(OUT, "moss_resolution_compare.jpg")
    canvas.save(out, quality=93)
    print("saved", out, canvas.size)


def make_base_vs_sft():
    """Base（上）vs SFT（下），同 seed 同 prompt"""
    base = os.path.join(RES, "base_1024_0.png")
    sft = os.path.join(RES, "sft_1024_0.png")
    if not (os.path.exists(base) and os.path.exists(sft)):
        print("skip base_vs_sft")
        return
    tw = 660
    a = Image.open(base).convert("RGB").resize((tw, tw), Image.LANCZOS)
    b = Image.open(sft).convert("RGB").resize((tw, tw), Image.LANCZOS)
    cap_h = 56
    pad = 14
    canvas = Image.new("RGB", (tw + pad * 2, tw * 2 + cap_h * 2 + pad * 3), (12, 16, 28))
    d = ImageDraw.Draw(canvas)
    d.text((pad + 8, pad), "MOSS-Image-Base", fill=(255, 210, 120), font=fontb)
    canvas.paste(a, (pad, pad + cap_h))
    d.rectangle([pad, pad + cap_h, pad + tw - 1, pad + cap_h + tw - 1], outline=(70, 90, 130), width=2)
    y2 = pad + cap_h + tw + pad
    d.text((pad + 8, y2), "MOSS-Image-SFT", fill=(140, 220, 255), font=fontb)
    canvas.paste(b, (pad, y2 + cap_h))
    d.rectangle([pad, y2 + cap_h, pad + tw - 1, y2 + cap_h + tw - 1], outline=(70, 90, 130), width=2)
    out = os.path.join(OUT, "moss_base_vs_sft.jpg")
    canvas.save(out, quality=93)
    print("saved", out, canvas.size)


def make_lora_compare():
    """Base（上）vs Base+LoRA（下），同 seed 同 prompt"""
    import glob
    pairs = []
    for i in (0, 1):
        b = os.path.join(RES, f"lora_base_{i}_0.png")
        l = os.path.join(RES, f"lora_lora_{i}_0.png")
        if os.path.exists(b) and os.path.exists(l):
            pairs.append((b, l))

    tw = 560
    cap_h = 52
    pad = 16
    canvas = Image.new("RGB", (tw * 2 + pad * 3,
                              (tw + cap_h) * len(pairs) + cap_h + pad * (len(pairs) + 1)),
                       (12, 16, 28))
    d = ImageDraw.Draw(canvas)
    d.text((pad, pad), "MOSS-Image-Base", fill=(255, 205, 110), font=fontb)
    d.text((pad * 2 + tw, pad), "+ KohakuMossTrainer LoRA", fill=(120, 220, 255), font=fontb)
    y = pad + cap_h
    for b, l in pairs:
        for src, x in ((b, pad), (l, pad * 2 + tw)):
            im = Image.open(src).convert("RGB").resize((tw, tw), Image.LANCZOS)
            canvas.paste(im, (x, y))
            d.rectangle([x, y, x + tw - 1, y + tw - 1], outline=(70, 90, 130), width=2)
        y += tw + cap_h
    out = os.path.join(OUT, "moss_lora_compare.jpg")
    canvas.save(out, quality=93)
    print("saved", out, canvas.size)


if __name__ == "__main__":
    make_res_comparison()
    make_base_vs_sft()
    make_lora_compare()
