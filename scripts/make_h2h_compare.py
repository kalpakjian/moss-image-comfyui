"""
MOSS-Image 5B vs Qwen-Image 2.1 — 對比圖產生器

輸出三種比較圖：
  1) 每個提示詞的並排對照（左 MOSS / 右 Qwen）
  2) 同一提示詞、不同解析度的橫向比較
  3) 數據總表（耗時 / 峰值 VRAM）
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def font(size, bold=False):
    try:
        name = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def side_by_side(a_path, b_path, a_label, b_label, caption, out_path, cell=760):
    """左右並排 + 上方標籤 + 下方說明"""
    ia = Image.open(a_path).convert("RGB")
    ib = Image.open(b_path).convert("RGB")
    f_lab = font(26, True)
    f_cap = font(21)
    f_sub = font(18)

    def fit(im):
        w, h = im.size
        s = cell / max(w, h)
        return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)

    ia, ib = fit(ia), fit(ib)
    pad, lab_h, cap_h = 18, 44, 70
    W = pad * 3 + ia.width + ib.width
    H = lab_h + max(ia.height, ib.height) + cap_h
    canvas = Image.new("RGB", (W, H), "#0e1116")
    d = ImageDraw.Draw(canvas)

    x = pad
    for im, lab in ((ia, a_label), (ib, b_label)):
        d.text((x, 10), lab, font=f_lab, fill="#e8edf2")
        canvas.paste(im, (x, lab_h))
        d.rectangle([x, lab_h, x + im.width - 1, lab_h + im.height - 1],
                    outline="#2b3540", width=2)
        x += im.width + pad

    # 說明（自動換行）
    words, line, lines = caption.split(" "), "", []
    for w_ in words:
        if d.textlength(line + w_, font=f_cap) > W - pad * 2:
            lines.append(line)
            line = w_ + " "
        else:
            line += w_ + " "
    lines.append(line)
    yy = lab_h + max(ia.height, ib.height) + 12
    for ln in lines[:3]:
        d.text((pad, yy), ln.strip(), font=f_cap, fill="#c7d0d9")
        yy += 26
    d.text((pad, yy + 2), "MOSS-Image 5B (int8)  vs  Qwen-Image 2.1 (int8)  —  RTX 4090, ComfyUI 0.42.5",
           font=f_sub, fill="#6f7d8a")
    canvas.save(out_path, quality=92)
    return canvas.size


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "h2h")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "output", "h2h_compare")
    os.makedirs(out, exist_ok=True)
    res_file = os.path.join(src, "results.json")
    results = json.load(open(res_file, encoding="utf-8")) if os.path.exists(res_file) else []
    ok = {r["tag"]: r for r in results if "error" not in r}

    labels = {"photo": "寫實室內場景", "cntext": "中文文字渲染海報", "entext": "英文招牌文字"}
    made = []
    for pid, pname in labels.items():
        for res in ("1024", "2048"):
            ta = f"{pid}_{res}_moss"
            tb = f"{pid}_{res}_qwen"
            if ta not in ok or tb not in ok:
                continue
            pa = os.path.join(src, ta + ".png")
            pb = os.path.join(src, tb + ".png")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            cap = (f"{pname} @{res}×{res}（{ok[ta]['mp']} MP）　"
                   f"MOSS {ok[ta]['seconds']}s / {round(ok[ta]['peak_vram_mib']/1024,1)} GiB　"
                   f"Qwen {ok[tb]['seconds']}s / {round(ok[tb]['peak_vram_mib']/1024,1)} GiB")
            dst = os.path.join(out, f"cmp_{pid}_{res}.jpg")
            size = side_by_side(pa, pb, "MOSS-Image 5B (SFT)", "Qwen-Image 2.1", cap, dst)
            made.append(dst)
            print("wrote", dst, size)

    # 數據總表圖
    if results:
        W, rowh = 1020, 42
        head = 78
        H = head + rowh * (len(results) + 1)
        c = Image.new("RGB", (W, H), "#ffffff")
        d = ImageDraw.Draw(c)
        f_h = font(26, True)
        f_r = font(20)
        d.text((24, 20), "MOSS-Image 5B vs Qwen-Image 2.1 — 同機實測", font=f_h, fill="#111111")
        cols = [(24, "提示詞", 200), (230, "模型", 90), (330, "解析度", 110), (450, "像素量", 90),
                (550, "耗時 (s)", 110), (670, "峰值 VRAM (GiB)", 180), (860, "冷/熱", 60),
                (930, "結果", 60)]
        for x, t, _w in cols:
            d.text((x, 52), t, font=f_r, fill="#444444")
        d.line([(24, 74), (W - 24, 74)], fill="#cccccc", width=2)
        yy = head
        for r in results:
            pname = labels.get(r.get("prompt_id", ""), r.get("prompt_id", ""))
            vals = [pname, r.get("model", ""), r.get("res", ""), f"{r.get('mp','-')} MP",
                    str(r.get("seconds", "-")), f"{round(r.get('peak_vram_mib',0)/1024,2)}",
                    "cold" if r.get("cold") else "warm",
                    "OK" if "error" not in r else "ERROR"]
            for (x, _t, _w), v in zip(cols, vals):
                d.text((x, yy + 10), v, font=f_r, fill="#111111")
            yy += rowh
        d.text((24, yy + 6), "同一台 RTX 4090 24GB / ComfyUI 0.42.5 / torch 2.16.0.dev+cu132，"
                             "每個模型都使用其官方推薦取樣參數", font=font(17), fill="#666666")
        dst = os.path.join(out, "h2h_table.jpg")
        c.save(dst, quality=93)
        print("wrote", dst, c.size)
        made.append(dst)


if __name__ == "__main__":
    main()
