"""MOSS-Image 實測 runner：送出工作流程 + 輪詢 + 取圖 + 記錄耗時"""
import argparse, json, os, sys, time, urllib.request, urllib.parse, uuid

SERVER = os.environ.get("MOSS_SERVER", "http://127.0.0.1:8188")
OUT = os.environ.get(
    "MOSS_OUT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")),
)
os.makedirs(OUT, exist_ok=True)


def get(path, timeout=15):
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_image(meta):
    qs = urllib.parse.urlencode({"filename": meta["filename"],
                                 "subfolder": meta.get("subfolder", ""),
                                 "type": meta.get("type", "output")})
    with urllib.request.urlopen(f"{SERVER}/view?{qs}") as r:
        return r.read()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True, help="workflow json")
    ap.add_argument("--tag", required=True, help="output tag")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    wf = json.load(open(a.wf, encoding="utf-8"))
    payload = {"prompt": wf["prompt"], "client_id": str(uuid.uuid4())}

    t0 = time.time()
    res = post("/prompt", payload)
    pid = res["prompt_id"]
    if res.get("node_errors"):
        print(f"NODE ERRORS: {json.dumps(res['node_errors'], ensure_ascii=False)[:800]}",
              flush=True)
        sys.exit(1)
    print(f"submitted {pid}  tag={a.tag}", flush=True)

    elapsed = None
    while time.time() - t0 < a.timeout:
        time.sleep(3)
        try:
            h = get(f"/history/{pid}")
        except Exception:
            continue
        if pid in h:
            entry = h[pid]
            elapsed = time.time() - t0
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print(f"ERROR: {json.dumps(status, ensure_ascii=False)[:800]}", flush=True)
                sys.exit(1)
            imgs = []
            for node_out in entry.get("outputs", {}).values():
                imgs += node_out.get("images", [])
            if not imgs:
                print("no images produced", flush=True)
                sys.exit(1)
            for i, meta in enumerate(imgs):
                raw = fetch_image(meta)
                p = os.path.join(OUT, f"{a.tag}_{i}.png")
                with open(p, "wb") as f:
                    f.write(raw)
                w = meta.get("metadata", {}).get("width")
                h_ = meta.get("metadata", {}).get("height")
                mp = (w * h_ / 1_000_000) if w and h_ else 0
                print(f"OK {p}  {w}x{h_} = {mp:.2f} MP  {len(raw)/1024/1024:.2f} MB  "
                      f"{elapsed:.1f}s", flush=True)
            break
    else:
        print("TIMEOUT", flush=True)
        sys.exit(1)

    with open(os.path.join(OUT, f"{a.tag}_result.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": a.tag, "prompt_id": pid, "elapsed_sec": round(elapsed, 1),
                   "outputs": imgs}, f, ensure_ascii=False, indent=2)
    print(f"DONE {a.tag} in {elapsed:.1f}s", flush=True)
