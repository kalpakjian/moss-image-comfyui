"""
驗證 ComfyUI 的「節點執行快取」是否會影響耗時量測

背景：先前 session 記錄到 MOSS-Image 1024×1024 只要 3.0 秒，
但重跑卻是 ~19 秒。ComfyUI 會依節點輸入雜湊做快取，
若整條工作流程（含 seed）與前一次完全相同，會直接回傳上次的結果。

本腳本：
  1. 用 seed=424242 跑一次（保證快取未命中）→ 基準
  2. 用完全相同的 workflow 再跑一次 → 驗證是否命中快取
  3. 換一個 seed 再跑 → 再次基準
"""
import json, os, sys, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_to_head as H  # noqa: E402


def timed_run(wf, out_dir, tag):
    t0 = time.time()
    res = H.post("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})
    pid = res["prompt_id"]
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            hist = H.get(f"/history/{pid}")
        except Exception:
            continue
        if pid in hist:
            e = hist[pid]
            st = e.get("status", {})
            msgs = st.get("messages", [])
            cached_nodes = [m[1].get("nodes") for m in msgs if m[0] == "execution_cached"]
            cached_nodes = [n for n in (cached_nodes[0] or []) if n] if cached_nodes else []
            t_start = t_ok = None
            for m in msgs:
                if m[0] == "execution_start":
                    t_start = m[1].get("timestamp")
                elif m[0] in ("execution_success", "execution_error"):
                    t_ok = m[1].get("timestamp")
            server_s = round((t_ok - t_start) / 1000, 2) if (t_start and t_ok) else None
            wall = round(time.time() - t0, 2)
            return {"tag": tag, "wall_s": wall, "server_s": server_s,
                    "cached_nodes": cached_nodes}
    return {"tag": tag, "wall_s": None, "server_s": None, "cached_nodes": None}


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "output/cache_test")
    os.makedirs(out_dir, exist_ok=True)
    W = H.build_moss("sft", 1024, 1024, "a red apple on a wooden table", 424242)

    print("=== 1. seed=424242 首次（快取必未命中）===", flush=True)
    r1 = timed_run(W, out_dir, "first_seed424242")
    print(json.dumps(r1, ensure_ascii=False), flush=True)

    print("=== 2. 完全相同的 workflow 再跑一次（測快取命中）===", flush=True)
    r2 = timed_run(W, out_dir, "second_same")
    print(json.dumps(r2, ensure_ascii=False), flush=True)

    print("=== 3. 換 seed=424243（快取必未命中）===", flush=True)
    W2 = H.build_moss("sft", 1024, 1024, "a red apple on a wooden table", 424243)
    r3 = timed_run(W2, out_dir, "third_seed424243")
    print(json.dumps(r3, ensure_ascii=False), flush=True)

    print("=== 4. 回到 seed=424242（此時已被快取住）===", flush=True)
    r4 = timed_run(W, out_dir, "fourth_seed424242_again")
    print(json.dumps(r4, ensure_ascii=False), flush=True)

    with open(os.path.join(out_dir, "cache_test.json"), "w", encoding="utf-8") as f:
        json.dump([r1, r2, r3, r4], f, ensure_ascii=False, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
