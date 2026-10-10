#!/usr/bin/env python3
"""
Baseline / comparison evaluation for the triage task.

Runs a model over data/test.jsonl and reports per-field accuracy, exact match,
JSON validity and latency. Run the SAME script before and after fine-tuning so
the two numbers are comparable — that comparison IS the deliverable.

    python3 evaluate.py                            # baseline: qwen3.5:9b
    python3 evaluate.py --limit 6                  # quick smoke test
    python3 evaluate.py --model qwen2.5:1.5b       # after fine-tuning
    python3 evaluate.py --out baseline.json        # save for later comparison
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import FIELDS, PREDICTION_SCHEMA, system_prompt  # noqa: E402

OLLAMA = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"
TIMEOUT = 300
MAX_ATTEMPTS = 3      # empty responses are intermittent, same as in synthesis


def call(model: str, question: str, system: str) -> str:
    prompt = f"{system}\n\n市民提問：{question}\n\n輸出："
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": PREDICTION_SCHEMA,        # single object, not an array
        "options": {
            "temperature": 0,     # evaluation must be deterministic
            # A label is ~40 tokens. Without a cap, a looping generation runs for
            # minutes (one item took 480s and burned two retries).
            "num_predict": 64,
            # prompt + label is ~430 tokens; the 4096 default KV cache pushed
            # VRAM to 96% of 8GB and caused stalls.
            "num_ctx": 2048,
        },
    }).encode()
    req = urllib.request.Request(OLLAMA, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())["response"]


def parse_prediction(raw: str) -> dict | None:
    """Return a complete label dict, or None if the output is unusable."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):                  # some models wrap in an array
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for field in FIELDS:
        value = data.get(field)
        if field == "escalate":
            if isinstance(value, bool):
                out[field] = value
            elif str(value).strip().lower() in ("true", "false"):
                out[field] = str(value).strip().lower() == "true"
            else:
                return None
        else:
            out[field] = str(value).strip() if value is not None else ""

    # NEVER use all(out.values()) here: escalate=False is a valid label but a
    # falsy value, so every "escalate": false prediction was scored as unusable.
    # That is what made the baseline read 16.7% while the model was correct.
    if not out["category"] or not out["urgency"]:
        return None
    return out


def load_test(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[!] {path} not found — run check_dataset.py first")
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        by_role = {m["role"]: m["content"] for m in json.loads(line)["messages"]}
        rows.append({"question": by_role["user"],
                     "gold": json.loads(by_role["assistant"])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--system", choices=("short", "full"), default="short",
                    help="'short' = the serving prompt; 'full' = the annotator's "
                         "rule set. A/B these to test whether the label/serving "
                         "prompt drift explains the accuracy gap.")
    args = ap.parse_args()

    system = system_prompt(args.system)
    rows = load_test(Path(args.data))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("[!] test set is empty")

    print(f"[i] model={args.model}  data={args.data}  n={len(rows)}  "
          f"system={args.system} ({len(system)} chars)\n")

    correct = {f: 0 for f in FIELDS}
    exact = json_ok = 0
    latencies: list[float] = []
    details = []

    for i, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        pred, error, raw, attempts = None, None, "", 0
        # Same intermittent-empty behaviour as synthesis: retry rather than
        # score an empty response as a wrong answer.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                raw = call(args.model, row["question"], system)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                continue
            error = None
            pred = parse_prediction(raw) if raw else None
            if pred:
                break
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        if pred:
            json_ok += 1
        hit = {f: bool(pred and pred[f] == row["gold"][f]) for f in FIELDS}
        for f in FIELDS:
            correct[f] += hit[f]
        all_ok = all(hit.values())
        exact += all_ok

        details.append({"question": row["question"], "gold": row["gold"],
                        "pred": pred, "hit": hit, "attempts": attempts,
                        "raw": raw[:300], "latency_ms": round(dt),
                        **({"error": error} if error else {})})

        # Never let a failure print as a bare ✗ again — show which field differs.
        note = ""
        if error:
            note = f"   ERROR {error}"
        elif not pred:
            note = f"   unusable raw ({len(raw)}B): {raw[:70]!r}"
        elif not all_ok:
            diffs = " | ".join(f"{f}: gold={row['gold'][f]!r}->pred={pred[f]!r}"
                               for f in FIELDS if not hit[f])
            note = f"   {diffs}"
        elif attempts > 1:
            note = f"   (retry x{attempts})"
        print(f"[{i:>3}/{len(rows)}] {'✓' if all_ok else '✗'} "
              f"{dt / 1000:5.1f}s  {row['question'][:40]}{note}")

    n = len(rows)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[min(n - 1, int(n * 0.95))]
    report = {
        "model": args.model, "n": n,
        "category_acc": round(correct["category"] / n, 4),
        "urgency_acc": round(correct["urgency"] / n, 4),
        "escalate_acc": round(correct["escalate"] / n, 4),
        "exact_match": round(exact / n, 4),
        "json_valid": round(json_ok / n, 4),
        "latency_p50_ms": round(p50),
        "latency_p95_ms": round(p95),
    }

    def pct(k: int) -> str:
        return f"{k}/{n} = {k / n:.1%}"

    print("\n=== 結果 ===")
    print(f"category_acc : {pct(correct['category'])}")
    print(f"urgency_acc  : {pct(correct['urgency'])}")
    print(f"escalate_acc : {pct(correct['escalate'])}")
    print(f"exact_match  : {pct(exact)}      <- 三欄全對（業務真正在意的）")
    print(f"json_valid   : {pct(json_ok)}")
    print(f"latency      : p50={p50:.0f}ms  p95={p95:.0f}ms")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"report": report, "details": details},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[i] 詳細結果 -> {args.out}")
    print("[next] 微調後跑同一腳本（--model 換成微調模型），兩份 report 相比")


if __name__ == "__main__":
    main()
