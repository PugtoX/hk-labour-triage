#!/usr/bin/env python3
"""
Synthesize triage training data with a local LLM (industrial practice:
generate with a big model, spot-check by hand — see SPEC.md).

    python synthesize.py                      # all categories, 20 each
    python synthesize.py --per-category 5     # quick trial run
    python synthesize.py --categories 假期與休假

Writes JSONL to data/categorical.jsonl. Re-running tops each category up to the
target instead of starting over, and never deletes other categories' rows.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import (CATEGORIES, CATEGORY_NAMES, ESCALATE_RULES,  # noqa: E402
                    URGENCIES, URGENCY_RULES, URGENCY_WARNING,
                    category_block_except)

OUT = ROOT / "data" / "synthetic.jsonl"

OLLAMA = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"
TIMEOUT = 600
MAX_ATTEMPTS = 3      # empty-array responses are intermittent, not prompt-driven
MAX_BATCH = 25        # one request for 69 items returns truncated JSON

# Constrain the RESPONSE SHAPE, not merely "valid JSON". With format="json" the
# model was free to return a single object instead of the array, and 13 of 16
# topics silently yielded 1 row instead of 20.
RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORY_NAMES},
            "urgency": {"type": "string", "enum": list(URGENCIES)},
            "escalate": {"type": "boolean"},
        },
        "required": ["text", "category", "urgency", "escalate"],
    },
}

PROMPT = """你是香港勞工諮詢熱線的數據標註員，正在為「分流模型」準備訓練數據。

## 目標類別

請生成 {n} 條市民的**口語提問**，每一條都必須**明確屬於**以下類別：

　【{category}】
　定義：{definition}

## 其他類別（供你判斷邊界）

{others}

你**只需要**生成「{category}」的問題，不必生成以上其他類別的問題。

## 輸出前自檢

逐條問自己：「這條提問是否明確屬於『{category}』？」
若某條偏向其他類別，**請改寫成明確屬於「{category}」的問題** ——
不要改標籤，也不要少寫一條。

## 必須遵守

必須輸出 **{n} 條**問題。**不可返回空數組，不可只返回 1 條。**
若一時想不到，就換個情境再寫一條，但數量必須達到 {n}。

## urgency 標註規則

{urgency_rules}

{urgency_warning}

## escalate 標註規則

{escalate_rules}

## 要求

1. 提問要像真人說話：可用粵語口語（「唔」「係咪」「邊度」「點算」）、繁體中文或簡體中文
2. 三種 urgency 都要出現，不要全部同一級
3. escalate 要有 true 也要有 false
4. 提問之間不要重複，涵蓋不同情境

## 輸出

只輸出 JSON 陣列，不要任何解釋或 markdown：

[{{"text": "老闆拖我三個月工資，我可以去邊度告佢？", "category": "{category}", "urgency": "high", "escalate": true}}]
"""


def call_ollama(model: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,                 # reasoning models would leave response empty
        "format": RESPONSE_SCHEMA,      # force an array of well-formed labels
        "options": {"temperature": 0.9},  # we want variety across the rows
    }).encode()
    req = urllib.request.Request(OLLAMA, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())["response"]


def parse_rows(raw: str, requested: str) -> list[dict]:
    """Accept a bare array or an object wrapping one; validate every field.

    The model's own `category` is kept (it is not forced to `requested`): when
    the generator judges a question to belong elsewhere, honouring that is the
    self-check working. Forcing the field is exactly what produced 30% wrong
    labels in the previous 16-topic version.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [!] {requested}: not JSON ({e})", file=sys.stderr)
        return []

    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
        else:
            print(f"  [!] {requested}: single object, not an array — check "
                  f"RESPONSE_SCHEMA", file=sys.stderr)
            data = [data]
    if not isinstance(data, list):
        print(f"  [!] {requested}: expected a list, got {type(data).__name__}",
              file=sys.stderr)
        return []

    rows, dropped, elsewhere = [], Counter(), 0
    for item in data:
        if not isinstance(item, dict):
            dropped["not an object"] += 1
            continue
        text = str(item.get("text", "")).strip()
        cat = str(item.get("category", "")).strip()
        urg = str(item.get("urgency", "")).strip().lower()
        esc = item.get("escalate")

        if len(text) < 6:
            dropped["text too short"] += 1
            continue
        if cat not in CATEGORY_NAMES:
            dropped[f"bad category {cat!r}"] += 1
            continue
        if urg not in URGENCIES:
            dropped[f"bad urgency {urg!r}"] += 1
            continue
        if not isinstance(esc, bool):
            if str(esc).strip().lower() in ("true", "false"):
                esc = str(esc).strip().lower() == "true"
            else:
                dropped[f"bad escalate {esc!r}"] += 1
                continue
        if cat != requested:
            elsewhere += 1
        rows.append({"text": text, "category": cat, "urgency": urg, "escalate": esc})

    if dropped:
        detail = ", ".join(f"{k}×{v}" for k, v in dropped.most_common(4))
        print(f"  [i] dropped {sum(dropped.values())}: {detail}")
    if elsewhere:
        print(f"  [i] {elsewhere} row(s) the model filed under another category "
              f"(self-check working; kept with its own label)")
    return rows


def load_existing() -> dict[str, list[dict]]:
    """category -> rows, so an interrupted run can resume."""
    by_cat: dict[str, list[dict]] = {}
    if not OUT.exists():
        return by_cat
    for line in OUT.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_cat.setdefault(row.get("category", "?"), []).append(row)
    return by_cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--per-category", type=int, default=20,
                    help="target rows per category (not a per-request batch)")
    ap.add_argument("--categories", nargs="*", default=CATEGORY_NAMES)
    args = ap.parse_args()

    unknown = [c for c in args.categories if c not in CATEGORIES]
    if unknown:
        sys.exit(f"[!] unknown categories: {unknown}\n    valid: {CATEGORY_NAMES}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing()
    all_rows: list[dict] = [r for rows in existing.values() for r in rows]
    # NOTE: every existing row is kept, including categories not in --categories.
    # The file is rewritten from all_rows at the end, so filtering here would
    # delete the other categories' data.

    print(f"[i] model={args.model}  per_category={args.per_category}  "
          f"categories={len(args.categories)}")
    print(f"[i] resuming with {len(all_rows)} rows already generated\n")

    seen = {r["text"] for r in all_rows}
    target = args.per_category

    for i, category in enumerate(args.categories, 1):
        have = sum(1 for r in all_rows if r["category"] == category)
        if have >= target:
            print(f"[{i}/{len(args.categories)}] {category}: already {have}, skip")
            continue

        # Ask for at most MAX_BATCH at a time. A single request for 69 items
        # comes back as truncated JSON ("Unterminated string"), which then gets
        # mis-reported as an empty response.
        need = target - have
        while need > 0:
            n = min(need, MAX_BATCH)
            capped = " (capped)" if n < need else ""
            print(f"[{i}/{len(args.categories)}] {category}: have {have}, "
                  f"requesting {n} more{capped} ...", flush=True)
            prompt = PROMPT.format(
                n=n,
                category=category,
                definition=CATEGORIES[category],
                others=category_block_except(category),
                urgency_rules=URGENCY_RULES,
                urgency_warning=URGENCY_WARNING,
                escalate_rules=ESCALATE_RULES,
            )
            # The model intermittently returns an empty array for the SAME prompt
            # (probed: all six prompt/format variants succeeded on a retry of a
            # category that had just failed). So retry rather than blame the prompt.
            rows: list[dict] = []
            raw = ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    raw = call_ollama(args.model, prompt)
                except urllib.error.URLError as e:
                    sys.exit(f"[!] cannot reach Ollama ({e.reason}). Is it running?  ollama list")
                except Exception as e:
                    print(f"  [!] {category}: {type(e).__name__}: {e}", file=sys.stderr)
                    break
                rows = parse_rows(raw, category)
                if rows:
                    break
                if attempt < MAX_ATTEMPTS:
                    # Distinguish the two failure modes: an empty reply and a
                    # truncated/unparsable one need different fixes.
                    stripped = raw.strip()
                    kind = ("empty response" if stripped in ("", "[]")
                            else f"unparsable JSON ({len(raw)} chars)")
                    print(f"  [i] attempt {attempt}/{MAX_ATTEMPTS}: {kind}, "
                          f"retrying", file=sys.stderr)
            if not rows:
                print(f"  [!] {category}: no usable rows after {MAX_ATTEMPTS} "
                      f"attempts. last raw ({len(raw)} chars): {raw[:200]!r}",
                      file=sys.stderr)
                break
            # Dedup must mark `seen` as each row is accepted — collecting first
            # and batch-adding later lets duplicates INSIDE one response through
            # (three identical questions reached the test set this way).
            fresh = []
            for r in rows:
                if r["text"] in seen:
                    continue
                seen.add(r["text"])
                fresh.append(r)
            all_rows.extend(fresh)
            print(f"  -> {len(rows)} valid, {len(fresh)} new")
            if not fresh:
                print(f"  [!] {category}: no new rows, stopping this category "
                      f"({have}/{target})", file=sys.stderr)
                break
            have += len(fresh)
            need -= len(fresh)

    order = {c: i for i, c in enumerate(CATEGORY_NAMES)}
    all_rows.sort(key=lambda r: (order.get(r["category"], 99), r["urgency"]))
    with OUT.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat = Counter(r["category"] for r in all_rows)
    print(f"\n[OK] {len(all_rows)} rows -> {OUT}")
    print(f"     per category: {dict(by_cat)}")
    print(f"     urgency:  {dict(Counter(r['urgency'] for r in all_rows))}")
    print(f"     escalate: {dict(Counter(r['escalate'] for r in all_rows))}")
    print("\n[next] python check_dataset.py")


if __name__ == "__main__":
    main()
