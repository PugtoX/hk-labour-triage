#!/usr/bin/env python3
"""
Validate the synthesized data and split it into train/val/test.

    python check_dataset.py
    python check_dataset.py --review 40      # size of the human spot-check file

Writes data/{train,val,test}.jsonl in chat format, plus data/review_sample.jsonl
for the mandatory human spot-check (SPEC.md §4). The split is stratified by topic
and seeded, so it is reproducible.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import CATEGORY_NAMES, URGENCIES, to_messages  # noqa: E402

DATA = ROOT / "data"
SRC = DATA / "synthetic.jsonl"


def load() -> list[dict]:
    if not SRC.exists():
        sys.exit(f"[!] {SRC} not found — run synthesize.py first")
    rows = []
    for i, line in enumerate(SRC.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"[!] {SRC}:{i} invalid JSON: {e}")
    if not rows:
        sys.exit(f"[!] {SRC} is empty")
    # Defensive dedup. synthesize.py used to collect new rows before marking
    # them seen, so duplicates INSIDE a single response slipped through and
    # reached the test set. Fixed at the source; this repairs older files.
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        unique.append(r)
    if len(unique) != len(rows):
        print(f"[i] dropped {len(rows) - len(unique)} duplicate text(s) "
              f"({len(rows)} -> {len(unique)})")
    return unique


def report(rows: list[dict]) -> None:
    print(f"[i] rows: {len(rows)}")
    per_cat = Counter(r["category"] for r in rows)
    thin = [c for c in CATEGORY_NAMES if per_cat.get(c, 0) < 10]
    print(f"    categories covered: {len(per_cat)}/{len(CATEGORY_NAMES)}")
    if thin:
        print(f"    [!] thin categories (<10 rows): {', '.join(thin)}")
    print(f"    urgency:  {dict(Counter(r['urgency'] for r in rows))}")
    print(f"    escalate: {dict(Counter(r['escalate'] for r in rows))}")

    lens = sorted(len(r["text"]) for r in rows)
    print(f"    text length: min={lens[0]} median={lens[len(lens) // 2]} max={lens[-1]}")

    # a text must not carry two different labels — that would be label noise
    seen: dict[str, dict] = {}
    conflicts = 0
    for r in rows:
        prev = seen.get(r["text"])
        if prev and (prev["category"], prev["urgency"], prev["escalate"]) != \
                    (r["category"], r["urgency"], r["escalate"]):
            conflicts += 1
        seen[r["text"]] = r
    dups = len(rows) - len(seen)
    print(f"    duplicate texts: {dups}   conflicting labels: {conflicts}")
    if conflicts:
        print("    [!] same text with different labels — check the synthesis prompt")


def balance(rows: list[dict], cap: int, seed: int = 0):
    """Down-sample categories above `cap`. `cap <= 0` means no cap.

    A classifier trained on 34 工資與扣薪 rows vs 5 假期與休假 rows learns to
    over-predict the majority. Capping (rather than oversampling the small
    categories) keeps every row a real synthesis output and is the simplest
    honest fix.

    Default is NO cap: a cap silently negates a deliberate data expansion.
    (Observed: 297 balanced rows were trimmed back to 120 because this defaulted
    to 20, so a run that "added data" trained on the exact same 84 rows.)
    """
    if cap <= 0:
        return rows, {}
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    kept, trimmed = [], {}
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        if len(items) > cap:
            rng.shuffle(items)
            trimmed[cat] = len(items) - cap
            items = items[:cap]
        kept.extend(items)
    return kept, trimmed


def allocate(items: list[dict], val_frac: float, test_frac: float):
    """Smallest category groups still yield one example for val and test."""
    n = len(items)
    if n < 3:
        return items, [], []
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    while n_val + n_test >= n:                 # never starve train
        if n_val > 1 and n_val >= n_test:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return (items[n_val + n_test:],          # train
            items[:n_val],                   # val
            items[n_val:n_val + n_test])     # test


def split(rows: list[dict], seed: int = 42):
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    train, val, test = [], [], []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        tr, va, te = allocate(items, 0.15, 0.15)
        train += tr
        val += va
        test += te
    for part in (train, val, test):
        part.sort(key=lambda r: (r["category"], r["text"]))
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=int, default=32,
                    help="rows to sample for the human spot-check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-category", type=int, default=0,
                    help="cap rows per category; 0 (default) = no cap")
    args = ap.parse_args()

    rows = load()
    report(rows)

    rows, trimmed = balance(rows, args.max_per_category, args.seed)
    if trimmed:
        detail = ", ".join(f"{c} -{n}" for c, n in sorted(trimmed.items()))
        print(f"[i] balanced to <= {args.max_per_category}/category: dropped "
              f"{sum(trimmed.values())} rows ({detail})")

    train, val, test = split(rows, args.seed)
    if not train or not val or not test:
        sys.exit("[!] a split came out empty — need more rows per category")

    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "val.jsonl", val)
    write_jsonl(DATA / "test.jsonl", test)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.review, len(rows)))
    with (DATA / "review_sample.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(sample, 1):
            r2 = dict(r, _review_id=i)
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

    print(f"\n[i] split: train={len(train)} val={len(val)} test={len(test)}")
    print(f"[i] wrote data/train.jsonl, val.jsonl, test.jsonl")
    print(f"[i] spot-check {len(sample)} rows -> data/review_sample.jsonl")
    print("\n[!] BEFORE TRAINING: read review_sample.jsonl and fix wrong labels.")
    print("    Synthetic data always contains some — SPEC.md requires >=10% review.")


if __name__ == "__main__":
    main()
