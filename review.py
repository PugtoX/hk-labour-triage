#!/usr/bin/env python3
"""
Human review helper — turn "looks OK" into a number with a pass/fail line.

    python review.py                       # print every row for judging
    python review.py --wrong 3,7,12        # after judging: report accuracy

Judging rule (see SPEC.md §5): for each row ask "is this question really in the
category it is labelled with?". Rows you judge wrong go into --wrong.

The pass line is CATEGORY_THRESHOLD below, fixed BEFORE looking at the data —
choosing it afterwards is how a bad dataset gets waved through.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import CATEGORIES  # noqa: E402  (shown alongside each row)

DEFAULT = ROOT / "data" / "synthetic.jsonl"

CATEGORY_THRESHOLD = 0.95     # label accuracy required for training
FIELD_THRESHOLD = 0.90        # urgency / escalate accuracy


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[!] {path} not found — run synthesize.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT))
    ap.add_argument("--wrong", default=None,
                    help="row numbers judged to have the wrong category. "
                         "Pass '-' to mean 'judged, zero wrong'.")
    ap.add_argument("--wrong-urgency", default=None,
                    help="row numbers with the wrong urgency label ('-' = zero wrong)")
    ap.add_argument("--wrong-escalate", default=None,
                    help="row numbers with the wrong escalate label ('-' = zero wrong)")
    args = ap.parse_args()

    rows = load(Path(args.file))
    n = len(rows)

    def parse_rows(spec: str) -> list[int]:
        if spec.strip() == "-":
            return []          # judged, and nothing was wrong
        nums = sorted({int(x) for x in spec.replace(" ", "").split(",") if x})
        bad = [i for i in nums if not 1 <= i <= n]
        if bad:
            sys.exit(f"[!] row numbers out of range: {bad}")
        return nums

    specs = [("category", args.wrong, CATEGORY_THRESHOLD),
             ("urgency", args.wrong_urgency, FIELD_THRESHOLD),
             ("escalate", args.wrong_escalate, FIELD_THRESHOLD)]

    # None = not judged. "" = judged with zero errors. Keeping those distinct is
    # what stops "two of three fields judged and passing" from printing ALL PASS.
    if all(spec is None for _, spec, _ in specs):
        print("審三輪，每輪只判一個欄位（混著判會把對的欄位也標成錯）：")
        print("  第 1 輪 category：這條提問真的屬於它標的類別嗎？")
        print("  第 2 輪 urgency：「我這個情況算不算／有無資格」= medium，不是 low")
        print("  第 3 輪 escalate：涉及具體金額／日期／已收通知 = true\n")
        for i, r in enumerate(rows, 1):
            print(f"[{i:>2}] {r['text']}")
            print(f"      category={r['category']}  urgency={r['urgency']}  "
                  f"escalate={r['escalate']}")
            print(f"      category 定義：{CATEGORIES[r['category']]}")
        print(f"\n{n} rows. 把每輪判錯的行號填進去（沒判的欄位會顯示 not judged）：")
        print("  python review.py --wrong 3,7")
        print("  python review.py --wrong-urgency 5,9")
        print("  python review.py --wrong-escalate 2,4")
        print(f"\nPass lines: category >= {CATEGORY_THRESHOLD:.0%}, "
              f"urgency/escalate >= {FIELD_THRESHOLD:.0%}")
        return

    print(f"rows judged: {n}\n")
    all_pass = True
    all_judged = True
    for field, spec, threshold in specs:
        if spec is None:
            print(f"{field:<9}: NOT JUDGED")
            all_judged = False
            continue
        bad = parse_rows(spec)
        acc = (n - len(bad)) / n
        ok = acc >= threshold
        all_pass = all_pass and ok
        print(f"{field:<9}: {n - len(bad)}/{n} = {acc:.1%}  (line {threshold:.0%})  "
              f"{'PASS' if ok else 'FAIL'}"
              + (f"   wrong: {bad}" if bad else ""))
    print()
    if all_pass and all_judged:
        print("ALL PASS — 這批數據可用於訓練")
    elif all_pass:
        print("INCOMPLETE — 有欄位未判，不可當作通過")
    else:
        print("NOT ALL PASS — 先修數據，不要拿去訓練")


if __name__ == "__main__":
    main()
