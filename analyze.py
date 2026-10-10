#!/usr/bin/env python3
"""
Compare per-item eval dumps produced by `train.py --out` / `evaluate.py --out`.

    python analyze.py baseline_9b.json ft_17b.json
    python analyze.py ft_17b.json

Reports what a single aggregate accuracy hides:

  1. per-class recall AND precision for urgency, with Wilson 95% CIs — a model
     that shoves everything into "medium" scores respectably overall while
     swallowing every urgent case
  2. a cost-weighted score (missing a high costs more than other errors)
  3. paired McNemar between two dumps — on n=46 an accuracy gap is usually NOT
     significant, and that has to be visible before anyone quotes the gap
  4. the gold `escalate x urgency` table — five of six cells populated means the
     two fields are largely the same signal labelled twice
  5. rows where both models agree but gold differs — candidate label errors
"""
import json
import sys
from collections import Counter
from math import comb

THREE = ("high", "medium", "low")

HIGH_MISS_COST = 3          # missing an urgent item costs 3x a normal mistake
COST_SWEEP = (2, 3, 5)      # sensitivity: the ranking must survive all of these


def load(path: str) -> list[dict]:
    # utf-8-sig: tolerate a BOM if the file was touched by an editor.
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)["details"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. At n=46 this is wide — which is the point."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return centre - half, centre + half


def mcnemar(hits_a: list[bool], hits_b: list[bool]) -> tuple[int, int, float]:
    """Paired comparison. Only the DISCORDANT pairs carry information; a 7:0
    split is decisive, a 9:7 split is not."""
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, p)


def cost_of(details: list[dict], high_cost: int = HIGH_MISS_COST) -> int:
    """Weighted error count: gold=high predicted as anything else costs
    high_cost, every other mistake costs 1."""
    total = 0
    for d in details:
        pred = d.get("pred") or {}
        if d["gold"]["urgency"] == "high" and pred.get("urgency") != "high":
            total += high_cost
        elif not d["hit"]["urgency"]:
            total += 1
    return total


def exact_hits(details: list[dict]) -> list[bool]:
    return [all(d["hit"].values()) for d in details]


def report(name: str, details: list[dict]) -> dict:
    n = len(details)
    print(f"\n=== {name}  (n={n}) ===")
    # An unusable prediction is a measurement failure, not a wrong answer. Say so
    # loudly — otherwise every class reads as 0% and a broken serving path looks
    # like a catastrophically bad model.
    unusable = sum(1 for d in details if not d.get("pred"))
    if unusable:
        print(f"  [!] {unusable}/{n} predictions unparseable — the scores below "
              f"are NOT a model result. Check the serving path first.")
        # evaluate.py already stores the raw reply on failure; surface one so the
        # cause is visible without re-running anything.
        for d in details:
            if not d.get("pred"):
                print(f"      sample raw: {d.get('raw', '')[:160]!r}")
                if d.get("error"):
                    print(f"      error: {d['error']}")
                break
    cm = Counter((d["gold"]["urgency"], (d["pred"] or {}).get("urgency"))
                 for d in details)
    hits = {f: [] for f in ("urgency",)}
    for g in THREE:
        total = sum(v for (gg, _), v in cm.items() if gg == g)
        hit = cm[(g, g)]
        lo, hi = wilson(hit, total)
        # precision for this class = correct / everything predicted as this class
        pred_total = sum(v for (_, p), v in cm.items() if p == g)
        prec = (hit / pred_total) if pred_total else 0.0
        row = {p: v for (gg, p), v in sorted(cm.items(), key=str) if gg == g}
        print(f"  urgency {g:<7} recall={hit:>3}/{total:<3} = "
              f"{hit / total if total else 0:6.1%} [{lo:.0%}-{hi:.0%}]"
              f"   precision={prec:5.1%} ({hit}/{pred_total})   {row}")
    for d in details:
        hits["urgency"].append(d["hit"]["urgency"])
    return {"hits": hits["urgency"], "cost": cost_of(details), "n": n}


def main() -> None:
    paths = sys.argv[1:] or ["ft_17b.json"]
    details = {p: load(p) for p in paths}
    summary = {name: report(name, d) for name, d in details.items()}

    print("\n=== cost (gold=high missed costs %dx) ===" % HIGH_MISS_COST)
    for name, s in summary.items():
        print(f"  {name:<24} cost={s['cost']:>3} over n={s['n']}  "
              f"= {100 * s['cost'] / s['n']:.0f} per 100")
    if len(summary) > 1:
        ranking = sorted(summary.items(), key=lambda kv: kv[1]["cost"])
        print(f"  -> lower cost is better: {[k for k, _ in ranking]}")
    print("  sensitivity (does the ranking survive other weights?):")
    for w in COST_SWEEP:
        costs = {name: cost_of(d, w) for name, d in details.items()}
        order = [k for k, _ in sorted(costs.items(), key=lambda kv: kv[1])]
        print(f"    x{w}: " + "  ".join(f"{k}={v}" for k, v in costs.items())
              + f"   order={order}")

    first = next(iter(details.values()))
    print("\n=== gold: escalate x urgency "
          "(six populated cells => the fields are independent) ===")
    table = Counter((d["gold"]["escalate"], d["gold"]["urgency"]) for d in first)
    for esc in (True, False):
        cells = {u: table[(esc, u)] for u in THREE if table[(esc, u)]}
        print(f"  escalate={str(esc):<5} {cells}")

    if len(details) > 1:
        names = list(details)
        first, second = names[0], names[1]
        print(f"\n=== paired McNemar: {first} vs {second} ===")
        for field, ha, hb in (
            ("urgency", summary[first]["hits"], summary[second]["hits"]),
            ("exact_match", exact_hits(details[first]),
             exact_hits(details[second])),
        ):
            only_a, only_b, p = mcnemar(ha, hb)
            verdict = "SIGNIFICANT" if p < 0.05 else "NOT significant"
            print(f"  {field:<12} {first} only={only_a}  {second} only={only_b}  "
                  f"McNemar p={p:.3f}  -> {verdict}")

        agree = sum(1 for x, y in zip(details[first], details[second])
                    if x["pred"] and y["pred"]
                    and x["pred"]["urgency"] == y["pred"]["urgency"])
        pairs = sum(1 for x, y in zip(details[first], details[second])
                    if x["pred"] and y["pred"])
        if pairs:
            print(f"  urgency agreement: {agree}/{pairs} = {agree / pairs:.1%}")
        else:
            print("  urgency agreement: n/a — no row has a parseable prediction "
                  "in both dumps")

        print("=== both models agree, gold differs (candidate label errors) ===")
        found = 0
        for i, (x, y) in enumerate(zip(details[first], details[second]), 1):
            if not (x["pred"] and y["pred"]):
                continue
            pu, yu, gu = (x["pred"]["urgency"], y["pred"]["urgency"],
                          x["gold"]["urgency"])
            if pu == yu and pu != gu:
                found += 1
                print(f"  [{i:>2}] gold={gu:<7} both_models={pu:<7} "
                      f"{x['question'][:52]}")
        if not found:
            print("  (none)")


if __name__ == "__main__":
    main()
