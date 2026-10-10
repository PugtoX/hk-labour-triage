#!/usr/bin/env python3
"""
Scan synthesized questions for terms that do not belong to Hong Kong labour law.

    python check_facts.py                       # every row in data/synthetic.jsonl
    python check_facts.py data/test.jsonl       # a specific split

SPEC §4.1 requires "虛構資訊 0 處" but nothing enforced it. This is a real legal
domain: inventing an agency or statute is the fastest way to lose a domain
interviewer, and the synthetic generator does produce mainland / Taiwan terms.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "data" / "synthetic.jsonl"

# regex -> why it is wrong here. Kept as data so it is easy to extend. Patterns
# rather than plain substrings because "最低工資法" is a prefix of the perfectly
# correct "最低工資法例" — a substring match produced a false positive.
FOREIGN = {
    "勞動局": "內地／台灣用語；香港是「勞工處」",
    "劳动局": "內地用語；香港是「勞工處」",
    "勞動處(?!分)": "香港是「勞工處」",
    "劳动处(?!分)": "香港是「勞工處」",
    "勞動基準法": "台灣法例；香港是《僱傭條例》",
    "劳动基准法": "台灣法例；香港是《僱傭條例》",
    "人社局": "內地機構",
    "社保局": "內地機構；香港是「積金局」",
    "勞動仲裁委員會": "內地機構；香港是「勞資審裁處」",
    "劳动仲裁": "內地用語；香港是「勞資審裁處」",
    "五險一金": "內地制度；香港是強積金",
    "五险一金": "內地制度；香港是強積金",
    "特休": "台灣用語；香港是「年假」",
    "法定工時": "台灣概念；香港沒有法定標準工時",
    "每週 ?40 ?小時": "內地概念；香港沒有此規定",
    "每周 ?40 ?小时": "內地概念；香港沒有此規定",
    "最低工資法(?!例|律|規)": "香港是《最低工資條例》",
    "最低工资法(?!例|律|规)": "香港是《最低工資條例》",
    "勞動合同法": "內地法例",
    "劳动合同法": "內地法例",
    "勞檢署": "不存在此機構；香港是「勞工處」",
    "差餉物業稅署": "不存在此機構",
}

PATTERNS = [(re.compile(pattern), why) for pattern, why in FOREIGN.items()]

# Real Hong Kong bodies / statutes. A question naming one of these is fine even
# if it also mentions something in FOREIGN (the foreign hit still wins).
HK_REAL = (
    "勞工處", "勞資審裁處", "小額薪酬索償仲裁處", "積金局", "強積金管理局",
    "入境事務處", "平等機會委員會", "僱傭條例", "最低工資條例",
    "強制性公積金計劃條例", "僱員補償條例", "性別歧視條例", "殘疾歧視條例",
    "家庭崗位歧視條例", "種族歧視條例", "僱員再培訓局",
)


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[!] {path} not found")
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    # --data/xxx.jsonl splits use the chat format; synthetic.jsonl uses flat rows
    rows = load(path)
    texts = []
    for r in rows:
        if "messages" in r:
            texts.append(r["messages"][1]["content"])
        else:
            texts.append(r["text"])

    hits = []
    for i, text in enumerate(texts, 1):
        for pattern, why in PATTERNS:
            m = pattern.search(text)
            if m:
                hits.append((i, m.group(0), why, text))

    print(f"[i] {path.name}: {len(texts)} rows, "
          f"{len(PATTERNS)} foreign-term patterns checked")
    if not hits:
        print("[OK] no foreign-jurisdiction or invented terms found")
        return

    print(f"[!] {len(hits)} hit(s):\n")
    for i, term, why, text in hits:
        print(f"  [{i:>3}] 「{term}」— {why}")
        print(f"        {text[:70]}")
    print(f"\n[!] SPEC §4.1 requires 0. These are generator artefacts — the fix is "
          f"to regenerate those rows (or delete them), not to relabel them.")
    sys.exit(1)


if __name__ == "__main__":
    main()
