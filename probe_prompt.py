#!/usr/bin/env python3
"""
Probe WHY the model returns an empty array for some categories.

Runs one failing category through several prompt/format combinations and prints
what each returns, so the cause is identified rather than guessed.

    python3 probe_prompt.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import (CATEGORIES, CATEGORY_NAMES, ESCALATE_RULES,  # noqa: E402
                    URGENCY_RULES, URGENCY_WARNING, category_block_except)
import synthesize as synth  # noqa: E402

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"
CAT = "假期與休假"
DEF = CATEGORIES[CAT]


def call(prompt: str, fmt=None) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "think": False,
               "options": {"temperature": 0.9}}
    if fmt is not None:
        payload["format"] = fmt
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


PROMPT_SIMPLE = f"""你是香港勞工諮詢熱線的數據標註員。

請生成 3 條市民的**口語提問**，每條都必須明確屬於類別「{CAT}」。
類別定義：{DEF}

urgency：high=已發生權益受損；medium=權益諮詢；low=定義理解。
escalate：true=涉及具體個案；false=一般諮詢。

只輸出 JSON 陣列，每項含 text / category / urgency / escalate 四個欄位，
category 一律填「{CAT}」。

[{{"text": "舉例：颱風假算唔算法定假日？", "category": "{CAT}", "urgency": "low", "escalate": false}}]
"""

PROMPT_NO_SELFCHECK = f"""你是香港勞工諮詢熱線的數據標註員。

請生成 3 條市民的**口語提問**，屬於類別「{CAT}」（{DEF}）。
可用粵語口語。urgency 為 high/medium/low，escalate 為 true/false。

只輸出 JSON 陣列：
[{{"text": "...", "category": "{CAT}", "urgency": "high", "escalate": true}}]
"""

PROMPT_FREE = f"""生成 6 條香港勞工諮詢的市民口語提問，涵蓋不同情境，並為每條標註：
- category（從以下選一個）：{"、".join(CATEGORY_NAMES)}
- urgency：high、medium 或 low
- escalate：true 或 false

只輸出 JSON 陣列。"""

variants = [
    ("A 完整prompt + schema", synth.PROMPT.format(
        n=3, category=CAT, definition=DEF, others=category_block_except(CAT),
        urgency_rules=URGENCY_RULES, urgency_warning=URGENCY_WARNING,
        escalate_rules=ESCALATE_RULES),
     synth.RESPONSE_SCHEMA),
    ("B 簡化prompt + schema", PROMPT_SIMPLE, synth.RESPONSE_SCHEMA),
    ("C 簡化prompt + json", PROMPT_SIMPLE, "json"),
    ("D 簡化prompt + 無format", PROMPT_SIMPLE, None),
    ("E 極簡prompt + 無format", PROMPT_NO_SELFCHECK, None),
    ("F 自由生成 + schema", PROMPT_FREE, synth.RESPONSE_SCHEMA),
]

def main() -> None:
    print(f"model={MODEL}  category={CAT}\n")
    for name, prompt, fmt in variants:
        try:
            raw = call(prompt, fmt)
        except Exception as e:
            print(f"{name:<26} ERROR {type(e).__name__}: {e}")
            continue
        tag = "EMPTY" if raw.strip() in ("[]", "") else "ok"
        print(f"{name:<26} [{tag}] len={len(raw):<5} {raw[:130]!r}")

    print("\n解讀：")
    print("  只有 A 是 EMPTY  -> prompt 的『其他類別/自檢』害的（改用 B 的寫法）")
    print("  A 和 B 都 EMPTY  -> schema 約束問題（改用 C 或 D 的寫法）")
    print("  全部 EMPTY       -> 該類別本身讓模型為難，看 F 是否正常")
    print("  全部 ok          -> 空回應是間歇性的，靠重試解決（不是 prompt 問題）")


if __name__ == "__main__":
    main()
