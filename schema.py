"""
Single source of truth for the triage label space.

Categories were MERGED from the 16 Labour Department FAQ topics after a manual
spot-check of synthetic data showed ~30% label errors. Root cause: four of the
original topics (僱傭保障 / 遣散費及長期服務金 / 受條例保障的僱員 /
終止僱傭合約) overlap too much to label reliably — by a model or a human.
Fewer, well-separated categories beat many unstable ones.

train / eval / inference must all use the SAME system prompt and label set.
"""
import json

# category -> one-line boundary definition. Used BOTH in the synthesis prompt
# (so the generator knows what belongs here) and in the system prompt (so the
# model is judged against the same definition it was trained on).
# category -> boundary definition. Used BOTH in the synthesis prompt (so the
# generator knows what belongs here) and in the system prompt (so the model is
# judged against the same definition it was trained on).
#
# "不包含" lines were added after a review found 3/36 mislabels, 2 of them caused
# by 離職補償 absorbing questions that belong elsewhere. Saying what a category
# is NOT is what stopped the bleeding.
CATEGORIES = {
    "工資與扣薪": (
        "工資的定義與支付、欠薪、工資扣除（不論以甚麼名目）、最低工資的適用範圍與計算。"
        "不包含：遣散費／長期服務金／年終酬金（屬「離職補償」）"
    ),
    "假期與休假": "休息日、法定假日、年假、病假、產假、侍產假的安排與薪酬",
    "合約與解僱": (
        "僱傭合約的形式與條款、通知期、解僱是否合法、不合理或不合法解僱。"
        "不包含：遣散費／長期服務金的金額與資格（屬「離職補償」）"
    ),
    "離職補償": (
        "遣散費、長期服務金、年終酬金的資格與計算。"
        "不包含：通知期與解僱是否合法（屬「合約與解僱」）、"
        "工資扣除（屬「工資與扣薪」）"
    ),
    "退休與福利": "強制性公積金（MPF）及其他僱員福利",
    "適用範圍": "誰受《僱傭條例》保障（兼職、臨時工、自僱人士的區分）",
}

CATEGORY_NAMES = list(CATEGORIES)

URGENCIES = ("high", "medium", "low")

FIELDS = ("category", "urgency", "escalate")

# Inference / evaluation contract: ONE label object per request. (Synthesis uses
# an array of these — see synthesize.RESPONSE_SCHEMA.) Both are built from the
# same FIELDS so the JSON shape cannot drift between training and serving.
PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORY_NAMES},
        "urgency": {"type": "string", "enum": list(URGENCIES)},
        "escalate": {"type": "boolean"},
    },
    "required": list(FIELDS),
}


def category_block() -> str:
    """The whole taxonomy, one line per category."""
    return "\n".join(f"- {name}：{desc}" for name, desc in CATEGORIES.items())


def category_block_except(name: str) -> str:
    """Every category except `name` — used to show the generator the boundaries."""
    return "\n".join(f"- {k}：{v}" for k, v in CATEGORIES.items() if k != name)


# urgency / escalate rules live here too, so the synthesis prompt (used to build
# data) and the SYSTEM message (used to train and to judge) cannot drift apart.
# The explicit "this situation" cue was added after a review found consultative
# questions ("呢個做法對唔對?") being labelled low instead of medium.
URGENCY_RULES = (
    "- high   = 已發生權益受損（欠薪、違法解僱、工傷、未付遣散費）／法定時效逼近／涉及具體金額爭議\n"
    "  例：「老闆拖我三個月工資」「我入職五年被突然辭退」\n"
    "- medium = 權益諮詢：問「我有沒有權」「點算先有資格」「咁樣做對唔對」\n"
    "  例：「兼職受唔受條例保障？」「我夠唔夠資格拿遣散費？」\n"
    "- low    = 純定義理解：問「甚麼是…」\n"
    "  例：「咩叫做長期服務金？」"
)

URGENCY_WARNING = (
    "⚠️ 只要問題在問「我這個情況算不算／有沒有資格／對不對」，就是 medium，不是 low。"
)

ESCALATE_RULES = (
    "- true  = 涉及具體個案（具體金額、具體日期、已收到通知、準備採取法律行動）\n"
    "  例：「公司欠我 25,000 元」「我上個月 15 號應該出糧但到今日都無」\n"
    "- false = 一般諮詢，市民可自行按條文處理\n"
    "  例：「兼職有無年假？」"
)

# Short keyword hints for the SERVING prompt. The full definitions above are for
# SYNTHESIS (where precise guidance is what keeps label quality high); once the
# rules are fine-tuned into the weights they no longer need to sit in every
# request. Measured: the long SYSTEM is ~955 tokens and pushed training VRAM to
# 8.18GB on an 8GB card, causing paging.
CATEGORY_HINTS = {
    "工資與扣薪": "工資、欠薪、扣薪、最低工資",
    "假期與休假": "休息日、假日、年假、病假、產假、侍產假",
    "合約與解僱": "合約條款、通知期、解僱是否合法",
    "離職補償": "遣散費、長期服務金、年終酬金",
    "退休與福利": "強積金 MPF、僱員福利",
    "適用範圍": "誰受保障：兼職、臨時工、自僱人士",
}

SYSTEM = (
    "你是香港勞工諮詢分流助手。只輸出一個 JSON 物件，不要任何其他文字。\n"
    '格式：{"category": "...", "urgency": "high|medium|low", "escalate": true|false}\n'
    "category（6 選 1）：\n"
    + "\n".join(f"- {k}：{v}" for k, v in CATEGORY_HINTS.items()) + "\n"
    "urgency：high=已受損或時效逼近；medium=權益諮詢（我算不算／有無資格／對唔對）；"
    "low=定義或資訊查詢\n"
    "escalate：涉及具體個案（金額、日期、已收通知）→ true；一般諮詢 → false"
)

# The prompt that actually produced the gold labels (synthesize.PROMPT) carries
# the full urgency rules WITH examples and the warning. SYSTEM above is the
# compressed serving version, so the model is asked to reproduce labels that were
# made under a stricter rule set. SYSTEM_FULL exists to A/B that gap: same
# categories, but urgency spelled out the way the annotator saw it.
SYSTEM_FULL = (
    "你是香港勞工諮詢分流助手。只輸出一個 JSON 物件，不要任何其他文字。\n"
    '格式：{"category": "...", "urgency": "high|medium|low", "escalate": true|false}\n'
    "category（6 選 1）：\n"
    + "\n".join(f"- {k}：{v}" for k, v in CATEGORY_HINTS.items()) + "\n"
    "\nurgency 標註規則：\n" + URGENCY_RULES + "\n" + URGENCY_WARNING + "\n"
    "\nescalate 標註規則：\n" + ESCALATE_RULES
)


def system_prompt(variant: str = "short") -> str:
    """'short' = serving prompt (default), 'full' = the annotator's rule set."""
    return SYSTEM_FULL if variant == "full" else SYSTEM


def to_json(row: dict) -> str:
    """Serialise a label dict with a fixed key order, so the model sees one form."""
    return json.dumps({k: row[k] for k in FIELDS}, ensure_ascii=False)


def to_messages(row: dict) -> dict:
    """One training example in chat format (trl / unsloth SFTTrainer style)."""
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": to_json(row)},
    ]}


def label_of(row: dict) -> dict:
    return {k: row[k] for k in FIELDS}
