# 香港勞工諮詢分流 · LoRA 微調

把市民口語化的香港勞工法諮詢問題,分到 6 個類別,並判斷緊急程度與是否需要轉人工。

**目標不是「比大模型更準」,而是「用更低的成本做到同級效果」—— 並用統計檢驗證明它做到了。**

```json
輸入: "老闆拖咗我三個月工資，我可以點算？"
輸出: {"category": "工資與扣薪", "urgency": "high", "escalate": true}
```

---

## 結果

同一份測試集(106 條,零重複),三條路徑:

| 指標 | A. `qwen3.5:9b` | B. `Qwen3-1.7B` base | **C. `Qwen3-1.7B` + LoRA** |
|---|---|---|---|
| `urgency_acc` | 62.3% | 42.5% | **68.9%** |
| **緊急個案召回** | 37.8% | 91.9% | **75.7%** |
| 緊急個案精確 | 93.3% | 45.3% | 80.0% |
| `low` 召回 | 66.7% | **0.0%** | 46.7% |
| **加權代價 /100** ↓ | 81 | 63 | **48** |

*加權代價:漏判一個緊急個案計 3 分,其餘錯誤計 1 分 —— 業務上漏掉 urgent 最貴。*

配對顯著性(McNemar,同一批題):

| 對比 | 指標 | 不一致 | p |
|---|---|---|---|
| **base → LoRA** | `exact_match` | **3 : 40** | **0.000** |
| **base → LoRA** | `urgency` | 6 : 34 | **0.000** |
| **9B vs LoRA** | `exact_match` | 11 : 24 | **0.041**(偏向 LoRA) |
| 9B vs LoRA | `urgency` | 15 : 22 | 0.324(不顯著) |

**三條結論:**

1. **微調極顯著有效** —— 相對同模型的 base,`exact_match` 是 3:40(p=0.000)。三次獨立測量全部確認。
2. **1.7B 在關鍵指標上超過 9B** —— 緊急個案召回 **37.8% → 75.7%**,加權代價 **低 41%**。
3. **模型小一半、代價低四成、準確率還更高** —— 這正是專案要證明的東西。

---

## 為什麼這個專案值得看:結論被推翻過兩次

**同一組模型,只換測試集,排名就翻轉了:**

| 協議 | n | 9B urgency | LoRA urgency | 當時的結論 |
|---|---|---|---|---|
| v1 | 46(含重複) | **73.9%** | 58.7% | 「9B 更好,1.7B 落後 15.2pp」 |
| v2 | 108(含重複) | 61.1% | **69.4%** | 「1.7B 反超」 |
| **v3** | **106(乾淨)** | 62.3% | **68.9%** | **「1.7B 反超,且顯著」** |

**9B 在 46 條上是 73.9%,在 106 條上是 62.3% —— 波動 11.6pp。** 中間那句「瓶頸精確在 urgency」
在 n=46 時 p=0.167(**不顯著**),**已被撤回**。

**這件事本身是這個專案最想展示的能力:知道自己什麼時候不該相信自己的數字。**

---

## 流程

```
                    ┌─────────────────────────────────────────┐
                    │  合成數據 (synthesize.py)                │
                    │  本地 qwen3.5:9b 生成問題 + 標籤          │
                    └────────────────┬────────────────────────┘
                                     ▼
        人工抽檢 (review.py) ──► 標籤體系 16 類 → 6 類(30% 標錯過)
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  劃分 (check_dataset.py)                 │
                    │  去重 · 分層 70/15/15 · 主題均衡          │
                    └────────────────┬────────────────────────┘
                                     ▼
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  基線 (evaluate.py)          微調 (train.py)              分析 (analyze.py)
  qwen3.5:9b + prompt         Qwen3-1.7B + fp16 LoRA        McNemar / CI / cost
  Ollama 路徑                  HF 路徑                       逐類 recall+precision
```

---

## 快速開始

```bash
# 環境:WSL2 + RTX 3060 Ti 8GB · Python 3.12
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch transformers peft bitsandbytes

python synthesize.py --per-category 120     # 合成 ~720 條
python check_dataset.py                     # 去重 + 劃分
python check_facts.py                       # 虛構機構／外地法例掃描

# 三方對比
python evaluate.py --data data/test.jsonl --out v3_9b.json
python train.py --eval-only --no-adapter --no-quant --base Qwen/Qwen3-1.7B --out v3_base.json
python train.py --eval-only --no-quant --base Qwen/Qwen3-1.7B --out v3_ft.json

# 全套統計
python analyze.py v3_9b.json v3_ft.json
python analyze.py v3_base.json v3_ft.json
```

沒有 GPU 也能跑 `synthesize.py` / `check_dataset.py` / `analyze.py` —— 它們只需要 Python 標準庫。

---

## 踩過的坑

| # | 坑 | 症狀 | 修法 |
|---|---|---|---|
| 1 | **標籤體系照搬業務目錄** | 16 類抽檢 30% 標錯 | 合併成 6 類,並補「不包含」條款 |
| 2 | **`format:"json"` 不約束結構** | 16 主題裡 13 個只出 1 條 | 改用 JSON Schema 強制陣列 |
| 3 | **QLoRA 在小模型上是負優化** | 4-bit 比 fp16 **慢 20 倍**、更費顯存 | 1.7B 以下用 fp16 LoRA |
| 4 | **缺控制組** | 差點認定「微調無效」 | 加 `--no-adapter` 跑同路徑 base |
| 5 | **自標自測** | 9B 的 73.9% 是「自一致率」而非準確率 | 已知限制,寫進文件 |
| 6 | **詳細規則反而有害** | 完整規則 vs 精簡規則 **p=0.003** | 保留精簡版;規則不是標籤的驅動因素 |
| 7 | **去重寫法錯誤** | 同一響應內的重複漏過,污染 1.8% | 逐條立即標記 `seen` |
| 8 | **預設值三連坑** | `MAX_LEN=512`(太短)、`--max-per-category=20`(砍掉 2/3 數據)、單次請求 69 條(JSON 截斷) | 全部改成「按實際規模」並加告警 |
| 9 | **`num_predict` 未設** | 單條推理跑 480 秒 | 限制輸出 64 token、`num_ctx` 降到 2048 |

---

## 數據質量檢查

| 檢查項 | 工具 | 結果 |
|---|---|---|
| 重複文本 | `check_dataset.py` | **0**(去重後 704 條) |
| 標籤衝突(同文本不同標籤) | `check_dataset.py` | **0** |
| **虛構機構／外地法例** | `check_facts.py`(22 條模式) | **0 處** ✅ 達 SPEC §4.1 |
| 人工抽檢 `category` | `review.py` | **36/36 = 100%** |
| 人工抽檢 `urgency` | `review.py` | **17/20 = 85%** ⚠️ 未達 90% 線 |

## 已知限制

1. **標籤質量 85%**(人工審 20 條,SPEC 要求 ≥90%)。且 gold 由 9B 自標,不是外部真值。
2. **延遲跨框架**:9B 走 Ollama、1.7B 走 HF,「快幾倍」混了框架差異。
3. **模型會向中間檔坍縮**:18 條「兩模型一致但與 gold 不同」中,9 條 high→medium、8 條 low→medium,
   雙向對稱 —— `low` 召回只有 46.7%。
4. **n=106**:`urgency` 的 95% CI 仍約 ±10pp。

---

## 檔案

| 檔案 | 作用 |
|---|---|
| `SPEC.md` | 任務規格、標籤定義、驗收線 |
| `BASELINE.md` | **完整實驗結果 + 統計檢驗 + 協議敏感性** |
| `schema.py` | 標籤空間的單一真相來源(訓練／評測／推理共用) |
| `synthesize.py` | LLM 合成數據(分批、去重、Schema 約束) |
| `check_dataset.py` | 去重、質量報告、分層劃分、主題均衡 |
| `check_facts.py` | 虛構機構／外地法例掃描 |
| `review.py` | 人工抽檢(三欄位各自對照驗收線) |
| `train.py` | fp16 LoRA 訓練 + 早停 + 逐條落盤 |
| `evaluate.py` | Ollama 側評測(含重試、可見錯誤) |
| `analyze.py` | per-class recall/precision、Wilson CI、McNemar、加權代價 |
| `run.sh` | 執行橋:同步程式碼 → 跑 → 輸出落到可讀取的日誌 |
| `tests/` | 4 套測試(解析器、劃分、續跑安全、評測) |
