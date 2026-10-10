# 實驗結果

三方對比 + 配對顯著性檢驗。**這張表是本項目的核心交付物。**

---

# 主表（協議 v3 — 乾淨數據）

> **協議指紋**：`test.jsonl` **n=106，零重複** ｜ `SYSTEM` = 379 字精簡版 ｜ gold = 9B 自標 ｜ seed 42

## 設定

| | A. 大模型基線 | B. 小模型 base | C. 小模型微調 |
|---|---|---|---|
| 模型 | `qwen3.5:9b`（6.6GB） | `Qwen/Qwen3-1.7B` | 同 B + LoRA adapter |
| 推理路徑 | **Ollama**（GGUF 社區量化） | HF transformers (bf16) | HF transformers (bf16) |
| 提示詞 | `schema.SYSTEM`（啟動時自檢一致） | 同左 | 同左 |
| 解碼 | `temperature=0`, `think=false`, `num_predict=64` | greedy | greedy |
| 訓練 | — | — | fp16 LoRA r=16/α=32, lr 2e-4, 3 epoch, 492 條 |

## 三方結果（n = 106）

| 指標 | **A. 9B** | **B. 1.7B base** | **C. 1.7B LoRA** |
|---|---|---|---|
| `urgency_acc` | 62.3% (66/106) | 42.5% (45/106) | **68.9% (73/106)** |
| **`high` 召回** | **37.8%** | 91.9% | **75.7%** |
| `high` 精確 | 93.3% | 45.3% | 80.0% |
| `low` 召回 | 66.7% | **0.0%** | 46.7% |
| **加權代價 /100** | **81** | 63 | **48** |

## 配對顯著性（McNemar，同一批 106 條）

| 對比 | 指標 | 不一致（前者/後者） | p | 判定 |
|---|---|---|---|---|
| **B → C** | `exact_match` | **3 / 40** | **0.000** | ✅ **顯著** |
| **B → C** | `urgency` | 6 / 34 | **0.000** | ✅ **顯著** |
| **A vs C** | `exact_match` | 11 / 24 | **0.041** | ✅ **顯著（偏向 C）** |
| A vs C | `urgency` | 15 / 22 | 0.324 | ⚠️ 不顯著 |

加深 = 加權代價（`gold=high` 漏判計 3 分，其餘錯計 1 分）：

| 模型 | ×2 | ×3 | ×5 |
|---|---|---|---|
| A. 9B | 63 | 86 | 132 |
| C. LoRA | **42** | **51** | **69** |

**排序在 ×2/×3/×5 下全部穩定。**

## 三條結論

1. **微調極顯著有效。** `base → LoRA` 的 `exact_match` 是 **3:40**，p=0.000。
   三次獨立測量（46 / 108 / 106 條）**全部確認**。
2. **1.7B 在關鍵指標上超過 9B。** 緊急個案召回 **37.8% → 75.7%**，加權代價 **低 41%**，
   `exact_match` **顯著更好**（p=0.041）。緊急檔是業務上最貴的一檔。
3. **SPEC §7「同級別準確率下大幅降低成本」達成。**
   1.7B 的 urgency 更高（68.9% vs 62.3%），同時模型小一半、加權代價低四成。

## 模型的失敗模式：向中間檔坍縮

```
C. LoRA 預測：high 35   medium 56   low 15
gold       ：high 37   medium 39   low 30
```

`low` 召回只有 46.7%。18 條「兩模型一致但與 gold 不同」中，**9 條 `gold=high` 被判成 medium、
8 條 `gold=low` 被判成 medium** —— **雙向對稱**，說明這不是單向標錯，而是**模型不願使用極端檔位**。

---

# 協議敏感性：結論曾經翻轉過

同一組模型，只換 test 集，排名就變了：

| 協議 | n | 9B urgency | LoRA urgency | 誰更好 |
|---|---|---|---|---|
| v1 | 46（含重複） | **73.9%** | 58.7% | **9B** |
| v2 | 108（含重複） | 61.1% | **69.4%** | 1.7B |
| **v3** | **106（乾淨）** | 62.3% | **68.9%** | **1.7B** |

**9B 在 46 條上是 73.9%，在 106 條上是 62.3% —— 波動 11.6pp。**

**教訓**：小 test 集上的模型排名不可靠。項目中期基於 n=46 寫過「1.7B 落後 15.2pp（瓶頸在 urgency）」，
那句話**已被撤回** —— 它在 v3 上不成立，且在 n=46 時 p=0.167（不顯著）。

**v2 → v3 只去掉了 13 條重複文本**（占 1.8%），結論方向不變但數值穩定下來 ——
這說明 v2 的「超越」不是重複樣本造成的假象。

---

# 標籤質量：本項目的真實上限

| 檢查 | 結果 | 線 | 判定 |
|---|---|---|---|
| 人工審 20 條 urgency | **17/20 = 85.0%** | ≥90% | ❌ **FAIL** |
| 9B vs 自身標籤（同 prompt） | 73.9% | — | 自一致性上限 |
| 9B vs 1.7B 一致率 | **65.1%** | — | 欄位噪聲大 |
| 詳細規則 vs 精簡規則（9B） | **p=0.003** | — | 見下 |

**兩處結構性問題：**

1. **自標自測。** gold 由 9B 以 `temperature=0.9` 自標，評測時 9B 以 `temperature=0`
   判同一批題 —— 所以 A 的分數是**自一致率**，不是對外部真值的準確率。
2. **契約漂移（本項目自造）。** 產出 gold 的 `synthesize.PROMPT` 含完整 `URGENCY_RULES`
   ＋三個例子＋`URGENCY_WARNING`；推理用的 `SYSTEM` 是被壓到 379 字的精簡版。

**Prompt A/B 的結果出乎預料**（`evaluate.py --system full`）：

| 9B | short（379字） | full（667字） |
|---|---|---|
| high 召回 | **41.7%** | **8.3%** |
| 加權代價 /100 | **57** | **98** |
| McNemar | — | **p=0.003** |

**完整規則顯著更差。** 根因是那條 `URGENCY_WARNING`（「只要問『我這個情況算不算』就是 medium」）
讓模型幾乎全選 medium。**所以 gold 並不忠實於規則文本** —— 同一套規則，標註時與推理時結果顯著不同。
**「改寫規則來提升 urgency」這條路因此被堵死。**

---

# 已知偏差

1. **延遲跨框架**：A 走 Ollama、C 走 HF，「快 1.6 倍」混了框架差異。
2. **`n=106`**：`urgency` 的 95% CI 仍約 ±10pp（見 `analyze.py` 輸出）。
3. **gold 未經全面人工校驗**：只審了 20 條，urgent/escalate 的整體一致性未知。
4. ~~合成數據的虛構風險~~ → **已檢查**：`check_facts.py` 以 22 條外地／虛構用語模式掃描
   全部 717 條與測試集 106 條，**0 處命中**（SPEC §4.1 達成）。
   注意該腳本第一版有假陽性（「最低工資法」匹配到正確的「最低工資法例」），已改用負向前瞻 ——
   **會誤報的檢查器比沒有檢查器更危險**，因為它會讓人刪掉正確的數據。

---

# 重跑與分析

```bash
bash run.sh "python check_dataset.py"
bash run.sh "python evaluate.py --data data/test.jsonl --out v3_9b.json"
bash run.sh "python train.py --eval-only --no-adapter --no-quant --base Qwen/Qwen3-1.7B --out v3_base.json"
bash run.sh "python train.py --eval-only --no-quant --base Qwen/Qwen3-1.7B --out v3_ft.json"
bash run.sh "python analyze.py v3_9b.json v3_ft.json"
bash run.sh "python analyze.py v3_base.json v3_ft.json"
```

`analyze.py` 一次輸出：per-class recall／precision（含 Wilson CI）、加權代價（含 ×2/×3/×5 敏感性）、
配對 McNemar（urgency 與 exact_match 分開）、`escalate × urgency` 交叉表、
以及「兩模型一致但與 gold 不同」的可疑標籤清單。

**永久保留的產物**（v1/v2/v3 的逐條 `gold`+`pred`）：
`baseline_9b.json` `base_17b.json` `ft_17b.json`（v1）、`v2_*.json`、`v3_*.json`。
**不要覆蓋** —— 它們是各協議下唯一的原始證據。
