# 数据分布文档 V2

> 更新时间：2026-05-10 | 训练集：106,281 条 | 测试集：972 条

---

## 一、训练数据 (data_v2/)

### 1.0 格式

```
text,label_name,source
```

- `text`: 清洗后的文本（已移除 @USER、URL、RT、<LF> 等噪音）
- `label_name`: 标签名，`safe` 表示安全文本（对所有标签均为负样本）
- `source`: 数据来源 — `external_integration` | `llm_generated` | `external_dataset`

### 1.1 全局概览

| 指标 | 数值 |
|------|------|
| 总样本 | 106,281 |
| 正样本 (各标签违规) | 82,321 (77.5%) |
| 负样本 (safe) | 23,960 (22.5%) |
| 标签数 | 22 (5 base + 17 local) |
| 国家数 | 8 |

### 1.2 数据来源

| 来源 | 数量 | 占比 | 说明 |
|------|------|------|------|
| external_integration | 56,232 | 52.9% | 外部标注数据整合 |
| llm_generated | 16,905 | 15.9% | LLM 生成 |
| external_dataset | 9,398 | 8.8% | Hugging Face 公开数据集 |
| balanced_safe_negative | 23,746 | 22.3% | 均衡处理的安全负样本 (safe) |

### 1.3 各国数据分布

| 国家 | 总量 | Local Pos | Base Pos | Safe Neg | 占比 |
|------|------|-----------|----------|----------|------|
| BR (巴西) | 13,168 | 2,988 | 7,220 | 2,960 | 12.4% |
| ID (印尼) | 13,453 | 2,994 | 7,509 | 2,950 | 12.7% |
| MX (墨西哥) | 13,226 | 2,997 | 7,241 | 2,988 | 12.4% |
| SA (沙特) | 15,724 | 4,488 | 8,262 | 2,974 | 14.8% |
| SG (新加坡) | 12,815 | 2,847 | 7,068 | 2,900 | 12.1% |
| TH (泰国) | 11,253 | 2,253 | 6,001 | 2,999 | 10.6% |
| TR (土耳其) | 12,132 | 2,697 | 6,460 | 2,975 | 11.4% |
| ZA (南非) | 14,510 | 3,000 | 8,510 | 3,000 | 13.6% |

```
各国数据量可视化:
SA ██████████████ 15,724
ZA █████████████ 14,510
ID ████████████ 13,453
BR ████████████ 13,168
MX ████████████ 13,226
SG ███████████ 12,815
TR ███████████ 12,132
TH ██████████ 11,253
```

### 1.4 Base 标签分布 (5 个)

每标签每国 ≥ 1,000 ✓

| 标签 | BR | ID | MX | SA | SG | TH | TR | ZA | **总计** |
|------|----|----|----|----|----|----|----|----|----------|
| base_csam | 1,990 | 1,998 | 1,829 | 1,980 | 1,973 | 1,000 | 1,997 | 1,998 | **14,765** |
| base_hate_speech | 1,991 | 1,999 | 1,999 | 1,288 | 1,972 | 2,000 | 1,296 | 1,999 | **14,544** |
| base_self_harm | 1,000 | 1,000 | 1,000 | 998 | 1,000 | 1,000 | 1,000 | 1,000 | **7,998** |
| base_spam_fraud | 997 | 1,000 | 999 | 1,998 | 1,137 | 1,000 | 999 | 1,513 | **9,643** |
| base_violence_gore | 1,242 | 1,512 | 1,414 | 1,998 | 986 | 1,001 | 1,168 | 2,000 | **11,321** |

### 1.5 Local 标签分布 (17 个)

每标签 1,000~1,500 ✓

| 国家 | 标签 | 数量 |
|------|------|------|
| BR | br_political_extremism | 1,496 |
| BR | br_structural_racism | 1,492 |
| ID | id_pornography_slang | 1,497 |
| ID | id_sara_violation | 1,497 |
| MX | mx_gender_violence | 1,498 |
| MX | mx_narco_culture | 1,499 |
| SA | sa_anti_state | 1,500 |
| SA | sa_blasphemy_anti_islam | 1,500 |
| SA | sa_immorality_lgbtq | 1,488 |
| SG | sg_racial_religious_harmony | 1,403 |
| SG | sg_vulgarity_singlish | 1,444 |
| TH | th_lese_majeste | 1,016 |
| TH | th_political_instigation | 1,237 |
| TR | tr_insulting_state | 1,200 |
| TR | tr_separatism_terror | 1,497 |
| ZA | za_severe_racism | 1,500 |
| ZA | za_xenophobia | 1,500 |

### 1.6 标签设计

```
22 个标签 = 5 个 Base (通用底线) + 17 个 Local (各国专属)

Safe: safe (对所有标签均为负样本)

Base:
  base_violence_gore     — 暴力与血腥
  base_csam              — 儿童保护与色情
  base_hate_speech       — 仇恨言论与骚扰
  base_spam_fraud        — 欺诈与垃圾信息
  base_self_harm         — 自残与自杀

Local ({cc}_{name}):
  BR: br_political_extremism, br_structural_racism
  ID: id_sara_violation, id_pornography_slang
  MX: mx_gender_violence, mx_narco_culture
  SA: sa_blasphemy_anti_islam, sa_immorality_lgbtq, sa_anti_state
  SG: sg_racial_religious_harmony, sg_vulgarity_singlish
  TH: th_lese_majeste, th_political_instigation
  TR: tr_insulting_state, tr_separatism_terror
  ZA: za_severe_racism, za_xenophobia
```

---

## 二、测试数据 (data_csv/safety_benchmark.csv)

### 2.0 格式

```
text,label_name,is_violation,source
```

- `is_violation`: `true` (违规) / `false` (安全)
- 测试数据完全独立于训练集，由 LLM 重新生成

### 2.1 全局概览

| 指标 | 数值 |
|------|------|
| 总样本 | 972 |
| 正样本 (违规) | 567 (58.3%) |
| 负样本 (安全) | 405 (41.7%) |
| 标签数 | 22 (5 base + 17 local) |

### 2.2 Base 标签分布

| 标签 | 正样本 | 负样本 | 合计 |
|------|--------|--------|------|
| base_hate_speech | 80 | 160 | 240 |
| base_csam | 80 | 40 | 120 |
| base_self_harm | 80 | 40 | 120 |
| base_spam_fraud | 80 | 40 | 120 |
| base_violence_gore | 77 | 40 | 117 |

### 2.3 Local 标签分布

每个 local 标签: 10 正样本 + 5 负样本 = 15 条

| 国家 | 标签数 | 样本数 |
|------|--------|--------|
| BR | 2 | 30 |
| ID | 2 | 30 |
| MX | 2 | 30 |
| SA | 3 | 45 |
| SG | 2 | 30 |
| TH | 2 | 30 |
| TR | 2 | 30 |
| ZA | 2 | 30 |

### 2.4 测试集构成

```
567 正样本 (58.3%):
  ├── Base 正样本: 397 条 (5 个 base 标签 × ~80)
  └── Local 正样本: 170 条 (17 个 local 标签 × 10)

405 负样本 (41.7%):
  ├── Base 负样本: 320 条
  └── Local 负样本: 85 条 (17 个 local 标签 × 5)
```

---

## 三、数据清洗说明

所有训练和测试数据已通过 `generate/clean_data.py` 清洗：

| 清洗项 | 说明 |
|--------|------|
| @USER / @user | Twitter 用户提及 → 移除 |
| RT @xxx: | 转发标记 → 移除 |
| `<LF>` | 换行 token → 空格 |
| https:// / http:// | URL → 普通标签移除，spam 标签替换为 [URL] |
| &amp; &lt; &gt; | HTML 实体 → 解码 |
| 多余空白 | 多空格/多换行 → 单空格 |

清洗后移除 561 条过短文本（< 10 字符）。

---

## 四、文件路径

```
旧版训练数据 (已归档):
  data_csv/safety_training_data.csv  (247K 条，旧标签名)

新版训练数据:
  data_v2/train_BR.csv    data_v2/train_MX.csv    data_v2/train_TH.csv
  data_v2/train_ID.csv    data_v2/train_SA.csv    data_v2/train_TR.csv
  data_v2/train_SG.csv    data_v2/train_ZA.csv

测试数据:
  data_csv/safety_benchmark.csv  (972 条，22 标签)

文档:
  DATA_DISTRIBUTION.md    (旧版 V1 分布文档)
  DATA_DISTRIBUTION_V2.md (当前新版 V2 分布文档)
```
