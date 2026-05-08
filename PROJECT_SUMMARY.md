# 文化感知跨语言安全对齐 — 项目总结

## 架构设计

```
用户文本
  │
  ├── Stage 1: 文化语境检测 (Context Detector, ~10M)
  │     └── 输出: cultural_context identifier
  │
  ├── Stage 2: LoRA 动态路由
  │     └── 根据文化语境挂载对应国家的 LoRA 模块
  │
  └── Stage 3: 分类判决
        ├── Base 通用分类器 (5 个全球标签)
        └── Local 特化分类器 (每国 2-3 个本地标签)
```

**模型架构**: Frozen xlm-roberta-base (270M) + LoRA Adapter (r=16, ~2M trainable) + Multi-label Classification Head

## 通用违规分类 (Base — 5 标签，所有国家共享)

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `base_violence_dangerous_behavior` | HIGH | contextual | 暴力与危险行为：煽动暴力、传播武器制作方法、极端血腥/酷刑描述、组织暴力活动 |
| 2 | `base_hate_speech_harassment` | HIGH | hybrid | 仇恨言论与严重骚扰：基于种族/宗教/性别等特征的攻击、系统性骚扰、人肉搜索 |
| 3 | `base_sexual_content_child_safety` | CRITICAL | hybrid | 色情与儿童保护：CSAM、未成年人性化描述、未经同意的私密影像传播、露骨色情 |
| 4 | `base_self_harm_suicide` | HIGH | contextual | 自残与生命安全：自杀方法描述、自残美化、自杀游戏组织、厌食症极端美化 |
| 5 | `base_spam_deceptive_practices` | LOW | keyword_sensitive | 欺诈与垃圾信息：钓鱼链接、虚假中奖、庞氏骗局/传销、批量刷屏、虚假医疗广告 |

## 各国独有分类 (Local Delta — 17 标签)

### 🇸🇬 新加坡 (SG) — 核心痛点：种族和谐与多语言混合

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_sg_racial_religious_harmony` | CRITICAL | contextual | 破坏种族/宗教和谐：CMIO 种族框架攻击、CECA 种族煽动、外籍劳工非人化 |
| 2 | `local_sg_vulgarity_singlish` | MEDIUM | hybrid | Singlish 本土粗口：福建话/马来语拼音缩写 (CB, KNN, LJ 系列) |

### 🇮🇩 印度尼西亚 (ID) — 核心痛点：SARA 原则与海量俚语

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_id_sara_violation` | CRITICAL | hybrid | 违反 SARA 原则：攻击 Suku(部落)、Agama(宗教)、Ras(血统)、Antargolongan(群体) |
| 2 | `local_id_pornography_slang` | HIGH | keyword_sensitive | 隐晦色情俚语：Bahasa Gaul 黑话 (bokep, open BO, VCS, lonte 等) |

### 🇹🇭 泰国 (TH) — 核心痛点：王室保护与政治极化

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_th_lese_majeste` | CRITICAL | contextual | 冒犯君主罪 (第112条)：对国王/王后/王储的诽谤侮辱，包括数字代号、emoji 暗讽 |
| 2 | `local_th_political_instigation` | HIGH | contextual | 政治煽动：煽动非法集会、暴力抗议、推翻宪政秩序 |

### 🇲🇽 墨西哥 (MX) — 核心痛点：毒枭文化与极端暴力

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_mx_narco_culture` | CRITICAL | contextual | 毒枭文化颂扬：赞美卡特尔(CJNG/CDS)、处决视频预告、招募成员、narcocorridos |
| 2 | `local_mx_gender_violence` | HIGH | contextual | 性别暴力/仇女：极端 machismo 言论、feminicidio 煽动、针对女性的暴力威胁 |

### 🇧🇷 巴西 (BR) — 核心痛点：政治撕裂与阶级歧视

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_br_political_extremism` | CRITICAL | contextual | 政治极端化：煽动军事政变、攻击电子投票系统(urnas eletrônicas)、攻击最高法院(STF) |
| 2 | `local_br_structural_racism` | CRITICAL | hybrid | 结构性种族主义：针对黑人/贫民窟(favela)居民的歧视，macaco 等极端种族侮辱 |

### 🇸🇦 沙特阿拉伯 (SA) — 核心痛点：宗教神圣与道德风纪

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_sa_blasphemy_anti_islam` | CRITICAL | contextual | 亵渎神明与反伊斯兰：侮辱先知/古兰经、传播无神论 (ilhad) |
| 2 | `local_sa_immorality_lgbtq` | HIGH | keyword_sensitive | 违反公共道德与 LGBTQ+ 宣扬：违反 modesty 标准、彩虹旗信号 |
| 3 | `local_sa_anti_state` | CRITICAL | contextual | 反国家/王室言论：攻击 Al Saud 王室成员或国家政策 |

### 🇹🇷 土耳其 (TR) — 核心痛点：国家尊严与地缘政治

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_tr_insulting_state` | CRITICAL | contextual | 侮辱国家与国父：侮辱土耳其共和国/总统/Atatürk (TCK 299/301) |
| 2 | `local_tr_separatism_terror` | CRITICAL | hybrid | 分裂主义与恐怖组织：宣传 PKK/KCK、FETÖ/PDY 等被认定为恐怖组织的团体 |

### 🇿🇦 南非 (ZA) — 核心痛点：种族隔离遗毒与排外

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `local_za_severe_racism` | CRITICAL | keyword_sensitive | 极端种族主义：K-word (Kaffir) 拦截，种族仇恨犯罪。严重度等同于美国 N-word |
| 2 | `local_za_xenophobia` | HIGH | contextual | 仇外/恐非症：针对非洲其他国家移民 (amakwerekwere) 的暴力煽动 |

## 训练数据进展

### 样本类型说明

| 类型 | 作用 | 示例 |
|------|------|------|
| **正样本 (Positive)** | 教模型识别违规内容 | "These kaffirs always complaining..." → is_violation=true |
| **对抗变体 (Adversarial)** | 教模型识别被刻意变形/绕过关键词过滤的内容 | 原文 "kaffir" 变体为 "k@ff!r" / "🐒" / "k​a​f​f​i​r" |
| **负样本 (Negative)** | 教模型区分"看起来像违规但实际不是"的边界场景 | 教授引用历史文献中的 K-word → is_violation=false |

### 数据统计

| 国家 | 总数 | Base 正样本 | Local 正样本 | 对抗变体 | 负样本 | 标签数 |
|------|------|------------|-------------|---------|--------|--------|
| 🇸🇦 沙特 | 1,581 | 550 | 291 | 199 | 322 | 8 (5B+3L) |
| 🇮🇩 印尼 | 1,434 | 550 | 259 | 438 | 187 | 7 (5B+2L) |
| 🇹🇷 土耳其 | 1,376 | 550 | 229 | 237 | 228 | 7 (5B+2L) |
| 🇲🇽 墨西哥 | 1,358 | 550 | 373 | 45 | 279 | 7 (5B+2L) |
| 🇧🇷 巴西 | 1,234 | 550 | 296 | 240 | 154 | 7 (5B+2L) |
| 🇿🇦 南非 | 1,252 | 550 | 329 | 240 | 149 | 7 (5B+2L) |
| 🇸🇬 新加坡 | 1,147 | 550 | 215 | 240 | 136 | 7 (5B+2L) |
| 🇹🇭 泰国 | 1,102 | 550 | 203 | 30 | 204 | 7 (5B+2L) |
| **总计** | **10,484** | **4,400** | **2,195** | **1,669** | **1,659** | **56** |

### 数据来源

| 模型 | 用途 | 生成标签 |
|------|------|---------|
| GPT-4o-mini | 初次生成 | 11 个 local 标签 |
| Gemini 2.5 Flash | 补齐低样本标签 + Base 标签 | 5×8 base + 6 个 local 标签翻倍 |

### 文件位置

```
data_csv/
├── safety_training_data.csv    # 8 国合并，10,484 行
├── safety_data_BR.csv           1,234 行
├── safety_data_ID.csv           1,434 行
├── safety_data_MX.csv           1,358 行
├── safety_data_SA.csv           1,581 行
├── safety_data_SG.csv           1,147 行
├── safety_data_TH.csv           1,102 行
├── safety_data_TR.csv           1,376 行
└── safety_data_ZA.csv           1,252 行
```

CSV 字段: `country_code, label, is_violation, text, severity, detection_type, language, generation_strategy, adversarial_technique`

## 下一步

训练代码位于 `train/` 目录下：

```bash
# 查看数据统计
python -m train --prepare-only

# 单国测试训练
bash train/run.sh --countries TH --epochs 3

# 全量训练 8 国
bash train/run.sh
```

## 增量添加数据

```bash
# 生成更多 base 样本
python -m generate.data_mgmt generate-base --countries TH --samples-per-label 50

# 生成更多 local 样本
python -m generate --labels local_th_lese_majeste --force

# 从外部 CSV 导入
python -m generate.data_mgmt add --file new_samples.csv

# 重新导出 CSV
python -m generate.data_mgmt export-csv
```
