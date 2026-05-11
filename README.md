# 文化感知跨语言安全对齐系统

基于 **Frozen Base + Regional Adapter + Country LoRA** 架构的内容安全分类器，覆盖 8 个国家（SG、ID、TH、MX、BR、SA、TR、ZA），支持文化特定的违规检测。

## 架构

```
用户文本
  │
  ├── 第一阶段：文化语境检测 (Context Detector, ~10M)
  │     └── 输出: cultural_context 标识符
  │
  ├── 第二阶段：LoRA 动态路由
  │     └── 根据文化语境挂载对应国家的 LoRA 模块
  │
  └── 第三阶段：分类判决
        ├── Base 通用分类器 — 5 个全球标签，所有国家共享
        └── Local 特化分类器 — 每国 2-3 个本地标签，共 17 个
```

**模型**: Frozen `xlm-roberta-base` (270M) + LoRA Adapter (r=16, ~2M 可训练参数) + 多标签分类头

## 三层标签体系

```
Base Violations (5 个全球标签)
  └── Regional Adapters (SEA / LATAM / MENA_AFRICA)
       └── Country LoRA Deltas (每国 2-3 个本地标签)
```

### 全球标签（5 个，所有国家共享）

| # | 标签名 | 严重度 | 检测类型 | 说明 |
|---|--------|--------|---------|------|
| 1 | `base_violence_dangerous_behavior` | HIGH | contextual | 煽动暴力、传播武器制作方法、极端血腥/酷刑描述 |
| 2 | `base_hate_speech_harassment` | HIGH | hybrid | 仇恨言论、系统性骚扰、人肉搜索 |
| 3 | `base_sexual_content_child_safety` | CRITICAL | hybrid | CSAM、未成年人性化描述、露骨色情 |
| 4 | `base_self_harm_suicide` | HIGH | contextual | 自杀方法描述、自残美化、自杀游戏组织 |
| 5 | `base_spam_deceptive_practices` | LOW | keyword_sensitive | 钓鱼链接、虚假中奖、庞氏骗局、虚假医疗广告 |

### 各国本地标签（共 17 个）

| 国家 | 标签 | 核心痛点 |
|------|------|---------|
| 🇸🇬 新加坡 | `local_sg_racial_religious_harmony`、`local_sg_vulgarity_singlish` | 种族宗教和谐、Singlish 本土粗口 |
| 🇮🇩 印尼 | `local_id_sara_violation`、`local_id_pornography_slang` | SARA 原则、Bahasa Gaul 色情黑话 |
| 🇹🇭 泰国 | `local_th_lese_majeste`、`local_th_political_instigation` | 冒犯君主罪（第112条）、政治煽动 |
| 🇲🇽 墨西哥 | `local_mx_narco_culture`、`local_mx_gender_violence` | 毒枭文化颂扬、性别暴力/仇女 |
| 🇧🇷 巴西 | `local_br_political_extremism`、`local_br_structural_racism` | 政治极端化、结构性种族主义 |
| 🇸🇦 沙特 | `local_sa_blasphemy_anti_islam`、`local_sa_immorality_lgbtq`、`local_sa_anti_state` | 亵渎神明、违反公共道德、反王室 |
| 🇹🇷 土耳其 | `local_tr_insulting_state`、`local_tr_separatism_terror` | 侮辱国家/Atatürk（TCK 299/301）、分裂主义宣传 |
| 🇿🇦 南非 | `local_za_severe_racism`、`local_za_xenophobia` | K-word 极端种族主义、仇外暴力 |

## 训练数据

### 数据统计

| 国家 | 总数 | Base 正样本 | Local 正样本 | 对抗变体 | 负样本 | 标签数 |
|------|------|------------|-------------|---------|--------|--------|
| 🇸🇦 沙特 | 1,581 | 550 | 291 | 199 | 322 | 8 |
| 🇮🇩 印尼 | 1,434 | 550 | 259 | 438 | 187 | 7 |
| 🇹🇷 土耳其 | 1,376 | 550 | 229 | 237 | 228 | 7 |
| 🇲🇽 墨西哥 | 1,358 | 550 | 373 | 45 | 279 | 7 |
| 🇧🇷 巴西 | 1,234 | 550 | 296 | 240 | 154 | 7 |
| 🇿🇦 南非 | 1,252 | 550 | 329 | 240 | 149 | 7 |
| 🇸🇬 新加坡 | 1,147 | 550 | 215 | 240 | 136 | 7 |
| 🇹🇭 泰国 | 1,102 | 550 | 203 | 30 | 204 | 7 |
| **总计** | **10,484** | **4,400** | **2,195** | **1,669** | **1,659** | **56** |

### 样本类型

- **正样本 (Positive)**: 违规内容，教模型识别什么该拦截
- **对抗变体 (Adversarial)**: 被刻意变形/绕过关键词过滤的内容（同形异义字、零宽字符、emoji 替换等）
- **负样本 (Negative)**: 边界案例——看起来像违规但实际不是（如学术讨论中引用违规词汇）

### 数据来源

通过 GPT-4o-mini 和 Gemini 2.5 Flash 生成，基于各国 taxonomy 定义进行校验。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM API
cp config.yaml.example config.yaml
# 编辑 config.yaml，填入 LLM API 密钥和地址

# 3. 单国测试训练
bash train/run.sh --countries TH --epochs 5

# 4. 全量训练 8 个国家
bash train/run.sh
```

## 常用命令

### 生成训练数据

```bash
# 为指定标签生成样本
python -m generate --labels local_th_lese_majeste --force

# 生成 base 违规样本
python -m generate.data_mgmt generate-base --countries TH --samples-per-label 50

# 从外部 CSV 导入数据
python -m generate.data_mgmt add --file new_samples.csv

# 导出所有数据到 CSV
python -m generate.data_mgmt export-csv
```

### 模型推理

```bash
# 用训练好的模型测试
python -m train.inference --model trained_models/best_model --text "输入文本"
```

### 跑基准测试

```bash
python -m generate.gen_benchmark
```

## 检测类型与训练策略

| 检测类型 | 损失函数 | γ | 策略 |
|----------|---------|---|------|
| `keyword_sensitive` | 标准交叉熵 | 0 | 基础 CE 损失 |
| `contextual` | Focal Loss | 2 | 聚焦难分样本 |
| `hybrid` | 两阶段损失 | 1-2 | 3:7 CE:Focal 加权 |

## 目录结构

```
├── taxonomy/                  # 违规标签体系（schema、标签定义、路由表）
│   ├── schema.json
│   ├── base_violations.json
│   ├── regional_adapters.json
│   ├── cultural_context_routing_map.json
│   └── countries/             # SG、ID、TH、MX、BR、SA、TR、ZA
├── prompts/                   # 数据生成的 LLM 提示词模板
├── generate/                  # 数据生成流水线
│   ├── generator.py           # 核心样本生成器
│   ├── adversarial.py         # 对抗变体生成器
│   ├── prompt_builder.py      # 标签定义 → LLM 提示词转换
│   ├── taxonomy_loader.py     # 标签体系 JSON 加载器
│   ├── data_mgmt.py           # 数据管理与 CSV 导出
│   └── balance_data.py        # 数据均衡工具
├── train/                     # 训练与推理
│   ├── model.py               # LoRA 模型构建
│   ├── dataset.py             # 多标签数据集
│   ├── losses.py              # Focal Loss 变体
│   ├── trainer.py             # 训练循环（含早停）
│   ├── inference.py           # 模型推理
│   ├── run.sh                 # 训练启动脚本
│   └── config.py              # 训练配置加载
├── data_csv/                  # 训练数据（CSV，已纳入 git）
├── data_v2/                   # V2 重建训练数据
├── data_kaggle/               # 公开多语言测试集
├── train_config.yaml          # 训练超参数配置
├── config.yaml.example        # LLM API 配置模板
└── requirements.txt
```

## 严重度分级

| 级别 | 动作 |
|------|------|
| CRITICAL | 自动拦截并上报，通知法务 |
| HIGH | 自动拦截，进入人工审核队列 |
| MEDIUM | 隔离待审，低置信度降级为 FLAG |
| LOW | 仅标记，用于统计与趋势监控 |

## 标签命名规范

```
local_{ISO3166-1}_{描述名}
```

示例: `local_th_lese_majeste`、`local_id_sara_violation`、`local_za_severe_racism`

## 核心设计决策

1. **语言 ≠ 国家** — `cultural_context_routing_map.json` 解耦语言与地理位置。一条关于泰国君主制的英文推文同样会激活 LoRA-TH。

2. **边界案例是数据，不是后处理** — 每个本地标签都包含 `boundary_cases`（看起来像违规但实际不是的场景）。数据合成时对每个边界案例生成等量的正负样本。

3. **Regional Adapter 实现复用** — `th.religious_insult` 和 `id.sara_violation` 共享区域语义子空间。新增国家只需训练 Country Delta，继承区域语义。

4. **检测类型决定损失策略** — 不同标签识别难度不同。`keyword_sensitive` 用标准 CE，`contextual` 用 Focal Loss (γ=2)，`hybrid` 混合两种策略。
