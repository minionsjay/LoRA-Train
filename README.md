# Culture-Aware Cross-Lingual Safety Alignment

A content safety classifier using a **Frozen Base + Regional Adapter + Country LoRA** architecture. Detects policy violations across 8 countries (SG, ID, TH, MX, BR, SA, TR, ZA) with culture-specific sensitivity.

## Architecture

```
User Text
  │
  ├── Stage 1: Cultural Context Detector (~10M)
  │     └── Output: cultural_context identifier
  │
  ├── Stage 2: LoRA Dynamic Router
  │     └── Mounts country-specific LoRA modules based on cultural context
  │
  └── Stage 3: Classification
        ├── Base Classifier — 5 global labels shared across all countries
        └── Local Classifier — 2-3 local labels per country (17 total)
```

**Model**: Frozen `xlm-roberta-base` (270M) + LoRA Adapter (r=16, ~2M trainable) + Multi-label Classification Head

## Three-Layer Taxonomy

```
Base Violations (5 global labels)
  └── Regional Adapters (SEA / LATAM / MENA_AFRICA)
       └── Country LoRA Deltas (2-3 local labels per country)
```

### Global Labels (5, shared across all countries)

| # | Label | Severity | Detection | Description |
|---|-------|----------|-----------|-------------|
| 1 | `base_violence_dangerous_behavior` | HIGH | contextual | Inciting violence, weapons, extreme gore |
| 2 | `base_hate_speech_harassment` | HIGH | hybrid | Hate speech, systematic harassment, doxxing |
| 3 | `base_sexual_content_child_safety` | CRITICAL | hybrid | CSAM, sexualized minors, explicit content |
| 4 | `base_self_harm_suicide` | HIGH | contextual | Suicide methods, self-harm glorification |
| 5 | `base_spam_deceptive_practices` | LOW | keyword_sensitive | Phishing, scams, pyramid schemes, spam |

### Country-Specific Labels (17 total)

| Country | Labels | Key Concerns |
|---------|--------|-------------|
| SG | `local_sg_racial_religious_harmony`, `local_sg_vulgarity_singlish` | Racial harmony, Singlish profanity |
| ID | `local_id_sara_violation`, `local_id_pornography_slang` | SARA principle, Bahasa Gaul slurs |
| TH | `local_th_lese_majeste`, `local_th_political_instigation` | Lèse-majesté (§112), political instigation |
| MX | `local_mx_narco_culture`, `local_mx_gender_violence` | Cartel glorification, feminicide |
| BR | `local_br_political_extremism`, `local_br_structural_racism` | Coup rhetoric, anti-favela racism |
| SA | `local_sa_blasphemy_anti_islam`, `local_sa_immorality_lgbtq`, `local_sa_anti_state` | Blasphemy, morality, anti-royal |
| TR | `local_tr_insulting_state`, `local_tr_separatism_terror` | State/Atatürk insults (TCK 299/301), PKK/FETÖ |
| ZA | `local_za_severe_racism`, `local_za_xenophobia` | K-word, xenophobic violence |

## Training Data

### Statistics

| Country | Total | Base Pos | Local Pos | Adversarial | Negatives | Labels |
|---------|-------|----------|-----------|-------------|-----------|--------|
| SA | 1,581 | 550 | 291 | 199 | 322 | 8 |
| ID | 1,434 | 550 | 259 | 438 | 187 | 7 |
| TR | 1,376 | 550 | 229 | 237 | 228 | 7 |
| MX | 1,358 | 550 | 373 | 45 | 279 | 7 |
| BR | 1,234 | 550 | 296 | 240 | 154 | 7 |
| ZA | 1,252 | 550 | 329 | 240 | 149 | 7 |
| SG | 1,147 | 550 | 215 | 240 | 136 | 7 |
| TH | 1,102 | 550 | 203 | 30 | 204 | 7 |
| **Total** | **10,484** | **4,400** | **2,195** | **1,669** | **1,659** | **56** |

### Sample Types

- **Positive**: Violations that teach the model what to catch
- **Adversarial**: Obfuscated variants (homoglyphs, zero-width chars, emoji substitution)
- **Negative**: Boundary cases — text that looks like a violation but isn't (e.g., academic discussion)

### Data Sources

Generated via GPT-4o-mini and Gemini 2.5 Flash, validated against per-country taxonomy definitions.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp config.yaml.example config.yaml
# Edit config.yaml with your LLM API key and endpoint

# 3. Train a single-country model
bash train/run.sh --countries TH --epochs 5

# 4. Full training across all 8 countries
bash train/run.sh
```

## Usage

### Generate Training Data

```bash
# Generate samples for a specific label
python -m generate --labels local_th_lese_majeste --force

# Generate base violation samples
python -m generate.data_mgmt generate-base --countries TH --samples-per-label 50

# Import external CSV data
python -m generate.data_mgmt add --file new_samples.csv

# Export all data to CSV
python -m generate.data_mgmt export-csv
```

### Run Inference

```bash
# Test a trained model
python -m train.inference --model trained_models/best_model --text "input text here"
```

### Benchmark

```bash
# Generate benchmark dataset
python -m generate.gen_benchmark
```

## Detection Types & Training Strategy

| Detection Type | Loss Function | γ | Strategy |
|----------------|---------------|----|----------|
| `keyword_sensitive` | Standard Cross Entropy | 0 | Base CE loss |
| `contextual` | Focal Loss | 2 | Heavy focus on hard examples |
| `hybrid` | Two-stage Loss | 1-2 | 3:7 CE:Focal weighting |

## Project Structure

```
├── taxonomy/                  # Violation taxonomy (schema, labels, routing)
│   ├── schema.json
│   ├── base_violations.json
│   ├── regional_adapters.json
│   ├── cultural_context_routing_map.json
│   └── countries/             # SG, ID, TH, MX, BR, SA, TR, ZA
├── prompts/                   # LLM prompt templates for data generation
├── generate/                  # Data generation pipeline
│   ├── generator.py           # Core sample generator
│   ├── adversarial.py         # Adversarial variant generator
│   ├── prompt_builder.py      # Taxonomy → prompt conversion
│   ├── taxonomy_loader.py     # Taxonomy JSON loader
│   ├── data_mgmt.py           # Data management & CSV export
│   └── balance_data.py        # Data balancing utilities
├── train/                     # Training & inference
│   ├── model.py               # LoRA model builder
│   ├── dataset.py             # Multi-label dataset
│   ├── losses.py              # Focal loss variants
│   ├── trainer.py             # Training loop with early stopping
│   ├── inference.py           # Model inference
│   ├── run.sh                 # Training launcher
│   └── config.py              # Training config loader
├── data_csv/                  # Training data (CSV, checked into git)
├── data_v2/                   # Rebuilt V2 training data
├── data_kaggle/               # Public multilingual test sets
├── train_config.yaml          # Training hyperparameters
├── config.yaml.example        # LLM API config template
└── requirements.txt
```

## Severity Levels

| Level | Action |
|-------|--------|
| CRITICAL | Auto-block + report, notify legal |
| HIGH | Auto-block, enter human review queue |
| MEDIUM | Quarantine, isolate for review |
| LOW | Flag for statistics and trend monitoring |

## Naming Convention

```
local_{ISO3166-1}_{descriptive_name}
```

Example: `local_th_lese_majeste`, `local_id_sara_violation`, `local_za_severe_racism`

## Key Design Decisions

1. **Language ≠ country** — The `cultural_context_routing_map.json` decouples language from geography. A Thai-context English tweet about the monarchy still activates LoRA-TH.

2. **Boundary cases are data, not postprocessing** — Each local label includes `boundary_cases` (scenarios that look like violations but aren't). Synthetic data generates equal positive/negative samples for each boundary case.

3. **Regional adapters enable reuse** — `th.religious_insult` and `id.sara_violation` share a regional subspace. New countries only train country deltas, inheriting regional semantics.

4. **Detection type drives loss** — Labels aren't equally hard. `keyword_sensitive` uses standard CE, `contextual` uses Focal Loss (γ=2), `hybrid` blends both strategies.
