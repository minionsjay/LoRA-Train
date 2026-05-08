# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

文化感知的跨语言安全对齐系统 — Content safety classifier using a **Frozen Base + Regional Adapter + Country LoRA** architecture. The system detects policy violations across 8 countries (SG, ID, TH, MX, BR, SA, TR, ZA) with culture-specific sensitivity.

## Architecture: Three-Layer Taxonomy

```
Base Violations (6 global labels)
  └── Regional Adapters (SEA / LATAM / MENA_AFRICA)
       └── Country LoRA Deltas (2-3 local labels per country)
```

- **Base**: `taxonomy/base_violations.json` — Global zero-tolerance categories (CSAM, terrorism, pornography, etc.)
- **Regional Adapters**: `taxonomy/regional_adapters.json` — Shared semantic subspaces within geographic regions
- **Country Deltas**: `taxonomy/countries/{code}.json` — Local cultural violations with triggers, adversarial variants, and boundary cases

## Key Design Decisions

1. **Detection types drive training strategy**: Each label has a `detection_type` field — `keyword_sensitive` (γ=0, standard CE), `contextual` (γ=2, Focal Loss + Hard Mining), or `hybrid` (two-stage, 3:7 loss weighting).

2. **Language ≠ country for routing**: The `cultural_context_routing_map.json` decouples language from geography. A Thai-context English tweet about the monarchy still activates LoRA-TH. Stage 1: lightweight Context Detector outputs cultural context. Stage 2: LoRA Dispatcher mounts the correct adapter.

3. **Boundary cases are first-class**: Every local label includes `boundary_cases` — scenarios that look like violations but aren't (e.g., academic discussion of Lese-Majeste law ≠ violation). Data synthesis must generate equal positive/negative samples for each boundary case.

4. **Regional adapters enable reuse**: `th.religious_insult` and `id.sara_violation` (religious dimension) share a regional subspace. New countries only train the Country Delta, inheriting regional semantics.

## Directory Structure

```
taxonomy/
├── schema.json                         # JSON Schema v2 for all taxonomy files
├── base_violations.json                # 6 global violation labels
├── regional_adapters.json              # SEA / LATAM / MENA_AFRICA shared subspaces
├── cultural_context_routing_map.json   # Two-stage routing table for LoRA dispatch
└── countries/
    ├── sg.json    # Singapore — racial harmony, Singlish vulgarity
    ├── id.json    # Indonesia — SARA, pornography slang (Bahasa Gaul)
    ├── th.json    # Thailand — Lese-Majeste, political instigation
    ├── mx.json    # Mexico — narco culture, gender violence
    ├── br.json    # Brazil — political extremism, structural racism
    ├── sa.json    # Saudi Arabia — blasphemy, immorality, anti-state
    ├── tr.json    # Turkey — insulting state/Atatürk, separatism
    └── za.json    # South Africa — severe racism (K-word), xenophobia
```

## Taxonomy Label Naming Convention

```
local_{ISO3166-1}_{descriptive_name}
```

Examples: `local_th_lese_majeste`, `local_id_sara_violation`, `local_za_severe_racism`

## Severity Levels

| Level | Action |
|-------|--------|
| CRITICAL | AUTO_BLOCK_AND_REPORT — zero tolerance, notify legal |
| HIGH | AUTO_BLOCK — enter human review queue |
| MEDIUM | QUARANTINE — isolate for review, downgrade to FLAG if low confidence |
| LOW | FLAG — statistics and trend monitoring only |
