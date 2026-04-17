# AQG Research Plan

## Goal

The project compares answer-aware question generation models under a shared pipeline:

```text
context + answer -> question
```

For user-facing generation from raw text, the system uses the same architecture as the earlier English prototype:

```text
raw text -> answer candidate extraction -> context + answer -> generated question
```

## Experiment Groups

Full experiments are listed in `configs/experiments.yaml` under `experiments`:

| id | train | eval | model | scenario |
| --- | --- | --- | --- | --- |
| `t5_squad_en` | SQuAD EN | SQuAD EN | `t5-base` | English monolingual baseline |
| `mt5_squad_en` | SQuAD EN | SQuAD EN | `google/mt5-base` | Multilingual model on English |
| `mt5_sberquad_ru` | SberQuAD RU | SberQuAD RU | `google/mt5-base` | Multilingual model on Russian |
| `rut5_sberquad_ru` | SberQuAD RU | SberQuAD RU | `ai-forever/ruT5-base` | Russian-specialized model |
| `mt5_en_to_ru_transfer` | SQuAD EN | SberQuAD RU | `google/mt5-base` | Cross-lingual transfer |

Fast experiments are CPU-friendly smoke tests and are not final research results.

## Commands

Run one fast experiment:

```powershell
python scripts/run_experiments.py --groups fast_experiments --ids t5_squad_en_fast --aggregate
```

Run all fast experiments:

```powershell
python scripts/run_experiments.py --groups fast_experiments --aggregate
```

Run one full experiment:

```powershell
python scripts/run_experiments.py --groups experiments --ids rut5_sberquad_ru --aggregate
```

Aggregate existing metrics without retraining:

```powershell
python scripts/aggregate_results.py --groups experiments fast_experiments
```

Outputs:

- `outputs/research_summary.csv`
- `outputs/research_summary.md`

## Interpretation

The main comparison should focus on:

- English monolingual quality: `t5_squad_en` vs `mt5_squad_en`
- Russian monolingual quality: `mt5_sberquad_ru` vs `rut5_sberquad_ru`
- Cross-lingual degradation: `mt5_squad_en` or `t5_squad_en` vs `mt5_en_to_ru_transfer`
- Practical text-to-questions pipeline: answer extraction quality plus QG model quality
