from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aqg.config_utils import load_config


METRIC_KEYS = [
    'sacrebleu',
    'rouge1',
    'rouge2',
    'rougeL',
    'rougeLsum',
    'meteor',
    'bertscore_f1',
]


def load_manifest(path: str | Path, groups: Iterable[str]) -> List[Dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    experiments: List[Dict[str, Any]] = []
    for group in groups:
        experiments.extend(manifest.get(group, []))
    return experiments


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_row(experiment: Dict[str, Any]) -> Dict[str, Any]:
    config = load_config(experiment['config'])
    output_dir = Path(config['output_dir'])
    metrics = read_json(output_dir / 'evaluation' / 'metrics.json')
    summary = read_json(output_dir / 'logs' / 'training_summary.json')

    row: Dict[str, Any] = {
        'id': experiment['id'],
        'title': experiment.get('title', ''),
        'scenario': experiment.get('scenario', ''),
        'model_family': experiment.get('model_family', ''),
        'model_name': experiment.get('model_name', config['model'].get('name', '')),
        'train_language': experiment.get('train_language', ''),
        'eval_language': experiment.get('eval_language', ''),
        'dataset': experiment.get('dataset', ''),
        'config': experiment['config'],
        'output_dir': str(output_dir),
        'checkpoint': str(output_dir / 'checkpoints' / 'best'),
        'best_val_loss': summary.get('best_val_loss', ''),
        'elapsed_hms': summary.get('elapsed_hms', ''),
        'device': summary.get('device', ''),
        'num_examples': metrics.get('num_examples', ''),
    }

    for key in METRIC_KEYS:
        row[key] = metrics.get(key, '')
    return row


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'id',
        'scenario',
        'model_name',
        'train_language',
        'eval_language',
        'sacrebleu',
        'rougeL',
        'meteor',
        'bertscore_f1',
        'best_val_loss',
    ]
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for row in rows:
        values = [_format_value(row.get(column, '')) for column in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate AQG experiment metrics')
    parser.add_argument('--manifest', default='configs/experiments.yaml')
    parser.add_argument('--groups', nargs='+', default=['experiments', 'fast_experiments'])
    parser.add_argument('--output_csv', default='outputs/research_summary.csv')
    parser.add_argument('--output_md', default='outputs/research_summary.md')
    args = parser.parse_args()

    experiments = load_manifest(args.manifest, args.groups)
    rows = [build_row(experiment) for experiment in experiments]
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))

    print(f'Wrote CSV: {args.output_csv}')
    print(f'Wrote Markdown: {args.output_md}')


if __name__ == '__main__':
    main()
