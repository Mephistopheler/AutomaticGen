from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aqg.config_utils import load_config


def load_manifest(path: str | Path, groups: Iterable[str]) -> List[Dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    experiments: List[Dict[str, Any]] = []
    for group in groups:
        experiments.extend(manifest.get(group, []))
    return experiments


def run_command(command: List[str], dry_run: bool = False) -> None:
    print(' '.join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run AQG research experiments')
    parser.add_argument('--manifest', default='configs/experiments.yaml')
    parser.add_argument('--groups', nargs='+', default=['fast_experiments'])
    parser.add_argument('--ids', nargs='*', help='Run only selected experiment ids')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    experiments = load_manifest(args.manifest, args.groups)
    if args.ids:
        selected = set(args.ids)
        experiments = [experiment for experiment in experiments if experiment['id'] in selected]

    if not experiments:
        raise ValueError('No experiments selected')

    for experiment in experiments:
        config_path = experiment['config']
        config = load_config(config_path)
        checkpoint = Path(config['output_dir']) / 'checkpoints' / 'best'

        print(f'\n=== {experiment["id"]}: {experiment.get("title", "")} ===')
        if not args.skip_train:
            run_command([sys.executable, 'train.py', '--config', config_path], dry_run=args.dry_run)

        if not args.skip_eval:
            run_command(
                [
                    sys.executable,
                    'evaluate_model.py',
                    '--config',
                    config_path,
                    '--checkpoint',
                    str(checkpoint),
                ],
                dry_run=args.dry_run,
            )

    if args.aggregate:
        run_command(
            [
                sys.executable,
                'scripts/aggregate_results.py',
                '--manifest',
                args.manifest,
                '--groups',
                *args.groups,
            ],
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
