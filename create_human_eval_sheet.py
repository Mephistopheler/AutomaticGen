from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.aqg.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description='Create human evaluation CSV from predictions JSONL')
    parser.add_argument('--predictions', required=True, help='Path to predictions.jsonl')
    parser.add_argument('--output', required=True, help='Path to CSV file')
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    df = pd.DataFrame(rows)
    for column in [
        'fluency_score_1_5',
        'relevance_score_1_5',
        'answerability_score_1_5',
        'pedagogical_value_score_1_5',
        'overall_score_1_5',
        'annotator_id',
        'notes',
    ]:
        if column not in df.columns:
            df[column] = ''

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Human evaluation sheet saved to: {out_path}')


if __name__ == '__main__':
    main()
