from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.aqg.config_utils import load_config, save_json
from src.aqg.data import load_dataset_group, tokenize_dataset
from src.aqg.metrics import compute_text_metrics
from src.aqg.transformers_text import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from src.aqg.utils import get_device, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate AQG model')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--split', default='evaluation_datasets', help='Config key with dataset specs')
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_specs = cfg['data'][args.split]
    template = cfg['data']['source_template']
    raw_dataset = load_dataset_group(dataset_specs, template=template)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    device = get_device()
    model.to(device)

    tokenized = tokenize_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        max_source_length=int(cfg['data']['max_source_length']),
        max_target_length=int(cfg['data']['max_target_length']),
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(
        tokenized,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        collate_fn=collator,
    )

    predictions: List[str] = []
    model.eval()
    for batch in tqdm(dataloader, desc='Evaluating'):
        batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        generated = model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_new_tokens=cfg['generation'].get('max_new_tokens', 64),
            num_beams=cfg['generation'].get('num_beams', 4),
            do_sample=cfg['generation'].get('do_sample', False),
            temperature=cfg['generation'].get('temperature', 1.0),
            top_p=cfg['generation'].get('top_p', 1.0),
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend([text.strip() for text in decoded])

    references = [row['target'] for row in raw_dataset]
    metrics = compute_text_metrics(
        predictions=predictions,
        references=references,
        bertscore_model_type=cfg['evaluation'].get('bertscore_model_type'),
        language=cfg['evaluation'].get('language', 'ru'),
    )

    out_dir = Path(cfg['output_dir']) / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / 'metrics.json')

    rows = []
    for row, pred in zip(raw_dataset, predictions):
        rows.append(
            {
                'example_id': row['example_id'],
                'dataset_name': row['dataset_name'],
                'context': row['context'],
                'answer_text': row['answer_text'],
                'reference_question': row['target'],
                'generated_question': pred,
            }
        )
    save_jsonl(rows, out_dir / 'predictions.jsonl')

    print('Metrics saved to:', out_dir / 'metrics.json')
    print(metrics)


if __name__ == '__main__':
    main()
