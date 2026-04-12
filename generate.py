from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.aqg.config_utils import load_config
from src.aqg.transformers_text import AutoModelForSeq2SeqLM, AutoTokenizer
from src.aqg.utils import get_device


def build_source(template: str, context: str, answer: str) -> str:
    return template.format(context=' '.join(context.split()), answer=' '.join(answer.split()))


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate questions for custom examples')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--context', help='Single context string')
    parser.add_argument('--answer', help='Single answer string')
    parser.add_argument('--input_jsonl', help='Path to JSONL with fields: context, answer_text')
    args = parser.parse_args()

    cfg = load_config(args.config)
    template = cfg['data']['source_template']
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    items: List[dict] = []
    if args.input_jsonl:
        with Path(args.input_jsonl).open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        if not args.context or not args.answer:
            raise ValueError('Either provide --input_jsonl or both --context and --answer')
        items.append({'context': args.context, 'answer_text': args.answer})

    for idx, item in enumerate(items, start=1):
        source = build_source(template, context=item['context'], answer=item['answer_text'])
        encoded = tokenizer(source, return_tensors='pt', truncation=True, max_length=int(cfg['data']['max_source_length']))
        encoded = {k: v.to(device) for k, v in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=cfg['generation'].get('max_new_tokens', 64),
            num_beams=cfg['generation'].get('num_beams', 4),
            do_sample=cfg['generation'].get('do_sample', False),
            temperature=cfg['generation'].get('temperature', 1.0),
            top_p=cfg['generation'].get('top_p', 1.0),
        )
        question = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        print(f'[{idx}] ANSWER: {item["answer_text"]}')
        print(f'[{idx}] QUESTION: {question}')
        print('-' * 80)


if __name__ == '__main__':
    main()
