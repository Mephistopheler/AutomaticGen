from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.aqg.answer_extraction import extract_answer_candidates
from src.aqg.config_utils import load_config
from src.aqg.transformers_text import AutoModelForSeq2SeqLM, AutoTokenizer
from src.aqg.utils import get_device, save_jsonl


def build_source(template: str, context: str, answer: str) -> str:
    return template.format(context=' '.join(context.split()), answer=' '.join(answer.split()))


def clean_question(text: str) -> str:
    text = text.replace('<extra_id_0>', '').strip()
    text = ' '.join(text.split())
    return text


def is_valid_question(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) < 5:
        return False
    if not any(char.isalpha() for char in normalized):
        return False
    if normalized.lower() in {'context:', 'answer:', 'generate question:'}:
        return False
    return True


def fallback_question(answer: str, kind: str) -> str:
    if kind == 'date_or_number':
        return f'\u041a\u043e\u0433\u0434\u0430 \u0443\u043f\u043e\u043c\u0438\u043d\u0430\u0435\u0442\u0441\u044f {answer}?'
    if kind == 'proper_name':
        return f'\u041a\u0442\u043e \u0438\u043b\u0438 \u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 {answer}?'
    if kind == 'abbreviation':
        return f'\u0427\u0442\u043e \u043e\u0431\u043e\u0437\u043d\u0430\u0447\u0430\u0435\u0442 {answer}?'
    return f'\u0427\u0442\u043e \u0441\u043a\u0430\u0437\u0430\u043d\u043e \u043e {answer}?'


def generate_question(
    model: Any,
    tokenizer: Any,
    source: str,
    device: Any,
    cfg: Dict[str, Any],
) -> str:
    encoded = tokenizer(
        source,
        return_tensors='pt',
        truncation=True,
        max_length=int(cfg['data']['max_source_length']),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=cfg['generation'].get('max_new_tokens', 64),
        num_beams=cfg['generation'].get('num_beams', 4),
        do_sample=cfg['generation'].get('do_sample', False),
        temperature=cfg['generation'].get('temperature', 1.0),
        top_p=cfg['generation'].get('top_p', 1.0),
    )
    return clean_question(tokenizer.decode(generated[0], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate questions from text with automatic answer selection')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--context', help='Single source text')
    parser.add_argument('--input_txt', help='Path to UTF-8 text file')
    parser.add_argument('--output_jsonl', help='Optional path for generated rows')
    parser.add_argument('--max_answers', type=int, default=5)
    args = parser.parse_args()

    if args.input_txt:
        context = Path(args.input_txt).read_text(encoding='utf-8')
    elif args.context:
        context = args.context
    else:
        raise ValueError('Provide --context or --input_txt')

    cfg = load_config(args.config)
    template = cfg['data']['source_template']
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    candidates = extract_answer_candidates(context, max_answers=args.max_answers)
    if not candidates:
        raise ValueError('No answer candidates found in the input text')

    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        source = build_source(template, context=context, answer=candidate.text)
        question = generate_question(model=model, tokenizer=tokenizer, source=source, device=device, cfg=cfg)
        used_fallback = False
        raw_question = question
        if not is_valid_question(question):
            question = fallback_question(candidate.text, candidate.kind)
            used_fallback = True
        row = {
            'rank': index,
            'answer_text': candidate.text,
            'answer_kind': candidate.kind,
            'answer_score': candidate.score,
            'generated_question': question,
            'model_question': raw_question,
            'used_fallback': used_fallback,
        }
        rows.append(row)
        print(f'[{index}] ANSWER ({candidate.kind}): {candidate.text}')
        if used_fallback:
            print(f'[{index}] MODEL QUESTION: {raw_question or "<empty>"}')
            print(f'[{index}] FALLBACK QUESTION: {question}')
        else:
            print(f'[{index}] QUESTION: {question}')
        print('-' * 80)

    if args.output_jsonl:
        save_jsonl(rows, args.output_jsonl)


if __name__ == '__main__':
    main()
