from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset


@dataclass
class NormalizedExample:
    example_id: str
    context: str
    answer_text: str
    question: str
    source: str
    target: str
    dataset_name: str
    language: str


def _normalize_text(value: Any) -> str:
    return ' '.join(_repair_mojibake(str(value or '')).split())


def _repair_mojibake(text: str) -> str:
    if '\u0420' not in text and '\u0421' not in text:
        return text

    raw = bytearray()
    for char in text:
        codepoint = ord(char)
        if codepoint <= 0xFF:
            raw.append(codepoint)
            continue
        try:
            raw.extend(char.encode('cp1251'))
        except UnicodeEncodeError:
            return text

    try:
        repaired = raw.decode('utf-8')
    except UnicodeDecodeError:
        return text
    if repaired.count('\ufffd') > text.count('\ufffd'):
        return text
    return repaired


def build_source_text(template: str, context: str, answer: str) -> str:
    return template.format(context=context, answer=answer)


def _from_squad_like_row(row: Dict[str, Any], dataset_name: str, language: str, template: str) -> Optional[NormalizedExample]:
    context = _normalize_text(row.get('context', ''))
    question = _normalize_text(row.get('question', ''))

    answers = row.get('answers')
    answer_text = ''
    if isinstance(answers, dict):
        texts = answers.get('text') or []
        if texts:
            answer_text = _normalize_text(texts[0])
    elif isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, dict):
            answer_text = _normalize_text(first.get('text', ''))
        else:
            answer_text = _normalize_text(first)

    if not context or not question or not answer_text:
        return None

    example_id = str(row.get('id', '')) or f'{dataset_name}-{abs(hash((context, question, answer_text)))}'
    source = build_source_text(template, context=context, answer=answer_text)
    return NormalizedExample(
        example_id=example_id,
        context=context,
        answer_text=answer_text,
        question=question,
        source=source,
        target=question,
        dataset_name=dataset_name,
        language=language,
    )


def _flatten_sberquad_like(dataset: Dataset, dataset_name: str, language: str, template: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for row in dataset:
        context = _normalize_text(row.get('context', ''))
        title = _normalize_text(row.get('title', ''))
        qas = row.get('qas') or []
        for qa in qas:
            question = _normalize_text(qa.get('question', ''))
            answers = qa.get('answers') or []
            answer_text = ''
            if answers:
                first = answers[0]
                if isinstance(first, dict):
                    answer_text = _normalize_text(first.get('text', ''))
                else:
                    answer_text = _normalize_text(first)
            if not context or not question or not answer_text:
                continue
            source = build_source_text(template, context=context, answer=answer_text)
            rows.append(
                {
                    'example_id': str(qa.get('id', '')) or f'{dataset_name}-{len(rows)}',
                    'context': context,
                    'answer_text': answer_text,
                    'question': question,
                    'source': source,
                    'target': question,
                    'dataset_name': dataset_name,
                    'language': language,
                    'title': title,
                }
            )
    return Dataset.from_list(rows)


def normalize_dataset(dataset: Dataset, dataset_name: str, language: str, template: str) -> Dataset:
    if 'qas' in dataset.column_names:
        return _flatten_sberquad_like(dataset, dataset_name, language, template)

    rows: List[Dict[str, Any]] = []
    for row in dataset:
        example = _from_squad_like_row(row, dataset_name=dataset_name, language=language, template=template)
        if example is None:
            continue
        rows.append(
            {
                'example_id': example.example_id,
                'context': example.context,
                'answer_text': example.answer_text,
                'question': example.question,
                'source': example.source,
                'target': example.target,
                'dataset_name': example.dataset_name,
                'language': example.language,
            }
        )
    return Dataset.from_list(rows)


def load_one_spec(spec: Dict[str, Any], template: str) -> Dataset:
    spec_type = spec.get('type', 'hf')
    dataset_name = spec.get('dataset_name') or spec.get('path') or spec.get('data_files', 'local')
    language = spec.get('language', 'unknown')

    if spec_type == 'hf':
        dataset = load_dataset(spec['path'], spec.get('name'), split=spec['split'])
        normalized = normalize_dataset(dataset, dataset_name=dataset_name, language=language, template=template)
        return _limit_dataset(normalized, spec.get('max_samples'))

    if spec_type == 'json':
        dataset = load_dataset('json', data_files=spec['data_files'], split=spec.get('split', 'train'))
        normalized = normalize_dataset(dataset, dataset_name=dataset_name, language=language, template=template)
        return _limit_dataset(normalized, spec.get('max_samples'))

    if spec_type == 'csv':
        dataset = load_dataset('csv', data_files=spec['data_files'], split=spec.get('split', 'train'))
        normalized = normalize_dataset(dataset, dataset_name=dataset_name, language=language, template=template)
        return _limit_dataset(normalized, spec.get('max_samples'))

    raise ValueError(f'Unsupported dataset spec type: {spec_type}')


def _limit_dataset(dataset: Dataset, max_samples: Any) -> Dataset:
    if max_samples is None:
        return dataset
    limit = min(int(max_samples), len(dataset))
    return dataset.select(range(limit))


def load_dataset_group(specs: Iterable[Dict[str, Any]], template: str) -> Dataset:
    datasets_list = [load_one_spec(spec, template=template) for spec in specs]
    if not datasets_list:
        raise ValueError('Dataset specs are empty')
    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(datasets_list)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, max_source_length: int, max_target_length: int) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            batch['source'],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch['target'],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    keep_columns = dataset.column_names
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=keep_columns)
    tokenized._keep_in_memory = False  # type: ignore[attr-defined]
    tokenized.info.description = f'Tokenized columns preserved from: {keep_columns}'
    return tokenized
