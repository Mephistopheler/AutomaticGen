from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence


from bert_score import score as bertscore_score
from sacrebleu import corpus_bleu


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _compute_text_metrics_impl(
    predictions: List[str],
    references: List[str],
    bertscore_model_type: Optional[str] = 'bert-base-multilingual-cased',
    language: str = 'ru',
) -> Dict[str, Any]:


    results: Dict[str, Any] = {
        'num_examples': len(predictions),
    }

    bleu = corpus_bleu(predictions, [references])
    results['sacrebleu'] = float(bleu.score)
    results['sacrebleu_signature'] = str(bleu.format(signature=True)).split(' = ')[0]

    try:
        import evaluate

        rouge = evaluate.load('rouge')
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        results.update({
            'rouge1': float(rouge_scores.get('rouge1', 0.0)),
            'rouge2': float(rouge_scores.get('rouge2', 0.0)),
            'rougeL': float(rouge_scores.get('rougeL', 0.0)),
            'rougeLsum': float(rouge_scores.get('rougeLsum', 0.0)),
        })
    except Exception as exc:  # pragma: no cover
        results['rouge_error'] = str(exc)

    try:
        import evaluate

        meteor = evaluate.load('meteor')
        meteor_score = meteor.compute(predictions=predictions, references=references)
        results['meteor'] = float(meteor_score.get('meteor', 0.0))
    except Exception as exc:  # pragma: no cover
        results['meteor_error'] = str(exc)

    try:
        if bertscore_model_type:
            _, _, f1 = bertscore_score(
                cands=predictions,
                refs=references,
                model_type=bertscore_model_type,
                verbose=False,
                batch_size=8,
                lang=None,
            )
        else:
            _, _, f1 = bertscore_score(
                cands=predictions,
                refs=references,
                lang=language,
                verbose=False,
                batch_size=8,
            )
        results['bertscore_f1'] = float(f1.mean().item())
    except Exception as exc:  # pragma: no cover
        results['bertscore_error'] = str(exc)

    return results


def compute_text_metrics(
    predictions: List[str],
    references: List[str],
    bertscore_model_type: Optional[str] = 'bert-base-multilingual-cased',
    language: str = 'ru',
    languages: Optional[Sequence[str]] = None,
    bertscore_model_by_language: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError('Predictions and references must have the same length')
    if languages is not None and len(languages) != len(predictions):
        raise ValueError('Languages and predictions must have the same length')

    results = _compute_text_metrics_impl(
        predictions=predictions,
        references=references,
        bertscore_model_type=bertscore_model_type,
        language=language,
    )

    if not languages:
        return results

    per_language: Dict[str, Any] = {}
    for current_language in sorted(set(languages)):
        indexes = [index for index, item_language in enumerate(languages) if item_language == current_language]
        if not indexes:
            continue
        current_model_type = (bertscore_model_by_language or {}).get(current_language, bertscore_model_type)
        per_language[current_language] = _compute_text_metrics_impl(
            predictions=[predictions[index] for index in indexes],
            references=[references[index] for index in indexes],
            bertscore_model_type=current_model_type,
            language=current_language,
        )

    results['languages'] = sorted(set(languages))
    results['metrics_by_language'] = per_language
    return results