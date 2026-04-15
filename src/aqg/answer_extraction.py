from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class AnswerCandidate:
    text: str
    score: float
    kind: str


_CYRILLIC = r'\u0400-\u04FF'
_WORD = rf'[A-Za-z{_CYRILLIC}][A-Za-z{_CYRILLIC}\-]*'
_CAPITALIZED = rf'[A-Z\u0410-\u042F\u0401][A-Za-z{_CYRILLIC}\-]*'

_STOPWORDS = {
    'and',
    'are',
    'but',
    'for',
    'from',
    'have',
    'that',
    'the',
    'this',
    'was',
    'were',
    '\u0431\u0435\u0437',
    '\u0431\u044b\u043b',
    '\u0431\u044b\u043b\u0430',
    '\u0431\u044b\u043b\u0438',
    '\u0432\u0441\u0435',
    '\u0433\u0434\u0435',
    '\u0434\u043b\u044f',
    '\u0435\u0433\u043e',
    '\u0435\u0435',
    '\u043a\u0430\u043a',
    '\u043a\u043e\u0433\u0434\u0430',
    '\u043a\u0442\u043e',
    '\u043d\u0430\u0434',
    '\u043f\u043e\u0434',
    '\u043f\u0440\u0438',
    '\u0447\u0442\u043e',
    '\u044d\u0442\u043e',
}


def extract_answer_candidates(context: str, max_answers: int = 5) -> List[AnswerCandidate]:
    context = _normalize(context)
    candidates: List[AnswerCandidate] = []

    candidates.extend(_extract_quoted(context))
    candidates.extend(_extract_dates_and_numbers(context))
    candidates.extend(_extract_abbreviations(context))
    candidates.extend(_extract_capitalized_phrases(context))
    candidates.extend(_extract_keyword_phrases(context))

    unique = _deduplicate(candidates)
    unique.sort(key=lambda item: item.score, reverse=True)
    return unique[:max_answers]


def _normalize(text: str) -> str:
    return ' '.join(text.split())


def _extract_quoted(context: str) -> Iterable[AnswerCandidate]:
    for match in re.finditer(r'[\"\u00ab](.{3,80}?)[\"\u00bb]', context):
        text = _clean_candidate(match.group(1))
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=4.0, kind='quoted')


def _extract_dates_and_numbers(context: str) -> Iterable[AnswerCandidate]:
    month = rf'[{_CYRILLIC}]{{3,12}}'
    patterns = [
        rf'\b\d{{1,2}}\s+{month}\s+\d{{4}}\b',
        rf'\b\d{{4}}\s*(?:\u0433\.?|\u0433\u043e\u0434[ауе]?|\u0432\u0435\u043a[ауе]?|\u0432\u0432?\.?)\b',
        r'\b\d+(?:[,.]\d+)?\s*(?:%|km|m|cm|kg|g|mln|million|billion)\b',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, context, flags=re.IGNORECASE):
            text = _clean_candidate(match.group(0))
            if _is_valid_candidate(text):
                yield AnswerCandidate(text=text, score=3.7, kind='date_or_number')


def _extract_abbreviations(context: str) -> Iterable[AnswerCandidate]:
    for match in re.finditer(rf'\b[A-Z\u0410-\u042F\u0401]{{2,}}(?:-[A-Z\u0410-\u042F\u0401]{{2,}})?\b', context):
        text = _clean_candidate(match.group(0))
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=3.4, kind='abbreviation')


def _extract_capitalized_phrases(context: str) -> Iterable[AnswerCandidate]:
    pattern = rf'\b{_CAPITALIZED}(?:\s+{_CAPITALIZED}){{0,4}}\b'
    for match in re.finditer(pattern, context):
        text = _clean_candidate(match.group(0))
        if len(text.split()) == 1 and _is_sentence_initial(context, match.start()):
            continue
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=3.0, kind='proper_name')


def _extract_keyword_phrases(context: str) -> Iterable[AnswerCandidate]:
    words = re.findall(_WORD, context)
    seen = set()
    for size in (3, 2, 1):
        for start in range(0, max(len(words) - size + 1, 0)):
            phrase_words = words[start:start + size]
            if not phrase_words:
                continue
            if phrase_words[0].lower() in _STOPWORDS:
                continue
            if any(len(word) < 5 for word in phrase_words):
                continue
            text = _clean_candidate(' '.join(phrase_words))
            key = text.lower()
            if key in seen or not _is_valid_candidate(text):
                continue
            seen.add(key)
            yield AnswerCandidate(text=text, score=1.0 + size * 0.2, kind='keyword')


def _clean_candidate(text: str) -> str:
    return text.strip(' \t\r\n,.;:!?()[]{}')


def _is_sentence_initial(context: str, start: int) -> bool:
    prefix = context[:start].rstrip()
    return not prefix or prefix[-1] in '.!?'


def _is_valid_candidate(text: str) -> bool:
    if not 2 <= len(text) <= 90:
        return False
    if text.lower() in _STOPWORDS:
        return False
    return any(char.isalpha() or char.isdigit() for char in text)


def _deduplicate(candidates: Iterable[AnswerCandidate]) -> List[AnswerCandidate]:
    by_key = {}
    for candidate in candidates:
        key = candidate.text.lower()
        existing = by_key.get(key)
        if existing is None or candidate.score > existing.score:
            by_key[key] = candidate
    return list(by_key.values())
