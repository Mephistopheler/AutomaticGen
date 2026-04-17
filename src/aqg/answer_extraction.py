from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class AnswerCandidate:
    text: str
    score: float
    kind: str
    sentence: str = ''


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
    '\u0432',
    '\u0432\u0441\u0435',
    '\u0433\u0434\u0435',
    '\u0434\u043b\u044f',
    '\u0435\u0433\u043e',
    '\u0435\u0435',
    '\u0438',
    '\u0438\u0437',
    '\u043a',
    '\u043a\u0430\u043a',
    '\u043a\u043e\u0433\u0434\u0430',
    '\u043a\u0442\u043e',
    '\u043d\u0430',
    '\u043d\u0430\u0434',
    '\u043e',
    '\u043e\u0431',
    '\u043f\u043e',
    '\u043f\u043e\u0434',
    '\u043f\u0440\u0438',
    '\u0441',
    '\u0447\u0442\u043e',
    '\u044d\u0442\u043e',
}


def extract_answer_candidates(context: str, max_answers: int = 5) -> List[AnswerCandidate]:
    context = _normalize(context)
    candidates: List[AnswerCandidate] = []

    for sentence in _split_sentences(context):
        candidates.extend(_extract_quoted(sentence))
        candidates.extend(_extract_dates_and_numbers(sentence))
        candidates.extend(_extract_abbreviations(sentence))
        candidates.extend(_extract_capitalized_phrases(sentence))
        candidates.extend(_extract_keyword_phrases(sentence))

    unique = _deduplicate(candidates)
    unique.sort(key=lambda item: item.score, reverse=True)
    return unique[:max_answers]


def _normalize(text: str) -> str:
    return ' '.join(text.split())


def _split_sentences(context: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', context)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extract_quoted(sentence: str) -> Iterable[AnswerCandidate]:
    for match in re.finditer(r'[\"\u00ab](.{3,80}?)[\"\u00bb]', sentence):
        text = _clean_candidate(match.group(1))
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=4.0, kind='quoted', sentence=sentence)


def _extract_dates_and_numbers(sentence: str) -> Iterable[AnswerCandidate]:
    month = rf'[{_CYRILLIC}]{{3,12}}'
    patterns = [
        rf'\b\d{{1,2}}\s+{month}\s+\d{{4}}\b',
        rf'\b\d{{4}}\s*(?:\u0433\.?|\u0433\u043e\u0434[\u0430\u0443\u0435]?|\u0432\u0435\u043a[\u0430\u0443\u0435]?|\u0432\u0432?\.?)\b',
        r'\b\d+(?:[,.]\d+)?\s*(?:%|km|m|cm|kg|g|mln|million|billion)\b',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            text = _clean_candidate(match.group(0))
            if _is_valid_candidate(text):
                yield AnswerCandidate(text=text, score=3.7, kind='date_or_number', sentence=sentence)


def _extract_abbreviations(sentence: str) -> Iterable[AnswerCandidate]:
    pattern = rf'\b[A-Z\u0410-\u042F\u0401]{{2,}}(?:-[A-Z\u0410-\u042F\u0401]{{2,}})?\b'
    for match in re.finditer(pattern, sentence):
        text = _clean_candidate(match.group(0))
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=3.4, kind='abbreviation', sentence=sentence)


def _extract_capitalized_phrases(sentence: str) -> Iterable[AnswerCandidate]:
    pattern = rf'\b{_CAPITALIZED}(?:\s+{_CAPITALIZED}){{0,4}}\b'
    for match in re.finditer(pattern, sentence):
        text = _clean_candidate(match.group(0))
        if len(text.split()) == 1 and _is_sentence_initial(sentence, match.start()):
            continue
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=3.0, kind='proper_name', sentence=sentence)


def _extract_keyword_phrases(sentence: str) -> Iterable[AnswerCandidate]:
    words = re.findall(_WORD, sentence)
    seen = set()
    for size in (3, 2, 1):
        for start in range(0, max(len(words) - size + 1, 0)):
            phrase_words = words[start:start + size]
            if not phrase_words:
                continue
            if phrase_words[0].lower() in _STOPWORDS:
                continue
            if any(word.lower() in _STOPWORDS or len(word) < 5 for word in phrase_words):
                continue
            text = _clean_candidate(' '.join(phrase_words))
            key = text.lower()
            if key in seen or not _is_valid_candidate(text):
                continue
            seen.add(key)
            yield AnswerCandidate(text=text, score=1.0 + size * 0.2, kind='keyword', sentence=sentence)


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
