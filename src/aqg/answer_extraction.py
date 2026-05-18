from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class AnswerCandidate:
    text: str
    score: float
    kind: str
    sentence: str = ''


_LETTER = r'[^\W\d_]'
_WORD = rf'{_LETTER}(?:[^\W_]|-)*'
_CAPITALIZED = rf'[A-ZА-ЯЁ](?:[^\W_]|-)*'
_SUPPORTED_SPACY_MODELS = {
    'en': 'en_core_web_sm',
    'ru': 'ru_core_news_sm',
}
_SINGLE_TOKEN_ENTITY_KINDS = {
    'spacy_person',
    'spacy_per',
    'spacy_gpe',
    'spacy_loc',
    'spacy_norp',
}
_CONTENT_SPAN_KINDS = {
    'spacy_noun_chunk',
    'spacy_dependency_span',
    'spacy_pos_phrase',
    'keyword',
}
_PARENTHETICAL_RE = re.compile(r'\([^)]*\)')
_ABBREVIATION_DOT_RE = re.compile(
    r'\b(фр|нид|англ|лат|рус|нем|исп|итал|см|рис|табл|стр|г|гг|т|д|е|им|ул|'
    r'Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.',
    flags=re.IGNORECASE,
)

_STOPWORDS = {
    # A compact fallback list for heuristic extraction. spaCy extraction relies
    # on each loaded language model's own token.is_stop values.
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'but',
    'by',
    'for',
    'from',
    'have',
    'in',
    'is',
    'it',
    'of',
    'on',
    'or',
    'that',
    'the',
    'this',
    'to',
    'was',
    'were',
    'with',
    'без',
    'был',
    'была',
    'были',
    'в',
    'все',
    'где',
    'для',
    'его',
    'ее',
    'и',
    'из',
    'к',
    'как',
    'когда',
    'кто',
    'на',
    'над',
    'о',
    'об',
    'по',
    'под',
    'при',
    'с',
    'что',
    'это',
}


def extract_answer_candidates(
    context: str,
    max_answers: int = 5,
    method: str = 'auto',
    language: str = 'auto',
    spacy_model: Optional[str] = None,
) -> List[AnswerCandidate]:
    context = _normalize(context)
    if method in {'auto', 'spacy'}:
        spacy_candidates = _extract_spacy_candidates(
            context=context,
            language=language,
            spacy_model=spacy_model,
        )
        if spacy_candidates or method == 'spacy':
            return spacy_candidates[:max_answers]

    return _extract_heuristic_candidates(context, max_answers=max_answers)


def _extract_heuristic_candidates(context: str, max_answers: int = 5) -> List[AnswerCandidate]:
    candidates: List[AnswerCandidate] = []

    for sentence in _split_sentences(context):
        candidates.extend(_extract_quoted(sentence))
        candidates.extend(_extract_dates_and_numbers(sentence))
        candidates.extend(_extract_abbreviations(sentence))
        candidates.extend(_extract_capitalized_phrases(sentence))
        candidates.extend(_extract_keyword_phrases(sentence))

    unique = _deduplicate(candidates)
    return _rank_candidates(unique)[:max_answers]


def _extract_spacy_candidates(
    context: str,
    language: str = 'auto',
    spacy_model: Optional[str] = None,
) -> List[AnswerCandidate]:
    try:
        import spacy
    except ImportError:
        return []

    model_name = spacy_model or _default_spacy_model(context=context, language=language)
    try:
        nlp = spacy.load(model_name)
    except OSError:
        return []

    doc = nlp(context)
    candidates: List[AnswerCandidate] = []

    candidates.extend(_spacy_noun_phrase_candidates(doc))
    candidates.extend(_spacy_entity_candidates(doc))

    unique = _deduplicate(candidates)
    return _rank_candidates(unique)


def _default_spacy_model(context: str, language: str) -> str:
    return _SUPPORTED_SPACY_MODELS[_resolve_language(context=context, language=language)]


def _resolve_language(context: str, language: str) -> str:
    if language != 'auto':
        return language
    return 'ru' if re.search(r'[А-Яа-яЁё]', context) else 'en'


def _spacy_entity_candidates(doc) -> Iterable[AnswerCandidate]:
    priority = {
        'DATE': 3.8,
        'TIME': 3.7,
        'MONEY': 3.6,
        'PERCENT': 3.6,
        'QUANTITY': 3.5,
        'ORG': 3.2,
        'EVENT': 3.1,
        'FAC': 3.0,
        'PERSON': 2.9,
        'PER': 2.9,
        'GPE': 2.8,
        'LOC': 2.8,
        'NORP': 2.7,
    }
    for ent in doc.ents:
        text = _clean_candidate(ent.text)
        if not _is_valid_candidate(text):
            continue
        sentence = ent.sent.text if ent.sent is not None else ''
        score = priority.get(ent.label_, 2.6)
        yield AnswerCandidate(text=text, score=score, kind=f'spacy_{ent.label_.lower()}', sentence=sentence)


def _spacy_noun_phrase_candidates(doc) -> Iterable[AnswerCandidate]:
    try:
        noun_chunks = list(doc.noun_chunks)
    except NotImplementedError:
        noun_chunks = []
    except ValueError:
        noun_chunks = []

    for chunk in noun_chunks:
        text = _clean_candidate(chunk.text)
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=3.3, kind='spacy_noun_chunk', sentence=chunk.sent.text)

    yield from _spacy_dependency_span_candidates(doc)
    yield from _spacy_pos_phrase_candidates(doc)

def _spacy_dependency_span_candidates(doc) -> Iterable[AnswerCandidate]:
        allowed_child_deps = {
            'acl',
            'adj',
            'amod',
            'appos',
            'compound',
            'flat',
            'fixed',
            'name',
            'nmod',
            'nummod',
            'obl',
        }
        head_pos = {'NOUN', 'PROPN'}
        modifier_pos = {'ADJ', 'NOUN', 'NUM', 'PROPN'}

        for token in doc:
            if token.pos_ not in head_pos or token.is_stop or token.is_punct:
                continue
            span_tokens = {token}
            stack = [token]
            while stack:
                current = stack.pop()
                for child in current.children:
                    if child.is_punct or child.is_space:
                        continue
                    if child.dep_ not in allowed_child_deps and child.pos_ not in modifier_pos:
                        continue
                    if child.pos_ not in modifier_pos:
                        continue
                    span_tokens.add(child)
                    stack.append(child)

            ordered = sorted(span_tokens, key=lambda item: item.i)
            groups = _contiguous_token_groups(ordered)
            for group in groups:
                if token not in group:
                    continue
                text = _clean_candidate(' '.join(item.text for item in group))
                if not _is_valid_candidate(text):
                    continue
                if len(text.split()) < 2:
                    continue
                sentence = token.sent.text if token.sent is not None else ''
                yield AnswerCandidate(text=text, score=3.2, kind='spacy_dependency_span', sentence=sentence)

def _spacy_pos_phrase_candidates(doc) -> Iterable[AnswerCandidate]:
    spans = []
    current = []
    allowed_pos = {'ADJ', 'NOUN', 'PROPN', 'NUM'}
    for token in doc:
        if token.pos_ in allowed_pos and not token.is_stop and not token.is_punct:
            current.append(token)
            continue
        if current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)

    for span_tokens in spans:
        text = _clean_candidate(' '.join(token.text for token in span_tokens))
        if _is_valid_candidate(text):
            sentence = span_tokens[0].sent.text if span_tokens[0].sent is not None else ''
            yield AnswerCandidate(text=text, score=2.8, kind='spacy_pos_phrase', sentence=sentence)

def _contiguous_token_groups(tokens) -> List[List]:
    if not tokens:
        return []
    groups = [[tokens[0]]]
    for token in tokens[1:]:
        if token.i == groups[-1][-1].i + 1:
            groups[-1].append(token)
        else:
            groups.append([token])
    return groups


def _normalize(text: str) -> str:
    return ' '.join(text.split())


def _split_sentences(context: str) -> List[str]:
    protected = _ABBREVIATION_DOT_RE.sub(lambda match: f'{match.group(1)}<dot>', context)
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    return [sentence.replace('<dot>', '.').strip() for sentence in sentences if sentence.strip()]


def _extract_quoted(sentence: str) -> Iterable[AnswerCandidate]:
    for match in re.finditer(r'["«](.{3,80}?)["»]', sentence):
        text = _clean_candidate(match.group(1))
        if _is_valid_candidate(text):
            yield AnswerCandidate(text=text, score=4.0, kind='quoted', sentence=sentence)


def _extract_dates_and_numbers(sentence: str) -> Iterable[AnswerCandidate]:
    month = rf'{_LETTER}{{3,12}}'
    patterns = [
        rf'\b\d{{1,2}}\s+{month}\s+\d{{4}}\b',
         rf'\b\d{{4}}\s*(?:г\.?|год[ауе]?|век[ауе]?|вв?\.?|year|years?|centur(?:y|ies)|AD|BC)?\b',
        r'\b\d+(?:[,.]\d+)?\s*(?:%|km|m|cm|kg|g|mln|million|billion)\b',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            text = _clean_candidate(match.group(0))
            if _is_valid_candidate(text):
                yield AnswerCandidate(text=text, score=3.7, kind='date_or_number', sentence=sentence)


def _extract_abbreviations(sentence: str) -> Iterable[AnswerCandidate]:
    pattern = r'\b[A-ZА-ЯЁ]{2,}(?:-[A-ZА-ЯЁ]{2,})?\b'
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
    if text.count('(') != text.count(')'):
        return False
    return any(char.isalpha() or char.isdigit() for char in text)


def _rank_candidates(candidates: Sequence[AnswerCandidate]) -> List[AnswerCandidate]:
    ranked = [
        AnswerCandidate(
            text=candidate.text,
            score=round(candidate.score + _quality_adjustment(candidate), 4),
            kind=candidate.kind,
            sentence=candidate.sentence,
        )
        for candidate in candidates
    ]
    ranked = _drop_contained_candidates(ranked)
    ranked.sort(key=lambda item: (item.score, len(item.text.split()), len(item.text)), reverse=True)
    return ranked


def _quality_adjustment(candidate: AnswerCandidate) -> float:
    text = candidate.text
    words = text.split()
    word_count = len(words)
    adjustment = 0.0

    if word_count >= 2:
        adjustment += min(0.9, 0.25 * word_count)
    elif candidate.kind in _SINGLE_TOKEN_ENTITY_KINDS:
        adjustment -= 0.8

    if candidate.kind in _CONTENT_SPAN_KINDS and word_count >= 2:
        adjustment += 0.6

    if _is_inside_parentheses(text, candidate.sentence):
        adjustment -= 0.9

    if word_count == 1 and len(text) < 5:
        adjustment -= 0.3

    return adjustment


def _is_inside_parentheses(text: str, sentence: str) -> bool:
    if not sentence:
        return False
    lowered = text.lower()
    return any(lowered in match.group(0).lower() for match in _PARENTHETICAL_RE.finditer(sentence))


def _drop_contained_candidates(candidates: Sequence[AnswerCandidate]) -> List[AnswerCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.score, len(item.text)), reverse=True)
    kept: List[AnswerCandidate] = []
    for candidate in ordered:
        normalized = candidate.text.lower()
        if any(
            normalized != other.text.lower()
            and normalized in other.text.lower()
            and other.score >= candidate.score - 0.2
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept



def _deduplicate(candidates: Iterable[AnswerCandidate]) -> List[AnswerCandidate]:
    by_key = {}
    for candidate in candidates:
        key = candidate.text.lower()
        existing = by_key.get(key)
        if existing is None or candidate.score > existing.score:
            by_key[key] = candidate
    return list(by_key.values())
