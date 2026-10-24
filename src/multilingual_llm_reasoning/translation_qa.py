"""Heuristic flags, never proof that a translation preserves mathematical meaning."""

import re
from collections import Counter
from collections.abc import Sequence

from .numeric import normalize_numeric

NUMERIC_TOKEN = re.compile(r"[+−-]?(?:[0-9]+(?:[,.][0-9]+)*|\.[0-9]+)(?:[eE][+−-]?[0-9]+)?")
UNIT_CURRENCY_TOKEN = re.compile(
    r"(?i)(?:[$€£¥%]|\b(?:usd|eur|gbp|dollars?|euros?|pounds?|yen|yuan|"
    r"kg|kilograms?|grams?|g|lbs?|pounds?|ounces?|oz|miles?|kilometers?|"
    r"kilometres?|meters?|metres?|feet|foot|inches?|liters?|litres?|"
    r"gallons?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|percent(?:age)?)\b)"
)

_RELATION_CUES = {
    "en": {
        "comparison": re.compile(r"(?i)\b(more|less|fewer|greater|smaller|larger|than)\b"),
        "multiplication": re.compile(r"(?i)\b(twice|double|triple|half|times|each|every)\b"),
        "condition_time": re.compile(r"(?i)\b(if|unless|before|after|remaining|left)\b"),
    },
    "zh": {
        "comparison": re.compile(r"比|多于|少于|更多|更少|大于|小于"),
        "multiplication": re.compile(r"倍|两倍|三倍|一半|每个|每"),
        "condition_time": re.compile(r"如果|除非|之前|以后|之后|剩下|剩余"),
    },
    "de": {
        "comparison": re.compile(r"(?i)\b(mehr|weniger|größer|kleiner|als)\b"),
        "multiplication": re.compile(r"(?i)\b(doppelt|dreifach|halb|mal|je|jeder)\b"),
        "condition_time": re.compile(r"(?i)\b(wenn|falls|bevor|nachdem|übrig|rest)\b"),
    },
    "ja": {
        "comparison": re.compile(r"より|多く|少なく|以上|以下|より多い|より少ない"),
        "multiplication": re.compile(r"倍|半分|各|それぞれ"),
        "condition_time": re.compile(r"場合|前に|後に|残り|残った|もし|ば|れば"),
    },
}
_LEAKAGE = re.compile(
    r"(?i)(?:####|FINAL_ANSWER\s*:|\bthe answer is\b|\bsolution\s*:|"
    r"答案是|解答是|答えは|答え：|die antwort ist|lösung\s*:)"
)
_ROUNDING_SOURCE = re.compile(
    r"(?i)\b(?:round(?:ed|ing)?\s+(?:to|up|down|off)|"
    r"nearest (?:whole )?integer|nearest cent)\b"
)
_ROUNDING_TARGET = {
    "zh": re.compile(r"四舍五入|舍入|取整|整数"),
    "de": re.compile(r"rund(?:en|et|ung)?|ganze Zahl|nächste(?:n)? Zahl|Cent", re.I),
    "ja": re.compile(r"四捨五入|丸め|整数|セント"),
}
_ROUNDING_DIRECTION = {
    "zh": re.compile(r"向上取整|向下取整|无条件进位|无条件舍去|进位|舍去"),
    "de": re.compile(r"aufgerund|abgerund|nach oben|nach unten", re.I),
    "ja": re.compile(r"切り上げ|切り捨て|上方向|下方向"),
}
_QUESTION_END = re.compile(r"[?？]\s*[\"'）)]*$")
_QUESTION_FINAL = {
    "zh": _QUESTION_END,
    "de": _QUESTION_END,
    "ja": re.compile(r"(?:[?？]\s*[\"'）)]*|か[。.!！?？]?)$"),
}
_CURRENCY_TARGET = {
    "zh": re.compile(r"美元|美金|欧元|人民币|元|英镑|日元|美元"),
    "de": re.compile(r"Dollar|Euro|Pfund|Yen|Yuan", re.I),
    "ja": re.compile(r"ドル|ユーロ|ポンド|円|元"),
}
_PERCENT_TARGET = {
    "zh": re.compile(r"百分之|百分比|百分率|％|%"),
    "de": re.compile(r"Prozent|%", re.I),
    "ja": re.compile(r"パーセント|百分率|確率|%|％"),
}
_EACH_TARGET = {
    "zh": re.compile(r"每|各"),
    "de": re.compile(r"jed(?:e|en|er|es|em)|pro", re.I),
    "ja": re.compile(r"それぞれ|各|ずつ"),
}
_EN_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
}


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?。！？]+", text.strip())))


def _normalized_numbers(text: str) -> set[str]:
    values = set()
    for token in NUMERIC_TOKEN.findall(text):
        try:
            values.add(normalize_numeric(token))
        except ValueError:
            continue
    return values


def qa_flags(
    source: str,
    translated: str,
    language: str,
    *,
    mock: bool = False,
    response_status: str = "completed",
    gold_answer: str | None = None,
) -> list[str]:
    """Return conservative review signals; an empty list never means approval."""
    if language not in _RELATION_CUES:
        raise ValueError(f"Unsupported translation language: {language}")
    flags = []
    if not translated.strip():
        flags.extend(("missing_text", "missing_translation"))
    if Counter(NUMERIC_TOKEN.findall(source)) != Counter(NUMERIC_TOKEN.findall(translated)):
        flags.append("changed_numeric_tokens")
    if Counter(UNIT_CURRENCY_TOKEN.findall(source)) != Counter(
        UNIT_CURRENCY_TOKEN.findall(translated)
    ):
        flags.append("unit_currency_token_mismatch")
    if len(translated.strip()) < max(1, len(source.strip()) * 0.2):
        flags.append("possibly_missing_text")
    if _sentence_count(translated) < _sentence_count(source):
        flags.append("possibly_incomplete_sentences")
    if language != "en" and translated.strip() == source.strip():
        flags.append("unchanged_english")
    if language == "zh" and not re.search(r"[\u4e00-\u9fff]", translated):
        flags.append("missing_target_script")
    if language == "ja" and not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", translated):
        flags.append("missing_target_script")
    source_cues = _RELATION_CUES["en"]
    target_cues = _RELATION_CUES[language]
    if any(
        pattern.search(source) and not target_cues[key].search(translated)
        for key, pattern in source_cues.items()
    ):
        flags.append("relational_cues_not_detected")
    if _LEAKAGE.search(translated):
        flags.append("possible_solution_leakage")
    if gold_answer is not None:
        try:
            gold = normalize_numeric(gold_answer)
        except ValueError:
            gold = None
        new_values = _normalized_numbers(translated) - _normalized_numbers(source)
        if gold is not None and gold in new_values:
            flags.append("possible_answer_leakage")
    if response_status != "completed":
        flags.append("provider_incomplete")
    if mock:
        flags.append("mock_not_translation")
    return flags


def mathematical_translation_flags(source: str, translated: str, language: str) -> list[str]:
    """Deterministic review heuristics for question structure and math-sensitive text."""
    if language not in _ROUNDING_TARGET:
        raise ValueError(f"Unsupported translation language: {language}")
    flags: list[str] = []
    text = translated.strip()
    if not text:
        return ["empty_translation", "possibly_truncated_output"]
    if text.endswith(("…", "...", "..")) or not re.search(r"[.!?。！？]$", text):
        flags.append("possibly_truncated_output")
    if source.rstrip().endswith("?") and not _QUESTION_FINAL[language].search(text):
        flags.append("missing_final_question")
    if _ROUNDING_SOURCE.search(source):
        if not _ROUNDING_TARGET[language].search(text):
            flags.append("missing_rounding_instruction")
        elif _ROUNDING_DIRECTION[language].search(text):
            flags.append("changed_rounding_direction")
    source_has_currency = bool(
        re.search(r"[$€£¥]|\b(?:dollars?|euros?|pounds?|yen|yuan)\b", source, re.I)
    )
    if source_has_currency and not _CURRENCY_TARGET[language].search(text):
        flags.append("missing_currency_reference")
    source_has_percent = bool(re.search(r"%|\bpercent(?:age)?\b", source, re.I))
    if source_has_percent and not _PERCENT_TARGET[language].search(text):
        flags.append("missing_percentage_reference")
    if re.search(r"\b(?:each|every)\b", source, re.I) and not _EACH_TARGET[language].search(text):
        flags.append("unclear_each_quantity_relation")
    source_numbers = Counter(_canonical_number_tokens(source, source_language="en"))
    target_numbers = Counter(_canonical_number_tokens(text, source_language=language))
    if source_numbers - target_numbers:
        flags.append("missing_numeric_values")
    if target_numbers - source_numbers:
        flags.append("duplicated_or_added_numeric_values")
    return flags


def _canonical_number_tokens(text: str, *, source_language: str) -> list[str]:
    values = []
    for token in NUMERIC_TOKEN.findall(text):
        try:
            values.append(normalize_numeric(token))
        except ValueError:
            continue
    if source_language == "en":
        text = re.sub(r"\beach one\b", "each", text, flags=re.I)
        words = re.compile(r"\b(?:" + "|".join(_EN_NUMBER_WORDS) + r")\b", re.I)
        values.extend(_EN_NUMBER_WORDS[word.casefold()] for word in words.findall(text))
    elif source_language == "de":
        german = {
            "null": "0",
            "eins": "1",
            "zwei": "2",
            "drei": "3",
            "vier": "4",
            "fünf": "5",
            "sechs": "6",
            "sieben": "7",
            "acht": "8",
            "neun": "9",
            "zehn": "10",
            "elf": "11",
            "zwölf": "12",
            "dreizehn": "13",
            "vierzehn": "14",
            "fünfzehn": "15",
            "sechzehn": "16",
            "siebzehn": "17",
            "achtzehn": "18",
            "neunzehn": "19",
            "zwanzig": "20",
        }
        words = re.compile(r"\b(?:" + "|".join(german) + r")\b", re.I)
        values.extend(german[word.casefold()] for word in words.findall(text))
        compounds = re.compile(r"\b(drei|vier|fünf|sechs|sieben|acht|neun|zehn)mal\b", re.I)
        values.extend(german[word.casefold()] for word in compounds.findall(text))
    if source_language in ("zh", "ja"):
        # These idioms contain the numeral 四 but mean "rounding" rather than
        # the number four. Remove them before scanning Chinese/Japanese numerals.
        cjk_text = re.sub(r"四舍五入|四捨五入", "", text)
        cjk_text = re.sub(r"一(?=(?:个|種|种|張|张|只|次|部|件|本|条|塊|块))", "", cjk_text)
        for token in re.findall(r"[零〇一二两三四五六七八九十百千]+", cjk_text):
            parsed = _parse_cjk_number(token)
            if parsed is not None:
                values.append(str(parsed))
    return values


def _parse_cjk_number(token: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if all(char in digits for char in token):
        value = 0
        for char in token:
            value = value * 10 + digits[char]
        return value
    section = 0
    number = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for char in token:
        if char in digits:
            number = digits[char]
        elif char in units:
            unit = units[char]
            if number == 0 and unit != 10:
                return None
            section += (number or (1 if unit == 10 else 0)) * unit
            number = 0
        else:
            return None
    return section + number


def duplicate_translation_flags(records: Sequence[dict[str, object]]) -> dict[tuple[str, str], str]:
    """Flag exact duplicate translated texts across different IDs within a language."""
    normalized: dict[tuple[str, str], list[str]] = {}
    for record in records:
        language = record.get("language")
        text = record.get("translated_text")
        identity = record.get("original_problem_id")
        if not all(isinstance(value, str) for value in (language, text, identity)):
            continue
        if not text.strip():
            continue
        key = (language, " ".join(text.casefold().split()))
        normalized.setdefault(key, []).append(identity)
    return {
        (identity, language): "duplicate_translated_question"
        for (language, _), identities in normalized.items()
        if len(set(identities)) > 1
        for identity in set(identities)
    }
