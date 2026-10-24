"""Exact, locale-explicit numeric normalization without floating-point rounding."""

import re
from decimal import Decimal

# GSM8K uses English/US notation: comma groups are thousands, dot is decimal.
_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)


def normalize_numeric(value: str) -> str:
    """Return a canonical decimal string; reject units, fractions, lists and bad grouping.

    Bounds limit resource use. No Decimal.normalize() is used: it can round long
    numbers to the active decimal context precision. Unicode minus is supported.
    """
    if not isinstance(value, str):
        raise ValueError("numeric answer must be a string")
    text = value.strip().replace("−", "-")
    if len(text) > 1000 or not _NUMBER.fullmatch(text):
        raise ValueError("expected one US-format number, without units or surrounding prose")
    exponent = re.split("[eE]", text)
    if len(exponent) == 2 and abs(int(exponent[1])) > 1000:
        raise ValueError("numeric exponent exceeds supported range [-1000, 1000]")
    number = Decimal(text.replace(",", ""))
    if abs(number.as_tuple().exponent) > 1000:
        raise ValueError("numeric exponent exceeds supported range [-1000, 1000]")
    if number.is_zero():
        return "0"
    result = format(number, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def parse_gsm8k_answer(answer: str) -> tuple[str, str]:
    """Extract rationale and exactly one final numeric answer after ####."""
    if not isinstance(answer, str) or answer.count("####") != 1:
        raise ValueError("answer must contain exactly one #### marker")
    rationale, numeric = answer.split("####")
    if not rationale.strip():
        raise ValueError("answer rationale must be nonempty")
    return rationale.strip(), normalize_numeric(numeric)
