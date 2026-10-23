"""Tokenization shared by BM25, the hashing embedder and the grounding check."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales cuando
    de del desde donde dos durante e el ella ellas ellos en entre era eran es esa esas ese eso esos
    esta estan estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mis
    muy no nos o os otra otras otro otros para pero por porque que quien quienes se ser si sin sobre
    su sus tambien te tiene tienen tu tus un una unas uno unos y ya yo
    cual cuanto cuantos cuanta cuantas puedo puede pueden debo debe deben hace hacer
    the and or of to in on for is are was were be been with as by at an a it this that these those
    from what which who how when where can do does
    """.split()
)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _light_stem(token: str) -> str:
    """Very small Spanish/English plural stemmer: enough to match 'facturas' with 'factura'."""
    if len(token) > 4 and token.endswith("es") and token[-3] not in "aeiou":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    return [stem for stem, _ in tokenize_with_surface(text, remove_stopwords)]


def tokenize_with_surface(text: str, remove_stopwords: bool = True) -> list[tuple[str, str]]:
    """(stem, original word) pairs: stems for matching, words for display."""
    words = _WORD.findall(strip_accents(text.lower()))
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [(_light_stem(w), w) for w in words]
