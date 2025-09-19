import pandas as pd
from typing import Tuple, Dict
from collections import Counter
import string
import unicodedata


def analyze_character_frequency(df: pd.DataFrame) -> Tuple[Dict[str, int], ...]:
    """Classify and count characters by category."""
    char_freq = Counter()
    for text in df["Contents"].astype(str):
        char_freq.update(text)

    lowercase, uppercase, digits, punct, spaces, others = {}, {}, {}, {}, {}, {}

    for ch, freq in char_freq.items():
        if ch.isalpha():
            (lowercase if ch.islower() else uppercase)[ch] = freq
        elif ch.isdigit():
            digits[ch] = freq
        elif ch in string.punctuation or ch in "«»–—’‘”“":
            punct[ch] = freq
        elif ch.isspace():
            spaces[ch] = freq
        else:
            others[ch] = freq

    return lowercase, uppercase, digits, punct, spaces, others


def print_top_n(d: Dict[str, int], title: str, n: int = 7) -> None:
    """Print top-N most frequent items in a dictionary."""
    print(f"\n{title} (Top: {n}, total: {len(d)})")
    for ch, f in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]:
        print(f"'{ch}': {f}")


def analyze_diacritics(lowercase_letters: Dict[str, int]) -> None:
    """Display usage of diacritic lowercase letters."""
    basic = set(string.ascii_lowercase)
    pure = {ch: freq for ch, freq in lowercase_letters.items() if ch in basic}
    dia = {ch: freq for ch, freq in lowercase_letters.items() if ch not in basic}

    for ch, f in sorted(dia.items()):
        name = unicodedata.name(ch, "UNKNOWN")
        print(f"'{ch}' (U+{ord(ch):04X}, {name}): {f}")

    total_basic = sum(pure.values())
    total_dia = sum(dia.values())
    total = total_basic + total_dia
    print(f"\n% of lowercase letters with diacritics: {total_dia / total * 100:.2f}%")
