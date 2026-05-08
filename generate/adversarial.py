import re
import random

# Homoglyph maps for character substitution
# Each character maps to a list of visually similar alternatives
HOMOGLYPHS = {
    'a': ['à', 'á', 'ä', '4', '@', 'а'],        # Cyrillic 'а' (U+0430)
    'b': ['8', 'ḅ', 'ɓ'],
    'c': ['с', 'ç', '¢', '©', 'ς'],              # Cyrillic 'с' (U+0441)
    'd': ['ԁ', 'ɗ', 'ḋ'],                         # Cyrillic 'ԁ'
    'e': ['è', 'é', 'ë', '3', '€', 'е'],          # Cyrillic 'е' (U+0435)
    'g': ['ǵ', 'ģ', '9'],
    'h': ['ḥ', 'ḫ', 'ḩ'],
    'i': ['ì', 'í', 'ï', '1', '!', 'і', 'ɩ'],    # Cyrillic 'і' (U+0456)
    'j': ['ј', 'ʝ'],                               # Cyrillic 'ј'
    'k': ['ķ', 'ḳ', 'ḵ'],
    'l': ['1', '|', 'ⅼ', 'ł'],
    'n': ['ñ', 'п', 'ṅ'],                          # Cyrillic 'п'
    'o': ['ò', 'ó', 'ö', '0', '°', 'о'],          # Cyrillic 'о' (U+043E)
    'p': ['р', 'þ'],                               # Cyrillic 'р'
    'r': ['ŕ', 'ř', 'ŗ'],
    's': ['ş', '5', '$', 'ѕ'],                     # Cyrillic 'ѕ' (U+0455)
    't': ['7', '†', 'т', 'ŧ'],                     # Cyrillic 'т'
    'u': ['ù', 'ú', 'ü', 'ų'],
    'x': ['х', 'ẋ'],                               # Cyrillic 'х'
    'y': ['ý', 'ÿ', 'у'],                          # Cyrillic 'у'
    'z': ['2', 'ż', 'ž'],
}

ADVERSARIAL_TECHNIQUES = [
    "leet_speak",
    "zero_width_space",
    "homoglyph_substitution",
    "syllable_split",
    "phonetic_respelling",
    "abbreviation_dots",
    "emoji_substitution",
    "mixed_script",
    "deliberate_misspelling",
    "diacritic_manipulation",
]


def apply_leet(text: str, intensity: float = 0.4) -> str:
    """Apply leet speak substitution to a portion of substitutable characters."""
    result = []
    for char in text:
        lower = char.lower()
        if lower in HOMOGLYPHS and random.random() < intensity:
            candidates = [c for c in HOMOGLYPHS[lower] if c.isascii() and len(c) == 1]
            if candidates:
                result.append(random.choice(candidates))
            else:
                result.append(char)
        else:
            result.append(char)
    return ''.join(result)


def apply_homoglyphs(text: str, intensity: float = 0.3) -> str:
    """Replace Latin characters with visually similar non-Latin characters."""
    result = []
    for char in text:
        lower = char.lower()
        if lower in HOMOGLYPHS and random.random() < intensity:
            non_latin = [c for c in HOMOGLYPHS[lower] if not c.isascii()]
            if non_latin:
                result.append(random.choice(non_latin))
            else:
                result.append(char)
        else:
            result.append(char)
    return ''.join(result)


def insert_zero_width_spaces(text: str, target_terms: list[str] | None = None) -> str:
    """Insert zero-width spaces into target terms to break keyword matching."""
    zws = '​'
    if not target_terms:
        return text

    result = text
    for term in sorted(target_terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(lambda m: zws.join(m.group()), result)
    return result


def syllable_split(text: str, target_terms: list[str] | None = None) -> str:
    """Split target terms with hyphens or periods."""
    if not target_terms:
        return text

    result = text
    for term in sorted(target_terms, key=len, reverse=True):
        mid = max(1, len(term) // 2)
        splits = [
            term[:mid] + '-' + term[mid:],
            term[:mid] + '.' + term[mid:],
            term[0] + '-' + term[1:],
        ]
        replacement = random.choice(splits)
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(replacement, result, count=1)
    return result


def apply_abbreviation_dots(text: str, target_terms: list[str] | None = None) -> str:
    """Convert trigger terms to dotted abbreviations (e.g., PKK → P.K.K)."""
    if not target_terms:
        return text

    result = text
    for term in sorted(target_terms, key=len, reverse=True):
        dotted = '.'.join(term)
        hyphenated = '-'.join(term)
        replacement = random.choice([dotted, hyphenated])
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(replacement, result, count=1)
    return result


def generate_adversarial_variants(
    text: str,
    trigger_keywords: list[str] | None = None,
    n_variants: int = 5,
) -> list[dict]:
    """Programmatically generate N adversarial variants of a text.

    Returns a list of {text, adversarial_technique} dicts.
    """
    techniques = [
        ("leet_speak", lambda t: apply_leet(t, random.uniform(0.3, 0.6))),
        ("homoglyph_substitution", lambda t: apply_homoglyphs(t, random.uniform(0.3, 0.6))),
        ("zero_width_space", lambda t: insert_zero_width_spaces(t, trigger_keywords)),
        ("syllable_split", lambda t: syllable_split(t, trigger_keywords)),
        ("abbreviation_dots", lambda t: apply_abbreviation_dots(t, trigger_keywords)),
    ]

    variants = []
    for i in range(n_variants):
        tech_name, func = techniques[i % len(techniques)]
        variant_text = func(text)
        if variant_text != text:
            variants.append({
                "text": variant_text,
                "adversarial_technique": tech_name,
            })

    return variants
