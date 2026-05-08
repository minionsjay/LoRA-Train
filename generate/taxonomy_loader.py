import json
from pathlib import Path
from .models import CountryTaxonomy


_TAXONOMY_DIR = Path(__file__).resolve().parent.parent / "taxonomy"


def load_all_countries(taxonomy_dir: str | None = None) -> list[CountryTaxonomy]:
    base = Path(taxonomy_dir) if taxonomy_dir else _TAXONOMY_DIR
    countries_dir = base / "countries"
    if not countries_dir.is_dir():
        raise FileNotFoundError(f"Countries taxonomy directory not found: {countries_dir}")

    countries = []
    for fpath in sorted(countries_dir.glob("*.json")):
        with open(fpath) as f:
            raw = json.load(f)
        countries.append(CountryTaxonomy(**raw))
    return countries


def load_country_codes(codes: list[str], taxonomy_dir: str | None = None) -> list[CountryTaxonomy]:
    all_countries = load_all_countries(taxonomy_dir)
    code_set = {c.upper() for c in codes}
    result = [c for c in all_countries if c.country_code in code_set]
    missing = code_set - {c.country_code for c in result}
    if missing:
        available = sorted(c.country_code for c in all_countries)
        raise ValueError(f"Unknown country codes: {missing}. Available: {available}")
    return result


def filter_labels(countries: list[CountryTaxonomy], labels: list[str]) -> list[CountryTaxonomy]:
    """Filter each country's local_violations to only include specified labels."""
    label_set = set(labels)
    for country in countries:
        country.local_violations = [v for v in country.local_violations if v.label in label_set]
    return [c for c in countries if c.local_violations]


DEFAULT_POSITIVE_COUNTS = {"keyword_sensitive": 150, "contextual": 150, "hybrid": 150}


def count_all_labels(countries: list[CountryTaxonomy]) -> dict[str, int]:
    """Return {label: sample_count_estimate} for all labels across all countries."""
    counts = {}
    for c in countries:
        for v in c.local_violations:
            pos = DEFAULT_POSITIVE_COUNTS.get(v.detection_type, 50)
            neg = len(v.boundary_cases) * 5 + 30
            counts[v.label] = pos + neg
    return counts
