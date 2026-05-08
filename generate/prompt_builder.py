from pathlib import Path
from .models import CountryTaxonomy, LocalViolation, BoundaryCase, TriggerData

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

DEFAULT_POSITIVE_COUNTS = {"keyword_sensitive": 150, "contextual": 150, "hybrid": 150}


def get_positive_count(detection_type: str) -> int:
    return DEFAULT_POSITIVE_COUNTS.get(detection_type, 50)


def _load_template(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


class PromptBuilder:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._positive_system = _load_template("positive_system.txt")
        self._positive_user_tpl = _load_template("positive_user.txt")
        self._negative_system = _load_template("negative_system.txt")
        self._negative_user_tpl = _load_template("negative_user.txt")
        self._adversarial_user_tpl = _load_template("adversarial_user.txt")

    # --- Templating helpers (simple Jinja2-like, no dependency needed) ---

    def _render(self, template: str, ctx: dict) -> str:
        """Simple {{ var }} template replacement."""
        result = template
        for key, value in ctx.items():
            placeholder = "{{ " + key + " }}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def _resolve_ctx(self, ctx: dict, template: str) -> str:
        result = template
        # Handle simple {{ var }} and {{ var | filter }}
        for key, value in ctx.items():
            placeholder = "{{ " + key + " }}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
            # Handle join filter
            join_placeholder = "{{ " + key + " | join(\", \") }}"
            if join_placeholder in result and isinstance(value, list):
                result = result.replace(join_placeholder, ", ".join(value))
        return result

    # --- Keyword directives per detection_type ---

    def _keyword_directive(self, dt: str, has_keywords: bool, has_adversarial: bool) -> str:
        if dt == "keyword_sensitive":
            return (
                "At least 50% of samples must use character-substitution adversarial variants "
                "(leet speak, homoglyphs, zero-width chars, phonetic respelling, emoji substitution) "
                "integrated naturally into the text. The remaining 50% use exact trigger keywords. "
                "Every sample must contain at least one trigger term or a recognizable variant of one.\n"
                + ("Trigger keywords to work with: YES — use and vary them extensively." if has_keywords
                   else "No specific keywords provided — invent locally-appropriate trigger terms.")
            )
        elif dt == "contextual":
            return (
                "Do NOT rely on explicit keywords. Create violations through context, implication, "
                "and culturally-situated references. The violation must be clear to a human reader "
                "familiar with the cultural context but may not contain any obvious trigger word. "
                "Use the behavioral patterns listed above as narrative templates. "
                "Character substitutions should be rare (at most 10% of samples) — "
                "focus instead on varied ways of implying the violation through euphemism, sarcasm, and cultural code words."
                + (f"\nWhen using adversarial variants, expand from these known patterns: {', '.join}." if has_adversarial else "")
            )
        else:  # hybrid
            return (
                "This label uses two-stage detection: keyword trigger + contextual confirmation. "
                "30% of samples should be keyword-heavy with adversarial substitutions (keyword is primary signal). "
                "70% should embed the keyword naturally in a broader context that confirms malicious intent — "
                "the keyword alone triggers review, the context confirms the violation. "
                "For the 30% keyword-heavy samples, apply character substitutions to the trigger terms. "
                "For the 70% contextual, use exact keywords but focus on making the surrounding context clearly violative."
            )

    def _perspective_suggestions(self, dt: str) -> str:
        if dt == "contextual":
            return (
                "- Academic/researcher discussing the topic neutrally\n"
                "- Journalist reporting facts without editorializing\n"
                "- Historian describing past events with scholarly distance\n"
                "- Personal narrative: someone describing being a victim/observer, not a perpetrator\n"
                "- Cultural/artistic discussion: analyzing a film/book/artwork that touches on the topic\n"
                "- Legal professional discussing applicable laws"
            )
        elif dt == "keyword_sensitive":
            return (
                "- Using the keyword as a quoted term in a metalinguistic discussion\n"
                "- Technical/medical/legal usage where the keyword has a different meaning\n"
                "- Reporting someone else's use of the term (with clear distancing)\n"
                "- Historical document quotation with academic framing"
            )
        else:  # hybrid
            return (
                "- Mix of contextual and keyword_sensitive perspectives above\n"
                "- Keyword present but in clearly benign context (different topic disambiguation)\n"
                "- Discussing the keyword as part of a policy debate without targeting anyone"
            )

    # --- Build methods ---

    def build_positive(self, country: CountryTaxonomy, violation: LocalViolation, count: int) -> dict:
        """Build system + user messages for positive sample generation."""
        triggers = violation.triggers or TriggerData()

        ctx = {
            "count": str(count),
            "country_name": country.country_name,
            "country_code": country.country_code,
            "region": country.region,
            "label": violation.label,
            "severity": violation.severity,
            "detection_type": violation.detection_type,
            "description": violation.description,
            "cultural_contexts": ", ".join(violation.cultural_context),
            "languages": ", ".join(country.languages),
            "triggers.keywords": ", ".join(triggers.keywords) if triggers.keywords else "(none — violation expressed through context/implication)",
            "triggers.patterns": "; ".join(triggers.patterns) if triggers.patterns else "(none specified)",
            "triggers.adversarial_variants": "; ".join(triggers.adversarial_variants) if triggers.adversarial_variants else "(none specified)",
            "keyword_directive": self._keyword_directive(
                violation.detection_type, bool(triggers.keywords), bool(triggers.adversarial_variants)
            ),
            "primary_languages": ", ".join(country.languages[:2]),
            "lang_primary_pct": "70",
            "lang_code_switched_pct": "30",
        }

        user_prompt = self._resolve_ctx(ctx, self._positive_user_tpl)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._positive_system},
                {"role": "user", "content": user_prompt},
            ],
        }

    def build_negative_boundary(self, country: CountryTaxonomy, violation: LocalViolation,
                                 boundary: BoundaryCase, count: int) -> dict:
        ctx = {
            "count": str(count),
            "country_name": country.country_name,
            "country_code": country.country_code,
            "label": violation.label,
            "description": violation.description,
            "scenario": boundary.scenario,
            "rationale": boundary.rationale,
            "languages": ", ".join(country.languages),
            "perspective_suggestions": self._perspective_suggestions(violation.detection_type),
        }

        user_prompt = self._resolve_ctx(ctx, self._negative_user_tpl)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._negative_system},
                {"role": "user", "content": user_prompt},
            ],
        }

    def build_general_negative(self, country: CountryTaxonomy, violation: LocalViolation, count: int) -> dict:
        languages = ", ".join(country.languages)
        system = (
            "You are a data generator for content safety classifier training. "
            "Generate completely benign, harmless social media posts that have NO relation to any content violation. "
            "The posts should be natural, realistic user-generated content in the specified languages."
        )
        user = (
            f"Generate {count} completely benign and harmless social media posts, comments, or messages "
            f"in {languages} (mix naturally, include some code-switching where realistic).\n\n"
            f"Topics: everyday life, food, technology, travel, family, sports, music, hobbies, work, school.\n"
            f"These posts must have absolutely nothing to do with {violation.label} or any other content violation.\n\n"
            f"Output exactly {count} JSON objects, one per line. Each object must have these EXACT fields:\n"
            f'{{"text": string, "label": "{violation.label}", "is_violation": false, '
            f'"country_code": "{country.country_code}", "language": string, '
            f'"generation_strategy": "general_negative", '
            f'"metadata": {{"rationale": "Random benign content unrelated to any violation category"}}}}'
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def build_adversarial(self, original_text: str, country: CountryTaxonomy, violation: LocalViolation,
                          count: int) -> dict:
        system = (
            "You are an adversarial text generator. Your task is to create character-level variants "
            "of given text that preserve meaning while evading keyword-based filters. Output ONLY valid JSON lines."
        )
        ctx = {
            "count": str(count),
            "original_text": original_text,
            "label": violation.label,
            "languages": ", ".join(country.languages),
            "country_name": country.country_name,
        }
        user = self._resolve_ctx(ctx, self._adversarial_user_tpl)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
