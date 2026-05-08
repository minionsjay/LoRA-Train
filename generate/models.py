from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Taxonomy models (parsed from JSON) ---

class TriggerData(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    adversarial_variants: list[str] = Field(default_factory=list)


class BoundaryCase(BaseModel):
    scenario: str
    should_trigger: bool
    rationale: str


class LocalViolation(BaseModel):
    label: str
    severity: str
    detection_type: str
    description: str
    cultural_context: list[str] = Field(default_factory=list)
    triggers: TriggerData = Field(default_factory=TriggerData)
    boundary_cases: list[BoundaryCase] = Field(default_factory=list)


class CountryTaxonomy(BaseModel):
    country_code: str
    country_name: str
    region: str
    languages: list[str]
    cultural_contexts: list[str]
    base_violations: list[str]
    regional_adapter: str
    local_violations: list[LocalViolation]


# --- Config models ---

class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    max_retries: int = 5
    timeout_seconds: int = 120
    max_concurrency: int = 10


class ProxyConfig(BaseModel):
    enabled: bool = False
    protocol: str = "http"
    host: str = "127.0.0.1"
    port: int = 7890
    username: str = ""
    password: str = ""


class PositiveCounts(BaseModel):
    keyword_sensitive: int = 60
    contextual: int = 50
    hybrid: int = 50


class NegativeCounts(BaseModel):
    per_boundary_case: int = 5
    general_non_violation: int = 30


class SamplesPerLabel(BaseModel):
    positive: PositiveCounts = Field(default_factory=PositiveCounts)
    negative: NegativeCounts = Field(default_factory=NegativeCounts)


class GenerationConfig(BaseModel):
    samples_per_label: SamplesPerLabel = Field(default_factory=SamplesPerLabel)
    temperature: float = 0.9
    top_p: float = 0.95
    force: bool = False
    seed: Optional[int] = 42


class OutputConfig(BaseModel):
    dir: str = "output"


class AppConfig(BaseModel):
    llm: LLMConfig
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# --- JSONL output sample model ---

class GeneratedSample(BaseModel):
    text: str
    label: str
    is_violation: bool
    severity: str
    country_code: str
    region: str
    detection_type: str
    language: str
    cultural_contexts: list[str] = Field(default_factory=list)
    adversarial_technique: Optional[str] = None
    generation_strategy: str  # "positive" | "adversarial_augmentation" | "boundary_negative" | "general_negative"
    model_used: str
    metadata: dict = Field(default_factory=dict)


# --- Generation job planning ---

class GenerationPlan(BaseModel):
    country: CountryTaxonomy
    violation: LocalViolation
    positive_count: int
    adversarial_count: int
    boundary_counts: dict[str, int]  # boundary_scenario -> count
    general_negative_count: int
    output_path: str


# --- Stats tracking ---

class GenerationStats(BaseModel):
    country_code: str
    label: str
    total_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    adversarial_samples: int = 0
    api_calls: int = 0
    tokens_used: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
