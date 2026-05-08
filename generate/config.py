import os
import re
import yaml
from pathlib import Path
from .models import AppConfig, LLMConfig, ProxyConfig, GenerationConfig, OutputConfig


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} placeholders with actual environment variable values.
    If the env var is not set, leaves the placeholder as-is (will be caught at API call time)."""
    def _replacer(match):
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            import warnings
            warnings.warn(
                f"Environment variable '{var_name}' is not set. "
                f"Set it via: export {var_name}=<value> before generating."
            )
            return ""  # Empty string — will fail at API call time with clear error
        return env_val
    return _ENV_VAR_RE.sub(_replacer, value)


def _resolve_dict_env_vars(obj):
    """Recursively resolve ${ENV_VAR} strings in a dict."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _resolve_dict_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_dict_env_vars(item) for item in obj]
    return obj


def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}. Copy config.yaml.example or create your own.")

    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _resolve_dict_env_vars(raw)

    llm = LLMConfig(**raw.get("llm", {}))
    proxy = ProxyConfig(**raw.get("proxy", {}))
    generation = GenerationConfig(**raw.get("generation", {}))
    output = OutputConfig(**raw.get("output", {}))

    return AppConfig(llm=llm, proxy=proxy, generation=generation, output=output)
