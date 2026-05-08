import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputWriter:
    def __init__(self, output_dir: str = "output", force: bool = False):
        self.output_dir = Path(output_dir)
        self.force = force
        self._files: dict[str, object] = {}  # path_str -> file handle
        self._counts: dict[str, int] = {}      # path_str -> line count
        self._seen_signatures: set[str] = set()

    def _open(self, path: Path) -> object:
        key = str(path)
        if key not in self._files:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if self.force else "a"
            self._files[key] = open(path, mode, encoding="utf-8")
            self._counts[key] = 0
        return self._files[key]

    def exists(self, country_code: str, label: str) -> bool:
        return (self.output_dir / country_code / f"{label}.jsonl").exists()

    def write(self, sample: dict, country_code: str, label: str):
        path = self.output_dir / country_code / f"{label}.jsonl"
        fh = self._open(path)
        line = json.dumps(sample, ensure_ascii=False)
        fh.write(line + "\n")
        fh.flush()
        self._counts[str(path)] += 1

    def write_samples(self, samples: list[dict], country_code: str, label: str):
        for sample in samples:
            sample.setdefault("model_used", "")
            sample.setdefault("label", label)
            sample.setdefault("country_code", country_code)
            self.write(sample, country_code, label)

    def write_stats(self, country_code: str, label: str, stats: dict):
        path = self.output_dir / country_code / f"{label}_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        stats["completed_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def close(self):
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)
