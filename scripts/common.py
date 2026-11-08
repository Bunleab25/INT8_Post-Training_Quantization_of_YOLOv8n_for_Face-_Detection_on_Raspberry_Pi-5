"""Shared utilities for Stage 1 and Stage 2 lifecycle scripts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF_RE = re.compile(r"^(?P<source>.*)\.rf\.[0-9a-f]+\.(?P<ext>[A-Za-z0-9]+)$")


def source_name(path: Path) -> str:
    """Return the base Roboflow source identifier before `.rf.<hash>.<ext>`."""
    match = RF_RE.match(path.name)
    if match:
        return match.group("source")
    name = path.stem
    if ".rf." in name:
        return name.split(".rf.", 1)[0]
    return name


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file using 1 MiB streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_checkpoint_hash(path: Path, checkpoint: Path | str) -> str | None:
    """Read expected SHA-256 hash for `checkpoint` from a sha256sum manifest file."""
    if not path.is_file():
        return None
    target = Path(checkpoint).as_posix()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("*") == target:
            return fields[0].lower()
    return None


def load_json(path: Path | str, default: Any = None) -> Any:
    """Load JSON from `path`, returning `default` (or `{}` if None) on error when default is supplied."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default if default is not None else {}


def write_json(path: Path | str, payload: Any) -> None:
    """Write `payload` as 2-space indented JSON with a trailing newline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric or None value for Markdown tables."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def last_epoch(results_csv: Path) -> int | None:
    """Extract the completed epoch count (1-indexed) from an Ultralytics `results.csv`."""
    if not results_csv.exists():
        return None
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    row = dict(zip([h.strip() for h in lines[0].split(",")], lines[-1].split(",")))
    try:
        return int(float(row.get("epoch", 0))) + 1
    except ValueError:
        return None
