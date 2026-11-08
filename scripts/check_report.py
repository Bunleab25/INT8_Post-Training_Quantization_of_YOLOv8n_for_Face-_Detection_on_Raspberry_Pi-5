#!/usr/bin/env python3
"""Validate the research report: every embedded image exists and figure numbers are sequential.

Usage: python scripts/check_report.py [docs/yolov8n_int8_ptq_pi5_report.md]
Exit code 1 on any problem (used by .github/workflows/report.yml).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "docs" / "yolov8n_int8_ptq_pi5_report.md"
IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?:<(?P<path_bracket>[^>]+)>|(?P<path>[^)\n]+))\)")
FIG_RE = re.compile(r"^Figure (\d+)\.")


def check(report: Path) -> list[str]:
    errors: list[str] = []
    text = report.read_text(encoding="utf-8")

    numbers: list[int] = []
    for m in IMG_RE.finditer(text):
        raw_path = m.group("path_bracket") or m.group("path")
        raw_path = raw_path.strip()
        target = (report.parent / raw_path).resolve()
        if not target.is_file() or target.stat().st_size == 0:
            errors.append(f"missing or empty image: {raw_path}")
        fig = FIG_RE.match(m.group("alt"))
        if fig:
            numbers.append(int(fig.group(1)))

    if numbers != list(range(1, len(numbers) + 1)):
        errors.append(f"figure numbers not sequential: {numbers}")
    if text.count("$$") % 2:
        errors.append("unbalanced $$ math delimiters")
    for stray in ("</Content>", "</invoke>"):
        if stray in text:
            errors.append(f"stray tag in report: {stray}")
    return errors


def main() -> int:
    report = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_REPORT
    errors = check(report)
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors))
        return 1
    print(f"OK: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
