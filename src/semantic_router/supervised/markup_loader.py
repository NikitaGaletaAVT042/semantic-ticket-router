from __future__ import annotations

import re
from pathlib import Path


TXSUP_RE = re.compile(r"TXSUP-\d+", re.IGNORECASE)


def load_key_to_label(path: Path) -> dict[str, str]:
    key_to_label: dict[str, str] = {}
    current_label: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        match = TXSUP_RE.search(line)
        if match:
            if current_label is not None:
                key_to_label[match.group(0).upper()] = current_label
            continue

        current_label = line

    return key_to_label