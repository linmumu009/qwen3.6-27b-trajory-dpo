#!/usr/bin/env python3
"""Select PI prompt rows by index while keeping row content server-side."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = list(dict.fromkeys(args.indices))
    if len(indices) != len(args.indices):
        raise ValueError("duplicate indices are not allowed")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise IndexError(f"indices outside dataset of {len(rows)} rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index in indices:
            handle.write(
                json.dumps(rows[index], ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "selected_indices": indices,
                "output_rows": len(indices),
                "output_sha256": sha256(output),
                "row_content_emitted": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
