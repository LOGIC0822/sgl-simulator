#!/usr/bin/env python3
"""Atomically merge supplemental AIC perf rows into an existing CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    header, existing_rows = read_table(args.existing)
    supplement_header, supplement_rows = read_table(args.supplement)
    if supplement_header != header:
        raise ValueError(
            f"Header mismatch:\nexisting={header}\nsupplement={supplement_header}"
        )
    key_fields = [field for field in header if field not in {"latency", "power", "energy"}]
    seen = {tuple(row.get(field, "") for field in key_fields) for row in existing_rows}
    new_rows = []
    for row in supplement_rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key not in seen:
            seen.add(key)
            new_rows.append(row)

    args.backup.parent.mkdir(parents=True, exist_ok=True)
    if not args.backup.exists():
        shutil.copy2(args.existing, args.backup)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{args.existing.name}.", dir=args.existing.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)
        os.replace(temporary_name, args.existing)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "existing": str(args.existing),
                "supplement": str(args.supplement),
                "backup": str(args.backup),
                "rows_before": len(existing_rows),
                "supplement_rows": len(supplement_rows),
                "rows_added": len(new_rows),
                "rows_after": len(existing_rows) + len(new_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
