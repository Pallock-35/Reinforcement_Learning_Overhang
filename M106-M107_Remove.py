#!/usr/bin/env python3
"""Remove M106/M107 fan commands from G-code file(s)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def is_fan_command(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(";"):
        return False
    first_token = stripped.split()[0].upper()
    return first_token in {"M106", "M107"}


def remove_fan_commands(lines: list[str]) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    for line in lines:
        if is_fan_command(line):
            removed += 1
        else:
            kept.append(line)
    return kept, removed


def process_gcode_file(input_path: Path, output_path: Path) -> int:
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    filtered, removed = remove_fan_commands(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(filtered), encoding="utf-8")
    return removed


def process_file(input_path: Path, output_path: Path | None) -> int:
    destination = output_path or input_path
    removed = process_gcode_file(input_path, destination)
    print(f"Removed {removed} fan command(s) from {input_path}")
    if output_path is not None:
        print(f"Wrote {destination}")
    return removed


def process_folder(input_dir: Path, output_dir: Path | None) -> int:
    gcode_files = sorted(input_dir.glob("*.gcode"))
    if not gcode_files:
        print(f"Error: no .gcode files found in {input_dir}", file=sys.stderr)
        return 1

    total_removed = 0
    for gcode_file in gcode_files:
        destination = (output_dir / gcode_file.name) if output_dir is not None else gcode_file
        removed = process_gcode_file(gcode_file, destination)
        total_removed += removed
        print(f"Removed {removed} fan command(s) from {gcode_file.name}")
        if output_dir is not None:
            print(f"  -> {destination}")

    print(
        f"Processed {len(gcode_files)} file(s), "
        f"removed {total_removed} fan command(s) total"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove M106/M107 fan commands from G-code file(s).",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="G-code file or folder containing .gcode files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (single input) or output folder (folder input)",
    )
    args = parser.parse_args()

    input_path = args.path
    if input_path.is_file():
        process_file(input_path, args.output)
        return 0
    if input_path.is_dir():
        return process_folder(input_path, args.output)

    print(f"Error: path not found: {input_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
