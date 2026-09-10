"""Print-run folder naming: print_NNN_<mode>_a<angle>."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PRINT_RUNS_DIR = "print_runs"
PRINT_PHOTOS_SUBDIR = "photos"

VALID_CONTROL_MODES = frozenset(
    {"fixed", "mpc", "mpc_temp"},
)

# print_001 | print 1 | print_001_mpc_a60 | print_001_mpc_temp_a62p5
PRINT_RUN_FOLDER_RE = re.compile(
    r"^print[_\s](?P<number>\d+)"
    r"(?:_(?P<mode>[a-z][a-z0-9_]+)_(?P<angle>a[\d.p]+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PrintRunFolderParts:
    print_number: int
    control_mode: str | None = None
    print_angle_deg: float | None = None


def normalize_control_mode_slug(control_mode: str) -> str:
    """Sanitize a control-mode slug for folder names.

    Thermal live modes are ``fixed``, ``mpc``, and ``mpc_temp``
    (see ``VALID_CONTROL_MODES``). Legacy Q-learning folders may still use
    ``rl`` / ``open_loop``; those are accepted for naming only.
    """
    slug = control_mode.strip().lower().replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    if not slug or not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
        raise ValueError(
            f"Invalid control_mode '{control_mode}'. "
            f"Expected a slug like: {', '.join(sorted(VALID_CONTROL_MODES))}"
        )
    return slug


def format_angle_slug(angle_deg: float) -> str:
    """Encode overhang angle for folder names, e.g. 60 -> a60, 62.5 -> a62p5."""
    text = f"{float(angle_deg):g}"
    return f"a{text.replace('.', 'p')}"


def parse_angle_slug(angle_slug: str) -> float:
    if not angle_slug.lower().startswith("a"):
        raise ValueError(f"Invalid angle slug: {angle_slug}")
    return float(angle_slug[1:].replace("p", "."))


def print_run_folder_name(
    print_number: int,
    *,
    control_mode: str | None = None,
    print_angle_deg: float | None = None,
) -> str:
    """Return a run folder name.

  Schema:
    print_{NNN}_{control_mode}_a{angle}

  Examples:
    print_001_fixed_a60
    print_002_mpc_a70
    print_003_mpc_temp_a62p5

  When control_mode or print_angle_deg is omitted, returns the legacy short
  form ``print_{NNN}`` for backward compatibility.
    """
    number_slug = f"print_{print_number:03d}"
    if control_mode is None or print_angle_deg is None:
        return number_slug
    mode_slug = normalize_control_mode_slug(control_mode)
    angle_slug = format_angle_slug(print_angle_deg)
    return f"{number_slug}_{mode_slug}_{angle_slug}"


def parse_print_run_folder_parts(key: str) -> PrintRunFolderParts | None:
    match = PRINT_RUN_FOLDER_RE.match(key.strip())
    if not match:
        return None
    print_number = int(match.group("number"))
    mode = match.group("mode")
    angle = match.group("angle")
    if mode is None or angle is None:
        return PrintRunFolderParts(print_number=print_number)
    return PrintRunFolderParts(
        print_number=print_number,
        control_mode=mode.lower(),
        print_angle_deg=parse_angle_slug(angle),
    )


def parse_print_number_from_run_key(key: str) -> int | None:
    text = key.strip()
    parts = parse_print_run_folder_parts(text)
    if parts is not None:
        return parts.print_number
    if text.isdigit():
        return int(text)
    return None


def normalize_print_run_key(key: str) -> str:
    text = key.strip()
    parts = parse_print_run_folder_parts(text)
    if parts is None:
        if text.isdigit():
            return print_run_folder_name(int(text))
        return text
    if parts.control_mode is None or parts.print_angle_deg is None:
        return print_run_folder_name(parts.print_number)
    return print_run_folder_name(
        parts.print_number,
        control_mode=parts.control_mode,
        print_angle_deg=parts.print_angle_deg,
    )


def print_run_dir(
    print_number: int,
    runs_root: Path | str = PRINT_RUNS_DIR,
    *,
    control_mode: str | None = None,
    print_angle_deg: float | None = None,
) -> Path:
    return Path(runs_root) / print_run_folder_name(
        print_number,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    )


def print_run_photos_dir(
    print_number: int,
    runs_root: Path | str = PRINT_RUNS_DIR,
    *,
    control_mode: str | None = None,
    print_angle_deg: float | None = None,
) -> Path:
    return print_run_dir(
        print_number,
        runs_root,
        control_mode=control_mode,
        print_angle_deg=print_angle_deg,
    ) / PRINT_PHOTOS_SUBDIR
