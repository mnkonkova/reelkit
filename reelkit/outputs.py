"""Versioned output folders: output/<slug>/vN/ with a `latest` pointer.

Re-rendering the same reel adds a new version beside the old one instead of
overwriting it, because the usual reason to re-render is to compare.
"""
from __future__ import annotations

import re
from pathlib import Path

VERSION_RE = re.compile(r"^v(\d+)$")
LATEST = "latest"


def existing_versions(base: Path) -> list[int]:
    if not base.is_dir():
        return []
    out = []
    for child in base.iterdir():
        if child.is_dir() and not child.is_symlink():
            m = VERSION_RE.match(child.name)
            if m:
                out.append(int(m.group(1)))
    return sorted(out)


def update_latest(base: Path, version: str) -> None:
    link = base / LATEST
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(version, target_is_directory=True)


def versioned_dir(base: Path, version: str | None = None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if version:
        name = version if VERSION_RE.match(version) else f"v{version}"
    else:
        nums = existing_versions(base)
        name = f"v{(nums[-1] + 1) if nums else 1}"
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    update_latest(base, name)
    return target
