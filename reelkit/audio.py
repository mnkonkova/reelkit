"""Loudness measurement, so a music bed can be matched to a target level.

Lifted from the parent project's `shnurok/audio_mix.py` rather than imported:
these two functions are the whole of what the renderer needed from it, and
copying them is what lets this package stand alone.
"""
from __future__ import annotations

import re
import subprocess

_I_RE = re.compile(r"\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS")


def parse_ebur128_i(stderr: str) -> float:
    matches = _I_RE.findall(stderr)
    if not matches:
        raise ValueError("ffmpeg ebur128 не вернул интегральную громкость")
    return float(matches[-1])


def measure_lufs(path, start: float | None = None, dur: float | None = None) -> float:
    af = "ebur128"
    if start is not None or dur is not None:
        trim = "atrim=" + (f"{start}" if start is not None else "0")
        if dur is not None:
            trim += f":{(start or 0) + dur}"
        af = f"{trim},asetpts=PTS-STARTPTS,ebur128"
    # NOTE: ebur128 reports at ffmpeg's *info* level; `-v error` swallows it.
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", af,
                        "-f", "null", "-"], capture_output=True, text=True)
    return parse_ebur128_i(r.stderr)


def gain_db(measured_lufs: float, target_lufs: float) -> float:
    return target_lufs - measured_lufs
