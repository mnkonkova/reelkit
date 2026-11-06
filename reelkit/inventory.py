"""What is actually in the dropped folder.

The engine is given a folder of clips and nothing else — no script, no naming
convention, no tags. Everything downstream (who narrates, what fills the
frame, where a circle can go) is decided from what this module measures.

The one classification that matters is talking-head vs b-roll, and the useful
signal turned out NOT to be sound. On the material this was built against the
presenter narrates continuously — including while filming the exhibits — so
every clip carries speech and speech cannot separate them. What separates them
is whether SHE IS BIG ENOUGH IN FRAME to be the subject (see
`face.FaceTrack.circleable`). Speech is still measured, because the audio of
every clip becomes part of the voiceover spine and a clip of pure room tone
should not drag silence into it.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .face import FaceModelMissing, FaceTrack, NoFaceError, track_face

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac"}
STILL_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
MUSIC_HINTS = ("music", "bgm", "track", "музык")

# The silence gate is set RELATIVE to each clip, not as a fixed dB. Measured
# on the museum material, a fixed -32 dB found no silence at all in any clip:
# that is roughly where those clips' own mean level sits, so the gate was
# under the room tone and everything read as speech. Phone audio from a quiet
# hall and audio from a street differ by far more than the margin between
# speech and pause, so only a per-clip gate works.
SILENCE_BELOW_MEAN_DB = 8.0   # gate = clip mean - this
SILENCE_MIN = 0.35            # shorter dips are breaths between words
# Telling an audio-only narration file from a music track. The first rule
# tried was "how much of the file is above the gate", and it put a quiet
# ambient track at 0.95 right next to speech — no margin to set a threshold
# in. Pause DENSITY separates them by an order of magnitude: measured across
# six library tracks and one narration, music produced 0.0-1.2 pauses per
# minute and speech produced 12.1. Someone reading without breathing would
# have to beat 4/min to be mistaken for music.
VOICE_PAUSES_PER_MIN = 4.0

# `voiceover` is an audio-only file carrying narration — the spine of an edit
# where the picture is all silent b-roll. It is kept apart from `music`
# because the two play opposite roles and are easy to confuse on disk.
Kind = Literal["talking_head", "broll", "voiceover", "music", "still"]


@dataclass(frozen=True)
class Clip:
    path: Path
    kind: Kind
    reason: str                 # why it landed in that kind, in plain words
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    speech_sec: float = 0.0     # non-silent seconds
    speech_start: float = 0.0   # first sound — where a trim should begin
    speech_end: float = 0.0
    # Spoken runs as (start, end). A single long take needs these to drop its
    # pauses; the head/tail pair above is not enough to cut inside a clip.
    speech_runs: tuple[tuple[float, float], ...] = ()
    face: FaceTrack | None = None

    @property
    def speech_ratio(self) -> float:
        return self.speech_sec / self.duration if self.duration else 0.0

    def as_dict(self) -> dict:
        """Plan-friendly form; the face track is summarised, not dumped."""
        d = {"file": self.path.name, "kind": self.kind, "reason": self.reason,
             "duration": round(self.duration, 3),
             "size": f"{self.width}x{self.height}" if self.width else None,
             "speech_sec": round(self.speech_sec, 2),
             "speech_span": [round(self.speech_start, 2), round(self.speech_end, 2)]}
        if self.face is not None:
            m = self.face.median
            d["face"] = {"size": round(m.size, 3), "presence": round(self.face.presence, 2),
                         "cx": round(m.cx, 3), "cy": round(m.cy, 3)}
        return d


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = 0.0
    if video and video.get("r_frame_rate", "0/0") != "0/0":
        num, den = (float(x) for x in video["r_frame_rate"].split("/"))
        fps = num / den if den else 0.0
    return {
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
        "width": int(video.get("width") or 0) if video else 0,
        "height": int(video.get("height") or 0) if video else 0,
        "fps": fps,
        "has_video": video is not None,
        "has_audio": audio is not None,
    }


def _mean_volume(path: Path) -> float | None:
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", res)
    return float(m.group(1)) if m else None


def _speech_profile(path: Path, duration: float):
    """(spoken seconds, first sound, last sound) via ffmpeg silencedetect.

    Cheap stand-in for a VAD: good enough to trim dead air off the head and
    tail of a clip and to notice a clip with no narration at all. It is not
    asked to tell speech from other sound, because in this material the only
    other sound is a quiet museum.
    """
    mean = _mean_volume(path)
    if mean is None:
        return duration, 0.0, duration, ((0.0, duration),)
    gate = mean - SILENCE_BELOW_MEAN_DB
    # NOTE: silencedetect reports at ffmpeg's *info* log level. Running it
    # with `-v error` (the habit everywhere else in this file) silently
    # swallows every silence_start line and the clip reads as 100% speech.
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path),
         "-af", f"silencedetect=noise={gate:.1f}dB:d={SILENCE_MIN}",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(m) for m in re.findall(r"silence_start: (-?[\d.]+)", res)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", res)]

    # Invert the silences into spoken runs.
    quiet: list[tuple[float, float]] = []
    for i, st in enumerate(starts):
        en = ends[i] if i < len(ends) else duration
        quiet.append((max(0.0, st), min(en, duration)))
    runs: list[tuple[float, float]] = []
    cursor = 0.0
    for qs, qe in quiet:
        if qs - cursor > 0.05:
            runs.append((cursor, qs))
        cursor = max(cursor, qe)
    if duration - cursor > 0.05:
        runs.append((cursor, duration))
    if not runs:
        runs = [(0.0, duration)]

    spoken = sum(e - s for s, e in runs)
    return spoken, runs[0][0], runs[-1][1], tuple(runs)


def scan(folder: Path, *, probe_faces: bool = True) -> list[Clip]:
    """Everything in `folder`, measured and classified, in filename order.

    Filename order is deliberate: camera files are numbered chronologically,
    and on a continuously-narrated shoot that order IS the order of the
    narration. Nothing downstream reorders clips without being told to.
    """
    clips: list[Clip] = []
    face_detection_off: str | None = None
    for path in sorted(Path(folder).iterdir(), key=lambda p: p.name.lower()):
        ext = path.suffix.lower()
        if path.name.startswith("."):
            continue
        if ext in STILL_EXTS:
            clips.append(Clip(path=path, kind="still", reason="картинка", duration=0.0))
            continue
        if ext in AUDIO_EXTS:
            info = _probe(path)
            spoken, head, tail, runs = _speech_profile(path, info["duration"])
            minutes = info["duration"] / 60 if info["duration"] else 0.0
            density = (len(runs) - 1) / minutes if minutes else 0.0
            if any(h in path.stem.lower() for h in MUSIC_HINTS):
                kind, why = "music", "музыка по имени файла"
            elif density >= VOICE_PAUSES_PER_MIN:
                kind, why = "voiceover", f"{density:.0f} пауз в минуту — похоже на озвучку"
            else:
                kind, why = "music", f"{density:.1f} пауз в минуту — похоже на музыку"
            clips.append(Clip(path=path, kind=kind, reason=why, duration=info["duration"],
                              has_audio=True, speech_sec=spoken, speech_start=head,
                              speech_end=tail, speech_runs=runs))
            continue
        if ext not in VIDEO_EXTS:
            continue

        info = _probe(path)
        if not info["has_video"]:
            clips.append(Clip(path=path, kind="music", reason="нет видеодорожки",
                              duration=info["duration"], has_audio=info["has_audio"]))
            continue
        if any(h in path.stem.lower() for h in MUSIC_HINTS):
            clips.append(Clip(path=path, kind="music", reason="музыка по имени файла",
                              duration=info["duration"], has_audio=info["has_audio"]))
            continue

        spoken = head = 0.0
        tail = info["duration"]
        runs: tuple[tuple[float, float], ...] = ((0.0, info["duration"]),)
        if info["has_audio"]:
            spoken, head, tail, runs = _speech_profile(path, info["duration"])

        face = None
        if probe_faces and not face_detection_off:
            try:
                face = track_face(path)
            except FaceModelMissing as exc:
                # Say it once, then stop trying: every later clip would fail
                # the same way and the folder would read as "no people here".
                face_detection_off = str(exc)
            except NoFaceError:
                face = None

        if face is not None and face.circleable():
            m = face.median
            kind, reason = "talking_head", (
                f"лицо {m.size:.0%} ширины кадра, видно в {face.presence:.0%} выборок")
        elif face is not None:
            m = face.median
            kind, reason = "broll", (
                f"лицо есть, но мелкое ({m.size:.0%}) или нестабильное "
                f"({face.presence:.0%}) — как b-roll")
        elif face_detection_off:
            kind, reason = "broll", "детектор лиц недоступен — всё как b-roll"
        else:
            kind, reason = "broll", "лиц нет"

        clips.append(Clip(path=path, kind=kind, reason=reason, duration=info["duration"],
                          width=info["width"], height=info["height"], fps=info["fps"],
                          has_audio=info["has_audio"], speech_sec=spoken,
                          speech_start=head, speech_end=tail, speech_runs=runs,
                          face=face))
    return clips


def face_detection_warning(clips: list[Clip]) -> str | None:
    """A line to print when nothing could be checked for faces."""
    if clips and all(c.face is None for c in clips if c.kind in ("talking_head", "broll")) \
            and any("детектор лиц недоступен" in c.reason for c in clips):
        return ("детектор лиц не работает: говорящий от перебивки не отличается, "
                "кружки выключены. См. README → «Модель лиц».")
    return None


def summary(clips: list[Clip]) -> str:
    counts: dict[str, int] = {}
    for c in clips:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    total = sum(c.duration for c in clips if c.kind != "still")
    parts = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    return f"{len(clips)} файлов ({parts}), всего {total:.0f} с"
