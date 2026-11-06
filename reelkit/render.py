"""Plan -> reel.

Deliberately dumb: every decision was already made in `plan.py`, so this
module only executes. That split is what makes a re-render cheap — nothing
here re-measures the footage.

Shots are rendered to normalised intermediates first and concatenated after,
rather than built as one giant filter_complex. With twenty-five shots the
single-graph version becomes unreadable, impossible to debug when one shot is
wrong, and re-renders everything when one shot changes. The intermediates are
all identical in codec, size and rate, so the concat demuxer stream-copies
them and the extra pass costs almost nothing.

Audio is assembled separately from picture, because they do not share cuts:
the narration runs continuously across a clip while the picture cuts away and
back (see plan.py). Each shot carries the audio span that belongs under it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from .audio import gain_db, measure_lufs
from .circle import render_shot as render_circle_shot
from .face import NoFaceError, track_face
from .plan import Plan, Shot
from .style import Style

# ffmpeg xfade names for the style's transition kinds.
XFADE = {"dissolve": "fade", "dipblack": "fadeblack", "dipwhite": "fadewhite",
         "flash": "fadewhite"}

ROOT = Path(__file__).resolve().parent.parent
LOOKS = ROOT / "styles" / "_looks.yaml"
FONTS = ROOT / "fonts"       # optional: drop .ttf files here for burned captions


def _probe_duration(path: Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True,
        check=True).stdout.strip() or 0.0)


def _fit_start(src: Path, start: float, length: float) -> float:
    """Shift a cut backwards so `length` seconds actually exist after it.

    A shot whose span ends at its clip's last frame cannot also supply the
    transition overlap. Rendering it short is the dangerous outcome: an xfade
    chain computes each offset from the previous output, so one short part
    throws off every join after it — measured, five parts 0.18s short turned a
    69-second reel into 39 seconds. Shifting the cut back by a fraction of a
    second changes nothing anyone can see.
    """
    total = _probe_duration(src)
    if start + length <= total:
        return start
    return max(0.0, total - length)


def _grade_filter(style: Style) -> str:
    """eq chain for the style's look, resolved against config.yaml profiles.

    Named profiles live in `styles/_looks.yaml` so several styles can share a
    grade; a style that wants its own numbers overrides any of them inline.
    """
    profiles = (yaml.safe_load(LOOKS.read_text(encoding="utf-8")) or {}).get("look_profiles", {})
    base = dict(profiles.get(style.look.profile) or {})
    look = style.look
    sat = look.saturation if look.saturation is not None else base.get("saturation", 1.0)
    con = look.contrast if look.contrast is not None else base.get("contrast", 1.0)
    gam = look.gamma if look.gamma is not None else base.get("gamma", 1.0)
    bri = base.get("brightness", 0.0)
    sharp = base.get("sharpness", 0.0)
    chain = f"eq=brightness={bri}:contrast={con}:saturation={sat}:gamma={gam}"
    if sharp:
        chain += f",unsharp=5:5:{sharp}"
    return chain


def _fill(style: Style) -> str:
    c = style.canvas
    return (f"scale={c.width}:{c.height}:force_original_aspect_ratio=increase,"
            f"crop={c.width}:{c.height},fps={c.fps},setsar=1")


def _render_plain(shot: Shot, folder: Path, out: Path, style: Style,
                  extra: float = 0.0) -> None:
    src = folder / shot.source
    span = shot.duration + extra
    start = _fit_start(src, shot.src_start, span)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
         "-t", f"{span:.3f}", "-i", str(src),
         "-an", "-vf", f"{_fill(style)},{_grade_filter(style)}",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(out)], check=True)


def _render_with_circle(shot: Shot, folder: Path, out: Path, style: Style,
                        extra: float = 0.0) -> str | None:
    """Circle shot; returns a note when it had to fall back to a plain cut."""
    assert shot.circle is not None
    head = folder / shot.circle.source
    span = shot.duration + extra
    head_start = _fit_start(head, shot.circle.src_start, span)
    try:
        track = track_face(head, head_start, span)
    except NoFaceError as exc:
        _render_plain(shot, folder, out, style, extra)
        return f"план {shot.index}: кружок отменён — {exc}"
    if not track.circleable():
        _render_plain(shot, folder, out, style, extra)
        return (f"план {shot.index}: кружок отменён — на этом куске лицо "
                f"{track.median.size:.0%} кадра, видно в {track.presence:.0%} выборок")

    background = folder / shot.source if shot.circle.mode == "pip" else None
    render_circle_shot(head, out, style=style, track=track,
                       start=head_start, dur=span, background=background)
    return None


def _combine(parts: list[Path], plan: Plan, style: Style, out: Path) -> None:
    """Join the shots — stream-copied hard cuts, or an xfade chain.

    With transitions every part was rendered `duration_sec` longer than its
    slot so the blend has material to chew on; the offsets below then put each
    cut back exactly where the plan put it, which is what keeps picture and
    narration together. Hard cuts skip all of that and stream-copy.
    """
    kind = style.transition.kind
    if kind == "cut" or len(parts) < 2:
        listing = out.parent / "concat.txt"
        # Paths inside a concat list resolve relative to the LIST FILE, not the
        # working directory — a relative path containing the parts directory
        # gets that directory prepended twice. Absolute only.
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", str(out)], check=True)
        listing.unlink(missing_ok=True)
        return

    d = style.transition.duration_sec
    effect = XFADE.get(kind, "fade")
    # Offsets are computed from the parts as rendered, never from the plan.
    # xfade's contract is len(out) = len(a) + len(b) - duration with the
    # transition starting at len(a) - duration, so the running length has to
    # be the real one or every later join lands in the wrong place.
    lengths = [_probe_duration(p) for p in parts]
    inputs: list[str] = []
    for p in parts:
        inputs += ["-i", str(p)]
    chain: list[str] = []
    label = "0:v"
    acc = lengths[0]
    for i in range(1, len(parts)):
        offset = max(0.0, acc - d)
        nxt = f"x{i}"
        chain.append(f"[{label}][{i}:v]xfade=transition={effect}:duration={d}:"
                     f"offset={offset:.3f}[{nxt}]")
        label = nxt
        acc = acc + lengths[i] - d
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs,
                    "-filter_complex", ";".join(chain), "-map", f"[{label}]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def _voice_track(plan: Plan, folder: Path, out: Path, style: Style) -> None:
    """Narration only, loudness-matched — the track captions are read from.

    Kept separate from the final mix on purpose: transcribing the mixed track
    put music under the speech and Whisper lost words to it ("находятся личные
    вещи Юрия Гагарина" came back as "находят Юрия Гагарина") and invented a
    line that was never said. Captions are read from clean speech.
    """
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for i, shot in enumerate(plan.shots):
        inputs += ["-i", str(folder / shot.audio_source)]
        end = shot.audio_start + shot.duration
        filters.append(
            f"[{i}:a]atrim={shot.audio_start:.3f}:{end:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
        labels.append(f"[a{i}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[voice]")
    filters.append(
        f"[voice]loudnorm=I={style.audio.voice_lufs}:TP=-1.5:LRA=11[a]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs,
                    "-filter_complex", ";".join(filters), "-map", "[a]",
                    "-c:a", "aac", "-b:a", "192k", str(out)], check=True)


def _mix_music(voice: Path, out: Path, style: Style, total: float) -> str | None:
    """Lay the style's music bed under the finished narration."""
    music = Path(style.audio.music_file) if style.audio.music_file else None
    if not music:
        return None
    if not music.exists():
        return f"музыка не найдена: {music}"
    mg = gain_db(measure_lufs(music), style.audio.music_lufs)
    fade_at = max(0.0, total - 1.5)
    filters = [
        f"[1:a]aloop=loop=-1:size=2e9,atrim=0:{total:.3f},asetpts=PTS-STARTPTS,"
        f"volume={mg:.2f}dB,afade=t=in:st=0:d=1.0,afade=t=out:st={fade_at:.3f}:d=1.5,"
        f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[music]"]
    if style.audio.ducking:
        # Music ducks under the voice instead of fighting it. The voice is
        # split because a sidechain cannot key off a stream it also outputs.
        filters.append("[0:a]asplit=2[vmain][vkey]")
        filters.append("[music][vkey]sidechaincompress=threshold=0.03:ratio=8:"
                       "attack=20:release=300[ducked]")
        filters.append("[vmain][ducked]amix=inputs=2:duration=first:"
                       "dropout_transition=0:normalize=0[mixed]")
    else:
        filters.append("[0:a][music]amix=inputs=2:duration=first:"
                       "dropout_transition=0:normalize=0[mixed]")
    filters.append(f"[mixed]alimiter=limit={style.audio.limiter}:level=disabled[a]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice), "-i", str(music),
                    "-filter_complex", ";".join(filters), "-map", "[a]",
                    "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    return None


def _maybe_captions(voice: Path, out_dir: Path, style: Style):
    """Subtitles if the style wants them AND the transcription stack is here.

    The whole captions path — ASR, the ASS writer, the correction table — is
    an OPTIONAL extra. Everything else in this package (cutting, grading,
    circles, music, transitions) is deterministic and needs no model at all,
    so a style with `captions.enabled: false` renders with nothing installed
    beyond ffmpeg. When a style does ask for captions and the extra is
    missing, that is said out loud rather than silently dropped.
    """
    if not style.captions.enabled:
        return None, []
    try:
        from .captions import load_caption_config, make_captions
    except ImportError as exc:
        return None, [f"субтитры пропущены: не установлен необязательный блок "
                      f"распознавания ({exc.name}). См. README → «Субтитры»."]
    cfg = load_caption_config()
    if cfg is None:
        return None, ["субтитры пропущены: нет config.yaml со стилями субтитров "
                      "(нужен основной репозиторий flexi). См. README → «Субтитры»."]
    return make_captions(voice, out_dir, style, cfg)


def render(plan: Plan, style: Style, out_dir: Path, *, keep_parts: bool = False) -> Path:
    """Render `plan` into `out_dir/final.mp4`. Returns the final path."""
    folder = Path(plan.folder)
    # Known gap, surfaced rather than swallowed: the renderer concatenates
    # hard cuts. A style asking for a dissolve would otherwise render as cuts
    # with nothing to say so, and the difference is easy to miss on review.
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # With a transition every shot but the last needs overlap material.
    overlap = style.transition.duration_sec if style.transition.kind != "cut" else 0.0
    parts: list[Path] = []
    for shot in plan.shots:
        part = parts_dir / f"shot_{shot.index:03d}.mp4"
        extra = 0.0 if shot.index == plan.shots[-1].index else overlap
        if shot.circle is not None:
            note = _render_with_circle(shot, folder, part, style, extra)
            if note:
                notes.append(note)
        else:
            _render_plain(shot, folder, part, style, extra)
        parts.append(part)

    silent = out_dir / "video_silent.mp4"
    _combine(parts, plan, style, silent)

    voice = out_dir / "voice.m4a"
    _voice_track(plan, folder, voice, style)

    ass, cap_notes = _maybe_captions(voice, out_dir, style)
    notes.extend(cap_notes)

    audio = out_dir / "audio_mixed.m4a"
    music_note = _mix_music(voice, audio, style, plan.total_sec)
    if music_note:
        notes.append(music_note)
    if not audio.exists():
        audio = voice

    final = out_dir / "final.mp4"
    if ass is None:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(audio),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
                        "-shortest", str(final)], check=True)
    else:
        # Burning needs a re-encode anyway, so the mux and the burn are one
        # pass. fontsdir keeps libass off the system font book: the bundled
        # families are the ones the styles name.
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(audio),
             "-map", "0:v", "-map", "1:a",
             "-vf", f"ass={ass}:fontsdir={FONTS}",
             "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(final)], check=True)

    if not keep_parts:
        for p in parts_dir.glob("shot_*.mp4"):
            p.unlink()
        parts_dir.rmdir()
        silent.unlink(missing_ok=True)

    (out_dir / "render_notes.json").write_text(
        json.dumps({"notes": notes, "warnings": plan.warnings}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    for n in notes:
        print(f"  ! {n}")
    return final
