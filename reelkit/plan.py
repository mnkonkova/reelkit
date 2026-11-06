"""Turn an inventory into an edit plan: what is on screen, second by second.

The plan is a plain JSON artifact on purpose. Analysis (face tracking,
transcription) is the slow part; rendering is the cheap part. Keeping the
decision in a file means a look can be re-rendered, a shot re-pointed by hand,
or a bad choice argued with, without re-measuring anything — the same split
that makes `capcut-scenario-builder`'s plan.json useful.

The structure it builds comes from the material rather than from a template:

* **Audio is the spine.** The presenter narrates continuously, in-camera, so
  every clip's own audio is part of the voiceover and the clips' order is the
  order of the narration. Nothing is re-timed against the picture; the picture
  is cut to fit the audio.
* **Picture cuts more often than the audio changes.** Measured on the
  reference edit: 74 seconds of narration over 9 source clips, but 22 shots.
  A clip whose narration runs longer than the style's shot budget keeps
  talking while the picture cuts away to other footage and (sometimes) back.
* **A talking-head clip is an opportunity, not an obligation.** With a circle
  style the presenter goes into the circle and b-roll takes the full frame —
  same audio, livelier picture.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .inventory import Clip
from .style import Style

PAD_SEC = 0.12      # breathing room around a trimmed speech span

# What kind of material is in the folder. The edit strategy follows from this,
# and getting it wrong produces a confidently wrong reel rather than an error,
# so it is decided once, explicitly, and written into the plan.
Shape = Literal["narrated_clips", "voiceover_plus_broll", "single_take"]
SINGLE_TAKE_SHARE = 0.70   # one clip holding this much of the footage is "the take"


@dataclass
class CirclePlan:
    source: str          # clip the head comes from
    src_start: float
    background: str | None   # clip filling the frame behind it (pip only)
    mode: str
    note: str = ""


@dataclass
class Shot:
    index: int
    kind: str            # talking | broll | cutaway
    source: str          # file shown full-frame (or behind the circle)
    src_start: float
    src_end: float
    timeline_start: float
    duration: float
    audio_source: str    # which clip's audio plays here — not always `source`
    audio_start: float
    circle: CirclePlan | None = None
    note: str = ""


@dataclass
class Plan:
    style: str
    shape: str
    shape_reason: str
    folder: str
    canvas: str
    total_sec: float
    shots: list[Shot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return path

    def describe(self) -> str:
        lines = [f"{len(self.shots)} планов, {self.total_sec:.1f} с, стиль {self.style}",
                 f"форма исходников: {self.shape} — {self.shape_reason}"]
        for s in self.shots:
            circ = ""
            if s.circle:
                circ = f"  ◍ {s.circle.mode} из {s.circle.source}"
            lines.append(
                f"  {s.index:2d}  {s.timeline_start:5.1f}с +{s.duration:4.1f}  "
                f"{s.kind:8} {s.source:14} [{s.src_start:4.1f}..{s.src_end:5.1f}]{circ}")
        return "\n".join(lines)


class _BrollPool:
    """Hands out b-roll slices, spreading use instead of repeating a favourite.

    Each clip is walked forward through its own footage so a second visit
    shows a different moment, and the least-recently-used clip is always
    offered first. Without this, every cutaway in a reel came from the same
    clip — the picker has no idea what a shot is *of*, only that it has not
    been shown lately.
    """

    def __init__(self, clips: list[Clip]):
        self._clips = list(clips)
        self._cursor = {c.path.name: c.speech_start for c in clips}
        self._last_used = {c.path.name: -1 for c in clips}
        self._last_handed: str | None = None
        self._tick = 0

    def take(self, want: float, *, avoid: str | None = None) -> tuple[Clip, float, float] | None:
        """Exactly `want` seconds of some clip, or None if no clip is that long.

        Returning a SHORTER slice was the original behaviour and it silently
        broke the timeline: the plan recorded the requested duration while the
        renderer produced a shorter piece, so picture and narration drifted
        apart by 2.3s over a 69s reel. A slice is now either the full length
        or refused, and a clip that has run out is rewound rather than
        shortened.
        """
        self._tick += 1
        # Exclude both the clip whose audio is running (showing it would make
        # the cutaway pointless) and whatever was handed out last. Without the
        # second rule the longest clip wins every time it still has footage
        # left, and the reel shows three cutaways from it back to back.
        banned = {avoid, self._last_handed} - {None}
        pool = [c for c in self._clips if c.path.name not in banned]
        pool = pool or [c for c in self._clips if c.path.name != avoid] or self._clips
        if not pool:
            return None
        pool.sort(key=lambda c: (self._last_used[c.path.name], c.path.name))
        long_enough = [c for c in pool if c.speech_end - c.speech_start >= want]
        if not long_enough:
            return None
        for clip in long_enough:
            start = self._cursor[clip.path.name]
            if clip.speech_end - start < want:
                start = clip.speech_end - want   # rewind to fit a full slice
            end = start + want
            self._cursor[clip.path.name] = end
            self._last_used[clip.path.name] = self._tick
            self._last_handed = clip.path.name
            return clip, start, end
        return None


def _slice_count(duration: float, target: float, maximum: float) -> int:
    """How many shots a span of narration should be cut into."""
    if duration <= maximum:
        return 1
    return max(2, round(duration / target))


def build_plan(clips: list[Clip], style: Style, folder: Path) -> Plan:
    """Detect what kind of material this is, then lay it out under `style`.

    One entry point for every source shape on purpose: the caller drops a
    folder and does not have to know which strategy it needs. The chosen shape
    and the reason for it go into the plan so a wrong guess is visible before
    the render rather than after.
    """
    shape, why = detect_shape(clips)
    plan = Plan(style=style.id, shape=shape, shape_reason=why, folder=str(folder),
                canvas=f"{style.canvas.width}x{style.canvas.height}@{style.canvas.fps}",
                total_sec=0.0)
    if shape == "voiceover_plus_broll":
        plan = _plan_voiceover(clips, style, folder, plan)
    elif shape == "single_take":
        plan = _plan_single_take(clips, style, folder, plan)
    else:
        plan = _plan_narrated(clips, style, folder, plan)
    return _check_plan(_merge_contiguous(plan), style)


def _merge_contiguous(plan: Plan) -> Plan:
    """Fuse neighbouring shots that are really one continuous piece of footage.

    When the b-roll pool runs dry a cutaway falls back to staying on the
    clip, and the plan then holds several shots from the same source at
    consecutive timecodes. On screen that is one unbroken shot, but the
    renderer treats each as a cut — with dissolves on, it would cross-fade a
    shot into itself four times mid-sentence. Merging keeps the audio
    untouched (the slices were contiguous there too) and removes the fake cuts.
    """
    merged: list[Shot] = []
    for shot in plan.shots:
        prev = merged[-1] if merged else None
        same = (prev is not None and prev.source == shot.source
                and prev.circle is None and shot.circle is None
                and abs(prev.src_end - shot.src_start) < 0.02
                and prev.audio_source == shot.audio_source
                and abs(prev.audio_start + prev.duration - shot.audio_start) < 0.02)
        if same:
            prev.src_end = round(shot.src_end, 3)
            prev.duration = round(prev.duration + shot.duration, 3)
            continue
        merged.append(shot)
    t = 0.0
    for i, shot in enumerate(merged):
        shot.index = i
        shot.timeline_start = round(t, 3)
        t += shot.duration
    plan.shots = merged
    plan.total_sec = round(t, 3)
    return plan


def _check_plan(plan: Plan, style: Style) -> Plan:
    """Invariants every strategy has to satisfy before anything is rendered."""
    # The renderer trusts `duration`; if a shot's source span does not supply
    # it, the picture comes out short and drifts against the narration.
    for s_ in plan.shots:
        have = s_.src_end - s_.src_start
        if abs(have - s_.duration) > 0.02:
            plan.warnings.append(
                f"план {s_.index}: {s_.source} даёт {have:.2f}с, а в плане {s_.duration:.2f}с "
                f"— картинка разъедется со звуком")
    short = [s.index for s in plan.shots if s.duration < style.pacing.shot_min_sec]
    if short:
        plan.warnings.append(
            f"планы короче {style.pacing.shot_min_sec}с: {short} — проверь, не рябит ли")
    # Audio slices of one source must tile without holes: a gap here silently
    # deletes narration while the picture covers for it.
    by_src: dict[str, list[Shot]] = {}
    for s_ in plan.shots:
        by_src.setdefault(s_.audio_source, []).append(s_)
    for name, group in by_src.items():
        group.sort(key=lambda x: x.audio_start)
        for a, b in zip(group, group[1:]):
            gap = b.audio_start - (a.audio_start + a.duration)
            if gap > 0.02:
                plan.warnings.append(
                    f"{name}: между планами {a.index} и {b.index} теряется {gap:.2f}с речи")
    return plan


def _plan_narrated(clips: list[Clip], style: Style, folder: Path, plan: Plan) -> Plan:
    """Several clips, each narrated in camera — the shape this engine began on."""
    usable = [c for c in clips if c.kind in ("talking_head", "broll") and c.duration > 0]
    if not usable:
        plan.warnings.append("в папке нет пригодных видеоклипов")
        return plan

    broll = [c for c in usable if c.kind == "broll"]
    if not broll:
        plan.warnings.append(
            "нет ни одного b-roll — перебивки брать неоткуда, "
            "длинные реплики останутся одним планом")
    pool = _BrollPool(broll) if broll else None
    pacing = style.pacing
    circle_mode = style.circle.mode

    # `circle.when` decides how many shots may carry a circle. `speaking` and
    # `always` currently coincide: the gate for a circle IS "a usable head is
    # in this clip" (face.circleable), and there is no per-clip speech signal
    # that would separate them — the presenter narrates every clip, on camera
    # or not. `hook_only` is a real restriction and is honoured below.
    circles_used = 0
    t = 0.0
    index = 0
    for clip in usable:
        span_start = max(0.0, clip.speech_start - PAD_SEC)
        span_end = min(clip.duration, clip.speech_end + PAD_SEC)
        span = span_end - span_start
        if span < 0.4:
            plan.warnings.append(f"{clip.path.name}: почти нет полезного звука, пропущен")
            continue

        n = _slice_count(span, pacing.shot_target_sec, pacing.shot_max_sec)
        if pool is None:
            n = 1

        # Pieces must TILE the span exactly. The opening shot of the reel is
        # held shorter (a reel that starts on a long establishing shot loses
        # people before the first sentence lands), and the time it gives up is
        # handed to the pieces after it rather than dropped. Shortening a
        # piece without redistributing used to delete 0.78s of narration at
        # the top of every reel — "находятся личные вещи" came out as
        # "находят", and the loss was invisible because the picture still
        # covered the gap.
        pieces = [span / n] * n
        if index == 0 and n > 1 and pieces[0] > pacing.opening_shot_sec:
            freed = pieces[0] - pacing.opening_shot_sec
            pieces[0] = pacing.opening_shot_sec
            for j in range(1, n):
                pieces[j] += freed / (n - 1)

        a_start = span_start
        for k in range(n):
            dur = pieces[k]

            circle = None
            circle_allowed = circle_mode != "off" and not (
                style.circle.when == "hook_only" and circles_used >= 1)
            if k == 0 and clip.kind == "talking_head" and circle_allowed:
                if circle_mode == "pip":
                    got = pool.take(dur, avoid=clip.path.name) if pool else None
                    if got is None:
                        shown, s0, s1 = clip, a_start, a_start + dur
                        note = "pip просили, но b-roll под него не нашлось — кадр как есть"
                    else:
                        bg, s0, s1 = got
                        shown = bg
                        circle = CirclePlan(source=clip.path.name, src_start=a_start,
                                            background=bg.path.name, mode="pip")
                        note = "ведущая в кружке поверх перебивки"
                else:  # main
                    shown, s0, s1 = clip, a_start, a_start + dur
                    circle = CirclePlan(source=clip.path.name, src_start=a_start,
                                        background=None, mode="main")
                    note = "ведущая крупным кругом, фон размыт"
                kind = "talking"
                if circle is not None:
                    circles_used += 1
            elif k == 0:
                shown, s0, s1 = clip, a_start, a_start + dur
                kind = "talking" if clip.kind == "talking_head" else "broll"
                note = ""
            else:
                # Mid-sentence cutaway: the audio keeps running on this clip,
                # the picture goes elsewhere.
                got = pool.take(dur, avoid=clip.path.name) if pool else None
                if got is None:
                    shown, s0, s1 = clip, a_start, a_start + dur
                    kind, note = "broll", "перебивка кончилась — остаёмся на клипе"
                else:
                    shown, s0, s1 = got
                    kind, note = "cutaway", "перебивка под продолжающуюся реплику"

            a_start_this = a_start
            a_start += dur
            plan.shots.append(Shot(
                index=index, kind=kind, source=shown.path.name,
                src_start=round(s0, 3), src_end=round(s1, 3),
                timeline_start=round(t, 3), duration=round(dur, 3),
                audio_source=clip.path.name, audio_start=round(a_start_this, 3),
                circle=circle, note=note))
            t += dur
            index += 1

    plan.total_sec = round(t, 3)
    return plan


def detect_shape(clips: list[Clip]) -> tuple[Shape, str]:
    """Which of the three source shapes this folder is, and why.

    Decided from measurements the inventory already took, in a fixed order —
    a separate narration file is the strongest signal there is, and one clip
    holding most of the footage is the next. Anything else is the case the
    engine was built on first: several clips, each narrated in camera.
    """
    voice = [c for c in clips if c.kind == "voiceover"]
    videos = [c for c in clips if c.kind in ("talking_head", "broll")]
    if not videos:
        return "narrated_clips", "видеоклипов нет"
    if voice:
        return ("voiceover_plus_broll",
                f"есть отдельная озвучка {voice[0].path.name} — она станет спиной монтажа, "
                f"картинка целиком уйдёт под неё")

    total = sum(c.duration for c in videos)
    longest = max(videos, key=lambda c: c.duration)
    if len(videos) == 1:
        return "single_take", f"один клип {longest.path.name} — его надо сокращать, а не раскладывать"
    if total and longest.duration / total >= SINGLE_TAKE_SHARE and longest.speech_ratio > 0.3:
        return ("single_take",
                f"{longest.path.name} занимает {longest.duration / total:.0%} материала и говорит — "
                f"это дубль, остальное перебивки")
    talking = sum(1 for c in videos if c.kind == "talking_head")
    return ("narrated_clips",
            f"{len(videos)} клипов, из них {talking} на камеру, звук в самих клипах")


def _emit(plan: Plan, pool: "_BrollPool | None", *, index: int, t: float,
          audio_src: str, audio_start: float, dur: float,
          shown: Clip, s0: float, s1: float, kind: str,
          circle: CirclePlan | None = None, note: str = "") -> float:
    plan.shots.append(Shot(
        index=index, kind=kind, source=shown.path.name,
        src_start=round(s0, 3), src_end=round(s1, 3),
        timeline_start=round(t, 3), duration=round(dur, 3),
        audio_source=audio_src, audio_start=round(audio_start, 3),
        circle=circle, note=note))
    return t + dur


def _plan_voiceover(clips: list[Clip], style: Style, folder: Path, plan: Plan) -> Plan:
    """Separate narration file drives everything; all footage is b-roll.

    Circles are refused in this shape no matter what the style asks. The
    voiceover is not the audio of any clip, so a talking head composited into
    a circle would mouth words the viewer is not hearing — the one failure a
    viewer notices instantly.
    """
    voice = next(c for c in clips if c.kind == "voiceover")
    footage = [c for c in clips if c.kind in ("talking_head", "broll")]
    if not footage:
        plan.warnings.append("есть озвучка, но нет ни одного клипа под неё")
        return plan
    if style.circle.mode != "off":
        plan.warnings.append(
            "кружок выключен: озвучка отдельным файлом, губы в кадре не совпадут со звуком")

    pool = _BrollPool(footage)
    start, end = voice.speech_start, voice.speech_end
    t = 0.0
    index = 0
    target = style.pacing.shot_target_sec
    while t < end - start - 0.2:
        want = min(target, (end - start) - t)
        if index == 0:
            want = min(want, style.pacing.opening_shot_sec)
        got = pool.take(want)
        if got is None:
            plan.warnings.append("перебивки короче запрошенного плана — остаток озвучки без картинки")
            break
        shown, s0, s1 = got
        t = _emit(plan, pool, index=index, t=t, audio_src=voice.path.name,
                  audio_start=start + t, dur=s1 - s0, shown=shown, s0=s0, s1=s1,
                  kind="broll")
        index += 1
    plan.total_sec = round(t, 3)
    return plan


def _plan_single_take(clips: list[Clip], style: Style, folder: Path, plan: Plan) -> Plan:
    """One long take: keep the speech, drop the pauses, cut away now and then.

    The take's own pauses are what gets removed — that is the whole point of
    this shape, and it is why the inventory measures speech runs rather than
    just the first and last sound.
    """
    videos = [c for c in clips if c.kind in ("talking_head", "broll")]
    take = max(videos, key=lambda c: c.duration)
    others = [c for c in videos if c.path != take.path]
    pool = _BrollPool(others) if others else None
    pacing = style.pacing
    circle_mode = style.circle.mode

    if pool is None:
        plan.warnings.append(
            f"{take.path.name} — один дубль без перебивок: картинка останется одним планом "
            f"на всю реплику. Для чистки живого дубля (паузы, запинки, дубли, кадрирование "
            f"по лицу) в репозитории есть отдельный маршрут "
            f"pipelines/render_talking_head_dynamic_clean.py")

    t = 0.0
    index = 0
    for run_start, run_end in take.speech_runs:
        span = run_end - run_start
        if span < 0.4:
            continue
        n = _slice_count(span, pacing.shot_target_sec, pacing.shot_max_sec)
        if pool is None:
            n = 1
        pieces = [span / n] * n
        a_start = run_start
        for k in range(n):
            dur = pieces[k]
            if index == 0 and n > 1 and dur > pacing.opening_shot_sec:
                freed = dur - pacing.opening_shot_sec
                dur = pacing.opening_shot_sec
                for j in range(k + 1, n):
                    pieces[j] += freed / (n - k - 1)
            circle = None
            if k == 0 and take.kind == "talking_head" and circle_mode != "off":
                circle = CirclePlan(source=take.path.name, src_start=a_start,
                                    background=None, mode=circle_mode)
            if k == 0 or pool is None:
                shown, s0, s1, kind = take, a_start, a_start + dur, "talking"
            else:
                got = pool.take(dur)
                if got is None:
                    shown, s0, s1, kind = take, a_start, a_start + dur, "talking"
                else:
                    shown, s0, s1 = got
                    kind = "cutaway"
            if circle is not None and circle.mode == "pip":
                # A pip needs something behind it; on a single take the only
                # candidate is a cutaway, and if there is none the shot stays
                # a plain cut rather than a circle over itself.
                got = pool.take(dur) if pool else None
                if got is None:
                    circle = None
                else:
                    shown, s0, s1 = got
                    kind = "talking"
            a_this = a_start
            a_start += dur
            t = _emit(plan, pool, index=index, t=t, audio_src=take.path.name,
                      audio_start=a_this, dur=dur, shown=shown, s0=s0, s1=s1,
                      kind=kind, circle=circle,
                      note="пауза дубля вырезана" if k == 0 and index else "")
            index += 1
    plan.total_sec = round(t, 3)
    if not plan.shots:
        plan.warnings.append(f"в {take.path.name} не нашлось речи — нечего монтировать")
    return plan
