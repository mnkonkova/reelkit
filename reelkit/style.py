"""The style library: one YAML file per look, loaded and validated.

Why files and not `config.yaml`: a look is the thing that changes most often
and is most worth sharing between projects. Today it is smeared across
`config.yaml` (`subtitle_styles` + `look_profiles` + `rhythm_profiles` +
`transition`) and, in the ref-style route, hardcoded into Python `Literal`
types — so adding a look means editing code. Here a look is one file, and the
set of looks is `ls styles/`.

Every style inherits `styles/_base.yaml` and overrides what it cares about, so
a new style is a handful of lines rather than a full copy. Unknown keys are
rejected: a typo'd `circel:` would otherwise be silently ignored and the
render would come out in the default look with no error to explain it.

Caption and grade names (`captions.style`, `look.profile`) are deliberately
references INTO `config.yaml` rather than copies of it — the ASS subtitle
engine and the grade profiles already live there and are shared with the
older routes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
STYLES_DIR = ROOT / "styles"
BASE_NAME = "_base"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Canvas(Strict):
    width: int = 1080
    height: int = 1920
    fps: int = 30


class Look(Strict):
    profile: str = "punchy"          # name in config.yaml -> look_profiles
    saturation: float | None = None  # set to override the profile's value
    contrast: float | None = None
    gamma: float | None = None
    lut_file: str | None = None


class Pacing(Strict):
    shot_min_sec: float = 1.4
    shot_target_sec: float = 2.6
    shot_max_sec: float = 4.0
    # A reel that opens on a 4-second establishing shot loses people. The
    # first shot is held shorter than the rest unless a style says otherwise.
    opening_shot_sec: float = 1.6

    @model_validator(mode="after")
    def _opening_not_below_floor(self) -> "Pacing":
        # A style that lowers shot_min_sec without touching opening_shot_sec
        # ends up flagging its own opening shot as too short on every render.
        # Caught here rather than as a per-render warning nobody acts on.
        if self.opening_shot_sec < self.shot_min_sec:
            raise ValueError(
                f"pacing.opening_shot_sec ({self.opening_shot_sec}) ниже "
                f"shot_min_sec ({self.shot_min_sec}) — первый план будет "
                f"короче собственного минимума стиля")
        return self


class Transition(Strict):
    kind: Literal["cut", "dissolve", "dipblack", "dipwhite", "flash"] = "cut"
    duration_sec: float = 0.0


class Ring(Strict):
    width_px: int = 0
    color: str = "#FFFFFF"
    opacity: float = 1.0


class Shadow(Strict):
    blur_px: int = 0
    opacity: float = 0.0
    dy_px: int = 0


class Circle(Strict):
    """The head circle.

    `pip` is the small circle parked over b-roll while the person talks; `main`
    is the Telegram-style circle that IS the frame, with the b-roll demoted to
    a background. They are the same renderer with different numbers, which is
    why one mode switch covers both rather than two features.
    """
    mode: Literal["off", "pip", "main"] = "off"
    diameter: float = Field(0.34, gt=0.0, le=1.2)   # share of canvas WIDTH
    center: tuple[float, float] = (0.74, 0.70)      # share of canvas w/h
    zoom: float = Field(2.2, ge=1.0, le=6.0)        # square side / face width
    follow: bool = False       # False parks on the median face for the shot
    ring: Ring = Ring()
    shadow: Shadow = Shadow()
    # What sits behind a `main` circle. `blur` reuses the same b-roll, which
    # keeps the shot coherent without a second asset to find.
    background: Literal["broll", "blur", "color"] = "broll"
    background_color: str = "#101010"
    background_blur: int = 28
    # How much the crop may be blown up to fill the circle before it reads as
    # mush. Lives here, not in the renderer, because a soft-focus look
    # tolerates what a crisp one does not — and because on a 720p source a
    # big circle simply cannot stay under 2.0 (measured: a 0.82 circle off a
    # 720-wide close-up needs 3.5x). A style that wants the big circle
    # anyway says so here instead of every call site passing allow_soft.
    max_upscale: float = Field(2.2, ge=1.0, le=9.0)
    # What to do when the crop has to be blown up past max_upscale.
    # Default is to warn, not refuse: measured on 720p sources, the zoom that
    # composes best (head filling the circle) is also the one that upscales
    # most, so refusing would block the look that is actually right. `error`
    # is for batch runs where silently soft output is worse than a stop.
    on_soft: Literal["warn", "error"] = "warn"
    # When the circle is allowed on screen. `speaking` needs a talking face
    # (see face.FaceTrack.is_presenter); `always` forces it wherever a face
    # exists at all, which is what a style with no voiceover wants.
    when: Literal["speaking", "always", "hook_only"] = "speaking"

    @model_validator(mode="after")
    def _main_is_big(self) -> "Circle":
        # A `main` circle smaller than half the frame is not a main circle, it
        # is a mis-set pip — and it would silently render as a tiny bubble in
        # the middle of an empty frame.
        if self.mode == "main" and self.diameter < 0.5:
            raise ValueError(
                f"circle.mode=main needs diameter >= 0.5 (got {self.diameter}); "
                "use mode=pip for a small circle")
        return self


class Captions(Strict):
    style: str = "white_soft_shadow"   # name in config.yaml -> subtitle_styles
    enabled: bool = True
    margin_v: int | None = None        # override the named style's position
    size: int | None = None
    max_words_per_chunk: int | None = None
    # config.yaml's word-by-word styles set this true, which pins every event
    # to a single content word no matter what max_words_per_chunk says. A
    # style wanting the reference's 3-5 word phrases has to turn it off.
    forbid_two_content_words: bool | None = None
    # Legibility over a bright shot. The reference's captions sit on museum
    # walls and white paper; at outline 1 they disappear into the paper.
    outline: int | None = None
    shadow: int | None = None


class TitleLine(Strict):
    font: str = "Bebas Neue Cyrillic"
    size: int = 118
    color: str = "#FFFFFF"
    italic: bool = False
    y: int = 225


class TitleCard(Strict):
    enabled: bool = False
    duration_sec: float = 2.6
    line1: TitleLine = TitleLine()
    line2: TitleLine | None = None


class Audio(Strict):
    voice_lufs: float = -16.0
    music_lufs: float = -26.0
    limiter: float = 0.95
    music_file: str | None = None
    ducking: bool = True


class Style(Strict):
    id: str
    title: str = ""
    canvas: Canvas = Canvas()
    look: Look = Look()
    pacing: Pacing = Pacing()
    transition: Transition = Transition()
    circle: Circle = Circle()
    captions: Captions = Captions()
    title_card: TitleCard = TitleCard()
    audio: Audio = Audio()

    @model_validator(mode="after")
    def _circle_clear_of_captions(self) -> "Style":
        """A pip circle must not sit on the caption line.

        Found by looking at a render: the circle at y=0.70 landed exactly on
        the subtitles and the two fought for the same band. Geometry knows
        this before the render does, so it is checked here rather than left
        for someone to notice in the output.
        """
        if self.circle.mode != "pip" or not self.captions.enabled:
            return self
        margin_v = self.captions.margin_v
        if margin_v is None:
            return self          # position comes from config.yaml's safe box
        radius = self.circle.diameter * self.canvas.width / 2
        centre_y = self.circle.center[1] * self.canvas.height
        cap_top = margin_v - self.captions.size      # one line of text
        cap_bottom = margin_v + self.captions.size
        if centre_y + radius > cap_top and centre_y - radius < cap_bottom:
            raise ValueError(
                f"кружок (центр y={centre_y:.0f}, радиус {radius:.0f}) перекрывает "
                f"строку субтитров ({cap_top:.0f}..{cap_bottom:.0f}). "
                f"Подними circle.center выше {(cap_top - radius) / self.canvas.height:.2f} "
                f"или опусти ниже {(cap_bottom + radius) / self.canvas.height:.2f}")
        return self


def _deep_merge(base: dict, over: dict) -> dict:
    """`over` wins, but nested dicts merge instead of replacing wholesale.

    Without this, a style that sets only `circle.mode` would wipe out every
    other circle default and render a 0-diameter circle.
    """
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def available(styles_dir: Path | str = STYLES_DIR) -> list[str]:
    """Style ids on disk — this is the library.

    Files starting with an underscore are shared fragments (`_base` defaults,
    `_looks` grade profiles), not styles anyone can render with.
    """
    d = Path(styles_dir)
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))


def load_style(style_id: str, styles_dir: Path | str = STYLES_DIR) -> Style:
    d = Path(styles_dir)
    path = d / f"{style_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"нет стиля {style_id!r}; есть: {', '.join(available(d)) or '(пусто)'}")
    base_path = d / f"{BASE_NAME}.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    data = _deep_merge(base or {}, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    data["id"] = style_id          # the filename is the id, always
    return Style.model_validate(data)
