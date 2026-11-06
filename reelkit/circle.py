"""The head circle: a round cut-out of the speaker, composited over a shot.

Two modes, one renderer. `pip` parks a small circle over b-roll while the
person talks; `main` makes the circle the frame itself, with the b-roll
demoted to a blurred or plain backdrop (the Telegram video-message look). They
differ only in diameter and placement, so they share every line below — which
is the point of making the circle a style parameter rather than two features.

Placement is static per shot by default. A circle that chases the head frame
by frame wobbles, and over the two to four seconds a reel shot actually lasts
the head does not leave a generous square anyway. `follow: true` turns on
tracking for the long takes where it does.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .face import FaceTrack
from .style import Canvas, Circle, Style

FEATHER_PX = 2.0     # anti-aliased edge; 1px shows stair-steps on the arc


class CircleTooSoft(RuntimeError):
    """The crop would have to be upscaled past MAX_UPSCALE to fill the circle."""


def _hex_to_ffmpeg(color: str) -> str:
    return "0x" + color.lstrip("#")


@dataclass(frozen=True)
class CirclePlacement:
    """Everything the filter graph needs, resolved to pixels."""
    src_x: int      # crop window in SOURCE pixels
    src_y: int
    side: int
    diameter: int   # on canvas
    x: int          # top-left of the circle on canvas
    y: int
    upscale: float


def place(circle: Circle, canvas: Canvas, track: FaceTrack) -> CirclePlacement:
    """Resolve the style's shares into a concrete crop and canvas position.

    The crop is clamped to the source frame: a head near an edge would
    otherwise produce a crop window hanging off the picture, which ffmpeg
    accepts and fills with garbage rather than refusing.
    """
    box = track.median
    cx, cy, side_share = box.square(circle.zoom)
    side = int(round(side_share * track.src_w))
    side = max(16, min(side, track.src_w, track.src_h))
    src_x = int(round(cx * track.src_w - side / 2))
    src_y = int(round(cy * track.src_h - side / 2))
    src_x = max(0, min(src_x, track.src_w - side))
    src_y = max(0, min(src_y, track.src_h - side))

    diameter = int(round(circle.diameter * canvas.width)) // 2 * 2
    return CirclePlacement(
        src_x=src_x, src_y=src_y, side=side, diameter=diameter,
        x=int(round(circle.center[0] * canvas.width - diameter / 2)),
        y=int(round(circle.center[1] * canvas.height - diameter / 2)),
        upscale=diameter / side,
    )


def _disc_alpha(diameter: int, feather: float = FEATHER_PX) -> str:
    """geq alpha expression turning a square into an anti-aliased disc."""
    r = diameter / 2
    return (f"clip(({r}-hypot(X-{r},Y-{r}))*{255.0 / feather:.3f},0,255)")


def circle_chain(place_: CirclePlacement, circle: Circle, *,
                 src_label: str, out_label: str) -> str:
    """Filter chain turning a source stream into an RGBA disc, ready to overlay.

    Built as one string so the caller can drop it into any filter_complex —
    the renderer composites several of these per shot.
    """
    d = place_.diameter
    ring = max(0, circle.ring.width_px)
    inner = d - 2 * ring
    parts = [
        f"[{src_label}]crop={place_.side}:{place_.side}:{place_.src_x}:{place_.src_y},"
        f"scale={inner}:{inner},format=rgba,"
        f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{_disc_alpha(inner)}'[cface]"
    ]
    if ring:
        # The ring is a filled disc of the ring colour with the face laid on
        # top, not a stroked outline: ffmpeg has no circle stroke, and a
        # filled disc one ring-width bigger is the same picture.
        parts.append(
            f"color=c={_hex_to_ffmpeg(circle.ring.color)}@{circle.ring.opacity}:"
            f"s={d}x{d}:d=1,format=rgba,"
            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{_disc_alpha(d)}'[cring]")
        parts.append(f"[cring][cface]overlay={ring}:{ring}:format=auto[{out_label}]")
    else:
        parts.append(f"[cface]null[{out_label}]")
    return ";".join(parts)


def shadow_chain(place_: CirclePlacement, circle: Circle, *, out_label: str) -> str | None:
    """A soft dark disc to sit under the circle, or None when the style has none."""
    sh = circle.shadow
    if sh.opacity <= 0 or sh.blur_px <= 0:
        return None
    d = place_.diameter
    pad = sh.blur_px * 3          # room for the blur to fall off inside the canvas
    box = d + pad * 2
    return (
        f"color=c=black@0:s={box}x{box}:d=1,format=rgba,"
        f"geq=r='0':g='0':b='0':a='{sh.opacity}*{_disc_alpha(d)}*"
        f"between(X,{pad},{pad + d})*between(Y,{pad},{pad + d})',"
        f"gblur=sigma={sh.blur_px / 2:.1f}[{out_label}]"
    )


def check_sharpness(pl: CirclePlacement, circle: Circle, track: FaceTrack) -> str | None:
    """Message about an over-stretched crop, or None when it is within budget.

    Returned rather than printed so the caller decides: a preview prints it,
    a batch with `on_soft: error` raises it, a plan records it next to the
    shot it belongs to.
    """
    if pl.upscale <= circle.max_upscale:
        return None
    ok_d = circle.diameter * circle.max_upscale / pl.upscale
    return (f"кружок {pl.diameter}px собирается из куска {pl.side}px — увеличение "
            f"{pl.upscale:.2f}x при бюджете {circle.max_upscale:.2f}x "
            f"(источник {track.src_w}x{track.src_h}, лицо {track.median.size:.3f} кадра). "
            f"Либо diameter <= {ok_d:.2f}, либо zoom "
            f"{circle.zoom * pl.upscale / circle.max_upscale:.1f}+ "
            f"(голова в круге станет мельче), либо снимать крупнее.")


def render_shot(src: Path, out: Path, *, style: Style, track: FaceTrack,
                start: float, dur: float, background: Path | None = None,
                allow_soft: bool = False) -> CirclePlacement:
    """Render one shot with the head circle burned in.

    `background` is the b-roll the circle sits on. In `pip` mode it is
    required — a pip over nothing is just a badly cropped talking head. In
    `main` mode it is optional and the style decides what fills the frame.
    """
    circle = style.circle
    if circle.mode == "off":
        raise ValueError("render_shot called with circle.mode=off")
    if circle.mode == "pip" and background is None:
        raise ValueError("pip-кружок нужно класть на b-roll: передай background")

    canvas = style.canvas
    pl = place(circle, canvas, track)
    soft = check_sharpness(pl, circle, track)
    if soft and not allow_soft:
        if circle.on_soft == "error":
            raise CircleTooSoft(soft)
        print(f"  ! {soft}")

    W, H, fps = canvas.width, canvas.height, canvas.fps
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps}"

    inputs: list[str] = []
    chains: list[str] = []

    # --- what fills the frame behind the circle -------------------------
    if circle.mode == "main" and circle.background == "color":
        inputs += ["-f", "lavfi", "-t", f"{dur:.3f}",
                   "-i", f"color=c={_hex_to_ffmpeg(circle.background_color)}:s={W}x{H}:r={fps}"]
        chains.append(f"[0:v]null[bg]")
        face_idx = 1
        inputs += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
    elif circle.mode == "main" and circle.background == "blur" and background is None:
        # Reuse the talking head itself, blurred: no second asset needed.
        inputs += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
        chains.append(f"[0:v]{fill},gblur=sigma={circle.background_blur}[bg]")
        face_idx = 0
    else:
        if background is None:
            raise ValueError("нужен background для этого режима круга")
        inputs += ["-t", f"{dur:.3f}", "-i", str(background)]
        blur = f",gblur=sigma={circle.background_blur}" if circle.background == "blur" else ""
        chains.append(f"[0:v]{fill}{blur}[bg]")
        inputs += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
        face_idx = 1

    chains.append(circle_chain(pl, circle, src_label=f"{face_idx}:v", out_label="disc"))

    shadow = shadow_chain(pl, circle, out_label="shadow")
    if shadow:
        pad = circle.shadow.blur_px * 3
        chains.append(shadow)
        chains.append(f"[bg][shadow]overlay={pl.x - pad}:{pl.y - pad + circle.shadow.dy_px}[bgs]")
        base = "bgs"
    else:
        base = "bg"

    chains.append(f"[{base}][disc]overlay={pl.x}:{pl.y}:format=auto,format=yuv420p[v]")

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(chains),
         "-map", "[v]", "-t", f"{dur:.3f}", "-r", str(fps),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return pl
