"""Where the head is, over time, in one clip.

The engine needs this for two different jobs: putting a talking head inside a
circle, and framing a vertical crop on the person. Both want the same thing —
a stable face box per moment — so it lives here once.

Lineage: the detector setup and the thresholds below are the ones already
proven in `pipelines/speaker_crop.py` on real footage (YuNet ONNX, analysis at
960px, score 0.75, faces under 2.5% of frame width are glare). The job here is
narrower than speaker_crop's: that module answers "which of several people is
talking"; this one answers "where is the one head in this clip", which is the
single-presenter case the reel engine actually sees.

A track is deliberately SPARSE (5 samples/sec) and SMOOTHED. Detection jitters
by a few pixels every frame, and a circle that follows raw detections wobbles
in a way that reads as cheap. Consumers get a settled path, or a single median
box when they only need to park a crop for the length of a shot.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/face_detection_yunet.onnx"

ANALYSE_W = 960      # detection resolution; 4K decode is slow and buys nothing
SCORE_MIN = 0.75     # below this YuNet starts "seeing" faces in reflections
MIN_FACE = 0.025     # share of frame width; smaller is not a person
TRACK_FPS = 5.0      # samples per second of track
SMOOTH_ALPHA = 0.35  # EMA factor; lower = calmer, slower to follow
CLUSTER_X = 0.08     # share of width a head can travel between samples
MAX_GAP_FRAMES = 3   # samples a track may go unseen before it is closed
EYE_FLOOR = 1.0      # keeps a motionless face from producing a huge ratio


class NoFaceError(RuntimeError):
    """No usable face in the requested span."""


class FaceModelMissing(NoFaceError):
    """The detector itself is absent — a different thing from "nobody here".

    Kept as its own type because the two are indistinguishable to a caller
    that only catches NoFaceError, and the consequences differ enormously: a
    clip with no face is ordinary b-roll, whereas a missing model silently
    turns EVERY clip into b-roll and quietly disables circles.
    """


@dataclass(frozen=True)
class FaceBox:
    """One sample of the head, in shares of frame width/height (0..1).

    Shares, not pixels, so a track survives the clip being scaled or re-probed
    at a different analysis width.
    """
    t: float       # seconds from the start of the span
    cx: float      # centre of the face box
    cy: float
    size: float    # face WIDTH as a share of frame width
    mouth_y: float = 0.0   # mouth centre, share of height (None-free: falls back to cy)

    def square(self, zoom: float) -> tuple[float, float, float]:
        """(centre_x, centre_y, side) of a square framing the head.

        `zoom` is the side as a multiple of face width: 1.0 is the face box
        itself (too tight — crops ears and chin), ~2.2 leaves head and
        shoulders, which is what a talking-head circle wants. The square is
        biased upward because a head sits above its own centre once you
        include shoulders.
        """
        side = self.size * zoom
        return self.cx, self.cy - side * 0.06, side


@dataclass(frozen=True)
class FaceTrack:
    boxes: list[FaceBox]
    src_w: int
    src_h: int
    talking_score: float = 0.0   # diagnostic only — see circleable() on why
    presence: float = 1.0        # share of samples where the face was actually seen

    def circleable(self, min_size: float = 0.10, min_presence: float = 0.50) -> bool:
        """Is there a head here worth putting in a circle?

        Note what this does NOT claim: that the head belongs to a person
        rather than to a photograph. On museum footage that question has no
        cheap answer — the walls are covered in portraits, and `talking_score`
        (mouth motion over eye motion) was measured at 0.82-1.21 for the
        presenter and 0.90-1.21 for framed photographs, i.e. it does not
        separate them at all in absolute terms. It only works the way
        `pipelines/speaker_crop.py` uses it: comparing several faces inside
        ONE frame, where the louder mouth wins.

        So the gate is the one the circle actually needs anyway. A face below
        ~10% of frame width occupies under a fifth of a pip circle and reads
        as a smudge, and a face seen in under half the samples will pop in and
        out. Both rejections are right for the feature even when the subject
        is a real person standing too far away — and they happen to reject
        wall portraits too, because those are either small or fleeting.

        A clip that passes this and should not have (a big, steady portrait
        filling the frame) is caught by the operator looking at one still, not
        by a better threshold. `plan.py` prints what it chose for that reason.
        """
        return self.median.size >= min_size and self.presence >= min_presence

    @property
    def median(self) -> FaceBox:
        """One box standing for the whole span.

        This is what a short shot should use: parking the crop on the median
        beats following the head, because over two or three seconds the head
        barely leaves a generous square anyway, and a static frame has no
        wobble to give itself away.
        """
        return FaceBox(
            t=0.0,
            cx=float(np.median([b.cx for b in self.boxes])),
            cy=float(np.median([b.cy for b in self.boxes])),
            size=float(np.median([b.size for b in self.boxes])),
            mouth_y=float(np.median([b.mouth_y for b in self.boxes])),
        )

    def at(self, t: float) -> FaceBox:
        """The sample nearest `t` (clamped to the ends of the track)."""
        return min(self.boxes, key=lambda b: abs(b.t - t))


def _frames(src: Path, start: float, dur: float, fps: float) -> np.ndarray:
    """Decode the span once, at analysis resolution, as BGR24."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(src)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split("x"))
    aw = ANALYSE_W if w > ANALYSE_W else w // 2 * 2
    ah = int(aw * h / w) // 2 * 2
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}",
         "-t", f"{dur:.3f}", "-i", str(src),
         "-vf", f"fps={fps},scale={aw}:{ah}", "-pix_fmt", "bgr24",
         "-f", "rawvideo", "-"], capture_output=True).stdout
    n = len(raw) // (aw * ah * 3)
    if n == 0:
        return np.zeros((0, ah, aw, 3), dtype=np.uint8), w, h
    return np.frombuffer(raw[: n * aw * ah * 3], dtype=np.uint8).reshape(n, ah, aw, 3), w, h


def _detect(frames: np.ndarray, cv2) -> list[list[tuple[float, ...]]]:
    """Per frame: [(cx, cy, face_w, mouth_x, mouth_y, eye_x, eye_y), ...] in px.

    YuNet returns five landmarks per face. Mouth and eyes are kept because
    they are what separates a person from a portrait on the wall: speech moves
    the mouth independently of the rest of the head (see `_talking_score`).
    """
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), score_threshold=SCORE_MIN)
    out = []
    for f in frames:
        det.setInputSize((w, h))
        _, faces = det.detect(np.ascontiguousarray(f))
        got = []
        if faces is not None:
            for d in faces:
                x, y, fw, fh = (float(v) for v in d[:4])
                if fw < w * MIN_FACE:
                    continue
                mx = float((d[8] + d[10]) / 2) if len(d) > 10 else x + fw / 2
                my = float((d[9] + d[11]) / 2) if len(d) > 11 else y + fh * 0.75
                ex = float((d[4] + d[6]) / 2) if len(d) > 7 else x + fw / 2
                ey = float((d[5] + d[7]) / 2) if len(d) > 7 else y + fh * 0.35
                got.append((x + fw / 2, y + fh / 2, fw, mx, my, ex, ey))
        out.append(got)
    return out


def _patch_motion(gray: np.ndarray, cy: float, cx: float, r: int) -> float:
    """Mean frame-to-frame change in a square patch — how much it moves."""
    h, w = gray.shape[1:]
    patch = gray[:, int(max(0, cy - r)):int(min(h, cy + r)),
                 int(max(0, cx - r)):int(min(w, cx + r))]
    if patch.shape[0] < 2 or patch.size == 0:
        return 0.0
    return float(np.abs(np.diff(patch, axis=0)).mean())


def _talking_score(gray: np.ndarray, boxes: list[tuple[float, ...]]) -> float:
    """How much the mouth moves relative to the rest of the head.

    A museum wall is covered in portraits and YuNet reports every one of them
    as a face, so "a face was found" cannot mean "a person is here". A talking
    person moves their mouth while the eyes stay put; a photograph (and a
    listener who merely nods) moves all of it together, or not at all. The
    ratio is near 1.0 for both of those and climbs well above it for speech.
    """
    if not boxes:
        return 0.0
    cx = float(np.median([b[0] for b in boxes]))
    size = float(np.median([b[2] for b in boxes]))
    my = float(np.median([b[4] for b in boxes]))
    ex = float(np.median([b[5] for b in boxes]))
    ey = float(np.median([b[6] for b in boxes]))
    r = max(4, int(size * 0.35))
    mouth = _patch_motion(gray, my, cx, r)
    eyes = _patch_motion(gray, ey, ex, r)
    return mouth / (eyes + EYE_FLOOR)


def _smooth(values: list[float]) -> list[float]:
    """Exponential moving average, run forwards then backwards.

    One pass alone lags: the circle trails the head and settles late. Running
    it again over the reversed series cancels that lag, so the path is calm
    AND lands on the head rather than behind it.
    """
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + SMOOTH_ALPHA * (v - out[-1]))
    back = [out[-1]]
    for v in reversed(out[:-1]):
        back.append(back[-1] + SMOOTH_ALPHA * (v - back[-1]))
    return list(reversed(back))


def track_face(src: Path, start: float = 0.0, dur: float | None = None,
               fps: float = TRACK_FPS) -> FaceTrack:
    """Follow the main head through a span. Raises NoFaceError if there is none.

    Callers that can live without a face (b-roll with nobody in it) should
    catch NoFaceError rather than pre-checking: whether a clip "has a person"
    is exactly what this answers.
    """
    if not MODEL.exists():
        raise FaceModelMissing(
            f"нет модели лиц: {MODEL}. Без неё движок не отличит говорящего "
            f"от перебивки и не поставит кружок. Как положить — см. README.")
    try:
        import cv2
    except ImportError as exc:
        raise FaceModelMissing("не установлен opencv-python") from exc

    if dur is None:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src)], capture_output=True, text=True,
            check=True).stdout.strip()) - start

    frames, src_w, src_h = _frames(src, start, dur, fps)
    if len(frames) == 0:
        raise NoFaceError("no frames decoded")
    h, w = frames[0].shape[:2]
    per_frame = _detect(frames, cv2)
    if not any(per_frame):
        raise NoFaceError("no faces detected")

    # Follow each face FRAME TO FRAME instead of binning detections by where
    # they sit in the frame.
    #
    # Binning by x was the first rule and it broke on a presenter who walks:
    # in IMG_5813 she crosses the frame and her detections fell into four
    # separate "people" at cx 0.07 / 0.22 / 0.35 / 0.43. The engine picked one
    # bin, parked a static crop at its median, and the circle spent most of
    # the shot holding a wall. Linking by proximity in TIME keeps a moving
    # subject as one track, which also makes `presence` mean what it says.
    jump = w * CLUSTER_X
    tracks: list[dict] = []
    for i, frame in enumerate(per_frame):
        taken: set[int] = set()
        for tr in tracks:
            if i - tr["last_i"] > MAX_GAP_FRAMES:
                continue
            best_j, best_d = None, jump
            for j, det_box in enumerate(frame):
                if j in taken:
                    continue
                dist = abs(det_box[0] - tr["last"][0]) + abs(det_box[1] - tr["last"][1])
                if dist < best_d:
                    best_j, best_d = j, dist
            if best_j is not None:
                taken.add(best_j)
                tr["dets"].append((i, frame[best_j]))
                tr["last"] = (frame[best_j][0], frame[best_j][1])
                tr["last_i"] = i
        for j, det_box in enumerate(frame):
            if j not in taken:
                tracks.append({"dets": [(i, det_box)], "last": (det_box[0], det_box[1]),
                               "last_i": i})

    gray_all = frames.astype(np.int16).mean(axis=3)
    # Rank by how much face a track actually delivers — samples times face
    # width — not by sample count. Length alone promotes a small background
    # detection that sits still for the whole clip over the presenter, whose
    # track breaks whenever she turns away.
    def mass(t: dict) -> float:
        return sum(d[2] for _, d in t["dets"])

    best_track = max(tracks, key=mass)
    seen = dict(best_track["dets"])

    # One detection per sample. Gaps (a turned head, a blur) are held from the
    # previous sample so the track has no holes to special-case downstream.
    raw: list[tuple[float, ...] | None] = [seen.get(i) for i in range(len(per_frame))]
    if not any(r is not None for r in raw):
        raise NoFaceError("main face never resolved")
    presence = sum(r is not None for r in raw) / len(raw)
    first = next(r for r in raw if r is not None)
    filled = []
    last = first
    for r in raw:
        last = r or last
        filled.append(last)

    talking = _talking_score(gray_all, [d for d in raw if d is not None])

    xs = _smooth([d[0] / w for d in filled])
    ys = _smooth([d[1] / h for d in filled])
    ss = _smooth([d[2] / w for d in filled])
    mys = _smooth([d[4] / h for d in filled])
    boxes = [FaceBox(t=i / fps, cx=x, cy=y, size=sz, mouth_y=my)
             for i, (x, y, sz, my) in enumerate(zip(xs, ys, ss, mys))]
    return FaceTrack(boxes=boxes, src_w=src_w, src_h=src_h,
                     talking_score=talking, presence=presence)
