"""Burned-in captions for a reel, from the assembled narration.

Transcription runs on the FINAL mixed voice track, not on the source clips.
That costs one more pass but it is the only way the word timings line up with
what the viewer hears: the plan trims and reorders spans, so timings measured
per source clip would need re-mapping through the edit, and every rounding
error there lands as a caption out of sync.

The ASS generation itself is `src/subtitles.py`, shared with the older routes
and not reimplemented here. This module's job is the three things around it:
getting a transcript, cleaning it, and resolving the style.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.schemas import Config, SubtitleStyle, Transcript, Word
from src.subtitles import apply_word_corrections, build_corrections_map, transcript_to_ass
from src.transcribe import transcribe

from .style import Style


def load_caption_config():
    """The parent project's config.yaml, or None when it is not installed."""
    try:
        from src.schemas import load_config
    except ImportError:
        return None
    try:
        return load_config()
    except Exception:
        return None

# Whisper fills trailing near-silence with the credits it saw a million times
# in its training data. Measured on this material: the end of IMG_5821 came
# back as "Субтитры сделал DimaTorzok" — words that were never spoken. Burning
# that into the frame is worse than losing a real trailing word, so the tail
# is checked against these and cut.
HALLUCINATION_TAIL = re.compile(
    r"(субтитр\w*\s+(сделал|создал|подготов|редактор)|"
    r"редактор\s+субтитр|"
    r"продолжение\s+следует|"
    r"спасибо\s+за\s+(просмотр|внимание)|"
    r"подписывайтесь\s+на\s+канал|"
    r"dimatorzok|корректор|amara\.org)",
    re.IGNORECASE)
TAIL_WORDS = 12     # how far back from the end to look for the artifact


def strip_hallucinated_tail(transcript: Transcript) -> tuple[Transcript, str | None]:
    """Drop invented closing credits. Returns (transcript, what was cut)."""
    words = list(transcript.words)
    if not words:
        return transcript, None
    window = words[-TAIL_WORDS:]
    text = " ".join(w.word for w in window)
    match = HALLUCINATION_TAIL.search(text)
    if not match:
        return transcript, None
    # Cut from the first word of the matched phrase onward: the artifact is
    # always the tail, never something embedded mid-sentence.
    start_char = match.start()
    consumed = 0
    cut_at = len(words)
    for i, w in enumerate(window):
        if consumed >= start_char:
            cut_at = len(words) - len(window) + i
            break
        consumed += len(w.word) + 1
    kept = words[:cut_at]
    if not kept:
        return transcript, None
    removed = " ".join(w.word for w in words[cut_at:])
    return Transcript(words=kept,
                      full_text=" ".join(w.word for w in kept),
                      duration=transcript.duration), removed


def correct_words(transcript: Transcript, cfg: Config) -> Transcript:
    """Apply config.yaml -> subtitle_corrections to every word."""
    mapping = build_corrections_map(cfg.subtitle_corrections)
    if not mapping:
        return transcript
    words = [Word(word=apply_word_corrections(w.word, mapping), start=w.start, end=w.end)
             for w in transcript.words]
    return Transcript(words=words, full_text=" ".join(w.word for w in words),
                      duration=transcript.duration)


def resolve_subtitle_style(style: Style, cfg: Config) -> SubtitleStyle:
    """The named config.yaml style with this reel style's overrides on top.

    Captions stay in config.yaml because the ASS engine and the older routes
    share them; a reel style names one and adjusts the handful of numbers that
    actually distinguish looks (where the line sits, how big, how many words).
    """
    name = style.captions.style
    if name not in cfg.subtitle_styles:
        raise KeyError(
            f"субтитры {name!r} не найдены в config.yaml → subtitle_styles "
            f"(есть: {', '.join(sorted(cfg.subtitle_styles))})")
    base = cfg.subtitle_styles[name]
    patch = {}
    if style.captions.margin_v is not None:
        patch["margin_v"] = style.captions.margin_v
    if style.captions.size is not None:
        patch["size"] = style.captions.size
    if style.captions.max_words_per_chunk is not None:
        patch["max_words_per_chunk"] = style.captions.max_words_per_chunk
    if style.captions.forbid_two_content_words is not None:
        patch["forbid_two_content_words"] = style.captions.forbid_two_content_words
    if style.captions.outline is not None:
        patch["outline"] = style.captions.outline
    if style.captions.shadow is not None:
        patch["shadow"] = style.captions.shadow
    return base.model_copy(update=patch) if patch else base


def caption_safe_box(style: Style, cfg: Config):
    """The shared safe box, moved to where this style puts its captions.

    `subtitles._position_tag` derives the caption's y from the safe box, not
    from the style's margin_v, so overriding margin_v alone does nothing — the
    museum style asked for y=1260 and rendered at 1460. Restating the request
    as a bottom padding is what the ASS writer actually reads, and it keeps
    the positioning logic in one place instead of forking it.
    """
    box = cfg.subtitle_safe_box
    if style.captions.margin_v is None:
        return box
    return box.model_copy(
        update={"bottom_padding_px": max(0, box.play_res_y - style.captions.margin_v)})


def make_captions(voice_path: Path, out_dir: Path, style: Style,
                  cfg: Config) -> tuple[Path | None, list[str]]:
    """Transcribe the reel's voice track and write subtitles.ass beside it."""
    notes: list[str] = []
    if not style.captions.enabled:
        return None, notes

    transcript = transcribe(voice_path, out_dir / "transcript.json",
                            language="ru", model_size=cfg.whisper.transcribe_model)
    transcript, removed = strip_hallucinated_tail(transcript)
    if removed:
        notes.append(f"с хвоста срезана галлюцинация Whisper: «{removed}»")
    transcript = correct_words(transcript, cfg)

    sub_style = resolve_subtitle_style(style, cfg)
    ass_path = out_dir / "subtitles.ass"
    warnings: list[str] = []
    transcript_to_ass(transcript, sub_style, ass_path,
                      safe_box=caption_safe_box(style, cfg), cfg=cfg,
                      layout_warnings=warnings, section="body")
    notes.extend(f"субтитры: {w}" for w in warnings)
    return ass_path, notes
