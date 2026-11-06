"""reelkit CLI — the style library and a circle preview.

    python -m src.reelkit styles
    python -m src.reelkit preview CLIP.MP4 --style museum_pip --bg BROLL.MP4

`preview` exists because the circle is the one thing you cannot judge from
numbers: whether the head sits right inside it is a look, not a measurement.
It renders a few seconds so the style can be tuned against real footage
before a whole reel is built on it.
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .outputs import versioned_dir
from .circle import place, render_shot
from .face import NoFaceError, track_face
from .inventory import face_detection_warning, scan, summary
from .plan import build_plan
from .render import render
from .style import available, load_style

app = typer.Typer(add_completion=False, help="Сборка роликов по исходникам: стили и кружки.")
console = Console()


@app.command()
def styles() -> None:
    """Что есть в библиотеке стилей."""
    ids = available()
    if not ids:
        console.print("[yellow]в styles/ пусто[/yellow]")
        raise typer.Exit(1)
    for sid in ids:
        s = load_style(sid)
        c = s.circle
        circle = "—" if c.mode == "off" else f"{c.mode} d={c.diameter:.2f} zoom={c.zoom:.1f}"
        console.print(f"[bold]{sid:16}[/bold] {s.title}")
        console.print(f"{'':16} грейд={s.look.profile}  план≈{s.pacing.shot_target_sec}с  "
                      f"переход={s.transition.kind}  круг={circle}")


@app.command()
def inspect(
    folder: Path = typer.Argument(..., help="Папка с исходниками"),
    no_faces: bool = typer.Option(False, "--no-faces", help="Быстро, без трекинга лиц"),
) -> None:
    """Что движок видит в папке и почему так решил."""
    clips = scan(folder, probe_faces=not no_faces)
    console.print(summary(clips))
    warn = face_detection_warning(clips)
    if warn:
        console.print(f"[yellow]![/yellow] {warn}")
    for c in clips:
        console.print(f"  [bold]{c.path.name:16}[/bold] {c.kind:12} {c.duration:5.1f}с  "
                      f"речь {c.speech_ratio:4.0%}  [dim]{c.reason}[/dim]")


@app.command()
def build(
    folder: Path = typer.Argument(..., help="Папка с исходниками"),
    style: str = typer.Option(..., "--style", "-s", help="id стиля из styles/"),
    slug: str = typer.Option(None, "--slug", help="Имя ролика (default: имя папки)"),
    version: str = typer.Option(None, "--version", help="Перезаписать конкретную vN"),
    plan_only: bool = typer.Option(False, "--plan-only", help="Только план, без рендера"),
    keep_parts: bool = typer.Option(False, "--keep-parts", help="Не удалять куски планов"),
) -> None:
    """Собрать ролик из папки исходников по стилю."""
    st = load_style(style)
    clips = scan(folder)
    console.print(summary(clips))
    warn = face_detection_warning(clips)
    if warn:
        console.print(f"[yellow]![/yellow] {warn}")
    plan = build_plan(clips, st, folder)
    if not plan.shots:
        for w in plan.warnings:
            console.print(f"[red]{w}[/red]")
        raise typer.Exit(1)

    out_dir = versioned_dir(Path("output") / (slug or folder.name), version)
    plan.to_json(out_dir / "plan.json")
    console.print(plan.describe())
    for w in plan.warnings:
        console.print(f"[yellow]! {w}[/yellow]")
    if plan_only:
        console.print(f"[green]план:[/green] {out_dir / 'plan.json'}")
        return
    final = render(plan, st, out_dir, keep_parts=keep_parts)
    console.print(f"[green]готово:[/green] {final}")


@app.command()
def preview(
    clip: Path = typer.Argument(..., help="Клип с говорящей головой"),
    style: str = typer.Option(..., "--style", "-s", help="id стиля из styles/"),
    bg: Path = typer.Option(None, "--bg", help="B-roll под кружок (обязателен для pip)"),
    start: float = typer.Option(0.0, help="С какой секунды клипа"),
    dur: float = typer.Option(3.0, help="Сколько секунд рендерить"),
    out: Path = typer.Option(None, "--out", "-o", help="Куда класть (default: output/preview/)"),
) -> None:
    """Собрать несколько секунд с кружком, чтобы посмотреть на стиль."""
    st = load_style(style)
    if st.circle.mode == "off":
        console.print(f"[yellow]у стиля {style} кружок выключен (circle.mode: off)[/yellow]")
        raise typer.Exit(1)

    try:
        track = track_face(clip, start, dur)
    except NoFaceError as exc:
        console.print(f"[red]лица в клипе нет:[/red] {exc}")
        raise typer.Exit(1)

    if not track.circleable():
        m = track.median
        console.print(
            f"[yellow]лицо есть, но для кружка не годится[/yellow] — "
            f"ширина {m.size:.3f} кадра, видно в {track.presence:.0%} выборок.\n"
            f"Кружку нужно хотя бы 0.10 и 50%: сними крупнее или возьми другой дубль.")
        raise typer.Exit(1)

    pl = place(st.circle, st.canvas, track)
    out = out or Path("output/preview") / f"{clip.stem}_{style}.mp4"
    render_shot(clip, out, style=st, track=track, start=start, dur=dur, background=bg)
    console.print(f"[green]{out}[/green]  круг {pl.diameter}px из куска {pl.side}px "
                  f"({pl.upscale:.2f}x)")


if __name__ == "__main__":
    app()
