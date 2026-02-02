from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class TraceError(RuntimeError):
    pass


def _escape(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _pitch_palette() -> dict[str, str]:
    # Keep in sync with `baseball.viz` (duplicated on purpose to keep modules independent).
    return {
        "FF": "#0ea5e9",  # 4-seam
        "SI": "#06b6d4",  # sinker
        "FC": "#22c55e",  # cutter
        "CH": "#10b981",  # changeup
        "FS": "#16a34a",  # splitter
        "SL": "#a855f7",  # slider
        "CU": "#f97316",  # curve
        "KC": "#fb7185",  # knuckle curve
        "ST": "#eab308",  # sweeper / other breaking
        "SV": "#f43f5e",  # (rare) slurve
        "FO": "#64748b",  # forkball / other
        "KN": "#94a3b8",  # knuckleball
        "OOV": "#cbd5e1",
    }


def iter_trace_events_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise TraceError(f"Missing trace JSONL: {p}")
    if p.is_dir():
        raise TraceError(f"Trace path is a directory (expected file): {p}")
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise TraceError(f"Invalid JSON at {p}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise TraceError(f"Expected object per line at {p}:{lineno}, got: {type(obj).__name__}")
            yield obj


def _stand_token(stand_id: int) -> str:
    return {1: "R", 2: "L", 3: "S"}.get(int(stand_id), "?")


def _throws_token(p_throws_id: int) -> str:
    return {1: "R", 2: "L"}.get(int(p_throws_id), "?")


def _topbot_token(inning_topbot_id: int) -> str:
    return {0: "Top", 1: "Bot"}.get(int(inning_topbot_id), "?")


def _bases_str(state: dict[str, Any]) -> str:
    o1 = int(state.get("on_1b_occ", 0) or 0)
    o2 = int(state.get("on_2b_occ", 0) or 0)
    o3 = int(state.get("on_3b_occ", 0) or 0)
    return f"{'1' if o1 else '-'}{'2' if o2 else '-'}{'3' if o3 else '-'}"


def _select_game(events: list[dict[str, Any]], game_pk: int | None) -> list[dict[str, Any]]:
    if game_pk is None:
        g = sorted({int(e.get("game_pk", -1)) for e in events if "game_pk" in e})
        if not g:
            raise TraceError("Trace JSONL did not include any `game_pk` fields.")
        if len(g) > 1:
            raise TraceError(f"Trace contains multiple games {g}; pass --game-pk to select one.")
        game_pk = g[0]
    out = [e for e in events if int(e.get("game_pk", -1)) == int(game_pk)]
    if not out:
        raise TraceError(f"No events found for game_pk={game_pk}.")
    return out


def _select_at_bat(events: list[dict[str, Any]], at_bat_number: int | None) -> tuple[int, list[dict[str, Any]]]:
    ab_all = sorted({int(e.get("at_bat_number", -1)) for e in events if "at_bat_number" in e})
    if not ab_all:
        raise TraceError("Trace events missing `at_bat_number`.")
    if at_bat_number is None:
        at_bat_number = ab_all[0]
    ab = int(at_bat_number)
    sub = [e for e in events if int(e.get("at_bat_number", -1)) == ab]
    if not sub:
        raise TraceError(f"No events found for at_bat_number={ab}. Available: {ab_all[:20]}{'…' if len(ab_all)>20 else ''}")
    sub.sort(key=lambda r: int(r.get("pitch_number", 0)))
    return ab, sub


def render_at_bat_strip_svg(
    *,
    events: list[dict[str, Any]],
    title: str,
    max_pitches: int = 12,
    x_range: tuple[float, float] = (-2.0, 2.0),
    z_range: tuple[float, float] = (0.5, 4.5),
    strike_zone: tuple[float, float, float, float] = (-0.83, 0.83, 1.5, 3.5),
) -> str:
    """
    Render a single at-bat as a horizontal "pitch strip" SVG.

    Each pitch panel shows:
      - count + inning + bases
      - top-K pitch type bars
      - strike-zone plot with actual location (dot) + predicted mean (cross)

    Inputs are the JSON objects emitted by `python -m baseball simulate --events-out ...`.
    """

    if not events:
        raise TraceError("No events provided to render.")

    palette = _pitch_palette()
    bg = "#ffffff"
    border = "#e2e8f0"
    text = "#0f172a"
    subtext = "#475569"
    pred_color = "#f97316"

    xmin, xmax = float(x_range[0]), float(x_range[1])
    zmin, zmax = float(z_range[0]), float(z_range[1])
    zx0, zx1, zz0, zz1 = strike_zone

    max_p = int(max_pitches)
    if max_p <= 0:
        max_p = len(events)
    evs = events[:max_p]

    # Layout
    pad = 18
    title_h = 46
    panel_w = 186
    panel_h = 214
    gap = 14

    zone_w = 120
    zone_h = 120
    zone_x_off = (panel_w - zone_w) // 2
    zone_y_off = 54

    bars_h = 54
    bar_h = 9
    bar_gap = 3

    n = len(evs)
    width = pad * 2 + n * panel_w + (n - 1) * gap
    height = pad * 2 + title_h + panel_h

    def fx(x: float) -> float:
        return (x - xmin) / (xmax - xmin) * zone_w

    def fz(z: float) -> float:
        # higher z should appear higher on page
        return (zmax - z) / (zmax - zmin) * zone_h

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')
    svg.append(f'<text x="{pad}" y="{pad + 22}" font-size="18" font-weight="900" fill="{text}">{_escape(title)}</text>')
    svg.append(
        f'<text x="{pad}" y="{pad + 40}" font-size="12" font-weight="650" fill="{subtext}">dot = actual pitch location, cross = predicted mean</text>'
    )

    y0 = pad + title_h
    for i, rec in enumerate(evs):
        x0 = pad + i * (panel_w + gap)

        svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="14" fill="{bg}" stroke="{border}"/>')

        try:
            state = dict(rec.get("state") or {})
            y = dict(rec.get("y") or {})
            pred = dict(rec.get("pred") or {})
        except Exception as e:
            raise TraceError(f"Malformed event at index {i}: {e}") from e

        balls = int(state.get("balls", 0) or 0)
        strikes = int(state.get("strikes", 0) or 0)
        outs = int(state.get("outs_when_up", 0) or 0)
        inning = int(state.get("inning", 0) or 0)
        topbot = _topbot_token(int(state.get("inning_topbot_id", 0) or 0))
        bases = _bases_str(state)
        count_str = f"{balls}-{strikes}"

        actual_type = str(y.get("pitch_type", "OOV"))
        actual_x = float(y.get("plate_x", float("nan")))
        actual_z = float(y.get("plate_z", float("nan")))
        pred_x = float(pred.get("plate_x", float("nan")))
        pred_z = float(pred.get("plate_z", float("nan")))
        loc_nll = pred.get("loc_nll", None)

        topk = pred.get("pitch_type_topk", []) or []
        top1 = topk[0] if topk else {"pitch_type": pred.get("pitch_type", "OOV"), "prob": None}
        top1_tok = str(top1.get("pitch_type", "OOV"))
        top1_prob = top1.get("prob", None)

        header = f"#{int(rec.get('pitch_number', i+1))}  {count_str}  {topbot} {inning}  outs {outs}  {bases}"
        svg.append(f'<text x="{x0 + 12}" y="{y0 + 22}" font-size="12" font-weight="800" fill="{text}">{_escape(header)}</text>')

        line2 = f"actual {actual_type} • pred {top1_tok}"
        if isinstance(top1_prob, (int, float)):
            line2 += f" ({float(top1_prob):.2f})"
        if isinstance(loc_nll, (int, float)):
            line2 += f" • loc_nll {float(loc_nll):.2f}"
        svg.append(f'<text x="{x0 + 12}" y="{y0 + 40}" font-size="11" font-weight="650" fill="{subtext}">{_escape(line2)}</text>')

        # Strike zone mini-plot.
        zx = x0 + zone_x_off
        zy = y0 + zone_y_off
        svg.append(f'<rect x="{zx}" y="{zy}" width="{zone_w}" height="{zone_h}" fill="#ffffff" stroke="{border}"/>')

        # Strike zone overlay
        sx = zx + fx(zx0)
        sy = zy + fz(zz1)
        sw = fx(zx1) - fx(zx0)
        sh = fz(zz0) - fz(zz1)
        svg.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" fill="none" stroke="{text}" stroke-width="1.2" />')

        # Predicted mean cross
        if pred_x == pred_x and pred_z == pred_z:
            px = zx + fx(pred_x)
            pz = zy + fz(pred_z)
            d = 6.0
            svg.append(f'<line x1="{px - d:.1f}" y1="{pz:.1f}" x2="{px + d:.1f}" y2="{pz:.1f}" stroke="{pred_color}" stroke-width="2" />')
            svg.append(f'<line x1="{px:.1f}" y1="{pz - d:.1f}" x2="{px:.1f}" y2="{pz + d:.1f}" stroke="{pred_color}" stroke-width="2" />')

        # Actual pitch dot
        if actual_x == actual_x and actual_z == actual_z:
            ax = zx + fx(actual_x)
            az = zy + fz(actual_z)
            c = palette.get(actual_type, "#94a3b8")
            svg.append(f'<circle cx="{ax:.1f}" cy="{az:.1f}" r="4.0" fill="{c}" stroke="#0f172a" stroke-width="1.0" />')

        # Top-K bars
        bars_x = x0 + 14
        bars_y = zy + zone_h + 12
        bar_label_w = 28
        bar_w = panel_w - 28 - bar_label_w
        bar_w = max(40, bar_w)

        # Only show first 5 to keep tiles compact.
        top_items = list(topk)[:5]
        if not top_items:
            top_items = [{"pitch_type": str(pred.get("pitch_type", "OOV")), "prob": float(pred.get("prob", 0.0) or 0.0)}]

        for j, item in enumerate(top_items):
            tok = str(item.get("pitch_type", "OOV"))
            prob = float(item.get("prob", 0.0) or 0.0)
            by = bars_y + j * (bar_h + bar_gap)
            svg.append(f'<text x="{bars_x}" y="{by + 8}" font-size="10" font-weight="800" fill="{subtext}">{_escape(tok)}</text>')
            fill = palette.get(tok, "#94a3b8")
            w = max(0.0, min(1.0, prob)) * bar_w
            svg.append(f'<rect x="{bars_x + bar_label_w}" y="{by}" width="{w:.1f}" height="{bar_h}" fill="{fill}" opacity="0.9" />')
            svg.append(f'<rect x="{bars_x + bar_label_w}" y="{by}" width="{bar_w}" height="{bar_h}" fill="none" stroke="{border}" />')
            svg.append(
                f'<text x="{bars_x + bar_label_w + bar_w + 6}" y="{by + 8}" font-size="10" font-weight="750" fill="{subtext}">{prob:.2f}</text>'
            )

    svg.append("</svg>")
    return "\n".join(svg)


def render_trace_html(
    *,
    trace_path: Path,
    game_pk: int | None = None,
    max_at_bats: int = 10,
    max_pitches_per_ab: int = 12,
) -> str:
    """
    Render a trace JSONL to a single self-contained HTML file.

    This is meant for quick sharing: open in a browser, scroll, and read.
    """

    events = list(iter_trace_events_jsonl(trace_path))
    events = _select_game(events, game_pk=game_pk)

    # Group by at_bat_number
    by_ab: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        ab = int(e.get("at_bat_number", -1))
        by_ab.setdefault(ab, []).append(e)
    abs_sorted = sorted(by_ab.keys())
    abs_sorted = abs_sorted[: max(0, int(max_at_bats))]

    if not abs_sorted:
        raise TraceError("No at-bats found after filtering.")

    first = by_ab[abs_sorted[0]][0]
    mode = str(first.get("mode", ""))
    gpk = int(first.get("game_pk", -1))
    pitcher_label = str(first.get("pitcher", first.get("pitcher_id", -1)))
    batter_label = str(first.get("batter", first.get("batter_id", -1)))
    stand = _stand_token(int(first.get("stand_id", 0)))
    pthrows = _throws_token(int(first.get("p_throws_id", 0)))

    title = f"Pitch-by-pitch replay ({mode}) — game {gpk}"
    subtitle = f"example matchup: pitcher {pitcher_label} ({pthrows}) vs batter {batter_label} ({stand})"

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8" />')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1" />')
    parts.append(f"<title>{_escape(title)}</title>")
    parts.append(
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#0f172a;}"
        "h1{margin:0 0 6px 0;font-size:22px;}"
        "h2{margin:24px 0 8px 0;font-size:16px;}"
        ".sub{color:#475569;margin:0 0 18px 0;}"
        ".card{border:1px solid #e2e8f0;border-radius:14px;padding:14px;margin:14px 0;}"
        ".meta{color:#475569;font-size:12px;margin-top:6px;}"
        "code{background:#f8fafc;padding:2px 6px;border-radius:6px;}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append(f"<h1>{_escape(title)}</h1>")
    parts.append(f'<p class="sub">{_escape(subtitle)}</p>')
    parts.append(
        "<div class=\"card\">"
        "<div><strong>How to read this:</strong> each at-bat is rendered as a left-to-right strip. "
        "For each pitch: the top shows count/inning/bases, the mini strike zone shows actual location (dot) and predicted mean (cross), "
        "and the bars show the model’s top pitch-type probabilities.</div>"
        "</div>"
    )

    for ab in abs_sorted:
        evs = by_ab[ab]
        evs.sort(key=lambda r: int(r.get("pitch_number", 0)))
        first_ab = evs[0]
        pitcher_label = str(first_ab.get("pitcher", first_ab.get("pitcher_id", -1)))
        batter_label = str(first_ab.get("batter", first_ab.get("batter_id", -1)))
        stand = _stand_token(int(first_ab.get("stand_id", 0)))
        pthrows = _throws_token(int(first_ab.get("p_throws_id", 0)))
        parts.append(f"<h2>At-bat {ab}</h2>")
        parts.append(f'<div class="meta">pitcher {pitcher_label} ({pthrows}) vs batter {batter_label} ({stand})</div>')
        parts.append("<div class=\"card\">")
        parts.append(
            render_at_bat_strip_svg(
                events=evs,
                title=f"At-bat {ab}",
                max_pitches=int(max_pitches_per_ab),
            )
        )
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def render_trace_file(
    *,
    trace_path: Path,
    out_path: Path,
    fmt: str,
    game_pk: int | None = None,
    at_bat_number: int | None = None,
    max_at_bats: int = 10,
    max_pitches_per_ab: int = 12,
) -> None:
    fmt = str(fmt).strip().lower()
    if fmt not in {"html", "svg"}:
        raise TraceError(f"Unsupported format: {fmt!r} (expected 'html' or 'svg')")

    op = Path(out_path)
    if str(op).strip() == "-" or op.as_posix().strip() == "-":
        raise TraceError("--out must be a file path (not '-')")
    op.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        html = render_trace_html(
            trace_path=Path(trace_path),
            game_pk=game_pk,
            max_at_bats=int(max_at_bats),
            max_pitches_per_ab=int(max_pitches_per_ab),
        )
        op.write_text(html + "\n", encoding="utf-8")
        return

    # SVG = single at-bat strip
    events = list(iter_trace_events_jsonl(Path(trace_path)))
    events = _select_game(events, game_pk=game_pk)
    ab, sub = _select_at_bat(events, at_bat_number=at_bat_number)
    title = f"Replay strip — game {int(sub[0].get('game_pk', -1))} — at-bat {ab}"
    svg = render_at_bat_strip_svg(events=sub, title=title, max_pitches=int(max_pitches_per_ab))
    op.write_text(svg + "\n", encoding="utf-8")
