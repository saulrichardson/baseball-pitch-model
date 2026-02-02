from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


class VizError(RuntimeError):
    pass


@dataclass(frozen=True)
class PitchTypeCols:
    token: str
    pred_col: str
    emp_col: str


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    s = str(h).lstrip("#")
    if len(s) != 6:
        raise VizError(f"Invalid hex color: {h!r}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    r = int(max(0, min(255, round(r * 255))))
    g = int(max(0, min(255, round(g * 255))))
    b = int(max(0, min(255, round(b * 255))))
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_hex(a: str, b: str, t: float) -> str:
    t = float(max(0.0, min(1.0, t)))
    ar, ag, ab = _hex_to_rgb01(a)
    br, bg, bb = _hex_to_rgb01(b)
    return _rgb01_to_hex((ar + t * (br - ar), ag + t * (bg - ag), ab + t * (bb - ab)))


def _escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def load_profile(profile_parquet: Path) -> pl.DataFrame:
    p = Path(profile_parquet)
    if not p.exists():
        raise VizError(f"Missing profile parquet: {p}")
    return pl.read_parquet(p)


def list_pitch_types(df: pl.DataFrame) -> list[PitchTypeCols]:
    cols = df.columns
    pred = [c for c in cols if c.startswith("pred_prob_")]
    out: list[PitchTypeCols] = []
    for pc in pred:
        tok = pc.removeprefix("pred_prob_")
        ec = f"emp_prob_{tok}"
        if ec in cols:
            out.append(PitchTypeCols(token=tok, pred_col=pc, emp_col=ec))
    if not out:
        raise VizError("No pitch type probability columns found (expected pred_prob_* / emp_prob_*)")
    return out


def _resolve_pitcher_df(df: pl.DataFrame, pitcher: str) -> pl.DataFrame:
    if "pitcher" not in df.columns and "pitcher_id" not in df.columns:
        raise VizError("Profile parquet must include pitcher and pitcher_id columns.")

    p = str(pitcher).strip()
    # Prefer exact match on the human-readable token column.
    if "pitcher" in df.columns:
        sub = df.filter(pl.col("pitcher") == p)
        if not sub.is_empty():
            return sub
    # Fallback: pitcher_id (vocab id), if user passed an int.
    try:
        pid = int(p)
    except ValueError:
        pid = None
    if pid is not None and "pitcher_id" in df.columns:
        sub = df.filter(pl.col("pitcher_id") == pid)
        if not sub.is_empty():
            return sub

    raise VizError(f"Pitcher not found in profile parquet: {pitcher!r}")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = float(weights.sum())
    if wsum <= 0:
        return float("nan")
    return float((values * weights).sum() / wsum)


def build_count_policy_grid(df_pitcher: pl.DataFrame, pitch_types: list[PitchTypeCols]) -> dict[tuple[int, int], dict[str, object]]:
    """
    Returns per-count (balls,strikes) summaries:
      - n_total
      - pred_top_token, pred_top_prob
      - emp_top_token, emp_top_prob
    """

    if "balls" not in df_pitcher.columns or "strikes" not in df_pitcher.columns:
        raise VizError("Profile parquet must include balls/strikes for count visualizations.")
    if "n" not in df_pitcher.columns:
        raise VizError("Profile parquet must include n for weighting.")

    out: dict[tuple[int, int], dict[str, object]] = {}
    for (b, s), g in df_pitcher.group_by(["balls", "strikes"], maintain_order=True):
        balls = int(g["balls"][0])
        strikes = int(g["strikes"][0])
        w = g["n"].to_numpy().astype(np.float64, copy=False)
        n_total = int(w.sum())

        pred_probs = []
        emp_probs = []
        for pt in pitch_types:
            pred_probs.append(_weighted_mean(g[pt.pred_col].to_numpy().astype(np.float64, copy=False), w))
            emp_probs.append(_weighted_mean(g[pt.emp_col].to_numpy().astype(np.float64, copy=False), w))

        pred_probs_arr = np.asarray(pred_probs, dtype=np.float64)
        emp_probs_arr = np.asarray(emp_probs, dtype=np.float64)
        pred_idx = int(np.nanargmax(pred_probs_arr))
        emp_idx = int(np.nanargmax(emp_probs_arr))

        out[(balls, strikes)] = {
            "n": n_total,
            "pred_top_token": pitch_types[pred_idx].token,
            "pred_top_prob": float(pred_probs_arr[pred_idx]),
            "emp_top_token": pitch_types[emp_idx].token,
            "emp_top_prob": float(emp_probs_arr[emp_idx]),
        }
    return out


def _pitch_palette() -> dict[str, str]:
    # Conservative palette: stable, high-contrast, readable in small tiles.
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
        "CS": "#94a3b8",  # called strike? (should not be a pitch type; safe fallback)
        "SC": "#94a3b8",  # screwball? (rare)
        "OOV": "#cbd5e1",
    }


def render_count_policy_svg(
    *,
    title: str,
    count_policy: dict[tuple[int, int], dict[str, object]],
    subtitle_left: str = "Predicted top pitch",
    subtitle_right: str = "Empirical top pitch",
) -> str:
    """
    SVG: 4x3 count grid, predicted vs empirical, categorical coloring by top pitch type.
    """

    palette = _pitch_palette()
    bg = "#ffffff"
    border = "#cbd5e1"
    text = "#0f172a"
    subtext = "#475569"

    # Layout
    cell = 62
    pad = 22
    grid_w = cell * 4
    grid_h = cell * 3
    gap = 34
    # Leave extra headroom so subtitles don't collide with the global subtitle line.
    title_h = 74
    footer_h = 28

    width = pad * 2 + grid_w * 2 + gap
    height = pad * 2 + title_h + grid_h + footer_h

    def cell_xy(col: int, row: int, *, grid_x0: int, grid_y0: int) -> tuple[int, int]:
        return (grid_x0 + col * cell, grid_y0 + row * cell)

    # strikes rows top->bottom: 2,1,0 for familiar "2 strikes at top" view
    strikes_order = [2, 1, 0]

    left_x0 = pad
    right_x0 = pad + grid_w + gap
    y0 = pad + title_h

    def render_grid(x0: int, key_prefix: str) -> list[str]:
        parts: list[str] = []
        for r, s in enumerate(strikes_order):
            for b in [0, 1, 2, 3]:
                rec = count_policy.get((b, s))
                if rec is None:
                    token = "—"
                    prob = float("nan")
                    n = 0
                else:
                    token = str(rec[f"{key_prefix}_top_token"])
                    prob = float(rec[f"{key_prefix}_top_prob"])
                    n = int(rec["n"])

                base = palette.get(token, "#94a3b8")
                # Intensity by probability, but keep readable even for small probs.
                t = 0.15 + 0.85 * float(max(0.0, min(1.0, prob))) if prob == prob else 0.0
                fill = _blend_hex("#ffffff", base, t)

                x, y = cell_xy(b, r, grid_x0=x0, grid_y0=y0)
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{border}" />')
                parts.append(
                    f'<text x="{x + cell/2:.1f}" y="{y + 26}" text-anchor="middle" font-size="14" font-weight="700" fill="{text}">{_escape(token)}</text>'
                )
                if prob == prob:
                    parts.append(
                        f'<text x="{x + cell/2:.1f}" y="{y + 46}" text-anchor="middle" font-size="12" font-weight="600" fill="{text}">{prob:.2f}</text>'
                    )
                parts.append(
                    f'<text x="{x + cell - 6}" y="{y + cell - 8}" text-anchor="end" font-size="10" font-weight="600" fill="{subtext}">n={n}</text>'
                )
        return parts

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')
    svg.append(
        f'<text x="{pad}" y="{pad + 22}" font-size="18" font-weight="800" fill="{text}">{_escape(title)}</text>'
    )
    svg.append(
        f'<text x="{pad}" y="{pad + 44}" font-size="12" font-weight="600" fill="{subtext}">balls × strikes grid (top = 2 strikes)</text>'
    )

    # Subtitles
    svg.append(
        f'<text x="{left_x0}" y="{y0 - 10}" font-size="12" font-weight="800" fill="{text}">{_escape(subtitle_left)}</text>'
    )
    svg.append(
        f'<text x="{right_x0}" y="{y0 - 10}" font-size="12" font-weight="800" fill="{text}">{_escape(subtitle_right)}</text>'
    )

    svg.extend(render_grid(left_x0, "pred"))
    svg.extend(render_grid(right_x0, "emp"))

    # Axes labels (balls)
    for i, b in enumerate([0, 1, 2, 3]):
        x = left_x0 + i * cell + cell / 2
        svg.append(
            f'<text x="{x:.1f}" y="{y0 + grid_h + 18}" text-anchor="middle" font-size="11" font-weight="700" fill="{subtext}">{b} balls</text>'
        )
        x2 = right_x0 + i * cell + cell / 2
        svg.append(
            f'<text x="{x2:.1f}" y="{y0 + grid_h + 18}" text-anchor="middle" font-size="11" font-weight="700" fill="{subtext}">{b} balls</text>'
        )

    # strikes labels
    for r, s in enumerate(strikes_order):
        y = y0 + r * cell + cell / 2 + 4
        svg.append(
            f'<text x="{left_x0 - 10}" y="{y:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{subtext}">{s} strikes</text>'
        )
        svg.append(
            f'<text x="{right_x0 - 10}" y="{y:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{subtext}">{s} strikes</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _gaussian_heatmap(
    *,
    xs: np.ndarray,
    zs: np.ndarray,
    weights: np.ndarray,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    nx: int,
    nz: int,
    sigma: float,
) -> np.ndarray:
    if xs.size == 0:
        return np.zeros((nz, nx), dtype=np.float64)
    if not (xs.shape == zs.shape == weights.shape):
        raise VizError("xs/zs/weights must have same shape")

    gx = np.linspace(float(x_min), float(x_max), int(nx), dtype=np.float64)
    gz = np.linspace(float(z_min), float(z_max), int(nz), dtype=np.float64)

    # Evaluate isotropic Gaussian kernels; small N (profile groups) so this is fine.
    out = np.zeros((int(nz), int(nx)), dtype=np.float64)
    inv2s2 = 1.0 / (2.0 * float(sigma) * float(sigma))
    for x, z, w in zip(xs, zs, weights):
        if not np.isfinite(x) or not np.isfinite(z) or not np.isfinite(w):
            continue
        if w <= 0:
            continue
        dx2 = (gx - float(x)) ** 2  # [nx]
        dz2 = (gz - float(z)) ** 2  # [nz]
        kern = np.exp(-(dz2[:, None] + dx2[None, :]) * inv2s2)
        out += float(w) * kern
    return out


def render_zone_heatmaps_svg(
    *,
    title: str,
    df_pitcher: pl.DataFrame,
    pitch_types: list[PitchTypeCols],
    pitch_tokens: Iterable[str],
    x_range: tuple[float, float] = (-2.0, 2.0),
    z_range: tuple[float, float] = (0.5, 4.5),
    bins: tuple[int, int] = (42, 42),
    sigma_ft: float = 0.22,
    strike_zone: tuple[float, float, float, float] = (-0.83, 0.83, 1.5, 3.5),
) -> str:
    """
    SVG small multiples: for each pitch token, show predicted vs empirical location heatmaps.

    Heatmaps are built from profile-group mean locations, weighted by:
      n * pred_prob_token (predicted)
      n * emp_prob_token  (empirical)
    """

    if "n" not in df_pitcher.columns:
        raise VizError("Profile parquet must include n")
    for c in ["pred_plate_x", "pred_plate_z", "emp_plate_x", "emp_plate_z"]:
        if c not in df_pitcher.columns:
            raise VizError(f"Profile parquet missing column: {c}")

    pt_map = {pt.token: pt for pt in pitch_types}
    tokens = [t for t in (str(x).strip() for x in pitch_tokens) if t]
    tokens = [t for t in tokens if t in pt_map]
    if not tokens:
        raise VizError("No requested pitch types found in profile parquet.")

    palette = _pitch_palette()
    border = "#cbd5e1"
    text = "#0f172a"
    subtext = "#475569"
    bg = "#ffffff"

    nx, nz = int(bins[0]), int(bins[1])
    xmin, xmax = float(x_range[0]), float(x_range[1])
    zmin, zmax = float(z_range[0]), float(z_range[1])

    # Layout
    pad = 22
    # Leave extra headroom so column headers don't collide with the global subtitle line.
    title_h = 74
    cell_px = 5  # pixel size per bin (small enough for GitHub, large enough to read)
    heat_w = nx * cell_px
    heat_h = nz * cell_px
    gap_x = 26
    gap_y = 34
    col_w = heat_w
    row_h = heat_h

    width = pad * 2 + col_w * 2 + gap_x
    height = pad * 2 + title_h + len(tokens) * row_h + (len(tokens) - 1) * gap_y + 22

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')
    svg.append(f'<text x="{pad}" y="{pad + 22}" font-size="18" font-weight="800" fill="{text}">{_escape(title)}</text>')
    svg.append(
        f'<text x="{pad}" y="{pad + 44}" font-size="12" font-weight="600" fill="{subtext}">batter-box heatmaps from profile-group means (smoothed)</text>'
    )

    # Column headers
    x_left = pad
    x_right = pad + col_w + gap_x
    y0 = pad + title_h
    svg.append(f'<text x="{x_left}" y="{y0 - 10}" font-size="12" font-weight="800" fill="{text}">Predicted</text>')
    svg.append(f'<text x="{x_right}" y="{y0 - 10}" font-size="12" font-weight="800" fill="{text}">Empirical</text>')

    def draw_heat(x0: int, y0: int, H: np.ndarray, base_color: str) -> None:
        vmax = float(np.max(H)) if H.size else 0.0
        if vmax <= 0:
            vmax = 1.0
        # Flip vertically so higher z is higher on page.
        H2 = H[::-1, :]
        for iz in range(H2.shape[0]):
            for ix in range(H2.shape[1]):
                v = float(H2[iz, ix]) / vmax
                fill = _blend_hex("#ffffff", base_color, min(1.0, max(0.0, v)))
                svg.append(
                    f'<rect x="{x0 + ix * cell_px}" y="{y0 + iz * cell_px}" width="{cell_px}" height="{cell_px}" fill="{fill}" />'
                )
        # Border
        svg.append(f'<rect x="{x0}" y="{y0}" width="{heat_w}" height="{heat_h}" fill="none" stroke="{border}" />')

        # Strike zone overlay (approx, in feet).
        zx0, zx1, zz0, zz1 = strike_zone
        # Map feet -> pixel
        def fx(x: float) -> float:
            return (x - xmin) / (xmax - xmin) * heat_w

        def fz(z: float) -> float:
            # y is flipped (higher z at top)
            return (zmax - z) / (zmax - zmin) * heat_h

        sx = x0 + fx(zx0)
        sy = y0 + fz(zz1)
        sw = fx(zx1) - fx(zx0)
        sh = fz(zz0) - fz(zz1)
        svg.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" fill="none" stroke="#0f172a" stroke-width="1.2" />')

    n_arr = df_pitcher["n"].to_numpy().astype(np.float64, copy=False)
    pred_x = df_pitcher["pred_plate_x"].to_numpy().astype(np.float64, copy=False)
    pred_z = df_pitcher["pred_plate_z"].to_numpy().astype(np.float64, copy=False)
    emp_x = df_pitcher["emp_plate_x"].to_numpy().astype(np.float64, copy=False)
    emp_z = df_pitcher["emp_plate_z"].to_numpy().astype(np.float64, copy=False)

    for i, tok in enumerate(tokens):
        pt = pt_map[tok]
        base = palette.get(tok, "#0ea5e9")

        p_pred = df_pitcher[pt.pred_col].to_numpy().astype(np.float64, copy=False)
        p_emp = df_pitcher[pt.emp_col].to_numpy().astype(np.float64, copy=False)

        w_pred = n_arr * p_pred
        w_emp = n_arr * p_emp

        H_pred = _gaussian_heatmap(
            xs=pred_x,
            zs=pred_z,
            weights=w_pred,
            x_min=xmin,
            x_max=xmax,
            z_min=zmin,
            z_max=zmax,
            nx=nx,
            nz=nz,
            sigma=float(sigma_ft),
        )
        H_emp = _gaussian_heatmap(
            xs=emp_x,
            zs=emp_z,
            weights=w_emp,
            x_min=xmin,
            x_max=xmax,
            z_min=zmin,
            z_max=zmax,
            nx=nx,
            nz=nz,
            sigma=float(sigma_ft),
        )

        row_y = y0 + i * (row_h + gap_y)

        svg.append(f'<text x="{pad}" y="{row_y - 8}" font-size="12" font-weight="800" fill="{text}">{_escape(tok)}</text>')
        draw_heat(x_left, row_y, H_pred, base)
        draw_heat(x_right, row_y, H_emp, base)

    svg.append("</svg>")
    return "\n".join(svg)
