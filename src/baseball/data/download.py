from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from baseball.data.raw_schema import RAW_REQUIRED_COLUMNS


class DownloadError(RuntimeError):
    pass


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date '{s}'. Expected YYYY-MM-DD.") from e


def _date_chunks(start: date, end_inclusive: date, chunk_days: int) -> list[tuple[date, date]]:
    if chunk_days <= 0:
        raise ValueError("--chunk-days must be > 0")
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end_inclusive:
        chunk_end = min(end_inclusive, cur + timedelta(days=chunk_days - 1))
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


@dataclass(frozen=True)
class DownloadResult:
    chunk_start: date
    chunk_end: date
    rows: int
    path: Path


def _fetch_statcast(start: date, end: date) -> pd.DataFrame:
    """
    Fetch Statcast pitch-by-pitch data for [start, end].

    Uses pybaseball's Statcast integration as the source of truth for request details.
    """

    try:
        from pybaseball import statcast  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise DownloadError(
            "pybaseball is required for downloads. Install deps from environment.yml."
        ) from e

    # pybaseball expects strings in YYYY-MM-DD format
    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    if not isinstance(df, pd.DataFrame):
        raise DownloadError("pybaseball.statcast returned unexpected type.")
    return df


def download_statcast_range(start: str, end: str, out_dir: Path, chunk_days: int = 7) -> list[DownloadResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_d = _parse_date(start)
    end_d = _parse_date(end)
    if end_d < start_d:
        raise ValueError("--end must be >= --start")

    chunks = _date_chunks(start_d, end_d, chunk_days=chunk_days)
    results: list[DownloadResult] = []

    def _write_empty_chunk(path: Path) -> None:
        # Some Statcast date ranges (offseason / no games) can return an empty DataFrame
        # with *no columns*. We still write an empty parquet with a stable schema so
        # downstream `prepare` can scan all raw chunks without failing on empty schemas.
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in RAW_REQUIRED_COLUMNS})
        df.to_parquet(path, index=False)

    for chunk_start, chunk_end in tqdm(chunks, desc="Downloading Statcast"):
        out_path = out_dir / f"statcast_{chunk_start.isoformat()}_{chunk_end.isoformat()}.parquet"
        if out_path.exists():
            # Explicit skip to support resuming large downloads.
            try:
                pf = pq.ParquetFile(out_path)
            except Exception:
                # Corrupt/partial file; force re-download.
                out_path.unlink(missing_ok=True)
            else:
                cols = list(getattr(pf.schema, "names", []) or [])
                rows = int(getattr(getattr(pf, "metadata", None), "num_rows", 0) or 0)
                if len(cols) > 0:
                    results.append(
                        DownloadResult(chunk_start=chunk_start, chunk_end=chunk_end, rows=rows, path=out_path)
                    )
                    continue
                # Empty-schema file (common in offseason); rewrite deterministically.
                out_path.unlink(missing_ok=True)

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                df = _fetch_statcast(chunk_start, chunk_end)
                if len(getattr(df, "columns", [])) == 0:
                    _write_empty_chunk(out_path)
                    rows = 0
                else:
                    df.to_parquet(out_path, index=False)
                    rows = len(df)
                results.append(
                    DownloadResult(chunk_start=chunk_start, chunk_end=chunk_end, rows=rows, path=out_path)
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Simple backoff; keep it explicit, fail if still broken.
                time.sleep(2**attempt)

        if last_err is not None:
            raise DownloadError(
                f"Failed downloading Statcast chunk {chunk_start}..{chunk_end} after 3 attempts: {last_err}"
            )

    return results
