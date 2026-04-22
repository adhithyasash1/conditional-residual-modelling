"""Download mixed-source market data and align it to observed SPX trading dates."""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from .config import (
    DATA_END_DATE,
    DATA_START_DATE,
    FRED_API_KEY,
    FRED_SERIES,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_COL,
    YAHOO_CLOSE_SERIES,
    YAHOO_OHLCV_SERIES,
    get_logger,
)

logger = get_logger(__name__)

FRED_JSON_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MONTHLY_COLS: tuple[str, ...] = ("fed_funds",)
RAW_PANEL_FILENAME = "market_macro_panel.csv"


class DataDownloadError(RuntimeError):
    """Raised when a required data source cannot be fetched."""


def _normalized_yahoo_end(end: Optional[str]) -> Optional[str]:
    if not end:
        return None
    return (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _download_yahoo_close(
    alias: str,
    ticker: str,
    start: str = DATA_START_DATE,
    end: Optional[str] = DATA_END_DATE,
    timeout: int = 30,
) -> pd.DataFrame:
    frame = yf.download(
        tickers=ticker,
        start=start,
        end=_normalized_yahoo_end(end),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
        repair=True,
        timeout=timeout,
    )
    if frame is None or frame.empty:
        raise DataDownloadError(f"Yahoo returned no rows for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    value_col = "Close" if "Close" in frame.columns else "Adj Close"
    if value_col not in frame.columns:
        raise DataDownloadError(f"Yahoo response for {ticker} missing close column.")
    out = frame[[value_col]].reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: "date", value_col: alias})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out[alias] = pd.to_numeric(out[alias], errors="coerce")
    return out.dropna(subset=[alias]).sort_values("date").reset_index(drop=True)


def _download_yahoo_ohlcv(
    alias: str,
    ticker: str,
    start: str = DATA_START_DATE,
    end: Optional[str] = DATA_END_DATE,
    timeout: int = 30,
) -> pd.DataFrame:
    frame = yf.download(
        tickers=ticker,
        start=start,
        end=_normalized_yahoo_end(end),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
        repair=True,
        timeout=timeout,
    )
    if frame is None or frame.empty:
        raise DataDownloadError(f"Yahoo returned no rows for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    out = frame.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    rename_map = {date_col: "date"}
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in out.columns:
            normalized = col.lower().replace("adj close", "adj_close")
            rename_map[col] = f"{alias}_{normalized}"
    out = out.rename(columns=rename_map)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    numeric_cols = [col for col in out.columns if col != "date"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if f"{alias}_close" not in out.columns:
        raise DataDownloadError(f"Yahoo response for {ticker} missing close column.")
    return out.dropna(subset=[f"{alias}_close"]).sort_values("date").reset_index(drop=True)


def _download_fred_series(
    alias: str,
    series_id: str,
    start: str = DATA_START_DATE,
    end: Optional[str] = DATA_END_DATE,
    timeout: int = 30,
) -> pd.DataFrame:
    if not FRED_API_KEY:
        logger.info("FRED_API_KEY not set. Falling back directly to the public CSV endpoint for %s.", series_id)
        return _download_fred_csv(alias=alias, series_id=series_id, start=start, end=end, timeout=timeout)

    params = {"series_id": series_id, "file_type": "json", "observation_start": start}
    params["api_key"] = FRED_API_KEY
    if end:
        params["observation_end"] = end

    try:
        response = requests.get(FRED_JSON_ENDPOINT, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", [])
        if observations:
            out = pd.DataFrame(observations)[["date", "value"]]
            out["date"] = pd.to_datetime(out["date"])
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out = out.dropna(subset=["value"]).rename(columns={"value": alias})
            return out.sort_values("date").reset_index(drop=True)
    except requests.RequestException as exc:
        logger.warning("FRED JSON request failed for %s: %s", series_id, exc)

    return _download_fred_csv(alias=alias, series_id=series_id, start=start, end=end, timeout=timeout)


def _download_fred_csv(
    alias: str,
    series_id: str,
    start: str = DATA_START_DATE,
    end: Optional[str] = DATA_END_DATE,
    timeout: int = 30,
) -> pd.DataFrame:
    csv_params = {"id": series_id, "cosd": start}
    if end:
        csv_params["coed"] = end
    csv_response = requests.get(FRED_CSV_ENDPOINT, params=csv_params, timeout=timeout)
    csv_response.raise_for_status()
    out = pd.read_csv(StringIO(csv_response.text))
    if out.empty or out.shape[1] < 2:
        raise DataDownloadError(f"FRED CSV returned no observations for {series_id}.")
    value_col = out.columns[1]
    out = out.rename(columns={out.columns[0]: "date", value_col: alias})
    out["date"] = pd.to_datetime(out["date"])
    out[alias] = pd.to_numeric(out[alias], errors="coerce")
    return out.dropna(subset=[alias]).sort_values("date").reset_index(drop=True)


def download_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download all configured Yahoo Finance and FRED series."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for alias, ticker in YAHOO_CLOSE_SERIES.items():
        logger.info("Downloading Yahoo close series %s (%s)", alias, ticker)
        frame = _download_yahoo_close(alias, ticker)
        frame.to_csv(raw_dir / f"yahoo_{alias}.csv", index=False)
        frames.append(frame.set_index("date"))

    for alias, ticker in YAHOO_OHLCV_SERIES.items():
        logger.info("Downloading Yahoo OHLCV series %s (%s)", alias, ticker)
        frame = _download_yahoo_ohlcv(alias, ticker)
        frame.to_csv(raw_dir / f"yahoo_{alias}.csv", index=False)
        frames.append(frame.set_index("date"))

    for alias, series_id in FRED_SERIES.items():
        logger.info("Downloading FRED series %s (%s)", alias, series_id)
        frame = _download_fred_series(alias, series_id)
        frame.to_csv(raw_dir / f"fred_{alias}.csv", index=False)
        frames.append(frame.set_index("date"))

    panel = pd.concat(frames, axis=1).sort_index()
    panel.index.name = "date"
    panel.to_csv(raw_dir / RAW_PANEL_FILENAME)
    logger.info("Saved raw panel with shape %s to %s", panel.shape, raw_dir / RAW_PANEL_FILENAME)
    return panel


def read_raw_panel(path: Path | None = None) -> pd.DataFrame:
    panel_path = path or (RAW_DIR / RAW_PANEL_FILENAME)
    if not panel_path.exists():
        raise FileNotFoundError(f"Raw panel not found at {panel_path}. Run download_data.py first.")
    return pd.read_csv(panel_path, parse_dates=["date"]).set_index("date").sort_index()


def align_to_spx_trading_dates(
    panel: pd.DataFrame,
    daily_ffill_limit: int = 10,
    monthly_cols: tuple[str, ...] = MONTHLY_COLS,
) -> pd.DataFrame:
    """Anchor the full feature panel on observed SPX trading dates."""
    if TARGET_COL not in panel.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in the raw panel.")

    spx = panel[TARGET_COL].dropna()
    spx = spx[spx > 0]
    if spx.empty:
        raise ValueError("SPX series is empty after removing missing and non-positive values.")

    trading_index = spx.index.sort_values().unique()
    aligned = pd.DataFrame(index=trading_index)
    aligned[TARGET_COL] = panel[TARGET_COL].reindex(trading_index)

    daily_cols = [col for col in panel.columns if col not in monthly_cols and col != TARGET_COL]
    if daily_cols:
        aligned[daily_cols] = panel[daily_cols].reindex(trading_index).ffill(limit=daily_ffill_limit)

    monthly_present = [col for col in monthly_cols if col in panel.columns]
    if monthly_present:
        aligned[monthly_present] = panel[monthly_present].reindex(trading_index).ffill()

    aligned = aligned.dropna(subset=[TARGET_COL]).copy()
    aligned = aligned[aligned[TARGET_COL] > 0]
    aligned.index.name = "date"
    return aligned


def build_master_dataset(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    download_if_missing: bool = False,
) -> pd.DataFrame:
    """Build the target-aware daily master dataset from raw files."""
    panel_path = raw_dir / RAW_PANEL_FILENAME
    if not panel_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"{panel_path} does not exist. Run scripts/download_data.py or set download_if_missing=True."
            )
        download_all(raw_dir=raw_dir)

    panel = read_raw_panel(panel_path)
    master = align_to_spx_trading_dates(panel)
    out_path = processed_dir / "master_daily.csv"
    master.to_csv(out_path)
    logger.info("Saved master daily dataset with shape %s to %s", master.shape, out_path)
    return master
