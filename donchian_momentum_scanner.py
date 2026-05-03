"""
Donchian + MACD + Volume momentum scanner

Purpose
- Finds BUY candidates using a short-term trend-following breakout rule.
- This script only generates signals. It does NOT place real orders.

Strategy summary
BUY candidate when all are true:
1) Close breaks above prior N-day Donchian high
2) MACD is above signal line and MACD histogram is rising for K days
3) Close is above 60-day moving average
4) Recent 5-day average volume is above prior 20-day average volume

Risk / management rules attached to each candidate:
- Time stop: if position does not reach +5% within 20 trading days, exit
- First take profit: +8%, sell 40%
- Second take profit: +20% to +30%, sell another 30% to 40%
- Stop loss: -12%, exit all
- Trailing exit: 5-day MA break, MACD bearish turn, or -10% from peak

Example:
python donchian_momentum_scanner.py --tickers 005930.KS 034020.KS 449450.KS --period 1y
python donchian_momentum_scanner.py --file tickers.txt --output signals.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from yahooquery import Ticker


@dataclass
class ScannerConfig:
    donchian_window: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_rising_days: int = 3
    trend_ma_window: int = 60
    recent_volume_window: int = 5
    prior_volume_window: int = 20
    min_volume_ratio: float = 1.0
    time_stop_days: int = 20
    time_stop_min_return: float = 5.0
    first_take_profit: float = 8.0
    first_take_profit_sell_pct: float = 40.0
    second_take_profit_low: float = 20.0
    second_take_profit_high: float = 30.0
    stop_loss: float = -12.0
    trailing_drawdown: float = -10.0


@dataclass
class SignalResult:
    ticker: str
    signal: str
    close: float
    signal_date: str
    donchian_high_prior: float
    breakout_pct: float
    macd: float
    signal_line: float
    macd_hist: float
    macd_hist_rising_days: int
    close_vs_60ma_pct: float
    volume_ratio_5d_vs_prior20d: float
    buy_reason: str
    management_rule: str


def normalize_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return single-ticker OHLCV DataFrame indexed by date."""
    if raw is None or raw.empty or isinstance(raw, dict):
        return pd.DataFrame()

    data = raw.reset_index()

    # yahooquery may return multi-symbol rows. Keep the requested symbol when available.
    if "symbol" in data.columns:
        data = data[data["symbol"] == ticker]

    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(data.columns)):
        return pd.DataFrame()

    data = data[["date", "open", "high", "low", "close", "volume"]].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce").dt.tz_localize(None)
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    data = data.set_index("date")

    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["high", "low", "close"])


def add_indicators(data: pd.DataFrame, cfg: ScannerConfig) -> pd.DataFrame:
    df = data.copy()

    # Prior Donchian high excludes today's high to avoid look-ahead bias.
    df["donchian_high_prior"] = df["high"].shift(1).rolling(cfg.donchian_window).max()
    df["breakout"] = df["close"] > df["donchian_high_prior"]
    df["breakout_pct"] = (df["close"] / df["donchian_high_prior"] - 1.0) * 100

    df["ema_fast"] = df["close"].ewm(span=cfg.macd_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.macd_slow, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["signal_line"] = df["macd"].ewm(span=cfg.macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["signal_line"]
    df["macd_bullish"] = df["macd"] > df["signal_line"]

    # True when MACD histogram has risen for K consecutive sessions.
    hist_diff = df["macd_hist"].diff()
    df["macd_hist_rising"] = hist_diff > 0
    df["macd_hist_rising_count"] = (
        df["macd_hist_rising"]
        .rolling(cfg.macd_rising_days)
        .sum()
        .fillna(0)
        .astype(int)
    )
    df["macd_rising_ok"] = df["macd_hist_rising_count"] >= cfg.macd_rising_days

    df["ma60"] = df["close"].rolling(cfg.trend_ma_window).mean()
    df["above_ma60"] = df["close"] > df["ma60"]
    df["close_vs_60ma_pct"] = (df["close"] / df["ma60"] - 1.0) * 100

    df["volume"] = df["volume"].fillna(0)
    df["vol_5d"] = df["volume"].rolling(cfg.recent_volume_window).mean()
    # Prior 20-day volume excludes the most recent 5 trading days.
    df["vol_prior20d"] = df["volume"].shift(cfg.recent_volume_window).rolling(cfg.prior_volume_window).mean()
    df["volume_ratio"] = df["vol_5d"] / df["vol_prior20d"]
    df["volume_ok"] = df["volume_ratio"] >= cfg.min_volume_ratio

    df["buy_signal"] = (
        df["breakout"]
        & df["macd_bullish"]
        & df["macd_rising_ok"]
        & df["above_ma60"]
        & df["volume_ok"]
    )

    return df


def evaluate_ticker(ticker: str, cfg: ScannerConfig, period: str = "1y") -> Optional[SignalResult]:
    try:
        raw = Ticker(ticker).history(period=period, interval="1d")
        data = normalize_history(raw, ticker)
        if data.empty or len(data) < max(cfg.trend_ma_window, cfg.macd_slow + cfg.macd_signal, cfg.donchian_window) + 10:
            return None

        df = add_indicators(data, cfg)
        latest = df.dropna().iloc[-1]

        signal = "BUY" if bool(latest["buy_signal"]) else "WATCH"
        reason_parts = []
        if bool(latest["breakout"]):
            reason_parts.append(f"{cfg.donchian_window}D Donchian breakout")
        if bool(latest["macd_bullish"]):
            reason_parts.append("MACD bullish")
        if bool(latest["macd_rising_ok"]):
            reason_parts.append(f"MACD histogram rising {cfg.macd_rising_days}D")
        if bool(latest["above_ma60"]):
            reason_parts.append("above 60MA")
        if bool(latest["volume_ok"]):
            reason_parts.append("volume expansion")

        return SignalResult(
            ticker=ticker,
            signal=signal,
            close=round(float(latest["close"]), 4),
            signal_date=latest.name.strftime("%Y-%m-%d"),
            donchian_high_prior=round(float(latest["donchian_high_prior"]), 4),
            breakout_pct=round(float(latest["breakout_pct"]), 2),
            macd=round(float(latest["macd"]), 4),
            signal_line=round(float(latest["signal_line"]), 4),
            macd_hist=round(float(latest["macd_hist"]), 4),
            macd_hist_rising_days=int(latest["macd_hist_rising_count"]),
            close_vs_60ma_pct=round(float(latest["close_vs_60ma_pct"]), 2),
            volume_ratio_5d_vs_prior20d=round(float(latest["volume_ratio"]), 2),
            buy_reason="; ".join(reason_parts) if reason_parts else "conditions not met",
            management_rule=(
                f"Time stop: +{cfg.time_stop_min_return:.0f}% within {cfg.time_stop_days}D, "
                f"TP1: +{cfg.first_take_profit:.0f}% sell {cfg.first_take_profit_sell_pct:.0f}%, "
                f"TP2: +{cfg.second_take_profit_low:.0f}~{cfg.second_take_profit_high:.0f}%, "
                f"SL: {cfg.stop_loss:.0f}%, trailing: {cfg.trailing_drawdown:.0f}% from peak"
            ),
        )
    except Exception as exc:
        print(f"[WARN] {ticker}: {exc}")
        return None


def load_tickers(file_path: Optional[str], tickers: Optional[Iterable[str]]) -> List[str]:
    output: List[str] = []
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            output.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])
    if tickers:
        output.extend([t.strip() for t in tickers if t.strip()])
    return list(dict.fromkeys(output))


def scan(tickers: List[str], cfg: ScannerConfig, period: str = "1y") -> pd.DataFrame:
    results = []
    for ticker in tickers:
        result = evaluate_ticker(ticker, cfg, period=period)
        if result is not None:
            results.append(asdict(result))

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["signal_rank"] = np.where(df["signal"] == "BUY", 0, 1)
    df = df.sort_values(
        by=["signal_rank", "volume_ratio_5d_vs_prior20d", "breakout_pct", "close_vs_60ma_pct"],
        ascending=[True, False, False, False],
    ).drop(columns=["signal_rank"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Donchian + MACD + Volume momentum scanner")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols, e.g. 005930.KS 034020.KS AAPL")
    parser.add_argument("--file", help="Text file with one ticker per line")
    parser.add_argument("--period", default="1y", help="Yahooquery period, e.g. 6mo, 1y, 2y")
    parser.add_argument("--output", default="donchian_signals.csv", help="CSV output path")
    parser.add_argument("--donchian", type=int, default=20, help="Donchian breakout window")
    parser.add_argument("--min-volume-ratio", type=float, default=1.0, help="5D volume / prior 20D volume threshold")
    parser.add_argument("--show-watch", action="store_true", help="Show WATCH rows as well as BUY rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = load_tickers(args.file, args.tickers)
    if not tickers:
        raise SystemExit("No tickers supplied. Use --tickers or --file.")

    cfg = ScannerConfig(donchian_window=args.donchian, min_volume_ratio=args.min_volume_ratio)
    df = scan(tickers, cfg, period=args.period)

    if df.empty:
        print("No usable ticker data found.")
        return

    output_df = df if args.show_watch else df[df["signal"] == "BUY"]
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    if output_df.empty:
        print("No BUY candidates today. WATCH list saved only if --show-watch is used.")
    else:
        print(output_df.to_string(index=False))
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
