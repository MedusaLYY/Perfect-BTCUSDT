import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from dateutil import parser as dt_parser
from tqdm import tqdm


BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def parse_time_to_ms(value: str) -> int:
    """
    Accepts:
    - 2026-01-01
    - 2026-01-01 00:00:00
    - 2026-01-01T00:00:00Z
    Returns UTC timestamp in milliseconds.
    """
    dt = dt_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def request_json(
    session: requests.Session,
    base_url: str,
    path: str,
    params: Dict[str, Any],
    max_retries: int = 8,
    sleep_base: float = 1.0,
) -> Any:
    url = base_url + path
    clean_params = {k: v for k, v in params.items() if v is not None}

    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=clean_params, timeout=20)

            if resp.status_code in (418, 429):
                wait = sleep_base * (2 ** attempt)
                print(f"[RATE_LIMIT] {resp.status_code} sleeping {wait:.1f}s: {url}")
                time.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                wait = sleep_base * (2 ** attempt)
                print(f"[SERVER_ERROR] {resp.status_code} sleeping {wait:.1f}s: {url}")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            wait = sleep_base * (2 ** attempt)
            print(f"[REQUEST_ERROR] {e} sleeping {wait:.1f}s: {url}")
            time.sleep(wait)

    raise RuntimeError(f"Request failed after retries: {url} params={clean_params}")


def append_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    if not rows:
        return

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def normalize_kline_row(symbol: str, interval: str, row: List[Any]) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "interval": interval,
        "open_time": int(row[0]),
        "open_time_utc": ms_to_utc(int(row[0])),
        "open": row[1],
        "high": row[2],
        "low": row[3],
        "close": row[4],
        "volume": row[5],
        "close_time": int(row[6]),
        "close_time_utc": ms_to_utc(int(row[6])),
        "quote_volume": row[7],
        "number_of_trades": row[8],
        "taker_buy_base_volume": row[9],
        "taker_buy_quote_volume": row[10],
    }


KLINE_FIELDS = [
    "symbol",
    "interval",
    "open_time",
    "open_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "close_time_utc",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def collect_klines(
    session: requests.Session,
    base_url: str,
    path: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    out_path: Path,
    label: str,
    sleep_sec: float,
) -> None:
    """
    Works for:
    - Spot:    GET /api/v3/klines
    - Futures: GET /fapi/v1/klines
    """
    ensure_dir(out_path.parent)

    step_ms = INTERVAL_MS[interval]
    current = start_ms

    pbar = tqdm(total=max(1, end_ms - start_ms), desc=label, unit="ms")

    while current < end_ms:
        data = request_json(
            session=session,
            base_url=base_url,
            path=path,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

        if not data:
            current += 1000 * step_ms
            pbar.update(min(1000 * step_ms, end_ms - current))
            time.sleep(sleep_sec)
            continue

        rows = [normalize_kline_row(symbol, interval, row) for row in data]
        append_rows_csv(out_path, rows, KLINE_FIELDS)

        last_open_time = int(data[-1][0])
        next_current = last_open_time + step_ms

        if next_current <= current:
            next_current = current + step_ms

        pbar.update(min(next_current - current, end_ms - current))
        current = next_current
        time.sleep(sleep_sec)

    pbar.close()


AGG_TRADE_FIELDS = [
    "symbol",
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "trade_time",
    "trade_time_utc",
    "is_buyer_maker",
    "is_best_match",
]


def normalize_agg_trade_row(symbol: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "agg_trade_id": row.get("a"),
        "price": row.get("p"),
        "quantity": row.get("q"),
        "first_trade_id": row.get("f"),
        "last_trade_id": row.get("l"),
        "trade_time": row.get("T"),
        "trade_time_utc": ms_to_utc(int(row.get("T"))),
        "is_buyer_maker": row.get("m"),
        "is_best_match": row.get("M"),
    }


def collect_spot_agg_trades(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    out_path: Path,
    sleep_sec: float,
) -> None:
    """
    Spot aggregate trades. This can become very large.
    For long historical ranges, Binance public data files are usually better.
    """
    ensure_dir(out_path.parent)
    current = start_ms

    pbar = tqdm(total=max(1, end_ms - start_ms), desc="spot_agg_trades", unit="ms")

    while current < end_ms:
        data = request_json(
            session=session,
            base_url=BINANCE_SPOT_BASE,
            path="/api/v3/aggTrades",
            params={
                "symbol": symbol,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

        if not data:
            # BTCUSDT usually has no empty minutes, but this keeps the loop safe.
            next_current = min(current + 60_000, end_ms)
            pbar.update(next_current - current)
            current = next_current
            time.sleep(sleep_sec)
            continue

        rows = [normalize_agg_trade_row(symbol, row) for row in data]
        append_rows_csv(out_path, rows, AGG_TRADE_FIELDS)

        last_t = int(data[-1]["T"])
        next_current = last_t + 1

        if next_current <= current:
            next_current = current + 1

        pbar.update(min(next_current - current, end_ms - current))
        current = next_current
        time.sleep(sleep_sec)

    pbar.close()


FUNDING_FIELDS = [
    "symbol",
    "funding_time",
    "funding_time_utc",
    "funding_rate",
    "mark_price",
]


def collect_funding_rate(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    out_path: Path,
    sleep_sec: float,
) -> None:
    ensure_dir(out_path.parent)
    current = start_ms

    pbar = tqdm(total=max(1, end_ms - start_ms), desc="funding_rate", unit="ms")

    while current < end_ms:
        data = request_json(
            session=session,
            base_url=BINANCE_FUTURES_BASE,
            path="/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            },
        )

        if not data:
            break

        rows = []
        for row in data:
            funding_time = int(row["fundingTime"])
            rows.append(
                {
                    "symbol": row.get("symbol", symbol),
                    "funding_time": funding_time,
                    "funding_time_utc": ms_to_utc(funding_time),
                    "funding_rate": row.get("fundingRate"),
                    "mark_price": row.get("markPrice"),
                }
            )

        append_rows_csv(out_path, rows, FUNDING_FIELDS)

        last_t = int(data[-1]["fundingTime"])
        next_current = last_t + 1

        if next_current <= current:
            break

        pbar.update(min(next_current - current, end_ms - current))
        current = next_current
        time.sleep(sleep_sec)

    pbar.close()


def generic_futures_data_collector(
    session: requests.Session,
    endpoint: str,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    out_path: Path,
    label: str,
    sleep_sec: float,
    limit: int = 500,
) -> None:
    """
    For endpoints under /futures/data/*:
    - openInterestHist
    - takerlongshortRatio
    - globalLongShortAccountRatio
    - topLongShortAccountRatio
    - topLongShortPositionRatio
    """
    ensure_dir(out_path.parent)
    current = start_ms

    pbar = tqdm(total=max(1, end_ms - start_ms), desc=label, unit="ms")
    header_written = False
    fieldnames: Optional[List[str]] = None

    while current < end_ms:
        data = request_json(
            session=session,
            base_url=BINANCE_FUTURES_BASE,
            path=endpoint,
            params={
                "symbol": symbol,
                "period": period,
                "startTime": current,
                "endTime": end_ms,
                "limit": limit,
            },
        )

        if not data:
            # Some Binance futures trading-data endpoints only expose recent history.
            # Stop rather than pretending older data exists.
            print(f"[EMPTY] {label}: no more data from {ms_to_utc(current)}")
            break

        rows = []
        for row in data:
            normalized = dict(row)
            normalized["symbol"] = normalized.get("symbol", symbol)

            if "timestamp" in normalized:
                normalized["timestamp"] = int(normalized["timestamp"])
                normalized["timestamp_utc"] = ms_to_utc(normalized["timestamp"])

            rows.append(normalized)

        if rows:
            if fieldnames is None:
                keys = set()
                for row in rows:
                    keys.update(row.keys())
                preferred = ["symbol", "timestamp", "timestamp_utc"]
                fieldnames = preferred + sorted([k for k in keys if k not in preferred])

            append_rows_csv(out_path, rows, fieldnames)

        timestamps = [int(row["timestamp"]) for row in rows if "timestamp" in row]
        if not timestamps:
            break

        last_t = max(timestamps)
        next_current = last_t + 1

        if next_current <= current:
            break

        pbar.update(min(next_current - current, end_ms - current))
        current = next_current
        time.sleep(sleep_sec)

    pbar.close()


def collect_orderbook_snapshots(
    session: requests.Session,
    symbol: str,
    out_path: Path,
    depth_limit: int,
    top_n: int,
    interval_sec: float,
    duration_sec: int,
) -> None:
    """
    Collects current order book snapshots periodically.
    This does NOT backfill historical order book.
    """
    ensure_dir(out_path.parent)

    fields = ["symbol", "fetch_time", "fetch_time_utc", "last_update_id"]
    for i in range(1, top_n + 1):
        fields += [f"bid_price_{i}", f"bid_qty_{i}", f"ask_price_{i}", f"ask_qty_{i}"]

    end_time = time.time() + duration_sec
    count = 0

    pbar = tqdm(total=duration_sec, desc="live_orderbook", unit="s")
    last_tick = time.time()

    while time.time() < end_time:
        fetch_time = int(time.time() * 1000)

        data = request_json(
            session=session,
            base_url=BINANCE_SPOT_BASE,
            path="/api/v3/depth",
            params={
                "symbol": symbol,
                "limit": depth_limit,
            },
        )

        row = {
            "symbol": symbol,
            "fetch_time": fetch_time,
            "fetch_time_utc": ms_to_utc(fetch_time),
            "last_update_id": data.get("lastUpdateId"),
        }

        bids = data.get("bids", [])[:top_n]
        asks = data.get("asks", [])[:top_n]

        for i in range(top_n):
            idx = i + 1

            if i < len(bids):
                row[f"bid_price_{idx}"] = bids[i][0]
                row[f"bid_qty_{idx}"] = bids[i][1]
            else:
                row[f"bid_price_{idx}"] = None
                row[f"bid_qty_{idx}"] = None

            if i < len(asks):
                row[f"ask_price_{idx}"] = asks[i][0]
                row[f"ask_qty_{idx}"] = asks[i][1]
            else:
                row[f"ask_price_{idx}"] = None
                row[f"ask_qty_{idx}"] = None

        append_rows_csv(out_path, [row], fields)

        count += 1
        now = time.time()
        pbar.update(min(duration_sec, int(now - last_tick)))
        last_tick = now

        time.sleep(interval_sec)

    pbar.close()
    print(f"[DONE] collected {count} orderbook snapshots -> {out_path}")


def deduplicate_csv_by_time(path: Path, time_col: str) -> None:
    if not path.exists():
        return

    df = pd.read_csv(path)
    if time_col not in df.columns:
        return

    before = len(df)
    df = df.drop_duplicates(subset=[time_col]).sort_values(time_col)
    after = len(df)
    df.to_csv(path, index=False)
    print(f"[DEDUP] {path.name}: {before} -> {after}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Binance market data for BTCUSDT 1m / 10m probability model."
    )

    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="UTC start time, e.g. 2026-01-01")
    parser.add_argument("--end", required=True, help="UTC end time, e.g. 2026-07-01")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--futures-period", default="5m")
    parser.add_argument("--out", default="data/raw/binance")
    parser.add_argument("--sleep", type=float, default=0.15)

    parser.add_argument("--skip-spot-klines", action="store_true")
    parser.add_argument("--skip-spot-agg-trades", action="store_true")
    parser.add_argument("--skip-futures-klines", action="store_true")
    parser.add_argument("--skip-futures-metrics", action="store_true")

    parser.add_argument(
        "--orderbook-duration-sec",
        type=int,
        default=0,
        help="If > 0, collect live spot orderbook snapshots for this many seconds.",
    )
    parser.add_argument("--orderbook-interval-sec", type=float, default=5.0)
    parser.add_argument("--orderbook-depth-limit", type=int, default=100)
    parser.add_argument("--orderbook-top-n", type=int, default=20)

    args = parser.parse_args()

    symbol = args.symbol.upper()
    start_ms = parse_time_to_ms(args.start)
    end_ms = parse_time_to_ms(args.end)

    if end_ms <= start_ms:
        raise ValueError("--end must be later than --start")

    if args.interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {args.interval}")

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "market-data-collector/1.0",
            "Accept": "application/json",
        }
    )

    print(f"[CONFIG] symbol={symbol}")
    print(f"[CONFIG] start={ms_to_utc(start_ms)}")
    print(f"[CONFIG] end={ms_to_utc(end_ms)}")
    print(f"[CONFIG] out={out_dir.resolve()}")

    if not args.skip_spot_klines:
        collect_klines(
            session=session,
            base_url=BINANCE_SPOT_BASE,
            path="/api/v3/klines",
            symbol=symbol,
            interval=args.interval,
            start_ms=start_ms,
            end_ms=end_ms,
            out_path=out_dir / f"{symbol}_spot_klines_{args.interval}.csv",
            label=f"spot_klines_{args.interval}",
            sleep_sec=args.sleep,
        )
        deduplicate_csv_by_time(out_dir / f"{symbol}_spot_klines_{args.interval}.csv", "open_time")

    if not args.skip_spot_agg_trades:
        collect_spot_agg_trades(
            session=session,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            out_path=out_dir / f"{symbol}_spot_agg_trades.csv",
            sleep_sec=args.sleep,
        )
        deduplicate_csv_by_time(out_dir / f"{symbol}_spot_agg_trades.csv", "agg_trade_id")

    if not args.skip_futures_klines:
        collect_klines(
            session=session,
            base_url=BINANCE_FUTURES_BASE,
            path="/fapi/v1/klines",
            symbol=symbol,
            interval=args.interval,
            start_ms=start_ms,
            end_ms=end_ms,
            out_path=out_dir / f"{symbol}_futures_klines_{args.interval}.csv",
            label=f"futures_klines_{args.interval}",
            sleep_sec=args.sleep,
        )
        deduplicate_csv_by_time(out_dir / f"{symbol}_futures_klines_{args.interval}.csv", "open_time")

    if not args.skip_futures_metrics:
        collect_funding_rate(
            session=session,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            out_path=out_dir / f"{symbol}_futures_funding_rate.csv",
            sleep_sec=args.sleep,
        )

        futures_jobs = [
            {
                "endpoint": "/futures/data/openInterestHist",
                "name": "futures_open_interest_hist",
            },
            {
                "endpoint": "/futures/data/takerlongshortRatio",
                "name": "futures_taker_buy_sell_volume",
            },
            {
                "endpoint": "/futures/data/globalLongShortAccountRatio",
                "name": "futures_global_long_short_ratio",
            },
            {
                "endpoint": "/futures/data/topLongShortAccountRatio",
                "name": "futures_top_trader_account_ratio",
            },
            {
                "endpoint": "/futures/data/topLongShortPositionRatio",
                "name": "futures_top_trader_position_ratio",
            },
        ]

        for job in futures_jobs:
            generic_futures_data_collector(
                session=session,
                endpoint=job["endpoint"],
                symbol=symbol,
                period=args.futures_period,
                start_ms=start_ms,
                end_ms=end_ms,
                out_path=out_dir / f"{symbol}_{job['name']}_{args.futures_period}.csv",
                label=job["name"],
                sleep_sec=args.sleep,
                limit=500,
            )

    if args.orderbook_duration_sec > 0:
        collect_orderbook_snapshots(
            session=session,
            symbol=symbol,
            out_path=out_dir / f"{symbol}_spot_orderbook_snapshots.csv",
            depth_limit=args.orderbook_depth_limit,
            top_n=args.orderbook_top_n,
            interval_sec=args.orderbook_interval_sec,
            duration_sec=args.orderbook_duration_sec,
        )

    print("[DONE] all requested data collection tasks finished.")


if __name__ == "__main__":
    main()