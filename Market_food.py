import json
import csv
import websocket
import math
import time
import os
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass
from collections import deque

BINANCE_ALL_TICKERS_URL = "wss://stream.binance.com:9443/ws/!ticker@arr"
BINANCE_REST_BASE_URL = "https://api.binance.com"
BINANCE_USER_STREAM_WS_BASE_URL = "wss://stream.binance.com:9443/ws"

BINANCE_API_KEY_INLINE = ""


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _binance_create_listen_key(api_key: str) -> str:
    req = urllib.request.Request(
        url=f"{BINANCE_REST_BASE_URL}/api/v3/userDataStream",
        method="POST",
        headers={
            "X-MBX-APIKEY": api_key,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"Binance listenKey HTTP error: {e.code} {e.reason} {body}")
    except Exception as e:
        raise RuntimeError(f"Binance listenKey error: {e}")

    listen_key = payload.get("listenKey")
    if not listen_key:
        raise RuntimeError(f"Binance listenKey missing in response: {payload}")
    return str(listen_key)


def _binance_keepalive_listen_key(api_key: str, listen_key: str, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        stop_event.wait(25 * 60)
        if stop_event.is_set():
            return

        req = urllib.request.Request(
            url=f"{BINANCE_REST_BASE_URL}/api/v3/userDataStream?listenKey={listen_key}",
            method="PUT",
            headers={
                "X-MBX-APIKEY": api_key,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15):
                pass
        except Exception as e:
            print("listenKey keepalive error:", e)

# -----------------------------
# DeltaFoldAgent (one per symbol)
# -----------------------------
@dataclass
class DeltaFoldAgent:
    symbol: str
    base: float
    delta: float = 0.0
    phase: float = 0.0
    fold: float = 0.0

    def update(self, price: float, field_dma: float):
        # local delta relative to locked base
        if self.base != 0:
            self.delta = (price - self.base) / self.base
        else:
            self.delta = 0.0

        # local phase evolution
        self.phase += self.delta + field_dma

        # fold lock (agent "answer")
        self.fold = math.sin(self.phase)

        return self.fold


# -----------------------------
# Universe = neural field
# -----------------------------
class Universe:
    def __init__(self):
        self.agents: dict[str, DeltaFoldAgent] = {}
        self.last_print = time.time()
        self._fib: list[int] = [1, 1]
        self.print_interval_s = float(os.environ.get("PRINT_INTERVAL", "2").strip() or "2")
        self.symbol_suffix = os.environ.get("SYMBOL_SUFFIX", "USDT").strip()
        self.regime_thr = float(os.environ.get("REGIME_THR", "0.12").strip() or "0.12")
        self.coil_std_delta_max = float(os.environ.get("COIL_STD_DELTA_MAX", "0.0015").strip() or "0.0015")
        self.leader_min_abs_delta = float(os.environ.get("LEADER_MIN_ABS_DELTA", "0.001").strip() or "0.001")
        self.basket_n = int(os.environ.get("BASKET_N", "5").strip() or "5")
        self.dma_hist = deque(maxlen=int(os.environ.get("DMA_HIST", "60").strip() or "60"))
        self._last_regime: str | None = None
        self._last_dma_sign: int = 0
        self.snapshot_dir = os.environ.get("SNAPSHOT_DIR", "").strip()
        self.sound_on_flip = os.environ.get("SOUND_ON_FLIP", "0").strip() == "1"
        self.entropy_high_thr = float(os.environ.get("ENTROPY_HIGH", "0.25").strip() or "0.25")
        self.entropy_low_thr = float(os.environ.get("ENTROPY_LOW", "0.05").strip() or "0.05")
        self.basket_persist_n = int(os.environ.get("BASKET_PERSIST_N", "3").strip() or "3")
        self._basket_streak_long: dict[str, int] = {}
        self._basket_streak_short: dict[str, int] = {}
        self.log_csv = os.environ.get("LOG_CSV", "1").strip() == "1"
        self.log_csv_path = os.environ.get("LOG_CSV_PATH", "field_log.csv").strip() or "field_log.csv"
        self._csv_header_written = False

    def _regime(self, dma: float, dma_slope: float, std_delta: float) -> str:
        if std_delta < 0.0005 and abs(dma_slope) < 0.002:
            return "COIL"
        if dma_slope > 0.005 and dma > 0:
            return "EXPANSION ↑"
        if dma_slope < -0.005 and dma < 0:
            return "EXPANSION ↓"
        if abs(dma_slope) < 0.002:
            return "STABLE"
        return "PIVOT"

    def _fold_entropy(self, dma: float) -> tuple[float, str]:
        agents = self.agents
        if not agents:
            return 0.0, "—"
        total = sum((a.fold - dma) ** 2 for a in agents.values())
        entropy = total / len(agents)
        if entropy <= self.entropy_low_thr:
            label = "LOW"
        elif entropy >= self.entropy_high_thr:
            label = "HIGH"
        else:
            label = "NEUTRAL"
        return entropy, label

    def _export_snapshot(self, ts: float, regime: str, dma: float, dma_slope: float, std_delta: float, entropy: float, entropy_label: str, long_syms: list[str], short_syms: list[str]):
        if not self.snapshot_dir:
            return
        row = {
            "ts": ts,
            "regime": regime,
            "dma": dma,
            "dma_slope": dma_slope,
            "std_delta": std_delta,
            "entropy": entropy,
            "entropy_label": entropy_label,
            "long": long_syms,
            "short": short_syms,
        }
        path = os.path.join(self.snapshot_dir, "field_snapshots.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            print(f"[snapshot write error] {e}")

    def _update_streaks(self, current: list[str], streaks: dict[str, int]) -> None:
        cur = set(current)
        for s in list(streaks.keys()):
            if s not in cur:
                del streaks[s]
        for s in current:
            streaks[s] = streaks.get(s, 0) + 1

    def _persist_basket(self, current: list[str], streaks: dict[str, int]) -> list[str]:
        self._update_streaks(current, streaks)
        n = self.basket_persist_n
        if n <= 1:
            return list(current)
        return [s for s in current if streaks.get(s, 0) >= n]

    def _log_csv_row(self, row: dict) -> None:
        if not self.log_csv:
            return
        try:
            write_header = False
            if not self._csv_header_written:
                try:
                    write_header = (not os.path.exists(self.log_csv_path)) or (os.path.getsize(self.log_csv_path) == 0)
                except Exception:
                    write_header = True

            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            self._csv_header_written = True
        except Exception as e:
            print(f"[csv log error] {e}")

    def _leaders_and_baskets(
        self,
        updates: list[tuple[DeltaFoldAgent, float, float]],
        dma: float,
    ) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]], list[str], list[str]]:
        aligned: list[tuple[str, float, float]] = []
        divergent: list[tuple[str, float, float]] = []
        want_suffix = self.symbol_suffix
        min_abs = self.leader_min_abs_delta

        if dma == 0.0:
            return [], [], [], []

        for agent, _price, d in updates:
            sym = agent.symbol
            if want_suffix and not sym.endswith(want_suffix):
                continue
            ad = abs(d)
            if ad < min_abs:
                continue
            same = (d >= 0.0 and dma >= 0.0) or (d <= 0.0 and dma <= 0.0)
            item = (sym, d, agent.fold)
            if same:
                aligned.append(item)
            else:
                divergent.append(item)

        aligned.sort(key=lambda x: abs(x[1]), reverse=True)
        divergent.sort(key=lambda x: abs(x[1]), reverse=True)

        long_syms: list[str] = []
        short_syms: list[str] = []
        if dma >= 0.0:
            long_syms = [s for (s, d, _f) in aligned if d > 0.0][: self.basket_n]
            short_syms = [s for (s, d, _f) in divergent if d < 0.0][: self.basket_n]
        else:
            short_syms = [s for (s, d, _f) in aligned if d < 0.0][: self.basket_n]
            long_syms = [s for (s, d, _f) in divergent if d > 0.0][: self.basket_n]

        return aligned[: self.basket_n], divergent[: self.basket_n], long_syms, short_syms

    def _ensure_fib(self, n: int) -> None:
        if n <= len(self._fib):
            return
        while len(self._fib) < n:
            self._fib.append(self._fib[-1] + self._fib[-2])

    def _compute_dma(self, F: list[float]) -> float:
        mode = os.environ.get("DMA_FOLD_MODE", "half").strip().lower()
        n = len(F)
        if n == 0:
            return 0.0

        if mode in ("inward", "in", "fib", "fibonacci"):
            pairs = n // 2
            if pairs <= 0:
                return math.sin(F[0])

            self._ensure_fib(pairs)
            weighted = 0.0
            wsum = 0.0
            for i in range(pairs):
                w = float(self._fib[i])
                weighted += (F[i] - F[-1 - i]) * w
                wsum += w
            if wsum == 0:
                wsum = 1.0
            return math.sin(weighted / wsum)

        mid = n // 2
        pos = sum(F[:mid])
        neg = sum(F[mid:])
        return math.sin(pos - neg)

    def ingest_market(self, tickers: list[dict]):
        agents = self.agents
        sin = math.sin

        updates: list[tuple[DeltaFoldAgent, float, float]] = []
        F: list[float] = []
        sum_d = 0.0
        sum_d2 = 0.0

        # 1) parse once, ensure agents exist, compute deltas once
        for t in tickers:
            sym = t.get("s")
            price_s = t.get("c")
            if not sym or price_s is None:
                continue

            try:
                price = float(price_s)
            except Exception:
                continue

            if price <= 0:
                continue

            agent = agents.get(sym)
            if agent is None:
                agent = DeltaFoldAgent(symbol=sym, base=price)
                agents[sym] = agent

            base = agent.base
            if base:
                d = (price - base) / base
            else:
                d = 0.0

            updates.append((agent, price, d))
            F.append(sin(d))
            sum_d += d
            sum_d2 += d * d

        # 2) compute global DMA field
        if not F:
            return

        dma = self._compute_dma(F)

        # 3) update every agent with field pressure (inline for speed)
        for agent, price, d in updates:
            agent.delta = d
            agent.phase += d + dma
            agent.fold = sin(agent.phase)

        # 4) optional visibility (not required for operation)
        now = time.time()
        if now - self.last_print > self.print_interval_s:
            self.last_print = now
            n = len(updates)
            if n <= 0:
                return
            mean_delta = sum_d / n
            var = (sum_d2 / n) - (mean_delta * mean_delta)
            if var < 0.0:
                var = 0.0
            std_delta = math.sqrt(var)
            self.print_state(dma, mean_delta, std_delta, updates)

    def print_state(self, dma: float, mean_delta: float, std_delta: float, updates: list[tuple[DeltaFoldAgent, float, float]]):
        # show a tiny slice so we don't spam
        sample = list(self.agents.values())[:5]

        self.dma_hist.append(dma)
        prev_dma = self.dma_hist[-2] if len(self.dma_hist) >= 2 else 0.0
        dma_slope = dma - prev_dma
        regime = self._regime(dma, dma_slope, std_delta)
        reversal = (self._last_regime is not None) and (regime != self._last_regime)
        self._last_regime = regime

        cur_sign = 1 if dma >= 0 else -1
        field_flip = (self._last_dma_sign != 0) and (cur_sign != self._last_dma_sign)
        self._last_dma_sign = cur_sign

        entropy, entropy_label = self._fold_entropy(dma)

        aligned, divergent, long_syms, short_syms = self._leaders_and_baskets(updates, dma)

        long_persist = self._persist_basket(long_syms, self._basket_streak_long)
        short_persist = self._persist_basket(short_syms, self._basket_streak_short)

        div_top = " ".join(f"{s}({d:+.3%})" for (s, d, _f) in divergent[:3])

        if self.sound_on_flip and (field_flip or reversal):
            print("\a", end="", flush=True)

        now_ts = time.time()
        self._export_snapshot(now_ts, regime, dma, dma_slope, std_delta, entropy, entropy_label, long_persist, short_persist)
        self._log_csv_row(
            {
                "ts": f"{now_ts:.3f}",
                "agents": str(len(self.agents)),
                "regime": regime,
                "dma": f"{dma:+.6f}",
                "ddma": f"{dma_slope:+.6f}",
                "sigma_delta": f"{std_delta:.6f}",
                "entropy": f"{entropy:.6f}",
                "entropy_label": entropy_label,
                "long": " ".join(long_persist),
                "short": " ".join(short_persist),
                "div_top3": div_top,
            }
        )

        print("\n=== LIVE FIELD SNAPSHOT ===")
        print(f"Agents: {len(self.agents)}")
        line = (
            f"REGIME: {regime}"
            f"  DMA={dma:+.6f}"
            f"  dDMA={dma_slope:+.6f}"
            f"  σΔ={std_delta:.6f}"
            f"  Entropy={entropy:.5f} [{entropy_label}]"
        )
        if reversal:
            line += "  REVERSAL"
        if field_flip:
            line += "  FLIP"
        print(line)

        if divergent:
            print(
                "DIVERGENCE: "
                + ", ".join(f"{s}({d:+.3%})" for (s, d, _f) in divergent)
            )
        if aligned:
            print(
                "ALIGNED:    "
                + ", ".join(f"{s}({d:+.3%})" for (s, d, _f) in aligned)
            )

        if long_persist or short_persist:
            print(
                f"BASKET: LONG[{len(long_persist)}]="
                + " ".join(long_persist)
                + f"  |  SHORT[{len(short_persist)}]="
                + " ".join(short_persist)
            )

        for a in sample:
            print(
                f"{a.symbol:10s} "
                f"Δ={a.delta:+.6f} "
                f"fold={a.fold:+.6f}"
            )


universe = Universe()

# -----------------------------
# WebSocket handlers
# -----------------------------
def on_message(ws, message):
    data = json.loads(message)
    if not isinstance(data, list):
        return

    universe.ingest_market(data)

def on_open(ws):
    print("[CONNECTED — LIVE AGENT FIELD RUNNING]")

def on_error(ws, error):
    if isinstance(error, KeyboardInterrupt):
        return
    print("WS error:", repr(error))

def on_close(ws, close_status_code=None, close_msg=None):
    print("WS closed", close_status_code, close_msg)

def run():
    backoff = 1.0
    while True:
        ws = websocket.WebSocketApp(
            BINANCE_ALL_TICKERS_URL,
            on_message=on_message,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close,
        )

        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except KeyboardInterrupt:
            try:
                ws.close()
            except Exception:
                pass
            break
        except Exception as e:
            print("WS run_forever error:", repr(e))

        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            break
        backoff = min(backoff * 2.0, 30.0)


def _on_user_stream_open(ws):
    print("[CONNECTED — BINANCE USER DATA STREAM]")


def _on_user_stream_message(ws, message: str):
    try:
        payload = json.loads(message)
    except Exception:
        print(message)
        return

    print("USER_STREAM:", payload)


def _on_user_stream_error(ws, error):
    if isinstance(error, KeyboardInterrupt):
        return
    print("USER_STREAM WS error:", repr(error))


def _on_user_stream_close(ws, *args):
    close_status_code = None
    close_msg = None
    if len(args) >= 1:
        close_status_code = args[0]
    if len(args) >= 2:
        close_msg = args[1]
    print("USER_STREAM WS closed", close_status_code, close_msg)


def run_user_stream():
    api_key = (os.environ.get("BINANCE_API_KEY") or BINANCE_API_KEY_INLINE).strip()
    if not api_key:
        raise RuntimeError("Missing Binance API key. Set BINANCE_API_KEY env var or BINANCE_API_KEY_INLINE.")
    backoff = 1.0
    while True:
        listen_key = _binance_create_listen_key(api_key)
        ws_url = f"{BINANCE_USER_STREAM_WS_BASE_URL}/{listen_key}"

        stop_event = threading.Event()
        keepalive_thread = threading.Thread(
            target=_binance_keepalive_listen_key,
            args=(api_key, listen_key, stop_event),
            daemon=True,
        )
        keepalive_thread.start()

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=_on_user_stream_message,
            on_open=_on_user_stream_open,
            on_error=_on_user_stream_error,
            on_close=_on_user_stream_close,
        )

        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except KeyboardInterrupt:
            try:
                ws.close()
            except Exception:
                pass
            stop_event.set()
            break
        except Exception as e:
            print("USER_STREAM run_forever error:", repr(e))
        finally:
            stop_event.set()

        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            break
        backoff = min(backoff * 2.0, 30.0)


if __name__ == "__main__":
    if os.environ.get("BINANCE_USER_STREAM", "").strip() == "1":
        threading.Thread(target=run_user_stream, daemon=True).start()
    run()
