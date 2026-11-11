#!/usr/bin/env python3
"""Roblox Up-and-Coming forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json (else /input) and writes
$MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json).

Method (v2, tuned on split backtests of the practice history):
  The chart's order is only loosely tied to player-count levels (in the
  practice history the Spearman correlation of rank with playing is only
  about -0.5, and one top-10 game held the chart's highest player count),
  while a game's current rank is highly persistent (consecutive-read rank
  correlation about 0.96). A pure "project playing, rank by playing"
  model (v1) therefore lost to the persistence baseline on one of two
  backtest splits. The forecaster below keeps the Monte Carlo skeleton
  but ranks simulations by a blended latent score:

      score(game, h) = LEVEL_W * (logP_last + shrunken drift(h))
                       + RANK_W * (-log(last_rank))
                       + sigma * sqrt(h) * gauss

  * Per game, a recent-weighted linear trend is fitted on log(playing);
    the drift is shrunk toward zero with horizon (mean reversion):
    the cumulative drift at horizon h is slope * TAU * (1 - exp(-h/TAU)).
  * -log(last_rank) is the persistence prior: it is 0 at rank 1 and
    grows more negative for lower ranks, in units comparable to the
    log-playing spread. RANK_W > LEVEL_W because rank persistence is
    the stronger signal in this chart; both weights, the smoothing
    mass and TAU were chosen on the two backtest splits (see below).
  * Age-out cliff: a game leaves the chart 28 days after its
    details.created timestamp; if that falls inside the scored window
    the game contributes zero occupancy afterwards. Missing `created`
    (the practice archive carries no details) means alive all window.
  * Monte Carlo tallies place occupancy (places 1..10) over window
    hours x simulations. Because one forecast is scored against all 72
    hourly reads, the time-averaged occupancy frequency is the target.
  * Frequencies are blended with a uniform smoothing mass over all
    candidates (SMOOTH_EPS = 0.15: heavy smoothing won on backtests --
    this chart churns hard over 72 h, and the log score punishes
    near-zero probabilities on games that do show up), trimmed to
    <=100 rows per place and rescaled so each place sums to at most
    0.98 (the rest stays unnamed, which the grader spreads over the
    read's chart).
  * top10 orders games by expected placement score
    sum_place P(game, place) * (11 - place), tie-broken by P(top 10)
    then universe_id, so output is fully deterministic.

  Backtests (fit on a prefix of the 34 practice reads, forecast the
  next 72 h, scored with the RULES log score against the later reads
  in the file; house baseline computed on the same splits):
    split A (fit 24 reads): baseline -56.911344, v1 -55.1203, v2 -50.644574
    split B (fit 28 reads): baseline -37.712654, v1 -43.4918, v2 -30.568980
  (v2 numbers are this exact configuration's, re-measured after the
  final constants were frozen; small wiggles of the weights move
  them by a few tenths.)

Standard library only, no network, seeded RNG: two runs on the same
history are byte-identical.
"""

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

# --- Tuned constants (see docstring / backtests) ------------------------------
SEED = 20261113
N_SIMS = 300               # Monte Carlo simulations per hour
TAU = 60.0                 # drift mean-reversion horizon (hours)
LEVEL_W = 0.35             # weight on projected log(playing)
RANK_W = 3.0               # weight on -log(last rank) persistence prior
RECENT_HOURS = 60.0        # candidate pool: chart reads within this span...
RECENT_READS_MIN = 10      # ...or at least this many latest reads
MAX_BEST_RANK = 55         # candidate if best recent rank <= this
LAST_READ_RANK = 45        # ... or ranked <= this in the very last read
TREND_POINTS = 20          # max observations used per trend fit
TREND_HALFLIFE_H = 30.0    # exponential weight half-life for the fit (hours)
MIN_PLAYING = 1.0
SMOOTH_EPS = 0.15          # uniform smoothing mass per place
PLACE_SUM_CAP = 0.98       # rescale each place's named sum to at most this
MIN_ROW_P = 0.001          # drop probability rows below this
SIGMA_FLOOR = 0.08
SIGMA_CAP = 0.85
DEFAULT_SIGMA = 0.35
AGE_LIMIT_DAYS = 28.0


def parse_ts(s):
    """Parse an ISO-8601 timestamp (with Z) to an aware datetime."""
    if s is None:
        return None
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def weighted_trend(ts, ys, halflife_h):
    """Weighted least squares y ~ a + b*t; returns (b, residual_sigma).

    ts are hours (floats), ys are log(playing). Weights decay with age
    relative to the newest point. Falls back to (0, None) when the fit
    is not identifiable.
    """
    n = len(ts)
    if n < 2:
        return 0.0, None
    t_new = ts[-1]
    ws = [0.5 ** ((t_new - t) / halflife_h) if halflife_h > 0 else 1.0 for t in ts]
    sw = sum(ws)
    mt = sum(w * t for w, t in zip(ws, ts)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (t - mt) ** 2 for w, t in zip(ws, ts))
    if sxx <= 1e-9:
        return 0.0, None
    sxy = sum(w * (t - mt) * (y - my) for w, t, y in zip(ws, ts, ys))
    b = sxy / sxx
    a = my - b * mt
    resid = [y - (a + b * t) for t, y in zip(ts, ys)]
    if n > 2:
        var = sum(r * r for r in resid) / (n - 2)
    else:
        var = sum(r * r for r in resid) / n
    return b, math.sqrt(max(var, 0.0))


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    window = task.get("window", {}) or {}
    win_from = parse_ts(window.get("from")) or parse_ts(task.get("history_until"))

    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    if not reads:
        raise SystemExit("empty history")

    read_times = [parse_ts(r.get("at")) for r in reads]
    t_last = read_times[-1]
    if win_from is None:
        win_from = t_last
    horizon0_h = (win_from - t_last).total_seconds() / 3600.0  # gap to window start
    n_slots = int(window.get("slots", 72))
    slot_minutes = float(window.get("slot_minutes", 60))
    # Hour offsets (from last read) for the middle of each scored slot.
    slot_hours = [horizon0_h + (i + 0.5) * (slot_minutes / 60.0) for i in range(n_slots)]

    # A read counts as "recent" for the candidate pool if it is within
    # RECENT_HOURS of the last read or among the RECENT_READS_MIN latest
    # reads (the final history is hourly; the practice archive is not).
    recent_flags = []
    for idx, rt in enumerate(read_times):
        age_h = (t_last - rt).total_seconds() / 3600.0
        recent_flags.append(age_h <= RECENT_HOURS or idx >= len(reads) - RECENT_READS_MIN)

    # ---- Collect per-game observations -------------------------------------
    t0 = read_times[0]
    series = {}
    best_rank_recent = {}
    last_rank = {}
    created_by_uid = {}
    on_chart_recent = set()
    aux_boost = set()  # on trending / playing-now in the last read
    for idx, read in enumerate(reads):
        chart = read.get("up_and_coming") or []
        rt = read_times[idx]
        hours = (rt - t0).total_seconds() / 3600.0
        for row in chart:
            uid = row.get("universe_id")
            if uid is None:
                continue
            uid = int(uid)
            playing = row.get("playing")
            if playing is not None:
                series.setdefault(uid, []).append((hours, math.log(max(float(playing), MIN_PLAYING))))
            if recent_flags[idx]:
                on_chart_recent.add(uid)
                rk = row.get("rank")
                if rk is not None:
                    rk = int(rk)
                    if uid not in best_rank_recent or rk < best_rank_recent[uid]:
                        best_rank_recent[uid] = rk
            if idx == len(reads) - 1:
                rk = row.get("rank")
                if rk is not None:
                    last_rank[uid] = int(rk)
        details = read.get("details")
        if isinstance(details, dict):
            for key, det in details.items():
                if not isinstance(det, dict):
                    continue
                created = parse_ts(det.get("created"))
                if created is not None:
                    try:
                        created_by_uid[int(key)] = created
                    except (TypeError, ValueError):
                        pass
    last_read = reads[-1]
    for aux_key in ("top_trending", "top_playing_now"):
        aux = last_read.get(aux_key) or []
        for row in aux[:40]:
            uid = row.get("universe_id")
            if uid is not None:
                aux_boost.add(int(uid))

    # ---- Candidate selection -------------------------------------------------
    candidates = []
    for uid in on_chart_recent:
        br = best_rank_recent.get(uid, 999)
        lr = last_rank.get(uid, 999)
        if br <= MAX_BEST_RANK or lr <= LAST_READ_RANK or uid in aux_boost:
            if uid in series and series[uid]:
                candidates.append(uid)
    candidates.sort()
    if not candidates:  # degenerate fallback: last chart top 10
        candidates = sorted(last_rank, key=lambda u: last_rank[u])[:10]

    last_hours = (t_last - t0).total_seconds() / 3600.0

    # ---- Per-game trend, sigma, age-out --------------------------------------
    band_resid = {0: [], 1: [], 2: []}

    def band_of(uid):
        br = best_rank_recent.get(uid, 999)
        if br <= 10:
            return 0
        if br <= 30:
            return 1
        return 2

    game_info = {}
    raw_fits = {}
    for uid in candidates:
        pts = series[uid][-TREND_POINTS:]
        ts = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        slope, sigma = weighted_trend(ts, ys, TREND_HALFLIFE_H)
        raw_fits[uid] = (slope, sigma, ys[-1])
        if sigma is not None:
            band_resid[band_of(uid)].append(sigma)
    band_sigma = {}
    for band, vals in band_resid.items():
        if vals:
            vals_sorted = sorted(vals)
            band_sigma[band] = vals_sorted[len(vals_sorted) // 2]
    all_sigmas = sorted(s for vals in band_resid.values() for s in vals)
    global_sigma = all_sigmas[len(all_sigmas) // 2] if all_sigmas else DEFAULT_SIGMA

    for uid in candidates:
        slope, sigma, logp_last = raw_fits[uid]
        if sigma is None or sigma <= 0:
            sigma = band_sigma.get(band_of(uid), global_sigma)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        expiry_h = None  # hours from t0 when the game ages out
        created = created_by_uid.get(uid)
        if created is not None:
            expiry = created + timedelta(days=AGE_LIMIT_DAYS)
            expiry_h = (expiry - t0).total_seconds() / 3600.0
        # Persistence prior from the last read's rank; games absent from
        # the last read fall back to their best recent rank (weaker prior).
        prior_rank = last_rank.get(uid, best_rank_recent.get(uid, 60))
        game_info[uid] = {
            "slope": slope,
            "sigma": sigma,
            "logp_last": logp_last,
            "expiry_h": expiry_h,
            "rank_prior": -math.log(max(int(prior_rank), 1)),
        }

    slot_abs_hours = [last_hours + h for h in slot_hours]  # hours since t0

    # ---- Monte Carlo -----------------------------------------------------------
    rng = random.Random(SEED)
    counts = {uid: [0] * places for uid in candidates}
    total_draws = N_SIMS * n_slots
    drift = {}
    noise = {}
    base = {}
    alive = {}
    for uid in candidates:
        info = game_info[uid]
        drift[uid] = [info["slope"] * TAU * (1.0 - math.exp(-h / TAU)) for h in slot_hours]
        noise[uid] = [info["sigma"] * math.sqrt(max(h, 0.01)) for h in slot_hours]
        base[uid] = LEVEL_W * info["logp_last"] + RANK_W * info["rank_prior"]
        if info["expiry_h"] is None:
            alive[uid] = [True] * n_slots
        else:
            alive[uid] = [slot_abs_hours[i] < info["expiry_h"] for i in range(n_slots)]

    gauss = rng.gauss
    for _ in range(N_SIMS):
        for si in range(n_slots):
            scored = []
            for uid in candidates:
                if not alive[uid][si]:
                    continue
                val = base[uid] + LEVEL_W * drift[uid][si] + noise[uid][si] * gauss(0.0, 1.0)
                scored.append((val, uid))
            scored.sort(key=lambda x: (-x[0], x[1]))
            for place_idx in range(min(places, len(scored))):
                counts[scored[place_idx][1]][place_idx] += 1

    # ---- Probabilities -----------------------------------------------------------
    n_cand = len(candidates)
    prob = {}  # (uid, place) -> p
    for uid in candidates:
        for place_idx in range(places):
            freq = counts[uid][place_idx] / total_draws if total_draws else 0.0
            p = (1.0 - SMOOTH_EPS) * freq + SMOOTH_EPS / n_cand
            if p >= MIN_ROW_P:
                prob[(uid, place_idx + 1)] = p
    rows = []
    for place in range(1, places + 1):
        place_rows = [(uid, p) for (uid, pl), p in prob.items() if pl == place]
        place_rows.sort(key=lambda x: (-x[1], x[0]))
        place_rows = place_rows[:100]
        s = sum(p for _, p in place_rows)
        if s > PLACE_SUM_CAP:
            scale = PLACE_SUM_CAP / s
            place_rows = [(uid, p * scale) for uid, p in place_rows]
        for uid, p in place_rows:
            if p > 0:
                rows.append({"universe_id": int(uid), "place": place, "p": round(float(p), 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # ---- top10 ---------------------------------------------------------------------
    pmap = {}
    for r in rows:
        pmap.setdefault(r["universe_id"], {})[r["place"]] = r["p"]

    def expected_score(uid):
        d = pmap.get(uid, {})
        return sum(d.get(pl, 0.0) * (11 - pl) for pl in range(1, places + 1))

    def p_top10(uid):
        d = pmap.get(uid, {})
        return sum(d.get(pl, 0.0) for pl in range(1, places + 1))

    ranked = sorted(candidates, key=lambda u: (-expected_score(u), -p_top10(u), u))
    top10 = [int(u) for u in ranked[:places]]
    if len(top10) < places:
        for uid in sorted(last_rank, key=lambda u: last_rank[u]):
            if uid not in top10:
                top10.append(int(uid))
            if len(top10) >= places:
                break

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    payload = {"probabilities": rows, "top10": top10[:places]}
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), sort_keys=False))
        fh.write("\n")


if __name__ == "__main__":
    main()
