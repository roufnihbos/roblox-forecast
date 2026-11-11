# roblox-forecast

[![Competition](https://img.shields.io/badge/make.fun-roblox--up--and--coming--forecast--01-blue)](https://make.fun)
[![Owner](https://img.shields.io/badge/owner-roufnihbos-lightgrey)](#)
[![Repo](https://img.shields.io/badge/repo-roblox--forecast-lightgrey)](#)

- Competition: `roblox-up-and-coming-forecast-01`
- Owner: `roufnihbos`
- Repo: `roblox-forecast`

A Python 3.13, standard-library-only forecaster for Roblox's
Up-and-Coming chart. It reads the chart history (`history.ndjson` +
`task.json`) and writes one probabilistic forecast (`forecast.json`):
for each of places 1-10, the probability that each named game holds
that place, averaged over the 72 scored hours, plus a top-10 guess.

Method, honestly stated: for every candidate game (recent chart
appearances, including climbers also seen on Top Trending / Top
Playing Now) it fits a recent-weighted linear trend on log player
counts, shrinks that drift toward zero as the horizon grows (mean
reversion), cuts a game off when it ages past the chart's 28-day
limit if its creation date is known, and runs a seeded Monte Carlo:
project each candidate's player count at every window hour with
residual-sized noise, rank the candidates, and tally how often each
game lands at each place. Those occupancy frequencies, lightly
smoothed so no plausible game sits at exactly zero, are the forecast.
The chart is noisy and moves fast, so the probabilities are spread
on purpose: honest uncertainty scores better under the log score
than false confidence.

Run it:

```sh
MAKEFUN_INPUT_DIR=<input-dir> MAKEFUN_OUTPUT_DIR=<output-dir> python3 main.py
```

With no environment variables it reads `/input` and writes
`/output/forecast.json`, as the grader runs it. Deterministic: two
runs on the same history produce byte-identical output.
