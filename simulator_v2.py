#!/usr/bin/env python3
"""
simulator_v2.py (version 2, calibrated on real runs) - fast stand-in for scheduler_v8.py.

It never touches MySQL. It makes up query sets, "runs" each set under every
setting (order x cap) on paper, and records which setting gave the best result.
Those rows are what the decision tree learns from.

Orders   fcfs      first come first served
         sjf       shortest predicted time first, with aging   (same rule as scheduler_v8.py)
         priority  lowest predicted group (1-4) first, with aging (same rule as scheduler_v8.py)
Caps     how many queries may run at the same time (default 8 and 32)
Metric   response time = finish time - arrival time, per query.
         Reported as the mean over all queries and as the 95th percentile.

Commands
  python simulator_v2.py validate
        Simulates your real "32 heavy + 50 light" test and prints it next to the
        numbers you measured. Run this FIRST. If it is far off, fix the settings below.
  python simulator_v2.py run --n 3000 --out train.csv
  python simulator_v2.py run --n 300 --profile unseen --seed 99 --out unseen.csv
        Generates query sets, simulates all settings, writes one CSV row per set.
  python simulator_v2.py summary train.csv
        Win counts, near-tie share, and how far the best FIXED setting is from the oracle.

Needs only Python 3 (no packages). One honest limit: the features written to the CSV
are computed over the whole query set. The live scheduler will compute them from the
current queue, so keep the definitions in features() and the live code identical.
"""
import argparse
import bisect
import csv
import math
import random
import statistics
import sys
import time

# ======================================================================
# SETTINGS TO CALIBRATE FROM REAL MYSQL RUNS  (edit these)
# ======================================================================
CORES = 8                 # your CPU: 8 cores, 8 logical processors
HEAVY_SOLO_S = 10.7       # seconds one heavy query takes at low load. From your clean runs:
#                           4 heavy at once took 10.7 s each. NOTE: your `check` run said 13.1 s
#                           for one query alone, but that is likely a cold start; this file only
#                           needs a number consistent with the THROUGHPUT table below.

# THROUGHPUT[n] = total work finished per second when n heavy queries run at once, in
# "solo-seconds per second" (= n * HEAVY_SOLO_S / time until all n finish).
# Measured from your clean runs (max total): n=4 -> 11.6 s, n=8 -> 12.9 s,
# n=16 -> 26.3 s, n=32 -> 52.8 s. It tops out near 6.5 from 8 upward, which fits
# 4 fast cores + 4 slower efficiency cores. The value at 32 is 6.9, a little above the
# 6.5 you get from the last finish time, because that last finish includes a slow tail of
# a few stragglers which the simulator models separately. n=1 and n=2 are ASSUMED.
THROUGHPUT = {1: 1.0, 2: 1.95, 4: 3.7, 8: 6.6, 16: 6.5, 32: 6.9}

# Above UNFAIR_ABOVE heavy queries running at once, MySQL shares the CPU unevenly. In your runs
# 32 at once finished between about 25 s (first) and 51 s (last), average 38 s, while 8 and 16
# at once finished almost together. Modelled by a random speed weight per query, drawn
# uniformly between UNFAIR_MIN and 1. FITTED to those three numbers, not measured directly.
UNFAIR_ABOVE = 16
UNFAIR_MIN = 0.02

LIGHT_MAX_S = 0.5         # work below this counts as "light": needs a slot but no CPU share
LIGHT_STALL_S = 0.0       # extra seconds a light query loses per heavy query running beyond
#                           CORES. Left at 0: your cap-32 results are explained by waiting for a
#                           slot alone, and any value above 0 made cap 32 look much worse than
#                           your real runs. It only mattered for plain MySQL with no cap, which
#                           is not one of the settings the scheduler can choose, so that case
#                           is NOT modelled (validate does not show it).

# Copied from scheduler_v8.py so the simulator ranks queries the same way
AGING = 0.1               # sjf: each second waited counts as 0.1 s off the prediction
PRIO_AGE_S = 30.0         # priority: each 30 s waited counts as one level

GROUP_CUTS = (0.01, 0.5, 5.0)     # seconds: group 1 <=0.01, 2 <=0.5, 3 <=5, 4 above
GROUP_TIME = {1: 0.001, 2: 0.05, 3: 2.0, 4: HEAVY_SOLO_S}   # typical time of each group

ORDERS = ("fcfs", "sjf", "priority")
CAPS = (8, 32)


def group_of(seconds):
    for g, cut in enumerate(GROUP_CUTS, start=1):
        if seconds <= cut:
            return g
    return 4


# ======================================================================
# How fast queries run when several share the CPU
# ======================================================================
_TP_KEYS = sorted(THROUGHPUT)


def _throughput(n):
    k = _TP_KEYS
    if n <= k[0]:
        return THROUGHPUT[k[0]] * n / k[0]
    if n >= k[-1]:
        return THROUGHPUT[k[-1]]
    i = bisect.bisect_left(k, n)
    if k[i] == n:
        return THROUGHPUT[n]
    k0, k1 = k[i - 1], k[i]
    v0, v1 = THROUGHPUT[k0], THROUGHPUT[k1]
    return v0 + (v1 - v0) * (n - k0) / (k1 - k0)


_TP = [0.0] + [_throughput(n) for n in range(1, 513)]
_RATE = [0.0] + [_throughput(n) / n for n in range(1, 513)]


def rate(n):
    """Speed of ONE running heavy query when n heavy queries run together."""
    return _RATE[n] if n < len(_RATE) else _RATE[-1]


# ======================================================================
# The simulation itself
# ======================================================================
class Workload:
    __slots__ = ("arr", "work", "timer", "ptime", "pgrp", "known", "wt", "gen")


def simulate(w, order, cap):
    """Return the finish time of every query of workload w under (order, cap).

    Heavy queries (work >= LIGHT_MAX_S) share the CPU equally (processor sharing).
    Light queries only need a free slot; their time is their tiny work plus a stall
    that grows when too many heavy queries are running. Nothing is preempted.
    """
    arr, work, timer = w.arr, w.work, w.timer
    ptime, pgrp, wt = w.ptime, w.pgrp, w.wt
    n = len(arr)
    finish = [0.0] * n
    pending = []
    cpu = {}              # running heavy query -> work left
    tim = {}              # running light query -> finish time
    t = arr[0]
    nxt = 0
    done = 0
    while done < n:
        while nxt < n and arr[nxt] <= t + 1e-12:
            pending.append(nxt)
            nxt += 1
        while pending and len(cpu) + len(tim) < cap:
            if order == "fcfs":
                i = pending.pop(0)
            else:
                if order == "sjf":
                    i = min(pending, key=lambda j: (ptime[j] - AGING * (t - arr[j]), arr[j], j))
                else:
                    i = min(pending, key=lambda j: (pgrp[j] - (t - arr[j]) / PRIO_AGE_S, arr[j], j))
                pending.remove(i)
            if timer[i]:
                tim[i] = t + work[i] + LIGHT_STALL_S * max(0, len(cpu) - CORES)
            else:
                cpu[i] = work[i]
        nt = arr[nxt] if nxt < n else float("inf")
        if tim:
            m = min(tim.values())
            if m < nt:
                nt = m
        r = 0.0
        rt = None
        if cpu:
            m = len(cpu)
            if m > UNFAIR_ABOVE:
                tp = _TP[min(m, 512)]
                sw = 0.0
                for i in cpu:
                    sw += wt[i]
                rt = {i: tp * wt[i] / sw for i in cpu}
                first = min(cpu, key=lambda i: cpu[i] / rt[i])
                ct = t + cpu[first] / rt[first]
            else:
                r = rate(m)
                first = min(cpu, key=cpu.get)
                ct = t + cpu[first] / r
            if ct < nt:
                nt = ct
        dt = nt - t
        if dt > 0 and cpu:
            if rt is None:
                d = r * dt
                for i in cpu:
                    cpu[i] -= d
            else:
                for i in cpu:
                    cpu[i] -= rt[i] * dt
        t = nt
        for i in [i for i, left in cpu.items() if left <= 1e-9]:
            finish[i] = t
            del cpu[i]
            done += 1
        for i in [i for i, ft in tim.items() if ft <= t + 1e-12]:
            finish[i] = t
            del tim[i]
            done += 1
    return finish


def metrics(arr, finish):
    resp = sorted(f - a for a, f in zip(arr, finish))
    n = len(resp)
    p95 = resp[min(n - 1, math.ceil(0.95 * n) - 1)]
    return sum(resp) / n, p95


# ======================================================================
# Making up query sets
# ======================================================================
def make_workload(rng, profile="train"):
    """profile 'train': the normal range.
       profile 'unseen': mostly-heavy, longer sets, and a ramp-up arrival pattern the
       training profile never produces. Use it to test on mixes the tree has not seen."""
    unseen = profile == "unseen"
    n = rng.randint(120, 200) if unseen else rng.randint(30, 120)
    heavy_share = rng.uniform(0.8, 0.95) if unseen else rng.uniform(0.0, 0.8)
    sigma = rng.uniform(0.05, 0.8)       # spread of runtimes inside each group
    rho = rng.uniform(0.3, 2.0)          # how loaded the CPU would be (1.0 = just full)
    pattern = rng.choice(["ramp", "ramp", "steady", "bursty"] if unseen
                         else ["steady", "bursty", "burst_trickle"])
    p_known = rng.uniform(0.2, 1.0)      # share of queries with real history
    acc = rng.uniform(0.6, 0.95)         # chance a guessed group is right
    big_frac = rng.random()              # of the heavy ones, how many are 'big'
    small_frac = rng.random()            # of the light ones, how many are 'small'

    p_big = heavy_share * big_frac
    p_med = heavy_share * (1 - big_frac)
    p_small = (1 - heavy_share) * small_frac
    work = []
    for _ in range(n):
        u = rng.random()
        if u < p_big:
            base = GROUP_TIME[4]
        elif u < p_big + p_med:
            base = GROUP_TIME[3]
        elif u < p_big + p_med + p_small:
            base = GROUP_TIME[2]
        else:
            base = GROUP_TIME[1]
        work.append(base * rng.lognormvariate(0.0, sigma))

    cpu_work = sum(x for x in work if x >= LIGHT_MAX_S)
    span = max(cpu_work / (CORES * rho), n / 50.0)    # seconds the arrivals are spread over
    if pattern == "steady":
        arr = [rng.uniform(0, span) for _ in range(n)]
    elif pattern == "ramp":
        arr = [span * math.sqrt(rng.random()) for _ in range(n)]
    elif pattern == "bursty":
        centers = [rng.uniform(0, span) for _ in range(rng.randint(2, 6))]
        arr = [rng.choice(centers) + rng.uniform(0, 0.2) for _ in range(n)]
    else:   # burst_trickle: like your real test, heavy ones at once, light ones trickle in
        gap = rng.uniform(0.02, 0.3)
        arr, k = [], 0
        for x in work:
            if x >= LIGHT_MAX_S:
                arr.append(rng.uniform(0, 0.1))
            else:
                arr.append(1.0 + gap * k)
                k += 1
    order = sorted(range(n), key=arr.__getitem__)
    arr = [arr[i] for i in order]
    work = [work[i] for i in order]

    known, ptime, pgrp = [], [], []
    for x in work:
        if rng.random() < p_known:                       # history: close to the truth
            known.append(True)
            p = x * rng.lognormvariate(0.0, 0.05)
            ptime.append(p)
            pgrp.append(group_of(p))
        else:                                            # guessed group (EXPLAIN rule)
            known.append(False)
            g = group_of(x)
            if rng.random() >= acc:
                g = g + 1 if g == 1 else g - 1 if g == 4 else g + rng.choice((-1, 1))
            ptime.append(GROUP_TIME[g])
            pgrp.append(g)

    w = Workload()
    w.arr, w.work, w.timer = arr, work, [x < LIGHT_MAX_S for x in work]
    w.ptime, w.pgrp, w.known = ptime, pgrp, known
    w.wt = [rng.uniform(UNFAIR_MIN, 1.0) for _ in range(n)]
    w.gen = {"gen_pattern": pattern, "gen_heavy_share": round(heavy_share, 3),
             "gen_sigma": round(sigma, 3), "gen_rho": round(rho, 3),
             "gen_group_acc": round(acc, 3), "gen_p_known": round(p_known, 3)}
    return w


def features(w):
    """Numbers describing the set, from PREDICTED information only (no true runtimes)."""
    n = len(w.arr)
    logs = [math.log10(p) for p in w.ptime]
    span = w.arr[-1] - w.arr[0]
    gaps = [b - a for a, b in zip(w.arr, w.arr[1:])]
    mg = statistics.fmean(gaps) if gaps else 0.0
    return {
        "n_queries": n,
        "share_long": round(sum(g == 4 for g in w.pgrp) / n, 4),
        "share_medium": round(sum(g == 3 for g in w.pgrp) / n, 4),
        "spread_log10": round(statistics.pstdev(logs), 4),
        "arrival_rate": round(n / max(span, 0.1), 4),
        "burst_cv": round(statistics.pstdev(gaps) / mg, 4) if mg > 0 else 0.0,
        "share_guessed": round(1 - sum(w.known) / n, 4),
    }


FEATURES = ["n_queries", "share_long", "share_medium", "spread_log10",
            "arrival_rate", "burst_cv", "share_guessed"]
GEN = ["gen_pattern", "gen_heavy_share", "gen_sigma", "gen_rho",
       "gen_group_acc", "gen_p_known"]


# ======================================================================
# Commands
# ======================================================================
def parse_list(text, cast):
    return tuple(cast(x) for x in text.split(","))


def cmd_run(args):
    orders = parse_list(args.orders, str)
    caps = parse_list(args.caps, int)
    settings = [(o, c) for o in orders for c in caps]
    names = [f"{o}_{c}" for o, c in settings]
    cols = (["id", "profile"] + FEATURES + GEN
            + [f"mean_{x}" for x in names] + [f"p95_{x}" for x in names]
            + ["winner_mean", "winner_p95", "margin_mean", "margin_p95"])
    rng = random.Random(args.seed)
    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for k in range(args.n):
            w = make_workload(rng, args.profile)
            row = {"id": k, "profile": args.profile}
            row.update(features(w))
            row.update(w.gen)
            mean_v, p95_v = {}, {}
            for (o, c), nm in zip(settings, names):
                mean_v[nm], p95_v[nm] = metrics(w.arr, simulate(w, o, c))
                row[f"mean_{nm}"] = round(mean_v[nm], 4)
                row[f"p95_{nm}"] = round(p95_v[nm], 4)
            for key, vals in (("mean", mean_v), ("p95", p95_v)):
                ranked = sorted(vals, key=vals.get)
                row[f"winner_{key}"] = ranked[0]
                row[f"margin_{key}"] = round(vals[ranked[1]] / vals[ranked[0]] - 1, 4) \
                    if vals[ranked[0]] > 0 else 0.0
            wr.writerow(row)
            if (k + 1) % 250 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{k + 1}/{args.n} sets done, {el:.0f}s elapsed, "
                      f"about {el / (k + 1) * (args.n - k - 1):.0f}s left", flush=True)
    print(f"Wrote {args.n} rows to {args.out} in {time.time() - t0:.0f}s")


def cmd_summary(args):
    with open(args.file, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("empty file")
        return
    print(f"{len(rows)} query sets in {args.file}")
    for key in ("mean", "p95"):
        names = [c[len(key) + 1:] for c in rows[0] if c.startswith(key + "_")]
        wins = {nm: 0 for nm in names}
        regret = {nm: 0.0 for nm in names}
        ties = 0
        for r in rows:
            vals = {nm: float(r[f"{key}_{nm}"]) for nm in names}
            best = min(vals.values())
            for nm in names:
                regret[nm] += (vals[nm] / best - 1) if best > 0 else 0.0
            wins[r[f"winner_{key}"]] += 1
            if float(r[f"margin_{key}"]) < 0.02:
                ties += 1
        print(f"\n--- metric: {key} response time ---")
        print(f"{'setting':<16}{'wins':>7}{'win %':>8}{'avg regret':>12}")
        for nm in sorted(names, key=lambda x: regret[x]):
            print(f"{nm:<16}{wins[nm]:>7}{100 * wins[nm] / len(rows):>7.1f}%"
                  f"{100 * regret[nm] / len(rows):>11.1f}%")
        bf = min(names, key=lambda x: regret[x])
        print(f"Best single fixed setting: {bf}, on average "
              f"{100 * regret[bf] / len(rows):.1f}% worse than picking the best per set "
              f"(the oracle). This is the most any model can gain.")
        print(f"Near-ties (best and second best within 2%): {100 * ties / len(rows):.1f}% of sets")


def _burst(n_heavy, n_light, rng):
    """The real test: n_heavy heavy queries at t=0, then n_light light ones, one per 0.1 s from t=1."""
    arr = [0.0] * n_heavy + [1.0 + 0.1 * k for k in range(n_light)]
    work = [HEAVY_SOLO_S] * n_heavy + [0.0005] * n_light
    w = Workload()
    w.arr, w.work, w.timer = arr, work, [x < LIGHT_MAX_S for x in work]
    w.ptime = list(work)
    w.pgrp = [group_of(x) for x in work]
    w.known = [True] * len(work)
    w.wt = [rng.uniform(UNFAIR_MIN, 1.0) for _ in work]
    w.gen = {}
    return w


def _burst_stats(n_heavy, n_light, order, cap, seeds=40):
    out = []
    for k in range(seeds):
        w = _burst(n_heavy, n_light, random.Random(k))
        fin = simulate(w, order, cap)
        light = [f - a for a, f, t in zip(w.arr, fin, w.timer) if t]
        heavy = [f - a for a, f, t in zip(w.arr, fin, w.timer) if not t]
        out.append((statistics.fmean(light) if light else 0.0, statistics.fmean(heavy),
                    max(fin) - min(w.arr), max(heavy)))
    return [statistics.fmean(c) for c in zip(*out)]


def cmd_validate(args):
    """Compare the simulator with every real measurement you have taken (sim / real)."""
    print("A) Heavy queries only, all at once (your clean demo runs). seconds, sim / real")
    print(f"{'heavy at once':<16}{'avg total':>18}{'last finish':>18}")
    real_demo = {4: (11.18, 11.58), 8: (12.78, 12.91), 16: (25.82, 26.32), 32: (39.22, 52.78)}
    for n, (ra, rm) in real_demo.items():
        _, h, _, m = _burst_stats(n, 0, "fcfs", n)
        print(f"{n:<16}{h:>9.1f} /{ra:>6.1f}{m:>11.1f} /{rm:>6.1f}")

    print("\nB) 32 heavy + 50 light through the scheduler (your compare runs). seconds, sim / real")
    print(f"{'setting':<20}{'light avg':>16}{'heavy avg':>16}{'batch':>16}")
    measured = [   # light avg, heavy avg, batch
        ("FCFS cap 8", "fcfs", 8, (43.6, 30.0, 48.3)),
        ("SJF cap 8", "sjf", 8, (8.5, 30.0, 47.8)),
        ("Priority cap 8", "priority", 8, (7.7, 29.8, 47.4)),
        ("FCFS cap 32", "fcfs", 32, (22.4, 37.8, 50.3)),
        ("SJF cap 32", "sjf", 32, (21.9, 37.6, 50.2)),
        ("Priority cap 32", "priority", 32, (22.1, 38.0, 50.7)),
    ]
    for name, o, c, m in measured:
        l, h, b, _ = _burst_stats(32, 50, o, c)
        print(f"{name:<20}{l:>8.1f} /{m[0]:>5.1f}{h:>9.1f} /{m[1]:>5.1f}{b:>9.1f} /{m[2]:>5.1f}")
    print("\nPlain MySQL with no cap is not modelled (see LIGHT_STALL_S at the top of this file).")
    print("Priority cap 8 light avg: real is about 1 s better than SJF; the simulator treats them alike here.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=3000, help="number of query sets")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--profile", choices=["train", "unseen"], default="train")
    r.add_argument("--orders", default=",".join(ORDERS))
    r.add_argument("--caps", default=",".join(str(c) for c in CAPS))
    r.add_argument("--out", default="sim_results.csv")
    s = sub.add_parser("summary")
    s.add_argument("file")
    args = ap.parse_args()
    {"validate": cmd_validate, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
