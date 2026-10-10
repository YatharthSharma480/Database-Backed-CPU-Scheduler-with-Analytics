"""
Scheduler v8: queue + concurrency cap + six classic scheduling algorithms,
with every query logged into MySQL.

Algorithms (policy name -> kind)
  fcfs        First Come First Served                  non-preemptive
  sjf         Shortest Job First (with aging)          non-preemptive
  srtf        Shortest Remaining Time First (+ aging)  preemptive
  rr          Round Robin (time quantum)               preemptive
  priority    Priority (with aging)                    non-preemptive
  priority_p  Priority (with aging)                    preemptive

How preemption works on MySQL
  A running MySQL query cannot be paused. For the preemptive policies a heavy query
  is therefore cut into slices (id ranges). Every slice is its own small query, and
  after each time quantum (QUANTUM slices) the scheduler may hand the worker to a
  different query. This only works for queries that can be cut up (COUNT, SUM, ...),
  and the slices of one query do not see one consistent snapshot. That is fine for
  the static test table but not for live data. The non-preemptive policies run the
  original query in one piece, exactly as plain MySQL would.

Usage
  python scheduler_v8.py check [slices]
  python scheduler_v8.py demo <cap> <n_heavy> <n_light> <run_label> [policy] [slices]
  python scheduler_v8.py compare <cap> <n_heavy> <n_light> <repeats> [slices]
  python scheduler_v8.py recover <cap>

Needs the 'sched_test' database from feasibility_test.py.
"""
import collections
import getpass
import hashlib
import itertools
import os
import queue
import random
import re
import statistics
import sys
import threading
import time
from datetime import datetime

import mysql.connector

HOST, PORT, USER = "127.0.0.1", 3306, "root"
TARGET_DB = "sched_test"
META_DB = "sched_meta"

UNKNOWN_MS = 5000    # predicted runtime for a query shape we have never seen
SLICES = 32          # preemptive policies: slices per heavy query
QUANTUM = 1          # preemptive policies: slices run before the scheduler may switch
AGING = 0.1          # sjf/srtf: each second of waiting counts as 100 ms off the prediction
PRIORITY = {"light": 1, "heavy": 5}   # priority policies: lower number runs first
PRIO_AGE_S = 30      # priority policies: each 30 s of waiting counts as one level

POLICIES = ("fcfs", "sjf", "srtf", "rr", "priority", "priority_p")
PREEMPTIVE = ("srtf", "rr", "priority_p")

HEAVY_SQL = (
    "SELECT COUNT(*) FROM big "
    "WHERE SHA2(SHA2(SHA2(CONCAT(payload, val), 256), 256), 256) LIKE '0%'"
)
LIGHT_SQL = "SELECT val FROM big WHERE id = {id}"

SCHEMA = [
    "CREATE DATABASE IF NOT EXISTS {m}",
    """CREATE TABLE IF NOT EXISTS {m}.query_log (
        id BIGINT PRIMARY KEY,
        run_id VARCHAR(40) NOT NULL,
        tag VARCHAR(32) NOT NULL DEFAULT '',
        fingerprint CHAR(32) NOT NULL,
        sql_text TEXT NOT NULL,
        status ENUM('queued','running','done','failed') NOT NULL DEFAULT 'queued',
        submitted_at DATETIME(3) NOT NULL,
        started_at DATETIME(3) NULL,
        finished_at DATETIME(3) NULL,
        wait_ms INT NULL,
        run_ms INT NULL,
        total_ms INT NULL,
        error VARCHAR(500) NULL,
        INDEX idx_status (status),
        INDEX idx_run_tag (run_id, tag),
        INDEX idx_fp (fingerprint)
    )""",
    """CREATE TABLE IF NOT EXISTS {m}.fingerprint (
        fingerprint CHAR(32) PRIMARY KEY,
        template TEXT NOT NULL,
        runs INT NOT NULL DEFAULT 0,
        avg_run_ms DOUBLE NOT NULL DEFAULT 0,
        last_seen DATETIME(3) NOT NULL
    )""",
]


def fingerprint(sql):
    """Replace literals with ? so similar queries share one fingerprint."""
    t = re.sub(r"\s+", " ", sql.strip().lower())
    t = re.sub(r"'[^']*'", "?", t)
    t = re.sub(r"\b\d+(\.\d+)?\b", "?", t)
    return hashlib.md5(t.encode()).hexdigest(), t


def dt(ts):
    return datetime.fromtimestamp(ts)


def make_slices(lo, hi, n):
    """Cut the id range lo..hi into n contiguous ranges."""
    size = -(-(hi - lo + 1) // n)
    return [(a, min(a + size - 1, hi)) for a in range(lo, hi + 1, size)]


class Job:
    """One logical query. Preemptive policies run it a few slices at a time."""

    def __init__(self, qid, sql, fp, tmpl, submitted, tag="", priority=3, slices=None):
        self.qid, self.sql, self.fp, self.tmpl = qid, sql, fp, tmpl
        self.submitted = submitted
        self.ready_at = submitted        # when it last started waiting (for aging)
        self.tag, self.priority = tag, priority
        self.sliced = bool(slices)
        self.chunks = ([f"{sql} AND id BETWEEN {a} AND {b}" for a, b in slices]
                       if slices else [sql])
        self.total_chunks = len(self.chunks)
        self.first_start = None
        self.service_ms = 0.0            # time actually spent executing on MySQL
        self.result = 0                  # sum of the slice results (COUNT queries)


class Scheduler:
    def __init__(self, password, cap, run_id, recover=False, policy="fcfs",
                 aging=AGING, quantum=QUANTUM):
        if policy == "fifo":
            policy = "fcfs"
        if policy not in POLICIES:
            raise ValueError("policy must be one of: " + ", ".join(POLICIES))
        self.password, self.cap, self.run_id = password, cap, run_id
        self.policy, self.aging, self.quantum = policy, aging, max(1, quantum)
        self.preemptive = policy in PREEMPTIVE
        self.history = {}
        self.results = {}                # qid -> summed result of finished sliced jobs
        self.cond = threading.Condition()
        self.pending = collections.deque()
        self.outstanding = 0
        self.stopping = False
        self.log_q = queue.Queue()

        # one-time setup on the main thread, before any other thread exists
        setup = self._connect(None)
        cur = setup.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt.format(m=META_DB))
        cur.execute(f"SELECT COALESCE(MAX(id), 0) FROM {META_DB}.query_log")
        self.ids = itertools.count(cur.fetchone()[0] + 1)
        # what the scheduler has learned: fingerprint -> [fastest run ms, runs],
        # rebuilt from the query log. The fastest run approximates a query's real
        # cost, because contention only ever makes a run slower (averages get polluted).
        cur.execute(
            "SELECT fingerprint, MIN(run_ms), COUNT(*) "
            f"FROM {META_DB}.query_log WHERE status='done' GROUP BY fingerprint")
        self.history = {fp: [float(fastest), n] for fp, fastest, n in cur.fetchall()}
        cur.execute(
            f"SELECT id, sql_text, submitted_at FROM {META_DB}.query_log "
            "WHERE status IN ('queued','running') ORDER BY id"
        )
        leftovers = cur.fetchall()
        if recover and leftovers:
            cur.execute(
                f"UPDATE {META_DB}.query_log SET status='queued', started_at=NULL "
                "WHERE status IN ('queued','running')"
            )
            setup.commit()
            for qid, sql, submitted_at in leftovers:
                fp, tmpl = fingerprint(sql)
                self.pending.append(Job(qid, sql, fp, tmpl, submitted_at.timestamp()))
            self.outstanding = len(leftovers)
            print(f"Recovered {len(leftovers)} unfinished queries from the log.")
        elif leftovers:
            print(f"Note: {len(leftovers)} unfinished queries from an earlier run. "
                  "Run 'python scheduler_v8.py recover <cap>' to finish them.")
        setup.close()

        self.meta = self._connect(META_DB)
        self.logger = threading.Thread(target=self._logger, daemon=True)
        self.logger.start()
        self.workers = [threading.Thread(target=self._worker, daemon=True)
                        for _ in range(cap)]
        for w in self.workers:
            w.start()

    def _connect(self, db):
        return mysql.connector.connect(
            host=HOST, port=PORT, user=USER, password=self.password, database=db
        )

    # ---- public API -------------------------------------------------------
    def submit(self, sql, tag="", priority=None, slices=None):
        """slices: list of (lo, hi) id ranges. Only used by the preemptive policies."""
        if not sql.strip().lower().startswith("select"):
            raise ValueError("only SELECT is accepted (safe to re-run after a crash)")
        fp, tmpl = fingerprint(sql)
        qid, ts = next(self.ids), time.time()
        prio = PRIORITY.get(tag, 3) if priority is None else priority
        job = Job(qid, sql, fp, tmpl, ts, tag, prio, slices)
        self.log_q.put(("submit", (qid, self.run_id, tag, fp, sql, ts)))
        with self.cond:
            self.pending.append(job)
            self.outstanding += 1
            self.cond.notify()
        return qid

    def wait_all(self):
        with self.cond:
            while self.outstanding > 0:
                self.cond.wait()

    def close(self):
        self.wait_all()
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        for w in self.workers:
            w.join()
        self.log_q.put(None)
        self.logger.join()
        self.meta.close()

    # ---- the one place the scheduling policy lives ------------------------
    def _predicted(self, fp):
        return self.history.get(fp, [UNKNOWN_MS, 0])[0]

    def _remaining_ms(self, job):
        """Predicted time still needed. Shrinks as slices complete."""
        return self._predicted(job.fp) * len(job.chunks) / job.total_chunks

    def _pick_next(self):
        """Called with self.cond held and self.pending non-empty. Removes and
        returns the job that should run next."""
        pol = self.policy
        if pol in ("fcfs", "rr"):
            # rr: preempted jobs go to the back of the queue, so popleft is the rotation
            return self.pending.popleft()
        now = time.time()
        if pol in ("sjf", "srtf"):
            # shortest (remaining) predicted time first. Waiting lowers the key, so a
            # long-waiting heavy query eventually outranks fresh light ones (no starvation).
            key = lambda j: self._remaining_ms(j) - self.aging * (now - j.ready_at) * 1000
        else:
            # lowest priority number first; PRIO_AGE_S seconds of waiting = one level
            key = lambda j: (j.priority - (now - j.ready_at) / PRIO_AGE_S, j.ready_at)
        best = min(self.pending, key=key)
        self.pending.remove(best)
        return best

    # ---- worker: runs queries on MySQL ------------------------------------
    def _worker(self):
        conn = self._connect(TARGET_DB)
        cur = conn.cursor()
        while True:
            with self.cond:
                job = None
                while True:
                    if self.pending:
                        job = self._pick_next()
                        break
                    if self.stopping:
                        break
                    self.cond.wait()
                if job is None:
                    break
                if self.preemptive and job.sliced:
                    stmts = job.chunks[:self.quantum]       # run one quantum, then re-decide
                    job.chunks = job.chunks[self.quantum:]
                else:
                    stmts = [job.sql]                       # run the whole query in one piece
                    job.chunks = []
            start = time.time()
            if job.first_start is None:
                job.first_start = start
                self.log_q.put(("start", (job.qid, start)))
            t0 = time.perf_counter()
            error = None
            try:
                for stmt in stmts:
                    cur.execute(stmt)
                    rows = cur.fetchall()
                    if job.sliced and self.preemptive and rows:
                        job.result += int(rows[0][0])
            except Exception as e:
                error = str(e)[:500]
            job.service_ms += (time.perf_counter() - t0) * 1000
            end = time.time()
            if error is None and job.chunks:
                with self.cond:                             # preempted: back into the queue
                    job.ready_at = end
                    self.pending.append(job)
                    self.cond.notify_all()
                continue
            if error is not None:
                self.log_q.put(("fail", (job.qid, end, error)))
            else:
                total_ms = int((end - job.submitted) * 1000)
                run_ms = int(job.service_ms)
                self.log_q.put(("done", (job.qid, job.fp, job.tmpl, end,
                                         max(0, total_ms - run_ms), run_ms, total_ms)))
            with self.cond:
                if error is None:
                    run_ms = int(job.service_ms)
                    if job.sliced and self.preemptive:
                        self.results[job.qid] = job.result
                    if job.fp in self.history:              # learn from this run
                        h = self.history[job.fp]
                        h[0] = min(h[0], run_ms)
                        h[1] += 1
                    else:
                        self.history[job.fp] = [float(run_ms), 1]
                self.outstanding -= 1
                self.cond.notify_all()
        conn.close()

    # ---- logger: writes events to MySQL in the background -----------------
    def _logger(self):
        cur = self.meta.cursor()
        while True:
            ev = self.log_q.get()
            if ev is None:
                break
            batch = [ev]
            while len(batch) < 50:
                try:
                    nxt = self.log_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self.log_q.put(None)
                    break
                batch.append(nxt)
            for kind, a in batch:
                try:
                    self._apply(cur, kind, a)
                except Exception as e:
                    print(f"[logger] could not write a {kind} event: {e}")
            self.meta.commit()

    def _apply(self, cur, kind, a):
        m = META_DB
        if kind == "submit":
            qid, run_id, tag, fp, sql, ts = a
            cur.execute(
                f"INSERT INTO {m}.query_log (id, run_id, tag, fingerprint, sql_text, "
                "status, submitted_at) VALUES (%s,%s,%s,%s,%s,'queued',%s)",
                (qid, run_id, tag, fp, sql, dt(ts)))
        elif kind == "start":
            qid, ts = a
            cur.execute(f"UPDATE {m}.query_log SET status='running', started_at=%s "
                        "WHERE id=%s", (dt(ts), qid))
        elif kind == "done":
            qid, fp, tmpl, ts, wait_ms, run_ms, total_ms = a
            cur.execute(
                f"UPDATE {m}.query_log SET status='done', finished_at=%s, wait_ms=%s, "
                "run_ms=%s, total_ms=%s WHERE id=%s",
                (dt(ts), wait_ms, run_ms, total_ms, qid))
            cur.execute(
                f"INSERT INTO {m}.fingerprint (fingerprint, template, runs, avg_run_ms, "
                "last_seen) VALUES (%s,%s,1,%s,%s) AS new (f_, t_, r_, a_, l_) "
                "ON DUPLICATE KEY UPDATE "
                "avg_run_ms = (avg_run_ms * runs + new.a_) / (runs + 1), "
                "runs = runs + 1, last_seen = new.l_",
                (fp, tmpl, run_ms, dt(ts)))
        elif kind == "fail":
            qid, ts, err = a
            cur.execute(f"UPDATE {m}.query_log SET status='failed', finished_at=%s, "
                        "error=%s WHERE id=%s", (dt(ts), err, qid))


# ---- helpers used by the commands below -----------------------------------
def get_bounds(password):
    """Smallest and largest id in the test table (ids can have gaps)."""
    conn = mysql.connector.connect(host=HOST, port=PORT, user=USER,
                                   password=password, database=TARGET_DB)
    cur = conn.cursor()
    cur.execute("SELECT MIN(id), MAX(id) FROM big")
    lo, hi = cur.fetchone()
    conn.close()
    return int(lo), int(hi)


def run_workload(s, n_heavy, n_light, bounds, slices):
    """n_heavy heavy queries at once, then n_light light ones, one every 0.1 s."""
    lo, hi = bounds
    sl = make_slices(lo, hi, slices)
    for _ in range(n_heavy):
        s.submit(HEAVY_SQL, "heavy", slices=sl)
    time.sleep(1)
    for _ in range(n_light):
        s.submit(LIGHT_SQL.format(id=random.randint(lo, hi)), "light")
        time.sleep(0.1)


def summary(password, run_id):
    conn = mysql.connector.connect(host=HOST, port=PORT, user=USER,
                                   password=password, database=META_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT tag, COUNT(*), ROUND(AVG(wait_ms)), ROUND(AVG(run_ms)), "
        "ROUND(AVG(total_ms)), MAX(total_ms) FROM query_log "
        "WHERE run_id=%s AND status='done' GROUP BY tag", (run_id,))
    print(f"\n=========== RESULT for run '{run_id}' (times in ms) ===========")
    print(f"{'tag':>8} {'count':>6} {'avg wait':>10} {'avg run':>9} "
          f"{'avg total':>10} {'max total':>10}")
    for tag, n, w, r, t, mx in cur.fetchall():
        print(f"{tag:>8} {n:>6} {w:>10} {r:>9} {t:>10} {mx:>10}")
    conn.close()


def run_stats(password, run_id):
    conn = mysql.connector.connect(host=HOST, port=PORT, user=USER,
                                   password=password, database=META_DB)
    cur = conn.cursor()
    cur.execute("SELECT tag, AVG(total_ms), MAX(total_ms) FROM query_log "
                "WHERE run_id=%s AND status='done' GROUP BY tag", (run_id,))
    stats = {tag: (float(avg), float(mx)) for tag, avg, mx in cur.fetchall()}
    cur.execute("SELECT TIMESTAMPDIFF(MICROSECOND, MIN(submitted_at), "
                "MAX(finished_at)) / 1000 FROM query_log "
                "WHERE run_id=%s AND status='done'", (run_id,))
    makespan = float(cur.fetchone()[0])
    conn.close()
    return stats, makespan


# ---- commands --------------------------------------------------------------
def check(password, slices):
    """Does the sliced heavy query give the same answer as the whole query?"""
    lo, hi = get_bounds(password)
    conn = mysql.connector.connect(host=HOST, port=PORT, user=USER,
                                   password=password, database=TARGET_DB)
    cur = conn.cursor()
    print(f"Ids run from {lo} to {hi}. Running the heavy query whole...")
    t0 = time.perf_counter()
    cur.execute(HEAVY_SQL)
    full = int(cur.fetchall()[0][0])
    t_full = time.perf_counter() - t0
    print(f"  whole query:  count {full}   ({t_full:.1f} s)")
    print(f"Running it again as {slices} slices...")
    t0 = time.perf_counter()
    total = 0
    for a, b in make_slices(lo, hi, slices):
        cur.execute(f"{HEAVY_SQL} AND id BETWEEN {a} AND {b}")
        total += int(cur.fetchall()[0][0])
    t_sliced = time.perf_counter() - t0
    print(f"  sliced query: count {total}   ({t_sliced:.1f} s)")
    print("\nRESULT: the counts " + ("MATCH." if full == total else "DO NOT MATCH. Tell me."))
    print(f"Slicing overhead: {t_sliced - t_full:+.2f} s over the whole query.")
    conn.close()


def demo(password, cap, n_heavy, n_light, run_id, policy="fcfs", slices=SLICES):
    bounds = get_bounds(password)
    s = Scheduler(password, cap, run_id, policy=policy)
    print(f"Run '{run_id}' ({policy}): cap {cap}, {n_heavy} heavy queries now, "
          f"then {n_light} light queries arriving every 0.1 s.")
    run_workload(s, n_heavy, n_light, bounds, slices)
    s.close()                       # waits for everything to finish
    time.sleep(0.5)
    summary(password, run_id)


def recover(password, cap):
    s = Scheduler(password, cap, "recovered", recover=True)
    s.close()
    print("Done. Check the sched_meta.query_log table for the results.")


# (name, policy, cap); a cap of None means "use the cap given on the command line"
CONFIGS = [
    ("Plain MySQL (no limit)", "fcfs", 100),
    ("FCFS", "fcfs", None),
    ("SJF (non-preemptive)", "sjf", None),
    ("SRTF (preemptive)", "srtf", None),
    ("Round Robin (preemptive)", "rr", None),
    ("Priority (non-preemptive)", "priority", None),
    ("Priority (preemptive)", "priority_p", None),
]


def warm_up(password, bounds):
    """One heavy and one light query alone, so every policy has a clean runtime estimate."""
    print("Warm-up: timing one heavy and one light query alone ...", flush=True)
    s = Scheduler(password, 1, "warmup", policy="fcfs")
    s.submit(HEAVY_SQL, "heavy")
    s.submit(LIGHT_SQL.format(id=bounds[0]), "light")
    s.close()
    time.sleep(0.5)


def compare(password, cap, n_heavy, n_light, repeats, slices=SLICES):
    bounds = get_bounds(password)
    warm_up(password, bounds)
    stamp = time.strftime("%m%d%H%M")
    results = {name: [] for name, _, _ in CONFIGS}
    for rep in range(1, repeats + 1):
        for idx, (name, policy, c) in enumerate(CONFIGS):
            run_id = f"cmp{stamp}_{idx}_r{rep}"
            print(f"[repeat {rep}/{repeats}] {name} ...", flush=True)
            s = Scheduler(password, cap if c is None else c, run_id, policy=policy)
            run_workload(s, n_heavy, n_light, bounds, slices)
            s.close()
            time.sleep(0.5)
            results[name].append(run_stats(password, run_id))
    print(f"\n=========== COMPARISON: {n_heavy} heavy + {n_light} light queries, "
          f"cap {cap}, {repeats} repeats (seconds) ===========")
    print(f"{'algorithm':<26} {'light avg':>10} {'light range':>13} "
          f"{'light worst':>12} {'heavy avg':>10} {'batch':>7}")
    for name, runs in results.items():
        la = [r[0]["light"][0] / 1000 for r in runs]
        lw = [r[0]["light"][1] / 1000 for r in runs]
        ha = [r[0]["heavy"][0] / 1000 for r in runs]
        mk = [r[1] / 1000 for r in runs]
        rng = f"{min(la):.1f}-{max(la):.1f}"
        print(f"{name:<26} {statistics.mean(la):>10.1f} {rng:>13} {max(lw):>12.1f} "
              f"{statistics.mean(ha):>10.1f} {statistics.mean(mk):>7.1f}")
    print("\n'light avg' = average submit-to-finish time of the small queries;")
    print("'light range' = lowest and highest average across repeats;")
    print("'batch' = time until the last query of the run finished.")
    print(f"Preemptive policies cut each heavy query into {slices} slices.")


USAGE = ("Use:  check [slices]\n"
         "      demo <cap> <n_heavy> <n_light> <run_label> [policy] [slices]\n"
         "      compare <cap> <n_heavy> <n_light> <repeats> [slices]\n"
         "      recover <cap>\n"
         "policies: " + ", ".join(POLICIES))


def _arg(i, default):
    return type(default)(sys.argv[i]) if len(sys.argv) > i else default


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("check", "demo", "compare", "recover"):
        print(USAGE)
        sys.exit(1)
    PASSWORD = os.environ.get("SCHED_PW") or getpass.getpass("MySQL root password: ")
    if mode == "check":
        check(PASSWORD, _arg(2, SLICES))
    elif mode == "demo":
        demo(PASSWORD, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
             _arg(6, "fcfs"), _arg(7, SLICES))
    elif mode == "compare":
        compare(PASSWORD, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                int(sys.argv[5]), _arg(6, SLICES))
    else:
        recover(PASSWORD, int(sys.argv[2]))
