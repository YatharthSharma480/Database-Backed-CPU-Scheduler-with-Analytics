#!/usr/bin/env python3
"""
train_tree.py - train a small decision tree that picks the scheduling setting for a query set,
and test it honestly against fixed choices, a hand-written rule, and the best possible pick.

Usage (needs scikit-learn):
    python train_tree.py train.csv unseen.csv

Both CSV files come from simulator_v2.py. Nothing here touches MySQL.

What it does, once for the mean metric and once for the p95 metric:
  1. Keeps four choices (the three cap-32 orders behave the same whenever no queue builds,
     so they are merged into one choice that runs SJF):
         fcfs_8   sjf_8   priority_8   sjf_32
  2. Splits train.csv 70% (to learn) / 30% (held-out test). unseen.csv is a second test.
  3. Picks the tree depth (1 to 5) by 5-fold cross-validation on the 70% only; if several
     depths are about equally good it takes the shallower one, so the tree stays easy to explain.
  4. Prints the tree as readable text.
  5. Scores every method by REGRET: how much worse (in %) its pick is than the best of the
     four, averaged over query sets. Lower is better, 0 is perfect. This matters more than
     plain accuracy, because picking the wrong one of two near-tied settings costs nothing.
  6. Prints a learning curve: does more training data help?

The tree learns from the 7 feature columns only (computed from predicted information). The
gen_* columns describe how a set was generated and are never used for training.

Limits to state in your report: the query sets come from your own generator, and labels come
from a simulator fitted to your laptop. The unseen file only tests mixes the generator makes
differently, not real traffic.
"""
import csv
import sys

import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ["n_queries", "share_long", "share_medium", "spread_log10",
            "arrival_rate", "burst_cv", "share_guessed"]
# label name -> simulator column suffix. The merged cap-32 choice runs SJF.
CHOICES = {"fcfs_8": "fcfs_8", "sjf_8": "sjf_8", "priority_8": "priority_8", "sjf_32": "sjf_32"}
NAMES = list(CHOICES)
SEED = 0
MIN_LEAF = 30                 # a leaf must hold at least this many training sets
HAND_RULE_HEAVY_LIMIT = 32    # hand rule: fewer predicted heavy queries than this -> cap 32


def load(path, metric):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = [f"{metric}_{CHOICES[n]}" for n in NAMES] + FEATURES
    missing = [c for c in need if c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing columns {missing}. Make it with simulator_v2.py using "
                 f"the default orders and caps.")
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    V = np.array([[float(r[f"{metric}_{CHOICES[n]}"]) for n in NAMES] for r in rows])
    return X, V


def regret(V, pick):
    """Per-set % by which the chosen setting is worse than the best of the four."""
    chosen = V[np.arange(len(V)), pick]
    best = V.min(axis=1)
    return np.where(best > 0, chosen / best - 1.0, 0.0) * 100.0


def labels_and_weights(V):
    """Winner per set, and a weight = how much the choice matters for that set.
    Near-ties get a tiny weight so the tree does not waste splits on noise."""
    order = np.sort(V, axis=1)
    best, second = order[:, 0], order[:, 1]
    margin = np.where(best > 0, second / best - 1.0, 0.0)
    return V.argmin(axis=1), np.clip(margin, 0.01, 1.0)


def fit_tree(X, V, depth):
    y, w = labels_and_weights(V)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=SEED)
    tree.fit(X, y, sample_weight=w)
    return tree


def predict(tree, X):
    return tree.predict(X)          # classes are the column indices 0..3 that occurred


def pick_depth(X, V):
    scores = {}
    for d in range(1, 6):
        folds = []
        for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
            t = fit_tree(X[tr], V[tr], d)
            folds.append(regret(V[va], predict(t, X[va])).mean())
        scores[d] = float(np.mean(folds))
    best = min(scores.values())
    for d in range(1, 6):                      # shallowest depth that is about as good
        if scores[d] <= best + 0.1:
            return d, scores
    return 5, scores


def hand_rule(X):
    """Hand-written baseline, fixed in advance: if the number of predicted heavy queries
    (groups 3 and 4) is below the cap-32 limit, nothing queues, so use cap 32; else SJF cap 8."""
    heavy = X[:, 0] * (X[:, 1] + X[:, 2])
    return np.where(heavy < HAND_RULE_HEAVY_LIMIT, NAMES.index("sjf_32"), NAMES.index("sjf_8"))


def repeat_splits(X, V, repeats=10):
    """One split can mislead. Repeat with 10 different splits and report mean and spread."""
    global SEED
    keep = SEED
    out = {"TREE": [], "hand rule": []}
    for n in NAMES:
        out[f"always {n}"] = []
    wins = 0
    for seed in range(repeats):
        SEED = seed
        idx = np.random.RandomState(seed).permutation(len(X))
        cut = int(0.7 * len(X))
        tr, te = idx[:cut], idx[cut:]
        d, _ = pick_depth(X[tr], V[tr])
        tree = fit_tree(X[tr], V[tr], d)
        out["TREE"].append(regret(V[te], tree.predict(X[te])).mean())
        out["hand rule"].append(regret(V[te], hand_rule(X[te])).mean())
        for i, n in enumerate(NAMES):
            out[f"always {n}"].append(regret(V[te], np.full(len(te), i)).mean())
        wins += out["TREE"][-1] < out["hand rule"][-1]
    SEED = keep
    print(f"\nSame test on {repeats} different random splits (average regret %, mean +- spread):")
    for k, v in out.items():
        print(f"  {k:<18}{np.mean(v):>7.2f}% +- {np.std(v):.2f}")
    print(f"  The tree beat the hand rule in {wins} of {repeats} splits.")


def report(metric, train_path, unseen_path):
    X, V = load(train_path, metric)
    Xu, Vu = load(unseen_path, metric)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    Xtr, Vtr, Xte, Vte = X[tr], V[tr], X[te], V[te]

    depth, cv = pick_depth(Xtr, Vtr)
    tree = fit_tree(Xtr, Vtr, depth)
    classes = list(tree.classes_)     # only the choices that actually won somewhere

    def tree_pick(Xs):
        return tree.predict(Xs)       # class ids are indices into NAMES

    print("=" * 74)
    print(f"METRIC: {metric} response time    ({len(Xtr)} train sets, {len(Xte)} held-out, "
          f"{len(Xu)} unseen)")
    print("=" * 74)
    print("Cross-validated regret by depth (lower is better): " +
          ", ".join(f"d{d}={cv[d]:.2f}%" for d in sorted(cv)))
    print(f"Chosen depth: {depth}\n")
    print(export_text(tree, feature_names=FEATURES,
                      class_names=[NAMES[c] for c in classes]))

    methods = [(f"always {n}", (lambda Xs, n=n: np.full(len(Xs), NAMES.index(n)))) for n in NAMES]
    methods.append(("hand rule", hand_rule))
    methods.append((f"TREE (depth {depth})", tree_pick))
    print(f"{'method':<24}{'held-out':>12}{'unseen':>12}   (average regret %, lower is better)")
    for name, fn in methods:
        print(f"{name:<24}{regret(Vte, fn(Xte)).mean():>11.2f}%{regret(Vu, fn(Xu)).mean():>11.2f}%")
    print(f"{'oracle (best per set)':<24}{0.0:>11.2f}%{0.0:>11.2f}%")

    yte, _ = labels_and_weights(Vte)
    yu, _ = labels_and_weights(Vu)
    pt, pu = tree_pick(Xte), tree_pick(Xu)
    print(f"\nTree picks the exact winner: {100 * np.mean(pt == yte):.1f}% held-out, "
          f"{100 * np.mean(pu == yu):.1f}% unseen")
    print(f"Tree pick within 2% of the best: {100 * np.mean(regret(Vte, pt) <= 2):.1f}% held-out, "
          f"{100 * np.mean(regret(Vu, pu) <= 2):.1f}% unseen")
    fixed = {n: regret(Vte, np.full(len(Xte), i)).mean() for i, n in enumerate(NAMES)}
    bf = min(fixed, key=fixed.get)
    gain = fixed[bf] - regret(Vte, pt).mean()
    room = fixed[bf]
    print(f"Best fixed choice on held-out: {bf} ({fixed[bf]:.2f}% regret). Tree improves on it by "
          f"{gain:.2f} points, out of {room:.2f} points that were possible.")

    print("\nLearning curve (held-out regret as the training set grows):")
    for frac in (0.1, 0.2, 0.4, 0.7, 1.0):
        m = max(MIN_LEAF * 2, int(frac * len(Xtr)))
        t = fit_tree(Xtr[:m], Vtr[:m], depth)
        print(f"  {m:>5} training sets -> {regret(Vte, t.predict(Xte)).mean():.2f}%")
    repeat_splits(X, V)
    with open(f"tree_{metric}.txt", "w") as fh:
        fh.write(export_text(tree, feature_names=FEATURES,
                             class_names=[NAMES[c] for c in classes]))
    print(f"\n(tree text saved to tree_{metric}.txt)\n")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python train_tree.py train.csv unseen.csv")
    for metric in ("mean", "p95"):
        report(metric, sys.argv[1], sys.argv[2])
    print("Reminder: the labels come from a simulator fitted to your laptop and the query sets "
          "from your own generator. Confirm the tree's picks with real MySQL runs.")


if __name__ == "__main__":
    main()
