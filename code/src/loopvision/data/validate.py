"""Gate G1: task validity.

**Pre-registered before any gate run.** These thresholds were written and
committed before the first training job was submitted, so that a failing
gate cannot be rescued by adjusting a number afterwards. If a check fails,
it fails, and the response is to deepen the task and re-run the gate with
both attempts recorded, per docs/findings.md.

The four checks (IMPLEMENTATION.md Section 4.6, as revised by
docs/decisions.md D-016):

  G1.1  Depth 3 unsolvable at k=1, by a looped model AND by a matched
        non-looped one. If either solves it, there is no loop-count curve
        to analyse and the task must be deepened.
  G1.2  Depth 1 solvable at k=1 to a high threshold. If not, the task is
        broken rather than deep, and the problem is rendering or capacity
        rather than composition.
  G1.3  The bag-of-operators probe must not exceed the order-blind
        ceiling, and that ceiling must sit below the M1 threshold tau.
        Replaces an earlier at-chance criterion that was unachievable.
  G1.4  Family C solvable at k=1 at every breadth. If counting needs
        loops, the breadth axis is contaminated by sequential structure
        and the H1 contrast is void.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopvision.data import groups as G

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------

TAU = 0.90  # the M1 threshold, IMPLEMENTATION.md Section 7

#: G1.1. Chance for family A is 1/48 = 0.0208. A model within about 2.4x of
#: chance after full training has not learned depth 3 composition. Set
#: above chance rather than at it because a converged model reliably picks
#: up small residual regularities, and the gate is about whether the task
#: is solvable, not whether accuracy is exactly chance.
G1_1_MAX_ACCURACY = 0.05

#: G1.2. Depth 1 is a single operator applied to a visible state. A model
#: that cannot reach tau here cannot see the glyphs, which would be a
#: rendering or capacity failure rather than evidence of depth.
G1_2_MIN_ACCURACY = TAU

#: G1.3. How far below tau the order-blind ceiling must sit. At 0.15 the
#: worst measured ceiling (0.652 at depth 2) clears the bar comfortably.
G1_3_CEILING_MARGIN = 0.15

#: G1.4. Family C is the H1 null arm and must be solvable in one pass at
#: every breadth, or breadth is not a null axis.
G1_4_MIN_ACCURACY = TAU


#: Which stored runs each check reads.
#:
#: **Attempt 1** used 20000 step runs (`g1_famA_*_s0`) and FAILED on G1.2:
#: family A depth 1 reached 0.0204 against a required 0.90. Attempt 2 reran
#: family A at 200000 steps per docs/decisions.md D-022 option 1.
#:
#: All three family A runs were rerun, not just the failing one. Comparing
#: depth 3 at 20000 steps against depth 1 at 200000 would not be a gate, it
#: would be an artefact of unequal training budgets.
#:
#: Family C is deliberately NOT rerun. It reached 1.000 at every breadth in
#: 20000 steps, and G1.4 asks whether the task is solvable at k=1. More
#: steps cannot make a solved task unsolved, so a rerun would cost GPU time
#: to confirm a saturated result.
GATE_RUNS = {
    "d3_looped": "g1_famA_d3_looped_long_s0",
    "d3_ffwd": "g1_famA_d3_ffwd_long_s0",
    "d1_looped": "g1_famA_d1_looped_long_s0",
    "famC": "g1_famC_looped_s0",
}

ATTEMPT_1_RUNS = {
    "d3_looped": "g1_famA_d3_looped_s0",
    "d3_ffwd": "g1_famA_d3_ffwd_s0",
    "d1_looped": "g1_famA_d1_looped_s0",
    "famC": "g1_famC_looped_s0",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    measured: float
    threshold: float
    detail: str = ""

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status}  {self.name}: {self.measured:.4f} (threshold {self.threshold:.4f}) {self.detail}"


# ---------------------------------------------------------------------------
# G1.3, the shortcut probe
# ---------------------------------------------------------------------------


def operator_count_features(depth: int, n: int, seed: int = 0):
    """Bag-of-operator-counts features and labels for family A.

    Features are counts of each of the 48 operator types, which is exactly
    what an order-blind model sees. Built from ground truth operator labels
    rather than from pixels, so the probe measures the task's shortcut
    rather than a vision model's ability to read glyphs.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, G.GROUP_ORDER), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ops, target = G.sample_operator_sequence(rng, depth)
        for op in ops:
            X[i, op] += 1.0
        y[i] = target
    return X, y


def bag_of_operators_probe(
    depth: int, n_train: int = 20000, n_test: int = 4000, epochs: int = 60, seed: int = 0
) -> float:
    """Held-out accuracy of a multinomial logistic regression on op counts.

    Implemented in numpy rather than pulling in scikit-learn, which would
    need a decisions.md dependency entry for one linear model.
    """
    Xtr, ytr = operator_count_features(depth, n_train, seed=seed)
    Xte, yte = operator_count_features(depth, n_test, seed=seed + 1000)

    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(G.GROUP_ORDER, G.GROUP_ORDER)).astype(np.float32)
    b = np.zeros(G.GROUP_ORDER, dtype=np.float32)
    onehot = np.eye(G.GROUP_ORDER, dtype=np.float32)[ytr]

    lr, batch = 0.5, 512
    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch):
            idx = order[start : start + batch]
            xb, yb = Xtr[idx], onehot[idx]
            logits = xb @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            grad = (probs - yb) / len(idx)
            W -= lr * (xb.T @ grad)
            b -= lr * grad.sum(axis=0)

    return float((np.argmax(Xte @ W + b, axis=1) == yte).mean())


def check_g1_3(depths=(2, 3, 4, 5, 6), trials: int = 3000, seed: int = 23) -> CheckResult:
    """The ceiling must clear tau, and the probe must not beat the ceiling."""
    rng = np.random.default_rng(seed)
    ceilings = {d: G.bag_of_operators_ceiling(rng, d, trials=trials) for d in depths}
    worst_ceiling = max(ceilings.values())

    probes = {d: bag_of_operators_probe(d) for d in depths}
    violations = [d for d in depths if probes[d] > ceilings[d] + 0.03]

    passed = worst_ceiling < TAU - G1_3_CEILING_MARGIN and not violations
    detail = "ceilings " + ", ".join(f"d{d}={ceilings[d]:.3f}" for d in depths)
    detail += " | probes " + ", ".join(f"d{d}={probes[d]:.3f}" for d in depths)
    if violations:
        detail += f" | PROBE ABOVE CEILING at depths {violations}"
    return CheckResult(
        "G1.3 order-blind ceiling below tau",
        passed,
        worst_ceiling,
        TAU - G1_3_CEILING_MARGIN,
        detail,
    )


# ---------------------------------------------------------------------------
# Reading trained gate runs
# ---------------------------------------------------------------------------


def read_summary(run_dir) -> dict:
    import json
    from pathlib import Path

    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Gate checks read stored runs, so the "
            f"training job must have completed first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def check_g1_1(looped_dir, ffwd_dir) -> list[CheckResult]:
    results = []
    for label, run_dir in (("looped k=1", looped_dir), ("matched feedforward", ffwd_dir)):
        summary = read_summary(run_dir)
        accuracy = summary["final_accuracy"]
        results.append(
            CheckResult(
                f"G1.1 depth 3 unsolvable at k=1, {label}",
                accuracy <= G1_1_MAX_ACCURACY,
                accuracy,
                G1_1_MAX_ACCURACY,
                f"chance {1 / 48:.4f}, n={summary['n_eval']}",
            )
        )
    return results


def check_g1_2(run_dir) -> CheckResult:
    summary = read_summary(run_dir)
    accuracy = summary["final_accuracy"]
    return CheckResult(
        "G1.2 depth 1 solvable at k=1",
        accuracy >= G1_2_MIN_ACCURACY,
        accuracy,
        G1_2_MIN_ACCURACY,
        f"n={summary['n_eval']}",
    )


def check_g1_4(run_dir) -> CheckResult:
    """Every breadth cell must clear the threshold, not just the average."""
    summary = read_summary(run_dir)
    cells = summary["by_cell"]
    worst_cell = min(cells, key=lambda c: cells[c]) if cells else None
    worst = cells[worst_cell] if worst_cell else 0.0
    return CheckResult(
        "G1.4 family C solvable at k=1 at every breadth",
        worst >= G1_4_MIN_ACCURACY,
        worst,
        G1_4_MIN_ACCURACY,
        f"worst cell {worst_cell}, all cells "
        + ", ".join(f"{c}={v:.3f}" for c, v in sorted(cells.items())),
    )
