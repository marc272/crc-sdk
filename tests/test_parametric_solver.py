"""The private solver interface and its HiGHS backend."""

from __future__ import annotations

import itertools
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import SolverUnavailableError, _solver


def _matrix(dense: list[list[float]]) -> _solver.SparseColumns:
    array = np.asarray(dense, dtype=np.float64)
    rows, columns = np.nonzero(array)
    return _solver.sparse_columns(rows, columns, array[rows, columns], array.shape)


def _lp(
    cost: list[float],
    dense: list[list[float]],
    upper_rows: list[float],
    *,
    col_upper: float = np.inf,
) -> _solver.Programme:
    n = len(cost)
    return _solver.Programme(
        cost=np.asarray(cost),
        col_lower=np.zeros(n),
        col_upper=np.full(n, col_upper),
        matrix=_matrix(dense),
        row_lower=np.full(len(upper_rows), -np.inf),
        row_upper=np.asarray(upper_rows),
    )


def test_sparse_columns() -> None:
    matrix = _matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])
    assert matrix.start.tolist() == [0, 1, 2, 4]
    assert matrix.index.tolist() == [0, 1, 0, 1]
    assert matrix.value.tolist() == [1.0, 3.0, 2.0, 4.0]
    assert matrix.n_columns == 3 and matrix.n_rows == 2
    with pytest.raises(ValueError, match="twice"):
        _solver.sparse_columns(
            np.array([0, 0]), np.array([1, 1]), np.array([1.0, 2.0]), (1, 2)
        )
    with pytest.raises(ValueError):
        _solver.SparseColumns(
            start=np.array([0, 1]), index=np.array([3]), value=np.array([1.0]), n_rows=2
        )
    with pytest.raises(ValueError):
        _solver.SparseColumns(
            start=np.array([1, 1]), index=np.array([0]), value=np.array([1.0]), n_rows=2
        )


def test_programme_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="one column"):
        _solver.Programme(
            cost=np.zeros(3),
            col_lower=np.zeros(3),
            col_upper=np.zeros(3),
            matrix=_matrix([[1.0, 1.0]]),
            row_lower=np.zeros(1),
            row_upper=np.zeros(1),
        )
    with pytest.raises(ValueError, match="finite"):
        _lp([np.nan, 1.0], [[1.0, 1.0]], [1.0])


def test_an_lp() -> None:
    # max x0 + 2 x1 subject to x0 + x1 <= 1: x = (0, 1).
    solution = _solver.solve_lp(_lp([-1.0, -2.0], [[1.0, 1.0]], [1.0]))
    assert solution.status == "optimal"
    assert solution.values is not None
    assert solution.values.tolist() == pytest.approx([0.0, 1.0])
    assert solution.objective == pytest.approx(-2.0)
    assert solution.message == "Optimal"
    assert solution.iterations >= 0 and solution.seconds >= 0.0


def test_infeasible_and_unbounded_lps() -> None:
    infeasible = _solver.Programme(
        cost=np.array([1.0]),
        col_lower=np.zeros(1),
        col_upper=np.full(1, np.inf),
        matrix=_matrix([[1.0]]),
        row_lower=np.array([2.0]),
        row_upper=np.array([1.0]),
    )
    solution = _solver.solve_lp(infeasible)
    assert solution.status in ("infeasible", "infeasible_or_unbounded")
    assert solution.values is None and solution.objective is None
    unbounded = _solver.solve_lp(_lp([-1.0, 0.0], [[0.0, 1.0]], [1.0]))
    assert unbounded.status in ("unbounded", "infeasible_or_unbounded")
    assert unbounded.values is None
    off = _solver.solve_lp(_lp([-1.0, 0.0], [[0.0, 1.0]], [1.0]), {"presolve": "off"})
    assert off.status == "unbounded"


def test_a_qp() -> None:
    # min x0^2 + x1^2 - 2 x0 - 2 x1 subject to x0 + x1 <= 1: x = (0.5, 0.5).
    hessian = _solver.sparse_columns(
        np.array([0, 1]), np.array([0, 1]), np.array([2.0, 2.0]), (2, 2)
    )
    programme = _solver.Programme(
        cost=np.array([-2.0, -2.0]),
        col_lower=np.zeros(2),
        col_upper=np.full(2, np.inf),
        matrix=_matrix([[1.0, 1.0]]),
        row_lower=np.array([-np.inf]),
        row_upper=np.array([1.0]),
        offset=2.0,
        hessian=hessian,
    )
    solution = _solver.solve_qp(programme)
    assert solution.status == "optimal"
    assert solution.values is not None
    assert solution.values.tolist() == pytest.approx([0.5, 0.5], abs=1e-9)
    assert solution.objective == pytest.approx(0.5)
    with pytest.raises(ValueError, match="hessian"):
        _solver.solve_lp(programme)
    with pytest.raises(ValueError, match="hessian"):
        _solver.solve_qp(_lp([1.0], [[1.0]], [1.0]))


def test_rejected_options_raise() -> None:
    with pytest.raises(ValueError, match="rejected the option"):
        _solver.solve_lp(_lp([1.0], [[1.0]], [1.0]), {"no_such_option": 1})


def test_identity() -> None:
    identity = _solver.identity()
    assert identity["name"] == "highs"
    major, minor, *_ = (int(part) for part in identity["version"].split("."))
    assert (major, minor) >= (1, 13)


def test_a_missing_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "highspy", None)
    with pytest.raises(SolverUnavailableError) as raised:
        _solver.require()
    assert raised.value.extra == "parametric"
    assert "pip install crc-sdk[parametric]" in str(raised.value)
    with pytest.raises(ImportError):
        _solver.solve_lp(_lp([1.0], [[1.0]], [1.0]))


def test_importing_the_interface_loads_no_solver() -> None:
    code = (
        "import sys, crc_sdk.parametric._solver, crc_sdk.parametric.design; "
        "print('highspy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


# -- limits and the dense active-set fallback --------------------------------

# Two QPs of an MSE design of a six-knot piecewise-linear payout under a budget
# and a limit, built from synthetic samples (Gumbel annual maxima through a
# tabulated damage curve, no basis risk), in scaled units: on them HiGHS 1.15's
# QP solver cycles until its iteration limit, or reports "unbounded" although
# every variable is bounded. Each is (cost, Hessian diagonal, Hessian below
# the diagonal, budget row, budget, upper bound).
STALLS = (
    [-0.01105739029501826, -0.03591255090484234, -0.047624440747157444]
    + [-0.08450488995076025, -0.10811452256107398, -0.020756010925984496],
    [0.09468294446334641, 0.20354686560211405, 0.12183508455511649]
    + [0.1439940830561492, 0.10792178428462679, 0.007875911899011701],
    [0.04530071578701637, 0.037344540044348305, 0.03933245975190873]
    + [0.03289342840755935, 0.01520051907898494],
    [0.0699918301251814, 0.14309606071673936, 0.09925604217568676]
    + [0.10810998560780863, 0.07800786588558553, 0.01153821548899832],
    0.015934528275732978,
    1.1935588452193666,
)
FALSE_UNBOUNDED = (
    [-0.016686303473545844, -0.046711983331344936, -0.04912441847203521]
    + [-0.0654177858272584, -0.06823000764017788, -0.01379980618529632],
    [0.12182608675966415, 0.25928888639147224, 0.14191770253564773]
    + [0.12160543969421655, 0.06938862459109667, 0.006120159486591107],
    [0.07814725509464474, 0.04546288200032437, 0.026698246775252774]
    + [0.021160977104313243, 0.008457189296120687],
    [0.09998667092715444, 0.19144951174322067, 0.10703941565561241]
    + [0.0847323317868913, 0.049503395495765296, 0.0072886743913558975],
    0.02732850397858923,
    1.1999329540205508,
)
# The last amount is reached by no draw: a zero row and column of the Hessian.
SEMIDEFINITE = (
    [-0.04799973441298773, -0.12170560558166621, -0.11897572060229658]
    + [-0.13740369954330306, -0.011441039807482034, 0.0],
    [0.1531680060438168, 0.30147750179173105, 0.1412533017547253]
    + [0.12654709999880898, 0.002277461399753872, 0.0],
    [0.10168234419902591, 0.049842056694223896, 0.036505199884597966]
    + [0.009608713727734258, 0.0],
    [0.12742517512142135, 0.22650095134249038, 0.11380027916677357]
    + [0.0863305068055706, 0.005943087563744065, 0.0],
    0.023242662644572344,
    2.692585289443289,
)


def _qp(
    hessian: np.ndarray,
    cost: list[float],
    dense: list[list[float]],
    row_upper: list[float],
    *,
    col_lower: list[float] | float = 0.0,
    col_upper: list[float] | float = np.inf,
    row_lower: list[float] | None = None,
) -> _solver.Programme:
    n = len(cost)
    lower = np.tril(np.asarray(hessian, dtype=np.float64))
    rows, columns = np.nonzero(lower)
    return _solver.Programme(
        cost=np.asarray(cost, dtype=np.float64),
        col_lower=np.broadcast_to(np.asarray(col_lower, dtype=np.float64), (n,)),
        col_upper=np.broadcast_to(np.asarray(col_upper, dtype=np.float64), (n,)),
        matrix=_matrix(dense)
        if dense
        else _solver.SparseColumns(
            np.zeros(n + 1, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0),
            0,
        ),
        row_lower=(
            np.full(len(row_upper), -np.inf)
            if row_lower is None
            else np.asarray(row_lower, dtype=np.float64)
        ),
        row_upper=np.asarray(row_upper, dtype=np.float64),
        hessian=_solver.sparse_columns(rows, columns, lower[rows, columns], (n, n)),
    )


def _design_qp(
    case: tuple[list[float], list[float], list[float], list[float], float, float],
) -> _solver.Programme:
    """A case's QP: budget row, then non-decreasing amounts, amounts in [0, upper]."""
    cost, diagonal, below, budget, bound, upper = case
    hessian = np.diag(diagonal) + np.diag(below, -1)
    order = [[0.0] * 6 for _ in range(5)]
    for i in range(5):
        order[i][i], order[i][i + 1] = 1.0, -1.0
    return _qp(hessian, cost, [budget, *order], [bound] + [0.0] * 5, col_upper=upper)


def _inequalities(programme: _solver.Programme) -> tuple[np.ndarray, np.ndarray]:
    """`G x <= h`, written out here rather than taken from the module."""
    n = programme.n_columns
    a = np.zeros((programme.matrix.n_rows, n))
    for j in range(n):
        span = slice(programme.matrix.start[j], programme.matrix.start[j + 1])
        a[programme.matrix.index[span], j] = programme.matrix.value[span]
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    for block, upper, lower in (
        (a, programme.row_upper, programme.row_lower),
        (np.eye(n), programme.col_upper, programme.col_lower),
    ):
        for row, high, low in zip(block, upper, lower, strict=True):
            if np.isfinite(high):
                rows.append(row)
                bounds.append(high)
            if np.isfinite(low):
                rows.append(-row)
                bounds.append(-low)
    return np.array(rows).reshape(-1, n), np.array(bounds)


def _hessian(programme: _solver.Programme) -> np.ndarray:
    assert programme.hessian is not None
    n = programme.n_columns
    full = np.zeros((n, n))
    for j in range(n):
        span = slice(programme.hessian.start[j], programme.hessian.start[j + 1])
        full[programme.hessian.index[span], j] = programme.hessian.value[span]
    return full + np.tril(full, -1).T


def _reference_optimum(programme: _solver.Programme) -> float:
    """The optimum by enumerating active sets: an exact, independent solve.

    Every set of at most n independent rows gives an equality-constrained
    QP; a solution of its KKT system that meets every row with multipliers
    >= 0 is a KKT point of the QP, and so a global minimum of the convex QP.
    """
    q, c = _hessian(programme), programme.cost
    g, h = _inequalities(programme)
    n, best = programme.n_columns, math.inf
    for size in range(n + 1):
        for chosen in itertools.combinations(range(h.shape[0]), size):
            rows = g[list(chosen)]
            if size and np.linalg.matrix_rank(rows) < size:
                continue
            kkt = np.block([[q, rows.T], [rows, np.zeros((size, size))]])
            rhs = np.concatenate([-c, h[list(chosen)]])
            solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
            if np.abs(kkt @ solution - rhs).max() > 1e-10:
                continue
            x, multipliers = solution[:n], solution[n:]
            if (g @ x - h).max(initial=-1.0) > 1e-10 or (multipliers < -1e-12).any():
                continue
            best = min(best, 0.5 * x @ q @ x + c @ x + programme.offset)
    return best


_REFERENCES: dict[int, float] = {}


def _case_optimum(case: Any) -> float:
    """`_reference_optimum` of a case's QP, computed once per case."""
    if id(case) not in _REFERENCES:
        _REFERENCES[id(case)] = _reference_optimum(_design_qp(case))
    return _REFERENCES[id(case)]


def _no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_solver, "ACTIVE_SET_MAX_COLUMNS", 0)


def _force_highs_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _solver, "QP_OPTIONS", {**_solver.QP_OPTIONS, "qp_iteration_limit": 1}
    )


def test_every_qp_solve_has_a_limit() -> None:
    assert _solver.QP_OPTIONS["qp_iteration_limit"] == _solver.QP_ITERATION_LIMIT
    assert _solver.QP_OPTIONS["time_limit"] == _solver.QP_TIME_LIMIT
    assert 0 < _solver.QP_ITERATION_LIMIT < 2**31 - 1
    assert 0.0 < _solver.QP_TIME_LIMIT < math.inf
    assert _solver.ACTIVE_SET_ITERATION_LIMIT > 0


@pytest.mark.skipif(sys.platform == "win32", reason="TEMPORARY diagnostic")
@pytest.mark.parametrize(
    "case", [STALLS, FALSE_UNBOUNDED, SEMIDEFINITE], ids=["stall", "unbounded", "psd"]
)
def test_qps_that_highs_leaves_unsolved_are_solved(case: Any) -> None:
    programme = _design_qp(case)
    solution = _solver.solve_qp(programme)
    assert solution.status == "optimal"
    assert solution.values is not None and solution.objective is not None
    # Whether HiGHS fails on these depends on its version; the outcome doesn't.
    assert solution.method in ("highs", "active_set")
    tolerance = 1e-12 if solution.method == "active_set" else 1e-7
    if solution.method == "active_set":
        assert "by the active-set fallback after HiGHS" in solution.message
    reference = _case_optimum(case)
    assert solution.objective == pytest.approx(reference, rel=tolerance, abs=1e-15)
    g, h = _inequalities(programme)
    assert (g @ solution.values - h).max() <= tolerance


def test_a_qp_stops_at_its_iteration_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_fallback(monkeypatch)
    solution = _solver.solve_qp(_design_qp(STALLS), {"qp_iteration_limit": 3})
    assert solution.status == "limit"
    assert solution.limit == "qp_iteration_limit"
    assert "qp_iteration_limit=3" in solution.message
    assert "more than the active-set fallback takes (0)" in solution.message
    assert solution.values is None and solution.method == "highs"
    assert solution.iterations >= 3


def test_a_qp_stops_at_its_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_fallback(monkeypatch)
    solution = _solver.solve_qp(_design_qp(STALLS), {"time_limit": 0.0})
    assert (solution.status, solution.limit) == ("limit", "time_limit")
    assert "time_limit=0.0" in solution.message


def test_an_lp_names_its_limit() -> None:
    solution = _solver.solve_lp(
        _lp([-1.0, -2.0], [[1.0, 1.0]], [1.0]),
        {"simplex_iteration_limit": 0, "presolve": "off"},
    )
    assert (solution.status, solution.limit) == ("limit", "simplex_iteration_limit")


def test_the_fallback_solves_a_qp_that_highs_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution = _solver.solve_qp(_design_qp(FALSE_UNBOUNDED), {"qp_iteration_limit": 1})
    assert (solution.status, solution.method) == ("optimal", "active_set")
    assert solution.message.startswith("Optimal, by the active-set fallback after")
    assert "qp_iteration_limit=1" in solution.message
    assert solution.iterations > 1
    assert solution.objective == pytest.approx(
        _case_optimum(FALSE_UNBOUNDED), rel=1e-12
    )


def test_the_fallback_has_its_own_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_solver, "ACTIVE_SET_ITERATION_LIMIT", 1)
    solution = _solver.solve_qp(_design_qp(STALLS), {"qp_iteration_limit": 1})
    assert (solution.status, solution.method) == ("limit", "active_set")
    assert solution.limit == "active_set_iteration_limit"
    assert "active_set_iteration_limit=1" in solution.message


def test_a_false_unbounded_is_solved_again(monkeypatch: pytest.MonkeyPatch) -> None:
    # HiGHS's QP outcome is replaced by "unbounded"; its LPs run as they are.
    real = _solver._run

    def unbounded(programme: _solver.Programme, *args: Any) -> _solver.Solution:
        if programme.hessian is None:
            return real(programme, *args)
        return _solver.Solution("unbounded", None, None, "Unbounded", 7, 0.0)

    monkeypatch.setattr(_solver, "_run", unbounded)
    solution = _solver.solve_qp(_design_qp(STALLS))
    assert (solution.status, solution.method) == ("optimal", "active_set")
    assert solution.message.endswith("after HiGHS: Unbounded")
    assert solution.objective == pytest.approx(_case_optimum(STALLS), rel=1e-12)


def _random_qp(rng: np.random.Generator) -> _solver.Programme:
    n = int(rng.integers(2, 5))
    rank = int(rng.integers(1, n + 1))
    factor = rng.normal(size=(rank, n))
    m = int(rng.integers(1, 4))
    dense = rng.normal(size=(m, n))
    dense[np.abs(dense) < 0.3] = 0.0
    # Rows through a known feasible point, some of them tight there, so that
    # optima at degenerate vertices are common.
    point = rng.uniform(0.0, 1.0, n)
    row_upper = dense @ point + rng.choice([0.0, 0.5], m)
    return _qp(
        factor.T @ factor,
        list(rng.normal(size=n)),
        dense.tolist(),
        list(row_upper),
        col_lower=list(np.where(rng.random(n) < 0.8, 0.0, -np.inf)),
        col_upper=list(np.where(rng.random(n) < 0.8, 2.0, np.inf)),
    )


def test_the_fallback_matches_an_exact_solve_on_random_qps() -> None:
    rng = np.random.default_rng(20261111)
    for _ in range(40):
        programme = _random_qp(rng)
        solution = _solver._active_set(programme, _solver.ACTIVE_SET_ITERATION_LIMIT)
        reference = _reference_optimum(programme)
        if math.isinf(reference):
            assert solution.status == "unbounded"
            continue
        assert solution.status == "optimal" and solution.objective is not None
        assert solution.objective == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_the_fallback_at_a_degenerate_vertex() -> None:
    # min (x0 - 1)^2 + (x1 - 1)^2 on x0 + x1 <= 1, x0 <= 0.5, x1 <= 0.5,
    # x0 - x1 <= 0 and x1 - x0 <= 0: four rows meet at the optimum (0.5, 0.5).
    programme = _qp(
        np.eye(2) * 2.0,
        [-2.0, -2.0],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 1.0]],
        [1.0, 0.5, 0.5, 0.0, 0.0],
        col_lower=-np.inf,
    )
    solution = _solver._active_set(programme, 100)
    assert solution.status == "optimal" and solution.values is not None
    assert solution.values.tolist() == pytest.approx([0.5, 0.5], abs=1e-12)


def test_the_fallback_follows_directions_without_curvature() -> None:
    # min x0^2 - x1: x1 has no curvature and is bounded above by x0 + x1 <= 3.
    hessian = np.diag([2.0, 0.0])
    bounded = _qp(hessian, [0.0, -1.0], [[1.0, 1.0]], [3.0], col_lower=-np.inf)
    solution = _solver._active_set(bounded, 100)
    assert solution.status == "optimal" and solution.values is not None
    assert solution.values.tolist() == pytest.approx([-0.5, 3.5], abs=1e-12)
    unbounded = _qp(hessian, [0.0, -1.0], [[1.0, 0.0]], [3.0], col_lower=-np.inf)
    assert _solver._active_set(unbounded, 100).status == "unbounded"


def test_the_fallback_reports_an_infeasible_qp() -> None:
    programme = _qp(np.eye(1), [0.0], [[1.0]], [-1.0])
    solution = _solver._active_set(programme, 100)
    assert (solution.status, solution.method) == ("infeasible", "active_set")
    assert solution.values is None


def test_the_fallback_without_constraints() -> None:
    programme = _qp(np.eye(2), [-1.0, 2.0], [], [], col_lower=-np.inf)
    solution = _solver._active_set(programme, 100)
    assert solution.status == "optimal" and solution.values is not None
    assert solution.values.tolist() == pytest.approx([1.0, -2.0], abs=1e-12)
