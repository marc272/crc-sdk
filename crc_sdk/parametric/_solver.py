"""The private solver interface of `design`, and its HiGHS backend.

`solve_lp` and `solve_qp` take a `Programme`: plain numpy arrays, with the
constraint matrix and the Hessian in compressed sparse column form
(`SparseColumns`: column pointers, row indices, values). No scipy. They
return a `Solution`: the status, the values, the objective and the solver's
own message, iterations and time, and on request an LP's optimal `Basis`.
An `LPSession` solves a sequence of LPs that differ only in their costs and
row bounds on one solver instance, each from the previous basis, and the
first from a given basis. `identity()` names the solver and its version for
the design record. So the backend can change without a change of the public
API.

The backend is HiGHS through `highspy`, from the optional extra
`crc-sdk[parametric]`. It is imported lazily: importing this module needs no
solver, and `require()` raises `SolverUnavailableError`, naming the extra,
when it is missing.

A programme is `minimise 1/2 x'Qx + c'x + offset` subject to `row_lower <=
A x <= row_upper` and `col_lower <= x <= col_upper`, with infinite bounds as
`inf`. `Q` is given as its lower triangle; an LP has none.

**Limits.** Every QP solve has a limit: HiGHS stops at `QP_ITERATION_LIMIT`
iterations or `QP_TIME_LIMIT` seconds, whichever comes first, with the status
`"limit"` and the limit's name in `Solution.limit`. HiGHS's active-set QP
solver can cycle at a degenerate vertex, and without a limit it never returns.

**The fallback.** When HiGHS's QP solver stops at its limit, or ends with any
outcome but optimal or infeasible (it has reported "unbounded" for a QP whose
variables are all bounded), `solve_qp` solves a QP of at most
`ACTIVE_SET_MAX_COLUMNS` columns again with a dense active-set method in numpy:
from a feasible point found by HiGHS's simplex, it minimises the objective on
the face of its working constraints, and at the minimum of a face it asks a
small LP, solved by HiGHS's simplex, for a feasible descent direction over
every active constraint. A degenerate vertex is thus left through an LP, whose
simplex does not cycle, and every step along an LP's direction lowers the
objective, so no face is visited twice. It stops when that LP finds no descent
direction, which is the optimality condition of a convex QP, or at
`ACTIVE_SET_ITERATION_LIMIT` iterations. `Solution.method` says which method
gave the outcome: `"highs"` or `"active_set"`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.errors import SolverUnavailableError

__all__ = [
    "ACTIVE_SET_ITERATION_LIMIT",
    "ACTIVE_SET_MAX_COLUMNS",
    "EXTRA",
    "LP_OPTIONS",
    "LPSession",
    "QP_ITERATION_LIMIT",
    "QP_OPTIONS",
    "QP_TIME_LIMIT",
    "RANGING_OPTIONS",
    "SOLVER_NAME",
    "Basis",
    "Method",
    "Programme",
    "Solution",
    "SolveStatus",
    "SparseColumns",
    "identity",
    "require",
    "solve_lp",
    "solve_qp",
    "sparse_columns",
]

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int32]

SOLVER_NAME = "highs"
"""The solver's name in design records."""

EXTRA = "parametric"
"""The optional dependency set that installs the solver."""

LP_OPTIONS: Mapping[str, bool | int | float | str] = {"solver": "simplex"}
"""HiGHS options of a design LP: the simplex, so a solution is a vertex."""

QP_ITERATION_LIMIT = 100_000
"""The most iterations of HiGHS's QP solver in one solve (`qp_iteration_limit`).

A design QP of one column per amount needs tens of iterations, and the QP of
a cost-of-capital premium at 50,000 draws about 1,300. A cycling solve runs
this many in about a quarter of a second on a QP of six amounts.
"""

QP_TIME_LIMIT = 600.0
"""The most seconds of one HiGHS QP solve (`time_limit`), a backstop."""

QP_OPTIONS: Mapping[str, bool | int | float | str] = {
    "qp_regularization_value": 1e-12,
    "qp_iteration_limit": QP_ITERATION_LIMIT,
    "time_limit": QP_TIME_LIMIT,
}
"""HiGHS options of a design QP.

HiGHS adds `qp_regularization_value` (1e-7 by default) to the Hessian's
diagonal, which moves a solution by about that share of its scale; 1e-12
keeps the semi-definite design QPs solvable without that bias. The limits
stop a solve that cycles (`QP_ITERATION_LIMIT`, `QP_TIME_LIMIT`).
"""

ACTIVE_SET_MAX_COLUMNS = 500
"""The most columns of a QP that the dense active-set fallback solves."""

ACTIVE_SET_ITERATION_LIMIT = 10_000
"""The most iterations of the dense active-set fallback in one solve."""

RANGING_OPTIONS: Mapping[str, bool | int | float | str] = {
    "solver": "simplex",
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}
"""HiGHS options of the LPs that range an optimal set (its defaults are 1e-7)."""

SolveStatus = Literal[
    "optimal", "infeasible", "unbounded", "infeasible_or_unbounded", "limit", "error"
]
"""A solve's outcome; only `"optimal"` comes with values.

`"limit"`: the solve stopped at an iteration or time limit, which
`Solution.limit` names.
"""

Method = Literal["highs", "active_set"]
"""What gave a solve's outcome: HiGHS, or the dense active-set fallback."""

_REJECTED = "HiGHS rejected the model"

# The dense active-set fallback works on rows of unit length. A row is active
# when its slack is at most _ACTIVE times max(1, |x|); rows are independent
# above _RANK times the largest singular value; a direction has no curvature
# below _CURVATURE times the Hessian's largest entry; and a slope below
# _SLOPE times the gradient's scale is no descent.
_ACTIVE = 1e-9
_RANK = 1e-10
_CURVATURE = 1e-12
_SLOPE = 1e-9
_KEEP = 1e-8


def require() -> ModuleType:
    """Return the `highspy` module, or raise `SolverUnavailableError`.

    The error is an `ImportError` that names the extra to install.
    """
    try:
        import highspy
    except ImportError as error:
        raise SolverUnavailableError(
            "designing a cover needs the HiGHS solver: "
            f"`pip install crc-sdk[{EXTRA}]` (it installs highspy)",
            extra=EXTRA,
        ) from error
    return highspy


def identity() -> dict[str, str]:
    """Return `{"name": "highs", "version": ...}` of the installed solver."""
    highspy = require()
    return {"name": SOLVER_NAME, "version": str(highspy.Highs().version())}


@dataclass(frozen=True, eq=False)
class SparseColumns:
    """A matrix in compressed sparse column form, as plain numpy arrays.

    Column `j` holds `value[start[j]:start[j + 1]]` in the rows
    `index[start[j]:start[j + 1]]`. `n_rows` is the number of rows.
    """

    start: IndexArray
    index: IndexArray
    value: FloatArray
    n_rows: int

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=np.int32)
        index = np.asarray(self.index, dtype=np.int32)
        value = np.asarray(self.value, dtype=np.float64)
        if start.ndim != 1 or start.shape[0] == 0 or start[0] != 0:
            raise ValueError("start must be a 1-D array that begins with 0")
        if (np.diff(start) < 0).any() or start[-1] != index.shape[0]:
            raise ValueError("start must be non-decreasing and end at the entries")
        if index.shape != value.shape or index.ndim != 1:
            raise ValueError("index and value must be 1-D arrays of one length")
        if index.size and (index.min() < 0 or index.max() >= self.n_rows):
            raise ValueError("row indices must lie in [0, n_rows)")
        if not np.isfinite(value).all():
            raise ValueError("matrix values must be finite")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "value", value)

    @property
    def n_columns(self) -> int:
        """The number of columns."""
        return int(self.start.shape[0] - 1)


def sparse_columns(
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    values: FloatArray,
    shape: tuple[int, int],
) -> SparseColumns:
    """Return the matrix of the entries `(rows[i], columns[i], values[i])`.

    Zero values are dropped; repeated positions are not allowed.
    """
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    keep = values != 0.0
    rows, columns, values = rows[keep], columns[keep], values[keep]
    order = np.lexsort((rows, columns))
    rows, columns, values = rows[order], columns[order], values[order]
    if rows.size > 1:
        same = (np.diff(rows) == 0) & (np.diff(columns) == 0)
        if same.any():
            raise ValueError("a matrix position is given twice")
    counts = np.bincount(columns, minlength=shape[1])
    start = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return SparseColumns(
        start=start,
        index=rows.astype(np.int32),
        value=values,
        n_rows=shape[0],
    )


@dataclass(frozen=True, eq=False)
class Programme:
    """`min 1/2 x'Qx + c'x + offset` s.t. `row_lower <= A x <= row_upper`, bounds.

    `cost` is `c`, one entry per column; `matrix` is `A`; `hessian` is the
    lower triangle of `Q` (square, one row and column per variable), or
    `None` for an LP. Infinite bounds are `inf` or `-inf`.
    """

    cost: FloatArray
    col_lower: FloatArray
    col_upper: FloatArray
    matrix: SparseColumns
    row_lower: FloatArray
    row_upper: FloatArray
    offset: float = 0.0
    hessian: SparseColumns | None = None

    def __post_init__(self) -> None:
        n = int(np.asarray(self.cost).shape[0])
        for name in ("cost", "col_lower", "col_upper"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (n,):
                raise ValueError(f"{name} must have one entry per column")
            object.__setattr__(self, name, array)
        if self.matrix.n_columns != n:
            raise ValueError("the matrix must have one column per variable")
        for name in ("row_lower", "row_upper"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (self.matrix.n_rows,):
                raise ValueError(f"{name} must have one entry per row")
            object.__setattr__(self, name, array)
        if not np.isfinite(self.cost).all() or not math.isfinite(self.offset):
            raise ValueError("cost and offset must be finite")
        if self.hessian is not None and (
            self.hessian.n_columns != n or self.hessian.n_rows != n
        ):
            raise ValueError("the hessian must be square, one row per variable")

    @property
    def n_columns(self) -> int:
        """The number of variables."""
        return int(self.cost.shape[0])


@dataclass(frozen=True, eq=False)
class Basis:
    """The optimal simplex basis of a solved LP, to start another solve from.

    It is opaque: `handle` is the solver's own copy of the basis, for a
    programme of `n_columns` variables and `n_rows` rows.
    """

    n_columns: int
    n_rows: int
    handle: Any = field(repr=False)


@dataclass(frozen=True, eq=False)
class Solution:
    """The outcome of one solve.

    `status` is `"optimal"` with `values` (one per variable) and `objective`
    (including the offset); otherwise both are `None`. `message` is the
    solver's own status text, `iterations` its iteration count (simplex,
    interior point and QP iterations together) and `seconds` the wall time.
    `basis` is the optimal basis of an LP solved with `keep_basis=True`, and
    `None` otherwise, or when the solver has no valid basis to give.
    `method` is `"highs"`, or `"active_set"` when the dense fallback gave the
    outcome (its `iterations` and `seconds` then include HiGHS's). `limit`
    names the limit a `"limit"` status stopped at: `"qp_iteration_limit"`,
    `"simplex_iteration_limit"`, `"time_limit"` or
    `"active_set_iteration_limit"`; the message gives its value.
    """

    status: SolveStatus
    values: FloatArray | None
    objective: float | None
    message: str
    iterations: int
    seconds: float
    basis: Basis | None = None
    method: Method = "highs"
    limit: str | None = None


def _options(
    highs: Any, base: Mapping[str, Any], options: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Set the options on a HiGHS instance; return them."""
    chosen = {"output_flag": False, **base, **(options or {})}
    for name, value in chosen.items():
        status = highs.setOptionValue(name, value)
        if status != require().HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected the option {name}={value!r}")
    return chosen


def _pass(highs: Any, programme: Programme) -> bool:
    """Pass `programme` to a HiGHS instance; return whether HiGHS accepted it."""
    highspy = require()
    matrix = programme.matrix
    hessian = programme.hessian
    integrality = np.zeros(programme.n_columns, dtype=np.int32)
    colwise = int(highspy.MatrixFormat.kColwise)
    minimise = int(highspy.ObjSense.kMinimize)
    common = (
        programme.cost,
        programme.col_lower,
        programme.col_upper,
        programme.row_lower,
        programme.row_upper,
        matrix.start,
        matrix.index,
        matrix.value,
    )
    if hessian is None:
        passed = highs.passModel(
            programme.n_columns,
            matrix.n_rows,
            int(matrix.index.shape[0]),
            colwise,
            minimise,
            float(programme.offset),
            *common,
            integrality,
        )
    else:
        passed = highs.passModel(
            programme.n_columns,
            matrix.n_rows,
            int(matrix.index.shape[0]),
            int(hessian.index.shape[0]),
            colwise,
            int(highspy.HessianFormat.kTriangular),
            minimise,
            float(programme.offset),
            *common,
            hessian.start,
            hessian.index,
            hessian.value,
            integrality,
        )
    return bool(passed != highspy.HighsStatus.kError)


def _solved(
    highs: Any,
    n_columns: int,
    chosen: Mapping[str, Any],
    *,
    keep_basis: bool = False,
    qp: bool = False,
) -> Solution:
    """Run a HiGHS instance that holds a model, and read its outcome.

    `chosen` are the options it runs with, to name a limit it stops at. With
    `keep_basis`, an optimal outcome carries the instance's basis.
    """
    highspy = require()
    began = time.perf_counter()
    ran = highs.run()
    seconds = time.perf_counter() - began
    model_status = highs.getModelStatus()
    message = str(highs.modelStatusToString(model_status))
    info = highs.getInfo()
    iterations = sum(
        max(0, int(getattr(info, name)))
        for name in (
            "simplex_iteration_count",
            "ipm_iteration_count",
            "qp_iteration_count",
            "crossover_iteration_count",
        )
    )
    statuses: dict[Any, SolveStatus] = {
        highspy.HighsModelStatus.kOptimal: "optimal",
        highspy.HighsModelStatus.kInfeasible: "infeasible",
        highspy.HighsModelStatus.kUnbounded: "unbounded",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "infeasible_or_unbounded",
        highspy.HighsModelStatus.kIterationLimit: "limit",
        highspy.HighsModelStatus.kTimeLimit: "limit",
    }
    status: SolveStatus = statuses.get(model_status, "error")
    if ran == highspy.HighsStatus.kError and status != "limit":
        status = "error"
    if status == "limit":
        limit = (
            "time_limit"
            if model_status == highspy.HighsModelStatus.kTimeLimit
            else "qp_iteration_limit"
            if qp
            else "simplex_iteration_limit"
        )
        return Solution(
            "limit",
            None,
            None,
            f"{message} (HiGHS, {limit}={chosen.get(limit)!r})",
            iterations,
            seconds,
            limit=limit,
        )
    if status != "optimal":
        return Solution(status, None, None, message, iterations, seconds)
    values = np.asarray(highs.getSolution().col_value, dtype=np.float64)
    objective = float(info.objective_function_value)
    if values.shape != (n_columns,) or not np.isfinite(values).all():
        return Solution(
            "error",
            None,
            None,
            f"{message}, without finite values",
            iterations,
            seconds,
        )
    basis: Basis | None = None
    if keep_basis:
        kept = highs.getBasis()
        if kept.valid:
            basis = Basis(n_columns, int(highs.getNumRow()), kept)
    return Solution(status, values, objective, message, iterations, seconds, basis)


def _run(
    programme: Programme,
    base: Mapping[str, Any],
    options: Mapping[str, Any] | None,
    keep_basis: bool = False,
) -> Solution:
    highs = require().Highs()
    chosen = _options(highs, base, options)
    if not _pass(highs, programme):
        return Solution("error", None, None, _REJECTED, 0, 0.0)
    return _solved(
        highs,
        programme.n_columns,
        chosen,
        keep_basis=keep_basis,
        qp=programme.hessian is not None,
    )


class LPSession:
    """One HiGHS instance for a sequence of LPs on one programme.

    The LPs differ only in their costs and row bounds. `LPSession(programme,
    options, basis)` passes the programme to HiGHS once; `add_row`,
    `change_row_bounds` and `change_cost` change the model in place, and
    `solve()` solves it from the basis of the previous solve, a warm start
    that skips the presolve. The first solve starts from `basis`, an optimal
    basis of the same programme (`solve_lp(..., keep_basis=True)`), if one is
    given, and from scratch otherwise; a row added after it enters the basis
    as a basic row. `options` are HiGHS options on top of `LP_OPTIONS`, as
    for `solve_lp`, and hold for every solve.

    Raises `SolverUnavailableError` without highspy, and `ValueError` for a
    programme with a Hessian, an option HiGHS rejects, a basis of another
    shape or one HiGHS rejects, or a change of the wrong shape. A model
    HiGHS rejects gives `solve()` the status `"error"`.
    """

    def __init__(
        self,
        programme: Programme,
        options: Mapping[str, bool | int | float | str] | None = None,
        basis: Basis | None = None,
    ) -> None:
        if programme.hessian is not None:
            raise ValueError("an LPSession takes a programme without a hessian")
        if basis is not None and (basis.n_columns, basis.n_rows) != (
            programme.n_columns,
            programme.matrix.n_rows,
        ):
            raise ValueError(
                f"the basis is for {basis.n_columns} columns and {basis.n_rows} "
                f"rows, the programme has {programme.n_columns} and "
                f"{programme.matrix.n_rows}"
            )
        self._highs = require().Highs()
        self._base: dict[str, Any] = {**LP_OPTIONS, **(options or {})}
        _options(self._highs, self._base, None)
        self._n_columns = programme.n_columns
        self._n_rows = programme.matrix.n_rows
        self._accepted = _pass(self._highs, programme)
        if self._accepted and basis is not None:
            self._check(self._highs.setBasis(basis.handle), "the basis")

    @property
    def n_rows(self) -> int:
        """The number of rows, the added ones included."""
        return self._n_rows

    def _check(self, status: Any, what: str) -> None:
        if status == require().HighsStatus.kError:
            raise ValueError(f"HiGHS rejected {what}")

    def add_row(self, coefficients: FloatArray, lower: float, upper: float) -> int:
        """Add the row `lower <= coefficients @ x <= upper`; return its index."""
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (self._n_columns,):
            raise ValueError("a row needs one coefficient per column")
        if not np.isfinite(coefficients).all():
            raise ValueError("row coefficients must be finite")
        nonzero = np.flatnonzero(coefficients)
        self._check(
            self._highs.addRow(
                float(lower),
                float(upper),
                int(nonzero.shape[0]),
                nonzero.astype(np.int32),
                coefficients[nonzero],
            ),
            "the row",
        )
        self._n_rows += 1
        return self._n_rows - 1

    def change_row_bounds(self, row: int, lower: float, upper: float) -> None:
        """Set the bounds of row `row` to `lower` and `upper`."""
        if not 0 <= row < self._n_rows:
            raise ValueError(f"row {row} is not in [0, {self._n_rows})")
        self._check(
            self._highs.changeRowBounds(int(row), float(lower), float(upper)),
            f"the bounds of row {row}",
        )

    def change_cost(self, cost: FloatArray) -> None:
        """Set the cost vector, one finite entry per column."""
        cost = np.asarray(cost, dtype=np.float64)
        if cost.shape != (self._n_columns,):
            raise ValueError("cost must have one entry per column")
        if not np.isfinite(cost).all():
            raise ValueError("cost must be finite")
        self._check(
            self._highs.changeColsCost(
                self._n_columns, np.arange(self._n_columns, dtype=np.int32), cost
            ),
            "the cost",
        )

    def solve(
        self, options: Mapping[str, bool | int | float | str] | None = None
    ) -> Solution:
        """Solve the current model; `options` hold for this solve only."""
        if not self._accepted:
            return Solution("error", None, None, _REJECTED, 0, 0.0)
        self._highs.resetOptions()
        chosen = _options(self._highs, self._base, options)
        return _solved(self._highs, self._n_columns, chosen)


def solve_lp(
    programme: Programme,
    options: Mapping[str, bool | int | float | str] | None = None,
    *,
    keep_basis: bool = False,
) -> Solution:
    """Solve a linear programme with HiGHS's simplex.

    `options` are HiGHS options on top of `LP_OPTIONS`. With `keep_basis`,
    an optimal solution carries its basis, for an `LPSession` to start from.
    Raises `SolverUnavailableError` without highspy, and `ValueError` for a
    programme with a Hessian or an option HiGHS rejects.
    """
    if programme.hessian is not None:
        raise ValueError("solve_lp takes a programme without a hessian")
    return _run(programme, LP_OPTIONS, options, keep_basis)


def solve_qp(
    programme: Programme, options: Mapping[str, bool | int | float | str] | None = None
) -> Solution:
    """Solve a convex quadratic programme with HiGHS, or with the fallback.

    The Hessian must be positive semi-definite. `options` are HiGHS options
    on top of `QP_OPTIONS`, whose limits stop a solve that cycles. When HiGHS
    ends with any outcome but optimal or infeasible, a programme of at most
    `ACTIVE_SET_MAX_COLUMNS` columns is solved again by the dense active-set
    fallback (the module docstring describes it), and its outcome is
    returned with `method="active_set"`; a larger one keeps HiGHS's outcome.
    Raises `SolverUnavailableError` without highspy, and `ValueError` for a
    programme without a Hessian or an option HiGHS rejects.
    """
    if programme.hessian is None:
        raise ValueError("solve_qp takes a programme with a hessian")
    highs = _run(programme, QP_OPTIONS, options)
    if highs.status in ("optimal", "infeasible") or highs.message == _REJECTED:
        return highs
    if programme.n_columns > ACTIVE_SET_MAX_COLUMNS:
        return replace(
            highs,
            message=f"{highs.message}; the QP has {programme.n_columns} columns, "
            f"more than the active-set fallback takes ({ACTIVE_SET_MAX_COLUMNS})",
        )
    fallback = _active_set(programme, ACTIVE_SET_ITERATION_LIMIT)
    return replace(
        fallback,
        message=f"{fallback.message}, by the active-set fallback after HiGHS: "
        f"{highs.message}",
        iterations=highs.iterations + fallback.iterations,
        seconds=highs.seconds + fallback.seconds,
    )


# -- the dense active-set fallback --------------------------------------------


def _dense(matrix: SparseColumns) -> FloatArray:
    """A `SparseColumns` matrix as a dense array."""
    dense = np.zeros((matrix.n_rows, matrix.n_columns), dtype=np.float64)
    columns = np.repeat(np.arange(matrix.n_columns), np.diff(matrix.start))
    dense[matrix.index, columns] = matrix.value
    return dense


def _inequalities(programme: Programme) -> tuple[FloatArray, FloatArray]:
    """The programme's finite row and column bounds as `G x <= h`.

    Each row of `G` has length 1; rows of zeros are left out.
    """
    n = programme.n_columns
    matrix = _dense(programme.matrix)
    identity = np.eye(n)
    rows: list[FloatArray] = [np.zeros((0, n))]
    bounds: list[FloatArray] = [np.zeros(0)]
    for block, bound, sign in (
        (matrix, programme.row_upper, 1.0),
        (matrix, programme.row_lower, -1.0),
        (identity, programme.col_upper, 1.0),
        (identity, programme.col_lower, -1.0),
    ):
        finite = np.isfinite(bound)
        rows.append(sign * block[finite])
        bounds.append(sign * bound[finite])
    g, h = np.vstack(rows), np.concatenate(bounds)
    norms = np.linalg.norm(g, axis=1)
    keep = norms > 0.0
    return g[keep] / norms[keep, None], h[keep] / norms[keep]


def _row_space(rows: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Orthonormal bases of the row space of `rows` and of its null space."""
    n = rows.shape[1]
    if rows.shape[0] == 0:
        return np.zeros((0, n)), np.eye(n)
    _, singular, vt = np.linalg.svd(rows)
    rank = int(np.count_nonzero(singular > _RANK * singular[0]))
    return vt[:rank], vt[rank:].T


def _face_step(
    hessian: FloatArray,
    gradient: FloatArray,
    rows: FloatArray,
    curvature: float,
    slope: float,
) -> tuple[FloatArray, bool]:
    """The step to the minimum on the face `rows @ step = 0`.

    Returns the Newton step and `False`; or, when a direction without
    curvature has a slope above `slope`, a descent direction within those
    directions and `True` (the face is then not bounded below by its own
    terms).
    """
    _, null = _row_space(rows)
    if null.shape[1] == 0:
        return np.zeros(gradient.shape[0]), False
    reduced = null.T @ hessian @ null
    values, vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    coordinates = vectors.T @ (null.T @ gradient)
    curved = values > curvature
    flat = coordinates[~curved]
    if (np.abs(flat) > slope).any():
        return -(null @ (vectors[:, ~curved] @ flat)), True
    newton = vectors[:, curved] @ (coordinates[curved] / values[curved])
    return -(null @ newton), False


def _line_minimum(
    hessian: FloatArray, gradient: FloatArray, direction: FloatArray, curvature: float
) -> float:
    """The step to the minimum along a descent direction, or `inf`."""
    bend = float(direction @ hessian @ direction)
    if bend <= curvature * float(direction @ direction):
        return math.inf
    return -float(gradient @ direction) / bend


def _ratio(
    rows: FloatArray,
    bounds: FloatArray,
    x: FloatArray,
    direction: FloatArray,
    exclude: Any,
    cap: float,
) -> tuple[float, int | None]:
    """The longest step up to `cap` along `direction` that keeps `rows @ x <= bounds`.

    Rows in `exclude` are not tested. Returns the step and the row that
    blocks it, or `None` when `cap` is reached first.
    """
    along = rows @ direction
    moving = along > 1e-14 * max(1.0, float(np.abs(direction).max(initial=0.0)))
    moving[np.asarray(exclude, dtype=np.int64)] = False
    if not moving.any():
        return cap, None
    candidates = np.flatnonzero(moving)
    slack = np.maximum(bounds[candidates] - rows[candidates] @ x, 0.0)
    ratios = slack / along[candidates]
    first = int(np.argmin(ratios))
    if ratios[first] < cap:
        return float(ratios[first]), int(candidates[first])
    return cap, None


def _descent(
    gradient: FloatArray, rows: FloatArray, scale: float
) -> tuple[FloatArray, float] | None:
    """The most descending direction in the box [-1, 1] with `rows @ d <= 0`.

    An LP, solved by HiGHS's simplex at tight tolerances. Returns the
    direction and its slope divided by `scale`, or `None` if the LP fails.
    """
    m, n = rows.shape
    where = np.nonzero(rows)
    lp = Programme(
        cost=gradient / scale,
        col_lower=np.full(n, -1.0),
        col_upper=np.full(n, 1.0),
        matrix=sparse_columns(where[0], where[1], rows[where], (m, n)),
        row_lower=np.full(m, -math.inf),
        row_upper=np.zeros(m),
    )
    solution = _run(lp, RANGING_OPTIONS, None)
    if solution.status != "optimal" or solution.values is None:
        return None
    return solution.values, float(gradient @ solution.values) / scale


def _start(programme: Programme) -> Solution:
    """A feasible point of the programme, from HiGHS's simplex."""
    feasibility = Programme(
        cost=np.zeros(programme.n_columns),
        col_lower=programme.col_lower,
        col_upper=programme.col_upper,
        matrix=programme.matrix,
        row_lower=programme.row_lower,
        row_upper=programme.row_upper,
    )
    solution = _run(feasibility, RANGING_OPTIONS, None)
    if solution.status == "infeasible_or_unbounded":
        solution = _run(feasibility, RANGING_OPTIONS, {"presolve": "off"})
    return solution


def _active_set(programme: Programme, limit: int) -> Solution:
    """Solve a small convex QP with the dense active-set method.

    The module docstring describes the method. Returns `"optimal"` with the
    values and the objective, `"infeasible"`, `"unbounded"` (a descent
    direction without curvature that no constraint blocks), `"limit"` after
    `limit` iterations, or `"error"` if an LP of the method fails.
    """
    assert programme.hessian is not None
    began = time.perf_counter()

    def outcome(
        status: SolveStatus,
        message: str,
        iterations: int,
        x: FloatArray | None = None,
        name: str | None = None,
    ) -> Solution:
        seconds = time.perf_counter() - began
        if x is None:
            return Solution(
                status,
                None,
                None,
                message,
                iterations,
                seconds,
                method="active_set",
                limit=name,
            )
        objective = 0.5 * float(x @ hessian @ x) + float(cost @ x) + programme.offset
        return Solution(
            status, x, objective, message, iterations, seconds, method="active_set"
        )

    start = _start(programme)
    if start.status == "infeasible":
        return outcome("infeasible", "Infeasible", 0)
    if start.status != "optimal" or start.values is None:
        return outcome("error", f"no feasible start ({start.message})", 0)
    lower = np.tril(_dense(programme.hessian))
    hessian = lower + np.tril(lower, -1).T
    cost = programme.cost
    rows, bounds = _inequalities(programme)
    curvature = _CURVATURE * max(float(np.abs(hessian).max(initial=0.0)), 1e-300)
    x = np.clip(start.values, programme.col_lower, programme.col_upper)

    def active(x: FloatArray) -> NDArray[np.int64]:
        slack = bounds - rows @ x
        tolerance = _ACTIVE * max(1.0, float(np.abs(x).max(initial=0.0)))
        return np.flatnonzero(slack <= tolerance)

    working = active(x)
    stationary = False
    for iteration in range(1, limit + 1):
        gradient = hessian @ x + cost
        scale = max(
            float(np.abs(cost).max(initial=0.0)),
            float(np.abs(hessian).max(initial=0.0))
            * max(1.0, float(np.abs(x).max(initial=0.0))),
            1e-300,
        )
        if not stationary:
            step, ray = _face_step(
                hessian, gradient, rows[working], curvature, _SLOPE * scale
            )
            size = float(np.abs(step).max(initial=0.0))
            if size > 1e-15 * max(1.0, float(np.abs(x).max(initial=0.0))):
                cap = _line_minimum(hessian, gradient, step, curvature) if ray else 1.0
                alpha, block = _ratio(rows, bounds, x, step, working, cap)
                if math.isinf(alpha):
                    return outcome("unbounded", "Unbounded", iteration)
                x = x + alpha * step
                if block is not None:
                    working = np.append(working, block)
                else:
                    stationary = not ray
                continue
        # x is the minimum on its face: is there a feasible descent direction?
        on = active(x)
        found = _descent(gradient, rows[on], scale)
        if found is None:
            return outcome("error", "a direction LP failed", iteration)
        direction, value = found
        if value >= -_SLOPE:
            return outcome("optimal", "Optimal", iteration, x)
        keep = on[rows[on] @ direction > -_KEEP]
        space, _ = _row_space(rows[keep])
        direction = direction - space.T @ (space @ direction)
        if float(gradient @ direction) >= -_SLOPE * scale:
            return outcome("optimal", "Optimal", iteration, x)
        cap = _line_minimum(hessian, gradient, direction, curvature)
        alpha, block = _ratio(rows, bounds, x, direction, keep, cap)
        if math.isinf(alpha):
            return outcome("unbounded", "Unbounded", iteration)
        x = x + alpha * direction
        working = keep if block is None else np.append(keep, block)
        stationary = False
    return outcome(
        "limit",
        f"Iteration limit reached (active set, active_set_iteration_limit={limit})",
        limit,
        name="active_set_iteration_limit",
    )
