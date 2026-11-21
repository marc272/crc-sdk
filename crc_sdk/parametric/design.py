"""Design a cover: choose its free parameters on a design sample.

`design(template, sample, objective, *, pricing, budget, limit, options)`
takes a template, a `Cover` with `Free` parameters, and returns a
`DesignResult`: the resolved cover and its design record. The design
minimises the objective (`MSE`, `ExpectedUncovered`, `WeightedCVaR` or a
`UserObjective`) on the sample, under the premium budget, the payout limit,
the bounds of every `Free` parameter and the payout family's own constraints
(non-negative amounts, non-decreasing levels under `monotone=True`).

**Routes.** A payout linear in its amounts (`LinearInAmounts`, which every
built-in family is) has its amounts solved exactly:

- E[U] and weighted CVaR: one linear programme (`route="lp"`), the CVaR in
  the Rockafellar-Uryasev form;
- MSE: one quadratic programme in the amounts (`route="qp"`);
- a cost-of-capital premium adds the Rockafellar-Uryasev auxiliaries of the
  premium's CVaR to the LP. The QP stays one in the amounts, and cutting
  planes on the premium's CVaR add the budget (see **Solver**);
- a premium that is a multiple of the expected payout plus an expense
  (`ExpectedPayoutMultiple`) keeps the budget one linear row in the amounts,
  `multiple * E[payout] <= bound - expense`, so the QP stays one in the
  amounts only. A budget below the expense raises `InfeasibleDesignError`
  before solving, on every route: a cover that never pays still costs it.

Free thresholds (attachment, exhaustion, breakpoints, knots, trigger) are
searched on their stated grids, with the LP or QP above at each grid point
(`route="grid"`). A `UserObjective`, or a payout that is not linear in its
amounts or has no amounts, is searched on a grid over every free parameter
and evaluated in numpy; candidates that break the budget or the limit are
dropped. On a finite sample the objective is piecewise constant or piecewise
smooth in the thresholds, so a stated grid is used rather than gradients; the
objective at every candidate is kept (`DesignResult.profile`).

**A local refinement.** `DesignOptions(refine=True)`, off by default, refines
the grid's best point, with the amounts solved exactly by the LP or QP at
every point it evaluates, as at a grid point. Each free threshold's window
is the best point's neighbours on its grid (or its bound, at an end of the
grid).

- A `StepPayout` or `BinaryPayout` pays from a threshold the years whose
  severity is at least the threshold's, so its objective changes only where
  a threshold crosses one of the sample's index values. Its thresholds are
  refined over those values (`_solver.coordinate_search`): threshold by
  threshold, the others held, it tries the sample's distinct index values in
  the window and the window's two ends, one point per set of years paid, and
  moves to the best of them when that lowers the objective by more than the
  grid's tie tolerance. A pass tries the values nearest each threshold's
  current one, 2 in the first pass and twice as many in each pass after, so
  that a cap on the points reaches every threshold. Passes repeat until one
  that tried every value moves no threshold, or until `refine_max_points`
  points. With one free threshold that is the best of every set of years a
  threshold in its window can pay; with several, a point that no single
  threshold can improve within its window.
- Every other family is refined by a compass search
  (`_solver.compass_search`): from a step per free threshold of half the
  larger side of its window, each round evaluates the point one step up and
  one step down along each threshold, clipped to its `Free` bounds. It moves
  to the best of them when that lowers the objective by more than the grid's
  tie tolerance, and halves every step otherwise. It stops when every step
  is at most `refine_tolerance` times its bounds' width, or after
  `refine_max_points` points, at a local optimum at the resolution of its
  last step, not a global one.

Either is deterministic and never ends worse than the grid's best point.
`DesignResult.refinement` records the method, its settings, the start, the
end, the gain, the points and why it stopped, and `refinement_profile` lists
the points. Where it ends, the amounts are solved again with the uniqueness
report; `ties` counts the grid's candidates within the tie tolerance of the
refined objective, and the search's own points are not counted, so whether
the thresholds themselves are unique is not checked. A design without free
thresholds is not refined. A grid evaluated in numpy (a `UserObjective`, or
a payout not linear in its amounts) has no exact amounts to refine around,
so the option raises `DesignError` there, as it does in `design_joint`,
which has no grid.

**Solver.** The LP and QP routes use HiGHS from the optional extra
`crc-sdk[parametric]` (`pip install crc-sdk[parametric]`), imported only
when needed; without it they raise `SolverUnavailableError`. A grid
evaluated in numpy needs no solver. HiGHS's active-set QP solver can cycle
at a degenerate vertex of a QP, and has reported a bounded one unbounded (on
Windows, corrupting the process's heap). So a QP of at most 100 columns (one
per amount) is solved by a dense active-set method in numpy, whose steps at a
degenerate vertex are LPs solved by HiGHS's simplex. A larger QP goes to
HiGHS with an iteration and a time limit; when HiGHS stops at its limit, or
ends with any outcome but optimal or infeasible, a QP of at most 500 columns
is solved again by the dense method.

An MSE design under a cost-of-capital budget is solved by cutting planes on
the premium's CVaR, which is convex and piecewise linear in the amounts:
each step solves the QP in the amounts, as above, with the budget's cuts
found so far, and adds the cut of its solution's tail (the largest `1 - q`
of the payout's probability), until the solution's premium exceeds the
budget by at most 1e-9 of the larger of the scale and the budget. Each QP
relaxes the budget, so the design's MSE is at most the optimum. The QPs have
one column per amount, not one per draw, so they reach HiGHS's QP solver only
with more than 100 amounts. The method stops after 1,000 QPs.

`DesignResult.solved_by` says which method solved the design (`"highs"`,
`"active_set"` or `"cutting_planes"`) and `fallbacks` how many of its solves
the dense method finished after HiGHS failed. A solve that stops at a limit
raises `SolverLimitError`, which names the limit.

**Scaling.** Losses and payouts are divided by the largest loss of a draw
(1 if that is at most 1e-6) before solving; results are in payout units.

**No silent failures.** After a solve, numpy recomputes the payout, the
objective, the premium, the budget and every constraint from the returned
parameters, with the relative tolerance `DesignOptions.feasibility_tolerance`
(relative to the larger of the bound and the scale). A violation, or a
solver status other than optimal, raises `DesignError`,
`InfeasibleDesignError` or `UnboundedDesignError`, with the asset id and the
solver's message. A mean squared error is never below 0, so a QP reported
unbounded raises `DesignError`, not `UnboundedDesignError`. Free amounts below
1e-9 of the scale are set to 0 and listed (`DesignResult.zeroed`).

**Non-unique optima** are reported, not hidden:

- an amount whose basis column is zero in every design year (a band that no
  year reaches) changes nothing on the sample; it is listed in
  `unidentified` and set by `DesignOptions.fill`: `"lowest"` (the default)
  takes the smallest value the constraints allow, `"highest"` the largest.
  Both keep the design exactly optimal;
- on the LP route, `DesignOptions(ranges=True)` ranges every identified
  amount over the near-optimal designs at two tolerances (1e-9 and 1e-11 of
  `max(1, |optimum|)`, in scaled units). An amount is pinned when its narrow
  range is at most half its wide one, or narrower than 1e-7 of the scale;
  `unique` is true when every identified amount is pinned. Without ranges,
  `unique` is `None` on the LP route. Each range end is the optimum of an LP
  over the near-optimal set, solved on a reduced LP: draws with the same
  basis row and loss are merged, and a draw away from the kinks of its
  uncovered loss and of the CVaR terms keeps the linear piece it is on at the
  design's optimum, summed into one row with the others. A fixed piece is
  below the function, so the reduced LP is a relaxation; when its solution
  keeps every fixed piece it solves the full LP, and otherwise the draws whose
  piece broke are freed and it is solved again. After 100 such LPs, or one
  that is not solved to optimality, the full LP solves that end;
- on the QP route, `unique` is true when `H = B' diag(p) B` over the
  identified amounts is positive definite (smallest eigenvalue above 1e-10
  of the largest);
- on a grid, `ties` counts the other candidates whose objective lies
  within `DesignOptions.feasibility_tolerance` times `max(1, |best|)` of the
  best, in scaled units; the first in lexicographic order wins. The LP and
  QP routes have one candidate, so `ties` is 0 there. `unique` is false
  with ties, else the inner solve's report
  (`None` for a grid evaluated in numpy).

For a weighted CVaR the design records `p_zero_loss` (the probability of a
zero loss), `tail_outcomes` (`(1 - beta) / p_min`) and `degenerate_tail`:
`"expected_uncovered"` when `beta <= p_zero_loss` (the objective is then a
multiple of E[U]), `"single_outcome"` when the tail holds at most one draw.

**Several hazards.** A cover with a `CompositeIndex` reads the components of
its names, in its order, so its payout sees an `(n, k)` index and everything
above applies. A cover on one index reads, on a sample with index components,
the component named its `hazard_name`. A design uses the loss of the sample
it is given: `design(template, sample)` the combined loss, and
`design(template, sample.component(name))` that hazard's own loss. Each read
has the component's tail: the view keeps it, so a template whose index has the
other tail raises `SampleError` there as on the sample of several hazards.

**Payouts over several components.** A `SumOf` of components with amounts is
one LP or QP on the block basis `[B_1 | ... | B_K]`. A `MaxOf` with a free scale
over fixed component amounts is one LP or QP in the scale, with basis
`max_k pi_k(s_k)`. Free thresholds are searched on their grids, as above, with
that LP or QP at each point. Every other `MaxOf`, and a `SumOf` with a component
without amounts, is searched in numpy on a grid over every free parameter.

A `StepGrid` is one LP or QP on the one-hot basis of its cells. Its free
thresholds are searched on their grids in the same way: every point of the
product over both axes counts towards `max_grid_points`, and a point with
unordered thresholds is recorded as `"invalid"`. A cell no design year reaches
is `unidentified`, and the fill rule runs on the order along both axes.

`Limit` on a `SumOf` bounds the year's total: for every choice of one amount
per component, the chosen amounts sum to at most the limit. That holds when the
rows of each component's basis sum to at most 1, so `design` raises
`DesignError` if a row sums above 1 on the design sample. On a `MaxOf` with a
free scale, with `G` the largest amount of any component, `scale <= limit / G`.

Everything here works in memory and does no I/O.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NoReturn, Self, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import Strict, model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric import _solver
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric._pairing import _check_reads
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import (
    CoverDefinitionError,
    DesignError,
    InfeasibleDesignError,
    SampleError,
    SolverLimitError,
    UnboundedDesignError,
)
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    Objective,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.parameters import Free, _DefinitionModel
from crc_sdk.parametric.payouts import (
    BinaryPayout,
    LinearConstraints,
    LinearInAmounts,
    StepPayout,
    SumOf,
    _has_amounts,
)
from crc_sdk.parametric.pricing import (
    _PRICINGS,
    _PRICINGS_TEXT,
    CostOfCapitalLoad,
    ExpectedPayoutMultiple,
    Pricing,
    PurePremium,
    _expense,
)
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _cover_index_values,
    _require_one_asset,
)

__all__ = [
    "DESIGN_SEMANTICS_VERSION",
    "DesignOptions",
    "DesignResult",
    "Fill",
    "Route",
    "design",
]

FloatArray = NDArray[np.float64]

DESIGN_SEMANTICS_VERSION = "crc-parametric-design/1"
"""The version of the document behind `DesignResult.design_id`."""

Route = Literal["lp", "qp", "grid"]
"""How a design was solved: one LP, one QP, or a grid search."""

Fill = Literal["lowest", "highest"]
"""The rule that sets unidentified amounts."""

SCALE_FLOOR = 1e-6
"""A largest loss at most this gives the scale 1."""

ZERO_AMOUNT = 1e-9
"""Free amounts below this share of the scale are set to 0."""

NEAR_OPTIMAL_TOLERANCES = (1e-9, 1e-11)
"""The wide and the narrow near-optimal tolerance of the ranges."""

PINNED_SHRINK = 0.5
"""An amount whose narrow range is at most this share of its wide one is pinned."""

PINNED_WIDTH = 1e-7
"""An amount whose narrow range is below this share of the scale is pinned."""

RANGING_KINK = 1e-9
"""A draw this close to a kink at the design's optimum starts free (scaled units).

The ranging LPs keep every other draw on the piece it is on there (see
`_ranges`); a draw that crosses its kink is freed when it does.
"""

RANGING_ROUND_LIMIT = 100
"""The most reduced LPs for one range end; past them the full LP solves it."""

MSE_EIGEN_TOLERANCE = 1e-10
"""QP route: unique when H's smallest eigenvalue exceeds this share of its largest."""

DEGENERATE_TAIL_TOLERANCE = 1e-9
"""The rounding allowance of the degenerate-tail comparisons."""

BASIS_ROW_TOLERANCE = 1e-9
"""The rounding allowance of a basis row summing to at most 1."""

REFINE_METHOD = "compass_search"
"""The method of `DesignOptions(refine=True)`, as the record names it."""

INDEX_VALUES_METHOD = "index_values"
"""The method of `DesignOptions(refine=True)` on a step or binary payout."""

REFINE_TOLERANCE = 1e-4
"""The default smallest step of a refinement, as a share of a parameter's bounds."""

REFINE_MAX_POINTS = 1_000
"""The default most points a refinement evaluates."""

StrictInt = Annotated[int, Strict()]
StrictBool = Annotated[bool, Strict()]


class DesignOptions(_DefinitionModel):
    """How `design` searches, checks and reports.

    - `max_grid_points`: the largest grid (product of the free parameters'
      grids) a design may search; a larger one raises `DesignError`.
    - `feasibility_tolerance`: the relative tolerance of the checks after a
      solve, relative to the larger of the bound and the scale; also the
      tie tolerance of a grid: candidates whose objective is within it times
      `max(1, |best|)` of the best tie with it (in scaled units for the
      built-in objectives, in the objective's own units for a user
      objective, whose scaling is unknown).
    - `fill`: `"lowest"` or `"highest"`, the value of unidentified amounts.
    - `ranges`: on the LP route, range every identified amount over the
      near-optimal designs and report `unique` (four range ends per amount,
      each an LP, solved on reduced LPs; see the module docstring).
    - `refine`: after a grid over free thresholds, refine its best point,
      with the amounts solved exactly at every point it evaluates: over the
      sample's index values for a `StepPayout` or `BinaryPayout`, by a
      compass search otherwise (see the module docstring).
      `refine_tolerance` is the compass search's smallest step, as a share of
      each parameter's bounds (`upper - lower`), and `refine_max_points` the
      most points either evaluates. Both apply only with `refine=True`.

    Raises `CoverDefinitionError` (a `ValueError`) for invalid values, and
    for `refine_tolerance` or `refine_max_points` changed without
    `refine=True`.
    """

    max_grid_points: StrictInt = 10_000
    feasibility_tolerance: FiniteFloat = 1e-7
    fill: Fill = "lowest"
    ranges: StrictBool = False
    refine: StrictBool = False
    refine_tolerance: FiniteFloat = REFINE_TOLERANCE
    refine_max_points: StrictInt = REFINE_MAX_POINTS

    def __init__(
        self,
        *,
        max_grid_points: int = 10_000,
        feasibility_tolerance: float = 1e-7,
        fill: Fill = "lowest",
        ranges: bool = False,
        refine: bool = False,
        refine_tolerance: float = REFINE_TOLERANCE,
        refine_max_points: int = REFINE_MAX_POINTS,
    ) -> None:
        super().__init__(
            max_grid_points=max_grid_points,
            feasibility_tolerance=feasibility_tolerance,
            fill=fill,
            ranges=ranges,
            refine=refine,
            refine_tolerance=refine_tolerance,
            refine_max_points=refine_max_points,
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_grid_points < 1:
            raise CoverDefinitionError("max_grid_points must be at least 1")
        if not 0.0 < self.feasibility_tolerance < 1.0:
            raise CoverDefinitionError("feasibility_tolerance must lie in (0, 1)")
        if not 0.0 < self.refine_tolerance < 1.0:
            raise CoverDefinitionError("refine_tolerance must lie in (0, 1)")
        if self.refine_max_points < 1:
            raise CoverDefinitionError("refine_max_points must be at least 1")
        if not self.refine and (
            self.refine_tolerance != REFINE_TOLERANCE
            or self.refine_max_points != REFINE_MAX_POINTS
        ):
            raise CoverDefinitionError(
                "refine_tolerance and refine_max_points apply only with refine=True"
            )
        return self

    def to_document(self) -> dict[str, Any]:
        """Return every option as plain JSON types.

        The refinement's options are left out when `refine` is false, so the
        document, and the `design_id` it enters, of a design without a
        refinement is that of a design made before the option existed.
        """
        document: dict[str, Any] = {
            "feasibility_tolerance": self.feasibility_tolerance,
            "fill": self.fill,
            "max_grid_points": self.max_grid_points,
            "ranges": self.ranges,
        }
        if self.refine:
            document |= {
                "refine": True,
                "refine_max_points": self.refine_max_points,
                "refine_tolerance": self.refine_tolerance,
            }
        return document


@dataclass(frozen=True, eq=False)
class DesignResult:
    """A designed cover and its design record.

    - `cover`: the template with every free parameter resolved;
      `template_hash` the template's `definition_hash`; `design_id` the
      `"sha256:"` hash of the template hash, the objective, the pricing, the
      budget, the limit, the options and the design sample's hash;
      `parameters` the designed value of each free parameter.
    - `objective` (document) and `objective_value`, on the design sample in
      payout units (squared for MSE); `premium`, the design premium (the
      contract price, priced on the design sample with `pricing`);
      `expected_loss`; `budget_bound`, the largest premium allowed, and
      `budget_use`, the premium's share of it.
    - `route`: `"lp"`, `"qp"` or `"grid"`; `amounts`: how the amounts were
      chosen (`"lp"`, `"qp"`, or `"grid"` when they were on the grid too).
    - `solver` (`{"name", "version"}`, or `None` for a grid in numpy),
      `status`, `solver_message`, `iterations`, `solves`, `seconds` (solver
      time) and `solver_options`.
    - `solved_by`: what solved the design's amounts, `"highs"`,
      `"active_set"` (the dense method of a small QP, or of one that HiGHS
      left unsolved) or `"cutting_planes"` (MSE under a cost-of-capital
      budget), or `None` for a grid in numpy; `fallbacks`: how many of the
      design's solves the dense method finished after HiGHS failed, over
      every candidate of a grid (a cutting-plane solve counts once, however
      many of its QPs did).
    - The non-uniqueness report: `unidentified` amounts and the `fill` rule
      that set them, `zeroed` amounts, `ranges` (wide ranges in payout
      units, with `ranges=True` on the LP route), `unique`, `ties`, and the
      grid's `profile` (a `pyarrow.Table`: one row per candidate, its values,
      `objective` and `status`) and `grid` counts.
    - The degenerate-tail flags `p_zero_loss`, `tail_outcomes` and
      `degenerate_tail`.
    - `design_sample`, the sample the cover was designed on.
    - `refinement`: with `DesignOptions(refine=True)` on a grid, the record
      of the refinement after it, as plain JSON types: `method` and its
      settings, `start` and `start_objective` (the grid's best point), `end`
      and `objective` (where it ended), `gain` (`start_objective -
      objective`, never below 0), `points` (how many points it evaluated,
      the start left out), `counts` (their statuses), `moves` and `stopped`.
      A compass search (`method="compass_search"`) adds `tolerance`,
      `max_points`, `initial_steps` and `final_steps`, and stops at
      `"step_tolerance"` or `"max_points"`; a search over the index values
      (`method="index_values"`, a step or binary payout) adds `max_points`,
      `windows` (each threshold's `[lower, upper]`), `candidates` (how many
      values each window holds) and `passes`, and stops at `"no_move"` or
      `"max_points"`. Objectives are in payout units (squared for MSE).
      `None` without a refinement. `refinement_profile` is a
      `pyarrow.Table` of the points it evaluated, in order: their values,
      `objective` and `status`, as in `profile`.

    `evaluate` and `compare` take a `DesignResult` in place of a cover: its
    rows use the design premium and carry `unique`, `ties` and
    `degenerate_tail`, and `to_record()` goes into their metadata.

    `result.cover.terms_text()` states the designed terms in words, every
    designed amount spelled exactly, for a report or a broker. The record says
    how they were chosen.
    """

    cover: Cover
    template_hash: str
    design_id: str
    parameters: Mapping[str, float]
    objective: Mapping[str, Any]
    objective_value: float
    pricing: Mapping[str, Any]
    budget: Mapping[str, Any] | None
    limit: Mapping[str, Any] | None
    options: DesignOptions
    premium: float
    expected_loss: float
    budget_bound: float | None
    budget_use: float | None
    route: Route
    amounts: Literal["lp", "qp", "grid"]
    solver: Mapping[str, str] | None
    status: str
    solver_message: str
    iterations: int
    solves: int
    seconds: float
    solver_options: Mapping[str, Any]
    unidentified: tuple[str, ...]
    fill: Fill
    zeroed: tuple[str, ...]
    ranges: Mapping[str, tuple[float, float]] | None
    unique: bool | None
    ties: int
    profile: Any | None
    grid: Mapping[str, int] | None
    p_zero_loss: float
    tail_outcomes: float | None
    degenerate_tail: str | None
    design_sample: OutcomeSample = field(repr=False)
    solved_by: _solver.Method | None = None
    fallbacks: int = 0
    refinement: Mapping[str, Any] | None = None
    refinement_profile: Any | None = None

    @property
    def sample_hash(self) -> str:
        """The design sample's `content_hash`."""
        return self.design_sample.content_hash

    def to_record(self) -> dict[str, Any]:
        """Return the design record as plain JSON types.

        The record holds the ids and hashes, the objective, pricing, budget,
        limit and options, the values, the solve (`solver` with the name and
        version; `solve` with the status, message, iterations, solves, time,
        options, `solved_by` and `fallbacks`), the non-uniqueness report and
        the degenerate-tail flags, and, after a refinement, its record under
        `refinement` (left out without one). `evaluate` writes it to its
        metadata's `designs`.
        """
        ranges = (
            None
            if self.ranges is None
            else {name: [low, high] for name, (low, high) in self.ranges.items()}
        )
        record: dict[str, Any] = {
            "design_id": self.design_id,
            "template_hash": self.template_hash,
            "cover_hash": self.cover.definition_hash,
            "sample_hash": self.sample_hash,
            "objective": copy.deepcopy(dict(self.objective)),
            "objective_value": self.objective_value,
            "pricing": dict(self.pricing),
            "budget": None if self.budget is None else dict(self.budget),
            "limit": None if self.limit is None else dict(self.limit),
            "options": self.options.to_document(),
            "parameters": dict(self.parameters),
            "premium": self.premium,
            "expected_loss": self.expected_loss,
            "budget_bound": self.budget_bound,
            "budget_use": self.budget_use,
            "route": self.route,
            "amounts": self.amounts,
            "solver": None if self.solver is None else dict(self.solver),
            "solve": {
                "status": self.status,
                "message": self.solver_message,
                "iterations": self.iterations,
                "solves": self.solves,
                "seconds": self.seconds,
                "options": copy.deepcopy(dict(self.solver_options)),
                "solved_by": self.solved_by,
                "fallbacks": self.fallbacks,
            },
            "unidentified": list(self.unidentified),
            "fill": self.fill,
            "zeroed": list(self.zeroed),
            "ranges": ranges,
            "unique": self.unique,
            "ties": self.ties,
            "grid": None if self.grid is None else dict(self.grid),
            "p_zero_loss": self.p_zero_loss,
            "tail_outcomes": self.tail_outcomes,
            "degenerate_tail": self.degenerate_tail,
        }
        if self.refinement is not None:
            record["refinement"] = copy.deepcopy(dict(self.refinement))
        return record


# -- the problem -------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Context:
    """The design problem in payout units, with its scale.

    `asset_id` is `None` when the problem is on several assets' total; its
    `index` then has one column per asset. A cover with a `CompositeIndex`
    has one column per component.

    A budget `bound` below the pricing's fixed expense raises
    `InfeasibleDesignError` here, before any route runs: a cover that never
    pays still costs the expense.
    """

    asset_id: str | None
    index: FloatArray
    loss: FloatArray
    probability: FloatArray
    scale: float
    expected_loss: float
    bound: float | None
    limit: float | None
    objective: Objective
    pricing: Pricing
    options: DesignOptions

    def __post_init__(self) -> None:
        expense = _expense(self.pricing)
        if self.bound is not None and self.bound < expense:
            raise self.error(
                InfeasibleDesignError,
                f"the budget {self.bound:.10g} is below the pricing's fixed expense "
                f"{expense:.10g}: a cover that never pays still costs the expense, "
                "so no payout keeps the premium within the budget",
                status="infeasible",
            )

    def scaled(self, value: float) -> float:
        """An objective value in scaled units."""
        if isinstance(self.objective, UserObjective):
            return value
        return value / self.scale**self.objective.homogeneity

    def slack(self, bound: float) -> float:
        """The allowance of a check against `bound`, in payout units."""
        return self.options.feasibility_tolerance * max(self.scale, abs(bound))

    def error(
        self,
        kind: type[DesignError],
        message: str,
        *,
        status: str | None = None,
        solver_message: str | None = None,
    ) -> DesignError:
        subject = (
            "the joint design" if self.asset_id is None else f"asset {self.asset_id!r}"
        )
        return kind(
            f"{subject}: {message}",
            asset_id=self.asset_id,
            status=status,
            solver_message=solver_message,
        )


@dataclass
class _Solves:
    """The solves of one design, for the record."""

    iterations: int = 0
    count: int = 0
    seconds: float = 0.0
    fallbacks: int = 0
    last: _solver.Solution | None = None

    def add(self, solution: _solver.Solution) -> _solver.Solution:
        self.iterations += solution.iterations
        self.count += 1
        self.seconds += solution.seconds
        self.fallbacks += int(solution.fallback)
        self.last = solution
        return solution


@dataclass(frozen=True, eq=False)
class _Amounts:
    """The amounts of one cover, solved by an LP or a QP."""

    cover: Cover
    values: dict[str, float]
    objective_value: float
    route: Literal["lp", "qp"]
    method: _solver.Method
    unidentified: tuple[str, ...]
    zeroed: tuple[str, ...]
    unique: bool | None
    ranges: dict[str, tuple[float, float]] | None


@dataclass(frozen=True, eq=False)
class _Block:
    """The amounts of a design, as columns: one cover's, or several side by side."""

    names: tuple[str, ...]
    basis: FloatArray
    constraints: LinearConstraints
    free: NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class _SolvedBlock:
    """A block's amounts after the solve, the repair and the fill rule."""

    amounts: FloatArray
    programme: _solver.Programme
    solution: _solver.Solution
    route: Literal["lp", "qp"]
    ranging: bool
    identified: NDArray[np.bool_]
    unidentified: NDArray[np.bool_]
    zeroed: NDArray[np.bool_]


class _Builder:
    """Assembles a programme column by column and row by row."""

    def __init__(self) -> None:
        self.cost: list[FloatArray] = []
        self.col_lower: list[FloatArray] = []
        self.col_upper: list[FloatArray] = []
        self.row_lower: list[FloatArray] = []
        self.row_upper: list[FloatArray] = []
        self.rows: list[NDArray[np.int64]] = []
        self.columns: list[NDArray[np.int64]] = []
        self.values: list[FloatArray] = []
        self.n_columns = 0
        self.n_rows = 0

    def variables(self, count: int, lower: Any, upper: Any, cost: Any) -> int:
        start = self.n_columns
        for target, value in (
            (self.col_lower, lower),
            (self.col_upper, upper),
            (self.cost, cost),
        ):
            target.append(np.broadcast_to(np.asarray(value, np.float64), (count,)))
        self.n_columns += count
        return start

    def constraints(self, count: int, lower: Any, upper: Any) -> int:
        start = self.n_rows
        self.row_lower.append(np.broadcast_to(np.asarray(lower, np.float64), (count,)))
        self.row_upper.append(np.broadcast_to(np.asarray(upper, np.float64), (count,)))
        self.n_rows += count
        return start

    def entries(self, rows: Any, columns: Any, values: Any) -> None:
        rows_array = np.asarray(rows, dtype=np.int64).reshape(-1)
        columns_array = np.asarray(columns, dtype=np.int64).reshape(-1)
        values_array = np.broadcast_to(
            np.asarray(values, dtype=np.float64), rows_array.shape
        )
        self.rows.append(rows_array)
        self.columns.append(columns_array)
        self.values.append(np.array(values_array))

    def programme(
        self, *, hessian: _solver.SparseColumns | None = None, offset: float = 0.0
    ) -> _solver.Programme:
        def joined(parts: list[Any], dtype: Any) -> Any:
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)

        matrix = _solver.sparse_columns(
            joined(self.rows, np.int64),
            joined(self.columns, np.int64),
            joined(self.values, np.float64),
            (self.n_rows, self.n_columns),
        )
        return _solver.Programme(
            cost=joined(self.cost, np.float64),
            col_lower=joined(self.col_lower, np.float64),
            col_upper=joined(self.col_upper, np.float64),
            matrix=matrix,
            row_lower=joined(self.row_lower, np.float64),
            row_upper=joined(self.row_upper, np.float64),
            offset=offset,
            hessian=hessian,
        )


def _block(builder: _Builder, first_row: int, basis: FloatArray, sign: float) -> None:
    """Add `sign * basis @ theta` to the rows from `first_row` on."""
    rows, columns = np.nonzero(basis)
    builder.entries(first_row + rows, columns, sign * basis[rows, columns])


def _lower_triangle(matrix: FloatArray, n_columns: int) -> _solver.SparseColumns:
    """The lower triangle of a square block in the top left of `n` columns."""
    rows, columns = np.nonzero(np.tril(matrix))
    return _solver.sparse_columns(
        rows, columns, matrix[rows, columns], (n_columns, n_columns)
    )


def _programme(
    basis: FloatArray, constraints: LinearConstraints, context: _Context
) -> tuple[_solver.Programme, Literal["lp", "qp"], _solver.CVaRRow | None]:
    """Build the LP or QP over the amounts, in scaled units.

    The amounts are the first columns, in `amount_names()` order. An MSE
    design under a cost-of-capital budget gets a QP in the amounts only,
    without the budget, and the budget as a `CVaRRow`, which cutting planes
    add (`_solver.solve_qp_cvar`); every other design gets no such row.
    """
    n, k = basis.shape
    scale = context.scale
    p = context.probability
    loss = context.loss / scale
    builder = _Builder()
    builder.variables(k, constraints.lower / scale, constraints.upper / scale, 0.0)
    objective = context.objective
    route: Literal["lp", "qp"]
    hessian: _solver.SparseColumns | None = None
    offset = 0.0
    tail_row: _solver.CVaRRow | None = None
    if isinstance(objective, MSE):
        route = "qp"
    else:
        route = "lp"
        alpha = objective.alpha if isinstance(objective, WeightedCVaR) else 0.0
        uncovered = builder.variables(n, 0.0, math.inf, (1.0 - alpha) * p)
        first = builder.constraints(n, loss, math.inf)
        _block(builder, first, basis, 1.0)
        builder.entries(first + np.arange(n), uncovered + np.arange(n), 1.0)
        if isinstance(objective, WeightedCVaR) and alpha > 0.0:
            beta = objective.beta
            zeta = builder.variables(1, -math.inf, math.inf, alpha)
            excess = builder.variables(n, 0.0, math.inf, alpha * p / (1.0 - beta))
            first = builder.constraints(n, 0.0, math.inf)
            rows = first + np.arange(n)
            builder.entries(rows, excess + np.arange(n), 1.0)
            builder.entries(rows, uncovered + np.arange(n), -1.0)
            builder.entries(rows, np.full(n, zeta), 1.0)
    if context.bound is not None:
        expected = p @ basis
        pricing = context.pricing
        if (
            isinstance(pricing, CostOfCapitalLoad)
            and pricing.rate > 0.0
            and route == "qp"
        ):
            # The premium's CVaR stays out of the QP, which keeps one column
            # per amount; cutting planes add it.
            tail_row = _solver.CVaRRow(
                linear=(1.0 - pricing.rate) * expected,
                basis=basis,
                probability=p,
                q=pricing.q,
                weight=pricing.rate,
                upper=context.bound / scale,
            )
        elif isinstance(pricing, CostOfCapitalLoad) and pricing.rate > 0.0:
            rate, q = pricing.rate, pricing.q
            eta = builder.variables(1, -math.inf, math.inf, 0.0)
            tail = builder.variables(n, 0.0, math.inf, 0.0)
            first = builder.constraints(n, 0.0, math.inf)
            rows = first + np.arange(n)
            builder.entries(rows, tail + np.arange(n), 1.0)
            _block(builder, first, basis, -1.0)
            builder.entries(rows, np.full(n, eta), 1.0)
            row = builder.constraints(1, -math.inf, context.bound / scale)
            builder.entries(np.full(k, row), np.arange(k), (1.0 - rate) * expected)
            builder.entries([row], [eta], rate)
            builder.entries(np.full(n, row), tail + np.arange(n), rate * p / (1.0 - q))
        elif isinstance(pricing, ExpectedPayoutMultiple):
            # multiple * E[payout] + expense <= bound, with bound >= expense.
            row = builder.constraints(
                1, -math.inf, (context.bound - pricing.expense) / scale
            )
            builder.entries(np.full(k, row), np.arange(k), pricing.multiple * expected)
        else:
            row = builder.constraints(1, -math.inf, context.bound / scale)
            builder.entries(np.full(k, row), np.arange(k), expected)
    m = constraints.a_ub.shape[0]
    if m:
        first = builder.constraints(m, -math.inf, constraints.b_ub / scale)
        _block(builder, first, constraints.a_ub, 1.0)
    if route == "qp":
        weighted = basis * p[:, None]
        gram = basis.T @ weighted
        builder.cost[0] = -2.0 * (weighted.T @ loss)
        hessian = _lower_triangle(2.0 * gram, builder.n_columns)
        offset = math.fsum((p * loss**2).tolist())
    return builder.programme(hessian=hessian, offset=offset), route, tail_row


# -- solving -----------------------------------------------------------------


def _solve(
    programme: _solver.Programme,
    route: Literal["lp", "qp"],
    context: _Context,
    solves: _Solves,
    options: Mapping[str, Any] | None = None,
    *,
    keep_basis: bool = False,
    tail_row: _solver.CVaRRow | None = None,
) -> _solver.Solution:
    """Solve; raise the typed error of any status other than optimal.

    With `keep_basis`, an LP's solution carries its optimal basis. A QP with
    a `tail_row` is solved by cutting planes. The QP route's objective, a
    mean squared error, is bounded below by 0.
    """

    def run(chosen: Mapping[str, Any] | None) -> _solver.Solution:
        if tail_row is not None:
            return _solver.solve_qp_cvar(programme, tail_row, chosen)
        if route == "qp":
            return _solver.solve_qp(programme, chosen)
        return _solver.solve_lp(programme, chosen, keep_basis=keep_basis)

    return _checked(run, context, solves, options, bounded_below=route == "qp")


def _checked(
    run: Callable[[Mapping[str, Any] | None], _solver.Solution],
    context: _Context,
    solves: _Solves,
    options: Mapping[str, Any] | None = None,
    *,
    bounded_below: bool = False,
) -> _solver.Solution:
    """Call `run(options)`; raise the typed error of any status but optimal.

    A solve stopped at a limit raises `SolverLimitError`, named by the limit.
    `bounded_below` says that the objective can't be unbounded (a mean squared
    error); an unbounded status is then a solver failure and raises
    `DesignError`.
    """
    solution = solves.add(run(options))
    if solution.status == "infeasible_or_unbounded":
        # Presolve may not tell the two apart; the solve without it does.
        solution = solves.add(run({**(options or {}), "presolve": "off"}))
    if solution.status == "optimal":
        return solution
    if solution.status == "limit":
        raise context.error(
            SolverLimitError,
            f"the solve stopped at its limit {solution.limit} before it found an "
            f"optimum ({solution.message})",
            status=solution.limit,
            solver_message=solution.message,
        )
    if solution.status == "unbounded" and bounded_below:
        raise context.error(
            DesignError,
            "the solver reported an unbounded objective, but a mean squared "
            f"error is never below 0: the solve failed ({solution.message})",
            status=solution.status,
            solver_message=solution.message,
        )
    kinds: dict[str, type[DesignError]] = {
        "infeasible": InfeasibleDesignError,
        "unbounded": UnboundedDesignError,
    }
    reasons = {
        "infeasible": "no payout satisfies the constraints",
        "unbounded": "the objective is unbounded under the constraints",
    }
    raise context.error(
        kinds.get(solution.status, DesignError),
        f"{reasons.get(solution.status, 'the solver found no optimal design')} "
        f"(HiGHS: {solution.message})",
        status=solution.status,
        solver_message=solution.message,
    )


def _order_edges(
    constraints: LinearConstraints,
) -> tuple[list[tuple[int, int]], NDArray[np.bool_]]:
    """The rows `a_i <= a_j` of the constraints, and the other rows.

    Returns the pairs `(i, j)` and a mask of the rows that are not of that
    form (all-zero rows are neither).
    """
    edges: list[tuple[int, int]] = []
    other = np.zeros(constraints.a_ub.shape[0], dtype=np.bool_)
    for row, (coefficients, bound) in enumerate(
        zip(constraints.a_ub, constraints.b_ub, strict=True)
    ):
        nonzero = np.flatnonzero(coefficients)
        if nonzero.size == 0:
            continue
        if nonzero.size == 2 and bound == 0.0:
            first, second = (float(coefficients[i]) for i in nonzero)
            if first == -second:
                low, high = (nonzero[0], nonzero[1])
                if first < 0:
                    low, high = high, low
                edges.append((int(low), int(high)))
                continue
        other[row] = True
    return edges, other


def _propagate(
    values: FloatArray,
    edges: Sequence[tuple[int, int]],
    *,
    up: NDArray[np.bool_] | None = None,
    down: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Enforce `a_i <= a_j` by raising `a_j` (where `up`) or lowering `a_i`."""
    result = values.copy()
    for _ in range(result.shape[0] + 1):
        changed = False
        for i, j in edges:
            if result[i] > result[j]:
                if up is not None and up[j]:
                    result[j] = result[i]
                    changed = True
                elif down is not None and down[i]:
                    result[i] = result[j]
                    changed = True
        if not changed:
            break
    return result


def _repair(
    values: FloatArray,
    constraints: LinearConstraints,
    edges: Sequence[tuple[int, int]],
) -> FloatArray:
    """Move solver values onto the bounds and order constraints exactly.

    A solver meets its constraints only to rounding, while a cover checks
    them exactly. The values are clipped to bounds tightened along the order
    constraints, then raised along them, which keeps every bound.
    """
    k = values.shape[0]
    everywhere = np.ones(k, dtype=np.bool_)
    lower = _propagate(constraints.lower.copy(), edges, up=everywhere)
    upper = _propagate(constraints.upper.copy(), edges, down=everywhere)
    clipped = np.minimum(np.maximum(values, lower), upper)
    return _propagate(clipped, edges, up=everywhere)


def _fill(
    values: FloatArray,
    unidentified: NDArray[np.bool_],
    constraints: LinearConstraints,
    edges: Sequence[tuple[int, int]],
    other: NDArray[np.bool_],
    context: _Context,
    solves: _Solves,
    names: Sequence[str],
) -> FloatArray:
    """Set the unidentified amounts by the fill rule.

    `"lowest"` takes the smallest value the constraints allow with the
    identified amounts fixed, `"highest"` the largest. Bounds and order
    constraints are propagated exactly; other constraints on an unidentified
    amount are solved as an LP.
    """
    if not unidentified.any():
        return values
    highest = context.options.fill == "highest"
    touches = (constraints.a_ub[other][:, unidentified] != 0).any()
    if not touches:
        start = values.copy()
        if highest:
            start[unidentified] = constraints.upper[unidentified]
            filled = _propagate(start, edges, down=unidentified)
        else:
            start[unidentified] = constraints.lower[unidentified]
            filled = _propagate(start, edges, up=unidentified)
    else:
        lower = np.where(unidentified, constraints.lower, values) / context.scale
        upper = np.where(unidentified, constraints.upper, values) / context.scale
        builder = _Builder()
        sign = -1.0 if highest else 1.0
        builder.variables(values.shape[0], lower, upper, sign * unidentified)
        m = constraints.a_ub.shape[0]
        first = builder.constraints(m, -math.inf, constraints.b_ub / context.scale)
        _block(builder, first, constraints.a_ub, 1.0)
        solution = _solve(builder.programme(), "lp", context, solves)
        assert solution.values is not None
        filled = np.where(unidentified, solution.values * context.scale, values)
        filled = _repair(filled, constraints, edges)
    if highest and not np.isfinite(filled[unidentified]).all():
        missing = [name for name, free in zip(names, unidentified, strict=True) if free]
        raise context.error(
            DesignError,
            f"fill='highest' needs an upper bound (a Free upper bound or a "
            f"Limit) above the unidentified amounts {missing}",
        )
    return filled


# -- ranging -----------------------------------------------------------------
#
# A range end is `min sign * theta_j` over the near-optimal set: the design
# LP's feasible set with the row `cost @ x <= upper`. In the amounts (and the
# free ζ and η), that row is `G(theta, ζ) <= upper`, a sum over the draws of
# p_d u_d, with u_d = (L_d - B_d theta)^+, and of the CVaR's p_d e_d, with e_d
# = (u_d - ζ)^+; a cost-of-capital budget sums t_d = (B_d theta - η)^+ in the
# same way. Each term is the larger of two linear pieces, and either piece is
# below it. So keeping a draw on one piece relaxes the LP, and summing the
# draws on fixed pieces into one row makes it small; only the draws near a
# kink keep their own columns and rows. If the reduced LP's solution keeps
# every fixed piece, it is feasible in the full LP, so it solves it.

_BELOW, _ABOVE, _FREE = 0, 1, 2
"""A draw's piece of `x^+`: `0` (x below the kink), `x` (above it), or free."""


@dataclass(eq=False)
class _Pieces:
    """The draws of a ranging, merged into groups, and each group's pieces.

    `basis`, `loss` (scaled) and `probability` are the groups of draws with
    one basis row and loss; `constant` marks a zero basis row, whose
    uncovered loss is the loss itself. `uncovered` is each group's piece of
    `u = (L - B theta)^+` and `excess` an uncovered group's piece of `e = (u -
    ζ)^+`. A covered group's `e = (-ζ)^+` is one column for all of them. For a
    cost-of-capital budget, `payout_basis` and `payout_probability` are the
    groups of draws with one basis row, and `tail` their pieces of `t = (B
    theta - η)^+`. A piece only ever becomes free.
    """

    basis: FloatArray
    loss: FloatArray
    probability: FloatArray
    constant: NDArray[np.bool_]
    uncovered: NDArray[np.int8]
    excess: NDArray[np.int8]
    payout_basis: FloatArray | None = None
    payout_probability: FloatArray | None = None
    tail: NDArray[np.int8] | None = None


def _merged(rows: FloatArray, probability: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The distinct rows, and the summed probability of the draws on each."""
    distinct, inverse = np.unique(rows, axis=0, return_inverse=True)
    summed = np.bincount(
        inverse.reshape(-1), weights=probability, minlength=distinct.shape[0]
    )
    return distinct, summed


def _tail_value(values: FloatArray, probability: FloatArray, level: float) -> float:
    """The value at which the largest values reach `1 - level` of the probability."""
    order = np.argsort(-values, kind="stable")
    mass = np.cumsum(probability[order])
    position = int(np.searchsorted(mass, (1.0 - level) * (1.0 - 1e-12)))
    return float(values[order[min(position, values.shape[0] - 1)]])


def _piece(margin: FloatArray) -> NDArray[np.int8]:
    """The piece of `margin^+` a draw is on, free within `RANGING_KINK` of 0."""
    return np.where(
        margin > RANGING_KINK,
        _ABOVE,
        np.where(margin < -RANGING_KINK, _BELOW, _FREE),
    ).astype(np.int8)


def _broken(piece: NDArray[np.int8], margin: FloatArray) -> NDArray[np.bool_]:
    """The draws whose fixed piece of `margin^+` is not the function's at `margin`."""
    above: NDArray[np.bool_] = (piece == _ABOVE) & (margin < 0.0)
    below: NDArray[np.bool_] = (piece == _BELOW) & (margin > 0.0)
    return above | below


def _cvar_weight(objective: Objective) -> float | None:
    """`alpha / (1 - beta)` of a weighted CVaR with a CVaR term, else `None`."""
    if isinstance(objective, WeightedCVaR) and objective.alpha > 0.0:
        return objective.alpha / (1.0 - objective.beta)
    return None


def _priced_tail(context: _Context) -> CostOfCapitalLoad | None:
    """The cost-of-capital pricing whose CVaR a budget adds to the LP, if any."""
    pricing = context.pricing
    if (
        context.bound is not None
        and isinstance(pricing, CostOfCapitalLoad)
        and pricing.rate > 0.0
    ):
        return pricing
    return None


def _pieces(block: _Block, theta: FloatArray, context: _Context) -> _Pieces:
    """Merge the draws and put each on its piece at the design's `theta`.

    ζ and η are the CVaR's value at risk of the uncovered loss and of the
    payout there; a draw within `RANGING_KINK` of a kink starts free.
    """
    k = block.basis.shape[1]
    rows, probability = _merged(
        np.column_stack([block.basis, context.loss / context.scale]),
        context.probability,
    )
    basis, loss = rows[:, :k], rows[:, k]
    constant = ~np.any(basis != 0.0, axis=1)
    margin = loss - basis @ theta
    uncovered = np.where(constant, np.int8(_ABOVE), _piece(margin)).astype(np.int8)
    excess = np.full(loss.shape[0], _BELOW, dtype=np.int8)
    objective = context.objective
    if _cvar_weight(objective) is not None:
        assert isinstance(objective, WeightedCVaR)
        zeta = _tail_value(np.maximum(margin, 0.0), probability, objective.beta)
        excess = _piece(margin - zeta)
    pieces = _Pieces(basis, loss, probability, constant, uncovered, excess)
    pricing = _priced_tail(context)
    if pricing is not None:
        payout_basis, payout_probability = _merged(block.basis, context.probability)
        payout = payout_basis @ theta
        eta = _tail_value(payout, payout_probability, pricing.q)
        pieces.payout_basis = payout_basis
        pieces.payout_probability = payout_probability
        pieces.tail = _piece(payout - eta)
    return pieces


def _fsum(values: FloatArray) -> float:
    return math.fsum(values.tolist())


def _reduced_programme(
    pieces: _Pieces, block: _Block, context: _Context, cost: FloatArray, upper: float
) -> tuple[_solver.Programme, int | None, int | None]:
    """The reduced ranging LP: `min cost @ theta` over the relaxed near-optimal set.

    The amounts are the first columns, as in `_programme`. Returns the
    programme and the columns of ζ and η (`None` without them).
    """
    k = block.basis.shape[1]
    scale = context.scale
    constraints = block.constraints
    objective = context.objective
    weight = _cvar_weight(objective)
    alpha = objective.alpha if isinstance(objective, WeightedCVaR) else 0.0
    basis, loss, p = pieces.basis, pieces.loss, pieces.probability
    builder = _Builder()
    builder.variables(k, constraints.lower / scale, constraints.upper / scale, cost)
    zeta = None if weight is None else builder.variables(1, -math.inf, math.inf, 0.0)
    # The near-optimal row: the fixed pieces summed, and the free columns.
    above = pieces.uncovered == _ABOVE
    on_theta = -(1.0 - alpha) * (p[above] @ basis[above])
    on_zeta = alpha
    bound = upper - (1.0 - alpha) * _fsum(p[above] * loss[above])
    near: list[tuple[NDArray[np.int64], FloatArray]] = []
    free = np.flatnonzero(pieces.uncovered == _FREE)
    if free.size:
        steps = np.arange(free.size)
        u = builder.variables(free.size, 0.0, math.inf, 0.0) + steps
        first = builder.constraints(free.size, loss[free], math.inf)
        _block(builder, first, basis[free], 1.0)
        builder.entries(first + steps, u, 1.0)
        near.append((u, (1.0 - alpha) * p[free]))
        if zeta is not None and weight is not None:
            e = builder.variables(free.size, 0.0, math.inf, 0.0) + steps
            rows = builder.constraints(free.size, 0.0, math.inf) + steps
            builder.entries(rows, e, 1.0)
            builder.entries(rows, u, -1.0)
            builder.entries(rows, np.full(free.size, zeta), 1.0)
            near.append((e, weight * p[free]))
    if zeta is not None and weight is not None:
        tail = above & (pieces.excess == _ABOVE)
        on_theta = on_theta - weight * (p[tail] @ basis[tail])
        on_zeta -= weight * _fsum(p[tail])
        bound -= weight * _fsum(p[tail] * loss[tail])
        kink = np.flatnonzero(above & (pieces.excess == _FREE))
        if kink.size:
            steps = np.arange(kink.size)
            e = builder.variables(kink.size, 0.0, math.inf, 0.0) + steps
            first = builder.constraints(kink.size, loss[kink], math.inf)
            _block(builder, first, basis[kink], 1.0)
            builder.entries(first + steps, e, 1.0)
            builder.entries(first + steps, np.full(kink.size, zeta), 1.0)
            near.append((e, weight * p[kink]))
        covered = pieces.uncovered == _BELOW
        if covered.any():
            e_covered = builder.variables(1, 0.0, math.inf, 0.0)
            row = builder.constraints(1, 0.0, math.inf)
            builder.entries([row, row], [e_covered, zeta], [1.0, 1.0])
            near.append((np.array([e_covered]), np.array([weight * _fsum(p[covered])])))
    row = builder.constraints(1, -math.inf, bound)
    builder.entries(np.full(k, row), np.arange(k), on_theta)
    if zeta is not None:
        builder.entries([row], [zeta], on_zeta)
    for columns, values in near:
        builder.entries(np.full(columns.size, row), columns, values)
    eta = _reduced_budget(builder, pieces, block, context)
    m = constraints.a_ub.shape[0]
    if m:
        first = builder.constraints(m, -math.inf, constraints.b_ub / scale)
        _block(builder, first, constraints.a_ub, 1.0)
    return builder.programme(), zeta, eta


def _reduced_budget(
    builder: _Builder, pieces: _Pieces, block: _Block, context: _Context
) -> int | None:
    """Add the budget's row as `_programme` does, a premium's CVaR reduced.

    Returns the column of η, or `None` without it.
    """
    if context.bound is None:
        return None
    k = block.basis.shape[1]
    scale = context.scale
    expected = context.probability @ block.basis
    pricing = context.pricing
    tail = _priced_tail(context)
    if tail is not None:
        assert pieces.payout_basis is not None
        assert pieces.payout_probability is not None and pieces.tail is not None
        basis, p = pieces.payout_basis, pieces.payout_probability
        weight = tail.rate / (1.0 - tail.q)
        above = pieces.tail == _ABOVE
        eta = builder.variables(1, -math.inf, math.inf, 0.0)
        row = builder.constraints(1, -math.inf, context.bound / scale)
        on_theta = (1.0 - tail.rate) * expected + weight * (p[above] @ basis[above])
        builder.entries(np.full(k, row), np.arange(k), on_theta)
        builder.entries([row], [eta], tail.rate - weight * _fsum(p[above]))
        free = np.flatnonzero(pieces.tail == _FREE)
        if free.size:
            steps = np.arange(free.size)
            t = builder.variables(free.size, 0.0, math.inf, 0.0) + steps
            first = builder.constraints(free.size, 0.0, math.inf)
            builder.entries(first + steps, t, 1.0)
            _block(builder, first, basis[free], -1.0)
            builder.entries(first + steps, np.full(free.size, eta), 1.0)
            builder.entries(np.full(free.size, row), t, weight * p[free])
        return eta
    if isinstance(pricing, ExpectedPayoutMultiple):
        row = builder.constraints(
            1, -math.inf, (context.bound - pricing.expense) / scale
        )
        builder.entries(np.full(k, row), np.arange(k), pricing.multiple * expected)
    else:
        row = builder.constraints(1, -math.inf, context.bound / scale)
        builder.entries(np.full(k, row), np.arange(k), expected)
    return None


def _release(
    pieces: _Pieces, values: FloatArray, k: int, zeta: int | None, eta: int | None
) -> int:
    """Free the groups whose fixed piece a reduced solution breaks; count them."""
    theta = values[:k]
    margin = pieces.loss - pieces.basis @ theta
    broken = _broken(pieces.uncovered, margin)
    count = int(broken.sum())
    if zeta is not None:
        excess = (pieces.uncovered == _ABOVE) & _broken(
            pieces.excess, margin - values[zeta]
        )
        pieces.excess[excess] = _FREE
        count += int(excess.sum())
    pieces.uncovered[broken] = _FREE
    if eta is not None:
        assert pieces.payout_basis is not None and pieces.tail is not None
        tail = _broken(pieces.tail, pieces.payout_basis @ theta - values[eta])
        pieces.tail[tail] = _FREE
        count += int(tail.sum())
    return count


class _FullRanging:
    """The range ends that the reduced LPs leave, solved on the full LP.

    One HiGHS instance holds the programme with the near-optimal row, made
    at the first such end; its first solve starts from the design LP's basis
    with the row basic, each later one from the previous solve's basis.
    """

    def __init__(
        self,
        programme: _solver.Programme,
        basis: _solver.Basis | None,
        context: _Context,
        solves: _Solves,
    ) -> None:
        self._programme = programme
        self._basis = basis
        self._context = context
        self._solves = solves
        self._session: _solver.LPSession | None = None
        self._near = 0

    def end(self, column: int, sign: float, upper: float) -> float:
        if self._session is None:
            self._session = _solver.LPSession(
                self._programme, _solver.RANGING_OPTIONS, basis=self._basis
            )
            self._near = self._session.add_row(
                self._programme.cost, -math.inf, math.inf
            )
        self._session.change_row_bounds(self._near, -math.inf, upper)
        cost = np.zeros(self._programme.n_columns)
        cost[column] = sign
        self._session.change_cost(cost)
        solution = _checked(self._session.solve, self._context, self._solves)
        assert solution.values is not None
        return float(solution.values[column])


def _range_end(
    pieces: _Pieces,
    block: _Block,
    context: _Context,
    solves: _Solves,
    full: _FullRanging,
    end: tuple[int, float, float],
) -> float:
    """The range end `(column, sign, upper)`, from reduced LPs or the full LP."""
    column, sign, upper = end
    k = block.basis.shape[1]
    cost = np.zeros(k)
    cost[column] = sign
    for _ in range(RANGING_ROUND_LIMIT):
        programme, zeta, eta = _reduced_programme(pieces, block, context, cost, upper)
        solution = solves.add(_solver.solve_lp(programme, _solver.RANGING_OPTIONS))
        if solution.status != "optimal" or solution.values is None:
            break
        if not _release(pieces, solution.values, k, zeta, eta):
            return float(solution.values[column])
    return full.end(column, sign, upper)


def _ranges(
    block: _Block,
    solved: _SolvedBlock,
    identified: Sequence[int],
    context: _Context,
    solves: _Solves,
) -> tuple[dict[int, tuple[float, float]], bool]:
    """Range the identified amounts over two near-optimal sets.

    `solved.solution` is the LP's optimal solution. Each near-optimal set is
    the LP's feasible set with one more row, `cost @ x <= optimum + tolerance
    * max(1, |optimum|)`; each amount's smallest and largest value there, at
    each tolerance, is the optimum of an LP. Each is solved on reduced LPs
    (see the comment above `_Pieces`), which share one set of pieces: a group
    freed for one end stays free for the next. An end whose reduced LP is not
    solved to optimality, or that takes `RANGING_ROUND_LIMIT` of them, is
    solved on the full LP (`_FullRanging`). Every solve is counted.

    Returns each amount's wide range in scaled units, and whether every one
    is pinned.
    """
    solution = solved.solution
    assert solution.objective is not None and solution.values is not None
    if not identified:
        return {}, True
    optimum = solution.objective
    pieces = _pieces(block, solution.values[: block.basis.shape[1]], context)
    full = _FullRanging(solved.programme, solution.basis, context, solves)
    found: list[dict[int, tuple[float, float]]] = []
    for tolerance in NEAR_OPTIMAL_TOLERANCES:
        upper = optimum + tolerance * max(1.0, abs(optimum))
        found.append(
            {
                column: (
                    _range_end(
                        pieces, block, context, solves, full, (column, 1.0, upper)
                    ),
                    _range_end(
                        pieces, block, context, solves, full, (column, -1.0, upper)
                    ),
                )
                for column in identified
            }
        )
    wide, narrow = found
    pinned = True
    for column in identified:
        width = wide[column][1] - wide[column][0]
        narrow_width = narrow[column][1] - narrow[column][0]
        if not (narrow_width <= PINNED_SHRINK * width or narrow_width < PINNED_WIDTH):
            pinned = False
    return wide, pinned


def _premium(context: _Context, payout: FloatArray) -> float:
    return context.pricing.premium(payout, context.probability)


def _objective_value(context: _Context, payout: FloatArray) -> float:
    return context.objective.value(context.loss, payout, context.probability)


def _over_budget(context: _Context, premium: float) -> str | None:
    """Describe a premium above the budget, or `None` if it is within it."""
    if context.bound is not None and premium - context.bound > context.slack(
        context.bound
    ):
        return f"the premium {premium:.10g} exceeds the budget {context.bound:.10g}"
    return None


def _verify(
    context: _Context,
    cover: Cover,
    amounts: FloatArray,
    basis: FloatArray | None,
    constraints: LinearConstraints | None,
) -> tuple[FloatArray, float, float]:
    """Recompute a design in numpy; raise `DesignError` on any violation.

    Returns the payout, the objective value and the premium.
    """
    payout = cover.payout_values(context.index)
    problems: list[str] = []
    if basis is not None:
        gap = float(np.max(np.abs(payout - basis @ amounts), initial=0.0))
        if gap > context.slack(0.0):
            problems.append(f"the payout differs from basis @ amounts by {gap:.3g}")
    if constraints is not None:
        below = constraints.lower - amounts
        above = amounts - constraints.upper
        for name, excess, bound in (
            ("a lower bound", below, constraints.lower),
            ("an upper bound", above, constraints.upper),
        ):
            finite = np.isfinite(bound)
            if (
                finite.any()
                and (
                    excess[finite] > np.array([context.slack(b) for b in bound[finite]])
                ).any()
            ):
                problems.append(f"an amount breaks {name}")
        if constraints.a_ub.shape[0]:
            activity = constraints.a_ub @ amounts
            slack = np.array([context.slack(b) for b in constraints.b_ub])
            if (activity - constraints.b_ub > slack).any():
                problems.append("the amounts break a linear constraint of the payout")
    premium = _premium(context, payout)
    over_budget = _over_budget(context, premium)
    if over_budget is not None:
        problems.append(over_budget)
    if context.limit is not None:
        largest = float(payout.max(initial=0.0))
        if largest - context.limit > context.slack(context.limit):
            problems.append(
                f"a payout of {largest:.10g} exceeds the limit {context.limit:.10g}"
            )
    if problems:
        raise context.error(
            DesignError,
            "the design failed its feasibility check: " + "; ".join(problems),
            status="infeasible_solution",
        )
    return payout, _objective_value(context, payout), premium


def _amount_values(
    cover: Cover, names: Sequence[str], amounts: FloatArray
) -> dict[str, float]:
    parameters = cover.payout.parameters()
    return {
        name: float(value) + 0.0
        for name, value in zip(names, amounts, strict=True)
        if isinstance(parameters[name], Free)
    }


def _check_limit_rows(payout: Any, context: _Context) -> None:
    """Raise `DesignError` if a `Limit` on a `SumOf` would not bound the year.

    The limit's rows take each component's largest amount as the most that
    component pays, which needs its basis rows to sum to at most 1.
    """
    if context.limit is None or not isinstance(payout, SumOf):
        return
    above = [
        name
        for name, largest in payout._largest_row_sums(context.index).items()
        if largest > 1.0 + BASIS_ROW_TOLERANCE
    ]
    if above:
        raise context.error(
            DesignError,
            f"the basis rows of the components {above} sum above 1 on the design "
            "sample, so a limit on the amounts does not bound the year's total",
        )


def _read_block(candidate: Cover, context: _Context) -> _Block:
    """Read a cover's basis and constraints; raise `DesignError` if unusable."""
    payout = cast(LinearInAmounts, candidate.payout)
    names = tuple(payout.amount_names())
    basis = np.asarray(payout.basis(context.index), dtype=np.float64)
    if basis.shape != (context.index.shape[0], len(names)):
        raise context.error(
            DesignError,
            f"the payout's basis has shape {basis.shape}, expected "
            f"{(context.index.shape[0], len(names))}",
        )
    if not np.isfinite(basis).all():
        raise context.error(DesignError, "the payout's basis is not finite")
    _check_limit_rows(payout, context)
    constraints = payout.amount_constraints(context.limit)
    if constraints.n_amounts != len(names):
        raise context.error(
            DesignError,
            f"the payout's amount constraints cover {constraints.n_amounts} "
            f"amounts, not {len(names)}",
        )
    empty = [
        name
        for name, low, high in zip(
            names, constraints.lower, constraints.upper, strict=True
        )
        if low > high
    ]
    if empty:
        raise context.error(
            InfeasibleDesignError,
            f"no value of {empty} satisfies its bounds: a fixed amount or a lower "
            "bound lies above the upper bound or the limit",
            status="infeasible",
        )
    parameters = payout.parameters()
    free = np.array([isinstance(parameters[name], Free) for name in names])
    return _Block(names, basis, constraints, free)


def _solve_block(
    block: _Block, context: _Context, solves: _Solves, *, report: bool
) -> _SolvedBlock:
    """Solve a block's amounts and apply the repair, the zeroing and the fill rule.

    With `report`, an LP keeps its basis for the ranges, when asked for.
    """
    names, basis, constraints, free = (
        block.names,
        block.basis,
        block.constraints,
        block.free,
    )
    if (
        isinstance(context.objective, ExpectedUncovered | WeightedCVaR)
        and context.bound is None
        and not np.isfinite(constraints.upper[free]).all()
    ):
        raise context.error(
            DesignError,
            "the expected uncovered loss and the weighted CVaR need a budget or "
            "bounded amounts (a Free upper bound or a Limit): without either, any "
            "payout at least as large as the loss is optimal",
        )
    programme, route, tail_row = _programme(basis, constraints, context)
    ranging = report and route == "lp" and context.options.ranges
    solution = _solve(
        programme, route, context, solves, keep_basis=ranging, tail_row=tail_row
    )
    assert solution.values is not None and solution.objective is not None
    edges, other = _order_edges(constraints)
    amounts = _repair(solution.values[: len(names)] * context.scale, constraints, edges)
    small = (
        free
        & (constraints.lower <= 0.0)
        & (np.abs(amounts) <= ZERO_AMOUNT * context.scale)
    )
    zeroed = small & (amounts != 0.0)
    amounts = _repair(np.where(small, 0.0, amounts), constraints, edges)
    identified = np.any(basis != 0.0, axis=0)
    unidentified = free & ~identified
    amounts = _fill(
        amounts, unidentified, constraints, edges, other, context, solves, names
    )
    return _SolvedBlock(
        amounts=amounts,
        programme=programme,
        solution=solution,
        route=route,
        ranging=ranging,
        identified=identified,
        unidentified=unidentified,
        zeroed=zeroed,
    )


def _resolve_amounts(
    candidate: Cover, names: Sequence[str], amounts: FloatArray, context: _Context
) -> tuple[Cover, dict[str, float]]:
    """Put solved amounts into a cover; raise `DesignError` if they don't fit."""
    values = _amount_values(candidate, names, amounts)
    try:
        return candidate.with_values(values), values
    except CoverDefinitionError as error:
        raise context.error(
            DesignError, f"the solved amounts do not make a valid cover: {error}"
        ) from error


def _check_objective(value: float, solved: _SolvedBlock, context: _Context) -> None:
    """Raise `DesignError` if numpy's objective differs from the solver's."""
    assert solved.solution.objective is not None
    scaled = context.scaled(value)
    if abs(
        scaled - solved.solution.objective
    ) > context.options.feasibility_tolerance * max(1.0, abs(scaled)):
        raise context.error(
            DesignError,
            f"the objective recomputed in numpy ({scaled:.12g}, scaled) differs "
            f"from the solver's ({solved.solution.objective:.12g})",
            status="inconsistent_objective",
        )


def _uniqueness(
    block: _Block,
    solved: _SolvedBlock,
    context: _Context,
    solves: _Solves,
    *,
    report: bool,
) -> tuple[bool | None, dict[int, tuple[float, float]] | None]:
    """The uniqueness report: whether the optimum is unique, and the ranges.

    On the QP route `unique` comes from the eigenvalue rule; on the LP route,
    when ranges are asked for, from the ranges, returned by column in payout
    units. Without `report`, or on the LP route without ranges, both are
    `None`.
    """
    ranged = [
        j for j in range(len(block.names)) if block.free[j] and solved.identified[j]
    ]
    if report and solved.route == "qp":
        if not ranged:
            return True, None
        gram = block.basis[:, ranged].T @ (
            block.basis[:, ranged] * context.probability[:, None]
        )
        eigen = np.linalg.eigvalsh(gram)
        return bool(eigen.min() > MSE_EIGEN_TOLERANCE * eigen.max()), None
    if solved.ranging:
        wide, unique = _ranges(block, solved, ranged, context, solves)
        return unique, {
            j: (low * context.scale + 0.0, high * context.scale + 0.0)
            for j, (low, high) in wide.items()
        }
    return None, None


def _masked_names(names: Sequence[str], mask: NDArray[np.bool_]) -> tuple[str, ...]:
    return tuple(name for name, flag in zip(names, mask, strict=True) if flag)


def _design_amounts(
    candidate: Cover,
    context: _Context,
    solves: _Solves,
    *,
    report: bool,
) -> _Amounts:
    """Solve the amounts of a cover whose thresholds are resolved.

    With `report`, compute the uniqueness report (ranges on the LP route,
    when asked for; the eigenvalue rule on the QP route).
    """
    block = _read_block(candidate, context)
    solved = _solve_block(block, context, solves, report=report)
    cover, values = _resolve_amounts(candidate, block.names, solved.amounts, context)
    _, value, _ = _verify(
        context, cover, solved.amounts, block.basis, block.constraints
    )
    _check_objective(value, solved, context)
    unique, by_column = _uniqueness(block, solved, context, solves, report=report)
    return _Amounts(
        cover=cover,
        values=values,
        objective_value=value,
        route=solved.route,
        method=solved.solution.method,
        unidentified=_masked_names(block.names, solved.unidentified),
        zeroed=_masked_names(block.names, solved.zeroed),
        unique=unique,
        ranges=(
            None
            if by_column is None
            else {block.names[j]: bounds for j, bounds in by_column.items()}
        ),
    )


# -- the grid ----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _GridOutcome:
    """A grid's chosen candidate, its report and its profile.

    `point` is the chosen candidate's values, `objective` its objective in
    payout units and `feasible` the objectives of every candidate that has
    one, scaled, for a later count of ties.
    """

    cover: Cover
    amounts: _Amounts | None
    ties: int
    profile: Any
    counts: dict[str, int]
    point: tuple[float, ...]
    objective: float
    feasible: tuple[float, ...]


def _solved_point(
    template: Cover,
    names: Sequence[str],
    point: Sequence[float],
    context: _Context,
    solves: _Solves,
) -> tuple[float | None, str, str | None]:
    """The objective of a point of the thresholds, its amounts solved exactly.

    Returns the objective in payout units and `"optimal"`; or `None` and
    `"invalid"` when the values make no valid cover (unordered thresholds),
    or `"infeasible"` with the error's message when no payout meets the
    constraints there.
    """
    try:
        candidate = template.with_values(dict(zip(names, point, strict=True)))
    except CoverDefinitionError:
        return None, "invalid", None
    try:
        amounts = _design_amounts(candidate, context, solves, report=False)
    except InfeasibleDesignError as error:
        return None, "infeasible", str(error)
    return amounts.objective_value, "optimal", None


def _grid(
    template: Cover,
    names: Sequence[str],
    context: _Context,
    solves: _Solves,
    *,
    inner: bool,
    finish: bool = True,
) -> _GridOutcome:
    """Search the grids of `names`; solve the amounts at each point if `inner`.

    With `finish`, an inner grid solves the chosen candidate's amounts again
    with the uniqueness report; without it, a refinement does so where it
    ends, and `amounts` is `None`.
    """
    parameters = template.payout.parameters()
    grids: list[tuple[float, ...]] = []
    for name in names:
        parameter = parameters[name]
        assert isinstance(parameter, Free)
        if parameter.grid is None:
            raise context.error(
                DesignError,
                f"a grid search needs a grid on every parameter it searches; "
                f"{name} has none (give Free(..., grid=...))",
            )
        grids.append(parameter.grid)
    size = math.prod(len(grid) for grid in grids)
    if size > context.options.max_grid_points:
        raise context.error(
            DesignError,
            f"the grid has {size} points, more than max_grid_points="
            f"{context.options.max_grid_points}",
        )
    linear = _has_amounts(template.payout)
    rows: list[tuple[tuple[float, ...], float | None, str]] = []
    first_infeasible: str | None = None
    began = time.perf_counter()
    for point in itertools.product(*grids):
        if inner:
            objective, status, message = _solved_point(
                template, names, point, context, solves
            )
            first_infeasible = first_infeasible or message
            rows.append((point, objective, status))
            continue
        try:
            candidate = template.with_values(dict(zip(names, point, strict=True)))
        except CoverDefinitionError:
            rows.append((point, None, "invalid"))
            continue
        payout = candidate.payout_values(context.index)
        status = _grid_status(context, template, candidate, payout, linear)
        value = _objective_value(context, payout) if status == "feasible" else None
        rows.append((point, value, status))
    if not inner:
        solves.seconds += time.perf_counter() - began
    feasible = [
        (position, context.scaled(value))
        for position, (_, value, _) in enumerate(rows)
        if value is not None
    ]
    counts = {
        status: sum(1 for row in rows if row[2] == status)
        for status in ("optimal", "feasible", "infeasible", "invalid")
        + ("over_budget", "over_limit")
    }
    counts = {"candidates": size, **{k: v for k, v in counts.items() if v}}
    if not feasible:
        reason = f"; the first: {first_infeasible}" if first_infeasible else ""
        raise context.error(
            InfeasibleDesignError,
            f"no grid candidate is feasible ({_counts_text(counts)}){reason}",
            status="infeasible",
        )
    best = min(value for _, value in feasible)
    allowance = context.options.feasibility_tolerance * max(1.0, abs(best))
    within = [position for position, value in feasible if value - best <= allowance]
    chosen = within[0]
    point, objective, _ = rows[chosen]
    assert objective is not None
    cover = template.with_values(dict(zip(names, point, strict=True)))
    chosen_amounts = (
        _design_amounts(cover, context, solves, report=True)
        if inner and finish
        else None
    )
    return _GridOutcome(
        cover=cover if chosen_amounts is None else chosen_amounts.cover,
        amounts=chosen_amounts,
        ties=len(within) - 1,
        profile=_profile_table(names, rows),
        counts=counts,
        point=point,
        objective=objective,
        feasible=tuple(value for _, value in feasible),
    )


def _profile_table(
    names: Sequence[str], rows: Sequence[tuple[Sequence[float], float | None, str]]
) -> Any:
    """One row per point: its values, `objective` and `status`."""
    columns: dict[str, Any] = {
        name: pa.array([row[0][i] for row in rows], pa.float64())
        for i, name in enumerate(names)
    }
    columns["objective"] = pa.array([row[1] for row in rows], pa.float64())
    columns["status"] = pa.array([row[2] for row in rows], pa.string())
    return pa.table(columns)


def _counts_text(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in counts.items())


def _breaks_constraints(
    context: _Context, values: FloatArray, constraints: LinearConstraints
) -> bool:
    """Whether resolved amounts break their upper bounds or the rows of `a_ub`."""
    above = values - constraints.upper
    if (above > np.array([context.slack(b) for b in constraints.upper])).any():
        return True
    if constraints.a_ub.shape[0] == 0:
        return False
    slack = np.array([context.slack(b) for b in constraints.b_ub])
    return bool((constraints.a_ub @ values - constraints.b_ub > slack).any())


def _grid_status(
    context: _Context,
    template: Cover,
    candidate: Cover,
    payout: FloatArray,
    linear: bool,
) -> str:
    """Whether a fully resolved candidate meets the budget and the limit."""
    if context.limit is not None:
        if linear:
            # A resolved `MaxOf` answers no amounts; its template states the bound.
            family = cast(
                LinearInAmounts,
                candidate.payout if _has_amounts(candidate.payout) else template.payout,
            )
            _check_limit_rows(candidate.payout, context)
            parameters = candidate.payout.parameters()
            values = np.array(
                [float(cast(float, parameters[name])) for name in family.amount_names()]
            )
            if _breaks_constraints(
                context, values, family.amount_constraints(context.limit)
            ):
                return "over_limit"
        elif float(payout.max(initial=0.0)) - context.limit > context.slack(
            context.limit
        ):
            return "over_limit"
    if context.bound is not None and _premium(context, payout) - context.bound > (
        context.slack(context.bound)
    ):
        return "over_budget"
    return "feasible"


# -- the refinement ----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Refined:
    """A refinement's outcome: the amounts where it ended, ties and its record."""

    amounts: _Amounts
    ties: int
    record: dict[str, Any]
    profile: Any


def _refinement_bounds(
    template: Cover, names: Sequence[str], context: _Context
) -> tuple[list[float], list[float]]:
    """The bounds of the parameters a refinement moves; each must be finite."""
    parameters = template.payout.parameters()
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        parameter = parameters[name]
        assert isinstance(parameter, Free)
        if parameter.upper is None:
            raise context.error(
                DesignError,
                f"DesignOptions(refine=True) moves {name} within its bounds, and it "
                "has no upper bound (give Free(lower, upper, grid=...))",
            )
        lower.append(parameter.lower)
        upper.append(parameter.upper)
    return lower, upper


def _window(
    value: float, grid: Sequence[float], lower: float, upper: float
) -> tuple[float, float]:
    """A grid value's neighbours on the grid; at either end, that side's bound."""
    position = list(grid).index(value)
    below = grid[position - 1] if position > 0 else lower
    above = grid[position + 1] if position + 1 < len(grid) else upper
    return below, above


def _named(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) + 0.0 for name, value in zip(names, values, strict=True)}


def _on_index_values(payout: Any, context: _Context) -> bool:
    """Whether a refinement searches `payout`'s thresholds over the index values.

    A step or binary payout pays from a threshold the years whose severity is
    at least the threshold's, so its objective changes only where a threshold
    crosses one of the sample's index values.
    """
    return type(payout) in (StepPayout, BinaryPayout) and context.index.ndim == 1


def _index_candidates(
    index: FloatArray, below: float, above: float
) -> tuple[float, ...]:
    """The sample's distinct index values in `[below, above]`, and both ends."""
    inside = np.unique(index[(index >= below) & (index <= above)])
    values = {float(value) + 0.0 for value in inside.tolist()}
    return tuple(sorted(values | {below + 0.0, above + 0.0}))


def _paying_years(
    template: Cover, names: Sequence[str], context: _Context
) -> Callable[[tuple[float, ...]], tuple[int, ...] | None]:
    """The class of a point of a step or binary payout's free thresholds.

    A threshold pays the years whose severity is at least its own (`>=`), so
    two values of it that leave the same number of years below them pay the
    same years: the class is that number for every free threshold. Two
    points of one class have the same basis, so the same amounts, objective
    and constraints. `None` when the values make no valid cover.
    """
    payout = cast(StepPayout | BinaryPayout, template.payout)
    sign = 1.0 if payout.tail == "upper" else -1.0
    severity = np.sort(sign * context.index)

    def key(point: tuple[float, ...]) -> tuple[int, ...] | None:
        try:
            template.with_values(_named(names, point))
        except CoverDefinitionError:
            return None
        thresholds = sign * np.asarray(point, dtype=np.float64)
        below = np.searchsorted(severity, thresholds, side="left")
        return tuple(int(count) for count in below.tolist())

    return key


def _refine(
    template: Cover,
    names: Sequence[str],
    outcome: _GridOutcome,
    bounds: tuple[list[float], list[float]],
    context: _Context,
    solves: _Solves,
) -> _Refined:
    """Refine a grid's chosen candidate over `names`.

    Each threshold's window is the chosen value's neighbours on its grid (or
    its bound, at an end of the grid). On a step or binary payout the search
    (`_solver.coordinate_search`) tries, threshold by threshold, the sample's
    distinct index values in the window and its two ends, one per set of
    paying years: in each pass the values nearest the current one, 2 in the
    first and twice as many in each pass after, until a pass that tried
    every value moves no threshold, or until `refine_max_points` points.
    Otherwise a compass search
    (`_solver.compass_search`) starts at the chosen candidate with a step per
    parameter of half the window's larger side, and stops when every step is
    at most `refine_tolerance` times its bounds' width, or after
    `refine_max_points` points.

    At every point either evaluates, the amounts are solved exactly, as at a
    grid point; a point whose values make no valid cover, or where no payout
    meets the constraints, has no objective. Either moves only to a point
    whose objective lies below the current one by more than the tie
    tolerance of the grid, so it never ends worse than the grid.

    Where it ends, the amounts are solved again with the uniqueness report.
    `ties` counts the grid's candidates within the tie tolerance of that
    objective, as on the grid; the search's own points are not counted.
    """
    options = context.options
    parameters = template.payout.parameters()
    lower, upper = bounds
    windows: list[tuple[float, float]] = []
    for name, value, low, high in zip(names, outcome.point, lower, upper, strict=True):
        grid = cast(Free, parameters[name]).grid
        assert grid is not None  # the grid has checked it
        windows.append(_window(value, grid, low, high))
    objectives: dict[tuple[float, ...], float | None] = {}

    def evaluate(point: tuple[float, ...]) -> tuple[float | None, str]:
        objective, status, _ = _solved_point(template, names, point, context, solves)
        objectives[point] = objective
        return (None if objective is None else context.scaled(objective)), status

    def allowance(value: float) -> float:
        return options.feasibility_tolerance * max(1.0, abs(value))

    start = context.scaled(outcome.objective)
    search: _solver.Search | _solver.CoordinateSearch
    if _on_index_values(template.payout, context):
        candidates = [
            _index_candidates(context.index, below, above) for below, above in windows
        ]
        search = _solver.coordinate_search(
            evaluate,
            outcome.point,
            start,
            candidates=candidates,
            key=_paying_years(template, names, context),
            max_points=options.refine_max_points,
            allowance=allowance,
        )
        settings: dict[str, Any] = {
            "method": INDEX_VALUES_METHOD,
            "max_points": options.refine_max_points,
            "windows": {
                name: [below + 0.0, above + 0.0]
                for name, (below, above) in zip(names, windows, strict=True)
            },
            "candidates": {
                name: len(values)
                for name, values in zip(names, candidates, strict=True)
            },
        }
        closing: dict[str, Any] = {"passes": search.passes, "stopped": search.stopped}
    else:
        steps = [
            max(value - below, above - value) / 2.0
            for value, (below, above) in zip(outcome.point, windows, strict=True)
        ]
        smallest = [
            options.refine_tolerance * (high - low)
            for low, high in zip(lower, upper, strict=True)
        ]
        compass = _solver.compass_search(
            evaluate,
            outcome.point,
            start,
            steps=steps,
            smallest=smallest,
            lower=lower,
            upper=upper,
            max_points=options.refine_max_points,
            allowance=allowance,
        )
        search = compass
        settings = {
            "method": REFINE_METHOD,
            "tolerance": options.refine_tolerance,
            "max_points": options.refine_max_points,
            "initial_steps": _named(names, steps),
        }
        closing = {
            "final_steps": _named(names, compass.steps),
            "stopped": compass.stopped,
        }
    moved = search.moves > 0
    cover = template.with_values(_named(names, search.best))
    amounts = _design_amounts(cover, context, solves, report=True)
    end = amounts.objective_value
    final = context.scaled(end)
    if final > search.objective + allowance(search.objective):
        raise context.error(
            DesignError,
            f"the refined point's objective solved again ({final:.12g}, scaled) "
            f"lies above the one the search found there ({search.objective:.12g})",
            status="inconsistent_objective",
        )
    if moved:
        ties = sum(1 for value in outcome.feasible if value - final <= allowance(final))
    else:
        ties = outcome.ties
    counts: dict[str, int] = {}
    for point in search.points:
        counts[point.status] = counts.get(point.status, 0) + 1
    record = {
        **settings,
        "start": _named(names, outcome.point),
        "start_objective": outcome.objective + 0.0,
        "end": _named(names, search.best),
        "objective": end + 0.0,
        "gain": max(0.0, outcome.objective - end) + 0.0,
        "points": len(search.points),
        "counts": dict(sorted(counts.items())),
        "moves": search.moves,
        **closing,
    }
    rows = [
        (point.values, objectives[point.values], point.status)
        for point in search.points
    ]
    return _Refined(amounts, ties, record, _profile_table(names, rows))


# -- the public function -----------------------------------------------------


def _solver_options(route: Literal["lp", "qp"], ranging: bool) -> dict[str, Any]:
    """The solver options of a design on the LP or QP route, for its record."""
    return {
        "solve": dict(_solver.QP_OPTIONS if route == "qp" else _solver.LP_OPTIONS),
        "ranging": dict(_solver.RANGING_OPTIONS) if ranging else None,
        "active_set": (
            {
                "first_columns": _solver.ACTIVE_SET_FIRST_COLUMNS,
                "max_columns": _solver.ACTIVE_SET_MAX_COLUMNS,
                "iteration_limit": _solver.ACTIVE_SET_ITERATION_LIMIT,
            }
            if route == "qp"
            else None
        ),
        "cutting_planes": (
            {
                "tolerance": _solver.CUTTING_PLANE_TOLERANCE,
                "limit": _solver.CUTTING_PLANE_LIMIT,
            }
            if route == "qp"
            else None
        ),
    }


def _check_inputs(
    template: Any,
    sample: Any,
    objective: Any,
    pricing: Any,
    budget: Any,
    limit: Any,
    options: Any,
) -> None:
    if not isinstance(template, Cover):
        raise TypeError(f"template must be a Cover, got {type(template).__name__}")
    _check_arguments(sample, objective, pricing, budget, limit, options)


def _check_terms(
    sample: Any,
    objective: Any,
    pricing: Any,
    budget: Any,
    limit: Any,
    options: Any,
) -> None:
    """Check the terms of a design without a grid route: `design_joint`'s.

    The arguments besides the templates are checked as `design` checks them.
    Its thresholds are fixed, so it has no grid point to refine, and
    `DesignOptions(refine=True)` raises `DesignError`.
    """
    _check_arguments(sample, objective, pricing, budget, limit, options)
    if options.refine:
        raise DesignError(
            "design_joint has no grid route: its thresholds are fixed (a free "
            "threshold raises), so DesignOptions(refine=True) has no grid point "
            "to refine; leave refine off, and compare threshold choices by running "
            "design_joint once per choice"
        )


def _check_arguments(
    sample: Any,
    objective: Any,
    pricing: Any,
    budget: Any,
    limit: Any,
    options: Any,
) -> None:
    """Check the arguments a design takes besides its template or templates."""
    checks: tuple[tuple[str, Any, Any, str], ...] = (
        ("sample", sample, OutcomeSample, "an OutcomeSample"),
        (
            "objective",
            objective,
            (MSE, ExpectedUncovered, WeightedCVaR, UserObjective),
            "MSE(), ExpectedUncovered(), WeightedCVaR(alpha, beta) or a UserObjective",
        ),
        ("pricing", pricing, _PRICINGS, _PRICINGS_TEXT),
        ("options", options, DesignOptions, "DesignOptions"),
    )
    for name, value, kind, text in checks:
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be {text}, got {type(value).__name__}")
    if budget is not None and not isinstance(budget, Budget):
        raise TypeError(f"budget must be a Budget or None, got {type(budget).__name__}")
    if limit is not None and not isinstance(limit, Limit):
        raise TypeError(f"limit must be a Limit or None, got {type(limit).__name__}")


def _check_template(template: Cover, sample: OutcomeSample) -> None:
    if not template.free_parameters():
        raise ValueError(
            "the template has no free parameters: there is nothing to design; "
            "evaluate the cover with evaluate()"
        )
    _check_units(template, sample)
    asset = sample.asset_ids[0]
    if template.asset_id is not None and template.asset_id != asset:
        raise SampleError(
            f"the cover is written for asset {template.asset_id!r}, but the "
            f"sample holds asset {asset!r}"
        )


def _check_units(
    template: Cover, sample: OutcomeSample, asset: str | None = None
) -> None:
    """Raise `SampleError` unless `template` reads `sample`, for `asset`'s column.

    The payout unit must be the loss unit, and the index must read what the
    sample holds: its unit, on a component's view (`sample.component(name)`)
    the component's tail, and, where the sample records one, its hazard.
    """
    if template.payout_unit != sample.loss_unit:
        raise SampleError(
            f"the cover pays in {template.payout_unit!r}, but the sample's loss "
            f"is in {sample.loss_unit!r}"
        )
    _check_reads(template.index_definition, sample, asset)


def _no_refinement(objective: Objective, context: _Context) -> NoReturn:
    """Raise `DesignError`: a grid in numpy has no exact amounts to refine around."""
    reason = (
        "a UserObjective is evaluated in numpy"
        if isinstance(objective, UserObjective)
        else "a payout that is not linear in its amounts is evaluated in numpy"
    )
    raise context.error(
        DesignError,
        "DesignOptions(refine=True) solves the amounts exactly, by an LP or a QP, "
        f"at every point it evaluates, but {reason} on a grid over every free "
        "parameter, its amounts included, so no LP or QP solves them; leave "
        "refine off, or search a finer grid",
    )


def _tail_flags(
    objective: Objective, loss: FloatArray, probability: FloatArray
) -> tuple[float, float | None, str | None]:
    p_zero = math.fsum(probability[loss == 0.0].tolist()) + 0.0
    if not isinstance(objective, WeightedCVaR):
        return p_zero, None, None
    tail = (1.0 - objective.beta) / float(probability.min())
    flag: str | None = None
    if objective.beta <= p_zero + DEGENERATE_TAIL_TOLERANCE:
        flag = "expected_uncovered"
    elif tail <= 1.0 + DEGENERATE_TAIL_TOLERANCE:
        flag = "single_outcome"
    return p_zero, tail, flag


def design(
    template: Cover,
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing = PurePremium(),
    budget: Budget | None = None,
    limit: Limit | None = None,
    options: DesignOptions = DesignOptions(),
) -> DesignResult:
    """Design a cover: choose the template's free parameters on `sample`.

    `template` is a `Cover` with `Free` parameters, written for the sample's
    asset (or for none), paying in the sample's loss unit on an index in the
    sample's index unit, or on the components of a sample with index
    components (see the module docstring). `objective` is `MSE()`,
    `ExpectedUncovered()`, `WeightedCVaR(alpha, beta)` or a `UserObjective`.
    `pricing` prices the premium on the sample; `budget` (`Budget(ratio=...)` or
    `Budget(amount=...)`) bounds it; `limit` (`Limit(amount)`) bounds every
    payout. `options` (`DesignOptions`) sets the grid cap, the tolerances,
    the fill rule of unidentified amounts, whether the LP route ranges its
    optimal set and whether a grid's best point is refined. The routes,
    checks and reports are described in the module docstring.

    The expected uncovered loss and the weighted CVaR need a budget or
    bounded amounts; MSE needs neither.

    Returns a `DesignResult` with the resolved cover and its record. Raises
    `SolverUnavailableError` without the `parametric` extra on the LP and QP
    routes; `InfeasibleDesignError` when no payout meets the constraints,
    such as a budget below the pricing's fixed expense;
    `UnboundedDesignError` for an unbounded objective (never for `MSE`);
    `SolverLimitError` when a solve stops at its iteration or time limit;
    `DesignError` for any other failed solve or check, and for
    `DesignOptions(refine=True)` on a grid evaluated in numpy or on a
    threshold without a finite upper bound; `SampleError` when the
    sample holds several assets (take one with `sample.asset(asset_id)`), the
    template's units or asset differ from the sample's, its index's tail
    differs from the component's on a component's view or on a sample of
    several hazards, its index's `hazard_name` differs from the hazard the
    sample records, or the sample lacks a component the template reads;
    `ValueError` for a template without free
    parameters; and `TypeError` for arguments of the wrong type.
    """
    _check_inputs(template, sample, objective, pricing, budget, limit, options)
    _require_one_asset(sample, "design")
    _check_template(template, sample)
    loss = np.array(sample.loss[:, 0], dtype=np.float64)
    index = np.array(
        _cover_index_values(template.index_definition, sample), dtype=np.float64
    )
    probability = np.array(sample.probability, dtype=np.float64)
    largest = float(loss.max())
    scale = largest if largest > SCALE_FLOOR else 1.0
    expected_loss = math.fsum((probability * loss).tolist()) + 0.0
    bound = None if budget is None else budget.bound(expected_loss)
    context = _Context(
        asset_id=sample.asset_ids[0],
        index=index,
        loss=loss,
        probability=probability,
        scale=scale,
        expected_loss=expected_loss,
        bound=bound,
        limit=None if limit is None else limit.amount,
        objective=objective,
        pricing=pricing,
        options=options,
    )
    payout = template.payout
    free = template.free_parameters()
    solves = _Solves()
    linear = _has_amounts(payout) and not isinstance(objective, UserObjective)
    if options.refine and not linear:
        _no_refinement(objective, context)
    ties = 0
    profile: Any = None
    counts: dict[str, int] | None = None
    amounts: _Amounts | None
    refined: _Refined | None = None
    if linear:
        _solver.require()
        amount_names = set(cast(LinearInAmounts, payout).amount_names())
        shape = [name for name in free if name not in amount_names]
        if shape:
            # A refinement's bounds are checked before the grid runs.
            bounds = (
                _refinement_bounds(template, shape, context) if options.refine else None
            )
            outcome = _grid(
                template, shape, context, solves, inner=True, finish=bounds is None
            )
            amounts, ties = outcome.amounts, outcome.ties
            profile, counts = outcome.profile, outcome.counts
            if bounds is not None:
                refined = _refine(template, shape, outcome, bounds, context, solves)
                amounts, ties = refined.amounts, refined.ties
        else:
            amounts = _design_amounts(template, context, solves, report=True)
        assert amounts is not None
        cover = amounts.cover
        route: Route = "grid" if shape else amounts.route
        amounts_route: Literal["lp", "qp", "grid"] = amounts.route
        unique = amounts.unique
        if ties:
            unique = False
        solver: dict[str, str] | None = _solver.identity()
        ranging = amounts.route == "lp" and options.ranges
        solver_options = _solver_options(amounts.route, ranging)
        message = "" if solves.last is None else solves.last.message
        solved_by: _solver.Method | None = amounts.method
    else:
        outcome = _grid(template, free, context, solves, inner=False)
        cover, ties, profile, counts = (
            outcome.cover,
            outcome.ties,
            outcome.profile,
            outcome.counts,
        )
        amounts = None
        route, amounts_route = "grid", "grid"
        unique = False if ties else None
        solver, solver_options, message = None, {}, "best grid candidate"
        solved_by = None
    _, objective_value, premium = _verify(context, cover, np.zeros(0), None, None)
    p_zero, tail, flag = _tail_flags(objective, loss, probability)
    values = cover.payout.parameters()
    parameters = {name: float(cast(float, values[name])) + 0.0 for name in free}
    budget_document = None if budget is None else budget.to_document()
    limit_document = None if limit is None else limit.to_document()
    design_id = content_hash(
        {
            "budget": budget_document,
            "limit": limit_document,
            "objective": objective.to_document(),
            "options": options.to_document(),
            "pricing": pricing.to_document(),
            "sample_hash": sample.content_hash,
            "semantics_version": DESIGN_SEMANTICS_VERSION,
            "template_hash": template.definition_hash,
        }
    )
    return DesignResult(
        cover=cover,
        template_hash=template.definition_hash,
        design_id=design_id,
        parameters=parameters,
        objective=objective.to_document(),
        objective_value=objective_value,
        pricing=pricing.to_document(),
        budget=budget_document,
        limit=limit_document,
        options=options,
        premium=premium,
        expected_loss=expected_loss,
        budget_bound=bound,
        budget_use=None if bound is None or bound <= 0.0 else premium / bound,
        route=route,
        amounts=amounts_route,
        solver=solver,
        status="optimal",
        solver_message=message,
        iterations=solves.iterations,
        solves=solves.count,
        seconds=solves.seconds,
        solver_options=solver_options,
        unidentified=() if amounts is None else amounts.unidentified,
        fill=options.fill,
        zeroed=() if amounts is None else amounts.zeroed,
        ranges=None if amounts is None else amounts.ranges,
        unique=unique,
        ties=ties,
        profile=profile,
        grid=counts,
        p_zero_loss=p_zero,
        tail_outcomes=tail,
        degenerate_tail=flag,
        design_sample=sample,
        solved_by=solved_by,
        fallbacks=solves.fallbacks,
        refinement=None if refined is None else refined.record,
        refinement_profile=None if refined is None else refined.profile,
    )
