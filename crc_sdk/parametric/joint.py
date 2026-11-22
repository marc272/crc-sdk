"""Design covers jointly: one objective on the asset owner's total, a cover per asset.

`design_joint(templates, sample, objective, *, pricing, budget, limit, options,
scope)` takes one template per asset of a sample of several assets and
returns a `JointDesignResult`: the resolved covers and the design's record.
`design` stays the function for one asset.

**What it optimises.** On each draw `d`, `L_d = sum_j L_{d,j}` is the owner's
total loss and `pi_d = sum_j pi_{d,j}` the total payout. The objective (`MSE`,
`ExpectedUncovered` or `WeightedCVaR`) and the premium (`PurePremium` or
`CostOfCapitalLoad`) are taken on `L_d` and `pi_d`. `Budget(ratio=k)` means
premium <= `k * E[L]` on the total, and `Budget(amount=B)` means premium <= `B`
on the total. Each asset keeps its own cover, on its own index, and each cover
pays by its own terms.

**How it is solved.** The LP or QP of `design`, on the block basis
`[B_1 | ... | B_m]` with the amounts in asset order, the covers' constraints
stacked block by block and the summed loss. The scale is the largest summed
loss of a draw. The QP is solved as `design` solves it: by the dense
active-set method up to 100 columns, and by HiGHS, with its limits and the
dense method as its fallback up to 500 columns, above that; under a
cost-of-capital budget, by cutting planes on the premium's CVaR, each step
such a QP in the amounts. A solve that stops at a limit raises
`SolverLimitError`. `solved_by` and `fallbacks` record it.

**Fixed thresholds.** Only amounts are free in a joint design. A free
threshold, a `UserObjective` or a payout without `LinearInAmounts` raises
`DesignError`, so a joint design has no grid: `ties` is 0 and `profile` and
`grid` are `None`. To compare threshold choices, run `design_joint` once per
choice.

**Non-unique optima.** The covers enter the objective only through their sum,
so several splits of the same total payout across the assets can be optimal,
and the amounts of one asset are often not identified on their own. The report
of `design` applies on the block basis: `fill` per asset,
`DesignOptions(ranges=True)` for every identified amount of every asset on the
LP route, and the eigenvalue rule on the QP route. Unless `unique` is true, the
per-asset amounts are one member of an optimal set.

**Clusters.** Under `scope=Clustered(assignment, values=..., joint=...)`,
`design_joint` takes one template per cluster of the owner's assets, and each
cluster's cover pays on the value-weighted mean of its assets' indices, a
`ClusterIndex`, against the sum of their losses. With `joint=True` the clusters
are designed as `Joint()` designs the assets, on the owner's total; with
`joint=False` each cluster is its own design problem, with its own budget. It
returns a `ClusteredDesignResult`, keyed by cluster. The assignment is stated:
the module clusters nothing itself.

Everything here works in memory and does no I/O.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric import _solver
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.design import (
    DESIGN_SEMANTICS_VERSION,
    SCALE_FLOOR,
    DesignOptions,
    Fill,
    _Block,
    _check_objective,
    _check_terms,
    _check_units,
    _Context,
    _masked_names,
    _objective_value,
    _over_budget,
    _premium,
    _read_block,
    _resolve_amounts,
    _solve_block,
    _solver_options,
    _Solves,
    _tail_flags,
    _uniqueness,
    _verify,
)
from crc_sdk.parametric.errors import DesignError, SampleError
from crc_sdk.parametric.index import ClusterIndex, CompositeIndex, IndexDefinition
from crc_sdk.parametric.objectives import Objective, UserObjective
from crc_sdk.parametric.parameters import _DefinitionModel
from crc_sdk.parametric.payouts import LinearConstraints, LinearInAmounts
from crc_sdk.parametric.pricing import Pricing, PurePremium
from crc_sdk.parametric.sample import OutcomeSample

__all__ = [
    "ClusterDesign",
    "Clustered",
    "ClusteredDesignResult",
    "Individual",
    "Joint",
    "JointDesignResult",
    "covers_hash",
    "design_joint",
]

FloatArray = NDArray[np.float64]

_ID = re.compile(r"^\S(?:[\s\S]*\S)?$")


class Individual(_DefinitionModel):
    """One design per asset, each with its own objective and budget."""

    kind: Literal["individual"] = "individual"

    def __init__(self) -> None:
        super().__init__()

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "individual"}`."""
        return {"kind": "individual"}


class Joint(_DefinitionModel):
    """One objective and one budget on the asset owner's total, one cover per asset."""

    kind: Literal["joint"] = "joint"

    def __init__(self) -> None:
        super().__init__()

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "joint"}`."""
        return {"kind": "joint"}


def covers_hash(covers: Mapping[str, Cover]) -> str:
    """Return the `"sha256:"` hash of a set of covers, one per asset or cluster.

    The hash is that of the canonical JSON object from asset id to the
    cover's `definition_hash`; canonical JSON sorts its keys, so the order of
    `covers` doesn't matter.
    """
    return content_hash(
        {asset: cover.definition_hash for asset, cover in covers.items()}
    )


@dataclass(frozen=True, eq=False)
class JointDesignResult:
    """Covers designed jointly on the asset owner's total, and the design's record.

    It holds the record of a `DesignResult`, with these differences:

    - `covers` maps each asset id to its resolved cover, in `asset_ids` order;
      `template_hashes` maps each asset id to its template's
      `definition_hash`; `covers_hash` is `covers_hash(covers)`; `design_id`
      is the `"sha256:"` hash of the template hashes in asset order, the
      scope, the objective, the pricing, the budget, the limit, the options
      and the design sample's hash;
    - `parameters`, `unidentified`, `zeroed` and `ranges` are keyed by asset
      id, then by the payout's flat parameter names (`"levels[0]"`, ...);
    - `objective_value`, `premium`, `expected_loss`, `budget_bound`,
      `budget_use`, `p_zero_loss`, `tail_outcomes` and `degenerate_tail` are
      on the total: the sum over the assets, draw by draw;
    - `route` and `amounts` are `"lp"` or `"qp"`; there is no grid, so `ties`
      is 0 and `profile` and `grid` are `None`;
    - `solved_by` is `"highs"`, `"active_set"` or `"cutting_planes"` and
      `fallbacks` is 0 or 1: the joint design is one solve.

    Unless `unique` is true, the per-asset amounts are one member of an
    optimal set: they are the solver's choice of how to split the total
    payout across the assets.
    """

    covers: Mapping[str, Cover]
    template_hashes: Mapping[str, str]
    covers_hash: str
    design_id: str
    scope: Mapping[str, Any]
    parameters: Mapping[str, Mapping[str, float]]
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
    route: Literal["lp", "qp"]
    amounts: Literal["lp", "qp"]
    solver: Mapping[str, str]
    status: str
    solver_message: str
    iterations: int
    solves: int
    seconds: float
    solver_options: Mapping[str, Any]
    unidentified: Mapping[str, tuple[str, ...]]
    fill: Fill
    zeroed: Mapping[str, tuple[str, ...]]
    ranges: Mapping[str, Mapping[str, tuple[float, float]]] | None
    unique: bool | None
    ties: int
    profile: None
    grid: None
    p_zero_loss: float
    tail_outcomes: float | None
    degenerate_tail: str | None
    design_sample: OutcomeSample = field(repr=False)
    solved_by: _solver.Method | None = None
    fallbacks: int = 0

    @property
    def asset_ids(self) -> tuple[str, ...]:
        """The assets of the design, in the design sample's order."""
        return tuple(self.covers)

    @property
    def sample_hash(self) -> str:
        """The design sample's `content_hash`."""
        return self.design_sample.content_hash

    def to_record(self) -> dict[str, Any]:
        """Return the design record as plain JSON types."""
        ranges = (
            None
            if self.ranges is None
            else {
                asset: {name: [low, high] for name, (low, high) in amounts.items()}
                for asset, amounts in self.ranges.items()
            }
        )
        return {
            "design_id": self.design_id,
            "scope": dict(self.scope),
            "asset_ids": list(self.asset_ids),
            "template_hashes": dict(self.template_hashes),
            "cover_hashes": {
                asset: cover.definition_hash for asset, cover in self.covers.items()
            },
            "covers_hash": self.covers_hash,
            "sample_hash": self.sample_hash,
            "objective": copy.deepcopy(dict(self.objective)),
            "objective_value": self.objective_value,
            "pricing": dict(self.pricing),
            "budget": None if self.budget is None else dict(self.budget),
            "limit": None if self.limit is None else dict(self.limit),
            "options": self.options.to_document(),
            "parameters": {
                asset: dict(values) for asset, values in self.parameters.items()
            },
            "premium": self.premium,
            "expected_loss": self.expected_loss,
            "budget_bound": self.budget_bound,
            "budget_use": self.budget_use,
            "route": self.route,
            "amounts": self.amounts,
            "solver": dict(self.solver),
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
            "unidentified": {
                asset: list(names) for asset, names in self.unidentified.items()
            },
            "fill": self.fill,
            "zeroed": {asset: list(names) for asset, names in self.zeroed.items()},
            "ranges": ranges,
            "unique": self.unique,
            "ties": self.ties,
            "grid": self.grid,
            "p_zero_loss": self.p_zero_loss,
            "tail_outcomes": self.tail_outcomes,
            "degenerate_tail": self.degenerate_tail,
        }


def _check_templates(templates: Any, scope: Any) -> tuple[Cover, ...]:
    if not isinstance(scope, (Joint, Clustered)):
        raise TypeError(
            f"scope must be Joint() or Clustered(...), got {type(scope).__name__}"
        )
    if isinstance(templates, (str, bytes, Cover, Mapping)) or not isinstance(
        templates, Sequence
    ):
        each = (
            "one per cluster, in the order of scope.cluster_ids"
            if isinstance(scope, Clustered)
            else "one per asset, in the order of sample.asset_ids"
        )
        raise TypeError(
            f"templates must be a sequence of Covers, {each}, got "
            f"{type(templates).__name__}"
        )
    for template in templates:
        if not isinstance(template, Cover):
            raise TypeError(
                f"each template must be a Cover, got {type(template).__name__}"
            )
    return tuple(templates)


def _check_problem(
    templates: Sequence[Cover], sample: OutcomeSample, objective: Objective
) -> None:
    """Check that the templates suit the sample and that the problem is linear."""
    assets = sample.asset_ids
    if len(assets) == 1:
        raise DesignError(
            f"design_joint designs several assets together, but this sample holds "
            f"one, {assets[0]!r}: use design",
            asset_id=assets[0],
        )
    if isinstance(objective, UserObjective):
        raise DesignError(
            "design_joint solves a linear or quadratic programme, so it takes "
            "MSE(), ExpectedUncovered() or WeightedCVaR(alpha, beta); a "
            "UserObjective is evaluated on a grid, and a joint design has none"
        )
    if len(templates) != len(assets):
        raise SampleError(
            f"design_joint takes one template per asset, but it got "
            f"{len(templates)} for the {len(assets)} assets {list(assets)}"
        )
    for asset, template in zip(assets, templates, strict=True):
        _check_units(template, sample, asset)
        if template.asset_id is not None and template.asset_id != asset:
            raise SampleError(
                f"the template for asset {asset!r} is written for asset "
                f"{template.asset_id!r}: the templates follow sample.asset_ids"
            )
        payout = template.payout
        if not isinstance(payout, LinearInAmounts):
            raise DesignError(
                f"asset {asset!r}: a joint design needs a payout linear in its "
                f"amounts (LinearInAmounts); {payout.kind!r} is not",
                asset_id=asset,
            )
        amounts = set(payout.amount_names())
        thresholds = [
            name for name in template.free_parameters() if name not in amounts
        ]
        if thresholds:
            raise DesignError(
                f"asset {asset!r}: a joint design takes fixed thresholds only, but "
                f"{thresholds} are free; to compare threshold choices, run "
                "design_joint once per choice",
                asset_id=asset,
            )
    if not any(template.free_parameters() for template in templates):
        raise ValueError(
            "no template has free parameters: there is nothing to design; "
            "evaluate the covers instead"
        )


def _for_asset(template: Cover, asset: str) -> Cover:
    """The template, written for `asset`."""
    if template.asset_id == asset:
        return template
    return Cover(
        template.index,
        template.payout,
        payout_unit=template.payout_unit,
        asset_id=asset,
        label=template.label,
    )


def _stacked(blocks: Sequence[_Block], assets: Sequence[str]) -> _Block:
    """Put the covers' blocks side by side, their constraints block by block."""
    n_amounts = [block.constraints.n_amounts for block in blocks]
    n_rows = [block.constraints.a_ub.shape[0] for block in blocks]
    a_ub = np.zeros((sum(n_rows), sum(n_amounts)), dtype=np.float64)
    row, column = 0, 0
    for block in blocks:
        rows, columns = block.constraints.a_ub.shape
        a_ub[row : row + rows, column : column + columns] = block.constraints.a_ub
        row += rows
        column += columns
    return _Block(
        names=tuple(
            f"{asset}: {name}"
            for asset, block in zip(assets, blocks, strict=True)
            for name in block.names
        ),
        basis=np.hstack([block.basis for block in blocks]),
        constraints=LinearConstraints(
            a_ub=a_ub,
            b_ub=np.concatenate([block.constraints.b_ub for block in blocks]),
            lower=np.concatenate([block.constraints.lower for block in blocks]),
            upper=np.concatenate([block.constraints.upper for block in blocks]),
        ),
        free=np.concatenate([block.free for block in blocks]),
    )


@dataclass(frozen=True, eq=False)
class _JointSolve:
    """One design problem over one or more covers, solved and checked."""

    covers: dict[str, Cover]
    parameters: dict[str, dict[str, float]]
    unidentified: dict[str, tuple[str, ...]]
    zeroed: dict[str, tuple[str, ...]]
    ranges: dict[str, dict[str, tuple[float, float]]] | None
    premium: float
    expected_loss: float
    bound: float | None
    objective_value: float
    unique: bool | None
    route: Literal["lp", "qp"]
    ranging: bool
    solves: _Solves
    method: _solver.Method
    p_zero: float
    tail: float | None
    flag: str | None

    @property
    def budget_use(self) -> float | None:
        bound = self.bound
        return None if bound is None or bound <= 0.0 else self.premium / bound


def _solve_joint(
    keys: Sequence[str],
    named: Sequence[Cover],
    index: FloatArray,
    loss: FloatArray,
    probability: FloatArray,
    *,
    objective: Objective,
    pricing: Pricing,
    budget: Budget | None,
    limit: Limit | None,
    options: DesignOptions,
    subject: str | None = None,
) -> _JointSolve:
    """Solve one objective and one budget on the sum of the covers' columns.

    `index` and `loss` have one column per cover, in the order of `keys`.
    `subject` names the problem in errors; `None` is the joint design.
    """
    total_loss: FloatArray = loss.sum(axis=1)
    largest = float(total_loss.max())
    scale = largest if largest > SCALE_FLOOR else 1.0
    expected_loss = math.fsum((probability * total_loss).tolist()) + 0.0
    bound = None if budget is None else budget.bound(expected_loss)
    limit_amount = None if limit is None else limit.amount

    # Each cover is read and checked on its own column, at the scale of the
    # total; the budget and the objective are the total's.
    parts = [
        _Context(
            asset_id=key,
            index=index[:, column],
            loss=loss[:, column],
            probability=probability,
            scale=scale,
            expected_loss=math.fsum((probability * loss[:, column]).tolist()) + 0.0,
            bound=None,
            limit=limit_amount,
            objective=objective,
            pricing=pricing,
            options=options,
        )
        for column, key in enumerate(keys)
    ]
    total = _Context(
        asset_id=subject,
        index=index,
        loss=total_loss,
        probability=probability,
        scale=scale,
        expected_loss=expected_loss,
        bound=bound,
        limit=None,
        objective=objective,
        pricing=pricing,
        options=options,
    )
    blocks = [
        _read_block(cover, part) for cover, part in zip(named, parts, strict=True)
    ]
    stacked = _stacked(blocks, keys)
    solves = _Solves()
    solved = _solve_block(stacked, total, solves, report=True)

    covers: dict[str, Cover] = {}
    parameters: dict[str, dict[str, float]] = {}
    unidentified: dict[str, tuple[str, ...]] = {}
    zeroed: dict[str, tuple[str, ...]] = {}
    positions: dict[int, tuple[str, str]] = {}
    total_payout: FloatArray = np.zeros_like(total_loss)
    start = 0
    for key, template, block, part in zip(keys, named, blocks, parts, strict=True):
        stop = start + len(block.names)
        amounts = solved.amounts[start:stop]
        cover, _ = _resolve_amounts(template, block.names, amounts, part)
        payout, _, _ = _verify(part, cover, amounts, block.basis, block.constraints)
        total_payout = total_payout + payout
        values = cover.payout.parameters()
        covers[key] = cover
        parameters[key] = {
            name: float(cast(float, values[name])) + 0.0
            for name in template.free_parameters()
        }
        unidentified[key] = _masked_names(block.names, solved.unidentified[start:stop])
        zeroed[key] = _masked_names(block.names, solved.zeroed[start:stop])
        for offset, name in enumerate(block.names):
            positions[start + offset] = (key, name)
        start = stop

    premium = _premium(total, total_payout)
    over_budget = _over_budget(total, premium)
    if over_budget is not None:
        raise total.error(
            DesignError,
            f"the design failed its feasibility check: {over_budget}",
            status="infeasible_solution",
        )
    objective_value = _objective_value(total, total_payout)
    _check_objective(objective_value, solved, total)
    unique, by_column = _uniqueness(stacked, solved, total, solves, report=True)
    ranges: dict[str, dict[str, tuple[float, float]]] | None = None
    if by_column is not None:
        ranges = {key: {} for key in keys}
        for column, bounds in by_column.items():
            key, name = positions[column]
            ranges[key][name] = bounds
    p_zero, tail, flag = _tail_flags(objective, total_loss, probability)
    return _JointSolve(
        covers=covers,
        parameters=parameters,
        unidentified=unidentified,
        zeroed=zeroed,
        ranges=ranges,
        premium=premium,
        expected_loss=expected_loss,
        bound=bound,
        objective_value=objective_value,
        unique=unique,
        route=solved.route,
        ranging=solved.ranging,
        solves=solves,
        method=solved.solution.method,
        p_zero=p_zero,
        tail=tail,
        flag=flag,
    )


def _design_assets(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing,
    budget: Budget | None,
    limit: Limit | None,
    options: DesignOptions,
    scope: Joint,
) -> JointDesignResult:
    """`design_joint` under `Joint()`: one cover per asset."""
    _check_problem(templates, sample, objective)
    _solver.require()
    assets = sample.asset_ids
    named = [
        _for_asset(template, asset)
        for asset, template in zip(assets, templates, strict=True)
    ]
    solve = _solve_joint(
        assets,
        named,
        np.array(sample.index, dtype=np.float64),
        np.array(sample.loss, dtype=np.float64),
        np.array(sample.probability, dtype=np.float64),
        objective=objective,
        pricing=pricing,
        budget=budget,
        limit=limit,
        options=options,
    )
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
            "scope": scope.to_document(),
            "semantics_version": DESIGN_SEMANTICS_VERSION,
            "template_hashes": [template.definition_hash for template in templates],
        }
    )
    solves = solve.solves
    return JointDesignResult(
        covers=solve.covers,
        template_hashes={
            asset: template.definition_hash
            for asset, template in zip(assets, templates, strict=True)
        },
        covers_hash=covers_hash(solve.covers),
        design_id=design_id,
        scope=scope.to_document(),
        parameters=solve.parameters,
        objective=objective.to_document(),
        objective_value=solve.objective_value,
        pricing=pricing.to_document(),
        budget=budget_document,
        limit=limit_document,
        options=options,
        premium=solve.premium,
        expected_loss=solve.expected_loss,
        budget_bound=solve.bound,
        budget_use=solve.budget_use,
        route=solve.route,
        amounts=solve.route,
        solver=_solver.identity(),
        status="optimal",
        solver_message="" if solves.last is None else solves.last.message,
        iterations=solves.iterations,
        solves=solves.count,
        seconds=solves.seconds,
        solver_options=_solver_options(solve.route, solve.ranging),
        unidentified=solve.unidentified,
        fill=options.fill,
        zeroed=solve.zeroed,
        ranges=solve.ranges,
        unique=solve.unique,
        ties=0,
        profile=None,
        grid=None,
        p_zero_loss=solve.p_zero,
        tail_outcomes=solve.tail,
        degenerate_tail=solve.flag,
        design_sample=sample,
        solved_by=solve.method,
        fallbacks=solves.fallbacks,
    )


# -- the cluster scope ---------------------------------------------------------


def _name(value: Any, what: str) -> str:
    """A valid asset or cluster id; raise `SampleError` otherwise."""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise SampleError(
            f"{what} must be a non-empty string without surrounding whitespace, "
            f"got {value!r}"
        )
    return value


@dataclass(frozen=True, eq=False, init=False)
class Clustered:
    """One cover per cluster of an owner's assets, on the cluster's mean index.

    `assignment` maps every asset id of the sample to its cluster's id, a label
    of your choice; the clusters follow the order in which the assignment first
    names them, and each cluster's assets the order in which it names them.
    `values` maps every asset id to its value, in the loss unit, and weights
    each cluster's index. `joint` says how the covers are designed:

    - `joint=True`: one design problem over every cluster. The objective, the
      premium and the budget are on the owner's total, the sum over the
      clusters of their losses and of their payouts, as under `Joint()` with
      the clusters as the covers;
    - `joint=False`: one design problem per cluster, on its own loss and
      payout, each with its own objective, premium and budget:
      `Budget(ratio=k)` means a premium of at most `k * E[L_c]` for cluster c,
      the rule of a design of one asset, applied per cluster.

    Both settings have the same clusters and the same index. A cluster's cover
    pays on the value-weighted mean of its assets' indices, `ClusterIndex`:
    xi_{d,c} = sum_{i in c} w_i x_{d,i}, with w_i = v_i / sum_{k in c} v_k. A
    cluster's loss is the sum of its assets' losses in each draw. So a
    cluster's cover pays on a trigger that none of its assets has alone, and a
    one-asset cluster's index is its asset's.

    `to_document()` holds the assignment and the setting, not the values,
    which a comparison never writes; the weights are part of each cluster
    cover's terms. Two scopes are equal when their assignments (in order),
    values and settings are.

    Raises `TypeError` for arguments of the wrong type and `SampleError` for
    an empty assignment, an invalid asset or cluster id, values that miss an
    assigned asset or name another, a negative or non-finite value, or a
    cluster whose values are all zero, which gives its index no weights.
    `design_joint` raises `SampleError` for an asset of the sample left out of
    the assignment and for an assigned asset that the sample doesn't hold,
    naming any cluster left without an asset of the sample.
    """

    assignment: Mapping[str, str]
    values: Mapping[str, float]
    joint: bool

    def __init__(
        self,
        assignment: Mapping[str, str],
        *,
        values: Mapping[str, float],
        joint: bool = True,
    ) -> None:
        if isinstance(assignment, (str, bytes)) or not isinstance(assignment, Mapping):
            raise TypeError(
                "assignment must map each asset id to its cluster's id, got "
                f"{type(assignment).__name__}"
            )
        if not assignment:
            raise SampleError("Clustered needs an assignment of one asset or more")
        checked = {
            _name(asset, "an asset id"): _name(cluster, "a cluster id")
            for asset, cluster in assignment.items()
        }
        if isinstance(values, (str, bytes)) or not isinstance(values, Mapping):
            raise TypeError(
                "values must map each asset id to its value, got "
                f"{type(values).__name__}"
            )
        missing = [asset for asset in checked if asset not in values]
        unknown = sorted(str(asset) for asset in values if asset not in checked)
        if missing or unknown:
            raise SampleError(
                "values= must give the value of every asset of the assignment, and "
                f"of no other: it lacks {missing} and names {unknown}"
            )
        stated: dict[str, float] = {}
        for asset in checked:
            value = values[asset]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise SampleError(
                    f"the value of asset {asset!r} must be a finite number >= 0, "
                    f"got {value!r}"
                )
            stated[asset] = float(value) + 0.0
        if not isinstance(joint, bool):
            raise TypeError(f"joint must be a bool, got {type(joint).__name__}")
        object.__setattr__(self, "assignment", MappingProxyType(checked))
        object.__setattr__(self, "values", MappingProxyType(stated))
        object.__setattr__(self, "joint", joint)
        for cluster, assets in self.members.items():
            if not math.fsum(stated[asset] for asset in assets) > 0.0:
                raise SampleError(
                    f"the values of cluster {cluster!r}'s assets {list(assets)} are "
                    "all zero, so they give its index no weights"
                )

    @property
    def cluster_ids(self) -> tuple[str, ...]:
        """The clusters, in the order the assignment first names them."""
        return tuple(dict.fromkeys(self.assignment.values()))

    @property
    def members(self) -> Mapping[str, tuple[str, ...]]:
        """Each cluster's assets, in the order the assignment names them."""
        return MappingProxyType(
            {
                cluster: tuple(
                    asset for asset, own in self.assignment.items() if own == cluster
                )
                for cluster in self.cluster_ids
            }
        )

    @property
    def weights(self) -> Mapping[str, Mapping[str, float]]:
        """Each cluster's weights: each asset's value over the cluster's total."""
        weights: dict[str, Mapping[str, float]] = {}
        for cluster, assets in self.members.items():
            total = math.fsum(self.values[asset] for asset in assets)
            weights[cluster] = MappingProxyType(
                {asset: self.values[asset] / total + 0.0 for asset in assets}
            )
        return MappingProxyType(weights)

    def index(self, cluster: str, index: IndexDefinition) -> ClusterIndex:
        """The `ClusterIndex` of `cluster`, on the `index` its assets read."""
        return ClusterIndex(index, self.weights[cluster])

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "clustered", "assignment": {...}, "joint": ...}`.

        The assignment maps each asset id to its cluster's id. The values are
        not part of the document.
        """
        return {
            "assignment": dict(self.assignment),
            "joint": self.joint,
            "kind": "clustered",
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clustered):
            return NotImplemented
        return (
            tuple(self.assignment.items()) == tuple(other.assignment.items())
            and tuple(self.values.items()) == tuple(other.values.items())
            and self.joint == other.joint
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.assignment.items()),
                tuple(self.values.items()),
                self.joint,
            )
        )

    def __repr__(self) -> str:
        return f"Clustered({dict(self.assignment)!r}, joint={self.joint})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _clustered,
            (dict(self.assignment), dict(self.values), self.joint),
        )


def _clustered(
    assignment: dict[str, str], values: dict[str, float], joint: bool
) -> Clustered:
    return Clustered(assignment, values=values, joint=joint)


@dataclass(frozen=True, eq=False)
class ClusterDesign:
    """One cluster's own design problem, under `Clustered(..., joint=False)`.

    Each cluster is designed on its own loss and payout, with its own
    objective, premium and budget: `objective_value`, `premium` (the design
    premium of the cluster's cover), `expected_loss`, `budget_bound` and
    `budget_use` are the cluster's, and so are `unique`, `p_zero_loss`,
    `tail_outcomes` and `degenerate_tail`, as `DesignResult` has them for one
    asset. The last fields record its solve.
    """

    objective_value: float
    premium: float
    expected_loss: float
    budget_bound: float | None
    budget_use: float | None
    unique: bool | None
    p_zero_loss: float
    tail_outcomes: float | None
    degenerate_tail: str | None
    solver_message: str
    iterations: int
    solves: int
    seconds: float
    solved_by: _solver.Method | None
    fallbacks: int

    def to_record(self) -> dict[str, Any]:
        """Return the cluster's design record as plain JSON types."""
        return {
            "budget_bound": self.budget_bound,
            "budget_use": self.budget_use,
            "degenerate_tail": self.degenerate_tail,
            "expected_loss": self.expected_loss,
            "objective_value": self.objective_value,
            "p_zero_loss": self.p_zero_loss,
            "premium": self.premium,
            "solve": {
                "fallbacks": self.fallbacks,
                "iterations": self.iterations,
                "message": self.solver_message,
                "seconds": self.seconds,
                "solved_by": self.solved_by,
                "solves": self.solves,
            },
            "tail_outcomes": self.tail_outcomes,
            "unique": self.unique,
        }


@dataclass(frozen=True, eq=False)
class ClusteredDesignResult:
    """Covers designed for the clusters of an owner's assets, and the record.

    It holds the record of a `JointDesignResult`, keyed by cluster:

    - `covers` maps each cluster id to its resolved cover, in the order of the
      clusters: a cover on the cluster's `ClusterIndex`, written for the
      cluster (`asset_id` is the cluster id); `template_hashes` maps each
      cluster id to its template's `definition_hash`; `covers_hash` is
      `covers_hash(covers)`; `scope` is `Clustered.to_document()`;
    - `parameters`, `unidentified`, `zeroed` and `ranges` are keyed by cluster
      id, then by the payout's flat parameter names;
    - `premium`, `expected_loss`, `budget_bound` and `budget_use` are on the
      owner's total: under `joint=True` the design premium of one contract on
      the total, and under `joint=False` the sum of the clusters' premiums,
      as each cluster is its own contract with its own budget, against the sum
      of their budget bounds;
    - `objective_value`, `p_zero_loss`, `tail_outcomes` and `degenerate_tail`
      are the total's under `joint=True`, and `None` under `joint=False`,
      where each cluster has its own: `clusters` maps each cluster id to its
      `ClusterDesign` under `joint=False`, and is `None` under `joint=True`;
    - `unique` is the design's under `joint=True`, and under `joint=False`
      true when every cluster's is, false when any cluster's is, and `None`
      otherwise; `iterations`, `solves`, `seconds` and `fallbacks` add up the
      clusters' solves, and `solved_by` is theirs if they share it;
    - `design_id` is the `"sha256:"` hash of the template hashes in cluster
      order, the scope, each cluster's index, the objective, the pricing, the
      budget, the limit, the options and the design sample's hash.

    Under `joint=True`, unless `unique` is true, the per-cluster amounts are
    one member of an optimal set, as a joint design's per-asset amounts are.
    """

    covers: Mapping[str, Cover]
    template_hashes: Mapping[str, str]
    covers_hash: str
    design_id: str
    scope: Mapping[str, Any]
    parameters: Mapping[str, Mapping[str, float]]
    objective: Mapping[str, Any]
    objective_value: float | None
    pricing: Mapping[str, Any]
    budget: Mapping[str, Any] | None
    limit: Mapping[str, Any] | None
    options: DesignOptions
    premium: float
    expected_loss: float
    budget_bound: float | None
    budget_use: float | None
    route: Literal["lp", "qp"]
    amounts: Literal["lp", "qp"]
    solver: Mapping[str, str]
    status: str
    solver_message: str
    iterations: int
    solves: int
    seconds: float
    solver_options: Mapping[str, Any]
    unidentified: Mapping[str, tuple[str, ...]]
    fill: Fill
    zeroed: Mapping[str, tuple[str, ...]]
    ranges: Mapping[str, Mapping[str, tuple[float, float]]] | None
    unique: bool | None
    ties: int
    profile: None
    grid: None
    p_zero_loss: float | None
    tail_outcomes: float | None
    degenerate_tail: str | None
    clusters: Mapping[str, ClusterDesign] | None
    design_sample: OutcomeSample = field(repr=False)
    solved_by: _solver.Method | None = None
    fallbacks: int = 0

    @property
    def joint(self) -> bool:
        """Whether the clusters were designed in one problem on the total."""
        return bool(self.scope["joint"])

    @property
    def cluster_ids(self) -> tuple[str, ...]:
        """The clusters, in the assignment's order."""
        return tuple(self.covers)

    @property
    def asset_ids(self) -> tuple[str, ...]:
        """The assets of the design, in the design sample's order."""
        return self.design_sample.asset_ids

    @property
    def members(self) -> Mapping[str, tuple[str, ...]]:
        """Each cluster's assets, in the order of its index."""
        return MappingProxyType(
            {
                cluster: _cluster_of(cover).members
                for cluster, cover in self.covers.items()
            }
        )

    @property
    def sample_hash(self) -> str:
        """The design sample's `content_hash`."""
        return self.design_sample.content_hash

    def to_record(self) -> dict[str, Any]:
        """Return the design record as plain JSON types."""
        ranges = (
            None
            if self.ranges is None
            else {
                cluster: {name: [low, high] for name, (low, high) in amounts.items()}
                for cluster, amounts in self.ranges.items()
            }
        )
        return {
            "design_id": self.design_id,
            "scope": copy.deepcopy(dict(self.scope)),
            "asset_ids": list(self.asset_ids),
            "cluster_ids": list(self.cluster_ids),
            "members": {
                cluster: list(assets) for cluster, assets in self.members.items()
            },
            "template_hashes": dict(self.template_hashes),
            "cover_hashes": {
                cluster: cover.definition_hash for cluster, cover in self.covers.items()
            },
            "covers_hash": self.covers_hash,
            "sample_hash": self.sample_hash,
            "objective": copy.deepcopy(dict(self.objective)),
            "objective_value": self.objective_value,
            "pricing": dict(self.pricing),
            "budget": None if self.budget is None else dict(self.budget),
            "limit": None if self.limit is None else dict(self.limit),
            "options": self.options.to_document(),
            "parameters": {
                cluster: dict(values) for cluster, values in self.parameters.items()
            },
            "premium": self.premium,
            "expected_loss": self.expected_loss,
            "budget_bound": self.budget_bound,
            "budget_use": self.budget_use,
            "route": self.route,
            "amounts": self.amounts,
            "solver": dict(self.solver),
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
            "unidentified": {
                cluster: list(names) for cluster, names in self.unidentified.items()
            },
            "fill": self.fill,
            "zeroed": {cluster: list(names) for cluster, names in self.zeroed.items()},
            "ranges": ranges,
            "unique": self.unique,
            "ties": self.ties,
            "grid": self.grid,
            "p_zero_loss": self.p_zero_loss,
            "tail_outcomes": self.tail_outcomes,
            "degenerate_tail": self.degenerate_tail,
            "clusters": (
                None
                if self.clusters is None
                else {
                    cluster: design.to_record()
                    for cluster, design in self.clusters.items()
                }
            ),
        }


def _cluster_of(cover: Cover) -> ClusterIndex:
    index = cover.cluster_index
    assert index is not None
    return index


def _cluster_index_values(index: ClusterIndex, sample: OutcomeSample) -> FloatArray:
    """A cluster's index on `sample`: its members' indices, weighted, `(n,)`."""
    columns = [sample.asset_ids.index(asset) for asset in index.members]
    weights = np.array(list(index.weights.values()), dtype=np.float64)
    index_values = np.asarray(sample.index, dtype=np.float64)
    values: FloatArray = index_values[:, columns] @ weights
    return values


def _cluster_loss(index: ClusterIndex, sample: OutcomeSample) -> FloatArray:
    """A cluster's loss on `sample`: the sum of its members' losses, `(n,)`."""
    columns = [sample.asset_ids.index(asset) for asset in index.members]
    loss: FloatArray = np.ascontiguousarray(
        np.asarray(sample.loss, dtype=np.float64)[:, columns]
    ).sum(axis=1)
    return loss


def _check_assignment(scope: Clustered, sample: OutcomeSample) -> None:
    """Raise `SampleError` unless the assignment holds exactly the sample's assets."""
    assets = sample.asset_ids
    left_out = [asset for asset in assets if asset not in scope.assignment]
    unknown = [asset for asset in scope.assignment if asset not in assets]
    if not left_out and not unknown:
        return
    empty = [
        cluster
        for cluster, members in scope.members.items()
        if all(asset in unknown for asset in members)
    ]
    message = (
        "Clustered assigns every asset of the sample to one cluster, but the "
        f"assignment leaves out {left_out} and names {unknown}, which the sample "
        f"{list(assets)} doesn't hold"
    )
    if empty:
        message += f"; so the clusters {empty} hold no asset of the sample"
    raise SampleError(message)


def _check_cluster_templates(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    scope: Clustered,
) -> None:
    """Check that the templates suit the clusters and that the problem is linear."""
    assets = sample.asset_ids
    if len(assets) == 1:
        raise DesignError(
            f"Clustered designs the clusters of several assets, but this sample "
            f"holds one, {assets[0]!r}: use design",
            asset_id=assets[0],
        )
    if isinstance(objective, UserObjective):
        raise DesignError(
            "design_joint solves a linear or quadratic programme, so it takes "
            "MSE(), ExpectedUncovered() or WeightedCVaR(alpha, beta); a "
            "UserObjective is evaluated on a grid, and a joint design has none"
        )
    if sample.index_components is not None:
        raise SampleError(
            "Clustered reads one index per asset, but this sample holds the index "
            f"components {list(sample.index_components)}"
        )
    _check_assignment(scope, sample)
    clusters = scope.cluster_ids
    if len(templates) != len(clusters):
        raise SampleError(
            f"design_joint takes one template per cluster under Clustered, but it "
            f"got {len(templates)} for the {len(clusters)} clusters {list(clusters)}"
        )
    for cluster, template in zip(clusters, templates, strict=True):
        if template.cluster_index is not None:
            raise SampleError(
                f"the template for cluster {cluster!r} pays on a ClusterIndex: write "
                "it on the IndexDefinition its assets read, and the design builds "
                "the cluster's index from the assignment and the values"
            )
        if isinstance(template.index_definition, CompositeIndex):
            raise SampleError(
                f"the template for cluster {cluster!r} pays on several components: "
                "a cluster's cover pays on the weighted mean of one index"
            )
        if template.asset_id is not None and template.asset_id != cluster:
            raise SampleError(
                f"the template for cluster {cluster!r} is written for "
                f"{template.asset_id!r}: the templates follow the clusters, in "
                f"the order {list(clusters)}"
            )
        for asset in scope.members[cluster]:
            _check_units(template, sample, asset)
        payout = template.payout
        if not isinstance(payout, LinearInAmounts):
            raise DesignError(
                f"cluster {cluster!r}: a joint design needs a payout linear in its "
                f"amounts (LinearInAmounts); {payout.kind!r} is not",
                asset_id=cluster,
            )
        amounts = set(payout.amount_names())
        thresholds = [
            name for name in template.free_parameters() if name not in amounts
        ]
        if thresholds:
            raise DesignError(
                f"cluster {cluster!r}: a joint design takes fixed thresholds only, "
                f"but {thresholds} are free; to compare threshold choices, run "
                "design_joint once per choice",
                asset_id=cluster,
            )
    if not any(template.free_parameters() for template in templates):
        raise ValueError(
            "no template has free parameters: there is nothing to design; "
            "evaluate the covers instead"
        )


def _for_cluster(template: Cover, cluster: str, index: ClusterIndex) -> Cover:
    """The template on its cluster's index, written for the cluster."""
    return Cover(
        index,
        template.payout,
        payout_unit=template.payout_unit,
        asset_id=cluster,
        label=template.label,
    )


def _all_unique(values: Sequence[bool | None]) -> bool | None:
    """True if every value is, false if any is, else `None`."""
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def _design_clustered(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing,
    budget: Budget | None,
    limit: Limit | None,
    options: DesignOptions,
    scope: Clustered,
) -> ClusteredDesignResult:
    """`design_joint` under `Clustered`: one cover per cluster."""
    _check_cluster_templates(templates, sample, objective, scope)
    _solver.require()
    clusters = scope.cluster_ids
    named = [
        _for_cluster(template, cluster, scope.index(cluster, template.index))
        for cluster, template in zip(clusters, templates, strict=True)
    ]
    indices = [_cluster_of(cover) for cover in named]
    index = np.column_stack([_cluster_index_values(each, sample) for each in indices])
    loss = np.column_stack([_cluster_loss(each, sample) for each in indices])
    probability = np.array(sample.probability, dtype=np.float64)
    terms: dict[str, Any] = {
        "objective": objective,
        "pricing": pricing,
        "budget": budget,
        "limit": limit,
        "options": options,
    }
    if scope.joint:
        solves = [_solve_joint(clusters, named, index, loss, probability, **terms)]
    else:
        solves = [
            _solve_joint(
                [cluster],
                [cover],
                index[:, [column]],
                loss[:, [column]],
                probability,
                subject=cluster,
                **terms,
            )
            for column, (cluster, cover) in enumerate(zip(clusters, named, strict=True))
        ]
    covers = {key: cover for solve in solves for key, cover in solve.covers.items()}
    ranged = [solve.ranges for solve in solves]
    ranges = (
        None
        if all(each is None for each in ranged)
        else {key: value for each in ranged for key, value in (each or {}).items()}
    )
    budget_document = None if budget is None else budget.to_document()
    limit_document = None if limit is None else limit.to_document()
    design_id = content_hash(
        {
            "budget": budget_document,
            "clusters": [
                {"cluster_id": cluster, "index": each.to_document()}
                for cluster, each in zip(clusters, indices, strict=True)
            ],
            "limit": limit_document,
            "objective": objective.to_document(),
            "options": options.to_document(),
            "pricing": pricing.to_document(),
            "sample_hash": sample.content_hash,
            "scope": scope.to_document(),
            "semantics_version": DESIGN_SEMANTICS_VERSION,
            "template_hashes": [template.definition_hash for template in templates],
        }
    )
    first = solves[0]
    methods = {solve.method for solve in solves}
    flags: tuple[float | None, float | None, str | None]
    if scope.joint:
        premium, expected_loss, bound = first.premium, first.expected_loss, first.bound
        objective_value: float | None = first.objective_value
        unique, flags = first.unique, (first.p_zero, first.tail, first.flag)
        per_cluster: dict[str, ClusterDesign] | None = None
    else:
        premium = math.fsum(solve.premium for solve in solves) + 0.0
        total_loss = np.asarray(sample.loss, dtype=np.float64).sum(axis=1)
        expected_loss = math.fsum((probability * total_loss).tolist()) + 0.0
        bounds = [solve.bound for solve in solves]
        bound = None if None in bounds else math.fsum(cast(list[float], bounds)) + 0.0
        objective_value = None
        unique = _all_unique([solve.unique for solve in solves])
        flags = (None, None, None)
        per_cluster = {
            cluster: ClusterDesign(
                objective_value=solve.objective_value,
                premium=solve.premium,
                expected_loss=solve.expected_loss,
                budget_bound=solve.bound,
                budget_use=solve.budget_use,
                unique=solve.unique,
                p_zero_loss=solve.p_zero,
                tail_outcomes=solve.tail,
                degenerate_tail=solve.flag,
                solver_message=(
                    "" if solve.solves.last is None else solve.solves.last.message
                ),
                iterations=solve.solves.iterations,
                solves=solve.solves.count,
                seconds=solve.solves.seconds,
                solved_by=solve.method,
                fallbacks=solve.solves.fallbacks,
            )
            for cluster, solve in zip(clusters, solves, strict=True)
        }
    last = solves[-1].solves.last
    return ClusteredDesignResult(
        covers=covers,
        template_hashes={
            cluster: template.definition_hash
            for cluster, template in zip(clusters, templates, strict=True)
        },
        covers_hash=covers_hash(covers),
        design_id=design_id,
        scope=scope.to_document(),
        parameters={
            key: value for solve in solves for key, value in solve.parameters.items()
        },
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
        route=first.route,
        amounts=first.route,
        solver=_solver.identity(),
        status="optimal",
        solver_message="" if last is None else last.message,
        iterations=sum(solve.solves.iterations for solve in solves),
        solves=sum(solve.solves.count for solve in solves),
        seconds=math.fsum(solve.solves.seconds for solve in solves) + 0.0,
        solver_options=_solver_options(
            first.route, any(solve.ranging for solve in solves)
        ),
        unidentified={
            key: value for solve in solves for key, value in solve.unidentified.items()
        },
        fill=options.fill,
        zeroed={key: value for solve in solves for key, value in solve.zeroed.items()},
        ranges=ranges,
        unique=unique,
        ties=0,
        profile=None,
        grid=None,
        p_zero_loss=flags[0],
        tail_outcomes=flags[1],
        degenerate_tail=flags[2],
        clusters=per_cluster,
        design_sample=sample,
        solved_by=methods.pop() if len(methods) == 1 else None,
        fallbacks=sum(solve.solves.fallbacks for solve in solves),
    )


@overload
def design_joint(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing = ...,
    budget: Budget | None = ...,
    limit: Limit | None = ...,
    options: DesignOptions = ...,
    scope: Joint = ...,
) -> JointDesignResult: ...


@overload
def design_joint(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing = ...,
    budget: Budget | None = ...,
    limit: Limit | None = ...,
    options: DesignOptions = ...,
    scope: Clustered,
) -> ClusteredDesignResult: ...


def design_joint(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing = PurePremium(),
    budget: Budget | None = None,
    limit: Limit | None = None,
    options: DesignOptions = DesignOptions(),
    scope: Joint | Clustered = Joint(),
) -> JointDesignResult | ClusteredDesignResult:
    """Design one cover per asset, or per cluster, for an objective on the total.

    Under `scope=Joint()`, the default, `templates` holds one `Cover` per
    asset, in `sample.asset_ids` order, each written for its position's asset
    or for none (the resolved cover then names that asset). The families may
    differ between assets; each must be linear in its amounts, and only
    amounts may be free. `objective`, `pricing` and `budget` are taken on the
    total loss and the total payout of each draw. `limit` is each cover's own
    limit: a limit on the total isn't offered. `options` sets the tolerances,
    the fill rule per asset and whether the LP route ranges the optimal set.
    It returns a `JointDesignResult`.

    Under `scope=Clustered(assignment, values=..., joint=...)`, `templates`
    holds one `Cover` per cluster, in the order of `scope.cluster_ids`, each
    on the `IndexDefinition` its assets read and written for its cluster or
    for none. Each resolved cover pays on its cluster's `ClusterIndex` and is
    written for the cluster. Under `joint=True` the objective, the premium
    and the budget are on the owner's total, the sum over the clusters; under
    `joint=False` each cluster has its own design problem, on its own loss
    and payout, with its own budget. The rules of `Joint()` hold per cover:
    fixed thresholds, linear payouts, `limit=` per cover. It returns a
    `ClusteredDesignResult`. Errors that name one problem's asset name the
    cluster in its place.

    The module docstring describes the problem, its solution and its report
    of non-unique optima. The expected uncovered loss and the weighted CVaR
    need a budget or bounded amounts; MSE needs neither.

    Raises `SolverUnavailableError` without the `parametric` extra;
    `InfeasibleDesignError` when no payout meets the constraints;
    `UnboundedDesignError` for an unbounded objective (never for `MSE`);
    `SolverLimitError` when the solve stops at its iteration or time limit;
    `DesignError` for a sample of one asset, a free threshold, a
    `UserObjective`, a payout without `LinearInAmounts`, or any failed solve
    or check; `SampleError` when the templates' number, assets or units don't
    suit the sample, or a template's index `hazard_name` differs from the
    hazard the sample records for its asset, and under `Clustered` for an
    assignment that leaves out an asset of the sample or names one it doesn't
    hold, and for a template on a `ClusterIndex` or a `CompositeIndex`;
    `ValueError` when no template has free parameters; and `TypeError` for
    arguments of the wrong type.
    """
    templates = _check_templates(templates, scope)
    _check_terms(sample, objective, pricing, budget, limit, options)
    terms: dict[str, Any] = {
        "pricing": pricing,
        "budget": budget,
        "limit": limit,
        "options": options,
    }
    if isinstance(scope, Clustered):
        return _design_clustered(templates, sample, objective, scope=scope, **terms)
    return _design_assets(templates, sample, objective, scope=scope, **terms)
