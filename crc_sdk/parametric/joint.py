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
loss of a draw.

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

Everything here works in memory and does no I/O.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

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
    _Solves,
    _tail_flags,
    _uniqueness,
    _verify,
)
from crc_sdk.parametric.errors import DesignError, SampleError
from crc_sdk.parametric.objectives import Objective, UserObjective
from crc_sdk.parametric.parameters import _DefinitionModel
from crc_sdk.parametric.payouts import LinearConstraints, LinearInAmounts
from crc_sdk.parametric.pricing import Pricing, PurePremium
from crc_sdk.parametric.sample import OutcomeSample

__all__ = [
    "Individual",
    "Joint",
    "JointDesignResult",
    "covers_hash",
    "design_joint",
]

FloatArray = NDArray[np.float64]


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
    """Return the `"sha256:"` hash of a set of covers, one per asset.

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
      is 0 and `profile` and `grid` are `None`.

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
    if not isinstance(scope, Joint):
        raise TypeError(f"scope must be Joint(), got {type(scope).__name__}")
    if isinstance(templates, (str, bytes, Cover, Mapping)) or not isinstance(
        templates, Sequence
    ):
        raise TypeError(
            "templates must be a sequence of Covers, one per asset, in the order "
            f"of sample.asset_ids, got {type(templates).__name__}"
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
        _check_units(template, sample)
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


def design_joint(
    templates: Sequence[Cover],
    sample: OutcomeSample,
    objective: Objective,
    *,
    pricing: Pricing = PurePremium(),
    budget: Budget | None = None,
    limit: Limit | None = None,
    options: DesignOptions = DesignOptions(),
    scope: Joint = Joint(),
) -> JointDesignResult:
    """Design one cover per asset for one objective on the owner's total.

    `templates` holds one `Cover` per asset, in `sample.asset_ids` order,
    each written for its position's asset or for none (the resolved cover
    then names that asset). The families may differ between assets; each must
    be linear in its amounts, and only amounts may be free. `objective`,
    `pricing` and `budget` are taken on the total loss and the total payout of
    each draw. `limit` is each cover's own limit: a limit on the total isn't
    offered. `options` sets the tolerances, the fill rule per asset and
    whether the LP route ranges the optimal set. `scope` is `Joint()`.

    The module docstring describes the problem, its solution and its report
    of non-unique optima. The expected uncovered loss and the weighted CVaR
    need a budget or bounded amounts; MSE needs neither.

    Returns a `JointDesignResult`. Raises `SolverUnavailableError` without the
    `parametric` extra; `InfeasibleDesignError` when no payout meets the
    constraints; `UnboundedDesignError` for an unbounded objective;
    `DesignError` for a sample of one asset, a free threshold, a
    `UserObjective`, a payout without `LinearInAmounts`, or any failed solve
    or check; `SampleError` when the templates' number, assets or units don't
    suit the sample; `ValueError` when no template has free parameters; and
    `TypeError` for arguments of the wrong type.
    """
    templates = _check_templates(templates, scope)
    _check_terms(sample, objective, pricing, budget, limit, options)
    _check_problem(templates, sample, objective)
    _solver.require()
    assets = sample.asset_ids
    named = [
        _for_asset(template, asset)
        for asset, template in zip(assets, templates, strict=True)
    ]
    index = np.array(sample.index, dtype=np.float64)
    loss = np.array(sample.loss, dtype=np.float64)
    probability = np.array(sample.probability, dtype=np.float64)
    total_loss: FloatArray = loss.sum(axis=1)
    largest = float(total_loss.max())
    scale = largest if largest > SCALE_FLOOR else 1.0
    expected_loss = math.fsum((probability * total_loss).tolist()) + 0.0
    bound = None if budget is None else budget.bound(expected_loss)
    limit_amount = None if limit is None else limit.amount

    # Each cover is read and checked on its own asset, at the scale of the
    # total; the budget and the objective are the total's.
    parts = [
        _Context(
            asset_id=asset,
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
        for column, asset in enumerate(assets)
    ]
    total = _Context(
        asset_id=None,
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
    stacked = _stacked(blocks, assets)
    solves = _Solves()
    solved = _solve_block(stacked, total, solves, report=True)

    covers: dict[str, Cover] = {}
    parameters: dict[str, dict[str, float]] = {}
    unidentified: dict[str, tuple[str, ...]] = {}
    zeroed: dict[str, tuple[str, ...]] = {}
    positions: dict[int, tuple[str, str]] = {}
    total_payout: FloatArray = np.zeros_like(total_loss)
    start = 0
    for asset, template, block, part in zip(assets, named, blocks, parts, strict=True):
        stop = start + len(block.names)
        amounts = solved.amounts[start:stop]
        cover, _ = _resolve_amounts(template, block.names, amounts, part)
        payout, _, _ = _verify(part, cover, amounts, block.basis, block.constraints)
        total_payout = total_payout + payout
        values = cover.payout.parameters()
        covers[asset] = cover
        parameters[asset] = {
            name: float(cast(float, values[name])) + 0.0
            for name in template.free_parameters()
        }
        unidentified[asset] = _masked_names(
            block.names, solved.unidentified[start:stop]
        )
        zeroed[asset] = _masked_names(block.names, solved.zeroed[start:stop])
        for offset, name in enumerate(block.names):
            positions[start + offset] = (asset, name)
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
        ranges = {asset: {} for asset in assets}
        for column, bounds in by_column.items():
            asset, name = positions[column]
            ranges[asset][name] = bounds

    p_zero, tail, flag = _tail_flags(objective, total_loss, probability)
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
    return JointDesignResult(
        covers=covers,
        template_hashes={
            asset: template.definition_hash
            for asset, template in zip(assets, templates, strict=True)
        },
        covers_hash=covers_hash(covers),
        design_id=design_id,
        scope=scope.to_document(),
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
        route=solved.route,
        amounts=solved.route,
        solver=_solver.identity(),
        status="optimal",
        solver_message="" if solves.last is None else solves.last.message,
        iterations=solves.iterations,
        solves=solves.count,
        seconds=solves.seconds,
        solver_options={
            "solve": dict(
                _solver.QP_OPTIONS if solved.route == "qp" else _solver.LP_OPTIONS
            ),
            "ranging": dict(_solver.RANGING_OPTIONS) if solved.ranging else None,
        },
        unidentified=unidentified,
        fill=options.fill,
        zeroed=zeroed,
        ranges=ranges,
        unique=unique,
        ties=0,
        profile=None,
        grid=None,
        p_zero_loss=p_zero,
        tail_outcomes=tail,
        degenerate_tail=flag,
        design_sample=sample,
    )
