"""A budget frontier: one design per budget, in one table.

`design_frontier(template, sample, objective, *, budgets, evaluate_on, split, ...)`
designs one template for one objective on `sample` once per budget, reads every
design on the block `evaluate_on`, and returns one table with one row per budget,
in the order given.

**One row per budget.** Row b is
`design(template, sample, objective, pricing=pricing, budget=b, limit=limit,
options=design_options)`, read on `evaluate_on` as `compare` reads a design. It has
`compare`'s columns and four more, which sit beside the identity columns:
`budget_ratio` and `budget_amount` (the budget as stated: exactly one is non-null),
`budget_use` (the design premium's share of the bound, null when the bound is 0)
and `objective_value` (the design's objective on the design sample, in payout
units, squared for MSE). `design_budget`, `compare`'s own column, is the bound in
payout units.

**Each row is solved on its own**, so it doesn't depend on the other budgets or on
their order. Amounts may jump between neighbouring budgets where a row isn't
unique. Each row says so in its own `unique`, `ties`, `degenerate_tail` and
`unidentified_share`. With `DesignOptions(ranges=True)` every row is ranged, at four
more LPs per amount; without it `unique` is null on the LP route.

**Nothing is ranked.** The table has no rank, no best flag, no marginal column and
no ordering by a metric. On the design sample the objective doesn't rise with the
bound, because the feasible sets are nested. On another block nothing guarantees
that. The difference between two rows describes that pair of budgets, on the block
stated.

One template and one objective per call: several go through several calls and
`EvaluationMetadata.merge`. Two calls that make the same designs, such as one
frontier read on two blocks, hold two records of each design that differ in their
solve time, and `merge` refuses that, so they stay in two files. The function works
in memory and does no I/O. Its LP and QP routes need the optional extra
`crc-sdk[parametric]`, as `design` does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import DesignError, EvaluationError
from crc_sdk.parametric.evaluate import (
    Split,
    _fold,
    _options,
    _sample,
    _split,
    compare,
)
from crc_sdk.parametric.metadata import (
    _FRONTIER_COLUMNS,
    EVALUATION_METADATA_KEY,
    EvaluationMetadata,
    _identity_columns,
    _replaced,
)
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import Objective
from crc_sdk.parametric.pricing import Pricing, PurePremium
from crc_sdk.parametric.sample import OutcomeSample

__all__ = ["design_frontier"]


def _stated(budget: Budget) -> str:
    """The budget as a caller wrote it, for an error message."""
    if budget.ratio is not None:
        return f"ratio={budget.ratio!r}"
    return f"amount={budget.amount!r}"


def _budgets(budgets: Any) -> list[Budget]:
    """The budgets of a call; raise if they are none, repeated or not budgets."""
    if isinstance(budgets, (str, bytes, Mapping)) or not isinstance(budgets, Sequence):
        raise TypeError(
            f"budgets must be a sequence of Budget, got {type(budgets).__name__}"
        )
    for budget in budgets:
        if not isinstance(budget, Budget):
            raise TypeError(
                f"budgets must hold Budget objects, got {type(budget).__name__}"
            )
    if not budgets:
        raise EvaluationError("design_frontier needs at least one budget")
    seen: set[tuple[float | None, float | None]] = set()
    for budget in budgets:
        key = (budget.ratio, budget.amount)
        if key in seen:
            raise EvaluationError(
                f"budgets repeat {_stated(budget)}, so two rows would repeat a key"
            )
        seen.add(key)
    return list(budgets)


def _designed(
    template: Cover,
    sample: OutcomeSample,
    objective: Objective,
    budget: Budget,
    *,
    pricing: Pricing,
    limit: Limit | None,
    options: DesignOptions,
) -> DesignResult:
    """The design at `budget`; a failed design raises its own error, naming it."""
    try:
        return design(
            template,
            sample,
            objective,
            pricing=pricing,
            budget=budget,
            limit=limit,
            options=options,
        )
    except DesignError as error:
        raise type(error)(
            f"at budget {_stated(budget)}: {error}",
            asset_id=error.asset_id,
            status=error.status,
            solver_message=error.solver_message,
        ) from error


def _with_budget_columns(
    table: Any, budgets: Sequence[Budget], results: Sequence[DesignResult]
) -> Any:
    """`table` with the frontier's four columns, before its first metric column."""
    values: Mapping[str, list[float | None]] = {
        "budget_ratio": [budget.ratio for budget in budgets],
        "budget_amount": [budget.amount for budget in budgets],
        "budget_use": [result.budget_use for result in results],
        "objective_value": [result.objective_value for result in results],
    }
    position = len(_identity_columns(table.column_names))
    for offset, name in enumerate(_FRONTIER_COLUMNS):
        table = table.add_column(
            position + offset,
            pa.field(name, pa.float64()),
            pa.array(values[name], pa.float64()),
        )
    return table


def design_frontier(
    template: Cover,
    sample: OutcomeSample,
    objective: Objective,
    *,
    budgets: Sequence[Budget],
    evaluate_on: OutcomeSample,
    split: Split,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    limit: Limit | None = None,
    design_options: DesignOptions = DesignOptions(),
    metric_options: MetricOptions = MetricOptions(),
) -> Any:
    """Design one template once per budget; return one row per budget, as given.

    `sample` is the design sample and `budgets` a sequence of `Budget`.
    `evaluate_on` is the block the rows are read on, and `split` says what it is
    (`"in_sample"` for `sample` itself, `"out_of_sample"` for a block held out from
    it), as in `compare`: it has no default, so a table states which it holds.
    `fold` is the fold of a k-fold split. `pricing`, `limit` and `design_options`
    are `design`'s, and `metric_options` are `compare`'s `options`.

    Row b is `design(template, sample, objective, pricing=pricing, budget=b,
    limit=limit, options=design_options)` read on `evaluate_on` by `compare`, so its
    premium is the design premium. Each row is solved on its own and doesn't depend
    on the other budgets or on their order. Amounts may jump between neighbouring
    budgets where a row isn't unique, and each row's `unique`, `ties`,
    `degenerate_tail` and `unidentified_share` say so. With
    `DesignOptions(ranges=True)` every row is ranged, and without it `unique` is
    null on the LP route.

    The table has `compare`'s columns, in `compare`'s order, and, between the
    identity columns and the metric columns, `budget_ratio` and `budget_amount`
    (float64, the budget as stated: exactly one is non-null), `budget_use` (float64,
    the design premium's share of the bound; null when the bound is 0) and
    `objective_value` (float64, the design's objective on the design sample, in
    payout units, squared for MSE). `to_long` keeps them as identity columns.
    `design_budget` is the bound in payout units. Its metadata holds every design's
    record and, in `setup.frontier`, one entry with the template's hash, the
    objective, the budgets, their designs, the design sample and the block read.

    There is no rank, no best flag, no marginal column and no ordering by a metric.
    On the design sample the objective doesn't rise with the bound, because the
    feasible sets are nested; on another block nothing guarantees that.

    One template and one objective per call. To put several in one file, concatenate
    the tables with `pyarrow.concat_tables` and pass `EvaluationMetadata.merge` of
    their metadata to `write_comparison`. Two calls that make the same designs, such
    as one frontier read on two blocks, hold two records of each design that differ
    in their solve time, and `merge` raises `EvaluationError` for that.

    Needs the optional extra `crc-sdk[parametric]` on the LP and QP routes, as
    `design` does. Raises `EvaluationError` for an empty `budgets`, a budget stated
    twice and invalid `split` or `fold`; `InfeasibleDesignError`, `UnboundedDesignError`
    or `DesignError`, naming the budget, for one that can't be designed, so no row
    is dropped quietly; `SampleError` for a sample of several assets, as `design`
    does; `LeakageError` and the other errors of `design` and `compare`; and
    `TypeError` for arguments of the wrong type.
    """
    checked = _budgets(budgets)
    evaluated = _sample(evaluate_on, "evaluate_on")
    read_as = _split(split)
    read_fold = _fold(fold)
    options = _options(metric_options)
    results = [
        _designed(
            template,
            sample,
            objective,
            budget,
            pricing=pricing,
            limit=limit,
            options=design_options,
        )
        for budget in checked
    ]
    table = compare(
        results,
        evaluated,
        split=read_as,
        fold=read_fold,
        pricing=pricing,
        options=options,
    )
    metadata = EvaluationMetadata.from_table(table)
    described = _replaced(
        metadata,
        setup=_replaced(
            metadata.setup,
            frontier=(
                {
                    "budgets": [budget.to_document() for budget in checked],
                    "design_ids": [result.design_id for result in results],
                    "design_sample": results[0].sample_hash,
                    "fold": read_fold,
                    "objective": dict(results[0].objective),
                    "sample_hash": evaluated.content_hash,
                    "split": read_as,
                    "template_hash": results[0].template_hash,
                },
            ),
        ),
    )
    framed = _with_budget_columns(table, checked, results)
    return framed.replace_schema_metadata(
        {
            **(framed.schema.metadata or {}),
            EVALUATION_METADATA_KEY.encode("utf-8"): described.to_json().encode(
                "utf-8"
            ),
        }
    )
