"""`design_frontier`: one design per budget, in one table.

Every number is synthetic and chosen so that it can be checked by hand or against
a stated rule. The tests check what the module computes and records; nothing here
is a finding about a cover, a budget or a design.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    Budget,
    CompositeIndex,
    CostOfCapitalLoad,
    Cover,
    DesignError,
    DesignOptions,
    EvaluationError,
    EvaluationMetadata,
    ExpectedPayoutMultiple,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    InfeasibleDesignError,
    LeakageError,
    Limit,
    LinearPayout,
    MetricOptions,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    SolverUnavailableError,
    StepPayout,
    SumOf,
    UserObjective,
    WeightedCVaR,
    compare,
    design,
    design_frontier,
    evaluate,
    holdout,
    setup_text,
    to_long,
    write_comparison,
)
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    EvaluationSetup,
    _identity_columns,
)
from crc_sdk.parametric.objectives import Objective

FRONTIER = ["budget_ratio", "budget_amount", "budget_use", "objective_value"]
INDEX = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
FEW = MetricOptions(betas=(0.9,), return_periods=(5.0, 50.0))
RATIOS = (0.25, 0.5, 1.0, 1.5)
OBJECTIVES: tuple[Objective, ...] = (
    ExpectedUncovered(),
    MSE(),
    WeightedCVaR(alpha=0.9, beta=0.9),
)

UNTYPED_FRONTIER: Any = design_frontier


def _provenance(**fields: Any) -> SampleProvenance:
    return SampleProvenance(
        source_kind="table",
        sources=({"role": "table", "name": "test", "licence": "CC0-1.0"},),
        semantics_source="caller",
        licences=({"role": "table", "source": "test", "licence": "CC0-1.0"},),
        **fields,
    )


def _small(
    index: Sequence[float],
    loss: Sequence[float],
    *,
    start: int = 0,
    asset: str = "A",
) -> OutcomeSample:
    n = len(index)
    return OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(start, start + n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=np.asarray(index, dtype=np.float64),
        loss=np.asarray(loss, dtype=np.float64),
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(),
    )


def _years(n: int = 40) -> OutcomeSample:
    # A wavy index, so that a block of neighbouring years holds high and low ones.
    years = np.arange(n, dtype=np.float64)
    index = 60.0 + 40.0 * np.sin(years * 1.7) + years
    return _small(
        index.tolist(), np.where(index > 60.0, (index - 60.0) * 10.0, 0.0).tolist()
    )


# Four years: no loss below the first breakpoint; band 0 holds the years with
# losses 10 and 30, band 1 the year with loss 40.
SAMPLE = _small([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 30.0, 40.0])


def _step(
    levels: Sequence[float | Free], breakpoints: Sequence[float | Free] = (1.0, 2.5)
) -> Cover:
    return Cover(
        INDEX, StepPayout(breakpoints, levels), payout_unit="EUR", asset_id="A"
    )


def _template() -> Cover:
    return _step([Free() for _ in range(4)], (60.0, 80.0, 100.0, 120.0))


def _ratios(values: Sequence[float] = RATIOS) -> list[Budget]:
    return [Budget(ratio=value) for value in values]


def _metadata(table: Any) -> EvaluationMetadata:
    return EvaluationMetadata.from_table(table)


def _frontier(
    objective: Objective = OBJECTIVES[0],
    *,
    budgets: Sequence[Budget] | None = None,
    **options: Any,
) -> tuple[Any, OutcomeSample, OutcomeSample]:
    train, test = holdout(_years())
    table = design_frontier(
        _template(),
        train,
        objective,
        budgets=_ratios() if budgets is None else budgets,
        evaluate_on=test,
        split="out_of_sample",
        metric_options=FEW,
        **options,
    )
    return table, train, test


# -- each row is a design at that budget --------------------------------------


@pytest.mark.parametrize(
    "objective", OBJECTIVES, ids=lambda item: item.to_document()["kind"]
)
def test_each_row_equals_design_and_evaluate_by_hand_out_of_sample(
    objective: Objective,
) -> None:
    table, train, test = _frontier(objective)
    assert table.num_rows == len(RATIOS)
    for position, budget in enumerate(_ratios()):
        result = design(_template(), train, objective, budget=budget)
        by_hand = evaluate(
            result, test, split="out_of_sample", options=FEW
        ).to_pylist()[0]
        row = table.slice(position, 1).to_pylist()[0]
        assert {key: row[key] for key in by_hand} == by_hand
        assert row["objective_value"] == result.objective_value


@pytest.mark.parametrize(
    "objective", OBJECTIVES, ids=lambda item: item.to_document()["kind"]
)
def test_each_row_equals_design_and_evaluate_by_hand_in_sample(
    objective: Objective,
) -> None:
    train, _ = holdout(_years())
    table = design_frontier(
        _template(),
        train,
        objective,
        budgets=_ratios(),
        evaluate_on=train,
        split="in_sample",
        metric_options=FEW,
    )
    for position, budget in enumerate(_ratios()):
        result = design(_template(), train, objective, budget=budget)
        by_hand = evaluate(result, train, split="in_sample", options=FEW)
        assert (
            table.slice(position, 1).drop_columns(FRONTIER).equals(by_hand.slice(0, 1))
        )


def test_the_table_is_compare_on_the_designs_with_four_more_columns() -> None:
    table, train, test = _frontier()
    results = [design(_template(), train, OBJECTIVES[0], budget=b) for b in _ratios()]
    by_hand = compare(results, test, split="out_of_sample", options=FEW)
    assert table.drop_columns(FRONTIER).equals(by_hand)
    assert table.drop_columns(FRONTIER).column_names == by_hand.column_names


def test_a_fold_a_limit_a_pricing_and_design_options_reach_every_row() -> None:
    train, test = holdout(_years())
    pricing = CostOfCapitalLoad(0.06, 0.9)
    limit = Limit(300.0)
    options = DesignOptions(ranges=True, fill="highest")
    table = design_frontier(
        _template(),
        train,
        OBJECTIVES[2],
        budgets=_ratios((0.5, 1.0)),
        evaluate_on=test,
        split="out_of_sample",
        pricing=pricing,
        limit=limit,
        design_options=options,
        metric_options=FEW,
    )
    for position, budget in enumerate(_ratios((0.5, 1.0))):
        result = design(
            _template(),
            train,
            OBJECTIVES[2],
            pricing=pricing,
            budget=budget,
            limit=limit,
            options=options,
        )
        by_hand = evaluate(result, test, split="out_of_sample", options=FEW)
        assert (
            table.slice(position, 1).drop_columns(FRONTIER).equals(by_hand.slice(0, 1))
        )
    metadata = _metadata(table)
    assert all(record["limit"] == {"amount": 300.0} for record in metadata.designs)
    assert all(record["options"]["ranges"] is True for record in metadata.designs)
    assert metadata.setup.evaluations[0]["pricing"] == pricing.to_document()
    assert table.column("premium_basis").to_pylist() == ["design", "design"]


def test_a_multiple_with_an_expense_prices_every_row_and_its_bound() -> None:
    train, test = holdout(_years())
    pricing = ExpectedPayoutMultiple(1.4, 25.0)
    budgets = _ratios((0.5, 1.0))
    table = design_frontier(
        _template(),
        train,
        OBJECTIVES[0],
        budgets=budgets,
        evaluate_on=test,
        split="out_of_sample",
        pricing=pricing,
        metric_options=FEW,
    )
    for position, budget in enumerate(budgets):
        result = design(
            _template(), train, OBJECTIVES[0], pricing=pricing, budget=budget
        )
        by_hand = evaluate(result, test, split="out_of_sample", options=FEW)
        assert (
            table.slice(position, 1).drop_columns(FRONTIER).equals(by_hand.slice(0, 1))
        )
        payout = result.cover.payout_values(train.index[:, 0])
        assert result.premium == pytest.approx(
            1.4 * float(train.probability @ payout) + 25.0
        )
    # E[U] spends the whole budget, expense included.
    assert table.column("budget_use").to_pylist() == pytest.approx([1.0, 1.0])
    metadata = _metadata(table)
    assert metadata.setup.evaluations[0]["pricing"] == pricing.to_document()
    assert all(
        record["pricing"] == pricing.to_document() for record in metadata.designs
    )
    assert "P = 1.4 E[payout] + 25" in setup_text(metadata)


def test_rows_keep_the_order_given_and_do_not_depend_on_it() -> None:
    forward, _, _ = _frontier()
    shuffled = (1.0, 0.25, 1.5, 0.5)
    backward, _, _ = _frontier(budgets=_ratios(shuffled))
    assert backward.column("budget_ratio").to_pylist() == list(shuffled)
    by_ratio = {row["budget_ratio"]: row for row in backward.to_pylist()}
    for row in forward.to_pylist():
        assert by_ratio[row["budget_ratio"]] == row


def test_a_sample_with_index_components_uses_its_design_budget() -> None:
    wind = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
    rain = IndexDefinition("rain", "mm", aggregation="annual_maximum")
    n = 12
    years = np.arange(n, dtype=np.float64)
    first = 10.0 + 5.0 * np.sin(years)
    second = 20.0 + 8.0 * np.cos(years * 1.3)
    sample = OutcomeSample(
        asset_ids=("A",),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=first,
        loss=first * 2.0 + second,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(
            hazard_dependence={"kind": "data"}, combine={"kind": "table"}
        ),
        index_components={
            "wind": IndexComponent(
                first, unit="m/s", tail="upper", hazard=None, own_loss=None
            ),
            "rain": IndexComponent(
                second, unit="mm", tail="upper", hazard=None, own_loss=None
            ),
        },
    )
    template = Cover(
        CompositeIndex({"wind": wind, "rain": rain}),
        SumOf(
            {
                "wind": StepPayout((10.0, 12.0), (Free(), Free())),
                "rain": StepPayout((20.0, 24.0), (Free(), Free())),
            }
        ),
        payout_unit="EUR",
        asset_id="A",
    )
    budgets = _ratios((0.25, 0.5))
    table = design_frontier(
        template,
        sample,
        OBJECTIVES[0],
        budgets=budgets,
        evaluate_on=sample,
        split="in_sample",
        metric_options=FEW,
    )
    for position, budget in enumerate(budgets):
        result = design(template, sample, OBJECTIVES[0], budget=budget)
        by_hand = evaluate(result, sample, split="in_sample", options=FEW).to_pylist()[
            0
        ]
        row = table.slice(position, 1).to_pylist()[0]
        assert {key: row[key] for key in by_hand} == by_hand
        assert row["design_budget"] == result.budget_bound
        assert row["components"] == "wind+rain"


# -- the columns --------------------------------------------------------------


def test_the_columns_state_the_budget_its_bound_its_use_and_the_objective() -> None:
    train, test = holdout(_years())
    budgets = [Budget(ratio=0.5), Budget(amount=150.0), Budget(ratio=1.0)]
    table, _, _ = _frontier(budgets=budgets)
    results = [design(_template(), train, OBJECTIVES[0], budget=b) for b in budgets]
    assert table.column("budget_ratio").to_pylist() == [0.5, None, 1.0]
    assert table.column("budget_amount").to_pylist() == [None, 150.0, None]
    assert table.column("design_budget").to_pylist() == [
        result.budget_bound for result in results
    ]
    assert table.column("budget_use").to_pylist() == [
        result.budget_use for result in results
    ]
    assert table.column("objective_value").to_pylist() == [
        result.objective_value for result in results
    ]
    assert table.column("premium").to_pylist() == [result.premium for result in results]
    for name in FRONTIER:
        assert table.schema.field(name).type == pa.float64()
        assert table.schema.field(name).nullable
    assert test.content_hash in table.column("sample_hash").to_pylist()


def test_budget_use_is_null_at_a_zero_bound() -> None:
    table = design_frontier(
        _step([Free(), Free()]),
        SAMPLE,
        OBJECTIVES[0],
        budgets=[Budget(amount=0.0), Budget(ratio=0.0)],
        evaluate_on=SAMPLE,
        split="in_sample",
        metric_options=FEW,
    )
    assert table.column("design_budget").to_pylist() == [0.0, 0.0]
    assert table.column("budget_use").to_pylist() == [None, None]
    assert table.column("budget_amount").to_pylist() == [0.0, None]
    assert table.column("budget_ratio").to_pylist() == [None, 0.0]


def test_the_columns_sit_between_the_identity_and_the_metric_columns() -> None:
    table, _, _ = _frontier()
    identity = _identity_columns(table.column_names)
    assert identity[-4:] == FRONTIER
    assert table.column_names[: len(identity)] == identity
    assert table.column_names[len(identity)] == "expected_loss"


def test_to_long_keeps_the_four_columns_beside_the_identity_columns() -> None:
    table, _, _ = _frontier()
    long = to_long(table)
    assert set(FRONTIER) <= set(long.column_names)
    assert not set(FRONTIER) & set(long.column("column").unique().to_pylist())
    premiums = long.filter(pc.field("column") == "premium")
    assert (
        premiums.column("budget_ratio").to_pylist()
        == table.column("budget_ratio").to_pylist()
    )
    assert (
        premiums.column("objective_value").to_pylist()
        == table.column("objective_value").to_pylist()
    )


def test_the_table_writes_and_reads_back(tmp_path: Path) -> None:
    table, _, _ = _frontier()
    _metadata(table).check_table(table)
    path = tmp_path / "frontier.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.equals(table)
    assert setup_text(read) == setup_text(table)


# -- the bound ----------------------------------------------------------------


@pytest.mark.parametrize(
    "objective", OBJECTIVES, ids=lambda item: item.to_document()["kind"]
)
def test_the_objective_does_not_rise_with_the_bound_on_the_design_sample(
    objective: Objective,
) -> None:
    ratios = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    table, _, _ = _frontier(objective, budgets=_ratios(ratios))
    values = table.column("objective_value").to_pylist()
    tolerance = 1e-6 * max(1.0, max(values))
    assert all(
        later <= earlier + tolerance
        for earlier, later in zip(values, values[1:], strict=False)
    )
    assert values[-1] < values[0]


# -- non-uniqueness and degenerate tails, per row -----------------------------


def test_non_uniqueness_is_reported_row_by_row() -> None:
    # A budget of 5 can be split between the bands in many ways; at 20 the two
    # levels are the two losses each band pays, so the design is pinned.
    budgets = [Budget(amount=5.0), Budget(amount=20.0)]
    table = design_frontier(
        _step([Free(), Free()]),
        SAMPLE,
        OBJECTIVES[0],
        budgets=budgets,
        evaluate_on=SAMPLE,
        split="in_sample",
        design_options=DesignOptions(ranges=True),
        metric_options=FEW,
    )
    assert table.column("unique").to_pylist() == [False, True]
    assert table.column("ties").to_pylist() == [0, 0]
    assert [record["unique"] for record in _metadata(table).designs] == [False, True]
    text = setup_text(table)
    assert "unique optimum: no" in text and "unique optimum: yes" in text


def test_without_ranges_unique_is_null_and_the_text_says_not_checked() -> None:
    table = design_frontier(
        _step([Free(), Free()]),
        SAMPLE,
        OBJECTIVES[0],
        budgets=[Budget(amount=5.0), Budget(amount=20.0)],
        evaluate_on=SAMPLE,
        split="in_sample",
        metric_options=FEW,
    )
    assert table.column("unique").to_pylist() == [None, None]
    assert setup_text(table).count("unique optimum: not checked") == 2


@pytest.mark.parametrize(
    ("beta", "flag"),
    [(0.2, "expected_uncovered"), (0.8, "single_outcome"), (0.5, None)],
)
def test_a_degenerate_tail_is_reported_on_every_row(
    beta: float, flag: str | None
) -> None:
    objective = WeightedCVaR(0.9, beta)
    budgets = [Budget(amount=5.0), Budget(amount=10.0), Budget(amount=20.0)]
    table = design_frontier(
        _step([Free(), Free()]),
        SAMPLE,
        objective,
        budgets=budgets,
        evaluate_on=SAMPLE,
        split="in_sample",
        metric_options=FEW,
    )
    assert table.column("degenerate_tail").to_pylist() == [flag] * 3
    for position, budget in enumerate(budgets):
        result = design(_step([Free(), Free()]), SAMPLE, objective, budget=budget)
        assert (
            table.column("degenerate_tail")[position].as_py() == result.degenerate_tail
        )


def test_the_unidentified_share_is_reported_row_by_row() -> None:
    # No design year reaches the band above 10, and half the held-out years do.
    template = _step([Free(), Free(), Free()], (1.0, 2.5, 10.0))
    held_out = _small([0.0, 2.0, 12.0, 20.0], [0.0, 30.0, 50.0, 60.0], start=100)
    budgets = [Budget(amount=5.0), Budget(amount=20.0)]
    table = design_frontier(
        template,
        SAMPLE,
        OBJECTIVES[0],
        budgets=budgets,
        evaluate_on=held_out,
        split="out_of_sample",
        metric_options=FEW,
    )
    assert table.column("unidentified_share").to_pylist() == [0.5, 0.5]
    in_sample = design_frontier(
        template,
        SAMPLE,
        OBJECTIVES[0],
        budgets=budgets,
        evaluate_on=SAMPLE,
        split="in_sample",
        metric_options=FEW,
    )
    assert in_sample.column("unidentified_share").to_pylist() == [0.0, 0.0]


def test_ties_are_reported_row_by_row_on_a_grid() -> None:
    # Both breakpoints split the years the same way, so they tie.
    template = _step(
        [Free(), Free()], breakpoints=(Free(0.0, 1.0, grid=[0.5, 0.8]), 2.5)
    )
    table = design_frontier(
        template,
        SAMPLE,
        MSE(),
        budgets=[Budget(amount=5.0), Budget(amount=10.0)],
        evaluate_on=SAMPLE,
        split="in_sample",
        design_options=DesignOptions(ranges=True),
        metric_options=FEW,
    )
    assert table.column("ties").to_pylist() == [1, 1]
    assert table.column("unique").to_pylist() == [False, False]


def test_the_numpy_grid_needs_no_solver_and_leaves_the_share_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "highspy", None)

    def mean_absolute(loss: Any, payout: Any, probability: Any) -> float:
        return float(np.sum(probability * np.abs(loss - payout)))

    objective = UserObjective(mean_absolute, name="mae", parameters={"power": 1})
    grid = [0.0, 10.0, 20.0, 30.0, 40.0]
    template = _step([Free(0.0, 40.0, grid=grid), Free(0.0, 40.0, grid=grid)])
    budgets = [Budget(amount=5.0), Budget(amount=15.0)]
    table = design_frontier(
        template,
        SAMPLE,
        objective,
        budgets=budgets,
        evaluate_on=SAMPLE,
        split="in_sample",
        metric_options=FEW,
    )
    results = [design(template, SAMPLE, objective, budget=b) for b in budgets]
    assert all(result.route == "grid" and result.solver is None for result in results)
    assert table.column("unidentified_share").to_pylist() == [None, None]
    assert table.column("unique").to_pylist() == [r.unique for r in results]
    assert table.column("ties").to_pylist() == [r.ties for r in results]


def test_the_lp_route_needs_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "highspy", None)
    with pytest.raises(SolverUnavailableError, match=r"crc-sdk\[parametric\]"):
        design_frontier(
            _step([Free(), Free()]),
            SAMPLE,
            OBJECTIVES[0],
            budgets=[Budget(amount=5.0)],
            evaluate_on=SAMPLE,
            split="in_sample",
        )


# -- no best point ------------------------------------------------------------


def test_the_table_has_no_rank_no_best_flag_and_no_ordering_by_a_metric() -> None:
    table, train, test = _frontier()
    by_hand = compare(
        [design(_template(), train, OBJECTIVES[0], budget=Budget(ratio=1.0))],
        test,
        split="out_of_sample",
        options=FEW,
    )
    # Nothing beside compare's columns and the four the issue names.
    assert set(table.column_names) == set(by_hand.column_names) | set(FRONTIER)
    for name in table.column_names:
        for word in ("best", "rank", "knee", "marginal", "optimal"):
            assert word not in name
    # The budgets are not in any metric's order: they stay as given.
    shuffled = (1.0, 0.25, 1.5, 0.5)
    mixed, _, _ = _frontier(budgets=_ratios(shuffled))
    assert mixed.column("budget_ratio").to_pylist() == list(shuffled)
    assert mixed.column("objective_value").to_pylist() != sorted(
        mixed.column("objective_value").to_pylist()
    )
    assert mixed.column("objective_value").to_pylist() != sorted(
        mixed.column("objective_value").to_pylist(), reverse=True
    )


def test_the_setup_text_names_no_best_budget() -> None:
    table, _, _ = _frontier()
    text = setup_text(table)
    assert "The table names no best budget." in text
    lowered = text.replace("names no best budget", "")
    assert "best" not in lowered.lower().split("frontier:")[1].split("\n")[0]


# -- failures -----------------------------------------------------------------


def _call(budgets: Any, **changes: Any) -> Any:
    arguments: dict[str, Any] = {
        "evaluate_on": SAMPLE,
        "split": "in_sample",
        "metric_options": FEW,
    }
    arguments.update(changes)
    return UNTYPED_FRONTIER(
        _step([Free(), Free()]),
        SAMPLE,
        OBJECTIVES[0],
        budgets=budgets,
        **arguments,
    )


def test_an_empty_list_of_budgets_raises() -> None:
    with pytest.raises(EvaluationError, match="at least one budget"):
        _call([])


@pytest.mark.parametrize(
    ("budgets", "named"),
    [
        ([Budget(ratio=0.5), Budget(ratio=0.5)], r"ratio=0\.5"),
        ([Budget(ratio=1), Budget(ratio=1.0)], r"ratio=1\.0"),
        ([Budget(amount=5.0), Budget(ratio=1.0), Budget(amount=5.0)], r"amount=5\.0"),
    ],
)
def test_a_repeated_budget_raises_naming_it(budgets: list[Budget], named: str) -> None:
    with pytest.raises(EvaluationError, match=f"repeat {named}"):
        _call(budgets)


@pytest.mark.parametrize(
    "budgets",
    [Budget(ratio=1.0), "ratio", {"ratio": 1.0}, iter(()), None, [1.0]],
)
def test_budgets_of_the_wrong_type_raise(budgets: Any) -> None:
    with pytest.raises(TypeError, match="budgets"):
        _call(budgets)


def test_a_budget_the_template_cannot_meet_raises_design_error_naming_it() -> None:
    template = _step([Free(lower=10.0), Free(lower=10.0)])
    with pytest.raises(InfeasibleDesignError, match=r"amount=1\.0") as raised:
        design_frontier(
            template,
            SAMPLE,
            OBJECTIVES[0],
            budgets=[Budget(amount=20.0), Budget(amount=1.0)],
            evaluate_on=SAMPLE,
            split="in_sample",
        )
    assert isinstance(raised.value, DesignError)
    assert raised.value.asset_id == "A"
    assert raised.value.status == "infeasible"
    assert raised.value.solver_message
    assert isinstance(raised.value.__cause__, InfeasibleDesignError)


def test_a_ratio_the_template_cannot_meet_is_named_too() -> None:
    template = _step([Free(lower=10.0), Free(lower=10.0)])
    with pytest.raises(InfeasibleDesignError, match=r"ratio=0\.01"):
        design_frontier(
            template,
            SAMPLE,
            OBJECTIVES[0],
            budgets=[Budget(ratio=0.01)],
            evaluate_on=SAMPLE,
            split="in_sample",
        )


def test_a_budget_below_the_pricings_expense_is_named() -> None:
    with pytest.raises(
        InfeasibleDesignError,
        match=r"at budget amount=5\.0: asset 'A': the budget 5 is below the "
        r"pricing's fixed expense 10",
    ) as raised:
        design_frontier(
            _step([Free(), Free()]),
            SAMPLE,
            OBJECTIVES[0],
            budgets=[Budget(amount=20.0), Budget(amount=5.0)],
            evaluate_on=SAMPLE,
            split="in_sample",
            pricing=ExpectedPayoutMultiple(1.5, 10.0),
        )
    assert raised.value.status == "infeasible"


def test_a_sample_of_several_assets_raises_sample_error() -> None:
    n = 4
    sample = OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=np.tile(np.array([[0.0], [1.0], [2.0], [3.0]]), (1, 2)),
        loss=np.tile(np.array([[0.0], [10.0], [30.0], [40.0]]), (1, 2)),
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(dependence={"kind": "data"}),
    )
    with pytest.raises(SampleError, match="several assets|one asset"):
        design_frontier(
            _step([Free(), Free()]),
            sample,
            OBJECTIVES[0],
            budgets=[Budget(ratio=0.5)],
            evaluate_on=sample,
            split="in_sample",
        )


def test_split_has_no_default() -> None:
    with pytest.raises(TypeError, match="split"):
        UNTYPED_FRONTIER(
            _step([Free(), Free()]),
            SAMPLE,
            OBJECTIVES[0],
            budgets=[Budget(ratio=0.5)],
            evaluate_on=SAMPLE,
        )


def test_budgets_and_the_block_are_keywords() -> None:
    with pytest.raises(TypeError):
        UNTYPED_FRONTIER(
            _step([Free(), Free()]), SAMPLE, OBJECTIVES[0], [Budget(ratio=0.5)], SAMPLE
        )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"split": "elsewhere"}, EvaluationError),
        ({"fold": -1}, EvaluationError),
        ({"evaluate_on": "SAMPLE"}, TypeError),
        ({"metric_options": {"betas": (0.9,)}}, TypeError),
    ],
)
def test_invalid_arguments_raise(
    changes: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _call([Budget(ratio=0.5)], **changes)


def test_an_out_of_sample_block_that_shares_periods_raises() -> None:
    with pytest.raises(LeakageError):
        _call([Budget(ratio=0.5)], split="out_of_sample")


def test_an_in_sample_row_must_read_the_design_sample() -> None:
    other = _small([0.0, 1.0, 2.0, 3.0], [5.0, 15.0, 30.0, 40.0], start=100)
    with pytest.raises(SampleError):
        _call([Budget(ratio=0.5)], evaluate_on=other)


def test_a_cover_with_nothing_to_design_raises_as_design_does() -> None:
    with pytest.raises(ValueError, match="no free parameters"):
        design_frontier(
            _step([10.0, 40.0]),
            SAMPLE,
            OBJECTIVES[0],
            budgets=[Budget(ratio=0.5)],
            evaluate_on=SAMPLE,
            split="in_sample",
        )


# -- the metadata -------------------------------------------------------------


def test_setup_frontier_holds_the_template_the_objective_and_the_budgets() -> None:
    train, test = holdout(_years())
    objective = OBJECTIVES[2]
    budgets = [Budget(ratio=0.5), Budget(amount=150.0)]
    table = design_frontier(
        _template(),
        train,
        objective,
        budgets=budgets,
        evaluate_on=test,
        split="out_of_sample",
        metric_options=FEW,
    )
    metadata = _metadata(table)
    assert metadata.setup.frontier is not None
    assert len(metadata.setup.frontier) == 1
    record = metadata.setup.frontier[0]
    assert record["template_hash"] == _template().definition_hash
    assert record["objective"] == objective.to_document()
    assert record["budgets"] == [{"ratio": 0.5}, {"amount": 150.0}]
    assert record["design_ids"] == table.column("design_id").to_pylist()
    assert record["design_sample"] == train.content_hash
    assert record["sample_hash"] == test.content_hash
    assert (record["split"], record["fold"]) == ("out_of_sample", 0)


def test_designs_holds_one_record_per_budget() -> None:
    table, _, _ = _frontier()
    metadata = _metadata(table)
    assert [record["design_id"] for record in metadata.designs] == (
        table.column("design_id").to_pylist()
    )
    assert [record["budget"] for record in metadata.designs] == [
        budget.to_document() for budget in _ratios()
    ]


def test_a_comparison_has_no_frontier_and_its_document_no_key() -> None:
    train, test = holdout(_years())
    result = design(_template(), train, OBJECTIVES[0], budget=Budget(ratio=0.5))
    table = compare([result], test, split="out_of_sample", options=FEW)
    assert _metadata(table).setup.frontier is None
    document = json.loads(table.schema.metadata[EVALUATION_METADATA_KEY.encode()])
    assert "frontier" not in document["setup"]
    assert "Frontier" not in setup_text(table)


@pytest.mark.parametrize(
    "frontier",
    [
        ({"budgets": [{"ratio": 1.0}]},),
        (
            {
                "budgets": [{"ratio": 1.0}, {"ratio": 2.0}],
                "design_ids": ["sha256:" + "1" * 64],
                "design_sample": "sha256:" + "2" * 64,
                "fold": 0,
                "objective": {"kind": "mse"},
                "sample_hash": "sha256:" + "3" * 64,
                "split": "in_sample",
                "template_hash": "sha256:" + "4" * 64,
            },
        ),
    ],
)
def test_an_invalid_frontier_record_raises(frontier: Any) -> None:
    with pytest.raises(EvaluationError, match="frontier"):
        EvaluationSetup(
            metric_options={}, samples=(), evaluations=(), frontier=frontier
        )


# -- the setup text -----------------------------------------------------------


def test_the_setup_text_has_a_frontier_line_with_the_blocks_and_every_budget() -> None:
    train, test = holdout(_years())
    table, _, _ = _frontier(budgets=[Budget(ratio=0.5), Budget(amount=150.0)])
    lines = [line for line in setup_text(table).splitlines() if "Frontier" in line]
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("Frontier: one design per budget (0.5 x E[L] and 150 EUR)")
    assert _template().definition_hash.removeprefix("sha256:")[:12] in line
    assert f"{len(train)} years" in line and "the design block" in line
    assert f"{len(test)} years" in line and "the held-out block" in line
    assert "(out of sample, fold 0)" in line
    assert line.endswith("The table names no best budget.")


def test_a_frontier_read_in_sample_states_its_split() -> None:
    train, _ = holdout(_years())
    table = design_frontier(
        _template(),
        train,
        OBJECTIVES[0],
        budgets=_ratios((0.5, 1.0)),
        evaluate_on=train,
        split="in_sample",
        metric_options=FEW,
    )
    text = setup_text(table)
    assert "(in sample, fold 0)" in text
    assert "In-sample numbers show the fit" in text


def test_every_design_has_its_line_in_the_setup_text() -> None:
    table, _, _ = _frontier()
    assert setup_text(table).count("Design ") == len(RATIOS)


def test_all_the_rows_give_the_text_of_the_metadata() -> None:
    table, _, _ = _frontier()
    assert setup_text(table) == setup_text(_metadata(table))


def test_filtered_rows_state_the_budgets_they_still_hold() -> None:
    table, _, _ = _frontier()
    kept = table.filter(pc.field("budget_ratio") == 0.5)
    text = setup_text(kept)
    assert "Frontier: one design per budget (0.5 x E[L])" in text
    assert "1.5 x E[L]" not in text
    assert text.count("Design ") == 1


def test_a_setup_alone_states_the_frontier() -> None:
    table, _, _ = _frontier()
    setup = _metadata(table).setup
    assert "Frontier: one design per budget" in setup_text(setup)
    assert "Frontier: one design per budget" in setup_text(
        setup.model_dump(mode="json")
    )


def test_a_setup_with_a_frontier_of_the_wrong_type_raises() -> None:
    with pytest.raises(EvaluationError, match="frontier"):
        setup_text({"frontier": "nope"})


def test_a_table_without_row_keys_states_the_whole_frontier() -> None:
    table, _, _ = _frontier()
    text = setup_text(table.drop_columns(["fold"]))
    assert text.startswith("Rows: not identified")
    assert "0.25 x E[L], 0.5 x E[L], 1 x E[L] and 1.5 x E[L]" in text


# -- several calls ------------------------------------------------------------


def _two_objectives() -> tuple[Any, Any]:
    first, _, _ = _frontier(OBJECTIVES[0], budgets=_ratios((0.5, 1.0)))
    second, _, _ = _frontier(OBJECTIVES[2], budgets=_ratios((0.5, 1.0)))
    return first, second


def test_several_objectives_go_through_several_calls_and_merge(tmp_path: Path) -> None:
    first, second = _two_objectives()
    merged = EvaluationMetadata.merge(_metadata(first), _metadata(second))
    assert merged.setup.frontier is not None
    assert [record["objective"]["kind"] for record in merged.setup.frontier] == [
        "expected_uncovered",
        "weighted_cvar",
    ]
    table = pa.concat_tables([first, second])
    merged.check_table(table)
    path = tmp_path / "frontiers.parquet"
    write_comparison(table, path, metadata=merged)
    read = pq.read_table(path)
    assert read.num_rows == 4
    assert setup_text(read).count("Frontier: one design per budget") == 2


def _read_on(block: OutcomeSample, split: Any, *, fold: int = 0) -> Any:
    # The same frontier each time: only the block it is read on changes.
    train, _ = holdout(_years())
    return design_frontier(
        _template(),
        train,
        OBJECTIVES[0],
        budgets=_ratios((0.5, 1.0)),
        evaluate_on=block,
        split=split,
        fold=fold,
        metric_options=FEW,
    )


def _block(kind: str) -> tuple[OutcomeSample, str, int]:
    train, test = holdout(_years())
    if kind == "in_sample":
        return train, "in_sample", 0
    # Another held-out block, of synthetic years no design sample holds.
    later = _small(test.index.tolist(), test.loss.tolist(), start=1000)
    return later, "out_of_sample", 1


@pytest.mark.parametrize("kind", ["in_sample", "another_fold"])
def test_one_frontier_read_on_two_blocks_goes_to_one_file(
    kind: str, tmp_path: Path
) -> None:
    _, test = holdout(_years())
    outside = _read_on(test, "out_of_sample")
    block, split, fold = _block(kind)
    other = _read_on(block, split, fold=fold)
    assert other.column("design_id").to_pylist() == (
        outside.column("design_id").to_pylist()
    )

    merged = EvaluationMetadata.merge(_metadata(other), _metadata(outside))

    assert merged.designs == _metadata(other).designs
    assert merged.setup.frontier is not None
    assert [record["split"] for record in merged.setup.frontier] == [
        split,
        "out_of_sample",
    ]
    assert [record["fold"] for record in merged.setup.frontier] == [fold, 0]
    table = pa.concat_tables([other, outside])
    path = tmp_path / "frontier_on_two_blocks.parquet"
    write_comparison(table, path, metadata=merged)
    read = pq.read_table(path)
    assert read.num_rows == 4
    assert read.equals(table)
    assert EvaluationMetadata.from_table(read) == merged
    assert setup_text(read).count("Frontier: one design per budget") == 2


def test_one_frontier_read_twice_with_one_row_key_is_refused_when_written(
    tmp_path: Path,
) -> None:
    _, test = holdout(_years())
    first = _read_on(test, "out_of_sample")
    second = _read_on(_block("another_fold")[0], "out_of_sample")
    merged = EvaluationMetadata.merge(_metadata(first), _metadata(second))
    path = tmp_path / "repeated_key.parquet"
    with pytest.raises(EvaluationError, match="repeats a row key"):
        write_comparison(pa.concat_tables([first, second]), path, metadata=merged)
    assert not path.exists()


def test_filtered_rows_of_a_merge_keep_one_frontier() -> None:
    first, second = _two_objectives()
    merged = EvaluationMetadata.merge(_metadata(first), _metadata(second))
    table = pa.concat_tables([first, second]).replace_schema_metadata(
        {EVALUATION_METADATA_KEY.encode(): merged.to_json().encode()}
    )
    kept = table.filter(pc.field("design_id") == first.column("design_id")[0].as_py())
    text = setup_text(kept)
    assert text.count("Frontier: one design per budget") == 1
    assert "E[U], the expected uncovered loss" in text.split("Frontier:")[1]


def test_a_merge_with_a_comparison_keeps_the_frontier() -> None:
    table, train, test = _frontier()
    result = design(_template(), train, OBJECTIVES[1], budget=Budget(ratio=0.5))
    plain = compare([result], test, split="out_of_sample", options=FEW)
    merged = EvaluationMetadata.merge(_metadata(table), _metadata(plain))
    assert merged.setup.frontier == _metadata(table).setup.frontier


def test_a_merge_of_comparisons_has_no_frontier() -> None:
    train, test = holdout(_years())
    result = design(_template(), train, OBJECTIVES[1], budget=Budget(ratio=0.5))
    plain = compare([result], test, split="out_of_sample", options=FEW)
    merged = EvaluationMetadata.merge(_metadata(plain), _metadata(plain))
    assert merged.setup.frontier is None
    assert "frontier" not in json.loads(merged.to_json())["setup"]


def test_a_merge_repeats_no_frontier_entry() -> None:
    table, _, _ = _frontier()
    merged = EvaluationMetadata.merge(_metadata(table), _metadata(table))
    assert merged.setup.frontier == _metadata(table).setup.frontier


def test_a_linear_payout_takes_a_frontier_too() -> None:
    # A second family, on the LP route, read by hand.
    template = Cover(
        INDEX, LinearPayout(1.0, 3.0, Free()), payout_unit="EUR", asset_id="A"
    )
    budgets = [Budget(amount=5.0), Budget(amount=15.0)]
    table = design_frontier(
        template,
        SAMPLE,
        OBJECTIVES[0],
        budgets=budgets,
        evaluate_on=SAMPLE,
        split="in_sample",
        metric_options=FEW,
    )
    for position, budget in enumerate(budgets):
        result = design(template, SAMPLE, OBJECTIVES[0], budget=budget)
        by_hand = evaluate(result, SAMPLE, split="in_sample", options=FEW)
        assert (
            table.slice(position, 1).drop_columns(FRONTIER).equals(by_hand.slice(0, 1))
        )
