"""`payout_frequency`, `unidentified_share` and the setup text on fresh draws.

The samples are small, hand-made and synthetic, chosen so that each number can
be counted by eye. The tests check what the module computes and records; nothing
here is a finding about a hazard, a cover or a design.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact
from numpy.typing import ArrayLike

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    CompositeIndex,
    Cover,
    CoverPackage,
    DesignOptions,
    DesignResult,
    EvaluationError,
    EvaluationMetadata,
    Free,
    IndexComponent,
    IndexDefinition,
    LossModel,
    MaxOf,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    PiecewiseLinearPayout,
    SampleProvenance,
    StepGrid,
    StepPayout,
    SumOf,
    UserObjective,
    compare,
    design,
    design_joint,
    evaluate,
    holdout,
    payout_breakdown,
    sample_curve,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY
from crc_sdk.parametric.metrics import metric_values
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
    ),
]

DOCS = Path(__file__).parent.parent / "docs" / "parametric.md"
UNTYPED_EVALUATE: Any = evaluate
UNTYPED_COMPARE: Any = compare

INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
TWO = CompositeIndex({"rain": RAIN, "heat": HEAT})

ADDED = ("payout_frequency", "unidentified_share")
SAME_CURVE = (
    "Out of sample here means new draws from the same curve, so these rows test "
    "the design's sampling error, not the curve."
)

# The design block reaches the first band of `_step_template` only, and the
# blocks below read the unreached bands or do not.
DESIGN_YEARS = np.array([5.0, 12.0, 14.0, 5.0, 16.0, 8.0])
DESIGN_LOSS = np.array([0.0, 4.0, 6.0, 0.0, 8.0, 0.0])
REACHES = np.array([5.0, 15.0, 25.0, 35.0])
STAYS_BELOW = np.array([5.0, 15.0, 18.0, 3.0])


def _provenance(**fields: Any) -> SampleProvenance:
    base: dict[str, Any] = {
        "source_kind": "table",
        "sources": ({"role": "table", "name": "test table", "licence": "CC0-1.0"},),
        "semantics_source": "caller",
        "licences": ({"role": "table", "source": "test table", "licence": "CC0-1.0"},),
    }
    return SampleProvenance(**{**base, **fields})


def _sample(
    index: ArrayLike,
    loss: ArrayLike | None = None,
    *,
    start: int = 2000,
    probability: ArrayLike | None = None,
) -> OutcomeSample:
    values = np.asarray(index, dtype=np.float64)
    n = values.shape[0]
    chance = np.full(n, 1.0 / n) if probability is None else np.asarray(probability)
    return OutcomeSample(
        asset_ids=("site",),
        period=np.arange(start, start + n),
        probability=chance,
        index=values,
        loss=np.full(n, 3.0) if loss is None else np.asarray(loss, dtype=np.float64),
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(),
    )


def _design_block() -> OutcomeSample:
    return _sample(DESIGN_YEARS, DESIGN_LOSS)


def _book(
    a: Sequence[float], b: Sequence[float], *, start: int = 2000
) -> OutcomeSample:
    """A sample of two assets, each with the loss of its own index."""
    index = np.column_stack([a, b]).astype(np.float64)
    return OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(start, start + index.shape[0]),
        probability=np.full(index.shape[0], 1.0 / index.shape[0]),
        index=index,
        loss=index * 0.5,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(dependence={"kind": "data"}),
    )


def _hazards(rain: ArrayLike, heat: ArrayLike, *, start: int = 2000) -> OutcomeSample:
    """A sample of one asset and two hazards, with the combined loss of both."""
    rain_values = np.asarray(rain, dtype=np.float64)
    heat_values = np.asarray(heat, dtype=np.float64)
    n = rain_values.shape[0]
    return OutcomeSample(
        asset_ids=("site",),
        period=np.arange(start, start + n),
        probability=np.full(n, 1.0 / n),
        index=rain_values,
        loss=rain_values + heat_values,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(
            hazard_dependence={"kind": "data"}, combine={"kind": "table"}
        ),
        index_components={
            "rain": IndexComponent(
                rain_values, unit="mm/day", tail="upper", own_loss=rain_values
            ),
            "heat": IndexComponent(
                heat_values, unit="degC", tail="upper", own_loss=heat_values
            ),
        },
    )


# The hazard design block, of eight years, and the four held-out years. Neither
# hazard reaches 20 in the design block. Of the held-out years, the first reads
# past 20 on rain only, the second on heat only, the third on both, the last on
# neither.
RAIN_YEARS = [5.0, 12.0, 15.0, 5.0, 12.0, 8.0, 14.0, 5.0, 25.0, 5.0, 25.0, 12.0]
HEAT_YEARS = [5.0, 5.0, 12.0, 14.0, 12.0, 5.0, 16.0, 11.0, 5.0, 25.0, 25.0, 12.0]
SPLIT_AT = 8 / 12


def _step_template(*, label: str = "step", **kwargs: Any) -> Cover:
    return Cover(
        INDEX,
        StepPayout([10.0, 20.0, 30.0], [Free(), Free(), Free()], **kwargs),
        payout_unit="EUR",
        label=label,
    )


def _two_bands(
    index: IndexDefinition = INDEX, *, asset: str | None = None, label: str = "two"
) -> Cover:
    return Cover(
        index,
        StepPayout([10.0, 20.0], [Free(), Free()]),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _designed(**options: Any) -> DesignResult:
    return design(
        _step_template(),
        _design_block(),
        MSE(),
        options=DesignOptions(**options),
    )


def _one(table: Any) -> dict[str, Any]:
    row: dict[str, Any] = table.to_pylist()[0]
    return row


def _row(table: Any, level: str, asset: str | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        row
        for row in table.to_pylist()
        if row["level"] == level and (asset is None or row["asset_id"] == asset)
    ]
    assert len(rows) == 1
    return rows[0]


def _out_of_sample(design_: Any, index: ArrayLike, **rest: Any) -> dict[str, Any]:
    block = _sample(index, start=2100, **rest)
    return _one(UNTYPED_EVALUATE(design_, block, split="out_of_sample"))


@dataclass(frozen=True)
class _Design:
    """A designed cover, as a user's design function would return it."""

    cover: Cover
    design_sample: OutcomeSample
    record: dict[str, Any] = field(default_factory=dict)
    premium: float = 1.0
    design_id: str = "sha256:" + "1" * 64
    unique: bool | None = None
    ties: int | None = 0
    degenerate_tail: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {"design_id": self.design_id, **self.record}


# -- payout_frequency ----------------------------------------------------------


def test_payout_frequency_is_the_probability_that_the_cover_pays() -> None:
    probability = np.array([0.1, 0.2, 0.3, 0.4])
    values = metric_values(
        np.zeros(4),
        np.array([0.0, 2.0, 0.0, 5.0]),
        probability,
        premium=0.0,
        options=MetricOptions(),
    )
    assert values["payout_frequency"] == pytest.approx(0.6, abs=1e-15)


def test_a_payout_at_the_threshold_counts_as_none() -> None:
    options = MetricOptions(payout_threshold=2.0)
    values = metric_values(
        np.zeros(4),
        np.array([1.0, 2.0, 2.5, 5.0]),
        np.full(4, 0.25),
        premium=0.0,
        options=options,
    )
    assert values["payout_frequency"] == 0.5


def test_payout_frequency_is_never_null() -> None:
    values = metric_values(
        np.zeros(2), np.zeros(2), np.full(2, 0.5), premium=0.0, options=MetricOptions()
    )
    assert values["payout_frequency"] == 0.0


def test_a_comparison_row_holds_the_frequency_of_its_payout() -> None:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    row = _one(evaluate(cover, _sample(REACHES), split="in_sample"))
    assert row["payout_frequency"] == 0.75


def test_an_allocated_row_holds_the_frequency_of_its_share() -> None:
    book = _book([0.0, 12.0, 0.0, 12.0], [0.0, 0.0, 12.0, 12.0])
    templates = [
        Cover(
            INDEX,
            StepPayout([10.0], [Free()]),
            payout_unit="EUR",
            asset_id=asset,
            label="one",
        )
        for asset in book.asset_ids
    ]
    table = compare(
        [design_joint(templates, book, MSE())],
        book,
        split="in_sample",
        allocation="loss",
        values={"A": 1.0, "B": 1.0},
    )
    # A share by loss goes to asset A in its two loss years, the years it pays.
    assert _row(table, "asset_by_loss", "A")["payout_frequency"] == 0.5
    assert _row(table, "total")["payout_frequency"] == 0.75


# -- the columns ---------------------------------------------------------------


def test_the_two_columns_follow_payout_to_loss_ratio_in_the_table() -> None:
    names = _one_table_names()
    at = names.index("payout_to_loss_ratio")
    assert tuple(names[at + 1 : at + 3]) == ADDED


def _one_table_names() -> list[str]:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    names: list[str] = evaluate(cover, _sample(REACHES), split="in_sample").column_names
    return names


def test_the_metadata_describes_the_two_columns() -> None:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    table = evaluate(
        cover,
        _sample(REACHES),
        split="in_sample",
        options=MetricOptions(payout_threshold=2.5),
    )
    columns = EvaluationMetadata.from_table(table).columns
    frequency = columns["payout_frequency"]
    assert (frequency.metric, frequency.parameter, frequency.unit) == (
        "payout_frequency",
        None,
        "1",
    )
    assert "P(payout > 2.5)" in frequency.definition
    share = columns["unidentified_share"]
    assert (share.metric, share.parameter, share.unit) == (
        "unidentified_share",
        None,
        "1",
    )
    assert "fill rule" in share.definition


def test_the_long_form_carries_the_two_columns() -> None:
    long = to_long(evaluate(_designed(), _design_block(), split="in_sample"))
    rows = [row for row in long.to_pylist() if row["column"] in ADDED]
    assert [row["column"] for row in rows] == list(ADDED)
    assert {row["unit"] for row in rows} == {"1"}
    assert all(row["parameter"] is None for row in rows)
    assert [row["value"] for row in rows] == [pytest.approx(0.5), 0.0]


def test_the_docs_list_the_two_columns() -> None:
    page = DOCS.read_text(encoding="utf-8")
    section = page[page.index("### Metrics") :]
    section = section[: section.index("\n### ")]
    for name in ADDED:
        assert f"| `{name}` |" in section


# -- metric_values -------------------------------------------------------------


def test_metric_values_sums_the_probability_of_the_draws_it_is_given() -> None:
    values = metric_values(
        np.zeros(4),
        np.zeros(4),
        np.array([0.1, 0.2, 0.3, 0.4]),
        premium=0.0,
        options=MetricOptions(),
        unidentified=[False, True, False, True],
    )
    assert values["unidentified_share"] == pytest.approx(0.6, abs=1e-15)


def test_metric_values_without_the_draws_reads_null() -> None:
    values = metric_values(
        np.zeros(2), np.zeros(2), np.full(2, 0.5), premium=0.0, options=MetricOptions()
    )
    assert values["unidentified_share"] is None


@pytest.mark.parametrize(
    "unidentified", [[True], [1, 0], [0.0, 1.0], [[True, False]], "no"]
)
def test_metric_values_refuses_anything_but_one_boolean_per_draw(
    unidentified: Any,
) -> None:
    with pytest.raises(EvaluationError, match="one boolean per draw"):
        metric_values(
            np.zeros(2),
            np.zeros(2),
            np.full(2, 0.5),
            premium=0.0,
            options=MetricOptions(),
            unidentified=unidentified,
        )


# -- a design on a block that reaches its unidentified amounts -----------------


def test_a_design_with_unidentified_amounts_read_on_a_block_that_reaches_them() -> None:
    result = _designed()
    assert result.unidentified == ("levels[1]", "levels[2]")
    assert _out_of_sample(result, REACHES)["unidentified_share"] == 0.5


def test_the_share_weighs_the_years_by_their_probability() -> None:
    probability = np.array([0.1, 0.2, 0.3, 0.4])
    row = _out_of_sample(_designed(), REACHES, probability=probability)
    assert row["unidentified_share"] == pytest.approx(0.7, abs=1e-15)


def test_the_share_is_zero_on_the_design_block() -> None:
    table = evaluate(_designed(), _design_block(), split="in_sample")
    assert _one(table)["unidentified_share"] == 0.0


def test_the_share_is_zero_on_a_block_that_does_not_reach_them() -> None:
    assert _out_of_sample(_designed(), STAYS_BELOW)["unidentified_share"] == 0.0


def test_a_design_with_every_amount_identified_reads_zero() -> None:
    result = design(
        Cover(INDEX, StepPayout([10.0], [Free()]), payout_unit="EUR", label="one"),
        _design_block(),
        MSE(),
    )
    assert result.unidentified == ()
    assert _out_of_sample(result, REACHES)["unidentified_share"] == 0.0


def test_a_held_out_block_of_one_sample_reads_the_share() -> None:
    sample = _sample(
        [5.0, 12.0, 14.0, 5.0, 16.0, 8.0, 5.0, 25.0], [0, 4, 6, 0, 8, 0, 0, 20]
    )
    train, test = holdout(sample, train_fraction=0.75)
    result = design(_step_template(), train, MSE())
    table = evaluate(result, test, split="out_of_sample")
    assert _one(table)["unidentified_share"] == 0.5


def test_the_filled_amount_does_not_matter_only_where_it_came_from() -> None:
    lowest = _designed()
    # With monotone levels the lowest value allowed is the level before it.
    values = lowest.cover.payout.parameters()
    assert values["levels[1]"] == values["levels[0]"]
    bounded = design(
        Cover(
            INDEX,
            StepPayout([10.0, 20.0, 30.0], [Free(0.0, 50.0)] * 3),
            payout_unit="EUR",
            label="step",
        ),
        _design_block(),
        MSE(),
        options=DesignOptions(fill="highest"),
    )
    assert bounded.cover.payout.parameters()["levels[2]"] == 50.0
    for result in (lowest, bounded):
        assert _out_of_sample(result, REACHES)["unidentified_share"] == 0.5


def test_a_year_that_only_partly_reads_a_filled_knot_counts_in_full() -> None:
    result = design(
        Cover(
            INDEX,
            PiecewiseLinearPayout([10.0, 20.0, 30.0], [Free(), Free(), Free()]),
            payout_unit="EUR",
            label="piecewise",
        ),
        _sample([5.0, 12.0, 15.0, 20.0, 18.0], [0.0, 3.0, 5.0, 9.0, 7.0]),
        MSE(),
    )
    assert result.unidentified == ("levels[2]",)
    # 20 reads knot 1 alone; 21 reads a tenth of knot 2, and 40 reads it whole.
    row = _out_of_sample(result, [5.0, 20.0, 21.0, 40.0])
    assert row["unidentified_share"] == 0.5


def test_a_fixed_amount_is_never_filled() -> None:
    result = design(
        Cover(
            INDEX,
            StepPayout([10.0, 20.0], [Free(), 5.0]),
            payout_unit="EUR",
            label="fixed",
        ),
        _design_block(),
        MSE(),
    )
    assert result.unidentified == ()
    row = _out_of_sample(result, REACHES)
    assert row["unidentified_share"] == 0.0
    assert row["payout_frequency"] == 0.75


def test_a_grid_over_the_shapes_reads_the_chosen_candidate() -> None:
    result = design(
        Cover(
            INDEX,
            StepPayout(
                [Free(6.0, 12.0, grid=[6.0, 12.0]), 20.0, 30.0],
                [Free(), Free(), Free()],
            ),
            payout_unit="EUR",
            label="shape",
        ),
        _design_block(),
        MSE(),
    )
    assert (result.route, result.amounts) == ("grid", "qp")
    assert result.unidentified == ("levels[1]", "levels[2]")
    assert _out_of_sample(result, REACHES)["unidentified_share"] == 0.5


# -- where the share is null ---------------------------------------------------


def test_a_cover_given_as_is_reads_null() -> None:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    block = _design_block()
    assert _one(evaluate(cover, block, split="in_sample"))["unidentified_share"] is None
    table = evaluate(
        cover,
        _sample(REACHES, start=2100),
        split="out_of_sample",
        design_sample=block,
    )
    assert _one(table)["unidentified_share"] is None


def test_the_numpy_grid_reads_null() -> None:
    grid = [0.0, 4.0, 8.0]

    def squared(loss: Any, payout: Any, probability: Any) -> float:
        return float(np.sum(probability * (loss - payout) ** 2))

    result = design(
        Cover(
            INDEX,
            StepPayout(
                [10.0, 20.0], [Free(0.0, 8.0, grid=grid), Free(0.0, 8.0, grid=grid)]
            ),
            payout_unit="EUR",
            label="grid",
        ),
        _design_block(),
        UserObjective(squared, name="squared"),
    )
    assert (result.route, result.amounts) == ("grid", "grid")
    assert result.unidentified == ()
    table = evaluate(result, _design_block(), split="in_sample")
    assert _one(table)["unidentified_share"] is None
    assert _out_of_sample(result, REACHES)["unidentified_share"] is None


def test_a_record_without_the_key_reads_null() -> None:
    cover = Cover(INDEX, StepPayout([10.0], [2.0]), payout_unit="EUR", label="step")
    stub = _Design(cover, _design_block())
    assert _out_of_sample(stub, REACHES)["unidentified_share"] is None


def test_a_record_of_a_grid_reads_null_even_with_the_key() -> None:
    cover = Cover(INDEX, StepPayout([10.0], [2.0]), payout_unit="EUR", label="step")
    stub = _Design(
        cover,
        _design_block(),
        {"unidentified": ["levels[0]"], "amounts": "grid"},
    )
    assert _out_of_sample(stub, REACHES)["unidentified_share"] is None


def test_a_record_that_names_amounts_the_cover_lacks_reads_null() -> None:
    cover = Cover(INDEX, StepPayout([10.0], [2.0]), payout_unit="EUR", label="step")
    stub = _Design(cover, _design_block(), {"unidentified": ["its_own_name"]})
    assert _out_of_sample(stub, REACHES)["unidentified_share"] is None


def test_a_user_payout_with_no_amounts_reads_null_only_if_it_lists_some() -> None:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    empty = _Design(cover, _design_block(), {"unidentified": []})
    assert _out_of_sample(empty, REACHES)["unidentified_share"] == 0.0


@pytest.mark.parametrize("names", ["levels[0]", 3, [1], {"levels[0]": True}])
def test_a_malformed_record_raises(names: Any) -> None:
    cover = Cover(INDEX, StepPayout([10.0], [2.0]), payout_unit="EUR", label="step")
    stub = _Design(cover, _design_block(), {"unidentified": names})
    with pytest.raises(EvaluationError, match="unidentified"):
        _out_of_sample(stub, REACHES)


# -- a joint design, a set of designs, a package -------------------------------

# Neither asset reaches 20 in the design block. In the evaluation block year 0
# reads past 20 on asset A only, year 1 on B only, year 2 on both, year 3 on none.
BOOK_DESIGN = (
    [0.0, 12.0, 15.0, 0.0, 5.0],
    [0.0, 0.0, 12.0, 15.0, 5.0],
)
BOOK_READ = (
    [25.0, 5.0, 25.0, 12.0],
    [5.0, 25.0, 25.0, 12.0],
)


def _joint_book() -> tuple[Any, OutcomeSample]:
    book = _book(*BOOK_DESIGN)
    templates = [_two_bands(asset=asset) for asset in book.asset_ids]
    return design_joint(templates, book, MSE()), book


def test_a_joint_design_counts_a_year_if_any_asset_reads_a_filled_amount() -> None:
    result, book = _joint_book()
    assert dict(result.unidentified) == {"A": ("levels[1]",), "B": ("levels[1]",)}
    block = _book(*BOOK_READ, start=2100)
    table = compare(
        [result],
        block,
        split="out_of_sample",
        allocation="value",
        values={"A": 3.0, "B": 1.0},
    )
    assert _row(table, "total")["unidentified_share"] == 0.75
    assert _row(table, "asset_by_value", "A")["unidentified_share"] == 0.75
    assert _row(table, "asset_by_value", "B")["unidentified_share"] == 0.75
    in_sample = compare([result], book, split="in_sample")
    assert _row(in_sample, "total")["unidentified_share"] == 0.0


def test_each_asset_of_a_joint_design_reads_its_own_amounts() -> None:
    # Asset B reaches past 20 in the design block, so only A has a filled amount.
    book = _book([0.0, 12.0, 15.0, 0.0, 5.0], [0.0, 0.0, 25.0, 15.0, 5.0])
    templates = [_two_bands(asset=asset) for asset in book.asset_ids]
    result = design_joint(templates, book, MSE())
    assert dict(result.unidentified) == {"A": ("levels[1]",), "B": ()}
    table = compare([result], _book(*BOOK_READ, start=2100), split="out_of_sample")
    assert _row(table, "total")["unidentified_share"] == 0.5


def _designs_of_each_asset() -> dict[str, Any]:
    book = _book(*BOOK_DESIGN)
    return {
        asset: design(_two_bands(asset=asset), book.asset(asset), MSE())
        for asset in book.asset_ids
    }


def test_a_set_of_designs_reads_each_asset_on_its_row_and_the_union_on_the_total() -> (
    None
):
    table = compare(
        [_designs_of_each_asset()],
        _book(*BOOK_READ, start=2100),
        split="out_of_sample",
    )
    assert _row(table, "asset", "A")["unidentified_share"] == 0.5
    assert _row(table, "asset", "B")["unidentified_share"] == 0.5
    assert _row(table, "total")["unidentified_share"] == 0.75


def test_a_set_of_covers_given_as_is_reads_null_on_every_row() -> None:
    book = _book(*BOOK_DESIGN)
    covers = {
        asset: Cover(
            INDEX,
            StepPayout([10.0], [2.0]),
            payout_unit="EUR",
            asset_id=asset,
            label="given",
        )
        for asset in book.asset_ids
    }
    table = compare([covers], book, split="in_sample")
    assert table.column("unidentified_share").to_pylist() == [None, None, None]


def test_a_total_with_a_contract_without_a_report_reads_null() -> None:
    designs = _designs_of_each_asset()
    grid = [0.0, 4.0, 8.0]

    def squared(loss: Any, payout: Any, probability: Any) -> float:
        return float(np.sum(probability * (loss - payout) ** 2))

    designs["B"] = design(
        Cover(
            INDEX,
            StepPayout(
                [10.0, 20.0], [Free(0.0, 8.0, grid=grid), Free(0.0, 8.0, grid=grid)]
            ),
            payout_unit="EUR",
            asset_id="B",
            label="two",
        ),
        _book(*BOOK_DESIGN).asset("B"),
        UserObjective(squared, name="squared"),
    )
    table = compare([designs], _book(*BOOK_READ, start=2100), split="out_of_sample")
    assert _row(table, "asset", "A")["unidentified_share"] == 0.5
    assert _row(table, "asset", "B")["unidentified_share"] is None
    assert _row(table, "total")["unidentified_share"] is None


def _hazard_blocks() -> tuple[OutcomeSample, OutcomeSample]:
    return holdout(_hazards(RAIN_YEARS, HEAT_YEARS), train_fraction=SPLIT_AT)


def test_a_package_reads_each_contract_on_its_row_and_the_union_on_its_own() -> None:
    train, test = _hazard_blocks()
    package = CoverPackage(
        {
            "rain": design(
                _two_bands(RAIN, label="rain"), train.component("rain"), MSE()
            ),
            "heat": design(
                _two_bands(HEAT, label="heat"), train.component("heat"), MSE()
            ),
        },
        label="both",
    )
    table = compare([package], test, split="out_of_sample")
    rows = {row["cover_label"]: row for row in table.to_pylist()}
    assert rows["rain"]["unidentified_share"] == 0.5
    assert rows["heat"]["unidentified_share"] == 0.5
    assert rows["both"]["level"] == "package"
    assert rows["both"]["unidentified_share"] == 0.75


def test_a_package_of_covers_given_as_is_reads_null() -> None:
    train, _ = _hazard_blocks()
    package = CoverPackage(
        {
            "rain": Cover(RAIN, BinaryPayout(10.0, 2.0), payout_unit="EUR"),
            "heat": Cover(HEAT, BinaryPayout(10.0, 2.0), payout_unit="EUR"),
        },
        label="both",
    )
    table = compare([package], train, split="in_sample")
    assert table.column("unidentified_share").to_pylist() == [None, None, None]


def test_a_sum_of_components_names_its_amounts_by_component() -> None:
    train, test = _hazard_blocks()
    template = Cover(
        TWO,
        SumOf(
            {
                "rain": StepPayout([10.0, 20.0], [Free(), Free()]),
                "heat": StepPayout([10.0, 20.0], [Free(), Free()]),
            }
        ),
        payout_unit="EUR",
        label="sum",
    )
    result = design(template, train, MSE())
    assert result.unidentified == ("rain.levels[1]", "heat.levels[1]")
    assert _one(evaluate(result, train, split="in_sample"))["unidentified_share"] == 0.0
    table = evaluate(result, test, split="out_of_sample")
    assert _one(table)["unidentified_share"] == 0.75


def test_a_grid_of_two_components_reads_its_cells() -> None:
    train, test = _hazard_blocks()
    template = Cover(
        TWO,
        StepGrid(
            {"rain": [10.0, 20.0], "heat": [10.0, 20.0]},
            [
                [0.0, Free(), Free()],
                [Free(), Free(), Free()],
                [Free(), Free(), Free()],
            ],
            monotone=False,
        ),
        payout_unit="EUR",
        label="grid",
    )
    result = design(template, train, MSE())
    assert result.unidentified == (
        "levels[0,2]",
        "levels[1,2]",
        "levels[2,0]",
        "levels[2,1]",
        "levels[2,2]",
    )
    assert _one(evaluate(result, train, split="in_sample"))["unidentified_share"] == 0.0
    table = evaluate(result, test, split="out_of_sample")
    assert _one(table)["unidentified_share"] == 0.75


def test_a_scale_no_design_year_reaches_is_read_through_the_largest_payout() -> None:
    train = _hazards([5.0, 5.0, 8.0], [5.0, 7.0, 9.0])
    template = Cover(
        TWO,
        MaxOf(
            {
                "rain": StepPayout([10.0], [1.0]),
                "heat": StepPayout([10.0], [1.0]),
            },
            scale=Free(),
        ),
        payout_unit="EUR",
        label="max",
    )
    result = design(template, train, MSE())
    assert result.unidentified == ("scale",)
    assert _one(evaluate(result, train, split="in_sample"))["unidentified_share"] == 0.0
    block = _hazards([12.0, 5.0, 3.0, 12.0], [5.0, 12.0, 3.0, 12.0], start=2100)
    table = evaluate(result, block, split="out_of_sample")
    assert _one(table)["unidentified_share"] == 0.75


# -- a table without the columns -----------------------------------------------


def _written_before_the_columns(table: Any, tmp_path: Path) -> Any:
    """`table` as written before the two columns existed, read from its file."""
    key = EVALUATION_METADATA_KEY.encode()
    document = json.loads(table.schema.metadata[key])
    for name in ADDED:
        del document["columns"][name]
    older = table.drop_columns(list(ADDED)).replace_schema_metadata(
        {**table.schema.metadata, key: json.dumps(document).encode()}
    )
    path = tmp_path / "older.parquet"
    pq.write_table(older, path)
    return pq.read_table(path)


def test_a_table_written_before_the_columns_reads_as_null(tmp_path: Path) -> None:
    table = evaluate(_designed(), _design_block(), split="in_sample")
    older = _written_before_the_columns(table, tmp_path)
    assert not set(ADDED) & set(older.column_names)
    long = to_long(older)
    rows = [row for row in long.to_pylist() if row["column"] in ADDED]
    assert [row["column"] for row in rows] == list(ADDED)
    assert [row["value"] for row in rows] == [None, None]
    assert {row["unit"] for row in rows} == {"1"}
    assert {row["metric"] for row in rows} == set(ADDED)
    # The long form keeps the order of the columns, and the other rows.
    columns = list(dict.fromkeys(long.column("column").to_pylist()))
    assert columns[5:8] == ["payout_to_loss_ratio", *ADDED]
    assert long.num_rows == to_long(table).num_rows
    assert setup_text(older) == setup_text(table)


def test_a_table_without_the_columns_is_written_as_it_stands(tmp_path: Path) -> None:
    table = evaluate(_designed(), _design_block(), split="in_sample")
    older = _written_before_the_columns(table, tmp_path)
    path = tmp_path / "again.parquet"
    write_comparison(older, path)
    assert not set(ADDED) & set(pq.read_table(path).column_names)


def test_dropping_only_the_columns_reads_as_null_too() -> None:
    table = evaluate(_designed(), _design_block(), split="in_sample")
    older = table.drop_columns(list(ADDED))
    EvaluationMetadata.from_table(older).check_table(older)
    values = {row["column"]: row["value"] for row in to_long(older).to_pylist()}
    assert values["payout_frequency"] is None
    assert values["unidentified_share"] is None


def test_a_table_written_earlier_concatenates_with_a_new_one_as_nulls(
    tmp_path: Path,
) -> None:
    result = _designed()
    new = evaluate(result, _sample(REACHES, start=2100), split="out_of_sample")
    table = evaluate(result, _design_block(), split="in_sample")
    older = _written_before_the_columns(table, tmp_path)
    together = pa.concat_tables([older, new], promote_options="default")
    assert together.column("payout_frequency").to_pylist() == [None, 0.75]
    assert together.column("unidentified_share").to_pylist() == [None, 0.5]
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(older), EvaluationMetadata.from_table(new)
    )
    merged.check_table(together)


def test_another_missing_metric_is_still_refused() -> None:
    table = evaluate(_designed(), _design_block(), split="in_sample")
    with pytest.raises(EvaluationError, match="absent"):
        to_long(table.drop_columns(["rmse"]))


def test_a_new_column_the_metadata_does_not_describe_is_still_refused(
    tmp_path: Path,
) -> None:
    table = evaluate(_designed(), _design_block(), split="in_sample")
    older = _written_before_the_columns(table, tmp_path)
    claimed = older.append_column("payout_frequency", pa.array([0.5], pa.float64()))
    with pytest.raises(EvaluationError, match="undescribed"):
        to_long(claimed)


def test_a_breakdown_keeps_its_own_frequency_column() -> None:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    table = payout_breakdown([cover], _sample(REACHES), split="in_sample")
    assert "unidentified_share" not in table.column_names
    assert "unidentified_share" not in EvaluationMetadata.from_table(table).columns
    assert "unidentified_share" not in to_long(table).column("column").to_pylist()
    with pytest.raises(EvaluationError, match="absent"):
        to_long(table.drop_columns(["payout_frequency"]))


def test_the_hashes_do_not_change() -> None:
    # Each value was read from the code as it stood before the two columns. A
    # solved cover's hash is left out: it holds the solver's floating-point amounts.
    block = _design_block()
    assert block.content_hash == HASHES["sample"]
    assert _step_template().definition_hash == HASHES["blank"]
    result = _designed()
    assert result.design_id == HASHES["design"]
    row = _one(evaluate(result, block, split="in_sample"))
    assert (row["sample_hash"], row["design_id"]) == (
        HASHES["sample"],
        HASHES["design"],
    )
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    row = _one(evaluate(cover, block, split="in_sample"))
    assert row["cover_hash"] == cover.definition_hash == HASHES["given"]


HASHES = {
    "sample": "sha256:5eb02d58809c7274d7a6d73cdcd808a11ed482f2139a028e730958d8d422d992",
    "blank": "sha256:6c2decd269621b6a39d8b99466530baa98c6a913f3bdcd67c3b0ecdbb1688fdb",
    "design": "sha256:c18127c17fc29eaeedfa63cd162070aa7af2879b5099c192f5e66b9e3ce32bcd",
    "given": "sha256:e1777894aa14ea78f40882bae40778aeead889ccddf6d537dfc25d1a83789219",
}


# -- the setup text on fresh draws ---------------------------------------------

GUMBEL = FittedDistribution.from_parameters("gumbel_r", location=100.0, scale=30.0)
ANOTHER = FittedDistribution.from_parameters("gumbel_r", location=140.0, scale=30.0)
RAMP = PiecewiseLinearImpact([80.0, 200.0], [0.0, 1.0])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
GIVEN = Cover(INDEX, BinaryPayout(150.0, 1e5), payout_unit="EUR", label="given")


def _curve_metadata() -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="synthetic",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic fixture",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
    )


def _draws(seed: int, curve: Any = GUMBEL) -> OutcomeSample:
    return sample_curve(
        curve,
        metadata=_curve_metadata(),
        n_draws=40,
        rng=np.random.default_rng(seed),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
    )


def _split_lines(metadata: Any) -> list[str]:
    return [
        line for line in setup_text(metadata).splitlines() if line.startswith("Split")
    ]


def test_fresh_draws_say_what_they_test() -> None:
    table = evaluate(GIVEN, _draws(2), split="out_of_sample", design_sample=_draws(1))
    (line,) = _split_lines(table)
    assert "on fresh Monte Carlo draws" in line
    assert line.endswith(SAME_CURVE)


def test_a_holdout_of_draws_says_so_too() -> None:
    train, test = holdout(_draws(1))
    table = evaluate(GIVEN, test, split="out_of_sample", design_sample=train)
    (line,) = _split_lines(table)
    assert "a holdout with train fraction 0.7" in line
    assert line.endswith(SAME_CURVE)


def test_a_k_fold_of_draws_says_so_on_each_fold() -> None:
    (train, test), *_ = year_block_kfold(_draws(1), 4)
    table = evaluate(GIVEN, test, split="out_of_sample", design_sample=train)
    (line,) = _split_lines(table)
    assert "of a 4-fold split" in line
    assert line.endswith(SAME_CURVE)


def test_every_form_of_the_metadata_says_the_same() -> None:
    table = evaluate(GIVEN, _draws(2), split="out_of_sample", design_sample=_draws(1))
    metadata = EvaluationMetadata.from_table(table)
    expected = _split_lines(table)
    assert _split_lines(metadata) == expected
    assert _split_lines(json.loads(metadata.to_json())) == expected
    assert _split_lines(metadata.setup) == expected


def test_in_sample_rows_on_draws_do_not_say_so() -> None:
    table = evaluate(GIVEN, _draws(1), split="in_sample")
    assert SAME_CURVE not in setup_text(table)


def test_a_holdout_of_years_does_not_say_so() -> None:
    train, test = holdout(_sample(np.linspace(5.0, 40.0, 10)))
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    table = evaluate(cover, test, split="out_of_sample", design_sample=train)
    (line,) = _split_lines(table)
    assert "a holdout" in line
    assert SAME_CURVE not in line


def test_draws_from_another_curve_do_not_say_so() -> None:
    table = evaluate(
        GIVEN, _draws(2, ANOTHER), split="out_of_sample", design_sample=_draws(1)
    )
    (line,) = _split_lines(table)
    assert "on fresh Monte Carlo draws" in line
    assert SAME_CURVE not in line


def test_a_design_sample_of_another_source_does_not_say_so() -> None:
    cover = Cover(INDEX, BinaryPayout(15.0, 10.0), payout_unit="EUR", label="binary")
    table = evaluate(
        cover,
        _sample(np.linspace(5.0, 40.0, 10)),
        split="out_of_sample",
        design_sample=_sample(np.linspace(5.0, 40.0, 10), start=2100),
    )
    assert SAME_CURVE not in setup_text(table)


def test_a_design_sample_that_was_not_drawn_does_not_say_so() -> None:
    table = evaluate(
        GIVEN,
        _draws(2),
        split="out_of_sample",
        design_sample=_sample(np.linspace(5.0, 40.0, 10), start=2100),
    )
    (line,) = _split_lines(table)
    assert "on a separate sample from another source" in line
    assert SAME_CURVE not in line


def test_each_evaluation_says_it_for_its_own_blocks() -> None:
    draws = evaluate(GIVEN, _draws(2), split="out_of_sample", design_sample=_draws(1))
    in_sample = evaluate(GIVEN, _draws(3), split="in_sample")
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(draws),
        EvaluationMetadata.from_table(in_sample),
    )
    assert [SAME_CURVE in line for line in _split_lines(merged)] == [True, False]


def _setup_mapping(entries: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "basis": "annual",
        "metric_options": {},
        "samples": [],
        "evaluations": [
            {
                "fold": 0,
                "pricing": {"kind": "pure"},
                "split": "out_of_sample",
                **entries,
            }
        ],
    }


def test_a_setup_alone_of_fresh_draws_says_so_with_no_sample_to_read() -> None:
    assert SAME_CURVE in setup_text(_setup_mapping({"method": "fresh_draws"}))


def test_a_setup_alone_in_sample_does_not_say_so() -> None:
    setup = _setup_mapping({"method": "fresh_draws", "split": "in_sample"})
    assert SAME_CURVE not in setup_text(setup)


def test_a_setup_alone_of_an_unknown_kind_of_holdout_does_not_say_so() -> None:
    setup = _setup_mapping({"method": "holdout", "details": {}})
    assert SAME_CURVE not in setup_text(setup)
