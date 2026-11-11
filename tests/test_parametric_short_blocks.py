"""The short-block flags of a comparison's rows, and their setup text.

On a block of n rows, a tail metric at beta with (1 - beta) * n <= 1, or at a
return period T > n, reads the top of the block. Every row of `compare`,
`evaluate` and `compare_pooled` says so per level, by the rule of
`bootstrap_intervals`' `short_block`. The samples are synthetic; nothing here is
a finding about a cover.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    BinaryPayout,
    Budget,
    Cover,
    CoverPackage,
    EvaluationMetadata,
    Free,
    IndexComponent,
    IndexDefinition,
    JointDesignResult,
    LinearPayout,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    SampleProvenance,
    ScenarioSet,
    StepPayout,
    bootstrap_intervals,
    compare,
    compare_pooled,
    compare_scenarios,
    design_frontier,
    design_joint,
    evaluate,
    holdout,
    payout_breakdown,
    sample_annual,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import (
    IDENTITY_COLUMNS,
    _is_short_block_column,
    _reaches_worst_row,
)

INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
DEFAULT_FLAGS = (
    "short_block_b0_9",
    "short_block_b0_95",
    "short_block_b0_99",
    "short_block_rp2",
    "short_block_rp5",
    "short_block_rp10",
    "short_block_rp25",
    "short_block_rp50",
    "short_block_rp100",
    "short_block_rp250",
)


def _provenance(**fields: Any) -> SampleProvenance:
    base: dict[str, Any] = {
        "source_kind": "table",
        "sources": ({"role": "table", "name": "test table", "licence": "CC0-1.0"},),
        "semantics_source": "caller",
        "licences": ({"role": "table", "source": "test table", "licence": "CC0-1.0"},),
    }
    return SampleProvenance(**{**base, **fields})


def _sample(n: int, *, asset: str = "a", shift: float = 0.0) -> OutcomeSample:
    # Distinct losses, so the worst and the second worst row differ.
    years = np.arange(n, dtype=np.float64)
    index = 60.0 + shift + 40.0 * np.sin(years * 1.7) + years
    return OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(1800, 1800 + n),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=np.where(index > 60.0, (index - 60.0) * 10.0, 0.0) + years * 1e-3,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(),
    )


def _cover(*, asset: str = "a", label: str = "linear") -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 140.0, 800.0),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _book(n: int) -> OutcomeSample:
    return OutcomeSample.join([_sample(n, asset="a"), _sample(n, asset="b", shift=9)])


def _joint(train: OutcomeSample) -> JointDesignResult:
    templates = [
        Cover(INDEX, StepPayout([70.0], [Free()]), payout_unit="EUR", asset_id=asset)
        for asset in train.asset_ids
    ]
    return design_joint(templates, train, MSE())


def _flags(row: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in row.items() if _is_short_block_column(name)}


def _expected(n: int, options: MetricOptions = MetricOptions()) -> dict[str, bool]:
    return {
        **{
            f"short_block_b{str(beta).replace('.', '_')}": (1.0 - beta) * n <= 1 + 1e-9
            for beta in options.betas
        },
        **{
            f"short_block_rp{period:g}": period > n for period in options.return_periods
        },
    }


def _short_line(table: Any) -> str | None:
    lines = [line for line in setup_text(table).splitlines() if "Short blocks" in line]
    assert len(lines) <= 1
    return lines[0] if lines else None


# -- the rule ------------------------------------------------------------------


def test_the_rule_is_short_block_s() -> None:
    assert _reaches_worst_row({"beta": 0.95}, 20) is True  # (1 - beta) * n == 1
    assert _reaches_worst_row({"beta": 0.95}, 21) is False
    assert _reaches_worst_row({"beta": 0.9}, 10) is True
    assert _reaches_worst_row({"return_period": 20.0}, 20) is False
    assert _reaches_worst_row({"return_period": 21.0}, 20) is True
    assert _reaches_worst_row({"loss_threshold": 0.0}, 20) is None
    assert _reaches_worst_row({}, 20) is None


def test_a_flag_says_what_its_columns_read() -> None:
    # On 20 rows: beta 0.95 holds one row, so CVaR is the worst row and VaR the
    # second worst; T = 21 > 20 reads the largest value, T = 20 does not.
    options = MetricOptions(betas=(0.9, 0.95), return_periods=(20.0, 21.0))
    sample = _sample(20)
    row = compare([_cover()], sample, split="in_sample", options=options).to_pylist()[0]
    ordered = np.sort(sample.loss[:, 0])
    assert _flags(row) == {
        "short_block_b0_9": False,
        "short_block_b0_95": True,
        "short_block_rp20": False,
        "short_block_rp21": True,
    }
    assert row["cvar_loss_b0_95"] == pytest.approx(ordered[-1])
    assert row["var_loss_b0_95"] == ordered[-2]
    assert row["loss_rp21"] == ordered[-1]
    assert row["loss_rp20"] == ordered[-2]
    assert row["cvar_loss_b0_9"] != pytest.approx(ordered[-1])


# -- every row of a short block, none of a long one ---------------------------


@pytest.mark.parametrize("n", [12, 20, 60, 300])
def test_a_block_s_rows_carry_its_flags(n: int) -> None:
    row = evaluate(_cover(), _sample(n), split="in_sample").to_pylist()[0]
    assert _flags(row) == _expected(n)
    assert row["n_rows"] == n


def test_a_long_block_flags_nothing_and_its_text_says_nothing() -> None:
    table = compare([_cover()], _sample(300), split="in_sample")
    assert not any(_flags(table.to_pylist()[0]).values())
    assert _short_line(table) is None


def test_the_flags_sit_after_the_rank_block_columns() -> None:
    table = compare([_cover()], _sample(20), split="in_sample")
    names = table.column_names
    start = len(IDENTITY_COLUMNS) + 4  # copula_reference_years and three betas
    assert tuple(names[start : start + len(DEFAULT_FLAGS)]) == DEFAULT_FLAGS
    assert names[start + len(DEFAULT_FLAGS)] == "expected_loss"
    for name in DEFAULT_FLAGS:
        field = table.schema.field(name)
        assert field.type == pa.bool_() and field.nullable
    assert not set(DEFAULT_FLAGS) & set(EvaluationMetadata.from_table(table).columns)


def test_the_flags_follow_the_metrics_labels() -> None:
    options = MetricOptions(betas=(0.875,), return_periods=(12.5,), loss_thresholds=())
    table = compare([_cover()], _sample(10), split="in_sample", options=options)
    flags = [name for name in table.column_names if _is_short_block_column(name)]
    assert flags == ["short_block_b0_875", "short_block_rp12_5"]
    assert "tail_coverage_b0_875" in table.column_names
    assert "loss_rp12_5" in table.column_names
    assert _flags(table.to_pylist()[0]) == {
        "short_block_b0_875": False,
        "short_block_rp12_5": True,
    }


def test_every_row_of_several_assets_carries_the_flags() -> None:
    book = _book(15)
    covers = {asset: _cover(asset=asset) for asset in book.asset_ids}
    table = compare([covers], book, split="in_sample")
    joint = compare(
        [_joint(book)],
        book,
        split="in_sample",
        allocation="loss",
        values={"a": 1.0, "b": 2.0},
    )
    rows = [*table.to_pylist(), *joint.to_pylist()]
    assert {row["level"] for row in rows} == {"asset", "total", "asset_by_loss"}
    for row in rows:
        assert _flags(row) == _expected(15)


def test_a_package_s_rows_carry_the_flags() -> None:
    n = 12
    rain = np.linspace(10.0, 80.0, n)
    heat = np.linspace(45.0, 18.0, n)
    rain_loss = np.where(rain > 50.0, rain - 50.0, 0.0)
    heat_loss = np.where(heat > 40.0, heat - 40.0, 0.0)
    sample = OutcomeSample(
        asset_ids=("site",),
        period=np.arange(2000, 2000 + n),
        probability=np.full(n, 1.0 / n),
        index=rain,
        loss=rain_loss + heat_loss,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=(),
            semantics_source="caller",
            hazard_dependence={"kind": "data"},
            combine={"kind": "table"},
        ),
        index_components={
            "rain": IndexComponent(
                rain, unit="mm/day", tail="upper", hazard=None, own_loss=rain_loss
            ),
            "heat": IndexComponent(
                heat, unit="degC", tail="upper", hazard=None, own_loss=heat_loss
            ),
        },
    )
    package = CoverPackage(
        {
            "rain": Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR"),
            "heat": Cover(HEAT, BinaryPayout(38.0, 5.0), payout_unit="EUR"),
        }
    )
    rows = compare([package], sample, split="in_sample").to_pylist()
    assert [row["level"] for row in rows] == ["asset", "asset", "package"]
    for row in rows:
        assert _flags(row) == _expected(n)
    # A breakdown has no tail metric, so it has no flag.
    parts = payout_breakdown([package], sample, split="in_sample")
    assert not any(_is_short_block_column(name) for name in parts.column_names)


def test_a_pooled_row_reads_every_held_out_row() -> None:
    # Three folds of 10 years: each block is short at T = 25, the 30 pooled rows
    # are not; beta 0.95 reaches the worst row of 10 rows but not of 30.
    folds = year_block_kfold(_sample(30), 3)
    pooled = compare_pooled([_cover()], folds).to_pylist()
    (row,) = pooled
    assert row["n_rows"] == 30 and row["fold"] is None
    assert _flags(row) == _expected(30)
    assert row["short_block_rp25"] is False and row["short_block_b0_95"] is False
    _, test = folds[0]
    one = compare([_cover()], test, split="out_of_sample", design_sample=folds[0][0])
    assert _flags(one.to_pylist()[0]) == _expected(10)
    assert one.to_pylist()[0]["short_block_rp25"] is True
    line = _short_line(compare_pooled([_cover()], folds))
    assert line is not None and "on 30 rows, beta 0.99 and return periods 50" in line


def test_the_flags_agree_with_the_intervals_short_block() -> None:
    sample = _sample(20)
    table = compare([_cover()], sample, split="in_sample")
    row = table.to_pylist()[0]
    intervals = bootstrap_intervals(
        [_cover()],
        sample,
        split="in_sample",
        rng=np.random.default_rng(3),
        n_bootstrap=10,
    )
    long = to_long(table)
    for item in intervals.to_pylist():
        parameter = item["parameter"]
        if parameter not in ("beta", "return_period"):
            assert item["short_block"] is None
            continue
        label = format(item["parameter_value"], "g").replace(".", "_")
        name = f"short_block_{'b' if parameter == 'beta' else 'rp'}{label}"
        assert item["short_block"] is row[name], item["column"]
        # The intervals table carries the wide flags too, from to_long.
        assert item[name] is row[name]
    assert set(DEFAULT_FLAGS) <= set(long.column_names)


def test_a_frontier_s_rows_carry_the_flags() -> None:
    pytest.importorskip("highspy")
    sample = _sample(20)
    train, test = holdout(sample, 0.5)
    template = Cover(
        INDEX, StepPayout([70.0], [Free()]), payout_unit="EUR", asset_id="a"
    )
    table = design_frontier(
        template,
        train,
        MSE(),
        budgets=[Budget(ratio=0.5), Budget(ratio=1.0)],
        evaluate_on=test,
        split="out_of_sample",
    )
    for row in table.to_pylist():
        assert _flags(row) == _expected(10)
    names = table.column_names
    assert names.index("short_block_rp250") < names.index("budget_ratio")


def _years(label: str, first: int) -> OutcomeSample:
    years = np.arange(first, first + 40, dtype=np.int64)
    series = AnnualSeries(
        years,
        np.random.default_rng(first).gumbel(60.0, 12.0, 40),
        "mm/day",
        {"kind": "table", "source": {"provider": "fixture", "recipe": "rx1day"}},
    )
    return sample_annual(
        series,
        loss_model=LossModel(
            PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
            value=1e6,
            value_unit="EUR",
        ),
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario=label,
    )


@pytest.mark.filterwarnings("ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning")
def test_rows_across_scenarios_carry_the_flags_and_one_short_blocks_line() -> None:
    train, test = holdout(_years("early", 2000))
    _, late = holdout(_years("late", 2060))
    table = compare_scenarios(
        [_cover()], ScenarioSet([test, late], reference="early"), design_sample=train
    )
    for row in table.to_pylist():
        assert _flags(row) == _expected(12)
    lines = [line for line in setup_text(table).splitlines() if "Short blocks" in line]
    # Across scenarios the line of compare_scenarios names each block's columns.
    assert len(lines) == 1
    assert lines[0].startswith("Short blocks: on the 12 rows of early and late")


# -- the long form, files and older tables ------------------------------------


def test_to_long_keeps_the_flags_as_identity_columns() -> None:
    table = compare([_cover()], _sample(12), split="in_sample")
    long = to_long(table)
    assert set(DEFAULT_FLAGS) <= set(long.column_names)
    assert "short_block_b0_99" not in set(long.column("column").to_pylist())
    rows = long.to_pylist()
    assert all(row["short_block_rp250"] is True for row in rows)


def test_a_table_written_before_the_flags_reads_null(tmp_path: Path) -> None:
    table = compare([_cover()], _sample(12), split="in_sample")
    older = table.drop_columns(list(DEFAULT_FLAGS))
    EvaluationMetadata.from_table(older).check_table(older)
    long = to_long(older)
    for name in DEFAULT_FLAGS:
        assert set(long.column(name).to_pylist()) == {None}
    # The text reads the metadata, so it states the same short blocks.
    assert setup_text(older) == setup_text(table)
    write_comparison(older, tmp_path / "older.parquet")


def test_the_flags_round_trip_through_a_file(tmp_path: Path) -> None:
    table = compare([_cover()], _sample(12), split="in_sample")
    write_comparison(table, tmp_path / "comparison.parquet")
    read = pq.read_table(tmp_path / "comparison.parquet")
    assert read.to_pylist() == table.to_pylist()
    assert setup_text(read) == setup_text(table)


# -- the setup text ------------------------------------------------------------


def test_the_setup_text_names_each_block_s_flagged_levels() -> None:
    sample = _sample(40)
    train, test = holdout(sample, 0.7)
    rows = [
        compare([_cover()], train, split="in_sample"),
        compare([_cover()], test, split="out_of_sample", design_sample=train),
    ]
    table = pa.concat_tables(rows)
    merged = EvaluationMetadata.merge(*map(EvaluationMetadata.from_table, rows))
    line = _short_line(merged)
    assert line is not None
    assert line.startswith(
        "Short blocks: these tail levels reach the worst row of their block, as the "
        "rows' short_block_b<beta> and short_block_rp<T> columns flag: on 28 rows, "
        "beta 0.99 and return periods 50, 100 and 250 years; on 12 rows, beta 0.95 "
        "and 0.99 and return periods 25, 50, 100 and 250 years."
    )
    assert "At such a beta, (1 - beta) * n <= 1" in line
    assert "At such a return period, T > n" in line
    # The rows of one block state that block's levels only.
    held_out = table.slice(1, 1).replace_schema_metadata(
        {b"crc.parametric.evaluation": merged.to_json().encode("utf-8")}
    )
    alone = _short_line(held_out)
    assert alone is not None and "on 12 rows" in alone and "on 28 rows" not in alone


def test_the_setup_text_explains_only_the_kinds_it_flags() -> None:
    options = MetricOptions(betas=(0.5,), return_periods=(100.0,))
    line = _short_line(
        compare([_cover()], _sample(40), split="in_sample", options=options)
    )
    assert line is not None
    assert "on 40 rows, return periods 100 years." in line
    assert "At such a beta" not in line
    options = MetricOptions(betas=(0.99,), return_periods=(2.0,))
    line = _short_line(
        compare([_cover()], _sample(40), split="in_sample", options=options)
    )
    assert line is not None
    assert "on 40 rows, beta 0.99." in line
    assert "At such a return period" not in line
