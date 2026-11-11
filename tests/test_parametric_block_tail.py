"""The rank-block columns of a comparison, on synthetic samples, offline.

`copula_reference_years` and `copula_block_tail_b<beta>` say where a row's tail
lies inside the assumption of an `EmpiricalCopula`. The numbers are synthetic and
chosen so that (1 - beta) * m can be read off. Nothing here is a finding about a
cover, a dependence model or a design.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    BinaryPayout,
    CappedSum,
    ComonotoneIndexWarning,
    Cover,
    CoverPackage,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    SampleProvenance,
    StepPayout,
    WeightedCVaR,
    bootstrap_intervals,
    compare,
    compare_pooled,
    design_joint,
    evaluate,
    payout_breakdown,
    sample_annual_hazards,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.design import DEGENERATE_TAIL_TOLERANCE
from crc_sdk.parametric.joint import JointDesignResult
from crc_sdk.parametric.metadata import (
    _BLOCK_TAIL_TOLERANCE,
    IDENTITY_COLUMNS,
    _identity_columns,
)
from crc_sdk.parametric.sources import AnnualSeries
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

UNTYPED_INTERVALS: Any = bootstrap_intervals
YEARS = "copula_reference_years"
FLAGS = (
    "copula_block_tail_b0_9",
    "copula_block_tail_b0_95",
    "copula_block_tail_b0_99",
)
WIND = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")


def _copula(m: Any = 20, **fields: Any) -> dict[str, Any]:
    return {
        "content_hash": "sha256:" + "a" * 64,
        "first_year": 1990,
        "k": 1,
        "kind": "empirical_copula",
        "last_year": 2009,
        "licences": [],
        "m": m,
        "sources": [],
        **fields,
    }


def _assets(dependence: Mapping[str, Any], n: int = 20) -> OutcomeSample:
    """Two assets over `n` draws, under a stated dependence record."""
    index = np.column_stack([np.arange(n), np.arange(n)[::-1]]).astype(np.float64)
    return OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=index * 2.0,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
            dependence=dict(dependence),
        ),
    )


def _hazards(model: Mapping[str, Any]) -> OutcomeSample:
    """One asset with two hazards, under a stated dependence between them."""
    n = 12
    rain = np.arange(n, dtype=np.float64) * 10.0
    heat = rain[::-1] / 2.0
    return OutcomeSample(
        asset_ids=("site",),
        period=np.arange(2000, 2000 + n),
        probability=np.full(n, 1.0 / n),
        index=rain,
        loss=rain + heat,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=(),
            semantics_source="caller",
            hazard_dependence=dict(model),
            combine={"kind": "table"},
        ),
        index_components={
            "rain": IndexComponent(
                rain, unit="mm/day", tail="upper", hazard=None, own_loss=rain
            ),
            "heat": IndexComponent(
                heat, unit="degC", tail="upper", hazard=None, own_loss=heat
            ),
        },
    )


def _step(level: Any, asset: str | None = None) -> Cover:
    return Cover(WIND, StepPayout([5.0], [level]), payout_unit="EUR", asset_id=asset)


COVERS = {"A": _step(10.0, "A"), "B": _step(10.0, "B")}


def _package() -> CoverPackage:
    return CoverPackage(
        {
            "rain": Cover(
                RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR", label="rain"
            ),
            "heat": Cover(
                HEAT, BinaryPayout(30.0, 5.0), payout_unit="EUR", label="heat"
            ),
        }
    )


def _joint(sample: OutcomeSample, objective: Any) -> JointDesignResult:
    templates = [_step(Free(upper=100.0), asset) for asset in sample.asset_ids]
    return design_joint(templates, sample, objective)


def _by_level(table: Any) -> dict[tuple[str, str | None], dict[str, Any]]:
    return {(row["level"], row["asset_id"]): row for row in table.to_pylist()}


def _line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("Rank blocks:"))


def _no_line(text: str) -> bool:
    return "Rank blocks:" not in text


def _flag(beta: float, m: int) -> Any:
    table = evaluate(
        COVERS,
        _assets(_copula(m)),
        split="in_sample",
        options=MetricOptions(betas=(beta,)),
    )
    total = _by_level(table)[("total", None)]
    assert total[YEARS] == m
    return total[next(n for n in table.column_names if n.startswith("copula_block"))]


# -- the columns ---------------------------------------------------------------


def test_the_columns_sit_beside_the_identity_columns() -> None:
    table = evaluate(COVERS, _assets(_copula()), split="in_sample")
    start = len(IDENTITY_COLUMNS)
    assert tuple(table.column_names[:start]) == IDENTITY_COLUMNS
    assert table.column_names[start : start + 4] == [YEARS, *FLAGS]
    # The short-block flags follow, one per beta and return period, then metrics.
    assert table.column_names[start + 4] == "short_block_b0_9"
    assert table.column_names[start + 14] == "expected_loss"
    assert table.schema.field(YEARS).type == pa.int32()
    assert all(table.schema.field(name).type == pa.bool_() for name in FLAGS)
    assert all(table.schema.field(name).nullable for name in (YEARS, *FLAGS))


def test_a_label_follows_the_metrics_rule() -> None:
    table = evaluate(
        COVERS,
        _assets(_copula()),
        split="in_sample",
        options=MetricOptions(betas=(0.875,), return_periods=(), loss_thresholds=()),
    )
    assert "copula_block_tail_b0_875" in table.column_names
    assert "tail_coverage_b0_875" in table.column_names


def test_they_are_not_metrics() -> None:
    table = evaluate(COVERS, _assets(_copula()), split="in_sample")
    metadata = EvaluationMetadata.from_table(table)
    assert not {YEARS, *FLAGS} & set(metadata.columns)


# -- at the boundary -----------------------------------------------------------


@pytest.mark.parametrize(
    ("beta", "m"),
    [(0.95, 20), (0.9, 10), (0.99, 100), (0.75, 4), (0.5, 2)],
)
def test_the_boundary_is_flagged(beta: float, m: int) -> None:
    # (1 - 0.95) * 20 is 1.0000000000000009 in floating point.
    assert abs((1.0 - beta) * m - 1.0) < 1e-12
    assert _flag(beta, m) is True


@pytest.mark.parametrize(("beta", "m"), [(0.95, 19), (0.9, 9), (0.99, 99), (0.9, 1)])
def test_below_the_boundary_is_flagged(beta: float, m: int) -> None:
    assert (1.0 - beta) * m < 1.0
    assert _flag(beta, m) is True


@pytest.mark.parametrize(("beta", "m"), [(0.95, 21), (0.9, 11), (0.99, 101), (0.5, 3)])
def test_above_the_boundary_is_not_flagged(beta: float, m: int) -> None:
    assert (1.0 - beta) * m > 1.0
    assert _flag(beta, m) is False


@pytest.mark.parametrize(
    ("excess", "flagged"), [(5e-10, True), (0.0, True), (1e-8, False)]
)
def test_the_tolerance_is_absolute_on_the_product(excess: float, flagged: bool) -> None:
    assert _flag(1.0 - (1.0 + excess) / 20, 20) is flagged


def test_the_tolerance_is_the_one_of_degenerate_tail() -> None:
    assert _BLOCK_TAIL_TOLERANCE == DEGENERATE_TAIL_TOLERANCE


def test_each_beta_is_flagged_on_its_own() -> None:
    table = evaluate(COVERS, _assets(_copula(30)), split="in_sample")
    total = _by_level(table)[("total", None)]
    assert total[YEARS] == 30
    assert [total[name] for name in FLAGS] == [False, False, True]


# -- which rows ----------------------------------------------------------------


def test_a_set_of_covers_fills_its_total_row_only() -> None:
    rows = _by_level(evaluate(COVERS, _assets(_copula()), split="in_sample"))
    assert rows[("total", None)][YEARS] == 20
    assert [rows[("total", None)][name] for name in FLAGS] == [False, True, True]
    for key in (("asset", "A"), ("asset", "B")):
        assert all(rows[key][name] is None for name in (YEARS, *FLAGS))


@pytest.mark.parametrize(
    ("rule", "level"), [("value", "asset_by_value"), ("loss", "asset_by_loss")]
)
def test_a_joint_design_fills_its_total_and_allocated_rows(
    rule: Any, level: str
) -> None:
    sample = _assets(_copula())
    table = compare(
        [_joint(sample, ExpectedUncovered())],
        sample,
        split="in_sample",
        allocation=rule,
        values={"A": 300.0, "B": 100.0},
    )
    rows = _by_level(table)
    for key in (("total", None), (level, "A"), (level, "B")):
        assert rows[key][YEARS] == 20
        assert [rows[key][name] for name in FLAGS] == [False, True, True]


def test_every_row_of_a_sample_of_several_hazards_is_filled() -> None:
    table = evaluate(_package(), _hazards(_copula(12)), split="in_sample")
    rows = table.to_pylist()
    assert sorted(row["level"] for row in rows) == ["asset", "asset", "package"]
    assert all(row[YEARS] == 12 for row in rows)
    # (1 - beta) * 12 is 1.2, 0.6 and 0.12.
    assert all([row[name] for name in FLAGS] == [False, True, True] for row in rows)


def test_a_component_view_reads_null() -> None:
    view = _hazards(_copula(12)).component("rain")
    cover = Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR")
    row = evaluate(cover, view, split="in_sample").to_pylist()[0]
    assert all(row[name] is None for name in (YEARS, *FLAGS))


@pytest.mark.parametrize(
    "model",
    [
        {"kind": "independent"},
        {"kind": "comonotonic"},
        {"kind": "gaussian_copula", "correlation": 0.5},
        {"kind": "data"},
        {"kind": "a_user_model", "m": 20},
    ],
    ids=["independent", "comonotonic", "gaussian", "data", "user"],
)
def test_other_models_read_null(model: Mapping[str, Any]) -> None:
    table = evaluate(COVERS, _assets(model), split="in_sample")
    for row in table.to_pylist():
        assert all(row[name] is None for name in (YEARS, *FLAGS))
    assert _no_line(setup_text(table))
    hazards = evaluate(_package(), _hazards(model), split="in_sample")
    for row in hazards.to_pylist():
        assert all(row[name] is None for name in (YEARS, *FLAGS))


def test_one_asset_and_one_hazard_reads_null() -> None:
    sample = _assets(_copula()).asset("A")
    row = evaluate(_step(10.0), sample, split="in_sample").to_pylist()[0]
    assert all(row[name] is None for name in (YEARS, *FLAGS))


def test_a_record_without_m_reads_null() -> None:
    record = _copula()
    del record["m"]
    table = evaluate(COVERS, _assets(record), split="in_sample")
    assert _by_level(table)[("total", None)][YEARS] is None
    assert _no_line(setup_text(table))


@pytest.mark.parametrize("m", [0, -3, True, "20", 20.0])
def test_a_record_with_a_bad_m_raises(m: Any) -> None:
    with pytest.raises(EvaluationError, match="m as a positive whole number"):
        evaluate(COVERS, _assets(_copula(m)), split="in_sample")


# -- the real routes -----------------------------------------------------------


def _dataset(unit: str = "mm/day", name: str = "rain") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit=unit,
        value_semantics=f"annual maximum {name}",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset=name,
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
    )


CURVE = FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0)
RAMP = PiecewiseLinearImpact([20.0, 140.0], [0.0, 1.0])
LOSS = LossModel(RAMP, value=100.0, value_unit="EUR", name="ramp")


def _reference_of_assets(m: int) -> OutcomeSample:
    rng = np.random.default_rng(5)
    index = rng.gamma(2.0, 20.0, (m, 2))
    return OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(1990, 1990 + m, dtype=np.int64),
        probability=np.full(m, 1.0 / m),
        index=index,
        loss=index,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "hazard", "name": "series"},),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )


def test_a_sample_of_curves_states_its_reference_s_years() -> None:
    sample = sample_curves(
        {"A": CURVE, "B": CURVE},
        dependence=EmpiricalCopula(_reference_of_assets(20)),
        metadata={"A": _dataset(), "B": _dataset()},
        n_draws=60,
        rngs={"A": np.random.default_rng(1), "B": np.random.default_rng(2)},
        loss_models={"A": LOSS, "B": LOSS},
        basis_risk=NoBasisRisk(),
    )
    index = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
    covers = {
        asset: Cover(
            index, StepPayout([40.0], [10.0]), payout_unit="EUR", asset_id=asset
        )
        for asset in ("A", "B")
    }
    rows = _by_level(evaluate(covers, sample, split="in_sample"))
    # m is the reference's 20 years, not the sample's 60 draws.
    assert rows[("total", None)][YEARS] == 20
    assert [rows[("total", None)][name] for name in FLAGS] == [False, True, True]
    assert rows[("asset", "A")][YEARS] is None


def test_a_sample_of_curve_hazards_states_its_reference_s_years() -> None:
    m = 30
    rng = np.random.default_rng(5)
    series = {
        name: AnnualSeries(
            np.arange(1990, 1990 + m),
            rng.gamma(2.0, 20.0, m),
            unit,
            {"kind": "table", "source": {"name": f"era5 {name}"}},
        )
        for name, unit in (("rain", "mm/day"), ("heat", "degC"))
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        reference = sample_annual_hazards(
            series,
            tails=dict.fromkeys(series, "upper"),
            loss_models={"rain": LOSS},
            basis_risk=NoBasisRisk(),
            asset_id="site",
        )
    sample = sample_curve_hazards(
        {"rain": CURVE, "heat": CURVE},
        dependence=EmpiricalCopula(reference),
        metadata={"rain": _dataset(), "heat": _dataset("degC", "heat")},
        n_draws=60,
        rngs={"rain": np.random.default_rng(1), "heat": np.random.default_rng(2)},
        loss_models={"rain": LOSS, "heat": LOSS},
        basis_risk=NoBasisRisk(),
        combine=CappedSum(100.0),
        asset_id="site",
    )
    table = evaluate(_package(), sample, split="in_sample")
    assert {row[YEARS] for row in table.to_pylist()} == {30}
    # (1 - beta) * 30 is 3, 1.5 and 0.3.
    for row in table.to_pylist():
        assert [row[name] for name in FLAGS] == [False, False, True]


# -- pooled rows and intervals -------------------------------------------------


def test_a_pooled_table_has_the_columns_and_reads_null() -> None:
    # A k-fold split refuses rank blocks, so a pooled row has none.
    sample = _assets({"kind": "data"}, n=12)
    folds = year_block_kfold(sample, 3)
    table = compare_pooled([COVERS], folds)
    assert {YEARS, *FLAGS} <= set(table.column_names)
    for row in table.to_pylist():
        assert all(row[name] is None for name in (YEARS, *FLAGS))


def test_an_intervals_table_keeps_the_columns() -> None:
    table = UNTYPED_INTERVALS(
        [COVERS],
        _assets(_copula()),
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=5,
    )
    assert {YEARS, *FLAGS} <= set(table.column_names)
    rows = [r for r in table.to_pylist() if r["level"] == "total"]
    assert rows and all(row[YEARS] == 20 for row in rows)


def test_to_long_refuses_an_intervals_table_by_name() -> None:
    table = UNTYPED_INTERVALS(
        [COVERS],
        _assets(_copula()),
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=5,
    )
    with pytest.raises(EvaluationError, match="bootstrap_intervals"):
        to_long(table)


# -- to_long, files and earlier tables -----------------------------------------


def test_to_long_keeps_the_columns() -> None:
    table = evaluate(COVERS, _assets(_copula()), split="in_sample")
    long = to_long(table)
    identity = _identity_columns(table.column_names)
    assert long.column_names[: len(identity)] == identity
    assert long.num_rows == 3 * (table.num_columns - len(identity))
    assert YEARS not in set(long.column("column").to_pylist())
    totals = [row for row in long.to_pylist() if row["level"] == "total"]
    assert {row[YEARS] for row in totals} == {20}
    assert {row["copula_block_tail_b0_95"] for row in totals} == {True}
    assets = [row for row in long.to_pylist() if row["level"] == "asset"]
    assert {row[YEARS] for row in assets} == {None}


def test_a_table_written_earlier_reads_as_null(tmp_path: Path) -> None:
    table = evaluate(COVERS, _assets(_copula()), split="in_sample")
    older = table.drop_columns([YEARS, *FLAGS])
    path = tmp_path / "older.parquet"
    write_comparison(older, path)
    read = pq.read_table(path)
    assert YEARS not in read.column_names
    long = to_long(read)
    assert {YEARS, *FLAGS} <= set(long.column_names)
    assert set(long.column(YEARS).to_pylist()) == {None}
    assert set(long.column("copula_block_tail_b0_99").to_pylist()) == {None}
    assert long.schema.field(YEARS).type == pa.int32()
    assert long.schema.field("copula_block_tail_b0_99").type == pa.bool_()


def test_a_written_table_keeps_the_columns(tmp_path: Path) -> None:
    table = evaluate(COVERS, _assets(_copula()), split="in_sample")
    path = tmp_path / "comparison.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.column(YEARS).to_pylist() == table.column(YEARS).to_pylist()
    assert read.schema.field(FLAGS[0]).type == pa.bool_()


def test_a_payout_breakdown_has_no_such_columns() -> None:
    sample = _hazards(_copula(12))
    table = payout_breakdown([_package()], sample, split="in_sample")
    assert not {YEARS, *FLAGS} & set(table.column_names)
    long = to_long(table)
    assert not {YEARS, *FLAGS} & set(long.column_names)
    assert _no_line(setup_text(table))


# -- the setup text ------------------------------------------------------------


def test_the_text_states_m_and_each_beta() -> None:
    table = evaluate(COVERS, _assets(_copula()), split="in_sample")
    line = _line(setup_text(table))
    assert "the total rows" in line
    assert "(m = 20)" in line
    assert "Flagged betas of the metric options: 0.95, 0.99." in line
    assert "Not flagged: 0.9." in line
    assert "wrong" not in line
    # The asset rows alone read no sum, so they have no such line.
    assert _no_line(setup_text(table.filter(pc.equal(table["level"], "asset"))))


def test_the_text_says_none_when_nothing_is_flagged() -> None:
    table = evaluate(COVERS, _assets(_copula(200)), split="in_sample")
    line = _line(setup_text(table))
    assert "Flagged betas of the metric options: none." in line
    assert "Not flagged: 0.9, 0.95, 0.99." in line


def test_the_text_names_a_model_year_of_a_pool() -> None:
    record = _copula(pool={"rows": 20})
    line = _line(setup_text(evaluate(COVERS, _assets(record), split="in_sample")))
    assert "reference model-years" in line


def test_the_text_of_a_sample_of_several_hazards() -> None:
    table = evaluate(_package(), _hazards(_copula(12)), split="in_sample")
    line = _line(setup_text(table))
    assert "the asset and package rows" in line
    assert "(m = 12)" in line


@pytest.mark.parametrize(
    ("beta", "answer"), [(0.95, "flagged too"), (0.9, "not flagged")]
)
def test_the_text_says_whether_the_design_s_beta_is_flagged(
    beta: float, answer: str
) -> None:
    sample = _assets(_copula())
    result = _joint(sample, WeightedCVaR(1.0, beta))
    line = _line(setup_text(compare([result], sample, split="in_sample")))
    assert f"beta {beta:g}, {answer}" in line
    assert "Weighted-CVaR beta of each row's design:" in line


def test_a_design_on_another_objective_has_no_such_statement() -> None:
    sample = _assets(_copula())
    result = _joint(sample, ExpectedUncovered())
    line = _line(setup_text(compare([result], sample, split="in_sample")))
    assert "Weighted-CVaR" not in line


def test_a_setup_before_execution_states_it_too() -> None:
    setup = {
        "metric_options": {"betas": [0.9, 0.99]},
        "samples": [{"asset_ids": ["A", "B"], "dependence": _copula(20)}],
        "evaluations": [{"level": ["total"], "scope": ["individual"]}],
    }
    line = _line(setup_text(setup))
    assert "(m = 20)" in line
    assert "Flagged betas of the metric options: 0.99." in line
