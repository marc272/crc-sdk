"""Scenario labels, the columns that say what a row read, and windows of horizons.

Every number here is synthetic: small made-up curves, series and tables.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    PiecewiseLinearImpact,
    TabulatedDistribution,
)

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    CappedSum,
    ComonotoneIndexWarning,
    Comonotonic,
    Cover,
    CoverPackage,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    Free,
    GaussianCopula,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LinearPayout,
    LossModel,
    MetricOptions,
    ModelRun,
    NoBasisRisk,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    ScenarioError,
    StepPayout,
    bootstrap_intervals,
    compare,
    compare_pooled,
    design,
    design_joint,
    evaluate,
    holdout,
    payout_breakdown,
    sample_annual,
    sample_annual_hazards,
    sample_annual_pool,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import (
    _ADDED_SCENARIO_COLUMNS,
    IDENTITY_COLUMNS,
    KEY_COLUMNS,
)
from crc_sdk.parametric.sample import _source_columns
from crc_sdk.pathways import register_pathway
from crc_sdk.types import CurveParameters, HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import ReturnPeriodExtrapolationWarning

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.distributions.ReturnPeriodExtrapolationWarning"
    ),
]

UNTYPED_CURVE: Any = sample_curve
UNTYPED_CURVES: Any = sample_curves
UNTYPED_HAZARDS: Any = sample_curve_hazards
UNTYPED_ANNUAL_HAZARDS: Any = sample_annual_hazards
UNTYPED_TABLE: Any = OutcomeSample.from_table
UNTYPED_PROVENANCE: Any = SampleProvenance

GUMBEL = FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0)
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
NOISE = ImpactNoise(0.3)
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "mm/day", aggregation="annual_maximum")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10.0,))
HORIZONS = (2045, 2050, 2055)
SEPARATED = {2045: 10.0, 2050: 100.0, 2055: 1000.0}
"""Locations far enough apart that a value says which member it was read from."""


def _dataset(provider: str = "fixture", **fields: Any) -> HazardDatasetMetadata:
    base: dict[str, Any] = {
        "h3_resolution": 7,
        "value_unit": "mm/day",
        "value_semantics": "annual maximum daily rainfall",
        "producer": "tests",
        "source": SourceProvenance(
            provider=provider,
            dataset="rx1day",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic fixture",
        ),
        "creation_version": "1",
        "probability_semantics": "annual_exceedance",
        "return_period_support": (2.0, 200.0),
    }
    base.update(fields)
    return HazardDatasetMetadata(**base)


def _row(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell_index": 613196570331971583,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 50.0,
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    row.update(fields)
    return row


def _member(
    horizon: int, location: float | None = None, **fields: Any
) -> dict[str, Any]:
    spot = 50.0 + (horizon - 2045) if location is None else location
    return _row(horizon=horizon, curve_location=spot, **fields)


def _window(
    horizons: tuple[int, ...] = HORIZONS, **fields: Any
) -> list[dict[str, Any]]:
    return [_member(horizon, **fields) for horizon in horizons]


def _separated(asset_scale: float = 1.0) -> list[dict[str, Any]]:
    return [
        _member(horizon, location=location * asset_scale, curve_scale=0.01)
        for horizon, location in SEPARATED.items()
    ]


def _member_of(values: Any) -> Any:
    """Which of `_separated`'s members each value was read from."""
    return np.rint(np.log10(np.asarray(values)) - 1.0).astype(np.int64)


def _curve(curve: Any = None, **options: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "metadata": _dataset(),
        "n_draws": 300,
        "rng": np.random.default_rng(7),
        "loss_model": LOSS,
        "basis_risk": NOISE,
        "asset_id": "a",
        **options,
    }
    sample: OutcomeSample = UNTYPED_CURVE(
        _row() if curve is None else curve, **arguments
    )
    return sample


def _curves(curves: Any = None, **options: Any) -> OutcomeSample:
    chosen = {"a": _row(), "b": _row(curve_location=60.0)} if curves is None else curves
    arguments: dict[str, Any] = {
        "dependence": Independent(),
        "metadata": dict.fromkeys(chosen, _dataset()),
        "n_draws": 300,
        "rngs": {
            name: np.random.default_rng(seed) for seed, name in enumerate(chosen, 1)
        },
        "loss_models": dict.fromkeys(chosen, LOSS),
        "basis_risk": NOISE,
        **options,
    }
    sample: OutcomeSample = UNTYPED_CURVES(chosen, **arguments)
    return sample


def _hazard_curves(curves: Any = None, **options: Any) -> OutcomeSample:
    chosen = (
        {"rain": _row(hazard_name="rain"), "heat": _row(hazard_name="heat")}
        if curves is None
        else curves
    )
    arguments: dict[str, Any] = {
        "dependence": Independent(),
        "metadata": dict.fromkeys(chosen, _dataset()),
        "n_draws": 300,
        "rngs": {
            name: np.random.default_rng(seed) for seed, name in enumerate(chosen, 11)
        },
        "loss_models": {next(iter(chosen)): LOSS},
        "basis_risk": NOISE,
        "asset_id": "a",
        **options,
    }
    sample: OutcomeSample = UNTYPED_HAZARDS(chosen, **arguments)
    return sample


def _years(count: int = 40, seed: int = 3) -> Any:
    return np.random.default_rng(seed).gumbel(60.0, 12.0, count)


def _series(
    source: Any = "a table",
    *,
    first: int = 2000,
    count: int = 40,
    seed: int = 3,
    **provenance: Any,
) -> AnnualSeries:
    years = np.arange(first, first + count, dtype=np.int64)
    return AnnualSeries(
        years,
        _years(count, seed),
        "mm/day",
        {"kind": "table", "source": source, **provenance},
    )


def _annual(**options: Any) -> OutcomeSample:
    series = options.pop("series", None)
    sample: OutcomeSample = sample_annual(
        _series() if series is None else series,
        loss_model=options.pop("loss_model", LOSS),
        basis_risk=options.pop("basis_risk", NoBasisRisk()),
        asset_id=options.pop("asset_id", "a"),
        **options,
    )
    return sample


def _run(model: str, source: Any = "a table") -> ModelRun:
    return ModelRun(
        _series(source, seed=sum(model.encode("utf-8"))),
        model=model,
        member="r1i1p1f1",
        pathway="ssp245",
    )


def _pool(runs: Any = None, **options: Any) -> OutcomeSample:
    sample: OutcomeSample = sample_annual_pool(
        [_run("M1"), _run("M2")] if runs is None else runs,
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        **options,
    )
    return sample


def _table_data(n: int = 20, **columns: Any) -> dict[str, Any]:
    rng = np.random.default_rng(5)
    return {
        "year": list(range(2000, 2000 + n)),
        "site": ["a"] * n,
        "x": rng.gumbel(50.0, 8.0, n).tolist(),
        "loss": rng.gamma(2.0, 100.0, n).tolist(),
        **columns,
    }


def _table(data: Any = None, **options: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "period": "year",
        "asset_id": "site",
        "index": "x",
        "loss": "loss",
        "semantics": "annual",
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        **options,
    }
    sample: OutcomeSample = UNTYPED_TABLE(
        _table_data() if data is None else data, **arguments
    )
    return sample


def _components_table(**options: Any) -> OutcomeSample:
    rng = np.random.default_rng(9)
    data = _table_data(rain=rng.gumbel(50.0, 8.0, 20).tolist())
    data["heat"] = rng.gumbel(30.0, 5.0, 20).tolist()
    return _table(
        data,
        index={"rain": "rain", "heat": "heat"},
        index_unit={"rain": "mm/day", "heat": "mm/day"},
        tails={"rain": "upper", "heat": "upper"},
        **options,
    )


def _cover(**fields: Any) -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        label=fields.pop("label", "linear"),
        **fields,
    )


def _template(**fields: Any) -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 90.0], [Free(), Free()]),
        payout_unit="EUR",
        label="step",
        **fields,
    )


def _first_row(table: Any) -> dict[str, Any]:
    row: dict[str, Any] = table.to_pylist()[0]
    return row


def _read(sample: OutcomeSample) -> tuple[Any, Any, Any]:
    row = _first_row(compare([_cover()], sample, split="in_sample", options=OPTIONS))
    return row["pathway"], row["horizon"], row["product"]


# -- the label ------------------------------------------------------------------


def test_without_a_label_a_curve_sample_is_baseline() -> None:
    assert _curve().scenario == "baseline"
    assert _curves().scenario == "baseline"
    assert _table().scenario == "baseline"


def test_a_label_reaches_the_hash_and_the_lineage_leaves_it_out() -> None:
    plain, labelled = _curve(), _curve(scenario="ssp245 2050")
    assert labelled.scenario == "ssp245 2050"
    assert labelled.content_hash != plain.content_hash
    assert labelled.lineage == plain.lineage
    assert np.array_equal(labelled.index, plain.index)


def test_a_label_on_several_assets() -> None:
    plain, labelled = _curves(), _curves(scenario="x")
    assert labelled.scenario == "x"
    assert labelled.content_hash != plain.content_hash
    assert labelled.lineage == plain.lineage


def test_a_label_on_a_table_keeps_its_lineage_with_or_without_a_source() -> None:
    for source in (None, "my table"):
        plain, labelled = _table(source=source), _table(source=source, scenario="x")
        assert labelled.scenario == "x"
        assert labelled.content_hash != plain.content_hash
        assert labelled.lineage == plain.lineage


def test_a_label_on_a_table_of_several_hazards() -> None:
    sample = _components_table(scenario="x")
    assert sample.scenario == "x"
    assert sample.lineage == _components_table().lineage


@pytest.mark.parametrize("label", ["", " x", "x "])
def test_an_invalid_label_is_refused_before_any_draw(label: str) -> None:
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    with pytest.raises(SampleError, match="scenario"):
        _curve(scenario=label, rng=rng)
    assert rng.bit_generator.state == state


def test_a_table_refuses_an_invalid_label() -> None:
    with pytest.raises(SampleError, match="scenario"):
        _table(scenario=" x")


# -- what a row read ------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: _curve(), ("ssp245", 2050, "fixture")),
        (lambda: _curve(GUMBEL), (None, None, "fixture")),
        (
            lambda: _curve(
                CurveParameters(
                    curve_type="gumbel_r", curve_location=50.0, curve_scale=10.0
                )
            ),
            (None, None, "fixture"),
        ),
        (
            lambda: _curve(_row(pathway=None, horizon=None)),
            (None, None, "fixture"),
        ),
        (lambda: _curve(_window()), ("ssp245", 2050, "fixture")),
        (lambda: _annual(), (None, 2019, None)),
        (
            lambda: _annual(series=_series({"provider": "p", "pathway": "ssp585"})),
            ("ssp585", 2019, "p"),
        ),
        (lambda: _pool(), ("ssp245", 2019, None)),
        (
            lambda: _pool(
                [_run("M1", {"provider": "p"}), _run("M2", {"provider": "p"})]
            ),
            ("ssp245", 2019, "p"),
        ),
        (lambda: _table(), (None, None, None)),
        (
            lambda: _table(source={"provider": "p", "pathway": "ssp126"}),
            ("ssp126", None, "p"),
        ),
    ],
)
def test_each_source_states_its_pathway_horizon_and_product(
    build: Any, expected: tuple[Any, Any, Any]
) -> None:
    assert _read(build()) == expected


def test_a_series_reads_the_one_scenario_of_a_table_of_runs() -> None:
    columns = {
        name: {"values": [value], "distinct": 1}
        for name, value in (("model", "M"), ("member", "r"), ("scenario", "ssp370"))
    }
    series = _series({"provider": "p"}, table_columns=columns)
    assert _read(_annual(series=series)) == ("ssp370", 2019, "p")


def test_a_series_with_only_a_scenario_column_states_no_pathway() -> None:
    columns = {"scenario": {"values": ["ssp370"], "distinct": 1}}
    series = _series({"provider": "p"}, table_columns=columns)
    assert _read(_annual(series=series)) == (None, 2019, "p")


def test_a_record_that_cannot_be_read_states_no_pathway() -> None:
    series = _series({"pathway": ["not", "a", "label"]})
    assert _read(_annual(series=series)) == (None, 2019, None)


def test_a_series_horizon_is_the_centre_of_its_whole_window() -> None:
    series = _series(count=30, first=1985)
    assert _read(_annual(series=series))[1] == 1999
    design_block, held_out = holdout(_annual(series=series), 0.7)
    assert _read(design_block)[1] == _read(held_out)[1] == 1999


def test_the_columns_never_read_the_label() -> None:
    labelled = _curve(scenario="ssp370 2099")
    assert _read(labelled) == ("ssp245", 2050, "fixture")
    assert _read(_annual(scenario="ssp370 2099")) == (None, 2019, None)


def test_the_index_series_is_not_read() -> None:
    hazard = _series({"provider": "hazard side", "pathway": "ssp245"})
    index = _series({"provider": "index side", "pathway": "ssp585"}, seed=8)
    sample = _annual(series=hazard, index=index)
    assert _read(sample) == ("ssp245", 2019, "hazard side")


def test_several_assets_agree() -> None:
    assert _read_book(_curves()) == ("ssp245", 2050, "fixture")


def _read_book(sample: OutcomeSample) -> tuple[Any, Any, Any]:
    covers = {asset: _cover() for asset in sample.asset_ids}
    row = _first_row(compare([covers], sample, split="in_sample", options=OPTIONS))
    return row["pathway"], row["horizon"], row["product"]


def _mismatched_book(**options: Any) -> OutcomeSample:
    return _curves(
        {"a": _row(horizon=2050), "b": _row(horizon=2080, curve_location=60.0)},
        **options,
    )


def test_unlabelled_sources_that_disagree_read_as_null() -> None:
    assert _read_book(_mismatched_book()) == ("ssp245", None, "fixture")


def test_labelled_sources_that_disagree_raise_and_say_which() -> None:
    sample = _mismatched_book(scenario="x")
    covers = {asset: _cover() for asset in sample.asset_ids}
    with pytest.raises(ScenarioError, match=r"horizon.*asset 'a'.*asset 'b'"):
        compare([covers], sample, split="in_sample", options=OPTIONS)


def test_a_silent_source_beside_a_stating_one_disagrees() -> None:
    curves = {"a": _row(), "b": GUMBEL}
    assert _read_book(_curves(curves))[:2] == (None, None)
    sample = _curves(curves, scenario="x", hazard_name="rx1day")
    covers = {asset: _cover() for asset in sample.asset_ids}
    with pytest.raises(ScenarioError, match="pathway"):
        compare([covers], sample, split="in_sample", options=OPTIONS)


def test_equivalent_spellings_agree_and_keep_the_first() -> None:
    curves = {"a": _row(pathway="historic"), "b": _row(pathway="historical")}
    sample = _curves(curves, scenario="x")
    assert _read_book(sample)[0] == "historic"


def test_products_of_several_assets_must_agree() -> None:
    other = _dataset("other")
    metadata = {"a": _dataset(), "b": other}
    assert _read_book(_curves(metadata=metadata))[2] is None
    sample = _curves(metadata=metadata, scenario="x")
    covers = {asset: _cover() for asset in sample.asset_ids}
    with pytest.raises(ScenarioError, match="product"):
        compare([covers], sample, split="in_sample", options=OPTIONS)


def test_a_sample_of_several_hazards_joins_its_products() -> None:
    curves = {"rain": _row(hazard_name="rain"), "heat": _row(hazard_name="heat")}
    sample = _hazard_curves(
        curves, metadata={"rain": _dataset("zeta"), "heat": _dataset("alpha")}
    )
    assert _source_columns(sample, strict=True).product == "alpha+zeta"
    same = _hazard_curves(curves, metadata=dict.fromkeys(curves, _dataset("p")))
    assert _source_columns(same, strict=True).product == "p"


def test_the_components_of_a_sample_must_agree_on_pathway_and_horizon() -> None:
    curves = {
        "rain": _row(hazard_name="rain", horizon=2050),
        "heat": _row(hazard_name="heat", horizon=2080),
    }
    unlabelled = _hazard_curves(curves)
    assert _source_columns(unlabelled, strict=False).horizon is None
    with pytest.raises(ScenarioError, match=r"horizon.*component 'rain'"):
        _source_columns(_hazard_curves(curves, scenario="x"), strict=True)


def test_a_view_of_an_asset_or_a_component_reads_its_own_source() -> None:
    book = _curves({"a": _row(horizon=2050), "b": _row(horizon=2080)})
    assert _source_columns(book.asset("b"), strict=True).horizon == 2080
    curves = {
        "rain": _row(hazard_name="rain", horizon=2050),
        "heat": _row(hazard_name="heat", horizon=2080),
    }
    combined = _hazard_curves(
        curves, loss_models={"rain": LOSS, "heat": LOSS}, combine=CappedSum(1.0)
    )
    view = combined.component("heat")
    assert _source_columns(view, strict=True).horizon == 2080


def test_a_join_reads_every_part() -> None:
    joined = OutcomeSample.join(
        [
            _annual(series=_series({"provider": "p"}), asset_id="a"),
            _annual(series=_series({"provider": "p"}, seed=4), asset_id="b"),
        ]
    )
    assert _read_book(joined) == (None, 2019, "p")


def test_a_labelled_unknown_pathway_raises_and_says_how_to_add_one() -> None:
    sample = _curve(_row(pathway="made-up"), scenario="x")
    with pytest.raises(ScenarioError, match="register_pathway"):
        compare([_cover()], sample, split="in_sample", options=OPTIONS)


def test_a_private_pathway_can_be_registered() -> None:
    register_pathway("private-pathway-registered")
    sample = _curve(_row(pathway="private-pathway-registered"), scenario="x")
    assert _read(sample)[0] == "private-pathway-registered"


def test_an_unlabelled_unknown_pathway_stands_as_spelled() -> None:
    assert _read(_curve(_row(pathway="made-up")))[0] == "made-up"


def test_a_strict_read_refuses_an_unknown_pathway_on_any_label() -> None:
    with pytest.raises(ScenarioError, match="made-up"):
        _source_columns(_curve(_row(pathway="made-up")), strict=True)


def test_a_sample_without_records_reads_null() -> None:
    sample = OutcomeSample(
        asset_ids=("a",),
        period=np.arange(3),
        probability=np.full(3, 1.0 / 3.0),
        index=np.array([1.0, 2.0, 3.0]),
        loss=np.array([0.0, 1.0, 2.0]),
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table", sources=(), semantics_source="caller"
        ),
    )
    assert _read(sample) == (None, None, None)


# -- the table ------------------------------------------------------------------


def test_the_columns_come_in_their_places() -> None:
    assert IDENTITY_COLUMNS[6:11] == (
        "scenario",
        "design_scenario",
        "pathway",
        "horizon",
        "product",
    )
    assert IDENTITY_COLUMNS.index("n_rows") == IDENTITY_COLUMNS.index("sample_hash") + 1
    table = compare([_cover()], _curve(), split="in_sample", options=OPTIONS)
    assert tuple(table.column_names[: len(IDENTITY_COLUMNS)]) == IDENTITY_COLUMNS
    assert table.schema.field("design_scenario").type == pa.string()
    assert table.schema.field("pathway").type == pa.string()
    assert table.schema.field("horizon").type == pa.int32()
    assert table.schema.field("product").type == pa.string()
    assert table.schema.field("n_rows").type == pa.int32()


def test_n_rows_is_the_number_of_rows_of_the_block() -> None:
    train, test = holdout(_curve(n_draws=300), 0.7)
    row = _first_row(
        compare([_cover()], test, split="out_of_sample", design_sample=train)
    )
    assert row["n_rows"] == 90


def test_design_scenario_is_null_for_a_cover_given_as_is() -> None:
    row = _first_row(evaluate(_cover(), _curve(scenario="x"), split="in_sample"))
    assert row["scenario"] == "x" and row["design_scenario"] is None


def test_design_scenario_is_the_label_of_the_designs_own_sample() -> None:
    train, test = holdout(_annual(scenario="a label"), 0.7)
    designed = design(_template(), train, MSE())
    row = _first_row(compare([designed], test, split="out_of_sample"))
    assert (row["scenario"], row["design_scenario"]) == ("a label", "a label")


def _book_blocks(**options: Any) -> tuple[OutcomeSample, OutcomeSample]:
    book = OutcomeSample.join(
        [
            _annual(series=_series(seed=1), asset_id="a", **options),
            _annual(series=_series(seed=2), asset_id="b", **options),
        ]
    )
    return holdout(book, 0.7)


def test_design_scenario_of_a_set_of_designs_is_their_shared_label() -> None:
    train, test = _book_blocks(scenario="x")
    designs = {
        asset: design(_template(asset_id=asset), train.asset(asset), MSE())
        for asset in train.asset_ids
    }
    rows = compare([designs], test, split="out_of_sample").to_pylist()
    assert {row["level"] for row in rows} == {"asset", "total"}
    assert {row["design_scenario"] for row in rows} == {"x"}


def test_design_scenario_of_a_set_of_designs_on_different_labels_is_null() -> None:
    train, test = _book_blocks()
    other = dataclasses.replace(train.asset("b"), scenario="another label")
    designs = {
        "a": design(_template(asset_id="a"), train.asset("a"), MSE()),
        "b": design(_template(asset_id="b"), other, MSE()),
    }
    rows = compare([designs], test, split="out_of_sample").to_pylist()
    total = next(row for row in rows if row["level"] == "total")
    assert total["design_scenario"] is None
    asset_b = next(row for row in rows if row["asset_id"] == "b")
    assert asset_b["design_scenario"] == "another label"


def test_design_scenario_of_a_set_of_covers_given_as_is_is_null() -> None:
    train, test = _book_blocks()
    covers = {asset: _cover(asset_id=asset) for asset in train.asset_ids}
    rows = compare(
        [covers], test, split="out_of_sample", design_sample=train
    ).to_pylist()
    assert {row["design_scenario"] for row in rows} == {None}


def test_design_scenario_of_a_joint_design() -> None:
    train, test = _book_blocks(scenario="x")
    joint = design_joint([_template(), _template()], train, MSE())
    rows = compare(
        [joint],
        test,
        split="out_of_sample",
        allocation="value",
        values={"a": 1e6, "b": 2e6},
    ).to_pylist()
    assert {row["design_scenario"] for row in rows} == {"x"}


def _package_blocks() -> tuple[OutcomeSample, OutcomeSample]:
    sample = sample_annual_hazards(
        {"rain": _series(seed=3), "heat": _series(seed=4)},
        tails={"rain": "upper", "heat": "upper"},
        loss_models={"rain": LOSS, "heat": LOSS},
        combine=CappedSum(1.0),
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario="x",
    )
    return holdout(sample, 0.7)


def _rain_step(*thresholds: float) -> Cover:
    return Cover(
        RAIN,
        StepPayout(list(thresholds), [Free() for _ in thresholds]),
        payout_unit="EUR",
    )


def _heat_step(*thresholds: float) -> Cover:
    return Cover(
        HEAT,
        StepPayout(list(thresholds), [Free() for _ in thresholds]),
        payout_unit="EUR",
    )


def _package(train: OutcomeSample) -> CoverPackage:
    return CoverPackage(
        {
            "rain": design(_rain_step(60.0, 90.0), train.component("rain"), MSE()),
            "heat": design(_heat_step(60.0, 90.0), train.component("heat"), MSE()),
        }
    )


def test_design_scenario_of_a_package() -> None:
    train, test = _package_blocks()
    rows = compare([_package(train)], test, split="out_of_sample").to_pylist()
    assert {row["level"] for row in rows} == {"asset", "package"}
    assert {row["design_scenario"] for row in rows} == {"x"}


def test_a_breakdown_has_the_new_columns() -> None:
    train, test = _package_blocks()
    table = payout_breakdown([_package(train)], test, split="out_of_sample")
    rows = table.to_pylist()
    assert {row["design_scenario"] for row in rows} == {"x"}
    assert {row["n_rows"] for row in rows} == {len(test)}


def test_the_key_holds_design_scenario() -> None:
    assert KEY_COLUMNS.index("design_scenario") == KEY_COLUMNS.index("scenario") + 1


def test_rows_that_differ_only_in_design_scenario_have_two_keys() -> None:
    table = compare([_cover()], _curve(), split="in_sample", options=OPTIONS)
    metadata = EvaluationMetadata.from_table(table)
    other = table.set_column(
        table.schema.get_field_index("design_scenario"),
        table.schema.field("design_scenario"),
        pa.array(["x"], pa.string()),
    )
    two = pa.concat_tables([table, other])
    metadata.check_table(two)
    with pytest.raises(EvaluationError, match="repeats a row key"):
        metadata.check_table(pa.concat_tables([table, table]))


def _without_the_new_columns(table: Any) -> Any:
    return table.drop_columns(list(_ADDED_SCENARIO_COLUMNS))


def test_a_table_without_the_new_columns_reads_as_null(tmp_path: Path) -> None:
    table = compare([_cover()], _curve(), split="in_sample", options=OPTIONS)
    old = _without_the_new_columns(table)
    assert not set(_ADDED_SCENARIO_COLUMNS) & set(old.column_names)
    EvaluationMetadata.from_table(old).check_table(old)
    long = to_long(old)
    for name in _ADDED_SCENARIO_COLUMNS:
        assert long.column(name).null_count == long.num_rows
    assert long.schema.field("horizon").type == pa.int32()
    assert long.schema.field("n_rows").type == pa.int32()
    assert long.column_names[: len(IDENTITY_COLUMNS)] == list(IDENTITY_COLUMNS)
    path = tmp_path / "old.parquet"
    write_comparison(old, path)
    read = pq.read_table(path)
    assert not set(_ADDED_SCENARIO_COLUMNS) & set(read.column_names)


def test_a_table_with_the_new_columns_keeps_them_in_its_long_form() -> None:
    table = compare([_cover()], _curve(), split="in_sample", options=OPTIONS)
    long = to_long(table)
    assert set(long.column("pathway").to_pylist()) == {"ssp245"}
    assert set(long.column("horizon").to_pylist()) == {2050}
    assert set(long.column("product").to_pylist()) == {"fixture"}
    assert set(long.column("n_rows").to_pylist()) == {300}


def test_a_table_keeps_a_partial_set_of_new_columns() -> None:
    table = compare([_cover()], _curve(), split="in_sample", options=OPTIONS)
    old = table.drop_columns(["pathway", "n_rows"])
    long = to_long(old)
    assert long.column("pathway").null_count == long.num_rows
    assert set(long.column("product").to_pylist()) == {"fixture"}


def test_the_setup_entry_states_what_the_sample_read() -> None:
    table = compare([_cover()], _curve(), split="in_sample", options=OPTIONS)
    entry = EvaluationMetadata.from_table(table).setup.samples[0]
    assert (entry["pathway"], entry["horizon"], entry["product"]) == (
        "ssp245",
        2050,
        "fixture",
    )
    assert "members" not in entry


def test_a_window_setup_entry_holds_its_members() -> None:
    sample = _curve(_window())
    table = compare([_cover()], sample, split="in_sample", options=OPTIONS)
    entry = EvaluationMetadata.from_table(table).setup.samples[0]
    assert entry["members"] == [["ssp245", 2045], ["ssp245", 2050], ["ssp245", 2055]]
    assert entry["horizon"] == 2050


def test_a_sample_with_no_records_adds_no_keys_to_its_setup_entry() -> None:
    sample = OutcomeSample(
        asset_ids=("a",),
        period=np.arange(3),
        probability=np.full(3, 1.0 / 3.0),
        index=np.array([1.0, 2.0, 3.0]),
        loss=np.array([0.0, 1.0, 2.0]),
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table", sources=(), semantics_source="caller"
        ),
    )
    table = compare([_cover()], sample, split="in_sample", options=OPTIONS)
    entry = EvaluationMetadata.from_table(table).setup.samples[0]
    assert not {"pathway", "horizon", "product", "members"} & set(entry)


# -- pooled rows ----------------------------------------------------------------


def test_a_pooled_row_sums_n_rows_and_keeps_the_blocks_values() -> None:
    sample = _annual(
        series=_series({"provider": "p", "pathway": "ssp245"}, count=23), scenario="x"
    )
    folds = year_block_kfold(sample, 5)
    designs = [design(_template(), train, MSE()) for train, _ in folds]
    row = _first_row(compare_pooled([designs], folds, options=OPTIONS))
    assert row["n_rows"] == 23
    assert (row["pathway"], row["horizon"], row["product"]) == ("ssp245", 2011, "p")
    assert (row["scenario"], row["design_scenario"]) == ("x", "x")


def test_a_pooled_cover_given_as_is_has_no_design_scenario() -> None:
    folds = year_block_kfold(_annual(), 4)
    row = _first_row(compare_pooled([_cover()], folds, options=OPTIONS))
    assert row["design_scenario"] is None
    assert row["n_rows"] == 40


def test_the_intervals_table_carries_the_columns() -> None:
    folds = year_block_kfold(_annual(), 4)
    designs = [design(_template(), train, MSE()) for train, _ in folds]
    table = bootstrap_intervals(
        [designs], folds, rng=np.random.default_rng(1), n_bootstrap=5, options=OPTIONS
    )
    assert set(table.column("n_rows").to_pylist()) == {40}
    assert table.column("design_scenario").null_count == 0


def test_the_setup_text_reads_a_window() -> None:
    train, test = holdout(_curve(_window(), n_draws=300), 0.7)
    table = compare([_cover()], test, split="out_of_sample", design_sample=train)
    assert isinstance(setup_text(table), str)


# -- a window of horizons -------------------------------------------------------


def test_a_row_reads_its_member_at_its_uniform() -> None:
    rows = _window()
    sample = _curve(
        rows, n_draws=300, basis_risk=NoBasisRisk(), rng=np.random.default_rng(3)
    )
    uniforms = np.random.default_rng(3).random(300)
    expected = np.empty(300)
    for j, row in enumerate(rows):
        positions = np.arange(j, 300, 3)
        distribution = FittedDistribution.from_parameters(
            "gumbel_r", location=row["curve_location"], scale=10.0
        )
        expected[positions] = distribution.quantiles(uniforms[positions])
    assert np.array_equal(sample.index[:, 0], expected)
    assert np.array_equal(sample.period, np.arange(300))
    assert np.array_equal(sample.probability, np.full(300, 1.0 / 300))


def test_a_window_has_the_one_hazard_record_of_a_single_curve() -> None:
    window = _curve(_window(), rng=np.random.default_rng(9), n_draws=300)
    single = _curve(_row(), rng=np.random.default_rng(9), n_draws=300)
    assert [record["step"] for record in window.provenance.rng] == [
        "hazard",
        "basis_risk",
    ]
    assert window.provenance.rng == single.provenance.rng
    assert np.array_equal(
        window.provenance.generator(0).random(300),
        np.random.default_rng(9).random(300),
    )


def test_the_noise_of_a_window_is_drawn_after_the_uniforms() -> None:
    def factors(curve: Any) -> Any:
        noisy = _curve(curve, rng=np.random.default_rng(4), basis_risk=NOISE)
        clean = _curve(curve, rng=np.random.default_rng(4), basis_risk=NoBasisRisk())
        reached = clean.loss[:, 0] > 0
        return noisy.loss[:, 0][reached] / clean.loss[:, 0][reached]

    window, single = _window(location=500.0), _row(curve_location=500.0)
    assert np.allclose(factors(window), factors(single))


def test_the_order_of_the_members_does_not_matter() -> None:
    forward = _curve(_window())
    backward = _curve(list(reversed(_window())))
    shuffled = _curve([_member(2050), _member(2055), _member(2045)])
    assert forward.content_hash == backward.content_hash == shuffled.content_hash
    assert forward.lineage == backward.lineage == shuffled.lineage


def test_a_window_takes_tuples_and_keeps_the_horizon_order_of_its_record() -> None:
    sample = _curve(tuple(reversed(_window())))
    horizons = [source["row"]["horizon"] for source in sample.provenance.sources]
    assert horizons == list(HORIZONS)


def test_there_is_one_hazard_source_per_member() -> None:
    sample = _curve(_window())
    sources = sample.provenance.sources
    assert [source["role"] for source in sources] == ["hazard"] * 3
    assert [source["curve"]["location"] for source in sources] == [50.0, 55.0, 60.0]
    assert [source["kind"] for source in sources] == ["curve"] * 3


def test_details_hold_the_members_and_the_cycle() -> None:
    details = _curve(_window()).provenance.details
    assert details["window"]["assignment"] == "cycle"
    assert [list(pair) for pair in details["window"]["members"]] == [
        ["ssp245", 2045],
        ["ssp245", 2050],
        ["ssp245", 2055],
    ]


def test_details_count_per_member() -> None:
    details = _curve(_window()).provenance.details
    assert len(details["support"]) == 3
    assert len(details["outside_support"]) == 3
    assert len(details["atom"]) == 3
    assert len(details["tabulated"]) == 3
    assert all(set(each) == {"below", "above"} for each in details["outside_support"])
    assert details["draws"] == 300


def test_outside_support_counts_each_member_at_its_own_uniforms() -> None:
    sample = _curve(_window(), n_draws=900, rng=np.random.default_rng(2))
    uniforms = np.random.default_rng(2).random(900)
    counts = sample.provenance.details["outside_support"]
    rare = 1.0 - 1.0 / 200.0
    for j in range(3):
        drawn = uniforms[np.arange(j, 900, 3)]
        assert counts[j]["above"] == int((drawn > rare).sum())


def test_a_hurdle_member_counts_its_own_atom() -> None:
    rows = [
        _member(2045),
        _member(
            2050,
            curve_kind="hurdle",
            curve_type="gumbel_r",
            curve_atom_probability=0.5,
            curve_atom_location=0.0,
        ),
    ]
    details = _curve(rows, n_draws=200).provenance.details
    assert details["atom"][0] is None
    assert details["atom"][1]["probability"] == 0.5
    assert details["atom"][1]["draws"] > 0


def test_a_tabulated_member_raises_naming_its_horizon() -> None:
    rows = [
        _member(2045),
        _row(
            horizon=2050,
            curve_kind="tabulated",
            curve_type="linear_probability",
            curve_location=None,
            curve_scale=None,
            curve_probabilities=[0.4, 0.6],
            curve_values=[50.0, 70.0],
        ),
    ]
    with pytest.raises(SampleError, match="horizon 2050: .*outside the tabulated"):
        _curve(rows, n_draws=200)


def test_a_tabulated_member_records_its_choice() -> None:
    rows = [
        _member(2045),
        _row(
            horizon=2050,
            curve_kind="tabulated",
            curve_type="linear_probability",
            curve_location=None,
            curve_scale=None,
            curve_probabilities=[0.4, 0.6],
            curve_values=[50.0, 70.0],
        ),
    ]
    sample = _curve(rows, n_draws=200, outside_support="clamp")
    details = sample.provenance.details
    assert details["tabulated"][0] is None
    assert details["tabulated"][1]["outside_support"] == "clamp"
    assert any(note.startswith("Horizon 2050: ") for note in sample.provenance.notes)


def test_a_curve_without_support_says_so_once() -> None:
    dataset = _dataset(return_period_support=None)
    notes = _curve(_window(), metadata=dataset).provenance.notes
    no_support = [note for note in notes if "declares no fitted support" in note]
    assert len(no_support) == 1


def test_the_notes_of_a_window_name_the_member() -> None:
    sample = _curve(_window(), n_draws=30_000, rng=np.random.default_rng(6))
    assert any(
        note.startswith("Horizon 2045: ") and "rarer than" in note
        for note in sample.provenance.notes
    )


def test_a_window_warns_of_the_draws_rarer_than_its_support() -> None:
    with pytest.warns(ReturnPeriodExtrapolationWarning, match=r"of 3000 draws"):
        _curve(_window(), n_draws=3000, rng=np.random.default_rng(6))


def test_the_lineage_holds_every_member_and_leaves_the_label_out() -> None:
    base = _curve(_window())
    assert base.lineage != _curve(_row()).lineage
    changed = _curve([_member(2045), _member(2050), _member(2055, location=999.0)])
    assert changed.lineage != base.lineage
    assert _curve(_window(), scenario="x").lineage == base.lineage


def test_a_window_splits_into_blocks_of_every_member() -> None:
    sample = _curve(_window(), n_draws=30_000)
    train, test = holdout(sample, 0.7)
    assert np.bincount(train.period % 3).tolist() == [7000, 7000, 7000]
    assert np.bincount(test.period % 3).tolist() == [3000, 3000, 3000]
    assert train.lineage == test.lineage == sample.lineage


@pytest.mark.parametrize(
    ("horizons", "centre"),
    [
        ((2045, 2050, 2055), 2050),
        ((2045, 2050, 2055, 2060), 2052),
        ((2031, 2040), 2035),
    ],
)
def test_the_horizon_of_a_window_is_the_centre_of_its_members(
    horizons: tuple[int, ...], centre: int
) -> None:
    n = len(horizons) * 100
    assert _read(_curve(_window(horizons), n_draws=n))[1] == centre


def test_a_centre_of_a_blocks_window_is_the_same() -> None:
    train, test = holdout(_curve(_window()), 0.7)
    assert _read(train)[1] == _read(test)[1] == 2050


def test_a_window_round_trips_through_its_table() -> None:
    sample = _curve(_window(), scenario="x")
    back = OutcomeSample.from_written_table(sample.to_table())
    assert back.content_hash == sample.content_hash
    assert back.provenance.to_document() == sample.provenance.to_document()


def test_equivalent_spellings_form_one_window_and_keep_the_first() -> None:
    rows = [_member(2045, pathway="historic"), _member(2050, pathway="historical")]
    sample = _curve(rows, n_draws=200, scenario="x")
    assert _read(sample)[0] == "historic"
    assert list(sample.provenance.details["window"]["members"][1]) == [
        "historical",
        2050,
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "two canonical rows or more, got 0"),
        ([_member(2045)], "two canonical rows or more, got 1"),
        ([_member(2045), GUMBEL], "must be canonical rows"),
        ([_member(2045), "row"], "must be canonical rows"),
        (
            [
                _member(2045),
                CurveParameters(
                    curve_type="gumbel_r", curve_location=50.0, curve_scale=10.0
                ),
            ],
            "must be canonical rows",
        ),
        (
            [_member(2045), _member(2050, hazard_name="rx5day")],
            "several hazards",
        ),
        ([_member(2045), _member(2050, cell_index=99)], "several cells"),
        (
            [_member(2045, pathway="ssp245"), _member(2050, pathway="ssp585")],
            "one pathway",
        ),
        ([_member(2045), _member(2045)], "repeat"),
        ([_member(2045, pathway=None), _member(2050)], "state its pathway"),
        ([_member(2045, pathway="made-up"), _member(2050)], "register_pathway"),
        ([_row(horizon=None), _member(2050)], "state its horizon"),
        ([_row(horizon="2050"), _member(2055)], "state its horizon"),
        ([_row(horizon=True), _member(2055)], "state its horizon"),
    ],
)
def test_a_window_is_refused(rows: Any, message: str) -> None:
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    with pytest.raises(ScenarioError, match=message):
        _curve(rows, rng=rng, n_draws=300)
    assert rng.bit_generator.state == state


def test_a_window_of_rows_that_name_no_hazard_takes_a_stated_one() -> None:
    rows = [_member(2045, hazard_name=None), _member(2050, hazard_name=None)]
    sample = _curve(rows, n_draws=200, hazard_name="rx1day")
    assert sample.provenance.hazard == {"name": "rx1day", "from": "stated"}


@pytest.mark.parametrize(
    "curve",
    [iter([_row(), _row(horizon=2055)]), np.array([1.0, 2.0]), 3.0, "a curve"],
    ids=["iterator", "array", "number", "text"],
)
def test_anything_but_a_row_a_distribution_or_a_window_is_refused(curve: Any) -> None:
    with pytest.raises(TypeError, match="curve must be"):
        _curve(curve)


def test_a_members_source_id_may_differ_within_one_dataset() -> None:
    rows = [_member(2045, source_id="pixel-a"), _member(2050, source_id="pixel-b")]
    assert _curve(rows, n_draws=200).provenance.details["window"] is not None


def test_a_no_data_member_is_refused_as_any_no_data_curve() -> None:
    rows = [
        _member(2045),
        _row(
            horizon=2050,
            curve_kind="no_data",
            curve_type="insufficient_informative_support",
            curve_location=None,
            curve_scale=None,
        ),
    ]
    with pytest.raises(SampleError, match="no data"):
        _curve(rows, n_draws=200)


@pytest.mark.parametrize(
    ("n_draws", "members", "lower", "upper"),
    [(100, 3, 99, 102), (101, 3, 99, 102), (2, 3, 3, 6), (7, 2, 6, 8)],
)
def test_draws_that_do_not_divide_name_the_two_nearest_counts(
    n_draws: int, members: int, lower: int, upper: int
) -> None:
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    horizons = (2045, 2050, 2055)[:members]
    with pytest.raises(
        ScenarioError,
        match=rf"got {n_draws}; the nearest valid counts are {lower} and {upper}",
    ):
        _curve(_window(horizons), n_draws=n_draws, rng=rng)
    assert rng.bit_generator.state == state


def test_a_single_row_is_not_changed_by_any_of_this() -> None:
    single = _curve(_row())
    assert "window" not in single.provenance.details
    assert set(single.provenance.details["outside_support"]) == {"below", "above"}
    assert len(single.provenance.sources) == 1


# -- a window across assets and hazards -----------------------------------------


def test_a_window_on_several_assets() -> None:
    curves = {"a": _separated(), "b": _separated(2.0)}
    sample = _curves(curves, basis_risk=NoBasisRisk())
    assert sample.provenance.details["window"]["assignment"] == "cycle"
    assert len(sample.provenance.sources) == 6
    for j in range(2):
        assert np.array_equal(
            _member_of(sample.index[:, j] / (j + 1)), sample.period % 3
        )
    assert _read_book(sample) == ("ssp245", 2050, "fixture")


@pytest.mark.parametrize(
    "dependence",
    [Comonotonic(), GaussianCopula(0.7), Independent()],
    ids=["comonotonic", "gaussian", "independent"],
)
def test_a_dependence_model_reorders_within_each_member(dependence: Any) -> None:
    curves = {"a": _separated(), "b": _separated(2.0)}
    sample = _curves(
        curves, dependence=dependence, basis_risk=NoBasisRisk(), n_draws=600
    )
    assert np.array_equal(_member_of(sample.index[:, 0]), sample.period % 3)
    assert np.array_equal(_member_of(sample.index[:, 1] / 2.0), sample.period % 3)
    assert "each of the window's 3 members" in sample.provenance.notes[0]


def test_comonotonic_ranks_align_within_each_member() -> None:
    curves = {"a": _separated(), "b": _separated(2.0)}
    sample = _curves(curves, dependence=Comonotonic(), basis_risk=NoBasisRisk())
    for j in range(3):
        rows = sample.period % 3 == j
        a, b = sample.index[rows, 0], sample.index[rows, 1]
        assert np.array_equal(np.argsort(a), np.argsort(b))


def _reference_book(years: int = 10) -> OutcomeSample:
    return OutcomeSample.join(
        [
            _annual(series=_series(seed=1, count=years), asset_id="a"),
            _annual(series=_series(seed=2, count=years), asset_id="b"),
        ]
    )


def test_empirical_copula_on_a_window_needs_a_multiple_of_members_and_years() -> None:
    curves = {"a": _window(), "b": _window()}
    with pytest.raises(
        ScenarioError, match="whole multiple of 30, .*counts are 300 and 330"
    ):
        _curves(curves, dependence=EmpiricalCopula(_reference_book()), n_draws=303)
    sample = _curves(curves, dependence=EmpiricalCopula(_reference_book()), n_draws=300)
    assert sample.provenance.dependence is not None
    assert sample.provenance.dependence["k"] == 10


def test_a_window_needs_the_same_horizons_for_every_asset() -> None:
    rngs = {"a": np.random.default_rng(1), "b": np.random.default_rng(2)}
    states = {name: rng.bit_generator.state for name, rng in rngs.items()}
    with pytest.raises(ScenarioError, match="same"):
        _curves(
            {"a": _window((2045, 2050)), "b": _window((2050, 2055))},
            rngs=rngs,
            n_draws=200,
        )
    assert all(rngs[name].bit_generator.state == states[name] for name in rngs)


def test_a_window_needs_the_same_pathway_for_every_asset() -> None:
    with pytest.raises(ScenarioError, match="same"):
        _curves(
            {"a": _window(pathway="ssp245"), "b": _window(pathway="ssp585")},
            n_draws=300,
        )


def test_a_window_cannot_stand_beside_a_single_curve() -> None:
    with pytest.raises(ScenarioError, match="single curves"):
        _curves({"a": _window(), "b": _row()})


def test_the_draws_of_assets_with_a_window_must_divide() -> None:
    with pytest.raises(ScenarioError, match="99 and 102"):
        _curves({"a": _window(), "b": _window()}, n_draws=100)


def test_the_view_of_an_asset_holds_its_window() -> None:
    curves = {"a": _window(), "b": _window()}
    view = _curves(curves).asset("a")
    assert view.provenance.details["window"]["assignment"] == "cycle"
    assert _source_columns(view, strict=True).horizon == 2050


def _hazard_window(name: str, scale: float = 1.0) -> list[dict[str, Any]]:
    return [dict(row, hazard_name=name) for row in _separated(scale)]


def test_a_window_on_several_hazards() -> None:
    curves = {"rain": _hazard_window("rain"), "heat": _hazard_window("heat", 2.0)}
    sample = _hazard_curves(curves, dependence=Comonotonic(), basis_risk=NoBasisRisk())
    components = sample.index_components
    assert components is not None
    assert np.array_equal(
        _member_of(components["rain"].values[:, 0]), sample.period % 3
    )
    assert np.array_equal(
        _member_of(components["heat"].values[:, 0] / 2.0), sample.period % 3
    )
    assert sample.provenance.details["window"]["assignment"] == "cycle"
    assert _source_columns(sample, strict=True).horizon == 2050


def test_a_window_on_several_hazards_needs_every_member_for_every_hazard() -> None:
    curves = {
        "rain": [dict(row, hazard_name="rain") for row in _window((2045, 2050))],
        "heat": [dict(row, hazard_name="heat") for row in _window((2050, 2055))],
    }
    with pytest.raises(ScenarioError, match="same"):
        _hazard_curves(curves, n_draws=200)


# -- the hazard a sample reads --------------------------------------------------


def test_a_canonical_row_names_its_hazard() -> None:
    assert _curve().provenance.hazard == {"name": "rx1day", "from": "curve_row"}


def test_a_stated_name_agrees_with_a_row_and_does_not_override_it() -> None:
    sample = _curve(hazard_name="rx1day")
    assert sample.provenance.hazard == {"name": "rx1day", "from": "curve_row"}


def test_a_name_a_row_contradicts_raises_naming_the_scenario_and_both_names() -> None:
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    with pytest.raises(
        ScenarioError, match=r"scenario 'x'.*hazard_name='rx5day'.*'rx1day'"
    ):
        _curve(hazard_name="rx5day", scenario="x", rng=rng)
    assert rng.bit_generator.state == state


def test_a_distribution_names_no_hazard_unless_stated() -> None:
    assert _curve(GUMBEL).provenance.hazard is None
    stated = _curve(GUMBEL, hazard_name="rx1day")
    assert stated.provenance.hazard == {"name": "rx1day", "from": "stated"}


def test_a_window_names_its_hazard_once() -> None:
    assert _curve(_window()).provenance.hazard == {
        "name": "rx1day",
        "from": "curve_row",
    }
    with pytest.raises(ScenarioError, match="contradicts"):
        _curve(_window(), hazard_name="rx5day")


def test_several_assets_share_a_hazard_name() -> None:
    assert _curves().provenance.hazard == {"name": "rx1day", "from": "curve_row"}
    parts = _curves().provenance.parts
    assert parts is not None
    assert parts["a"]["hazard"] == {"name": "rx1day", "from": "curve_row"}


def test_several_assets_with_one_stated_name() -> None:
    sample = _curves({"a": GUMBEL, "b": GUMBEL}, hazard_name="rx1day")
    assert sample.provenance.hazard == {"name": "rx1day", "from": "stated"}


def test_a_row_and_a_stated_name_for_a_silent_source_say_stated() -> None:
    sample = _curves({"a": _row(), "b": GUMBEL}, hazard_name="rx1day")
    assert sample.provenance.hazard == {"name": "rx1day", "from": "stated"}


def test_assets_that_name_two_hazards_record_none_unless_labelled() -> None:
    curves = {"a": _row(hazard_name="rx1day"), "b": _row(hazard_name="rx5day")}
    assert _curves(curves).provenance.hazard is None
    with pytest.raises(ScenarioError, match="must read one hazard"):
        _curves(curves, scenario="x")


def test_several_assets_refuse_a_contradicted_name_before_any_draw() -> None:
    rngs = {"a": np.random.default_rng(1), "b": np.random.default_rng(2)}
    state = rngs["a"].bit_generator.state
    with pytest.raises(ScenarioError, match="contradicts"):
        _curves(
            {"a": GUMBEL, "b": _row()},
            rngs=rngs,
            hazard_name="rx5day",
        )
    assert rngs["a"].bit_generator.state == state


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("a table", None),
        (None, None),
        ({"provider": "p"}, None),
        ({"recipe": "rx1day"}, {"name": "rx1day", "from": "source_plan"}),
        ({"hazard_name": "rx1day"}, {"name": "rx1day", "from": "source_plan"}),
        (
            {"recipe": "rx1day", "hazard_name": "other"},
            {"name": "rx1day", "from": "source_plan"},
        ),
        ({"recipe": " "}, None),
    ],
)
def test_a_series_names_its_hazard_in_its_source(source: Any, expected: Any) -> None:
    assert _annual(series=_series(source)).provenance.hazard == expected


def test_a_series_reads_a_one_valued_recipe_column() -> None:
    columns = {"recipe": {"values": ["rx1day"], "distinct": 1}}
    sample = _annual(series=_series("a table", table_columns=columns))
    assert sample.provenance.hazard == {"name": "rx1day", "from": "table_column"}


def test_a_recipe_column_with_several_values_names_no_hazard() -> None:
    columns = {"recipe": {"values": ["a", "b"], "distinct": 2}}
    series = _series("a table", table_columns=columns)
    assert _annual(series=series).provenance.hazard is None
    stated = _annual(series=series, hazard_name="rx1day")
    assert stated.provenance.hazard == {"name": "rx1day", "from": "stated"}


def test_a_recipe_record_without_a_value_names_no_hazard() -> None:
    columns = {"recipe": {"distinct": 1}}
    assert _annual(series=_series(table_columns=columns)).provenance.hazard is None


def test_the_index_series_does_not_name_the_hazard() -> None:
    index = _series({"recipe": "other"}, seed=8)
    sample = _annual(series=_series(), index=index)
    assert sample.provenance.hazard is None


def test_a_stated_name_for_a_silent_series_is_the_callers() -> None:
    sample = _annual(hazard_name="rx1day")
    assert sample.provenance.hazard == {"name": "rx1day", "from": "stated"}


def test_a_series_refuses_a_contradicted_name() -> None:
    with pytest.raises(ScenarioError, match=r"scenario 'x'.*'b'.*'a'"):
        _annual(series=_series({"recipe": "a"}), hazard_name="b", scenario="x")


def test_a_stated_name_must_be_a_name() -> None:
    with pytest.raises(SampleError, match="hazard_name"):
        _annual(hazard_name=" ")


def test_a_pool_names_one_hazard() -> None:
    runs = [_run("M1", {"recipe": "rx1day"}), _run("M2", {"recipe": "rx1day"})]
    assert _pool(runs).provenance.hazard == {"name": "rx1day", "from": "source_plan"}
    assert _pool().provenance.hazard is None
    assert _pool(hazard_name="rx1day").provenance.hazard == {
        "name": "rx1day",
        "from": "stated",
    }


def test_a_pool_with_a_silent_run_records_none_and_takes_a_stated_name() -> None:
    runs = [_run("M1", {"recipe": "rx1day"}), _run("M2")]
    assert _pool(runs).provenance.hazard is None
    assert _pool(runs, hazard_name="rx1day").provenance.hazard == {
        "name": "rx1day",
        "from": "stated",
    }


def test_a_pool_refuses_a_name_a_run_contradicts() -> None:
    runs = [_run("M1", {"recipe": "rx1day"}), _run("M2")]
    with pytest.raises(ScenarioError, match="contradicts"):
        _pool(runs, hazard_name="rx5day")


def test_a_pool_of_two_hazard_names_records_none_unless_labelled() -> None:
    runs = [_run("M1", {"hazard_name": "a"}), _run("M2", {"hazard_name": "b"})]
    assert _pool(runs).provenance.hazard is None
    with pytest.raises(ScenarioError, match="must read one hazard"):
        _pool(runs, scenario="x")


@pytest.mark.parametrize(
    ("data", "options", "expected"),
    [
        (None, {}, None),
        (None, {"source": {"recipe": "r"}}, {"name": "r", "from": "source_plan"}),
        (None, {"hazard_name": "r"}, {"name": "r", "from": "stated"}),
        (
            _table_data(recipe=["r"] * 20),
            {},
            {"name": "r", "from": "table_column"},
        ),
        (_table_data(recipe=["r", "s"] * 10), {}, None),
        (
            _table_data(recipe=["r"] * 20),
            {"source": {"hazard_name": "q"}},
            {"name": "q", "from": "source_plan"},
        ),
        (_table_data(recipe=[None] * 20), {}, None),
    ],
)
def test_a_table_names_its_hazard(data: Any, options: Any, expected: Any) -> None:
    assert _table(data, **options).provenance.hazard == expected


def test_a_table_refuses_a_contradicted_name() -> None:
    with pytest.raises(ScenarioError, match="contradicts"):
        _table(source={"recipe": "r"}, hazard_name="s")


def test_a_table_of_several_assets_records_one_name() -> None:
    n = 10
    data = {
        "year": list(range(2000, 2000 + n)) * 2,
        "site": ["a"] * n + ["b"] * n,
        "x": list(range(n)) * 2,
        "loss": [1.0] * (2 * n),
    }
    sample = _table(data, hazard_name="r")
    assert sample.provenance.hazard == {"name": "r", "from": "stated"}
    assert sample.asset("a").provenance.hazard == {"name": "r", "from": "stated"}


def test_a_table_of_several_hazards_names_them_by_component() -> None:
    sample = _components_table(hazard_name={"rain": "rx1day"})
    assert sample.provenance.hazard is None
    components = sample.provenance.components
    assert components is not None
    assert components["rain"]["hazard"] == {"name": "rx1day", "from": "stated"}
    assert "hazard" not in components["heat"]


def test_a_table_of_several_hazards_refuses_one_name() -> None:
    with pytest.raises(SampleError, match="mapping by component"):
        _components_table(hazard_name="rx1day")


def test_a_table_of_one_hazard_refuses_a_mapping() -> None:
    with pytest.raises(SampleError, match="hazard_name is a mapping only"):
        _table(hazard_name={"rain": "rx1day"})


def test_the_curves_of_several_hazards_name_each_component() -> None:
    sample = _hazard_curves()
    assert sample.provenance.hazard is None
    components = sample.provenance.components
    assert components is not None
    assert components["rain"]["hazard"] == {"name": "rain", "from": "curve_row"}
    assert components["heat"]["hazard"] == {"name": "heat", "from": "curve_row"}


def test_a_component_name_is_not_a_hazard_name() -> None:
    sample = _hazard_curves({"rain": GUMBEL, "heat": GUMBEL})
    components = sample.provenance.components
    assert components is not None
    assert "hazard" not in components["rain"]


def test_hazard_names_state_a_silent_component() -> None:
    sample = _hazard_curves(
        {"rain": GUMBEL, "heat": GUMBEL}, hazard_names={"rain": "rx1day"}
    )
    components = sample.provenance.components
    assert components is not None
    assert components["rain"]["hazard"] == {"name": "rx1day", "from": "stated"}
    assert "hazard" not in components["heat"]


def test_hazard_names_refuse_a_contradiction_before_any_draw() -> None:
    rngs = {"rain": np.random.default_rng(1), "heat": np.random.default_rng(2)}
    state = rngs["rain"].bit_generator.state
    with pytest.raises(ScenarioError, match="contradicts"):
        _hazard_curves(
            {"rain": GUMBEL, "heat": _row(hazard_name="heat")},
            rngs=rngs,
            hazard_names={"heat": "other"},
        )
    assert rngs["rain"].bit_generator.state == state


def test_hazard_names_are_keyed_by_the_components() -> None:
    with pytest.raises(SampleError, match="hazard_names"):
        _hazard_curves(hazard_names={"snow": "x"})


def _annual_hazards(**options: Any) -> OutcomeSample:
    sample: OutcomeSample = UNTYPED_ANNUAL_HAZARDS(
        {
            "rain": options.pop("rain", _series({"recipe": "rx1day"}, seed=1)),
            "heat": options.pop("heat", _series(seed=2)),
        },
        tails={"rain": "upper", "heat": "upper"},
        loss_models={"rain": LOSS},
        basis_risk=NoBasisRisk(),
        asset_id="a",
        **options,
    )
    return sample


def test_per_year_hazards_name_each_component() -> None:
    sample = _annual_hazards(hazard_names={"heat": "txx"})
    assert sample.provenance.hazard is None
    components = sample.provenance.components
    assert components is not None
    assert components["rain"]["hazard"] == {"name": "rx1day", "from": "source_plan"}
    assert components["heat"]["hazard"] == {"name": "txx", "from": "stated"}


def test_a_component_view_keeps_its_hazard() -> None:
    sample = _annual_hazards()
    view = sample.component("rain")
    assert view.provenance.hazard == {"name": "rx1day", "from": "source_plan"}


def test_per_year_hazards_refuse_a_contradicted_name() -> None:
    with pytest.raises(ScenarioError, match="contradicts"):
        _annual_hazards(hazard_names={"rain": "other"})


def test_a_join_records_the_name_its_parts_share() -> None:
    joined = OutcomeSample.join(
        [
            _annual(series=_series({"recipe": "r"}, seed=1), asset_id="a"),
            _annual(series=_series({"recipe": "r"}, seed=2), asset_id="b"),
        ]
    )
    assert joined.provenance.hazard == {"name": "r", "from": "source_plan"}


def test_a_join_with_a_silent_part_records_none() -> None:
    joined = OutcomeSample.join(
        [
            _annual(series=_series({"recipe": "r"}, seed=1), asset_id="a"),
            _annual(series=_series(seed=2), asset_id="b"),
        ]
    )
    assert joined.provenance.hazard is None


def test_a_join_of_two_names_records_none_unless_labelled() -> None:
    def parts(**options: Any) -> list[OutcomeSample]:
        return [
            _annual(series=_series({"recipe": "r"}, seed=1), asset_id="a", **options),
            _annual(series=_series({"recipe": "s"}, seed=2), asset_id="b", **options),
        ]

    assert OutcomeSample.join(parts()).provenance.hazard is None
    with pytest.raises(ScenarioError, match="must read one hazard"):
        OutcomeSample.join(parts(scenario="x"))


def test_the_record_stays_out_of_hash_lineage_and_design_id() -> None:
    plain, named = _annual(), _annual(hazard_name="rx1day")
    assert plain.content_hash == named.content_hash
    assert plain.lineage == named.lineage
    plain_design = design(_template(), plain, MSE())
    named_design = design(_template(), named, MSE())
    assert plain_design.design_id == named_design.design_id


def test_a_split_keeps_the_record() -> None:
    train, test = holdout(_curve(), 0.7)
    assert (
        train.provenance.hazard == test.provenance.hazard == _curve().provenance.hazard
    )


def test_a_document_leaves_out_a_missing_record() -> None:
    assert "hazard" not in _curve(GUMBEL).provenance.to_document()
    assert _curve().provenance.to_document()["hazard"] == {
        "name": "rx1day",
        "from": "curve_row",
    }


def test_the_record_round_trips_through_a_written_table() -> None:
    sample = _curve()
    back = OutcomeSample.from_written_table(sample.to_table())
    assert back.provenance.hazard == sample.provenance.hazard
    metadata = json.loads(sample.to_table().schema.metadata[b"crc.parametric.sample"])
    assert metadata["provenance"]["hazard"]["name"] == "rx1day"


@pytest.mark.parametrize(
    "hazard",
    [
        {"name": "r"},
        {"name": " r", "from": "stated"},
        {"name": "r", "from": "elsewhere"},
        {"name": "r", "from": "stated", "extra": 1},
    ],
)
def test_the_provenance_refuses_an_invalid_record(hazard: Any) -> None:
    with pytest.raises(SampleError, match="hazard"):
        UNTYPED_PROVENANCE(
            source_kind="table", sources=(), semantics_source="caller", hazard=hazard
        )


def test_the_provenance_refuses_a_record_that_is_no_mapping() -> None:
    with pytest.raises(SampleError, match="hazard must be a mapping"):
        UNTYPED_PROVENANCE(
            source_kind="table", sources=(), semantics_source="caller", hazard="r"
        )


def test_builders_still_warn_about_a_comonotone_index() -> None:
    with pytest.warns(ComonotoneIndexWarning):
        _curve(_window(), basis_risk=NoBasisRisk())


def test_hurdle_and_tabulated_distributions_stand_as_single_curves() -> None:
    hurdle = HurdleDistribution(
        FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0),
        atom_probability=0.2,
        atom_location=0.0,
    )
    tabulated = TabulatedDistribution(
        np.array([0.0, 0.5, 1.0]), np.array([10.0, 50.0, 90.0])
    )
    for curve in (hurdle, tabulated):
        assert "window" not in _curve(curve).provenance.details
