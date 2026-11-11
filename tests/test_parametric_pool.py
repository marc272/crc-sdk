"""Pooled model-years: `ModelRun`, `sample_annual_pool`, `ModelPool` and `pool=`."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact
from crc_framework import impacts as framework_impacts
from crc_framework.distributions import FittedDistribution

from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AsRecorded,
    Budget,
    ComonotoneIndexWarning,
    Comonotonic,
    Cover,
    EmpiricalCopula,
    EvaluationMetadata,
    Free,
    FreshDraws,
    H3Cell,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    Individual,
    Joint,
    KFold,
    LeakageError,
    LinearPayout,
    LinearTrend,
    LossModel,
    MetricOptions,
    ModelPool,
    ModelRun,
    NearestCell,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    RecentYears,
    SampleError,
    SampleProvenance,
    ScenarioError,
    StationColumn,
    StepPayout,
    evaluate,
    holdout,
    sample_annual,
    sample_annual_pool,
    sample_curves,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY
from crc_sdk.parametric.sample import SAMPLE_METADATA_KEY
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = pytest.mark.filterwarnings(
    "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
)

UNTYPED_POOL: Any = sample_annual_pool
UNTYPED_RUN: Any = ModelRun
UNTYPED_PLAN: Any = ParametricPlan
YEARS = np.arange(2015, 2045)
MODELS = ("GFDL-ESM4", "MIROC6", "NorESM2-LM")
MEMBER = "r1i1p1f1"
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
LINEAR = Cover(INDEX, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR", label="linear")
STEP = Cover(
    INDEX,
    StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
    payout_unit="EUR",
    label="step",
)
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
ARCHIVE = SourceProvenance(
    provider="example",
    dataset="synthetic NEX-shaped rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic annual maxima",
)
CELLS = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
SITES = {"a": (6.96, 50.94), "b": (7.10, 51.10)}
STATEMENTS = (
    "A draw mixes model spread with year-to-year variability.",
    "The rows are not independent years: rows of one run share that model's "
    "climate and rows of one year the pathway's forcing, so the number of rows is "
    "not an effective sample size.",
)


def _values(model: str, years: Any = YEARS, shift: float = 0.0) -> Any:
    rng = np.random.default_rng(sum(model.encode("utf-8")))
    return rng.gumbel(60.0 + shift, 12.0, len(years))


def _series(
    model: str,
    *,
    years: Any = YEARS,
    values: Any = None,
    unit: str = "mm/day",
    **provenance: Any,
) -> AnnualSeries:
    return AnnualSeries(
        np.asarray(years),
        _values(model, years) if values is None else values,
        unit,
        {"source": {"name": f"synthetic {model}"}, **provenance},
    )


def _run(
    model: str, *, member: str = MEMBER, pathway: str = "ssp245", **series: Any
) -> ModelRun:
    return ModelRun(
        _series(model, **series), model=model, member=member, pathway=pathway
    )


def _pool(runs: Any = None, **options: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "loss_model": LOSS,
        "basis_risk": NoBasisRisk(),
        "asset_id": "site-a",
    }
    arguments.update(options)
    sample: OutcomeSample = UNTYPED_POOL(
        [_run(model) for model in MODELS] if runs is None else runs, **arguments
    )
    return sample


def _document(sample: OutcomeSample) -> dict[str, Any]:
    pool: dict[str, Any] = sample.provenance.to_document()["pool"]
    return pool


def _lines(text: str, label: str) -> list[str]:
    return [line for line in text.splitlines() if line.split(":")[0].startswith(label)]


def _table_of(
    models: Sequence[str] = MODELS[:2],
    *,
    years: Any = YEARS,
    cells: Sequence[tuple[float, float]] = CELLS,
    scenario: str = "ssp245",
    member: str = MEMBER,
    recipe: str = "rx1day",
    held_out_shift: float = 0.0,
) -> Any:
    """The layout of NEX-GDDP-CMIP6's `annual_extremes()` batches, in one table."""
    tables: list[Any] = []
    for model in models:
        count = len(years) * len(cells)
        rng = np.random.default_rng(sum(model.encode("utf-8")))
        cell_years = np.repeat(years, len(cells))
        tables.append(
            pa.table(
                {
                    "year": pa.array(cell_years, pa.int32()),
                    "latitude": np.tile([lat for lat, _ in cells], len(years)),
                    "longitude": np.tile([lon for _, lon in cells], len(years)),
                    "value": rng.gumbel(60.0, 12.0, count)
                    + held_out_shift * (cell_years >= 2037),
                    "unit": ["mm/day"] * count,
                    "recipe": [recipe] * count,
                    "model": [model] * count,
                    "scenario": [scenario] * count,
                    "member": [member] * count,
                    "retrieved_at": [f"2026-11-08T00:00:0{len(tables)}Z"] * count,
                    "archive_licence": ["CC0-1.0"] * count,
                    "attribution": ["NASA NEX-GDDP-CMIP6"] * count,
                }
            )
        )
    return pa.concat_tables(tables)


def _assets(ids: Sequence[str] = ("a", "b")) -> Any:
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [SITES[i][0] for i in ids],
            "latitude": [SITES[i][1] for i in ids],
            "value_eur": [2e6 for _ in ids],
        }
    )


def _planned(source: Any = None, *, assets: Any = None, **options: Any) -> Any:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "provenance": ARCHIVE,
        "value_column": "value_eur",
        "value_unit": "EUR",
        "seed": 42,
        "pool": ModelPool(),
    }
    arguments.update(options)
    return UNTYPED_PLAN.for_annual_extremes(
        _table_of() if source is None else source,
        _assets() if assets is None else assets,
        **arguments,
    )


def _resolved(plan: Any) -> Any:
    return plan.basis_risk(ImpactNoise(0.3, upper=1.0)).covers(LINEAR).metrics(OPTIONS)


@dataclass
class _NexPlan:
    """A plan whose `annual_extremes()` yields batches, as NEX-GDDP-CMIP6's does."""

    table: Any

    def annual_extremes(self) -> Iterator[Any]:
        yield from self.table.to_batches(max_chunksize=7)


@dataclass
class _Cmip6Plan:
    """A CMIP6-shaped plan: its table names no run, and its `explain()` does."""

    model: str
    table: Any = field(default=None)
    member: str = MEMBER
    pathway: str = "ssp245"
    recipe: str = "rx1day"
    reads: list[int] = field(default_factory=list)

    def annual_extremes(self) -> Any:
        self.reads.append(1)
        if self.table is not None:
            return self.table
        return _table_of([self.model], member=self.member).select(
            ["year", "latitude", "longitude", "value"]
        )

    def explain(self, *, format: str = "text") -> dict[str, Any]:
        del format
        return {
            "model": self.model,
            "member": self.member,
            "pathway": self.pathway,
            "recipe": self.recipe,
            "unit": "mm/day",
        }


# -- the rows ------------------------------------------------------------------


def test_three_models_of_n_years_give_3n_rows_at_1_over_3n() -> None:
    sample = _pool()
    assert len(sample) == 3 * len(YEARS) == 90
    assert np.array_equal(sample.probability, np.full(90, 1.0 / 90))
    assert np.array_equal(sample.period, np.arange(90))
    assert sample.provenance.source_kind == "annual"
    assert sample.provenance.semantics_source == "data"


def test_rows_are_in_year_model_member_order_with_names_in_byte_order() -> None:
    names = ["z-model", "ä-model", "B-model"]
    sample = _pool([_run(name) for name in names])
    pool = _document(sample)
    assert [run["model"] for run in pool["runs"]] == ["B-model", "z-model", "ä-model"]
    assert pool["year"][:6] == [2015, 2015, 2015, 2016, 2016, 2016]
    assert pool["run"][:6] == [0, 1, 2, 0, 1, 2]
    expected = np.column_stack(
        [_values(name) for name in ("B-model", "z-model", "ä-model")]
    ).reshape(-1)
    assert np.array_equal(sample.index[:, 0], expected)


def test_each_row_s_identity_is_recorded() -> None:
    sample = _pool()
    pool = _document(sample)
    assert pool["pathway"] == "ssp245"
    assert len(pool["year"]) == len(pool["run"]) == len(sample)
    for run in pool["runs"]:
        assert {"model", "member", "pathway", "lineage", "series"} <= set(run)
        assert run["series"]["source"]["name"] == f"synthetic {run['model']}"
    assert sample.provenance.sources[0]["model"] == MODELS[0]
    assert [source["role"] for source in sample.provenance.sources] == ["hazard"] * 3


def test_the_order_of_the_runs_does_not_change_the_sample() -> None:
    one = _pool([_run(model) for model in MODELS])
    other = _pool([_run(model) for model in reversed(MODELS)])
    assert one.content_hash == other.content_hash
    assert one.lineage == other.lineage


def test_a_written_pool_keeps_each_row_s_identity() -> None:
    sample = _pool(trend=LinearTrend())
    read = OutcomeSample.from_written_table(sample.to_table())
    assert read.provenance.to_document() == sample.provenance.to_document()
    assert _document(read)["year"] == _document(sample)["year"]


def test_every_other_sample_records_no_pool() -> None:
    plain = sample_annual(
        _series("X"), loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="a"
    )
    assert plain.provenance.pool is None


def test_runs_may_be_any_iterable() -> None:
    assert len(_pool(_run(model) for model in MODELS)) == 90


def test_the_hazard_is_kept_when_the_noise_leaves_the_index_alone() -> None:
    noisy = _pool(basis_risk=ImpactNoise(0.3), rng=np.random.default_rng(1))
    assert noisy.hazard is None
    assert noisy.index_unit == "mm/day" and noisy.loss_unit == "EUR"


# -- refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: [_run("A", pathway="ssp245"), _run("B", pathway="ssp585")],
            "one pathway",
        ),
        (
            lambda: [_run("A", pathway="historical"), _run("B", pathway="ssp245")],
            "historical",
        ),
        (lambda: [_run("A"), _run("B", unit="mm")], "one unit"),
        (
            lambda: [
                _run("A", source={"name": "a", "recipe": "rx1day"}),
                _run("B", source={"name": "b", "recipe": "rx5day"}),
            ],
            "one recipe",
        ),
        (
            lambda: [
                _run(
                    "A",
                    table_columns={"recipe": {"values": ["rx1day"], "distinct": 1}},
                ),
                _run(
                    "B",
                    table_columns={"recipe": {"values": ["rx5day"], "distinct": 1}},
                ),
            ],
            "one recipe",
        ),
        (
            lambda: [_run("A", member="r1i1p1f1"), _run("A", member="r2i1p1f1")],
            "whatever its member",
        ),
        (lambda: [_run("A"), _run("A")], "whatever its member"),
        (lambda: [_run("A")], "two runs or more"),
        (list, "two runs or more"),
        (
            lambda: [_run("A"), _run("B", years=np.arange(2015, 2041))],
            "must hold the same years",
        ),
    ],
)
def test_a_pool_it_cannot_hold_raises(build: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _pool(build())


def test_different_years_list_each_run_s_years() -> None:
    with pytest.raises(SampleError) as error:
        _pool([_run("A"), _run("B", years=np.arange(2015, 2041))])
    text = str(error.value)
    assert "A r1i1p1f1: 2015 to 2044" in text
    assert "B r1i1p1f1: 2015 to 2040" in text
    assert "years=" in text


def test_a_run_that_states_no_recipe_is_not_compared() -> None:
    runs = [_run("A", source={"name": "a", "recipe": "rx1day"}), _run("B")]
    assert len(_pool(runs)) == 60


ONE_RECIPE = {"recipe": {"values": ["rx5day"], "distinct": 1}}


@pytest.mark.parametrize(
    "runs",
    [
        lambda: [
            _run("A", source={"name": "a", "hazard_name": "rx1day"}),
            _run("B", source={"name": "b", "hazard_name": "rx5day"}),
        ],
        lambda: [
            _run("A", source={"name": "a", "recipe": "rx1day"}),
            _run("B", source={"name": "b", "hazard_name": "rx5day"}),
        ],
        lambda: [
            _run("A", source={"name": "a", "hazard_name": "rx1day"}),
            _run("B", table_columns=ONE_RECIPE),
        ],
    ],
    ids=["two-hazard-names", "a-recipe-and-a-name", "a-name-and-a-column"],
)
@pytest.mark.parametrize("scenario", ["baseline", "ssp245 pool"])
def test_runs_that_name_two_hazards_are_refused_whatever_the_label(
    runs: Any, scenario: str
) -> None:
    # Each run names its hazard where no recipe does, so the recipes don't refuse it.
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    with pytest.raises(SampleError) as error:
        _pool(runs(), scenario=scenario, basis_risk=ImpactNoise(0.2), rng=rng)
    assert str(error.value).startswith(
        "a pool reads one hazard in every run, but its runs name "
        "['rx1day', 'rx5day'] (run A r1i1p1f1 names 'rx1day', run B r1i1p1f1 "
        "names 'rx5day')"
    )
    assert rng.bit_generator.state == state


def test_a_run_that_names_no_hazard_is_not_compared_and_the_pool_records_none() -> None:
    runs = [
        _run("A", source={"name": "a", "hazard_name": "rx1day"}),
        _run("B", source={"name": "b", "hazard_name": "rx1day"}),
        _run("C"),
    ]
    assert _pool(runs).provenance.hazard is None
    assert _pool(runs[:2]).provenance.hazard == {
        "name": "rx1day",
        "from": "source_plan",
    }


def test_runs_that_name_one_hazard_in_different_ways_record_it() -> None:
    runs = [
        _run("A", source={"name": "a", "hazard_name": "rx5day"}),
        _run("B", table_columns=ONE_RECIPE),
    ]
    assert _pool(runs).provenance.hazard == {"name": "rx5day", "from": "source_plan"}


def test_a_table_of_several_models_at_one_location_still_raises_as_one_series() -> None:
    table = _table_of(cells=CELLS[:1])
    with pytest.raises(SampleError, match="several rows"):
        AnnualSeries.from_table(table, unit="mm/day")


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda: UNTYPED_RUN([1.0], model="A", member="r", pathway="p"), TypeError),
        (
            lambda: UNTYPED_RUN(_series("A"), model=" A", member="r", pathway="p"),
            SampleError,
        ),
        (
            lambda: UNTYPED_RUN(_series("A"), model="A", member="", pathway="p"),
            SampleError,
        ),
        (
            lambda: UNTYPED_RUN(_series("A"), model="A", member="r", pathway=None),
            SampleError,
        ),
        (
            lambda: UNTYPED_POOL(
                "runs", loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="a"
            ),
            TypeError,
        ),
        (
            lambda: UNTYPED_POOL(
                ["run"], loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="a"
            ),
            TypeError,
        ),
        (lambda: _pool(missing="zero"), SampleError),
        (lambda: _pool(trend="none"), TypeError),
        (lambda: _pool(scenario=" x"), SampleError),
        (lambda: _pool(rng=1), TypeError),
    ],
)
def test_invalid_arguments_raise(build: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        build()


def test_the_loss_model_checks_the_unit_of_the_runs() -> None:
    expression = ImpactExpression.linear(
        0.01,
        calibration=ImpactCalibration(
            exposure_variable="tmax",
            exposure_unit="degC",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        clamp={"minimum": 0.0, "maximum": 1.0},
    )
    with pytest.raises(SampleError, match="degC"):
        _pool(loss_model=LossModel(expression))


# -- the same years, missing years and the window ------------------------------


def _gap(model: str, year: int) -> Any:
    values = _values(model).copy()
    values[year - 2015] = np.nan
    return values


def test_a_missing_year_raises_and_names_the_run() -> None:
    with pytest.raises(SampleError) as error:
        _pool([_run("A", values=_gap("A", 2020)), _run("B")])
    assert "A r1i1p1f1: 2020" in str(error.value)
    assert "drop_year" in str(error.value)


def test_drop_year_drops_a_year_from_every_run_and_lists_it_with_the_run() -> None:
    sample = _pool(
        [_run("A", values=_gap("A", 2020)), _run("B", values=_gap("B", 2031))],
        missing="drop_year",
    )
    pool = _document(sample)
    assert len(sample) == 2 * 28
    assert 2020 not in pool["year"] and 2031 not in pool["year"]
    runs = {run["model"]: run for run in pool["runs"]}
    assert runs["A"]["dropped_years"] == [2020]
    assert runs["B"]["dropped_years"] == [2031]
    assert sample.provenance.details["dropped_years"] == (2020, 2031)
    assert any("dropped from every run" in note for note in sample.provenance.notes)


def test_a_year_a_run_does_not_hold_is_not_dropped_for_it() -> None:
    with pytest.raises(SampleError, match="must hold the same years"):
        _pool(
            [_run("A"), _run("B", years=np.arange(2015, 2041))],
            missing="drop_year",
        )


def test_no_year_with_a_value_in_every_run_raises() -> None:
    all_gone = np.full(len(YEARS), np.nan)
    with pytest.raises(SampleError, match="no year has a value"):
        _pool([_run("A", values=all_gone), _run("B")], missing="drop_year")


def test_recent_years_takes_the_last_n_of_each_run() -> None:
    sample = _pool(
        [_run("A"), _run("B", years=np.arange(2005, 2045))], trend=RecentYears(30)
    )
    assert _document(sample)["year"][0] == 2015
    assert len(sample) == 60


def test_a_window_that_leaves_different_years_raises() -> None:
    with pytest.raises(SampleError, match="must hold the same years"):
        _pool(
            [_run("A"), _run("B", years=np.arange(2005, 2035))],
            trend=RecentYears(10),
        )


def test_recent_years_above_a_run_s_length_names_the_run() -> None:
    with pytest.raises(SampleError, match="run A r1i1p1f1: RecentYears"):
        _pool(
            [_run("A", years=np.arange(2035, 2045)), _run("B")], trend=RecentYears(30)
        )


# -- the trend -----------------------------------------------------------------


def test_each_run_has_its_own_fit_on_the_same_years_and_reference_year() -> None:
    sample = _pool(trend=LinearTrend())
    trend = sample.provenance.to_document()["trend"]
    assert trend["reference_year"] == 2044
    assert trend["fit_years"] == YEARS.tolist()
    assert [fit["model"] for fit in trend["fits"]] == list(MODELS)
    assert [fit["role"] for fit in trend["fits"]] == ["hazard"] * 3
    assert len({fit["slope"] for fit in trend["fits"]}) == 3
    for position, model in enumerate(MODELS):
        single = sample_annual(
            _series(model),
            loss_model=LOSS,
            basis_risk=NoBasisRisk(),
            asset_id="site-a",
            trend=LinearTrend(),
        )
        assert np.allclose(
            sample.index[position :: len(MODELS), 0], single.index[:, 0], atol=1e-9
        )
        assert trend["fits"][position]["slope"] == pytest.approx(
            single.provenance.to_document()["trend"]["fits"][0]["slope"]
        )


def test_the_reference_year_is_the_last_of_the_pool_s_years_when_it_is_dropped() -> (
    None
):
    sample = _pool(
        [_run("A", values=_gap("A", 2044)), _run("B")],
        trend=LinearTrend(),
        missing="drop_year",
    )
    trend = sample.provenance.to_document()["trend"]
    assert trend["reference_year"] == 2044
    assert trend["fit_years"][-1] == 2043


def test_a_trend_error_names_the_run() -> None:
    rising = 50.0 + 0.5 * (YEARS - 2015)
    with pytest.raises(SampleError, match="hazard of run A r1i1p1f1"):
        _pool(
            [_run("A", values=rising), _run("B", values=rising + 1.0)],
            trend=LinearTrend(reference_year=1900),
        )


def test_the_setup_text_names_each_run_s_fit() -> None:
    table = evaluate(LINEAR, _pool(trend=LinearTrend()), split="in_sample")
    line = _lines(setup_text(table), "Trend")[0]
    for model in MODELS:
        assert f"hazard ({model} {MEMBER}) slope" in line


# -- basis risk and the scenario label -----------------------------------------


def test_no_basis_risk_warns_as_for_one_series() -> None:
    with pytest.warns(ComonotoneIndexWarning):
        _pool()


def test_impact_noise_draws_one_factor_per_row_in_row_order() -> None:
    model = ImpactNoise(0.3)
    sample = _pool(basis_risk=model, rng=np.random.default_rng(9))
    plain = _pool()
    rng = sample.provenance.generator()
    factor = np.exp(0.3 * rng.standard_normal(90) - 0.5 * 0.3**2)
    assert np.allclose(sample.loss[:, 0], plain.loss[:, 0] * factor, atol=1e-6)
    assert np.array_equal(sample.index, plain.index)
    assert sample.provenance.rng[0]["step"] == "basis_risk"


def test_impact_noise_rebuilds_from_the_recorded_state() -> None:
    sample = _pool(basis_risk=ImpactNoise(0.3), rng=np.random.default_rng(9))
    again = _pool(basis_risk=ImpactNoise(0.3), rng=sample.provenance.generator())
    assert sample.content_hash == again.content_hash


def test_the_scenario_label_names_the_rows_and_is_out_of_the_lineage() -> None:
    labelled = _pool(scenario="ssp245 pool")
    assert labelled.scenario == "ssp245 pool"
    assert _pool().scenario == "baseline"
    assert labelled.content_hash != _pool().content_hash
    assert labelled.lineage == _pool().lineage


# -- splits --------------------------------------------------------------------


def test_holdout_cuts_years_not_rows() -> None:
    sample = _pool()
    design, held_out = holdout(sample, 0.75)
    design_years = _document(design)["year"]
    held_years = _document(held_out)["year"]
    assert len(design) == 3 * 22 and len(held_out) == 3 * 8
    assert set(design_years) == set(range(2015, 2037))
    assert set(held_years) == set(range(2037, 2045))
    # A cut of the rows at 0.75 would put 2037 on both sides.
    rows = _document(sample)["year"]
    assert rows[66] == 2037 and rows[67] == 2037
    assert not set(design_years) & set(held_years)


def test_holdout_holds_out_the_latest_years_of_every_model() -> None:
    _, held_out = holdout(_pool(), 0.75)
    pool = _document(held_out)
    for position in range(3):
        kept = [
            year
            for year, run in zip(pool["year"], pool["run"], strict=True)
            if run == position
        ]
        assert kept == list(range(2037, 2045))


def test_a_block_s_probabilities_are_rescaled_and_its_rows_kept_in_order() -> None:
    sample = _pool()
    design, held_out = holdout(sample, 0.75)
    assert np.array_equal(design.period, np.arange(66))
    assert np.array_equal(held_out.period, np.arange(66, 90))
    assert design.probability.sum() == pytest.approx(1.0)
    assert np.allclose(held_out.probability, 1.0 / 24)
    assert design.provenance.selection[-1]["block"] == "train"


def test_kfold_holds_out_the_same_years_of_every_run_in_each_fold() -> None:
    folds = year_block_kfold(_pool(), 4)
    blocks = np.array_split(YEARS, 4)
    assert len(folds) == 4
    for block, (design, held_out) in zip(blocks, folds, strict=True):
        pool = _document(held_out)
        assert set(pool["year"]) == set(block.tolist())
        for position in range(3):
            years = [
                year
                for year, run in zip(pool["year"], pool["run"], strict=True)
                if run == position
            ]
            assert years == block.tolist()
        assert not set(_document(design)["year"]) & set(pool["year"])
    assert set(_document(folds[-1][1])["year"]) == set(blocks[-1].tolist())


def test_k_is_at_most_the_number_of_years() -> None:
    assert len(year_block_kfold(_pool(), 30)) == 30
    with pytest.raises(SampleError, match=r"\[2, 30\]"):
        year_block_kfold(_pool(), 31)


def test_a_fraction_that_leaves_a_block_empty_raises_in_years() -> None:
    with pytest.raises(SampleError, match="of 30 years"):
        holdout(_pool(), 0.01)


def test_select_keeps_the_identity_of_its_rows() -> None:
    sample = _pool()
    selected = sample.select([0, 1, 5, 89])
    pool = _document(selected)
    assert pool["year"] == [2015, 2015, 2016, 2044]
    assert pool["run"] == [0, 1, 2, 2]
    assert selected.provenance.selection[-1]["kind"] == "select"


# -- leakage -------------------------------------------------------------------


def test_the_blocks_of_a_pool_pass_the_leakage_check() -> None:
    design, held_out = holdout(_pool(), 0.75)
    table = evaluate(LINEAR, held_out, split="out_of_sample", design_sample=design)
    assert table.num_rows == 1
    entry = EvaluationMetadata.from_table(table).setup.evaluations[0]
    assert entry["method"] == "holdout" and entry["leakage_check"] == "passed"


def test_a_pool_evaluated_on_its_own_rows_raises() -> None:
    design, _ = holdout(_pool(), 0.75)
    with pytest.raises(LeakageError, match="shares 66 model-years"):
        evaluate(LINEAR, design, split="out_of_sample", design_sample=design)


def test_pools_of_different_runs_that_share_one_run_raise() -> None:
    one = _pool([_run("A"), _run("B")])
    other = _pool([_run("B"), _run("C")])
    assert one.lineage != other.lineage
    with pytest.raises(LeakageError, match="shares 30 model-years"):
        evaluate(LINEAR, other, split="out_of_sample", design_sample=one)


def test_one_shared_model_year_is_named_in_the_singular() -> None:
    first = _pool([_run("A", years=YEARS[:6]), _run("B", years=YEARS[:6])])
    second = _pool([_run("A", years=YEARS[5:11]), _run("C", years=YEARS[5:11])])
    with pytest.raises(LeakageError, match="shares 1 model-year with"):
        evaluate(LINEAR, second, split="out_of_sample", design_sample=first)


def test_pools_of_other_runs_are_no_leak() -> None:
    one = _pool([_run("A"), _run("B")])
    other = _pool([_run("C"), _run("D")])
    table = evaluate(LINEAR, other, split="out_of_sample", design_sample=one)
    assert table.num_rows == 1


def test_a_pool_and_one_of_its_runs_raise_either_way() -> None:
    pool = _pool()
    plain = sample_annual(
        _series(MODELS[0]),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="site-a",
    )
    with pytest.raises(LeakageError, match="shares 30 model-years"):
        evaluate(LINEAR, plain, split="out_of_sample", design_sample=pool)
    with pytest.raises(LeakageError, match="shares 30 model-years"):
        evaluate(LINEAR, pool, split="out_of_sample", design_sample=plain)


def test_a_run_on_other_years_than_a_pool_is_no_leak() -> None:
    design, held_out = holdout(_pool(), 0.75)
    plain = sample_annual(
        _series(MODELS[0]),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="site-a",
    )
    _, late = holdout(plain, 22 / 30)
    assert late.period[0] == 2037
    table = evaluate(LINEAR, late, split="out_of_sample", design_sample=design)
    assert table.num_rows == 1
    assert held_out.lineage == design.lineage


def test_a_trend_fitted_on_all_the_years_of_a_pool_is_refused() -> None:
    design, held_out = holdout(_pool(trend=LinearTrend()), 0.75)
    with pytest.raises(LeakageError, match="fit_years"):
        evaluate(LINEAR, held_out, split="out_of_sample", design_sample=design)


def test_a_trend_fitted_on_the_design_years_of_a_pool_passes() -> None:
    fitted = _pool(trend=LinearTrend(fit_years=YEARS[:22]))
    design, held_out = holdout(fitted, 0.75)
    table = evaluate(LINEAR, held_out, split="out_of_sample", design_sample=design)
    assert table.num_rows == 1


def test_the_lineage_leaves_a_pool_s_trend_out() -> None:
    assert _pool(trend=LinearTrend()).lineage == _pool().lineage
    assert (
        _document(_pool(trend=LinearTrend()))["runs"][0]["lineage"]
        == _document(_pool())["runs"][0]["lineage"]
    )


# -- join ----------------------------------------------------------------------


def _other_site(asset: str = "site-b", **options: Any) -> OutcomeSample:
    runs = [
        ModelRun(
            _series(model, values=_values(model, shift=5.0)),
            model=model,
            member=MEMBER,
            pathway="ssp245",
        )
        for model in MODELS
    ]
    return _pool(runs, asset_id=asset, **options)


def test_join_keeps_the_pool_and_states_the_data_s_dependence() -> None:
    joined = OutcomeSample.join([_pool(), _other_site()])
    pool = _document(joined)
    assert joined.asset_ids == ("site-a", "site-b")
    assert joined.provenance.to_document()["dependence"] == {"kind": "data"}
    assert [run["model"] for run in pool["runs"]] == list(MODELS)
    assert pool["year"] == _document(_pool())["year"]
    assert joined.provenance.source_kind == "annual"


def test_join_needs_the_same_rows() -> None:
    two = _pool([_run(model) for model in MODELS[:2]], asset_id="site-b")
    with pytest.raises(SampleError, match="identical period"):
        OutcomeSample.join([_pool([_run(m) for m in MODELS]), two])
    other = _pool([_run("X"), _run(MODELS[1]), _run(MODELS[2])], asset_id="site-b")
    with pytest.raises(SampleError, match=r"same \(year, model, member\)"):
        OutcomeSample.join([_pool(), other])


def test_join_of_a_member_of_another_label_raises() -> None:
    other = [
        ModelRun(_series(model), model=model, member="r2i1p1f1", pathway="ssp245")
        for model in MODELS
    ]
    with pytest.raises(SampleError, match=r"same \(year, model, member\)"):
        OutcomeSample.join([_pool(), _pool(other, asset_id="site-b")])


def test_join_of_years_that_differ_raises() -> None:
    other = _pool(
        [_run(model, years=np.arange(2016, 2046)) for model in MODELS],
        asset_id="site-b",
    )
    with pytest.raises(SampleError, match=r"same \(year, model, member\)"):
        OutcomeSample.join([_pool(), other])


def test_join_of_a_pool_and_a_sample_that_is_not_one_raises() -> None:
    plain = OutcomeSample(
        asset_ids=("site-b",),
        period=np.arange(90),
        probability=np.full(90, 1.0 / 90),
        index=_pool().index,
        loss=_pool().loss,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="annual", sources=(), semantics_source="data"
        ),
    )
    with pytest.raises(SampleError, match="pooled in every sample or in none"):
        OutcomeSample.join([_pool(), plain])


def test_join_of_pools_under_two_pathways_raises() -> None:
    other = [
        ModelRun(_series(model), model=model, member=MEMBER, pathway="ssp585")
        for model in MODELS
    ]
    with pytest.raises(SampleError, match="one pathway"):
        OutcomeSample.join([_pool(), _pool(other, asset_id="site-b")])


def test_asset_returns_the_part_s_own_trend_and_pool() -> None:
    first, second = _pool(trend=LinearTrend()), _other_site(trend=LinearTrend())
    joined = OutcomeSample.join([first, second])
    taken = joined.asset("site-b")
    assert _document(taken) == _document(second)
    assert (
        taken.provenance.to_document()["trend"]
        == (second.provenance.to_document()["trend"])
    )
    assert taken.content_hash == second.content_hash
    assert taken.lineage == second.lineage


def test_a_split_of_a_joined_pool_cuts_years_and_asset_keeps_those_rows() -> None:
    joined = OutcomeSample.join([_pool(), _other_site()])
    design, held_out = holdout(joined, 0.75)
    assert len(held_out) == 24
    taken = held_out.asset("site-a")
    pool = _document(taken)
    assert len(pool["year"]) == 24
    assert set(pool["year"]) == set(range(2037, 2045))
    assert pool["runs"][0]["series"]["source"]["name"] == f"synthetic {MODELS[0]}"
    table = evaluate(
        {"site-a": LINEAR, "site-b": LINEAR},
        held_out,
        split="out_of_sample",
        design_sample=design,
    )
    assert table.num_rows >= 1


def test_a_selection_of_runs_joins_and_comes_back_to_its_own_runs() -> None:
    first = _pool().select(np.arange(0, 90, 3))
    second = _other_site().select(np.arange(0, 90, 3))
    joined = OutcomeSample.join([first, second])
    assert [run["model"] for run in _document(joined)["runs"]] == [MODELS[0]]
    assert _document(joined.asset("site-b"))["run"] == [0] * 30
    assert len(_document(joined.asset("site-b"))["runs"]) == 3


def test_a_joined_pool_on_its_own_rows_raises_leakage() -> None:
    joined = OutcomeSample.join([_pool(), _other_site()])
    design, _ = holdout(joined, 0.75)
    with pytest.raises(LeakageError, match="model-years"):
        evaluate(
            {"site-a": LINEAR, "site-b": LINEAR},
            design,
            split="out_of_sample",
            design_sample=design,
        )


def test_a_joined_run_counts_as_one_run_for_a_joined_plain_sample() -> None:
    pool = OutcomeSample.join([_pool(), _other_site()])
    plain = OutcomeSample.join(
        [
            sample_annual(
                _series(MODELS[0]),
                loss_model=LOSS,
                basis_risk=NoBasisRisk(),
                asset_id="site-a",
            ),
            sample_annual(
                _series(MODELS[0], values=_values(MODELS[0], shift=5.0)),
                loss_model=LOSS,
                basis_risk=NoBasisRisk(),
                asset_id="site-b",
            ),
        ]
    )
    with pytest.raises(LeakageError, match="shares 30 model-years"):
        evaluate(
            {"site-a": LINEAR, "site-b": LINEAR},
            plain,
            split="out_of_sample",
            design_sample=pool,
        )


# -- the plan ------------------------------------------------------------------


def test_a_nex_shaped_plan_gives_one_60_row_sample_per_asset() -> None:
    samples = _resolved(_planned()).samples()
    assert list(samples) == ["a", "b"]
    for sample in samples.values():
        assert len(sample) == 2 * 30 == 60
        assert np.allclose(sample.probability, 1.0 / 60)
        pool = _document(sample)
        assert [run["model"] for run in pool["runs"]] == list(MODELS[:2])
        assert pool["pathway"] == "ssp245"


def test_a_nex_shaped_plan_states_the_pooling(tmp_path: Path) -> None:
    result = _resolved(_planned()).write_parquet(tmp_path / "pool.parquet")
    text = setup_text(result.metadata)
    line = _lines(text, "Pool")[0]
    assert "2 runs under pathway ssp245" in line
    assert f"{MODELS[0]} {MEMBER}" in line and f"{MODELS[1]} {MEMBER}" in line
    assert "the same weight" in line
    assert "30 years (2015 to 2044), 60 rows" in line
    for statement in STATEMENTS:
        assert statement in line
    assert "60 pooled model-years (2015 to 2044)" in text
    assert "pooled per-year values" in text
    assert "improve" not in line and "better" not in line


def test_the_plan_s_sample_is_the_one_built_by_hand() -> None:
    table = _table_of()
    samples = _resolved(_planned(table)).samples()
    children = np.random.SeedSequence(42).spawn(2)
    for position, asset in enumerate(("a", "b")):
        runs = [
            ModelRun(
                AnnualSeries.nearest_cell(
                    table.filter(pc.equal(table.column("model"), model)),
                    longitude=SITES[asset][0],
                    latitude=SITES[asset][1],
                    unit="mm/day",
                    source=ARCHIVE,
                ),
                model=model,
                member=MEMBER,
                pathway="ssp245",
            )
            for model in MODELS[:2]
        ]
        sample = sample_annual_pool(
            runs,
            loss_model=LossModel(RAMP, value=2e6, value_unit="EUR"),
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id=asset,
            rng=np.random.default_rng(children[position]),
        )
        assert sample.content_hash == samples[asset].content_hash
        assert sample.lineage == samples[asset].lineage


def test_a_labelled_plan_refuses_a_run_its_table_contradicts() -> None:
    table = _table_of(scenario="ssp585").append_column(
        "experiment", pa.array(["ssp245"] * len(_table_of()))
    )
    plan = _resolved(
        _planned(table, pool=ModelPool(pathway="experiment"), scenario="x")
    )
    with pytest.raises(ScenarioError, match="'x'.*'ssp245'.*scenario column.*'ssp585'"):
        plan.samples()


def test_a_labelled_plan_whose_runs_agree_keeps_the_unlabelled_rows() -> None:
    unlabelled = _resolved(_planned()).samples()["a"]
    labelled = _resolved(_planned(scenario="x")).samples()["a"]
    assert labelled.lineage == unlabelled.lineage
    assert labelled.loss.tobytes() == unlabelled.loss.tobytes()


def test_a_plan_on_batches_of_a_stub_nex_plan_gives_the_same_rows() -> None:
    table = _table_of()
    from_table = _resolved(_planned(table)).samples()
    from_plan = _resolved(_planned(_NexPlan(table))).samples()
    assert from_plan["a"].content_hash == from_table["a"].content_hash


def test_a_sequence_of_sources_one_per_run_gives_the_same_rows() -> None:
    sources = [_table_of([model]) for model in MODELS[:2]]
    from_sequence = _resolved(_planned(sources)).samples()
    from_table = _resolved(_planned(_table_of())).samples()
    assert from_sequence["a"].content_hash == from_table["a"].content_hash


def test_a_file_source_pools_too(tmp_path: Path) -> None:
    path = tmp_path / "nex.parquet"
    pq.write_table(_table_of(), path)
    assert len(_resolved(_planned(path)).samples()["a"]) == 60


def test_a_run_s_identity_comes_from_its_plan_s_explain() -> None:
    plans = [_Cmip6Plan(model) for model in MODELS[:2]]
    samples = _resolved(_planned(plans)).samples()
    pool = _document(samples["a"])
    assert [run["model"] for run in pool["runs"]] == list(MODELS[:2])
    assert pool["pathway"] == "ssp245"
    assert pool["runs"][0]["series"]["pathway"] == "ssp245"
    assert all(len(plan.reads) == 1 for plan in plans)


def test_columns_win_over_a_plan_s_explain() -> None:
    table = _table_of(MODELS[:1], scenario="ssp585")
    plan = _Cmip6Plan("other", table=table)
    other = _Cmip6Plan(MODELS[1], pathway="ssp585")
    pool = _document(_resolved(_planned([plan, other])).samples()["a"])
    assert [run["model"] for run in pool["runs"]] == list(MODELS[:2])


def test_a_source_that_states_no_run_raises() -> None:
    bare = _table_of().select(["year", "latitude", "longitude", "value"])
    with pytest.raises(SampleError, match="states no model"):
        _resolved(_planned(bare)).samples()


def test_a_partly_named_source_needs_the_rest_from_its_plan() -> None:
    table = _table_of().drop_columns(["member"])
    with pytest.raises(SampleError, match="states no member"):
        _resolved(_planned(table)).samples()


def test_columns_can_be_renamed() -> None:
    table = _table_of().rename_columns(
        [
            {"model": "source_id", "member": "variant", "scenario": "experiment"}.get(
                name, name
            )
            for name in _table_of().column_names
        ]
    )
    plan = _planned(
        table,
        pool=ModelPool(model="source_id", member="variant", pathway="experiment"),
    )
    assert len(_resolved(plan).samples()["a"]) == 60


def test_a_dictionary_run_column_is_read() -> None:
    table = _table_of()
    position = table.schema.get_field_index("model")
    table = table.set_column(
        position, "model", table.column("model").dictionary_encode()
    )
    assert len(_resolved(_planned(table)).samples()["a"]) == 60


def test_a_run_column_of_numbers_or_nulls_raises() -> None:
    table = _table_of()
    numbers = table.set_column(
        table.schema.get_field_index("model"),
        "model",
        pa.array(np.arange(table.num_rows)),
    )
    with pytest.raises(SampleError, match="must hold strings"):
        _resolved(_planned(numbers)).samples()
    nulls = table.set_column(
        table.schema.get_field_index("member"),
        "member",
        pa.array([None] * table.num_rows, pa.string()),
    )
    with pytest.raises(SampleError, match="null values"):
        _resolved(_planned(nulls)).samples()


def test_an_empty_source_raises() -> None:
    empty = _table_of().slice(0, 0)
    with pytest.raises(SampleError, match="no rows"):
        _resolved(_planned(empty)).samples()


def test_a_source_of_a_sequence_that_holds_two_runs_raises() -> None:
    with pytest.raises(SampleError, match="each source of a sequence is one run"):
        _resolved(_planned([_table_of(), _table_of(MODELS[2:])])).samples()


def test_a_sequence_of_two_plans_with_different_recipes_raises() -> None:
    plans = [_Cmip6Plan(MODELS[0]), _Cmip6Plan(MODELS[1], recipe="rx5day")]
    with pytest.raises(SampleError, match="one recipe"):
        _resolved(_planned(plans)).samples()


def test_a_table_under_two_pathways_raises() -> None:
    table = pa.concat_tables(
        [
            _table_of(MODELS[:1], scenario="ssp245"),
            _table_of(MODELS[1:2], scenario="ssp585"),
        ]
    )
    with pytest.raises(SampleError, match="one pathway"):
        _resolved(_planned(table)).samples()


def test_the_same_model_twice_in_a_table_raises_whatever_its_member() -> None:
    table = pa.concat_tables(
        [
            _table_of(MODELS[:1], member="r1i1p1f1"),
            _table_of(MODELS[:1], member="r2i1p1f1"),
        ]
    )
    with pytest.raises(SampleError, match="whatever its member"):
        _resolved(_planned(table)).samples()


def test_a_table_of_several_models_without_pool_still_raises() -> None:
    plan = _planned(_table_of(), pool=None)
    with pytest.raises(SampleError, match="several rows"):
        _resolved(plan).samples()


def test_each_run_is_located_on_its_own_grid_and_recorded() -> None:
    coarse = ((50.75, 6.75), (50.75, 7.25), (51.25, 6.75), (51.25, 7.25))
    sources = [_table_of(MODELS[:1], cells=CELLS), _table_of(MODELS[1:2], cells=coarse)]
    pool = _document(_resolved(_planned(sources)).samples()["a"])
    fine, wide = pool["runs"][0]["series"], pool["runs"][1]["series"]
    assert fine["cell"] == {"latitude": 50.875, "longitude": 6.875}
    assert fine["spacing_degrees"] == {"latitude": 0.25, "longitude": 0.25}
    assert wide["cell"] == {"latitude": 50.75, "longitude": 6.75}
    assert wide["spacing_degrees"] == {"latitude": 0.5, "longitude": 0.5}
    assert "distance_km" not in fine


def test_a_run_s_own_columns_are_recorded_with_the_archive_s_terms() -> None:
    sample = _resolved(_planned()).samples()["a"]
    series = _document(sample)["runs"][0]["series"]
    assert series["table_columns"]["model"]["values"] == [MODELS[0]]
    assert series["source"]["licence"] == "CC0-1.0"
    assert sample.provenance.licences[0]["licence"] == "CC0-1.0"
    assert len(sample.provenance.licences) == 1


def test_years_missing_and_the_trend_apply_to_every_run() -> None:
    plan = _resolved(
        _planned(years=list(range(2020, 2030)), trend=LinearTrend(min_years=3))
    )
    pool = _document(plan.samples()["a"])
    assert sorted(set(pool["year"])) == list(range(2020, 2030))
    assert len(pool["year"]) == 20


def test_a_stated_years_resolves_runs_of_different_lengths() -> None:
    table = pa.concat_tables(
        [_table_of(MODELS[:1]), _table_of(MODELS[1:2], years=np.arange(2015, 2041))]
    )
    with pytest.raises(SampleError, match="must hold the same years"):
        _resolved(_planned(table)).samples()
    plan = _resolved(_planned(table, years=list(range(2015, 2041))))
    assert len(plan.samples()["a"]) == 52


def test_cell_and_pool_may_go_together() -> None:
    plan = _planned(impact=framework_impacts.for_factor("rx1day"), cell=H3Cell(7))
    sample = plan.basis_risk(NoBasisRisk()).samples()["a"]
    assert len(sample) == 60
    assert any("H3 cell" in note for note in sample.provenance.notes)


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        (lambda: _planned(pool="model"), TypeError, "ModelPool"),
        (
            lambda: _planned(location=StationColumn("station")),
            ValueError,
            "models have no stations",
        ),
        (lambda: _planned([]), ValueError, "at least one"),
        (lambda: _planned([1.0]), TypeError, "annual_extremes"),
        (lambda: _planned(_table_of().to_pylist()), TypeError, "annual_extremes"),
        (lambda: ModelPool(model=""), ValueError, "model"),
    ],
)
def test_invalid_pools_raise_at_the_builder(
    build: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build()


def test_a_sequence_without_pool_raises_as_before() -> None:
    with pytest.raises(TypeError, match="annual_extremes"):
        _planned([_table_of()], pool=None)


def test_index_is_refused_on_a_pool() -> None:
    with pytest.raises(ValueError, match="share no weather"):
        _planned().index(
            _table_of(),
            location=NearestCell(),
            unit="mm/day",
            provenance=ARCHIVE,
        )


def test_fresh_draws_are_refused_on_a_pool() -> None:
    with pytest.raises(TypeError, match="fresh draws"):
        _resolved(_planned()).split(FreshDraws(100, seed=1))


def test_the_dependence_builder_is_refused_on_a_pool() -> None:
    with pytest.raises(ValueError, match="takes no dependence model"):
        _planned().dependence(Comonotonic())


def test_explain_states_the_pool_s_columns_and_reads_nothing() -> None:
    plans = [_Cmip6Plan(model) for model in MODELS[:2]]
    plan = _resolved(_planned(plans, trend=LinearTrend(), scenario="ssp245 pool"))
    document: Any = plan.explain(format="json")
    assert all(not each.reads for each in plans)
    sample = document["setup"]["samples"][0]
    assert sample["pool"] == {
        "kind": "model_pool",
        "model_column": "model",
        "member_column": "member",
        "pathway_column": "scenario",
    }
    assert sample["trend"]["kind"] == "linear_trend"
    assert sample["scenario"] == "ssp245 pool"
    assert [entry["role"] for entry in document["sources"]] == ["hazard", "hazard"]
    assert all(entry["pool"]["kind"] == "model_pool" for entry in document["sources"])
    line = _lines(setup_text(document), "Pool")[0]
    assert "the columns model, member and scenario" in line
    assert "not known yet" in line
    for statement in STATEMENTS:
        assert statement in line


def test_explain_of_one_table_describes_it_once() -> None:
    document: Any = _resolved(_planned()).explain(format="json")
    assert len(document["sources"]) == 1
    assert document["sources"][0]["rows"] == _table_of().num_rows


def test_explain_of_a_plan_without_pool_has_no_pool_key() -> None:
    plain = UNTYPED_PLAN.for_annual_extremes(
        _table_of(MODELS[:1]),
        _assets(),
        impact=RAMP,
        unit="mm/day",
        provenance=ARCHIVE,
        value_column="value_eur",
        value_unit="EUR",
    )
    document: Any = (
        plain.basis_risk(NoBasisRisk()).covers(LINEAR).explain(format="json")
    )
    assert "pool" not in document["setup"]["samples"][0]
    assert "pool" not in document["sources"][0]


def test_the_written_metadata_holds_the_pool_without_its_rows(tmp_path: Path) -> None:
    result = (
        _resolved(_planned()).split(Holdout(0.75)).write_parquet(tmp_path / "p.parquet")
    )
    entries = result.metadata.setup.samples
    assert all("pool" not in entry["details"] for entry in entries)
    for entry in entries:
        pool = entry["pool"]
        assert {"pathway", "rows", "runs", "years"} == set(pool)
        assert "year" not in pool and "run" not in pool
        assert all("series" not in run for run in pool["runs"])
        assert pool["rows"] == 2 * len(pool["years"])
        assert [run["model"] for run in pool["runs"]] == list(MODELS[:2])


def test_the_arrays_are_in_the_sample_s_own_metadata() -> None:
    sample = _resolved(_planned()).samples()["a"]
    document = json.loads(
        sample.to_table().schema.metadata[SAMPLE_METADATA_KEY.encode()]
    )
    assert len(document["provenance"]["pool"]["year"]) == 60


def test_sources_has_one_hazard_entry_per_run_and_one_record(tmp_path: Path) -> None:
    table = _table_of()
    result = _resolved(_planned(table)).write_parquet(tmp_path / "p.parquet")
    document = json.loads(result.metadata.to_json())
    hazards = [entry for entry in document["sources"] if entry["role"] == "hazard"]
    assert {entry["model"] for entry in hazards} == set(MODELS[:2])
    assert {entry["annual_table_sha256"] for entry in hazards} == {
        document["source_records"][0]["annual_table"]["table_sha256"]
    }
    assert len(document["source_records"]) == 1
    source = _lines(setup_text(result.metadata), "Source")[0]
    assert f"the hazard series of {MODELS[0]} {MEMBER}" in source
    assert "pathway ssp245" in source


def test_a_sequence_has_one_record_each_named_by_its_runs(tmp_path: Path) -> None:
    sources = [_table_of([model]) for model in MODELS[:2]]
    result = _resolved(_planned(sources)).write_parquet(tmp_path / "p.parquet")
    document = json.loads(result.metadata.to_json())
    records = [entry["annual_table"] for entry in document["source_records"]]
    assert len(records) == 2
    named = {
        entry["model"]: entry["annual_table_sha256"]
        for entry in document["sources"]
        if entry["role"] == "hazard"
    }
    assert named[MODELS[0]] != named[MODELS[1]]
    assert set(named.values()) == {record["table_sha256"] for record in records}


def test_the_split_lines_state_the_years_of_each_block(tmp_path: Path) -> None:
    result = (
        _resolved(_planned()).split(Holdout(0.75)).write_parquet(tmp_path / "p.parquet")
    )
    text = setup_text(result.metadata)
    assert "16 pooled model-years (2037 to 2044), the held-out block" in text
    assert "44 pooled model-years (2015 to 2036), the design block" in text
    pool_lines = _lines(text, "Pool")
    assert len(pool_lines) == 1
    assert "30 years (2015 to 2044), 60 rows" in pool_lines[0]


def test_a_plan_with_a_kfold_states_one_pool_line(tmp_path: Path) -> None:
    result = _resolved(_planned()).split(KFold(3)).write_parquet(tmp_path / "p.parquet")
    assert len(_lines(setup_text(result.metadata), "Pool")) == 1


def test_samples_of_other_runs_have_a_line_each() -> None:
    first = evaluate(LINEAR, _pool(asset_id="a"), split="in_sample")
    other = evaluate(
        LINEAR,
        _pool([_run("X"), _run("Y")], asset_id="b"),
        split="in_sample",
    )
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(first), EvaluationMetadata.from_table(other)
    )
    lines = _lines(setup_text(merged), "Pool")
    assert [line.split(":")[0] for line in lines] == ["Pool of a", "Pool of b"]


def test_a_selection_of_rows_says_so() -> None:
    selected = _pool().select(np.arange(0, 44))
    table = evaluate(LINEAR, selected, split="in_sample")
    assert "on a selection of the rows" in _lines(setup_text(table), "Pool")[0]


def test_a_trend_is_fitted_per_fold_on_a_pool(tmp_path: Path) -> None:
    result = (
        _resolved(_planned(trend=LinearTrend()))
        .split(Holdout(0.75))
        .write_parquet(tmp_path / "p.parquet")
    )
    entries = result.metadata.setup.samples
    assert entries
    for entry in entries:
        trend = entry["trend"]
        assert trend["fit_years"] == list(range(2015, 2037))
        assert trend["reference_year"] == 2044
        assert len(trend["fits"]) == 2
    assert _document(result.samples["a"])["year"][-1] == 2044
    all_years = result.samples["a"].provenance.to_document()["trend"]
    assert all_years["fit_years"] == YEARS.tolist()


def _in_sample_hashes(table: Any) -> set[str]:
    rows = table.filter(pc.equal(table.column("split"), "in_sample"))
    return set(rows.column("sample_hash").to_pylist())


def test_a_changed_held_out_value_leaves_the_design_rows_alone(tmp_path: Path) -> None:
    def run(shift: float, name: str) -> Any:
        return (
            _resolved(_planned(_table_of(held_out_shift=shift), trend=LinearTrend()))
            .split(Holdout(0.75))
            .write_parquet(tmp_path / name)
        )

    base, shifted = run(0.0, "a.parquet"), run(500.0, "b.parquet")
    assert _in_sample_hashes(base.table) == _in_sample_hashes(shifted.table)
    fits = [entry["trend"]["fits"] for entry in base.metadata.setup.samples]
    other = [entry["trend"]["fits"] for entry in shifted.metadata.setup.samples]
    assert fits == other


def test_every_fold_starts_from_the_asset_s_same_generator_state() -> None:
    plan = _resolved(_planned(trend=LinearTrend())).split(KFold(3))
    built = plan._build_samples(folds=True)
    assert built.folds is not None
    states = {fold.provenance.rng[0]["state"] for fold in built.folds["a"]}
    assert len(states) == 1
    assert built.samples["a"].provenance.rng[0]["state"] in states


def test_too_few_design_years_raise_naming_the_fold() -> None:
    plan = _resolved(_planned(trend=LinearTrend(min_years=25))).split(Holdout(0.75))
    with pytest.raises(SampleError, match="fold 0"):
        plan.write_parquet("unused.parquet")


def test_a_design_on_a_pool_runs_out_of_sample(tmp_path: Path) -> None:
    plan = (
        _planned()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(STEP)
        .objective(MSE(), budget=Budget(ratio=0.8))
        .split(Holdout(0.75))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "p.parquet")
    splits = result.table.column("split").to_pylist()
    assert "out_of_sample" in splits and "in_sample" in splits
    assert all(
        design.design_sample.provenance.pool is not None for design in result.designs
    )


# -- Joint() with pool= --------------------------------------------------------


def test_joint_works_with_a_pool(tmp_path: Path) -> None:
    plan = (
        _planned()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(STEP)
        .objective(MSE(), budget=Budget(ratio=0.8))
        .split(Holdout(0.75))
        .scopes(Joint())
        .metrics(OPTIONS)
    )
    joint = plan.joint_sample()
    assert joint.asset_ids == ("a", "b")
    assert len(joint) == 60
    assert joint.provenance.to_document()["dependence"] == {"kind": "data"}
    result = plan.write_parquet(tmp_path / "p.parquet")
    assert "total" in result.table.column("level").to_pylist()
    assert result.joint_sample is not None
    assert result.joint_sample.provenance.pool is not None
    views = result.samples
    assert _document(views["b"])["runs"][0]["series"]["cell"] is not None
    text = setup_text(result.metadata)
    assert "Dependence between the assets: the data's own" in text
    assert len(_lines(text, "Pool")) == 1


def test_joint_with_a_trend_and_a_split_joins_per_fold(tmp_path: Path) -> None:
    plan = (
        _planned(trend=LinearTrend())
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(LINEAR)
        .split(KFold(3))
        .scopes(Joint())
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "p.parquet")
    assert result.joint_sample is not None
    for entry in result.metadata.setup.samples:
        if len(entry["asset_ids"]) == 2:
            assert len(entry["trend"]["fits"]) == 4
            assert {fit["asset_id"] for fit in entry["trend"]["fits"]} == {"a", "b"}
    assert _document(result.joint_sample)["year"][-1] == 2044


def test_a_pool_records_draw_numbers_as_periods_and_calendar_years_in_pool(
    tmp_path: Path,
) -> None:
    result = _resolved(_planned()).write_parquet(tmp_path / "p.parquet")
    sample = next(s for s in result.metadata.setup.samples if s["asset_ids"] == ["a"])
    # Two runs of 30 years: 60 rows, numbered from 0.
    assert sample["periods"] == [0, 59]
    assert sample["pool"]["years"] == [int(year) for year in YEARS]


def _seven_sites() -> Any:
    sites = {f"s{k}": (6.875, 50.125 + 0.25 * k) for k in range(7)}
    return pa.table(
        {
            "asset_id": list(sites),
            "longitude": [point[0] + 0.01 for point in sites.values()],
            "latitude": [point[1] + 0.01 for point in sites.values()],
            "value_eur": [2e6 for _ in sites],
        }
    )


def _seven_cells() -> Any:
    return tuple((50.125 + 0.25 * k, 6.875) for k in range(7))


def test_joint_above_five_assets_summarises_the_trend_and_the_location(
    tmp_path: Path,
) -> None:
    plan = (
        _planned(
            _table_of(cells=_seven_cells()),
            assets=_seven_sites(),
            trend=LinearTrend(),
            location=NearestCell(spacing=(0.25, 0.25)),
        )
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(LINEAR)
        .split(Holdout(0.75))
        .scopes(Joint())
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "p.parquet")
    text = setup_text(result.metadata)
    trend = _lines(text, "Trend")
    assert len(trend) == 1
    assert " of s0" not in trend[0] and "GFDL-ESM4" not in trend[0]
    assert re.search(
        r"where each intercept is the fitted value: hazard slope -?[\d.]+ to -?[\d.]+ "
        r"mm/day per year and intercept [\d.]+ to [\d.]+ mm/day, over 14 series of 7 "
        r"assets; the sample's trend record holds each series' fit\.$",
        trend[0],
    )
    assert _lines(text, "Location") == [
        "Location: hazard series, located by a point, in the nearest grid cell (grid "
        "step 0.25 by 0.25 degrees): 7 assets in 7 cells, with centres at latitudes "
        "50.125 to 51.625 and longitudes 6.875."
    ]
    # The record keeps every series' fit.
    entry = next(s for s in result.metadata.setup.samples if len(s["asset_ids"]) == 7)
    assert len(entry["trend"]["fits"]) == 14


def test_a_pool_read_at_the_same_cell_names_each_asset_once(tmp_path: Path) -> None:
    result = _resolved(_planned()).write_parquet(tmp_path / "p.parquet")
    assert _lines(setup_text(result.metadata), "Location") == [
        "Location: hazard series, located by a point, in the nearest grid cell (grid "
        "step 0.25 by 0.25 degrees): a at (50.875, 6.875) and b at (51.125, 7.125), "
        "the cell centres as (latitude, longitude)."
    ]


def test_individual_and_joint_together_on_a_pool(tmp_path: Path) -> None:
    plan = (
        _planned()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(STEP)
        .objective(MSE(), budget=Budget(ratio=0.8))
        .split(Holdout(0.75))
        .scopes(Individual(), Joint())
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "p.parquet")
    assert set(result.table.column("scope").to_pylist()) == {"individual", "joint"}


# -- an empirical copula on a pooled reference ----------------------------------


def _pooled_reference() -> OutcomeSample:
    return OutcomeSample.join([_pool(), _other_site()])


def test_a_pooled_reference_records_its_first_and_last_calendar_year() -> None:
    document = EmpiricalCopula(_pooled_reference()).to_document()
    assert document["first_year"] == 2015
    assert document["last_year"] == 2044
    assert document["m"] == 90
    assert document["pool"] == {"runs": 3, "years": 30}


def test_any_other_reference_records_its_periods_and_no_pool() -> None:
    first = sample_annual(
        _series("A"), loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="site-a"
    )
    other = sample_annual(
        _series("B", values=_values("B", shift=5.0)),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="site-b",
    )
    document = EmpiricalCopula(OutcomeSample.join([first, other])).to_document()
    assert document["first_year"] == 2015 and document["last_year"] == 2044
    assert document["m"] == 30
    assert "pool" not in document


def test_m_of_a_pooled_reference_is_its_number_of_rows() -> None:
    copula = EmpiricalCopula(_pooled_reference())
    assert copula.record(180, ("site-a", "site-b"))["k"] == 2
    # 120 is a multiple of the 30 years, but not of the 90 rows.
    with pytest.raises(SampleError, match="reference's 90 model-years"):
        copula.record(120, ("site-a", "site-b"))


def test_the_rank_blocks_of_a_pooled_reference_are_its_rows() -> None:
    reference = _pooled_reference()
    copula = EmpiricalCopula(reference)
    uniforms = np.random.default_rng(4).random((180, 2))
    order = copula.reorder(uniforms, ("site-a", "site-b"))
    values = reference.index[:, 0]
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    sorted_draws = np.argsort(uniforms[:, 0], kind="stable")
    for row, rank in enumerate(ranks):
        block = set(sorted_draws[2 * rank : 2 * rank + 2].tolist())
        assert set(order[2 * row : 2 * row + 2, 0].tolist()) == block


def test_the_setup_sentence_names_a_pooled_reference_s_calendar_years() -> None:
    copula = EmpiricalCopula(_pooled_reference())
    statement = copula.statement(180, "assets")
    assert "90 model-years from 2015 to 2044" in statement
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="rx1day",
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
    )
    curve = FittedDistribution.from_parameters("gumbel_r", location=60.0, scale=10.0)
    sample = sample_curves(
        {"site-a": curve, "site-b": curve},
        dependence=copula,
        metadata={"site-a": metadata, "site-b": metadata},
        n_draws=180,
        rngs={"site-a": np.random.default_rng(1), "site-b": np.random.default_rng(2)},
        loss_models={"site-a": LOSS, "site-b": LOSS},
        basis_risk=NoBasisRisk(),
    )
    table = evaluate({"site-a": LINEAR, "site-b": LINEAR}, sample, split="in_sample")
    text = setup_text(table)
    assert "the ranks of 90 reference model-years, 2015 to 2044" in text
    assert "90 model-years from 2015 to 2044" in text
    assert "2 draws per model-year" in text


# -- metadata written before the key -------------------------------------------


def _written_before_the_key(table: Any, tmp_path: Path) -> Any:
    """`table` as written before a setup sample had `pool`, read from its file."""
    key = EVALUATION_METADATA_KEY.encode()
    document = json.loads(table.schema.metadata[key])
    for sample in document["setup"]["samples"]:
        del sample["pool"]
    older = table.replace_schema_metadata(
        {**table.schema.metadata, key: json.dumps(document).encode()}
    )
    path = tmp_path / "older.parquet"
    pq.write_table(older, path)
    return pq.read_table(path)


def _plain_sample() -> OutcomeSample:
    return sample_annual(
        _series("A"), loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="a"
    )


def test_a_comparison_written_before_the_key_reads_as_null(tmp_path: Path) -> None:
    table = evaluate(LINEAR, _plain_sample(), split="in_sample")
    assert EvaluationMetadata.from_table(table).setup.samples[0]["pool"] is None
    older = _written_before_the_key(table, tmp_path)
    metadata = EvaluationMetadata.from_table(older)
    assert "pool" not in metadata.setup.samples[0]
    assert metadata.setup.samples[0].get("pool") is None
    for text in (
        setup_text(metadata),
        setup_text(older),
        setup_text(json.loads(metadata.to_json())),
        setup_text(json.loads(metadata.to_json())["setup"]),
    ):
        assert _lines(text, "Pool") == []
        assert "40 pooled" not in text and "30 recorded years" in text


def test_a_sample_table_written_before_the_field_reads_as_null() -> None:
    table = _plain_sample().to_table()
    key = SAMPLE_METADATA_KEY.encode()
    document = json.loads(table.schema.metadata[key])
    del document["provenance"]["pool"]
    read = OutcomeSample.from_written_table(
        table.replace_schema_metadata({key: json.dumps(document).encode()})
    )
    assert read.provenance.pool is None
    written = evaluate(LINEAR, read, split="in_sample")
    assert EvaluationMetadata.from_table(written).setup.samples[0]["pool"] is None


def test_a_written_provenance_from_before_the_field_reads_as_no_record() -> None:
    document = _plain_sample().provenance.to_document()
    del document["pool"]
    assert SampleProvenance(**document).pool is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"pool": "x"}, "pool must be a mapping"),
        (
            {"pool": {"pathway": "p", "runs": "x", "year": [], "run": []}},
            "pool's runs must be a list",
        ),
        (
            {
                "pool": {
                    "pathway": "p",
                    "runs": [{"model": "m", "member": "r", "lineage": "l"}],
                    "year": [2015],
                    "run": ["0"],
                }
            },
            "outside its 1 runs",
        ),
        ({"pool": {"runs": []}}, "pool must hold"),
        (
            {
                "pool": {
                    "pathway": "p",
                    "runs": [{"model": "m"}],
                    "year": [],
                    "run": [],
                }
            },
            "each run of a pool",
        ),
        (
            {
                "pool": {
                    "pathway": "p",
                    "runs": [{"model": "m", "member": "r", "lineage": "l"}],
                    "year": [2015],
                    "run": [],
                }
            },
            "one year and one run per row",
        ),
        (
            {
                "pool": {
                    "pathway": "p",
                    "runs": [{"model": "m", "member": "r", "lineage": "l"}],
                    "year": [2015],
                    "run": [1],
                }
            },
            "outside its 1 runs",
        ),
    ],
)
def test_an_invalid_pool_record_raises(fields: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {
        "source_kind": "annual",
        "sources": (),
        "semantics_source": "data",
        **fields,
    }
    with pytest.raises(SampleError, match=message):
        SampleProvenance(**arguments)


def test_a_pool_record_of_another_length_than_the_sample_raises() -> None:
    sample = _pool()
    with pytest.raises(SampleError, match="names 90 rows"):
        OutcomeSample(
            asset_ids=sample.asset_ids,
            period=np.arange(2),
            probability=np.full(2, 0.5),
            index=sample.index[:2],
            loss=sample.loss[:2],
            hazard=None,
            index_unit=sample.index_unit,
            loss_unit=sample.loss_unit,
            semantics="annual",
            provenance=sample.provenance,
        )


def test_as_recorded_is_the_default_on_a_pool() -> None:
    sample = _pool(trend=AsRecorded())
    assert sample.provenance.to_document()["trend"] == {"kind": "as_recorded"}
    assert sample.content_hash == _pool().content_hash
