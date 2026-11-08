"""The stated climate-trend option of per-year samples, and `scenario=`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact
from crc_framework.distributions import FittedDistribution

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AsRecorded,
    ComonotoneIndexWarning,
    Cover,
    EvaluationError,
    EvaluationMetadata,
    Free,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    KFold,
    LeakageError,
    LinearPayout,
    LinearTrend,
    LossModel,
    MetricOptions,
    NearestCell,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    RecentYears,
    SampleError,
    SampleProvenance,
    StationColumn,
    StepPayout,
    compare,
    design,
    design_joint,
    evaluate,
    holdout,
    sample_annual,
    sample_curve,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY
from crc_sdk.parametric.sample import SAMPLE_METADATA_KEY
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = pytest.mark.filterwarnings(
    "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
)

UNTYPED_ANNUAL: Any = sample_annual
UNTYPED_RECENT: Any = RecentYears
UNTYPED_LINEAR: Any = LinearTrend
UNTYPED_PLAN: Any = ParametricPlan
YEARS = np.arange(1981, 2021)
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
LINEAR = Cover(INDEX, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR", label="linear")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
HAZARD = SourceProvenance(
    provider="example",
    dataset="synthetic rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic annual maxima",
)
GAUGES = SourceProvenance(
    provider="example",
    dataset="synthetic gauges",
    version="2",
    licence="CC-BY-4.0",
    attribution="Synthetic gauge network",
)
CELLS = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
POINTS = {"a": (6.96, 50.94), "b": (7.10, 51.10)}


def _line(years: Any = YEARS) -> Any:
    return 50.0 + 0.5 * (np.asarray(years) - 1981)


def _noisy(seed: int = 1, years: Any = YEARS) -> Any:
    rng = np.random.default_rng(seed)
    return 60.0 + 0.4 * (np.asarray(years) - 1981) + rng.gumbel(0.0, 8.0, len(years))


def _series(
    values: Any = None,
    years: Any = YEARS,
    unit: str = "mm/day",
    name: str = "synthetic series",
) -> AnnualSeries:
    return AnnualSeries(
        np.asarray(years),
        _line(years) if values is None else values,
        unit,
        {"source": {"name": name}},
    )


def _annual(values: Any = None, **options: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "loss_model": LOSS,
        "basis_risk": NoBasisRisk(),
        "asset_id": "site-a",
    }
    arguments.update(options)
    sample: OutcomeSample = UNTYPED_ANNUAL(_series(values), **arguments)
    return sample


def _text_lines(text: str, label: str) -> list[str]:
    return [line for line in text.splitlines() if line.split(":")[0].startswith(label)]


# -- the three options ---------------------------------------------------------


def test_a_straight_line_is_constant_after_the_adjustment() -> None:
    sample = _annual(trend=LinearTrend())
    assert np.ptp(sample.index) <= 1e-12
    np.testing.assert_allclose(sample.index[:, 0], 69.5, atol=1e-12, rtol=0)


def test_a_straight_line_is_constant_on_a_design_block_fit() -> None:
    sample = _annual(trend=LinearTrend(fit_years=YEARS[:28]))
    assert np.ptp(sample.index) <= 1e-12


def test_the_record_holds_method_fit_years_slope_intercept_and_reference_year() -> None:
    trend = _annual(trend=LinearTrend()).provenance.to_document()["trend"]
    fit = trend.pop("fits")
    assert trend == {
        "kind": "linear_trend",
        "method": "ols_additive",
        "reference_year": 2020,
        "min_years": 10,
        "fit_years": YEARS.tolist(),
    }
    assert len(fit) == 1
    assert fit[0]["role"] == "hazard"
    assert fit[0]["unit"] == "mm/day"
    assert fit[0]["slope"] == pytest.approx(0.5, abs=1e-12)
    assert fit[0]["intercept"] == pytest.approx(69.5, abs=1e-12)


def test_the_fit_is_ordinary_least_squares_on_the_stated_reference_year() -> None:
    values = _noisy()
    sample = _annual(values, trend=LinearTrend(2000))
    slope, intercept = np.polyfit((YEARS - 2000).astype(float), values, 1)
    fit = sample.provenance.trend["fits"][0]  # type: ignore[index]
    assert fit["slope"] == pytest.approx(slope, rel=1e-12)
    assert fit["intercept"] == pytest.approx(intercept, rel=1e-12)
    np.testing.assert_allclose(sample.index[:, 0], values - slope * (YEARS - 2000))


def test_the_adjusted_fit_years_have_the_intercept_as_their_mean() -> None:
    sample = _annual(_noisy(), trend=LinearTrend(fit_years=YEARS[:30]))
    fit = sample.provenance.trend["fits"][0]  # type: ignore[index]
    mean = float(sample.select(YEARS[:30]).index[:, 0].mean())
    assert mean == pytest.approx(fit["intercept"], abs=1e-12)


def test_the_spread_around_the_line_is_not_touched() -> None:
    values = _noisy()
    sample = _annual(values, trend=LinearTrend())
    slope = sample.provenance.trend["fits"][0]["slope"]  # type: ignore[index]
    moved = values - sample.index[:, 0]
    np.testing.assert_allclose(moved, slope * (YEARS - 2020), atol=1e-9)


def test_a_reference_year_may_lie_outside_the_record() -> None:
    sample = _annual(trend=LinearTrend(2050))
    assert sample.provenance.trend["reference_year"] == 2050  # type: ignore[index]
    np.testing.assert_allclose(sample.index[:, 0], 84.5, atol=1e-12, rtol=0)


def test_the_default_reference_year_is_the_last_entry_with_a_value_or_not() -> None:
    values = _noisy()
    values[-1] = np.nan
    trend = _annual(values, missing="drop_year", trend=LinearTrend()).provenance.trend
    assert trend is not None
    assert trend["reference_year"] == 2020
    assert trend["fit_years"][-1] == 2019


def test_the_loss_reads_the_adjusted_hazard() -> None:
    values = _noisy()
    sample = _annual(values, trend=LinearTrend())
    expected = LOSS.loss_values(LOSS.impact_values(sample.index[:, 0]))
    np.testing.assert_allclose(sample.loss[:, 0], expected)
    assert sample.hazard is None
    assert sample.provenance.source_kind == "annual"


def test_as_recorded_is_the_default_and_leaves_the_rows_alone() -> None:
    values = _noisy()
    default = _annual(values)
    stated = _annual(values, trend=AsRecorded())
    assert default.content_hash == stated.content_hash
    np.testing.assert_array_equal(default.index[:, 0], values)
    assert default.provenance.trend == {"kind": "as_recorded"}


def test_a_trend_draws_no_random_numbers() -> None:
    values = _noisy()
    noisy: dict[str, Any] = {"basis_risk": ImpactNoise(0.3)}
    recorded = _annual(values, rng=np.random.default_rng(7), **noisy)
    adjusted = _annual(
        values, rng=np.random.default_rng(7), trend=LinearTrend(), **noisy
    )
    assert recorded.provenance.rng == adjusted.provenance.rng
    assert not np.array_equal(recorded.loss, adjusted.loss)


@pytest.mark.parametrize(
    "option",
    [AsRecorded(), RecentYears(3), LinearTrend(2000, min_years=3, fit_years=(1, 2, 3))],
)
def test_every_option_states_a_document_and_is_hashable(option: Any) -> None:
    json.dumps(option.to_document())
    assert hash(option) == hash(type(option)(**vars(option)))


def test_fit_years_are_kept_sorted_and_may_be_an_array() -> None:
    assert LinearTrend(fit_years=np.array([2003, 2001, 2002])).fit_years == (
        2001,
        2002,
        2003,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: UNTYPED_RECENT(0),
        lambda: UNTYPED_RECENT(-1),
        lambda: UNTYPED_RECENT(True),
        lambda: UNTYPED_RECENT(2.5),
        lambda: UNTYPED_LINEAR(reference_year=2000.5),
        lambda: UNTYPED_LINEAR(reference_year=True),
        lambda: UNTYPED_LINEAR(min_years=2),
        lambda: UNTYPED_LINEAR(min_years=True),
        lambda: UNTYPED_LINEAR(fit_years=[]),
        lambda: UNTYPED_LINEAR(fit_years=[2001, 2001, 2002]),
        lambda: UNTYPED_LINEAR(fit_years=[2001.5]),
        lambda: UNTYPED_LINEAR(fit_years=[[2001, 2002]]),
    ],
)
def test_invalid_options_raise_at_construction(build: Any) -> None:
    with pytest.raises(SampleError):
        build()


def test_a_trend_of_another_type_raises() -> None:
    with pytest.raises(TypeError, match="trend must be"):
        _annual(trend="linear")


# -- the years -----------------------------------------------------------------


def test_too_few_fit_years_raise_with_the_count() -> None:
    with pytest.raises(SampleError, match="at least 10 fit years .* has 5"):
        _annual(trend=LinearTrend(fit_years=YEARS[:5]))
    assert (
        _annual(trend=LinearTrend(min_years=5, fit_years=YEARS[:5])).provenance.trend
        is not None
    )


def test_a_fit_year_the_series_does_not_hold_raises() -> None:
    with pytest.raises(SampleError, match="does not hold: 1970"):
        _annual(trend=LinearTrend(fit_years=[1970, *YEARS[:20].tolist()]))


def test_a_fit_year_dropped_as_missing_is_left_out_of_the_record() -> None:
    values = _noisy()
    values[3] = np.nan
    trend = _annual(
        values, missing="drop_year", trend=LinearTrend(fit_years=YEARS[:20])
    ).provenance.trend
    assert trend is not None
    assert 1984 not in trend["fit_years"]
    assert len(trend["fit_years"]) == 19


def test_too_few_fit_years_count_the_years_with_a_value() -> None:
    values = _noisy()
    values[:31] = np.nan
    with pytest.raises(SampleError, match="at least 10 fit years .* has 9"):
        _annual(values, missing="drop_year", trend=LinearTrend())


def test_recent_years_keep_the_last_entries_and_change_no_value() -> None:
    values = _noisy()
    sample = _annual(values, trend=RecentYears(10))
    assert sample.period.tolist() == YEARS[-10:].tolist()
    np.testing.assert_array_equal(sample.index[:, 0], values[-10:])
    assert sample.provenance.to_document()["trend"] == {
        "kind": "recent_years",
        "n": 10,
        "years": [2011, 2020],
    }
    np.testing.assert_allclose(sample.probability, 0.1)


def test_a_gap_before_the_window_does_not_matter() -> None:
    values = _noisy()
    values[5] = np.nan
    with pytest.raises(SampleError, match="1 year has no value"):
        _annual(values)
    assert len(_annual(values, trend=RecentYears(10))) == 10


def test_a_gap_inside_the_window_is_a_missing_year() -> None:
    values = _noisy()
    values[-3] = np.nan
    with pytest.raises(SampleError, match=r"1 year has no value \(hazard \[2018\]\)"):
        _annual(values, trend=RecentYears(10))
    kept = _annual(values, trend=RecentYears(10), missing="drop_year")
    assert len(kept) == 9
    assert kept.provenance.details["dropped_years"] == (2018,)
    assert kept.provenance.trend["years"] == (2011, 2020)  # type: ignore[index]


def test_recent_years_above_the_length_raise() -> None:
    with pytest.raises(SampleError, match=r"RecentYears\(41\) needs 41 years"):
        _annual(trend=RecentYears(41))


def test_a_split_divides_the_window() -> None:
    train, test = holdout(_annual(_noisy(), trend=RecentYears(20)), 0.75)
    assert train.period.tolist() == YEARS[-20:-5].tolist()
    assert test.period.tolist() == YEARS[-5:].tolist()


# -- no clipping ---------------------------------------------------------------


def test_a_non_finite_adjustment_raises() -> None:
    huge = np.tile([1.7e308, -1.7e308], 20)
    with pytest.raises(SampleError, match="non-finite"):
        _annual(huge, trend=LinearTrend(2500))


def test_a_newly_negative_value_raises_without_clipping() -> None:
    values = 1.0 + (YEARS - 1981.0)
    with pytest.raises(SampleError, match=r"non-negative hazard values negative"):
        _annual(values, trend=LinearTrend(1970))


def test_a_value_negative_as_recorded_may_stay_negative() -> None:
    values = -5.0 + 0.1 * (YEARS - 1981)
    sample = _annual(values, trend=LinearTrend())
    assert (sample.index < 0).all()


# -- a separate index series ---------------------------------------------------


def test_the_hazard_and_the_index_have_a_fit_each() -> None:
    sample = _annual(
        _line(),
        index=_series(20.0 + 0.2 * (YEARS - 1981.0), unit="m", name="gauge"),
        trend=LinearTrend(),
    )
    document = sample.provenance.to_document()["trend"]
    assert [fit["role"] for fit in document["fits"]] == ["hazard", "index"]
    assert [fit["unit"] for fit in document["fits"]] == ["mm/day", "m"]
    assert document["fits"][0]["slope"] == pytest.approx(0.5)
    assert document["fits"][1]["slope"] == pytest.approx(0.2)
    assert sample.hazard is not None
    assert np.ptp(sample.hazard) <= 1e-12
    assert np.ptp(sample.index) <= 1e-12


def test_a_negative_index_value_is_checked_on_its_own_role() -> None:
    index = _series(1.0 + (YEARS - 1981.0), unit="m", name="gauge")
    with pytest.raises(SampleError, match="non-negative index values"):
        _annual(_line() + 500.0, index=index, trend=LinearTrend(1970))


# -- the leakage check ---------------------------------------------------------


def test_a_holdout_of_a_sample_fitted_on_all_years_is_refused() -> None:
    design_block, held_out = holdout(_annual(_noisy(), trend=LinearTrend()), 0.7)
    with pytest.raises(LeakageError, match="fit_years"):
        evaluate(LINEAR, held_out, split="out_of_sample", design_sample=design_block)


def test_a_kfold_of_a_sample_fitted_on_all_years_is_refused() -> None:
    train, test = year_block_kfold(_annual(_noisy(), trend=LinearTrend()), 4)[1]
    with pytest.raises(LeakageError, match="fit_years"):
        compare([LINEAR], test, split="out_of_sample", fold=1, design_sample=train)


def test_a_fit_on_the_design_years_passes_the_check() -> None:
    sample = _annual(_noisy(), trend=LinearTrend(fit_years=YEARS[:28]))
    train, test = holdout(sample, 0.7)
    table = evaluate(LINEAR, test, split="out_of_sample", design_sample=train)
    assert table.column("split").to_pylist() == ["out_of_sample"]


def test_an_evaluation_sample_fitted_on_its_own_years_is_refused() -> None:
    values = _noisy()
    design_block = _annual(values).select(YEARS[:28])
    evaluated = _annual(values, trend=LinearTrend()).select(YEARS[28:])
    with pytest.raises(LeakageError, match="evaluation sample's linear trend"):
        evaluate(LINEAR, evaluated, split="out_of_sample", design_sample=design_block)


def test_the_lineage_leaves_the_trend_out() -> None:
    values = _noisy()
    adjusted = _annual(values, trend=LinearTrend())
    recorded = _annual(values)
    assert adjusted.lineage == recorded.lineage
    with pytest.raises(LeakageError, match="shares 40 periods"):
        evaluate(LINEAR, recorded, split="out_of_sample", design_sample=adjusted)


def test_a_block_of_another_source_is_no_leak() -> None:
    other: dict[str, Any] = {"trend": LinearTrend()}
    design_block = _annual(_noisy(), **other)
    series = _series(_noisy(2), name="another series")
    evaluated = sample_annual(
        series,
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="site-a",
        trend=LinearTrend(),
    )
    table = evaluate(
        LINEAR, evaluated, split="out_of_sample", design_sample=design_block
    )
    assert table.num_rows == 1


# -- the provenance ------------------------------------------------------------


def test_a_split_and_a_selection_carry_the_trend() -> None:
    sample = _annual(_noisy(), trend=LinearTrend())
    train, test = holdout(sample, 0.7)
    for block in (train, test, sample.select(YEARS[:5])):
        assert block.provenance.trend == sample.provenance.trend


def test_a_written_sample_keeps_its_trend() -> None:
    sample = _annual(_noisy(), trend=LinearTrend())
    read = OutcomeSample.from_written_table(sample.to_table())
    assert read.provenance.to_document() == sample.provenance.to_document()


def test_a_written_sample_from_before_the_field_reads_as_no_record() -> None:
    sample = _annual(_noisy())
    document = sample.provenance.to_document()
    del document["trend"]
    assert SampleProvenance(**document).trend is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"trend": "as_recorded"}, "trend must be a mapping"),
        ({"trend": {"n": 3}}, "trend must name its kind"),
        ({"trend": {"kind": "x", "fit_years": [float("nan")]}}, "plain JSON"),
    ],
)
def test_an_invalid_trend_record_raises(fields: Any, message: str) -> None:
    arguments: Any = {
        "source_kind": "annual",
        "sources": (),
        "semantics_source": "data",
        **fields,
    }
    with pytest.raises(SampleError, match=message):
        SampleProvenance(**arguments)


def test_monte_carlo_and_table_samples_record_no_trend() -> None:
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=HAZARD,
        creation_version="1",
        probability_semantics="annual_exceedance",
    )
    drawn = sample_curve(
        FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=12.0),
        metadata=metadata,
        n_draws=5,
        rng=np.random.default_rng(1),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="site-a",
    )
    assert drawn.provenance.trend is None
    written = evaluate(LINEAR, drawn, split="in_sample")
    assert EvaluationMetadata.from_table(written).setup.samples[0]["trend"] is None
    table = OutcomeSample.from_table(
        pa.table(
            {
                "period": [1, 2],
                "asset_id": ["a", "a"],
                "index": [1.0, 2.0],
                "loss": [0.0, 1.0],
            }
        ),
        period="period",
        asset_id="asset_id",
        index="index",
        loss="loss",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
    )
    assert table.provenance.trend is None
    written = evaluate(LINEAR, table, split="in_sample")
    assert EvaluationMetadata.from_table(written).setup.samples[0]["trend"] is None


# -- join ----------------------------------------------------------------------


def _part(asset: str, trend: Any, seed: int = 1) -> OutcomeSample:
    return sample_annual(
        _series(_noisy(seed), name=f"series {asset}"),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id=asset,
        trend=trend,
    )


def test_join_needs_one_kind_of_trend() -> None:
    with pytest.raises(SampleError, match="trend's kind of asset 'b'"):
        OutcomeSample.join([_part("a", LinearTrend()), _part("b", AsRecorded())])


def test_join_needs_one_reference_year() -> None:
    with pytest.raises(SampleError, match="trend's reference year of asset 'b'"):
        OutcomeSample.join([_part("a", LinearTrend()), _part("b", LinearTrend(2010))])


def test_join_needs_the_same_fit_years() -> None:
    first = LinearTrend(fit_years=YEARS[:30])
    second = LinearTrend(fit_years=YEARS[:29])
    with pytest.raises(SampleError, match="trend's fit years of asset 'b'"):
        OutcomeSample.join([_part("a", first), _part("b", second)])


def test_join_shares_the_basis_and_names_each_part_s_fit() -> None:
    a, b = _part("a", LinearTrend(), seed=1), _part("b", LinearTrend(), seed=2)
    joined = OutcomeSample.join([a, b])
    trend = joined.provenance.to_document()["trend"]
    assert trend["kind"] == "linear_trend"
    assert trend["reference_year"] == 2020
    assert trend["fit_years"] == YEARS.tolist()
    assert [fit["asset_id"] for fit in trend["fits"]] == ["a", "b"]
    assert joined.asset("a").provenance.trend == a.provenance.trend
    assert joined.asset("b").provenance.trend == b.provenance.trend
    assert "trend" not in joined.provenance.details


def test_a_held_out_block_of_a_join_fitted_on_all_years_is_refused() -> None:
    a, b = _part("a", LinearTrend(), seed=1), _part("b", LinearTrend(), seed=2)
    train, test = holdout(OutcomeSample.join([a, b]), 0.7)
    with pytest.raises(LeakageError, match="fit_years"):
        evaluate(
            LINEAR,
            test.asset("a"),
            split="out_of_sample",
            design_sample=train.asset("a"),
        )


def _joint_templates() -> list[Cover]:
    return [
        Cover(
            INDEX,
            StepPayout([60.0, 90.0, 120.0], [Free(), Free(), Free()]),
            payout_unit="EUR",
            asset_id=asset,
            label="step",
        )
        for asset in ("a", "b")
    ]


def test_a_joint_comparison_states_each_part_s_fit() -> None:
    fit = LinearTrend(fit_years=YEARS[:28])
    joined = OutcomeSample.join([_part("a", fit, seed=1), _part("b", fit, seed=2)])
    train, test = holdout(joined, 0.7)
    designed = design_joint(_joint_templates(), train, MSE())
    table = compare([designed], test, split="out_of_sample")
    line = _text_lines(setup_text(EvaluationMetadata.from_table(table)), "Trend")
    assert len(line) == 1
    assert "fitted on 28 years (1981 to 2008), reference year 2020" in line[0]
    assert "hazard of a slope" in line[0] and "hazard of b slope" in line[0]


def test_a_joint_design_fitted_on_all_years_is_refused_out_of_sample() -> None:
    parts = [_part("a", LinearTrend(), seed=1), _part("b", LinearTrend(), seed=2)]
    train, test = holdout(OutcomeSample.join(parts), 0.7)
    designed = design_joint(_joint_templates(), train, MSE())
    with pytest.raises(LeakageError, match="fit_years"):
        compare([designed], test, split="out_of_sample")


def test_join_of_recent_years_keeps_what_the_parts_state_alike() -> None:
    joined = OutcomeSample.join(
        [_part("a", RecentYears(10)), _part("b", RecentYears(10))]
    )
    assert joined.provenance.to_document()["trend"] == {
        "kind": "recent_years",
        "n": 10,
        "years": [2011, 2020],
    }


def test_join_of_as_recorded_parts_records_as_recorded() -> None:
    joined = OutcomeSample.join([_part("a", AsRecorded()), _part("b", AsRecorded())])
    assert joined.provenance.trend == {"kind": "as_recorded"}


def test_a_part_with_no_record_joins_an_as_recorded_part() -> None:
    table = OutcomeSample.from_table(
        pa.table(
            {
                "period": YEARS.astype("int32"),
                "asset_id": ["b"] * 40,
                "index": _noisy(3),
                "loss": _noisy(4) * 0.0 + 1.0,
            }
        ),
        period="period",
        asset_id="asset_id",
        index="index",
        loss="loss",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
    )
    joined = OutcomeSample.join([_part("a", AsRecorded()), table])
    assert joined.provenance.trend == {"kind": "as_recorded"}


# -- scenario= -----------------------------------------------------------------


def test_scenario_labels_the_rows_and_stays_out_of_the_lineage() -> None:
    values = _noisy()
    labelled = _annual(values, scenario="warm world")
    plain = _annual(values)
    assert plain.scenario == "baseline"
    assert labelled.scenario == "warm world"
    assert labelled.to_table().column("scenario").to_pylist()[0] == "warm world"
    assert labelled.content_hash != plain.content_hash
    assert labelled.lineage == plain.lineage
    with pytest.raises(LeakageError, match="shares 40 periods"):
        evaluate(LINEAR, plain, split="out_of_sample", design_sample=labelled)


def test_scenario_reaches_the_comparison() -> None:
    table = evaluate(
        LINEAR, _annual(_noisy(), scenario="warm world"), split="in_sample"
    )
    assert table.column("scenario").to_pylist() == ["warm world"]


@pytest.mark.parametrize("scenario", ["", " padded ", None, 3])
def test_an_invalid_scenario_raises(scenario: Any) -> None:
    with pytest.raises(SampleError, match="scenario must be"):
        _annual(scenario=scenario)


# -- the setup text ------------------------------------------------------------


def _setup(sample: OutcomeSample) -> str:
    return setup_text(
        EvaluationMetadata.from_table(evaluate(LINEAR, sample, split="in_sample"))
    )


def test_as_recorded_has_its_sentence() -> None:
    assert _text_lines(_setup(_annual(_noisy())), "Trend") == [
        "Trend: the years are used as recorded; no trend is removed."
    ]


def test_recent_years_have_their_sentence() -> None:
    assert _text_lines(_setup(_annual(_noisy(), trend=RecentYears(10))), "Trend") == [
        "Trend: only the last 10 years of the series are kept (2011 to 2020), as "
        "recorded; no value changes."
    ]


def test_a_linear_trend_states_fit_years_slope_intercept_and_reference_year() -> None:
    sample = _annual(
        trend=LinearTrend(fit_years=[*range(1981, 1989), *range(1997, 2021)])
    )
    assert _text_lines(_setup(sample), "Trend") == [
        "Trend: an additive linear trend in the mean is removed, by ordinary least "
        "squares on the year, fitted on 32 years (1981 to 1988 and 1997 to 2020), "
        "reference year 2020, where each intercept is the fitted value: hazard "
        "slope 0.5 mm/day per year, intercept 69.5 mm/day."
    ]


def test_a_linear_trend_states_each_series() -> None:
    sample = _annual(
        _line(),
        index=_series(20.0 + 0.2 * (YEARS - 1981.0), unit="m", name="gauge"),
        trend=LinearTrend(),
    )
    table = compare(
        [
            Cover(
                IndexDefinition("rx1day", "m", aggregation="annual_maximum"),
                LinearPayout(10.0, 30.0, 5e5),
                payout_unit="EUR",
                label="gauge",
            )
        ],
        sample,
        split="in_sample",
    )
    line = _text_lines(setup_text(EvaluationMetadata.from_table(table)), "Trend")[0]
    assert "hazard slope 0.5 mm/day per year, intercept 69.5 mm/day" in line
    assert "index slope 0.2 m per year, intercept 27.8 m" in line


AS_RECORDED_LINE = "Trend: the years are used as recorded; no trend is removed."


def test_each_setup_sample_holds_the_record_under_trend() -> None:
    sample = _annual(_noisy(), trend=LinearTrend())
    table = evaluate(LINEAR, sample, split="in_sample")
    entry = EvaluationMetadata.from_table(table).setup.samples[0]
    assert entry["trend"] == sample.provenance.to_document()["trend"]
    assert entry["trend"]["kind"] == "linear_trend"
    assert "trend" not in entry["details"]
    assert "trend" not in sample.provenance.details


def _written_before_the_key(table: Any, tmp_path: Path) -> Any:
    """`table` as written before a setup sample had `trend`, read from its file."""
    key = EVALUATION_METADATA_KEY.encode()
    document = json.loads(table.schema.metadata[key])
    for sample in document["setup"]["samples"]:
        del sample["trend"]
    older = table.replace_schema_metadata(
        {**table.schema.metadata, key: json.dumps(document).encode()}
    )
    path = tmp_path / "older.parquet"
    pq.write_table(older, path)
    return pq.read_table(path)


def test_a_comparison_written_before_the_key_reads_as_null(tmp_path: Path) -> None:
    table = evaluate(LINEAR, _annual(_noisy()), split="in_sample")
    older = _written_before_the_key(table, tmp_path)
    metadata = EvaluationMetadata.from_table(older)
    assert "trend" not in metadata.setup.samples[0]
    assert metadata.setup.samples[0].get("trend") is None
    assert _text_lines(setup_text(metadata), "Trend") == [AS_RECORDED_LINE]
    assert _text_lines(setup_text(older), "Trend") == [AS_RECORDED_LINE]
    document = json.loads(metadata.to_json())
    assert _text_lines(setup_text(document), "Trend") == [AS_RECORDED_LINE]
    assert _text_lines(setup_text(document["setup"]), "Trend") == [AS_RECORDED_LINE]


def test_a_sample_table_written_before_the_field_reads_as_null() -> None:
    table = _annual(_noisy()).to_table()
    key = SAMPLE_METADATA_KEY.encode()
    document = json.loads(table.schema.metadata[key])
    del document["provenance"]["trend"]
    read = OutcomeSample.from_written_table(
        table.replace_schema_metadata({key: json.dumps(document).encode()})
    )
    assert read.provenance.trend is None
    written = evaluate(LINEAR, read, split="in_sample")
    metadata = EvaluationMetadata.from_table(written)
    assert metadata.setup.samples[0]["trend"] is None
    assert _text_lines(setup_text(metadata), "Trend") == [AS_RECORDED_LINE]


def test_a_record_in_details_is_not_read() -> None:
    setup = {
        "basis": "annual",
        "metric_options": {},
        "samples": [
            {
                "sample_hash": "x",
                "roles": [],
                "source_kind": "annual",
                "details": {"trend": {"kind": "recent_years", "n": 7}},
            }
        ],
        "evaluations": [],
    }
    assert _text_lines(setup_text(setup), "Trend") == [AS_RECORDED_LINE]


def test_the_setup_text_reads_the_trend_key() -> None:
    metadata = EvaluationMetadata.from_table(
        evaluate(LINEAR, _annual(_noisy()), split="in_sample")
    )
    document = json.loads(metadata.to_json())
    document["setup"]["samples"][0]["trend"] = {"kind": "recent_years", "n": 7}
    assert _text_lines(setup_text(document["setup"]), "Trend") == [
        "Trend: only the last 7 years of the series are kept, as recorded; no "
        "value changes."
    ]


def test_curve_samples_have_no_trend_line() -> None:
    setup = {
        "basis": "annual",
        "metric_options": {},
        "samples": [{"sample_hash": "x", "roles": [], "source_kind": "monte_carlo"}],
        "evaluations": [],
    }
    assert _text_lines(setup_text(setup), "Trend") == []


def test_samples_that_differ_are_named_by_asset() -> None:
    a, b = _part("a", LinearTrend(), seed=1), _part("b", LinearTrend(), seed=2)
    tables = [evaluate(LINEAR, sample, split="in_sample") for sample in (a, b)]
    merged = EvaluationMetadata.merge(
        *(EvaluationMetadata.from_table(table) for table in tables)
    )
    lines = _text_lines(setup_text(merged), "Trend")
    assert [line.split(":")[0] for line in lines] == ["Trend of a", "Trend of b"]


def test_a_joined_sample_states_each_part_s_fit() -> None:
    a, b = _part("a", LinearTrend(), seed=1), _part("b", LinearTrend(), seed=2)
    joined = OutcomeSample.join([a, b])
    document = joined.provenance.to_document()
    setup = {
        "basis": "annual",
        "metric_options": {},
        "samples": [
            {
                "sample_hash": "x",
                "roles": ["evaluation"],
                "source_kind": "annual",
                "trend": document["trend"],
            }
        ],
        "evaluations": [],
    }
    line = _text_lines(setup_text(setup), "Trend")[0]
    assert "hazard of a slope" in line
    assert "hazard of b slope" in line


@pytest.mark.parametrize(
    ("trend", "wrong"),
    [
        ({"kind": "recent_years", "n": "ten"}, r"\.n has the wrong type"),
        (
            {"kind": "linear_trend", "fit_years": [2000], "reference_year": "x"},
            r"\.reference_year has the wrong type",
        ),
        (
            {
                "kind": "linear_trend",
                "fit_years": [2000],
                "reference_year": 2000,
                "fits": [{"slope": "steep", "intercept": 1.0}],
            },
            r"\.slope must be a number",
        ),
    ],
)
def test_a_trend_record_of_the_wrong_type_raises(trend: Any, wrong: str) -> None:
    setup = {
        "basis": "annual",
        "metric_options": {},
        "samples": [
            {"sample_hash": "x", "roles": [], "source_kind": "annual", "trend": trend}
        ],
        "evaluations": [],
    }
    with pytest.raises(EvaluationError, match=wrong):
        setup_text(setup)


def test_an_unknown_trend_kind_is_stated_as_it_is() -> None:
    setup = {
        "basis": "annual",
        "metric_options": {},
        "samples": [
            {
                "sample_hash": "x",
                "roles": [],
                "source_kind": "annual",
                "trend": {"kind": "quadratic"},
            }
        ],
        "evaluations": [],
    }
    assert _text_lines(setup_text(setup), "Trend") == ["Trend: quadratic, as stated."]


# -- the plan ------------------------------------------------------------------


def _grid(years: Any = YEARS, seed: int = 1, rise: float = 0.4) -> Any:
    """A per-cell annual table in the layout of ERA5's and CMIP6's."""
    rng = np.random.default_rng(seed)
    count = len(years) * len(CELLS)
    return pa.table(
        {
            "year": pa.array(np.repeat(years, len(CELLS)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in CELLS], len(years)),
            "longitude": np.tile([lon for _, lon in CELLS], len(years)),
            "value": rng.gumbel(60.0, 12.0, count)
            + rise * (np.repeat(years, len(CELLS)) - 1981.0),
        }
    )


def _changed(table: Any, year: int, by: float) -> Any:
    years = table.column("year").to_numpy()
    values = table.column("value").to_numpy().copy()
    values[years == year] += by
    return table.set_column(
        table.column_names.index("value"), "value", pa.array(values, pa.float64())
    )


def _assets(*ids: str) -> Any:
    ids = ids or ("a", "b")
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [POINTS[i][0] for i in ids],
            "latitude": [POINTS[i][1] for i in ids],
            "value_eur": [2e6 + 1e5 * k for k in range(len(ids))],
            "station_id": ["S1" for _ in ids],
        }
    )


def _plan(source: Any = None, assets: Any = None, **options: Any) -> ParametricPlan:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "provenance": HAZARD,
        "value_column": "value_eur",
        "value_unit": "EUR",
        "seed": 42,
    }
    arguments.update(options)
    plan: ParametricPlan = UNTYPED_PLAN.for_annual_extremes(
        _grid() if source is None else source,
        _assets() if assets is None else assets,
        **arguments,
    )
    return plan


TEMPLATE = Cover(
    INDEX,
    StepPayout([60.0, 90.0, 120.0], [Free(), Free(), Free()]),
    payout_unit="EUR",
    label="step",
)


def _designed(plan: ParametricPlan) -> ParametricPlan:
    return (
        plan.basis_risk(NoBasisRisk())
        .covers(TEMPLATE, LINEAR)
        .objective(MSE())
        .metrics(OPTIONS)
    )


def test_the_plan_defaults_as_recorded_and_baseline() -> None:
    plan = _plan()
    assert plan.trend == AsRecorded()
    assert plan.scenario == "baseline"
    sample = plan.basis_risk(NoBasisRisk()).samples()["a"]
    assert sample.provenance.trend == {"kind": "as_recorded"}
    assert sample.scenario == "baseline"


def test_the_plan_refuses_stated_fit_years() -> None:
    with pytest.raises(ValueError, match="plan sets them per fold"):
        _plan(trend=LinearTrend(fit_years=YEARS[:20]))


def test_the_plan_refuses_another_type_of_trend() -> None:
    with pytest.raises(TypeError, match="trend must be"):
        _plan(trend="recent")


def test_the_plan_refuses_an_invalid_scenario() -> None:
    with pytest.raises(SampleError, match="scenario must be"):
        _plan(scenario=" ")


def test_explain_states_the_option_and_the_scenario_before_anything_is_read() -> None:
    plan = _plan(trend=LinearTrend(2015, min_years=12), scenario="warm world")
    document = plan.basis_risk(NoBasisRisk()).explain(format="json")
    assert isinstance(document, dict)
    samples = document["setup"]["samples"]
    assert samples[0]["trend"] == {
        "kind": "linear_trend",
        "method": "ols_additive",
        "reference_year": 2015,
        "min_years": 12,
        "fit_years": None,
    }
    assert samples[0]["scenario"] == "warm world"
    text = setup_text(document)
    assert "Scenario: warm world." in text
    assert _text_lines(text, "Trend") == [
        "Trend: an additive linear trend in the mean is removed, by ordinary least "
        "squares on the year, with reference year 2015 and at least 12 fit years; "
        "the fit years, slope and intercept are not known yet."
    ]


@pytest.mark.parametrize(
    ("trend", "line"),
    [
        (AsRecorded(), "the years are used as recorded"),
        (RecentYears(5), "only the last 5 years of the series are kept, as recorded"),
        (LinearTrend(), "with reference year the last year of the series"),
    ],
)
def test_explain_states_each_option(trend: Any, line: str) -> None:
    plan = _plan(trend=trend).basis_risk(NoBasisRisk())
    assert line in _text_lines(setup_text(plan.explain(format="json")), "Trend")[0]


def test_samples_scenario_and_all_years_fit() -> None:
    samples = (
        _plan(trend=LinearTrend(), scenario="warm world")
        .basis_risk(NoBasisRisk())
        .split(Holdout(0.7))
        .samples()
    )
    for sample in samples.values():
        assert sample.scenario == "warm world"
        assert sample.provenance.trend["fit_years"] == tuple(YEARS)  # type: ignore[index]
        assert len(sample) == 40


def test_recent_years_on_the_plan_come_after_years() -> None:
    samples = (
        _plan(years=list(range(1990, 2011)), trend=RecentYears(10))
        .basis_risk(NoBasisRisk())
        .samples()
    )
    assert samples["a"].period.tolist() == list(range(2001, 2011))


def test_recent_years_on_the_plan_are_not_rebuilt_per_fold() -> None:
    plan = _designed(_plan(trend=RecentYears(20))).split(Holdout(0.75))
    designs = plan.designs()
    assert len(designs) == 2
    samples = plan.samples()
    train, _ = holdout(samples["a"], 0.75)
    assert designs[0].design_sample.content_hash == train.content_hash


def test_a_holdout_fits_on_the_design_years_only(tmp_path: Path) -> None:
    plan = _designed(_plan(trend=LinearTrend())).split(Holdout(0.7))
    result = plan.write_parquet(tmp_path / "out.parquet")
    table = result.table
    assert set(table.column("split").to_pylist()) == {"in_sample", "out_of_sample"}
    samples = result.metadata.setup.samples
    fit = [entry["trend"]["fit_years"] for entry in samples]
    assert all(years == YEARS[:28].tolist() for years in fit)
    held_out = [entry for entry in samples if "design" not in entry["roles"]]
    assert all(entry["periods"] == [2009, 2020] for entry in held_out)
    assert (
        result.samples["a"].provenance.trend["fit_years"]  # type: ignore[index]
        == tuple(YEARS)
    )


def test_a_plan_s_rows_equal_a_per_fold_sample_built_by_hand(tmp_path: Path) -> None:
    source = _grid()
    plan = _designed(_plan(source, _assets("a"), trend=LinearTrend())).split(
        Holdout(0.7)
    )
    written = plan.write_parquet(tmp_path / "out.parquet")
    series = AnnualSeries.nearest_cell(
        source, longitude=6.96, latitude=50.94, unit="mm/day", source=HAZARD
    )
    loss = LossModel(RAMP, value=2e6, value_unit="EUR")
    by_hand = sample_annual(
        series,
        loss_model=loss,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        trend=LinearTrend(fit_years=YEARS[:28]),
    )
    train, test = holdout(by_hand, 0.7)
    hashes = {entry["sample_hash"] for entry in written.metadata.setup.samples}
    assert {train.content_hash, test.content_hash} <= hashes


def test_kfold_fits_each_fold_on_its_own_design_years(tmp_path: Path) -> None:
    plan = _designed(_plan(trend=LinearTrend())).split(KFold(4))
    result = plan.write_parquet(tmp_path / "out.parquet")
    designs = [e for e in result.metadata.setup.samples if "design" in e["roles"]]
    fits = {tuple(e["trend"]["fit_years"]) for e in designs}
    blocks = np.array_split(YEARS, 4)
    expected = {
        tuple(int(year) for year in YEARS if year not in block) for block in blocks
    }
    assert fits == expected
    reference = {e["trend"]["reference_year"] for e in designs}
    assert reference == {2020}


def test_each_fold_s_setup_text_states_its_own_fit(tmp_path: Path) -> None:
    plan = _designed(_plan(assets=_assets("a"), trend=LinearTrend())).split(KFold(4))
    result = plan.write_parquet(tmp_path / "out.parquet")
    lines = _text_lines(setup_text(result.metadata), "Trend")
    assert len(lines) == 4
    assert all(line.startswith("Trend of a: an additive") for line in lines)


@pytest.mark.parametrize("rule", [Holdout(0.7), KFold(4)])
def test_a_changed_held_out_year_leaves_the_design_unchanged(rule: Any) -> None:
    source = _grid()
    plan = _designed(_plan(source, _assets("a"), trend=LinearTrend())).split(rule)
    # Under Holdout(0.7) the held-out years are 2009-2020; a k-fold's last fold's
    # design block is 1981-2010, so the year changed is held out there.
    held_out_year = 2015
    designs = plan.designs()
    other = _designed(
        _plan(_changed(source, held_out_year, 500.0), _assets("a"), trend=LinearTrend())
    ).split(rule)
    changed = other.designs()
    fold_of_last_block = -1
    assert (
        designs[fold_of_last_block].design_id == changed[fold_of_last_block].design_id
    )
    assert (
        designs[fold_of_last_block].design_sample.content_hash
        == changed[fold_of_last_block].design_sample.content_hash
    )
    assert designs[fold_of_last_block].cover.definition_hash == (
        changed[fold_of_last_block].cover.definition_hash
    )


def test_a_changed_held_out_year_leaves_a_layer_one_design_unchanged() -> None:
    values = _noisy()
    other = values.copy()
    other[-1] += 500.0
    trend = LinearTrend(fit_years=YEARS[:28])
    first, _ = holdout(_annual(values, trend=trend), 0.7)
    second, held_out = holdout(_annual(other, trend=trend), 0.7)
    assert first.content_hash == second.content_hash
    assert design(TEMPLATE, first, MSE()).design_id == (
        design(TEMPLATE, second, MSE()).design_id
    )
    assert first.provenance.trend == second.provenance.trend
    assert held_out.index[-1, 0] != first.index[-1, 0]


def test_a_changed_held_out_year_moves_only_itself() -> None:
    values = _noisy()
    other = values.copy()
    other[-1] += 500.0
    trend = LinearTrend(fit_years=YEARS[:28])
    moved = _annual(other, trend=trend).index[:, 0]
    first = _annual(values, trend=trend).index[:, 0]
    np.testing.assert_array_equal(moved[:-1], first[:-1])
    assert moved[-1] == pytest.approx(first[-1] + 500.0)


def test_a_fold_with_too_few_design_years_raises_with_its_fold_and_count() -> None:
    plan = _designed(
        _plan(
            _grid(np.arange(2001, 2021)),
            trend=LinearTrend(min_years=14),
        )
    ).split(Holdout(0.5))
    with pytest.raises(
        SampleError,
        match=r"asset 'a', fold 0: LinearTrend needs at least 14 fit years "
        r"\(min_years\), and the design block holds 10",
    ):
        plan.designs()


def test_samples_do_not_look_at_the_folds() -> None:
    plan = _designed(
        _plan(_grid(np.arange(2001, 2021)), trend=LinearTrend(min_years=14))
    ).split(Holdout(0.5))
    assert len(plan.samples()["a"]) == 20


def test_every_fold_starts_from_the_same_generator_state(tmp_path: Path) -> None:
    source = _grid()
    plan = (
        _plan(source, _assets("a"), trend=LinearTrend())
        .basis_risk(ImpactNoise(0.3))
        .covers(LINEAR)
        .metrics(OPTIONS)
        .split(KFold(4))
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    states = {record["state"] for record in result.metadata.rng}
    assert len(states) == 1
    assert json.loads(next(iter(states)))["bit_generator"] == "PCG64"


def test_a_noise_factor_does_not_depend_on_the_fold() -> None:
    source = _grid()
    plan = (
        _plan(source, _assets("a"), trend=LinearTrend(), seed=11)
        .basis_risk(ImpactNoise(0.3))
        .covers(TEMPLATE)
        .objective(MSE())
        .split(KFold(4))
    )
    designs = plan.designs()
    ratios: dict[int, set[float]] = {}
    series = AnnualSeries.nearest_cell(
        source, longitude=6.96, latitude=50.94, unit="mm/day", source=HAZARD
    )
    all_years = sample_annual(
        series,
        loss_model=LossModel(RAMP, value=2e6, value_unit="EUR"),
        basis_risk=NoBasisRisk(),
        asset_id="a",
    )
    assert len(all_years) == 40
    for result in designs:
        sample = result.design_sample
        impact = LossModel(RAMP, value=2e6, value_unit="EUR")
        hazard = sample.hazard if sample.hazard is not None else sample.index
        base = impact.loss_values(impact.impact_values(hazard[:, 0]))
        for year, loss, zero in zip(
            sample.period, sample.loss[:, 0], base, strict=True
        ):
            if zero > 0:
                ratios.setdefault(int(year), set()).add(round(float(loss / zero), 9))
    assert ratios
    assert all(len(values) == 1 for values in ratios.values())


def test_the_comonotone_warning_is_not_repeated_per_fold(tmp_path: Path) -> None:
    plan = _designed(_plan(assets=_assets("a"), trend=LinearTrend())).split(KFold(4))
    with pytest.warns(ComonotoneIndexWarning) as caught:
        plan.write_parquet(tmp_path / "out.parquet")
    assert len([w for w in caught if w.category is ComonotoneIndexWarning]) == 1


def test_the_plan_fits_hazard_and_index_separately(tmp_path: Path) -> None:
    stations = pa.table(
        {
            "station": ["S1"] * 40,
            "year": pa.array(YEARS, pa.int32()),
            "value": _noisy(5),
        }
    )
    plan = (
        _plan(assets=_assets("a"), trend=LinearTrend())
        .index(
            stations,
            location=StationColumn("station_id"),
            unit="mm/day",
            provenance=GAUGES,
        )
        .basis_risk(NoBasisRisk())
    )
    sample = plan.samples()["a"]
    document = sample.provenance.to_document()["trend"]
    assert [fit["role"] for fit in document["fits"]] == ["hazard", "index"]
    assert document["reference_year"] == 2020
    assert any("read at station S1" in note for note in sample.provenance.notes)
    written = (plan.covers(LINEAR).metrics(OPTIONS).split(Holdout(0.7))).write_parquet(
        tmp_path / "out.parquet"
    )
    for entry in written.metadata.setup.samples:
        assert [f["role"] for f in entry["trend"]["fits"]] == [
            "hazard",
            "index",
        ]
        assert any("read at station S1" in note for note in entry["notes"])


def test_the_index_is_adjusted_on_the_same_years_and_reference_year() -> None:
    table = _grid()
    index = _grid(seed=9)
    plan = (
        _plan(table, _assets("a"), trend=LinearTrend())
        .index(index, location=NearestCell(), unit="mm/day", provenance=GAUGES)
        .basis_risk(NoBasisRisk())
    )
    sample = plan.samples()["a"]
    series = AnnualSeries.nearest_cell(
        table, longitude=6.96, latitude=50.94, unit="mm/day", source=HAZARD
    )
    other = AnnualSeries.nearest_cell(
        index, longitude=6.96, latitude=50.94, unit="mm/day", source=GAUGES
    )
    by_hand = sample_annual(
        series,
        index=other,
        loss_model=LossModel(RAMP, value=2e6, value_unit="EUR"),
        basis_risk=NoBasisRisk(),
        asset_id="a",
        trend=LinearTrend(),
    )
    assert sample.content_hash == by_hand.content_hash


def test_the_plan_written_scenario_is_in_every_row(tmp_path: Path) -> None:
    plan = _designed(_plan(scenario="warm world")).split(Holdout(0.7))
    result = plan.write_parquet(tmp_path / "out.parquet")
    assert set(result.table.column("scenario").to_pylist()) == {"warm world"}
    read = pq.read_table(tmp_path / "out.parquet")
    metadata = json.loads(read.schema.metadata[EVALUATION_METADATA_KEY.encode()])
    assert {s["scenario"] for s in metadata["setup"]["samples"]} == {"warm world"}


def test_a_plan_as_recorded_keeps_its_rows(tmp_path: Path) -> None:
    source = _grid()
    without = _designed(_plan(source)).split(Holdout(0.7))
    stated = _designed(_plan(source, trend=AsRecorded())).split(Holdout(0.7))
    first = without.write_parquet(tmp_path / "a.parquet").table
    second = stated.write_parquet(tmp_path / "b.parquet").table
    assert first.drop_columns(["cover_hash"]).equals(
        second.drop_columns(["cover_hash"])
    )
