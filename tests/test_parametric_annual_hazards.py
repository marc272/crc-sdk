"""`sample_annual_hazards`: the same years of several hazards at one asset.

The numbers are synthetic and chosen so that sums can be read off. The tests
check what the module computes and records; nothing here is a finding about a
hazard, a rule or a cover form.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from numpy.typing import NDArray

from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AsRecorded,
    CappedSum,
    ComonotoneIndexWarning,
    CompositeIndex,
    CompoundDamage,
    Cover,
    CoverPackage,
    Free,
    ImpactNoise,
    IndexDefinition,
    LeakageError,
    LinearTrend,
    LossModel,
    MaxOf,
    NoBasisRisk,
    OutcomeSample,
    RecentYears,
    SampleError,
    StepPayout,
    SumOf,
    compare,
    design,
    holdout,
    sample_annual,
    sample_annual_hazards,
    setup_text,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
)

UNTYPED_HAZARDS: Any = sample_annual_hazards

YEARS = list(range(2000, 2008))
RAIN = [70.0, 10.0, 30.0, 90.0, 20.0, 60.0, 5.0, 40.0]
HEAT = [60.0, 20.0, 80.0, 50.0, 10.0, 70.0, 15.0, 30.0]
RAIN_INDEX = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT_INDEX = IndexDefinition("heat", "degC", aggregation="annual_maximum")
NOISE = ImpactNoise(0.3, upper=1.0)


def _hundredth(hazard: NDArray[np.float64]) -> NDArray[np.float64]:
    return hazard / 100.0


def _ramp(name: str = "ramp", **fields: Any) -> LossModel:
    arguments: dict[str, Any] = {"value": 1000.0, "value_unit": "EUR", **fields}
    return LossModel(_hundredth, name=name, **arguments)


def _series(
    values: list[float | None],
    *,
    unit: str = "mm/day",
    first: int = 2000,
    **provenance: Any,
) -> AnnualSeries:
    data = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    return AnnualSeries(np.arange(first, first + len(values)), data, unit, provenance)


def _two_series() -> dict[str, AnnualSeries]:
    return {"rain": _series([*RAIN]), "heat": _series([*HEAT], unit="degC")}


TRENDED_RAIN = [50.0, 55.0, 52.0, 60.0, 58.0, 65.0, 63.0, 70.0]
TRENDED_HEAT = [70.0, 65.0, 68.0, 60.0, 62.0, 55.0, 57.0, 50.0]
"""Far enough from zero that removing their trends leaves them positive."""


def _trended_series() -> dict[str, AnnualSeries]:
    return {
        "rain": _series([*TRENDED_RAIN]),
        "heat": _series([*TRENDED_HEAT], unit="degC"),
    }


def _sample(**overrides: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "series": _two_series(),
        "tails": {"rain": "upper", "heat": "upper"},
        "loss_models": {"rain": _ramp("rain-ramp"), "heat": _ramp("heat-ramp")},
        "combine": CappedSum(1.0),
        "basis_risk": NoBasisRisk(),
        "asset_id": "site",
    }
    arguments.update(overrides)
    series = arguments.pop("series")
    sample: OutcomeSample = UNTYPED_HAZARDS(series, **arguments)
    return sample


def _one_hazard(series: AnnualSeries, name: str, **overrides: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "loss_model": _ramp(name),
        "basis_risk": NoBasisRisk(),
        "asset_id": "site",
    }
    arguments.update(overrides)
    sample: OutcomeSample = sample_annual(series, **arguments)
    return sample


def _rain_cover() -> Cover:
    return Cover(RAIN_INDEX, StepPayout([50.0], [Free()]), payout_unit="EUR")


def _heat_cover() -> Cover:
    return Cover(HEAT_INDEX, StepPayout([50.0], [Free()]), payout_unit="EUR")


def _fixed_cover() -> Cover:
    return Cover(RAIN_INDEX, StepPayout([50.0], [800.0]), payout_unit="EUR", label="r")


def _text(sample: OutcomeSample) -> str:
    return setup_text(compare([_fixed_cover()], sample, split="in_sample"))


def _line(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(start))


ANNUAL_MAXIMA = (
    "Two annual maxima of one year may come from different events: the rows are "
    "the same years, not the same events."
)


# -- rows, the rule and the combined loss ----------------------------------------


def test_the_years_become_the_rows() -> None:
    sample = _sample()
    assert sample.asset_ids == ("site",)
    assert sample.period.tolist() == YEARS
    np.testing.assert_allclose(sample.probability, 1 / 8)
    assert sample.semantics == "annual"
    assert sample.loss_unit == "EUR"
    assert sample.scenario == "baseline"
    assert sample.provenance.source_kind == "annual"
    assert sample.provenance.semantics_source == "data"


def test_the_first_series_is_the_primary_and_the_order_of_series_sets_the_rest() -> (
    None
):
    sample = _sample()
    assert sample.index_components is not None
    assert list(sample.index_components) == ["rain", "heat"]
    assert sample.index[:, 0].tolist() == RAIN
    assert sample.index_unit == "mm/day"
    assert sample.hazard is None
    series = _two_series()
    swapped = _sample(series={"heat": series["heat"], "rain": series["rain"]})
    assert swapped.index_components is not None
    assert list(swapped.index_components) == ["heat", "rain"]
    assert swapped.index[:, 0].tolist() == HEAT
    assert swapped.index_unit == "degC"


def test_the_components_hold_their_values_units_and_tails() -> None:
    sample = _sample(tails={"rain": "upper", "heat": "lower"})
    components = sample.index_components
    assert components is not None
    assert components["rain"].values[:, 0].tolist() == RAIN
    assert components["heat"].values[:, 0].tolist() == HEAT
    assert (components["rain"].unit, components["rain"].tail) == ("mm/day", "upper")
    assert (components["heat"].unit, components["heat"].tail) == ("degC", "lower")
    assert components["rain"].hazard is None


def test_a_capped_sum_of_0_7_and_0_6_is_the_whole_value() -> None:
    sample = _sample(combine=CappedSum(1.0))
    assert sample.loss[0, 0] == 1000.0


def test_compound_damage_of_0_7_and_0_6_is_0_88_of_the_value() -> None:
    sample = _sample(combine=CompoundDamage())
    assert sample.loss[0, 0] == pytest.approx(880.0)


def test_the_rule_acts_on_impacts_before_the_value() -> None:
    capped = _sample(combine=CappedSum(0.5))
    np.testing.assert_allclose(
        capped.loss[:, 0], np.minimum((np.array(RAIN) + HEAT) / 100.0, 0.5) * 1000.0
    )
    compounded = _sample(combine=CompoundDamage())
    kept = (1.0 - np.array(RAIN) / 100.0) * (1.0 - np.array(HEAT) / 100.0)
    np.testing.assert_allclose(compounded.loss[:, 0], (1.0 - kept) * 1000.0)


def test_three_hazards_combine() -> None:
    series = {**_two_series(), "wind": _series([*HEAT], unit="m/s")}
    sample = _sample(
        series=series,
        tails=dict.fromkeys(series, "upper"),
        loss_models={name: _ramp(name) for name in series},
        combine=CompoundDamage(),
    )
    kept = (1 - np.array(RAIN) / 100) * (1 - np.array(HEAT) / 100) ** 2
    np.testing.assert_allclose(sample.loss[:, 0], (1 - kept) * 1000.0)


class Worst:
    """A user's rule: the larger impact of the year."""

    def to_document(self) -> dict[str, Any]:
        return {"kind": "worst_of", "version": 1}

    def statement(self) -> str:
        return "The largest impact of the year is taken"

    def combine(self, impacts: Mapping[str, Any]) -> Any:
        return np.stack(list(impacts.values())).max(axis=0)


def test_a_user_rule_makes_the_loss_and_is_recorded() -> None:
    sample = _sample(combine=Worst())
    np.testing.assert_allclose(
        sample.loss[:, 0], np.maximum(RAIN, HEAT) / 100.0 * 1000.0
    )
    assert sample.provenance.combine == {
        "kind": "worst_of",
        "version": 1,
        "statement": "The largest impact of the year is taken",
    }
    assert "Combined loss: The largest impact of the year is taken." in _text(sample)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: type(
                "Short",
                (Worst,),
                {"combine": lambda self, impacts: np.zeros(3)},
            )(),
            r"must return one value per year, shape \(8,\)",
            id="shape",
        ),
        pytest.param(
            lambda: type(
                "Bad",
                (Worst,),
                {"combine": lambda self, impacts: np.full(8, np.nan)},
            )(),
            "negative or non-finite",
            id="nan",
        ),
        pytest.param(
            lambda: type(
                "Claims",
                (Worst,),
                {"to_document": lambda self: {"kind": "capped_sum", "cap": 1.0}},
            )(),
            "kind of a built-in",
            id="built-in-kind",
        ),
        pytest.param(
            lambda: type("Mute", (Worst,), {"statement": lambda self: ""})(),
            "must state itself",
            id="no-sentence",
        ),
    ],
)
def test_a_rule_that_fails_the_gate_stops_the_sample(build: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _sample(combine=build())


def test_the_rule_is_recorded_with_its_sentence() -> None:
    assert _sample(combine=CappedSum(1.0)).provenance.combine == {
        "kind": "capped_sum",
        "cap": 1.0,
        "statement": CappedSum(1.0).statement(),
    }
    assert _sample(combine=CompoundDamage()).provenance.combine == {
        "kind": "compound_damage",
        "statement": CompoundDamage().statement(),
    }


# -- the rule is required, and only when it is needed ----------------------------


def test_two_impacts_without_a_rule_raise_naming_the_rules() -> None:
    with pytest.raises(
        SampleError,
        match=r"\['rain', 'heat'\] each have an impact.*none is assumed.*"
        r"CappedSum\(cap\).*CompoundDamage\(\).*CombineRule",
    ):
        _sample(combine=None)


def test_a_rule_with_one_impact_raises_rather_than_being_ignored() -> None:
    with pytest.raises(SampleError, match="only component 'heat' has one"):
        _sample(loss_models={"heat": _ramp()}, combine=CappedSum(1.0))


@pytest.mark.parametrize("models", [{}, None], ids=["empty", "none"])
def test_a_sample_with_no_impact_raises(models: Any) -> None:
    with pytest.raises(SampleError, match="would have no loss"):
        _sample(loss_models=models, combine=None)


@pytest.mark.parametrize(
    "rule",
    [lambda impacts: impacts["rain"], max, "capped_sum"],
    ids=["callable", "builtin", "string"],
)
def test_a_bare_callable_as_a_rule_is_a_type_error(rule: Any) -> None:
    with pytest.raises(TypeError, match="every rule has a document"):
        _sample(combine=rule)


def test_a_component_with_no_impact_needs_no_rule() -> None:
    sample = _sample(loss_models={"heat": _ramp("heat-ramp")}, combine=None)
    assert sample.index_components is not None
    assert sample.index_components["rain"].own_loss is None
    own_heat = sample.index_components["heat"].own_loss
    assert own_heat is not None
    np.testing.assert_allclose(sample.loss[:, 0], np.array(HEAT) * 10.0)
    np.testing.assert_allclose(own_heat[:, 0], np.array(HEAT) * 10.0)
    assert sample.index[:, 0].tolist() == RAIN
    assert sample.provenance.combine == {
        "kind": "one_impact",
        "component": "heat",
        "statement": (
            "Only component heat has an impact, so the combined loss is its own loss."
        ),
    }


# -- one value ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("second", "message"),
    [
        pytest.param(
            {"value": 2000.0},
            r"must have one value, but those of \['rain', 'heat'\] differ",
            id="value",
        ),
        pytest.param({"value_unit": "USD"}, "one value_unit", id="unit"),
    ],
)
def test_the_loss_models_must_have_one_value(second: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _sample(loss_models={"rain": _ramp(), "heat": _ramp(**second)})


def test_a_refused_value_is_not_in_the_message() -> None:
    with pytest.raises(SampleError) as caught:
        _sample(loss_models={"rain": _ramp(), "heat": _ramp(value=2345.0)})
    assert "2345" not in str(caught.value)


def test_the_loss_models_must_have_one_value_column() -> None:
    assets = pa.table({"asset_id": ["site"], "a": [10.0], "b": [10.0]})

    def bound(column: str) -> LossModel:
        return LossModel(_hundredth, value_column=column, value_unit="EUR").for_asset(
            assets, "site"
        )

    with pytest.raises(SampleError, match="one value_column"):
        _sample(loss_models={"rain": bound("a"), "heat": bound("b")})
    sample = _sample(loss_models={"rain": bound("a"), "heat": bound("a")})
    assert sample.loss[0, 0] == pytest.approx(10.0)


def test_an_unbound_loss_model_raises_before_any_draw() -> None:
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    unbound = LossModel(_hundredth, value_column="a", value_unit="EUR")
    with pytest.raises(SampleError, match="for_asset"):
        _sample(loss_models={"rain": _ramp(), "heat": unbound}, rng=rng)
    assert rng.bit_generator.state == before


def test_a_model_bound_to_another_asset_raises() -> None:
    assets = pa.table({"asset_id": ["other"], "a": [10.0]})
    model = LossModel(_hundredth, value_column="a", value_unit="EUR").for_asset(
        assets, "other"
    )
    with pytest.raises(SampleError, match="bound to asset 'other'"):
        _sample(loss_models={"rain": model, "heat": model})


# -- the same years ------------------------------------------------------------------


def test_mismatched_years_raise_listing_each_series() -> None:
    series = {
        "rain": _series([1.0, 2.0, 3.0, 4.0]),
        "heat": _series([1.0, 2.0, 3.0, 4.0], first=2001, unit="degC"),
    }
    with pytest.raises(
        SampleError,
        match=r"must hold the same years, so that a row is one year of every hazard; "
        r"they hold rain: 2000 to 2003; heat: 2001 to 2004$",
    ):
        _sample(series=series)


def test_mismatched_years_raise_under_drop_year_too() -> None:
    series = {
        "rain": _series([1.0, 2.0, 3.0]),
        "heat": _series([1.0, 2.0], unit="degC"),
    }
    with pytest.raises(SampleError, match="rain: 2000 to 2002; heat: 2000 to 2001"):
        _sample(series=series, missing="drop_year")


def test_a_gap_in_the_years_is_listed() -> None:
    gapped = AnnualSeries(
        np.array([2000, 2002, 2003]), np.array([1.0, 2.0, 3.0]), "mm/day", {}
    )
    series = {"rain": gapped, "heat": _series([1.0, 2.0, 3.0], unit="degC")}
    with pytest.raises(SampleError, match="rain: 2000 and 2002 to 2003; heat"):
        _sample(series=series)


def test_the_dependence_is_the_datas_own() -> None:
    assert _sample().provenance.hazard_dependence == {"kind": "data"}


def test_missing_years_raise_naming_each_series() -> None:
    series = {
        "rain": _series([1.0, None, 3.0, 4.0]),
        "heat": _series([1.0, 2.0, 3.0, None], unit="degC"),
    }
    with pytest.raises(
        SampleError,
        match=r"^2 years have no value \(rain \[2001\]; heat \[2003\]\); pass "
        r"missing='drop_year' to drop them, listed in the provenance$",
    ):
        _sample(series=series)


def test_a_year_missing_from_one_series_is_dropped_from_all_and_listed() -> None:
    series = {
        "rain": _series([10.0, None, 30.0, 40.0]),
        "heat": _series([10.0, 20.0, 30.0, None], unit="degC"),
    }
    sample = _sample(series=series, missing="drop_year")
    assert sample.period.tolist() == [2000, 2002]
    np.testing.assert_allclose(sample.probability, 0.5)
    assert sample.provenance.details["dropped_years"] == (2001, 2003)
    assert "2 years without a value were dropped: 2001, 2003." in (
        sample.provenance.notes
    )
    heat = sample.component("heat")
    assert heat.period.tolist() == [2000, 2002]
    assert heat.provenance.details["dropped_years"] == (2001, 2003)


def test_no_year_with_a_value_in_every_series_raises() -> None:
    series = {
        "rain": _series([1.0, None]),
        "heat": _series([None, 2.0], unit="degC"),
    }
    with pytest.raises(SampleError, match="no year has a value in every series"):
        _sample(series=series, missing="drop_year")


# -- the trend -----------------------------------------------------------------------


def test_recent_years_keeps_the_same_window_of_every_series() -> None:
    series = {
        "rain": _series([None, *RAIN[1:]]),
        "heat": _series([None, *HEAT[1:]], unit="degC"),
    }
    sample = _sample(series=series, trend=RecentYears(3))
    assert sample.period.tolist() == [2005, 2006, 2007]
    assert sample.index[:, 0].tolist() == RAIN[-3:]
    assert sample.provenance.trend == {
        "kind": "recent_years",
        "n": 3,
        "years": (2005, 2007),
    }


def test_a_linear_trend_fits_each_series_on_the_same_years() -> None:
    trend = LinearTrend(min_years=3)
    sample = _sample(series=_trended_series(), trend=trend)
    record = sample.provenance.trend
    assert record is not None
    assert [fit["component"] for fit in record["fits"]] == ["rain", "heat"]
    assert {fit["role"] for fit in record["fits"]} == {"hazard"}
    assert record["fit_years"] == tuple(YEARS)
    assert record["reference_year"] == 2007
    series = _trended_series()
    for name, fit in zip(("rain", "heat"), record["fits"], strict=True):
        alone = _one_hazard(series[name], name, trend=trend)
        assert alone.provenance.trend is not None
        own = alone.provenance.trend["fits"][0]
        assert (fit["slope"], fit["intercept"], fit["unit"]) == (
            own["slope"],
            own["intercept"],
            own["unit"],
        )


def test_the_impact_reads_the_adjusted_hazard() -> None:
    trend = LinearTrend(min_years=3)
    sample = _sample(series=_trended_series(), trend=trend, combine=CompoundDamage())
    components = sample.index_components
    assert components is not None
    rain, heat = components["rain"], components["heat"]
    kept = (1 - rain.values[:, 0] / 100) * (1 - heat.values[:, 0] / 100)
    np.testing.assert_allclose(sample.loss[:, 0], (1 - kept) * 1000.0)
    assert not np.array_equal(rain.values[:, 0], TRENDED_RAIN)


def test_a_component_keeps_its_own_fit() -> None:
    sample = _sample(series=_trended_series(), trend=LinearTrend(min_years=3))
    trend = sample.component("heat").provenance.trend
    assert trend is not None
    assert [fit["component"] for fit in trend["fits"]] == ["heat"]
    assert trend["fit_years"] == tuple(YEARS)


def test_a_trend_error_names_the_series() -> None:
    with pytest.raises(SampleError, match="non-negative heat values negative"):
        _sample(
            series={
                "rain": _series([*TRENDED_RAIN]),
                "heat": _series(
                    [5.0, 140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0], unit="degC"
                ),
            },
            trend=LinearTrend(min_years=3),
        )


# -- the basis-risk model ------------------------------------------------------------


def test_no_basis_risk_warns_once() -> None:
    with pytest.warns(ComonotoneIndexWarning) as recorded:
        _sample()
    assert len(recorded) == 1


def test_a_basis_risk_model_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComonotoneIndexWarning)
        _sample(basis_risk=NOISE, rng=np.random.default_rng(0))


def test_impact_noise_needs_a_generator() -> None:
    with pytest.raises(SampleError, match="pass rng="):
        _sample(basis_risk=NOISE)


def test_the_model_draws_in_component_order() -> None:
    series = _two_series()
    sample = _sample(basis_risk=NOISE, rng=np.random.default_rng(3), series=series)
    assert (
        sample.component("rain").content_hash
        == _one_hazard(
            series["rain"],
            "rain-ramp",
            basis_risk=NOISE,
            rng=np.random.default_rng(3),
        ).content_hash
    )
    after_rain = np.random.default_rng(3)
    after_rain.normal(size=8)
    assert (
        sample.component("heat").content_hash
        == _one_hazard(
            series["heat"], "heat-ramp", basis_risk=NOISE, rng=after_rain
        ).content_hash
    )


def test_the_rule_acts_after_the_basis_risk_model() -> None:
    sample = _sample(
        basis_risk=NOISE, rng=np.random.default_rng(3), combine=CompoundDamage()
    )
    components = sample.index_components
    assert components is not None
    own_rain, own_heat = components["rain"].own_loss, components["heat"].own_loss
    assert own_rain is not None and own_heat is not None
    kept = (1 - own_rain[:, 0] / 1000) * (1 - own_heat[:, 0] / 1000)
    np.testing.assert_allclose(sample.loss[:, 0], (1 - kept) * 1000.0)
    assert not np.allclose(own_rain[:, 0], np.array(RAIN) * 10.0)


def test_compound_damage_refuses_an_impact_a_model_pushed_above_one() -> None:
    class Doubled(NoBasisRisk):
        def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
            return index, impact * 2.0

    with pytest.raises(SampleError, match="ImpactNoise without upper=1"):
        _sample(basis_risk=Doubled(), combine=CompoundDamage())


def test_the_generator_records_name_their_components_and_rebuild_the_sample() -> None:
    series = _two_series()
    sample = _sample(basis_risk=NOISE, rng=np.random.default_rng(9), series=series)
    records = sample.provenance.rng
    assert [record["component"] for record in records] == ["rain", "heat"]
    assert {record["step"] for record in records} == {"basis_risk"}
    again = _sample(basis_risk=NOISE, rng=sample.provenance.generator(0), series=series)
    assert again.content_hash == sample.content_hash


def test_a_component_with_no_impact_draws_nothing() -> None:
    series = _two_series()
    sample = _sample(
        loss_models={"heat": _ramp("heat-ramp")},
        combine=None,
        basis_risk=NOISE,
        rng=np.random.default_rng(3),
        series=series,
    )
    assert [record["component"] for record in sample.provenance.rng] == ["heat"]
    assert (
        sample.component("heat").content_hash
        == _one_hazard(
            series["heat"], "heat-ramp", basis_risk=NOISE, rng=np.random.default_rng(3)
        ).content_hash
    )


def test_a_model_that_changes_the_index_keeps_the_hazard() -> None:
    class Shifted(NoBasisRisk):
        def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
            return index + 1.0, impact

    sample = _sample(basis_risk=Shifted())
    components = sample.index_components
    assert components is not None
    hazard = components["rain"].hazard
    assert hazard is not None
    assert hazard[:, 0].tolist() == RAIN
    assert sample.hazard is not None
    assert sample.index[:, 0].tolist() == [value + 1.0 for value in RAIN]


# -- a component's own loss ----------------------------------------------------------


def test_a_component_holds_its_own_hazards_loss() -> None:
    sample = _sample()
    rain = sample.component("rain")
    np.testing.assert_allclose(rain.loss[:, 0], np.array(RAIN) * 10.0)
    assert rain.index[:, 0].tolist() == RAIN
    assert rain.index_unit == "mm/day"
    assert rain.loss_unit == "EUR"
    heat = sample.component("heat")
    np.testing.assert_allclose(heat.loss[:, 0], np.array(HEAT) * 10.0)
    assert heat.index_unit == "degC"
    assert heat.index_components is None
    assert rain.period.tolist() == YEARS


@pytest.mark.parametrize("name", ["rain", "heat"])
def test_a_component_has_the_hash_of_its_one_hazard_sample(name: str) -> None:
    series = _two_series()
    assert (
        _sample(series=series).component(name).content_hash
        == _one_hazard(series[name], f"{name}-ramp").content_hash
    )


def test_a_component_with_combined_loss_keeps_the_combined_loss() -> None:
    sample = _sample(combine=CompoundDamage())
    view = sample.component("heat", loss="combined")
    assert view.index[:, 0].tolist() == HEAT
    np.testing.assert_array_equal(view.loss, sample.loss)


def test_the_view_of_a_component_with_no_impact() -> None:
    sample = _sample(loss_models={"heat": _ramp()}, combine=None)
    with pytest.raises(SampleError, match="loss='combined'"):
        sample.component("rain")
    view = sample.component("rain", loss="combined")
    np.testing.assert_array_equal(view.loss, sample.loss)
    assert view.provenance.loss_model is None


def test_a_component_records_the_step_and_its_own_source() -> None:
    sample = _sample(
        series={
            "rain": _series([*RAIN], source={"name": "rain table"}),
            "heat": _series([*HEAT], unit="degC", source={"name": "heat table"}),
        }
    )
    heat = sample.component("heat").provenance
    assert heat.selection == ({"kind": "component", "name": "heat", "loss": "own"},)
    assert [source["component"] for source in heat.sources] == ["heat"]
    assert heat.sources[0]["series"]["source"]["name"] == "heat table"
    assert heat.loss_model is not None
    assert heat.loss_model["impact"]["name"] == "heat-ramp"
    assert heat.hazard_dependence is None
    assert heat.combine is None
    assert heat.components is None
    assert heat.lineage == sample.lineage


def test_a_separate_cover_can_be_designed_on_a_components_own_loss() -> None:
    sample = _sample()
    rain = design(_rain_cover(), sample.component("rain"), MSE())
    # Rain reaches 50 in 2000, 2003 and 2005, with own losses 700, 900 and 600.
    assert rain.cover.payout_values(np.array([60.0]))[0] == pytest.approx(2200.0 / 3)
    on_combined = design(
        _rain_cover(), sample.component("rain", loss="combined"), MSE()
    )
    assert on_combined.cover.payout_values(np.array([60.0]))[0] == pytest.approx(1000.0)


# -- the record ----------------------------------------------------------------------


def test_the_sources_name_their_components() -> None:
    sample = _sample()
    assert [
        (source["role"], source["component"], source["unit"])
        for source in sample.provenance.sources
    ] == [("hazard", "rain", "mm/day"), ("hazard", "heat", "degC")]


def test_each_licence_is_listed_once() -> None:
    same = {"name": "one", "licence": "CC-BY-4.0", "attribution": "A"}
    other = {"name": "two", "licence": "CC0", "attribution": "B"}
    shared = _sample(
        series={
            "rain": _series([*RAIN], source=same),
            "heat": _series([*HEAT], unit="degC", source=same),
        }
    )
    assert [entry["licence"] for entry in shared.provenance.licences] == ["CC-BY-4.0"]
    different = _sample(
        series={
            "rain": _series([*RAIN], source=same),
            "heat": _series([*HEAT], unit="degC", source=other),
        }
    )
    assert [entry["licence"] for entry in different.provenance.licences] == [
        "CC-BY-4.0",
        "CC0",
    ]


def test_each_components_loss_model_is_recorded_in_order() -> None:
    document = _sample().provenance.loss_model
    assert document is not None
    assert document["kind"] == "components"
    assert [entry["name"] for entry in document["components"]] == ["rain", "heat"]
    assert [
        entry["loss_model"]["impact"]["name"] for entry in document["components"]
    ] == ["rain-ramp", "heat-ramp"]


def test_the_values_of_the_assets_are_not_recorded() -> None:
    document = _sample(
        loss_models={
            "rain": _ramp(value=2345.5),
            "heat": _ramp(value=2345.5),
        }
    ).provenance.to_document()
    assert "2345.5" not in repr(document)


def test_the_notes_name_their_components() -> None:
    notes = _sample().provenance.notes
    assert notes == (
        "Component rain: No basis-risk model: the index and the loss are comonotone.",
        "Component heat: No basis-risk model: the index and the loss are comonotone.",
    )
    assert _sample().component("rain").provenance.notes == (
        "No basis-risk model: the index and the loss are comonotone.",
    )


def test_the_details_are_each_components_own() -> None:
    sample = _sample(basis_risk=NOISE, rng=np.random.default_rng(1))
    details = sample.provenance.details
    assert details["years"] == (2000, 2007)
    assert details["n_years"] == 8
    assert details["missing"] == "raise"
    assert details["dropped_years"] == ()
    assert set(details["basis_risk"]) == {"rain", "heat"}
    own = sample.component("rain").provenance.details["basis_risk"]
    assert own == details["basis_risk"]["rain"]


def test_the_lineage_names_the_source_of_every_series_and_not_the_scenario() -> None:
    assert _sample().lineage == _sample().lineage
    assert _sample().lineage == _sample(scenario="other").lineage
    assert _sample().lineage == _sample(trend=RecentYears(4)).lineage
    changed = {**_two_series(), "heat": _series([*RAIN], unit="degC")}
    assert _sample().lineage != _sample(series=changed).lineage


def test_the_scenario_is_the_callers() -> None:
    assert _sample(scenario="warm").scenario == "warm"
    assert _sample(scenario="warm").component("rain").scenario == "warm"


def test_a_written_sample_reads_back() -> None:
    sample = _sample(combine=CompoundDamage())
    back = OutcomeSample.from_written_table(sample.to_table())
    assert back.content_hash == sample.content_hash
    assert back.provenance.combine == sample.provenance.combine
    assert back.provenance.hazard_dependence == {"kind": "data"}
    assert back.lineage == sample.lineage
    assert back.component("heat").content_hash == sample.component("heat").content_hash


def test_a_sample_holds_the_same_hash_under_the_same_series() -> None:
    assert _sample().content_hash == _sample().content_hash
    assert _sample().content_hash != _sample(combine=CompoundDamage()).content_hash


# -- splits and comparisons ------------------------------------------------------------


def test_a_holdout_keeps_the_components_and_the_lineage() -> None:
    sample = _sample()
    train, test = holdout(sample, 0.75)
    assert train.period.tolist() == YEARS[:6]
    assert test.period.tolist() == YEARS[6:]
    assert train.lineage == sample.lineage == test.lineage
    assert test.provenance.hazard_dependence == {"kind": "data"}
    assert train.index_components is not None
    assert train.index_components["heat"].values[:, 0].tolist() == HEAT[:6]


def test_a_design_on_a_component_is_checked_for_leakage() -> None:
    sample = _sample()
    train, test = holdout(sample, 0.75)
    rain = design(_rain_cover(), train.component("rain"), MSE())
    compare([rain], test, split="out_of_sample")
    with pytest.raises(LeakageError):
        compare([rain], train, split="out_of_sample")


def test_a_holistic_cover_and_separate_covers_in_one_comparison() -> None:
    train, test = holdout(_sample(combine=CompoundDamage()), 0.75)
    holistic = design(
        Cover(
            CompositeIndex({"rain": RAIN_INDEX, "heat": HEAT_INDEX}),
            SumOf(
                {
                    "rain": StepPayout([50.0], [Free()]),
                    "heat": StepPayout([50.0], [Free()]),
                }
            ),
            payout_unit="EUR",
            label="holistic",
        ),
        train,
        MSE(),
    )
    rain = design(_rain_cover(), train.component("rain"), MSE())
    heat = design(_heat_cover(), train.component("heat"), MSE())
    table = compare(
        [holistic, CoverPackage({"rain": rain, "heat": heat}, label="separate")],
        test,
        split="out_of_sample",
    )
    assert table.column("level").to_pylist() == ["asset", "asset", "asset", "package"]
    assert len(set(table.column("sample_hash").to_pylist())) == 1
    text = setup_text(table)
    assert "designed on the combined loss" in text
    assert "designed on the own loss of component rain" in text


def test_a_max_of_cover_reads_the_components() -> None:
    cover = Cover(
        CompositeIndex({"rain": RAIN_INDEX, "heat": HEAT_INDEX}),
        MaxOf(
            {
                "rain": StepPayout([50.0], [100.0]),
                "heat": StepPayout([50.0], [200.0]),
            }
        ),
        payout_unit="EUR",
    )
    table = compare([cover], _sample(), split="in_sample")
    assert table.column("components").to_pylist() == ["rain+heat"]


# -- the setup text --------------------------------------------------------------------


def test_the_setup_text_states_the_rule_and_the_annual_maxima() -> None:
    text = _text(_sample(combine=CappedSum(1.0)))
    sample_line = _line(text, "Sample:")
    assert (
        "Index components: rain (in mm/day, upper tail, with a loss of its own) and "
        "heat (in degC, upper tail, with a loss of its own); the first is the "
        "sample's index." in sample_line
    )
    assert (
        "Combined loss: The impacts are summed and capped at 1.0, then multiplied by "
        "the value." in sample_line
    )
    assert (
        "Dependence between the hazards: the data's own, the rows being the same "
        f"years of the data. {ANNUAL_MAXIMA}" in sample_line
    )


def test_the_setup_text_states_compound_damage() -> None:
    assert (
        "Combined loss: Each hazard damages what the others leave: 1 minus the "
        "product of (1 minus each impact), then multiplied by the value."
    ) in _text(_sample(combine=CompoundDamage()))


def test_the_setup_text_states_each_components_loss_model() -> None:
    assert (
        "Loss: rain: callable impact rain-ramp times a stated asset value, in EUR; "
        "heat: callable impact heat-ramp times a stated asset value, in EUR."
    ) in _text(_sample())


def test_the_setup_text_of_one_impact() -> None:
    text = _text(_sample(loss_models={"heat": _ramp("heat-ramp")}, combine=None))
    assert (
        "Combined loss: Only component heat has an impact, so the combined loss is "
        "its own loss."
    ) in text
    assert "rain (in mm/day, upper tail, with no loss of its own)" in text
    assert "Loss: heat: callable impact heat-ramp" in text


def test_the_setup_text_of_a_table_says_so_with_the_annual_maxima() -> None:
    table = {
        "year": YEARS,
        "site": ["site"] * 8,
        "rain": RAIN,
        "heat": HEAT,
        "loss": [float(i) for i in range(8)],
    }
    sample = OutcomeSample.from_table(
        table,
        period="year",
        asset_id="site",
        index={"rain": "rain", "heat": "heat"},
        loss="loss",
        semantics="annual",
        index_unit={"rain": "mm/day", "heat": "degC"},
        loss_unit="EUR",
        tails={"rain": "upper", "heat": "upper"},
    )
    text = _text(sample)
    assert "Combined loss: as the table states it." in text
    assert ANNUAL_MAXIMA in text


def test_the_annual_maxima_sentence_is_for_the_datas_own_dependence() -> None:
    sample = _sample()
    drawn = dataclasses.replace(
        sample,
        provenance=dataclasses.replace(
            sample.provenance, hazard_dependence={"kind": "independent"}
        ),
    )
    text = _text(drawn)
    assert "Independent(), a stated assumption" in text
    assert "annual maxima" not in text


def test_the_setup_text_says_mixed_tails_are_values_not_severities() -> None:
    mixed = _text(_sample(tails={"rain": "upper", "heat": "lower"}))
    assert (
        f"{ANNUAL_MAXIMA} With mixed tails, that dependence is between values, not "
        "severities."
    ) in mixed
    assert "mixed tails" not in _text(_sample())


def test_a_rule_without_a_final_stop_reads_as_a_sentence() -> None:
    text = _text(_sample(combine=Worst()))
    assert "is taken. Dependence between the hazards" in text


def _grid_table(values: list[float]) -> Any:
    return pa.table(
        {
            "year": pa.array([year for year in YEARS for _ in range(2)], pa.int32()),
            "latitude": [50.0] * 16,
            "longitude": [7.0, 7.25] * 8,
            "value": [value for value in values for _ in range(2)],
        }
    )


def _located_sample() -> OutcomeSample:
    rain = AnnualSeries.nearest_cell(
        _grid_table(RAIN),
        longitude=7.01,
        latitude=50.0,
        unit="mm/day",
        spacing=(0.25, 0.25),
        source={"provider": "era5", "dataset": "rx1day"},
    )
    stations = pa.table(
        {"year": pa.array(YEARS, pa.int32()), "station": ["S1"] * 8, "value": HEAT}
    )
    heat = AnnualSeries.from_table(
        stations,
        where={"station": "S1"},
        unit="degC",
        source={"provider": "noaa", "dataset": "txx"},
    )
    return _sample(series={"rain": rain, "heat": heat})


def test_the_setup_text_states_each_components_source_and_location() -> None:
    text = _text(_located_sample())
    assert (
        "Source: the hazard series of component rain from provider era5 and dataset "
        "rx1day, with horizon and pathway not recorded; the hazard series of "
        "component heat from provider noaa and dataset txx, with horizon and pathway "
        "not recorded."
    ) in text
    assert (
        "Location: hazard series of component rain, located by a point, in the nearest "
        "grid cell (grid step 0.25 by 0.25 degrees): site at (50, 7), the cell centres "
        "as (latitude, longitude)."
    ) in text
    assert (
        "Location: hazard series of component heat, the rows where station = S1."
        in (text)
    )


def test_a_one_hazard_series_with_a_filter_has_no_location_line() -> None:
    stations = pa.table(
        {"year": pa.array(YEARS, pa.int32()), "station": ["S1"] * 8, "value": RAIN}
    )
    series = AnnualSeries.from_table(stations, where={"station": "S1"}, unit="mm/day")
    table = compare([_fixed_cover()], _one_hazard(series, "r"), split="in_sample")
    assert "Location" not in setup_text(table)


def test_the_setup_text_names_each_components_trend() -> None:
    text = _text(_sample(series=_trended_series(), trend=LinearTrend(min_years=3)))
    line = _line(text, "Trend:")
    assert re.search(r"hazard of component rain slope .* mm/day per year", line)
    assert re.search(r"hazard of component heat slope .* degC per year", line)


def test_the_setup_text_of_a_trend_as_recorded_is_one_line() -> None:
    assert "Trend: the years are used as recorded; no trend is removed." in _text(
        _sample(trend=AsRecorded())
    )


def test_the_setup_text_of_a_view_says_it_holds_the_combined_loss() -> None:
    sample = _sample(loss_models={"heat": _ramp()}, combine=None)
    train, test = holdout(sample, 0.75)
    design_on_view = design(
        _rain_cover(), train.component("rain", loss="combined"), MSE()
    )
    text = setup_text(compare([design_on_view], test, split="out_of_sample"))
    assert "Loss: the combined loss of the hazards, in EUR." in text


def test_the_setup_text_of_a_view_with_an_impact_says_so_too() -> None:
    train, test = holdout(_sample(), 0.75)
    design_on_view = design(
        _rain_cover(), train.component("rain", loss="combined"), MSE()
    )
    text = setup_text(compare([design_on_view], test, split="out_of_sample"))
    assert "Loss: the combined loss of the hazards, in EUR." in text


def test_the_setup_text_of_a_design_on_its_own_loss_names_its_model() -> None:
    train, test = holdout(_sample(), 0.75)
    rain = design(_rain_cover(), train.component("rain"), MSE())
    text = setup_text(compare([rain], test, split="out_of_sample"))
    assert "Loss: callable impact rain-ramp times a stated asset value, in EUR." in text


def test_the_setup_text_of_a_loss_model_of_the_wrong_shape_raises() -> None:
    sample = _sample()
    broken = dataclasses.replace(
        sample,
        provenance=dataclasses.replace(
            sample.provenance, loss_model={"kind": "components", "components": [1]}
        ),
    )
    with pytest.raises(ValueError, match="list of objects"):
        _text(broken)


# -- the arguments -----------------------------------------------------------------


def test_a_hazard_unit_the_impact_is_not_calibrated_for_raises_before_any_draw() -> (
    None
):
    model = LossModel(
        ImpactExpression.linear(
            0.01,
            calibration=ImpactCalibration(
                exposure_variable="rx1day",
                exposure_unit="mm/day",
                impact_metric="damage_ratio",
                impact_unit="1",
            ),
            clamp={"minimum": 0.0, "maximum": 1.0},
        ),
        value=1000.0,
        value_unit="EUR",
    )
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    with pytest.raises(SampleError, match="'degC', but impact 'linear'"):
        _sample(loss_models={"rain": model, "heat": model}, rng=rng)
    assert rng.bit_generator.state == before
    assert _sample(loss_models={"rain": model, "heat": _ramp()}).loss_unit == "EUR"


def test_a_component_with_no_impact_is_not_checked_for_its_unit() -> None:
    sample = _sample(
        series={
            "rain": _series([*RAIN], unit="anything"),
            "heat": _series([*HEAT], unit="degC"),
        },
        loss_models={"heat": _ramp()},
        combine=None,
    )
    assert sample.index_unit == "anything"


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param(
            {"series": [1, 2]}, TypeError, "series must map", id="series-list"
        ),
        pytest.param(
            {"series": {"rain": _series([1.0])}},
            SampleError,
            "two hazards or more, got 1; for one hazard use sample_annual",
            id="one-series",
        ),
        pytest.param(
            {"series": {"rain": 1, "heat": 2}},
            TypeError,
            "series\\['rain'\\] must be an AnnualSeries",
            id="not-a-series",
        ),
        pytest.param(
            {"series": {"a.b": _series([1.0]), "c": _series([1.0])}},
            SampleError,
            "without '\\.'",
            id="bad-name",
        ),
        pytest.param(
            {"tails": {"rain": "upper"}},
            SampleError,
            "tails is keyed by the components",
            id="tails-missing",
        ),
        pytest.param(
            {"tails": None}, SampleError, "tails must be a mapping", id="tails-none"
        ),
        pytest.param(
            {"tails": {"rain": "upper", "heat": "up"}},
            SampleError,
            "tails\\['heat'\\] must be 'upper' or 'lower'",
            id="bad-tail",
        ),
        pytest.param(
            {"loss_models": {"rain": _ramp(), "wind": _ramp()}},
            SampleError,
            "loss_models is keyed by the components",
            id="unknown-model",
        ),
        pytest.param(
            {"loss_models": {"rain": _ramp(), "heat": 1}},
            TypeError,
            "loss_models\\['heat'\\] must be a LossModel",
            id="not-a-model",
        ),
        pytest.param(
            {"basis_risk": None}, TypeError, "basis_risk must be", id="no-basis-risk"
        ),
        pytest.param(
            {"asset_id": " a"}, SampleError, "asset_id must be", id="bad-asset"
        ),
        pytest.param({"rng": 1}, TypeError, "rng must be", id="bad-rng"),
        pytest.param(
            {"missing": "fill"}, SampleError, "missing must be", id="bad-missing"
        ),
        pytest.param({"trend": "none"}, TypeError, "trend must be", id="bad-trend"),
        pytest.param(
            {"scenario": ""}, SampleError, "scenario must be", id="bad-scenario"
        ),
    ],
)
def test_invalid_arguments_raise(
    overrides: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _sample(**overrides)


def test_a_recent_years_window_longer_than_the_series_raises() -> None:
    with pytest.raises(SampleError, match=r"RecentYears\(9\) needs 9 years"):
        _sample(trend=RecentYears(9))
