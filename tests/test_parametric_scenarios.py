"""What a `ScenarioSet` holds, and each thing it refuses.

Every number here is synthetic: small made-up curves, series and tables.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import numpy as np
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    AnnualSeries,
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    CappedSum,
    Comonotonic,
    CompositeIndex,
    CompoundDamage,
    Cover,
    EmpiricalCopula,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LinearPayout,
    LinearTrend,
    LossModel,
    ModelRun,
    NoBasisRisk,
    OutcomeSample,
    RecentYears,
    SampleError,
    ScenarioError,
    ScenarioSet,
    SumOf,
    holdout,
    sample_annual,
    sample_annual_hazards,
    sample_annual_pool,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric.scenarios import _check_bound_cover, _check_pairs_held
from crc_sdk.pathways import register_pathway
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance, TemporalWindow

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.distributions.ReturnPeriodExtrapolationWarning"
    ),
]

UNTYPED_SET: Any = ScenarioSet
UNTYPED_CURVE: Any = sample_curve
UNTYPED_CURVES: Any = sample_curves
UNTYPED_HAZARDS: Any = sample_curve_hazards
UNTYPED_TABLE: Any = OutcomeSample.from_table

SEED = 7
CELL = 613196570331971583
CELL_HEX = format(CELL, "x")
OTHER_CELL = CELL + 1
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
NOISE = ImpactNoise(0.3)
WINDOW = (2045, 2050, 2055)
BOTH = ("'a'", "'b'")


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


def _row(
    horizon: int = 2050,
    *,
    pathway: str = "ssp245",
    location: float | None = None,
    **fields: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell_index": CELL,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": (50.0 + (horizon - 2020) if location is None else location),
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    row.update(fields)
    return row


def _curve(label: str, curve: Any = None, **options: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "metadata": _dataset(),
        "n_draws": 120,
        "rng": np.random.default_rng(SEED),
        "loss_model": LOSS,
        "basis_risk": NOISE,
        "asset_id": "a",
        "scenario": label,
        **options,
    }
    sample: OutcomeSample = UNTYPED_CURVE(
        _row() if curve is None else curve, **arguments
    )
    return sample


def _window(
    label: str, horizons: tuple[int, ...] = WINDOW, **options: Any
) -> OutcomeSample:
    fields = options.pop("fields", {})
    return _curve(label, [_row(horizon, **fields) for horizon in horizons], **options)


def _curves(label: str, curves: Any = None, **options: Any) -> OutcomeSample:
    chosen = {"a": _row(), "b": _row(location=90.0)} if curves is None else curves
    arguments: dict[str, Any] = {
        "dependence": Independent(),
        "metadata": dict.fromkeys(chosen, _dataset()),
        "n_draws": 120,
        "rngs": {
            name: np.random.default_rng(seed) for seed, name in enumerate(chosen, 1)
        },
        "loss_models": dict.fromkeys(chosen, LOSS),
        "basis_risk": NOISE,
        "scenario": label,
        **options,
    }
    sample: OutcomeSample = UNTYPED_CURVES(chosen, **arguments)
    return sample


def _book_window(
    label: str, horizons: tuple[int, ...], **options: Any
) -> OutcomeSample:
    return _curves(
        label,
        {
            "a": [_row(horizon) for horizon in horizons],
            "b": [_row(horizon, location=90.0 + horizon) for horizon in horizons],
        },
        **options,
    )


def _hazards(label: str, horizon: int = 2050, **options: Any) -> OutcomeSample:
    curves = options.pop(
        "curves",
        {
            "rain": _row(horizon, hazard_name="rain"),
            "heat": _row(horizon, hazard_name="heat", location=horizon - 2000.0),
        },
    )
    arguments: dict[str, Any] = {
        "dependence": Independent(),
        "metadata": dict.fromkeys(curves, _dataset()),
        "n_draws": 120,
        "rngs": {
            name: np.random.default_rng(seed) for seed, name in enumerate(curves, 11)
        },
        "loss_models": {next(iter(curves)): LOSS},
        "basis_risk": NOISE,
        "asset_id": "a",
        "scenario": label,
        **options,
    }
    sample: OutcomeSample = UNTYPED_HAZARDS(curves, **arguments)
    return sample


def _series(
    source: Any = None,
    *,
    first: int = 2000,
    count: int = 40,
    seed: int = 3,
    **provenance: Any,
) -> AnnualSeries:
    years = np.arange(first, first + count, dtype=np.int64)
    values = np.random.default_rng(seed).gumbel(60.0, 12.0, count)
    document = {"provider": "fixture", "recipe": "rx1day"} if source is None else source
    return AnnualSeries(
        years, values, "mm/day", {"kind": "table", "source": document, **provenance}
    )


def _stated(pathway: str, **fields: Any) -> dict[str, Any]:
    return {"provider": "fixture", "recipe": "rx1day", "pathway": pathway, **fields}


def _annual(label: str, series: AnnualSeries | None = None, **options: Any) -> Any:
    return sample_annual(
        _series() if series is None else series,
        loss_model=options.pop("loss_model", LOSS),
        basis_risk=options.pop("basis_risk", NoBasisRisk()),
        asset_id=options.pop("asset_id", "a"),
        scenario=label,
        **options,
    )


def _years(pathway: str, first: int, label: str, **options: Any) -> Any:
    return _annual(label, _series(_stated(pathway), first=first), **options)


def _run(model: str, pathway: str = "ssp245", **options: Any) -> ModelRun:
    series = _series(seed=sum(model.encode("utf-8")), **options)
    return ModelRun(series, model=model, member="r1i1p1f1", pathway=pathway)


def _pool(label: str, runs: Any = None, **options: Any) -> Any:
    return sample_annual_pool(
        [_run("M1"), _run("M2")] if runs is None else runs,
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario=label,
        **options,
    )


def _table(label: str, **options: Any) -> OutcomeSample:
    rng = np.random.default_rng(5)
    data = {
        "year": list(range(2000, 2020)),
        "site": ["a"] * 20,
        "x": rng.gumbel(50.0, 8.0, 20).tolist(),
        "loss": rng.gamma(2.0, 100.0, 20).tolist(),
    }
    arguments: dict[str, Any] = {
        "period": "year",
        "asset_id": "site",
        "index": "x",
        "loss": "loss",
        "semantics": "annual",
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "hazard_name": "rx1day",
        "scenario": label,
        **options,
    }
    sample: OutcomeSample = UNTYPED_TABLE(data, **arguments)
    return sample


def _components_table(label: str, **options: Any) -> OutcomeSample:
    rng = np.random.default_rng(9)
    data = {
        "year": list(range(2000, 2020)),
        "site": ["a"] * 20,
        "rain": rng.gumbel(50.0, 8.0, 20).tolist(),
        "heat": rng.gumbel(30.0, 5.0, 20).tolist(),
        "loss": rng.gamma(2.0, 100.0, 20).tolist(),
    }
    sample: OutcomeSample = UNTYPED_TABLE(
        data,
        period="year",
        asset_id="site",
        index={"rain": "rain", "heat": "heat"},
        loss="loss",
        semantics="annual",
        index_unit={"rain": "mm/day", "heat": "mm/day"},
        loss_unit="EUR",
        tails={"rain": "upper", "heat": "upper"},
        scenario=label,
        **options,
    )
    return sample


def _edited(sample: OutcomeSample, edit: Any) -> OutcomeSample:
    """`sample` with its provenance's document changed by `edit`, in place."""
    document = sample.provenance.to_document()
    edit(document)
    provenance = type(sample.provenance)(**document)
    return dataclasses.replace(sample, provenance=provenance)


def _refused(entries: Any, *needles: str, **options: Any) -> str:
    """Build the set, reference first, and check that its error names `needles`."""
    reference = options.pop("reference", entries[0].scenario)
    with pytest.raises(ScenarioError) as caught:
        ScenarioSet(entries, reference=reference, **options)
    message = str(caught.value)
    for needle in needles:
        assert needle in message, message
    return message


def _cover(location: Any = None, **fields: Any) -> Cover:
    index = IndexDefinition(
        "rx1day",
        "mm/day",
        aggregation="annual_maximum",
        location=AtAsset() if location is None else location,
    )
    return Cover(
        index,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        label=fields.pop("label", "bound"),
        **fields,
    )


# -- the set -------------------------------------------------------------------


def test_a_set_holds_its_samples_under_a_reference() -> None:
    a, b = _curve("a", _row(2030)), _curve("b", _row(2050))
    held = ScenarioSet([b, a], reference="a")
    assert held.reference == "a"
    assert held.entries == (b, a)
    assert held.mixed_products is False
    assert held.sample("a") is a
    assert held.sample("b") is b


def test_labels_hold_the_reference_first_then_as_given() -> None:
    a, b, c = (
        _curve("a", _row(2030)),
        _curve("b", _row(2050)),
        _curve("c", _row(2070)),
    )
    assert ScenarioSet([c, a, b], reference="a").labels == ("a", "c", "b")
    assert ScenarioSet([c, a, b], reference="b").labels == ("b", "c", "a")


def test_the_reference_alone_is_a_set() -> None:
    assert ScenarioSet([_curve("a")], reference="a").labels == ("a",)


def test_a_set_is_frozen() -> None:
    held = ScenarioSet([_curve("a")], reference="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        held.reference = "b"  # type: ignore[misc]


def test_a_set_says_what_it_holds() -> None:
    text = repr(ScenarioSet([_curve("a")], reference="a"))
    assert text == "ScenarioSet(1 samples, reference='a', mixed_products=False)"


def test_a_label_the_set_does_not_hold_raises() -> None:
    held = ScenarioSet([_curve("a")], reference="a")
    with pytest.raises(ScenarioError, match="no scenario 'z'.*'a'"):
        held.sample("z")


def test_the_reference_is_required() -> None:
    with pytest.raises(TypeError, match="reference"):
        UNTYPED_SET([_curve("a")])


def test_the_reference_must_be_a_member() -> None:
    a, b = _curve("a", _row(2030)), _curve("b", _row(2050))
    _refused([a, b], "reference", "'z'", "'a'", "'b'", reference="z")


@pytest.mark.parametrize("reference", [None, 2030])
def test_a_reference_that_is_not_a_string_raises(reference: Any) -> None:
    with pytest.raises(TypeError, match="reference"):
        UNTYPED_SET([_curve("a")], reference=reference)


@pytest.mark.parametrize("reference", ["", " a"])
def test_an_invalid_reference_raises(reference: str) -> None:
    with pytest.raises(ScenarioError, match="reference"):
        ScenarioSet([_curve("a")], reference=reference)


def test_two_samples_with_one_label_raise_and_name_scenario() -> None:
    message = _refused(
        [_curve("a", _row(2030)), _curve("a", _row(2050))], "'a'", "scenario="
    )
    assert "two samples" in message


def test_scenario_error_is_a_sample_error_and_a_value_error() -> None:
    with pytest.raises(SampleError):
        ScenarioSet([_curve("a")], reference="z")
    with pytest.raises(ValueError):
        ScenarioSet([_curve("a")], reference="z")


@pytest.mark.parametrize("entries", ["a", 3, {"a": 1}, [object()], ["a"]], ids=repr)
def test_entries_of_the_wrong_type_raise(entries: Any) -> None:
    with pytest.raises(TypeError, match="entries"):
        UNTYPED_SET(entries, reference="a")


def test_a_single_sample_is_not_entries() -> None:
    with pytest.raises(TypeError, match="entries"):
        UNTYPED_SET(_curve("a"), reference="a")


def test_mixed_products_must_be_a_bool() -> None:
    with pytest.raises(TypeError, match="mixed_products"):
        UNTYPED_SET([_curve("a")], reference="a", mixed_products=1)


def test_a_generator_of_entries_is_read_once() -> None:
    a, b = _curve("a", _row(2030)), _curve("b", _row(2050))
    assert ScenarioSet((s for s in (a, b)), reference="a").entries == (a, b)


# -- what a plan states --------------------------------------------------------


def test_a_set_without_entries_states_only_the_reference() -> None:
    held = ScenarioSet(reference="ssp585_2030")
    assert held.reference == "ssp585_2030"
    assert held.entries == ()


def test_an_empty_set_raises_where_samples_are_needed() -> None:
    held = ScenarioSet(reference="a")
    with pytest.raises(ScenarioError, match="no built samples"):
        _ = held.labels
    with pytest.raises(ScenarioError, match="no built samples"):
        held.sample("a")


def test_labels_map_a_label_to_its_pairs_in_horizon_order() -> None:
    held = ScenarioSet(
        reference="2050s",
        labels={
            "2050s": [("ssp585", 2055), ("ssp585", 2045), ("ssp585", 2050)],
            "mid": [("ssp585", 2060)],
        },
    )
    assert held._pairs == {
        "2050s": (("ssp585", 2045), ("ssp585", 2050), ("ssp585", 2055)),
        "mid": (("ssp585", 2060),),
    }
    assert "2050s" in repr(held)


def test_a_set_of_labels_holds_no_samples() -> None:
    held = ScenarioSet(reference="x", labels={"x": [("ssp585", 2050)]})
    with pytest.raises(TypeError, match="holds no built"):
        _ = held.labels
    with pytest.raises(TypeError, match="holds no built"):
        held.sample("x")


def test_labels_beside_samples_raise() -> None:
    with pytest.raises(TypeError, match="never both"):
        ScenarioSet([_curve("a")], reference="a", labels={"a": [("ssp245", 2050)]})


def test_the_reference_must_be_one_of_the_labels() -> None:
    with pytest.raises(ScenarioError, match="reference 'z'.*'x'"):
        ScenarioSet(reference="z", labels={"x": [("ssp585", 2050)]})


def test_two_pathways_in_a_window_of_labels_raise() -> None:
    with pytest.raises(ScenarioError, match="'x'.*one pathway.*ssp245.*ssp585"):
        ScenarioSet(reference="x", labels={"x": [("ssp245", 2045), ("ssp585", 2050)]})


def test_equivalent_spellings_are_one_pathway_in_a_window_of_labels() -> None:
    held = ScenarioSet(
        reference="x", labels={"x": [("historic", 1990), ("historical", 2000)]}
    )
    assert held._pairs == {"x": (("historic", 1990), ("historical", 2000))}


def test_a_horizon_twice_in_a_window_of_labels_raises() -> None:
    with pytest.raises(ScenarioError, match="'x'.*horizon.*2050.*repeat"):
        ScenarioSet(reference="x", labels={"x": [("ssp585", 2050), ("ssp585", 2050)]})


def test_an_unknown_pathway_in_labels_raises() -> None:
    with pytest.raises(ScenarioError, match="'x'.*'not-a-pathway'"):
        ScenarioSet(reference="x", labels={"x": [("not-a-pathway", 2050)]})


def test_two_labels_with_the_same_members_raise_in_any_order() -> None:
    with pytest.raises(ScenarioError, match="'x' and 'y'.*pathway, horizon"):
        ScenarioSet(
            reference="x",
            labels={
                "x": [("ssp585", 2045), ("ssp585", 2050)],
                "y": [("ssp585", 2050), ("ssp585", 2045)],
            },
        )


def test_two_labels_with_equivalent_spellings_raise() -> None:
    with pytest.raises(ScenarioError, match="'x' and 'y'"):
        ScenarioSet(
            reference="x",
            labels={"x": [("historic", 2000)], "y": [("historical", 2000)]},
        )


def test_two_windows_of_labels_with_one_centre_pass() -> None:
    ScenarioSet(
        reference="x",
        labels={
            "x": [("ssp585", 2045), ("ssp585", 2055)],
            "y": [("ssp585", 2040), ("ssp585", 2060)],
            "z": [("ssp585", 2050)],
        },
    )


def test_a_label_with_no_pair_raises() -> None:
    with pytest.raises(ScenarioError, match="'x'.*no \\(pathway, horizon\\)"):
        ScenarioSet(reference="x", labels={"x": []})


@pytest.mark.parametrize(
    ("labels", "where"),
    [
        ([("x", 1)], "map"),
        ({"x": "ssp585"}, "iterable"),
        ({"x": ["ssp585"]}, "pairs"),
        ({"x": [("ssp585", 2050, 1)]}, "pairs"),
        ({"x": [(1, 2050)]}, "pathway"),
        ({"x": [("ssp585", "2050")]}, "horizon"),
        ({"x": [("ssp585", True)]}, "horizon"),
        ({1: [("ssp585", 2050)]}, "label"),
    ],
    ids=repr,
)
def test_labels_of_the_wrong_type_raise(labels: Any, where: str) -> None:
    with pytest.raises(TypeError, match=where):
        UNTYPED_SET(reference="x", labels=labels)


# -- what is refused, field by field -------------------------------------------


def test_assets_differ() -> None:
    _refused(
        [_curve("a", _row(2030)), _curve("b", _row(2050), asset_id="z")],
        *BOTH,
        "asset_ids",
    )


def test_assets_compare_in_order() -> None:
    swapped = {"b": _row(location=90.0), "a": _row()}
    _refused(
        [
            _curves("a", {"a": _row(2030), "b": _row(2030, location=90.0)}),
            _curves("b", swapped),
        ],
        *BOTH,
        "asset_ids",
    )


def test_curves_beside_years_raise() -> None:
    _refused([_curve("a"), _annual("b")], *BOTH, "source_kind")


def test_curves_beside_a_table_raise() -> None:
    _refused([_curve("a"), _table("b")], *BOTH, "source_kind", "monte_carlo", "table")


def test_a_series_beside_a_table_is_one_kind_and_states_no_loss_model() -> None:
    _refused(
        [_years("ssp245", 2000, "a"), _table("b", source={"provider": "fixture"})],
        *BOTH,
        "loss_model",
    )


def test_two_tables_stand_side_by_side() -> None:
    held = ScenarioSet(
        [
            _table("a", source=_stated("ssp245")),
            _table("b", source=_stated("ssp585")),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_a_hazard_that_differs_raises_by_exact_name() -> None:
    _refused(
        [_curve("a", _row(2030)), _curve("b", _row(2050, hazard_name="rx5day"))],
        *BOTH,
        "hazard_name",
        "rx1day",
        "rx5day",
    )


def test_hazard_names_are_compared_as_spelled() -> None:
    _refused(
        [_curve("a", _row(2030)), _curve("b", _row(2050, hazard_name="RX1DAY"))],
        *BOTH,
        "hazard_name",
    )


def test_per_year_hazards_that_differ_raise() -> None:
    other = _series({"provider": "fixture", "recipe": "rx5day", "pathway": "ssp585"})
    _refused(
        [_years("ssp245", 2000, "a"), _annual("b", other)],
        *BOTH,
        "hazard_name",
    )


def test_a_curve_that_names_no_hazard_raises() -> None:
    bare = FittedDistribution.from_parameters("gumbel_r", location=99.0, scale=10.0)
    message = _refused([_curve("a"), _curve("b", bare)], "'b'", "hazard_name")
    assert "'a'" not in message


def test_a_stated_name_stands_for_a_curve_that_names_none() -> None:
    bare = FittedDistribution.from_parameters("gumbel_r", location=99.0, scale=10.0)
    held = ScenarioSet(
        [_curve("a"), _curve("b", bare, hazard_name="rx1day")], reference="a"
    )
    assert held.labels == ("a", "b")


@pytest.mark.parametrize(
    "build",
    [
        lambda: _annual("a", _series("a table")),
        lambda: _pool("a", [_run("M1", source="t"), _run("M2", source="t")]),
        lambda: _table("a", hazard_name=None),
    ],
    ids=["series", "pool", "table"],
)
def test_a_per_year_scenario_that_names_no_hazard_raises(build: Any) -> None:
    _refused([build()], "'a'", "hazard_name")


def test_index_units_differ() -> None:
    _refused(
        [
            _curve("a", _row(2030)),
            _curve("b", _row(2050), metadata=_dataset(value_unit="mm/d")),
        ],
        *BOTH,
        "index_unit",
    )


def test_loss_units_differ() -> None:
    other = LossModel(RAMP, value=1e6, value_unit="USD")
    _refused(
        [_curve("a", _row(2030)), _curve("b", _row(2050), loss_model=other)],
        *BOTH,
        "loss_unit",
    )


def test_loss_models_differ() -> None:
    other = LossModel(RAMP, value=2e6, value_unit="EUR")
    _refused(
        [_curve("a", _row(2030)), _curve("b", _row(2050), loss_model=other)],
        *BOTH,
        "loss_model",
    )


def test_the_loss_model_of_one_asset_differs() -> None:
    other = LossModel(RAMP, value=2e6, value_unit="EUR")
    first = {"a": _row(2030), "b": _row(2030, location=90.0)}
    _refused(
        [
            _curves("a", first),
            _curves(
                "b",
                {"a": _row(2050), "b": _row(2050, location=95.0)},
                loss_models={"a": LOSS, "b": other},
            ),
        ],
        *BOTH,
        "loss_model",
        "asset 'b'",
    )


def _joined(label: str, first: int, value: float = 1e6) -> OutcomeSample:
    other = LossModel(RAMP, value=value, value_unit="EUR")
    return OutcomeSample.join(
        [
            _annual(label, _series(_stated("ssp245"), first=first), asset_id="a"),
            _annual(
                label,
                _series(_stated("ssp245"), first=first, seed=4),
                asset_id="b",
                loss_model=other,
            ),
        ]
    )


def test_a_joined_sample_compares_each_parts_loss_model() -> None:
    held = ScenarioSet([_joined("a", 2000), _joined("b", 2041)], reference="a")
    assert held.labels == ("a", "b")
    _refused(
        [_joined("a", 2000), _joined("b", 2041, 2e6)],
        *BOTH,
        "loss_model",
        "asset 'b'",
    )


def test_basis_risk_models_differ() -> None:
    _refused(
        [
            _curve("a", _row(2030)),
            _curve("b", _row(2050), basis_risk=ImpactNoise(0.2)),
        ],
        *BOTH,
        "basis_risk",
    )


def test_a_trend_option_differs() -> None:
    _refused(
        [
            _years("ssp245", 2000, "a"),
            _years("ssp585", 2000, "b", trend=RecentYears(20)),
        ],
        *BOTH,
        "trend",
    )


def test_the_n_of_a_trend_differs() -> None:
    _refused(
        [
            _years("ssp245", 2000, "a", trend=RecentYears(10)),
            _years("ssp585", 2000, "b", trend=RecentYears(20)),
        ],
        *BOTH,
        "trend",
        "'n'",
    )


def test_the_min_years_of_a_trend_differs() -> None:
    _refused(
        [
            _years("ssp245", 2000, "a", trend=LinearTrend(min_years=10)),
            _years("ssp585", 2000, "b", trend=LinearTrend(min_years=20)),
        ],
        *BOTH,
        "trend",
        "min_years",
    )


def test_a_reference_year_and_fitted_slopes_may_differ() -> None:
    held = ScenarioSet(
        [
            _years("ssp245", 2000, "a", trend=LinearTrend()),
            _years("ssp585", 2041, "b", trend=LinearTrend(reference_year=2060)),
        ],
        reference="a",
    )
    assert held.sample("a").provenance.trend is not None


def test_a_pools_models_differ() -> None:
    _refused(
        [
            _pool("a"),
            _pool(
                "b",
                [_run("M1", pathway="ssp585"), _run("M9", pathway="ssp585")],
            ),
        ],
        *BOTH,
        "(model, member)",
    )


def test_a_pools_models_differ_even_under_mixed_products() -> None:
    _refused(
        [
            _pool("a"),
            _pool(
                "b",
                [_run("M1", pathway="ssp585"), _run("M9", pathway="ssp585")],
            ),
        ],
        *BOTH,
        "(model, member)",
        mixed_products=True,
    )


def test_a_pool_of_the_same_models_in_another_order_passes() -> None:
    held = ScenarioSet(
        [
            _pool("a", [_run("M1"), _run("M2")]),
            _pool(
                "b",
                [_run("M2", pathway="ssp585"), _run("M1", pathway="ssp585")],
            ),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_a_series_beside_a_pool_raises_unless_mixed_products() -> None:
    entries = [_pool("a"), _years("ssp585", 2000, "b")]
    _refused(entries, *BOTH, "pool", "mixed_products=True")
    held = ScenarioSet(entries, reference="a", mixed_products=True)
    assert held.mixed_products is True


def test_a_pool_beside_a_series_raises_with_the_series_as_reference() -> None:
    _refused(
        [_years("ssp585", 2000, "b"), _pool("a")],
        *BOTH,
        "pool",
    )


def test_a_stated_pathway_a_row_contradicts_raises() -> None:
    _refused(
        [_curve("a", _row(2050), details={"pathway": "ssp585"})],
        "'a'",
        "pathway",
        "ssp245",
        "ssp585",
    )


def test_a_stated_pathway_of_another_spelling_agrees() -> None:
    held = ScenarioSet(
        [
            _curve(
                "a",
                _row(2000, pathway="historical"),
                details={"pathway": "historic"},
            )
        ],
        reference="a",
    )
    assert held.labels == ("a",)


def test_a_series_that_contradicts_its_source_raises() -> None:
    _refused(
        [_annual("a", _series(_stated("ssp245"), pathway="ssp585"))],
        "'a'",
        "pathway",
        "ssp245",
        "ssp585",
    )


def test_a_model_run_that_contradicts_its_series_raises() -> None:
    run = ModelRun(
        _series(_stated("ssp585")), model="M1", member="r1i1p1f1", pathway="ssp245"
    )
    other = ModelRun(
        _series(_stated("ssp585"), seed=4),
        model="M2",
        member="r1i1p1f1",
        pathway="ssp245",
    )
    # A pool with a label is refused at the builder; this one reaches the set.
    _refused(
        [_pool("baseline", [run, other])],
        "'baseline'",
        "pathway",
        "ssp245",
        "ssp585",
    )


def test_a_table_scenario_column_that_contradicts_its_source_raises() -> None:
    column = {"values": ["x"], "distinct": 1}
    columns = {
        "model": column,
        "member": column,
        "scenario": {"values": ["ssp585"], "distinct": 1},
    }
    _refused(
        [_annual("a", _series(_stated("ssp245"), table_columns=columns))],
        "'a'",
        "pathway",
        "ssp585",
    )


def test_an_unknown_pathway_raises_and_a_registered_one_passes() -> None:
    _refused([_curve("a", _row(2050, pathway="my-path"))], "'a'", "my-path")
    register_pathway("my-registered-path")
    ScenarioSet([_curve("a", _row(2050, pathway="my-registered-path"))], reference="a")


def test_one_pathway_horizon_and_product_raises() -> None:
    _refused(
        [_curve("a", _row(2050)), _curve("b", _row(2050, location=95.0))],
        *BOTH,
        "pathway, horizon and product",
    )


def test_equivalent_spellings_are_one_pathway() -> None:
    _refused(
        [
            _curve("a", _row(2000, pathway="historic")),
            _curve("b", _row(2000, pathway="historical", location=95.0)),
        ],
        *BOTH,
        "pathway, horizon and product",
    )


def test_a_null_equals_a_null() -> None:
    _refused(
        [
            _table("a", source={"provider": "fixture"}),
            _table("b", source={"provider": "fixture"}),
        ],
        *BOTH,
        "pathway, horizon and product",
    )


def test_two_series_with_one_centre_raise() -> None:
    _refused(
        [
            _annual("a", _series(_stated("ssp245"), first=1985, count=30)),
            _annual("b", _series(_stated("ssp245"), first=1990, count=20)),
        ],
        *BOTH,
        "pathway, horizon and product",
    )


def test_every_pair_of_scenarios_is_compared() -> None:
    _refused(
        [
            _curve("r", _row(2030)),
            _curve("a", _row(2050)),
            _curve("b", _row(2050, location=95.0)),
        ],
        *BOTH,
        "pathway, horizon and product",
    )


def test_two_scenarios_that_differ_in_one_of_the_three_pass() -> None:
    held = ScenarioSet(
        [
            _curve("a", _row(2050, pathway="ssp245")),
            _curve("b", _row(2050, pathway="ssp585", location=95.0)),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_equal_curves_at_two_horizons_raise() -> None:
    _refused(
        [
            _curve("a", _row(2030, location=50.0)),
            _curve("b", _row(2050, location=50.0)),
        ],
        *BOTH,
        "equal curve",
        "2030",
        "2050",
    )


def test_equal_curves_of_any_one_asset_raise() -> None:
    _refused(
        [
            _curves("a", {"a": _row(2030), "b": _row(2030, location=90.0)}),
            _curves("b", {"a": _row(2050), "b": _row(2050, location=90.0)}),
        ],
        *BOTH,
        "equal curve",
        "asset 'b'",
    )


def test_curves_of_another_product_at_one_pair_are_not_compared() -> None:
    held = ScenarioSet(
        [
            _curve("a", _row(2050, location=50.0)),
            _curve("b", _row(2050, location=50.0), metadata=_dataset("other")),
        ],
        reference="a",
        mixed_products=True,
    )
    assert held.labels == ("a", "b")


def test_a_curve_its_record_cannot_describe_is_not_compared() -> None:
    def undescribed(document: dict[str, Any]) -> None:
        document["sources"][0]["curve"] = {"kind": "other", "type": "Mine"}

    held = ScenarioSet(
        [
            _edited(_curve("a", _row(2030)), undescribed),
            _edited(_curve("b", _row(2050)), undescribed),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_a_horizon_outside_the_window_raises() -> None:
    window = TemporalWindow(start_year=2020, end_year=2040)
    message = _refused(
        [_curve("a", _row(2050), metadata=_dataset(temporal_window=window))],
        "'a'",
        "temporal_window",
    )
    assert "2050" in message


def test_a_horizon_inside_the_window_passes() -> None:
    window = TemporalWindow(start_year=2020, end_year=2060)
    held = ScenarioSet(
        [_curve("a", _row(2050), metadata=_dataset(temporal_window=window))],
        reference="a",
    )
    assert held.labels == ("a",)


def test_a_horizon_other_than_a_time_invariant_year_raises() -> None:
    window = TemporalWindow(kind="time_invariant", reference_year=2050)
    _refused(
        [_curve("a", _row(2030), metadata=_dataset(temporal_window=window))],
        "'a'",
        "temporal_window",
        "time-invariant",
    )


def test_a_pair_the_data_does_not_hold_raises() -> None:
    with pytest.raises(ScenarioError, match="'x'.*ssp585.*2050.*asset 'b'"):
        _check_pairs_held(
            "x",
            [("ssp585", 2050)],
            {"asset 'a'": [("ssp585", 2050)], "asset 'b'": [("ssp585", 2045)]},
        )


def test_a_pair_is_held_under_an_equivalent_spelling() -> None:
    _check_pairs_held("x", [("historic", 2000)], {"a": [("historical", 2000)]})


def test_two_products_raise_unless_stated() -> None:
    entries = [
        _curve("a", _row(2050)),
        _curve("b", _row(2050, location=95.0), metadata=_dataset("other")),
    ]
    _refused(entries, *BOTH, "product", "fixture", "other", "mixed_products=True")
    assert ScenarioSet(entries, reference="a", mixed_products=True).labels == (
        "a",
        "b",
    )


def test_two_products_raise_whichever_is_the_reference() -> None:
    entries = [
        _curve("a", _row(2050), metadata=_dataset("other")),
        _curve("b", _row(2050, location=95.0)),
        _curve("c", _row(2070), metadata=_dataset("other")),
    ]
    _refused(entries, *BOTH, "product")


def test_a_null_product_beside_one_is_two_products() -> None:
    _refused(
        [
            _table("a", source={"provider": "fixture"}),
            _table("b", source={"pathway": "ssp245"}),
        ],
        *BOTH,
        "product",
    )


def test_a_different_generator_raises() -> None:
    _refused(
        [
            _curve("a", _row(2030)),
            _curve("b", _row(2050), rng=np.random.default_rng(8)),
        ],
        *BOTH,
        "rng",
    )


def test_reusing_one_generator_raises() -> None:
    generator = np.random.default_rng(SEED)
    _refused(
        [
            _curve("a", _row(2030), rng=generator),
            _curve("b", _row(2050), rng=generator),
        ],
        *BOTH,
        "rng",
        "seed sequence",
    )


def test_a_generator_of_one_asset_differs() -> None:
    rngs = {"a": np.random.default_rng(1), "b": np.random.default_rng(9)}
    _refused(
        [
            _curves("a", {"a": _row(2030), "b": _row(2030, location=90.0)}),
            _curves(
                "b",
                {"a": _row(2050), "b": _row(2050, location=95.0)},
                rngs=rngs,
            ),
        ],
        *BOTH,
        "rng",
        "asset 'b'",
    )


def test_no_generator_record_raises() -> None:
    def no_records(document: dict[str, Any]) -> None:
        document["rng"] = ()

    _refused(
        [
            _edited(_curve("a", _row(2030)), no_records),
            _edited(_curve("b", _row(2050)), no_records),
        ],
        *BOTH,
        "rng",
    )


def test_other_periods_raise() -> None:
    _refused(
        [_curve("a", _row(2030)), _curve("b", _row(2050), n_draws=60)],
        *BOTH,
        "period",
    )


def test_blocks_of_a_holdout_share_the_draws_of_the_same_side() -> None:
    a = holdout(_curve("a", _row(2030)))
    b = holdout(_curve("b", _row(2050)))
    held = ScenarioSet([a[1], b[1]], reference="a")
    assert held.labels == ("a", "b")


def test_blocks_of_opposite_sides_of_a_holdout_raise() -> None:
    a = holdout(_curve("a", _row(2030)))
    b = holdout(_curve("b", _row(2050)))
    _refused([a[0], b[1]], *BOTH, "period")


def test_years_need_not_share_rows() -> None:
    held = ScenarioSet(
        [_years("ssp245", 2000, "a"), _years("ssp585", 2041, "b")], reference="a"
    )
    assert held.labels == ("a", "b")


def test_dependence_differs() -> None:
    first = {"a": _row(2030), "b": _row(2030, location=90.0)}
    second = {"a": _row(2050), "b": _row(2050, location=95.0)}
    _refused(
        [
            _curves("a", first),
            _curves("b", second, dependence=Comonotonic()),
        ],
        *BOTH,
        "dependence",
    )


def _reference(seed: int) -> OutcomeSample:
    return OutcomeSample.join(
        [
            _annual("baseline", _series(seed=seed), asset_id="a"),
            _annual("baseline", _series(seed=seed + 1), asset_id="b"),
        ]
    )


def test_an_empirical_copula_of_another_reference_differs() -> None:
    first = {"a": _row(2030), "b": _row(2030, location=90.0)}
    second = {"a": _row(2050), "b": _row(2050, location=95.0)}
    _refused(
        [
            _curves("a", first, dependence=EmpiricalCopula(_reference(3))),
            _curves("b", second, dependence=EmpiricalCopula(_reference(30))),
        ],
        *BOTH,
        "dependence",
    )


def test_a_window_beside_a_single_horizon_under_an_empirical_copula_passes() -> None:
    copula = EmpiricalCopula(_reference(3))
    window = _book_window("a", WINDOW, dependence=copula)
    single = _curves(
        "b",
        {"a": _row(2080), "b": _row(2080, location=170.0)},
        dependence=copula,
    )
    assert window.provenance.dependence is not None
    assert single.provenance.dependence is not None
    assert window.provenance.dependence["k"] != single.provenance.dependence["k"]
    assert ScenarioSet([window, single], reference="a").labels == ("a", "b")


# -- how each refusal reads on a window ----------------------------------------


def test_two_labels_with_the_same_members_raise() -> None:
    _refused(
        [_window("a"), _window("b", fields={"curve_scale": 12.0})],
        *BOTH,
        "pathway, horizon and product",
        "2055",
    )


def test_the_order_members_are_given_in_does_not_matter() -> None:
    _refused(
        [_window("a"), _window("b", WINDOW[::-1], fields={"curve_scale": 12.0})],
        *BOTH,
        "pathway, horizon and product",
    )


def test_two_windows_with_one_centre_and_different_members_pass() -> None:
    held = ScenarioSet(
        [_window("a", (2045, 2050, 2055)), _window("b", (2040, 2050, 2060))],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_a_single_horizon_at_a_windows_centre_passes() -> None:
    held = ScenarioSet(
        [_window("a"), _curve("b", _row(2050, location=999.0))], reference="a"
    )
    assert held.labels == ("a", "b")


def test_a_window_with_equivalent_spelling_is_the_same_members() -> None:
    _refused(
        [
            _window("a", (1990, 2000), fields={"pathway": "historic"}),
            _window(
                "b",
                (1990, 2000),
                fields={"pathway": "historical", "curve_scale": 12.0},
            ),
        ],
        *BOTH,
        "pathway, horizon and product",
    )


def test_a_window_over_a_time_invariant_layer_raises() -> None:
    message = _refused(
        [_curve("a", [_row(2045, location=50.0), _row(2055, location=50.0)])],
        "'a'",
        "equal curve",
        "2045",
        "2055",
    )
    assert "scenarios 'a'" not in message


def test_two_windows_do_not_raise_for_the_horizons_they_share() -> None:
    held = ScenarioSet(
        [_window("a", (2045, 2050, 2055)), _window("b", (2050, 2055, 2060))],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_a_member_equal_to_another_scenarios_curve_raises() -> None:
    _refused(
        [
            _window("a", (2045, 2050)),
            _curve("b", _row(2080, location=50.0 + (2050 - 2020))),
        ],
        *BOTH,
        "equal curve",
        "2050",
        "2080",
    )


def test_equal_curves_read_every_asset_of_a_window() -> None:
    _refused(
        [
            _curves(
                "a",
                {
                    "a": [_row(2045), _row(2055)],
                    "b": [_row(2045, location=9.0), _row(2055, location=9.0)],
                },
            )
        ],
        "'a'",
        "equal curve",
        "asset 'b'",
    )


def test_every_member_is_inside_the_window() -> None:
    window = TemporalWindow(start_year=2020, end_year=2050)
    _refused(
        [_window("a", metadata=_dataset(temporal_window=window))],
        "'a'",
        "temporal_window",
        "2055",
    )


def test_a_member_the_data_of_an_asset_does_not_hold_raises() -> None:
    def lose_a_member(document: dict[str, Any]) -> None:
        part = document["parts"]["b"]
        part["sources"] = part["sources"][:-1]

    _refused(
        [_edited(_book_window("a", WINDOW), lose_a_member)],
        "'a'",
        "(pathway, horizon)",
        "2055",
        "asset 'b'",
    )


def test_every_member_is_checked_for_a_stated_pathway() -> None:
    _refused(
        [_window("a", details={"pathway": "ssp585"})],
        "'a'",
        "pathway",
        "ssp245",
        "ssp585",
    )


def test_two_pathways_in_a_window_raise() -> None:
    with pytest.raises(ScenarioError, match="one pathway"):
        _curve("a", [_row(2045), _row(2050, pathway="ssp585")])


def test_two_pathways_in_a_windows_record_raise() -> None:
    def two_pathways(document: dict[str, Any]) -> None:
        document["details"]["window"]["members"][1][0] = "ssp585"

    _refused(
        [_edited(_window("a"), two_pathways)],
        "'a'",
        "one pathway",
        "ssp245",
        "ssp585",
    )


def test_a_member_of_an_unknown_pathway_in_a_record_raises() -> None:
    def unknown(document: dict[str, Any]) -> None:
        document["details"]["window"]["members"][1][0] = "my-other-path"

    _refused([_edited(_window("a"), unknown)], "'a'", "my-other-path")


def test_a_horizon_twice_in_a_windows_record_raises() -> None:
    def repeated(document: dict[str, Any]) -> None:
        document["details"]["window"]["members"][1][1] = 2045

    _refused([_edited(_window("a"), repeated)], "'a'", "horizon", "2045")


@pytest.mark.parametrize(
    ("field", "value"),
    [("hazard_name", "other"), ("cell_index", "1234567890abcde")],
)
def test_members_that_differ_in_more_than_horizon_raise(field: str, value: str) -> None:
    def differs(document: dict[str, Any]) -> None:
        document["sources"][1]["row"][field] = value

    _refused([_edited(_window("a"), differs)], "'a'", field, "asset 'a'")


def test_members_of_another_dataset_raise() -> None:
    def differs(document: dict[str, Any]) -> None:
        document["sources"][1]["metadata"]["producer"] = "someone else"

    _refused([_edited(_window("a"), differs)], "'a'", "dataset")


def test_a_windows_draws_are_a_whole_multiple_of_its_members() -> None:
    def fewer(document: dict[str, Any]) -> None:
        document["details"]["draws"] = 100

    _refused(
        [_edited(_window("a"), fewer)],
        "'a'",
        "n_draws",
        "whole multiple of 3",
        "100",
    )


def test_a_hazard_that_differs_raises_for_a_window() -> None:
    _refused(
        [
            _window("a", (2045, 2050)),
            _window("b", (2055, 2060), fields={"hazard_name": "x"}),
        ],
        *BOTH,
        "hazard_name",
    )


def test_units_differ_for_a_window() -> None:
    _refused(
        [
            _window("a", (2045, 2050)),
            _window("b", (2055, 2060), metadata=_dataset(value_unit="mm/d")),
        ],
        *BOTH,
        "index_unit",
    )


def test_loss_models_differ_for_a_window() -> None:
    other = LossModel(RAMP, value=2e6, value_unit="EUR")
    _refused(
        [_window("a", (2045, 2050)), _window("b", (2055, 2060), loss_model=other)],
        *BOTH,
        "loss_model",
    )


def test_basis_risk_models_differ_for_a_window() -> None:
    _refused(
        [
            _window("a", (2045, 2050)),
            _window("b", (2055, 2060), basis_risk=ImpactNoise(0.2)),
        ],
        *BOTH,
        "basis_risk",
    )


def test_a_window_that_names_no_hazard_raises() -> None:
    _refused([_window("a", fields={"hazard_name": None})], "'a'", "hazard_name")


def test_two_products_raise_for_a_window() -> None:
    entries = [
        _window("a"),
        _window("b", metadata=_dataset("other"), fields={"curve_scale": 12.0}),
    ]
    _refused(entries, *BOTH, "product")
    assert ScenarioSet(entries, reference="a", mixed_products=True).labels == (
        "a",
        "b",
    )


def test_draws_are_shared_across_widths() -> None:
    held = ScenarioSet(
        [
            _window("a", (2045, 2050)),
            _window("b", (2055, 2060, 2065)),
            _curve("c", _row(2080)),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b", "c")


def test_a_window_with_another_generator_raises() -> None:
    _refused(
        [
            _window("a", (2045, 2050)),
            _window("b", (2055, 2060), rng=np.random.default_rng(8)),
        ],
        *BOTH,
        "rng",
    )


def test_a_window_with_other_periods_raises() -> None:
    _refused(
        [_window("a", (2045, 2050)), _window("b", (2055, 2060), n_draws=60)],
        *BOTH,
        "period",
    )


def test_a_window_keeps_its_dependence_comparison() -> None:
    _refused(
        [
            _book_window("a", (2045, 2050)),
            _book_window("b", (2055, 2060), dependence=Comonotonic()),
        ],
        *BOTH,
        "dependence",
    )


def test_an_empirical_copula_of_another_reference_differs_for_windows() -> None:
    _refused(
        [
            _book_window("a", WINDOW, dependence=EmpiricalCopula(_reference(3))),
            _book_window(
                "b", (2060, 2065, 2070), dependence=EmpiricalCopula(_reference(30))
            ),
        ],
        *BOTH,
        "dependence",
    )


# -- a bound cover -------------------------------------------------------------


def _at_cell(longitude: float, latitude: float, label: str = "a") -> Any:
    return _annual(
        label,
        _series(
            None, cell={"latitude": latitude, "longitude": longitude}, kind="nearest"
        ),
    )


def _at_station(station: str, label: str = "a") -> Any:
    return _annual(label, _series(None, where={"station": station}))


def test_a_cover_at_the_asset_reads_any_scenario() -> None:
    for sample in (_curve("a"), _annual("a"), _table("a"), _pool("a")):
        _check_bound_cover(_cover(), sample)


def test_a_cover_at_a_point_reads_the_series_centred_there() -> None:
    _check_bound_cover(_cover(AtPoint(-79.25, 43.75)), _at_cell(-79.25, 43.75))


def test_a_grid_that_runs_to_360_is_one_point() -> None:
    _check_bound_cover(_cover(AtPoint(-79.25, 43.75)), _at_cell(280.75, 43.75))


@pytest.mark.parametrize(
    "build",
    [
        lambda: _at_cell(-80.0, 43.75),
        lambda: _curve("a"),
        lambda: _table("a"),
        lambda: _annual("a"),
    ],
    ids=["elsewhere", "curve", "table", "no-cell"],
)
def test_a_cover_at_a_point_refuses_what_is_not_centred_there(build: Any) -> None:
    with pytest.raises(ScenarioError, match="'a'.*'bound'.*location"):
        _check_bound_cover(_cover(AtPoint(-79.25, 43.75)), build())


def test_a_cover_at_a_station_reads_that_stations_series() -> None:
    _check_bound_cover(_cover(AtStation("S1")), _at_station("S1"))


def test_a_cover_at_a_station_reads_a_one_valued_station_column() -> None:
    columns = {"station_id": {"values": ["S1"], "distinct": 1}}
    series = _series(None, table_columns=columns)
    _check_bound_cover(_cover(AtStation("S1")), _annual("a", series))


@pytest.mark.parametrize(
    "build",
    [
        lambda: _at_station("S2"),
        lambda: _curve("a"),
        lambda: _annual("a"),
    ],
    ids=["elsewhere", "curve", "no-station"],
)
def test_a_cover_at_a_station_refuses_what_is_not_there(build: Any) -> None:
    with pytest.raises(ScenarioError, match="'a'.*'bound'.*location"):
        _check_bound_cover(_cover(AtStation("S1")), build())


def test_a_cover_at_a_cell_reads_a_curve_there() -> None:
    _check_bound_cover(_cover(AtCell(CELL)), _curve("a"))
    _check_bound_cover(_cover(AtCell(CELL_HEX)), _curve("a"))


@pytest.mark.parametrize(
    "build",
    [
        lambda: _curve("a", _row(cell_index=OTHER_CELL)),
        lambda: _annual("a"),
        lambda: _curve(
            "a", FittedDistribution.from_parameters("gumbel_r", location=9.0, scale=1.0)
        ),
    ],
    ids=["elsewhere", "series", "no-cell"],
)
def test_a_cover_at_a_cell_refuses_what_is_not_there(build: Any) -> None:
    with pytest.raises(ScenarioError, match="'a'.*'bound'.*location"):
        _check_bound_cover(_cover(AtCell(CELL)), build())


def test_every_member_of_a_window_is_at_the_cell() -> None:
    _check_bound_cover(_cover(AtCell(CELL)), _window("a"))
    with pytest.raises(ScenarioError, match="location"):
        _check_bound_cover(_cover(AtCell(OTHER_CELL)), _window("a"))


def test_a_member_at_another_cell_is_refused() -> None:
    def elsewhere(document: dict[str, Any]) -> None:
        document["sources"][1]["row"]["cell_index"] = format(OTHER_CELL, "x")

    with pytest.raises(ScenarioError, match="location"):
        _check_bound_cover(_cover(AtCell(CELL)), _edited(_window("a"), elsewhere))


def test_a_separate_index_series_says_where_the_index_is_read() -> None:
    hazard = _series(None, where={"station": "S1"})
    index = _series(None, where={"station": "S2"}, seed=4)
    sample = _annual("a", hazard, index=index)
    _check_bound_cover(_cover(AtStation("S2")), sample)
    with pytest.raises(ScenarioError, match="location"):
        _check_bound_cover(_cover(AtStation("S1")), sample)


def test_every_run_of_a_pool_is_at_the_point() -> None:
    cell = {"latitude": 43.75, "longitude": -79.25}
    runs = [_run("M1", cell=cell), _run("M2", cell=cell)]
    _check_bound_cover(_cover(AtPoint(-79.25, 43.75)), _pool("a", runs))
    other = [_run("M1", cell=cell), _run("M2", cell={**cell, "longitude": -80.0})]
    with pytest.raises(ScenarioError, match="location"):
        _check_bound_cover(_cover(AtPoint(-79.25, 43.75)), _pool("a", other))


def test_a_cover_on_one_asset_of_several_reads_that_asset() -> None:
    sample = _curves(
        "a", {"a": _row(), "b": _row(cell_index=OTHER_CELL, location=90.0)}
    )
    _check_bound_cover(_cover(AtCell(CELL), asset_id="a"), sample)
    with pytest.raises(ScenarioError, match="asset 'b'"):
        _check_bound_cover(_cover(AtCell(CELL), asset_id="b"), sample)


def test_a_cover_with_no_asset_reads_every_asset() -> None:
    sample = _curves(
        "a", {"a": _row(), "b": _row(cell_index=OTHER_CELL, location=90.0)}
    )
    with pytest.raises(ScenarioError, match="asset 'b'"):
        _check_bound_cover(_cover(AtCell(CELL)), sample)


def _composite(rain: Any, heat: Any) -> Cover:
    index = CompositeIndex(
        {
            "rain": IndexDefinition(
                "rain", "mm/day", aggregation="annual_maximum", location=rain
            ),
            "heat": IndexDefinition(
                "heat", "mm/day", aggregation="annual_maximum", location=heat
            ),
        }
    )
    payout = SumOf(
        {
            "rain": LinearPayout(60.0, 120.0, 5e5),
            "heat": LinearPayout(60.0, 120.0, 5e5),
        }
    )
    return Cover(index, payout, payout_unit="EUR", label="bound")


def test_a_composite_cover_reads_each_components_location() -> None:
    sample = _hazards("a")
    _check_bound_cover(_composite(AtCell(CELL), AtAsset()), sample)
    with pytest.raises(ScenarioError, match="component 'heat'"):
        _check_bound_cover(_composite(AtCell(CELL), AtCell(OTHER_CELL)), sample)


def test_a_one_index_cover_reads_the_component_it_names() -> None:
    index = IndexDefinition(
        "heat", "mm/day", aggregation="annual_maximum", location=AtCell(OTHER_CELL)
    )
    cover = Cover(index, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR", label="c")
    with pytest.raises(ScenarioError, match="component 'heat'"):
        _check_bound_cover(cover, _hazards("a"))


# -- several hazards -----------------------------------------------------------


def test_samples_of_several_hazards_stand_side_by_side() -> None:
    held = ScenarioSet([_hazards("a", 2030), _hazards("b", 2050)], reference="a")
    assert held.labels == ("a", "b")


def test_components_differ_in_order() -> None:
    swapped = {
        "heat": _row(2050, hazard_name="heat", location=30.0),
        "rain": _row(2050, hazard_name="rain"),
    }
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, curves=swapped)],
        *BOTH,
        "components",
    )


def test_a_sample_of_several_hazards_beside_one_hazard_raises() -> None:
    _refused([_hazards("a", 2030), _curve("b", _row(2050))], *BOTH, "components")


def test_a_components_unit_differs() -> None:
    metadata = {"rain": _dataset(), "heat": _dataset(value_unit="mm/d")}
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, metadata=metadata)],
        *BOTH,
        "unit",
    )


def test_a_components_tail_differs() -> None:
    metadata = {"rain": _dataset(), "heat": _dataset(return_period_tail="lower")}
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, metadata=metadata)],
        *BOTH,
        "tail",
    )


def test_a_components_hazard_name_differs() -> None:
    curves = {
        "rain": _row(2050, hazard_name="rx1day"),
        "heat": _row(2050, hazard_name="heat", location=30.0),
    }
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, curves=curves)],
        *BOTH,
        "hazard_name",
        "component 'rain'",
    )


def test_a_component_with_an_impact_must_name_its_hazard() -> None:
    def silent(document: dict[str, Any]) -> None:
        document["components"]["rain"].pop("hazard")

    _refused([_edited(_hazards("a"), silent)], "'a'", "component 'rain'")


def test_a_component_with_no_impact_needs_no_name() -> None:
    def silent(document: dict[str, Any]) -> None:
        document["components"]["heat"].pop("hazard")

    held = ScenarioSet([_edited(_hazards("a"), silent)], reference="a")
    assert held.labels == ("a",)


def test_a_table_of_several_hazards_names_each() -> None:
    _refused([_components_table("a")], "'a'", "component 'rain'")
    held = ScenarioSet(
        [_components_table("a", hazard_name={"rain": "rain", "heat": "heat"})],
        reference="a",
    )
    assert held.labels == ("a",)


def test_a_components_loss_model_differs() -> None:
    other = {"rain": LossModel(RAMP, value=2e6, value_unit="EUR")}
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, loss_models=other)],
        *BOTH,
        "loss_model",
    )


def test_the_rule_differs() -> None:
    both = {"rain": LOSS, "heat": LOSS}
    _refused(
        [
            _hazards("a", 2030, loss_models=both, combine=CappedSum(1.0)),
            _hazards("b", 2050, loss_models=both, combine=CompoundDamage()),
        ],
        *BOTH,
        "combine",
    )


def test_hazard_dependence_differs() -> None:
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, dependence=Comonotonic())],
        *BOTH,
        "hazard_dependence",
    )


def test_a_components_generator_differs() -> None:
    rngs = {"rain": np.random.default_rng(11), "heat": np.random.default_rng(99)}
    _refused(
        [_hazards("a", 2030), _hazards("b", 2050, rngs=rngs)],
        *BOTH,
        "rng",
        "component 'heat'",
    )


def test_equal_curves_of_any_one_component_raise() -> None:
    first = {
        "rain": _row(2030, hazard_name="rain"),
        "heat": _row(2030, hazard_name="heat", location=30.0),
    }
    second = {
        "rain": _row(2050, hazard_name="rain"),
        "heat": _row(2050, hazard_name="heat", location=30.0),
    }
    _refused(
        [_hazards("a", curves=first), _hazards("b", curves=second)],
        *BOTH,
        "equal curve",
        "component 'heat'",
    )


def test_a_window_beside_one_horizon_passes_under_a_copula_of_hazards() -> None:
    reference = sample_annual_hazards(
        {"rain": _series(), "heat": _series(seed=4)},
        tails={"rain": "upper", "heat": "upper"},
        loss_models={"rain": LOSS},
        basis_risk=NoBasisRisk(),
        asset_id="a",
    )
    copula = EmpiricalCopula(reference)
    window = _hazards(
        "a",
        dependence=copula,
        curves={
            "rain": [_row(year, hazard_name="rain") for year in WINDOW],
            "heat": [
                _row(year, hazard_name="heat", location=9.0 + year) for year in WINDOW
            ],
        },
    )
    single = _hazards("b", 2080, dependence=copula)
    assert ScenarioSet([window, single], reference="a").labels == ("a", "b")


def test_two_products_compare_the_joined_providers() -> None:
    metadata = {"rain": _dataset("one"), "heat": _dataset("two")}
    entries = [
        _hazards("a", 2030, metadata=metadata),
        _hazards("b", 2050),
    ]
    _refused(entries, *BOTH, "product", "one+two")
    held = ScenarioSet(entries, reference="a", mixed_products=True)
    assert held.labels == ("a", "b")


# -- the cell each asset is read in ------------------------------------------------

TORONTO = (43.65, -79.38)
HAMILTON = (43.26, -79.87)


def test_curves_of_one_asset_from_two_cells_raise() -> None:
    entries = [
        _curve("2030s", _row(2030)),
        _curve("2050s", _row(2050, cell_index=OTHER_CELL)),
    ]
    _refused(
        entries,
        "'2030s'",
        "'2050s'",
        "cell_index",
        "asset 'a'",
        repr(CELL_HEX),
        repr(format(OTHER_CELL, "x")),
    )


def test_a_window_beside_a_horizon_in_another_cell_raises() -> None:
    entries = [_window("a"), _curve("b", _row(2080, cell_index=OTHER_CELL))]
    _refused(entries, *BOTH, "cell_index", "asset 'a'")


def test_each_asset_of_a_book_is_read_in_one_cell() -> None:
    moved = {"a": _row(2080), "b": _row(2080, cell_index=OTHER_CELL, location=90.0)}
    _refused([_curves("a"), _curves("b", moved)], *BOTH, "cell_index", "asset 'b'")


def test_each_hazard_of_an_asset_is_read_in_one_cell() -> None:
    moved = {
        "rain": _row(2080, hazard_name="rain"),
        "heat": _row(2080, hazard_name="heat", location=80.0, cell_index=OTHER_CELL),
    }
    entries = [_hazards("a"), _hazards("b", 2080, curves=moved)]
    _refused(entries, *BOTH, "cell_index", "component 'heat'")


def test_a_curve_without_a_row_has_no_cell_to_compare() -> None:
    bare = FittedDistribution.from_parameters("gumbel_r", location=75.0, scale=10.0)
    entries = [_curve("a", _row(2050)), _curve("b", bare, hazard_name="rx1day")]
    assert ScenarioSet(entries, reference="a").labels == ("a", "b")


def _h3_cell(h3: Any, point: tuple[float, float], resolution: int) -> int:
    return int(h3.str_to_int(h3.latlng_to_cell(*point, resolution)))


def _at(label: str, horizon: int, cell: int, resolution: int) -> OutcomeSample:
    return _curve(
        label,
        _row(horizon, cell_index=cell),
        metadata=_dataset(h3_resolution=resolution),
    )


def test_another_place_at_one_resolution_raises() -> None:
    h3 = pytest.importorskip("h3")
    toronto, hamilton = (_h3_cell(h3, place, 8) for place in (TORONTO, HAMILTON))
    entries = [_at("2030s", 2030, toronto, 8), _at("2050s", 2050, hamilton, 8)]
    message = _refused(entries, "'2030s'", "'2050s'", "cell_index")
    assert "two resolutions" not in message


def test_cells_at_two_resolutions_are_compared_at_the_coarser() -> None:
    h3 = pytest.importorskip("h3")
    fine = _h3_cell(h3, TORONTO, 8)
    parent = h3.cell_to_parent(h3.int_to_str(fine), 5)
    beside = next(cell for cell in h3.grid_disk(parent, 1) if cell != parent)
    # A point's cells at two resolutions nest up to one neighbouring cell.
    for coarse in (parent, beside):
        entries = [
            _at("a", 2030, fine, 8),
            _at("b", 2050, int(h3.str_to_int(coarse)), 5),
        ]
        assert ScenarioSet(entries, reference="a").labels == ("a", "b")
    elsewhere = _h3_cell(h3, HAMILTON, 5)
    entries = [_at("a", 2030, fine, 8), _at("b", 2050, elsewhere, 5)]
    _refused(entries, *BOTH, "cell_index", "two resolutions")


def test_a_cell_that_is_no_h3_cell_is_another_place() -> None:
    pytest.importorskip("h3")
    # The resolution bits say 5, but the digits below it are not H3's.
    invalid = (CELL & ~(0xF << 52)) | (5 << 52)
    entries = [_at("a", 2030, CELL, 8), _at("b", 2050, invalid, 5)]
    _refused(entries, *BOTH, "cell_index")


def test_two_resolutions_need_the_geometry_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h3 = pytest.importorskip("h3")
    fine = _h3_cell(h3, TORONTO, 8)
    coarse = int(h3.str_to_int(h3.cell_to_parent(h3.int_to_str(fine), 5)))
    entries = [_at("a", 2030, fine, 8), _at("b", 2050, coarse, 5)]
    monkeypatch.setitem(sys.modules, "h3", None)
    with pytest.raises(ImportError, match=r"crc-sdk\[geometry\]"):
        ScenarioSet(entries, reference="a")


# -- the place each per-year series is read at -------------------------------------

STEP = {"latitude": 0.25, "longitude": 0.25}


def _cell(latitude: float, longitude: float, step: Any = STEP) -> dict[str, Any]:
    """A nearest-cell record: the matched centre and, if given, the grid step."""
    record: dict[str, Any] = {
        "kind": "nearest_cell",
        "cell": {"latitude": latitude, "longitude": longitude},
    }
    if step is not None:
        record["spacing_degrees"] = step
    return record


def _placed(label: str, first: int, **provenance: Any) -> Any:
    """A per-year scenario of one product whose series records `provenance`."""
    return _annual(label, _series(_stated("ssp245"), first=first, **provenance))


def _at_cells(*cells: dict[str, Any]) -> list[Any]:
    """One scenario per cell record, labelled a, b, c, ..., forty years apart."""
    return [
        _placed(label, 1980 + 40 * i, **cell)
        for i, (label, cell) in enumerate(zip("abcdefgh", cells, strict=False))
    ]


def test_series_from_one_cell_stand_side_by_side() -> None:
    held = ScenarioSet(
        _at_cells(_cell(43.625, -79.375), _cell(43.625, -79.375)), reference="a"
    )
    assert held.labels == ("a", "b")


def test_another_cell_of_one_grid_raises() -> None:
    # The neighbouring cell to the west: one step, centres one step apart.
    message = _refused(
        _at_cells(_cell(43.625, -79.375), _cell(43.625, -79.625)),
        *BOTH,
        "differ in cell",
        "asset 'a'",
        "latitude 43.625, longitude -79.375",
        "latitude 43.625, longitude -79.625",
        "two cells of one grid",
    )
    assert "the hazard series" in message


def test_a_grid_that_runs_to_360_is_one_grid() -> None:
    held = ScenarioSet(
        _at_cells(_cell(43.625, -79.375), _cell(43.625, 280.625)), reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        _at_cells(_cell(43.625, -79.375), _cell(43.625, 280.375)),
        *BOTH,
        "two cells of one grid",
    )


def test_cells_of_two_grids_must_overlap() -> None:
    # A 1-degree cell holds the 0.25-degree cell; a cell two degrees away doesn't.
    coarse = {"latitude": 1.0, "longitude": 1.0}
    held = ScenarioSet(
        _at_cells(_cell(43.625, -79.375), _cell(43.5, -79.5, coarse)), reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        _at_cells(_cell(43.625, -79.375), _cell(45.5, -79.5, coarse)),
        *BOTH,
        "differ in cell",
        "(step 1 by 1 degrees)",
        "no point lies within half a grid step of both centres",
    )


def test_two_grids_of_one_step_offset_by_half_a_step_must_overlap() -> None:
    # As ERA5's centres at whole quarters beside NEX-GDDP-CMIP6's at odd eighths.
    held = ScenarioSet(
        _at_cells(_cell(43.625, -79.375), _cell(43.5, -79.25)), reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        _at_cells(_cell(43.625, -79.375), _cell(43.25, -79.25)),
        *BOTH,
        "no point lies within half a grid step of both centres",
    )


def test_a_cell_without_a_step_is_a_point() -> None:
    held = ScenarioSet(
        _at_cells(_cell(43.625, -79.375), _cell(43.7, -79.3, step=None)),
        reference="a",
    )
    assert held.labels == ("a", "b")
    _refused(
        _at_cells(_cell(43.625, -79.375), _cell(43.8, -79.3, step=None)),
        *BOTH,
        "differ in cell",
    )
    _refused(
        _at_cells(_cell(43.7, -79.3, step=None), _cell(43.70001, -79.3, step=None)),
        *BOTH,
        "differ in cell",
    )


def test_every_pair_of_per_year_scenarios_is_compared() -> None:
    # Each fine cell lies in the coarse reference's, but they are two cells.
    coarse = {"latitude": 1.0, "longitude": 1.0}
    _refused(
        _at_cells(
            _cell(43.5, -79.5, coarse),
            _cell(43.625, -79.375),
            _cell(43.875, -79.375),
        ),
        "'b'",
        "'c'",
        "two cells of one grid",
    )


def test_another_station_raises() -> None:
    held = ScenarioSet(
        [
            _placed("a", 1980, where={"station": "S1"}),
            _placed(
                "b",
                2020,
                table_columns={"station_id": {"values": ["S1"], "distinct": 1}},
            ),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")
    _refused(
        [
            _placed("a", 1980, where={"station": "S1"}),
            _placed("b", 2020, where={"station": "S2"}),
        ],
        *BOTH,
        "differ in station",
        "station 'S1'",
        "station 'S2'",
    )


def test_a_filter_on_other_columns_names_no_place() -> None:
    held = ScenarioSet(
        [
            _placed("a", 1980, where={"model": "M1", "scenario": "historical"}),
            _placed("b", 2020, where={"model": "M1", "scenario": "ssp245"}),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_a_cell_beside_a_station_is_not_compared() -> None:
    held = ScenarioSet(
        [
            _placed("a", 1980, **_cell(43.625, -79.375)),
            _placed("b", 2020, where={"station": "S1"}),
        ],
        reference="a",
    )
    assert held.labels == ("a", "b")


def test_the_index_series_is_compared_with_the_index_series() -> None:
    def scenario(label: str, first: int, station: str) -> Any:
        hazard = _series(_stated("ssp245"), first=first, **_cell(43.625, -79.375))
        index = _series(None, first=first, seed=4, where={"station": station})
        return _annual(label, hazard, index=index)

    # The hazard is read at a cell and the index at a station: each by its own.
    held = ScenarioSet(
        [scenario("a", 1980, "S1"), scenario("b", 2020, "S1")], reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        [scenario("a", 1980, "S1"), scenario("b", 2020, "S2")],
        *BOTH,
        "the index series of asset 'a'",
        "station 'S2'",
    )


def test_each_run_of_a_pool_is_compared() -> None:
    def pool(label: str, pathway: str, longitude: float) -> Any:
        runs = [
            _run("M1", pathway, **_cell(43.625, -79.375)),
            _run("M2", pathway, **_cell(43.625, longitude)),
        ]
        return _pool(label, runs)

    held = ScenarioSet(
        [pool("a", "ssp245", -79.375), pool("b", "ssp585", -79.375)], reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        [pool("a", "ssp245", -79.375), pool("b", "ssp585", -79.625)],
        *BOTH,
        "differ in cell",
        "(run M2 r1i1p1f1) in the grid cell centred at latitude 43.625, "
        "longitude -79.625",
    )


def test_runs_on_their_own_grids_overlap() -> None:
    # CMIP6 models each on its own grid: run M2's coarse cell holds M1's.
    coarse = {"latitude": 1.25, "longitude": 1.875}
    runs = [
        _run("M1", **_cell(43.625, -79.375)),
        _run("M2", **_cell(43.75, -79.6875, coarse)),
    ]
    later = [
        _run("M1", "ssp585", **_cell(43.625, -79.375)),
        _run("M2", "ssp585", **_cell(43.75, -79.6875, coarse)),
    ]
    held = ScenarioSet([_pool("a", runs), _pool("b", later)], reference="a")
    assert held.labels == ("a", "b")


def test_each_asset_of_a_joined_sample_is_compared() -> None:
    def joined(label: str, first: int, longitude: float) -> OutcomeSample:
        return OutcomeSample.join(
            [
                _annual(
                    label,
                    _series(_stated("ssp245"), first=first, **_cell(43.625, -79.375)),
                    asset_id="a",
                ),
                _annual(
                    label,
                    _series(
                        _stated("ssp245"),
                        first=first,
                        seed=4,
                        **_cell(43.625, longitude),
                    ),
                    asset_id="b",
                ),
            ]
        )

    held = ScenarioSet(
        [joined("a", 1980, -79.125), joined("b", 2020, -79.125)], reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        [joined("a", 1980, -79.125), joined("b", 2020, -78.875)],
        *BOTH,
        "asset 'b'",
    )


def test_each_component_of_per_year_hazards_is_compared() -> None:
    def hazards(label: str, first: int, station: str) -> OutcomeSample:
        sample: OutcomeSample = sample_annual_hazards(
            {
                "rain": _series(
                    _stated("ssp245"), first=first, where={"station": "S1"}
                ),
                "heat": _series(
                    {**_stated("ssp245"), "recipe": "heat"},
                    first=first,
                    seed=4,
                    where={"station": station},
                ),
            },
            tails={"rain": "upper", "heat": "upper"},
            loss_models={"rain": LOSS},
            basis_risk=NoBasisRisk(),
            asset_id="a",
            scenario=label,
        )
        return sample

    held = ScenarioSet(
        [hazards("a", 1980, "S1"), hazards("b", 2020, "S1")], reference="a"
    )
    assert held.labels == ("a", "b")
    _refused(
        [hazards("a", 1980, "S1"), hazards("b", 2020, "S2")],
        *BOTH,
        "component 'heat'",
    )
