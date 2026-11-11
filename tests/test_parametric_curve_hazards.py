"""`sample_curve_hazards` and the stated dependence across hazards, offline.

The numbers are synthetic. The tests check what the module computes and records;
nothing here is a finding about a hazard, a dependence model or a cover form.
"""

from __future__ import annotations

import inspect
import json
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact
from numpy.typing import NDArray

from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    AnnualSeries,
    CappedSum,
    ComonotoneIndexWarning,
    Comonotonic,
    CompoundDamage,
    Cover,
    DependenceModel,
    EmpiricalCopula,
    GaussianCopula,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LeakageError,
    LossModel,
    NoBasisRisk,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    compare,
    holdout,
    sample_annual_hazards,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import ReturnPeriodExtrapolationWarning

UNTYPED_HAZARDS: Any = sample_curve_hazards  # for deliberately invalid arguments
UNTYPED_ASSETS: Any = sample_curves
UNTYPED_GAUSSIAN: Any = GaussianCopula
NAMES = ("rain", "heat")
SEEDS = {"rain": 11, "heat": 22, "wind": 33}
UNITS = {"rain": "mm/day", "heat": "degC", "wind": "m/s"}
RAMP = PiecewiseLinearImpact([20.0, 140.0], [0.0, 1.0])
NOISE = ImpactNoise(0.3, upper=1.0)
ASSET = "site"


def _metadata(name: str = "rain", **fields: Any) -> HazardDatasetMetadata:
    base: dict[str, Any] = {
        "h3_resolution": 7,
        "value_unit": UNITS[name],
        "value_semantics": f"annual maximum {name}",
        "producer": "tests",
        "source": SourceProvenance(
            provider="fixture",
            dataset=name,
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        "creation_version": "1",
        "probability_semantics": "annual_exceedance",
    }
    base.update(fields)
    return HazardDatasetMetadata(**base)


def _curve(name: str) -> FittedDistribution:
    return FittedDistribution.from_parameters(
        "gumbel_r", location={"rain": 50.0, "heat": 80.0, "wind": 30.0}[name], scale=10
    )


def _loss(name: str) -> LossModel:
    return LossModel(RAMP, value=1e6, value_unit="EUR", name=f"loss of {name}")


def _rngs(
    names: Sequence[str] = NAMES, shift: int = 0
) -> dict[str, np.random.Generator]:
    return {name: np.random.default_rng(SEEDS[name] + shift) for name in names}


def _options(names: Sequence[str] = NAMES) -> dict[str, Any]:
    return {
        "curves": {name: _curve(name) for name in names},
        "metadata": {name: _metadata(name) for name in names},
        "rngs": _rngs(names),
        "loss_models": {name: _loss(name) for name in names},
        "basis_risk": NOISE,
        "combine": CappedSum(1.0),
        "asset_id": ASSET,
    }


def _draws(dependence: Any, n: int = 300, **overrides: Any) -> OutcomeSample:
    arguments = {**_options(), **overrides}
    sample: OutcomeSample = UNTYPED_HAZARDS(
        arguments.pop("curves"), dependence=dependence, n_draws=n, **arguments
    )
    return sample


def _single(name: str, n: int = 300) -> OutcomeSample:
    return sample_curve(
        _curve(name),
        metadata=_metadata(name),
        n_draws=n,
        rng=np.random.default_rng(SEEDS[name]),
        loss_model=_loss(name),
        basis_risk=NOISE,
        asset_id=ASSET,
    )


def _uniforms(names: Sequence[str] = NAMES, n: int = 300) -> NDArray[np.float64]:
    return np.column_stack(
        [np.random.default_rng(SEEDS[name]).random(n) for name in names]
    )


def _ranks(values: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")


def _spearman(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    return float(np.corrcoef(_ranks(first), _ranks(second))[0, 1])


def _reference(
    m: int = 30, *, seed: int = 5, names: Sequence[str] = NAMES
) -> OutcomeSample:
    """A per-year sample of several hazards, whose columns have their own ranks."""
    rng = np.random.default_rng(seed)
    series = {
        name: AnnualSeries(
            np.arange(1990, 1990 + m),
            rng.gamma(2.0, 20.0, m),
            UNITS[name],
            {"kind": "table", "source": {"name": f"era5 {name}"}},
        )
        for name in names
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        return sample_annual_hazards(
            series,
            tails=dict.fromkeys(names, "upper"),
            loss_models={names[0]: _loss(names[0])},
            basis_risk=NoBasisRisk(),
            asset_id=ASSET,
        )


def _models() -> list[Any]:
    """Every built-in model, each fit for the default 300 draws."""
    return [
        Independent(),
        Comonotonic(),
        EmpiricalCopula(_reference()),
        GaussianCopula(0.6),
    ]


def _rows(sample: OutcomeSample, name: str) -> list[tuple[float, float, float]]:
    """A component's (hazard, index, own loss) rows, sorted to compare as multisets."""
    assert sample.index_components is not None
    component = sample.index_components[name]
    hazard = component.values if component.hazard is None else component.hazard
    assert component.own_loss is not None
    return sorted(
        zip(
            hazard[:, 0].tolist(),
            component.values[:, 0].tolist(),
            component.own_loss[:, 0].tolist(),
            strict=True,
        )
    )


def _own_rows(single: OutcomeSample) -> list[tuple[float, float, float]]:
    hazard = single.index if single.hazard is None else single.hazard
    return sorted(
        zip(
            hazard[:, 0].tolist(),
            single.index[:, 0].tolist(),
            single.loss[:, 0].tolist(),
            strict=True,
        )
    )


def _values(sample: OutcomeSample, name: str) -> NDArray[np.float64]:
    assert sample.index_components is not None
    values: NDArray[np.float64] = sample.index_components[name].values[:, 0]
    return values


def _text(sample: OutcomeSample) -> str:
    cover = Cover(
        IndexDefinition("rain", "mm/day", aggregation="annual_maximum"),
        StepPayout([60.0], [800.0]),
        payout_unit="EUR",
        label="r",
    )
    return setup_text(compare([cover], sample, split="in_sample"))


def _line(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(start))


# -- no default --------------------------------------------------------------


def test_the_dependence_has_no_default() -> None:
    parameter = inspect.signature(sample_curve_hazards).parameters["dependence"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    options = _options()
    with pytest.raises(TypeError, match="dependence"):
        UNTYPED_HAZARDS(options.pop("curves"), n_draws=10, **options)


@pytest.mark.parametrize("value", [None, "independent", object(), Independent])
def test_the_dependence_must_be_a_stated_model(value: Any) -> None:
    with pytest.raises(TypeError, match="no default"):
        _draws(value)


def test_a_bare_callable_is_refused() -> None:
    def reorder(uniforms: Any, columns: Any) -> Any:
        return np.argsort(uniforms, axis=0)

    with pytest.raises(TypeError, match="DependenceModel"):
        _draws(reorder)


# -- each component holds sample_curve's draws in another order ---------------


def test_the_rows_are_draws_of_one_asset() -> None:
    sample = _draws(Independent())
    assert sample.asset_ids == (ASSET,)
    assert sample.period.tolist() == list(range(300))
    np.testing.assert_allclose(sample.probability, 1 / 300)
    assert sample.semantics == "annual"
    assert sample.loss_unit == "EUR"
    assert sample.provenance.source_kind == "monte_carlo"
    assert sample.provenance.semantics_source == "dataset"
    assert sample.provenance.probability_semantics == "annual_exceedance"


def test_the_order_of_the_curves_sets_the_components() -> None:
    sample = _draws(Independent())
    assert sample.index_components is not None
    assert list(sample.index_components) == ["rain", "heat"]
    assert np.array_equal(sample.index[:, 0], _values(sample, "rain"))
    assert sample.index_unit == "mm/day"
    curves = {name: _curve(name) for name in ("heat", "rain")}
    swapped = _draws(Independent(), curves=curves)
    assert swapped.index_components is not None
    assert list(swapped.index_components) == ["heat", "rain"]
    assert swapped.index_unit == "degC"
    assert np.array_equal(swapped.index[:, 0], _values(swapped, "heat"))


def test_each_component_states_its_unit_and_tail() -> None:
    metadata = {
        "rain": _metadata("rain"),
        "heat": _metadata("heat", return_period_tail="lower"),
    }
    sample = _draws(Independent(), metadata=metadata)
    components = sample.index_components
    assert components is not None
    assert (components["rain"].unit, components["rain"].tail) == ("mm/day", "upper")
    assert (components["heat"].unit, components["heat"].tail) == ("degC", "lower")


@pytest.mark.parametrize("name", NAMES)
def test_every_model_holds_each_components_draws_in_another_order(name: str) -> None:
    single = _own_rows(_single(name))
    for model in _models():
        assert _rows(_draws(model), name) == single


@pytest.mark.parametrize("name", NAMES)
def test_under_independent_a_component_is_sample_curves_sample(name: str) -> None:
    view = _draws(Independent()).component(name)
    single = _single(name)
    assert view.content_hash == single.content_hash
    assert np.array_equal(view.index, single.index)
    assert np.array_equal(view.loss, single.loss)
    assert view.index_unit == single.index_unit


def test_the_reordering_models_do_move_rows() -> None:
    for model in (Comonotonic(), EmpiricalCopula(_reference()), GaussianCopula(0.6)):
        sample = _draws(model)
        assert not np.array_equal(
            _values(sample, "heat"), _single("heat").index[:, 0]
        ), model


def test_a_rows_hazard_index_and_own_loss_stay_together() -> None:
    class ShiftsTheIndex:
        name = "shifts_the_index"

        def to_document(self) -> dict[str, Any]:
            return {"kind": self.name}

        def apply(
            self, index: Any, hazard: Any, impact: Any, rng: Any
        ) -> tuple[Any, Any]:
            del rng
            return index + 3.0, impact

    model: Any = ShiftsTheIndex()
    sample = _draws(Comonotonic(), basis_risk=model)
    components = sample.index_components
    assert components is not None
    for name in NAMES:
        component = components[name]
        assert component.hazard is not None
        np.testing.assert_allclose(component.values - component.hazard, 3.0)
        loss_model = _loss(name)
        assert component.own_loss is not None
        np.testing.assert_allclose(
            component.own_loss[:, 0],
            loss_model.loss_values(loss_model.impact_values(component.hazard[:, 0])),
        )


def test_the_first_component_is_the_primary() -> None:
    sample = _draws(Comonotonic())
    assert np.array_equal(sample.index[:, 0], _values(sample, "rain"))
    assert sample.hazard is None


def test_the_models_use_no_random_numbers_beyond_sample_curve() -> None:
    for model in _models():
        rngs = _rngs()
        _draws(model, rngs=rngs)
        for name in NAMES:
            own = np.random.default_rng(SEEDS[name])
            sample_curve(
                _curve(name),
                metadata=_metadata(name),
                n_draws=300,
                rng=own,
                loss_model=_loss(name),
                basis_risk=NOISE,
                asset_id=ASSET,
            )
            assert rngs[name].bit_generator.state == own.bit_generator.state


def test_a_sample_can_be_rebuilt_from_its_records() -> None:
    reference = _reference()
    sample = _draws(EmpiricalCopula(reference))
    rngs = {name: sample.component(name).provenance.generator(0) for name in NAMES}
    again = _draws(EmpiricalCopula(reference), rngs=rngs)
    assert again.content_hash == sample.content_hash


# -- the combined loss ---------------------------------------------------------


def _impact(sample: OutcomeSample, name: str) -> NDArray[np.float64]:
    assert sample.index_components is not None
    own = sample.index_components[name].own_loss
    assert own is not None
    impact: NDArray[np.float64] = own[:, 0] / 1e6
    return impact


@pytest.mark.parametrize("model", [Comonotonic(), GaussianCopula(-0.5)])
def test_the_combined_loss_is_made_after_the_reorder(model: Any) -> None:
    capped = _draws(model, combine=CappedSum(0.5))
    np.testing.assert_allclose(
        capped.loss[:, 0],
        np.minimum(_impact(capped, "rain") + _impact(capped, "heat"), 0.5) * 1e6,
    )
    compounded = _draws(model, combine=CompoundDamage())
    kept = (1.0 - _impact(compounded, "rain")) * (1.0 - _impact(compounded, "heat"))
    np.testing.assert_allclose(compounded.loss[:, 0], (1.0 - kept) * 1e6)


def test_the_rule_is_recorded() -> None:
    combine = _draws(Independent(), combine=CappedSum(1.0)).provenance.combine
    assert combine is not None
    assert combine["kind"] == "capped_sum"
    assert "capped at 1.0" in combine["statement"]


def test_two_impacts_need_a_rule() -> None:
    rngs = _rngs()
    before = rngs["rain"].bit_generator.state
    with pytest.raises(SampleError, match="combined loss must be stated"):
        _draws(Independent(), combine=None, rngs=rngs)
    assert rngs["rain"].bit_generator.state == before


def test_a_component_with_no_impact_needs_no_rule_and_draws_nothing_more() -> None:
    rngs = _rngs()
    sample = _draws(
        Comonotonic(), loss_models={"rain": _loss("rain")}, combine=None, rngs=rngs
    )
    components = sample.index_components
    assert components is not None
    assert components["heat"].own_loss is None
    assert components["heat"].hazard is None
    assert components["rain"].own_loss is not None
    # The loss is rain's own: every row of it, however the rows were moved.
    assert np.array_equal(sample.loss, components["rain"].own_loss)
    assert sample.provenance.combine is not None
    assert sample.provenance.combine["kind"] == "one_impact"
    # Heat's generator made only its uniforms.
    own = np.random.default_rng(SEEDS["heat"])
    own.random(300)
    assert rngs["heat"].bit_generator.state == own.bit_generator.state
    steps = [
        dict(entry) for entry in sample.provenance.rng if entry["component"] == "heat"
    ]
    assert [step["step"] for step in steps] == ["hazard"]
    with pytest.raises(SampleError, match="no loss of its own"):
        sample.component("heat")
    combined = sample.component("heat", loss="combined")
    assert np.array_equal(combined.loss, sample.loss)


def test_no_impact_at_all_is_refused() -> None:
    with pytest.raises(SampleError, match="no component has an impact"):
        _draws(Independent(), loss_models={}, combine=None)


def test_one_impact_refuses_a_rule() -> None:
    with pytest.raises(SampleError, match="combines two impacts"):
        _draws(Independent(), loss_models={"rain": _loss("rain")})


def test_the_loss_models_need_one_value() -> None:
    other = LossModel(RAMP, value=2.0, value_unit="EUR")
    with pytest.raises(SampleError, match="one value"):
        _draws(Independent(), loss_models={"rain": _loss("rain"), "heat": other})
    dollars = LossModel(RAMP, value=1e6, value_unit="USD")
    with pytest.raises(SampleError, match="value_unit"):
        _draws(Independent(), loss_models={"rain": _loss("rain"), "heat": dollars})


# -- the empirical copula on components ----------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_a_copula_with_one_draw_per_year_has_the_references_ranks(name: str) -> None:
    reference = _reference()
    sample = _draws(EmpiricalCopula(reference), n=30)
    assert np.array_equal(
        _ranks(_values(sample, name)), _ranks(_values(reference, name))
    )


@pytest.mark.parametrize("k", [1, 3])
def test_a_copula_keeps_the_references_ranks_at_block_level(k: int) -> None:
    reference = _reference()
    m = len(reference)
    n = k * m
    uniforms = _uniforms(n=n)
    order = EmpiricalCopula(reference).reorder(uniforms, NAMES)
    for j, name in enumerate(NAMES):
        assert sorted(order[:, j].tolist()) == list(range(n))
        rank_of_draw = np.empty(n, dtype=np.int64)
        rank_of_draw[np.argsort(uniforms[:, j], kind="stable")] = np.arange(n)
        blocks = rank_of_draw[order[:, j]] // k
        assert np.array_equal(blocks, np.repeat(_ranks(_values(reference, name)), k))


@pytest.mark.parametrize("k", [1, 3])
def test_a_copula_sample_reproduces_the_references_ranks(k: int) -> None:
    reference = _reference()
    m = len(reference)
    sample = _draws(EmpiricalCopula(reference), n=k * m)
    for name in NAMES:
        by_rank = _values(sample, name).reshape(m, k)[
            np.argsort(_ranks(_values(reference, name)), kind="stable")
        ]
        # Every draw of a year lies below every draw of the next rank's year.
        assert (by_rank[:-1].max(axis=1) <= by_rank[1:].min(axis=1)).all()


def test_a_copula_ranks_a_components_hazard_when_it_has_one() -> None:
    m = 30
    index = np.random.default_rng(1).gamma(2.0, 20.0, m)
    hazard = np.random.default_rng(2).gamma(2.0, 20.0, m)
    table = pa.table(
        {
            "year": list(range(1990, 1990 + m)),
            "site": [ASSET] * m,
            "rain_index": index.tolist(),
            "rain_hazard": hazard.tolist(),
            "heat_index": index[::-1].tolist(),
            "loss": hazard.tolist(),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        reference = OutcomeSample.from_table(
            table,
            period="year",
            asset_id="site",
            index={"rain": "rain_index", "heat": "heat_index"},
            loss="loss",
            hazard={"rain": "rain_hazard"},
            semantics="annual",
            index_unit={"rain": "mm/day", "heat": "degC"},
            loss_unit="EUR",
            tails={"rain": "upper", "heat": "upper"},
            source="a table of mine",
        )
    sample = _draws(EmpiricalCopula(reference), n=30)
    assert np.array_equal(_ranks(_values(sample, "rain")), _ranks(hazard))
    assert np.array_equal(_ranks(_values(sample, "heat")), _ranks(index[::-1]))


def test_a_copula_records_its_reference() -> None:
    reference = _reference(40)
    sample = _draws(EmpiricalCopula(reference), n=200)
    document = sample.provenance.to_document()["hazard_dependence"]
    assert document["kind"] == "empirical_copula"
    assert document["content_hash"] == reference.content_hash
    assert document["first_year"] == 1990
    assert document["last_year"] == 2029
    assert document["m"] == 40
    assert document["k"] == 5
    roles = {entry["role"]: entry for entry in sample.provenance.licences}
    assert "hazard" in roles
    assert "dependence" in roles


def test_a_copula_refuses_another_components_reference() -> None:
    rngs = _rngs()
    before = rngs["rain"].bit_generator.state
    with pytest.raises(SampleError, match="wind"):
        _draws(EmpiricalCopula(_reference(names=("rain", "wind"))), rngs=rngs)
    assert rngs["rain"].bit_generator.state == before


def test_a_copula_needs_a_whole_multiple_of_its_years() -> None:
    with pytest.raises(SampleError, match="30 and 60"):
        _draws(EmpiricalCopula(_reference()), n=31)


def test_a_copula_across_hazards_refuses_a_reference_of_assets() -> None:
    frame = _reference()
    assets = OutcomeSample(
        asset_ids=NAMES,
        period=frame.period,
        probability=frame.probability,
        index=np.column_stack([_values(frame, name) for name in NAMES]),
        loss=np.zeros((30, 2)),
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="annual",
            sources=tuple(frame.provenance.sources),
            semantics_source="data",
            dependence={"kind": "data"},
        ),
    )
    with pytest.raises(SampleError, match="reference holds several assets"):
        _draws(EmpiricalCopula(assets))


def test_a_copula_across_assets_refuses_a_reference_of_hazards() -> None:
    with pytest.raises(SampleError, match="reference holds several hazards"):
        _assets_sample(EmpiricalCopula(_reference()))


def _assets_sample(model: Any, n: int = 300, **overrides: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "metadata": {name: _metadata("rain") for name in NAMES},
        "rngs": _rngs(),
        "loss_models": {name: _loss(name) for name in NAMES},
        "basis_risk": NOISE,
    }
    arguments.update(overrides)
    sample: OutcomeSample = UNTYPED_ASSETS(
        {name: _curve(name) for name in NAMES},
        dependence=model,
        n_draws=n,
        **arguments,
    )
    return sample


# -- the Gaussian copula -------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 300, 5_000])
def test_rho_zero_is_independent_row_for_row(n: int) -> None:
    uniforms = _uniforms(n=n)
    assert np.array_equal(
        GaussianCopula(0.0).reorder(uniforms, NAMES),
        Independent().reorder(uniforms, NAMES),
    )


@pytest.mark.parametrize("n", [1, 2, 300, 5_000])
def test_rho_one_is_comonotonic_row_for_row(n: int) -> None:
    uniforms = _uniforms(n=n)
    assert np.array_equal(
        GaussianCopula(1.0).reorder(uniforms, NAMES),
        Comonotonic().reorder(uniforms, NAMES),
    )


def test_the_identity_matrix_is_independent_row_for_row() -> None:
    names = ("rain", "heat", "wind")
    uniforms = _uniforms(names)
    assert np.array_equal(
        GaussianCopula(np.eye(3)).reorder(uniforms, names),
        Independent().reorder(uniforms, names),
    )


def _uniforms_that_round_alike() -> NDArray[np.float64]:
    """Adjacent doubles, which the normal quantile can tie or invert, and repeats."""
    rng = np.random.default_rng(3)
    columns = []
    for _ in range(2):
        start = 0.3 + 0.4 * rng.random()
        values = [start]
        for _ in range(399):
            values.append(float(np.nextafter(values[-1], 2.0)))
        values += [0.0, 0.0, 1.0 - 2.0**-53]
        columns.append(rng.permutation(np.array(values, dtype=np.float64)))
    return np.column_stack(columns)


def test_the_reductions_are_exact_where_the_quantile_rounds_alike() -> None:
    uniforms = _uniforms_that_round_alike()
    assert np.array_equal(
        GaussianCopula(0.0).reorder(uniforms, NAMES),
        Independent().reorder(uniforms, NAMES),
    )
    assert np.array_equal(
        GaussianCopula(1.0).reorder(uniforms, NAMES),
        Comonotonic().reorder(uniforms, NAMES),
    )


@pytest.mark.parametrize(("rho", "model"), [(0.0, Independent()), (1.0, Comonotonic())])
def test_the_reductions_hold_on_the_sample(rho: float, model: Any) -> None:
    assert _draws(GaussianCopula(rho)).content_hash == _draws(model).content_hash


@pytest.mark.parametrize("rho", [-0.7, -0.3, 0.3, 0.6, 0.9])
def test_the_spearman_correlation_is_that_of_the_copula(rho: float) -> None:
    sample = _draws(GaussianCopula(rho), n=100_000)
    expected = 6.0 / math.pi * math.asin(rho / 2.0)
    measured = _spearman(_values(sample, "rain"), _values(sample, "heat"))
    assert abs(measured - expected) < 0.01


def test_three_columns_take_the_matrix_in_column_order() -> None:
    names = ("rain", "heat", "wind")
    matrix = [[1.0, 0.5, 0.2], [0.5, 1.0, -0.4], [0.2, -0.4, 1.0]]
    uniforms = _uniforms(names, n=100_000)
    order = GaussianCopula(matrix).reorder(uniforms, names)
    moved = np.column_stack(
        [uniforms[order[:, j], j] for j in range(uniforms.shape[1])]
    )
    for (a, b), rho in {(0, 1): 0.5, (0, 2): 0.2, (1, 2): -0.4}.items():
        expected = 6.0 / math.pi * math.asin(rho / 2.0)
        assert abs(_spearman(moved[:, a], moved[:, b]) - expected) < 0.01


def test_the_first_column_stays_in_draw_order() -> None:
    uniforms = _uniforms()
    assert np.array_equal(
        GaussianCopula(0.6).reorder(uniforms, NAMES)[:, 0], np.arange(300)
    )


def test_minus_one_reverses_the_ranks() -> None:
    uniforms = _uniforms()
    order = GaussianCopula(-1.0).reorder(uniforms, NAMES)
    moved = np.column_stack([uniforms[order[:, j], j] for j in range(2)])
    assert np.array_equal(_ranks(moved[:, 1]), 299 - _ranks(moved[:, 0]))


def test_a_uniform_of_exactly_zero_is_read_inside_the_interval() -> None:
    uniforms = np.array([[0.0, 0.2], [0.5, 0.0], [0.9, 0.7]])
    order = GaussianCopula(0.5).reorder(uniforms, NAMES)
    assert sorted(order[:, 1].tolist()) == [0, 1, 2]


def test_sample_curves_takes_the_gaussian_copula() -> None:
    sample = _assets_sample(GaussianCopula(0.6), n=2_000)
    assert sample.provenance.dependence is not None
    assert dict(sample.provenance.dependence) == {
        "kind": "gaussian_copula",
        "correlation": 0.6,
    }
    assert sample.provenance.notes[0].startswith(
        "The dependence between the assets is an assumption"
    )
    assert _spearman(sample.index[:, 0], sample.index[:, 1]) > 0.4


def test_the_model_is_frozen_and_documented() -> None:
    assert GaussianCopula(0.5) == GaussianCopula(0.5)
    assert GaussianCopula(0.5) != GaussianCopula(0.6)
    matrix = [[1.0, 0.5], [0.5, 1.0]]
    assert GaussianCopula(np.array(matrix)) == GaussianCopula(matrix)
    assert hash(GaussianCopula(matrix)) == hash(GaussianCopula(np.array(matrix)))
    assert GaussianCopula(0.5).to_document() == {
        "kind": "gaussian_copula",
        "correlation": 0.5,
    }
    document = GaussianCopula(matrix).to_document()
    assert document == {"kind": "gaussian_copula", "correlation": matrix}
    json.dumps(document)
    assert isinstance(GaussianCopula(np.float64(0.5)).correlation, float)
    assert isinstance(GaussianCopula(0.5), DependenceModel)


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        -1.01,
        float("nan"),
        True,
        "0.5",
        [0.1, 0.2],
        [[1.0, 0.5, 0.1], [0.5, 1.0, 0.1]],
        [[1.0]],
        [[1.0, 0.5], [0.4, 1.0]],
        [[2.0, 0.5], [0.5, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, float("nan")], [float("nan"), 1.0]],
        [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]],
    ],
)
def test_the_correlation_must_be_a_number_in_range_or_a_correlation_matrix(
    value: Any,
) -> None:
    with pytest.raises(SampleError, match="correlation"):
        UNTYPED_GAUSSIAN(value)


def test_a_matrix_within_rounding_of_symmetric_is_taken() -> None:
    GaussianCopula([[1.0, 0.5 + 1e-14], [0.5, 1.0]])


def test_a_number_states_two_columns_only() -> None:
    names = ("rain", "heat", "wind")
    options = _options(names)
    rngs = options["rngs"]
    before = rngs["rain"].bit_generator.state
    with pytest.raises(SampleError, match="two columns"):
        UNTYPED_HAZARDS(
            options.pop("curves"), dependence=GaussianCopula(0.5), n_draws=10, **options
        )
    assert rngs["rain"].bit_generator.state == before


def test_a_matrix_must_fit_the_columns() -> None:
    names = ("rain", "heat", "wind")
    options = _options(names)
    with pytest.raises(SampleError, match="has 2 columns"):
        UNTYPED_HAZARDS(
            options.pop("curves"),
            dependence=GaussianCopula([[1.0, 0.5], [0.5, 1.0]]),
            n_draws=10,
            **options,
        )


# -- a user's model goes through the same gate ---------------------------------


class Reversed:
    """A user's model: the second column's draws, in reverse draw order."""

    kind = "reversed_second"

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.kind, "version": 1}

    def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
        return {**self.to_document(), "draws": n_draws}

    def statement(self, n_draws: int, between: str) -> str:
        del n_draws
        return f"the second of the {between} is drawn in reverse."

    def reorder(self, uniforms: Any, columns: Sequence[str]) -> Any:
        n, d = uniforms.shape
        order = np.tile(np.arange(n)[:, np.newaxis], (1, d))
        order[:, 1] = order[::-1, 1]
        return order


def test_a_users_model_is_a_dependence_model() -> None:
    assert isinstance(Reversed(), DependenceModel)


def test_a_users_model_reorders_and_is_recorded() -> None:
    sample = _draws(Reversed())
    assert np.array_equal(_values(sample, "rain"), _single("rain").index[:, 0])
    assert np.array_equal(_values(sample, "heat"), _single("heat").index[::-1, 0])
    assert sample.provenance.hazard_dependence is not None
    assert dict(sample.provenance.hazard_dependence) == {
        "kind": "reversed_second",
        "version": 1,
        "draws": 300,
    }


def test_the_module_writes_the_opening_of_a_users_sentence() -> None:
    assert _draws(Reversed()).provenance.notes[0] == (
        "The dependence between the hazards is an assumption: the second of the "
        "hazards is drawn in reverse."
    )


def test_the_module_does_not_repeat_an_opening_a_user_wrote() -> None:
    class Opens(Reversed):
        def statement(self, n_draws: int, between: str) -> str:
            del n_draws
            return f"The dependence between the {between} is an assumption: mine."

    assert _draws(Opens()).provenance.notes[0] == (
        "The dependence between the hazards is an assumption: mine."
    )


def test_sample_curves_takes_a_users_model() -> None:
    sample = _assets_sample(Reversed())
    assert sample.provenance.dependence is not None
    assert sample.provenance.dependence["kind"] == "reversed_second"
    assert sample.provenance.notes[0].startswith(
        "The dependence between the assets is an assumption: the second of the assets"
    )


@pytest.mark.parametrize(
    "kind",
    ["independent", "comonotonic", "empirical_copula", "gaussian_copula", "data"],
)
def test_a_users_model_cannot_take_a_built_ins_kind(kind: str) -> None:
    class Impostor(Reversed):
        def to_document(self) -> dict[str, Any]:
            return {"kind": kind}

    with pytest.raises(SampleError, match="cannot take the kind of a built-in"):
        _draws(Impostor())


def test_a_subclass_of_a_built_in_is_a_users_model() -> None:
    class Sub(Comonotonic):
        pass

    with pytest.raises(SampleError, match="cannot take the kind of a built-in"):
        _draws(Sub())


@pytest.mark.parametrize(
    "document",
    [
        "a text",
        {},
        {"kind": ""},
        {"kind": 3},
        {"kind": "mine", "bad": object()},
        {"kind": "mine", "bad": float("nan")},
    ],
)
def test_a_document_must_be_plain_json_with_a_kind(document: Any) -> None:
    class Odd(Reversed):
        def to_document(self) -> dict[str, Any]:
            return document  # type: ignore[no-any-return]

        def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
            return {"kind": "mine"}

    rngs = _rngs()
    before = rngs["rain"].bit_generator.state
    with pytest.raises(SampleError, match=r"to_document\(\)"):
        _draws(Odd(), rngs=rngs)
    assert rngs["rain"].bit_generator.state == before


def test_a_record_must_keep_the_kind() -> None:
    class Other(Reversed):
        def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
            return {"kind": "something_else"}

    with pytest.raises(SampleError, match="must name the kind"):
        _draws(Other())


def test_a_record_cannot_take_a_built_ins_kind() -> None:
    class Hidden(Reversed):
        def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
            return {"kind": "independent"}

    with pytest.raises(SampleError, match="cannot take the kind"):
        _draws(Hidden())


@pytest.mark.parametrize("statement", ["", "   ", None, 3])
def test_a_model_must_state_itself(statement: Any) -> None:
    class Silent(Reversed):
        def statement(self, n_draws: int, between: str) -> str:
            return statement  # type: ignore[no-any-return]

    with pytest.raises(SampleError, match="must state itself"):
        _draws(Silent())


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (np.zeros((300, 2)), "integers"),
        (np.zeros((300, 2), dtype=bool), "integers"),
        (np.zeros((299, 2), dtype=np.int64), "one draw per row"),
        (np.zeros((300, 3), dtype=np.int64), "one draw per row"),
        (np.zeros((300, 2), dtype=np.int64), "permutation"),
    ],
)
def test_a_reorder_must_hold_each_draw_once(result: Any, message: str) -> None:
    class Bad(Reversed):
        def reorder(self, uniforms: Any, columns: Sequence[str]) -> Any:
            return result

    with pytest.raises(SampleError, match=message):
        _draws(Bad())


def test_a_reorder_cannot_drop_a_draw() -> None:
    class Drops(Reversed):
        def reorder(self, uniforms: Any, columns: Sequence[str]) -> Any:
            order = super().reorder(uniforms, columns)
            order[0, 1] = order[1, 1]
            return order

    with pytest.raises(SampleError, match="permutation"):
        _draws(Drops())


def test_two_calls_on_the_same_uniforms_must_agree() -> None:
    class Stateful(Reversed):
        def __init__(self) -> None:
            self.generator = np.random.default_rng(0)

        def reorder(self, uniforms: Any, columns: Sequence[str]) -> Any:
            n, d = uniforms.shape
            return np.column_stack([self.generator.permutation(n) for _ in range(d)])

    with pytest.raises(SampleError, match="different rows"):
        _draws(Stateful())


def test_a_model_receives_read_only_uniforms() -> None:
    class Writes(Reversed):
        def reorder(self, uniforms: Any, columns: Sequence[str]) -> Any:
            uniforms[0, 0] = 0.5
            return super().reorder(uniforms, columns)

    with pytest.raises(ValueError, match="read-only"):
        _draws(Writes())


@pytest.mark.parametrize("model", [Independent(), Comonotonic(), GaussianCopula(0.2)])
def test_the_built_ins_pass_the_same_gate(model: Any) -> None:
    assert isinstance(model, DependenceModel)
    sample = _draws(model)
    assert sample.provenance.notes[0].startswith(
        "The dependence between the hazards is an assumption:"
    )


# -- the record and the setup text ---------------------------------------------


@pytest.mark.parametrize(
    ("model", "kind"),
    [
        (Independent(), "independent"),
        (Comonotonic(), "comonotonic"),
    ],
)
def test_the_provenance_states_the_model(model: Any, kind: str) -> None:
    sample = _draws(model)
    assert sample.provenance.hazard_dependence is not None
    assert dict(sample.provenance.hazard_dependence) == {"kind": kind}
    assert sample.provenance.dependence is None


def _models_and_words() -> list[tuple[Any, str, str]]:
    return [
        (Independent(), "they are independent", "Independent(), a stated assumption"),
        (Comonotonic(), "they are comonotonic", "Comonotonic(), a stated assumption"),
        (
            EmpiricalCopula(_reference()),
            "ranks of the stated reference",
            "EmpiricalCopula, a stated assumption: the ranks of 30 reference years",
        ),
        (
            GaussianCopula(0.6),
            "a Gaussian copula with correlation 0.6",
            "GaussianCopula, a stated assumption: correlation 0.6",
        ),
        (Reversed(), "is drawn in reverse", "reversed_second, a stated assumption"),
    ]


@pytest.mark.parametrize("position", range(5))
def test_the_setup_text_calls_the_dependence_an_assumption(position: int) -> None:
    model, sentence, on_the_sample = _models_and_words()[position]
    text = _text(_draws(model))
    assert f"Dependence between the hazards: {on_the_sample}" in _line(text, "Sample:")
    notes = _line(text, "Sample notes")
    assert "The dependence between the hazards is an assumption:" in notes
    assert sentence in notes


def test_the_setup_text_of_a_matrix_says_the_order() -> None:
    names = ("rain", "heat", "wind")
    matrix = [[1.0, 0.5, 0.2], [0.5, 1.0, -0.4], [0.2, -0.4, 1.0]]
    options = _options(names)
    options["combine"] = CompoundDamage()
    sample: OutcomeSample = UNTYPED_HAZARDS(
        options.pop("curves"), dependence=GaussianCopula(matrix), n_draws=50, **options
    )
    assert "in column order" in _line(_text(sample), "Sample:")
    assert "in column order" in _line(_text(sample), "Sample notes")


def test_the_gaussian_statement_gives_the_rows_rank_correlation() -> None:
    statement = GaussianCopula(0.6).statement(100, "hazards")
    assert statement.startswith("The dependence between the hazards is an assumption:")
    assert "Spearman rank correlation of about 0.582" in statement


def test_the_copula_note_states_the_limit_of_the_reference() -> None:
    note = _draws(EmpiricalCopula(_reference()), n=90).provenance.notes[0]
    assert "30 years from 1990 to 2019" in note
    assert "3 draws per year" in note
    assert "dependence finer than one year in 30 is assumed, not observed" in note


def test_mixed_tails_are_stated_as_values_not_severities() -> None:
    metadata = {
        "rain": _metadata("rain"),
        "heat": _metadata("heat", return_period_tail="lower"),
    }
    text = _text(_draws(Comonotonic(), metadata=metadata))
    assert "With mixed tails, that dependence is between values, not severities." in (
        _line(text, "Sample:")
    )


def test_equal_tails_say_nothing_of_severities() -> None:
    assert "severities" not in _text(_draws(Comonotonic()))


def test_the_setup_text_names_each_component_and_source() -> None:
    text = _text(_draws(Comonotonic()))
    sample_line = _line(text, "Sample:")
    assert "rain (in mm/day, upper tail, with a loss of its own)" in sample_line
    assert "Combined loss: The impacts are summed" in sample_line
    source = _line(text, "Source")
    assert "the hazard curve of component rain" in source
    assert "the hazard curve of component heat" in source


def test_no_statement_says_which_model_is_better() -> None:
    for model in [*_models(), Reversed()]:
        statement = model.statement(300, "hazards").lower()
        for word in ("better", "worse", "outperform", "superior", "should"):
            assert word not in statement


# -- the provenance ------------------------------------------------------------


def test_the_provenance_keeps_each_components_own_record() -> None:
    sample = _draws(Comonotonic())
    provenance = sample.provenance
    assert provenance.basis_risk is not None
    assert provenance.basis_risk["kind"] == "impact_noise"
    assert provenance.loss_model is not None
    assert provenance.loss_model["kind"] == "components"
    assert [entry["name"] for entry in provenance.loss_model["components"]] == [
        "rain",
        "heat",
    ]
    assert [source["component"] for source in provenance.sources] == list(NAMES)
    assert sorted(record["component"] for record in provenance.rng) == sorted(NAMES * 2)
    assert provenance.details["draws"] == 300
    assert provenance.details["periods"] == (0, 299)
    assert provenance.components is not None
    assert set(provenance.components) == set(NAMES)
    assert all(not note.startswith("Asset") for note in provenance.notes)
    for name in NAMES:
        own = _single(name).provenance.to_document()
        part = sample.component(name).provenance.to_document()
        for field in ("sources", "rng", "loss_model", "basis_risk", "details"):
            assert part[field] == own[field], field
    json.dumps(provenance.to_document())


def test_a_component_records_that_its_rows_were_moved() -> None:
    sample = _draws(Comonotonic())
    assert sample.provenance.components is not None
    assert list(sample.provenance.components["rain"]["selection"]) == []
    assert [
        dict(step) for step in sample.provenance.components["heat"]["selection"]
    ] == [{"kind": "reorder", "dependence": "comonotonic"}]
    steps = sample.component("heat").provenance.selection
    assert [step["kind"] for step in steps] == ["reorder", "component"]


def test_the_sample_round_trips_through_its_table() -> None:
    for model in (GaussianCopula(0.5), EmpiricalCopula(_reference()), Reversed()):
        sample = _draws(model)
        back = OutcomeSample.from_written_table(sample.to_table())
        assert back.content_hash == sample.content_hash
        assert back.provenance.to_document() == sample.provenance.to_document()


def test_details_reach_every_component() -> None:
    sample = _draws(Independent(), details={"file_sha256": "abc"})
    assert sample.provenance.components is not None
    for name in NAMES:
        assert sample.provenance.components[name]["sources"][0]["details"] == {
            "file_sha256": "abc"
        }


def test_the_scenario_names_the_rows() -> None:
    assert _draws(Independent()).scenario == "baseline"
    assert _draws(Independent(), scenario="ssp245").scenario == "ssp245"
    with pytest.raises(SampleError, match="scenario"):
        _draws(Independent(), scenario="  ")


def test_mixed_semantics_are_stated_side_by_side() -> None:
    metadata = {
        "rain": _metadata("rain"),
        "heat": _metadata("heat", probability_semantics="annual_value_distribution"),
    }
    provenance = _draws(Independent(), metadata=metadata).provenance
    assert (
        provenance.probability_semantics
        == "annual_exceedance, annual_value_distribution"
    )


# -- splits and leakage --------------------------------------------------------


def _holdout(sample: OutcomeSample) -> Any:
    return holdout(sample)


def _kfold(sample: OutcomeSample) -> Any:
    return year_block_kfold(sample, 3)


SPLITS = [(_holdout, "holdout"), (_kfold, "year_block_kfold")]


@pytest.mark.parametrize(("split", "name"), SPLITS)
def test_a_split_of_a_copula_sample_is_refused(split: Any, name: str) -> None:
    sample = _draws(EmpiricalCopula(_reference()))
    with pytest.raises(SampleError, match=rf"{name} cannot split .*FreshDraws"):
        split(sample)
    with pytest.raises(SampleError, match=name):
        split(sample.component("heat"))
    with pytest.raises(SampleError, match=name):
        split(OutcomeSample.from_written_table(sample.to_table()))


def test_a_split_is_refused_on_a_copula_component_whose_rows_stayed_in_place() -> None:
    m = 30
    uniforms = _uniforms(n=m)
    series = {
        "rain": AnnualSeries(
            np.arange(1990, 1990 + m), uniforms[:, 0] * 100.0, "mm/day", {}
        ),
        "heat": AnnualSeries(
            np.arange(1990, 1990 + m), uniforms[::-1, 1] * 100.0, "degC", {}
        ),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        reference = sample_annual_hazards(
            series,
            tails=dict.fromkeys(NAMES, "upper"),
            loss_models={"rain": _loss("rain")},
            basis_risk=NoBasisRisk(),
            asset_id=ASSET,
        )
    sample = _draws(EmpiricalCopula(reference), n=m)
    # The reference's ranks of rain are those of its uniforms: no row moves.
    assert np.array_equal(_values(sample, "rain"), _single("rain", m).index[:, 0])
    with pytest.raises(SampleError, match="holdout cannot split"):
        holdout(sample.component("rain"))


@pytest.mark.parametrize("model", [Independent(), Comonotonic(), GaussianCopula(0.4)])
def test_the_other_models_split_on_every_array(model: Any) -> None:
    sample = _draws(model)
    design, held_out = holdout(sample, train_fraction=0.7)
    assert (len(design), len(held_out)) == (210, 90)
    assert np.array_equal(_values(design, "heat"), _values(sample, "heat")[:210])
    assert np.array_equal(design.loss, sample.loss[:210])
    assert design.lineage == held_out.lineage == sample.lineage
    with pytest.raises(LeakageError):
        _check_out_of_sample(design, sample)
    # One hazard's own rows are out of sample under every model.
    _check_out_of_sample(design.component("heat"), held_out.component("heat"))
    if isinstance(model, Independent):
        _check_out_of_sample(design, held_out)
        return
    # The heat draws were placed by the ranks of every row of both blocks.
    with pytest.raises(LeakageError, match="depend on the other block"):
        _check_out_of_sample(design, held_out)


def test_the_lineage_links_the_same_draws_under_every_model() -> None:
    lineages = {_draws(model).lineage for model in _models()}
    assert len(lineages) == 1
    assert _draws(Independent(), rngs=_rngs(shift=100)).lineage not in lineages


def test_draws_shared_between_models_are_a_leak() -> None:
    with pytest.raises(LeakageError):
        _check_out_of_sample(_draws(GaussianCopula(0.5)), _draws(Independent()))
    _check_out_of_sample(
        _draws(GaussianCopula(0.5)), _draws(GaussianCopula(0.5), rngs=_rngs(shift=100))
    )


def test_a_components_view_keeps_the_samples_lineage() -> None:
    sample = _draws(Comonotonic())
    view = sample.component("heat")
    assert view.lineage == sample.lineage
    with pytest.raises(LeakageError):
        _check_out_of_sample(sample, view)


# -- arguments -----------------------------------------------------------------


def test_one_hazard_is_refused_naming_sample_curve() -> None:
    options = _options(("rain",))
    options.pop("curves")
    with pytest.raises(SampleError, match="use sample_curve"):
        UNTYPED_HAZARDS(
            {"rain": _curve("rain")},
            dependence=Independent(),
            n_draws=10,
            **{**options, "combine": None},
        )


@pytest.mark.parametrize("name", ["a.b", " a", "a+b", "x[0]"])
def test_a_component_name_must_be_valid(name: str) -> None:
    curves = {name: _curve("rain"), "heat": _curve("heat")}
    with pytest.raises(SampleError, match="name"):
        _draws(Independent(), curves=curves)


def test_curves_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="curves"):
        _draws(Independent(), curves=[_curve("rain")])


@pytest.mark.parametrize("key", ["metadata", "rngs"])
def test_a_mapping_must_name_every_component(key: str) -> None:
    options = _options()
    partial = dict(options[key])
    del partial["heat"]
    changed: dict[str, Any] = {key: partial}
    with pytest.raises(SampleError, match=key):
        _draws(Independent(), **changed)


@pytest.mark.parametrize("key", ["metadata", "rngs", "loss_models"])
def test_a_mapping_cannot_name_another_component(key: str) -> None:
    options = _options()
    changed: dict[str, Any] = {key: {**options[key], "wind": options[key]["rain"]}}
    with pytest.raises(SampleError, match=key):
        _draws(Independent(), **changed)


def test_wrong_value_types_are_refused() -> None:
    with pytest.raises(TypeError, match="HazardDatasetMetadata"):
        _draws(Independent(), metadata=dict.fromkeys(NAMES, "x"))
    with pytest.raises(TypeError, match="Generator"):
        _draws(Independent(), rngs=dict.fromkeys(NAMES, "x"))
    with pytest.raises(TypeError, match="LossModel"):
        _draws(Independent(), loss_models=dict.fromkeys(NAMES, "x"))
    with pytest.raises(TypeError, match="combine"):
        _draws(Independent(), combine=min)


def test_a_basis_risk_model_is_required() -> None:
    with pytest.raises(TypeError, match="basis_risk"):
        _draws(Independent(), basis_risk=None)


@pytest.mark.parametrize("n", [0, True, 1.5])
def test_invalid_counts_are_refused(n: Any) -> None:
    with pytest.raises(SampleError, match="n_draws"):
        _draws(Independent(), n=n)


def test_one_shared_generator_is_refused() -> None:
    shared = np.random.default_rng(5)
    before = shared.bit_generator.state
    with pytest.raises(SampleError, match="own generator"):
        _draws(Independent(), rngs=dict.fromkeys(NAMES, shared))
    assert shared.bit_generator.state == before


def test_a_hazard_unit_the_impact_is_not_calibrated_for_is_refused() -> None:
    calibrated = ImpactExpression.linear(
        0.01,
        calibration=ImpactCalibration(
            exposure_variable="rx1day",
            exposure_unit="mm/h",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
        clamp={"minimum": 0.0, "maximum": 1.0},
    )
    models = {
        "rain": LossModel(calibrated, value=1e6, value_unit="EUR"),
        "heat": _loss("heat"),
    }
    rngs = _rngs()
    before = rngs["heat"].bit_generator.state
    with pytest.raises(SampleError, match="unit"):
        _draws(Independent(), loss_models=models, rngs=rngs)
    assert rngs["heat"].bit_generator.state == before


# -- warnings ------------------------------------------------------------------


def test_the_comonotone_warning_reaches_the_caller() -> None:
    with pytest.warns(ComonotoneIndexWarning):
        _draws(Independent(), basis_risk=NoBasisRisk())


def test_draws_beyond_the_support_warn_naming_the_component() -> None:
    metadata = {
        name: _metadata(name, return_period_support=(2.0, 10.0)) for name in NAMES
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample = _draws(Independent(), metadata=metadata)
    rarer = [
        warning
        for warning in caught
        if issubclass(warning.category, ReturnPeriodExtrapolationWarning)
    ]
    assert [str(warning.message).split(":")[0] for warning in rarer] == [
        "component rain",
        "component heat",
    ]
    assert all(warning.filename == __file__ for warning in rarer)
    notes = sample.provenance.notes
    assert any(note.startswith("Component heat: ") for note in notes)
