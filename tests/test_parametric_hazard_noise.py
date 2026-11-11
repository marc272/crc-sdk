"""`HazardNoise` and the samplers' step before the impact, offline.

The numbers are synthetic. The tests check what the module computes and records;
nothing here is a finding about a hazard, a basis-risk model or a cover form, and
none says that one model suits an asset better than another.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact
from numpy.typing import NDArray

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    CappedSum,
    ComonotoneIndexWarning,
    Comonotonic,
    Cover,
    EvaluationError,
    EvaluationMetadata,
    Free,
    HazardNoise,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LeakageError,
    LinearTrend,
    LossModel,
    MetricOptions,
    ModelRun,
    NoBasisRisk,
    OutcomeSample,
    SampleError,
    StepPayout,
    design,
    evaluate,
    holdout,
    sample_annual,
    sample_annual_hazards,
    sample_annual_pool,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    year_block_kfold,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import ReturnPeriodExtrapolationWarning

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
    ),
]

FloatArray = NDArray[np.float64]

GUMBEL = FittedDistribution.from_parameters("gumbel_r", location=100.0, scale=30.0)
RAMP = PiecewiseLinearImpact([80.0, 200.0], [0.0, 1.0])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
SIGMA = 0.25
MODEL = HazardNoise(SIGMA)
N = 400
SEED = 7
YEARS = np.arange(1981, 2021)
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
STEP = Cover(
    INDEX, StepPayout([90.0, 120.0], [2e5, 5e5]), payout_unit="EUR", label="step"
)
ASSETS = ("a", "b")
NAMES = ("rain", "wind")
UNITS = {"rain": "mm/day", "wind": "m/s"}


def _metadata(unit: str = "mm/day") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit=unit,
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


def _values(seed: int = 1, n: int = YEARS.size) -> FloatArray:
    return np.random.default_rng(seed).gumbel(100.0, 30.0, n)


def _series(
    values: Any = None, *, unit: str = "mm/day", name: str = "s"
) -> AnnualSeries:
    return AnnualSeries(
        YEARS,
        _values() if values is None else np.asarray(values, dtype=np.float64),
        unit,
        {"source": {"name": f"synthetic {name}"}},
    )


# -- the six samplers, one argument apart --------------------------------------


def _curve(model: Any, seed: int = SEED, n: int = N) -> OutcomeSample:
    return sample_curve(
        GUMBEL,
        metadata=_metadata(),
        n_draws=n,
        rng=np.random.default_rng(seed),
        loss_model=LOSS,
        basis_risk=model,
        asset_id="a",
    )


def _annual(model: Any, seed: int = SEED, **options: Any) -> OutcomeSample:
    arguments = {"rng": np.random.default_rng(seed), **options}
    return sample_annual(
        _series(), loss_model=LOSS, basis_risk=model, asset_id="a", **arguments
    )


def _run(model: str, seed: int) -> ModelRun:
    return ModelRun(
        _series(_values(seed), name=model),
        model=model,
        member="r1i1p1f1",
        pathway="ssp245",
    )


def _pool(model: Any, seed: int = SEED) -> OutcomeSample:
    return sample_annual_pool(
        [_run("M1", 1), _run("M2", 2)],
        loss_model=LOSS,
        basis_risk=model,
        asset_id="a",
        rng=np.random.default_rng(seed),
    )


def _generators(seed: int, names: tuple[str, ...]) -> dict[str, np.random.Generator]:
    return {name: np.random.default_rng(seed + i) for i, name in enumerate(names)}


def _assets(
    model: Any, seed: int = SEED, n: int = N, *, dependence: Any = None
) -> OutcomeSample:
    return sample_curves(
        dict.fromkeys(ASSETS, GUMBEL),
        dependence=Independent() if dependence is None else dependence,
        metadata=dict.fromkeys(ASSETS, _metadata()),
        n_draws=n,
        rngs=_generators(seed, ASSETS),
        loss_models=dict.fromkeys(ASSETS, LOSS),
        basis_risk=model,
    )


def _annual_hazards(
    model: Any, seed: int = SEED, *, impacted: tuple[str, ...] = NAMES
) -> OutcomeSample:
    return sample_annual_hazards(
        {
            "rain": _series(_values(1), name="rain"),
            "wind": _series(_values(2), unit="m/s", name="wind"),
        },
        tails=dict.fromkeys(NAMES, "upper"),
        loss_models=dict.fromkeys(impacted, LOSS),
        combine=CappedSum(1.0) if len(impacted) > 1 else None,
        basis_risk=model,
        asset_id="a",
        rng=np.random.default_rng(seed),
    )


def _curve_hazards(
    model: Any, seed: int = SEED, *, impacted: tuple[str, ...] = NAMES
) -> OutcomeSample:
    return sample_curve_hazards(
        dict.fromkeys(NAMES, GUMBEL),
        dependence=Independent(),
        metadata={name: _metadata(UNITS[name]) for name in NAMES},
        n_draws=N,
        rngs=_generators(seed, NAMES),
        loss_models=dict.fromkeys(impacted, LOSS),
        combine=CappedSum(1.0) if len(impacted) > 1 else None,
        basis_risk=model,
        asset_id="a",
    )


ROUTES = [_curve, _annual, _pool, _assets, _annual_hazards, _curve_hazards]
Route = Callable[..., OutcomeSample]


def _metadata_of(table: Any) -> EvaluationMetadata:
    return EvaluationMetadata.from_table(table)


def _sample_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith("Sample")]
    assert lines
    return lines[0]


def _column(table: Any, name: str) -> float:
    value: float = table.column(name)[0].as_py()
    return value


# -- what the model does to a sample --------------------------------------------


def test_the_cover_pays_on_the_index_and_the_loss_reads_the_hazard_at_the_asset() -> (
    None
):
    sample = _curve(MODEL)
    rng = np.random.default_rng(SEED)
    index = GUMBEL.quantiles(rng.random(N))
    factor = np.exp(rng.normal(-0.5 * SIGMA**2, SIGMA, N))
    np.testing.assert_array_equal(sample.index[:, 0], index)
    assert sample.hazard is not None
    np.testing.assert_array_equal(sample.hazard[:, 0], index * factor)
    np.testing.assert_array_equal(
        sample.loss[:, 0], LOSS.loss_values(LOSS.impact_values(index * factor))
    )


def test_both_failures_occur_for_one_cover() -> None:
    sample = _curve(MODEL, n=5000)
    assert sample.hazard is not None
    index, loss = sample.index[:, 0], sample.loss[:, 0]
    # The cover attaches at 90: a year can pay with no loss, and lose with no pay.
    assert ((index >= 90.0) & (loss == 0.0)).any()
    assert ((index < 90.0) & (loss > 0.0)).any()
    row = evaluate(STEP, sample, split="in_sample")
    assert _column(row, "false_negative_rate_l0") > 0.0
    assert _column(row, "false_positive_rate_l0") > 0.0


def test_the_failure_rates_respond_to_sigma() -> None:
    def rates(model: Any) -> tuple[float, float]:
        row = evaluate(STEP, _curve(model, n=5000), split="in_sample")
        return (
            _column(row, "false_negative_rate_l0"),
            _column(row, "false_positive_rate_l0"),
        )

    plain = rates(NoBasisRisk())
    assert plain[1] == 0.0
    low, high = rates(HazardNoise(0.1)), rates(HazardNoise(0.5))
    assert plain[1] < low[1] < high[1]
    assert low[0] != high[0]


def test_impact_noise_cant_change_the_failure_rates_at_zero() -> None:
    options = MetricOptions(loss_thresholds=(0.0, 3e5))

    def row(model: Any) -> Any:
        return evaluate(STEP, _curve(model, n=5000), split="in_sample", options=options)

    plain, noisy = row(NoBasisRisk()), row(ImpactNoise(0.5))
    for name in ("false_negative_rate_l0", "false_positive_rate_l0"):
        assert _column(noisy, name) == _column(plain, name)
    label = next(
        name
        for name in noisy.column_names
        if name.startswith("false_negative_rate_l") and name != "false_negative_rate_l0"
    )
    # A threshold above zero: the factor can carry a loss across it.
    assert _column(noisy, label) != _column(plain, label)


def test_the_provenance_records_the_model_and_keeps_the_lineage() -> None:
    sample = _curve(MODEL)
    assert sample.provenance.basis_risk == {"kind": "hazard_noise", "sigma": SIGMA}
    assert sample.lineage == _curve(NoBasisRisk()).lineage


def test_the_recorded_generator_rebuilds_the_sample() -> None:
    curve = _curve(MODEL)
    again = sample_curve(
        GUMBEL,
        metadata=_metadata(),
        n_draws=N,
        rng=curve.provenance.generator(),
        loss_model=LOSS,
        basis_risk=MODEL,
        asset_id="a",
    )
    assert again.content_hash == curve.content_hash
    annual = _annual(MODEL)
    rebuilt = _annual(MODEL, rng=annual.provenance.generator())
    assert rebuilt.content_hash == annual.content_hash


def test_a_written_sample_keeps_the_hazard_at_the_asset() -> None:
    sample = _curve(MODEL)
    back = OutcomeSample.from_written_table(sample.to_table())
    assert back.content_hash == sample.content_hash
    assert back.hazard is not None
    assert sample.hazard is not None
    np.testing.assert_array_equal(back.hazard, sample.hazard)


# -- the draw order of section 4.4 ----------------------------------------------


def test_the_hazard_draws_first_and_the_model_after() -> None:
    sample = _curve(MODEL)
    plain = _curve(NoBasisRisk())
    # The same uniforms: the model reads the index the plain route draws.
    np.testing.assert_array_equal(sample.index, plain.index)
    assert sample.provenance.rng[0] == plain.provenance.rng[0]
    assert [record["step"] for record in sample.provenance.rng] == [
        "hazard",
        "basis_risk",
    ]
    rng = np.random.default_rng(SEED)
    rng.random(N)
    resumed = sample.provenance.generator(1)
    assert resumed.bit_generator.state == rng.bit_generator.state


def test_one_generator_record_is_taken_at_entry_to_the_model() -> None:
    sample = _annual(MODEL)
    assert [record["step"] for record in sample.provenance.rng] == ["basis_risk"]
    rng = np.random.default_rng(SEED)
    assert sample.provenance.generator().bit_generator.state == rng.bit_generator.state


def test_the_impact_draws_nothing() -> None:
    seen: list[dict[str, Any]] = []

    class _Watches:
        """Records the generator's state each time the model is called."""

        name = "watches"

        def to_document(self) -> dict[str, Any]:
            return {"kind": self.name}

        def apply_hazard(self, index: Any, hazard: Any, rng: Any) -> Any:
            hazard = hazard * np.exp(rng.normal(0.0, 0.1, size=hazard.shape))
            seen.append(rng.bit_generator.state)
            return index, hazard

        def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
            seen.append(rng.bit_generator.state)
            return index, impact

    _annual(_Watches())
    assert seen[0] == seen[1]


@pytest.mark.parametrize("route", ROUTES)
def test_a_sweep_over_sigma_reads_one_set_of_draws(route: Route) -> None:
    low, high = route(HazardNoise(0.1)), route(HazardNoise(0.5))
    assert low.hazard is not None and high.hazard is not None
    z_low = (np.log(low.hazard / low.index) + 0.5 * 0.1**2) / 0.1
    z_high = (np.log(high.hazard / high.index) + 0.5 * 0.5**2) / 0.5
    np.testing.assert_allclose(z_low, z_high, atol=1e-9)


@pytest.mark.parametrize("sigma", [0.0, 0.5])
def test_every_sigma_leaves_the_generator_in_one_state(sigma: float) -> None:
    def state(value: float) -> Any:
        rng = np.random.default_rng(SEED)
        sample_annual(
            _series(),
            loss_model=LOSS,
            basis_risk=HazardNoise(value),
            asset_id="a",
            rng=rng,
        )
        return rng.bit_generator.state

    assert state(sigma) == state(0.25)


# -- sigma zero ------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_at_sigma_zero_the_rows_are_those_of_no_basis_risk(route: Route) -> None:
    plain, zero = route(NoBasisRisk()), route(HazardNoise(0.0))
    np.testing.assert_array_equal(zero.index, plain.index)
    np.testing.assert_array_equal(zero.loss, plain.loss)
    assert zero.hazard is None
    assert zero.content_hash == plain.content_hash


@pytest.mark.parametrize("route", ROUTES)
def test_at_sigma_zero_it_warns_as_no_basis_risk_does(route: Route) -> None:
    with pytest.warns(ComonotoneIndexWarning):
        route(HazardNoise(0.0))


@pytest.mark.parametrize("route", ROUTES)
def test_above_zero_it_does_not_warn(route: Route) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComonotoneIndexWarning)
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        route(MODEL)


def test_at_sigma_zero_the_ratios_are_one_and_the_notes_say_so() -> None:
    sample = _annual(HazardNoise(0.0))
    summary = sample.provenance.details["basis_risk"]
    assert summary["hazard_mean_ratio"] == 1.0
    assert summary["impact_mean_ratio"] == 1.0
    assert any("comonotone" in note for note in sample.provenance.notes)


# -- models without apply_hazard ------------------------------------------------


class _Holds:
    """A user model that holds the damage function, and no `apply_hazard`."""

    name = "holds_damage"

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.name, "sigma": self.sigma}

    def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
        del impact
        log_factor = rng.normal(-0.5 * self.sigma**2, self.sigma, size=hazard.shape)
        return index, LOSS.impact_values(hazard * np.exp(log_factor))


@pytest.mark.parametrize(
    "model",
    [NoBasisRisk(), ImpactNoise(0.3, upper=1.0), _Holds(0.2)],
    ids=lambda m: m.name,
)
def test_a_model_without_apply_hazard_gives_todays_rows(model: Any) -> None:
    rng = np.random.default_rng(SEED)
    hazard = GUMBEL.quantiles(rng.random(N))
    state = rng.bit_generator.state
    index, impact = model.apply(hazard, hazard, LOSS.impact_values(hazard), rng)
    sample = _curve(model)
    np.testing.assert_array_equal(sample.index[:, 0], index)
    np.testing.assert_array_equal(sample.loss[:, 0], LOSS.loss_values(impact))
    assert sample.hazard is None
    assert sample.provenance.generator(1).bit_generator.state == state
    assert "hazard_mean_ratio" not in sample.provenance.details["basis_risk"]


@pytest.mark.parametrize("route", [_curve, _annual, _pool])
def test_a_stand_in_that_holds_the_damage_function_gives_the_same_rows(
    route: Route,
) -> None:
    built_in, stand_in = route(HazardNoise(0.2)), route(_Holds(0.2))
    np.testing.assert_array_equal(built_in.index, stand_in.index)
    np.testing.assert_array_equal(built_in.loss, stand_in.loss)


def test_a_model_may_have_both_methods_and_apply_hazard_runs_first() -> None:
    calls: list[str] = []
    seen: dict[str, FloatArray] = {}

    class _Both:
        name = "both"

        def to_document(self) -> dict[str, Any]:
            return {"kind": self.name}

        def apply_hazard(self, index: Any, hazard: Any, rng: Any) -> Any:
            calls.append("apply_hazard")
            return index + 1.0, hazard * 2.0

        def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
            calls.append("apply")
            seen.update(index=index, hazard=hazard, impact=impact)
            return index, impact * 0.5

    plain = _annual(NoBasisRisk())
    sample = _annual(_Both())
    assert calls == ["apply_hazard", "apply"]
    hazard = plain.index[:, 0]
    np.testing.assert_array_equal(seen["index"], hazard + 1.0)
    np.testing.assert_array_equal(seen["hazard"], hazard * 2.0)
    np.testing.assert_array_equal(seen["impact"], LOSS.impact_values(hazard * 2.0))
    np.testing.assert_array_equal(sample.index[:, 0], hazard + 1.0)
    assert sample.hazard is not None
    np.testing.assert_array_equal(sample.hazard[:, 0], hazard * 2.0)
    expected = 0.5 * LOSS.impact_values(hazard * 2.0)
    np.testing.assert_array_equal(sample.loss[:, 0], LOSS.loss_values(expected))
    summary = sample.provenance.details["basis_risk"]
    assert summary["index_changed"] is True
    assert summary["hazard_mean_ratio"] == pytest.approx(2.0)
    assert summary["impact_mean_ratio"] == pytest.approx(
        expected.sum() / LOSS.impact_values(hazard).sum()
    )


# -- the gate -------------------------------------------------------------------


def _returning(value: Callable[[FloatArray, FloatArray], Any]) -> Any:
    class _Returns:
        name = "returns"

        def to_document(self) -> dict[str, Any]:
            return {"kind": self.name}

        def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
            return index, impact

        def apply_hazard(self, index: Any, hazard: Any, rng: Any) -> Any:
            return value(index, hazard)

    return _Returns()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (lambda i, h: h, "must return"),
        (lambda i, h: (i, h, h), "must return"),
        (lambda i, h: (i, h[:-1]), "changed the shape"),
        (lambda i, h: (i[:-1], h), "changed the shape"),
        (lambda i, h: (i, np.full_like(h, np.nan)), "non-finite"),
        (lambda i, h: (np.full_like(i, np.inf), h), "non-finite"),
        (lambda i, h: (i, ["a"] * h.size), "non-numbers"),
    ],
)
def test_the_gate_refuses_a_bad_apply_hazard(result: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        _annual(_returning(result))


# -- what it refuses ------------------------------------------------------------


def test_a_negative_hazard_is_refused_and_a_zero_stays_zero() -> None:
    values = _values()
    values[3] = -1.0
    with pytest.raises(SampleError, match="ratio scale"):
        sample_annual(
            _series(values),
            loss_model=LOSS,
            basis_risk=MODEL,
            asset_id="a",
            rng=np.random.default_rng(SEED),
        )
    values[3] = 0.0
    sample = sample_annual(
        _series(values),
        loss_model=LOSS,
        basis_risk=MODEL,
        asset_id="a",
        rng=np.random.default_rng(SEED),
    )
    assert sample.hazard is not None
    assert sample.hazard[3, 0] == 0.0


@pytest.mark.parametrize("sigma", [0.0, SIGMA])
@pytest.mark.parametrize("same_values", [True, False])
def test_a_separate_index_series_is_refused_before_any_draw(
    sigma: float, same_values: bool
) -> None:
    rng = np.random.default_rng(SEED)
    state = rng.bit_generator.state
    other = _series(None if same_values else _values(9), name="elsewhere")
    with pytest.raises(SampleError, match="separate index"):
        sample_annual(
            _series(),
            loss_model=LOSS,
            basis_risk=HazardNoise(sigma),
            asset_id="a",
            index=other,
            rng=rng,
        )
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("sigma", [0.0, SIGMA])
def test_a_missing_generator_is_refused_at_every_sigma(sigma: float) -> None:
    with pytest.raises(SampleError, match="pass rng="):
        sample_annual(
            _series(), loss_model=LOSS, basis_risk=HazardNoise(sigma), asset_id="a"
        )


# -- the mean is on the hazard ---------------------------------------------------


def test_the_two_mean_ratios_are_recorded() -> None:
    sample = _annual(MODEL)
    assert sample.hazard is not None
    summary = sample.provenance.details["basis_risk"]
    before = LOSS.impact_values(sample.index[:, 0])
    after = LOSS.impact_values(sample.hazard[:, 0])
    assert summary == {
        "hazard_mean_ratio": pytest.approx(
            sample.hazard[:, 0].sum() / sample.index[:, 0].sum()
        ),
        "impact_mean_ratio": pytest.approx(after.sum() / before.sum()),
        "index_changed": False,
    }


def test_no_impact_without_the_factor_leaves_the_ratio_undefined() -> None:
    sample = sample_annual(
        _series(np.full(YEARS.size, 10.0)),
        loss_model=LOSS,
        basis_risk=HazardNoise(0.01),
        asset_id="a",
        rng=np.random.default_rng(SEED),
    )
    assert sample.provenance.details["basis_risk"]["impact_mean_ratio"] is None


# -- with trends, pools, assets and components ----------------------------------


def test_under_a_trend_the_factor_multiplies_the_adjusted_hazard() -> None:
    values = 100.0 + 2.0 * (YEARS - YEARS[0]) + _values(4) / 10.0
    plain = sample_annual(
        _series(values),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        trend=LinearTrend(),
    )
    sample = sample_annual(
        _series(values),
        loss_model=LOSS,
        basis_risk=MODEL,
        asset_id="a",
        trend=LinearTrend(),
        rng=np.random.default_rng(SEED),
    )
    np.testing.assert_array_equal(sample.index, plain.index)
    factor = np.exp(
        np.random.default_rng(SEED).normal(-0.5 * SIGMA**2, SIGMA, YEARS.size)
    )
    assert sample.hazard is not None
    np.testing.assert_array_equal(sample.hazard[:, 0], plain.index[:, 0] * factor)


def test_a_pool_draws_one_factor_per_row_in_row_order() -> None:
    plain, sample = _pool(NoBasisRisk()), _pool(MODEL)
    assert sample.provenance.rng[0]["step"] == "basis_risk"
    rows = plain.index.shape[0]
    assert rows == 2 * YEARS.size
    factor = np.exp(np.random.default_rng(SEED).normal(-0.5 * SIGMA**2, SIGMA, rows))
    assert sample.hazard is not None
    np.testing.assert_array_equal(sample.index, plain.index)
    np.testing.assert_array_equal(sample.hazard[:, 0], plain.index[:, 0] * factor)


def test_each_asset_draws_its_factors_from_its_own_generator() -> None:
    sample = _assets(MODEL)
    assert sample.hazard is not None
    hazards = sample.hazard
    for j, asset in enumerate(ASSETS):
        own = _curve(MODEL, seed=SEED + j)
        assert own.hazard is not None
        np.testing.assert_array_equal(hazards[:, j], own.hazard[:, 0])
        assert asset in sample.asset_ids
    assert not np.array_equal(hazards[:, 0], hazards[:, 1])


def test_a_dependence_model_moves_index_hazard_and_loss_together() -> None:
    sample = _assets(MODEL, dependence=Comonotonic())
    independent = _assets(MODEL)
    assert sample.hazard is not None and independent.hazard is not None
    for j in range(2):
        # The rows are a permutation, and each row still holds one draw's values.
        moved = {
            (float(x), float(h), float(y))
            for x, h, y in zip(
                sample.index[:, j], sample.hazard[:, j], sample.loss[:, j], strict=True
            )
        }
        kept = {
            (float(x), float(h), float(y))
            for x, h, y in zip(
                independent.index[:, j],
                independent.hazard[:, j],
                independent.loss[:, j],
                strict=True,
            )
        }
        assert moved == kept
    np.testing.assert_array_equal(
        np.argsort(sample.index[:, 0]), np.argsort(sample.index[:, 1])
    )


def test_an_assets_own_sample_has_its_ratios() -> None:
    sample = _assets(MODEL)
    recorded = sample.provenance.details["basis_risk"]
    assert set(recorded) == set(ASSETS)
    for asset in ASSETS:
        assert sample.asset(asset).provenance.details["basis_risk"] == recorded[asset]


def test_components_draw_in_component_order_per_year() -> None:
    sample = _annual_hazards(MODEL)
    assert sample.index_components is not None
    rng = np.random.default_rng(SEED)
    for name, seed in (("rain", 1), ("wind", 2)):
        factor = np.exp(rng.normal(-0.5 * SIGMA**2, SIGMA, YEARS.size))
        component = sample.index_components[name]
        np.testing.assert_array_equal(component.values[:, 0], _values(seed))
        assert component.hazard is not None
        np.testing.assert_array_equal(component.hazard[:, 0], _values(seed) * factor)
        assert component.own_loss is not None
        # A component's own loss reads its hazard at the asset.
        np.testing.assert_array_equal(
            component.own_loss[:, 0],
            LOSS.loss_values(LOSS.impact_values(component.hazard[:, 0])),
        )


@pytest.mark.parametrize("route", [_annual_hazards, _curve_hazards])
def test_a_component_with_no_impact_still_draws(route: Route) -> None:
    both, only_wind = route(MODEL), route(MODEL, impacted=("wind",))
    assert both.index_components is not None
    assert only_wind.index_components is not None
    first, second = both.index_components, only_wind.index_components
    # The other component's draws don't depend on which of them has an impact.
    assert second["rain"].own_loss is None
    assert second["rain"].hazard is not None
    assert second["wind"].hazard is not None and first["wind"].hazard is not None
    np.testing.assert_array_equal(second["wind"].hazard, first["wind"].hazard)
    np.testing.assert_array_equal(second["rain"].hazard, first["rain"].hazard)
    summary = only_wind.provenance.details["basis_risk"]
    assert summary["rain"]["impact_mean_ratio"] is None
    assert summary["rain"]["hazard_mean_ratio"] == pytest.approx(
        float(second["rain"].hazard.sum() / second["rain"].values.sum())
    )
    assert summary["wind"]["impact_mean_ratio"] is not None
    assert "basis_risk" in {r["step"] for r in only_wind.provenance.rng}


def test_a_component_with_no_impact_draws_nothing_for_other_models() -> None:
    sample = _annual_hazards(ImpactNoise(0.3), impacted=("wind",))
    assert [record["component"] for record in sample.provenance.rng] == ["wind"]
    assert sample.index_components is not None
    assert sample.index_components["rain"].hazard is None


# -- the setup text --------------------------------------------------------------


STATED = (
    "hazard noise, the hazard at the asset being the index times a mean-one "
    "lognormal factor with sigma 0.25, one per year and asset, as an assumption "
    "the caller states"
)


def test_the_setup_text_states_the_model_and_the_two_ratios() -> None:
    sample = _annual(MODEL)
    summary = sample.provenance.details["basis_risk"]
    line = _sample_line(
        setup_text(_metadata_of(evaluate(STEP, sample, split="in_sample")))
    )
    assert STATED in line
    # The expected loss against sigma 0 stands next to sigma, in the same statement.
    assert (
        f"{STATED}; over the rows as built, with the model over without it (sigma 0, "
        f"on the same draws), the expected loss is "
        f"{summary['impact_mean_ratio']:.6g} and the mean hazard is "
        f"{summary['hazard_mean_ratio']:.6g}"
    ) in line


def test_the_setup_text_of_a_plan_not_yet_run_states_the_model_alone() -> None:
    setup = {
        "samples": [
            {
                "basis_risk": {"kind": "hazard_noise", "sigma": 0.25},
                "source_kind": "monte_carlo",
            }
        ]
    }
    line = _sample_line(setup_text(setup))
    assert STATED in line
    assert "mean hazard" not in line


def test_the_setup_text_says_the_factors_are_independent_between_assets() -> None:
    sample = _assets(MODEL)
    designs = {asset: STEP for asset in ASSETS}
    line = _sample_line(
        setup_text(_metadata_of(evaluate(designs, sample, split="in_sample")))
    )
    assert f"{STATED}, the factors independent between assets" in line
    assert "the expected loss and the mean hazard are" in line
    for asset in ASSETS:
        summary = sample.provenance.details["basis_risk"][asset]
        assert (
            f"{summary['impact_mean_ratio']:.6g} and "
            f"{summary['hazard_mean_ratio']:.6g} for {asset}"
        ) in line


def _joined(model: Any) -> OutcomeSample:
    return OutcomeSample.join(
        [
            sample_annual(
                _series(_values(seed), name=asset),
                loss_model=LOSS,
                basis_risk=model,
                asset_id=asset,
                rng=np.random.default_rng(seed),
            )
            for asset, seed in (("a", 1), ("b", 2))
        ]
    )


def test_a_joined_sample_states_each_assets_ratios() -> None:
    joined = _joined(MODEL)
    metadata = _metadata_of(
        evaluate(dict.fromkeys(ASSETS, STEP), joined, split="in_sample")
    )
    line = _sample_line(setup_text(metadata))
    assert f"{STATED}, the factors independent between assets" in line
    assert "the expected loss and the mean hazard are" in line
    assert joined.provenance.parts is not None
    for asset in ASSETS:
        summary = joined.provenance.parts[asset]["details"]["basis_risk"]
        assert (
            f"{summary['impact_mean_ratio']:.6g} and "
            f"{summary['hazard_mean_ratio']:.6g} for {asset}"
        ) in line
    assert "own provenance" not in line


@pytest.mark.parametrize(
    "model",
    [NoBasisRisk(), ImpactNoise(0.3, upper=1.0), _Holds(0.2)],
    ids=lambda m: m.name,
)
def test_a_joined_sample_of_another_model_keeps_its_setup_document(model: Any) -> None:
    joined = _joined(model)
    metadata = _metadata_of(
        evaluate(dict.fromkeys(ASSETS, STEP), joined, split="in_sample")
    )
    (entry,) = metadata.setup.samples
    documents = joined.provenance.to_document()["parts"]
    assert entry["parts"] == {
        asset: {
            "basis_risk": documents[asset]["basis_risk"],
            "loss_model": documents[asset]["loss_model"],
        }
        for asset in ASSETS
    }
    assert "mean hazard" not in _sample_line(setup_text(metadata))


def test_a_joined_sample_above_five_assets_gives_the_ranges() -> None:
    names = [f"a{k}" for k in range(6)]
    setup = {
        "samples": [
            {
                "asset_ids": names,
                "details": {"draws": 40, "periods": [1981, 2020]},
                "parts": {
                    name: {
                        "basis_risk": {"kind": "hazard_noise", "sigma": 0.25},
                        "details": {
                            "basis_risk": {
                                "hazard_mean_ratio": 1.0 + k / 100,
                                "impact_mean_ratio": 1.5 + k / 10,
                            }
                        },
                    }
                    for k, name in enumerate(names)
                },
                "source_kind": "annual",
            }
        ]
    }
    assert (
        "the expected loss is 1.5 to 2 and the mean hazard is 1 to 1.05 across 6 assets"
    ) in _sample_line(setup_text(setup))


def test_the_warning_is_true_of_both_models_and_names_hazard_noise_first() -> None:
    with pytest.warns(ComonotoneIndexWarning) as plain:
        _annual(NoBasisRisk())
    with pytest.warns(ComonotoneIndexWarning) as zero:
        _annual(HazardNoise(0.0))
    assert str(plain[0].message) == str(zero[0].message)
    message = str(zero[0].message)
    assert "without a basis-risk model" not in message
    assert message.index("HazardNoise") < message.index("ImpactNoise")


def test_the_setup_text_gives_the_range_above_five_assets() -> None:
    names = [f"a{k}" for k in range(6)]
    recorded = {
        name: {"hazard_mean_ratio": 1.0 + k / 100, "impact_mean_ratio": 1.5 + k / 10}
        for k, name in enumerate(names)
    }
    setup = {
        "samples": [
            {
                "asset_ids": names,
                "details": {"basis_risk": recorded},
                "parts": {
                    name: {"basis_risk": {"kind": "hazard_noise", "sigma": 0.25}}
                    for name in names
                },
                "source_kind": "monte_carlo",
            }
        ]
    }
    line = _sample_line(setup_text(setup))
    assert (
        "the expected loss is 1.5 to 2 and the mean hazard is 1 to 1.05 across 6 assets"
    ) in line


def test_the_setup_text_names_a_components_ratios_and_its_undefined_impact() -> None:
    sample = _annual_hazards(MODEL, impacted=("wind",))
    line = _sample_line(
        setup_text(
            {
                "samples": [
                    {
                        "basis_risk": sample.provenance.basis_risk,
                        "components": [{"name": name} for name in NAMES],
                        "details": sample.provenance.details,
                        "source_kind": "annual",
                    }
                ]
            }
        )
    )
    assert f"{STATED}, each component with factors of its own" in line
    rain = sample.provenance.details["basis_risk"]["rain"]["hazard_mean_ratio"]
    assert "each component's own expected loss and the mean hazard are" in line
    assert f"not defined and {rain:.6g} for rain" in line


def test_two_drawn_samples_of_one_size_each_state_their_own_ratios() -> None:
    template = Cover(
        INDEX, StepPayout([90.0, 120.0], [Free(), Free()]), payout_unit="EUR"
    )
    drawn, fresh = _curve(MODEL, seed=1), _curve(MODEL, seed=2)
    table = evaluate(design(template, drawn, MSE()), fresh, split="out_of_sample")
    lines = {
        line.split(":")[0]: line
        for line in setup_text(_metadata_of(table)).splitlines()
        if line.startswith("Sample of")
    }
    span = "Sample of 400 simulated years (draws 0 to 399)"
    assert set(lines) == {
        f"{span}, the design sample",
        f"{span}, the evaluation sample",
    }
    for sample, role in ((drawn, "design"), (fresh, "evaluation")):
        ratio = sample.provenance.details["basis_risk"]["impact_mean_ratio"]
        assert (
            f"the expected loss is {ratio:.6g}" in lines[f"{span}, the {role} sample"]
        )


def test_a_block_carries_the_ratios_of_the_sample_it_was_cut_from() -> None:
    sample = _annual(MODEL)
    train, test = holdout(sample)
    assert (
        train.provenance.details["basis_risk"]
        == test.provenance.details["basis_risk"]
        == sample.provenance.details["basis_risk"]
    )


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            {"samples": [{"basis_risk": {"kind": "hazard_noise", "sigma": "a"}}]},
            r"basis_risk\.sigma must be a number",
        ),
        (
            {"samples": [{"details": {"basis_risk": "x"}}]},
            r"details\.basis_risk has the wrong type",
        ),
    ],
)
def test_a_malformed_setup_is_refused(setup: Any, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        setup_text(setup)


# -- holding the hazard's distribution -------------------------------------------


HELD = HazardNoise(SIGMA, hold="hazard_distribution")
ONE_HAZARD = [_curve, _annual, _pool, _assets]
COMPONENTS = [_annual_hazards, _curve_hazards]
SIGMAS = (0.1, 0.25, 0.4, 1.0)
# The loss of one hazard holds to this relative tolerance across sigma; the
# ratios the samplers record are exactly 1.
TOLERANCE = 1e-12


def _hazards(sample: OutcomeSample) -> list[tuple[FloatArray, FloatArray]]:
    """Each (index, hazard at the asset) the sample holds, column by column."""
    if sample.index_components is not None:
        pairs = []
        for component in sample.index_components.values():
            assert component.hazard is not None
            pairs.append((component.values[:, 0], component.hazard[:, 0]))
        return pairs
    assert sample.hazard is not None
    return [
        (sample.index[:, j], sample.hazard[:, j]) for j in range(len(sample.asset_ids))
    ]


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r.__name__)
def test_the_default_and_its_hashes_are_unchanged(route: Route) -> None:
    # The hashes rest on floating-point draws, which differ between platforms,
    # so they aren't pinned: the tests above check the default's rows against
    # its formula, and here the default stated or not gives one sample.
    default = route(MODEL)
    stated = route(HazardNoise(SIGMA, hold="hazard_mean"))
    assert stated.content_hash == default.content_hash
    assert stated.lineage == default.lineage
    assert stated.provenance.to_document() == default.provenance.to_document()
    assert HazardNoise(SIGMA, hold="hazard_mean") == MODEL
    assert MODEL.to_document() == {"kind": "hazard_noise", "sigma": SIGMA}


def test_the_form_is_recorded_in_the_provenance() -> None:
    assert HELD.to_document() == {
        "kind": "hazard_noise",
        "sigma": SIGMA,
        "hold": "hazard_distribution",
    }
    sample = _curve(HELD)
    assert sample.provenance.basis_risk == HELD.to_document()
    back = OutcomeSample.from_written_table(sample.to_table())
    assert back.provenance.basis_risk == HELD.to_document()
    assert back.content_hash == sample.content_hash
    # The rows differ from the default's, the source of the rows does not.
    assert sample.content_hash != _curve(MODEL).content_hash
    assert sample.lineage == _curve(MODEL).lineage


@pytest.mark.parametrize("hold", ["mean", "", None, 1, "HAZARD_MEAN"])
def test_an_unknown_form_is_refused(hold: Any) -> None:
    with pytest.raises(SampleError, match="hold must be one of"):
        HazardNoise(SIGMA, hold=hold)


@pytest.mark.parametrize("route", ONE_HAZARD, ids=lambda r: r.__name__)
@pytest.mark.parametrize("sigma", SIGMAS)
def test_the_held_form_holds_the_expected_loss_across_sigma(
    route: Route, sigma: float
) -> None:
    plain = route(NoBasisRisk())
    held = route(HazardNoise(sigma, hold="hazard_distribution"))
    np.testing.assert_array_equal(held.index, plain.index)
    expected = plain.probability @ plain.loss
    np.testing.assert_allclose(held.probability @ held.loss, expected, rtol=TOLERANCE)
    # The hazard at the asset is a reordering of the index, column by column.
    for index, hazard in _hazards(held):
        np.testing.assert_array_equal(np.sort(hazard), np.sort(index))
        assert not np.array_equal(hazard, index)
    recorded = held.provenance.details["basis_risk"]
    one = len(held.asset_ids) == 1
    for summary in [recorded] if one else [recorded[a] for a in held.asset_ids]:
        assert summary["hazard_mean_ratio"] == 1.0
        assert summary["impact_mean_ratio"] == 1.0


@pytest.mark.parametrize("sigma", SIGMAS)
def test_the_default_form_moves_the_expected_loss_with_sigma(sigma: float) -> None:
    ratio = _curve(HazardNoise(sigma)).provenance.details["basis_risk"][
        "impact_mean_ratio"
    ]
    assert abs(ratio - 1.0) > 1e-3


@pytest.mark.parametrize("route", COMPONENTS, ids=lambda r: r.__name__)
def test_each_component_holds_its_own_loss(route: Route) -> None:
    plain, held = route(NoBasisRisk()), route(HELD)
    assert plain.index_components is not None
    assert held.index_components is not None
    for name, component in held.index_components.items():
        before = plain.index_components[name]
        assert component.hazard is not None
        np.testing.assert_array_equal(
            np.sort(component.hazard[:, 0]), np.sort(before.values[:, 0])
        )
        assert component.own_loss is not None and before.own_loss is not None
        np.testing.assert_allclose(
            component.own_loss.mean(), before.own_loss.mean(), rtol=TOLERANCE
        )
        summary = held.provenance.details["basis_risk"][name]
        assert summary["hazard_mean_ratio"] == 1.0
        assert summary["impact_mean_ratio"] == 1.0


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r.__name__)
def test_the_held_form_keeps_the_ranks_and_the_draws_of_the_default(
    route: Route,
) -> None:
    default, held = route(MODEL), route(HELD)
    for (index, mean_form), (same_index, held_form) in zip(
        _hazards(default), _hazards(held), strict=True
    ):
        np.testing.assert_array_equal(index, same_index)
        # Row d has the rank in both forms that the index times its factor has.
        np.testing.assert_array_equal(
            np.argsort(mean_form, kind="stable"), np.argsort(held_form, kind="stable")
        )
    assert held.provenance.rng == default.provenance.rng


@pytest.mark.parametrize("hold", ["hazard_mean", "hazard_distribution"])
def test_both_forms_leave_the_generator_in_one_state(hold: Any) -> None:
    def state(model: Any) -> Any:
        rng = np.random.default_rng(SEED)
        sample_annual(
            _series(), loss_model=LOSS, basis_risk=model, asset_id="a", rng=rng
        )
        return rng.bit_generator.state

    assert state(HazardNoise(0.4, hold=hold)) == state(MODEL)


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r.__name__)
def test_the_held_form_at_sigma_zero_gives_the_rows_of_no_basis_risk(
    route: Route,
) -> None:
    plain = route(NoBasisRisk())
    with pytest.warns(ComonotoneIndexWarning):
        zero = route(HazardNoise(0.0, hold="hazard_distribution"))
    np.testing.assert_array_equal(zero.index, plain.index)
    np.testing.assert_array_equal(zero.loss, plain.loss)
    assert zero.hazard is None
    assert zero.content_hash == plain.content_hash


def test_the_rule_by_hand() -> None:
    index = np.array([30.0, 0.0, 10.0, 20.0, 0.0, 40.0])
    rng = np.random.default_rng(SEED)
    _, held = HELD.apply_hazard(index, index.copy(), rng)
    factor = np.exp(np.random.default_rng(SEED).normal(-0.5 * SIGMA**2, SIGMA, 6))
    ranked = index * factor
    expected = np.empty(6)
    expected[np.lexsort((index, ranked))] = np.sort(index)
    np.testing.assert_array_equal(held, expected)
    assert (held[index == 0.0] == 0.0).all()


def test_a_zero_stays_zero_when_a_factor_underflows() -> None:
    # sigma 40 makes most factors exp(-800) = 0, so many products tie at zero.
    index = np.array([0.0, 5.0, 0.0, 7.0, 3.0, 0.0, 9.0, 1.0])
    model = HazardNoise(40.0, hold="hazard_distribution")
    _, held = model.apply_hazard(index, index.copy(), np.random.default_rng(SEED))
    _, plain = HazardNoise(40.0).apply_hazard(
        index, index.copy(), np.random.default_rng(SEED)
    )
    assert (plain[index > 0] == 0.0).any()
    np.testing.assert_array_equal(held[index == 0.0], 0.0)
    np.testing.assert_array_equal(np.sort(held), np.sort(index))


def test_the_rule_ranks_each_column_of_a_matrix() -> None:
    index = np.arange(1.0, 13.0).reshape(6, 2)
    _, held = HELD.apply_hazard(index, index.copy(), np.random.default_rng(SEED))
    assert held.shape == index.shape
    for column in range(2):
        np.testing.assert_array_equal(
            np.sort(held[:, column]), np.sort(index[:, column])
        )
    scalar = np.array(5.0)
    _, one = HELD.apply_hazard(scalar, scalar.copy(), np.random.default_rng(SEED))
    assert one.shape == () and float(one) == 5.0


def test_the_failure_rates_respond_to_sigma_at_one_loss_level() -> None:
    def row(sigma: float) -> Any:
        model = HazardNoise(sigma, hold="hazard_distribution")
        return evaluate(STEP, _curve(model, n=5000), split="in_sample")

    low, high = row(0.1), row(0.5)
    assert _column(low, "expected_loss") == pytest.approx(
        _column(high, "expected_loss"), rel=TOLERANCE
    )
    assert 0.0 < _column(low, "false_positive_rate_l0")
    assert _column(low, "false_positive_rate_l0") < _column(
        high, "false_positive_rate_l0"
    )
    assert _column(low, "false_negative_rate_l0") < _column(
        high, "false_negative_rate_l0"
    )


# -- the held form and the leakage rules ----------------------------------------


@pytest.mark.parametrize("route", ROUTES, ids=lambda r: r.__name__)
def test_a_held_sample_is_not_split(route: Route) -> None:
    sample = route(HELD)
    with pytest.raises(SampleError, match="holdout cannot split this sample"):
        holdout(sample)
    with pytest.raises(SampleError, match="year_block_kfold cannot split"):
        year_block_kfold(sample, 4)


def test_an_assets_own_sample_and_a_component_are_not_split_either() -> None:
    with pytest.raises(SampleError, match="cannot split"):
        holdout(_assets(HELD).asset("b"))
    with pytest.raises(SampleError, match="cannot split"):
        holdout(_annual_hazards(HELD).component("wind"))
    # A joined sample with one held asset among its parts.
    held = sample_annual(
        _series(_values(2), name="b"),
        loss_model=LOSS,
        basis_risk=HELD,
        asset_id="b",
        rng=np.random.default_rng(SEED),
    )
    joined = OutcomeSample.join([_annual(MODEL), held])
    with pytest.raises(SampleError, match="cannot split"):
        year_block_kfold(joined, 2)


def test_blocks_selected_from_one_held_sample_are_refused() -> None:
    template = Cover(
        INDEX, StepPayout([90.0, 120.0], [Free(), Free()]), payout_unit="EUR"
    )
    sample = _curve(HELD)
    first, second = sample.select(np.arange(0, 200)), sample.select(np.arange(200, N))
    designed = design(template, first, MSE())
    with pytest.raises(LeakageError, match="cut from one sample"):
        evaluate(designed, second, split="out_of_sample")
    # The default's factors are each row's own: its blocks pass, as before.
    plain = _curve(MODEL)
    designed = design(template, plain.select(np.arange(0, 200)), MSE())
    evaluate(designed, plain.select(np.arange(200, N)), split="out_of_sample")


def test_separate_calls_each_hold_their_own_rows_and_pass() -> None:
    template = Cover(
        INDEX, StepPayout([90.0, 120.0], [Free(), Free()]), payout_unit="EUR"
    )
    drawn, fresh = _curve(HELD, seed=1), _curve(HELD, seed=2)
    table = evaluate(design(template, drawn, MSE()), fresh, split="out_of_sample")
    assert table.num_rows == 1
    for sample in (drawn, fresh):
        assert sample.hazard is not None
        np.testing.assert_array_equal(
            np.sort(sample.hazard[:, 0]), np.sort(sample.index[:, 0])
        )
    # Per-year samples of separate years of one series.
    values = _values()

    def years(rows: slice) -> OutcomeSample:
        series = AnnualSeries(
            YEARS[rows], values[rows], "mm/day", {"source": {"name": "synthetic s"}}
        )
        return sample_annual(
            series,
            loss_model=LOSS,
            basis_risk=HELD,
            asset_id="a",
            rng=np.random.default_rng(SEED),
        )

    early, late = years(slice(0, 28)), years(slice(28, None))
    assert early.lineage == late.lineage
    table = evaluate(design(template, early, MSE()), late, split="out_of_sample")
    assert table.num_rows == 1


# -- the held form in the setup text ---------------------------------------------


HELD_STATED = (
    "hazard noise that holds the hazard's distribution, the hazard at the asset "
    "being the index's own values, reordered by the rank of the index times a "
    "mean-one lognormal factor with sigma 0.25, one per year and asset, as an "
    "assumption the caller states"
)


def test_the_setup_text_states_the_held_form_and_its_loss_level() -> None:
    sample = _annual(HELD)
    line = _sample_line(
        setup_text(_metadata_of(evaluate(STEP, sample, split="in_sample")))
    )
    assert (
        f"{HELD_STATED}; over the rows as built, with the model over without it "
        "(sigma 0, on the same draws), the expected loss is 1 and the mean hazard "
        "is 1"
    ) in line


def test_the_setup_text_names_another_models_reference_plainly() -> None:
    setup = {
        "samples": [
            {
                "basis_risk": {"kind": "user_model"},
                "details": {
                    "basis_risk": {"hazard_mean_ratio": 2.0, "impact_mean_ratio": 3.0}
                },
                "source_kind": "annual",
            }
        ]
    }
    line = _sample_line(setup_text(setup))
    assert (
        "with the model over without it, the expected loss is 3 and the mean "
        "hazard is 2"
    ) in line


def test_an_unknown_form_in_a_setup_is_refused() -> None:
    setup = {
        "samples": [
            {"basis_risk": {"kind": "hazard_noise", "sigma": 0.25, "hold": "x"}}
        ]
    }
    with pytest.raises(EvaluationError, match=r"basis_risk\.hold must be"):
        setup_text(setup)
