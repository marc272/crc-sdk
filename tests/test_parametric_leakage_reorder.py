"""Draws a dependence model moved are out of sample only where their rows say so.

A curve sample's period is its draw number only while its column is in draw
order. A dependence model other than `Independent()` moves a column's draws to
the rows its ranks over every row choose. Two samples built by separate calls
from the same generators, one reordered and the other not, or reordered by two
models, then hold the same draws in disjoint periods. And two cuts of one
sample whose columns `Comonotonic()` or `GaussianCopula` moved depend on each
other wherever a cut reads several columns, while one column's own rows are a
random share of its draws. The tests below show both, then check that
`_check_out_of_sample` refuses them, through views, written tables, the
generator records and `evaluate`, and that fresh draws and cuts of an
`Independent()` sample still pass. Every number is synthetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact
from numpy.typing import NDArray

from crc_sdk.parametric import (
    MSE,
    CappedSum,
    Comonotonic,
    Cover,
    EmpiricalCopula,
    Free,
    GaussianCopula,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LeakageError,
    LossModel,
    OutcomeSample,
    SampleProvenance,
    StepPayout,
    design,
    evaluate,
    holdout,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    year_block_kfold,
)
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

FloatArray = NDArray[np.float64]
ASSETS = ("site-b", "site-c", "site-a")
SEEDS = {"site-b": 11, "site-c": 22, "site-a": 33, "rain": 11, "heat": 22}
LOCATIONS = {"site-b": 50.0, "site-c": 80.0, "site-a": 30.0, "rain": 50.0, "heat": 80.0}
UNITS = {"rain": "mm/day", "heat": "degC"}
HAZARDS = ("rain", "heat")
RAMP = PiecewiseLinearImpact([20.0, 140.0], [0.0, 1.0])
NOISE = ImpactNoise(0.3, upper=1.0)
M = 30
N = 300
HALF = N // 2
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
APART = "their rows were placed apart"
RANKED = "depend on the other block"


def _metadata(unit: str = "mm/day") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit=unit,
        value_semantics="annual maximum",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="synthetic",
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
    )


def _curve(name: str, shift: float = 0.0) -> FittedDistribution:
    return FittedDistribution.from_parameters(
        "gumbel_r", location=LOCATIONS[name] + shift, scale=10.0
    )


def _loss(name: str) -> LossModel:
    return LossModel(RAMP, value=1e6, value_unit="EUR", name=f"loss of {name}")


def _reference(seed: int = 5) -> OutcomeSample:
    """A per-year sample of the three assets, whose columns have their own ranks."""
    rng = np.random.default_rng(seed)
    return OutcomeSample(
        asset_ids=ASSETS,
        period=np.arange(1990, 1990 + M),
        probability=np.full(M, 1.0 / M),
        index=rng.gamma(2.0, 20.0, (M, len(ASSETS))),
        loss=rng.gamma(2.0, 5.0, (M, len(ASSETS))),
        hazard=None,
        index_unit="degC",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="annual",
            sources=({"role": "hazard", "name": "era5 series"},),
            semantics_source="data",
            dependence={"kind": "data"},
        ),
    )


def _gaussian(rho: float) -> GaussianCopula:
    return GaussianCopula(np.eye(3) * (1.0 - rho) + rho)


def _assets(
    model: Any, *, n: int = N, shift: int = 0, moved: float = 0.0
) -> OutcomeSample:
    """Three assets' draws; `moved` shifts every curve, `shift` every seed."""
    return sample_curves(
        {asset: _curve(asset, moved) for asset in ASSETS},
        dependence=model,
        metadata={asset: _metadata() for asset in ASSETS},
        n_draws=n,
        rngs={a: np.random.default_rng(SEEDS[a] + shift) for a in ASSETS},
        loss_models={asset: _loss(asset) for asset in ASSETS},
        basis_risk=NOISE,
    )


def _single(asset: str) -> OutcomeSample:
    """One asset's draws from the generator `_assets` gives it."""
    return sample_curve(
        _curve(asset),
        metadata=_metadata(),
        n_draws=N,
        rng=np.random.default_rng(SEEDS[asset]),
        loss_model=_loss(asset),
        basis_risk=NOISE,
        asset_id=asset,
    )


def _hazards(model: Any, *, shift: int = 0) -> OutcomeSample:
    return sample_curve_hazards(
        {name: _curve(name) for name in HAZARDS},
        dependence=model,
        metadata={name: _metadata(UNITS[name]) for name in HAZARDS},
        n_draws=N,
        rngs={name: np.random.default_rng(SEEDS[name] + shift) for name in HAZARDS},
        loss_models={name: _loss(name) for name in HAZARDS},
        basis_risk=NOISE,
        combine=CappedSum(1.0),
        asset_id="site",
    )


def _first(sample: OutcomeSample) -> OutcomeSample:
    return sample.select(np.arange(HALF))


def _second(sample: OutcomeSample) -> OutcomeSample:
    return sample.select(np.arange(HALF, N))


def _shared(first: FloatArray, second: FloatArray) -> int:
    """How many values two columns of draws hold in common."""
    return int(np.intersect1d(first, second).size)


def _refused(designed: OutcomeSample, evaluation: OutcomeSample, match: str) -> str:
    with pytest.raises(LeakageError, match=match) as raised:
        _check_out_of_sample(designed, evaluation)
    return str(raised.value)


class Reversed:
    """A model of the caller's: the second column's draws in reverse draw order."""

    def to_document(self) -> dict[str, Any]:
        return {"kind": "reversed_second"}

    def record(self, n_draws: int, columns: Any) -> dict[str, Any]:
        return self.to_document()

    def statement(self, n_draws: int, between: str) -> str:
        return "the second column is drawn in reverse."

    def reorder(self, uniforms: FloatArray, columns: Any) -> NDArray[np.int64]:
        n, d = uniforms.shape
        order = np.tile(np.arange(n, dtype=np.int64)[:, np.newaxis], (1, d))
        order[:, 1] = order[::-1, 1]
        return order


# -- one lineage placed apart: the same generators under two models ---------------


def test_a_copula_cut_and_an_independent_cut_of_the_same_draws_share_draws() -> None:
    copula, independent = _assets(EmpiricalCopula(_reference())), _assets(Independent())
    first, second = _first(copula), _second(independent)
    assert np.intersect1d(first.period, second.period).size == 0
    # Disjoint periods, yet 65 of site-b's 150 draws are on both sides.
    assert _shared(first.index[:, 0], second.index[:, 0]) == 65
    message = _refused(first, second, APART)
    assert "reordered by EmpiricalCopula in the design sample" in message
    assert "in draw order under Independent() in the evaluation sample" in message
    message = _refused(second, first, APART)
    assert "in draw order under Independent() in the design sample" in message


PAIRS = [
    (EmpiricalCopula(_reference()), Independent()),
    (Comonotonic(), Independent()),
    (_gaussian(0.5), Independent()),
    (Comonotonic(), _gaussian(0.5)),
    (_gaussian(0.3), _gaussian(0.6)),
    (EmpiricalCopula(_reference()), EmpiricalCopula(_reference(seed=6))),
    (EmpiricalCopula(_reference()), Comonotonic()),
]
PAIR_IDS = [
    "copula-independent",
    "comonotonic-independent",
    "gaussian-independent",
    "comonotonic-gaussian",
    "gaussian-gaussian",
    "copula-copula",
    "copula-comonotonic",
]


@pytest.mark.parametrize(("one", "other"), PAIRS, ids=PAIR_IDS)
def test_cuts_placed_by_two_models_are_refused_both_ways(one: Any, other: Any) -> None:
    ours, theirs = _assets(one), _assets(other)
    assert ours.lineage == theirs.lineage
    # A moved asset's draws land on both sides of the cut.
    assert _shared(_first(ours).index[:, 1], _second(theirs).index[:, 1]) > 0
    _refused(_first(ours), _second(theirs), APART)
    _refused(_second(theirs), _first(ours), APART)


def test_one_moved_asset_against_its_own_draws_in_draw_order_is_refused() -> None:
    view = _assets(Comonotonic()).asset("site-c")
    single = _single("site-c")
    assert view.lineage == single.lineage
    message = _refused(_first(view), _second(single), APART)
    assert "reordered by Comonotonic() in the design sample" in message
    assert "in draw order in the evaluation sample" in message
    _refused(_first(single), _second(view), APART)


def test_an_asset_left_in_draw_order_passes_under_either_model() -> None:
    # Comonotonic() keeps the first asset in draw order: its period is its draw.
    kept = _assets(Comonotonic()).asset("site-b")
    for other in (_assets(Independent()).asset("site-b"), _single("site-b")):
        assert _shared(_first(kept).index, _second(other).index) == 0
        _check_out_of_sample(_first(kept), _second(other))
        _check_out_of_sample(_first(other), _second(kept))


def test_a_sample_ranked_on_the_shared_draws_is_refused() -> None:
    # Only site-b's draws are shared, in draw order on both sides, but
    # Comonotonic() placed the other assets by the ranks of site-b's draws on
    # every row, the evaluation's included.
    joint = _assets(Comonotonic())
    _refused(_first(joint), _second(_single("site-b")), APART)
    _refused(_first(_single("site-b")), _second(joint), APART)


def test_hazards_placed_apart_are_refused_through_every_view() -> None:
    moved, kept = _hazards(Comonotonic()), _hazards(Independent())
    _refused(_first(moved), _second(kept), APART)
    _refused(_first(moved).component("heat"), _second(kept).component("heat"), APART)
    # Rain stayed in place, so its own rows pass, but its view on the combined
    # loss reads heat's impacts as Comonotonic() placed them.
    _check_out_of_sample(
        _first(moved).component("rain"), _second(kept).component("rain")
    )
    combined = _first(moved).component("rain", loss="combined")
    assert {"kind": "reorder", "dependence": "comonotonic"} in [
        dict(step) for step in combined.provenance.selection
    ]
    _refused(combined, _second(kept).component("rain", loss="combined"), APART)


def test_written_cuts_placed_apart_are_refused() -> None:
    first = _first(_assets(Comonotonic()))
    second = _second(_assets(Independent()))
    written = [OutcomeSample.from_written_table(s.to_table()) for s in (first, second)]
    _refused(written[0], written[1], APART)


def test_shared_generators_under_other_curves_placed_apart_are_refused() -> None:
    # Every curve differs, so no source is shared, but every generator is.
    moved = _assets(Comonotonic(), moved=1.0)
    kept = _assets(Independent())
    assert moved.lineage != kept.lineage
    message = _refused(_first(moved), _second(kept), APART)
    assert "share a generator state" in message
    _check_out_of_sample(_first(_assets(Independent(), moved=1.0)), _second(kept))


def test_evaluate_refuses_a_design_read_on_draws_placed_apart() -> None:
    template = Cover(
        INDEX, StepPayout([60.0, 80.0], [Free(), Free()]), payout_unit="EUR"
    )
    view = _assets(EmpiricalCopula(_reference())).asset("site-c")
    designed = design(template, _first(view), MSE())
    with pytest.raises(LeakageError, match=APART):
        evaluate(designed, _second(_single("site-c")), split="out_of_sample")
    fresh = _assets(Independent(), shift=1000).asset("site-c")
    assert evaluate(designed, fresh, split="out_of_sample").num_rows == 1


# -- cuts of one sample: the measured dependence -----------------------------------


def _spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def test_the_blocks_of_a_comonotonic_cut_depend_on_each_other() -> None:
    # Over 200 seeds of 200 rows: under Comonotonic() a block's moved site-c
    # values are site-c's draws at the ranks site-b's values take among both
    # blocks', so they fall as the other block's site-b values rise. One
    # asset's own rows, and an Independent() sample, show no such link.
    by_model: dict[str, list[tuple[float, float, float]]] = {}
    for name, model in (("comonotonic", Comonotonic()), ("independent", Independent())):
        rows = []
        for seed in range(200):
            sample = _assets(model, n=200, shift=1000 * (seed + 1))
            first, second = sample.index[:100], sample.index[100:]
            rows.append(
                (
                    float(first[:, 1].mean()),
                    float(second[:, 0].mean()),
                    float(second[:, 1].mean()),
                )
            )
        by_model[name] = rows
    moved = by_model["comonotonic"]
    across = _spearman([r[0] for r in moved], [r[1] for r in moved])
    own = _spearman([r[0] for r in moved], [r[2] for r in moved])
    kept = by_model["independent"]
    control = _spearman([r[0] for r in kept], [r[1] for r in kept])
    assert across < -0.3
    assert abs(own) < 0.2
    assert abs(control) < 0.2


MODELS = [Comonotonic(), _gaussian(0.5)]
MODEL_IDS = ["comonotonic", "gaussian"]


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_cuts_of_several_moved_assets_are_refused(model: Any) -> None:
    sample = _assets(model)
    message = _refused(_first(sample), _second(sample), RANKED)
    assert "FreshDraws" in message
    train, test = holdout(sample, 0.7)
    _refused(train, test, RANKED)
    for train, test in year_block_kfold(sample, 3):
        _refused(train, test, RANKED)


@pytest.mark.parametrize("model", [Comonotonic(), GaussianCopula(0.5)], ids=MODEL_IDS)
def test_cuts_of_several_moved_hazards_are_refused(model: Any) -> None:
    sample = _hazards(model)
    train, test = holdout(sample, 0.7)
    _refused(train, test, RANKED)
    _refused(
        train.component("rain", loss="combined"),
        test.component("rain", loss="combined"),
        RANKED,
    )
    _refused(train.component("heat"), test.component("rain"), RANKED)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_the_same_columns_rows_of_each_cut_pass(model: Any) -> None:
    sample = _assets(model)
    first, second = _first(sample), _second(sample)
    for asset in ASSETS:
        _check_out_of_sample(first.asset(asset), second.asset(asset))
    train, test = holdout(sample.asset("site-c"), 0.7)
    _check_out_of_sample(train, test)
    hazards = _hazards(Comonotonic())
    for name in HAZARDS:
        _check_out_of_sample(
            _first(hazards).component(name), _second(hazards).component(name)
        )


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_a_cut_of_several_assets_against_one_assets_cut_is_refused(model: Any) -> None:
    sample = _assets(model)
    first, second = _first(sample), _second(sample)
    _refused(first, second.asset("site-c"), RANKED)
    _refused(first.asset("site-b"), second, RANKED)


def test_a_model_of_the_callers_is_refused_even_on_one_column() -> None:
    sample = _assets(Reversed())
    first, second = _first(sample), _second(sample)
    message = _refused(first.asset("site-c"), second.asset("site-c"), "cannot tell")
    assert "'reversed_second'" in message
    # Its first and third columns stayed in draw order.
    _check_out_of_sample(first.asset("site-b"), second.asset("site-b"))


# -- what stays out of sample ------------------------------------------------------


@pytest.mark.parametrize(("one", "other"), PAIRS, ids=PAIR_IDS)
def test_fresh_draws_pass_whatever_the_models(one: Any, other: Any) -> None:
    ours, fresh = _assets(one), _assets(other, shift=1000)
    _check_out_of_sample(ours, fresh)
    _check_out_of_sample(_first(ours), _second(fresh))
    _check_out_of_sample(_first(fresh).asset("site-c"), ours.asset("site-c"))


def test_cuts_of_one_independent_sample_pass() -> None:
    sample = _assets(Independent())
    _check_out_of_sample(_first(sample), _second(sample))
    _check_out_of_sample(*holdout(sample, 0.7))
    for train, test in year_block_kfold(sample, 3):
        _check_out_of_sample(train, test)
    hazards = _hazards(Independent())
    _check_out_of_sample(
        _first(hazards).component("rain", loss="combined"),
        _second(hazards).component("rain", loss="combined"),
    )
