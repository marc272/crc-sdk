"""Cuts of one `EmpiricalCopula` sample, taken with `select`, are not out of sample.

An `EmpiricalCopula` deals each column's draws to the reference years by their
rank over every row of the sample: the k draws of rank band r go to the rows of
the reference year of rank r. The tests below show the two ways a cut of such a
sample fails, then check that `_check_out_of_sample` refuses both, on either
axis, through one asset's or one hazard's view, after a written table, and
through `evaluate`. Fresh draws with the same reference, and cuts of the other
models, still pass. The same detection sets the resampling unit of
`bootstrap_intervals`. Every number is synthetic.
"""

from __future__ import annotations

import warnings
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
    EmpiricalCopula,
    Free,
    GaussianCopula,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LeakageError,
    LossModel,
    NoBasisRisk,
    OutcomeSample,
    SampleProvenance,
    StepPayout,
    design,
    evaluate,
    sample_annual_hazards,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric.intervals import _resampling_unit
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

FloatArray = NDArray[np.float64]

ASSETS = ("site-b", "site-c", "site-a")
LOCATIONS = {"site-b": 50.0, "site-c": 80.0, "site-a": 30.0, "rain": 50.0, "heat": 80.0}
UNITS = {"rain": "mm/day", "heat": "degC"}
HAZARDS = ("rain", "heat")
RAMP = PiecewiseLinearImpact([20.0, 140.0], [0.0, 1.0])
NOISE = ImpactNoise(0.3, upper=1.0)
M = 30
K = 10
N = K * M
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
MESSAGE = "cut from one sample, and an EmpiricalCopula"


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


def _curve(name: str) -> FittedDistribution:
    return FittedDistribution.from_parameters(
        "gumbel_r", location=LOCATIONS[name], scale=10.0
    )


def _loss(name: str) -> LossModel:
    return LossModel(RAMP, value=1e6, value_unit="EUR", name=f"loss of {name}")


def _reference(m: int = M, seed: int = 5) -> OutcomeSample:
    """A per-year sample of the three assets, whose columns have their own ranks."""
    rng = np.random.default_rng(seed)
    return OutcomeSample(
        asset_ids=ASSETS,
        period=np.arange(1990, 1990 + m),
        probability=np.full(m, 1.0 / m),
        index=rng.gamma(2.0, 20.0, (m, len(ASSETS))),
        loss=rng.gamma(2.0, 5.0, (m, len(ASSETS))),
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


def _assets(model: Any, n: int = N, shift: int = 0) -> OutcomeSample:
    return sample_curves(
        {asset: _curve(asset) for asset in ASSETS},
        dependence=model,
        metadata={asset: _metadata() for asset in ASSETS},
        n_draws=n,
        rngs={
            asset: np.random.default_rng(11 * (j + 1) + shift)
            for j, asset in enumerate(ASSETS)
        },
        loss_models={asset: _loss(asset) for asset in ASSETS},
        basis_risk=NOISE,
    )


def _hazard_reference() -> OutcomeSample:
    rng = np.random.default_rng(5)
    series = {
        name: AnnualSeries(
            np.arange(1990, 1990 + M),
            rng.gamma(2.0, 20.0, M),
            UNITS[name],
            {"kind": "table", "source": {"name": f"era5 {name}"}},
        )
        for name in HAZARDS
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        return sample_annual_hazards(
            series,
            tails=dict.fromkeys(HAZARDS, "upper"),
            loss_models={"rain": _loss("rain")},
            basis_risk=NoBasisRisk(),
            asset_id="site",
        )


def _hazards(model: Any, shift: int = 0) -> OutcomeSample:
    return sample_curve_hazards(
        {name: _curve(name) for name in HAZARDS},
        dependence=model,
        metadata={name: _metadata(UNITS[name]) for name in HAZARDS},
        n_draws=N,
        rngs={
            name: np.random.default_rng(11 * (j + 1) + shift)
            for j, name in enumerate(HAZARDS)
        },
        loss_models={name: _loss(name) for name in HAZARDS},
        basis_risk=NOISE,
        combine=CappedSum(1.0),
        asset_id="site",
    )


def _ranks(values: FloatArray) -> NDArray[np.int64]:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")


def _whole_years(sample: OutcomeSample) -> tuple[OutcomeSample, OutcomeSample]:
    """The first half of the reference years, and the second."""
    half = len(sample) // 2
    return sample.select(sample.period[:half]), sample.select(sample.period[half:])


def _slots(sample: OutcomeSample, k: int = K) -> tuple[OutcomeSample, OutcomeSample]:
    """The first half of every reference year's rows, and the second half."""
    first = np.arange(len(sample)) % k < k // 2
    return sample.select(sample.period[first]), sample.select(sample.period[~first])


# -- why a cut is not out of sample ----------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_whole_reference_years_hold_ranks_the_other_block_lacks(seed: int) -> None:
    # One draw per reference year: the years above the median of a column's
    # reference take that column's larger half of the draws, whatever the seed.
    reference = _reference()
    sample = _assets(EmpiricalCopula(reference), n=M, shift=100 * seed)
    years = np.argsort(reference.index[:, 0], kind="stable")
    low, high = np.sort(years[: M // 2]), np.sort(years[M // 2 :])
    assert sample.index[low, 0].max() < sample.index[high, 0].min()


def test_one_reference_year_on_both_sides_holds_one_rank_band() -> None:
    sample = _assets(EmpiricalCopula(_reference()))
    first, second = _slots(sample)
    for column in range(len(ASSETS)):
        band = _ranks(sample.index[:, column]) // K
        rows = band.reshape(M, K)
        # Every row of a reference year has that year's band, on either side.
        assert (rows == rows[:, :1]).all()
        np.testing.assert_array_equal(
            band[np.isin(sample.period, first.period)].reshape(M, K // 2)[:, 0],
            band[np.isin(sample.period, second.period)].reshape(M, K // 2)[:, 0],
        )


# -- the check refuses them --------------------------------------------------------


@pytest.mark.parametrize("cut", [_whole_years, _slots], ids=["whole_years", "slots"])
def test_cuts_of_one_copula_sample_are_refused(cut: Any) -> None:
    first, second = cut(_assets(EmpiricalCopula(_reference())))
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(first, second)
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(second, first)


def test_cuts_of_one_assets_rows_are_refused() -> None:
    sample = _assets(EmpiricalCopula(_reference()))
    first, second = _whole_years(sample)
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(first.asset("site-c"), second.asset("site-c"))
    view = sample.asset("site-c")
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(view.select(first.period), view.select(second.period))
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(first, second.asset("site-c"))


@pytest.mark.parametrize("cut", [_whole_years, _slots], ids=["whole_years", "slots"])
def test_cuts_of_a_copula_sample_across_hazards_are_refused(cut: Any) -> None:
    first, second = cut(_hazards(EmpiricalCopula(_hazard_reference())))
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(first, second)
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(first.component("rain"), second.component("rain"))


def test_written_cuts_are_refused() -> None:
    first, second = _whole_years(_assets(EmpiricalCopula(_reference())))
    written = [OutcomeSample.from_written_table(s.to_table()) for s in (first, second)]
    with pytest.raises(LeakageError, match=MESSAGE):
        _check_out_of_sample(*written)


def test_evaluate_refuses_a_design_on_one_cut_read_on_the_other() -> None:
    template = Cover(
        INDEX, StepPayout([60.0, 80.0], [Free(), Free()]), payout_unit="EUR"
    )
    view = _assets(EmpiricalCopula(_reference())).asset("site-b")
    first, second = _whole_years(view)
    designed = design(template, first, MSE())
    with pytest.raises(LeakageError, match=MESSAGE):
        evaluate(designed, second, split="out_of_sample")
    # A fresh sample with new generators and the same reference passes.
    fresh = _assets(EmpiricalCopula(_reference()), shift=1000).asset("site-b")
    assert evaluate(designed, fresh, split="out_of_sample").num_rows == 1


def test_fresh_draws_with_the_same_reference_pass() -> None:
    copula = EmpiricalCopula(_reference())
    first, _ = _whole_years(_assets(copula))
    fresh = _assets(copula, shift=1000)
    _check_out_of_sample(first, fresh)
    _check_out_of_sample(first, _whole_years(fresh)[1])


@pytest.mark.parametrize(
    "model",
    [Independent(), Comonotonic(), GaussianCopula(np.eye(3) * 0.5 + 0.5)],
    ids=["independent", "comonotonic", "gaussian"],
)
def test_cuts_of_one_assets_rows_of_the_other_models_still_pass(model: Any) -> None:
    first, second = _whole_years(_assets(model))
    _check_out_of_sample(first.asset("site-c"), second.asset("site-c"))


def test_cuts_of_every_asset_pass_only_in_draw_order() -> None:
    first, second = _whole_years(_assets(Independent()))
    _check_out_of_sample(first, second)
    for model in (Comonotonic(), GaussianCopula(np.eye(3) * 0.5 + 0.5)):
        first, second = _whole_years(_assets(model))
        with pytest.raises(LeakageError, match="depend on the other block"):
            _check_out_of_sample(first, second)


# -- the resampling unit of bootstrap_intervals ------------------------------------


def test_every_view_of_a_copula_sample_is_resampled_by_row() -> None:
    assets = _assets(EmpiricalCopula(_reference()))
    hazards = _hazards(EmpiricalCopula(_hazard_reference()))
    first, _ = _whole_years(assets)
    for sample in (
        assets,
        first,
        first.asset("site-c"),
        assets.asset("site-b"),
        hazards,
        hazards.component("heat"),
        _whole_years(hazards)[1].component("rain"),
        OutcomeSample.from_written_table(first.to_table()),
    ):
        assert _resampling_unit(sample) == "row"
    for model in (Independent(), Comonotonic()):
        assert _resampling_unit(_whole_years(_assets(model))[0]) == "year"
