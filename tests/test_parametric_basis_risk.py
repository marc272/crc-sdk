"""Basis-risk models: the protocols, no basis risk and lognormal noise."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import (
    BasisRisk,
    HazardBasisRisk,
    HazardNoise,
    ImpactNoise,
    NoBasisRisk,
    SampleError,
)

UNTYPED_NOISE: Any = ImpactNoise  # for calls with deliberately invalid arguments
UNTYPED_HAZARD_NOISE: Any = HazardNoise


def test_built_in_models_meet_the_protocol() -> None:
    assert isinstance(NoBasisRisk(), BasisRisk)
    assert isinstance(ImpactNoise(0.2), BasisRisk)
    assert isinstance(HazardNoise(0.2), BasisRisk)


def test_only_a_model_with_apply_hazard_meets_the_hazard_protocol() -> None:
    assert isinstance(HazardNoise(0.2), HazardBasisRisk)
    assert not isinstance(ImpactNoise(0.2), HazardBasisRisk)
    assert not isinstance(NoBasisRisk(), HazardBasisRisk)


def test_no_basis_risk_changes_nothing_and_draws_nothing() -> None:
    index = np.array([1.0, 2.0, 3.0])
    impact = np.array([0.0, 0.1, 0.5])
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    new_index, new_impact = NoBasisRisk().apply(index, index, impact, rng)
    assert new_index is index and new_impact is impact
    assert rng.bit_generator.state == state
    assert NoBasisRisk().to_document() == {"kind": "no_basis_risk"}
    assert NoBasisRisk().name == "no_basis_risk"


def test_impact_noise_is_the_mean_one_lognormal_factor() -> None:
    sigma = 0.4
    impact = np.full(200_000, 0.25)
    index = np.arange(impact.size, dtype=np.float64)
    new_index, noisy = ImpactNoise(sigma).apply(
        index, index, impact, np.random.default_rng(11)
    )
    assert new_index is index
    factor = noisy / impact
    # One rng.normal(-sigma**2/2, sigma) call, as the thesis noise layer.
    expected = np.exp(
        np.random.default_rng(11).normal(-0.5 * sigma**2, sigma, impact.size)
    )
    np.testing.assert_array_equal(factor, expected)
    assert abs(factor.mean() - 1.0) < 0.01
    assert abs(np.log(factor).std() - sigma) < 0.01


def test_impact_noise_keeps_zero_impact_and_caps() -> None:
    impact = np.array([0.0, 0.9, 0.95, 0.99] * 1000)
    _, noisy = ImpactNoise(0.5, upper=1.0).apply(
        impact, impact, impact, np.random.default_rng(0)
    )
    assert (noisy[impact == 0.0] == 0.0).all()
    assert noisy.max() == 1.0
    assert (noisy == 1.0).sum() > 0
    assert ImpactNoise(0.5, upper=1.0).to_document() == {
        "kind": "impact_noise",
        "sigma": 0.5,
        "upper": 1.0,
    }


def test_impact_noise_needs_a_generator() -> None:
    impact = np.array([0.1, 0.2])
    with pytest.raises(SampleError, match="rng"):
        ImpactNoise(0.1).apply(impact, impact, impact, None)


@pytest.mark.parametrize("sigma", [0.0, -0.1, math.inf, math.nan, True, "0.2"])
def test_impact_noise_needs_a_positive_sigma(sigma: Any) -> None:
    with pytest.raises(SampleError, match="sigma"):
        UNTYPED_NOISE(sigma)


@pytest.mark.parametrize("upper", [0.0, -1.0, math.inf])
def test_impact_noise_needs_a_positive_cap(upper: float) -> None:
    with pytest.raises(SampleError, match="upper"):
        ImpactNoise(0.2, upper=upper)


def test_impact_noise_is_frozen() -> None:
    model = ImpactNoise(0.2)
    with pytest.raises(AttributeError):
        model.sigma = 0.3  # type: ignore[misc]


def test_impact_noise_says_what_it_cant_change() -> None:
    assert ImpactNoise.__doc__ is not None
    text = " ".join(ImpactNoise.__doc__.split())
    assert "it can't change whether a year pays or whether a loss occurs" in text
    assert "`false_negative_rate_l0` and `false_positive_rate_l0` don't move" in text
    assert "With a loss threshold above zero they can" in text


def test_hazard_noise_documents_itself() -> None:
    assert HazardNoise(0.25).to_document() == {"kind": "hazard_noise", "sigma": 0.25}
    assert HazardNoise(0.25).name == "hazard_noise"
    assert HazardNoise(1).sigma == 1.0


def test_hazard_noise_apply_returns_its_inputs_and_draws_nothing() -> None:
    index = np.array([1.0, 2.0, 3.0])
    impact = np.array([0.0, 0.1, 0.5])
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    new_index, new_impact = HazardNoise(0.2).apply(index, index, impact, rng)
    assert new_index is index and new_impact is impact
    assert rng.bit_generator.state == state


def test_hazard_noise_is_the_mean_one_lognormal_factor_on_the_index() -> None:
    sigma = 0.4
    index = np.full(200_000, 50.0)
    new_index, at_asset = HazardNoise(sigma).apply_hazard(
        index, index, np.random.default_rng(11)
    )
    assert new_index is index
    # One rng.normal(-sigma**2/2, sigma) call, as in ImpactNoise.
    expected = np.exp(
        np.random.default_rng(11).normal(-0.5 * sigma**2, sigma, index.size)
    )
    np.testing.assert_array_equal(at_asset, index * expected)
    factor = at_asset / index
    assert abs(factor.mean() - 1.0) < 0.01
    assert abs(np.log(factor).std() - sigma) < 0.01


def test_hazard_noise_at_sigma_zero_is_exactly_one_and_still_draws() -> None:
    index = np.array([0.0, 1.5, 1e6])
    rng = np.random.default_rng(5)
    _, at_asset = HazardNoise(0.0).apply_hazard(index, index, rng)
    np.testing.assert_array_equal(at_asset, index)
    drawn = np.random.default_rng(5)
    drawn.normal(0.0, 1.0, 3)
    assert rng.bit_generator.state == drawn.bit_generator.state


def test_hazard_noise_shares_its_normals_across_sigmas() -> None:
    index = np.full(1000, 10.0)
    logs = [
        np.log(
            HazardNoise(sigma).apply_hazard(index, index, np.random.default_rng(2))[1]
            / index
        )
        for sigma in (0.1, 0.7)
    ]
    np.testing.assert_allclose(
        (logs[0] + 0.5 * 0.1**2) / 0.1, (logs[1] + 0.5 * 0.7**2) / 0.7, atol=1e-9
    )


def test_hazard_noise_keeps_a_zero_and_refuses_a_negative() -> None:
    index = np.array([0.0, 5.0])
    _, at_asset = HazardNoise(2.0).apply_hazard(index, index, np.random.default_rng(1))
    assert at_asset[0] == 0.0
    negative = np.array([-1.0, 5.0])
    with pytest.raises(SampleError, match="ratio scale"):
        HazardNoise(0.2).apply_hazard(negative, negative, np.random.default_rng(1))


def test_hazard_noise_refuses_an_index_that_is_not_the_hazard() -> None:
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    with pytest.raises(SampleError, match="differ"):
        HazardNoise(0.2).apply_hazard(np.array([1.0]), np.array([2.0]), rng)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("sigma", [0.0, 0.2])
def test_hazard_noise_needs_a_generator_at_every_sigma(sigma: float) -> None:
    index = np.array([0.1, 0.2])
    with pytest.raises(SampleError, match="rng"):
        HazardNoise(sigma).apply_hazard(index, index, None)


@pytest.mark.parametrize("sigma", [-0.1, math.inf, math.nan, True, "0.2", None])
def test_hazard_noise_needs_a_finite_sigma_that_is_not_negative(sigma: Any) -> None:
    with pytest.raises(SampleError, match="sigma"):
        UNTYPED_HAZARD_NOISE(sigma)


def test_hazard_noise_is_frozen() -> None:
    model = HazardNoise(0.2)
    with pytest.raises(AttributeError):
        model.sigma = 0.3  # type: ignore[misc]
