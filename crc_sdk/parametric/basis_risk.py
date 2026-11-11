"""Basis-risk models: how an outcome sample's index and loss differ.

A cover pays on its index; the owner bears the loss. Basis risk is the gap
between the two. Every sample states the model it was built with, and none is
assumed: the samplers take `basis_risk` as a required argument.

- `HazardNoise(sigma)`: the hazard at the asset is the index times a mean-one
  lognormal factor `exp(sigma * Z - sigma**2 / 2)`, one draw per year and
  asset. The cover pays on the index and the loss model reads the hazard at
  the asset, so a year can pay with no loss and a loss can occur with no
  payout. `sigma` is an assumption the caller states. The factor widens the
  hazard, so the expected loss moves with `sigma`; with
  `hold="hazard_distribution"` the hazard at the asset keeps the index's own
  values, reassigned to the years by the rank of the index times the factor,
  so the loss of one hazard keeps its distribution at every `sigma`.
- `ImpactNoise(sigma, *, upper=None)`: multiplies each year's impact by a
  mean-one lognormal factor `exp(sigma * Z - sigma**2 / 2)`, one draw per
  year, then caps it at `upper` if given (1 for a damage ratio). The index is
  unchanged. The cap lowers the mean impact a little; the samplers record the
  realised ratio.
- `NoBasisRisk()`: the index is the hazard the loss model reads, so the index
  and the loss move together (comonotone) and a cover can track the loss
  almost exactly. The samplers warn with `ComonotoneIndexWarning` when it is
  used without a separate index series.
- A user model implements the `BasisRisk` protocol, and `HazardBasisRisk` if
  it acts before the impact is computed.

A per-year index read from another cell, station or hazard than the loss is
not a model here: pass it to `sample_annual` as `index=`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, Protocol, get_args, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.errors import SampleError

__all__ = ["BasisRisk", "HazardBasisRisk", "HazardNoise", "ImpactNoise", "NoBasisRisk"]

FloatArray = NDArray[np.float64]

_Hold = Literal["hazard_mean", "hazard_distribution"]
"""What `HazardNoise` holds as `sigma` moves: the mean hazard, or its distribution."""


@runtime_checkable
class BasisRisk(Protocol):
    """What every basis-risk model provides.

    `name` identifies the model in the sample's provenance, and
    `to_document()` returns its parameters as plain JSON types, including a
    `kind` key.

    `apply(index, hazard, impact, rng)` takes one value per draw: the index
    the cover would pay on, the hazard the loss model read, and the impact
    (usually a damage ratio) the loss model computed from it. It returns the
    new `(index, impact)`, each of the input's shape, finite, and the impact
    non-negative. The loss is the returned impact times the asset value.
    `rng` is the sample's generator, or `None` when the caller gave none; a
    model that draws must raise without one. The samplers call `apply` after
    the hazard draws, so the draw order is fixed: hazard first, then the
    basis-risk model.
    """

    @property
    def name(self) -> str: ...

    def to_document(self) -> dict[str, Any]: ...

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]: ...


@runtime_checkable
class HazardBasisRisk(BasisRisk, Protocol):
    """A basis-risk model that also acts before the impact is computed.

    `apply_hazard(index, hazard, rng)` takes one value per draw, the index the
    cover would pay on and the hazard as read or drawn, and returns the new
    `(index, hazard at the asset)`, each of the input's shape and finite. The
    samplers then compute the impact from the hazard at the asset with the
    loss model, so the model never holds the damage function, and call
    `apply` as for any model. The order is fixed: the hazard, `apply_hazard`,
    the impact, which draws nothing, and `apply`.

    A model that doesn't implement it skips that step.
    """

    def apply_hazard(
        self,
        index: FloatArray,
        hazard: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]: ...


@dataclass(frozen=True)
class NoBasisRisk:
    """No basis risk: the index is the hazard at the asset.

    The loss is then a function of the index, so a cover can track it almost
    exactly. Write it out to state that choice; the samplers warn with
    `ComonotoneIndexWarning` when the index is not read from another series.
    """

    @property
    def name(self) -> str:
        """`"no_basis_risk"`."""
        return "no_basis_risk"

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "no_basis_risk"}`."""
        return {"kind": self.name}

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the index and the impact unchanged; draws nothing."""
        del hazard, rng
        return index, impact


def _real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a real number, got {value!r}")
    return float(value)


def _positive(name: str, value: Any) -> float:
    number = _real(name, value)
    if not math.isfinite(number) or number <= 0.0:
        raise SampleError(f"{name} must be finite and positive, got {value!r}")
    return number


def _not_negative(name: str, value: Any) -> float:
    number = _real(name, value)
    if not math.isfinite(number) or number < 0.0:
        raise SampleError(f"{name} must be finite and not negative, got {value!r}")
    return number


@dataclass(frozen=True)
class HazardNoise:
    """The hazard at the asset is the index times a mean-one lognormal factor.

    In year `d` the hazard at the asset is
    `h_d = x_d * exp(sigma * Z_d - sigma**2 / 2)`, with `x_d` the index and
    `Z_d` standard normal, one draw per year and asset, so the factor has mean
    1 and `sigma` is the standard deviation of its logarithm. The index does
    not change: the cover pays on `x_d` and the loss model reads `h_d`. For one
    cover, a year can then pay with no loss and a loss can occur with no
    payout, so `false_negative_rate_l<l>` and `false_positive_rate_l<l>`
    respond to `sigma`. The sample's `hazard` holds `h_d`.

    `sigma` is an assumption the caller states, and the module does not
    estimate it. It is not the same physical quantity as the `sigma` of
    `ImpactNoise`.

    **What `sigma` does to the loss level.** The mean is on the hazard, not on
    the loss. The factor keeps the mean hazard but widens the hazard at the
    asset, so more years fall beyond a damage function's threshold or cap. With
    a damage function that is convex, or zero below a threshold that most
    years don't reach, the expected loss rises with `sigma`; with one that is
    capped where most years lie, it can fall. A sweep over `sigma` then moves
    the loss level as well as the gap between index and loss. The samplers
    record both means as `hazard_mean_ratio` and `impact_mean_ratio` in the
    provenance's `details["basis_risk"]`: over the rows as built, the mean with
    the model over the mean on the same draws without it, which is the sample
    at `sigma=0`. The setup text states them next to `sigma`.

    **`hold="hazard_distribution"`** holds the loss level for such a sweep. The
    factors are drawn as above, and `y_d = x_d * exp(sigma * Z_d - sigma**2 /
    2)` ranks the rows, ties by the index and then by row. Row `d` then takes as
    its hazard at the asset the index value of the rank that `y_d` has:
    `h_d = x_(r_d)`, with `x_(k)` the `k`-th smallest index value of the rows
    and `r_d` the rank of `y_d`. The hazard at the asset is a reordering of the
    index's own values, so its distribution over the rows, and that of any
    impact computed from it, is exactly the one at `sigma=0`, whatever the
    damage function, and both mean ratios are 1. Only how the index and the
    hazard at the asset are paired changes with `sigma`. A zero stays zero. On
    a sample of several hazards each component keeps its own distribution and
    its own loss; the combined loss keeps its mean when the rule adds the
    impacts, and not in general.

    This form assigns values across the rows of one call, as an
    `EmpiricalCopula` does, so a block of its rows would hold values of the
    other block's: `holdout` and `year_block_kfold` refuse to split its sample,
    and an out-of-sample evaluation between two samples cut from one such
    sample raises `LeakageError`. Build the design and the evaluation samples
    with separate calls (fresh draws from a curve, or per-year samples of
    separate years); each then holds its own rows' distribution and reads no
    value of the other's. The default, `hold="hazard_mean"`, has no such step,
    as each row's factor is its own. Both forms draw the same `Z_d`, so the two
    forms of one `sigma` from one generator state read one set of draws, and
    `to_document()` records `hold` only when it is not the default, so the
    default's documents, rows and hashes don't change.

    The model suits a hazard on a ratio scale, where zero means none (rain,
    wind, depth). A negative hazard is refused, because a factor would move it
    the wrong way, and a zero stays zero. The module cannot tell that a
    temperature in degrees Celsius is not on such a scale.

    The factors are drawn with one call `rng.normal(-sigma**2 / 2, sigma,
    size=n)`, at every `sigma`, so a sweep over `sigma` from one generator
    state reads one set of `Z_d`. `sigma=0` is allowed: the factor is exactly 1,
    the rows equal those of `NoBasisRisk()` in either form, and the samplers
    warn with `ComonotoneIndexWarning` as they do for it.

    The samplers refuse it next to a separate index series. The index and the
    hazard then already differ and the hazard at the asset is data, so a
    factor would contradict it.

    Raises `SampleError` unless `sigma` is finite and not negative and `hold` is
    `"hazard_mean"` or `"hazard_distribution"`.
    """

    sigma: float
    hold: _Hold = field(default="hazard_mean", kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _not_negative("sigma", self.sigma))
        if not isinstance(self.hold, str) or self.hold not in get_args(_Hold):
            raise SampleError(
                f"hold must be one of {get_args(_Hold)}, got {self.hold!r}"
            )

    @property
    def name(self) -> str:
        """`"hazard_noise"`."""
        return "hazard_noise"

    def to_document(self) -> dict[str, Any]:
        """Return the kind, `sigma`, and `hold` when it is not the default."""
        document: dict[str, Any] = {"kind": self.name, "sigma": self.sigma}
        if self.hold != "hazard_mean":
            document["hold"] = self.hold
        return document

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the index and the impact unchanged; draws nothing."""
        del hazard, rng
        return index, impact

    def apply_hazard(
        self,
        index: FloatArray,
        hazard: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the index unchanged and the hazard at the asset.

        Under `hold="hazard_distribution"` the hazard at the asset is the index's
        values reordered by the rank of the index times the factor, along the
        first axis. Raises `SampleError` without a `numpy.random.Generator`, at
        every `sigma`, for a negative value, or when the index and the hazard
        differ: they are one value here, and the factor multiplies it.
        """
        if not isinstance(rng, np.random.Generator):
            raise SampleError(
                "HazardNoise draws one factor per year and asset; pass rng= a "
                "numpy.random.Generator"
            )
        if not np.array_equal(index, hazard):
            raise SampleError(
                "HazardNoise multiplies an index that is the hazard at the asset, "
                "but the index and the hazard differ"
            )
        if (index < 0).any():
            raise SampleError(
                "HazardNoise needs a hazard on a ratio scale, where zero means "
                "none, but a value is negative: a factor would move it the wrong "
                "way"
            )
        sigma = self.sigma
        log_factor = rng.normal(loc=-0.5 * sigma * sigma, scale=sigma, size=index.shape)
        values = np.asarray(index, dtype=np.float64)
        at_asset = values * np.exp(log_factor)
        if self.hold == "hazard_mean":
            return index, at_asset
        return index, _reordered_by_rank(values, at_asset)


def _reordered_by_rank(values: FloatArray, ranked: FloatArray) -> FloatArray:
    """`values` reordered along the first axis so that their ranks are `ranked`'s.

    Along each column, the row with the `k`-th smallest `ranked` gets the `k`-th
    smallest of `values`; ties in `ranked` go by `values`, then by row, so a
    zero of `values` keeps a zero wherever `ranked` is zero too.
    """
    rows = values.shape[0] if values.ndim else 1
    flat_values = values.reshape(rows, -1)
    flat_ranked = ranked.reshape(flat_values.shape)
    result = np.empty_like(flat_values)
    for column in range(flat_values.shape[1]):
        own = flat_values[:, column]
        order = np.lexsort((own, flat_ranked[:, column]))
        result[order, column] = np.sort(own)
    return result.reshape(values.shape)


@dataclass(frozen=True)
class ImpactNoise:
    """Mean-one lognormal noise on each year's impact.

    Each draw's impact is multiplied by `exp(sigma * Z - sigma**2 / 2)` with
    `Z` standard normal, one draw per year, so the factor has mean 1 and
    `sigma` is the standard deviation of its logarithm. With `upper`, the
    noisy impact is then capped at `upper` (1 for a damage ratio), which
    lowers its mean a little. The index is unchanged.

    The factor multiplies an impact the index has already fixed, and a year
    with zero impact keeps zero impact. So it can't change whether a year pays
    or whether a loss occurs, and `false_negative_rate_l0` and
    `false_positive_rate_l0` don't move with sigma. With a loss threshold
    above zero they can, because the factor can carry a loss across the
    threshold.

    The factors are drawn with one call `rng.normal(-sigma**2 / 2, sigma,
    size=n)`, the noise layer of the thesis code (homoscedastic) with one
    draw per year.

    Raises `SampleError` unless `sigma` and `upper` are finite and positive.
    """

    sigma: float
    upper: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive("sigma", self.sigma))
        if self.upper is not None:
            object.__setattr__(self, "upper", _positive("upper", self.upper))

    @property
    def name(self) -> str:
        """`"impact_noise"`."""
        return "impact_noise"

    def to_document(self) -> dict[str, Any]:
        """Return the kind, `sigma` and `upper`."""
        return {"kind": self.name, "sigma": self.sigma, "upper": self.upper}

    def apply(
        self,
        index: FloatArray,
        hazard: FloatArray,
        impact: FloatArray,
        rng: np.random.Generator | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the index unchanged and the noisy, capped impact.

        Raises `SampleError` without a `numpy.random.Generator`.
        """
        del hazard
        if not isinstance(rng, np.random.Generator):
            raise SampleError(
                "ImpactNoise draws one factor per year; pass rng= a "
                "numpy.random.Generator"
            )
        sigma = self.sigma
        log_factor = rng.normal(
            loc=-0.5 * sigma * sigma, scale=sigma, size=impact.shape
        )
        noisy = np.asarray(impact, dtype=np.float64) * np.exp(log_factor)
        if self.upper is not None:
            noisy = np.minimum(noisy, self.upper)
        return index, noisy
