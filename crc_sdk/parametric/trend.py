"""A stated treatment of the climate trend in per-year values.

The years of a per-year record carry whatever trend the climate had over them,
so using them as they are is a choice like any other, and a sample states it.
`trend=` on `sample_annual` and on `ParametricPlan.for_annual_extremes` takes
one of three options:

- `AsRecorded()`, the default, uses the years as they are.
- `RecentYears(n)` keeps the last `n` entries of the series' years. A gap
  before them does not matter, and a gap among them is a missing year. No value
  changes.
- `LinearTrend(reference_year=None, *, min_years=10, fit_years=None)` removes
  an additive linear trend in the mean. On the fit years it fits
  `x_t = a + b (t - t0) + e_t` by ordinary least squares, with `t` the calendar
  year and `t0` the reference year, and replaces every year's value by
  `x_t - b (t - t0)`. So `a`, the recorded intercept, is the fitted value at
  the reference year, and the adjusted fit years have mean `a`. The spread
  around the line is not touched, so a change in the variance or the tail stays
  in the data.

Each series has its own fit: the hazard, and a separate index series, on the
same years with the same reference year. By default the reference year is the
last entry of the series' years, whether or not it has a value, so it does not
change with the years a split holds out. Nothing is clipped: a value that
leaves the fit non-finite, or was non-negative as recorded and is negative
after the adjustment, raises `SampleError`, and one that is negative as
recorded (a temperature) may stay negative.

The steps come in a fixed order: the series' years; the window of
`RecentYears`; the missing years; the fit and the adjustment; the impact on the
adjusted hazard; the basis-risk model. A trend draws no random numbers.

Each option's `to_document()` states what was asked for. What a sample used is
in its provenance's `trend`, with the fit years, one slope and intercept per
series, and the reference year. Which option suits a hazard is the caller's
choice, and the module says nothing about it. This module needs numpy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.errors import SampleError

__all__ = ["AsRecorded", "LinearTrend", "RecentYears", "Trend"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_SMALLEST_MIN_YEARS = 3
"""Fewer fit years than this would leave a straight line no residual."""


class _Series(NamedTuple):
    """One series a trend is applied to: its role, its unit, its kept values."""

    role: str
    unit: str
    values: FloatArray


def _year_list(years: Any) -> str:
    """Up to ten years, for an error message."""
    shown = [int(year) for year in years]
    return ", ".join(str(year) for year in shown[:10]) + (
        ", ..." if len(shown) > 10 else ""
    )


def _whole_number(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SampleError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _stated_years(value: Any) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.ndim != 1 or array.size == 0:
        raise SampleError(
            f"fit_years must be a non-empty sequence of integer years, got {value!r}"
        )
    unique = np.unique(array.astype(np.int64))
    if unique.size != array.size:
        raise SampleError("fit_years must not repeat")
    return tuple(int(year) for year in unique)


def _least_squares(elapsed: FloatArray, values: FloatArray) -> tuple[float, float]:
    """The ordinary least squares slope, and the fitted value at `elapsed = 0`.

    With exact sums, so the result does not depend on the order of the years.
    Returns NaNs for values so large that the sums overflow.
    """
    count = elapsed.shape[0]
    try:
        mean_elapsed = math.fsum(elapsed.tolist()) / count
        mean_value = math.fsum(values.tolist()) / count
        spread = elapsed - mean_elapsed
        with np.errstate(over="ignore", invalid="ignore"):
            products = spread * (values - mean_value)
        slope = math.fsum(products.tolist()) / math.fsum((spread * spread).tolist())
    except (OverflowError, ValueError):
        return math.nan, math.nan
    return slope + 0.0, mean_value - slope * mean_elapsed + 0.0


def _check_adjusted(
    series: _Series, years: IntArray, adjusted: FloatArray, slope: float
) -> None:
    """Raise `SampleError` unless the adjusted values are valid; none is clipped."""
    invalid = ~np.isfinite(adjusted)
    if invalid.any() or not math.isfinite(slope):
        raise SampleError(
            f"LinearTrend gives the {series.role} series non-finite values "
            f"(slope {slope!r}, years {_year_list(years[invalid])}); nothing is clipped"
        )
    negative = (series.values >= 0.0) & (adjusted < 0.0)
    if negative.any():
        raise SampleError(
            f"LinearTrend makes {int(negative.sum())} non-negative {series.role} "
            f"values negative (years {_year_list(years[negative])}); nothing is "
            "clipped, so state another reference_year or trend"
        )


@dataclass(frozen=True)
class AsRecorded:
    """Use the years as they are; no trend is removed."""

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "as_recorded"}`."""
        return {"kind": "as_recorded"}

    def _first_row(self, count: int) -> int:
        del count
        return 0

    def _apply(
        self, years: IntArray, kept: IntArray, series: Sequence[_Series]
    ) -> tuple[list[FloatArray], dict[str, Any]]:
        del years, kept
        return [each.values for each in series], self.to_document()


@dataclass(frozen=True)
class RecentYears:
    """Keep the last `n` entries of the series' years; no value changes.

    The window is taken after a stated `years=` and before the missing years
    are looked at, so a gap before it does not matter and a gap inside it is a
    missing year. A split divides the window, so the held-out block is the
    latest years inside it.

    Raises `SampleError` unless `n` is a positive integer, and at sampling if
    `n` is above the length of the series.
    """

    n: int

    def __post_init__(self) -> None:
        n = _whole_number("n", self.n)
        if n < 1:
            raise SampleError(f"n must be a positive integer, got {self.n!r}")
        object.__setattr__(self, "n", n)

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "recent_years", "n": n}`."""
        return {"kind": "recent_years", "n": self.n}

    def _first_row(self, count: int) -> int:
        if self.n > count:
            raise SampleError(
                f"RecentYears({self.n}) needs {self.n} years, and the series has "
                f"{count}"
            )
        return count - self.n

    def _apply(
        self, years: IntArray, kept: IntArray, series: Sequence[_Series]
    ) -> tuple[list[FloatArray], dict[str, Any]]:
        del kept
        record = {**self.to_document(), "years": [int(years[0]), int(years[-1])]}
        return [each.values for each in series], record


@dataclass(frozen=True)
class LinearTrend:
    """Remove an additive linear trend in the mean, fitted by least squares.

    On the fit years, `x_t = a + b (t - t0) + e_t` is fitted by ordinary least
    squares, and every kept year's value becomes `x_t - b (t - t0)`.

    - `reference_year` is `t0`. By default, the last entry of the series'
      years, whether or not it has a value; a stated year may lie outside the
      record.
    - `min_years` is the least number of fit years. Fewer raise `SampleError`.
      The default of 10 is a floor against a slope that is nearly all noise,
      not a statement that ten years are enough.
    - `fit_years` states the fit years; by default they are every kept year. A
      year the series does not hold raises `SampleError`. A year dropped as
      missing is left out, so the record lists the years actually used. On a
      plan it raises `ValueError`, because the plan sets it per fold.

    Raises `SampleError` for a `reference_year` that is not an integer, a
    `min_years` below 3, which would leave no residual, and `fit_years` that
    are empty, repeat or are not integers.
    """

    reference_year: int | None = None
    min_years: int = field(default=10, kw_only=True)
    fit_years: tuple[int, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.reference_year is not None:
            object.__setattr__(
                self,
                "reference_year",
                _whole_number("reference_year", self.reference_year),
            )
        min_years = _whole_number("min_years", self.min_years)
        if min_years < _SMALLEST_MIN_YEARS:
            raise SampleError(
                f"min_years must be at least {_SMALLEST_MIN_YEARS}, which leaves a "
                f"straight line a residual; got {self.min_years!r}"
            )
        object.__setattr__(self, "min_years", min_years)
        if self.fit_years is not None:
            object.__setattr__(self, "fit_years", _stated_years(self.fit_years))

    def to_document(self) -> dict[str, Any]:
        """Return the kind, the method and the terms stated."""
        return {
            "kind": "linear_trend",
            "method": "ols_additive",
            "reference_year": self.reference_year,
            "min_years": self.min_years,
            "fit_years": None if self.fit_years is None else list(self.fit_years),
        }

    def _first_row(self, count: int) -> int:
        del count
        return 0

    def _years_to_fit(self, years: IntArray, kept: IntArray) -> IntArray:
        """The fit years: those stated and kept, or every kept year."""
        if self.fit_years is None:
            chosen = kept
        else:
            stated = np.array(self.fit_years, dtype=np.int64)
            unknown = np.setdiff1d(stated, years)
            if unknown.size:
                raise SampleError(
                    f"fit_years names years the series does not hold: "
                    f"{_year_list(unknown)}"
                )
            chosen = np.intersect1d(stated, kept)
        if chosen.shape[0] < self.min_years:
            raise SampleError(
                f"LinearTrend needs at least {self.min_years} fit years (min_years), "
                f"and has {chosen.shape[0]}"
            )
        return chosen

    def _apply(
        self, years: IntArray, kept: IntArray, series: Sequence[_Series]
    ) -> tuple[list[FloatArray], dict[str, Any]]:
        reference = (
            int(years[-1]) if self.reference_year is None else self.reference_year
        )
        fit_years = self._years_to_fit(years, kept)
        elapsed = (kept - reference).astype(np.float64)
        on_fit = np.isin(kept, fit_years)
        adjusted: list[FloatArray] = []
        fits: list[dict[str, Any]] = []
        for each in series:
            slope, intercept = _least_squares(elapsed[on_fit], each.values[on_fit])
            with np.errstate(over="ignore", invalid="ignore"):
                values = each.values - slope * elapsed
            _check_adjusted(each, kept, values, slope)
            adjusted.append(values)
            fits.append(
                {
                    "role": each.role,
                    "slope": slope,
                    "intercept": intercept,
                    "unit": each.unit,
                }
            )
        return adjusted, {
            "kind": "linear_trend",
            "method": "ols_additive",
            "reference_year": reference,
            "min_years": self.min_years,
            "fit_years": [int(year) for year in fit_years],
            "fits": fits,
        }


Trend = AsRecorded | RecentYears | LinearTrend
"""The options `trend=` takes."""


def _require_trend(trend: Any) -> Trend:
    if not isinstance(trend, (AsRecorded, RecentYears, LinearTrend)):
        raise TypeError(
            "trend must be AsRecorded(), RecentYears(n) or LinearTrend(...), got "
            f"{type(trend).__name__}"
        )
    return trend
