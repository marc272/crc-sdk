"""Stated dependence between the columns of a sample drawn from fitted curves.

A fitted curve describes one place, and says nothing about how two places move
together. Curve draws are made column by column, each from its own generator,
so putting them side by side as they are would impose independence without
saying so. A dependence model says how the columns are put side by side. It is
an assumption the caller states: there is no default, and the sample records
it.

Every model only reorders. It takes the uniforms each column's draws came from
and says which draw goes in which row: a row's values stay together and no
value changes, so each column holds the same draws under every model, and the
models use no random numbers of their own.

- `Independent()` keeps every column in draw order. The columns are independent
  because each has its own generator.
- `Comonotonic()` keeps the first column in draw order and moves every other
  column's draw of rank r to the row of the first column's draw of rank r.
- `EmpiricalCopula(reference)` puts each column's draws in the ranks of a stated
  per-year sample, in blocks of draws per reference year.

`crc_sdk.parametric.sample_curves` applies them. Ranks are those of the
uniforms, so a hurdle curve's atom adds no ties. This module needs numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.errors import SampleError
from crc_sdk.parametric.sample import (
    _PROBABILITY_TOLERANCE,
    OutcomeSample,
    _sample_years,
)

__all__ = ["Comonotonic", "Dependence", "EmpiricalCopula", "Independent"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _assumption(between: str, what: str) -> str:
    return f"The dependence between the {between} is an assumption: {what}"


@dataclass(frozen=True)
class Independent:
    """Every column stays in draw order.

    The columns are independent because each has its own generator; putting
    them side by side in draw order is what this model states.
    """

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "independent"}`."""
        return {"kind": "independent"}

    def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
        """Return what a sample of `n_draws` rows records: `to_document()`."""
        del n_draws, columns
        return self.to_document()

    def statement(self, n_draws: int, between: str) -> str:
        """Return the sentence that states this model next to a number."""
        del n_draws
        return _assumption(
            between, "they are independent, each drawn from its own generator."
        )

    def reorder(self, uniforms: FloatArray, columns: Sequence[str]) -> IntArray:
        """Return the draw of each row and column: `(n, d)`, all in draw order."""
        del columns
        n, d = uniforms.shape
        return np.tile(np.arange(n, dtype=np.int64)[:, np.newaxis], (1, d))


@dataclass(frozen=True)
class Comonotonic:
    """Every column's draw of rank r shares a row with the first column's.

    The first column stays in draw order. Ranks are those of the uniforms, so
    the rank correlation of the uniforms between any two columns is 1, whatever
    atoms or flat stretches the curves have.
    """

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "comonotonic"}`."""
        return {"kind": "comonotonic"}

    def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
        """Return what a sample of `n_draws` rows records: `to_document()`."""
        del n_draws, columns
        return self.to_document()

    def statement(self, n_draws: int, between: str) -> str:
        """Return the sentence that states this model next to a number."""
        del n_draws
        return _assumption(
            between,
            "they are comonotonic, so draws of equal rank share a row.",
        )

    def reorder(self, uniforms: FloatArray, columns: Sequence[str]) -> IntArray:
        """Return the draw of each row and column: `(n, d)`, ranks aligned."""
        del columns
        first = np.argsort(uniforms[:, 0], kind="stable")
        order = np.empty(uniforms.shape, dtype=np.int64)
        for j in range(uniforms.shape[1]):
            order[first, j] = np.argsort(uniforms[:, j], kind="stable")
        return order


def _rank_blocks(uniforms: FloatArray, reference: FloatArray, k: int) -> IntArray:
    """One column's draws, in the rows of `reference`'s years, `k` to a year.

    The draws, sorted by their uniforms, make `reference.size` consecutive
    blocks of `k`, and a year of rank r takes block r, in draw order. Tied
    years pool their blocks and are dealt them in draw order, `k` each, in
    year order. The stable sort keeps tied years in year order, so a year
    without a tie is the same step with a pool of one block.
    """
    draws = np.argsort(uniforms, kind="stable")
    years = np.argsort(reference, kind="stable")
    ranked = reference[years]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    stops = np.r_[starts[1:], ranked.size]
    order = np.empty(uniforms.size, dtype=np.int64)
    for start, stop in zip(starts, stops, strict=True):
        rows = (years[start:stop, np.newaxis] * k + np.arange(k)).ravel()
        order[rows] = np.sort(draws[start * k : stop * k])
    return order


@dataclass(frozen=True)
class EmpiricalCopula:
    """Each column's draws in the ranks of a stated per-year sample.

    `reference` is an `OutcomeSample` of `m` observed or modelled years with
    equal probabilities, holding two columns or more, with the sample's own
    ids. A Monte Carlo sample has no dependence of its own to lend and raises.
    Ranks are read from its `hazard`, or from its `index` when it has no
    separate hazard. Only the ranks are used, so its unit may differ from the
    curves'. A larger reference value takes a larger draw; negate a reference
    that moves the other way.

    `n_draws` must be a whole multiple `k·m` of the reference's years. Each
    column's draws, sorted by their uniforms, make `m` consecutive blocks of
    `k`. A reference year of rank r in that column takes block r, in the order
    the draws were made. The rows are in reference-year order: the first `k`
    rows are the first reference year, and so on. With `k = 1` the sample's
    ranks are the reference's. Years with equal values in a column pool their
    blocks, which are dealt in draw order, `k` each, in year order, so a tie
    adds no dependence between columns.

    Inside a block the reference says nothing, so the columns are independent
    there: dependence finer than one year in `m` is assumed, not observed.
    Because the rows come in reference-year blocks, `holdout` and
    `year_block_kfold` raise `SampleError` on such a sample. Its out-of-sample
    test is a fresh sample with new generators and the same reference, which
    is part of the stated model.

    A reference may be a pool of model-years: `m` is then its number of rows,
    each model-year taking a block of its own, and the first and last year in
    its record are the pool's calendar years.

    Raises `SampleError` for a reference it cannot use, and `TypeError` for
    anything but an `OutcomeSample`.
    """

    reference: OutcomeSample

    def __post_init__(self) -> None:
        reference = self.reference
        if not isinstance(reference, OutcomeSample):
            raise TypeError(
                "reference must be an OutcomeSample of observed or modelled years, "
                f"got {type(reference).__name__}"
            )
        kind = reference.provenance.source_kind
        if kind == "monte_carlo":
            raise SampleError(
                "a Monte Carlo sample cannot be the reference: it has no dependence "
                "of its own to lend; use a sample of observed or modelled years"
            )
        if len(reference.asset_ids) < 2:
            raise SampleError(
                "the reference must hold two columns or more to have a dependence "
                f"between them, got {list(reference.asset_ids)}"
            )
        if float(np.ptp(reference.probability)) > _PROBABILITY_TOLERANCE:
            raise SampleError(
                "the reference's years must have equal probabilities: only then "
                "does each rank stand for the same share of the draws"
            )

    def to_document(self) -> dict[str, Any]:
        """Return the kind, and the reference's hash, sources, licences and years."""
        reference = self.reference
        document = reference.provenance.to_document()
        years = _sample_years(reference)
        return {
            "kind": "empirical_copula",
            "content_hash": reference.content_hash,
            "sources": document["sources"],
            "licences": document["licences"],
            "first_year": int(years[0]),
            "last_year": int(years[-1]),
            "m": len(reference),
            **_pooled(reference),
        }

    def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
        """Return `to_document()` and `k`, the draws per reference year.

        Raises `SampleError` unless `n_draws` is a whole multiple of the
        reference's years, naming the two nearest valid counts, and unless
        `columns` are the reference's columns.
        """
        m = len(self.reference)
        if n_draws % m:
            lower = max(n_draws // m, 1) * m
            raise SampleError(
                f"EmpiricalCopula needs n_draws to be a whole multiple of the "
                f"reference's {m} {_rows_are(self.reference)}s, got {n_draws}; the "
                "nearest valid counts "
                f"are {lower} and {lower + m}"
            )
        held = sorted(self.reference.asset_ids)
        if sorted(columns) != held:
            raise SampleError(
                f"the reference holds {held}, but the sample holds {sorted(columns)}: "
                "they must be the same"
            )
        return {**self.to_document(), "k": n_draws // m}

    def statement(self, n_draws: int, between: str) -> str:
        """Return the sentences that state this model next to a number."""
        reference = self.reference
        m = len(reference)
        years = _sample_years(reference)
        row = _rows_are(reference)
        return _assumption(
            between,
            f"their draws are reordered to the ranks of the stated reference, "
            f"{m} {row}s from {years[0]} to {years[-1]} "
            f"({reference.content_hash[:19]}), {n_draws // m} draws per {row}. "
            f"Inside a reference {row}'s block the {between} are independent: "
            f"dependence finer than one {row} in {m} is assumed, not observed. The "
            "reference is part of this stated model, so a design and its fresh "
            "draws share it.",
        )

    def reorder(self, uniforms: FloatArray, columns: Sequence[str]) -> IntArray:
        """Return the draw of each row and column: `(n, d)`, by reference year."""
        reference = self.reference
        ranked = reference.index if reference.hazard is None else reference.hazard
        k = uniforms.shape[0] // len(reference)
        order = np.empty(uniforms.shape, dtype=np.int64)
        for j, column in enumerate(columns):
            values = ranked[:, reference.asset_ids.index(column)]
            order[:, j] = _rank_blocks(uniforms[:, j], values, k)
        return order


def _rows_are(reference: OutcomeSample) -> str:
    """What one row of a reference is."""
    return "year" if reference.provenance.pool is None else "model-year"


def _pooled(reference: OutcomeSample) -> dict[str, Any]:
    """What a pooled reference holds, in its record; nothing for any other."""
    pool = reference.provenance.pool
    if pool is None:
        return {}
    return {
        "pool": {
            "runs": len(pool["runs"]),
            "years": int(_sample_years(reference).shape[0]),
        }
    }


Dependence = Independent | Comonotonic | EmpiricalCopula
"""A stated dependence model; `sample_curves` requires one and has no default."""
