"""Stated dependence between the columns of a sample drawn from fitted curves.

A fitted curve describes one hazard at one place, and says nothing about how two
places or two hazards move together. Curve draws are made column by column, each
from its own generator, so putting them side by side as they are would impose
independence without saying so. A dependence model says how the columns are put
side by side. It is an assumption the caller states: there is no default, and
the sample records it.

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
- `GaussianCopula(correlation)` moves each column's draw of rank r to the row
  where correlated normal scores of the columns' own uniforms have rank r.
- A model of the caller's implements `DependenceModel`.

Every model, built in or not, passes one gate before its sample is returned.
Its document and record are plain JSON with a `kind`, which a user's model cannot
take from a built-in. Its sentence is a non-empty string, written after the
opening that calls the dependence an assumption. Its reorder holds each column's
draws, once each, and gives the same rows when it is called again.

`crc_sdk.parametric.sample_curves` applies them across assets and
`crc_sdk.parametric.sample_curve_hazards` across the hazards of one asset, with
the columns named by asset or by component. Ranks are those of the uniforms, so
a hurdle curve's atom adds no ties, and the models order the draws by value and
not by severity. This module needs numpy and the standard library only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from crc_sdk.parametric.errors import SampleError
from crc_sdk.parametric.sample import (
    _PROBABILITY_TOLERANCE,
    OutcomeSample,
    _sample_years,
)

__all__ = [
    "Comonotonic",
    "Dependence",
    "DependenceModel",
    "EmpiricalCopula",
    "GaussianCopula",
    "Independent",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_CORRELATION_TOLERANCE = 1e-12
_SMALLEST_UNIFORM = 2.0**-53
"""How far from 0 and 1 the normal quantile is read: `1 - 2**-53` is a double."""


@runtime_checkable
class DependenceModel(Protocol):
    """What every stated dependence model provides.

    `to_document()` returns the model's terms as plain JSON types, with a
    `kind` that is not a built-in's. `record(n_draws, columns)` returns what a
    sample records, which is that document with anything this call adds, and
    raises for a call the model cannot serve; it runs before any draw.
    `statement(n_draws, between)` returns the sentence that states the model
    next to a number, `between` being `"assets"` or `"hazards"`. The module
    writes it after its own opening, which calls the dependence an assumption,
    unless it opens so already.

    `reorder(uniforms, columns)` takes each column's uniforms in draw order,
    read-only and `(n, d)`, and says which draw each row takes: integers of
    shape `(n, d)`, each column a permutation of 0 to `n - 1`. Two calls on the
    same uniforms must agree, so a model may hold its parameters but no
    generator its document doesn't describe.
    """

    def to_document(self) -> dict[str, Any]: ...

    def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]: ...

    def statement(self, n_draws: int, between: str) -> str: ...

    def reorder(self, uniforms: FloatArray, columns: Sequence[str]) -> IntArray: ...


def _opening(between: str) -> str:
    return f"The dependence between the {between} is an assumption:"


def _assumption(between: str, what: str) -> str:
    return f"{_opening(between)} {what}"


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
    ids: for columns of assets, those assets; for columns of hazards, one
    asset's `index_components` of those names. A Monte Carlo sample has no
    dependence of its own to lend and raises. Ranks are read from a column's
    `hazard`, or from its index when it has no separate hazard. Only the ranks
    are used, so its unit may differ from the curves'. A larger reference value
    takes a larger draw; negate a reference that moves the other way.

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
    `year_block_kfold` raise `SampleError` on such a sample, and two blocks cut
    from it with `select` raise `LeakageError` at an out-of-sample evaluation.
    Each column's draws were dealt to the reference years by their rank over
    every row, so a block of whole reference years holds the rank blocks the
    other lacks, and a reference year with rows in both blocks puts draws of one
    rank block in each. Its out-of-sample test is a fresh sample with new
    generators and the same reference, which is part of the stated model.

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
        columns = _reference_columns(reference)
        if len(columns) < 2:
            raise SampleError(
                "the reference must hold two columns or more to have a dependence "
                f"between them, got {list(columns)}"
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
        held = sorted(_reference_columns(self.reference))
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
        k = uniforms.shape[0] // len(reference)
        order = np.empty(uniforms.shape, dtype=np.int64)
        for j, column in enumerate(columns):
            values = _reference_values(reference, column)
            order[:, j] = _rank_blocks(uniforms[:, j], values, k)
        return order


def _reference_columns(reference: OutcomeSample) -> tuple[str, ...]:
    """A reference's columns: its components, or else its assets."""
    components = reference.index_components
    return reference.asset_ids if components is None else tuple(components)


def _reference_values(reference: OutcomeSample, column: str) -> FloatArray:
    """The values a reference's column is ranked on: its hazard, or else its index."""
    components = reference.index_components
    if components is None:
        ranked = reference.index if reference.hazard is None else reference.hazard
        values: FloatArray = ranked[:, reference.asset_ids.index(column)]
        return values
    component = components[column]
    held = component.values if component.hazard is None else component.hazard
    ranked_values: FloatArray = held[:, 0]
    return ranked_values


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


def _correlation(value: Any) -> float | tuple[tuple[float, ...], ...]:
    """`value` as a number in [-1, 1], or a positive definite correlation matrix."""
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not -1.0 <= number <= 1.0:
            raise SampleError(f"correlation must lie in [-1, 1], got {value!r}")
        return number
    try:
        matrix = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError(
            "correlation must be a number in [-1, 1] for two columns or a "
            f"correlation matrix, got {value!r}"
        ) from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise SampleError(
            "a correlation matrix must be square, with two columns or more; "
            f"got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all() or (
        np.abs(matrix - matrix.T).max() > _CORRELATION_TOLERANCE
    ):
        raise SampleError("a correlation matrix must be finite and symmetric")
    if np.abs(np.diag(matrix) - 1.0).max() > _CORRELATION_TOLERANCE:
        raise SampleError("a correlation matrix must have ones on its diagonal")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise SampleError(
            "a correlation matrix must be positive definite; a number in [-1, 1] "
            "states two columns, the perfectly dependent ones included"
        ) from error
    return tuple(tuple(row) for row in matrix.tolist())


def _normal_scores(uniforms: FloatArray, draws: IntArray) -> FloatArray:
    """The normal scores of one column's uniforms; `draws` sorts the uniforms.

    The quantile is the standard library's, on uniforms clipped to the open
    interval. It increases only up to rounding, so each score is kept above the
    last: the scores strictly increase with the draws' ranks, and their ranks
    are exactly those of the uniforms.
    """
    quantile = NormalDist().inv_cdf
    clipped = np.clip(uniforms[draws], _SMALLEST_UNIFORM, 1.0 - _SMALLEST_UNIFORM)
    ascending: list[float] = []
    score = -math.inf
    for uniform in clipped.tolist():
        score = max(quantile(uniform), math.nextafter(score, math.inf))
        ascending.append(score)
    scores = np.empty(uniforms.shape, dtype=np.float64)
    scores[draws] = ascending
    return scores


@dataclass(frozen=True)
class GaussianCopula:
    """Each column's draws in the ranks of correlated normal scores.

    `correlation` is a number in [-1, 1] for two columns, or a positive
    definite matrix for two columns or more, in column order. With `z_j` the
    normal scores of column j's own uniforms and `y = z L^T`, `L` being the
    Cholesky factor of the correlation matrix, column j's draw of rank r goes
    to the row where `y_j` has rank r. For two columns, `y_1 = z_1` and
    `y_2 = rho z_1 + sqrt(1 - rho^2) z_2`, so the first column stays in draw
    order.

    The model is a function of the uniforms and uses no random numbers beyond
    the draws. `rho = 0` is `Independent()` and `rho = 1` is `Comonotonic()`,
    row for row. `correlation` is the copula's parameter, and the rows' Spearman
    rank correlation is about `6 / pi * arcsin(rho / 2)`.

    Raises `SampleError` for any other `correlation`.
    """

    correlation: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlation", _correlation(self.correlation))

    def to_document(self) -> dict[str, Any]:
        """Return the kind and the correlation, in column order."""
        correlation = self.correlation
        return {
            "kind": "gaussian_copula",
            "correlation": (
                correlation
                if isinstance(correlation, float)
                else np.asarray(correlation).tolist()
            ),
        }

    def _factor(self, columns: Sequence[str]) -> FloatArray:
        """The Cholesky factor of the correlation of `columns`, lower triangular."""
        correlation = self.correlation
        count = len(columns)
        if isinstance(correlation, float):
            if count != 2:
                raise SampleError(
                    f"a number states the correlation of two columns, but the sample "
                    f"holds {count}: {list(columns)}; state a matrix"
                )
            remainder = math.sqrt(1.0 - correlation * correlation)
            return np.array([[1.0, 0.0], [correlation, remainder]], dtype=np.float64)
        matrix = np.array(correlation, dtype=np.float64)
        if matrix.shape[0] != count:
            raise SampleError(
                f"the correlation matrix has {matrix.shape[0]} columns, but the "
                f"sample holds {count}: {list(columns)}"
            )
        factor: FloatArray = np.linalg.cholesky(matrix)
        return factor

    def record(self, n_draws: int, columns: Sequence[str]) -> dict[str, Any]:
        """Return what a sample of `n_draws` rows records: `to_document()`.

        Raises `SampleError` unless the correlation is for these columns.
        """
        del n_draws
        self._factor(columns)
        return self.to_document()

    def statement(self, n_draws: int, between: str) -> str:
        """Return the sentence that states this model next to a number."""
        del n_draws
        correlation = self.correlation
        if isinstance(correlation, float):
            spearman = 6.0 / math.pi * math.asin(correlation / 2.0)
            stated = (
                f"correlation {correlation!r}, a Spearman rank correlation of about "
                f"{spearman:.3f} between the rows"
            )
        else:
            stated = f"the correlation matrix {self.to_document()['correlation']}"
            stated += ", in column order"
        return _assumption(
            between,
            f"their draws are reordered under a Gaussian copula with {stated}. The "
            "correlation is a stated parameter of the copula.",
        )

    def reorder(self, uniforms: FloatArray, columns: Sequence[str]) -> IntArray:
        """Return the draw of each row and column: `(n, d)`, by the mixed ranks."""
        factor = self._factor(columns)
        draws = np.argsort(uniforms, axis=0, kind="stable")
        scores = np.column_stack(
            [
                _normal_scores(uniforms[:, j], draws[:, j])
                for j in range(uniforms.shape[1])
            ]
        )
        mixed = scores @ factor.T
        order = np.empty(uniforms.shape, dtype=np.int64)
        for j in range(uniforms.shape[1]):
            order[np.argsort(mixed[:, j], kind="stable"), j] = draws[:, j]
        return order


Dependence = Independent | Comonotonic | EmpiricalCopula | GaussianCopula
"""A built-in dependence model; none is a default."""

_BUILT_IN_MODELS = (Independent, Comonotonic, EmpiricalCopula, GaussianCopula)
_BUILT_IN_KINDS = frozenset(
    {"comonotonic", "data", "empirical_copula", "gaussian_copula", "independent"}
)
"""The kinds a user's model cannot take, so that none is recorded as one of these."""


def _require_model(model: Any, between: str) -> None:
    """Raise `TypeError` unless `model` is a model: every model has a document."""
    if isinstance(model, type) or not isinstance(model, DependenceModel):
        shown = model.__name__ if isinstance(model, type) else type(model).__name__
        raise TypeError(
            "dependence must be stated, as Independent(), Comonotonic() or another "
            "DependenceModel, such as EmpiricalCopula(reference) or "
            f"GaussianCopula(correlation): curves say nothing about how {between} "
            f"move together, so there is no default; got {shown}"
        )


def _document_kind(model: Any, document: Any, call: str) -> str:
    """Return the `kind` of a document of `model`, after checking the document."""
    if not isinstance(document, Mapping):
        raise SampleError(
            f"a dependence model's {call} must return a mapping, got {document!r}"
        )
    try:
        json.dumps(document, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise SampleError(
            f"a dependence model's {call} must hold plain JSON values: {error}"
        ) from error
    kind = document.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise SampleError(
            f"a dependence model's {call} must name its kind, got {kind!r}"
        )
    if kind in _BUILT_IN_KINDS and type(model) not in _BUILT_IN_MODELS:
        raise SampleError(
            f"a user's dependence model cannot take the kind of a built-in, {kind!r}: "
            "name another, so that the record says which model made the rows"
        )
    return kind


def _model_record(
    model: Any, n_draws: int, columns: Sequence[str], between: str
) -> tuple[dict[str, Any], str]:
    """Check what a model states before any draw.

    Returns what the sample records and the sentence for its notes. Raises
    `SampleError` for a document, a record or a sentence that doesn't pass the
    gate, and for an `EmpiricalCopula` whose reference holds the other axis.
    """
    if isinstance(model, EmpiricalCopula):
        holds = "assets" if model.reference.index_components is None else "hazards"
        if holds != between:
            raise SampleError(
                f"the reference holds several {holds}, but this sample's columns are "
                f"{between}: state a reference that holds those {between}"
            )
    kind = _document_kind(model, model.to_document(), "to_document()")
    record = model.record(n_draws, tuple(columns))
    if _document_kind(model, record, "record()") != kind:
        raise SampleError(
            f"dependence model {kind!r}'s record() must name the kind of its "
            f"to_document(), got {record['kind']!r}"
        )
    statement = model.statement(n_draws, between)
    if not isinstance(statement, str) or not statement.strip():
        raise SampleError(
            f"dependence model {kind!r} must state itself: statement() must return "
            f"a non-empty string, got {statement!r}"
        )
    if not statement.startswith(_opening(between)):
        statement = _assumption(between, statement.strip())
    return dict(record), statement


def _checked_order(
    model: DependenceModel, kind: str, uniforms: FloatArray, columns: Sequence[str]
) -> IntArray:
    """Return `model`'s reorder of `uniforms`, after checking it.

    Raises `SampleError` unless it holds integers of the uniforms' shape, each
    column a permutation of the draws, and a second call gives the same rows.
    """
    held = uniforms.copy()
    held.setflags(write=False)
    names = tuple(columns)
    order = np.asarray(model.reorder(held, names))
    if order.dtype.kind not in "iu":
        raise SampleError(
            f"dependence model {kind!r} must return integers from reorder(), got "
            f"dtype {order.dtype}"
        )
    if order.shape != uniforms.shape:
        raise SampleError(
            f"dependence model {kind!r} must return one draw per row and column from "
            f"reorder(), shape {uniforms.shape}; got {order.shape}"
        )
    n = uniforms.shape[0]
    if not (np.sort(order, axis=0) == np.arange(n)[:, np.newaxis]).all():
        raise SampleError(
            f"dependence model {kind!r} must hold each column's draws once each: "
            f"every column of reorder() must be a permutation of 0 to {n - 1}, so no "
            "draw is dropped, repeated or changed"
        )
    if not np.array_equal(order, np.asarray(model.reorder(held, names))):
        raise SampleError(
            f"dependence model {kind!r} gave different rows on the same uniforms: "
            "reorder() may hold its parameters, but no generator its document "
            "doesn't describe"
        )
    return order.astype(np.int64)
