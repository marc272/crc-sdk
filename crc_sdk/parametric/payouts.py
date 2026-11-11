"""Payout functions: the protocols, the built-in families and their registry.

A payout function maps one year's index value to the year's payout. Every
family pays more as the index moves to the severe side. With
`tail="upper"` larger index values are more severe; with `tail="lower"`
smaller ones are. Thresholds are given in index units in order of increasing
severity: increasing for `"upper"`, decreasing for `"lower"`. Internally a
family works on the severity `s = x` (upper) or `s = -x` (lower).

- `LinearPayout(attachment, exhaustion, limit)`: `limit * clip((s - a) /
  (e - a), 0, 1)`.
- `StepPayout(breakpoints, levels)`: `levels[k]` for `b_k <= s < b_{k+1}`,
  the last level at or beyond the last breakpoint.
- `PiecewiseLinearPayout(knots, levels)`: the points `(c_k, levels[k])`
  joined by straight lines, the last level at or beyond the last knot.
- `BinaryPayout(trigger, amount)`: `amount` for `s >= t`.

Thresholds (attachment, exhaustion, breakpoints, knots, trigger) are in index
units; amounts (limit, levels, amount) in payout units.

- Every family pays nothing below its first threshold.
- A severity equal to a threshold belongs to the more severe side (half-open
  bins).
- Amounts are non-negative. `monotone=True`, the default for steps and
  piecewise linear payouts, also requires non-decreasing levels.

Each parameter is a float (fixed) or `Free` (chosen by a design). Parameter
names are flat strings: `attachment`, `levels[2]`, `knots[1]`, ... A user
family implements `PayoutFunction` (and `LinearInAmounts` for payouts linear
in their amounts) and is made readable from cover documents with
`register_payout`. A user family on several index components has several
`index_names`, a `tail` of one entry per name, and takes an `(n, k)` index.

Two built-in families combine one-index payouts, one per index component, and
go on a cover with a `CompositeIndex` of those names:

- `SumOf(components)`: the sum of the components' payouts. A year pays on
  each component that pays, independently of the others.
- `MaxOf(components, scale=1.0)`: `scale` times the largest of the components'
  payouts. One year pays one amount.

A component's parameter is named `<component>.<parameter>`, such as
`rx1day.levels[2]`, and a `MaxOf`'s own is `scale`. Component payouts are read
through the registry, so a registered user family can be a component.

`StepGrid(breakpoints, levels)` is one table over exactly two components: a
year pays the level of the cell its two severities fall in, and only the cell
below both first thresholds pays nothing. Its parameters are named
`<component>.breakpoints[k]` and `levels[i,j]`.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Protocol,
    Self,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator, Strict, model_validator

from crc_sdk.parametric._canonical import canonical_json
from crc_sdk.parametric.errors import (
    CoverDefinitionError,
    SampleError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.index import Tail, _component_name, _number, _quantity
from crc_sdk.parametric.parameters import Free, _DefinitionModel, _resolve_values

__all__ = [
    "BinaryPayout",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "MaxOf",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "StepGrid",
    "StepPayout",
    "SumOf",
    "register_payout",
]

FloatArray = NDArray[np.float64]


@runtime_checkable
class PayoutFunction(Protocol):
    """What every payout function provides.

    `kind` is the registry key written into the cover document. `tail` is
    the severe side of the index; a cover requires it to equal its index's.
    `index_names` names the index components the payout reads, in the order
    of the columns of a multi-component index; a cover on one index reads
    `("index",)`.

    `parameters()` returns every parameter by flat name, in a fixed order,
    each a float or `Free`. `with_values` returns a copy with values for
    some free parameters. `evaluate` maps index values to payouts, same
    shape and non-negative, and needs every parameter resolved. The document
    holds `kind` and every parameter, with `Free` ones in their document
    form, so that `from_document(to_document())` rebuilds the payout.

    A payout on several components names them in `index_names` and has as
    `tail` a tuple of one `Tail` per name, in that order. It goes on a cover
    with a `CompositeIndex` of those names, its `evaluate` takes `(n, k)`
    index values, one column per name, and returns `(n,)`.
    """

    kind: ClassVar[str]

    @property
    def tail(self) -> Tail | tuple[Tail, ...]: ...

    @property
    def index_names(self) -> tuple[str, ...]: ...

    def parameters(self) -> Mapping[str, float | Free]: ...

    def with_values(self, values: Mapping[str, float]) -> Self: ...

    def evaluate(self, index: FloatArray) -> FloatArray: ...

    def to_document(self) -> dict[str, Any]: ...

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self: ...


@runtime_checkable
class LinearInAmounts(PayoutFunction, Protocol):
    """A payout that is linear in its amounts once its thresholds are fixed.

    The payout is `basis(index) @ amounts`, with the amounts in
    `amount_names()` order. `basis` returns one row per index value and one
    column per amount, and needs the thresholds resolved; the amounts may
    still be free. `amount_constraints(limit)` states the bounds and linear
    constraints on the amounts, with every amount at most `limit` when one
    is given. Designs use these to solve for the amounts as a linear or
    quadratic programme, when there is at least one amount; a payout with no
    amounts is searched on the grids of its free parameters.

    On several index components `basis` takes the `(n, k)` index of
    `PayoutFunction.evaluate` and still returns `(n, n_amounts)`.
    """

    def amount_names(self) -> tuple[str, ...]: ...

    def basis(self, index: FloatArray) -> FloatArray: ...

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints: ...


def _has_amounts(payout: Any) -> bool:
    """Whether a payout's amounts can be solved: `LinearInAmounts` with at least one."""
    return isinstance(payout, LinearInAmounts) and bool(payout.amount_names())


def _float_array(name: str, values: ArrayLike, ndim: int) -> FloatArray:
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CoverDefinitionError(f"{name} must be numeric") from error
    if array.ndim != ndim:
        raise CoverDefinitionError(f"{name} must be {ndim}-D, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class LinearConstraints:
    """Linear constraints on an amount vector `a`.

    `a_ub @ a <= b_ub` and `lower <= a <= upper`, where `upper` is `inf` for
    an amount without an upper bound. `a_ub` has one row per constraint and
    one column per amount. The arrays are read-only copies.

    Raises `CoverDefinitionError` for mismatched shapes or non-finite values
    (other than an infinite upper bound).
    """

    a_ub: FloatArray
    b_ub: FloatArray
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        a_ub = _float_array("a_ub", self.a_ub, 2)
        b_ub = _float_array("b_ub", self.b_ub, 1)
        lower = _float_array("lower", self.lower, 1)
        upper = _float_array("upper", self.upper, 1)
        n_amounts = lower.shape[0]
        if upper.shape != (n_amounts,) or a_ub.shape[1:] != (n_amounts,):
            raise CoverDefinitionError(
                "lower, upper and the columns of a_ub must have one entry per amount"
            )
        if b_ub.shape != (a_ub.shape[0],):
            raise CoverDefinitionError("b_ub must have one entry per row of a_ub")
        if not (np.isfinite(a_ub).all() and np.isfinite(b_ub).all()):
            raise CoverDefinitionError("a_ub and b_ub must be finite")
        if not np.isfinite(lower).all() or np.isnan(upper).any():
            raise CoverDefinitionError("lower must be finite and upper not NaN")
        if np.isneginf(upper).any():
            raise CoverDefinitionError("upper must not be -inf")
        object.__setattr__(self, "a_ub", a_ub)
        object.__setattr__(self, "b_ub", b_ub)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def n_amounts(self) -> int:
        """The number of amounts the constraints are over."""
        return int(self.lower.shape[0])


def _index_values(index: ArrayLike, *, ndim: int | None = None) -> FloatArray:
    """Return index values as a float array; raise `SampleError` if invalid."""
    try:
        values = np.asarray(index, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError("index values must be numeric") from error
    if ndim is not None and values.ndim != ndim:
        raise SampleError(f"index values must be {ndim}-D, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise SampleError("index values must be finite")
    return values


def _parameter_value(value: Any) -> Any:
    if isinstance(value, Free):
        return value
    if isinstance(value, Mapping):
        return Free.from_document(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CoverDefinitionError(f"must be a number or Free, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise CoverDefinitionError(f"must be finite, got {number}")
    return number + 0.0


def _parameter_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CoverDefinitionError(f"must be a sequence, got {value!r}")
    return value


Parameter = Annotated[float | Free, BeforeValidator(_parameter_value)]
"""A fixed value or a `Free` parameter."""

Parameters = Annotated[tuple[Parameter, ...], BeforeValidator(_parameter_list)]
"""A sequence of parameters, such as the levels of a step payout."""

StrictBool = Annotated[bool, Strict()]


def _severity_candidates(value: float | Free, sign: float) -> list[float]:
    """The severities a threshold can take: its value, or its grid."""
    if isinstance(value, Free):
        return sorted(sign * point for point in value.grid or ())
    return [sign * value]


def _check_thresholds(
    names: Sequence[str], values: Sequence[float | Free], tail: Tail, what: str
) -> None:
    """Thresholds must be strictly increasing in severity for some choice."""
    for name, value in zip(names, values, strict=True):
        if isinstance(value, Free) and (value.upper is None or value.grid is None):
            raise CoverDefinitionError(
                f"{name} is a threshold in index units: a free threshold needs a "
                "finite upper bound and a grid"
            )
    sign = 1.0 if tail == "upper" else -1.0
    previous = -math.inf
    for value in values:
        later = [s for s in _severity_candidates(value, sign) if s > previous]
        if not later:
            direction = "increasing" if tail == "upper" else "decreasing"
            choice = (
                "; no choice of the free values is ordered"
                if any(isinstance(item, Free) for item in values)
                else ""
            )
            raise CoverDefinitionError(
                f"{what} must be strictly {direction} for tail={tail!r}{choice}"
            )
        previous = later[0]


def _least_amount(value: float | Free, previous: float) -> float | None:
    """The smallest value an amount can take at or above `previous`, if it has one."""
    if not isinstance(value, Free):
        return value if value >= previous else None
    lowest = max(previous, value.lower)
    if value.grid is not None:
        return next((point for point in value.grid if point >= lowest), None)
    return lowest if value.upper is None or value.upper >= lowest else None


def _amount_bounds(
    values: Sequence[float | Free], limit: float | None
) -> tuple[FloatArray, FloatArray]:
    """The lower and upper bounds of amounts, each upper bound capped at `limit`."""
    lower = np.array(
        [value.lower if isinstance(value, Free) else value for value in values]
    )
    upper = np.array(
        [
            (math.inf if value.upper is None else value.upper)
            if isinstance(value, Free)
            else value
            for value in values
        ]
    )
    if limit is not None:
        upper = np.minimum(upper, _checked_limit(limit))
    return lower, upper


def _check_amounts(
    names: Sequence[str], values: Sequence[float | Free], *, monotone: bool, what: str
) -> None:
    """Amounts must be non-negative and, if monotone, non-decreasing."""
    for name, value in zip(names, values, strict=True):
        lowest = value.lower if isinstance(value, Free) else value
        if lowest < 0:
            bound = "the lower bound of " if isinstance(value, Free) else ""
            raise CoverDefinitionError(
                f"{bound}{name} must be >= 0: amounts are non-negative"
            )
    if not monotone:
        return
    previous = 0.0
    for value in values:
        least = _least_amount(value, previous)
        if least is None:
            raise CoverDefinitionError(
                f"{what} must be non-decreasing when monotone=True"
            )
        previous = least


def _location(name: str) -> tuple[str, int | None]:
    field, _, rest = name.partition("[")
    return (field, int(rest[:-1])) if rest else (field, None)


class _BuiltinPayout(_DefinitionModel):
    """Shared parameter handling of the built-in families."""

    kind: ClassVar[str]
    _threshold_fields: ClassVar[tuple[str, ...]]
    _amount_fields: ClassVar[tuple[str, ...]]

    tail: Tail = "upper"

    def __init__(self, **data: Any) -> None:
        # Explicit, so type checkers do not take pydantic's field-only signature.
        super().__init__(**data)

    @property
    def index_names(self) -> tuple[str, ...]:
        """The index components the payout reads: `("index",)`."""
        return ("index",)

    def _field_items(self, fields: Sequence[str]) -> list[tuple[str, float | Free]]:
        items: list[tuple[str, float | Free]] = []
        for field in fields:
            value = getattr(self, field)
            if isinstance(value, tuple):
                items.extend((f"{field}[{i}]", item) for i, item in enumerate(value))
            else:
                items.append((field, value))
        return items

    def parameters(self) -> dict[str, float | Free]:
        """Every parameter by flat name: thresholds first, then amounts."""
        return dict(self._field_items(self._threshold_fields + self._amount_fields))

    def amount_names(self) -> tuple[str, ...]:
        """The names of the amounts, in the column order of `basis`."""
        return tuple(name for name, _ in self._field_items(self._amount_fields))

    def with_values(self, values: Mapping[str, float]) -> Self:
        """Return a copy with values for some of the free parameters.

        Each name must be a free parameter, and each value a finite number
        within its bounds (the grid is not enforced). The copy is validated
        again, so the resolved thresholds must be ordered. Raises
        `CoverDefinitionError` otherwise.
        """
        checked = _resolve_values(self.parameters(), values)
        data: dict[str, Any] = {
            field: list(value) if isinstance(value, tuple) else value
            for field, value in self._fields_data().items()
        }
        for name, value in checked.items():
            field, position = _location(name)
            if position is None:
                data[field] = value
            else:
                data[field][position] = value
        return type(self)._validated(data)

    def _fields_data(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in type(self).model_fields}

    def _unresolved(self, fields: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            name for name, value in self._field_items(fields) if isinstance(value, Free)
        )

    def _require_resolved(self, fields: Sequence[str]) -> None:
        free = self._unresolved(fields)
        if free:
            raise UnresolvedParameterError(
                f"the {self.kind} payout has free parameters: {', '.join(free)}"
            )

    def _sign(self) -> float:
        return 1.0 if self.tail == "upper" else -1.0

    def _thresholds(self) -> FloatArray:
        """The resolved thresholds as severities, strictly increasing."""
        self._require_resolved(self._threshold_fields)
        values = [value for _, value in self._field_items(self._threshold_fields)]
        return self._sign() * np.asarray(values, dtype=np.float64)

    def _amounts(self) -> FloatArray:
        self._require_resolved(self._amount_fields)
        values = [value for _, value in self._field_items(self._amount_fields)]
        return np.asarray(values, dtype=np.float64)

    def _monotone(self) -> bool:
        return False

    def evaluate(self, index: FloatArray) -> FloatArray:
        """Return the payout of each index value, same shape.

        Raises `UnresolvedParameterError` if a parameter is free, and
        `SampleError` for non-numeric or non-finite index values.
        """
        self._require_resolved(self._threshold_fields + self._amount_fields)
        values = _index_values(index)
        severity = self._sign() * values.reshape(-1)
        payout = self._payout(severity, self._thresholds(), self._amounts())
        return payout.reshape(values.shape)

    def basis(self, index: FloatArray) -> FloatArray:
        """Return the basis matrix, `(n, len(amount_names()))`.

        The payout is `basis(index) @ amounts`. Rows are non-negative and
        sum to at most 1, so every payout is at most the largest amount.
        Needs the thresholds resolved; raises `UnresolvedParameterError`
        otherwise, and `SampleError` unless `index` is 1-D and finite.
        """
        self._require_resolved(self._threshold_fields)
        severity = self._sign() * _index_values(index, ndim=1)
        return self._basis(severity, self._thresholds())

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        """Return the bounds and constraints on the amounts.

        A fixed amount is bounded by its value on both sides; a free one by
        its `Free` bounds. `limit` caps every amount; with `monotone=True`
        each level is at most the next. Raises `CoverDefinitionError` for a
        negative or non-finite limit.
        """
        items = self._field_items(self._amount_fields)
        lower, upper = _amount_bounds([value for _, value in items], limit)
        n_amounts = len(items)
        if self._monotone() and n_amounts > 1:
            a_ub = np.zeros((n_amounts - 1, n_amounts))
            rows = np.arange(n_amounts - 1)
            a_ub[rows, rows] = 1.0
            a_ub[rows, rows + 1] = -1.0
        else:
            a_ub = np.zeros((0, n_amounts))
        return LinearConstraints(
            a_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]), lower=lower, upper=upper
        )

    def to_document(self) -> dict[str, Any]:
        """Return `kind` and every field; `Free` parameters in document form."""
        document: dict[str, Any] = {"kind": self.kind}
        for field, value in self._fields_data().items():
            document[field] = _document_value(value)
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a payout written by `to_document`.

        Raises `CoverDefinitionError` for another `kind`, unknown or missing
        fields, or invalid values.
        """
        if not isinstance(document, Mapping):
            raise CoverDefinitionError("a payout document must be an object")
        if document.get("kind") != cls.kind:
            raise CoverDefinitionError(
                f"expected a payout of kind {cls.kind!r}, got {document.get('kind')!r}"
            )
        return cls._validated(
            {key: value for key, value in document.items() if key != "kind"}
        )

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        raise NotImplementedError

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        raise NotImplementedError


def _parameter_or_none(value: Any) -> float | None:
    if not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _checked_limit(limit: Any) -> float:
    """Return a limit as a float; raise `CoverDefinitionError` if it is invalid."""
    cap = None if isinstance(limit, bool) else _parameter_or_none(limit)
    if cap is None or cap < 0:
        raise CoverDefinitionError(f"limit must be a finite number >= 0, got {limit!r}")
    return cap


def _document_value(value: Any) -> Any:
    if isinstance(value, Free):
        return value.to_document()
    if isinstance(value, tuple):
        return [_document_value(item) for item in value]
    return value


class LinearPayout(_BuiltinPayout):
    """Pays in proportion to the severity between attachment and exhaustion.

    The payout is `limit * clip((s - a) / (e - a), 0, 1)` in severity `s`:
    nothing at or below the `attachment`, the full `limit` at or beyond the
    `exhaustion`, linear in between. `attachment` and `exhaustion` are in
    index units, with the exhaustion more severe (larger for `tail="upper"`,
    smaller for `"lower"`). `limit` is in payout units and non-negative.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "linear"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("attachment", "exhaustion")
    _amount_fields: ClassVar[tuple[str, ...]] = ("limit",)

    attachment: Parameter
    exhaustion: Parameter
    limit: Parameter

    def __init__(
        self,
        attachment: float | Free,
        exhaustion: float | Free,
        limit: float | Free,
        *,
        tail: Tail = "upper",
    ) -> None:
        super().__init__(
            attachment=attachment, exhaustion=exhaustion, limit=limit, tail=tail
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_thresholds(
            ("attachment", "exhaustion"),
            (self.attachment, self.exhaustion),
            self.tail,
            "attachment and exhaustion",
        )
        _check_amounts(("limit",), (self.limit,), monotone=False, what="limit")
        return self

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        payout: FloatArray = amounts[0] * self._basis(severity, thresholds)[:, 0]
        return payout

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        attachment, exhaustion = float(thresholds[0]), float(thresholds[1])
        share = (severity - attachment) / (exhaustion - attachment)
        basis: FloatArray = np.clip(share, 0.0, 1.0).reshape(-1, 1)
        return basis


class StepPayout(_BuiltinPayout):
    """Pays a level per band of severity.

    With breakpoints b_0 < ... < b_{n-1} in severity, a year pays `levels[k]`
    when b_k <= s < b_{k+1}, `levels[n-1]` when s >= b_{n-1}, and nothing
    below b_0, the attachment. A severity equal to a breakpoint falls into
    the band above it. `breakpoints` are in index units, increasing for
    `tail="upper"` and decreasing for `"lower"`; `levels` are in payout
    units, one per breakpoint. `monotone=True` requires non-decreasing
    levels.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "step"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("breakpoints",)
    _amount_fields: ClassVar[tuple[str, ...]] = ("levels",)

    breakpoints: Parameters
    levels: Parameters
    monotone: StrictBool = True

    def __init__(
        self,
        breakpoints: Sequence[float | Free] | NDArray[np.float64],
        levels: Sequence[float | Free] | NDArray[np.float64],
        *,
        tail: Tail = "upper",
        monotone: bool = True,
    ) -> None:
        super().__init__(
            breakpoints=breakpoints, levels=levels, tail=tail, monotone=monotone
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_banded(self, "breakpoints", minimum=1)
        return self

    def _monotone(self) -> bool:
        return self.monotone

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        band = np.searchsorted(thresholds, severity, side="right") - 1
        return np.where(band >= 0, amounts[np.maximum(band, 0)], 0.0)

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        band = np.searchsorted(thresholds, severity, side="right") - 1
        matrix = np.zeros((severity.shape[0], thresholds.shape[0]))
        rows = np.flatnonzero(band >= 0)
        matrix[rows, band[rows]] = 1.0
        return matrix


class PiecewiseLinearPayout(_BuiltinPayout):
    """Pays along straight lines between knots.

    With knots c_0 < ... < c_{m-1} in severity (m >= 2), a year pays nothing
    below c_0, the linear interpolation of the points (c_k, levels[k]) on
    [c_0, c_{m-1}], and `levels[m-1]` at or above c_{m-1}. At c_0 the payout
    is `levels[0]`, so a positive first level is a jump at the attachment.
    `knots` are in index units, increasing for `tail="upper"` and decreasing
    for `"lower"`; `levels` are in payout units, one per knot.
    `monotone=True` requires non-decreasing levels.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "piecewise_linear"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("knots",)
    _amount_fields: ClassVar[tuple[str, ...]] = ("levels",)

    knots: Parameters
    levels: Parameters
    monotone: StrictBool = True

    def __init__(
        self,
        knots: Sequence[float | Free] | NDArray[np.float64],
        levels: Sequence[float | Free] | NDArray[np.float64],
        *,
        tail: Tail = "upper",
        monotone: bool = True,
    ) -> None:
        super().__init__(knots=knots, levels=levels, tail=tail, monotone=monotone)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_banded(self, "knots", minimum=2)
        return self

    def _monotone(self) -> bool:
        return self.monotone

    @staticmethod
    def _segments(
        severity: FloatArray, knots: FloatArray
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], FloatArray, NDArray[np.intp]]:
        """Rows inside [c_0, c_{m-1}), their segment and weight; rows beyond."""
        segment = np.searchsorted(knots, severity, side="right") - 1
        last = knots.shape[0] - 1
        inside = np.flatnonzero((segment >= 0) & (segment < last))
        left = segment[inside]
        weight = (severity[inside] - knots[left]) / (knots[left + 1] - knots[left])
        beyond = np.flatnonzero(segment >= last)
        return inside, left, weight, beyond

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        inside, left, weight, beyond = self._segments(severity, thresholds)
        low, high = amounts[left], amounts[left + 1]
        # low + w * (high - low) is exact at a flat segment and monotone in w;
        # the clip keeps rounding from leaving the segment's range.
        line = np.clip(
            low + weight * (high - low), np.minimum(low, high), np.maximum(low, high)
        )
        payout = np.zeros(severity.shape[0])
        payout[inside] = line
        payout[beyond] = amounts[-1]
        return payout

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        inside, left, weight, beyond = self._segments(severity, thresholds)
        matrix = np.zeros((severity.shape[0], thresholds.shape[0]))
        matrix[inside, left] = 1.0 - weight
        matrix[inside, left + 1] = weight
        matrix[beyond, -1] = 1.0
        return matrix


class BinaryPayout(_BuiltinPayout):
    """Pays a fixed amount when the index reaches the trigger.

    A year pays `amount` when its severity is at or beyond `trigger` (at or
    above it for `tail="upper"`, at or below it for `"lower"`), and nothing
    otherwise. `trigger` is in index units, `amount` in payout units and
    non-negative.

    Raises `CoverDefinitionError` for invalid parameters.
    """

    kind: ClassVar[str] = "binary"
    _threshold_fields: ClassVar[tuple[str, ...]] = ("trigger",)
    _amount_fields: ClassVar[tuple[str, ...]] = ("amount",)

    trigger: Parameter
    amount: Parameter

    def __init__(
        self, trigger: float | Free, amount: float | Free, *, tail: Tail = "upper"
    ) -> None:
        super().__init__(trigger=trigger, amount=amount, tail=tail)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_thresholds(("trigger",), (self.trigger,), self.tail, "trigger")
        _check_amounts(("amount",), (self.amount,), monotone=False, what="amount")
        return self

    def _payout(
        self, severity: FloatArray, thresholds: FloatArray, amounts: FloatArray
    ) -> FloatArray:
        payout: FloatArray = np.where(severity >= thresholds[0], amounts[0], 0.0)
        return payout

    def _basis(self, severity: FloatArray, thresholds: FloatArray) -> FloatArray:
        basis: FloatArray = (severity >= thresholds[0]).astype(np.float64)
        return basis.reshape(-1, 1)


def _check_banded(
    payout: StepPayout | PiecewiseLinearPayout, field: str, *, minimum: int
) -> None:
    thresholds: tuple[float | Free, ...] = getattr(payout, field)
    if len(thresholds) < minimum:
        raise CoverDefinitionError(
            f"{field} must hold at least {minimum} value{'s' if minimum > 1 else ''}"
        )
    if len(payout.levels) != len(thresholds):
        raise CoverDefinitionError(
            f"levels must hold one value per entry of {field}: "
            f"got {len(payout.levels)} levels and {len(thresholds)} {field}"
        )
    names = [f"{field}[{i}]" for i in range(len(thresholds))]
    _check_thresholds(names, thresholds, payout.tail, field)
    _check_amounts(
        [f"levels[{i}]" for i in range(len(payout.levels))],
        payout.levels,
        monotone=payout.monotone,
        what="levels",
    )


_COMPONENT_ENTRY_KEYS = frozenset({"name", "payout"})


def _read_components(components: Any, kind: str) -> Mapping[str, PayoutFunction]:
    """Return the components, read-only; raise `CoverDefinitionError` if invalid."""
    if not isinstance(components, Mapping):
        raise CoverDefinitionError(
            f"a {kind} payout maps each component name to a one-index payout, "
            f"got {type(components).__name__}"
        )
    if len(components) < 2:
        raise CoverDefinitionError(
            f"a {kind} payout holds two components or more, got {len(components)}"
        )
    for name, payout in components.items():
        _component_name(name)
        if not isinstance(payout, PayoutFunction):
            raise CoverDefinitionError(
                f"component {name!r} must implement PayoutFunction, "
                f"got {type(payout).__name__}"
            )
        if tuple(payout.index_names) != ("index",) or not isinstance(payout.tail, str):
            raise CoverDefinitionError(
                f"component {name!r} must be a payout on one index, "
                f"index_names == ('index',); got {payout.index_names!r}"
            )
    return MappingProxyType(dict(components))


def _components_from_document(
    document: Any, kind: str, keys: frozenset[str]
) -> dict[str, PayoutFunction]:
    """Read the components of a document that `to_document` wrote."""
    if not isinstance(document, Mapping):
        raise CoverDefinitionError("a payout document must be an object")
    if document.get("kind") != kind:
        raise CoverDefinitionError(
            f"expected a payout of kind {kind!r}, got {document.get('kind')!r}"
        )
    if set(document) != keys:
        raise CoverDefinitionError(f"a {kind} document has the keys {sorted(keys)}")
    entries = document["components"]
    if isinstance(entries, str | bytes | Mapping) or not isinstance(entries, Sequence):
        raise CoverDefinitionError(f"a {kind} document's components is a list")
    components: dict[str, PayoutFunction] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _COMPONENT_ENTRY_KEYS:
            raise CoverDefinitionError(
                f"each component of a {kind} document has the keys "
                f"{sorted(_COMPONENT_ENTRY_KEYS)}"
            )
        name = _component_name(entry["name"])
        if name in components:
            raise CoverDefinitionError(f"a {kind} document repeats component {name!r}")
        payout = entry["payout"]
        if not isinstance(payout, Mapping):
            raise CoverDefinitionError("a payout document must be an object")
        components[name] = _payout_reader(payout.get("kind"))(payout)
    return components


def _payout_from_document(document: Mapping[str, Any]) -> Any:
    """Read a payout document through the registry."""
    return _payout_reader(document["kind"])(document)


def _index_columns(index: ArrayLike, names: Sequence[str]) -> FloatArray:
    """Return an `(n, k)` index with one column per name; raise `SampleError` if not."""
    values = _index_values(index, ndim=2)
    if values.shape[1] != len(names):
        raise SampleError(
            f"index values must have one column per component "
            f"{list(names)}, got shape {values.shape}"
        )
    return values


def _no_constraints() -> LinearConstraints:
    return LinearConstraints(
        a_ub=np.zeros((0, 0)),
        b_ub=np.zeros(0),
        lower=np.zeros(0),
        upper=np.zeros(0),
    )


def _is_ordered(constraints: LinearConstraints) -> bool:
    """Whether the constraints hold `a_i <= a_{i+1}` for every pair of amounts."""
    needed = {(i, i + 1) for i in range(constraints.n_amounts - 1)}
    for row, bound in zip(constraints.a_ub, constraints.b_ub, strict=True):
        nonzero = np.flatnonzero(row)
        if (
            nonzero.size == 2
            and bound == 0.0
            and row[nonzero[0]] == 1.0
            and row[nonzero[1]] == -1.0
        ):
            needed.discard((int(nonzero[0]), int(nonzero[1])))
    return not needed


def _total_limit_rows(parts: Sequence[LinearConstraints]) -> FloatArray:
    """The rows that keep every choice of one amount per component within a limit.

    A component whose amounts are ordered pays at most its last one, so it
    offers only that; any other offers each of its amounts.
    """
    choices: list[list[int]] = []
    start = 0
    for part in parts:
        end = start + part.n_amounts
        choices.append([end - 1] if _is_ordered(part) else list(range(start, end)))
        start = end
    picks = np.array(list(itertools.product(*choices)), dtype=np.int64)
    rows = np.zeros((picks.shape[0], start))
    rows[np.arange(picks.shape[0])[:, None], picks] = 1.0
    return rows


@dataclass(frozen=True, eq=False)
class _OfComponents:
    """What `SumOf` and `MaxOf` share: one-index payouts, by component name."""

    kind: ClassVar[str]

    components: Mapping[str, PayoutFunction]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "components", _read_components(self.components, self.kind)
        )

    @property
    def tail(self) -> tuple[Tail, ...]:
        """Each component's severe side, in order."""
        return tuple(cast(Tail, payout.tail) for payout in self.components.values())

    @property
    def index_names(self) -> tuple[str, ...]:
        """The components, in the order of the columns of the index."""
        return tuple(self.components)

    def _component_parameters(self) -> dict[str, float | Free]:
        return {
            f"{name}.{key}": value
            for name, payout in self.components.items()
            for key, value in payout.parameters().items()
        }

    def parameters(self) -> dict[str, float | Free]:
        """Every parameter by `<component>.<its name>`, in component order."""
        return self._component_parameters()

    def _with_component_values(
        self, values: Mapping[str, float]
    ) -> dict[str, PayoutFunction]:
        """The components with `values`, keyed `<component>.<parameter>`."""
        parts: dict[str, dict[str, float]] = {name: {} for name in self.components}
        for key, value in values.items():
            name, _, parameter = key.partition(".")
            parts[name][parameter] = value
        return {
            name: payout.with_values(parts[name]) if parts[name] else payout
            for name, payout in self.components.items()
        }

    def _require_resolved(
        self, parameters: Mapping[str, float | Free] | None = None
    ) -> None:
        chosen = self.parameters() if parameters is None else parameters
        free = [key for key, value in chosen.items() if isinstance(value, Free)]
        if free:
            raise UnresolvedParameterError(
                f"the {self.kind} payout has free parameters: {', '.join(free)}"
            )

    def _columns(self, index: ArrayLike) -> FloatArray:
        return _index_columns(index, tuple(self.components))

    def _component_payouts(self, index: ArrayLike) -> FloatArray:
        """Each component's payout: `(n, k)`, one column per component."""
        values = self._columns(index)
        columns: list[FloatArray] = []
        for column, (name, payout) in enumerate(self.components.items()):
            paid = np.asarray(payout.evaluate(values[:, column]), dtype=np.float64)
            if paid.shape != (values.shape[0],):
                raise CoverDefinitionError(
                    f"component {name!r} returned shape {paid.shape} for "
                    f"{values.shape[0]} index values"
                )
            columns.append(paid)
        return np.column_stack(columns)

    def _components_document(self) -> list[dict[str, Any]]:
        # A list, because canonical JSON sorts the keys of an object and the
        # order of the components is part of the terms.
        return [
            {"name": name, "payout": dict(payout.to_document())}
            for name, payout in self.components.items()
        ]

    def to_document(self) -> dict[str, Any]:
        """Return `kind` and the components, each with its own document."""
        return {"components": self._components_document(), "kind": self.kind}

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, _OfComponents)
        return canonical_json(self.to_document()) == canonical_json(other.to_document())

    def __hash__(self) -> int:
        return hash(canonical_json(self.to_document()))

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return (_payout_from_document, (self.to_document(),))


@dataclass(frozen=True, eq=False)
class SumOf(_OfComponents):
    """Pays the sum of one-index payouts, one per index component.

    `components` maps each component's name to a one-index payout (a built-in
    family or a registered user family), two or more, in order. A year pays
    `sum_k pi_k(s_k)`, each component on its own index, so its parameters are
    named `<component>.<parameter>`. It goes on a cover with a `CompositeIndex`
    of the same names.

    The payout is linear in every amount when every component is
    `LinearInAmounts` with at least one amount: `basis` is then the block
    `[B_1 | ... | B_K]`. Otherwise `amount_names()` is empty and a design takes
    the numpy grid over every free parameter.

    A limit bounds the year's total, not each amount: see
    `amount_constraints`.

    Raises `CoverDefinitionError` for fewer than two components, an invalid
    name, or a component that is not a one-index payout.
    """

    kind: ClassVar[str] = "sum_of"

    def with_values(self, values: Mapping[str, float]) -> Self:
        """Return a copy with values for some of the free parameters.

        Raises `CoverDefinitionError` for an unknown or fixed parameter, a
        value outside its bounds, or a component that becomes invalid.
        """
        checked = _resolve_values(self.parameters(), values)
        return type(self)(self._with_component_values(checked))

    def evaluate(self, index: FloatArray) -> FloatArray:
        """Return the payout of each row of an `(n, k)` index: `(n,)`.

        Raises `UnresolvedParameterError` if a parameter is free, and
        `SampleError` for an index of the wrong shape or with non-finite
        values.
        """
        self._require_resolved()
        payout: FloatArray = self._component_payouts(index).sum(axis=1)
        return payout

    def breakdown(self, index: FloatArray) -> dict[str, FloatArray]:
        """Return each component's payout of an `(n, k)` index, by name."""
        self._require_resolved()
        paid = self._component_payouts(index)
        return {name: paid[:, j] for j, name in enumerate(self.components)}

    def _linear(self) -> bool:
        return all(_has_amounts(payout) for payout in self.components.values())

    def amount_names(self) -> tuple[str, ...]:
        """The amounts, in the column order of `basis`; empty if not linear."""
        if not self._linear():
            return ()
        return tuple(
            f"{name}.{amount}"
            for name, payout in self.components.items()
            for amount in cast(LinearInAmounts, payout).amount_names()
        )

    def _bases(self, index: ArrayLike) -> list[FloatArray]:
        values = self._columns(index)
        bases: list[FloatArray] = []
        for column, (name, payout) in enumerate(self.components.items()):
            family = cast(LinearInAmounts, payout)
            basis = np.asarray(family.basis(values[:, column]), dtype=np.float64)
            if basis.shape != (values.shape[0], len(family.amount_names())):
                raise CoverDefinitionError(
                    f"component {name!r} returned a basis of shape {basis.shape}"
                )
            bases.append(basis)
        return bases

    def basis(self, index: FloatArray) -> FloatArray:
        """Return the block basis `[B_1 | ... | B_K]`, `(n, len(amount_names()))`.

        Needs the thresholds resolved; raises `UnresolvedParameterError`
        otherwise, and `SampleError` for an index of the wrong shape.
        """
        if not self._linear():
            return np.zeros((self._columns(index).shape[0], 0))
        return np.hstack(self._bases(index))

    def _largest_row_sums(self, index: ArrayLike) -> dict[str, float]:
        """The largest sum of a row of each component's basis, by name."""
        return {
            name: float(basis.sum(axis=1).max(initial=0.0))
            for name, basis in zip(self.components, self._bases(index), strict=True)
        }

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        """Return the bounds and constraints of the components, side by side.

        `limit` bounds the year's total: for every choice of one amount per
        component, the chosen amounts sum to at most `limit`. Each component
        pays at most its largest amount and any can reach it, so that is the
        worst year. A component with ordered amounts offers its last amount,
        so with every component monotone this is one row, and otherwise up to
        the product of the numbers of amounts. No amount can then exceed the
        limit either, so every upper bound is capped at it. Raises
        `CoverDefinitionError` for a negative or non-finite limit.
        """
        if not self._linear():
            return _no_constraints()
        cap = None if limit is None else _checked_limit(limit)
        parts = [
            cast(LinearInAmounts, payout).amount_constraints(cap)
            for payout in self.components.values()
        ]
        n_amounts = sum(part.n_amounts for part in parts)
        blocks: list[FloatArray] = []
        bounds = [part.b_ub for part in parts]
        start = 0
        for part in parts:
            block = np.zeros((part.a_ub.shape[0], n_amounts))
            block[:, start : start + part.n_amounts] = part.a_ub
            blocks.append(block)
            start += part.n_amounts
        if cap is not None:
            rows = _total_limit_rows(parts)
            blocks.append(rows)
            bounds.append(np.full(rows.shape[0], cap))
        return LinearConstraints(
            a_ub=np.vstack(blocks),
            b_ub=np.concatenate(bounds),
            lower=np.concatenate([part.lower for part in parts]),
            upper=np.concatenate([part.upper for part in parts]),
        )

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a payout written by `to_document`.

        Raises `UnknownPayoutKindError` for a component of an unregistered
        kind, and `CoverDefinitionError` for another `kind` or invalid keys.
        """
        return cls(
            _components_from_document(
                document, cls.kind, frozenset({"components", "kind"})
            )
        )


@dataclass(frozen=True, eq=False)
class MaxOf(_OfComponents):
    """Pays a scale times the largest of one-index payouts, one per component.

    `components` is as for `SumOf`, and a year pays
    `scale * max_k pi_k(s_k)`: one contract, one amount. `scale` is a float, or
    `Free` for a design to choose it.

    The payout is linear in the scale when the scale is free and every
    component is `LinearInAmounts` with fixed amounts: `amount_names()` is then
    `("scale",)` and `basis` the column `max_k pi_k(s_k)`. In every other case
    `amount_names()` is empty and a design takes the numpy grid over every free
    parameter: the objective is not convex in the component amounts, so they
    have no linear or quadratic programme.

    On a free scale a limit bounds the largest year: with `G` the largest
    amount of any component, `scale <= limit / G`.

    Raises `CoverDefinitionError` as `SumOf` does, for a negative or non-finite
    scale, and for a free scale beside a free component amount, because the
    scale would multiply amounts that are free already, so it is not
    identified.
    """

    kind: ClassVar[str] = "max_of"

    scale: float | Free = dataclasses.field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            scale = _parameter_value(self.scale)
        except CoverDefinitionError as error:
            raise CoverDefinitionError(f"scale {error}") from error
        _check_amounts(("scale",), (scale,), monotone=False, what="scale")
        object.__setattr__(self, "scale", scale)
        if isinstance(scale, Free):
            free = self._free_amounts()
            if free:
                raise CoverDefinitionError(
                    "a free scale beside free component amounts is not identified: "
                    f"it would multiply {free}, which are free already; fix the "
                    "scale or those amounts"
                )

    def _free_amounts(self) -> list[str]:
        amounts: list[str] = []
        for name, payout in self.components.items():
            if isinstance(payout, LinearInAmounts):
                values = payout.parameters()
                amounts += [
                    f"{name}.{amount}"
                    for amount in payout.amount_names()
                    if isinstance(values[amount], Free)
                ]
        return amounts

    def parameters(self) -> dict[str, float | Free]:
        """Every parameter: the components', then `scale`."""
        return {**super().parameters(), "scale": self.scale}

    def with_values(self, values: Mapping[str, float]) -> Self:
        """Return a copy with values for some of the free parameters.

        Raises `CoverDefinitionError` for an unknown or fixed parameter, a
        value outside its bounds, or a component that becomes invalid.
        """
        checked = _resolve_values(self.parameters(), values)
        scale = checked.pop("scale", self.scale)
        return type(self)(self._with_component_values(checked), scale=scale)

    def _largest(self, index: ArrayLike) -> FloatArray:
        largest: FloatArray = self._component_payouts(index).max(axis=1)
        return largest

    def evaluate(self, index: FloatArray) -> FloatArray:
        """Return the payout of each row of an `(n, k)` index: `(n,)`.

        Raises `UnresolvedParameterError` if a parameter is free, and
        `SampleError` for an index of the wrong shape or with non-finite
        values.
        """
        self._require_resolved()
        payout: FloatArray = cast(float, self.scale) * self._largest(index)
        return payout

    def breakdown(self, index: FloatArray) -> dict[str, FloatArray]:
        """Return an attribution of each year's payout to the components.

        The year's payout goes to the component paying most, in equal shares
        when several tie. It is an attribution: one contract pays one amount.
        """
        self._require_resolved()
        paid = self._component_payouts(index)
        largest = paid.max(axis=1, keepdims=True)
        paying = paid == largest
        share = paying / paying.sum(axis=1, keepdims=True)
        attributed = cast(float, self.scale) * largest * share
        return {name: attributed[:, j] for j, name in enumerate(self.components)}

    def _linear(self) -> bool:
        return isinstance(self.scale, Free) and all(
            _has_amounts(payout) for payout in self.components.values()
        )

    def amount_names(self) -> tuple[str, ...]:
        """`("scale",)` when the payout is linear in a free scale, else empty."""
        return ("scale",) if self._linear() else ()

    def basis(self, index: FloatArray) -> FloatArray:
        """Return the column `max_k pi_k(s_k)`, `(n, len(amount_names()))`.

        Needs every component parameter resolved; raises
        `UnresolvedParameterError` otherwise, and `SampleError` for an index of
        the wrong shape.
        """
        if not self._linear():
            return np.zeros((self._columns(index).shape[0], 0))
        self._require_resolved(self._component_parameters())
        return self._largest(index).reshape(-1, 1)

    def _largest_amount(self) -> float:
        """The largest amount of any component; every amount is fixed."""
        amounts: list[float] = []
        for payout in self.components.values():
            family = cast(LinearInAmounts, payout)
            values = family.parameters()
            amounts += [
                float(cast(float, values[amount])) for amount in family.amount_names()
            ]
        return max(amounts)

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        """Return the bounds of a free scale, under `limit` if one is given.

        With `G` the largest amount of any component, `limit` adds
        `scale <= limit / G`. Raises `CoverDefinitionError` for a negative or
        non-finite limit.
        """
        if not self._linear():
            return _no_constraints()
        scale = cast(Free, self.scale)
        upper = math.inf if scale.upper is None else scale.upper
        if limit is not None:
            cap = _checked_limit(limit)
            largest = self._largest_amount()
            if largest > 0.0:
                upper = min(upper, cap / largest)
        return LinearConstraints(
            a_ub=np.zeros((0, 1)),
            b_ub=np.zeros(0),
            lower=np.array([scale.lower]),
            upper=np.array([upper]),
        )

    def to_document(self) -> dict[str, Any]:
        """Return `kind`, the components and the scale."""
        return {
            **super().to_document(),
            "scale": _document_value(self.scale),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a payout written by `to_document`.

        Raises `UnknownPayoutKindError` for a component of an unregistered
        kind, and `CoverDefinitionError` for another `kind`, invalid keys or
        an invalid scale.
        """
        components = _components_from_document(
            document, cls.kind, frozenset({"components", "kind", "scale"})
        )
        return cls(components, scale=document["scale"])


_GRID_KEYS = frozenset({"breakpoints", "kind", "levels", "monotone", "tail"})
_GRID_AXIS_KEYS = frozenset({"name", "thresholds"})


def _grid_parameters(what: str, values: Any) -> tuple[float | Free, ...]:
    """Read a sequence of parameters; an error names `what`."""
    try:
        return tuple(_parameter_value(item) for item in _parameter_list(values))
    except CoverDefinitionError as error:
        raise CoverDefinitionError(f"{what} {error}") from error


def _grid_tails(tail: Any) -> tuple[Tail, Tail]:
    if (
        isinstance(tail, str)
        or not isinstance(tail, Sequence)
        or len(tail) != 2
        or any(side not in ("upper", "lower") for side in tail)
    ):
        raise CoverDefinitionError(
            f"tail must hold two of 'upper' and 'lower', one per component, "
            f"got {tail!r}"
        )
    return cast("tuple[Tail, Tail]", tuple(tail))


def _grid_breakpoints(
    breakpoints: Any, tails: tuple[Tail, Tail]
) -> dict[str, tuple[float | Free, ...]]:
    """Read the thresholds of the two components; raise if they are invalid."""
    if not isinstance(breakpoints, Mapping):
        raise CoverDefinitionError(
            "a step grid maps each of its two component names to its thresholds, "
            f"got {type(breakpoints).__name__}"
        )
    if len(breakpoints) != 2:
        raise CoverDefinitionError(
            f"a step grid holds exactly two components, got {len(breakpoints)}"
        )
    axes: dict[str, tuple[float | Free, ...]] = {}
    for (name, thresholds), tail in zip(breakpoints.items(), tails, strict=True):
        _component_name(name)
        read = _grid_parameters(f"the breakpoints of {name!r}", thresholds)
        if not read:
            raise CoverDefinitionError(
                f"the breakpoints of {name!r} must hold at least 1 value"
            )
        names = [f"{name}.breakpoints[{k}]" for k in range(len(read))]
        _check_thresholds(names, read, tail, f"the breakpoints of {name!r}")
        axes[name] = read
    return axes


def _grid_levels(
    levels: Any, n_rows: int, n_columns: int
) -> tuple[tuple[float | Free, ...], ...]:
    """Read the table; raise unless it has its shape and pays nothing at (0, 0)."""
    try:
        rows = [_parameter_list(row) for row in _parameter_list(levels)]
    except CoverDefinitionError as error:
        raise CoverDefinitionError(f"levels {error}") from error
    if len(rows) != n_rows or any(len(row) != n_columns for row in rows):
        raise CoverDefinitionError(
            f"levels must have {n_rows} rows of {n_columns} values, one row per bin "
            "of the first component and one value per bin of the second"
        )
    table = tuple(_grid_parameters(f"levels[{i}]", row) for i, row in enumerate(rows))
    if isinstance(table[0][0], Free) or table[0][0] != 0.0:
        raise CoverDefinitionError(
            "levels[0][0] must be 0.0: the cell below both first thresholds pays "
            f"nothing, got {table[0][0]!r}"
        )
    return table


def _check_grid_order(levels: tuple[tuple[float | Free, ...], ...]) -> None:
    """Levels must be non-decreasing along both axes, for some choice of the free ones.

    The least value of each cell given its two predecessors is the least of
    every feasible table, so a table is feasible exactly when this finds one.
    """
    least: list[list[float]] = []
    for i, row in enumerate(levels):
        found: list[float] = []
        for j, value in enumerate(row):
            above = least[i - 1][j] if i else 0.0
            left = found[j - 1] if j else 0.0
            smallest = _least_amount(value, max(above, left))
            if smallest is None:
                raise CoverDefinitionError(
                    "levels must be non-decreasing along both axes when "
                    f"monotone=True; no value of levels[{i},{j}] is"
                )
            found.append(smallest)
        least.append(found)


@dataclass(frozen=True, eq=False, init=False)
class StepGrid:
    """Pays a level per cell of a table over two index components.

    `breakpoints` maps exactly two component names to their thresholds, in
    increasing severity (increasing for `"upper"`, decreasing for `"lower"`),
    and `tail` holds each component's severe side, in that order. On each axis
    bin 0 lies below the first threshold, bin k holds `b_{k-1} <= s < b_k`, and
    the last bin runs from the last threshold up; a severity equal to a
    threshold belongs to the more severe bin.

    `levels` is the table of amounts: with m thresholds on the first component
    and q on the second, m + 1 rows of q + 1 values, rows along the first
    component. A year pays `levels[i][j]`, with i and j the bins of its two
    severities. Only the cell below both first thresholds pays nothing, so
    `levels[0][0]` must be 0.0 and cannot be `Free`, and a year can pay on one
    component alone, in the cells `(i, 0)` and `(0, j)`.

    `monotone=True`, the default, requires levels non-decreasing along both
    axes. Each axis works on its own severity, so mixed tails need nothing
    more. `monotone=False` allows any non-negative levels and is recorded.

    Parameters are named `<component>.breakpoints[k]` and `levels[i,j]`, with
    no `levels[0,0]`. A free threshold needs a finite upper bound and a grid.
    The payout is linear in every level: `basis` is one-hot on the
    `(m + 1)(q + 1) - 1` cells, and a design solves the levels as one linear or
    quadratic programme. A limit bounds every cell.

    A resolved `SumOf` of two step payouts with levels `theta` and `phi` equals
    the grid with levels `theta_i + phi_j`, and a `MaxOf` the grid with
    `max(theta_i, phi_j)`, taking `theta_0 = phi_0 = 0`.

    Raises `CoverDefinitionError` for anything but two valid component names,
    unordered thresholds, a table of the wrong shape, `levels[0][0]` other than
    0.0, a negative amount, or levels that break `monotone=True`.
    """

    kind: ClassVar[str] = "step_grid"

    breakpoints: Mapping[str, tuple[float | Free, ...]]
    levels: tuple[tuple[float | Free, ...], ...]
    tail: tuple[Tail, Tail]
    monotone: bool

    def __init__(
        self,
        breakpoints: Mapping[str, Sequence[float | Free] | NDArray[np.float64]],
        levels: Sequence[Sequence[float | Free]] | NDArray[np.float64],
        *,
        tail: tuple[Tail, Tail] = ("upper", "upper"),
        monotone: bool = True,
    ) -> None:
        if not isinstance(monotone, bool):
            raise CoverDefinitionError(f"monotone must be a bool, got {monotone!r}")
        tails = _grid_tails(tail)
        axes = _grid_breakpoints(breakpoints, tails)
        first, second = axes.values()
        table = _grid_levels(levels, len(first) + 1, len(second) + 1)
        object.__setattr__(self, "breakpoints", MappingProxyType(axes))
        object.__setattr__(self, "levels", table)
        object.__setattr__(self, "tail", tails)
        object.__setattr__(self, "monotone", monotone)
        amounts = self._amount_parameters()
        _check_amounts(
            list(amounts), list(amounts.values()), monotone=False, what="levels"
        )
        if monotone:
            _check_grid_order(table)

    @property
    def index_names(self) -> tuple[str, ...]:
        """The two components, in the order of the columns of the index."""
        return tuple(self.breakpoints)

    def _threshold_parameters(self) -> dict[str, float | Free]:
        return {
            f"{name}.breakpoints[{k}]": value
            for name, thresholds in self.breakpoints.items()
            for k, value in enumerate(thresholds)
        }

    def _amount_parameters(self) -> dict[str, float | Free]:
        return {
            f"levels[{i},{j}]": value
            for i, row in enumerate(self.levels)
            for j, value in enumerate(row)
            if (i, j) != (0, 0)
        }

    def parameters(self) -> dict[str, float | Free]:
        """Every parameter by flat name: the thresholds, then the levels by row."""
        return {**self._threshold_parameters(), **self._amount_parameters()}

    def amount_names(self) -> tuple[str, ...]:
        """The levels, by row, in the column order of `basis`."""
        return tuple(self._amount_parameters())

    def with_values(self, values: Mapping[str, float]) -> Self:
        """Return a copy with values for some of the free parameters.

        Raises `CoverDefinitionError` for an unknown or fixed parameter, a
        value outside its bounds, or a table that becomes invalid, such as
        unordered thresholds.
        """
        checked = _resolve_values(self.parameters(), values)
        axes = {name: list(items) for name, items in self.breakpoints.items()}
        table = [list(row) for row in self.levels]
        for key, value in checked.items():
            head, _, position = key[:-1].partition("[")
            if head == "levels":
                row, column = position.split(",")
                table[int(row)][int(column)] = value
            else:
                axes[head.partition(".")[0]][int(position)] = value
        return type(self)(axes, table, tail=self.tail, monotone=self.monotone)

    def _require_resolved(self, parameters: Mapping[str, float | Free]) -> None:
        free = [key for key, value in parameters.items() if isinstance(value, Free)]
        if free:
            raise UnresolvedParameterError(
                f"the {self.kind} payout has free parameters: {', '.join(free)}"
            )

    def _bins(self, index: ArrayLike) -> list[NDArray[np.intp]]:
        """Each row's bin on each axis: 0 below the first threshold."""
        values = _index_columns(index, self.index_names)
        bins: list[NDArray[np.intp]] = []
        for column, (items, tail) in enumerate(
            zip(self.breakpoints.values(), self.tail, strict=True)
        ):
            sign = 1.0 if tail == "upper" else -1.0
            thresholds = sign * np.asarray(items, dtype=np.float64)
            bins.append(
                np.searchsorted(thresholds, sign * values[:, column], side="right")
            )
        return bins

    def evaluate(self, index: FloatArray) -> FloatArray:
        """Return the payout of each row of an `(n, 2)` index: `(n,)`.

        Raises `UnresolvedParameterError` if a parameter is free, and
        `SampleError` for an index of the wrong shape or with non-finite
        values.
        """
        self._require_resolved(self.parameters())
        first, second = self._bins(index)
        paid: FloatArray = np.asarray(self.levels, dtype=np.float64)[first, second]
        return paid

    def breakdown(self, index: FloatArray) -> dict[str, FloatArray]:
        """Return the payout by region of an `(n, 2)` index.

        `only:<first>` and `only:<second>` hold the years where one index is
        past its first threshold and the other is not, and `both` those where
        both are. A grid has no payout per component.
        """
        paid = self.evaluate(index)
        first, second = self._bins(index)
        names = self.index_names
        return {
            f"only:{names[0]}": np.where((first > 0) & (second == 0), paid, 0.0),
            f"only:{names[1]}": np.where((first == 0) & (second > 0), paid, 0.0),
            "both": np.where((first > 0) & (second > 0), paid, 0.0),
        }

    def basis(self, index: FloatArray) -> FloatArray:
        """Return the one-hot basis of the cells, `(n, len(amount_names()))`.

        The payout is `basis(index) @ amounts`, each row holding at most one
        1. Needs the thresholds resolved; raises `UnresolvedParameterError`
        otherwise, and `SampleError` for an index of the wrong shape.
        """
        self._require_resolved(self._threshold_parameters())
        first, second = self._bins(index)
        n_columns = len(self.levels[0])
        cell = first * n_columns + second - 1
        matrix = np.zeros((cell.shape[0], len(self.levels) * n_columns - 1))
        rows = np.flatnonzero(cell >= 0)
        matrix[rows, cell[rows]] = 1.0
        return matrix

    def _order_rows(self) -> FloatArray:
        """The rows `levels[i,j] <= next` along each axis, leaving out cell (0, 0)."""
        n_rows, n_columns = len(self.levels), len(self.levels[0])
        pairs = [
            ((i, j), (i + 1, j))
            for i in range(n_rows - 1)
            for j in range(n_columns)
            if (i, j) != (0, 0)
        ] + [
            ((i, j), (i, j + 1))
            for i in range(n_rows)
            for j in range(n_columns - 1)
            if (i, j) != (0, 0)
        ]
        a_ub = np.zeros((len(pairs), n_rows * n_columns - 1))
        for row, ((i, j), (k, m)) in enumerate(pairs):
            a_ub[row, i * n_columns + j - 1] = 1.0
            a_ub[row, k * n_columns + m - 1] = -1.0
        return a_ub

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        """Return the bounds and constraints on the levels.

        A fixed level is bounded by its value on both sides, a free one by its
        `Free` bounds, and `limit` caps every level. With `monotone=True` each
        level is at most the next along both axes. Raises
        `CoverDefinitionError` for a negative or non-finite limit.
        """
        lower, upper = _amount_bounds(list(self._amount_parameters().values()), limit)
        a_ub = self._order_rows() if self.monotone else np.zeros((0, lower.shape[0]))
        return LinearConstraints(
            a_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]), lower=lower, upper=upper
        )

    def to_document(self) -> dict[str, Any]:
        """Return `kind` and every term; `Free` parameters in document form."""
        # A list, because canonical JSON sorts the keys of an object and the
        # order of the components is part of the terms.
        return {
            "breakpoints": [
                {"name": name, "thresholds": _document_value(thresholds)}
                for name, thresholds in self.breakpoints.items()
            ],
            "kind": self.kind,
            "levels": _document_value(self.levels),
            "monotone": self.monotone,
            "tail": list(self.tail),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a payout written by `to_document`.

        Raises `CoverDefinitionError` for another `kind`, invalid keys or
        invalid values.
        """
        if not isinstance(document, Mapping):
            raise CoverDefinitionError("a payout document must be an object")
        if document.get("kind") != cls.kind:
            raise CoverDefinitionError(
                f"expected a payout of kind {cls.kind!r}, got {document.get('kind')!r}"
            )
        if set(document) != _GRID_KEYS:
            raise CoverDefinitionError(
                f"a {cls.kind} document has the keys {sorted(_GRID_KEYS)}"
            )
        entries = document["breakpoints"]
        if isinstance(entries, str | bytes | Mapping) or not isinstance(
            entries, Sequence
        ):
            raise CoverDefinitionError(f"a {cls.kind} document's breakpoints is a list")
        axes: dict[str, Any] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != _GRID_AXIS_KEYS:
                raise CoverDefinitionError(
                    f"each axis of a {cls.kind} document has the keys "
                    f"{sorted(_GRID_AXIS_KEYS)}"
                )
            name = _component_name(entry["name"])
            if name in axes:
                raise CoverDefinitionError(
                    f"a {cls.kind} document repeats component {name!r}"
                )
            axes[name] = entry["thresholds"]
        return cls(
            axes,
            document["levels"],
            tail=document["tail"],
            monotone=document["monotone"],
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, StepGrid)
        return canonical_json(self.to_document()) == canonical_json(other.to_document())

    def __hash__(self) -> int:
        return hash(canonical_json(self.to_document()))

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return (_payout_from_document, (self.to_document(),))


# -- the terms in words ---------------------------------------------------------

_Axis = tuple[str, str]
"""What a threshold is read on: the index's name, as a band names it, and its unit."""


class _Words:
    """Spells a payout's values, and records the free parameters it meets.

    A fixed value is spelled with its unit, and a free one by its name; the
    free one's `Free` and unit are kept, for the cover's list of free
    parameters.
    """

    def __init__(self, payout_unit: str) -> None:
        self.payout_unit = payout_unit
        self.free: dict[str, tuple[Free, str | None]] = {}

    def value(self, name: str, value: Any, unit: str | None) -> str:
        if isinstance(value, Free):
            self.free[name] = (value, unit)
            return name
        if isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return _quantity(number, unit)
        return repr(value)

    def amount(self, name: str, value: Any) -> str:
        return self.value(name, value, self.payout_unit)


def _sides(tail: Tail) -> tuple[str, str]:
    """The words for the more severe and the less severe side of a threshold."""
    return ("above", "below") if tail == "upper" else ("below", "above")


def _bands(subject: str, thresholds: Sequence[str], tail: Tail) -> list[str]:
    """Where a value lies: before the first threshold, between two, from the last on.

    A value equal to a threshold belongs to the more severe band, so a band
    runs from "at or above" one threshold (for `tail="upper"`) to "below" the
    next.
    """
    more, less = _sides(tail)
    return [
        f"{subject} is {less} {thresholds[0]}",
        *(
            f"{subject} is at or {more} {low} and {less} {high}"
            for low, high in itertools.pairwise(thresholds)
        ),
        f"{subject} is at or {more} {thresholds[-1]}",
    ]


def _order(monotone: bool, along: str = "") -> str:
    return (
        f"the levels never decrease{along}" if monotone else "the levels in any order"
    )


def _role(text: str, value: Any, role: str) -> str:
    """`text` with its role in the terms, unless it is a free parameter's name."""
    return text if isinstance(value, Free) else f"{text} (the {role})"


def _linear_words(
    payout: LinearPayout, words: _Words, axis: _Axis, prefix: str
) -> tuple[str, list[str]]:
    subject, unit = axis
    more, less = _sides(payout.tail)
    attachment = words.value(f"{prefix}attachment", payout.attachment, unit)
    exhaustion = words.value(f"{prefix}exhaustion", payout.exhaustion, unit)
    limit = words.amount(f"{prefix}limit", payout.limit)
    return (
        "in proportion between an attachment and an exhaustion, up to a limit "
        '(kind "linear")',
        [
            f"- nothing when {subject} is at or {less} "
            f"{_role(attachment, payout.attachment, 'attachment')}",
            f"- in a straight line from nothing at {attachment} to {limit} at "
            f"{exhaustion}",
            f"- {_role(limit, payout.limit, 'limit')} when {subject} is at or "
            f"{more} {_role(exhaustion, payout.exhaustion, 'exhaustion')}",
        ],
    )


def _step_words(
    payout: StepPayout, words: _Words, axis: _Axis, prefix: str
) -> tuple[str, list[str]]:
    subject, unit = axis
    thresholds = [
        words.value(f"{prefix}breakpoints[{k}]", value, unit)
        for k, value in enumerate(payout.breakpoints)
    ]
    levels = [
        words.amount(f"{prefix}levels[{k}]", value)
        for k, value in enumerate(payout.levels)
    ]
    first, *bands = _bands(subject, thresholds, payout.tail)
    return (
        f'steps, {_order(payout.monotone)} (kind "step", monotone={payout.monotone})',
        [
            f"- nothing when {first}",
            *(
                f"- {level} when {band}"
                for level, band in zip(levels, bands, strict=True)
            ),
        ],
    )


def _piecewise_words(
    payout: PiecewiseLinearPayout, words: _Words, axis: _Axis, prefix: str
) -> tuple[str, list[str]]:
    subject, unit = axis
    more, less = _sides(payout.tail)
    knots = [
        words.value(f"{prefix}knots[{k}]", value, unit)
        for k, value in enumerate(payout.knots)
    ]
    levels = [
        words.amount(f"{prefix}levels[{k}]", value)
        for k, value in enumerate(payout.levels)
    ]
    return (
        f"straight lines between knots, {_order(payout.monotone)} "
        f'(kind "piecewise_linear", monotone={payout.monotone})',
        [
            f"- nothing when {subject} is {less} {knots[0]}",
            *(
                f"- in a straight line from {levels[k]} at {knots[k]} to "
                f"{levels[k + 1]} at {knots[k + 1]}"
                for k in range(len(knots) - 1)
            ),
            f"- {levels[-1]} when {subject} is at or {more} {knots[-1]}",
        ],
    )


def _binary_words(
    payout: BinaryPayout, words: _Words, axis: _Axis, prefix: str
) -> tuple[str, list[str]]:
    subject, unit = axis
    more, less = _sides(payout.tail)
    trigger = words.value(f"{prefix}trigger", payout.trigger, unit)
    amount = words.amount(f"{prefix}amount", payout.amount)
    return (
        'one amount from a trigger on (kind "binary")',
        [
            f"- nothing when {subject} is {less} {trigger}",
            f"- {amount} when {subject} is at or {more} "
            f"{_role(trigger, payout.trigger, 'trigger')}",
        ],
    )


def _component_words(
    payout: _OfComponents, words: _Words, axes: Mapping[str, _Axis], prefix: str
) -> list[str]:
    """Each component's payout, under its name, its parameters `<name>.<its own>`."""
    lines: list[str] = []
    for name, component in payout.components.items():
        head, items = _payout_words(
            component, words, {"index": axes[name]}, f"{prefix}{name}."
        )
        lines.append(f"- {name}: {head}")
        lines += [f"  {item}" for item in items]
    return lines


def _grid_words(
    payout: StepGrid, words: _Words, axes: Mapping[str, _Axis], prefix: str
) -> tuple[str, list[str]]:
    names = payout.index_names
    bins: list[list[str]] = []
    for name, thresholds, tail in zip(
        names, payout.breakpoints.values(), payout.tail, strict=True
    ):
        subject, unit = axes[name]
        spelled = [
            words.value(f"{prefix}{name}.breakpoints[{k}]", value, unit)
            for k, value in enumerate(thresholds)
        ]
        bins.append(_bands(subject, spelled, tail))
    cells: list[str] = []
    for i, row in enumerate(payout.levels):
        for j, level in enumerate(row):
            # The cell below both first thresholds pays nothing, by construction.
            amount = (
                "nothing"
                if (i, j) == (0, 0)
                else words.amount(f"{prefix}levels[{i},{j}]", level)
            )
            cells.append(f"- {amount} when {bins[0][i]} and {bins[1][j]}")
    return (
        f"one amount per cell of a table over {names[0]} and {names[1]}, "
        f"{_order(payout.monotone, ' along either')} "
        f'(kind "step_grid", monotone={payout.monotone})',
        cells,
    )


def _user_words(payout: Any, words: _Words, prefix: str) -> tuple[str, list[str]]:
    """A family of your own: its parameters, as it names them."""
    amounts = (
        set(payout.amount_names()) if isinstance(payout, LinearInAmounts) else set()
    )
    lines: list[str] = []
    for name, value in payout.parameters().items():
        unit = words.payout_unit if name in amounts else None
        spelled = words.value(f"{prefix}{name}", value, unit)
        lines.append(f"- {name}: {'free' if isinstance(value, Free) else spelled}")
    return (
        f'a family of your own (kind "{payout.kind}"), with its parameters as it '
        "names them",
        lines,
    )


def _payout_words(
    payout: Any, words: _Words, axes: Mapping[str, _Axis], prefix: str = ""
) -> tuple[str, list[str]]:
    """A payout's terms in words: a headline, and one line per band or part.

    `axes` maps each of the payout's `index_names` to the name a band reads
    and the unit of its thresholds. The lines start with `"- "`; a part's own
    lines are indented below it. A built-in family is described by its
    terms; a family of your own, or a subclass of a built-in one, by its
    parameters.
    """
    kind = type(payout)
    if kind is LinearPayout:
        return _linear_words(payout, words, axes["index"], prefix)
    if kind is StepPayout:
        return _step_words(payout, words, axes["index"], prefix)
    if kind is PiecewiseLinearPayout:
        return _piecewise_words(payout, words, axes["index"], prefix)
    if kind is BinaryPayout:
        return _binary_words(payout, words, axes["index"], prefix)
    if kind is SumOf:
        return (
            "the sum of one payout per component, each on its own index "
            '(kind "sum_of")',
            _component_words(payout, words, axes, prefix),
        )
    if kind is MaxOf:
        scale = words.value(f"{prefix}scale", payout.scale, None)
        return (
            f"the largest of one payout per component, times {scale}: one amount a "
            'year (kind "max_of")',
            _component_words(payout, words, axes, prefix),
        )
    if kind is StepGrid:
        return _grid_words(payout, words, axes, prefix)
    return _user_words(payout, words, prefix)


_BUILT_IN_ONE_INDEX = (LinearPayout, StepPayout, PiecewiseLinearPayout, BinaryPayout)


def _built_in(payout: Any) -> bool:
    """Whether the payout and every component of it is a built-in family."""
    if type(payout) in (SumOf, MaxOf):
        return all(_built_in(part) for part in payout.components.values())
    return type(payout) in (*_BUILT_IN_ONE_INDEX, StepGrid)


def _most(payout: Any) -> float | None:
    """The most a built-in payout pays in a year; `None` while an amount is free.

    Thresholds don't change it: every family reaches its largest amount from
    some severity on, a `SumOf` the sum of its components' and a `MaxOf` its
    scale times the largest of theirs.
    """
    kind = type(payout)
    values: list[Any]
    if kind in _BUILT_IN_ONE_INDEX:
        values = [value for _, value in payout._field_items(payout._amount_fields)]
    elif kind is StepGrid:
        values = [value for row in payout.levels for value in row]
    elif kind in (SumOf, MaxOf):
        parts = [_most(part) for part in payout.components.values()]
        if any(part is None for part in parts) or isinstance(
            getattr(payout, "scale", 1.0), Free
        ):
            return None
        known = cast(list[float], parts)
        if kind is SumOf:
            return math.fsum(known)
        return float(payout.scale) * max(known)
    else:
        return None
    if any(isinstance(value, Free) for value in values):
        return None
    return float(max(values))


def _most_words(payout: Any, payout_unit: str) -> str:
    """The most a year pays, in words, or why it is not known."""
    if not _built_in(payout):
        return "not known: a family of your own states none"
    most = _most(payout)
    if most is None:
        return "not known until the free amounts are chosen"
    return _quantity(most, payout_unit)


def _free_words(free: Free, unit: str | None) -> str:
    """A free parameter's bounds and grid, in words."""
    lower = _quantity(free.lower, unit)
    text = (
        f"from {lower}, with no upper bound"
        if free.upper is None
        else f"from {lower} to {_quantity(free.upper, unit)}"
    )
    grid = free.grid
    if grid is None:
        return text
    if len(grid) <= 10:
        return f"{text}, on the grid {', '.join(_number(value) for value in grid)}"
    return (
        f"{text}, on a grid of {len(grid)} values from {_number(grid[0])} to "
        f"{_number(grid[-1])}"
    )


_KIND = re.compile(r"^\S(?:.*\S)?$")
_PAYOUT_METHODS = ("parameters", "with_values", "evaluate", "to_document")
_REGISTRY: dict[str, type[Any]] = {}

PayoutClass = TypeVar("PayoutClass", bound=type[Any])


def register_payout(cls: PayoutClass) -> PayoutClass:
    """Register a payout class under its `kind`, so covers can read it back.

    Use it as a class decorator on a user family that implements
    `PayoutFunction`. `Cover.from_json` looks the document's `kind` up here.
    Registering the same class again does nothing. Raises
    `CoverDefinitionError` if `cls` is not a class with a non-empty string
    `kind` and the protocol's methods, or if another class already holds its
    `kind`.
    """
    if not isinstance(cls, type):
        raise CoverDefinitionError(f"register_payout takes a class, got {cls!r}")
    kind = getattr(cls, "kind", None)
    if not isinstance(kind, str) or not _KIND.fullmatch(kind):
        raise CoverDefinitionError(
            f"{cls.__qualname__}.kind must be a non-empty string without "
            "surrounding whitespace"
        )
    missing = [
        name
        for name in (*_PAYOUT_METHODS, "from_document")
        if not callable(getattr(cls, name, None))
    ]
    if missing:
        raise CoverDefinitionError(
            f"{cls.__qualname__} does not implement PayoutFunction: "
            f"missing {', '.join(missing)}"
        )
    registered = _REGISTRY.get(kind)
    if registered is not None and registered is not cls:
        raise CoverDefinitionError(
            f"payout kind {kind!r} is already registered by {registered.__qualname__}"
        )
    _REGISTRY[kind] = cls
    return cls


def _payout_reader(kind: Any) -> Callable[[Mapping[str, Any]], Any]:
    """Return the registered `from_document` of `kind`."""
    if not isinstance(kind, str):
        raise CoverDefinitionError(
            f"a payout document needs a string kind, got {kind!r}"
        )
    cls = _REGISTRY.get(kind)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownPayoutKindError(
            f"unknown payout kind {kind!r}; registered kinds: {known}. "
            "Register a user family with register_payout."
        )
    reader: Callable[[Mapping[str, Any]], Any] = cls.from_document
    return reader


for _family in (
    LinearPayout,
    StepPayout,
    PiecewiseLinearPayout,
    BinaryPayout,
    SumOf,
    MaxOf,
    StepGrid,
):
    register_payout(_family)
del _family
