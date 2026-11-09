"""The rule that makes one year's loss out of the impacts of several hazards.

A sample of several hazards has one loss per year, and how the hazards' impacts
make that loss is a statement about the asset, so a sample states it and no rule
is assumed. Each component's impact comes from its `LossModel`, on its hazard,
after the basis-risk model. A rule takes those impacts, one array per component,
and the combined loss is the rule's result times the value.

- `CappedSum(cap)` is the sum of the impacts, at most `cap`.
- `CompoundDamage()` is `1 - product of (1 - impact)`: each hazard damages what
  the others leave. Every impact must lie in [0, 1].
- A user rule implements `CombineRule`.

So `CappedSum(1.0)` of 0.7 and 0.6 is 1.0, and `CompoundDamage()` of them is
`1 - 0.3 * 0.4 = 0.88`. Which rule suits an asset is the caller's choice, and the
module says nothing about it. A rule that caps or compounds makes the combined
loss smaller than the sum of the hazards' own losses in some years.

Every rule, built in or not, passes one gate before its sample is returned. Its
document is plain JSON with a `kind`, its sentence is a non-empty string and its
result has one finite, non-negative value per year. A user rule cannot take the
`kind` of a built-in, so nothing else can be recorded as one. A sample records
the document and the sentence as its provenance's `combine`, and `setup_text`
prints the sentence next to the numbers.

This module needs numpy only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from crc_sdk.parametric.basis_risk import _positive
from crc_sdk.parametric.errors import SampleError

__all__ = ["CappedSum", "CombineRule", "CompoundDamage"]

FloatArray = NDArray[np.float64]

_ONE_IMPACT = "one_impact"
"""The `combine` kind of a sample in which only one component has an impact."""

_STATEMENT = "statement"
"""The key of a recorded rule that holds its sentence."""


@runtime_checkable
class CombineRule(Protocol):
    """What every rule for a combined loss provides.

    `to_document()` returns the rule's terms as plain JSON types, with a
    `kind` that is not a built-in's and no `statement` key. `statement()`
    returns one sentence for the setup text. `combine(impacts)` takes each
    component's impact, by name in component order, as read-only `(n,)` arrays,
    and returns the rule's result: `(n,)`, finite and non-negative. The combined
    loss is that result times the value.
    """

    def to_document(self) -> dict[str, Any]: ...

    def statement(self) -> str: ...

    def combine(self, impacts: Mapping[str, FloatArray]) -> FloatArray: ...


@dataclass(frozen=True)
class CappedSum:
    """The sum of the impacts, at most `cap`, in the impacts' unit.

    Raises `SampleError` unless `cap` is finite and positive.
    """

    cap: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cap", _positive("cap", self.cap))

    def to_document(self) -> dict[str, Any]:
        """Return the kind and the cap."""
        return {"kind": "capped_sum", "cap": self.cap}

    def statement(self) -> str:
        """Return the sentence that states this rule next to a number."""
        return (
            f"The impacts are summed and capped at {self.cap!r}, then multiplied by "
            "the value."
        )

    def combine(self, impacts: Mapping[str, FloatArray]) -> FloatArray:
        """Return `min(sum of the impacts, cap)`, one value per year."""
        total = np.stack(list(impacts.values())).sum(axis=0)
        capped: FloatArray = np.minimum(total, self.cap)
        return capped


@dataclass(frozen=True)
class CompoundDamage:
    """`1 - product of (1 - impact)`: each hazard damages what the others leave.

    Every impact must lie in [0, 1], so that it is a share of the value.
    """

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "compound_damage"}`."""
        return {"kind": "compound_damage"}

    def statement(self) -> str:
        """Return the sentence that states this rule next to a number."""
        return (
            "Each hazard damages what the others leave: 1 minus the product of "
            "(1 minus each impact), then multiplied by the value."
        )

    def combine(self, impacts: Mapping[str, FloatArray]) -> FloatArray:
        """Return `1 - product of (1 - impact)`, one value per year.

        Raises `SampleError` if an impact lies outside [0, 1], as `ImpactNoise`
        without `upper=1` can make one.
        """
        for name, impact in impacts.items():
            if not ((impact >= 0.0) & (impact <= 1.0)).all():
                raise SampleError(
                    f"CompoundDamage needs every impact in [0, 1], but component "
                    f"{name!r} has one outside it; ImpactNoise without upper=1 can "
                    "put one there"
                )
        kept = np.prod(1.0 - np.stack(list(impacts.values())), axis=0)
        combined: FloatArray = 1.0 - kept
        return combined


_BUILT_IN_RULES = (CappedSum, CompoundDamage)
_BUILT_IN_KINDS = frozenset({"capped_sum", "compound_damage", _ONE_IMPACT})


def _one_impact_record(name: str) -> dict[str, Any]:
    """What a sample records when only component `name` has an impact."""
    return {
        "kind": _ONE_IMPACT,
        "component": name,
        _STATEMENT: (
            f"Only component {name} has an impact, so the combined loss is its "
            "own loss."
        ),
    }


def _rule_record(rule: Any) -> dict[str, Any]:
    """Check a rule's document and sentence; return what a sample records.

    Raises `TypeError` for anything but a `CombineRule`, and `SampleError` for
    a document or a sentence that doesn't pass the gate.
    """
    if not isinstance(rule, CombineRule):
        raise TypeError(
            "combine must be a CombineRule, such as CappedSum(cap) or "
            "CompoundDamage(), with to_document, statement and combine: every "
            f"rule has a document; got {type(rule).__name__}"
        )
    document = rule.to_document()
    if not isinstance(document, Mapping):
        raise SampleError(
            f"a rule's to_document() must return a mapping, got {document!r}"
        )
    try:
        json.dumps(document, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise SampleError(
            f"a rule's document must hold plain JSON values: {error}"
        ) from error
    kind = document.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise SampleError(f"a rule's document must name its kind, got {kind!r}")
    if kind in _BUILT_IN_KINDS and type(rule) not in _BUILT_IN_RULES:
        raise SampleError(
            f"a user's rule cannot take the kind of a built-in, {kind!r}: name "
            "another, so that the record says which rule made the loss"
        )
    if _STATEMENT in document:
        raise SampleError(
            f"a rule's document must not hold a {_STATEMENT!r} key: statement() "
            "gives the sentence"
        )
    statement = rule.statement()
    if not isinstance(statement, str) or not statement.strip():
        raise SampleError(
            f"rule {kind!r} must state itself: statement() must return a non-empty "
            f"string, got {statement!r}"
        )
    return {**document, _STATEMENT: statement}


def _combined_impact(
    rule: CombineRule, kind: str, impacts: Mapping[str, FloatArray]
) -> FloatArray:
    """Return the checked result of `rule` on the components' impacts.

    Raises `SampleError` unless it has one finite, non-negative value per year.
    """
    n = next(iter(impacts.values())).shape[0]
    arrays = {name: impact.copy() for name, impact in impacts.items()}
    for array in arrays.values():
        array.setflags(write=False)
    result = rule.combine(MappingProxyType(arrays))
    try:
        combined = np.array(result, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError(f"rule {kind!r} returned non-numbers") from error
    if combined.shape != (n,):
        raise SampleError(
            f"rule {kind!r} must return one value per year, shape ({n},); got "
            f"{combined.shape}"
        )
    if not np.isfinite(combined).all() or (combined < 0).any():
        raise SampleError(f"rule {kind!r} returned negative or non-finite values")
    return combined
