"""Pricing: the premium of a cover's payouts on an outcome sample.

- `PurePremium()`: `P = E[payout]`.
- `CostOfCapitalLoad(rate, q)`: `P = (1 - rate) * E[payout] + rate *
  CVaR_q(payout)`, with `rate` in [0, 1] and `q` in (0, 1). The load is the
  cost of holding capital against the payout's tail. It is convex in the
  payout, so a design under it stays a linear or quadratic programme.
- `ExpectedPayoutMultiple(multiple, expense=0.0)`: `P = multiple *
  E[payout] + expense`, with `multiple >= 1` and `expense >= 0`, a fixed
  amount in the payout unit of the covers it prices. It is the price an
  insurer or a broker usually states first: a multiple of the expected
  payout and a fixed expense per contract and year. It is linear in the
  payout, so a design's budget stays one linear row.

The CVaR is over the years of the sample, as everywhere in this module (see
`crc_sdk.parametric.metrics`); it is not crc-framework's CVaR over binary
outcomes. The pricings are frozen value objects with a JSON document
(`to_document()`), which evaluation metadata records next to every premium.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self, TypeAlias

from numpy.typing import ArrayLike
from pydantic import model_validator

from crc_sdk.impacts._types import FiniteFloat
from crc_sdk.parametric._risk import Draws
from crc_sdk.parametric.errors import CoverDefinitionError
from crc_sdk.parametric.parameters import _DefinitionModel

__all__ = [
    "CostOfCapitalLoad",
    "ExpectedPayoutMultiple",
    "Pricing",
    "PurePremium",
    "pricing_from_document",
]


class PurePremium(_DefinitionModel):
    """The pure premium: the expected payout, `P = E[payout]`."""

    kind: Literal["pure"] = "pure"

    def __init__(self) -> None:
        super().__init__()

    def premium(self, payout: ArrayLike, probability: ArrayLike) -> float:
        """Return `sum(probability * payout)`.

        Raises `ValueError` for arrays of different shapes, non-finite
        payouts or non-positive probabilities.
        """
        return Draws(payout, probability).mean()

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "pure"}`."""
        return {"kind": "pure"}


class CostOfCapitalLoad(_DefinitionModel):
    """The pure premium plus a cost-of-capital load on the payout's tail.

    `P = (1 - rate) * E[payout] + rate * CVaR_q(payout)`, with `rate` in
    [0, 1] and `q` in (0, 1). Raises `CoverDefinitionError` otherwise.
    """

    kind: Literal["cost_of_capital"] = "cost_of_capital"
    rate: FiniteFloat
    q: FiniteFloat

    def __init__(self, rate: float, q: float) -> None:
        super().__init__(rate=rate, q=q)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 0.0 <= self.rate <= 1.0:
            raise CoverDefinitionError(f"rate must lie in [0, 1], got {self.rate!r}")
        if not 0.0 < self.q < 1.0:
            raise CoverDefinitionError(f"q must lie in (0, 1), got {self.q!r}")
        return self

    def premium(self, payout: ArrayLike, probability: ArrayLike) -> float:
        """Return `(1 - rate) * E[payout] + rate * CVaR_q(payout)`.

        Raises `ValueError` for arrays of different shapes, non-finite
        payouts or non-positive probabilities.
        """
        draws = Draws(payout, probability)
        return (1.0 - self.rate) * draws.mean() + self.rate * draws.cvar(self.q) + 0.0

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "cost_of_capital", "q": q, "rate": rate}`."""
        return {"kind": "cost_of_capital", "q": self.q, "rate": self.rate}


class ExpectedPayoutMultiple(_DefinitionModel):
    """A multiple of the expected payout plus a fixed expense.

    `P = multiple * E[payout] + expense`, with `multiple >= 1` and
    `expense >= 0`. `expense` is a fixed amount per contract and year, in the
    payout unit of the covers it prices; it defaults to 0. `multiple` has no
    default: 1 with no expense is the pure premium, which `PurePremium()`
    states. Raises `CoverDefinitionError` for terms out of range.
    """

    kind: Literal["expected_payout_multiple"] = "expected_payout_multiple"
    multiple: FiniteFloat
    expense: FiniteFloat = 0.0

    def __init__(self, multiple: float, expense: float = 0.0) -> None:
        super().__init__(multiple=multiple, expense=expense)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.multiple < 1.0:
            raise CoverDefinitionError(
                f"multiple must be at least 1, got {self.multiple!r}"
            )
        if self.expense < 0.0:
            raise CoverDefinitionError(f"expense must be >= 0, got {self.expense!r}")
        return self

    def premium(self, payout: ArrayLike, probability: ArrayLike) -> float:
        """Return `multiple * sum(probability * payout) + expense`.

        Raises `ValueError` for arrays of different shapes, non-finite
        payouts or non-positive probabilities.
        """
        return self.multiple * Draws(payout, probability).mean() + self.expense + 0.0

    def to_document(self) -> dict[str, Any]:
        """Return the kind, `expense` and `multiple`, as plain JSON types."""
        return {
            "expense": self.expense,
            "kind": "expected_payout_multiple",
            "multiple": self.multiple,
        }


Pricing: TypeAlias = PurePremium | CostOfCapitalLoad | ExpectedPayoutMultiple
"""The pricings a cover's premium can be computed with."""

_PRICINGS = (PurePremium, CostOfCapitalLoad, ExpectedPayoutMultiple)
_PRICINGS_TEXT = (
    "PurePremium(), CostOfCapitalLoad(rate, q) or "
    "ExpectedPayoutMultiple(multiple, expense)"
)


def _check_pricing(pricing: Any) -> Pricing:
    """Return `pricing` if it is one of the pricings; raise `TypeError` if not."""
    if not isinstance(pricing, _PRICINGS):
        raise TypeError(
            f"pricing must be {_PRICINGS_TEXT}, got {type(pricing).__name__}"
        )
    return pricing


def _expense(pricing: Pricing) -> float:
    """The fixed part of a pricing's premium: what a cover that never pays costs."""
    return pricing.expense if isinstance(pricing, ExpectedPayoutMultiple) else 0.0


def pricing_from_document(document: Any) -> Pricing:
    """Read a pricing document, as `to_document()` writes it.

    Raises `CoverDefinitionError` for an unknown kind or invalid terms.
    """
    if not isinstance(document, Mapping):
        raise CoverDefinitionError("a pricing document must be an object")
    kind = document.get("kind")
    if kind == "pure":
        if set(document) != {"kind"}:
            raise CoverDefinitionError('a pure premium document is {"kind": "pure"}')
        return PurePremium()
    if kind == "cost_of_capital":
        if set(document) != {"kind", "q", "rate"}:
            raise CoverDefinitionError(
                "a cost-of-capital document has the keys kind, q and rate"
            )
        return CostOfCapitalLoad(document["rate"], document["q"])
    if kind == "expected_payout_multiple":
        if set(document) != {"expense", "kind", "multiple"}:
            raise CoverDefinitionError(
                "an expected-payout-multiple document has the keys expense, kind "
                "and multiple"
            )
        return ExpectedPayoutMultiple(document["multiple"], document["expense"])
    raise CoverDefinitionError(f"unknown pricing kind {kind!r}")
