"""Pricing: the pure premium, the cost-of-capital load, a multiple and an expense."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import (
    CostOfCapitalLoad,
    CoverDefinitionError,
    ExpectedPayoutMultiple,
    PurePremium,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.pricing import pricing_from_document

UNTYPED_LOAD: Any = CostOfCapitalLoad
UNTYPED_MULTIPLE: Any = ExpectedPayoutMultiple
PAYOUT = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
PROBABILITY = np.full(5, 0.2)


def test_pure_premium_is_the_expected_payout() -> None:
    assert PurePremium().premium(PAYOUT, PROBABILITY) == pytest.approx(2.0)
    assert PurePremium().to_document() == {"kind": "pure"}
    assert PurePremium() == PurePremium()


def test_cost_of_capital_adds_a_load_on_the_payout_tail() -> None:
    pricing = CostOfCapitalLoad(0.5, 0.8)
    # CVaR_0.8 of the payout is the worst 20 %: the year paying 10.
    assert pricing.premium(PAYOUT, PROBABILITY) == pytest.approx(0.5 * 2 + 0.5 * 10)
    assert CostOfCapitalLoad(0.0, 0.9).premium(PAYOUT, PROBABILITY) == pytest.approx(
        2.0
    )
    assert CostOfCapitalLoad(rate=1.0, q=0.5).premium(PAYOUT, PROBABILITY) == (
        pytest.approx(4.0)
    )
    assert pricing.to_document() == {"kind": "cost_of_capital", "q": 0.8, "rate": 0.5}


def test_a_multiple_of_the_expected_payout_plus_an_expense() -> None:
    # E[payout] = 2: 1.5 x 2 + 100.
    pricing = ExpectedPayoutMultiple(1.5, 100.0)
    assert pricing.premium(PAYOUT, PROBABILITY) == pytest.approx(103.0)
    assert ExpectedPayoutMultiple(multiple=2.0, expense=0.5).premium(
        PAYOUT, PROBABILITY
    ) == pytest.approx(4.5)
    # A cover that never pays costs the expense alone.
    assert pricing.premium(np.zeros(5), PROBABILITY) == 100.0
    # The expense defaults to 0; a multiple of 1 without it is the pure premium.
    assert ExpectedPayoutMultiple(3.0) == ExpectedPayoutMultiple(3.0, 0.0)
    assert ExpectedPayoutMultiple(3.0).premium(PAYOUT, PROBABILITY) == 6.0
    assert ExpectedPayoutMultiple(1.0).premium(PAYOUT, PROBABILITY) == (
        PurePremium().premium(PAYOUT, PROBABILITY)
    )
    assert pricing.to_document() == {
        "expense": 100.0,
        "kind": "expected_payout_multiple",
        "multiple": 1.5,
    }
    # Whole numbers are floats in the document, as for the other pricings.
    assert ExpectedPayoutMultiple(2, 10).to_document() == {
        "expense": 10.0,
        "kind": "expected_payout_multiple",
        "multiple": 2.0,
    }


def test_the_premium_is_linear_in_the_expected_payout() -> None:
    rng = np.random.default_rng(7)
    payout = rng.gamma(0.5, 30.0, 40)
    probability = np.full(40, 1 / 40)
    pricing = ExpectedPayoutMultiple(1.8, 25.0)
    expected = math.fsum((probability * payout).tolist())
    assert pricing.premium(payout, probability) == pytest.approx(
        1.8 * expected + 25.0, rel=1e-15
    )
    doubled = pricing.premium(2.0 * payout, probability)
    assert doubled - 25.0 == pytest.approx(2.0 * (1.8 * expected), rel=1e-15)


def test_the_documents_and_hashes_of_the_other_pricings_are_unchanged() -> None:
    # A design's id hashes its pricing's document, so these must not move.
    assert content_hash(PurePremium().to_document()) == content_hash({"kind": "pure"})
    assert CostOfCapitalLoad(0.06, 0.99).to_document() == {
        "kind": "cost_of_capital",
        "q": 0.99,
        "rate": 0.06,
    }
    documents = {
        json.dumps(pricing.to_document(), sort_keys=True)
        for pricing in (
            PurePremium(),
            CostOfCapitalLoad(0.0, 0.9),
            ExpectedPayoutMultiple(1.0),
            ExpectedPayoutMultiple(1.0, 1.0),
        )
    }
    assert len(documents) == 4


@pytest.mark.parametrize(
    "document",
    [
        {"kind": "pure"},
        {"kind": "cost_of_capital", "q": 0.99, "rate": 0.06},
        {"expense": 0.0, "kind": "expected_payout_multiple", "multiple": 1.0},
        {"expense": 2500.0, "kind": "expected_payout_multiple", "multiple": 1.35},
    ],
)
def test_documents_round_trip(document: dict[str, Any]) -> None:
    assert pricing_from_document(document).to_document() == document


@pytest.mark.parametrize(
    ("rate", "q"),
    [
        (-0.1, 0.9),
        (1.1, 0.9),
        (0.1, 0.0),
        (0.1, 1.0),
        (math.nan, 0.9),
        (True, 0.9),
        ("0.1", 0.9),
    ],
)
def test_invalid_terms_raise(rate: Any, q: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        UNTYPED_LOAD(rate, q)


@pytest.mark.parametrize(
    ("multiple", "expense", "match"),
    [
        (0.99, 0.0, "multiple must be at least 1"),
        (0.0, 0.0, "multiple must be at least 1"),
        (-1.5, 0.0, "multiple must be at least 1"),
        (1.5, -0.01, "expense must be >= 0"),
        (math.inf, 0.0, None),
        (1.5, math.nan, None),
        (True, 0.0, None),
        (1.5, "10", None),
        (None, 0.0, None),
    ],
)
def test_invalid_multiples_and_expenses_raise(
    multiple: Any, expense: Any, match: str | None
) -> None:
    with pytest.raises(CoverDefinitionError, match=match):
        UNTYPED_MULTIPLE(multiple, expense)


def test_a_multiple_needs_its_multiple() -> None:
    with pytest.raises(TypeError):
        UNTYPED_MULTIPLE()


@pytest.mark.parametrize(
    "document",
    [
        None,
        {"kind": "loaded"},
        {"kind": "pure", "rate": 0.1},
        {"kind": "cost_of_capital", "rate": 0.1},
        {"kind": "cost_of_capital", "q": 2.0, "rate": 0.1},
        {"kind": "expected_payout_multiple", "multiple": 1.5},
        {"expense": 1.0, "kind": "expected_payout_multiple"},
        {"expense": 1.0, "kind": "expected_payout_multiple", "multiple": 0.5},
        {"expense": -1.0, "kind": "expected_payout_multiple", "multiple": 1.5},
        {"expense": 1.0, "kind": "expected_payout_multiple", "multiple": 1.5, "q": 1},
    ],
)
def test_invalid_documents_raise(document: Any) -> None:
    with pytest.raises(CoverDefinitionError):
        pricing_from_document(document)


def test_invalid_payouts_raise() -> None:
    with pytest.raises(ValueError):
        PurePremium().premium(np.array([1.0, np.nan]), np.full(2, 0.5))
    with pytest.raises(ValueError):
        CostOfCapitalLoad(0.1, 0.9).premium(np.ones(3), np.full(2, 0.5))
    with pytest.raises(ValueError):
        ExpectedPayoutMultiple(1.5, 1.0).premium(np.array([1.0, np.inf]), np.ones(2))
    with pytest.raises(ValueError):
        ExpectedPayoutMultiple(1.5).premium(np.ones(2), np.array([1.0, 0.0]))
