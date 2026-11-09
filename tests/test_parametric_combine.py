"""`CappedSum`, `CompoundDamage`, `CombineRule` and the gate every rule passes."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from crc_sdk.parametric import CappedSum, CombineRule, CompoundDamage, SampleError
from crc_sdk.parametric.combine import _combined_impact, _rule_record

UNTYPED_CAPPED: Any = CappedSum


def _both(first: list[float], second: list[float]) -> dict[str, NDArray[np.float64]]:
    return {
        "rain": np.array(first, dtype=np.float64),
        "heat": np.array(second, dtype=np.float64),
    }


class Rule:
    """A user's rule, whose answers a test can replace one at a time."""

    def __init__(
        self,
        *,
        document: Any = None,
        statement: Any = "The largest impact of the year.",
        combine: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.document = {"kind": "worst"} if document is None else document
        self.sentence = statement
        self.rule = combine

    def to_document(self) -> Any:
        return self.document

    def statement(self) -> Any:
        return self.sentence

    def combine(self, impacts: Mapping[str, Any]) -> Any:
        if self.rule is not None:
            return self.rule(impacts)
        return np.stack(list(impacts.values())).max(axis=0)


# -- the built-in rules ---------------------------------------------------------


def test_a_capped_sum_of_0_7_and_0_6_is_1() -> None:
    assert CappedSum(1.0).combine(_both([0.7], [0.6])).tolist() == [1.0]


def test_compound_damage_of_0_7_and_0_6_is_0_88() -> None:
    assert CompoundDamage().combine(_both([0.7], [0.6]))[0] == pytest.approx(0.88)


def test_a_capped_sum_adds_below_its_cap() -> None:
    combined = CappedSum(0.9).combine(_both([0.1, 0.5, 0.0], [0.2, 0.5, 0.0]))
    np.testing.assert_allclose(combined, [0.3, 0.9, 0.0])


def test_compound_damage_reduces_to_the_one_hazard_that_hits() -> None:
    combined = CompoundDamage().combine(_both([0.4, 0.0, 1.0], [0.0, 0.0, 0.5]))
    np.testing.assert_allclose(combined, [0.4, 0.0, 1.0])


def test_compound_damage_of_three_hazards() -> None:
    impacts = {**_both([0.5], [0.5]), "wind": np.array([0.5])}
    assert CompoundDamage().combine(impacts)[0] == pytest.approx(0.875)


@pytest.mark.parametrize(
    "impact", [-0.1, 1.1, float("nan")], ids=["negative", "above-1", "nan"]
)
def test_compound_damage_refuses_an_impact_outside_0_and_1(impact: float) -> None:
    with pytest.raises(SampleError, match=r"component 'heat' has one outside it"):
        CompoundDamage().combine(_both([0.1], [impact]))


@pytest.mark.parametrize(
    "cap", [0.0, -1.0, float("inf"), float("nan"), True, "1"], ids=repr
)
def test_a_cap_is_finite_and_positive(cap: Any) -> None:
    with pytest.raises(SampleError, match="cap must be"):
        UNTYPED_CAPPED(cap)


def test_the_documents_say_what_each_rule_is() -> None:
    assert CappedSum(2).to_document() == {"kind": "capped_sum", "cap": 2.0}
    assert CompoundDamage().to_document() == {"kind": "compound_damage"}


def test_the_rules_state_themselves() -> None:
    assert CappedSum(1.0).statement() == (
        "The impacts are summed and capped at 1.0, then multiplied by the value."
    )
    assert CompoundDamage().statement() == (
        "Each hazard damages what the others leave: 1 minus the product of "
        "(1 minus each impact), then multiplied by the value."
    )


@pytest.mark.parametrize("rule", [CappedSum(1.0), CompoundDamage()], ids=repr)
def test_the_rules_are_frozen_values_that_pickle(rule: Any) -> None:
    assert isinstance(rule, CombineRule)
    assert pickle.loads(pickle.dumps(rule)) == rule
    with pytest.raises(AttributeError):
        setattr(rule, "cap", 2.0)


# -- the gate -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "kept"),
    [
        (CappedSum(1.0), {"kind": "capped_sum", "cap": 1.0}),
        (CompoundDamage(), {"kind": "compound_damage"}),
        (Rule(document={"kind": "worst", "terms": [1, 2]}), None),
    ],
    ids=["capped-sum", "compound-damage", "user"],
)
def test_every_rule_is_recorded_with_its_sentence(rule: Any, kept: Any) -> None:
    record = _rule_record(rule)
    assert record == {
        **(kept if kept is not None else rule.to_document()),
        "statement": rule.statement(),
    }


@pytest.mark.parametrize(
    "rule",
    [lambda impacts: impacts, np.minimum, "capped_sum", None, 1.0],
    ids=["callable", "ufunc", "string", "none", "number"],
)
def test_anything_but_a_rule_is_a_type_error(rule: Any) -> None:
    with pytest.raises(TypeError, match="every rule has a document"):
        _rule_record(rule)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        pytest.param(Rule(document=[1]), "must return a mapping", id="not-mapping"),
        pytest.param(Rule(document={"kind": object()}), "plain JSON", id="not-json"),
        pytest.param(
            Rule(document={"kind": "k", "x": float("nan")}), "plain JSON", id="nan"
        ),
        pytest.param(Rule(document={"cap": 1.0}), "name its kind", id="no-kind"),
        pytest.param(Rule(document={"kind": " "}), "name its kind", id="blank-kind"),
        pytest.param(Rule(document={"kind": 3}), "name its kind", id="kind-not-str"),
        pytest.param(
            Rule(document={"kind": "capped_sum", "cap": 1.0}),
            "kind of a built-in",
            id="capped-sum-kind",
        ),
        pytest.param(
            Rule(document={"kind": "compound_damage"}),
            "kind of a built-in",
            id="compound-kind",
        ),
        pytest.param(
            Rule(document={"kind": "one_impact"}),
            "kind of a built-in",
            id="one-impact-kind",
        ),
        pytest.param(
            Rule(document={"kind": "k", "statement": "x"}),
            "must not hold a 'statement' key",
            id="statement-key",
        ),
        pytest.param(Rule(statement=""), "must state itself", id="empty-statement"),
        pytest.param(Rule(statement="  "), "must state itself", id="blank-statement"),
        pytest.param(Rule(statement=1), "must state itself", id="statement-not-str"),
    ],
)
def test_a_document_or_sentence_that_fails_the_gate_raises(
    rule: Any, message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _rule_record(rule)


def test_a_subclass_of_a_built_in_is_a_users_rule() -> None:
    class Changed(CappedSum):
        def combine(self, impacts: Mapping[str, Any]) -> Any:
            return np.zeros(1)

    with pytest.raises(SampleError, match="kind of a built-in"):
        _rule_record(Changed(1.0))


@pytest.mark.parametrize(
    ("result", "message"),
    [
        pytest.param([0.1, 0.2, 0.3], r"shape \(2,\); got \(3,\)", id="too-long"),
        pytest.param([[0.1], [0.2]], r"shape \(2,\); got \(2, 1\)", id="two-d"),
        pytest.param(0.5, r"shape \(2,\); got \(\)", id="scalar"),
        pytest.param([0.1, float("nan")], "negative or non-finite", id="nan"),
        pytest.param([0.1, float("inf")], "negative or non-finite", id="inf"),
        pytest.param([0.1, -0.2], "negative or non-finite", id="negative"),
        pytest.param(["a", "b"], "non-numbers", id="strings"),
        pytest.param(None, "non-finite|shape", id="none"),
    ],
)
def test_a_result_that_fails_the_gate_raises(result: Any, message: str) -> None:
    rule = Rule(combine=lambda impacts: result)
    with pytest.raises(SampleError, match=message):
        _combined_impact(rule, "worst", _both([0.1, 0.2], [0.3, 0.4]))


@pytest.mark.parametrize(
    "rule",
    [CappedSum(1.0), CompoundDamage(), Rule()],
    ids=["capped", "compound", "user"],
)
def test_the_gate_passes_built_in_and_user_rules_alike(rule: Any) -> None:
    combined = _combined_impact(rule, "k", _both([0.1, 0.2], [0.3, 0.4]))
    assert combined.shape == (2,)
    assert (combined >= 0).all()


def test_a_rule_reads_read_only_copies_in_component_order() -> None:
    seen: list[str] = []

    def combine(impacts: Mapping[str, Any]) -> Any:
        seen.extend(impacts)
        with pytest.raises(ValueError, match="read-only"):
            impacts["heat"][0] = 9.0
        with pytest.raises(TypeError):
            impacts["rain"] = 0  # type: ignore[index]
        return impacts["heat"]

    given = _both([0.1], [0.2])
    combined = _combined_impact(Rule(combine=combine), "k", given)
    assert seen == ["rain", "heat"]
    assert given["heat"].flags.writeable
    assert combined.flags.writeable
    assert not np.shares_memory(combined, given["heat"])
