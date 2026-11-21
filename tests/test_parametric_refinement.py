"""The local refinement after a grid: `DesignOptions(refine=True)`.

From the grid's best point over the free thresholds, with the amounts solved
exactly at every point it evaluates: over the sample's index values for a
step or binary payout, by a compass search otherwise. The cases are
synthetic: losses made by a payout of the same family, so the optimum is
known in closed form (an objective of 0 at the generating thresholds).
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CompositeIndex,
    CostOfCapitalLoad,
    Cover,
    DesignError,
    DesignOptions,
    DesignResult,
    EvaluationMetadata,
    ExpectedPayoutMultiple,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    Joint,
    Limit,
    LinearConstraints,
    LinearPayout,
    MaxOf,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    PiecewiseLinearPayout,
    PurePremium,
    SampleProvenance,
    StepGrid,
    StepPayout,
    SumOf,
    UserObjective,
    WeightedCVaR,
    _solver,
    design,
    design_frontier,
    design_joint,
    evaluate,
    setup_text,
)
from crc_sdk.parametric.design import (
    INDEX_VALUES_METHOD,
    REFINE_MAX_POINTS,
    REFINE_METHOD,
    REFINE_TOLERANCE,
)
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.objectives import Objective
from crc_sdk.parametric.parameters import _DefinitionModel
from crc_sdk.types import SourceProvenance

DESIGN_MODULE = importlib.import_module("crc_sdk.parametric.design")
INDEX = IndexDefinition("wind", "m/s")
ATTACHMENT, EXHAUSTION, AMOUNT = 3.7, 8.0, 100.0


def _sample(index: Sequence[float], loss: Sequence[float]) -> OutcomeSample:
    n = len(index)
    return OutcomeSample(
        asset_ids=("A",),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=np.asarray(index, dtype=np.float64),
        loss=np.asarray(loss, dtype=np.float64),
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
        ),
    )


def _linear_sample(
    attachment: float = ATTACHMENT, exhaustion: float = EXHAUSTION, seed: int = 1
) -> OutcomeSample:
    """200 years whose loss is a linear payout of the index: its optimum is known."""
    index = np.random.default_rng(seed).uniform(0.0, 10.0, 200)
    loss = AMOUNT * np.clip((index - attachment) / (exhaustion - attachment), 0, 1)
    return _sample(index.tolist(), loss.tolist())


SAMPLE = _linear_sample()
GRID = [0.0, 2.5, 5.0, 7.5]


def _linear(
    attachment: float | Free = Free(0.0, 10.0, grid=GRID),
    exhaustion: float | Free = EXHAUSTION,
) -> Cover:
    return Cover(INDEX, LinearPayout(attachment, exhaustion, Free()), payout_unit="EUR")


OBJECTIVES: list[tuple[Objective, dict[str, Any]]] = [
    (MSE(), {}),
    (ExpectedUncovered(), {"budget": Budget(ratio=1.0)}),
    (WeightedCVaR(0.5, 0.9), {"budget": Budget(ratio=1.0)}),
]
OBJECTIVE_IDS = ["mse", "eu", "wcvar"]
REFINE = DesignOptions(refine=True)


def _scaled(result: DesignResult, value: float) -> float:
    """An objective value in the scaled units the design compares in."""
    scale = float(result.design_sample.loss.max())
    homogeneity = 2 if result.objective["kind"] == "mse" else 1
    return value / scale**homogeneity


def _rows(result: DesignResult) -> list[dict[str, Any]]:
    assert result.refinement_profile is not None
    rows: list[dict[str, Any]] = result.refinement_profile.to_pylist()
    return rows


# -- the option --------------------------------------------------------------


def test_the_option_is_off_and_leaves_the_options_document_as_it_was() -> None:
    assert DesignOptions().to_document() == {
        "feasibility_tolerance": 1e-7,
        "fill": "lowest",
        "max_grid_points": 10_000,
        "ranges": False,
    }
    assert REFINE.to_document() == {
        "feasibility_tolerance": 1e-7,
        "fill": "lowest",
        "max_grid_points": 10_000,
        "ranges": False,
        "refine": True,
        "refine_max_points": REFINE_MAX_POINTS,
        "refine_tolerance": REFINE_TOLERANCE,
    }
    assert (REFINE_TOLERANCE, REFINE_MAX_POINTS) == (1e-4, 1_000)


@pytest.mark.parametrize(
    "bad",
    [
        {"refine": 1},
        {"refine": True, "refine_tolerance": 0.0},
        {"refine": True, "refine_tolerance": 1.0},
        {"refine": True, "refine_tolerance": math.nan},
        {"refine": True, "refine_max_points": 0},
        {"refine": True, "refine_max_points": 2.0},
        {"refine_tolerance": 1e-6},
        {"refine_max_points": 10},
    ],
)
def test_invalid_refinement_options_raise(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DesignOptions(**bad)


def test_without_the_option_a_grid_design_is_not_refined() -> None:
    plain = design(_linear(), SAMPLE, MSE())
    assert plain.refinement is None and plain.refinement_profile is None
    assert "refinement" not in plain.to_record()
    assert plain.parameters["attachment"] == 2.5
    # The id and the solves of this design before the option existed.
    assert plain.design_id == (
        "sha256:5c62ad607bd73926d6383887cbd07d664a5dde2ff0c6ddb787b01d6584b72ac2"
    )
    assert plain.solves == 5


def test_the_option_enters_the_design_id() -> None:
    plain = design(_linear(), SAMPLE, MSE())
    refined = design(_linear(), SAMPLE, MSE(), options=REFINE)
    finer = design(
        _linear(),
        SAMPLE,
        MSE(),
        options=DesignOptions(refine=True, refine_tolerance=1e-6),
    )
    assert len({plain.design_id, refined.design_id, finer.design_id}) == 3


# -- the search --------------------------------------------------------------


@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_an_optimum_between_grid_points_is_found(
    objective: Objective, terms: dict[str, Any]
) -> None:
    # The loss is the linear payout attached at 3.7, between the grid's 2.5 and
    # 5.0: there the objective is 0, and anywhere else it is above 0.
    plain = design(_linear(), SAMPLE, objective, **terms)
    refined = design(_linear(), SAMPLE, objective, options=REFINE, **terms)
    assert plain.parameters["attachment"] == 2.5
    assert refined.route == "grid" and refined.amounts == plain.amounts
    # Found to a stated tolerance: the attachment within 1e-3 of its bounds'
    # width, and the objective within 1e-5 of the optimum, 0, in units of the
    # largest loss. MSE stops at the tie tolerance (1e-7) first: its gains near
    # the optimum shrink with the square of the distance.
    assert refined.parameters["attachment"] == pytest.approx(ATTACHMENT, abs=1e-2)
    assert refined.parameters["limit"] == pytest.approx(AMOUNT, rel=1e-3)
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-5
    assert refined.objective_value < plain.objective_value
    record = refined.refinement
    assert record is not None
    assert record["method"] == "compass_search"
    assert record["start"] == {"attachment": 2.5}
    assert record["start_objective"] == pytest.approx(plain.objective_value)
    assert record["end"] == {"attachment": refined.parameters["attachment"]}
    assert record["objective"] == refined.objective_value
    assert record["gain"] == pytest.approx(
        plain.objective_value - refined.objective_value
    )
    assert record["moves"] > 0 and record["stopped"] == "step_tolerance"
    assert record["points"] == len(_rows(refined)) == record["counts"]["optimal"]
    assert record["initial_steps"] == {"attachment": 1.25}
    assert record["final_steps"]["attachment"] <= REFINE_TOLERANCE * 10.0
    # The grid's profile is the grid's alone.
    assert refined.profile is not None and plain.profile is not None
    assert refined.profile.equals(plain.profile)


def test_a_smaller_tolerance_gets_closer() -> None:
    terms: dict[str, Any] = {"budget": Budget(ratio=1.0)}
    coarse = design(_linear(), SAMPLE, ExpectedUncovered(), options=REFINE, **terms)
    fine = design(
        _linear(),
        SAMPLE,
        ExpectedUncovered(),
        options=DesignOptions(refine=True, refine_tolerance=1e-7),
        **terms,
    )
    assert abs(fine.parameters["attachment"] - ATTACHMENT) <= 1e-5
    assert abs(fine.parameters["attachment"] - ATTACHMENT) < abs(
        coarse.parameters["attachment"] - ATTACHMENT
    )
    assert fine.objective_value < coarse.objective_value
    assert fine.refinement is not None and coarse.refinement is not None
    assert fine.refinement["points"] > coarse.refinement["points"]


def test_two_free_thresholds_are_refined_together() -> None:
    sample = _linear_sample(attachment=3.7, exhaustion=8.3)
    template = _linear(
        Free(0.0, 10.0, grid=[0.0, 2.5, 5.0, 7.5]),
        Free(0.0, 10.0, grid=[6.0, 10.0]),
    )
    plain = design(template, sample, MSE())
    refined = design(template, sample, MSE(), options=REFINE)
    assert refined.parameters["attachment"] == pytest.approx(3.7, abs=1e-2)
    assert refined.parameters["exhaustion"] == pytest.approx(8.3, abs=1e-2)
    assert _scaled(refined, refined.objective_value) <= 1e-7
    assert refined.objective_value < plain.objective_value
    assert refined.refinement is not None
    assert set(refined.refinement["start"]) == {"attachment", "exhaustion"}


@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_a_design_at_its_optimum_is_not_moved(
    objective: Objective, terms: dict[str, Any]
) -> None:
    # The grid holds the generating attachment, so its best point is optimal.
    template = _linear(Free(0.0, 10.0, grid=[0.0, ATTACHMENT, 7.5]))
    plain = design(template, SAMPLE, objective, **terms)
    refined = design(template, SAMPLE, objective, options=REFINE, **terms)
    assert plain.parameters["attachment"] == ATTACHMENT
    assert refined.parameters == plain.parameters
    assert refined.cover.definition_hash == plain.cover.definition_hash
    assert refined.objective_value == plain.objective_value
    assert refined.premium == plain.premium
    assert (refined.ties, refined.unique) == (plain.ties, plain.unique)
    record = refined.refinement
    assert record is not None
    assert record["moves"] == 0 and record["gain"] == 0.0
    assert record["end"] == record["start"] == {"attachment": ATTACHMENT}
    assert record["points"] > 0  # it looked, and found nothing better


def test_a_flat_optimum_of_a_step_payout_is_not_moved() -> None:
    # Between two years' index values a step payout's objective is constant:
    # every breakpoint in (1, 2] splits these four years alike. Both grid
    # points lie there, so they tie, and no other breakpoint does better.
    sample = _sample([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 30.0, 40.0])
    template = Cover(
        INDEX,
        StepPayout([Free(0.0, 3.0, grid=[1.5, 1.8]), 2.5], [Free(), Free()]),
        payout_unit="EUR",
    )
    plain = design(template, sample, MSE())
    refined = design(template, sample, MSE(), options=REFINE)
    assert (plain.ties, plain.unique) == (1, False)
    assert refined.refinement is not None and refined.refinement["moves"] == 0
    assert refined.refinement["method"] == INDEX_VALUES_METHOD
    assert refined.parameters == plain.parameters
    assert (refined.ties, refined.unique) == (1, False)


def test_the_search_stays_within_the_bounds() -> None:
    # The loss attaches at 3.7, above the attachment's upper bound of 3: the
    # search ends on that bound, and no point it evaluates leaves [0.2, 3].
    template = _linear(Free(0.2, 3.0, grid=[0.5, 1.5]))
    refined = design(template, SAMPLE, MSE(), options=REFINE)
    assert refined.parameters["attachment"] == 3.0
    values = [row["attachment"] for row in _rows(refined)]
    assert values and all(0.2 <= value <= 3.0 for value in values)
    assert 3.0 in values


def test_two_thresholds_stay_within_their_bounds_and_in_order() -> None:
    # The attachment must stay below the exhaustion: points where it does not
    # make no valid cover, are recorded as "invalid" and never moved to.
    sample = _linear_sample(attachment=5.5, exhaustion=6.0)
    template = _linear(
        Free(1.0, 7.0, grid=[2.0, 4.0, 6.0]), Free(5.0, 9.0, grid=[6.0, 8.0])
    )
    refined = design(template, sample, MSE(), options=REFINE)
    rows = _rows(refined)
    assert all(1.0 <= row["attachment"] <= 7.0 for row in rows)
    assert all(5.0 <= row["exhaustion"] <= 9.0 for row in rows)
    for row in rows:
        if row["attachment"] >= row["exhaustion"]:
            assert row["status"] == "invalid" and row["objective"] is None
        else:
            assert row["status"] == "optimal" and row["objective"] is not None
    record = refined.refinement
    assert record is not None and record["counts"]["invalid"] > 0
    assert refined.parameters["attachment"] < refined.parameters["exhaustion"]
    assert refined.parameters["attachment"] == pytest.approx(5.5, abs=1e-2)
    assert refined.parameters["exhaustion"] == pytest.approx(6.0, abs=1e-2)


def test_a_point_without_a_feasible_payout_is_not_moved_to() -> None:
    # A fixed first level of 20 and a second level of at least 20 (monotone):
    # with the breakpoint at 1 or below, the first level pays years 1 and 2,
    # and the premium of at least (20 + 20 + 20) / 4 = 15 breaks the budget
    # of 10. Above 2.8 the breakpoints are out of order. A step payout is
    # refined over the index values: 1 and 0 have no feasible payout, 3 makes
    # no valid cover, and 2 is the start's set of years.
    sample = _sample([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 30.0, 40.0])
    template = Cover(
        INDEX,
        StepPayout([Free(0.0, 3.0, grid=[2.0]), 2.8], [20.0, Free()]),
        payout_unit="EUR",
    )
    terms: dict[str, Any] = {"budget": Budget(amount=10.0)}
    plain = design(template, sample, ExpectedUncovered(), **terms)
    refined = design(template, sample, ExpectedUncovered(), options=REFINE, **terms)
    rows = _rows(refined)
    by_status = {row["status"] for row in rows}
    assert {"infeasible", "invalid"} <= by_status
    for row in rows:
        if row["status"] == "infeasible":
            assert row["objective"] is None and row["breakpoints[0]"] <= 1.0
        if row["status"] == "invalid":
            assert row["objective"] is None and row["breakpoints[0]"] >= 2.8
    record = refined.refinement
    assert record is not None and record["counts"]["infeasible"] > 0
    assert record["method"] == INDEX_VALUES_METHOD and record["moves"] == 0
    assert refined.objective_value <= plain.objective_value
    assert refined.premium <= 10.0 * (1.0 + 1e-7)
    assert refined.parameters["breakpoints[0]"] > 1.0


def test_the_compass_search_does_not_move_to_a_point_without_a_feasible_payout() -> (
    None
):
    # The limit is at least 80: below an attachment of about 4.5 its premium
    # breaks the budget of 30, so the search's first step down, to 3.75, has
    # no feasible payout.
    template = Cover(
        INDEX,
        LinearPayout(Free(0.0, 10.0, grid=GRID), EXHAUSTION, Free(80.0, 100.0)),
        payout_unit="EUR",
    )
    terms: dict[str, Any] = {"budget": Budget(amount=30.0)}
    plain = design(template, SAMPLE, MSE(), **terms)
    refined = design(template, SAMPLE, MSE(), options=REFINE, **terms)
    rows = _rows(refined)
    assert {row["status"] for row in rows} == {"infeasible", "optimal"}
    assert rows[1] == {"attachment": 3.75, "objective": None, "status": "infeasible"}
    record = refined.refinement
    assert record is not None and record["method"] == REFINE_METHOD
    assert record["moves"] > 0 and record["counts"]["infeasible"] > 0
    assert refined.objective_value < plain.objective_value
    assert refined.premium <= 30.0 * (1.0 + 1e-7)


def test_the_point_cap_stops_the_search() -> None:
    plain = design(_linear(), SAMPLE, MSE())
    capped = design(
        _linear(),
        SAMPLE,
        MSE(),
        options=DesignOptions(refine=True, refine_max_points=3),
    )
    record = capped.refinement
    assert record is not None
    assert record["points"] == 3 and record["stopped"] == "max_points"
    assert capped.objective_value <= plain.objective_value


def test_a_refinement_is_never_worse_than_its_grid() -> None:
    # Random samples, step and linear covers, every built-in objective.
    rng = np.random.default_rng(11)
    for case in range(12):
        index = rng.gumbel(5.0, 1.5, 60)
        loss = np.maximum(0.0, index - rng.uniform(3.0, 7.0)) * rng.lognormal(
            0.0, 0.3, 60
        )
        sample = _sample(index.tolist(), loss.tolist())
        grid = np.linspace(2.0, 9.0, 4).tolist()
        template = (
            _linear(Free(0.0, 10.0, grid=grid), 12.0)
            if case % 2
            else Cover(
                INDEX,
                StepPayout([Free(0.0, 10.0, grid=grid), 10.5], [Free(), Free()]),
                payout_unit="EUR",
            )
        )
        objective, terms = OBJECTIVES[case % 3]
        terms = {**terms, "limit": Limit(float(loss.max()))}
        plain = design(template, sample, objective, **terms)
        refined = design(template, sample, objective, options=REFINE, **terms)
        assert refined.objective_value <= plain.objective_value, case
        assert refined.refinement is not None
        assert refined.refinement["gain"] == pytest.approx(
            plain.objective_value - refined.objective_value, abs=1e-12
        )


def test_the_refinement_is_deterministic() -> None:
    first = design(
        _linear(),
        SAMPLE,
        WeightedCVaR(0.5, 0.9),
        budget=Budget(ratio=1.0),
        options=REFINE,
    )
    second = design(
        _linear(),
        SAMPLE,
        WeightedCVaR(0.5, 0.9),
        budget=Budget(ratio=1.0),
        options=REFINE,
    )
    assert first.refinement == second.refinement
    assert first.parameters == second.parameters
    assert first.refinement_profile is not None
    assert first.refinement_profile.equals(second.refinement_profile)
    assert first.design_id == second.design_id


# -- ties and non-uniqueness at the refined point ------------------------------


def _fixed_at(result: DesignResult) -> Cover:
    return _linear(result.parameters["attachment"])


def test_the_lp_ranges_are_those_of_the_refined_thresholds() -> None:
    options = DesignOptions(refine=True, ranges=True)
    terms: dict[str, Any] = {"budget": Budget(ratio=1.0)}
    refined = design(_linear(), SAMPLE, ExpectedUncovered(), options=options, **terms)
    assert refined.refinement is not None and refined.refinement["moves"] > 0
    fixed = design(
        _fixed_at(refined),
        SAMPLE,
        ExpectedUncovered(),
        options=DesignOptions(ranges=True),
        **terms,
    )
    assert refined.ties == 0
    assert refined.unique == fixed.unique is not None
    assert refined.ranges == fixed.ranges
    assert refined.parameters["limit"] == fixed.parameters["limit"]


def test_the_qp_report_is_that_of_the_refined_thresholds() -> None:
    refined = design(_linear(), SAMPLE, MSE(), options=REFINE)
    fixed = design(_fixed_at(refined), SAMPLE, MSE())
    assert refined.ties == 0
    assert refined.unique is True and fixed.unique is True
    assert refined.parameters["limit"] == fixed.parameters["limit"]


def test_a_refined_point_that_solves_worse_again_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The search believes one point is better than the grid; solved again
    # there, it is not, and the design raises rather than return it.
    solved_point = DESIGN_MODULE._solved_point

    def flattering(*args: Any) -> tuple[float | None, str, str | None]:
        objective, status, message = solved_point(*args)
        if objective is not None and args[2][0] == 3.75:
            return 0.0, status, message
        return objective, status, message

    monkeypatch.setattr(DESIGN_MODULE, "_solved_point", flattering)
    with pytest.raises(DesignError, match="solved again") as raised:
        design(_linear(), SAMPLE, MSE(), options=REFINE)
    assert raised.value.status == "inconsistent_objective"


# -- where the option does not apply -------------------------------------------


def test_a_design_without_free_thresholds_is_not_refined() -> None:
    template = _linear(2.5)
    plain = design(template, SAMPLE, MSE())
    refined = design(template, SAMPLE, MSE(), options=REFINE)
    assert refined.route == "qp"
    assert refined.refinement is None and refined.refinement_profile is None
    assert refined.parameters == plain.parameters
    assert refined.objective_value == plain.objective_value


def _mean_absolute(loss: Any, payout: Any, probability: Any) -> float:
    return float(np.sum(probability * np.abs(loss - payout)))


def test_a_user_objective_raises_and_names_the_reason() -> None:
    objective = UserObjective(_mean_absolute, name="mae")
    template = Cover(
        INDEX,
        LinearPayout(Free(0.0, 10.0, grid=GRID), EXHAUSTION, Free(0.0, 100.0, grid=5)),
        payout_unit="EUR",
    )
    design(template, SAMPLE, objective)  # the grid in numpy works without it
    with pytest.raises(DesignError, match="UserObjective is evaluated in numpy"):
        design(template, SAMPLE, objective, options=REFINE)


def test_a_payout_without_amounts_raises_and_names_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A payout that `design` cannot solve for amounts is searched in numpy.
    monkeypatch.setattr(DESIGN_MODULE, "_has_amounts", lambda payout: False)
    with pytest.raises(DesignError, match="not linear in its amounts"):
        design(_linear(), SAMPLE, MSE(), options=REFINE)


def test_design_joint_raises() -> None:
    n = 40
    rng = np.random.default_rng(3)
    index = rng.uniform(0.0, 10.0, (n, 2))
    loss = np.clip(index - 4.0, 0.0, None) * 10.0
    sample = OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=loss,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )
    templates = [
        Cover(INDEX, StepPayout([4.0, 7.0], [Free(), Free()]), payout_unit="EUR")
        for _ in range(2)
    ]
    design_joint(templates, sample, MSE())  # the same design without it
    with pytest.raises(DesignError, match="design_joint has no grid route"):
        design_joint(templates, sample, MSE(), options=REFINE)


class OpenBinary(_DefinitionModel):
    """A user family linear in its amount: `amount` where the index reaches `trigger`.

    Unlike `BinaryPayout`, it takes a free trigger without an upper bound.
    """

    kind: ClassVar[str] = "test_open_binary"
    trigger: Any
    amount: Any
    tail: Tail = "upper"

    def __init__(self, trigger: Any, amount: Any) -> None:
        super().__init__(trigger=trigger, amount=amount)

    @property
    def index_names(self) -> tuple[str, ...]:
        return ("index",)

    def parameters(self) -> dict[str, float | Free]:
        return {"trigger": self.trigger, "amount": self.amount}

    def with_values(self, values: Mapping[str, float]) -> Self:
        merged = {**self.parameters(), **values}
        return type(self)(merged["trigger"], merged["amount"])

    def amount_names(self) -> tuple[str, ...]:
        return ("amount",)

    def basis(self, index: Any) -> Any:
        reached = np.asarray(index, dtype=np.float64) >= self.trigger
        return reached.astype(np.float64)[:, None]

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        free = self.amount if isinstance(self.amount, Free) else None
        low = self.amount if free is None else free.lower
        high = self.amount if free is None else free.upper
        upper = min(
            math.inf if high is None else high, math.inf if limit is None else limit
        )
        return LinearConstraints(
            a_ub=np.zeros((0, 1)),
            b_ub=np.zeros(0),
            lower=np.array([low]),
            upper=np.array([upper]),
        )

    def evaluate(self, index: Any) -> Any:
        return self.basis(index)[:, 0] * self.amount

    def to_document(self) -> dict[str, Any]:
        def document(value: Any) -> Any:
            return value.to_document() if isinstance(value, Free) else value

        return {
            "kind": self.kind,
            "amount": document(self.amount),
            "tail": self.tail,
            "trigger": document(self.trigger),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(document["trigger"], document["amount"])


def test_a_user_family_linear_in_its_amounts_is_refined() -> None:
    template = Cover(
        INDEX, OpenBinary(Free(0.0, 10.0, grid=GRID), Free()), payout_unit="EUR"
    )
    plain = design(template, SAMPLE, MSE())
    refined = design(template, SAMPLE, MSE(), options=REFINE)
    assert refined.route == "grid" and refined.amounts == "qp"
    assert refined.refinement is not None
    assert refined.objective_value <= plain.objective_value


def test_a_threshold_without_an_upper_bound_raises() -> None:
    template = Cover(
        INDEX, OpenBinary(Free(0.0, None, grid=GRID), Free()), payout_unit="EUR"
    )
    design(template, SAMPLE, MSE())  # the grid alone needs no bound
    with pytest.raises(DesignError, match="trigger within its bounds"):
        design(template, SAMPLE, MSE(), options=REFINE)


# -- the record, the metadata and the setup text --------------------------------


def test_the_record_holds_the_refinement_as_plain_json() -> None:
    refined = design(_linear(), SAMPLE, MSE(), options=REFINE)
    record = refined.to_record()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["refinement"] == refined.refinement
    assert record["options"]["refine"] is True
    assert set(record["refinement"]) == {
        "method",
        "tolerance",
        "max_points",
        "initial_steps",
        "start",
        "start_objective",
        "end",
        "objective",
        "gain",
        "points",
        "counts",
        "moves",
        "final_steps",
        "stopped",
    }
    assert record["solve"]["solves"] > record["refinement"]["points"]


def _design_line(result: DesignResult, sample: OutcomeSample = SAMPLE) -> str:
    table = evaluate(result, sample, split="in_sample")
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.designs[0].get("refinement") == result.refinement
    (line,) = [
        line for line in setup_text(table).splitlines() if line.startswith("Design ")
    ]
    return line


def test_the_setup_text_states_the_refinement() -> None:
    refined = design(_linear(), SAMPLE, MSE(), options=REFINE)
    record = refined.refinement
    assert record is not None
    line = _design_line(refined)
    assert (
        "route: grid; refined by a compass search from the grid's best point " in line
    )
    assert "(attachment 2.5, objective " in line
    assert f"to attachment {record['end']['attachment']:.6g}, objective " in line
    assert ", a gain of " in line
    assert (
        f"after {record['points']} points (smallest step 0.0001 of each "
        "parameter's bounds, at most 1,000 points)" in line
    )
    assert "cap" not in line


def test_the_setup_text_says_when_the_search_did_not_move_or_hit_its_cap() -> None:
    template = _linear(Free(0.0, 10.0, grid=[0.0, ATTACHMENT, 7.5]))
    line = _design_line(design(template, SAMPLE, MSE(), options=REFINE))
    # At the optimum the objective is 0 to rounding, which differs by platform.
    assert "from the grid's best point (attachment 3.7, objective " in line
    assert "), not moved, after " in line and "a gain of" not in line
    capped = design(
        _linear(),
        SAMPLE,
        MSE(),
        options=DesignOptions(refine=True, refine_max_points=2),
    )
    line = _design_line(capped)
    assert (
        "after 2 points (" in line
        and "at most 2 points; it stopped at that cap)" in line
    )


def test_a_design_without_refinement_has_no_refinement_in_its_line() -> None:
    assert "refined" not in _design_line(design(_linear(), SAMPLE, MSE()))


# -- the frontier and the plan pass the option on ------------------------------


def test_a_budget_frontier_refines_every_budget() -> None:
    table = design_frontier(
        _linear(),
        SAMPLE,
        ExpectedUncovered(),
        budgets=[Budget(ratio=0.5), Budget(ratio=1.0)],
        evaluate_on=SAMPLE,
        split="in_sample",
        design_options=REFINE,
    )
    metadata = EvaluationMetadata.from_table(table)
    records = [record for record in metadata.designs if "refinement" in record]
    assert len(records) == 2


def _annual(years: Any) -> Any:
    rng = np.random.default_rng(1)
    cells = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
    n = len(years) * len(cells)
    return pa.table(
        {
            "year": pa.array(np.repeat(years, len(cells)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in cells], len(years)),
            "longitude": np.tile([lon for _, lon in cells], len(years)),
            "value": rng.gumbel(50.0, 12.0, n),
        }
    )


def _plan() -> Any:
    from crc_framework import PiecewiseLinearImpact

    assets = pa.table(
        {
            "asset_id": ["warehouse-a", "depot-b"],
            "longitude": [6.96, 7.10],
            "latitude": [50.94, 51.10],
            "value_eur": [2e6, 2.1e6],
        }
    )
    plan: Any = ParametricPlan
    return plan.for_annual_extremes(
        _annual(np.arange(1981, 2021)),
        assets,
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
        unit="mm/day",
        provenance=SourceProvenance(
            provider="example",
            dataset="synthetic rx1day",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic annual maxima",
        ),
        value_column="value_eur",
        value_unit="EUR",
        seed=42,
    ).basis_risk(NoBasisRisk())


def _rain_step() -> Cover:
    return Cover(
        IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum"),
        StepPayout(
            [Free(40.0, 100.0, grid=[50.0, 70.0, 90.0]), 110.0], [Free(), Free()]
        ),
        payout_unit="EUR",
        label="step",
    )


@pytest.mark.filterwarnings("ignore::crc_sdk.parametric.ComonotoneIndexWarning")
def test_a_plan_passes_the_option_to_every_design() -> None:
    designs = (
        _plan()
        .covers(_rain_step())
        .objective(ExpectedUncovered(), budget=Budget(ratio=0.5), options=REFINE)
        .designs()
    )
    assert len(designs) == 2
    for result in designs:
        assert result.refinement is not None
        assert result.options.refine is True


@pytest.mark.filterwarnings("ignore::crc_sdk.parametric.ComonotoneIndexWarning")
def test_a_joint_plan_raises() -> None:
    plan = (
        _plan()
        .covers(
            Cover(
                IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum"),
                StepPayout([60.0, 90.0], [Free(), Free()]),
                payout_unit="EUR",
                label="step",
            )
        )
        .objective(MSE(), options=REFINE)
        .scopes(Joint())
    )
    with pytest.raises(DesignError, match="design_joint has no grid route"):
        plan.designs()


# -- a step or binary payout: the search over the index values ---------------

# Ten years. A threshold pays the years at or beyond it, so the best one pays
# the years from 4.31 on: between the grid's 0 and 5, and in a set of years
# that no compass step from 5 reaches (every threshold in (4.31, 6] pays the
# years from 6 on, and the steps from 5 stay there).
STEP_INDEX = [1.0, 2.0, 3.0, 4.0, 4.3, 4.31, 6.0, 7.0, 8.0, 9.0]
STEP_LOSS = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0, 100.0]
BINARY_LOSS = [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0]
LOWER = IndexDefinition("wind", "m/s", tail="lower")


def _step(grid: Sequence[float] = (0.0, 5.0, 8.0)) -> Cover:
    return Cover(
        INDEX,
        StepPayout([Free(0.0, 8.0, grid=grid), 8.5], [Free(), Free()]),
        payout_unit="EUR",
    )


def _binary(index: IndexDefinition = INDEX) -> Cover:
    tail: Tail = index.tail
    return Cover(
        index,
        BinaryPayout(Free(0.0, 10.0, grid=[0.0, 5.0, 10.0]), Free(), tail=tail),
        payout_unit="EUR",
    )


def _compass(
    monkeypatch: pytest.MonkeyPatch, *arguments: Any, **terms: Any
) -> DesignResult:
    """The same design, its step or binary payout refined by a compass search."""
    with monkeypatch.context() as patched:
        patched.setattr(DESIGN_MODULE, "_on_index_values", lambda *_: False)
        return design(*arguments, options=REFINE, **terms)


@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_a_step_threshold_between_grid_points_is_found_exactly(
    objective: Objective, terms: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The loss is a step payout of 50 from 4.31 and 100 from 8.5: the
    # objective is 0 with the first breakpoint at 4.31, exactly, and above 0
    # at any breakpoint that pays another set of years.
    sample = _sample(STEP_INDEX, STEP_LOSS)
    plain = design(_step(), sample, objective, **terms)
    refined = design(_step(), sample, objective, options=REFINE, **terms)
    compass = _compass(monkeypatch, _step(), sample, objective, **terms)
    assert refined.parameters["breakpoints[0]"] == 4.31
    assert refined.parameters["levels[0]"] == pytest.approx(50.0, rel=1e-9)
    assert refined.parameters["levels[1]"] == pytest.approx(100.0, rel=1e-9)
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12
    assert plain.objective_value > 0.0
    # The compass search from the same grid point misses that set of years.
    assert compass.refinement is not None
    assert compass.refinement["method"] == REFINE_METHOD
    assert compass.parameters["breakpoints[0]"] != 4.31
    assert _scaled(compass, compass.objective_value) > 0.01
    record = refined.refinement
    assert record is not None
    assert record["method"] == INDEX_VALUES_METHOD
    assert record["start"] == {"breakpoints[0]": plain.parameters["breakpoints[0]"]}
    assert record["end"] == {"breakpoints[0]": 4.31}
    assert record["moves"] >= 1 and record["stopped"] == "no_move"
    assert record["gain"] == pytest.approx(
        plain.objective_value - refined.objective_value
    )


@pytest.mark.parametrize("index", [INDEX, LOWER], ids=["upper", "lower"])
@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_a_binary_trigger_between_grid_points_is_found_exactly(
    objective: Objective,
    terms: dict[str, Any],
    index: IndexDefinition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loss is 100 from 4.31 up; with a lower tail the index is mirrored,
    # 10 - x, and the loss is 100 from 5.69 down. A trigger pays the years at
    # or beyond it, so the exact trigger is that index value itself.
    upper = index.tail == "upper"
    values = STEP_INDEX if upper else [10.0 - x for x in STEP_INDEX]
    trigger = 4.31 if upper else 10.0 - 4.31
    sample = _sample(values, BINARY_LOSS)
    plain = design(_binary(index), sample, objective, **terms)
    refined = design(_binary(index), sample, objective, options=REFINE, **terms)
    compass = _compass(monkeypatch, _binary(index), sample, objective, **terms)
    assert trigger in values
    assert refined.parameters["trigger"] == trigger
    assert refined.parameters["amount"] == pytest.approx(100.0, rel=1e-9)
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12
    assert plain.objective_value > 0.0
    assert compass.parameters["trigger"] != trigger
    assert _scaled(compass, compass.objective_value) > 0.01
    assert refined.refinement is not None
    assert refined.refinement["method"] == INDEX_VALUES_METHOD
    low, high = refined.refinement["windows"]["trigger"]
    assert low <= plain.parameters["trigger"] <= high and low <= trigger <= high


def test_each_set_of_paying_years_is_tried_once() -> None:
    # The window of 5 is the grid's 0 and 8: the index values 1 to 8 and the
    # end 0, ten values. The first pass tries the 2 nearest 5, 4.31 and 4.3,
    # and moves to 4.31; each pass after tries twice as many around 4.31: 4
    # and 3 (and 6, which pays the same years as 5, the start), then 2, 7, 1
    # and 8. The fourth covers the window, finds 0 paying the same years as
    # 1, solves nothing and moves nothing.
    sample = _sample(STEP_INDEX, STEP_LOSS)
    refined = design(_step(), sample, MSE(), options=REFINE)
    record = refined.refinement
    assert record is not None
    assert record["windows"] == {"breakpoints[0]": [0.0, 8.0]}
    assert record["candidates"] == {"breakpoints[0]": 10}
    tried = [row["breakpoints[0]"] for row in _rows(refined)]
    assert tried == [4.31, 4.3, 4.0, 3.0, 2.0, 7.0, 1.0, 8.0]
    assert record["points"] == len(tried) == record["counts"]["optimal"]
    assert (record["moves"], record["passes"]) == (1, 4)


def test_several_thresholds_are_refined_in_passes_until_none_moves() -> None:
    rng = np.random.default_rng(4)
    index = rng.gumbel(5.0, 1.5, 80)
    loss = 20.0 * (index >= 4.2) + 30.0 * (index >= 6.1) + rng.normal(0.0, 3.0, 80)
    sample = _sample(index.tolist(), np.maximum(loss, 0.0).tolist())
    template = Cover(
        INDEX,
        StepPayout(
            [
                Free(2.0, 6.0, grid=[2.0, 3.5, 5.0]),
                Free(5.5, 9.0, grid=[5.5, 7.0, 9.0]),
            ],
            [Free(), Free()],
        ),
        payout_unit="EUR",
    )
    plain = design(template, sample, MSE())
    refined = design(template, sample, MSE(), options=REFINE)
    record = refined.refinement
    assert record is not None and record["stopped"] == "no_move"
    assert record["passes"] >= 2 and record["moves"] >= 2
    assert refined.objective_value < plain.objective_value
    names = ["breakpoints[0]", "breakpoints[1]"]
    end = [refined.parameters[name] for name in names]
    # No single threshold improves on the end within its window by more than
    # the tie tolerance: every index value there, and the window's ends, give
    # an objective at least as high, to that tolerance (scaled units).
    best = _scaled(refined, refined.objective_value)
    for i, name in enumerate(names):
        low, high = record["windows"][name]
        assert low <= end[i] <= high
        assert end[i] in index.tolist() or end[i] in (low, high)
        values = {x for x in index.tolist() if low <= x <= high} | {low, high}
        for value in sorted(values):
            point = list(end)
            point[i] = value
            if point[0] >= point[1]:
                continue
            fixed = Cover(INDEX, StepPayout(point, [Free(), Free()]), payout_unit="EUR")
            other = design(fixed, sample, MSE())
            gap = _scaled(other, other.objective_value) - best
            assert gap >= -1e-7 * max(1.0, best), (name, value)


def test_the_point_cap_stops_the_index_value_search() -> None:
    sample = _sample(STEP_INDEX, STEP_LOSS)
    plain = design(_step(), sample, MSE())
    capped = design(
        _step(), sample, MSE(), options=DesignOptions(refine=True, refine_max_points=2)
    )
    record = capped.refinement
    assert record is not None
    assert record["points"] == 2 and record["stopped"] == "max_points"
    assert record["passes"] == 2
    # The first pass tried the two values nearest 5 and moved to the better;
    # the second stopped at its first new value.
    assert capped.parameters["breakpoints[0]"] == 4.31
    assert capped.objective_value <= plain.objective_value
    line = _design_line(capped, sample)
    assert "after 2 points in 2 passes (" in line
    assert "at most 2 points; it stopped at that cap)" in line


def test_the_index_value_search_is_deterministic() -> None:
    sample = _sample(STEP_INDEX, STEP_LOSS)
    terms: dict[str, Any] = {"budget": Budget(ratio=1.0)}
    first = design(_step(), sample, WeightedCVaR(0.5, 0.9), options=REFINE, **terms)
    second = design(_step(), sample, WeightedCVaR(0.5, 0.9), options=REFINE, **terms)
    assert first.refinement == second.refinement
    assert first.parameters == second.parameters
    assert first.refinement_profile is not None
    assert first.refinement_profile.equals(second.refinement_profile)
    assert first.design_id == second.design_id


def test_the_record_of_an_index_value_search_is_plain_json() -> None:
    sample = _sample(STEP_INDEX, STEP_LOSS)
    record = design(_step(), sample, MSE(), options=REFINE).to_record()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert set(record["refinement"]) == {
        "method",
        "max_points",
        "windows",
        "candidates",
        "start",
        "start_objective",
        "end",
        "objective",
        "gain",
        "points",
        "counts",
        "moves",
        "passes",
        "stopped",
    }


def test_the_amounts_report_is_that_of_the_refined_thresholds() -> None:
    sample = _sample(STEP_INDEX, STEP_LOSS)
    terms: dict[str, Any] = {"budget": Budget(ratio=1.0)}
    options = DesignOptions(refine=True, ranges=True)
    refined = design(_step(), sample, ExpectedUncovered(), options=options, **terms)
    assert refined.refinement is not None and refined.refinement["moves"] == 1
    fixed = design(
        Cover(INDEX, StepPayout([4.31, 8.5], [Free(), Free()]), payout_unit="EUR"),
        sample,
        ExpectedUncovered(),
        options=DesignOptions(ranges=True),
        **terms,
    )
    assert refined.ties == 0
    assert refined.unique == fixed.unique is not None
    assert refined.ranges == fixed.ranges


def test_the_setup_text_states_the_index_value_search() -> None:
    sample = _sample(STEP_INDEX, STEP_LOSS)
    line = _design_line(design(_step(), sample, MSE(), options=REFINE), sample)
    # At the optimum the MSE is 0 to rounding, which differs by platform.
    assert (
        "route: grid; refined over the sample's index values from the grid's best "
        "point (breakpoints[0] 5, objective 250) to breakpoints[0] 4.31, "
        "objective " in line
    )
    assert (
        ", a gain of 250, after 8 points in 4 passes (each threshold over the "
        "index values between its grid neighbours: breakpoints[0] from 0 to 8; "
        "at most 1,000 points); unique optimum: yes; ties: 0." in line
    )


def test_without_the_option_step_and_binary_designs_are_unchanged() -> None:
    # The ids, values and solves of these designs before the option existed.
    sample = _sample(STEP_INDEX, STEP_LOSS)
    step = design(_step(), sample, MSE())
    assert step.refinement is None and step.refinement_profile is None
    assert step.parameters["breakpoints[0]"] == 5.0 and step.solves == 4
    assert step.design_id == (
        "sha256:fc9745dbf4e558b0e936f99fdf6490b7d559c70f1ac20bc95643e6e6fc16efc1"
    )
    binary = design(_binary(), _sample(STEP_INDEX, BINARY_LOSS), MSE())
    assert binary.refinement is None
    assert binary.parameters["trigger"] == 5.0 and binary.solves == 4
    assert binary.design_id == (
        "sha256:4aa0ffc8feada3b3275da1f32c5a6dc691e0ed642f3972f104fa2e19e9d4513d"
    )


def test_other_families_keep_the_compass_search() -> None:
    linear = design(_linear(), SAMPLE, MSE(), options=REFINE)
    piecewise = design(
        Cover(
            INDEX,
            PiecewiseLinearPayout(
                [Free(0.0, 6.0, grid=[0.0, 2.5, 5.0]), 8.0], [Free(), Free()]
            ),
            payout_unit="EUR",
        ),
        SAMPLE,
        MSE(),
        options=REFINE,
    )
    user = design(
        Cover(INDEX, OpenBinary(Free(0.0, 10.0, grid=GRID), Free()), payout_unit="EUR"),
        SAMPLE,
        MSE(),
        options=REFINE,
    )
    for result in (linear, piecewise, user):
        assert result.refinement is not None
        assert result.refinement["method"] == REFINE_METHOD


# -- the compass search itself -------------------------------------------------


def _counting(
    function: Callable[[tuple[float, ...]], float],
) -> tuple[Callable[[tuple[float, ...]], tuple[float | None, str]], list[Any]]:
    calls: list[tuple[float, ...]] = []

    def evaluate(point: tuple[float, ...]) -> tuple[float | None, str]:
        calls.append(point)
        return function(point), "optimal"

    return evaluate, calls


def _search(
    function: Callable[[tuple[float, ...]], float],
    start: Sequence[float],
    **settings: Any,
) -> tuple[_solver.Search, list[Any]]:
    evaluate, calls = _counting(function)
    n = len(start)
    arguments: dict[str, Any] = {
        "steps": [1.0] * n,
        "smallest": [1e-6] * n,
        "lower": [-10.0] * n,
        "upper": [10.0] * n,
        "max_points": 1_000,
        "allowance": lambda value: 1e-12,
    }
    arguments.update(settings)
    found = _solver.compass_search(evaluate, start, function(tuple(start)), **arguments)
    return found, calls


def test_the_compass_search_finds_a_smooth_minimum() -> None:
    def bowl(x: tuple[float, ...]) -> float:
        return (x[0] - 1.3) ** 2 + 2.0 * (x[1] + 0.4) ** 2

    found, calls = _search(bowl, [3.0, 3.0])
    assert found.best == pytest.approx((1.3, -0.4), abs=1e-5)
    assert found.objective == bowl(found.best)
    assert found.stopped == "step_tolerance"
    assert all(step <= 1e-6 for step in found.steps)
    assert len(found.points) == len(calls) == len(set(calls))  # each once
    assert [point.values for point in found.points] == calls


def test_the_compass_search_keeps_its_bounds_and_clips_to_them() -> None:
    found, calls = _search(lambda x: -x[0], [0.0], lower=[-1.0], upper=[2.5])
    assert found.best == (2.5,)
    assert all(-1.0 <= point[0] <= 2.5 for point in calls)


def test_the_compass_search_does_not_move_from_a_minimum() -> None:
    found, _ = _search(lambda x: abs(x[0]), [0.0])
    assert found.best == (0.0,) and found.moves == 0
    assert found.objective == 0.0


def test_the_compass_search_moves_only_on_a_gain_above_the_allowance() -> None:
    # Every other point is better by 1e-3, less than the allowance of 1e-2.
    def nearly_flat(x: tuple[float, ...]) -> float:
        return 0.0 if x == (0.0,) else -1e-3

    found, _ = _search(nearly_flat, [0.0], allowance=lambda value: 1e-2)
    assert found.moves == 0 and found.best == (0.0,)


def test_the_compass_search_skips_points_without_an_objective() -> None:
    def evaluate(point: tuple[float, ...]) -> tuple[float | None, str]:
        if point[0] > 0.0:
            return None, "invalid"
        return point[0], "optimal"

    found = _solver.compass_search(
        evaluate,
        [0.0],
        0.0,
        steps=[1.0],
        smallest=[1e-3],
        lower=[-2.0],
        upper=[2.0],
        max_points=100,
        allowance=lambda value: 1e-12,
    )
    assert found.best == (-2.0,)
    assert any(point.status == "invalid" for point in found.points)


def test_the_compass_search_stops_at_its_cap_with_its_best_point() -> None:
    found, calls = _search(lambda x: (x[0] - 5.0) ** 2, [0.0], max_points=4)
    assert found.stopped == "max_points" and len(calls) == 4
    assert found.objective <= 25.0 and found.objective == (found.best[0] - 5.0) ** 2


def test_a_parameter_without_room_is_not_polled() -> None:
    found, calls = _search(
        lambda x: (x[0] - 1.0) ** 2 + x[1] ** 2,
        [0.0, 0.0],
        steps=[1.0, 0.0],
        smallest=[1e-6, 0.0],
    )
    assert all(point[1] == 0.0 for point in calls)
    assert found.best[0] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "bad",
    [
        {"steps": [1.0, 1.0]},
        {"max_points": 0},
        {"steps": [-1.0]},
        {"smallest": [-1.0]},
        {"lower": [1.0]},
    ],
)
def test_the_compass_search_checks_its_inputs(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _search(lambda x: x[0] ** 2, [0.0], **bad)


# -- the coordinate search itself ----------------------------------------------


def _coordinate(
    function: Callable[[tuple[float, ...]], float | None],
    start: Sequence[float],
    candidates: Sequence[Sequence[float]],
    **settings: Any,
) -> tuple[_solver.CoordinateSearch, list[Any]]:
    calls: list[tuple[float, ...]] = []

    def evaluate(point: tuple[float, ...]) -> tuple[float | None, str]:
        calls.append(point)
        value = function(point)
        return value, "optimal" if value is not None else "invalid"

    arguments: dict[str, Any] = {
        "key": lambda point: point,
        "max_points": 1_000,
        "allowance": lambda value: 1e-12,
    }
    arguments.update(settings)
    value = function(tuple(start))
    assert value is not None
    found = _solver.coordinate_search(
        evaluate, start, value, candidates=candidates, **arguments
    )
    return found, calls


def test_the_coordinate_search_finds_the_best_candidate() -> None:
    found, calls = _coordinate(
        lambda x: (x[0] - 3.3) ** 2, [8.0], [[float(i) for i in range(11)]]
    )
    assert found.best == (3.0,) and found.objective == pytest.approx(0.09)
    assert found.moves == 3 and found.passes == 4 and found.stopped == "no_move"
    # The 2 values nearest 8, then 4 nearest 7, 8 nearest 5 and all 10 others
    # nearest 3, each nearest first and the lower first at equal distance,
    # each value evaluated once.
    assert calls == [(x,) for x in (7.0, 9.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 10.0)]
    assert len(calls) == 10 == len(found.points)


def test_a_capped_coordinate_search_reaches_every_parameter() -> None:
    # The first pass tries 2 values of each parameter, so a cap of 4 points
    # moves both.
    grid = [float(i) for i in range(-10, 11)]
    found, calls = _coordinate(
        lambda x: x[0] ** 2 + x[1] ** 2, [5.0, 5.0], [grid, grid], max_points=4
    )
    assert calls == [(4.0, 5.0), (6.0, 5.0), (4.0, 4.0), (4.0, 6.0)]
    assert found.best == (4.0, 4.0) and found.stopped == "max_points"


def test_the_coordinate_search_takes_the_nearest_of_equal_candidates() -> None:
    found, _ = _coordinate(
        lambda x: min(abs(x[0] - 1.0), abs(x[0] - 6.0)), [4.0], [[1.0, 6.0, 9.0]]
    )
    assert found.best == (6.0,)
    found, _ = _coordinate(
        lambda x: min(abs(x[0] - 2.0), abs(x[0] - 6.0)), [4.0], [[2.0, 6.0]]
    )
    assert found.best == (2.0,)  # at equal distance, the lower one


def test_the_coordinate_search_evaluates_each_class_once() -> None:
    # The objective depends only on the integer part: the start's class (2)
    # is skipped, and 0.5 shares the class of 0.25.
    found, calls = _coordinate(
        lambda x: (math.floor(x[0]) - 0.6) ** 2,
        [2.5],
        [[0.25, 0.5, 1.0, 2.0, 2.75, 3.0]],
        key=lambda point: math.floor(point[0]),
    )
    assert found.best == (1.0,)
    assert calls == [(3.0,), (1.0,), (0.5,)]


def test_the_coordinate_search_moves_each_parameter_in_passes() -> None:
    grid = [float(i) for i in range(-5, 6)]

    def coupled(x: tuple[float, ...]) -> float:
        return (x[0] - 2.0) ** 2 + 0.5 * (x[1] - x[0]) ** 2

    # Each pass moves x0 with x1 held, then x1: pass 1 (2 values each) to
    # (-3, 3), pass 2 (4 values) to (-1, 1), pass 3 (8 values) to (2, 2), and
    # pass 4, which tries every value, moves neither.
    found, _ = _coordinate(coupled, [-5.0, 5.0], [grid, grid])
    assert found.best == (2.0, 2.0) and found.objective == 0.0
    assert (found.moves, found.passes, found.stopped) == (6, 4, "no_move")


def test_the_coordinate_search_does_not_move_to_a_point_without_an_objective() -> None:
    found, _ = _coordinate(
        lambda x: None if x[0] < 0.0 else x[0], [2.0], [[-3.0, -1.0, 1.0, 2.0]]
    )
    assert found.best == (1.0,)
    assert [point.status for point in found.points] == ["optimal", "invalid", "invalid"]


def test_the_coordinate_search_moves_only_on_a_gain_above_the_allowance() -> None:
    found, _ = _coordinate(
        lambda x: 0.0 if x == (0.0,) else -1e-3,
        [0.0],
        [[-1.0, 1.0]],
        allowance=lambda value: 1e-2,
    )
    assert found.moves == 0 and found.best == (0.0,)


def test_the_coordinate_search_stops_at_its_cap_with_its_best_point() -> None:
    found, calls = _coordinate(
        lambda x: (x[0] - 5.0) ** 2, [0.0], [[1.0, 2.0, 5.0]], max_points=2
    )
    assert found.stopped == "max_points" and len(calls) == 2
    assert found.best == (2.0,) and found.passes == 2


@pytest.mark.parametrize("bad", [{"candidates": [[1.0], [2.0]]}, {"max_points": 0}])
def test_the_coordinate_search_checks_its_inputs(bad: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "candidates": [[1.0]],
        "key": lambda point: point,
        "max_points": 10,
        "allowance": lambda value: 0.0,
    }
    arguments.update(bad)
    with pytest.raises(ValueError):
        _solver.coordinate_search(
            lambda point: (0.0, "optimal"), [0.0], 0.0, **arguments
        )


def test_the_coordinate_search_tries_extra_candidates_of_the_point() -> None:
    # The extra candidate of the first parameter is the second's value minus
    # 0.5, so it moves with it: at (0, 3) the first may take 2.5 too.
    found, calls = _coordinate(
        lambda x: abs(x[0] - 2.5) + abs(x[1] - 3.0),
        [0.0, 3.0],
        [[0.0, 1.0, 2.0], [3.0]],
        extra=lambda i, point: (point[1] - 0.5,) if i == 0 else (),
    )
    assert found.best == (2.5, 3.0) and found.objective == 0.0
    assert (2.5, 3.0) in calls


def test_the_coordinate_search_reports_its_moves() -> None:
    moves: list[tuple[float, ...]] = []
    found, _ = _coordinate(
        lambda x: (x[0] - 3.0) ** 2 + (x[1] - 1.0) ** 2,
        [0.0, 0.0],
        [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]],
        moved=moves.append,
    )
    assert found.best == (3.0, 1.0)
    assert len(moves) == found.moves and moves[-1] == found.best


# -- payouts built from steps ---------------------------------------------------

HEAT = IndexDefinition("heat", "degC")
HEAT_INDEX = [6.0, 1.0, 2.0, 7.0, 3.0, 8.0, 1.0, 2.0, 9.0, 3.0]
HOT = [30.0 if x >= 5.0 else 0.0 for x in HEAT_INDEX]
# Rain pays 50 from 4.31 and 100 from 8.5, as above; heat 30 from 5.
RAIN_STEP = [50.0 * (x >= 4.31) + 50.0 * (x >= 8.5) for x in STEP_INDEX]


def _two(
    rain: Sequence[float],
    loss: Sequence[float],
    tail: Tail = "upper",
    heat: Sequence[float] = HEAT_INDEX,
) -> Any:
    n = len(rain)
    fields: dict[str, Any] = {
        "asset_ids": ("A",),
        "period": np.arange(n, dtype=np.int64),
        "probability": np.full(n, 1.0 / n),
        "index": np.asarray(rain, dtype=np.float64),
        "loss": np.asarray(loss, dtype=np.float64),
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": SampleProvenance(
            source_kind="table",
            sources=(),
            semantics_source="caller",
            hazard_dependence={"kind": "data"},
            combine={"kind": "table"},
        ),
        "index_components": {
            "rain": IndexComponent(
                np.asarray(rain, dtype=np.float64),
                unit="mm/day",
                tail=tail,
                hazard=None,
                own_loss=None,
            ),
            "heat": IndexComponent(
                np.asarray(heat, dtype=np.float64),
                unit="degC",
                tail="upper",
                hazard=None,
                own_loss=None,
            ),
        },
    }
    untyped: Any = OutcomeSample
    return untyped(**fields)


def _two_cover(payout: Any, tail: Tail = "upper") -> Cover:
    rain = IndexDefinition("rain", "mm/day", tail=tail)
    return Cover(
        CompositeIndex({"rain": rain, "heat": HEAT}), payout, payout_unit="EUR"
    )


def _rain_part(
    levels: Sequence[float | Free] = (Free(), Free()), tail: Tail = "upper"
) -> StepPayout:
    if tail == "upper":
        return StepPayout([Free(0.0, 8.0, grid=[0.0, 5.0, 8.0]), 8.5], list(levels))
    return StepPayout(
        [Free(2.0, 10.0, grid=[2.0, 5.0, 10.0]), 1.5], list(levels), tail="lower"
    )


def _composites() -> dict[str, tuple[Cover, Any, str]]:
    """Payouts built from steps, their samples and the rain threshold's name."""
    both = [a + b for a, b in zip(RAIN_STEP, HOT, strict=True)]
    larger = [max(a, b) for a, b in zip(RAIN_STEP, HOT, strict=True)]
    grid_loss = [50.0 * (x >= 4.31) + h for x, h in zip(STEP_INDEX, HOT, strict=True)]
    return {
        "sum_of": (
            _two_cover(
                SumOf({"rain": _rain_part(), "heat": BinaryPayout(5.0, Free())})
            ),
            _two(STEP_INDEX, both),
            "rain.breakpoints[0]",
        ),
        "sum_of_lower": (
            _two_cover(
                SumOf(
                    {
                        "rain": _rain_part(tail="lower"),
                        "heat": BinaryPayout(5.0, Free()),
                    }
                ),
                tail="lower",
            ),
            _two([10.0 - x for x in STEP_INDEX], both, tail="lower"),
            "rain.breakpoints[0]",
        ),
        "max_of": (
            _two_cover(
                MaxOf(
                    {
                        "rain": _rain_part([50.0, 100.0]),
                        "heat": BinaryPayout(5.0, 30.0),
                    },
                    scale=Free(),
                )
            ),
            _two(STEP_INDEX, larger),
            "rain.breakpoints[0]",
        ),
        "step_grid": (
            _two_cover(
                StepGrid(
                    {"rain": [Free(0.0, 8.0, grid=[0.0, 5.0, 8.0])], "heat": [5.0]},
                    [[0.0, Free()], [Free(), Free()]],
                )
            ),
            _two(STEP_INDEX, grid_loss),
            "rain.breakpoints[0]",
        ),
    }


@pytest.mark.parametrize("family", ["sum_of", "sum_of_lower", "max_of", "step_grid"])
@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_a_payout_built_from_steps_is_refined_over_the_index_values(
    family: str,
    objective: Objective,
    terms: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each loss is the family's payout with the rain threshold at 4.31 (5.69
    # on the mirrored rain of a lower tail), between the grid's points and in
    # a set of years that no compass step from the grid's best point reaches.
    template, sample, name = _composites()[family]
    exact = 4.31 if family != "sum_of_lower" else 10.0 - 4.31
    plain = design(template, sample, objective, **terms)
    refined = design(template, sample, objective, options=REFINE, **terms)
    compass = _compass(monkeypatch, template, sample, objective, **terms)
    record = refined.refinement
    assert record is not None and record["method"] == INDEX_VALUES_METHOD
    assert refined.parameters[name] == exact
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12
    assert _scaled(plain, plain.objective_value) > 1e-3
    assert record["end"] == {name: exact} and record["stopped"] == "no_move"
    low, high = record["windows"][name]
    assert low <= plain.parameters[name] <= high and low <= exact <= high
    assert compass.refinement is not None
    assert compass.refinement["method"] == REFINE_METHOD
    assert compass.parameters[name] != exact
    assert _scaled(compass, compass.objective_value) > 1e-3


def test_a_payout_with_another_free_threshold_keeps_the_compass_search() -> None:
    # A linear component's free attachment is not a step's threshold, so the
    # whole payout keeps the compass search.
    template = _two_cover(
        SumOf(
            {
                "rain": _rain_part(),
                "heat": LinearPayout(Free(0.0, 9.0, grid=[1.0, 5.0]), 9.5, Free()),
            }
        )
    )
    both = [a + b for a, b in zip(RAIN_STEP, HOT, strict=True)]
    refined = design(template, _two(STEP_INDEX, both), MSE(), options=REFINE)
    assert refined.refinement is not None
    assert refined.refinement["method"] == REFINE_METHOD


def test_a_payout_whose_steps_have_fixed_thresholds_keeps_the_compass_search() -> None:
    # Only a linear component's threshold is free: the compass search.
    template = _two_cover(
        SumOf(
            {
                "rain": StepPayout([4.31, 8.5], [Free(), Free()]),
                "heat": LinearPayout(Free(0.0, 9.0, grid=[1.0, 5.0]), 9.5, Free()),
            }
        )
    )
    both = [a + b for a, b in zip(RAIN_STEP, HOT, strict=True)]
    refined = design(template, _two(STEP_INDEX, both), MSE(), options=REFINE)
    assert refined.refinement is not None
    assert refined.refinement["method"] == REFINE_METHOD


# -- a band left without a year -------------------------------------------------

# Six years, a loss of 100 in the last. Every breakpoint at an index value
# below the fixed 5.5 puts the year at 5 in the first band, which pays its
# level of at least 20 on a year without a loss; one between 5 and 5.5 leaves
# that band empty, and the second band pays the loss exactly.
EMPTY_INDEX = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
EMPTY_LOSS = [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]


def _emptied(level: float | Free, tail: Tail = "upper") -> Cover:
    if tail == "upper":
        payout = StepPayout([Free(0.0, 6.0, grid=[1.0, 3.0]), 5.5], [level, Free()])
        return Cover(INDEX, payout, payout_unit="EUR")
    payout = StepPayout(
        [Free(4.0, 10.0, grid=[7.0, 9.0]), 4.5], [level, Free()], tail="lower"
    )
    return Cover(LOWER, payout, payout_unit="EUR")


@pytest.mark.parametrize("tail", ["upper", "lower"])
@pytest.mark.parametrize("level", [20.0, Free(20.0, 50.0)], ids=["fixed", "bounded"])
@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_an_empty_band_is_tried_with_a_fixed_or_bounded_level(
    objective: Objective, terms: dict[str, Any], level: float | Free, tail: Tail
) -> None:
    index = EMPTY_INDEX if tail == "upper" else [10.0 - x for x in EMPTY_INDEX]
    sample = _sample(index, EMPTY_LOSS)
    plain = design(_emptied(level, tail), sample, objective, **terms)
    refined = design(_emptied(level, tail), sample, objective, options=REFINE, **terms)
    middle = 5.25 if tail == "upper" else 4.75
    assert refined.parameters["breakpoints[0]"] == middle
    assert refined.parameters["levels[1]"] == pytest.approx(100.0, rel=1e-9)
    assert "levels[0]" in refined.unidentified or not isinstance(level, Free)
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12
    assert _scaled(plain, plain.objective_value) > 1e-3
    record = refined.refinement
    assert record is not None and record["stopped"] == "no_move"
    # The index values and the window's ends; the empty band's value is extra.
    assert record["candidates"] == {"breakpoints[0]": 6}
    # Without that value, the best is the index value next to it, which pays
    # the level on the year without a loss.
    best_index_value = min(
        (row["objective"], row["breakpoints[0]"])
        for row in _rows(refined)
        if row["objective"] is not None and row["breakpoints[0]"] != middle
    )
    assert _scaled(refined, best_index_value[0]) > 1e-3


@pytest.mark.parametrize(("objective", "terms"), OBJECTIVES, ids=OBJECTIVE_IDS)
def test_an_empty_band_is_not_tried_with_free_levels(
    objective: Objective, terms: dict[str, Any]
) -> None:
    # With the first level free, a breakpoint at any index value up to 5 pays
    # it 0 and does as well, so only the index values and the window's ends
    # are tried: the grid's best point, 1, ties with them and is kept.
    sample = _sample(EMPTY_INDEX, EMPTY_LOSS)
    refined = design(_emptied(Free()), sample, objective, options=REFINE, **terms)
    assert refined.parameters["breakpoints[0]"] in EMPTY_INDEX
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12
    assert {row["breakpoints[0]"] for row in _rows(refined)} <= set(EMPTY_INDEX)


def test_the_empty_band_follows_the_next_threshold() -> None:
    # Two free breakpoints, the first with a fixed level: its empty band's
    # value lies between the last index value below the second and the
    # second, wherever the second is.
    template = Cover(
        INDEX,
        StepPayout(
            [Free(0.0, 6.0, grid=[1.0, 3.0]), Free(3.0, 7.0, grid=[4.5, 6.5])],
            [20.0, Free()],
        ),
        payout_unit="EUR",
    )
    sample = _sample(EMPTY_INDEX, EMPTY_LOSS)
    context: Any = SimpleNamespace(index=np.asarray(EMPTY_INDEX))
    names = ["breakpoints[0]", "breakpoints[1]"]
    axes = DESIGN_MODULE._threshold_axes(template.payout, names, context)
    extra = DESIGN_MODULE._empty_bands(
        template, names, axes, [(1.0, 6.0), (4.5, 6.5)], context
    )
    assert extra is not None
    assert extra(0, (3.0, 5.5)) == (5.25,)
    assert extra(0, (3.0, 4.5)) == (4.25,)
    assert extra(0, (3.0, 6.5)) == ()  # 6.5 lies beyond the first's window
    assert extra(1, (3.0, 5.5)) == ()  # the last breakpoint closes no band
    refined = design(template, sample, MSE(), options=REFINE)
    end = refined.parameters
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12
    assert end["breakpoints[0]"] < end["breakpoints[1]"]
    assert not any(
        end["breakpoints[0]"] <= x < end["breakpoints[1]"] for x in EMPTY_INDEX
    )


def test_an_empty_band_of_a_step_grid_is_tried_with_fixed_levels() -> None:
    # The grid's second row has fixed levels of 20, which the years of rain
    # 5 would pay: a rain breakpoint between 5 and the fixed 5.5 empties it.
    template = _two_cover(
        StepGrid(
            {"rain": [Free(0.0, 6.0, grid=[1.0, 3.0]), 5.5], "heat": [5.0]},
            [[0.0, Free()], [20.0, 20.0], [Free(), Free()]],
        )
    )
    sample = _two(EMPTY_INDEX, EMPTY_LOSS, heat=[1.0] * len(EMPTY_INDEX))
    refined = design(template, sample, MSE(), options=REFINE)
    assert refined.parameters["rain.breakpoints[0]"] == 5.25
    assert 0.0 <= _scaled(refined, refined.objective_value) <= 1e-12


# -- warm starts ------------------------------------------------------------------

WARM_TERMS: list[tuple[str, Objective, dict[str, Any]]] = [
    ("eu-pure", ExpectedUncovered(), {"budget": Budget(ratio=0.6)}),
    (
        "wcvar-coc",
        WeightedCVaR(0.7, 0.9),
        {"pricing": CostOfCapitalLoad(0.06, 0.95), "budget": Budget(ratio=0.6)},
    ),
    (
        "eu-multiple",
        ExpectedUncovered(),
        {"pricing": ExpectedPayoutMultiple(1.3, 1000.0), "budget": Budget(ratio=0.6)},
    ),
    ("wcvar-limit", WeightedCVaR(1.0, 0.8), {"limit": Limit(1.5e6)}),
]


def _warm_battery() -> dict[str, tuple[Cover, OutcomeSample]]:
    rng = np.random.default_rng(5)
    x = rng.gumbel(50.0, 12.0, 300)
    loss = 2e6 * np.clip((x - 45.0) / 100.0, 0.0, 0.6) * rng.lognormal(0.0, 0.3, 300)
    sample = _sample(x.tolist(), loss.tolist())
    lower = _sample((200.0 - x).tolist(), loss.tolist())
    return {
        "step": (
            Cover(
                INDEX,
                StepPayout(
                    [
                        Free(40.0, 80.0, grid=5),
                        90.0,
                        Free(95.0, 140.0, grid=[100.0, 120.0]),
                    ],
                    [Free(), Free(), Free()],
                ),
                payout_unit="EUR",
            ),
            sample,
        ),
        "binary_lower": (
            Cover(
                LOWER,
                BinaryPayout(Free(100.0, 160.0, grid=13), Free(), tail="lower"),
                payout_unit="EUR",
            ),
            lower,
        ),
        "step_fixed_level": (
            Cover(
                INDEX,
                StepPayout(
                    [Free(40.0, 80.0, grid=5), Free(85.0, 120.0, grid=3)],
                    [2e5, Free()],
                ),
                payout_unit="EUR",
            ),
            sample,
        ),
    }


@pytest.mark.parametrize("terms_id", [item[0] for item in WARM_TERMS])
def test_warm_starts_give_the_designs_of_solves_from_scratch(
    terms_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, objective, terms = next(item for item in WARM_TERMS if item[0] == terms_id)
    warm_solves: list[int] = []
    solve = DESIGN_MODULE._WarmLP.solve

    def counted(self: Any, *arguments: Any) -> Any:
        found = solve(self, *arguments)
        warm_solves.append(found is not None)
        return found

    for name, (template, sample) in _warm_battery().items():
        with monkeypatch.context() as patched:
            patched.setattr(DESIGN_MODULE._WarmLP, "solve", counted)
            warm = design(template, sample, objective, options=REFINE, **terms)
        with monkeypatch.context() as patched:
            patched.setattr(DESIGN_MODULE, "WARM_SEATS", 0)
            cold = design(template, sample, objective, options=REFINE, **terms)
        assert warm.design_id == cold.design_id, name
        assert warm.parameters == cold.parameters, name
        assert warm.cover.definition_hash == cold.cover.definition_hash, name
        assert warm.objective_value == cold.objective_value, name
        assert warm.refinement == cold.refinement, name
        assert (warm.ties, warm.unique) == (cold.ties, cold.unique), name
        warm_rows, cold_rows = _rows(warm), _rows(cold)
        assert len(warm_rows) == len(cold_rows), name
        for one, other in zip(warm_rows, cold_rows, strict=True):
            assert one["status"] == other["status"], name
            if one["objective"] is None:
                assert other["objective"] is None
            else:
                assert one["objective"] == pytest.approx(other["objective"], rel=1e-9)
    assert sum(warm_solves) > 0.9 * len(warm_solves)


def test_the_warm_lp_has_the_optimum_of_the_design_lp() -> None:
    # Over the years with a loss, with a shared excess column for the others:
    # on random step bases, every objective and pricing gives the optimum of
    # the full design LP.
    rng = np.random.default_rng(2)
    n = 120
    loss = np.where(rng.uniform(size=n) < 0.6, 0.0, rng.lognormal(0.0, 1.0, n))
    index = rng.uniform(0.0, 10.0, n)
    template = StepPayout([Free(0.0, 10.0, grid=3), 7.0], [Free(), Free(0.0, 9.0)])
    for objective, pricing, ratio in [
        (ExpectedUncovered(), PurePremium(), 0.5),
        (WeightedCVaR(0.6, 0.85), PurePremium(), 0.5),
        (WeightedCVaR(1.0, 0.9), CostOfCapitalLoad(0.1, 0.9), 0.7),
        (ExpectedUncovered(), ExpectedPayoutMultiple(1.2, 0.01), 0.8),
        (WeightedCVaR(0.5, 0.9), PurePremium(), None),
    ]:
        expected_loss = float(np.mean(loss))
        context = DESIGN_MODULE._Context(
            asset_id="A",
            index=index,
            loss=loss,
            probability=np.full(n, 1.0 / n),
            scale=float(loss.max()),
            expected_loss=expected_loss,
            bound=None if ratio is None else ratio * expected_loss,
            limit=None,
            objective=objective,
            pricing=pricing,
            options=DesignOptions(),
        )
        for first in rng.uniform(0.0, 6.9, 5):
            payout = template.with_values({"breakpoints[0]": float(first)})
            basis = payout.basis(index)
            constraints = payout.amount_constraints(None)
            full, _, _ = DESIGN_MODULE._programme(basis, constraints, context)
            reduced, layout = DESIGN_MODULE._warm_programme(basis, constraints, context)
            assert reduced.n_columns < full.n_columns
            one = _solver.solve_lp(full)
            other = _solver.solve_lp(reduced)
            assert one.status == other.status == "optimal"
            assert one.objective is not None and other.objective is not None
            assert other.objective == pytest.approx(one.objective, rel=1e-9, abs=1e-12)
            assert (layout.budget is None) == (ratio is None)


def test_a_warm_solve_that_fails_is_solved_from_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every third warm solve reports an error: the point is solved from
    # scratch, its seat dropped, and the design is that of solves from scratch.
    template, sample = _warm_battery()["step"]
    terms: dict[str, Any] = {"budget": Budget(ratio=0.6)}
    calls = [0]
    solve = _solver.LPSession.solve

    def failing(self: Any, *arguments: Any) -> Any:
        calls[0] += 1
        found = solve(self, *arguments)
        if calls[0] % 3 == 0:
            return dataclasses.replace(found, status="error", values=None)
        return found

    with monkeypatch.context() as patched:
        patched.setattr(_solver.LPSession, "solve", failing)
        warm = design(template, sample, ExpectedUncovered(), options=REFINE, **terms)
    with monkeypatch.context() as patched:
        patched.setattr(DESIGN_MODULE, "WARM_SEATS", 0)
        cold = design(template, sample, ExpectedUncovered(), options=REFINE, **terms)
    assert calls[0] > 3
    assert warm.parameters == cold.parameters
    assert warm.refinement == cold.refinement


def test_a_warm_solution_that_fails_its_checks_is_solved_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A warm solution whose objective is wrong fails the check against numpy;
    # the point is solved again from scratch, and the design is unchanged.
    template, sample = _warm_battery()["step"]
    terms: dict[str, Any] = {"budget": Budget(ratio=0.6)}
    solve = DESIGN_MODULE._WarmLP.solve
    spoiled = [0]

    def spoiling(self: Any, *arguments: Any) -> Any:
        found = solve(self, *arguments)
        if found is None:
            return None
        spoiled[0] += 1
        return dataclasses.replace(found, objective=found.objective + 1.0)

    with monkeypatch.context() as patched:
        patched.setattr(DESIGN_MODULE._WarmLP, "solve", spoiling)
        warm = design(template, sample, ExpectedUncovered(), options=REFINE, **terms)
    with monkeypatch.context() as patched:
        patched.setattr(DESIGN_MODULE, "WARM_SEATS", 0)
        cold = design(template, sample, ExpectedUncovered(), options=REFINE, **terms)
    assert spoiled[0] > 0
    assert warm.parameters == cold.parameters
    assert warm.refinement == cold.refinement


def test_warm_starts_keep_a_few_seats_and_skip_the_qp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template, sample = _warm_battery()["step"]
    seats: list[int] = []
    new = DESIGN_MODULE._WarmLP._new

    def counted(self: Any, *arguments: Any) -> Any:
        seat = new(self, *arguments)
        seats.append(len(self._seats))
        return seat

    monkeypatch.setattr(DESIGN_MODULE._WarmLP, "_new", counted)
    design(
        template,
        sample,
        ExpectedUncovered(),
        budget=Budget(ratio=0.6),
        options=REFINE,
    )
    assert 1 <= len(seats) and max(seats) <= DESIGN_MODULE.WARM_SEATS
    seats.clear()
    design(template, sample, MSE(), options=REFINE)
    assert seats == []
