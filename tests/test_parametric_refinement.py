"""The local refinement after a grid: `DesignOptions(refine=True)`.

From the grid's best point over the free thresholds, with the amounts solved
exactly at every point it evaluates: over the sample's index values for a
step or binary payout, by a compass search otherwise. The cases are
synthetic: losses made by a payout of the same family, so the optimum is
known in closed form (an objective of 0 at the generating thresholds).
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Self

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    Cover,
    DesignError,
    DesignOptions,
    DesignResult,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    Joint,
    Limit,
    LinearConstraints,
    LinearPayout,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    PiecewiseLinearPayout,
    SampleProvenance,
    StepPayout,
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
    assert record["moves"] == 1 and record["passes"] == 2
    assert record["stopped"] == "no_move"
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
    # end 0, ten values, tried nearest 5 first. 6 pays the same years as 5,
    # the start, and 0 those of 1: neither is solved. The second pass finds
    # every set of years tried.
    sample = _sample(STEP_INDEX, STEP_LOSS)
    refined = design(_step(), sample, MSE(), options=REFINE)
    record = refined.refinement
    assert record is not None
    assert record["windows"] == {"breakpoints[0]": [0.0, 8.0]}
    assert record["candidates"] == {"breakpoints[0]": 10}
    tried = [row["breakpoints[0]"] for row in _rows(refined)]
    assert tried == [4.31, 4.3, 4.0, 3.0, 7.0, 2.0, 8.0, 1.0]
    assert record["points"] == len(tried) == record["counts"]["optimal"]


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
    assert record["passes"] == 1
    # The two nearest values were tried, and the better one moved to.
    assert capped.parameters["breakpoints[0]"] == 4.31
    assert capped.objective_value <= plain.objective_value
    line = _design_line(capped, sample)
    assert "after 2 points in 1 pass (" in line
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
        ", a gain of 250, after 8 points in 2 passes (each threshold over the "
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
    assert found.moves == 1 and found.passes == 2 and found.stopped == "no_move"
    # Nearest the start first, the lower one first at equal distance; the
    # second pass finds every candidate evaluated.
    assert calls[:4] == [(7.0,), (9.0,), (6.0,), (10.0,)]
    assert len(calls) == 10 == len(found.points)


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

    # Pass 1 moves x0 to 3 (x1 held at 5), then x1 to 3; pass 2 moves both
    # to 2; pass 3 moves neither.
    found, _ = _coordinate(coupled, [-5.0, 5.0], [grid, grid])
    assert found.best == (2.0, 2.0) and found.objective == 0.0
    assert (found.moves, found.passes, found.stopped) == (4, 3, "no_move")


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
    assert found.best == (2.0,) and found.passes == 1


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
