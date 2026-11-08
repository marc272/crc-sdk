"""Joint designs: parity with the golden fixtures, the total, the report.

The fixture cases are read as JSON with plain numpy, as
`tests/fixtures/parametric/README.md` describes; no reference code is
imported. Every design case of several assets in the folder is judged, so new
ones are picked up when they are copied. The metrics of the total are in
`tests/test_parametric_evaluate_joint.py`.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pytest
from numpy.typing import ArrayLike

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CostOfCapitalLoad,
    Cover,
    DesignError,
    DesignOptions,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    Individual,
    InfeasibleDesignError,
    Joint,
    JointDesignResult,
    Limit,
    LinearPayout,
    OutcomeSample,
    PurePremium,
    SampleError,
    SampleProvenance,
    SolverUnavailableError,
    StepPayout,
    UserObjective,
    WeightedCVaR,
    _solver,
    design,
    design_joint,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.constraints import budget_from_document, limit_from_document
from crc_sdk.parametric.design import DESIGN_SEMANTICS_VERSION
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.joint import covers_hash
from crc_sdk.parametric.objectives import objective_from_document
from crc_sdk.parametric.parameters import _DefinitionModel
from crc_sdk.parametric.pricing import Pricing, pricing_from_document

DESIGN_MODULE = sys.modules[design.__module__]
CASES = Path(__file__).parent / "fixtures" / "parametric" / "cases"


# -- the fixture cases -------------------------------------------------------


def _case(path: Path) -> dict[str, Any]:
    case: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return case


def _joint_cases() -> list[Path]:
    return [
        path for path in sorted(CASES.glob("*.json")) if _case(path)["scope"] == "joint"
    ]


JOINT_CASES = _joint_cases()


def _case_sample(case: Mapping[str, Any]) -> OutcomeSample:
    block = case["sample"]
    return OutcomeSample(
        asset_ids=tuple(block["asset_ids"]),
        period=np.asarray(block["period"], dtype=np.int64),
        probability=np.asarray(block["probability"], dtype=np.float64),
        index=np.asarray(block["index"], dtype=np.float64),
        loss=np.asarray(block["loss"], dtype=np.float64),
        hazard=None,
        index_unit=block["index_unit"],
        loss_unit=block["loss_unit"],
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": case["case_id"]},),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )


def _templates(case: Mapping[str, Any]) -> list[Cover]:
    return [Cover.from_document(document) for document in case["cover"]]


def _design_case(
    case: Mapping[str, Any], sample: OutcomeSample, templates: Sequence[Cover]
) -> JointDesignResult:
    return design_joint(
        templates,
        sample,
        objective_from_document(case["objective"]),
        pricing=pricing_from_document(case["pricing"]),
        budget=budget_from_document(case["budget"]),
        limit=limit_from_document(case["limit"]),
        options=DesignOptions(ranges=True),
    )


def _scale(case: Mapping[str, Any]) -> float:
    return float(max(sum(row) for row in case["sample"]["loss"]))


def _total_payout(
    covers: Sequence[Cover], sample: OutcomeSample
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.sum(
        [
            cover.payout_values(sample.index[:, column])
            for column, cover in enumerate(covers)
        ],
        axis=0,
    )


def test_the_joint_cases_are_present() -> None:
    names = {path.stem for path in JOINT_CASES}
    objectives = ("mse", "eu", "wcvar-b0_9", "wcvar-b0_99")
    assert names == {
        f"step-{objective}-{pricing}-joint-{data}"
        for objective in objectives
        for pricing in ("pure", "coc")
        for data in ("synthetic", "thesis")
    }
    cases = [_case(path) for path in JOINT_CASES]
    modes = [case["compare_parameters"] for case in cases]
    assert (
        modes.count("exact"),
        modes.count("within_ranges"),
        modes.count("none"),
    ) == (
        5,
        9,
        2,
    )
    for case in cases:
        assert case["kind"] == "design" and case["split"] is None
        assert len(case["sample"]["asset_ids"]) == 3
        assert len(case["cover"]) == 3
        for document in case["cover"]:
            # Thresholds are fixed in every joint case.
            assert all(isinstance(b, float) for b in document["payout"]["breakpoints"])


@pytest.mark.parametrize("path", JOINT_CASES, ids=lambda path: path.stem)
def test_joint_design_parity(path: Path) -> None:
    case = _case(path)
    expected = case["expected"]
    tolerances = case["tolerances"]
    scale = _scale(case)
    sample = _case_sample(case)
    templates = _templates(case)
    result = _design_case(case, sample, templates)
    assert result.asset_ids == sample.asset_ids
    assert set(expected["parameters"]) == set(sample.asset_ids)

    # 1. The objective value.
    power = 2 if case["objective"]["kind"] == "mse" else 1
    target = expected["objective_value"]
    assert abs(result.objective_value - target) <= (
        tolerances["objective_value"] * abs(target) + 1e-9 * scale**power
    )

    # 2. The parameters.
    allowance = tolerances["parameters"] * scale
    mode = case["compare_parameters"]
    for asset, values in expected["parameters"].items():
        unidentified = expected["unidentified"][asset]
        assert set(result.unidentified[asset]) == set(unidentified), asset
        for name, value in values.items():
            if name in unidentified:
                continue
            if mode == "exact":
                assert abs(result.parameters[asset][name] - value) <= allowance, name
            elif mode == "within_ranges":
                assert result.ranges is not None
                low, high = result.ranges[asset][name]
                assert low - allowance <= value <= high + allowance, (asset, name)
            else:
                assert mode == "none"
        # Unidentified levels take the fill rule's value, per asset.
        levels = result.parameters[asset]
        for position, name in enumerate(levels):
            if name in result.unidentified[asset]:
                floor = levels[f"levels[{position - 1}]"] if position else 0.0
                assert levels[name] == floor, (asset, name)

    # 3. The premium: of the designed covers if the optimum is unique, else of
    # the templates resolved with the case's parameters, on the total payout.
    pricing = pricing_from_document(case["pricing"])
    if expected["unique"]:
        premium = result.premium
    else:
        resolved = [
            template.with_values(expected["parameters"][asset])
            for asset, template in zip(sample.asset_ids, templates, strict=True)
        ]
        premium = pricing.premium(_total_payout(resolved, sample), sample.probability)
    assert abs(premium - expected["premium"]) <= (
        tolerances["premium"] * abs(expected["premium"]) + 1e-9 * scale
    )

    # 4. The module's uniqueness never claims more than the case's.
    if expected["unique"] is False:
        assert result.unique is not True
    if expected["unique"] and mode == "exact":
        assert result.unique is True


@pytest.mark.parametrize("path", JOINT_CASES, ids=lambda path: path.stem)
def test_the_premium_and_the_budget_are_on_the_total(path: Path) -> None:
    case = _case(path)
    sample = _case_sample(case)
    result = _design_case(case, sample, _templates(case))
    total_loss = sample.loss.sum(axis=1)
    probability = sample.probability
    expected_loss = float(probability @ total_loss)
    pricing = pricing_from_document(case["pricing"])
    payout = _total_payout(list(result.covers.values()), sample)

    assert result.expected_loss == pytest.approx(expected_loss, rel=1e-12)
    assert result.budget_bound is not None
    assert result.budget_bound == pytest.approx(0.8 * expected_loss, rel=1e-12)
    assert result.premium == pytest.approx(
        pricing.premium(payout, probability), rel=1e-12
    )
    assert result.premium <= result.budget_bound * (1.0 + 1e-7)
    assert result.budget_use == pytest.approx(result.premium / result.budget_bound)
    objective = objective_from_document(case["objective"])
    assert result.objective_value == pytest.approx(
        objective.value(total_loss, payout, probability), rel=1e-12
    )
    assert result.p_zero_loss == pytest.approx(
        float(probability[total_loss == 0.0].sum()), abs=1e-12
    )
    if isinstance(objective, WeightedCVaR):
        n = probability.shape[0]
        assert result.tail_outcomes == pytest.approx((1.0 - objective.beta) * n)
    else:
        assert result.tail_outcomes is None
    assert result.degenerate_tail is None
    amounts = "qp" if case["objective"]["kind"] == "mse" else "lp"
    assert result.route == result.amounts == amounts
    assert (result.ties, result.profile, result.grid) == (0, None, None)
    assert result.solver["name"] == "highs"


# -- small samples -----------------------------------------------------------


def _sample(
    index: ArrayLike,
    loss: ArrayLike,
    *,
    asset_ids: tuple[str, ...] = ("A", "B"),
    index_unit: str = "m/s",
    loss_unit: str = "EUR",
) -> OutcomeSample:
    values = np.asarray(index, dtype=np.float64)
    n = values.shape[0]
    return OutcomeSample(
        asset_ids=asset_ids,
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=values,
        loss=np.asarray(loss, dtype=np.float64),
        hazard=None,
        index_unit=index_unit,
        loss_unit=loss_unit,
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )


INDEX = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
# Asset A pays in year 1 and asset B in year 2; each covers its own loss.
SEPARATE = _sample(
    [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
    [[0.0, 0.0], [40.0, 0.0], [0.0, 20.0], [0.0, 0.0]],
)


def _step(
    levels: Sequence[float | Free],
    breakpoints: Sequence[float | Free] = (1.0,),
    *,
    asset: str | None = None,
    label: str | None = None,
) -> Cover:
    return Cover(
        INDEX,
        StepPayout(breakpoints, levels),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _pair() -> list[Cover]:
    return [_step([Free()], asset="A"), _step([Free()], asset="B")]


def _level(result: JointDesignResult, asset: str, position: int = 0) -> float:
    return result.parameters[asset][f"levels[{position}]"]


# -- what it optimises -------------------------------------------------------


def test_a_payout_offsets_the_total_loss_of_its_draw() -> None:
    # Only A's index reaches its trigger, in a year when A has no loss and B
    # has 30. On the total, A's cover pays against that loss. B's cover never
    # pays, so its level is unidentified.
    sample = _sample(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 30.0], [0.0, 0.0], [0.0, 0.0]],
    )
    result = design_joint(
        _pair(), sample, ExpectedUncovered(), budget=Budget(ratio=1.0)
    )
    assert _level(result, "A") == pytest.approx(30.0)
    assert result.objective_value == pytest.approx(0.0, abs=1e-9)
    assert result.premium == pytest.approx(7.5)
    assert result.budget_bound == pytest.approx(7.5)
    assert result.unidentified == {"A": (), "B": ("levels[0]",)}
    assert _level(result, "B") == 0.0


@pytest.mark.parametrize(
    "budget", [Budget(ratio=0.5), Budget(amount=7.5)], ids=["ratio", "amount"]
)
def test_one_budget_is_shared_and_its_split_is_reported(budget: Budget) -> None:
    # E[L] of the total is 15, so both budgets are 7.5: the two levels may sum
    # to 30. A's level lies in [10, 40] and B's in [0, 20], and every split
    # with a + b = 30 gives E[U] = (60 - 30) / 4.
    result = design_joint(
        _pair(),
        SEPARATE,
        ExpectedUncovered(),
        budget=budget,
        options=DesignOptions(ranges=True),
    )
    assert result.objective_value == pytest.approx(7.5)
    assert result.premium == pytest.approx(7.5)
    assert result.budget_bound == pytest.approx(7.5)
    assert result.budget_use == pytest.approx(1.0)
    assert _level(result, "A") + _level(result, "B") == pytest.approx(30.0)
    assert result.unique is False
    assert result.ranges is not None
    assert result.ranges["A"]["levels[0]"] == pytest.approx((10.0, 30.0), abs=1e-4)
    assert result.ranges["B"]["levels[0]"] == pytest.approx((0.0, 20.0), abs=1e-4)


def test_without_ranges_the_lp_route_reports_no_uniqueness() -> None:
    result = design_joint(
        _pair(), SEPARATE, ExpectedUncovered(), budget=Budget(amount=7.5)
    )
    assert result.unique is None and result.ranges is None


def test_a_loaded_premium_is_priced_on_the_total_payout() -> None:
    # The covers pay in different years, so the CVaR of the total at q = 0.75
    # is the larger of the two levels: the premium is (a + b) / 8 + max(a, b) / 2.
    # A budget of 15 and E[U] = (120 - a - b) / 4 give a = b = 20.
    sample = _sample(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
        [[0.0, 0.0], [60.0, 0.0], [0.0, 60.0], [0.0, 0.0]],
    )
    pricing = CostOfCapitalLoad(0.5, 0.75)
    result = design_joint(
        _pair(),
        sample,
        ExpectedUncovered(),
        pricing=pricing,
        budget=Budget(amount=15.0),
    )
    assert _level(result, "A") == pytest.approx(20.0)
    assert _level(result, "B") == pytest.approx(20.0)
    assert result.premium == pytest.approx(15.0)
    assert result.objective_value == pytest.approx(20.0)


def test_the_covers_stay_separate_contracts_with_their_own_terms() -> None:
    # MSE, closed form per band, unique. A's two bands are pooled by its
    # monotone levels (30 and 10 become 20 and 20); B's are not pooled, and
    # its first level lies below A's last.
    sample = _sample(
        [[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [0.0, 1.5], [0.0, 3.0], [0.0, 0.0]],
        [[0.0, 0.0], [30.0, 0.0], [10.0, 0.0], [0.0, 5.0], [0.0, 20.0], [0.0, 0.0]],
    )
    templates = [
        _step([Free(), Free()], (1.0, 2.5), asset="A"),
        _step([Free(), Free()], (1.0, 2.5), asset="B"),
    ]
    result = design_joint(templates, sample, MSE())
    assert result.parameters["A"] == pytest.approx(
        {"levels[0]": 20.0, "levels[1]": 20.0}
    )
    assert result.parameters["B"] == pytest.approx(
        {"levels[0]": 5.0, "levels[1]": 20.0}
    )
    assert result.unique is True
    assert result.route == "qp"


def test_families_may_differ_between_assets() -> None:
    sample = _sample(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 0.0]],
        [[0.0, 0.0], [10.0, 0.0], [0.0, 20.0], [0.0, 0.0]],
    )
    templates = [
        Cover(INDEX, BinaryPayout(1.0, Free()), payout_unit="EUR", asset_id="A"),
        Cover(INDEX, LinearPayout(0.0, 10.0, Free()), payout_unit="EUR", asset_id="B"),
    ]
    result = design_joint(templates, sample, MSE())
    assert result.parameters["A"] == pytest.approx({"amount": 10.0})
    assert result.parameters["B"] == pytest.approx({"limit": 40.0})
    assert result.covers["A"].payout.kind == "binary"
    assert result.covers["B"].payout.kind == "linear"
    assert result.objective_value == pytest.approx(0.0, abs=1e-9)


def test_a_limit_is_each_covers_own() -> None:
    # Both covers pay in the same year. A limit of 30 caps each cover, so the
    # total payout of that year is 60.
    sample = _sample([[0.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [40.0, 40.0]])
    result = design_joint(_pair(), sample, MSE(), limit=Limit(30.0))
    assert _level(result, "A") == pytest.approx(30.0)
    assert _level(result, "B") == pytest.approx(30.0)
    payout = _total_payout(list(result.covers.values()), sample)
    assert payout.max() == pytest.approx(60.0)
    assert result.limit == {"amount": 30.0}


def test_a_cover_with_fixed_terms_stays_as_written() -> None:
    # B's cover is fixed at 12; A's covers what is left of the total.
    sample = _sample([[0.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [20.0, 20.0]])
    templates = [_step([Free()], asset="A"), _step([12.0], asset="B")]
    result = design_joint(templates, sample, MSE())
    assert _level(result, "A") == pytest.approx(28.0)
    assert result.parameters["B"] == {}
    assert result.covers["B"] == templates[1]


def test_the_scale_is_the_largest_summed_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[tuple[int, int], float, list[float]]] = []
    real = DESIGN_MODULE._programme

    def spy(basis: Any, constraints: Any, context: Any) -> Any:
        seen.append((basis.shape, context.scale, context.loss.tolist()))
        return real(basis, constraints, context)

    monkeypatch.setattr(DESIGN_MODULE, "_programme", spy)
    sample = _sample(
        [[0.0, 0.0], [3.0, 3.0], [3.0, 0.0]], [[0.0, 0.0], [30.0, 25.0], [40.0, 0.0]]
    )
    design_joint(_pair(), sample, MSE())
    assert seen == [((3, 2), 55.0, [0.0, 55.0, 40.0])]


def test_a_template_for_no_asset_is_written_for_its_position() -> None:
    template = _step([Free()], label="storm")
    result = design_joint([template, template], SEPARATE, MSE())
    assert result.covers["A"].asset_id == "A"
    assert result.covers["B"].asset_id == "B"
    assert result.covers["A"].label == "storm"
    assert result.template_hashes == {
        "A": template.definition_hash,
        "B": template.definition_hash,
    }


# -- several optimal designs -------------------------------------------------


def test_a_singular_gram_matrix_is_not_unique() -> None:
    # Both covers pay in the same year, so only their sum is determined.
    sample = _sample([[0.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [20.0, 10.0]])
    result = design_joint(_pair(), sample, MSE())
    assert result.objective_value == pytest.approx(0.0, abs=1e-9)
    assert _level(result, "A") + _level(result, "B") == pytest.approx(30.0)
    assert result.unique is False


@pytest.mark.parametrize(("fill", "expected"), [("lowest", 10.0), ("highest", 40.0)])
def test_unidentified_amounts_are_filled_per_asset(fill: Any, expected: float) -> None:
    # A's middle band holds no year; B's bands all do. The covers pay in
    # different years, so every identified level is determined.
    sample = _sample(
        [[0.0, 0.0], [1.5, 0.0], [5.0, 0.0], [0.0, 1.5], [0.0, 3.0], [0.0, 5.0]],
        [[0.0, 0.0], [10.0, 0.0], [40.0, 0.0], [0.0, 5.0], [0.0, 10.0], [0.0, 15.0]],
    )
    levels = [Free(0.0, 50.0) for _ in range(3)]
    templates = [
        _step(levels, (1.0, 2.5, 4.0), asset="A"),
        _step(levels, (1.0, 2.5, 4.0), asset="B"),
    ]
    result = design_joint(templates, sample, MSE(), options=DesignOptions(fill=fill))
    assert result.unidentified == {"A": ("levels[1]",), "B": ()}
    assert result.fill == fill
    assert _level(result, "A", 0) == pytest.approx(10.0)
    assert _level(result, "A", 1) == pytest.approx(expected)
    assert _level(result, "A", 2) == pytest.approx(40.0)
    assert _level(result, "B", 1) == pytest.approx(10.0)


def test_amounts_near_zero_are_set_to_zero_and_listed_per_asset() -> None:
    result = design_joint(
        _pair(), SEPARATE, ExpectedUncovered(), budget=Budget(amount=1e-12)
    )
    assert result.parameters == {"A": {"levels[0]": 0.0}, "B": {"levels[0]": 0.0}}
    assert set(result.zeroed) == {"A", "B"}
    assert all(names in ((), ("levels[0]",)) for names in result.zeroed.values())


# -- the individual designs as a point of the joint problem ------------------


def _book(seed: int, n: int = 60) -> OutcomeSample:
    """A synthetic book of three assets sharing an annual shock."""
    rng = np.random.default_rng(seed)
    shock = 22.0 + rng.gamma(2.0, 7.0, (n, 1))
    hazard = shock + rng.normal(0.0, 3.0, (n, 3))
    index = np.maximum(hazard + rng.normal(0.0, 3.0, (n, 3)), 0.0)
    excess = np.maximum(hazard - 25.7, 0.0) / 49.0
    loss = np.array([10.0, 6.0, 3.0]) * excess**3 / (1.0 + excess**3)
    return _sample(index, loss, asset_ids=("A1", "A2", "A3"))


OBJECTIVES = [MSE(), ExpectedUncovered(), WeightedCVaR(0.9, 0.9)]
PRICINGS: list[Pricing] = [PurePremium(), CostOfCapitalLoad(0.06, 0.95)]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("pricing", PRICINGS, ids=["pure", "coc"])
@pytest.mark.parametrize("objective", OBJECTIVES, ids=["mse", "eu", "wcvar"])
def test_under_a_ratio_budget_the_individual_designs_are_a_feasible_point(
    objective: MSE | ExpectedUncovered | WeightedCVaR, pricing: Pricing, seed: int
) -> None:
    """Covers designed asset by asset are feasible for the joint problem.

    Under `Budget(ratio=k)` each asset's design has a premium of at most `k`
    times its own expected loss. The premium of the total payout is at most
    the sum of the premiums, because the premium is subadditive in the
    payout, and the expected losses sum to the expected total loss. So the
    premium of the total is within `k` times the expected total loss, and the
    covers' own constraints are those of the joint problem: together they are
    a feasible point. A minimum is at most the objective at any feasible
    point, so the joint optimum is at most their objective on the total.
    """
    book = _book(seed)
    kappa = 0.4 + 0.1 * seed
    template = Cover(
        INDEX,
        StepPayout((25.0, 30.0, 36.0, 43.0, 51.0, 60.0), [Free() for _ in range(6)]),
        payout_unit="EUR",
    )
    individual = [
        design(
            template,
            book.asset(asset),
            objective,
            pricing=pricing,
            budget=Budget(ratio=kappa),
        )
        for asset in book.asset_ids
    ]
    payout = _total_payout([result.cover for result in individual], book)
    probability = book.probability
    total_loss = book.loss.sum(axis=1)
    scale = float(total_loss.max())
    premium = pricing.premium(payout, probability)
    assert premium <= math.fsum(result.premium for result in individual) + 1e-9 * scale
    assert premium <= kappa * float(probability @ total_loss) + 1e-6 * scale

    joint = design_joint(
        [template, template, template],
        book,
        objective,
        pricing=pricing,
        budget=Budget(ratio=kappa),
    )
    at_the_point = objective.value(total_loss, payout, probability)
    assert joint.objective_value <= at_the_point + 1e-6 * scale**objective.homogeneity


def test_under_an_amount_budget_the_individual_premiums_sum_to_more_than_it() -> None:
    """The feasibility above needs a ratio budget.

    Under `Budget(amount=B)` each asset's design may use all of `B`, so the
    premiums of the covers sum to more than `B`, and together the covers are
    not a feasible point of a joint problem with that budget.
    """
    budget = 2.5
    individual = [
        design(
            _step([Free()]),
            SEPARATE.asset(asset),
            ExpectedUncovered(),
            budget=Budget(amount=budget),
        )
        for asset in SEPARATE.asset_ids
    ]
    assert [result.premium for result in individual] == pytest.approx([2.5, 2.5])
    payout = _total_payout([result.cover for result in individual], SEPARATE)
    premium = PurePremium().premium(payout, SEPARATE.probability)
    assert premium == pytest.approx(5.0)
    assert premium > budget


# -- errors ------------------------------------------------------------------


def _no_solve(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the design must fail before solving")


def test_a_sample_of_one_asset_raises_and_names_design() -> None:
    with pytest.raises(DesignError, match=r": use design") as raised:
        design_joint([_step([Free()], asset="A")], SEPARATE.asset("A"), MSE())
    assert raised.value.asset_id == "A"


def test_a_free_threshold_raises_before_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_solver, "solve_qp", _no_solve)
    monkeypatch.setattr(_solver, "solve_lp", _no_solve)
    templates = [
        _step([Free()], asset="A"),
        _step([Free()], (Free(0.5, 2.0, grid=[0.5, 1.0, 2.0]),), asset="B"),
    ]
    with pytest.raises(DesignError, match=r"breakpoints\[0\]") as raised:
        design_joint(templates, SEPARATE, MSE())
    assert "fixed thresholds" in str(raised.value)
    assert raised.value.asset_id == "B"


def test_a_user_objective_raises() -> None:
    objective = UserObjective(lambda loss, payout, p: 0.0, name="zero")
    with pytest.raises(DesignError, match="UserObjective"):
        design_joint(_pair(), SEPARATE, objective)


class Curved(_DefinitionModel):
    """A test family that is not linear in its amounts."""

    kind: ClassVar[str] = "test_joint_curved"
    amount: Any
    tail: Tail = "upper"

    def __init__(self, amount: Any) -> None:
        super().__init__(amount=amount)

    @property
    def index_names(self) -> tuple[str, ...]:
        return ("index",)

    def parameters(self) -> dict[str, float | Free]:
        return {"amount": self.amount}

    def with_values(self, values: Mapping[str, float]) -> Self:
        return type(self)({**self.parameters(), **values}["amount"])

    def evaluate(self, index: Any) -> Any:
        return self.amount * np.sqrt(np.maximum(index, 0.0))

    def to_document(self) -> dict[str, Any]:
        value = self.amount
        return {
            "kind": self.kind,
            "amount": value.to_document() if isinstance(value, Free) else value,
            "tail": self.tail,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(document["amount"])


def test_a_payout_without_linear_amounts_raises() -> None:
    templates = [
        _step([Free()], asset="A"),
        Cover(INDEX, Curved(Free()), payout_unit="EUR", asset_id="B"),
    ]
    with pytest.raises(DesignError, match="LinearInAmounts") as raised:
        design_joint(templates, SEPARATE, MSE())
    assert raised.value.asset_id == "B"


def test_the_templates_must_suit_the_sample() -> None:
    with pytest.raises(SampleError, match="one template per asset"):
        design_joint([_step([Free()], asset="A")], SEPARATE, MSE())
    with pytest.raises(SampleError, match="one template per asset"):
        design_joint([], SEPARATE, MSE())
    with pytest.raises(SampleError, match="follow sample.asset_ids"):
        design_joint(_pair()[::-1], SEPARATE, MSE())
    other_unit = Cover(INDEX, StepPayout([1.0], [Free()]), payout_unit="USD")
    with pytest.raises(SampleError, match="USD"):
        design_joint([_step([Free()]), other_unit], SEPARATE, MSE())
    other_index = Cover(
        IndexDefinition("rain", "mm/day", aggregation="annual_maximum"),
        StepPayout([1.0], [Free()]),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match="mm/day"):
        design_joint([_step([Free()]), other_index], SEPARATE, MSE())


def test_a_design_of_nothing_raises() -> None:
    templates = [_step([1.0], asset="A"), _step([2.0], asset="B")]
    with pytest.raises(ValueError, match="no template has free parameters"):
        design_joint(templates, SEPARATE, MSE())


@pytest.mark.parametrize(
    "arguments",
    [
        {"templates": _step([Free()])},
        {"templates": "AB"},
        {"templates": {"A": _step([Free()])}},
        {"templates": [_step([Free()]), "B"]},
        {"sample": "S"},
        {"objective": "mse"},
        {"pricing": "pure"},
        {"budget": 3.0},
        {"limit": 3.0},
        {"options": {}},
        {"scope": "joint"},
        {"scope": Individual()},
    ],
)
def test_arguments_of_the_wrong_type(arguments: dict[str, Any]) -> None:
    given: dict[str, Any] = {
        "templates": _pair(),
        "sample": SEPARATE,
        "objective": MSE(),
        **arguments,
    }
    templates, sample, objective = (
        given.pop("templates"),
        given.pop("sample"),
        given.pop("objective"),
    )
    with pytest.raises(TypeError):
        design_joint(templates, sample, objective, **given)


def test_unbounded_payouts_need_a_budget() -> None:
    with pytest.raises(DesignError, match="need a budget or bounded amounts"):
        design_joint(_pair(), SEPARATE, ExpectedUncovered())
    bounded = [_step([Free(0.0, 15.0)], asset=asset) for asset in ("A", "B")]
    result = design_joint(bounded, SEPARATE, ExpectedUncovered())
    assert (_level(result, "A"), _level(result, "B")) == pytest.approx((15.0, 15.0))


def test_infeasible_designs_raise_for_the_total() -> None:
    # Each level is at least 20, so the premium is at least 10, above the budget.
    templates = [_step([Free(20.0)], asset=asset) for asset in ("A", "B")]
    with pytest.raises(InfeasibleDesignError) as raised:
        design_joint(templates, SEPARATE, MSE(), budget=Budget(amount=5.0))
    assert raised.value.asset_id is None
    assert str(raised.value).startswith("the joint design: ")


def test_an_empty_bound_names_its_asset() -> None:
    templates = [_step([Free()], asset="A"), _step([Free(20.0)], asset="B")]
    with pytest.raises(InfeasibleDesignError, match="levels") as raised:
        design_joint(templates, SEPARATE, MSE(), limit=Limit(10.0))
    assert raised.value.asset_id == "B"


def test_a_solution_that_breaks_the_budget_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _solver.solve_lp

    def inflated(
        programme: _solver.Programme,
        options: Mapping[str, Any] | None = None,
        *,
        keep_basis: bool = False,
    ) -> _solver.Solution:
        solution = real(programme, options, keep_basis=keep_basis)
        assert solution.values is not None
        values = solution.values.copy()
        values[:2] *= 1.01
        return _solver.Solution(
            "optimal",
            values,
            solution.objective,
            solution.message,
            solution.iterations,
            solution.seconds,
        )

    monkeypatch.setattr(_solver, "solve_lp", inflated)
    with pytest.raises(DesignError, match="exceeds the budget") as raised:
        design_joint(_pair(), SEPARATE, ExpectedUncovered(), budget=Budget(amount=7.5))
    assert raised.value.status == "infeasible_solution"
    assert raised.value.asset_id is None


def test_a_missing_solver_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "highspy", None)
    with pytest.raises(SolverUnavailableError, match=r"crc-sdk\[parametric\]"):
        design_joint(_pair(), SEPARATE, MSE())


# -- the record --------------------------------------------------------------


def test_the_scopes_are_documents() -> None:
    assert Joint().to_document() == {"kind": "joint"}
    assert Individual().to_document() == {"kind": "individual"}
    assert Joint() == Joint()


def test_covers_hash_is_of_the_assets_and_their_covers() -> None:
    result = design_joint(_pair(), SEPARATE, MSE())
    covers = result.covers
    assert result.covers_hash == covers_hash(covers)
    assert result.covers_hash == content_hash(
        {asset: cover.definition_hash for asset, cover in covers.items()}
    )
    assert covers_hash(dict(reversed(list(covers.items())))) == result.covers_hash


def test_the_design_id_follows_the_inputs() -> None:
    templates = _pair()
    budget = Budget(amount=7.5)
    first = design_joint(templates, SEPARATE, ExpectedUncovered(), budget=budget)
    assert first.design_id == content_hash(
        {
            "budget": {"amount": 7.5},
            "limit": None,
            "objective": {"kind": "expected_uncovered"},
            "options": DesignOptions().to_document(),
            "pricing": {"kind": "pure"},
            "sample_hash": SEPARATE.content_hash,
            "scope": {"kind": "joint"},
            "semantics_version": DESIGN_SEMANTICS_VERSION,
            "template_hashes": [template.definition_hash for template in templates],
        }
    )
    second = design_joint(templates, SEPARATE, ExpectedUncovered(), budget=budget)
    assert (second.design_id, second.parameters, second.covers) == (
        first.design_id,
        first.parameters,
        first.covers,
    )
    other = design_joint(
        templates, SEPARATE, ExpectedUncovered(), budget=Budget(amount=7.0)
    )
    assert other.design_id != first.design_id


def test_the_record_is_plain_json() -> None:
    result = design_joint(
        _pair(),
        SEPARATE,
        ExpectedUncovered(),
        pricing=CostOfCapitalLoad(0.06, 0.75),
        budget=Budget(ratio=0.5),
        options=DesignOptions(ranges=True),
    )
    record = result.to_record()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["scope"] == {"kind": "joint"}
    assert record["asset_ids"] == ["A", "B"]
    assert record["covers_hash"] == result.covers_hash
    assert record["cover_hashes"] == {
        asset: cover.definition_hash for asset, cover in result.covers.items()
    }
    assert record["template_hashes"] == {
        template.asset_id: template.definition_hash for template in _pair()
    }
    assert record["sample_hash"] == SEPARATE.content_hash == result.sample_hash
    assert set(record["parameters"]) == {"A", "B"}
    assert record["unidentified"] == {"A": [], "B": []}
    assert result.ranges is not None
    assert record["ranges"]["A"]["levels[0]"] == list(result.ranges["A"]["levels[0]"])
    assert record["solve"]["status"] == "optimal"
    assert record["solve"]["options"]["ranging"] is not None
    assert record["solver"] == {"name": "highs", "version": record["solver"]["version"]}
    assert (record["ties"], record["grid"]) == (0, None)
    assert record["premium"] == result.premium
    assert record["budget"] == {"ratio": 0.5}
