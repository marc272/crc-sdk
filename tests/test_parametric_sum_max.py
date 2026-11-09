"""`SumOf` and `MaxOf`: the families, their designs, reductions and closed forms.

The numbers are synthetic and chosen so that the optima have closed forms or
can be checked by brute force. The reductions to the golden fixtures are in
`tests/test_parametric_design.py`. The tests check what the module computes;
nothing here is a finding about a cover.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import pickle
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar, Self

import numpy as np
import pytest

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CompositeIndex,
    CostOfCapitalLoad,
    Cover,
    CoverDefinitionError,
    DesignError,
    DesignOptions,
    DesignResult,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    InfeasibleDesignError,
    Limit,
    LinearConstraints,
    LinearInAmounts,
    LinearPayout,
    MaxOf,
    OutcomeSample,
    PayoutFunction,
    PiecewiseLinearPayout,
    SampleError,
    SampleProvenance,
    SolverUnavailableError,
    StepPayout,
    SumOf,
    UnknownPayoutKindError,
    UnresolvedParameterError,
    UserObjective,
    WeightedCVaR,
    _solver,
    design,
    register_payout,
)
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.payouts import _REGISTRY

RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
TWO = CompositeIndex({"rain": RAIN, "heat": HEAT})

RAIN_VALUES = np.array([10.0, 60.0, 20.0, 80.0, 30.0, 15.0, 70.0, 25.0])
HEAT_VALUES = np.array([30.0, 25.0, 40.0, 20.0, 35.0, 42.0, 45.0, 38.0])
LOSS = np.array([0.0, 10.0, 4.0, 6.0, 3.0, 2.0, 20.0, 6.0])
"""Rain reaches 50 in years 1, 3, 6, heat 38 in 2, 5, 6, 7: year 6 is both."""


def _provenance() -> SampleProvenance:
    return SampleProvenance(
        source_kind="table",
        sources=(),
        semantics_source="caller",
        hazard_dependence={"kind": "data"},
        combine={"kind": "table"},
    )


def _sample(
    rain: Any = RAIN_VALUES, heat: Any = HEAT_VALUES, loss: Any = LOSS
) -> OutcomeSample:
    rain, heat, loss = map(np.asarray, (rain, heat, loss))
    n = rain.shape[0]
    fields: dict[str, Any] = {
        "asset_ids": ("site",),
        "period": np.arange(2000, 2000 + n),
        "probability": np.full(n, 1 / n),
        "index": rain,
        "loss": loss,
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": _provenance(),
        "index_components": {
            "rain": IndexComponent(
                rain, unit="mm/day", tail="upper", hazard=None, own_loss=None
            ),
            "heat": IndexComponent(
                heat, unit="degC", tail="upper", hazard=None, own_loss=None
            ),
        },
    }
    untyped: Any = OutcomeSample
    built: OutcomeSample = untyped(**fields)
    return built


def _cover(payout: PayoutFunction, *, index: CompositeIndex = TWO) -> Cover:
    return Cover(index, payout, payout_unit="EUR", label="two")


def _index(n: int = 8) -> Any:
    return np.column_stack([RAIN_VALUES[:n], HEAT_VALUES[:n]])


def _sum_of(rain: Any = None, heat: Any = None) -> SumOf:
    return SumOf(
        {
            "rain": BinaryPayout(50.0, Free()) if rain is None else rain,
            "heat": BinaryPayout(38.0, Free()) if heat is None else heat,
        }
    )


def _max_of(*, rain: float = 4.0, heat: float = 8.0, scale: Any = None) -> MaxOf:
    return MaxOf(
        {"rain": BinaryPayout(50.0, rain), "heat": BinaryPayout(38.0, heat)},
        scale=Free() if scale is None else scale,
    )


class Doubled:
    """A one-index user family whose basis rows sum to 2, so a limit cannot hold."""

    kind: ClassVar[str] = "test_doubled"
    tail: ClassVar[Tail] = "upper"
    index_names: ClassVar[tuple[str, ...]] = ("index",)

    def __init__(self, amount: Any) -> None:
        self.amount = amount

    def parameters(self) -> dict[str, float | Free]:
        return {"amount": self.amount}

    def with_values(self, values: Mapping[str, float]) -> Self:
        return type(self)(values.get("amount", self.amount))

    def amount_names(self) -> tuple[str, ...]:
        return ("amount",)

    def basis(self, index: Any) -> Any:
        return (np.asarray(index) >= 50.0).astype(np.float64).reshape(-1, 1) * 2.0

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        lower = self.amount.lower if isinstance(self.amount, Free) else self.amount
        upper = (
            (math.inf if self.amount.upper is None else self.amount.upper)
            if isinstance(self.amount, Free)
            else self.amount
        )
        return LinearConstraints(
            a_ub=np.zeros((0, 1)),
            b_ub=np.zeros(0),
            lower=np.array([lower]),
            upper=np.array([upper if limit is None else min(upper, limit)]),
        )

    def evaluate(self, index: Any) -> Any:
        amount = self.amount
        return self.basis(index)[:, 0] * amount

    def to_document(self) -> dict[str, Any]:
        amount = self.amount
        return {
            "amount": amount.to_document() if isinstance(amount, Free) else amount,
            "kind": self.kind,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        amount = document["amount"]
        return cls(
            Free.from_document(amount) if isinstance(amount, Mapping) else amount
        )


class Capped:
    """A one-index user family that isn't linear in its parameter."""

    kind: ClassVar[str] = "test_capped"
    tail: ClassVar[Tail] = "upper"
    index_names: ClassVar[tuple[str, ...]] = ("index",)

    def __init__(self, slope: Any) -> None:
        self.slope = slope

    def parameters(self) -> dict[str, float | Free]:
        return {"slope": self.slope}

    def with_values(self, values: Mapping[str, float]) -> Self:
        return type(self)(values.get("slope", self.slope))

    def evaluate(self, index: Any) -> Any:
        slope = self.slope
        return np.minimum(np.maximum(np.asarray(index) - 50.0, 0.0) * slope, 10.0)

    def to_document(self) -> dict[str, Any]:
        slope = self.slope
        return {
            "kind": self.kind,
            "slope": slope.to_document() if isinstance(slope, Free) else slope,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        slope = document["slope"]
        return cls(Free.from_document(slope) if isinstance(slope, Mapping) else slope)


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    families: tuple[type[Any], ...] = (Doubled, Capped)
    for family in families:
        register_payout(family)
    yield
    for family in families:
        del _REGISTRY[family.kind]


# -- the families ------------------------------------------------------------


def test_sum_of_pays_the_sum_of_its_components() -> None:
    payout = _sum_of(BinaryPayout(50.0, 4.0), BinaryPayout(38.0, 8.0))
    paid = payout.evaluate(_index())
    assert paid.tolist() == [0.0, 4.0, 8.0, 4.0, 0.0, 8.0, 12.0, 8.0]
    assert payout.index_names == ("rain", "heat")
    assert payout.tail == ("upper", "upper")


def test_max_of_pays_the_scale_times_the_largest() -> None:
    payout = MaxOf(
        {"rain": BinaryPayout(50.0, 4.0), "heat": BinaryPayout(38.0, 8.0)}, scale=0.5
    )
    assert payout.evaluate(_index()).tolist() == [
        0.0,
        2.0,
        4.0,
        2.0,
        0.0,
        4.0,
        4.0,
        4.0,
    ]


def test_a_cover_takes_the_families_and_writes_their_terms() -> None:
    cover = _cover(_sum_of())
    assert cover.free_parameters() == ("rain.amount", "heat.amount")
    assert cover.to_document()["payout"]["kind"] == "sum_of"
    assert [entry["name"] for entry in cover.to_document()["payout"]["components"]] == [
        "rain",
        "heat",
    ]


@pytest.mark.parametrize(
    "payout",
    [
        pytest.param(_sum_of(StepPayout([50.0, 70.0], [Free(), Free()])), id="sum_of"),
        pytest.param(_max_of(), id="max_of-free-scale"),
        pytest.param(_max_of(scale=2.5), id="max_of-fixed-scale"),
        pytest.param(
            MaxOf({"rain": Capped(0.5), "heat": BinaryPayout(38.0, 8.0)}),
            id="max_of-user-component",
        ),
    ],
)
def test_the_families_round_trip_through_a_cover(payout: Any) -> None:
    cover = _cover(payout)
    read = Cover.from_json(cover.to_json())
    assert read.definition_hash == cover.definition_hash
    assert read.payout == payout
    assert hash(read.payout) == hash(payout)
    assert pickle.loads(pickle.dumps(payout)) == payout
    assert pickle.loads(pickle.dumps(cover)).definition_hash == cover.definition_hash


def test_the_order_of_the_components_is_part_of_the_terms() -> None:
    other = CompositeIndex({"heat": HEAT, "rain": RAIN})
    first = _cover(_sum_of())
    second = _cover(
        SumOf({"heat": BinaryPayout(38.0, Free()), "rain": BinaryPayout(50.0, Free())}),
        index=other,
    )
    assert first.definition_hash != second.definition_hash
    assert _sum_of() != second.payout


def test_an_unknown_component_kind_raises_when_read() -> None:
    document = _sum_of().to_document()
    document["components"][0]["payout"] = {"kind": "nobody"}
    with pytest.raises(UnknownPayoutKindError, match="nobody"):
        SumOf.from_document(document)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(lambda: SumOf([]), "maps each component", id="not-a-mapping"),  # type: ignore[arg-type]
        pytest.param(
            lambda: SumOf({"rain": BinaryPayout(1.0, 1.0)}), "two components", id="one"
        ),
        pytest.param(
            lambda: SumOf({"a.b": BinaryPayout(1.0, 1.0), "c": BinaryPayout(1.0, 1.0)}),
            "component name",
            id="bad-name",
        ),
        pytest.param(
            lambda: SumOf({"a": 3.0, "c": BinaryPayout(1.0, 1.0)}),  # type: ignore[dict-item]
            "PayoutFunction",
            id="not-a-payout",
        ),
        pytest.param(
            lambda: SumOf({"a": _sum_of(), "c": BinaryPayout(1.0, 1.0)}),
            "one index",
            id="nested",
        ),
        pytest.param(lambda: _max_of(scale=-1.0), "non-negative", id="negative-scale"),
        pytest.param(
            lambda: _max_of(scale=Free(-1.0)), "non-negative", id="negative-free-scale"
        ),
        pytest.param(lambda: _max_of(scale="a"), "scale must be", id="bad-scale"),
        pytest.param(
            lambda: MaxOf(
                {"rain": BinaryPayout(50.0, Free()), "heat": BinaryPayout(38.0, 8.0)},
                scale=Free(),
            ),
            "not identified",
            id="free-scale-beside-free-amount",
        ),
    ],
)
def test_invalid_families_raise(build: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        build()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: {**d, "kind": "max_of"}, "expected a payout"),
        (lambda d: {**d, "extra": 1}, "has the keys"),
        (lambda d: {**d, "components": {}}, "is a list"),
        (lambda d: {**d, "components": [{"name": "a"}]}, "has the keys"),
        (
            lambda d: {**d, "components": [d["components"][0], d["components"][0]]},
            "repeats",
        ),
        (
            lambda d: {
                **d,
                "components": [{"name": "a", "payout": 1}, d["components"][1]],
            },
            "must be an object",
        ),
    ],
)
def test_an_invalid_document_raises(change: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        SumOf.from_document(change(_sum_of().to_document()))


def test_a_document_must_be_an_object() -> None:
    with pytest.raises(CoverDefinitionError, match="must be an object"):
        SumOf.from_document(1)  # type: ignore[arg-type]


def test_with_values_resolves_by_flat_name() -> None:
    payout = _sum_of().with_values({"rain.amount": 2.0})
    assert payout.parameters()["rain.amount"] == 2.0
    assert isinstance(payout.parameters()["heat.amount"], Free)
    resolved = payout.with_values({"heat.amount": 3.0})
    assert resolved.evaluate(_index())[6] == 5.0


def test_max_of_with_values_resolves_the_scale_and_components() -> None:
    payout = MaxOf(
        {
            "rain": BinaryPayout(Free(40.0, 60.0, grid=[40.0, 60.0]), 4.0),
            "heat": BinaryPayout(38.0, 8.0),
        },
        scale=Free(),
    )
    resolved = payout.with_values({"rain.trigger": 60.0, "scale": 0.5})
    assert resolved.scale == 0.5
    assert resolved.components["rain"].parameters()["trigger"] == 60.0


@pytest.mark.parametrize("family", [_sum_of, _max_of])
def test_a_template_does_not_pay(family: Any) -> None:
    with pytest.raises(UnresolvedParameterError, match="free parameters"):
        family().evaluate(_index())


@pytest.mark.parametrize("family", [_sum_of, _max_of])
@pytest.mark.parametrize(
    "index", [np.zeros(3), np.zeros((3, 3)), np.array([[np.nan, 1.0]])]
)
def test_an_index_of_the_wrong_shape_raises(family: Any, index: Any) -> None:
    payout = family()
    with pytest.raises(SampleError):
        payout.basis(index)


def test_a_component_of_the_wrong_shape_raises() -> None:
    class Wrong(Capped):
        kind: ClassVar[str] = "test_wrong"

        def evaluate(self, index: Any) -> Any:
            return np.zeros(3)

    payout = SumOf({"rain": Wrong(0.5), "heat": BinaryPayout(38.0, 8.0)})
    with pytest.raises(CoverDefinitionError, match="returned shape"):
        payout.evaluate(_index())


def test_sum_of_is_linear_in_every_amount_of_its_components() -> None:
    payout = _sum_of(StepPayout([50.0, 70.0], [Free(), 3.0]))
    assert isinstance(payout, LinearInAmounts)
    assert payout.amount_names() == ("rain.levels[0]", "rain.levels[1]", "heat.amount")
    basis = payout.basis(_index())
    assert basis.shape == (8, 3)
    assert basis[6].tolist() == [0.0, 1.0, 1.0]


def test_sum_of_with_a_component_not_linear_has_no_amounts() -> None:
    payout = SumOf({"rain": Capped(Free(0, 1)), "heat": BinaryPayout(38.0, Free())})
    assert payout.amount_names() == ()
    assert payout.basis(_index()).shape == (8, 0)
    assert payout.amount_constraints(1.0).n_amounts == 0


def test_max_of_is_linear_only_in_a_free_scale_over_fixed_amounts() -> None:
    assert _max_of().amount_names() == ("scale",)
    assert _max_of().basis(_index())[:, 0].tolist() == [
        0.0,
        4.0,
        8.0,
        4.0,
        0.0,
        8.0,
        8.0,
        8.0,
    ]
    assert _max_of(scale=2.0).amount_names() == ()
    assert _max_of(scale=2.0).basis(_index()).shape == (8, 0)
    assert _max_of(scale=2.0).amount_constraints().n_amounts == 0
    free_amount = MaxOf(
        {"rain": BinaryPayout(50.0, Free()), "heat": BinaryPayout(38.0, 8.0)}
    )
    assert free_amount.amount_names() == ()
    with_user = MaxOf(
        {"rain": Capped(0.5), "heat": BinaryPayout(38.0, 8.0)}, scale=Free()
    )
    assert with_user.amount_names() == ()


def test_max_of_basis_needs_its_thresholds() -> None:
    payout = MaxOf(
        {
            "rain": BinaryPayout(Free(40.0, 60.0, grid=2), 4.0),
            "heat": BinaryPayout(38.0, 8.0),
        },
        scale=Free(),
    )
    with pytest.raises(UnresolvedParameterError, match="rain.trigger"):
        payout.basis(_index())


# -- the constraints ---------------------------------------------------------


def _monotone(n: int) -> StepPayout:
    return StepPayout(np.linspace(40.0, 80.0, n), [Free()] * n)


def _free_levels(n: int) -> StepPayout:
    return StepPayout(np.linspace(40.0, 80.0, n), [Free()] * n, monotone=False)


def _limit_rows(constraints: LinearConstraints) -> Any:
    """The rows of `a_ub` with no negative coefficient: those of the limit."""
    keep = (constraints.a_ub >= 0).all(axis=1)
    return constraints.a_ub[keep], constraints.b_ub[keep]


def test_monotone_components_give_one_limit_row_on_the_last_amounts() -> None:
    payout = _sum_of(_monotone(3), _monotone(2))
    rows, bounds = _limit_rows(payout.amount_constraints(8.0))
    assert rows.tolist() == [[0.0, 0.0, 1.0, 0.0, 1.0]]
    assert bounds.tolist() == [8.0]


def test_components_not_monotone_give_one_row_per_combination() -> None:
    payout = _sum_of(_free_levels(3), _free_levels(2))
    rows, bounds = _limit_rows(payout.amount_constraints(8.0))
    assert rows.shape == (6, 5)
    assert bounds.tolist() == [8.0] * 6
    assert (rows.sum(axis=1) == 2.0).all()
    assert {tuple(np.flatnonzero(row).tolist()) for row in rows} == {
        (i, j) for i, j in itertools.product(range(3), (3, 4))
    }


def test_a_monotone_component_offers_only_its_last_amount() -> None:
    payout = _sum_of(_monotone(3), _free_levels(2))
    rows, _ = _limit_rows(payout.amount_constraints(8.0))
    assert rows.shape[0] == 2
    assert (rows[:, 2] == 1.0).all()


def test_without_a_limit_the_constraints_are_the_components_side_by_side() -> None:
    constraints = _sum_of(_monotone(2), _monotone(3)).amount_constraints()
    assert constraints.n_amounts == 5
    assert constraints.a_ub.tolist() == [
        [1.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, -1.0],
    ]
    assert np.isinf(constraints.upper).all()


def test_a_limit_caps_every_amount_too() -> None:
    constraints = _sum_of(_monotone(2), _monotone(2)).amount_constraints(8.0)
    assert constraints.upper.tolist() == [8.0] * 4


def test_limit_on_a_free_scale_divides_by_the_largest_amount() -> None:
    constraints = _max_of(rain=4.0, heat=8.0).amount_constraints(2.0)
    assert constraints.upper.tolist() == [0.25]
    free = Free(0.0, 0.1)
    assert _max_of(scale=free).amount_constraints(2.0).upper.tolist() == [0.1]


def test_limit_on_a_free_scale_over_zero_amounts_is_no_bound() -> None:
    constraints = _max_of(rain=0.0, heat=0.0).amount_constraints(2.0)
    assert constraints.upper.tolist() == [math.inf]


@pytest.mark.parametrize("family", [_sum_of, _max_of])
@pytest.mark.parametrize("limit", [-1.0, math.nan, True])
def test_an_invalid_limit_raises(family: Any, limit: Any) -> None:
    with pytest.raises(CoverDefinitionError, match="limit must be"):
        family().amount_constraints(limit)


# -- SumOf's designs ---------------------------------------------------------


def test_a_sum_takes_one_qp_on_the_block_basis() -> None:
    result = design(_cover(_sum_of()), _sample(), MSE())
    assert (result.route, result.amounts) == ("qp", "qp")
    assert result.unique is True
    basis = np.column_stack(
        [(RAIN_VALUES >= 50.0).astype(float), (HEAT_VALUES >= 38.0).astype(float)]
    )
    expected = np.linalg.solve(basis.T @ basis, basis.T @ LOSS)
    assert result.parameters["rain.amount"] == pytest.approx(expected[0], abs=1e-6)
    assert result.parameters["heat.amount"] == pytest.approx(expected[1], abs=1e-6)


def test_a_sum_takes_one_lp_under_the_expected_uncovered() -> None:
    result = design(
        _cover(_sum_of()), _sample(), ExpectedUncovered(), budget=Budget(amount=3.0)
    )
    assert (result.route, result.amounts) == ("lp", "lp")
    assert result.premium <= 3.0 + 1e-6


def test_a_sum_reads_each_components_own_side() -> None:
    dry = IndexDefinition("heat", "degC", tail="lower", aggregation="annual_minimum")
    sample = _sample()
    components = sample.index_components
    assert components is not None
    lower = dataclasses.replace(components["heat"], tail="lower")
    mixed = dataclasses.replace(
        sample, index_components={"rain": components["rain"], "heat": lower}
    )
    payout = SumOf(
        {
            "rain": BinaryPayout(50.0, Free()),
            "heat": BinaryPayout(21.0, Free(), tail="lower"),
        }
    )
    index = CompositeIndex({"rain": RAIN, "heat": dry})
    assert payout.tail == ("upper", "lower")
    result = design(_cover(payout, index=index), mixed, MSE())
    basis = payout.basis(_index())
    assert basis[3].tolist() == [1.0, 1.0]
    assert basis[0].tolist() == [0.0, 0.0]
    assert set(result.parameters) == {"rain.amount", "heat.amount"}


def test_linearly_dependent_components_are_reported() -> None:
    sample = _sample(heat=RAIN_VALUES)
    payout = SumOf(
        {"rain": BinaryPayout(50.0, Free()), "heat": BinaryPayout(50.0, Free())}
    )
    assert design(_cover(payout), sample, MSE()).unique is False


def test_free_thresholds_of_a_sum_take_the_grid_route() -> None:
    template = _cover(
        _sum_of(BinaryPayout(Free(40.0, 60.0, grid=[40.0, 50.0, 60.0]), Free()))
    )
    result = design(template, _sample(), MSE())
    assert (result.route, result.amounts) == ("grid", "qp")
    assert result.grid == {"candidates": 3, "optimal": 3}
    assert result.parameters["rain.trigger"] in (40.0, 50.0, 60.0)


def test_a_sum_with_a_component_not_linear_takes_the_numpy_grid() -> None:
    template = _cover(
        SumOf(
            {
                "rain": Capped(Free(0.1, 0.5, grid=[0.1, 0.5])),
                "heat": BinaryPayout(38.0, Free(0.0, 8.0, grid=[0.0, 8.0])),
            }
        )
    )
    result = design(template, _sample(), MSE())
    assert result.route == "grid" and result.solver is None
    assert result.grid == {"candidates": 4, "feasible": 4}


def test_the_numpy_grid_checks_the_limit_on_a_component_not_linear() -> None:
    template = _cover(
        SumOf(
            {
                "rain": Capped(Free(0.1, 0.5, grid=[0.1, 0.5])),
                "heat": BinaryPayout(38.0, Free(0.0, 8.0, grid=[0.0, 8.0])),
            }
        )
    )
    result = design(template, _sample(), MSE(), limit=Limit(11.0))
    assert result.grid is not None and result.grid.get("over_limit", 0) >= 1
    largest = result.cover.payout_values(_index()).max()
    assert largest <= 11.0 + 1e-9


# -- the limit on a SumOf ----------------------------------------------------


@pytest.mark.parametrize(
    ("objective", "budget"),
    [
        pytest.param(MSE(), None, id="mse"),
        pytest.param(ExpectedUncovered(), None, id="eu-no-budget"),
        pytest.param(WeightedCVaR(0.5, 0.75), Budget(amount=50.0), id="wcvar"),
    ],
)
def test_the_limit_bounds_the_years_total(objective: Any, budget: Any) -> None:
    result = design(
        _cover(_sum_of()),
        _sample(),
        objective,
        budget=budget,
        limit=Limit(8.0),
    )
    payout = result.cover.payout_values(_index())
    assert payout.max() == pytest.approx(8.0, abs=1e-6)
    total = result.parameters["rain.amount"] + result.parameters["heat.amount"]
    assert total == pytest.approx(8.0, abs=1e-6)


def test_without_the_limit_the_years_total_is_above_it() -> None:
    result = design(_cover(_sum_of()), _sample(), MSE())
    assert result.cover.payout_values(_index()).max() > 8.0


def test_the_limit_on_steps_that_are_not_monotone_bounds_every_year() -> None:
    levels = StepPayout([50.0, 70.0], [Free(), Free()], monotone=False)
    other = StepPayout([38.0, 42.0], [Free(), Free()], monotone=False)
    result = design(_cover(_sum_of(levels, other)), _sample(), MSE(), limit=Limit(9.0))
    values = result.cover.payout.parameters()
    worst = max(
        float(values[f"rain.levels[{i}]"]) + float(values[f"heat.levels[{j}]"])  # type: ignore[arg-type]
        for i in range(2)
        for j in range(2)
    )
    assert worst <= 9.0 + 1e-6


def test_the_lp_with_a_limit_matches_a_brute_force_search() -> None:
    sample = _sample()
    result = design(_cover(_sum_of()), sample, ExpectedUncovered(), limit=Limit(8.0))
    rain, heat = RAIN_VALUES >= 50.0, HEAT_VALUES >= 38.0
    best = math.inf
    for a in np.linspace(0.0, 8.0, 321):
        for b in np.linspace(0.0, 8.0 - a, 161):
            uncovered = np.maximum(LOSS - a * rain - b * heat, 0.0)
            best = min(best, float(uncovered.mean()))
    assert result.objective_value == pytest.approx(best, abs=2e-3)
    assert result.objective_value <= best + 1e-9


@pytest.mark.parametrize(("fill", "total"), [("lowest", 5.0), ("highest", 8.0)])
def test_the_fill_rule_under_a_limit_row(fill: Any, total: float) -> None:
    template = _cover(
        _sum_of(
            StepPayout([50.0, 1000.0], [Free(5.0, 5.0), Free()]),
            BinaryPayout(1000.0, Free(0.0, 8.0)),
        )
    )
    result = design(
        template, _sample(), MSE(), limit=Limit(8.0), options=DesignOptions(fill=fill)
    )
    assert set(result.unidentified) == {"rain.levels[1]", "heat.amount"}
    values = result.parameters
    if fill == "lowest":
        assert values["rain.levels[1]"] == pytest.approx(5.0)
        assert values["heat.amount"] == pytest.approx(0.0)
    else:
        assert values["rain.levels[1]"] + values["heat.amount"] == pytest.approx(
            total, abs=1e-6
        )


def test_a_fixed_amount_above_the_limit_is_infeasible() -> None:
    template = _cover(_sum_of(BinaryPayout(50.0, 9.0)))
    with pytest.raises(InfeasibleDesignError, match="satisfies its bounds"):
        design(template, _sample(), MSE(), limit=Limit(8.0))


def test_fixed_amounts_that_sum_above_the_limit_are_infeasible() -> None:
    template = _cover(
        SumOf(
            {
                "rain": BinaryPayout(50.0, 5.0),
                "heat": StepPayout([38.0, 42.0], [Free(5.0), Free(5.0)]),
            }
        )
    )
    with pytest.raises(InfeasibleDesignError):
        design(template, _sample(), MSE(), limit=Limit(8.0))


def test_basis_rows_above_one_refuse_a_limit() -> None:
    template = _cover(_sum_of(Doubled(Free(0.0, 20.0))))
    with pytest.raises(DesignError, match="sum above 1"):
        design(template, _sample(), MSE(), limit=Limit(8.0))
    assert design(template, _sample(), MSE()).route == "qp"


def test_basis_rows_above_one_refuse_a_limit_on_the_numpy_grid() -> None:
    template = _cover(
        _sum_of(Doubled(Free(0.0, 20.0, grid=[0.0, 20.0])), BinaryPayout(38.0, 1.0))
    )
    objective = UserObjective(_absolute, name="abs")
    with pytest.raises(DesignError, match="sum above 1"):
        design(template, _sample(), objective, limit=Limit(8.0))


def _absolute(loss: Any, payout: Any, probability: Any) -> float:
    return float((probability * np.abs(loss - payout)).sum())


def test_the_numpy_grid_reads_the_limit_rows_of_a_sum() -> None:
    template = _cover(
        _sum_of(
            BinaryPayout(50.0, Free(0.0, 12.0, grid=[0.0, 6.0, 12.0])),
            BinaryPayout(38.0, Free(0.0, 12.0, grid=[0.0, 6.0, 12.0])),
        )
    )
    result = design(
        template, _sample(), UserObjective(_absolute, name="abs"), limit=Limit(12.0)
    )
    # (6, 12), (12, 6) and (12, 12) break the total of 12, though no amount does.
    assert result.grid == {"candidates": 9, "feasible": 6, "over_limit": 3}
    assert result.cover.payout_values(_index()).max() <= 12.0


def test_a_user_objective_on_a_free_scale_checks_the_limit_by_amounts() -> None:
    template = _cover(_max_of(scale=Free(0.0, 2.0, grid=[0.0, 0.5, 1.0, 2.0])))
    result = design(
        template, _sample(), UserObjective(_absolute, name="abs"), limit=Limit(4.0)
    )
    # scale <= 4 / 8, whatever the sample's years pay.
    assert result.grid == {"candidates": 4, "feasible": 2, "over_limit": 2}
    assert result.parameters["scale"] <= 0.5


# -- MaxOf's designs ---------------------------------------------------------


def test_max_of_one_scale_mse_no_budget_is_the_closed_form() -> None:
    generator = np.random.default_rng(20261109)
    n = 200
    rain, heat = generator.gamma(2.0, 30.0, n), generator.gamma(3.0, 12.0, n)
    loss = generator.exponential(5.0, n)
    sample = _sample(rain, heat, loss)
    template = _cover(
        MaxOf(
            {
                "rain": LinearPayout(40.0, 120.0, 6.0),
                "heat": StepPayout([25.0, 45.0], [2.0, 9.0]),
            },
            scale=Free(),
        )
    )
    result = design(template, sample, MSE())
    assert (result.route, result.amounts) == ("qp", "qp")
    assert result.unique is True
    index = np.column_stack([rain, heat])
    payouts = template.payout
    assert isinstance(payouts, MaxOf)
    g = payouts.basis(index)[:, 0]
    p = sample.probability
    closed_form = float((p * loss * g).sum() / (p * g * g).sum())
    assert result.parameters["scale"] == pytest.approx(closed_form, rel=1e-9)


def test_max_of_counts_a_tie_in_the_largest_only_once() -> None:
    template = _cover(_max_of(rain=8.0, heat=8.0))
    result = design(template, _sample(), MSE())
    g = 8.0 * ((RAIN_VALUES >= 50.0) | (HEAT_VALUES >= 38.0))
    assert result.parameters["scale"] == pytest.approx(
        float((LOSS * g).sum() / (g * g).sum()), rel=1e-9
    )


def test_max_of_scale_stays_within_the_limit() -> None:
    result = design(_cover(_max_of()), _sample(), MSE(), limit=Limit(4.0))
    assert result.parameters["scale"] == pytest.approx(0.5, abs=1e-9)
    assert result.cover.payout_values(_index()).max() == pytest.approx(4.0, abs=1e-6)


def test_max_of_scale_under_a_budget_and_loaded_pricing() -> None:
    result = design(
        _cover(_max_of()),
        _sample(),
        ExpectedUncovered(),
        pricing=CostOfCapitalLoad(0.2, 0.9),
        budget=Budget(amount=2.0),
    )
    assert result.route == "lp"
    assert result.premium == pytest.approx(2.0, abs=1e-6)


def test_max_of_scale_with_thresholds_on_a_grid() -> None:
    template = _cover(
        MaxOf(
            {
                "rain": BinaryPayout(Free(40.0, 60.0, grid=[40.0, 50.0, 60.0]), 4.0),
                "heat": BinaryPayout(38.0, 8.0),
            },
            scale=Free(),
        )
    )
    result = design(template, _sample(), MSE())
    assert (result.route, result.amounts) == ("grid", "qp")
    assert set(result.parameters) == {"rain.trigger", "scale"}


def test_max_of_reports_a_scale_no_year_identifies() -> None:
    template = _cover(
        MaxOf(
            {"rain": BinaryPayout(1000.0, 4.0), "heat": BinaryPayout(1000.0, 8.0)},
            scale=Free(0.0, 3.0),
        )
    )
    assert design(template, _sample(), MSE()).unidentified == ("scale",)


def test_a_max_of_with_a_fixed_scale_takes_the_numpy_grid_without_a_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> None:
        raise SolverUnavailableError("no solver")

    monkeypatch.setattr(_solver, "require", missing)
    template = _cover(
        MaxOf(
            {
                "rain": BinaryPayout(Free(40.0, 75.0, grid=[40.0, 75.0]), 4.0),
                "heat": BinaryPayout(38.0, 8.0),
            }
        )
    )
    result = design(template, _sample(), MSE())
    assert result.route == "grid" and result.solver is None
    assert result.ties == 0 and result.unique is None


def test_a_free_scale_takes_the_solver_route(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise SolverUnavailableError("no solver")

    monkeypatch.setattr(_solver, "require", missing)
    with pytest.raises(SolverUnavailableError):
        design(_cover(_max_of()), _sample(), MSE())


def test_free_component_amounts_search_a_grid_because_the_objective_is_not_convex() -> (
    None
):
    # One year, loss 1, two unit payouts: (1, 0) and (0, 1) leave nothing
    # uncovered and their midpoint leaves 0.5.
    sample = _sample([60.0], [45.0], [1.0])
    template = _cover(
        MaxOf(
            {
                "rain": BinaryPayout(50.0, Free(0.0, 1.0, grid=[0.0, 0.5, 1.0])),
                "heat": BinaryPayout(38.0, Free(0.0, 1.0, grid=[0.0, 0.5, 1.0])),
            }
        )
    )
    result = design(template, sample, ExpectedUncovered())
    assert result.route == "grid" and result.solver is None
    assert result.objective_value == 0.0
    assert max(result.parameters.values()) == 1.0
    assert result.profile is not None
    profile = {
        (row["rain.amount"], row["heat.amount"]): row["objective"]
        for row in result.profile.to_pylist()
    }
    assert profile[(0.5, 0.5)] == 0.5
    assert profile[(1.0, 0.0)] == 0.0 and profile[(0.0, 1.0)] == 0.0


def test_a_free_scale_without_a_grid_needs_one_for_a_user_objective() -> None:
    with pytest.raises(DesignError, match="needs a grid"):
        design(_cover(_max_of()), _sample(), UserObjective(_absolute, name="abs"))


def test_a_max_of_design_records_the_cover() -> None:
    result: DesignResult = design(_cover(_max_of()), _sample(), MSE())
    assert result.cover.to_document()["payout"]["kind"] == "max_of"
    assert result.cover.free_parameters() == ()


def test_max_of_with_a_piecewise_component_is_a_column_of_the_largest() -> None:
    payout = MaxOf(
        {
            "rain": PiecewiseLinearPayout([40.0, 80.0], [0.0, 10.0]),
            "heat": BinaryPayout(38.0, 2.0),
        },
        scale=Free(),
    )
    column = payout.basis(_index())[:, 0]
    assert column[3] == pytest.approx(10.0)
    assert column[5] == pytest.approx(2.0)
