"""`StepGrid`: the family, its designs, its special cases and its breakdown.

The numbers are synthetic and chosen so that the optima have closed forms or
can be checked by brute force. The reduction to the golden fixtures is in
`tests/test_parametric_design.py`. The tests check what the module computes;
nothing here is a finding about a cover.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CompositeIndex,
    CostOfCapitalLoad,
    Cover,
    CoverDefinitionError,
    CoverPackage,
    DesignError,
    DesignOptions,
    DesignResult,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    InfeasibleDesignError,
    Limit,
    LinearInAmounts,
    MaxOf,
    OutcomeSample,
    PayoutFunction,
    PurePremium,
    SampleError,
    SampleProvenance,
    StepGrid,
    StepPayout,
    SumOf,
    UnresolvedParameterError,
    UserObjective,
    WeightedCVaR,
    compare,
    design,
    evaluate,
    holdout,
    payout_breakdown,
    setup_text,
    split_budget,
    write_comparison,
)
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.payouts import _REGISTRY

Tails = tuple[Tail, Tail]

TAILS: list[Tails] = [
    ("upper", "upper"),
    ("upper", "lower"),
    ("lower", "upper"),
    ("lower", "lower"),
]
RAIN_BINS = (40.0, 70.0)
HEAT_BINS = (30.0, 80.0)


def _side(values: Any, tail: Tail) -> Any:
    """Values on the side of `tail`: the lower tail mirrors them, severity intact."""
    array = np.asarray(values, dtype=np.float64)
    return array if tail == "upper" else 100.0 - array


def _composite(tails: Tails = ("upper", "upper")) -> CompositeIndex:
    return CompositeIndex(
        {
            "rain": IndexDefinition(
                "rain", "mm/day", tail=tails[0], aggregation="annual_maximum"
            ),
            "heat": IndexDefinition(
                "heat", "degC", tail=tails[1], aggregation="annual_maximum"
            ),
        }
    )


def _cover(
    payout: PayoutFunction,
    tails: Tails = ("upper", "upper"),
    *,
    label: str = "grid",
) -> Cover:
    return Cover(_composite(tails), payout, payout_unit="EUR", label=label)


def _sample(
    rain: Any,
    heat: Any,
    loss: Any,
    *,
    tails: Tails = ("upper", "upper"),
    probability: Any = None,
    own_loss: tuple[Any, Any] | None = None,
) -> OutcomeSample:
    """A sample of `rain` and `heat`, put on the sides of `tails`."""
    first, second = _side(rain, tails[0]), _side(heat, tails[1])
    n = first.shape[0]
    probabilities = (
        np.full(n, 1 / n) if probability is None else np.asarray(probability, float)
    )
    rain_loss, heat_loss = (None, None) if own_loss is None else own_loss
    fields: dict[str, Any] = {
        "asset_ids": ("site",),
        "period": np.arange(2000, 2000 + n),
        "probability": probabilities,
        "index": first,
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
                first, unit="mm/day", tail=tails[0], hazard=None, own_loss=rain_loss
            ),
            "heat": IndexComponent(
                second, unit="degC", tail=tails[1], hazard=None, own_loss=heat_loss
            ),
        },
    }
    untyped: Any = OutcomeSample
    built: OutcomeSample = untyped(**fields)
    return built


def _free_table(n_rows: int, n_columns: int, upper: float | None = None) -> list[Any]:
    return [
        [0.0 if (i, j) == (0, 0) else Free(0.0, upper) for j in range(n_columns)]
        for i in range(n_rows)
    ]


def _grid(
    tails: Tails = ("upper", "upper"),
    *,
    monotone: bool = True,
    upper: float | None = None,
) -> StepGrid:
    """A 3 x 3 table on `RAIN_BINS` and `HEAT_BINS`, every level free."""
    return StepGrid(
        {"rain": _side(RAIN_BINS, tails[0]), "heat": _side(HEAT_BINS, tails[1])},
        _free_table(3, 3, upper),
        tail=tails,
        monotone=monotone,
    )


def _bin(value: float, thresholds: Sequence[float]) -> int:
    """The bin of one value, worked out by hand: the thresholds it has passed."""
    return sum(1 for threshold in thresholds if value >= threshold)


def _draws(n: int = 200, seed: int = 20261109) -> tuple[Any, Any, Any, Any]:
    """Rain, heat, loss and probability, with no year in the cell (2, 2)."""
    rng = np.random.default_rng(seed)
    rain = rng.uniform(0.0, 100.0, n).round(1)
    heat = rng.uniform(0.0, 100.0, n).round(1)
    loss = rng.gamma(2.0, 3.0, n).round(3)
    weight = rng.uniform(1.0, 3.0, n)
    keep = ~((rain >= RAIN_BINS[1]) & (heat >= HEAT_BINS[1]))
    return rain[keep], heat[keep], loss[keep], weight[keep] / weight[keep].sum()


def _table(result: DesignResult) -> Any:
    return np.asarray(cast(StepGrid, result.cover.payout).levels, dtype=np.float64)


def _column(table: Any, name: str) -> list[Any]:
    values: list[Any] = table.column(name).to_pylist()
    return values


def _row(table: Any, **match: Any) -> dict[str, Any]:
    rows = [
        row
        for row in table.to_pylist()
        if all(row[key] == value for key, value in match.items())
    ]
    assert len(rows) == 1, rows
    return cast(dict[str, Any], rows[0])


# -- the family --------------------------------------------------------------


def test_a_year_pays_the_level_of_its_cell() -> None:
    grid = StepGrid(
        {"rain": [40.0, 70.0], "heat": [30.0]},
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
    )
    index = np.array(
        [
            [10.0, 10.0],  # below both first thresholds
            [10.0, 30.0],  # at the first heat threshold: the more severe bin
            [40.0, 29.9],  # at the first rain threshold
            [69.9, 99.0],
            [70.0, 0.0],  # at the last rain threshold: the last bin
            [1e6, 1e6],  # the last bin runs from the last threshold up
        ]
    )
    assert grid.evaluate(index).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_each_axis_works_on_its_own_severity() -> None:
    grid = StepGrid(
        {"rain": [40.0, 70.0], "drought": [30.0, 10.0]},
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        tail=("upper", "lower"),
    )
    index = np.array(
        [
            [10.0, 31.0],
            [10.0, 30.0],  # at a lower-tail threshold: the more severe bin
            [10.0, 10.0],
            [50.0, 20.0],
            [90.0, 5.0],
        ]
    )
    assert grid.evaluate(index).tolist() == [0.0, 1.0, 2.0, 4.0, 8.0]


def test_a_year_can_pay_on_one_hazard_alone() -> None:
    grid = StepGrid({"rain": [40.0], "heat": [30.0]}, [[0.0, 1.0], [2.0, 3.0]])
    assert grid.evaluate(np.array([[50.0, 0.0], [0.0, 50.0]])).tolist() == [2.0, 1.0]


def test_the_grid_is_a_payout_linear_in_its_levels() -> None:
    grid = _grid()
    assert isinstance(grid, PayoutFunction)
    assert isinstance(grid, LinearInAmounts)
    assert _REGISTRY["step_grid"] is StepGrid
    assert grid.index_names == ("rain", "heat")
    assert grid.tail == ("upper", "upper")


def test_the_parameters_are_named_and_ordered() -> None:
    grid = StepGrid(
        {"rain": [Free(0.0, 50.0, grid=[10.0, 20.0]), 40.0], "heat": [30.0]},
        [[0.0, Free()], [1.0, Free()], [Free(), Free()]],
    )
    assert list(grid.parameters()) == [
        "rain.breakpoints[0]",
        "rain.breakpoints[1]",
        "heat.breakpoints[0]",
        "levels[0,1]",
        "levels[1,0]",
        "levels[1,1]",
        "levels[2,0]",
        "levels[2,1]",
    ]
    assert grid.amount_names() == (
        "levels[0,1]",
        "levels[1,0]",
        "levels[1,1]",
        "levels[2,0]",
        "levels[2,1]",
    )


def test_with_values_resolves_some_of_the_parameters() -> None:
    template = StepGrid(
        {"rain": [Free(0.0, 50.0, grid=[10.0, 20.0]), 40.0], "heat": [30.0]},
        [[0.0, Free()], [1.0, Free()], [Free(), Free()]],
    )
    resolved = template.with_values({"rain.breakpoints[0]": 20.0, "levels[2,1]": 7.0})
    assert resolved.breakpoints["rain"] == (20.0, 40.0)
    assert resolved.levels[2][1] == 7.0
    assert isinstance(resolved.levels[0][1], Free)
    assert template.breakpoints["rain"][0] != 20.0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param({"levels[0,0]": 1.0}, "unknown parameter", id="the-empty-cell"),
        pytest.param({"levels[1,0]": 1.0}, "is fixed", id="a-fixed-level"),
        pytest.param({"levels[2,0]": math.inf}, "finite", id="not-finite"),
        pytest.param({"levels[2,0]": -1.0}, "outside", id="out-of-bounds"),
        pytest.param({"rain.breakpoints[0]": 60.0}, "strictly", id="unordered"),
    ],
)
def test_with_values_refuses_what_is_not_a_valid_table(
    values: Mapping[str, float], message: str
) -> None:
    template = StepGrid(
        {"rain": [Free(0.0, 90.0, grid=[20.0, 60.0]), 40.0], "heat": [30.0]},
        [[0.0, 2.0], [1.0, 2.0], [Free(), 2.0]],
        monotone=False,
    )
    with pytest.raises(CoverDefinitionError, match=message):
        template.with_values(values)


def test_a_resolved_level_must_keep_the_order() -> None:
    template = StepGrid({"rain": [40.0], "heat": [30.0]}, [[0.0, 2.0], [1.0, Free()]])
    with pytest.raises(CoverDefinitionError, match="non-decreasing"):
        template.with_values({"levels[1,1]": 1.5})


@pytest.mark.parametrize(
    "tails", [("upper", "upper"), ("lower", "upper")], ids=lambda tails: tails[0]
)
def test_the_basis_is_one_hot_on_the_cells(tails: Tails) -> None:
    grid = _grid(tails)
    rain, heat, _, _ = _draws()
    index = np.column_stack([_side(rain, tails[0]), _side(heat, tails[1])])
    basis = grid.basis(index)
    assert basis.shape == (index.shape[0], 8)
    assert set(np.unique(basis)) <= {0.0, 1.0}
    assert basis.sum(axis=1).max() == 1.0
    amounts = np.arange(1.0, 9.0)
    names = grid.amount_names()
    paid = grid.with_values(dict(zip(names, amounts.tolist(), strict=True))).evaluate(
        index
    )
    np.testing.assert_array_equal(basis @ amounts, paid)
    below = (rain < RAIN_BINS[0]) & (heat < HEAT_BINS[0])
    assert below.any() and (basis[below] == 0.0).all()


@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (1, 3), (3, 2)])
def test_the_constraints_have_the_stated_rows(shape: tuple[int, int]) -> None:
    m, q = shape
    grid = StepGrid(
        {
            "rain": [float(k + 1) for k in range(m)],
            "heat": [float(k + 1) for k in range(q)],
        },
        _free_table(m + 1, q + 1),
    )
    constraints = grid.amount_constraints()
    assert constraints.n_amounts == (m + 1) * (q + 1) - 1
    assert constraints.a_ub.shape[0] == m * (q + 1) + q * (m + 1) - 2
    names = grid.amount_names()
    found = {
        (names[int(row.argmax())], names[int(row.argmin())]) for row in constraints.a_ub
    }
    expected = {
        (f"levels[{i},{j}]", f"levels[{k},{n}]")
        for i, j in itertools.product(range(m + 1), range(q + 1))
        for k, n in ((i + 1, j), (i, j + 1))
        if k <= m and n <= q and (i, j) != (0, 0)
    }
    assert found == expected
    assert (constraints.b_ub == 0.0).all()


def test_monotone_false_has_no_order_rows_and_is_recorded() -> None:
    free = _grid(monotone=False)
    assert free.amount_constraints().a_ub.shape == (0, 8)
    assert free.to_document()["monotone"] is False
    assert _cover(free).definition_hash != _cover(_grid()).definition_hash


def test_the_bounds_read_the_cells_and_the_limit() -> None:
    grid = StepGrid(
        {"rain": [40.0], "heat": [30.0]},
        [[0.0, Free(1.0, 5.0)], [2.0, Free()]],
        monotone=False,
    )
    plain = grid.amount_constraints()
    assert plain.lower.tolist() == [1.0, 2.0, 0.0]
    assert plain.upper.tolist() == [5.0, 2.0, math.inf]
    assert grid.amount_constraints(3.0).upper.tolist() == [3.0, 2.0, 3.0]
    with pytest.raises(CoverDefinitionError, match="limit must be"):
        grid.amount_constraints(-1.0)


@pytest.mark.parametrize(
    "payout",
    [
        pytest.param(_grid(), id="every-level-free"),
        pytest.param(_grid(("upper", "lower"), monotone=False), id="mixed-tails"),
        pytest.param(
            StepGrid(
                {"rain": [Free(0.0, 50.0, grid=[10.0, 20.0]), 40.0], "heat": [30.0]},
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            ),
            id="a-free-breakpoint",
        ),
    ],
)
def test_the_grid_round_trips_through_a_cover(payout: StepGrid) -> None:
    tails = payout.tail
    cover = _cover(payout, tails)
    read = Cover.from_json(cover.to_json())
    assert read.definition_hash == cover.definition_hash
    assert read.payout == payout
    assert hash(read.payout) == hash(payout)
    assert pickle.loads(pickle.dumps(payout)) == payout
    assert pickle.loads(pickle.dumps(cover)).definition_hash == cover.definition_hash


def test_the_document_keeps_the_order_of_the_components() -> None:
    document = _grid().to_document()
    assert [entry["name"] for entry in document["breakpoints"]] == ["rain", "heat"]
    swapped = StepGrid({"heat": HEAT_BINS, "rain": RAIN_BINS}, _free_table(3, 3))
    assert _grid() != swapped


def test_the_grid_is_read_only() -> None:
    grid = _grid()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, grid).monotone = False
    with pytest.raises(TypeError):
        cast(Any, grid.breakpoints)["rain"] = (1.0,)


def test_a_cover_needs_the_names_and_tails_of_its_index() -> None:
    with pytest.raises(CoverDefinitionError, match="in that order"):
        Cover(
            CompositeIndex(
                {
                    "heat": IndexDefinition(
                        "heat", "degC", aggregation="annual_maximum"
                    ),
                    "rain": IndexDefinition(
                        "rain", "mm/day", aggregation="annual_maximum"
                    ),
                }
            ),
            _grid(),
            payout_unit="EUR",
        )
    with pytest.raises(CoverDefinitionError, match="tails"):
        _cover(_grid(("upper", "lower")), ("upper", "upper"))


def test_a_grid_is_no_component_of_a_sum() -> None:
    with pytest.raises(CoverDefinitionError, match="one index"):
        SumOf({"a": _grid(), "b": BinaryPayout(1.0, 1.0)})


def _invalid(*args: Any, **kwargs: Any) -> Any:
    untyped: Any = StepGrid
    return lambda: untyped(*args, **kwargs)


ONE_BY_ONE = [[0.0, 1.0], [2.0, 3.0]]
AXES: dict[str, Any] = {"rain": [40.0], "heat": [30.0]}


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(_invalid([], ONE_BY_ONE), "maps each of its two", id="no-mapping"),
        pytest.param(_invalid({"a": [1.0]}, ONE_BY_ONE), "exactly two", id="one"),
        pytest.param(
            _invalid({"a": [1.0], "b": [1.0], "c": [1.0]}, ONE_BY_ONE),
            "exactly two",
            id="three",
        ),
        pytest.param(
            _invalid({"a.b": [1.0], "c": [1.0]}, ONE_BY_ONE),
            "component name",
            id="bad-name",
        ),
        pytest.param(
            _invalid({"a": [], "b": [1.0]}, [[0.0, 1.0]]), "at least 1", id="no-bin"
        ),
        pytest.param(
            _invalid({"a": "ab", "b": [1.0]}, ONE_BY_ONE), "sequence", id="a-string"
        ),
        pytest.param(
            _invalid({"a": [2.0, 1.0], "b": [1.0]}, _free_table(3, 2)),
            "strictly increasing",
            id="unordered",
        ),
        pytest.param(
            _invalid({"a": [1.0, 1.0], "b": [1.0]}, _free_table(3, 2)),
            "strictly increasing",
            id="repeated",
        ),
        pytest.param(
            _invalid(
                {"a": [1.0, 2.0], "b": [1.0]},
                _free_table(3, 2),
                tail=("lower", "upper"),
            ),
            "strictly decreasing",
            id="lower-tail-increasing",
        ),
        pytest.param(
            _invalid({"a": [Free(0.0, 9.0)], "b": [1.0]}, ONE_BY_ONE),
            "needs a finite upper bound and a grid",
            id="a-free-threshold-without-a-grid",
        ),
        pytest.param(
            _invalid(
                {
                    "a": [Free(0.0, 9.0, grid=[3.0]), Free(0.0, 9.0, grid=[1.0])],
                    "b": [1.0],
                },
                _free_table(3, 2),
            ),
            "no choice of the free values",
            id="no-ordered-choice",
        ),
        pytest.param(
            _invalid(AXES, 3.0), "levels must be a sequence", id="not-a-table"
        ),
        pytest.param(_invalid(AXES, [1.0, 2.0]), "sequence", id="flat"),
        pytest.param(_invalid(AXES, [[0.0, 1.0]]), "2 rows of 2", id="too-few-rows"),
        pytest.param(
            _invalid(AXES, [[0.0], [1.0]]), "2 rows of 2", id="too-few-columns"
        ),
        pytest.param(
            _invalid(AXES, [[0.0, 1.0], [2.0, "x"]]),
            "levels\\[1\\]",
            id="a-string-level",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, 1.0], [2.0, True]]), "a number", id="a-bool-level"
        ),
        pytest.param(_invalid(AXES, [[0.0, 1.0], [2.0, math.inf]]), "finite", id="inf"),
        pytest.param(
            _invalid(AXES, [[1.0, 1.0], [2.0, 3.0]]),
            "levels\\[0\\]\\[0\\] must be 0.0",
            id="the-first-cell-pays",
        ),
        pytest.param(
            _invalid(AXES, [[Free(), 1.0], [2.0, 3.0]]),
            "levels\\[0\\]\\[0\\] must be 0.0",
            id="the-first-cell-is-free",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, -1.0], [2.0, 3.0]], monotone=False),
            "levels\\[0,1\\] must be >= 0",
            id="negative",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, Free(-1.0)], [2.0, 3.0]], monotone=False),
            "the lower bound of levels\\[0,1\\]",
            id="a-negative-lower-bound",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, 1.0], [2.0, 0.5]]),
            "non-decreasing along both axes",
            id="breaks-the-order-of-the-rows",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, 5.0], [6.0, 5.5]]),
            "levels\\[1,1\\]",
            id="breaks-the-order-of-the-columns",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, 3.0], [1.0, Free(0.0, 2.0)]]),
            "levels\\[1,1\\]",
            id="a-free-upper-bound-below-its-predecessor",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, 3.0], [1.0, Free(0.0, 9.0, grid=[1.0, 2.0])]]),
            "levels\\[1,1\\]",
            id="a-grid-below-its-predecessor",
        ),
        pytest.param(
            _invalid(AXES, [[0.0, Free(4.0)], [1.0, 3.0]]),
            "levels\\[1,1\\]",
            id="a-fixed-level-below-a-free-lower-bound",
        ),
        pytest.param(_invalid(AXES, ONE_BY_ONE, tail="upper"), "tail must", id="str"),
        pytest.param(
            _invalid(AXES, ONE_BY_ONE, tail=("upper",)), "tail must", id="one-tail"
        ),
        pytest.param(
            _invalid(AXES, ONE_BY_ONE, tail=("upper", "both")), "tail must", id="side"
        ),
        pytest.param(
            _invalid(AXES, ONE_BY_ONE, monotone=1), "monotone must", id="monotone"
        ),
    ],
)
def test_an_invalid_grid_raises(build: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        build()


def test_an_ordered_table_with_free_levels_is_accepted() -> None:
    table: list[list[float | Free]] = [
        [0.0, Free(1.0)],
        [Free(0.0, 9.0, grid=[2.0, 4.0]), Free()],
    ]
    assert StepGrid(AXES, table).levels[0][0] == 0.0


def test_an_array_table_is_read() -> None:
    grid = StepGrid(
        {"rain": np.array([40.0]), "heat": np.array([30.0])},
        np.array([[0.0, 1.0], [2.0, 3.0]]),
    )
    assert grid.levels == ((0.0, 1.0), (2.0, 3.0))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(lambda d: d.update(kind="step"), "expected a payout", id="kind"),
        pytest.param(lambda d: d.pop("tail"), "has the keys", id="missing-key"),
        pytest.param(lambda d: d.update(extra=1), "has the keys", id="extra-key"),
        pytest.param(
            lambda d: d.update(breakpoints={"rain": [1.0]}), "is a list", id="a-mapping"
        ),
        pytest.param(
            lambda d: d["breakpoints"].__setitem__(0, {"name": "rain"}),
            "has the keys",
            id="an-axis-without-thresholds",
        ),
        pytest.param(
            lambda d: d["breakpoints"][1].update(name="rain"),
            "repeats component",
            id="a-repeated-name",
        ),
    ],
)
def test_an_invalid_document_raises(edit: Any, message: str) -> None:
    document = _grid().to_document()
    edit(document)
    with pytest.raises(CoverDefinitionError, match=message):
        StepGrid.from_document(document)


def test_a_document_must_be_an_object() -> None:
    with pytest.raises(CoverDefinitionError, match="must be an object"):
        StepGrid.from_document(cast(Any, []))


def test_a_template_pays_nothing_until_resolved() -> None:
    template = _grid()
    index = np.array([[50.0, 50.0]])
    with pytest.raises(UnresolvedParameterError, match="levels\\[0,1\\]"):
        template.evaluate(index)
    with pytest.raises(UnresolvedParameterError):
        template.breakdown(index)
    assert template.basis(index).shape == (1, 8)
    free_threshold = StepGrid(
        {"rain": [Free(0.0, 50.0, grid=[10.0, 20.0])], "heat": [30.0]},
        [[0.0, Free()], [Free(), Free()]],
    )
    with pytest.raises(UnresolvedParameterError, match="rain.breakpoints"):
        free_threshold.basis(index)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(np.array([1.0, 2.0]), id="one-dimensional"),
        pytest.param(np.ones((2, 3)), id="three-columns"),
        pytest.param(np.array([[np.nan, 1.0]]), id="not-finite"),
    ],
)
def test_an_invalid_index_raises(index: Any) -> None:
    grid = StepGrid(AXES, ONE_BY_ONE)
    with pytest.raises(SampleError):
        grid.evaluate(index)
    with pytest.raises(SampleError):
        grid.basis(index)


# -- the special cases -------------------------------------------------------


def _lattice(tails: Tails, rain: Sequence[float], heat: Sequence[float]) -> Any:
    """Every pair of the two axes' values, at each threshold and between them."""

    def points(thresholds: Sequence[float]) -> list[float]:
        below = [thresholds[0] - 1.0]
        return below + [
            value for threshold in thresholds for value in (threshold, threshold + 0.5)
        ]

    pairs = list(itertools.product(points(rain), points(heat)))
    return np.column_stack(
        [
            _side([pair[0] for pair in pairs], tails[0]),
            _side([pair[1] for pair in pairs], tails[1]),
        ]
    )


THETA = (1.5, 4.0)
PHI = (0.25, 2.0, 7.0)
HEAT_THREE = (20.0, 30.0, 60.0)


def _steps(
    tails: Tails, theta: Sequence[float], phi: Sequence[float], *, monotone: bool = True
) -> dict[str, PayoutFunction]:
    return {
        "rain": StepPayout(
            _side(RAIN_BINS, tails[0]), theta, tail=tails[0], monotone=monotone
        ),
        "heat": StepPayout(
            _side(HEAT_THREE, tails[1]), phi, tail=tails[1], monotone=monotone
        ),
    }


def _combined_grid(
    tails: Tails,
    theta: Sequence[float],
    phi: Sequence[float],
    rule: Any,
    *,
    monotone: bool = True,
) -> StepGrid:
    table = [[rule(t, f) for f in (0.0, *phi)] for t in (0.0, *theta)]
    return StepGrid(
        {"rain": _side(RAIN_BINS, tails[0]), "heat": _side(HEAT_THREE, tails[1])},
        table,
        tail=tails,
        monotone=monotone,
    )


@pytest.mark.parametrize("tails", TAILS)
@pytest.mark.parametrize(
    ("theta", "phi", "monotone"),
    [
        pytest.param(THETA, PHI, True, id="monotone"),
        pytest.param((4.0, 1.5), (7.0, 0.25, 2.0), False, id="not-monotone"),
    ],
)
def test_a_sum_of_steps_is_the_grid_of_summed_levels(
    tails: Tails, theta: Sequence[float], phi: Sequence[float], monotone: bool
) -> None:
    index = _lattice(tails, RAIN_BINS, HEAT_THREE)
    summed = SumOf(_steps(tails, theta, phi, monotone=monotone))
    grid = _combined_grid(tails, theta, phi, lambda a, b: a + b, monotone=monotone)
    np.testing.assert_array_equal(grid.evaluate(index), summed.evaluate(index))
    assert summed.evaluate(index).max() > 0.0


@pytest.mark.parametrize("tails", TAILS)
@pytest.mark.parametrize(
    ("theta", "phi", "monotone"),
    [
        pytest.param(THETA, PHI, True, id="monotone"),
        pytest.param((4.0, 1.5), (7.0, 0.25, 2.0), False, id="not-monotone"),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_a_maximum_of_steps_is_the_grid_of_the_larger_levels(
    tails: Tails,
    theta: Sequence[float],
    phi: Sequence[float],
    monotone: bool,
    scale: float,
) -> None:
    index = _lattice(tails, RAIN_BINS, HEAT_THREE)
    largest = MaxOf(_steps(tails, theta, phi, monotone=monotone), scale=scale)
    grid = _combined_grid(
        tails, theta, phi, lambda a, b: scale * max(a, b), monotone=monotone
    )
    np.testing.assert_array_equal(grid.evaluate(index), largest.evaluate(index))


# -- designs: reached and unreached cells --------------------------------------


@pytest.mark.parametrize("tails", TAILS)
def test_under_mse_each_reached_level_is_its_cells_mean_loss(tails: Tails) -> None:
    rain, heat, loss, probability = _draws()
    sample = _sample(rain, heat, loss, tails=tails, probability=probability)
    result = design(_cover(_grid(tails, monotone=False), tails), sample, MSE())
    assert result.route == "qp"
    assert result.unique is True
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for values in zip(rain, heat, loss, probability, strict=True):
        cell = (_bin(values[0], RAIN_BINS), _bin(values[1], HEAT_BINS))
        cells.setdefault(cell, []).append((values[2], values[3]))
    assert len(cells) == 8, sorted(cells)
    table = _table(result)
    for (i, j), rows in cells.items():
        if (i, j) != (0, 0):
            mean = sum(a * p for a, p in rows) / sum(p for _, p in rows)
            assert result.parameters[f"levels[{i},{j}]"] == pytest.approx(
                mean, rel=1e-8
            )
    assert result.unidentified == ("levels[2,2]",)
    assert result.parameters["levels[2,2]"] == 0.0
    # The means break the order, so this says something about monotone=False.
    assert (np.diff(table, axis=0) < 0).any() or (np.diff(table, axis=1) < 0).any()


@pytest.mark.parametrize("flip", [False, True], ids=["across", "down"])
def test_the_order_pools_the_cells_that_break_it(flip: bool) -> None:
    # Two years of loss 10 in one cell, one of loss 4 in the cell after it.
    first = [10.0, 10.0, 60.0, 60.0, 10.0]
    second = [60.0, 60.0, 60.0, 10.0, 10.0]
    loss = [10.0, 10.0, 4.0, 2.0, 0.0]
    rain, heat = (second, first) if flip else (first, second)
    grid = StepGrid({"rain": [40.0], "heat": [40.0]}, _free_table(2, 2))
    result = design(_cover(grid), _sample(rain, heat, loss), MSE())
    pooled = (10.0 + 10.0 + 4.0) / 3
    assert result.parameters["levels[0,1]" if not flip else "levels[1,0]"] == (
        pytest.approx(pooled, rel=1e-7)
    )
    assert result.parameters["levels[1,1]"] == pytest.approx(pooled, rel=1e-7)
    assert result.parameters["levels[1,0]" if not flip else "levels[0,1]"] == (
        pytest.approx(2.0, rel=1e-7)
    )


@pytest.mark.parametrize(
    "objective",
    [
        pytest.param(MSE(), id="mse"),
        pytest.param(ExpectedUncovered(), id="eu"),
        pytest.param(WeightedCVaR(0.5, 0.9), id="wcvar"),
    ],
)
@pytest.mark.parametrize(
    "pricing",
    [
        pytest.param(PurePremium(), id="pure"),
        pytest.param(CostOfCapitalLoad(0.1, 0.9), id="coc"),
    ],
)
@pytest.mark.parametrize("fill", ["lowest", "highest"])
@pytest.mark.parametrize("tails", TAILS)
def test_levels_are_monotone_on_both_axes(
    objective: Any, pricing: Any, fill: Any, tails: Tails
) -> None:
    rain, heat, loss, probability = _draws()
    sample = _sample(rain, heat, loss, tails=tails, probability=probability)
    result = design(
        _cover(_grid(tails, upper=float(loss.max())), tails),
        sample,
        objective,
        pricing=pricing,
        budget=Budget(ratio=0.8),
        options=DesignOptions(fill=fill),
    )
    table = _table(result)
    assert (np.diff(table, axis=0) >= 0.0).all()
    assert (np.diff(table, axis=1) >= 0.0).all()
    assert result.unidentified == ("levels[2,2]",)
    assert result.premium <= cast(float, result.budget_bound) * (1.0 + 1e-7)
    assert result.route == ("qp" if isinstance(objective, MSE) else "lp")
    # The payout rises with each severity, whichever side each index is on.
    paid = result.cover.payout_values(
        np.column_stack(
            [
                _side(np.repeat([10.0, 50.0, 90.0], 3), tails[0]),
                _side(np.tile([10.0, 50.0, 90.0], 3), tails[1]),
            ]
        )
    ).reshape(3, 3)
    np.testing.assert_array_equal(paid, table)


def test_a_mixed_tail_design_equals_the_upper_tail_design() -> None:
    rain, heat, loss, probability = _draws()
    designs = [
        design(
            _cover(_grid(tails), tails),
            _sample(rain, heat, loss, tails=tails, probability=probability),
            MSE(),
        )
        for tails in TAILS
    ]
    for result in designs[1:]:
        np.testing.assert_allclose(_table(result), _table(designs[0]), rtol=1e-8)
        assert result.unidentified == designs[0].unidentified


def _all_but_the_middle(
    loss: Mapping[tuple[int, int], float],
) -> tuple[list[float], list[float], list[float]]:
    points = (10.0, 50.0, 90.0)
    rain = [points[i] for i, _ in loss]
    heat = [points[j] for _, j in loss]
    return rain, heat, list(loss.values())


MEANS = {
    (0, 0): 0.0,
    (0, 1): 2.0,
    (0, 2): 4.0,
    (1, 0): 3.0,
    (1, 2): 7.0,
    (2, 0): 5.0,
    (2, 1): 9.0,
    (2, 2): 12.0,
}


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        pytest.param("lowest", 3.0, id="lowest-is-the-larger-predecessor"),
        pytest.param("highest", 7.0, id="highest-is-the-smaller-successor"),
    ],
)
def test_the_fill_rule_runs_on_the_order_of_both_axes(
    fill: Any, expected: float
) -> None:
    rain, heat, loss = _all_but_the_middle(MEANS)
    grid = StepGrid(
        {"rain": RAIN_BINS, "heat": (30.0, 80.0)}, _free_table(3, 3, upper=100.0)
    )
    result = design(
        _cover(grid), _sample(rain, heat, loss), MSE(), options=DesignOptions(fill=fill)
    )
    assert result.unidentified == ("levels[1,1]",)
    assert result.fill == fill
    assert result.parameters["levels[1,1]"] == pytest.approx(expected, rel=1e-7)


def test_fill_highest_needs_a_bound_above_an_unreached_top_cell() -> None:
    rain, heat, loss, _ = _draws()
    with pytest.raises(DesignError, match="needs an upper bound"):
        design(
            _cover(_grid()),
            _sample(rain, heat, loss),
            MSE(),
            options=DesignOptions(fill="highest"),
        )


def test_every_cell_is_unidentified_when_no_year_pays() -> None:
    grid = StepGrid({"rain": [1000.0], "heat": [1000.0]}, _free_table(2, 2, upper=9.0))
    result = design(_cover(grid), _sample([1.0, 2.0], [1.0, 2.0], [3.0, 4.0]), MSE())
    assert result.unidentified == ("levels[0,1]", "levels[1,0]", "levels[1,1]")
    assert set(result.parameters.values()) == {0.0}


def test_a_fixed_cell_stays_and_is_no_parameter() -> None:
    rain, heat, loss = _all_but_the_middle(MEANS)
    grid = StepGrid(
        {"rain": RAIN_BINS, "heat": (30.0, 80.0)},
        [
            [0.0, 1.0, Free()],
            [Free(), Free(), Free()],
            [Free(), 50.0, Free()],
        ],
        monotone=False,
    )
    result = design(_cover(grid), _sample(rain, heat, loss), MSE())
    assert "levels[0,1]" not in result.parameters
    assert "levels[2,1]" not in result.parameters
    table = _table(result)
    assert table[0, 1] == 1.0 and table[2, 1] == 50.0
    assert table[2, 2] == pytest.approx(12.0, rel=1e-7)


def test_mse_under_a_binding_budget_lowers_every_cell_equally() -> None:
    # Equal probabilities and means above the shift: theta_c = mean_c - lambda.
    rain, heat, loss = _all_but_the_middle({**MEANS, (1, 1): 6.0})
    keep = [i for i, (r, h) in enumerate(zip(rain, heat, strict=True)) if r + h > 20.0]
    rain = [rain[i] for i in keep]
    heat = [heat[i] for i in keep]
    loss = [loss[i] for i in keep]
    mean = sum(loss) / len(loss)
    bound = mean - 1.0  # lambda = 1 with every one of the n cells paying
    grid = StepGrid({"rain": RAIN_BINS, "heat": (30.0, 80.0)}, _free_table(3, 3))
    result = design(
        _cover(grid), _sample(rain, heat, loss), MSE(), budget=Budget(amount=bound)
    )
    assert result.premium == pytest.approx(bound, rel=1e-7)
    for (i, j), level in zip(
        [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        [2.0, 4.0, 3.0, 6.0, 7.0, 5.0, 9.0, 12.0],
        strict=True,
    ):
        assert result.parameters[f"levels[{i},{j}]"] == pytest.approx(
            level - 1.0, rel=1e-6
        ), (i, j)


def test_a_limit_bounds_every_cell() -> None:
    rain, heat, loss = _all_but_the_middle(MEANS)
    grid = StepGrid(
        {"rain": RAIN_BINS, "heat": (30.0, 80.0)},
        _free_table(3, 3),
        monotone=False,
    )
    result = design(_cover(grid), _sample(rain, heat, loss), MSE(), limit=Limit(6.0))
    table = _table(result)
    assert table.max() == pytest.approx(6.0, rel=1e-9)
    assert table[2, 2] == pytest.approx(6.0, rel=1e-9)
    assert table[0, 2] == pytest.approx(4.0, rel=1e-7)


def test_a_fixed_level_above_the_limit_is_infeasible() -> None:
    rain, heat, loss = _all_but_the_middle(MEANS)
    grid = StepGrid(
        {"rain": RAIN_BINS, "heat": (30.0, 80.0)},
        [[0.0, 9.0, Free()], [Free(), Free(), Free()], [Free(), Free(), Free()]],
        monotone=False,
    )
    with pytest.raises(InfeasibleDesignError, match="levels\\[0,1\\]"):
        design(_cover(grid), _sample(rain, heat, loss), MSE(), limit=Limit(6.0))


def test_the_ranges_report_a_pinned_design() -> None:
    rain, heat, loss, probability = _draws()
    result = design(
        _cover(_grid(upper=float(loss.max()))),
        _sample(rain, heat, loss, probability=probability),
        ExpectedUncovered(),
        budget=Budget(ratio=0.5),
        options=DesignOptions(ranges=True),
    )
    assert result.route == "lp"
    assert result.ranges is not None
    assert set(result.ranges) == set(result.parameters) - set(result.unidentified)


# -- designs: free breakpoints ------------------------------------------------


def _free_breakpoint_grid() -> StepGrid:
    return StepGrid(
        {
            "rain": [
                Free(10.0, 90.0, grid=[30.0, 50.0, 70.0]),
                Free(10.0, 90.0, grid=[30.0, 50.0, 70.0]),
            ],
            "heat": [Free(10.0, 90.0, grid=[40.0, 60.0])],
        },
        _free_table(3, 2),
        monotone=False,
    )


def _threshold_names() -> list[str]:
    return ["rain.breakpoints[0]", "rain.breakpoints[1]", "heat.breakpoints[0]"]


def test_free_breakpoints_search_the_ordered_points_of_the_product() -> None:
    rain, heat, loss, probability = _draws()
    sample = _sample(rain, heat, loss, probability=probability)
    result = design(_cover(_free_breakpoint_grid()), sample, MSE())
    assert result.route == "grid" and result.amounts == "qp"
    assert result.grid == {"candidates": 18, "optimal": 6, "invalid": 12}
    best: tuple[float, tuple[float, ...]] = (math.inf, ())
    for rain_low, rain_high, heat_first in itertools.product(
        (30.0, 50.0, 70.0), (30.0, 50.0, 70.0), (40.0, 60.0)
    ):
        if rain_low >= rain_high:
            continue
        fixed = StepGrid(
            {"rain": [rain_low, rain_high], "heat": [heat_first]},
            _free_table(3, 2),
            monotone=False,
        )
        value = design(_cover(fixed), sample, MSE()).objective_value
        if value < best[0]:
            best = (value, (rain_low, rain_high, heat_first))
    assert result.objective_value == pytest.approx(best[0], rel=1e-9)
    assert tuple(result.parameters[name] for name in _threshold_names()) == best[1]
    profile: Any = result.profile
    assert profile.column_names == [*_threshold_names(), "objective", "status"]
    assert set(profile.column("status").to_pylist()) == {"optimal", "invalid"}
    assert min(
        value for value in profile.column("objective").to_pylist() if value is not None
    ) == pytest.approx(best[0], rel=1e-9)


def test_free_breakpoints_stay_under_the_grid_cap() -> None:
    rain, heat, loss, _ = _draws()
    with pytest.raises(DesignError, match="18 points, more than max_grid_points=17"):
        design(
            _cover(_free_breakpoint_grid()),
            _sample(rain, heat, loss),
            MSE(),
            options=DesignOptions(max_grid_points=17),
        )


def test_free_breakpoints_work_on_a_lower_tail() -> None:
    rain, heat, loss, probability = _draws()
    tails: Tails = ("upper", "lower")
    grid = StepGrid(
        {
            "rain": [40.0],
            "heat": [Free(0.0, 90.0, grid=[60.0, 40.0][::-1])],
        },
        _free_table(2, 2),
        tail=tails,
        monotone=False,
    )
    result = design(
        _cover(grid, tails),
        _sample(rain, heat, loss, tails=tails, probability=probability),
        MSE(),
    )
    assert result.route == "grid"
    assert result.grid == {"candidates": 2, "optimal": 2}
    assert result.parameters["heat.breakpoints[0]"] in {40.0, 60.0}


def test_a_chosen_breakpoint_decides_which_cells_are_unidentified() -> None:
    # At 80 no rain year passes the first threshold, so 20 fits the loss better;
    # no year has heat at 90 or above, so the second column is never reached.
    grid = StepGrid(
        {"rain": [Free(0.0, 90.0, grid=[20.0, 80.0])], "heat": [90.0]},
        _free_table(2, 2),
        monotone=False,
    )
    sample = _sample(
        [50.0, 60.0, 10.0, 12.0], [10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 0.0, 0.0]
    )
    result = design(_cover(grid), sample, MSE())
    assert result.parameters["rain.breakpoints[0]"] == 20.0
    assert result.unidentified == ("levels[0,1]", "levels[1,1]")
    assert result.parameters["levels[1,0]"] == pytest.approx(5.0, rel=1e-7)


def test_an_lp_runs_at_each_point_of_free_breakpoints() -> None:
    rain, heat, loss, probability = _draws()
    result = design(
        _cover(_free_breakpoint_grid()),
        _sample(rain, heat, loss, probability=probability),
        WeightedCVaR(0.5, 0.9),
        budget=Budget(ratio=0.8),
    )
    assert result.route == "grid" and result.amounts == "lp"
    assert result.grid == {"candidates": 18, "optimal": 6, "invalid": 12}


def test_a_user_objective_searches_the_levels_too() -> None:
    def absolute(loss: Any, payout: Any, probability: Any) -> float:
        return float(np.sum(probability * np.abs(loss - payout)))

    grid_points = [0.0, 2.0, 4.0]
    grid = StepGrid(
        {"rain": [40.0], "heat": [40.0]},
        [
            [0.0, Free(0.0, 4.0, grid=grid_points)],
            [Free(0.0, 4.0, grid=grid_points), Free(0.0, 4.0, grid=grid_points)],
        ],
    )
    sample = _sample(
        [10.0, 60.0, 60.0, 10.0], [60.0, 10.0, 60.0, 10.0], [1.9, 3.8, 3.0, 0.0]
    )
    result = design(_cover(grid), sample, UserObjective(absolute, name="abs"))
    assert result.route == "grid" and result.solver is None
    best = min(
        (
            0.25 * (abs(1.9 - a) + abs(3.8 - b) + abs(3.0 - c))
            for a, b, c in itertools.product(grid_points, repeat=3)
            if a <= c and b <= c
        )
    )
    assert result.objective_value == pytest.approx(best, abs=1e-12)
    assert result.parameters == {
        "levels[0,1]": 2.0,
        "levels[1,0]": 4.0,
        "levels[1,1]": 4.0,
    }
    # Candidates that break the order are refused by the table, and recorded.
    counts = cast(Mapping[str, int], result.grid)
    assert counts["candidates"] == 27 and counts["invalid"] > 0


def test_a_limit_drops_user_objective_candidates() -> None:
    def absolute(loss: Any, payout: Any, probability: Any) -> float:
        return float(np.sum(probability * np.abs(loss - payout)))

    grid = StepGrid(
        {"rain": [40.0], "heat": [40.0]},
        [[0.0, 1.0], [1.0, Free(0.0, 9.0, grid=[2.0, 8.0])]],
    )
    sample = _sample([10.0, 60.0], [60.0, 60.0], [1.0, 8.0])
    result = design(
        _cover(grid), sample, UserObjective(absolute, name="abs"), limit=Limit(4.0)
    )
    assert result.parameters == {"levels[1,1]": 2.0}
    assert cast(Mapping[str, int], result.grid)["over_limit"] == 1


# -- evaluation and the breakdown ----------------------------------------------


BREAKDOWN_RAIN = np.array([10.0, 60.0, 20.0, 80.0, 30.0, 15.0, 70.0, 25.0])
BREAKDOWN_HEAT = np.array([30.0, 25.0, 40.0, 20.0, 35.0, 42.0, 45.0, 38.0])
BREAKDOWN_LOSS = np.array([0.0, 10.0, 4.0, 6.0, 3.0, 2.0, 20.0, 6.0])
"""Rain reaches 50 in years 1, 3, 6, heat 38 in 2, 5, 6, 7: year 6 is both."""


def _breakdown_cover(
    table: Sequence[Sequence[float]] = ((0.0, 4.0), (5.0, 9.0)),
    tails: Tails = ("upper", "upper"),
    *,
    monotone: bool = True,
) -> Cover:
    grid = StepGrid(
        {"rain": _side([50.0], tails[0]), "heat": _side([38.0], tails[1])},
        table,
        tail=tails,
        monotone=monotone,
    )
    return _cover(grid, tails)


def _breakdown_sample(tails: Tails = ("upper", "upper")) -> OutcomeSample:
    return _sample(BREAKDOWN_RAIN, BREAKDOWN_HEAT, BREAKDOWN_LOSS, tails=tails)


@pytest.mark.parametrize("tails", TAILS)
def test_a_grid_breaks_down_into_its_regions(tails: Tails) -> None:
    cover = _breakdown_cover(tails=tails)
    table = payout_breakdown([cover], _breakdown_sample(tails), split="in_sample")
    assert _column(table, "part") == ["only:rain", "only:heat", "both"]
    assert set(_column(table, "part_kind")) == {"region"}
    rain_alone = _row(table, part="only:rain")
    assert rain_alone["expected_payout"] == pytest.approx(2 * 5.0 / 8)
    assert rain_alone["payout_frequency"] == pytest.approx(2 / 8)
    heat_alone = _row(table, part="only:heat")
    assert heat_alone["expected_payout"] == pytest.approx(3 * 4.0 / 8)
    assert heat_alone["payout_frequency"] == pytest.approx(3 / 8)
    both = _row(table, part="both")
    assert both["expected_payout"] == pytest.approx(9.0 / 8)
    assert both["payout_frequency"] == pytest.approx(1 / 8)
    assert sum(_column(table, "share")) == pytest.approx(1.0, abs=1e-12)
    assert rain_alone["components"] == "rain+heat"


def test_the_regions_add_up_to_the_expected_payout() -> None:
    cover = _breakdown_cover()
    sample = _breakdown_sample()
    compared = evaluate(cover, sample, split="in_sample").to_pylist()[0]
    table = payout_breakdown([cover], sample, split="in_sample")
    assert sum(_column(table, "expected_payout")) == pytest.approx(
        compared["expected_payout"], abs=1e-12
    )
    assert compared["expected_payout"] == pytest.approx((10.0 + 12.0 + 9.0) / 8)


def test_a_region_a_cell_of_zero_leaves_empty() -> None:
    cover = _breakdown_cover(((0.0, 0.0), (5.0, 9.0)))
    table = payout_breakdown([cover], _breakdown_sample(), split="in_sample")
    alone = _row(table, part="only:heat")
    assert alone["expected_payout"] == 0.0 and alone["share"] == 0.0
    assert alone["payout_frequency"] == 0.0


def test_a_grid_that_pays_nothing_has_no_shares() -> None:
    cover = _breakdown_cover(((0.0, 0.0), (0.0, 0.0)))
    table = payout_breakdown([cover], _breakdown_sample(), split="in_sample")
    assert _column(table, "share") == [None, None, None]


def test_the_breakdown_of_a_design_carries_its_identity() -> None:
    grid = StepGrid({"rain": [50.0], "heat": [38.0]}, _free_table(2, 2))
    result = design(_cover(grid), _breakdown_sample(), MSE())
    table = payout_breakdown([result], _breakdown_sample(), split="in_sample")
    assert set(_column(table, "premium_basis")) == {"design"}
    assert set(_column(table, "design_id")) == {result.design_id}


def test_a_breakdown_of_regions_writes_and_reads(tmp_path: Path) -> None:
    table = payout_breakdown(
        [_breakdown_cover()], _breakdown_sample(), split="in_sample"
    )
    path = tmp_path / "breakdown.parquet"
    write_comparison(table, path)
    metadata = EvaluationMetadata.from_table(pq.read_table(path))
    assert metadata.setup.evaluations[0]["part_kinds"] == ["region"]


def test_the_setup_text_of_regions_claims_no_attribution() -> None:
    text = setup_text(
        payout_breakdown([_breakdown_cover()], _breakdown_sample(), split="in_sample")
    )
    assert "a payout breakdown holds no premiums" in text
    assert "attribution" not in text


def test_a_grid_is_a_row_of_a_comparison_beside_separate_covers() -> None:
    rain, heat, loss, _ = _draws()
    own = (0.6 * loss, 0.6 * loss)
    sample = _sample(rain, heat, loss, own_loss=own)
    train, test = holdout(sample, train_fraction=0.75)
    budget = Budget(ratio=0.8)
    grid = design(_cover(_grid(), label="holistic"), train, MSE(), budget=budget)
    budgets = split_budget(budget, train, shares="own_loss")
    separate = {
        name: design(
            Cover(
                IndexDefinition(name, unit, aggregation="annual_maximum"),
                StepPayout(thresholds, [Free(), Free()]),
                payout_unit="EUR",
                label=name,
            ),
            train.component(name),
            MSE(),
            budget=budgets[name],
        )
        for name, unit, thresholds in (
            ("rain", "mm/day", RAIN_BINS),
            ("heat", "degC", HEAT_BINS),
        )
    }
    table = compare(
        [grid, CoverPackage(separate, label="separate")], test, split="out_of_sample"
    )
    holistic = _row(table, cover_label="holistic")
    package = _row(table, level="package")
    assert holistic["components"] == "rain+heat"
    assert package["components"] == "rain+heat"
    assert holistic["sample_hash"] == package["sample_hash"]
    assert holistic["design_budget"] == pytest.approx(package["design_budget"])
