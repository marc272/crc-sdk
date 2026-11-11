"""A cover's terms in words: `Cover.terms_text()`."""

from __future__ import annotations

import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pytest

from crc_sdk.parametric import (
    AtCell,
    AtPoint,
    AtStation,
    BinaryPayout,
    CompositeIndex,
    Cover,
    Free,
    IndexDefinition,
    LinearConstraints,
    LinearPayout,
    MaxOf,
    PiecewiseLinearPayout,
    StepGrid,
    StepPayout,
    SumOf,
)
from crc_sdk.parametric.index import Tail, _number

DOCS = Path(__file__).resolve().parents[1] / "docs" / "parametric.md"

RAIN = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("txx", "degC", aggregation="annual_maximum")
BOTH = CompositeIndex({"rx1day": RAIN, "txx": HEAT})

STEP = Cover(
    RAIN,
    StepPayout(breakpoints=[60.0, 80.0, 100.0], levels=[1e3, 5e3, 2e4]),
    payout_unit="EUR",
    asset_id="warehouse-a",
)
STEP_HASH = "sha256:d6d4fb6ee74366eb0fe226b0dcecf07f992c23e4d5051953a9b0c9ffef9cf1e5"
STEP_TERMS = f"""\
Asset: warehouse-a
Index: rx1day, in mm/day, read at the asset; larger values are more severe (tail upper); aggregation: annual_maximum; source: not stated
Payout per year, in EUR: steps, the levels never decrease (kind "step", monotone=True)
  - nothing when rx1day is below 60 mm/day
  - 1,000 EUR when rx1day is at or above 60 mm/day and below 80 mm/day
  - 5,000 EUR when rx1day is at or above 80 mm/day and below 100 mm/day
  - 20,000 EUR when rx1day is at or above 100 mm/day
Most paid in a year: 20,000 EUR
Free parameters: none
Hash: {STEP_HASH}, of the cover document (schema 1.0, semantics crc-parametric-cover/1)"""  # noqa: E501

GRID = Cover(
    BOTH,
    StepGrid({"rx1day": [70.0], "txx": [36.0]}, [[0.0, Free()], [Free(), Free()]]),
    payout_unit="EUR",
    label="grid",
).with_values({"levels[0,1]": 1e5, "levels[1,0]": 2e5, "levels[1,1]": 4e5})
GRID_HASH = "sha256:8f7878026907fb2fbc65871f217450a717a359a633f7079c0c3562aa7cfda09c"
GRID_TERMS = f"""\
Label: grid (a display name, not part of the terms)
Asset: not named
Index: 2 components, in this order
  - rx1day: rx1day, in mm/day, read at the asset; larger values are more severe (tail upper); aggregation: annual_maximum; source: not stated
  - txx: txx, in degC, read at the asset; larger values are more severe (tail upper); aggregation: annual_maximum; source: not stated
Payout per year, in EUR: one amount per cell of a table over rx1day and txx, the levels never decrease along either (kind "step_grid", monotone=True)
  - nothing when rx1day is below 70 mm/day and txx is below 36 degC
  - 100,000 EUR when rx1day is below 70 mm/day and txx is at or above 36 degC
  - 200,000 EUR when rx1day is at or above 70 mm/day and txx is below 36 degC
  - 400,000 EUR when rx1day is at or above 70 mm/day and txx is at or above 36 degC
Most paid in a year: 400,000 EUR
Free parameters: none
Hash: {GRID_HASH}, of the cover document (schema 1.1, semantics crc-parametric-cover/1)"""  # noqa: E501


def _lines(cover: Cover) -> list[str]:
    return cover.terms_text().splitlines()


def _line(cover: Cover, start: str) -> str:
    (line,) = [line for line in _lines(cover) if line.startswith(start)]
    return line


def _payout_lines(cover: Cover) -> list[str]:
    """The indented lines under `Payout per year`, without their indent."""
    lines = _lines(cover)
    start = next(i for i, line in enumerate(lines) if line.startswith("Payout"))
    items: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        items.append(line[2:])
    return items


def _free_lines(cover: Cover) -> list[str]:
    lines = _lines(cover)
    start = next(i for i, line in enumerate(lines) if line.startswith("Free"))
    return [line[4:] for line in lines[start + 1 :] if line.startswith("  - ")]


# -- one index -------------------------------------------------------------------


def test_a_step_cover_states_its_terms() -> None:
    assert STEP.definition_hash == STEP_HASH
    assert STEP.terms_text() == STEP_TERMS


def test_a_linear_cover_states_attachment_exhaustion_and_limit() -> None:
    cover = Cover(
        IndexDefinition("rx1day", "mm/day"),
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        label="linear",
    )
    lines = _lines(cover)
    assert lines[0] == "Label: linear (a display name, not part of the terms)"
    assert lines[1] == "Asset: not named"
    assert lines[2] == (
        "Index: rx1day, in mm/day, read at the asset; larger values are more "
        "severe (tail upper); aggregation: not stated, so the terms don't say how "
        "a year's value is made; source: not stated"
    )
    assert lines[3] == (
        "Payout per year, in EUR: in proportion between an attachment and an "
        'exhaustion, up to a limit (kind "linear")'
    )
    assert _payout_lines(cover) == [
        "- nothing when rx1day is at or below 60 mm/day (the attachment)",
        "- in a straight line from nothing at 60 mm/day to 500,000 EUR at 120 mm/day",
        "- 500,000 EUR (the limit) when rx1day is at or above 120 mm/day "
        "(the exhaustion)",
    ]
    assert _line(cover, "Most") == "Most paid in a year: 500,000 EUR"


def test_a_lower_tail_reads_the_other_way() -> None:
    spei = IndexDefinition(
        "spei",
        "1",
        tail="lower",
        location=AtPoint(-0.1276, 51.5072, "london"),
        aggregation="annual_minimum",
        source="era5",
    )
    cover = Cover(
        spei,
        PiecewiseLinearPayout([-1.0, -1.5, -2.0], [1e-7, 2.5e5, 1e16], tail="lower"),
        payout_unit="GBP",
        asset_id="farm-7",
    )
    assert _line(cover, "Index") == (
        "Index: spei, dimensionless (unit 1), read at the point london "
        "(longitude -0.1276, latitude 51.5072, WGS84); smaller values are more "
        "severe (tail lower); aggregation: annual_minimum; source: era5"
    )
    assert _line(cover, "Payout").endswith(
        "straight lines between knots, the levels never decrease (kind "
        '"piecewise_linear", monotone=True)'
    )
    assert _payout_lines(cover) == [
        "- nothing when spei is above -1",
        "- in a straight line from 0.0000001 GBP at -1 to 250,000 GBP at -1.5",
        "- in a straight line from 250,000 GBP at -1.5 to "
        "10,000,000,000,000,000 GBP at -2",
        "- 10,000,000,000,000,000 GBP when spei is at or below -2",
    ]
    assert _line(cover, "Most") == ("Most paid in a year: 10,000,000,000,000,000 GBP")


def test_a_lower_tail_linear_and_binary_cover() -> None:
    dry = IndexDefinition("spi", "1", tail="lower")
    linear = Cover(dry, LinearPayout(-1.0, -2.0, 10.0, tail="lower"), payout_unit="1")
    assert _line(linear, "Payout").startswith("Payout per year, dimensionless (unit 1)")
    assert _payout_lines(linear) == [
        "- nothing when spi is at or above -1 (the attachment)",
        "- in a straight line from nothing at -1 to 10 at -2",
        "- 10 (the limit) when spi is at or below -2 (the exhaustion)",
    ]
    binary = Cover(dry, BinaryPayout(-1.5, 3e4, tail="lower"), payout_unit="USD")
    assert _payout_lines(binary) == [
        "- nothing when spi is above -1.5",
        "- 30,000 USD when spi is at or below -1.5 (the trigger)",
    ]


def test_a_binary_cover_at_a_station() -> None:
    cover = Cover(
        IndexDefinition("wind", "m/s", location=AtStation("ST-1"), source="synop"),
        BinaryPayout(30.0, 1e5),
        payout_unit="USD",
        asset_id="port",
    )
    assert "read at the station ST-1" in _line(cover, "Index")
    assert _line(cover, "Index").endswith("; source: synop")
    assert _line(cover, "Payout") == (
        'Payout per year, in USD: one amount from a trigger on (kind "binary")'
    )
    assert _payout_lines(cover) == [
        "- nothing when wind is below 30 m/s",
        "- 100,000 USD when wind is at or above 30 m/s (the trigger)",
    ]
    assert _line(cover, "Most") == "Most paid in a year: 100,000 USD"


@pytest.mark.parametrize(
    ("location", "words"),
    [
        (
            AtPoint(8.5, 47.25),
            "read at the point (longitude 8.5, latitude 47.25, WGS84)",
        ),
        (AtCell(0x872A1072BFFFFFF), "read in the H3 cell 872a1072bffffff"),
    ],
)
def test_the_index_says_where_it_is_read(location: Any, words: str) -> None:
    cover = Cover(
        IndexDefinition("rx1day", "mm/day", location=location),
        StepPayout([60.0], [1.0]),
        payout_unit="EUR",
    )
    assert f"Index: rx1day, in mm/day, {words}; " in cover.terms_text()


# -- templates ---------------------------------------------------------------------


def test_a_template_names_its_free_parameters_and_their_bounds() -> None:
    cover = Cover(
        IndexDefinition("wind", "m/s"),
        BinaryPayout(Free(20.0, 40.0, grid=5), Free(upper=1e6)),
        payout_unit="USD",
    )
    assert _payout_lines(cover) == [
        "- nothing when wind is below trigger",
        "- amount when wind is at or above trigger",
    ]
    assert _line(cover, "Most") == (
        "Most paid in a year: not known until the free amounts are chosen"
    )
    assert _line(cover, "Free") == (
        "Free parameters: a design chooses them, and the cover pays nothing until "
        "they are resolved"
    )
    assert _free_lines(cover) == [
        "trigger: from 20 m/s to 40 m/s, on the grid 20, 25, 30, 35, 40",
        "amount: from 0 USD to 1,000,000 USD",
    ]


def test_free_parameters_follow_the_cover_order() -> None:
    cover = Cover(
        RAIN,
        StepPayout(
            [Free(50.0, 70.0, grid=[50.0, 60.0, 70.0]), 80.0],
            [Free(upper=5e4, grid=list(range(0, 50001, 2500))), Free()],
            monotone=False,
        ),
        payout_unit="EUR",
    )
    assert _line(cover, "Payout").endswith(
        'steps, the levels in any order (kind "step", monotone=False)'
    )
    assert _payout_lines(cover) == [
        "- nothing when rx1day is below breakpoints[0]",
        "- levels[0] when rx1day is at or above breakpoints[0] and below 80 mm/day",
        "- levels[1] when rx1day is at or above 80 mm/day",
    ]
    assert [line.partition(":")[0] for line in _free_lines(cover)] == list(
        cover.free_parameters()
    )
    assert _free_lines(cover) == [
        "breakpoints[0]: from 50 mm/day to 70 mm/day, on the grid 50, 60, 70",
        "levels[0]: from 0 EUR to 50,000 EUR, on a grid of 21 values from 0 to 50,000",
        "levels[1]: from 0 EUR, with no upper bound",
    ]


def test_free_thresholds_with_fixed_amounts_still_state_the_most() -> None:
    cover = Cover(
        RAIN,
        PiecewiseLinearPayout([Free(50.0, 60.0, grid=2), 80.0], [0.0, 4e4]),
        payout_unit="EUR",
    )
    assert _line(cover, "Most") == "Most paid in a year: 40,000 EUR"
    assert _free_lines(cover) == [
        "knots[0]: from 50 mm/day to 60 mm/day, on the grid 50, 60"
    ]


def test_a_free_linear_cover_names_its_parameters() -> None:
    cover = Cover(
        RAIN,
        LinearPayout(Free(40.0, 60.0, grid=3), Free(80.0, 120.0, grid=3), Free()),
        payout_unit="EUR",
    )
    assert _payout_lines(cover) == [
        "- nothing when rx1day is at or below attachment",
        "- in a straight line from nothing at attachment to limit at exhaustion",
        "- limit when rx1day is at or above exhaustion",
    ]
    assert [line.partition(":")[0] for line in _free_lines(cover)] == [
        "attachment",
        "exhaustion",
        "limit",
    ]


# -- several indices --------------------------------------------------------------


def test_a_step_grid_states_every_cell() -> None:
    assert GRID.definition_hash == GRID_HASH
    assert GRID.terms_text() == GRID_TERMS


def test_a_step_grid_template_and_mixed_tails() -> None:
    dry = IndexDefinition("spei", "1", tail="lower")
    cover = Cover(
        CompositeIndex({"rx1day": RAIN, "spei": dry}),
        StepGrid(
            {"rx1day": [60.0, 80.0], "spei": [-1.0]},
            [[0.0, Free()], [Free(), Free()], [Free(), Free()]],
            tail=("upper", "lower"),
        ),
        payout_unit="EUR",
    )
    cells = _payout_lines(cover)
    assert len(cells) == 6
    assert cells[0] == "- nothing when rx1day is below 60 mm/day and spei is above -1"
    assert cells[3] == (
        "- levels[1,1] when rx1day is at or above 60 mm/day and below 80 mm/day "
        "and spei is at or below -1"
    )
    assert [line.partition(":")[0] for line in _free_lines(cover)] == list(
        cover.free_parameters()
    )


def test_a_sum_of_states_each_component() -> None:
    cover = Cover(
        BOTH,
        SumOf(
            {
                "rx1day": StepPayout([70.0, 90.0], [1e5, 2e5]),
                "txx": BinaryPayout(38.0, 5e4),
            }
        ),
        payout_unit="EUR",
        asset_id="warehouse-a",
    )
    assert _line(cover, "Index") == "Index: 2 components, in this order"
    assert _line(cover, "Payout") == (
        "Payout per year, in EUR: the sum of one payout per component, each on its "
        'own index (kind "sum_of")'
    )
    assert _payout_lines(cover) == [
        '- rx1day: steps, the levels never decrease (kind "step", monotone=True)',
        "  - nothing when rx1day is below 70 mm/day",
        "  - 100,000 EUR when rx1day is at or above 70 mm/day and below 90 mm/day",
        "  - 200,000 EUR when rx1day is at or above 90 mm/day",
        '- txx: one amount from a trigger on (kind "binary")',
        "  - nothing when txx is below 38 degC",
        "  - 50,000 EUR when txx is at or above 38 degC (the trigger)",
    ]
    assert _line(cover, "Most") == "Most paid in a year: 250,000 EUR"


def test_a_sum_of_template_names_component_parameters() -> None:
    cover = Cover(
        BOTH,
        SumOf({"rx1day": StepPayout([70.0], [Free()]), "txx": BinaryPayout(38.0, 5e4)}),
        payout_unit="EUR",
    )
    assert "  - rx1day.levels[0] when rx1day is at or above 70 mm/day" in _payout_lines(
        cover
    )
    assert _free_lines(cover) == ["rx1day.levels[0]: from 0 EUR, with no upper bound"]
    assert _line(cover, "Most") == (
        "Most paid in a year: not known until the free amounts are chosen"
    )


def test_a_max_of_states_its_scale() -> None:
    components = {
        "rx1day": LinearPayout(60.0, 120.0, 1e5),
        "txx": LinearPayout(35.0, 42.0, 2e5),
    }
    fixed = Cover(BOTH, MaxOf(components, scale=1.5), payout_unit="EUR")
    assert _line(fixed, "Payout") == (
        "Payout per year, in EUR: the largest of one payout per component, times "
        '1.5: one amount a year (kind "max_of")'
    )
    assert _payout_lines(fixed)[0] == (
        "- rx1day: in proportion between an attachment and an exhaustion, up to a "
        'limit (kind "linear")'
    )
    assert _line(fixed, "Most") == "Most paid in a year: 300,000 EUR"
    free = Cover(BOTH, MaxOf(components, scale=Free(upper=2.0)), payout_unit="EUR")
    assert "times scale: one amount a year" in _line(free, "Payout")
    assert _free_lines(free) == ["scale: from 0 to 2"]
    assert _line(free, "Most") == (
        "Most paid in a year: not known until the free amounts are chosen"
    )


# -- families of your own -----------------------------------------------------------


class Tiered:
    """A test family on one index, linear in its amount, with an extra parameter."""

    kind: ClassVar[str] = "test_terms_tiered"

    def __init__(self, trigger: Any, amount: Any, *, tail: Tail = "upper") -> None:
        self.trigger, self.amount, self._tail = trigger, amount, tail

    @property
    def tail(self) -> Tail:
        return self._tail

    @property
    def index_names(self) -> tuple[str, ...]:
        return ("index",)

    def parameters(self) -> dict[str, float | Free]:
        return {"trigger": self.trigger, "amount": self.amount, "count": 3}

    def with_values(self, values: Mapping[str, float]) -> Self:
        return type(self)(
            values.get("trigger", self.trigger), values.get("amount", self.amount)
        )

    def amount_names(self) -> tuple[str, ...]:
        return ("amount",)

    def basis(self, index: Any) -> Any:
        return (np.asarray(index) >= self.trigger).astype(np.float64).reshape(-1, 1)

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        return LinearConstraints(
            a_ub=np.zeros((0, 1)), b_ub=np.zeros(0), lower=np.zeros(1), upper=np.ones(1)
        )

    def evaluate(self, index: Any) -> Any:
        return self.basis(index)[:, 0] * self.amount

    def to_document(self) -> dict[str, Any]:
        def value(item: Any) -> Any:
            return item.to_document() if isinstance(item, Free) else item

        return {
            "amount": value(self.amount),
            "kind": self.kind,
            "tail": self._tail,
            "trigger": value(self.trigger),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(document["trigger"], document["amount"], tail=document["tail"])


def test_a_family_of_your_own_states_its_parameters() -> None:
    cover = Cover(RAIN, Tiered(60.0, Free(upper=1e4)), payout_unit="EUR")
    assert _line(cover, "Payout") == (
        'Payout per year, in EUR: a family of your own (kind "test_terms_tiered"), '
        "with its parameters as it names them"
    )
    assert _payout_lines(cover) == ["- trigger: 60", "- amount: free", "- count: 3"]
    assert _free_lines(cover) == ["amount: from 0 EUR to 10,000 EUR"]
    assert _line(cover, "Most") == (
        "Most paid in a year: not known: a family of your own states none"
    )


def test_a_family_of_your_own_inside_a_sum_of() -> None:
    cover = Cover(
        BOTH,
        SumOf({"rx1day": Tiered(60.0, 5e3), "txx": BinaryPayout(38.0, 5e4)}),
        payout_unit="EUR",
    )
    lines = _payout_lines(cover)
    assert lines[0] == (
        '- rx1day: a family of your own (kind "test_terms_tiered"), with its '
        "parameters as it names them"
    )
    assert lines[1:4] == ["  - trigger: 60", "  - amount: 5,000 EUR", "  - count: 3"]
    assert _line(cover, "Most") == (
        "Most paid in a year: not known: a family of your own states none"
    )


def test_a_subclass_of_a_built_in_family_states_its_parameters() -> None:
    class Renamed(BinaryPayout):
        kind: ClassVar[str] = "test_terms_renamed"

    # BinaryPayout's own __init__ takes the parameters by position.
    renamed: Any = Renamed
    cover = Cover(RAIN, renamed(70.0, 2e4), payout_unit="EUR")
    assert '(kind "test_terms_renamed")' in _line(cover, "Payout")
    assert _payout_lines(cover) == ["- trigger: 70", "- amount: 20,000 EUR"]


# -- a view only -------------------------------------------------------------------


COVERS = {
    "step": STEP,
    "grid": GRID,
    "linear": Cover(RAIN, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR"),
    "piecewise": Cover(
        RAIN, PiecewiseLinearPayout([60.0, 90.0], [1e3, 3e4]), payout_unit="EUR"
    ),
    "binary": Cover(RAIN, BinaryPayout(70.0, Free()), payout_unit="EUR"),
    "sum_of": Cover(
        BOTH,
        SumOf({"rx1day": StepPayout([70.0], [1e5]), "txx": BinaryPayout(38.0, 5e4)}),
        payout_unit="EUR",
        label="sum",
    ),
    "max_of": Cover(
        BOTH,
        MaxOf(
            {"rx1day": LinearPayout(60.0, 120.0, 1.0), "txx": BinaryPayout(38.0, 1.0)},
            scale=Free(),
        ),
        payout_unit="EUR",
    ),
}


@pytest.mark.parametrize("name", sorted(COVERS))
def test_the_terms_are_a_view_of_the_document(name: str) -> None:
    cover = COVERS[name]
    document, text, digest = cover.to_document(), cover.to_json(), cover.definition_hash
    terms = cover.terms_text()
    assert (cover.to_document(), cover.to_json(), cover.definition_hash) == (
        document,
        text,
        digest,
    )
    assert terms.splitlines()[-1].startswith(f"Hash: {digest}, of the cover document")
    # The same terms give the same text, however the cover was made.
    assert Cover.from_json(text, label=cover.label).terms_text() == terms
    assert pickle.loads(pickle.dumps(cover)).terms_text() == terms
    assert cover.terms_text() == terms


def test_the_label_is_the_only_line_outside_the_terms() -> None:
    labelled = Cover(
        STEP.index,
        STEP.payout,
        payout_unit="EUR",
        asset_id="warehouse-a",
        label="rain step",
    )
    assert labelled == STEP
    assert labelled.terms_text() == (
        "Label: rain step (a display name, not part of the terms)\n" + STEP_TERMS
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (60.0, "60"),
        (-0.0, "0"),
        (0.1, "0.1"),
        (-1.5, "-1.5"),
        (1234.5, "1,234.5"),
        (-1234567.25, "-1,234,567.25"),
        (1e-7, "0.0000001"),
        (1.5e-7, "0.00000015"),
        (1e16, "10,000,000,000,000,000"),
        (123456.78901234567, "123,456.78901234567"),
    ],
)
def test_numbers_are_spelled_exactly(value: float, text: str) -> None:
    assert _number(value) == text


def test_every_spelled_number_reads_back_as_itself() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(0.0, 1e6, 200), rng.lognormal(0.0, 8.0, 200), [5e-324, 1.7e308]]
    )
    for value in values:
        spelled = _number(float(value))
        assert re.fullmatch(r"-?\d{1,3}(,\d{3})*(\.\d+)?", spelled), spelled
        assert float(spelled.replace(",", "")) == value


def test_the_docs_page_shows_these_terms() -> None:
    page = DOCS.read_text(encoding="utf-8")
    for terms in (STEP_TERMS, GRID_TERMS):
        assert f"```text\n{terms}\n```" in page
