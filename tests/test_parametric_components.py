"""Named index components: composite covers, multi-hazard samples and their use.

The numbers are synthetic and chosen so that the optima have closed forms. The
tests check what the module computes and records; nothing here is a finding
about a cover.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AtPoint,
    BinaryPayout,
    Budget,
    ComonotoneIndexWarning,
    CompositeIndex,
    Cover,
    CoverDefinitionError,
    DesignError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    LeakageError,
    LinearConstraints,
    LinearInAmounts,
    LinearPayout,
    OutcomeSample,
    ParametricPlan,
    PayoutFunction,
    SampleError,
    SampleProvenance,
    StepPayout,
    UnresolvedParameterError,
    UserObjective,
    WeightedCVaR,
    _solver,
    compare,
    design,
    design_joint,
    evaluate,
    holdout,
    register_payout,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.constraints import Limit
from crc_sdk.parametric.cover import (
    COMPOSITE_COVER_SCHEMA_VERSION,
    COVER_SCHEMA_VERSION,
)
from crc_sdk.parametric.errors import SolverUnavailableError
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.metadata import _identity_columns
from crc_sdk.parametric.parameters import _resolve_values
from crc_sdk.parametric.payouts import _REGISTRY
from crc_sdk.parametric.sample import (
    SAMPLE_COMPONENTS_SCHEMA_VERSION,
    SAMPLE_METADATA_KEY,
    SAMPLE_SCHEMA_VERSION,
)
from crc_sdk.types import SourceProvenance

RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
DRY = IndexDefinition("heat", "degC", tail="lower", aggregation="annual_minimum")
TWO = CompositeIndex({"rain": RAIN, "heat": HEAT})


def _document_value(value: Any) -> Any:
    return value.to_document() if isinstance(value, Free) else value


def _read_value(value: Any) -> Any:
    return Free.from_document(value) if isinstance(value, Mapping) else value


class TwoTriggers:
    """A test family on two components: one amount per component's trigger.

    The payout is the sum, over the components, of `amount` in the years the
    component reaches its `trigger`.
    """

    kind: ClassVar[str] = "test_two_triggers"

    def __init__(
        self,
        triggers: tuple[Any, Any],
        amounts: tuple[Any, Any],
        *,
        tail: tuple[Tail, Tail] = ("upper", "upper"),
        names: tuple[str, str] = ("rain", "heat"),
    ) -> None:
        self.triggers = triggers
        self.amounts = amounts
        self._tail = tail
        self._names = names

    @property
    def tail(self) -> tuple[Tail, Tail]:
        return self._tail

    @property
    def index_names(self) -> tuple[str, str]:
        return self._names

    def _amount_keys(self) -> tuple[str, str]:
        return (f"{self._names[0]}.amount", f"{self._names[1]}.amount")

    def _trigger_keys(self) -> tuple[str, str]:
        return (f"{self._names[0]}.trigger", f"{self._names[1]}.trigger")

    def parameters(self) -> dict[str, float | Free]:
        return {
            **dict(zip(self._trigger_keys(), self.triggers, strict=True)),
            **dict(zip(self._amount_keys(), self.amounts, strict=True)),
        }

    def with_values(self, values: Mapping[str, float]) -> Self:
        merged = {**self.parameters(), **_resolve_values(self.parameters(), values)}
        return type(self)(
            (merged[self._trigger_keys()[0]], merged[self._trigger_keys()[1]]),
            (merged[self._amount_keys()[0]], merged[self._amount_keys()[1]]),
            tail=self._tail,
            names=self._names,
        )

    def amount_names(self) -> tuple[str, ...]:
        return self._amount_keys()

    def basis(self, index: Any) -> Any:
        columns = []
        for column, (trigger, tail) in enumerate(
            zip(self.triggers, self._tail, strict=True)
        ):
            sign = 1.0 if tail == "upper" else -1.0
            columns.append(sign * index[:, column] >= sign * trigger)
        return np.column_stack(columns).astype(np.float64)

    def amount_constraints(self, limit: float | None = None) -> LinearConstraints:
        lower = [
            amount.lower if isinstance(amount, Free) else amount
            for amount in self.amounts
        ]
        upper = [
            (math.inf if amount.upper is None else amount.upper)
            if isinstance(amount, Free)
            else amount
            for amount in self.amounts
        ]
        if limit is not None:
            upper = [min(value, limit) for value in upper]
        return LinearConstraints(
            a_ub=np.zeros((0, 2)),
            b_ub=np.zeros(0),
            lower=np.array(lower),
            upper=np.array(upper),
        )

    def evaluate(self, index: Any) -> Any:
        return self.basis(index) @ np.array(self.amounts, dtype=np.float64)

    def to_document(self) -> dict[str, Any]:
        return {
            "amounts": [_document_value(amount) for amount in self.amounts],
            "kind": self.kind,
            "names": list(self._names),
            "tail": list(self._tail),
            "triggers": [_document_value(trigger) for trigger in self.triggers],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        amounts = [_read_value(value) for value in document["amounts"]]
        triggers = [_read_value(value) for value in document["triggers"]]
        return cls(
            (triggers[0], triggers[1]),
            (amounts[0], amounts[1]),
            tail=(document["tail"][0], document["tail"][1]),
            names=(document["names"][0], document["names"][1]),
        )


class NoAmounts(TwoTriggers):
    """Linear in form, but answers `amount_names()` with none."""

    kind: ClassVar[str] = "test_no_amounts"

    def amount_names(self) -> tuple[str, ...]:
        return ()

    def evaluate(self, index: Any) -> Any:
        return self.basis(index) @ np.array(self.amounts, dtype=np.float64)


class Scaled:
    """A vector family that isn't linear in its amounts: `min(scale * sum, cap)`."""

    kind: ClassVar[str] = "test_scaled"

    def __init__(self, scale: Any) -> None:
        self.scale = scale

    @property
    def tail(self) -> tuple[Tail, Tail]:
        return ("upper", "upper")

    @property
    def index_names(self) -> tuple[str, str]:
        return ("rain", "heat")

    def parameters(self) -> dict[str, float | Free]:
        return {"scale": self.scale}

    def with_values(self, values: Mapping[str, float]) -> Self:
        return type(self)(values.get("scale", self.scale))

    def evaluate(self, index: Any) -> Any:
        return np.minimum(self.scale * index.sum(axis=1), 12.0)

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.kind, "scale": _document_value(self.scale)}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(_read_value(document["scale"]))


def _two(payout: PayoutFunction, *, index: CompositeIndex = TWO) -> Cover:
    return Cover(index, payout, payout_unit="EUR", label="two")


def _template() -> Cover:
    return _two(TwoTriggers((50.0, 38.0), (Free(), Free())))


PERIOD = np.arange(2000, 2008)
RAIN_VALUES = np.array([10.0, 60.0, 20.0, 80.0, 30.0, 15.0, 70.0, 25.0])
HEAT_VALUES = np.array([30.0, 25.0, 40.0, 20.0, 35.0, 42.0, 28.0, 38.0])
RAIN_LOSS = np.array([0.0, 10.0, 0.0, 6.0, 0.0, 0.0, 8.0, 0.0])
HEAT_LOSS = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 6.0])
COMBINED = RAIN_LOSS + HEAT_LOSS + np.array([1.0, 0, 0, 0, 3.0, 0, 0, 0])
"""Rain pays in years 1, 3 and 6 and heat in 2, 5 and 7, so the columns are disjoint."""


def _component(
    values: Any, unit: str = "mm/day", *, own_loss: Any = None, **rest: Any
) -> IndexComponent:
    fields = {"tail": "upper", "hazard": None, **rest}
    return IndexComponent(values, unit=unit, own_loss=own_loss, **fields)


def _provenance(**overrides: Any) -> SampleProvenance:
    fields: dict[str, Any] = {
        "source_kind": "table",
        "sources": (),
        "semantics_source": "caller",
        "hazard_dependence": {"kind": "data"},
        "combine": {"kind": "table"},
    }
    fields.update(overrides)
    return SampleProvenance(**fields)


def _fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "asset_ids": ("site",),
        "period": PERIOD,
        "probability": np.full(8, 1 / 8),
        "index": RAIN_VALUES,
        "loss": COMBINED,
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": _provenance(),
        "index_components": {
            "rain": _component(RAIN_VALUES, own_loss=RAIN_LOSS),
            "heat": _component(HEAT_VALUES, "degC", own_loss=HEAT_LOSS),
        },
    }
    fields.update(overrides)
    return fields


UNTYPED_SAMPLE: Any = OutcomeSample


def _sample(**overrides: Any) -> OutcomeSample:
    sample: OutcomeSample = UNTYPED_SAMPLE(**_fields(**overrides))
    return sample


TABLE = {
    "year": [int(year) for year in PERIOD],
    "site": ["site"] * 8,
    "rain_index": RAIN_VALUES.tolist(),
    "heat_index": HEAT_VALUES.tolist(),
    "rain_loss": RAIN_LOSS.tolist(),
    "heat_loss": HEAT_LOSS.tolist(),
    "loss": COMBINED.tolist(),
}


def _from_table(**overrides: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "period": "year",
        "asset_id": "site",
        "index": {"rain": "rain_index", "heat": "heat_index"},
        "loss": "loss",
        "own_loss": {"rain": "rain_loss", "heat": "heat_loss"},
        "semantics": "annual",
        "index_unit": {"rain": "mm/day", "heat": "degC"},
        "loss_unit": "EUR",
        "tails": {"rain": "upper", "heat": "upper"},
        "source": "a user table",
    }
    arguments.update(overrides)
    table = arguments.pop("table", TABLE)
    sample: OutcomeSample = OutcomeSample.from_table(table, **arguments)
    return sample


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    """Register the test families for one test, leaving the registry as it was."""
    families: tuple[type[Any], ...] = (TwoTriggers, NoAmounts, Scaled)
    for family in families:
        register_payout(family)
    yield
    for family in families:
        del _REGISTRY[family.kind]


# -- CompositeIndex ----------------------------------------------------------


def test_a_composite_index_names_its_components_in_order() -> None:
    assert TWO.names == ("rain", "heat")
    assert TWO.tails == ("upper", "upper")
    assert list(TWO.components) == ["rain", "heat"]
    assert TWO.components["heat"] == HEAT
    with pytest.raises(TypeError):
        TWO.components["rain"] = HEAT  # type: ignore[index]


def test_a_composite_index_document_is_a_list_in_order() -> None:
    assert TWO.to_document() == {
        "components": [
            {"index": RAIN.to_document(), "name": "rain"},
            {"index": HEAT.to_document(), "name": "heat"},
        ],
        "kind": "composite",
    }
    assert CompositeIndex.from_document(TWO.to_document()) == TWO


def test_the_order_of_the_components_is_part_of_the_terms() -> None:
    swapped = CompositeIndex({"heat": HEAT, "rain": RAIN})
    assert swapped != TWO and TWO != "rain"
    assert hash(TWO) == hash(CompositeIndex({"rain": RAIN, "heat": HEAT}))


def test_a_composite_index_pickles() -> None:
    assert pickle.loads(pickle.dumps(TWO)) == TWO


@pytest.mark.parametrize(
    ("components", "message"),
    [
        pytest.param([("rain", RAIN)], "map each", id="not a mapping"),
        pytest.param({"rain": RAIN}, "two components or more", id="one"),
        pytest.param({"a.b": RAIN, "c": HEAT}, "name", id="dot"),
        pytest.param({"a+b": RAIN, "c": HEAT}, "name", id="plus"),
        pytest.param({"a[0]": RAIN, "c": HEAT}, "name", id="bracket"),
        pytest.param({" a": RAIN, "c": HEAT}, "name", id="whitespace"),
        pytest.param({"": RAIN, "c": HEAT}, "name", id="empty"),
        pytest.param({1: RAIN, "c": HEAT}, "name", id="not a string"),
        pytest.param({"a": "rain", "c": HEAT}, "IndexDefinition", id="not an index"),
    ],
)
def test_an_invalid_composite_index_raises(components: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        CompositeIndex(components)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param([], "has the keys", id="not an object"),
        pytest.param({"kind": "composite"}, "has the keys", id="no components"),
        pytest.param({"components": [], "kind": "other"}, "composite", id="kind"),
        pytest.param({"components": {}, "kind": "composite"}, "list", id="not a list"),
        pytest.param(
            {"components": [{"name": "a"}], "kind": "composite"},
            "has the keys",
            id="a bad component",
        ),
        pytest.param(
            {
                "components": [
                    {"index": RAIN.to_document(), "name": "a"},
                    {"index": HEAT.to_document(), "name": "a"},
                ],
                "kind": "composite",
            },
            "repeats",
            id="a repeated name",
        ),
    ],
)
def test_an_invalid_composite_document_raises(document: Any, message: str) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        CompositeIndex.from_document(document)


# -- covers ------------------------------------------------------------------


COMPOSITE_JSON = (
    '{"asset_id":"site","index":{"components":[{"index":{"aggregation":'
    '"annual_maximum","hazard_name":"rain","location":{"kind":"asset"},'
    '"source":null,"tail":"upper","unit":"mm/day"},"name":"rain"},{"index":'
    '{"aggregation":"annual_maximum","hazard_name":"heat","location":{"kind":'
    '"asset"},"source":null,"tail":"upper","unit":"degC"},"name":"heat"}],'
    '"kind":"composite"},"payout":{"amounts":[1.0,2.0],"kind":"test_two_triggers",'
    '"names":["rain","heat"],"tail":["upper","upper"],"triggers":[50.0,38.0]},'
    '"payout_unit":"EUR","schema_version":"1.1",'
    '"semantics_version":"crc-parametric-cover/1"}'
)


def _resolved() -> Cover:
    return Cover(
        TWO,
        TwoTriggers((50.0, 38.0), (1.0, 2.0)),
        payout_unit="EUR",
        asset_id="site",
    )


def test_a_composite_cover_writes_its_terms_as_schema_1_1() -> None:
    cover = _resolved()
    assert cover.to_json() == COMPOSITE_JSON
    assert cover.to_document()["schema_version"] == COMPOSITE_COVER_SCHEMA_VERSION
    assert set(cover.to_document()) == {
        "asset_id",
        "index",
        "payout",
        "payout_unit",
        "schema_version",
        "semantics_version",
    }


def test_a_composite_cover_round_trips() -> None:
    cover = _resolved()
    restored = Cover.from_json(cover.to_json(), label="back")
    assert restored == cover and restored.definition_hash == cover.definition_hash
    assert restored.index_definition == TWO and restored.label == "back"
    assert pickle.loads(pickle.dumps(cover)) == cover


def test_a_composite_template_resolves() -> None:
    template = _template()
    assert template.free_parameters() == ("rain.amount", "heat.amount")
    cover = template.with_values({"rain.amount": 3.0, "heat.amount": 4.0})
    assert cover.index_definition is template.index_definition
    assert cover.label == "two"
    index = np.column_stack([RAIN_VALUES, HEAT_VALUES])
    np.testing.assert_array_equal(
        cover.payout_values(index),
        [0.0, 3.0, 4.0, 3.0, 0.0, 4.0, 3.0, 4.0],
    )


def test_the_order_of_the_components_changes_the_hash() -> None:
    swapped = Cover(
        CompositeIndex({"heat": HEAT, "rain": RAIN}),
        TwoTriggers((38.0, 50.0), (2.0, 1.0), names=("heat", "rain")),
        payout_unit="EUR",
        asset_id="site",
    )
    assert swapped.definition_hash != _resolved().definition_hash


@pytest.mark.parametrize(
    ("index", "payout", "message"),
    [
        pytest.param(
            TWO,
            TwoTriggers((1.0, 1.0), (1.0, 1.0), names=("heat", "rain")),
            "components",
            id="names in another order",
        ),
        pytest.param(
            TWO,
            TwoTriggers((1.0, 1.0), (1.0, 1.0), names=("rain", "wind")),
            "components",
            id="another name",
        ),
        pytest.param(
            TWO,
            TwoTriggers((1.0, 1.0), (1.0, 1.0), tail=("upper", "lower")),
            "tails",
            id="another tail",
        ),
        pytest.param(
            TWO, LinearPayout(1.0, 2.0, 1.0), "components", id="a one-index payout"
        ),
        pytest.param(
            RAIN,
            TwoTriggers((1.0, 1.0), (1.0, 1.0)),
            "tail",
            id="a vector payout on one index",
        ),
    ],
)
def test_the_gate_of_a_composite_cover(
    index: IndexDefinition | CompositeIndex, payout: PayoutFunction, message: str
) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        Cover(index, payout, payout_unit="EUR")


def test_a_payout_with_a_single_tail_cannot_go_on_a_composite() -> None:
    class SingleTail(TwoTriggers):
        @property
        def tail(self) -> Any:
            return "upper"

    with pytest.raises(CoverDefinitionError, match="tuple of one tail"):
        Cover(TWO, SingleTail((1.0, 1.0), (1.0, 1.0)), payout_unit="EUR")


def test_a_cover_needs_one_kind_of_index() -> None:
    with pytest.raises(CoverDefinitionError, match="IndexDefinition or a Composite"):
        Cover("rain", LinearPayout(1.0, 2.0, 1.0), payout_unit="EUR")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param({"schema_version": "1.0"}, "unknown fields", id="a 1.0 reader"),
        pytest.param({"schema_version": "2.0"}, "schema_version", id="a later one"),
    ],
)
def test_the_schema_decides_how_the_index_is_read(
    edit: dict[str, Any], message: str
) -> None:
    with pytest.raises(CoverDefinitionError, match=message):
        Cover.from_document({**_resolved().to_document(), **edit})


def test_a_composite_schema_refuses_a_single_index() -> None:
    plain = Cover(RAIN, LinearPayout(1.0, 2.0, 1.0), payout_unit="EUR").to_document()
    with pytest.raises(CoverDefinitionError, match="has the keys"):
        Cover.from_document({**plain, "schema_version": "1.1"})


def test_the_index_of_a_composite_cover_is_its_definition() -> None:
    cover = _resolved()
    assert cover.index_definition == TWO
    with pytest.raises(CoverDefinitionError, match="index_definition"):
        _ = cover.index


def test_a_plan_refuses_a_composite_cover_for_now() -> None:
    """The plan reads one index; until it reads components it says so."""
    untyped: Any = ParametricPlan
    plan = untyped.for_annual_extremes(
        pa.table({"year": [2000], "value": [1.0], "cell": ["a"]}),
        pa.table(
            {
                "asset_id": ["a"],
                "longitude": [7.0],
                "latitude": [51.0],
                "value_eur": [1.0],
            }
        ),
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
        unit="mm/day",
        provenance=SourceProvenance(
            provider="example",
            dataset="synthetic",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic",
        ),
        value_column="value_eur",
        value_unit="EUR",
        seed=42,
    )
    with pytest.raises(CoverDefinitionError, match="ParametricPlan takes"):
        plan.covers(_resolved())


def test_a_one_index_cover_keeps_its_index() -> None:
    cover = Cover(RAIN, LinearPayout(1.0, 2.0, 1.0), payout_unit="EUR")
    assert cover.index is RAIN and cover.index_definition is RAIN
    assert cover.to_document()["schema_version"] == COVER_SCHEMA_VERSION


def test_payout_values_of_a_composite_cover_read_columns_in_order() -> None:
    cover = _resolved()
    index = np.array([[60.0, 10.0], [10.0, 40.0], [60.0, 40.0]])
    np.testing.assert_array_equal(cover.payout_values(index), [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(np.zeros(3), id="1-D"),
        pytest.param(np.zeros((3, 3)), id="too many columns"),
        pytest.param(np.array([[np.nan, 1.0]]), id="not finite"),
    ],
)
def test_payout_values_of_a_composite_cover_check_the_shape(index: Any) -> None:
    with pytest.raises(SampleError):
        _resolved().payout_values(index)


def test_a_composite_template_does_not_pay() -> None:
    with pytest.raises(UnresolvedParameterError):
        _template().payout_values(np.zeros((1, 2)))


def test_a_payout_of_the_wrong_shape_is_refused() -> None:
    class Wide(TwoTriggers):
        def evaluate(self, index: Any) -> Any:
            return np.zeros(index.shape)

    with pytest.raises(CoverDefinitionError, match="returned shape"):
        _two(Wide((1.0, 1.0), (1.0, 1.0))).payout_values(np.zeros((2, 2)))


def test_a_vector_family_passes_the_protocols() -> None:
    assert isinstance(TwoTriggers((1.0, 1.0), (1.0, 1.0)), PayoutFunction)
    assert isinstance(TwoTriggers((1.0, 1.0), (1.0, 1.0)), LinearInAmounts)


# -- one-hazard samples and covers do not change ------------------------------


_DIRECT = OutcomeSample(
    asset_ids=("a",),
    period=np.array([2000, 2001, 2002, 2003]),
    probability=np.full(4, 0.25),
    index=np.array([[1.0], [2.0], [3.0], [4.0]]),
    loss=np.array([[0.0], [1.0], [0.5], [2.0]]),
    hazard=None,
    index_unit="mm/day",
    loss_unit="EUR",
    semantics="annual",
    provenance=SampleProvenance(
        source_kind="table", sources=(), semantics_source="caller"
    ),
)
_DIRECT_HASH = "sha256:6cd8f74c885b9bd8883ea0b0e125295b1b3afd1788c74d8eac8ac02db026424a"
_TABLE_HASH = "sha256:1b97ebc0fa5ff2739f7ee060731aa3e318f7ee763d54261d39242ba944a4cae8"
_TABLE_LINEAGE = (
    "sha256:6e4b00652af56a29a6c230f5d6df6ec4888a0aa02b95fe353e709879da07656c"
)


def test_a_one_hazard_sample_keeps_its_hash_and_table() -> None:
    """The hashes and the metadata text are those of the code before components."""
    table = _DIRECT.to_table()
    assert _DIRECT.content_hash == _DIRECT_HASH
    assert table.schema.names == [
        "scenario",
        "period",
        "asset_id",
        "probability",
        "index",
        "loss",
        "hazard",
    ]
    assert table.schema.metadata[SAMPLE_METADATA_KEY.encode()].decode() == (
        '{"asset_ids":["a"],"content_hash":"' + _DIRECT_HASH + '",'
        '"index_unit":"mm/day","loss_unit":"EUR","provenance":{"basis_risk":null,'
        '"dependence":null,"details":{},"licences":[],"lineage":null,'
        '"loss_model":null,"notes":[],"parent":null,"parts":null,"pool":null,'
        '"probability_semantics":null,"rng":[],"selection":[],'
        '"semantics_source":"caller","source_kind":"table","sources":[],'
        '"trend":null},"scenario":"baseline","schema_version":"1.0",'
        '"semantics":"annual","semantics_version":"crc-parametric-sample/1"}'
    )
    read = OutcomeSample.from_written_table(table)
    assert read.content_hash == _DIRECT_HASH and read.index_components is None


def test_a_table_of_one_hazard_keeps_its_hash_and_lineage() -> None:
    sample = OutcomeSample.from_table(
        {
            "year": [2001, 2000, 2002, 2003],
            "site": ["a"] * 4,
            "x": [3.0, 1.0, 2.0, 1.5],
            "loss": [1.0, 0.5, 0.0, 2.0],
            "h": [1.0, 2.0, 3.0, 4.0],
        },
        period="year",
        asset_id="site",
        index="x",
        loss="loss",
        hazard="h",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
        source="my table",
    )
    assert sample.content_hash == _TABLE_HASH and sample.lineage == _TABLE_LINEAGE
    assert sample.provenance.hazard_dependence is None
    assert "hazard_dependence" not in sample.provenance.to_document()


def test_a_one_hazard_setup_entry_keeps_its_keys() -> None:
    cover = Cover(RAIN, LinearPayout(1.0, 4.0, 2.0), payout_unit="EUR")
    table = compare([cover], _DIRECT, split="in_sample")
    entry = EvaluationMetadata.from_table(table).setup.samples[0]
    assert sorted(entry) == [
        "asset_ids",
        "basis_risk",
        "dependence",
        "details",
        "draws",
        "index_unit",
        "lineage",
        "loss_model",
        "loss_unit",
        "notes",
        "parent",
        "parts",
        "periods",
        "pool",
        "probability_semantics",
        "roles",
        "sample_hash",
        "scenario",
        "selection",
        "semantics",
        "semantics_source",
        "source_kind",
        "trend",
    ]
    assert "Combined loss" not in setup_text(table)


# -- IndexComponent ----------------------------------------------------------


def test_a_component_reads_one_column() -> None:
    component = IndexComponent(
        np.array([1.0, 2.0]),
        unit="mm/day",
        tail="lower",
        hazard=np.array([[3.0], [4.0]]),
        own_loss=np.array([0.0, 1.0]),
    )
    assert component.values.shape == (2, 1)
    assert component.hazard is not None and component.hazard.shape == (2, 1)
    assert component.own_loss is not None and component.own_loss.shape == (2, 1)
    with pytest.raises(ValueError, match="read-only"):
        component.values[0, 0] = 9.0


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"values": np.zeros((2, 2))}, "shape", id="two columns"),
        pytest.param({"values": np.zeros(0)}, "shape", id="empty"),
        pytest.param({"values": [np.nan, 1.0]}, "finite", id="not finite"),
        pytest.param({"values": ["a", "b"]}, "numeric", id="not numeric"),
        pytest.param({"unit": ""}, "unit", id="no unit"),
        pytest.param({"tail": "both"}, "tail", id="a bad tail"),
        pytest.param({"hazard": [1.0]}, "shape", id="hazard of another length"),
        pytest.param({"own_loss": [1.0]}, "shape", id="own loss of another length"),
        pytest.param({"own_loss": [-1.0, 1.0]}, "non-negative", id="negative"),
    ],
)
def test_an_invalid_component_raises(fields: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {
        "values": [1.0, 2.0],
        "unit": "mm/day",
        "tail": "upper",
        **fields,
    }
    values = arguments.pop("values")
    with pytest.raises(SampleError, match=message):
        IndexComponent(values, **arguments)


# -- the sample --------------------------------------------------------------


def test_the_first_component_is_the_primary() -> None:
    sample = _sample()
    assert sample.index_components is not None
    assert list(sample.index_components) == ["rain", "heat"]
    np.testing.assert_array_equal(sample.index[:, 0], RAIN_VALUES)
    assert sample.index_unit == "mm/day"
    with pytest.raises(TypeError):
        sample.index_components["rain"] = None  # type: ignore[index]


def _other_components(**overrides: Any) -> dict[str, IndexComponent]:
    return {
        "rain": _component(RAIN_VALUES, own_loss=RAIN_LOSS),
        "heat": _component(HEAT_VALUES, "degC", own_loss=HEAT_LOSS),
        **overrides,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"index_components": []}, "must map", id="not a mapping"),
        pytest.param(
            {"index_components": {"rain": _component(RAIN_VALUES)}},
            "two components or more",
            id="one",
        ),
        pytest.param(
            {"index_components": _other_components(**{"a.b": _component(RAIN_VALUES)})},
            "name",
            id="a bad name",
        ),
        pytest.param(
            {"index_components": _other_components(heat=None)},
            "IndexComponent",
            id="not a component",
        ),
        pytest.param(
            {"index_components": _other_components(heat=_component(np.zeros(3)))},
            "rows",
            id="another length",
        ),
        pytest.param({"index": HEAT_VALUES}, "first component", id="the index differs"),
        pytest.param({"index_unit": "m"}, "first component", id="the unit differs"),
        pytest.param({"hazard": RAIN_VALUES}, "first component", id="a hazard extra"),
        pytest.param(
            {"provenance": _provenance(hazard_dependence=None)},
            "dependence between",
            id="dependence not stated",
        ),
        pytest.param(
            {"provenance": _provenance(components={"rain": {}})},
            "components are for",
            id="other documents",
        ),
    ],
)
def test_an_invalid_sample_with_components_raises(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _sample(**overrides)


def test_a_sample_with_components_holds_one_asset() -> None:
    with pytest.raises(SampleError, match="exactly one asset"):
        _sample(
            asset_ids=("a", "b"),
            index=np.column_stack([RAIN_VALUES, RAIN_VALUES]),
            loss=np.column_stack([COMBINED, COMBINED]),
            provenance=_provenance(dependence={"kind": "data"}),
        )


def test_a_primary_hazard_must_match() -> None:
    components = _other_components(
        rain=_component(RAIN_VALUES, own_loss=RAIN_LOSS, hazard=RAIN_VALUES + 1)
    )
    sample = _sample(index_components=components, hazard=RAIN_VALUES + 1)
    assert sample.hazard is not None
    with pytest.raises(SampleError, match="first component"):
        _sample(index_components=components, hazard=RAIN_VALUES)
    with pytest.raises(SampleError, match="first component"):
        _sample(index_components=components)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"hazard_dependence": {"kind": "data"}}, id="dependence"),
        pytest.param({"combine": {"kind": "table"}}, id="combine"),
        pytest.param({"components": {"a": {}}}, id="components"),
    ],
)
def test_a_one_hazard_sample_states_nothing_about_hazards(
    fields: dict[str, Any],
) -> None:
    provenance = SampleProvenance(
        source_kind="table", sources=(), semantics_source="caller", **fields
    )
    with pytest.raises(SampleError, match="without index components"):
        _sample(index_components=None, provenance=provenance)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"hazard_dependence": {"model": "x"}}, "kind", id="no kind"),
        pytest.param({"combine": {"rule": "x"}}, "kind", id="a rule with no kind"),
        pytest.param({"components": {"a": 1}}, "provenance document", id="not a doc"),
        pytest.param({"combine": 1}, "mapping", id="not a mapping"),
    ],
)
def test_invalid_provenance_of_components_raises(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _provenance(**fields)


def test_the_hash_reads_every_component_field() -> None:
    base = _sample().content_hash
    assert (
        base
        != _sample(index_components=None, provenance=_DIRECT.provenance).content_hash
    )
    changed: list[OutcomeSample] = [
        _sample(
            index_components=_other_components(
                heat=_component(HEAT_VALUES + 1, "degC", own_loss=HEAT_LOSS)
            )
        ),
        _sample(
            index_components=_other_components(
                heat=_component(HEAT_VALUES, "K", own_loss=HEAT_LOSS)
            )
        ),
        _sample(
            index_components=_other_components(
                heat=_component(HEAT_VALUES, "degC", own_loss=HEAT_LOSS, tail="lower")
            )
        ),
        _sample(
            index_components=_other_components(heat=_component(HEAT_VALUES, "degC"))
        ),
        _sample(
            index_components=_other_components(
                heat=_component(
                    HEAT_VALUES, "degC", own_loss=HEAT_LOSS, hazard=HEAT_VALUES + 1
                )
            )
        ),
    ]
    assert len({base, *(sample.content_hash for sample in changed)}) == 6


def test_the_table_of_a_sample_with_components() -> None:
    sample = _sample(
        index_components=_other_components(
            heat=_component(HEAT_VALUES, "degC", tail="lower")
        )
    )
    table = sample.to_table()
    assert table.schema.names[-6:] == [
        "index[rain]",
        "hazard[rain]",
        "own_loss[rain]",
        "index[heat]",
        "hazard[heat]",
        "own_loss[heat]",
    ]
    assert table.column("index[heat]").to_pylist() == HEAT_VALUES.tolist()
    assert table.column("own_loss[heat]").null_count == 8
    metadata = json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])
    assert metadata["schema_version"] == SAMPLE_COMPONENTS_SCHEMA_VERSION
    assert metadata["components"] == [
        {"name": "rain", "tail": "upper", "unit": "mm/day"},
        {"name": "heat", "tail": "lower", "unit": "degC"},
    ]
    assert SAMPLE_SCHEMA_VERSION != SAMPLE_COMPONENTS_SCHEMA_VERSION


def test_a_sample_with_components_round_trips(tmp_path: Path) -> None:
    sample = _from_table()
    path = tmp_path / "sample.parquet"
    pq.write_table(sample.to_table(), path)
    read = OutcomeSample.from_written_table(pq.read_table(path))
    assert read.content_hash == sample.content_hash
    assert read.provenance.to_document() == sample.provenance.to_document()
    assert read.index_components is not None and sample.index_components is not None
    assert list(read.index_components) == list(sample.index_components)
    for name, component in read.index_components.items():
        own = sample.index_components[name]
        assert (component.unit, component.tail) == (own.unit, own.tail)
        np.testing.assert_array_equal(component.values, own.values)
        assert component.own_loss is not None and own.own_loss is not None
        np.testing.assert_array_equal(component.own_loss, own.own_loss)
    assert read.lineage == sample.lineage


def _metadata_edited(table: Any, **changes: Any) -> Any:
    document = json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])
    for key, value in changes.items():
        if value is None:
            del document[key]
        else:
            document[key] = value
    return table.replace_schema_metadata(
        {SAMPLE_METADATA_KEY.encode(): json.dumps(document).encode()}
    )


def test_a_reader_of_1_0_refuses_a_written_sample_with_components() -> None:
    table = _sample().to_table()
    with pytest.raises(SampleError, match="unknown keys"):
        OutcomeSample.from_written_table(
            _metadata_edited(table, schema_version=SAMPLE_SCHEMA_VERSION)
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"schema_version": "2.0"}, "schema_version", id="a later one"),
        pytest.param({"components": None}, "lacks", id="no components"),
        pytest.param({"components": [{"name": "a"}]}, "name, a unit", id="bad"),
        pytest.param({"components": "rain"}, "must be a list", id="not a list"),
        pytest.param(
            {"semantics_version": "crc-parametric-sample/2"},
            "semantics_version",
            id="semantics",
        ),
    ],
)
def test_a_written_sample_with_invalid_metadata_raises(
    changes: dict[str, Any], message: str
) -> None:
    table = _sample().to_table()
    with pytest.raises(SampleError, match=message):
        OutcomeSample.from_written_table(_metadata_edited(table, **changes))


def test_a_written_sample_with_a_changed_component_is_refused() -> None:
    table = _sample().to_table()
    position = table.schema.get_field_index("index[heat]")
    edited = table.set_column(
        position,
        "index[heat]",
        pa.array((HEAT_VALUES + 1).tolist(), pa.float64()),
    )
    with pytest.raises(SampleError, match="content hash"):
        OutcomeSample.from_written_table(edited)


def test_a_written_component_is_null_in_every_row_or_in_none() -> None:
    table = _sample().to_table()
    position = table.schema.get_field_index("own_loss[heat]")
    edited = table.set_column(
        position,
        "own_loss[heat]",
        pa.array([None, *HEAT_LOSS[1:].tolist()], pa.float64()),
    )
    with pytest.raises(SampleError, match="null in 1 of 8"):
        OutcomeSample.from_written_table(edited)


def test_join_refuses_a_sample_with_components() -> None:
    other = dataclass_with_asset(_sample(), "other")
    with pytest.raises(SampleError, match="index components"):
        OutcomeSample.join([_sample(), other])


def dataclass_with_asset(sample: OutcomeSample, asset: str) -> OutcomeSample:
    return dataclasses.replace(sample, asset_ids=(asset,))


def test_asset_returns_the_sample_itself() -> None:
    sample = _sample()
    assert sample.asset("site") is sample


def test_a_selection_keeps_every_array_and_the_lineage() -> None:
    sample = _from_table()
    chosen = sample.select([2002, 2005])
    assert chosen.lineage == sample.lineage
    assert chosen.index_components is not None
    np.testing.assert_array_equal(
        chosen.index_components["heat"].values[:, 0], HEAT_VALUES[[2, 5]]
    )
    own = chosen.index_components["heat"].own_loss
    assert own is not None
    np.testing.assert_array_equal(own[:, 0], HEAT_LOSS[[2, 5]])
    np.testing.assert_array_equal(chosen.index[:, 0], RAIN_VALUES[[2, 5]])


def test_holdout_and_kfold_act_on_every_array() -> None:
    sample = _from_table()
    design_block, held_out = holdout(sample, 0.75)
    assert design_block.index_components is not None
    assert held_out.index_components is not None
    np.testing.assert_array_equal(
        held_out.index_components["heat"].values[:, 0], HEAT_VALUES[6:]
    )
    assert design_block.lineage == held_out.lineage == sample.lineage
    folds = year_block_kfold(sample, 4)
    assert all(train.index_components is not None for train, _ in folds)


def test_a_split_refuses_an_empirical_copula_across_hazards() -> None:
    sample = _sample(
        provenance=_provenance(
            hazard_dependence={"kind": "empirical_copula", "k": 1, "m": 8}
        )
    )
    with pytest.raises(SampleError, match="EmpiricalCopula"):
        holdout(sample)
    with pytest.raises(SampleError, match="EmpiricalCopula"):
        year_block_kfold(sample, 2)
    with pytest.raises(SampleError, match="EmpiricalCopula"):
        holdout(sample.component("rain"))


# -- component() -------------------------------------------------------------


def test_a_component_holds_its_own_loss_by_default() -> None:
    sample = _from_table()
    view = sample.component("heat")
    np.testing.assert_array_equal(view.index[:, 0], HEAT_VALUES)
    np.testing.assert_array_equal(view.loss[:, 0], HEAT_LOSS)
    np.testing.assert_array_equal(view.period, sample.period)
    np.testing.assert_array_equal(view.probability, sample.probability)
    assert view.index_unit == "degC" and view.index_components is None
    assert view.hazard is None


def test_a_component_can_keep_the_combined_loss() -> None:
    view = _from_table().component("heat", loss="combined")
    np.testing.assert_array_equal(view.loss[:, 0], COMBINED)


def test_a_component_keeps_its_own_hazard() -> None:
    components = _other_components(
        heat=_component(HEAT_VALUES, "degC", hazard=HEAT_VALUES + 2, own_loss=HEAT_LOSS)
    )
    view = _sample(index_components=components).component("heat")
    assert view.hazard is not None
    np.testing.assert_array_equal(view.hazard[:, 0], HEAT_VALUES + 2)


def test_a_component_records_the_step_and_keeps_the_lineage() -> None:
    sample = _from_table()
    view = sample.component("heat", loss="combined")
    assert view.lineage == sample.lineage
    assert view.provenance.parent == sample.content_hash
    assert view.provenance.selection[-1] == {
        "kind": "component",
        "name": "heat",
        "loss": "combined",
    }
    document = view.provenance.to_document()
    assert document["sources"][0]["component"] == "heat"
    assert document["sources"][0]["columns"]["index"] == "heat_index"
    assert "hazard_dependence" not in document and "combine" not in document


def test_a_component_commutes_with_a_split() -> None:
    sample = _from_table()
    design_block, _ = holdout(sample, 0.75)
    own_first = design_block.component("heat")
    split_first, _ = holdout(sample.component("heat"), 0.75)
    assert own_first.content_hash == split_first.content_hash
    assert own_first.provenance.selection[-1]["kind"] == "holdout"
    assert split_first.provenance.selection[-1]["kind"] == "holdout"


def test_a_component_without_documents_uses_the_samples_provenance() -> None:
    view = _sample().component("rain")
    assert view.provenance.source_kind == "table"
    assert view.provenance.hazard_dependence is None
    assert view.provenance.selection == (
        {"kind": "component", "name": "rain", "loss": "own"},
    )


def test_a_component_with_no_impact_has_no_own_loss() -> None:
    sample = _sample(
        index_components=_other_components(heat=_component(HEAT_VALUES, "degC"))
    )
    with pytest.raises(SampleError, match="loss='combined'"):
        sample.component("heat")
    assert sample.component("heat", loss="combined").loss.shape == (8, 1)


@pytest.mark.parametrize(
    ("sample", "name", "loss", "message"),
    [
        pytest.param(_DIRECT, "rain", "own", "one hazard", id="one-hazard sample"),
        pytest.param(_sample(), "wind", "own", "no component", id="unknown name"),
        pytest.param(_sample(), "rain", "total", "loss must be", id="unknown loss"),
    ],
)
def test_a_bad_component_call_raises(
    sample: OutcomeSample, name: str, loss: Any, message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        sample.component(name, loss=loss)


# -- a user's table ----------------------------------------------------------


def test_a_table_with_several_index_columns() -> None:
    sample = _from_table()
    assert sample.index_components is not None
    assert list(sample.index_components) == ["rain", "heat"]
    np.testing.assert_array_equal(sample.loss[:, 0], COMBINED)
    provenance = sample.provenance
    assert provenance.combine is not None and provenance.combine["kind"] == "table"
    assert provenance.hazard_dependence is not None
    assert provenance.hazard_dependence["kind"] == "data"
    assert provenance.components is not None
    assert set(provenance.components) == {"rain", "heat"}
    columns = provenance.sources[0]["columns"]
    assert columns["index"] == {"rain": "rain_index", "heat": "heat_index"}
    assert sample.provenance.lineage is not None


def test_the_rows_of_a_table_are_sorted_by_period() -> None:
    shuffled = {key: list(reversed(values)) for key, values in TABLE.items()}
    sample = _from_table(table=shuffled)
    assert sample.content_hash == _from_table().content_hash


def test_a_table_states_probabilities_hazards_and_no_own_loss() -> None:
    table = {
        **TABLE,
        "p": [0.125] * 8,
        "rain_hazard": (RAIN_VALUES + 1).tolist(),
    }
    sample = _from_table(
        table=table,
        probability="p",
        hazard={"rain": "rain_hazard"},
        own_loss={"rain": "rain_loss"},
    )
    assert sample.index_components is not None
    assert sample.hazard is not None
    np.testing.assert_array_equal(sample.hazard[:, 0], RAIN_VALUES + 1)
    assert sample.index_components["heat"].hazard is None
    assert sample.index_components["heat"].own_loss is None
    assert sample.provenance.details["probability"] == "column"


def test_a_table_can_leave_the_own_losses_out() -> None:
    sample = _from_table(own_loss=None)
    assert sample.index_components is not None
    assert sample.index_components["heat"].own_loss is None


def test_a_comonotone_component_warns_and_is_noted() -> None:
    table = {**TABLE, "loss": (RAIN_VALUES * 2).tolist()}
    with pytest.warns(ComonotoneIndexWarning, match="components rain"):
        sample = _from_table(table=table, own_loss=None)
    assert sample.provenance.notes == (
        "Component rain: The table's loss is a monotone function of its index: "
        "index and loss are comonotone.",
    )
    assert len(sample.component("rain", loss="combined").provenance.notes) == 1


def test_a_comonotone_own_loss_warns() -> None:
    table = {**TABLE, "heat_loss": (HEAT_VALUES * 2).tolist()}
    with pytest.warns(ComonotoneIndexWarning, match="components heat"):
        _from_table(table=table)


def test_a_table_with_noise_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _from_table()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"index": {"rain": "rain_index"}}, "two components", id="one"),
        pytest.param(
            {"index": {"a.b": "rain_index", "heat": "heat_index"}}, "name", id="name"
        ),
        pytest.param({"tails": None}, "tails must be a mapping", id="no tails"),
        pytest.param(
            {"tails": {"rain": "upper"}}, "missing", id="tails of one component"
        ),
        pytest.param({"index_unit": "mm/day"}, "index_unit must be", id="one unit"),
        pytest.param(
            {"index_unit": {"rain": "mm/day"}}, "missing", id="units of one component"
        ),
        pytest.param({"hazard": "rain_index"}, "hazard must be", id="a hazard column"),
        pytest.param({"own_loss": {"wind": "loss"}}, "unknown", id="unknown own loss"),
        pytest.param(
            {"tails": {"rain": "upper", "heat": "x"}}, "tail", id="a bad tail"
        ),
        pytest.param(
            {"index": {"rain": "nope", "heat": "heat_index"}}, "no column", id="col"
        ),
        pytest.param(
            {"table": {**TABLE, "site": ["a", "b"] * 4}},
            "exactly one asset",
            id="two assets",
        ),
        pytest.param(
            {"table": {**TABLE, "year": [2000] * 8}}, "more than once", id="rep"
        ),
        pytest.param(
            {
                "table": {key: [] for key in TABLE},
                "tails": {"rain": "upper", "heat": "upper"},
            },
            "no rows|must hold",
            id="no rows",
        ),
    ],
)
def test_an_invalid_table_with_several_index_columns_raises(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _from_table(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"tails": {"x": "upper"}}, "tails and own_loss", id="tails"),
        pytest.param({"own_loss": {"x": "loss"}}, "tails and own_loss", id="own loss"),
        pytest.param({"hazard": {"x": "loss"}}, "hazard and index_unit", id="hazard"),
        pytest.param({"index_unit": {"x": "m"}}, "hazard and index_unit", id="unit"),
    ],
)
def test_one_index_column_refuses_the_arguments_of_several(
    overrides: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {
        "period": "year",
        "asset_id": "site",
        "index": "rain_index",
        "loss": "loss",
        "semantics": "annual",
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        **overrides,
    }
    with pytest.raises(SampleError, match=message):
        OutcomeSample.from_table(TABLE, **arguments)


# -- design: what a cover reads ----------------------------------------------


def test_a_registered_family_designs_on_the_qp_route() -> None:
    """Disjoint triggers make each amount the mean loss of its own years."""
    result = design(_template(), _sample(), MSE())
    assert result.route == "qp" and result.amounts == "qp"
    assert result.parameters["rain.amount"] == pytest.approx(8.0, abs=1e-6)
    assert result.parameters["heat.amount"] == pytest.approx(4.0, abs=1e-6)
    assert result.unique is True


def test_a_registered_family_designs_on_the_lp_route() -> None:
    """Under a budget of 3, a unit of premium buys one unit of covered loss first."""
    result = design(
        _template(), _sample(), ExpectedUncovered(), budget=Budget(amount=3)
    )
    assert result.route == "lp" and result.solver is not None
    assert result.parameters["rain.amount"] == pytest.approx(6.0, abs=1e-6)
    assert result.parameters["heat.amount"] == pytest.approx(2.0, abs=1e-6)
    assert result.objective_value == pytest.approx(2.0, abs=1e-6)
    assert result.premium == pytest.approx(3.0, abs=1e-6)


def test_a_weighted_cvar_takes_the_lp_route_on_a_vector_family() -> None:
    result = design(
        _template(), _sample(), WeightedCVaR(0.5, 0.75), budget=Budget(amount=3)
    )
    assert result.route == "lp"
    assert result.premium <= 3.0 + 1e-6


@pytest.mark.parametrize(
    ("objective", "budget"),
    [
        pytest.param(MSE(), None, id="mse"),
        pytest.param(ExpectedUncovered(), Budget(amount=3.0), id="eu"),
        pytest.param(WeightedCVaR(0.5, 0.75), Budget(amount=3.0), id="wcvar"),
    ],
)
def test_a_second_component_that_never_pays_reduces_to_one_index(
    objective: Any, budget: Budget | None
) -> None:
    sample = _sample()
    composite = design(
        _two(TwoTriggers((50.0, 1000.0), (Free(), Free()))),
        sample,
        objective,
        budget=budget,
    )
    alone = design(
        Cover(RAIN, StepPayout([50.0], [Free()]), payout_unit="EUR"),
        sample.component("rain", loss="combined"),
        objective,
        budget=budget,
    )
    assert composite.objective_value == pytest.approx(alone.objective_value, abs=1e-6)
    assert composite.parameters["rain.amount"] == pytest.approx(
        alone.parameters["levels[0]"], abs=1e-6
    )
    assert composite.unidentified == ("heat.amount",)


def test_free_thresholds_of_a_vector_family_take_the_grid_route() -> None:
    template = _two(
        TwoTriggers((Free(40.0, 70.0, grid=[40.0, 50.0, 70.0]), 38.0), (Free(), Free()))
    )
    result = design(template, _sample(), MSE())
    assert result.route == "grid" and result.amounts == "qp"
    assert result.ties == 1 and result.unique is False
    assert result.parameters["rain.trigger"] == 40.0
    assert result.profile is not None and result.profile.num_rows == 3


def test_mixed_tails_read_each_components_own_side() -> None:
    """Heat is written with its sign turned, so the lower tail holds the same years."""
    index = CompositeIndex({"rain": RAIN, "heat": DRY})
    sample = _sample(
        index_components=_other_components(
            heat=_component(-HEAT_VALUES, "degC", tail="lower", own_loss=HEAT_LOSS)
        )
    )
    template = Cover(
        index,
        TwoTriggers((50.0, -38.0), (Free(), Free()), tail=("upper", "lower")),
        payout_unit="EUR",
    )
    result = design(template, sample, MSE())
    assert result.parameters["rain.amount"] == pytest.approx(8.0, abs=1e-6)
    assert result.parameters["heat.amount"] == pytest.approx(4.0, abs=1e-6)


def test_a_design_records_the_composite_cover() -> None:
    result = design(_template(), _sample(), MSE())
    assert isinstance(result.cover.index_definition, CompositeIndex)
    assert result.template_hash == _template().definition_hash
    assert result.sample_hash == _sample().content_hash


def test_a_family_with_no_amounts_takes_the_numpy_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise SolverUnavailableError("no solver", extra="parametric")

    monkeypatch.setattr(_solver, "require", missing)
    payout = NoAmounts(
        (50.0, 38.0),
        (Free(0.0, 8.0, grid=[0.0, 4.0, 8.0]), Free(0.0, 4.0, grid=[0.0, 4.0])),
    )
    result = design(_two(payout), _sample(), MSE())
    assert result.route == "grid" and result.solver is None
    assert result.parameters == {"rain.amount": 8.0, "heat.amount": 4.0}


def test_a_family_with_no_amounts_is_checked_against_the_limit() -> None:
    payout = NoAmounts(
        (50.0, 38.0),
        (Free(0.0, 8.0, grid=[0.0, 4.0, 8.0]), Free(0.0, 4.0, grid=[0.0, 4.0])),
    )
    result = design(_two(payout), _sample(), MSE(), limit=Limit(4.0))
    assert result.parameters["rain.amount"] == 4.0
    assert result.profile is not None
    statuses = result.profile.column("status").to_pylist()
    assert "over_limit" in statuses


def test_a_vector_family_not_linear_in_its_amounts_takes_the_numpy_grid() -> None:
    result = design(
        _two(Scaled(Free(0.0, 0.2, grid=[0.0, 0.1, 0.2]))), _sample(), MSE()
    )
    assert result.route == "grid" and result.solver is None


def test_a_user_objective_on_a_vector_family_takes_the_numpy_grid() -> None:
    def objective(loss: Any, payout: Any, probability: Any) -> float:
        return float(probability @ np.abs(loss - payout))

    template = _two(
        TwoTriggers(
            (50.0, 38.0),
            (Free(0.0, 8.0, grid=[0.0, 8.0]), Free(0.0, 4.0, grid=[0.0, 4.0])),
        )
    )
    result = design(template, _sample(), UserObjective(objective, name="abs"))
    assert result.route == "grid"
    assert result.parameters == {"rain.amount": 8.0, "heat.amount": 4.0}


def _one_index(hazard: str = "rain", unit: str = "mm/day", **index: Any) -> Cover:
    definition = IndexDefinition(hazard, unit, aggregation="annual_maximum", **index)
    return Cover(
        definition,
        StepPayout([50.0], [Free()], tail=definition.tail),
        payout_unit="EUR",
    )


def test_a_one_index_design_reads_the_component_of_its_hazard() -> None:
    """On the whole sample: the first component's or the second's index."""
    rain = design(_one_index(), _sample(), MSE())
    assert rain.parameters["levels[0]"] == pytest.approx(8.0, abs=1e-6)
    heat = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        _sample(),
        MSE(),
    )
    assert heat.parameters["levels[0]"] == pytest.approx(4.0, abs=1e-6)


def test_a_design_uses_the_loss_of_the_sample_it_is_given() -> None:
    combined = design(_one_index(), _sample(), MSE())
    own = design(_one_index(), _sample().component("rain"), MSE())
    assert combined.parameters["levels[0]"] == pytest.approx(8.0, abs=1e-6)
    assert own.parameters["levels[0]"] == pytest.approx(8.0, abs=1e-6)
    heat = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        _sample().component("heat"),
        MSE(),
    )
    assert heat.expected_loss == pytest.approx(HEAT_LOSS.mean())


@pytest.mark.parametrize(
    ("cover", "sample", "message"),
    [
        pytest.param(_template(), _DIRECT, "one index", id="a composite on one hazard"),
        pytest.param(
            _two(
                TwoTriggers((1.0, 1.0), (Free(), Free()), names=("rain", "wind")),
                index=CompositeIndex(
                    {
                        "rain": RAIN,
                        "wind": IndexDefinition(
                            "wind", "m/s", aggregation="annual_maximum"
                        ),
                    }
                ),
            ),
            _sample(),
            r"holds no \['wind'\]",
            id="a component the sample lacks",
        ),
        pytest.param(
            _two(
                TwoTriggers((1.0, 1.0), (Free(), Free())),
                index=CompositeIndex(
                    {
                        "rain": RAIN,
                        "heat": IndexDefinition(
                            "heat", "K", aggregation="annual_maximum"
                        ),
                    }
                ),
            ),
            _sample(),
            "'K'",
            id="another unit",
        ),
        pytest.param(
            _two(
                TwoTriggers((1.0, 1.0), (Free(), Free()), tail=("upper", "lower")),
                index=CompositeIndex({"rain": RAIN, "heat": DRY}),
            ),
            _sample(),
            "tail",
            id="another tail",
        ),
        pytest.param(
            _one_index("wind", "m/s"),
            _sample(),
            r"component\(name, loss='combined'\)",
            id="a one-index cover names no component",
        ),
        pytest.param(_one_index(unit="m"), _sample(), "'m'", id="a one-index unit"),
        pytest.param(
            _one_index(tail="lower"), _sample(), "tail", id="a one-index tail"
        ),
    ],
)
def test_what_a_cover_cannot_read_raises_in_design(
    cover: Cover, sample: OutcomeSample, message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        design(cover, sample, MSE())


def test_a_joint_design_refuses_a_composite_template() -> None:
    base = dataclass_with_asset(_DIRECT, "a")
    other = dataclass_with_asset(_DIRECT, "b")
    joined = OutcomeSample.join([base, other])
    with pytest.raises(SampleError, match="one index"):
        design_joint([_template(), _template()], joined, MSE())


def test_a_joint_design_refuses_a_sample_with_components() -> None:
    with pytest.raises(DesignError, match="use design"):
        design_joint([_template()], _sample(), MSE())


# -- evaluate and compare ----------------------------------------------------


def _column(table: Any, name: str) -> list[Any]:
    values: list[Any] = table.column(name).to_pylist()
    return values


def test_both_kinds_of_cover_share_one_table() -> None:
    sample = _sample()
    composite = _two(TwoTriggers((50.0, 38.0), (8.0, 4.0)))
    single = Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR", label="rain")
    other = Cover(HEAT, BinaryPayout(38.0, 4.0), payout_unit="EUR", label="heat")
    table = compare([composite, single, other], sample, split="in_sample")
    assert table.num_rows == 3
    assert len(set(_column(table, "sample_hash"))) == 1
    expected = {
        "two": (RAIN_VALUES >= 50.0) * 8.0 + (HEAT_VALUES >= 38.0) * 4.0,
        "rain": (RAIN_VALUES >= 50.0) * 8.0,
        "heat": (HEAT_VALUES >= 38.0) * 4.0,
    }
    for row, label in zip(
        table.to_pylist(), _column(table, "cover_label"), strict=True
    ):
        payout = expected[label]
        assert row["expected_payout"] == pytest.approx(payout.mean())
        assert row["expected_loss"] == pytest.approx(COMBINED.mean())


def test_a_table_of_both_kinds_writes_and_reads(tmp_path: Path) -> None:
    table = compare(
        [
            _two(TwoTriggers((50.0, 38.0), (8.0, 4.0))),
            Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR", label="rain"),
        ],
        _from_table(),
        split="in_sample",
    )
    path = tmp_path / "comparison.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.num_rows == 2
    n_identity = len(_identity_columns(read.column_names))
    assert to_long(read).num_rows == 2 * (read.num_columns - n_identity)


def test_a_composite_document_goes_into_the_metadata() -> None:
    table = evaluate(_resolved(), _sample(), split="in_sample")
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.covers[0].document["schema_version"] == "1.1"
    assert metadata.covers[0].document["index"]["kind"] == "composite"


def test_what_a_cover_cannot_read_raises_in_evaluate() -> None:
    with pytest.raises(SampleError, match="one index"):
        evaluate(_resolved(), _DIRECT, split="in_sample")
    single = Cover(
        IndexDefinition("wind", "m/s", aggregation="annual_maximum"),
        BinaryPayout(1.0, 1.0),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match=r"component\(name"):
        evaluate(single, _sample(), split="in_sample")


def test_evaluate_on_a_component_reads_its_own_loss() -> None:
    single = Cover(HEAT, BinaryPayout(38.0, 4.0), payout_unit="EUR")
    row = evaluate(single, _sample().component("heat"), split="in_sample").to_pylist()[
        0
    ]
    assert row["expected_loss"] == pytest.approx(HEAT_LOSS.mean())


def test_a_design_on_a_component_is_checked_against_that_view() -> None:
    sample = _from_table()
    own = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        sample.component("heat"),
        MSE(),
    )
    table = compare([own], sample, split="in_sample")
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["expected_loss"] == pytest.approx(COMBINED.mean())
    assert row["premium_basis"] == "design"
    assert row["premium"] == pytest.approx(own.premium)
    metadata = EvaluationMetadata.from_table(table)
    assert {tuple(entry["roles"]) for entry in metadata.setup.samples} == {
        ("design",),
        ("evaluation",),
    }


def test_a_stated_design_sample_is_compared_with_the_view() -> None:
    sample = _from_table()
    own = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        sample.component("heat"),
        MSE(),
    )
    table = compare([own], sample, split="in_sample", design_sample=sample)
    assert table.num_rows == 1


def test_a_design_on_a_combined_view_is_checked_against_that_view() -> None:
    sample = _from_table()
    own = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        sample.component("heat", loss="combined"),
        MSE(),
    )
    assert compare([own], sample, split="in_sample").num_rows == 1


def test_a_design_on_another_blocks_view_is_not_in_sample() -> None:
    sample = _from_table()
    train, test = holdout(sample, 0.75)
    own = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        train.component("heat"),
        MSE(),
    )
    with pytest.raises(SampleError, match="in sample means"):
        compare([own], test, split="in_sample")


def test_out_of_sample_on_the_held_out_block_of_both_kinds() -> None:
    train, test = holdout(_from_table(), 0.75)
    composite = design(_template(), train, MSE())
    one = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        train.component("heat"),
        MSE(),
    )
    table = compare([composite, one], test, split="out_of_sample")
    assert table.num_rows == 2
    assert _column(table, "split") == ["out_of_sample", "out_of_sample"]
    assert len(set(_column(table, "expected_loss"))) == 1


def test_a_design_on_a_component_sees_a_shared_period_as_leakage() -> None:
    sample = _from_table()
    train, _ = holdout(sample, 0.75)
    own = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        train.component("heat"),
        MSE(),
    )
    with pytest.raises(LeakageError):
        compare([own], sample, split="out_of_sample")


def test_a_set_of_covers_refuses_a_sample_with_components() -> None:
    cover = Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR")
    with pytest.raises(SampleError, match="mapping of covers"):
        evaluate({"site": cover}, _sample(), split="in_sample")


# -- the setup ---------------------------------------------------------------


def test_the_setup_states_the_components() -> None:
    table = compare(
        [_two(TwoTriggers((50.0, 38.0), (8.0, 4.0)))], _from_table(), split="in_sample"
    )
    entry = EvaluationMetadata.from_table(table).setup.samples[0]
    assert entry["components"] == [
        {"name": "rain", "own_loss": True, "tail": "upper", "unit": "mm/day"},
        {"name": "heat", "own_loss": True, "tail": "upper", "unit": "degC"},
    ]
    assert entry["combine"] == {"kind": "table"}
    assert entry["hazard_dependence"] == {"kind": "data"}
    text = setup_text(table)
    assert (
        "Index components: rain (in mm/day, upper tail, with a loss of its own) and "
        "heat (in degC, upper tail, with a loss of its own); the first is the "
        "sample's index." in text
    )
    assert "Combined loss: as the table states it." in text
    assert "Dependence between the hazards: the data's own" in text
    assert "Combined loss: every row's metrics are on the combined loss" in text


def test_the_setup_says_what_a_component_has_not() -> None:
    sample = _from_table(own_loss=None)
    table = compare(
        [_two(TwoTriggers((50.0, 38.0), (8.0, 4.0)))], sample, split="in_sample"
    )
    assert "with no loss of its own" in setup_text(table)


def test_the_setup_says_what_each_design_read() -> None:
    sample = _from_table()
    on_combined = design(_template(), sample, MSE())
    on_own = design(
        Cover(HEAT, StepPayout([38.0], [Free()]), payout_unit="EUR"),
        sample.component("heat"),
        MSE(),
    )
    on_view = design(
        Cover(RAIN, StepPayout([50.0], [Free()]), payout_unit="EUR"),
        sample.component("rain", loss="combined"),
        MSE(),
    )
    text = setup_text(
        compare([on_combined, on_own, on_view], sample, split="in_sample")
    )
    assert "designed on the combined loss" in text
    assert "designed on the own loss of component heat" in text
    assert "designed on the index of component rain, on the combined loss" in text


def test_a_rule_other_than_the_table_is_named() -> None:
    sample = _sample(provenance=_provenance(combine={"kind": "capped_sum", "cap": 1.0}))
    cover = _two(TwoTriggers((50.0, 38.0), (8.0, 4.0)))
    text = setup_text(compare([cover], sample, split="in_sample"))
    assert "Combined loss: capped_sum, as stated." in text


def test_the_setup_of_a_one_index_design_is_unchanged() -> None:
    own = design(_one_index(), _sample().component("rain"), MSE())
    alone = design(
        Cover(RAIN, LinearPayout(1.0, 4.0, Free()), payout_unit="EUR"), _DIRECT, MSE()
    )
    assert "designed on" not in setup_text(compare([alone], _DIRECT, split="in_sample"))
    assert own.design_sample.index_components is None


def test_at_point_stays_in_a_composite_definition() -> None:
    placed = IndexDefinition(
        "rain",
        "mm/day",
        aggregation="annual_maximum",
        location=AtPoint(1.0, 2.0, name="p"),
    )
    composite = CompositeIndex({"rain": placed, "heat": HEAT})
    assert CompositeIndex.from_document(composite.to_document()) == composite
