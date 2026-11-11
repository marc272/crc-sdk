"""What a parametric cover pays on: the index definition and its location.

`IndexDefinition` is one index. `CompositeIndex` names several, for a cover
that pays on several hazards at once.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, field_validator

from crc_sdk.impacts._types import FiniteFloat, Identifier
from crc_sdk.parametric.errors import CoverDefinitionError
from crc_sdk.parametric.parameters import _DefinitionModel

__all__ = [
    "AtAsset",
    "AtCell",
    "AtPoint",
    "AtStation",
    "CompositeIndex",
    "IndexDefinition",
    "Location",
    "Tail",
]

Tail = Literal["upper", "lower"]
"""Which side of the index is severe: larger values (`"upper"`) or smaller."""

_H3_CELL = re.compile(r"^[0-9a-f]{15}$")
_COMPONENT_NAME = re.compile(r"[^\s.+\[\]](?:[^.+\[\]]*[^\s.+\[\]])?")


def _component_name(
    name: Any, error: type[Exception] = CoverDefinitionError, what: str = "a component"
) -> str:
    """Return a valid index component name; raise `error` otherwise.

    Parameter names and table columns are built from component names, so a
    name holds no `.`, `+`, `[` or `]` and no surrounding whitespace.
    """
    if not isinstance(name, str) or not _COMPONENT_NAME.fullmatch(name):
        raise error(
            f"{what} name must be a non-empty string without surrounding "
            f"whitespace and without '.', '+', '[' or ']', got {name!r}"
        )
    return name


class AtAsset(_DefinitionModel):
    """The index is read at the cover's asset."""

    kind: Literal["asset"] = "asset"

    def __init__(self) -> None:
        super().__init__()


class AtPoint(_DefinitionModel):
    """The index is read at a WGS84 point, for example a reference site."""

    kind: Literal["point"] = "point"
    longitude: Annotated[FiniteFloat, Field(ge=-180.0, le=180.0)]
    latitude: Annotated[FiniteFloat, Field(ge=-90.0, le=90.0)]
    name: Identifier | None = None

    def __init__(
        self, longitude: float, latitude: float, name: str | None = None
    ) -> None:
        super().__init__(longitude=longitude, latitude=latitude, name=name)


class AtStation(_DefinitionModel):
    """The index is read at an observation station."""

    kind: Literal["station"] = "station"
    station_id: Identifier

    def __init__(self, station_id: str) -> None:
        super().__init__(station_id=station_id)


class AtCell(_DefinitionModel):
    """The index is read in one H3 cell.

    `cell_index` takes the canonical unsigned integer (the `cell_index`
    column of a canonical dataset) or its 15-digit hexadecimal string. The
    document stores the string, because JSON numbers are binary64 floats and
    cannot hold every 64-bit cell index.
    """

    kind: Literal["cell"] = "cell"
    cell_index: str

    def __init__(self, cell_index: int | str) -> None:
        super().__init__(cell_index=cell_index)

    @field_validator("cell_index", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise CoverDefinitionError("cell_index must be an H3 cell, not a bool")
        if isinstance(value, int | np.integer):
            if not 0 < int(value) < 2**64:
                raise CoverDefinitionError(
                    "cell_index must be an unsigned 64-bit H3 cell"
                )
            value = format(int(value), "x")
        if not isinstance(value, str) or not _H3_CELL.fullmatch(value.lower()):
            raise CoverDefinitionError(
                f"cell_index must be an H3 cell (15 hexadecimal digits), got {value!r}"
            )
        return value.lower()


Location = Annotated[
    AtAsset | AtPoint | AtStation | AtCell, Field(discriminator="kind")
]
"""Where the index is read: `AtAsset`, `AtPoint`, `AtStation` or `AtCell`."""

_LOCATIONS: dict[str, type[AtAsset | AtPoint | AtStation | AtCell]] = {
    "asset": AtAsset,
    "point": AtPoint,
    "station": AtStation,
    "cell": AtCell,
}


def _number(value: float) -> str:
    """A number in words: its exact shortest spelling, grouped by thousands.

    `60.0` is `60`, `20000.0` is `20,000` and `1e-07` is `0.0000001`: the
    value as the cover document holds it, without an exponent or rounding.
    """
    text = format(Decimal(repr(float(value) + 0.0)), "f")
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, _, fraction = digits.partition(".")
    fraction = fraction.rstrip("0")
    return f"{sign}{int(whole):,}" + (f".{fraction}" if fraction else "")


def _quantity(value: float, unit: str | None) -> str:
    """A number with its unit; a dimensionless unit (`None` or `"1"`) adds none."""
    number = _number(value)
    return number if unit is None or unit == "1" else f"{number} {unit}"


def _unit_words(unit: str) -> str:
    return "dimensionless (unit 1)" if unit == "1" else f"in {unit}"


def _location_words(location: AtAsset | AtPoint | AtStation | AtCell) -> str:
    """Where the index is read, in words."""
    if isinstance(location, AtPoint):
        where = (
            f"longitude {_number(location.longitude)}, "
            f"latitude {_number(location.latitude)}, WGS84"
        )
        if location.name is None:
            return f"read at the point ({where})"
        return f"read at the point {location.name} ({where})"
    if isinstance(location, AtStation):
        return f"read at the station {location.station_id}"
    if isinstance(location, AtCell):
        return f"read in the H3 cell {location.cell_index}"
    return "read at the asset"


def _index_words(index: IndexDefinition) -> str:
    """An index's terms in words: hazard, unit, place, severe side, aggregation, source.

    The aggregation and the source say plainly when the terms state none.
    """
    severe = "larger" if index.tail == "upper" else "smaller"
    aggregation = (
        "not stated, so the terms don't say how a year's value is made"
        if index.aggregation is None
        else index.aggregation
    )
    source = "not stated" if index.source is None else index.source
    return (
        f"{index.hazard_name}, {_unit_words(index.unit)}, "
        f"{_location_words(index.location)}; {severe} values are more severe "
        f"(tail {index.tail}); aggregation: {aggregation}; source: {source}"
    )


def _location_from_document(document: Mapping[str, Any]) -> Any:
    kind = document.get("kind")
    location = _LOCATIONS.get(kind) if isinstance(kind, str) else None
    if location is None:
        raise CoverDefinitionError(
            f"location kind must be one of {sorted(_LOCATIONS)}, got {kind!r}"
        )
    return location._validated(
        {key: value for key, value in document.items() if key != "kind"}
    )


class IndexDefinition(_DefinitionModel):
    """The index a cover pays on.

    `hazard_name` and `unit` name the hazard and its unit, as a canonical
    dataset's `hazard_name` and `value_unit` do. `tail` says which side is
    severe: `"upper"` for hazards where larger values are worse (rainfall,
    wind), `"lower"` where smaller values are worse (a drought index); it
    follows the dataset's `return_period_tail`. `location` says where the
    index is read (`AtAsset()` by default). `source` optionally names the
    data source.

    `aggregation` optionally says how one year's values become the year's
    index, for example `"annual_maximum"`. It is part of the terms, so the
    cover document and its hash hold it, but nothing checks it: a stated
    value is free text, and no design or evaluation reads it, because a
    sample holds one index value per year whatever made it. Left out
    (`None`), the terms
    don't say how the year's value was made, and the document holds
    `"aggregation": null`, so its hash differs from that of the same terms
    with an aggregation stated.

    Raises `CoverDefinitionError` for empty names or units, an unknown tail
    or an invalid location, and from `from_document` for a document without
    the `aggregation` key.
    """

    hazard_name: Identifier
    unit: Identifier
    tail: Tail = "upper"
    location: Location = AtAsset()
    aggregation: Identifier | None = None
    source: Identifier | None = None

    def __init__(
        self,
        hazard_name: str,
        unit: str,
        *,
        tail: Tail = "upper",
        location: AtAsset | AtPoint | AtStation | AtCell = AtAsset(),
        aggregation: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            hazard_name=hazard_name,
            unit=unit,
            tail=tail,
            location=location,
            aggregation=aggregation,
            source=source,
        )

    @field_validator("location", mode="before")
    @classmethod
    def _read_location(cls, value: Any) -> Any:
        return _location_from_document(value) if isinstance(value, Mapping) else value

    def to_document(self) -> dict[str, Any]:
        """Return the definition as plain JSON types, as the cover stores it."""
        document: dict[str, Any] = self.model_dump(mode="json")
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a definition written by `to_document`.

        The document holds `aggregation`, `null` when the terms leave it
        unstated, as every document `to_document` writes does.
        """
        if isinstance(document, Mapping) and "aggregation" not in document:
            fields = cls.model_fields
            unknown = sorted(str(key) for key in document if key not in fields)
            missing = [
                name
                for name, info in fields.items()
                if (info.is_required() or name == "aggregation")
                and name not in document
            ]
            raise CoverDefinitionError(
                f"a {cls.__name__} document has unknown fields {unknown} and lacks "
                f"the fields {missing}; aggregation is null when the terms don't "
                "state one"
            )
        return cls._validated(document)


_COMPOSITE_KEYS = frozenset({"components", "kind"})
_COMPOSITE_COMPONENT_KEYS = frozenset({"index", "name"})


@dataclass(frozen=True, eq=False)
class CompositeIndex:
    """Several named indices a cover pays on together.

    `components` maps each component's name to its `IndexDefinition`, two or
    more, in order: the order is part of the terms, and the columns of the
    index values a payout reads follow it. A sample's component of the same
    name supplies each one's values.

    Two composite indices are equal when they hold the same names and
    definitions in the same order. Raises `CoverDefinitionError` for fewer
    than two components, an invalid name or a value that is not an
    `IndexDefinition`.
    """

    components: Mapping[str, IndexDefinition]

    def __post_init__(self) -> None:
        components = self.components
        if not isinstance(components, Mapping):
            raise CoverDefinitionError(
                "components must map each component name to an IndexDefinition, "
                f"got {type(components).__name__}"
            )
        if len(components) < 2:
            raise CoverDefinitionError(
                "a CompositeIndex holds two components or more, got "
                f"{len(components)}; a cover on one index takes its IndexDefinition"
            )
        for name, index in components.items():
            _component_name(name)
            if not isinstance(index, IndexDefinition):
                raise CoverDefinitionError(
                    f"component {name!r} must be an IndexDefinition, "
                    f"got {type(index).__name__}"
                )
        object.__setattr__(self, "components", MappingProxyType(dict(components)))

    @property
    def names(self) -> tuple[str, ...]:
        """The component names, in order."""
        return tuple(self.components)

    @property
    def tails(self) -> tuple[Tail, ...]:
        """Each component's severe side, in order."""
        return tuple(index.tail for index in self.components.values())

    def to_document(self) -> dict[str, Any]:
        """Return the definition as plain JSON types, as the cover stores it.

        The components are a list, because canonical JSON sorts the keys of
        an object and their order is part of the terms.
        """
        return {
            "components": [
                {"index": index.to_document(), "name": name}
                for name, index in self.components.items()
            ],
            "kind": "composite",
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        """Read a definition written by `to_document`.

        Raises `CoverDefinitionError` for another `kind`, unknown or missing
        keys, or a component name that repeats.
        """
        if not isinstance(document, Mapping) or set(document) != _COMPOSITE_KEYS:
            raise CoverDefinitionError(
                f"a CompositeIndex document has the keys {sorted(_COMPOSITE_KEYS)}"
            )
        if document["kind"] != "composite":
            raise CoverDefinitionError(
                f"expected an index of kind 'composite', got {document['kind']!r}"
            )
        entries = document["components"]
        if isinstance(entries, (str, bytes, Mapping)) or not isinstance(
            entries, Sequence
        ):
            raise CoverDefinitionError(
                "a CompositeIndex document's components is a list"
            )
        components: dict[str, IndexDefinition] = {}
        for entry in entries:
            if (
                not isinstance(entry, Mapping)
                or set(entry) != _COMPOSITE_COMPONENT_KEYS
            ):
                raise CoverDefinitionError(
                    "each component of a CompositeIndex document has the keys "
                    f"{sorted(_COMPOSITE_COMPONENT_KEYS)}"
                )
            name = _component_name(entry["name"])
            if name in components:
                raise CoverDefinitionError(
                    f"a CompositeIndex document repeats the component {name!r}"
                )
            components[name] = IndexDefinition.from_document(entry["index"])
        return cls(components)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeIndex):
            return NotImplemented
        return tuple(self.components.items()) == tuple(other.components.items())

    def __hash__(self) -> int:
        return hash(tuple(self.components.items()))

    def __reduce__(self) -> tuple[Any, tuple[dict[str, IndexDefinition]]]:
        return (type(self), (dict(self.components),))
