"""The parametric cover: an index, a payout function and the contract terms."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from crc_sdk.parametric._canonical import canonical_json, content_hash
from crc_sdk.parametric.errors import (
    CoverDefinitionError,
    SampleError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.index import (
    CompositeIndex,
    IndexDefinition,
    _index_words,
    _unit_words,
)
from crc_sdk.parametric.parameters import Free, _resolve_values
from crc_sdk.parametric.payouts import (
    PayoutFunction,
    _Axis,
    _free_words,
    _index_values,
    _most_words,
    _payout_reader,
    _payout_words,
    _Words,
)

__all__ = [
    "COMPOSITE_COVER_SCHEMA_VERSION",
    "COVER_SCHEMA_VERSION",
    "COVER_SEMANTICS_VERSION",
    "Cover",
]

COVER_SCHEMA_VERSION = "1.0"
"""The `schema_version` of the document of a cover on one index."""

COMPOSITE_COVER_SCHEMA_VERSION = "1.1"
"""The `schema_version` of the document of a cover on a `CompositeIndex`."""

COVER_SEMANTICS_VERSION = "crc-parametric-cover/1"
"""The `semantics_version` of cover documents."""

_DOCUMENT_KEYS = frozenset(
    {
        "asset_id",
        "index",
        "payout",
        "payout_unit",
        "schema_version",
        "semantics_version",
    }
)
_IDENTIFIER = re.compile(r"^\S(?:[\s\S]*\S)?$")


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CoverDefinitionError(
            f"{name} must be a non-empty string without surrounding whitespace, "
            f"got {value!r}"
        )
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise CoverDefinitionError(f"cover JSON repeats the key {key!r}")
        document[key] = value
    return document


def _reject_constant(name: str) -> NoReturn:
    raise CoverDefinitionError(f"cover JSON must not contain {name}")


def _check_payout_reads(
    payout: PayoutFunction, index: IndexDefinition | CompositeIndex
) -> None:
    """Raise `CoverDefinitionError` unless `payout` reads what `index` holds."""
    if isinstance(index, CompositeIndex):
        if tuple(payout.index_names) != index.names:
            raise CoverDefinitionError(
                f"the payout reads the components {payout.index_names!r}, but the "
                f"index holds {index.names!r}, in that order"
            )
        if payout.tail != index.tails:
            raise CoverDefinitionError(
                f"the payout's tails {payout.tail!r} differ from the index's "
                f"tails {index.tails!r}: a payout on several components has a "
                "tuple of one tail per component, in order"
            )
        return
    if payout.tail != index.tail:
        raise CoverDefinitionError(
            f"the payout's tail {payout.tail!r} differs from the index's "
            f"tail {index.tail!r}"
        )
    if tuple(payout.index_names) != ("index",):
        raise CoverDefinitionError(
            "a cover on one index reads one index component, index_names == "
            f"('index',); got {payout.index_names!r}"
        )


class Cover:
    """A parametric cover: what it pays on, how much, and for whom.

    `index` is the `IndexDefinition` the cover pays on and `payout` a payout
    function (`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`,
    `BinaryPayout`, or a registered user family) whose `tail` equals the
    index's. `payout_unit` is the unit of the payouts, for example a
    currency. `asset_id` names the asset the cover is written for; the same
    terms on another asset are another contract. `label` is a display name
    only.

    `index` may instead be a `CompositeIndex`, for a cover on several hazards.
    The payout's `index_names` must then equal its names and its `tail` be the
    tuple of its tails, in order, and the cover document writes
    `COMPOSITE_COVER_SCHEMA_VERSION`, so that a reader of the one-index
    document refuses it.

    A cover with `Free` parameters is a template: it has a document and a
    hash, but `payout_values` raises until `with_values` (or a design)
    resolves it.

    The cover document holds every term except the label, as canonical JSON
    (`to_json`). `definition_hash` is `"sha256:"` plus the SHA-256 of that
    text, the format of `ImpactExpression.definition_hash`. Covers compare
    and hash by `definition_hash`, so two covers with the same terms are
    equal whatever their labels, and they pickle through their JSON.

    Raises `CoverDefinitionError` for an invalid index, payout, unit, asset
    id or label, or a payout whose tail or index names differ from the
    index's.
    """

    __slots__ = (
        "_asset_id",
        "_document",
        "_hash",
        "_index",
        "_json",
        "_label",
        "_payout",
        "_payout_unit",
    )

    def __init__(
        self,
        index: IndexDefinition | CompositeIndex,
        payout: PayoutFunction,
        *,
        payout_unit: str,
        asset_id: str | None = None,
        label: str | None = None,
    ) -> None:
        if not isinstance(index, IndexDefinition | CompositeIndex):
            raise CoverDefinitionError(
                "index must be an IndexDefinition or a CompositeIndex, got "
                f"{type(index).__name__}"
            )
        if not isinstance(payout, PayoutFunction):
            raise CoverDefinitionError(
                f"payout must implement PayoutFunction, got {type(payout).__name__}"
            )
        _check_payout_reads(payout, index)
        payout_document = payout.to_document()
        if (
            not isinstance(payout_document, Mapping)
            or payout_document.get("kind") != payout.kind
        ):
            raise CoverDefinitionError(
                f"the payout document must be an object whose kind is {payout.kind!r}"
            )
        unit = _identifier("payout_unit", payout_unit)
        asset = None if asset_id is None else _identifier("asset_id", asset_id)
        document: dict[str, Any] = {
            "asset_id": asset,
            "index": index.to_document(),
            "payout": dict(payout_document),
            "payout_unit": unit,
            "schema_version": (
                COMPOSITE_COVER_SCHEMA_VERSION
                if isinstance(index, CompositeIndex)
                else COVER_SCHEMA_VERSION
            ),
            "semantics_version": COVER_SEMANTICS_VERSION,
        }
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise CoverDefinitionError(
                f"label must be a non-empty string, got {label!r}"
            )
        try:
            text = canonical_json(document)
        except (TypeError, ValueError) as error:
            raise CoverDefinitionError(
                f"the cover document has no canonical JSON form: {error}"
            ) from error
        self._index = index
        self._payout = payout
        self._payout_unit = unit
        self._asset_id = asset
        self._label = label
        self._document = document
        self._json = text
        self._hash = content_hash(document)

    @property
    def index(self) -> IndexDefinition:
        """The index of a cover on one index.

        Raises `CoverDefinitionError` on a cover with a `CompositeIndex`,
        which has no one index: read `index_definition`.
        """
        if isinstance(self._index, CompositeIndex):
            raise CoverDefinitionError(
                f"this cover pays on the components {list(self._index.names)}, not "
                "on one index: read cover.index_definition. ParametricPlan takes "
                "covers on one index for now; design and evaluate this cover with "
                "the functions"
            )
        return self._index

    @property
    def index_definition(self) -> IndexDefinition | CompositeIndex:
        """What the cover pays on: an `IndexDefinition` or a `CompositeIndex`."""
        return self._index

    @property
    def payout(self) -> PayoutFunction:
        """The payout function and its parameters."""
        return self._payout

    @property
    def payout_unit(self) -> str:
        """The unit of the payouts."""
        return self._payout_unit

    @property
    def asset_id(self) -> str | None:
        """The asset the cover is written for, if named."""
        return self._asset_id

    @property
    def label(self) -> str | None:
        """The display name; not part of the document or the hash."""
        return self._label

    @property
    def definition_hash(self) -> str:
        """`"sha256:<hex>"` of the canonical cover document."""
        return self._hash

    def free_parameters(self) -> tuple[str, ...]:
        """The names of the free parameters, in the payout's order."""
        return tuple(
            name
            for name, value in self._payout.parameters().items()
            if isinstance(value, Free)
        )

    def with_values(self, values: Mapping[str, float]) -> Cover:
        """Return the cover with values for some or all free parameters.

        Each name must be a free parameter and each value a finite number
        within its bounds. The index, unit, asset and label are kept. Raises
        `CoverDefinitionError` otherwise, or if the resolved payout is
        invalid (for example, unordered thresholds).
        """
        checked = _resolve_values(self._payout.parameters(), values)
        return Cover(
            self._index,
            self._payout.with_values(checked),
            payout_unit=self._payout_unit,
            asset_id=self._asset_id,
            label=self._label,
        )

    def payout_values(self, index_values: ArrayLike) -> NDArray[np.float64]:
        """Return the payout for each index value, in `payout_unit`.

        The result has the shape of `index_values`. On a cover with a
        `CompositeIndex`, `index_values` has shape `(n, k)`, one row per draw
        and one column per component in `index_names` order, and the result
        `(n,)`. Raises `UnresolvedParameterError` on a template, and
        `SampleError` for non-numeric or non-finite index values or, on a
        composite cover, the wrong shape.
        """
        free = self.free_parameters()
        if free:
            raise UnresolvedParameterError(
                f"the cover is a template; resolve its free parameters first: "
                f"{', '.join(free)}"
            )
        if isinstance(self._index, CompositeIndex):
            values = _index_values(index_values, ndim=2)
            names = self._index.names
            if values.shape[1] != len(names):
                raise SampleError(
                    f"index values must have one column per component {list(names)}, "
                    f"got shape {values.shape}"
                )
            shape = values.shape[:1]
        else:
            values = _index_values(index_values)
            shape = values.shape
        payout = np.asarray(self._payout.evaluate(values), dtype=np.float64)
        if payout.shape != shape:
            raise CoverDefinitionError(
                f"payout {self._payout.kind!r} returned shape {payout.shape} "
                f"for index values of shape {values.shape}"
            )
        if not np.isfinite(payout).all() or (payout < 0).any():
            raise CoverDefinitionError(
                f"payout {self._payout.kind!r} returned negative or non-finite values"
            )
        return payout

    def terms_text(self) -> str:
        """Return the cover's terms in plain text, for a report or a broker.

        One statement a line, in this order:

        - `Label`, when the cover has one: a display name, not a term;
        - `Asset`: the asset the cover is written for, or that it names none;
        - `Index`: the hazard, its unit, where it is read, its severe side,
          the aggregation and the source, saying plainly when the terms state
          no aggregation or source; on a `CompositeIndex`, one indented line
          per component, in order;
        - `Payout per year`: the payout unit and the family, with its `kind`,
          then one indented line per band, knot or cell, from nothing to the
          most severe, in index and payout units. A value equal to a threshold
          belongs to the more severe band. A `SumOf` or `MaxOf` gives each
          component's payout under its name, a `StepGrid` one line per cell,
          and a family of your own (or a subclass of a built-in one) its
          parameters as it names them;
        - `Most paid in a year`: the largest amount the terms pay, which a
          `LinearPayout` calls its limit, or why it is not known yet;
        - `Free parameters`: none, or, on a template, each free parameter's
          bounds and grid, in its unit. A free value appears in the lines
          above by its name, as `free_parameters()` lists it;
        - `Hash`: `definition_hash`, with the document's schema and semantics
          versions.

        Numbers are spelled exactly as the cover document holds them, grouped
        by thousands and without an exponent, so the text states the terms
        that the hash identifies. A rounded amount is another contract: build
        the cover with it, and it has its own hash.

        The text is a view of the terms: it depends only on them and the
        label, so the same cover gives the same text, and it reads or writes
        nothing.
        """
        index = self._index
        lines = (
            []
            if self._label is None
            else [f"Label: {self._label} (a display name, not part of the terms)"]
        )
        lines.append(
            f"Asset: {'not named' if self._asset_id is None else self._asset_id}"
        )
        axes: dict[str, _Axis]
        if isinstance(index, CompositeIndex):
            lines.append(f"Index: {len(index.names)} components, in this order")
            lines += [
                f"  - {name}: {_index_words(component)}"
                for name, component in index.components.items()
            ]
            axes = {
                name: (name, component.unit)
                for name, component in index.components.items()
            }
        else:
            lines.append(f"Index: {_index_words(index)}")
            axes = {"index": (index.hazard_name, index.unit)}
        words = _Words(self._payout_unit)
        head, items = _payout_words(self._payout, words, axes)
        lines.append(f"Payout per year, {_unit_words(self._payout_unit)}: {head}")
        lines += [f"  {item}" for item in items]
        lines.append(
            f"Most paid in a year: {_most_words(self._payout, self._payout_unit)}"
        )
        free = self.free_parameters()
        if not free:
            lines.append("Free parameters: none")
        else:
            lines.append(
                "Free parameters: a design chooses them, and the cover pays nothing "
                "until they are resolved"
            )
            parameters = self._payout.parameters()
            for name in free:
                value, unit = words.free.get(name, (parameters[name], None))
                lines.append(f"  - {name}: {_free_words(cast(Free, value), unit)}")
        lines.append(
            f"Hash: {self._hash}, of the cover document (schema "
            f"{self._document['schema_version']}, semantics "
            f"{COVER_SEMANTICS_VERSION})"
        )
        return "\n".join(lines)

    def to_document(self) -> dict[str, Any]:
        """Return a copy of the cover document, as plain JSON types."""
        return copy.deepcopy(self._document)

    def to_json(self) -> str:
        """Return the canonical JSON text of the cover document."""
        return self._json

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, label: str | None = None
    ) -> Cover:
        """Read a cover document, as `to_document` returns it.

        The payout's `kind` is looked up among the registered payout
        classes. A document of `COVER_SCHEMA_VERSION` holds one index, and one
        of `COMPOSITE_COVER_SCHEMA_VERSION` a `CompositeIndex`. Raises
        `UnknownPayoutKindError` for an unregistered kind, and
        `CoverDefinitionError` for missing or unknown keys, another schema or
        semantics version, or invalid terms.
        """
        if not isinstance(document, Mapping):
            raise CoverDefinitionError("a cover document must be an object")
        keys = set(document)
        if keys != _DOCUMENT_KEYS:
            missing = ", ".join(sorted(_DOCUMENT_KEYS - keys)) or "none"
            unknown = ", ".join(sorted(map(str, keys - _DOCUMENT_KEYS))) or "none"
            raise CoverDefinitionError(
                f"a cover document has the keys {sorted(_DOCUMENT_KEYS)}; "
                f"missing: {missing}; unknown: {unknown}"
            )
        version = document["schema_version"]
        if version not in (COVER_SCHEMA_VERSION, COMPOSITE_COVER_SCHEMA_VERSION):
            raise CoverDefinitionError(
                f"unsupported cover schema_version {document['schema_version']!r}"
            )
        if document["semantics_version"] != COVER_SEMANTICS_VERSION:
            raise CoverDefinitionError(
                f"unsupported cover semantics_version {document['semantics_version']!r}"
            )
        payout_document = document["payout"]
        if not isinstance(payout_document, Mapping):
            raise CoverDefinitionError("a payout document must be an object")
        payout = _payout_reader(payout_document.get("kind"))(payout_document)
        return cls(
            (
                CompositeIndex.from_document(document["index"])
                if version == COMPOSITE_COVER_SCHEMA_VERSION
                else IndexDefinition.from_document(document["index"])
            ),
            payout,
            payout_unit=document["payout_unit"],
            asset_id=document["asset_id"],
            label=label,
        )

    @classmethod
    def from_json(cls, text: str, *, label: str | None = None) -> Cover:
        """Read a cover from its JSON text, as `to_json` writes it.

        `label` sets the display name, which the JSON does not hold. Raises
        `CoverDefinitionError` for invalid JSON (including repeated keys,
        NaN and infinity) and as `from_document` does.
        """
        if not isinstance(text, str):
            raise CoverDefinitionError(f"cover JSON must be a str, got {type(text)}")
        try:
            document = json.loads(
                text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except json.JSONDecodeError as error:
            raise CoverDefinitionError(f"cover JSON is not valid: {error}") from error
        return cls.from_document(document, label=label)

    @classmethod
    def _from_pickle(cls, text: str, label: str | None) -> Cover:
        return cls.from_json(text, label=label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cover):
            return NotImplemented
        return self._hash == other._hash

    def __hash__(self) -> int:
        return hash(self._hash)

    def __reduce__(self) -> tuple[Any, tuple[str, str | None]]:
        return (type(self)._from_pickle, (self._json, self._label))

    def __repr__(self) -> str:
        label = f"{self._label!r}, " if self._label is not None else ""
        return f"Cover({label}{self._payout.kind}, {self._hash})"
