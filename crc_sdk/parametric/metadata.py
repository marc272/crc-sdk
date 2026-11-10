"""The metadata of a comparison table: the setup behind every number.

A comparison table (`compare`, `evaluate`) carries one JSON document under
the Parquet schema metadata key `crc.parametric.evaluation`
(`EVALUATION_METADATA_KEY`), validated by `EvaluationMetadata` and written
with sorted keys. It holds:

- the environment: `schema_version`, `producer`, `creation_version` (the
  crc-sdk version), `crc_framework_version`, `numpy_version`, `platform`,
  `created_at` (UTC), and `solvers`, the solvers of any design evaluated;
- `setup`, the setup a published number needs next to it: the basis (one
  draw is one year), the metric options, every sample (source kind, draws
  and periods, semantics and who stated them, the basis-risk and loss
  models, units, lineage and split steps, the dependence between its assets,
  its trend record and its pool of model-years, without the row arrays, and
  the `pathway`, `horizon` and `product` its source records state; a window of
  horizons also its `members`; a sample with index components also its
  `components`, each with its name, unit, tail and whether it has a loss of its
  own, its `combine` and its `hazard_dependence`)
  and every evaluation (sample, split, fold, how the split was made, the
  design samples checked for leakage, the pricing, the levels and scopes of
  its rows and the allocation rule). `compare_pooled` adds each block and one
  sample entry for the pooled rows, and writes one evaluation with `fold` null,
  `method` `"kfold_pooled"` and `details` with `k` and the held-out blocks'
  hashes in fold order. `bootstrap_intervals` adds `intervals`: the method, the
  resampling unit, the number of resamples, the level and the columns whose
  level reaches the block's worst row. `intervals` is null otherwise.
  `design_frontier` adds `frontier`, one entry per call: the template's hash, the
  objective, the budgets as stated and each one's design, the design sample and
  the block read, with its split and fold.
  `compare_scenarios` adds `scenarios`: the reference, the labels,
  `mixed_products`, `design_on`, which scenarios share draws and the generator
  record they share, and per scenario its source kind, `hazard`, `pathway`,
  `horizon`, `product`, `members` (null for a per-year scenario), where it reads
  the index, `n_rows` and its `blocks`, one per fold: the block's rows, how it
  was made, a window's rows per member and the columns whose level reaches its
  worst row. Each of its evaluations also has `scenario`, `design_scenario` and
  `premium_bases`, and `method` may be `"other_scenario"` and `leakage_check`
  `"different_sources"`. `to_drift` adds `drift`, with the reference used.
  `scenarios`, `drift` and `frontier` are absent, and read as null, on any other
  table;
- `columns`: for every metric column, its metric, parameter, unit and a
  one-line definition;
- `covers`: every cover's label, hash and full document, the hash of every
  set of covers a `"total"` or allocated row names, with the asset ids and
  cover hashes it is made of, and the hash of every package a `"package"` row
  names, with the component names and cover hashes it is made of;
- `designs`: the record of every design evaluated, with the label of its
  cover as evaluated (`cover_label`), and for a pooled row an entry of `kind`
  `"pooled"` under the row's `design_id` that lists the folds' designs, as
  `covers` lists the folds' covers;
- `sources`, `licences` and `rng`: the samples' source documents, their
  licences and attributions, and the generator state before every random
  step, each tagged with its sample's hash, the step `"bootstrap"` included;
- `source_records`: what a plan read, once for all its samples (the
  canonical file, or the annual table), empty for a comparison that
  `compare` or `evaluate` builds;
- `assets`: the number of assets; the id column and the hash of the
  (asset id, value) pairs when the caller knows them, as a plan does and as
  an allocation's `values=` state them. Asset values are never written.

A document or table without the `scope`, `components` or `design_budget` column,
or without an evaluation's `level`, `scope` and `allocation` or a sample's
`dependence`, reads as rows of single assets: `level` `["asset"]`, `scope`
`["individual"]`, no allocation, no dependence and no component. A sample
without `trend` has none recorded, which on per-year values reads as the years
as recorded, and a sample without `pool` is no pool. A comparison without the
metric column `payout_frequency` or `unidentified_share` reads as null there, as
does one without the column `design_scenario`, `pathway`, `horizon`, `product` or
`n_rows`, or without the rank-block columns `copula_reference_years` and
`copula_block_tail_b<beta>`, and a document without `intervals` has no intervals.

The module claims no licence of its own for a comparison: it is derived data
under its inputs' terms, which `licences` lists.

`setup_text(metadata)` states the setup in plain text, one statement a line,
for a notebook or a report to print next to the numbers. It also takes a
comparison table, and states the setup of the evaluations its rows name, and
a plan's `explain()` document, before execution. Invalid metadata raises
`EvaluationError`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Self, TypeGuard

import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import (
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from crc_sdk.impacts._types import ContentHash, StrictFrozenModel
from crc_sdk.parametric.errors import EvaluationError

__all__ = [
    "EVALUATION_METADATA_KEY",
    "EVALUATION_SCHEMA_VERSION",
    "ColumnEntry",
    "CoverEntry",
    "EvaluationMetadata",
    "EvaluationSetup",
    "setup_text",
]

EVALUATION_METADATA_KEY = "crc.parametric.evaluation"
"""The Parquet schema metadata key of a comparison table."""

EVALUATION_SCHEMA_VERSION = "1.0"
"""The `schema_version` of `EvaluationMetadata`."""

IDENTITY_COLUMNS = (
    "cover_label",
    "cover_hash",
    "design_id",
    "asset_id",
    "level",
    "scope",
    "scenario",
    "design_scenario",
    "pathway",
    "horizon",
    "product",
    "split",
    "fold",
    "sample_hash",
    "n_rows",
    "premium_basis",
    "unique",
    "ties",
    "degenerate_tail",
    "components",
    "design_budget",
)
"""The columns of a comparison table before its metric columns.

A comparison's rank-block columns (`_is_block_column`) follow them, and are
identity columns too.
"""

_ADDED_SCENARIO_COLUMNS = (
    "design_scenario",
    "pathway",
    "horizon",
    "product",
    "n_rows",
)
"""The identity columns of scenario analysis, null in a table written without them."""

_ADDED_IDENTITY_COLUMNS = (
    "scope",
    "components",
    "design_budget",
    *_ADDED_SCENARIO_COLUMNS,
)
"""The identity columns that a table written earlier doesn't have."""

_ADDED_METRIC_COLUMNS = ("payout_frequency", "unidentified_share")
"""The metric columns that a comparison written earlier doesn't have, in order."""

_BREAKDOWN_COLUMNS = ("part", "part_kind")
"""The columns of a payout breakdown beside the identity columns."""

_FRONTIER_COLUMNS = ("budget_ratio", "budget_amount", "budget_use", "objective_value")
"""The columns of a budget frontier beside the identity columns."""

_INTERVAL_COLUMNS = (
    "column",
    "metric",
    "parameter",
    "parameter_value",
    "unit",
    "value",
    "ci_low",
    "ci_high",
    "ci_level",
    "n_bootstrap",
    "n_undefined",
    "short_block",
)
"""The columns of an intervals table beside the identity columns."""

KEY_COLUMNS = (
    "cover_hash",
    "design_id",
    "asset_id",
    "level",
    "scenario",
    "design_scenario",
    "split",
    "fold",
)
"""The columns whose values identify a row of a comparison table."""


_REFERENCE_YEARS_COLUMN = "copula_reference_years"
"""The column of m, the rows of an `EmpiricalCopula` reference."""

_BLOCK_TAIL_PREFIX = "copula_block_tail_b"
"""The start of a rank-block flag's name; the metrics' label of beta follows."""

_BLOCK_TAIL_TOLERANCE = 1e-9
"""The rounding allowance of the rank-block flag, the one of `degenerate_tail`."""

_SUMMED_LEVELS = frozenset({"total", "asset_by_value", "asset_by_loss"})
"""The levels whose rows read a sum over a sample's reordered columns."""


def _is_block_column(name: str) -> bool:
    """Whether `name` is `copula_reference_years` or a rank-block flag."""
    return name == _REFERENCE_YEARS_COLUMN or name.startswith(_BLOCK_TAIL_PREFIX)


def _tail_in_one_block(beta: float, years: int) -> bool:
    """Whether the tail at `beta` holds at most one of `years` reference blocks."""
    return (1.0 - beta) * years <= 1.0 + _BLOCK_TAIL_TOLERANCE


def _reference_years(
    level: str, *, dependence: Any, hazard_dependence: Any, components: bool
) -> int | None:
    """m where a row at `level` reads a sum over rank blocks, else `None`.

    Every row of a sample of several hazards reads the combined loss, and on a
    sample of several assets the `"total"` and allocated rows read a sum. A
    record of an `EmpiricalCopula` without `m` has none to state. Raises
    `EvaluationError` for an `m` that is not a positive whole number.
    """
    if components:
        model = hazard_dependence
    else:
        model = dependence if level in _SUMMED_LEVELS else None
    if not isinstance(model, Mapping) or model.get("kind") != "empirical_copula":
        return None
    years = model.get("m")
    if years is None:
        return None
    if isinstance(years, bool) or not isinstance(years, int) or years < 1:
        raise EvaluationError(
            f"an EmpiricalCopula's record must state m as a positive whole number, "
            f"got {years!r}"
        )
    return years


def _identity_columns(names: Iterable[str]) -> list[str]:
    """The identity columns among `names`, those added since optional.

    A comparison's rank-block flags, a budget frontier's four columns and a
    payout breakdown's `part` and `part_kind` are identity columns too.
    """
    listed = list(names)
    present = set(listed)
    return [
        *(
            name
            for name in IDENTITY_COLUMNS
            if name in present or name not in _ADDED_IDENTITY_COLUMNS
        ),
        *(name for name in listed if _is_block_column(name)),
        *(name for name in _FRONTIER_COLUMNS if name in present),
        *(name for name in _BREAKDOWN_COLUMNS if name in present),
    ]


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def _plain(name: str, value: Any) -> None:
    try:
        _json_text(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must hold plain, finite JSON values: {error}"
        ) from error


def _evaluation_error(error: ValidationError) -> EvaluationError:
    """Turn a pydantic validation error into an `EvaluationError`."""
    messages: list[str] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        original = (item.get("ctx") or {}).get("error")
        message = str(original) if isinstance(original, Exception) else item["msg"]
        messages.append(f"{location}: {message}" if location else message)
    return EvaluationError(f"invalid {error.title}: {'; '.join(messages)}")


class _EvaluationModel(StrictFrozenModel):
    """Frozen model whose validation failures raise `EvaluationError`."""

    # Hidden from type checkers, so that they keep the fields' signature.
    if not TYPE_CHECKING:

        def __init__(self, **data: Any) -> None:
            try:
                super().__init__(**data)
            except ValidationError as error:
                raise _evaluation_error(error) from error


def _unique(entries: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The entries without repeats, in first-seen order."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for entry in entries:
        text = _json_text(entry)
        if text not in seen:
            seen.add(text)
            kept.append(dict(entry))
    return tuple(kept)


def _replaced(model: Any, **fields: Any) -> Any:
    """`model` with `fields` replaced, validated again."""
    return type(model)(**{**dict(model), **fields})


def _merged_intervals(
    records: Sequence[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """One `setup.intervals` for several tables; `None` if none has any."""
    stated = [record for record in records if record is not None]
    if not stated:
        return None
    first = stated[0]
    agreed = ("ci_level", "method", "n_bootstrap", "unit")
    for record in stated[1:]:
        if any(record.get(key) != first.get(key) for key in agreed):
            raise EvaluationError(
                "the metadata differ in their intervals: their method, unit, number "
                "of resamples or level"
            )
    flagged = dict.fromkeys(
        column
        for record in stated
        for column in record.get("short_block_columns") or ()
    )
    return {**first, "short_block_columns": list(flagged)}


_FRONTIER_KEYS = frozenset(
    {
        "budgets",
        "design_ids",
        "design_sample",
        "fold",
        "objective",
        "sample_hash",
        "split",
        "template_hash",
    }
)
"""The keys of one entry of `setup.frontier`."""


def _check_frontier_record(record: Mapping[str, Any]) -> None:
    """Raise `ValueError` unless a `setup.frontier` entry has what is read."""
    missing = sorted(_FRONTIER_KEYS - set(record))
    if missing:
        raise ValueError(f"a setup.frontier entry lacks {missing}")
    budgets, designs = record["budgets"], record["design_ids"]
    if not isinstance(budgets, list) or not isinstance(designs, list):
        raise ValueError("a setup.frontier entry's budgets and design_ids are lists")
    if len(budgets) != len(designs):
        raise ValueError("a setup.frontier entry needs one design per budget")


def _merged_frontier(
    records: Sequence[tuple[dict[str, Any], ...] | None],
) -> tuple[dict[str, Any], ...] | None:
    """One `setup.frontier` for several tables; `None` if none has any."""
    return _unique(entry for record in records for entry in record or ()) or None


_BLOCK_FIELDS = ("n_rows", "blocks")
"""The fields of a scenario in `setup.scenarios` that differ between folds."""


def _check_scenarios_record(record: Mapping[str, Any]) -> None:
    """Raise `ValueError` unless `setup.scenarios` has the keys that are read."""
    entries = record.get("scenarios")
    if not isinstance(record.get("reference"), str) or not isinstance(entries, Mapping):
        raise ValueError("setup.scenarios needs a reference and its scenarios")
    if record["reference"] not in entries:
        raise ValueError("setup.scenarios must describe its reference")
    for label, entry in entries.items():
        blocks = entry.get("blocks") if isinstance(entry, Mapping) else None
        if not isinstance(blocks, list) or not all(
            isinstance(block, Mapping) and {"n_rows", "sample_hash"} <= set(block)
            for block in blocks
        ):
            raise ValueError(f"setup.scenarios[{label!r}] needs its blocks")


def _without_blocks(record: Mapping[str, Any]) -> dict[str, Any]:
    """`record` without what differs between the folds of one set."""
    return {
        **record,
        "scenarios": {
            label: {
                key: value for key, value in entry.items() if key not in _BLOCK_FIELDS
            }
            for label, entry in record["scenarios"].items()
        },
    }


def _merged_scenarios(
    records: Sequence[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """One `setup.scenarios` for several tables; `None` if none has any.

    Every table must state one set of scenarios. The folds of a set differ in
    their blocks only: the blocks are combined without repeats, and a scenario's
    `n_rows` is the sum over them.
    """
    if all(record is None for record in records):
        return None
    if any(record is None for record in records):
        raise EvaluationError(
            "the metadata differ in their scenarios: every table of a merge states "
            "one setup.scenarios, or none does"
        )
    stated = [record for record in records if record is not None]
    if len({_json_text(_without_blocks(record)) for record in stated}) > 1:
        raise EvaluationError(
            "the metadata differ in their scenarios: only the blocks of one set of "
            "scenarios may differ between tables"
        )
    first = stated[0]
    merged: dict[str, Any] = {}
    for label, entry in first["scenarios"].items():
        blocks = list(
            _unique(
                block
                for record in stated
                for block in record["scenarios"][label]["blocks"]
            )
        )
        merged[label] = {
            **entry,
            "blocks": blocks,
            "n_rows": sum(int(block["n_rows"]) for block in blocks),
        }
    return {**first, "scenarios": merged}


def _merged_drift(records: Sequence[dict[str, Any] | None]) -> dict[str, Any] | None:
    """One `setup.drift` for several tables; `None` if none has any."""
    if len({_json_text(record) for record in records}) > 1:
        raise EvaluationError(
            "the metadata differ in their drift: they take differences from different "
            "references"
        )
    return records[0]


class ColumnEntry(_EvaluationModel):
    """One metric column: its metric, parameter, unit and definition."""

    metric: str
    parameter: dict[str, float] | None
    unit: str
    definition: str


class CoverEntry(_EvaluationModel):
    """One cover: its label, its `definition_hash` and its cover document."""

    label: str | None
    hash: ContentHash
    document: dict[str, Any]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _plain("a cover document", self.document)
        return self


class EvaluationSetup(_EvaluationModel):
    """The setup of a comparison: basis, metric options, samples, evaluations.

    `samples` holds one document per sample (each with its `sample_hash`
    and `roles`, `"evaluation"` and/or `"design"`), and `evaluations` one per
    evaluation run (each with its `sample_hash`, `split` and `fold`, and, in
    what `compare` and `evaluate` write, the `level` and `scope` of its rows
    and its `allocation`). `intervals` is what `bootstrap_intervals` records,
    and null on any other table. `scenarios` is what `compare_scenarios` records,
    `drift` what `to_drift` records and `frontier` what `design_frontier` records,
    one entry per call; each is null on any other table and left out of its
    document, so such a table's document is what it was before they existed.
    Raises `EvaluationError` for a sample or an evaluation without those keys.
    """

    basis: Literal["annual"] = "annual"
    metric_options: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    intervals: dict[str, Any] | None = None
    scenarios: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None
    frontier: tuple[dict[str, Any], ...] | None = None

    @model_serializer(mode="wrap")
    def _without_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        document: dict[str, Any] = handler(self)
        for key in ("scenarios", "drift", "frontier"):
            if document.get(key) is None:
                document.pop(key, None)
        return document

    @model_validator(mode="after")
    def _check(self) -> Self:
        _plain("setup", self.model_dump(mode="json"))
        for sample in self.samples:
            if "sample_hash" not in sample or "roles" not in sample:
                raise ValueError("every setup sample needs a sample_hash and roles")
        for evaluation in self.evaluations:
            if not {"sample_hash", "split", "fold"} <= set(evaluation):
                raise ValueError(
                    "every setup evaluation needs a sample_hash, a split and a fold"
                )
        if self.scenarios is not None:
            _check_scenarios_record(self.scenarios)
        for record in self.frontier or ():
            _check_frontier_record(record)
        return self


class EvaluationMetadata(_EvaluationModel):
    """The metadata of a comparison table, under `crc.parametric.evaluation`.

    `compare` and `evaluate` build it and attach it to their table;
    `from_table` reads it back, `merge` combines the metadata of several
    tables, `write_comparison` checks it against the table before writing,
    and `setup_text` states its setup in words. Raises `EvaluationError` for
    invalid fields and values that are not plain, finite JSON.
    """

    schema_version: Literal["1.0"] = "1.0"
    producer: Literal["crc_sdk.parametric"] = "crc_sdk.parametric"
    creation_version: str
    crc_framework_version: str | None
    numpy_version: str
    platform: str
    created_at: str
    solvers: tuple[dict[str, Any], ...] = ()
    setup: EvaluationSetup
    columns: dict[str, ColumnEntry]
    covers: tuple[CoverEntry, ...]
    designs: tuple[dict[str, Any], ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    source_records: tuple[dict[str, Any], ...] = ()
    assets: dict[str, Any]
    licences: tuple[dict[str, Any], ...] = ()
    rng: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name in (
            "solvers",
            "designs",
            "sources",
            "source_records",
            "assets",
            "licences",
            "rng",
        ):
            _plain(name, getattr(self, name))
        ids = [design.get("design_id") for design in self.designs]
        if not all(isinstance(value, str) for value in ids):
            raise ValueError("every design record needs a design_id")
        if len(set(ids)) != len(ids):
            raise ValueError("design records repeat a design_id")
        return self

    def to_json(self) -> str:
        """Return the metadata as JSON text with sorted keys."""
        return _json_text(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, text: str | bytes) -> EvaluationMetadata:
        """Read the metadata from its JSON text. Raises `EvaluationError`."""
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"evaluation metadata is not valid JSON: {error}"
            ) from error
        if not isinstance(document, dict):
            raise EvaluationError(
                "evaluation metadata must be a JSON object, got "
                f"{type(document).__name__}"
            )
        return cls(**document)

    @classmethod
    def from_table(cls, table: Any) -> EvaluationMetadata:
        """Read the metadata a comparison table carries.

        Raises `EvaluationError` if the table's schema has no
        `crc.parametric.evaluation` key, or its value is invalid.
        """
        metadata = getattr(getattr(table, "schema", None), "metadata", None) or {}
        text = metadata.get(EVALUATION_METADATA_KEY.encode("utf-8"))
        if text is None:
            raise EvaluationError(
                f"the table has no {EVALUATION_METADATA_KEY!r} metadata; build it "
                "with compare or evaluate"
            )
        return cls.from_json(text)

    @classmethod
    def merge(cls, *items: EvaluationMetadata) -> EvaluationMetadata:
        """Combine the metadata of several comparison tables into one.

        Use it with `pyarrow.concat_tables` to write the rows of several
        `compare` calls (in and out of sample, several folds or assets) into
        one file. The environments, the basis and the metric options must
        agree, and a column or design id must mean the same in each; samples,
        evaluations, covers, designs, sources, source records, licences and
        generator states are combined without repeats. Intervals of several
        tables must have one method, unit, number of resamples and level, and
        their flagged columns are combined. Tables of `compare_scenarios` must
        state one set of scenarios, of which only the blocks may differ, and
        those are combined. The entries of `setup.frontier` are combined without
        repeats. `created_at` is the latest one.

        Raises `EvaluationError` if the items disagree, and `TypeError` for an
        item that is not `EvaluationMetadata`.
        """
        if not items:
            raise EvaluationError("merge needs at least one EvaluationMetadata")
        for item in items:
            if not isinstance(item, EvaluationMetadata):
                raise TypeError(
                    f"merge takes EvaluationMetadata, got {type(item).__name__}"
                )
        first = items[0]
        for field in (
            "creation_version",
            "crc_framework_version",
            "numpy_version",
            "platform",
        ):
            values = {getattr(item, field) for item in items}
            if len(values) > 1:
                raise EvaluationError(
                    f"the metadata differ in {field}: {sorted(map(str, values))}"
                )
        options = {_json_text(item.setup.metric_options) for item in items}
        if len(options) > 1:
            raise EvaluationError(
                "the metadata differ in their metric options, so their tables "
                "have different columns"
            )
        columns: dict[str, ColumnEntry] = {}
        for item in items:
            for name, entry in item.columns.items():
                if name in columns and columns[name] != entry:
                    raise EvaluationError(
                        f"column {name!r} differs between the metadata (its unit "
                        f"{columns[name].unit!r} or {entry.unit!r}); write one table "
                        "per payout unit"
                    )
                columns.setdefault(name, entry)
        designs: dict[str, dict[str, Any]] = {}
        for item in items:
            for record in item.designs:
                design_id = record["design_id"]
                if design_id in designs and _json_text(
                    designs[design_id]
                ) != _json_text(record):
                    raise EvaluationError(
                        f"design {design_id!r} has two different records"
                    )
                designs.setdefault(design_id, record)
        samples: dict[str, dict[str, Any]] = {}
        for item in items:
            for sample in item.setup.samples:
                identity = _json_text({k: v for k, v in sample.items() if k != "roles"})
                if identity in samples:
                    roles = set(samples[identity]["roles"]) | set(sample["roles"])
                    samples[identity] = {**samples[identity], "roles": sorted(roles)}
                else:
                    samples[identity] = dict(sample)
        asset_ids = {
            asset_id for sample in samples.values() for asset_id in sample["asset_ids"]
        }
        # An evaluation without an allocation needs no asset values, so it
        # states none: a hash that another one states stands for both.
        stated = next(
            (
                item.assets["values_sha256"]
                for item in items
                if item.assets.get("values_sha256") is not None
            ),
            None,
        )

        def asset_record(item: EvaluationMetadata) -> dict[str, Any]:
            kept = {key: value for key, value in item.assets.items() if key != "count"}
            if stated is not None and kept.get("values_sha256") is None:
                kept["values_sha256"] = stated
            return kept

        assets = asset_record(first)
        for item in items[1:]:
            if _json_text(asset_record(item)) != _json_text(assets):
                raise EvaluationError("the metadata differ in their asset records")
        return cls(
            creation_version=first.creation_version,
            crc_framework_version=first.crc_framework_version,
            numpy_version=first.numpy_version,
            platform=first.platform,
            created_at=max(item.created_at for item in items),
            solvers=_unique(entry for item in items for entry in item.solvers),
            setup=EvaluationSetup(
                basis=first.setup.basis,
                metric_options=first.setup.metric_options,
                samples=tuple(samples.values()),
                evaluations=_unique(
                    entry for item in items for entry in item.setup.evaluations
                ),
                intervals=_merged_intervals([item.setup.intervals for item in items]),
                scenarios=_merged_scenarios([item.setup.scenarios for item in items]),
                drift=_merged_drift([item.setup.drift for item in items]),
                frontier=_merged_frontier([item.setup.frontier for item in items]),
            ),
            columns=columns,
            covers=tuple(
                {
                    _json_text(entry.model_dump(mode="json")): entry
                    for item in items
                    for entry in item.covers
                }.values()
            ),
            designs=tuple(designs.values()),
            sources=_unique(entry for item in items for entry in item.sources),
            source_records=_unique(
                entry for item in items for entry in item.source_records
            ),
            assets={"count": len(asset_ids), **assets},
            licences=_unique(entry for item in items for entry in item.licences),
            rng=_unique(entry for item in items for entry in item.rng),
        )

    def check_table(self, table: Any) -> None:
        """Check that this metadata describes `table`, a comparison table.

        The table must have the identity columns and exactly the metric
        columns of `columns`; a table written before `scope`, `components`,
        `design_budget`, `design_scenario`, `pathway`, `horizon`, `product`,
        `n_rows` or the rank-block columns were added may lack them, and a
        comparison written before `payout_frequency` or `unidentified_share` was
        added may lack that column. An intervals table has one row per wide row
        and metric column instead, and its `column` holds metric columns of
        `columns`. Its key
        (`cover_hash`, `design_id`, `asset_id`, `level`, `scenario`,
        `design_scenario`, `split`, `fold`, `part` on a payout breakdown and
        `column` on an intervals table, a column it lacks reading as null) must
        be unique; and every cover, design, sample and evaluation it names must
        be described here. Raises `EvaluationError` otherwise.
        """
        names = list(getattr(table, "column_names", ()))
        identity = _identity_columns(names)
        missing = [name for name in identity if name not in names]
        if missing:
            raise EvaluationError(f"the comparison table lacks the columns {missing}")
        intervals = "ci_low" in names
        if intervals:
            lacking = [name for name in _INTERVAL_COLUMNS if name not in names]
            if lacking:
                raise EvaluationError(
                    f"the intervals table lacks the columns {lacking}"
                )
            metrics = table.column("column").unique().to_pylist()
        else:
            metrics = [name for name in names if name not in identity]
        optional = () if "part" in names else _ADDED_METRIC_COLUMNS
        unknown = sorted(set(metrics) - set(self.columns))
        # The rows of an intervals table may be filtered by metric.
        absent = (
            []
            if intervals
            else sorted(set(self.columns) - set(metrics) - set(optional))
        )
        if unknown or absent:
            raise EvaluationError(
                "the table's metric columns differ from the metadata's: "
                f"undescribed {unknown}, absent {absent}"
            )
        rows = table.select(
            [*identity, "column"] if intervals else identity
        ).to_pylist()
        key_columns = (*KEY_COLUMNS, "part") if "part" in names else KEY_COLUMNS
        if intervals:
            key_columns = (*key_columns, "column")
        keys = [tuple(row.get(name) for name in key_columns) for row in rows]
        if len(set(keys)) != len(keys):
            raise EvaluationError("the comparison table repeats a row key")
        covers = {entry.hash for entry in self.covers}
        designs = {record["design_id"] for record in self.designs}
        samples = {sample["sample_hash"] for sample in self.setup.samples}
        evaluations = {
            (entry["sample_hash"], entry["split"], entry["fold"])
            for entry in self.setup.evaluations
        }
        for row in rows:
            if row["cover_hash"] not in covers:
                raise EvaluationError(
                    f"cover {row['cover_hash']} is not in the metadata"
                )
            if row["design_id"] is not None and row["design_id"] not in designs:
                raise EvaluationError(
                    f"design {row['design_id']} is not in the metadata"
                )
            if row["sample_hash"] not in samples:
                raise EvaluationError(
                    f"sample {row['sample_hash']} is not in the metadata"
                )
            if (row["sample_hash"], row["split"], row["fold"]) not in evaluations:
                raise EvaluationError(
                    f"no evaluation of sample {row['sample_hash']} "
                    f"({row['split']}, fold {row['fold']}) is in the metadata"
                )


# -- the setup in words --------------------------------------------------------


def _number(value: Any, where: str) -> str:
    """A finite number in at most six significant digits, without an exponent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise EvaluationError(f"{where} must be finite, got {value!r}")
    text = format(Decimal(repr(float(f"{value:.6g}") + 0.0)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, point, fraction = digits.partition(".")
    return f"{sign}{int(whole):,}{point}{fraction}"


def _numbers(values: Any, where: str) -> str:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EvaluationError(f"{where} must be a list of numbers")
    if not values:
        return "none"
    return ", ".join(_number(value, f"{where}[{i}]") for i, value in enumerate(values))


def _get(
    entry: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str
) -> Any:
    """`entry[key]` if it has the type `kind`; `None` when absent or null."""
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise EvaluationError(f"{where}.{key} has the wrong type: {value!r}")
    return value


def _list(entry: Mapping[str, Any], key: str, where: str) -> list[Any]:
    """`entry[key]` as a list; empty when absent or null."""
    values = entry.get(key)
    if values is None:
        return []
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise EvaluationError(f"{where}.{key} must be a list")
    return list(values)


def _objects(entry: Mapping[str, Any], key: str, where: str) -> list[Mapping[str, Any]]:
    values = _list(entry, key, where)
    if not all(isinstance(value, Mapping) for value in values):
        raise EvaluationError(f"{where}.{key} must be a list of objects")
    return values


def _join(words: Sequence[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    if len(words) <= 2:
        return " and ".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def _short(value: Any) -> str:
    """The first 12 hex digits of a `sha256:` hash."""
    return str(value).removeprefix("sha256:")[:12]


def _first_seen(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _years(sample: Mapping[str, Any] | None, where: str) -> str:
    """How many years a sample holds, which ones, and which block it is."""
    if sample is None:
        return "a sample the setup does not describe"
    kind = _get(sample, "source_kind", str, where)
    draws = _get(sample, "draws", int, where)
    periods = _list(sample, "periods", where)
    if periods and (
        len(periods) != 2
        or any(isinstance(p, bool) or not isinstance(p, int) for p in periods)
    ):
        raise EvaluationError(f"{where}.periods must be two integers")
    count = "an unknown number of" if draws is None else str(draws)
    span = f"{periods[0]} to {periods[1]}" if periods else ""
    if kind == "monte_carlo":
        text = f"{count} simulated years" + (f" (draws {span})" if span else "")
    elif kind == "annual" and _get(sample, "pool", Mapping, where) is not None:
        years = _list(sample["pool"], "years", f"{where}.pool")
        text = f"{count} pooled model-years" + (
            f" ({_year_spans(years)})" if years else ""
        )
    elif kind == "annual":
        text = f"{count} recorded years" + (f" ({span})" if span else "")
    else:
        text = f"{count} years" + (f" (periods {span})" if span else "")
    steps = _objects(sample, "selection", where)
    step = steps[-1] if steps else {}
    block = step.get("block")
    if step.get("kind") == "holdout" and block in ("train", "test"):
        text += ", the design block" if block == "train" else ", the held-out block"
    elif step.get("kind") == "kfold" and block in ("train", "test"):
        text += (
            ", every block but the held-out one"
            if block == "train"
            else ", the held-out block"
        )
    elif step.get("kind") == "select":
        text += ", a selection of periods"
    return text


def _source(sample: Mapping[str, Any], where: str) -> str:
    kind = _get(sample, "source_kind", str, where)
    if kind == "monte_carlo":
        semantics = _get(sample, "probability_semantics", str, where)
        stated = _get(sample, "semantics_source", str, where)
        who = {"dataset": "as the dataset declares", "caller": "as the caller states"}
        if semantics is None and stated == "dataset":
            # A plan's setup before execution: the dataset is not read yet.
            return (
                "Monte Carlo draws from a hazard curve, with the probability "
                "semantics the dataset declares, not known yet"
            )
        return (
            f"Monte Carlo draws from a hazard curve with {semantics or 'unstated'} "
            f"semantics, {who.get(stated or '', 'stated by no one recorded')}"
        )
    if kind == "annual":
        if _get(sample, "pool", Mapping, where) is not None:
            return "pooled per-year values, one row per model-year"
        return "per-year values, one row per recorded year"
    if kind == "table":
        return "rows of a user table, each a year as the caller states"
    return "a source of unknown kind"


def _basis_risk(sample: Mapping[str, Any], where: str) -> str:
    if "basis_risk" not in sample:
        return "not stated"
    model = sample["basis_risk"]
    if model is None:
        return "not modelled; the index and the loss are used as given"
    if not isinstance(model, Mapping):
        raise EvaluationError(f"{where}.basis_risk must be an object")
    kind = model.get("kind")
    if kind == "no_basis_risk":
        return "no model"
    if kind == "hazard_noise":
        sigma = _number(model.get("sigma"), f"{where}.basis_risk.sigma")
        return (
            "hazard noise, the hazard at the asset being the index times a mean-one "
            f"lognormal factor with sigma {sigma}, one per year and asset, as an "
            "assumption the caller states"
        )
    if kind == "impact_noise":
        sigma = _number(model.get("sigma"), f"{where}.basis_risk.sigma")
        text = f"impact noise, a mean-one lognormal factor with sigma {sigma}"
        if model.get("upper") is not None:
            upper = _number(model["upper"], f"{where}.basis_risk.upper")
            text += f", the noisy impact capped at {upper}"
        return text
    terms = [
        f"{key} {json.dumps(model[key], sort_keys=True)}"
        for key in sorted(model)
        if key != "kind"
    ]
    return str(kind) + (f" ({', '.join(terms)})" if terms else "")


def _states_hazard_noise(sample: Mapping[str, Any]) -> bool:
    """Whether the sample, or one of its assets, states `HazardNoise`."""
    parts = sample.get("parts") or {}
    models = [sample.get("basis_risk"), *(p.get("basis_risk") for p in parts.values())]
    return any(
        isinstance(model, Mapping) and model.get("kind") == "hazard_noise"
        for model in models
    )


def _hazard_noise_scope(sample: Mapping[str, Any]) -> str:
    """How the factors of `HazardNoise` relate across assets or components."""
    if not _states_hazard_noise(sample):
        return ""
    if sample.get("parts"):
        return ", the factors independent between assets"
    if sample.get("components"):
        return ", each component with factors of its own"
    return ""


def _has_mean_ratios(record: Any) -> bool:
    return isinstance(record, Mapping) and "hazard_mean_ratio" in record


def _recorded_mean_ratios(
    sample: Mapping[str, Any], where: str
) -> dict[str, Mapping[str, Any]]:
    """The records of the two mean ratios by name; `""` for a sample's own.

    `details.basis_risk` holds the record of a sample, or of each of its assets
    or components by name. A joined sample holds each asset's in that part.
    """
    details = _get(sample, "details", Mapping, where) or {}
    recorded = _get(details, "basis_risk", Mapping, f"{where}.details") or {}
    if _has_mean_ratios(recorded):
        return {"": recorded}
    named = {str(k): each for k, each in recorded.items() if _has_mean_ratios(each)}
    if named:
        return named
    for asset, part in (_get(sample, "parts", Mapping, where) or {}).items():
        own = _get(part, "details", Mapping, f"{where}.parts.{asset}") or {}
        if _has_mean_ratios(own.get("basis_risk")):
            named[str(asset)] = own["basis_risk"]
    return named


def _mean_ratios(sample: Mapping[str, Any], where: str) -> str:
    """What a model that acts on the hazard did to the means, in words.

    Blocks of one sample carry its ratios, which are over the rows as built.
    Above `_MANY_ASSETS` names the text gives the ranges.
    """
    named = _recorded_mean_ratios(sample, where)
    if not named:
        return ""
    place = f"{where}.details.basis_risk"

    def say(value: Any, at: str) -> str:
        return "not defined" if value is None else _number(value, at)

    lead = "; over the rows as built, with the model over without it, "
    if "" in named:
        own = named[""]
        hazard = say(own.get("hazard_mean_ratio"), f"{place}.hazard_mean_ratio")
        impact = say(own.get("impact_mean_ratio"), f"{place}.impact_mean_ratio")
        return f"{lead}the mean hazard is {hazard} and the mean impact {impact}"
    if len(named) <= _MANY_ASSETS:
        each = [
            f"{say(entry.get('hazard_mean_ratio'), f'{place}.{name}')} and "
            f"{say(entry.get('impact_mean_ratio'), f'{place}.{name}')} for {name}"
            for name, entry in named.items()
        ]
        return f"{lead}the mean hazard and the mean impact are {', '.join(each)}"

    def ranged(key: str) -> str:
        values = [entry[key] for entry in named.values() if entry.get(key) is not None]
        if not values:
            return "not defined"
        return _span(values, lambda value: _number(value, f"{place}.{key}"))

    who = "assets" if sample.get("parts") else "components"
    return (
        f"{lead}the mean hazard is {ranged('hazard_mean_ratio')} and the mean impact "
        f"{ranged('impact_mean_ratio')} across {len(named)} {who}"
    )


def _loss(sample: Mapping[str, Any], where: str) -> str:
    unit = _get(sample, "loss_unit", str, where) or "an unstated unit"
    if any(
        step.get("kind") == "component" and step.get("loss") == "combined"
        for step in _objects(sample, "selection", where)
    ):
        return f"the combined loss of the hazards, in {unit}"
    if "loss_model" not in sample:
        return f"model not stated, in {unit}"
    model = sample["loss_model"]
    if model is None:
        return f"from the table, in {unit}"
    if not isinstance(model, Mapping):
        raise EvaluationError(f"{where}.loss_model must be an object")
    place = f"{where}.loss_model"
    if model.get("kind") != "components":
        return _loss_model(model, unit, place)
    described: list[str] = []
    for i, entry in enumerate(_objects(model, "components", place)):
        at = f"{place}.components[{i}]"
        each = _get(entry, "loss_model", Mapping, at) or {}
        described.append(f"{entry.get('name')}: {_loss_model(each, unit, at)}")
    return "; ".join(described)


def _loss_model(model: Mapping[str, Any], unit: str, where: str) -> str:
    """One loss model's document in words, with the unit of its loss."""
    impact = _get(model, "impact", Mapping, where) or {}
    value = _get(model, "value", Mapping, where) or {}
    kind = {
        "expression": "impact expression",
        "function": "impact function",
        "callable": "callable impact",
    }.get(str(impact.get("kind")), "impact")
    text = f"{kind} {impact.get('name')}"
    if impact.get("definition_hash"):
        text += f" ({_short(impact['definition_hash'])})"
    if impact.get("exposure_unit"):
        text += f" for hazard values in {impact['exposure_unit']}"
    if value.get("source") == "column":
        text += f" times the asset value in column {value.get('column')}"
    elif value.get("source") == "constant":
        text += " times a stated asset value"
    else:
        text += ", the damage ratio"
    return f"{text}, in {unit}"


def _for_each_part(
    sample: Mapping[str, Any],
    where: str,
    describe: Callable[[Mapping[str, Any], str], str],
) -> str:
    """`describe` of a sample, or of each of its assets when they have parts.

    On a sample of several assets the loss and basis-risk models are per
    asset. Parts that read the same are written once.
    """
    parts = _get(sample, "parts", Mapping, where)
    if not parts:
        return describe(sample, where)
    texts = {
        str(asset): describe(
            {**part, "loss_unit": sample.get("loss_unit")}, f"{where}.parts.{asset}"
        )
        for asset, part in parts.items()
    }
    if len(set(texts.values())) == 1:
        return next(iter(texts.values()))
    return "; ".join(f"{asset}: {text}" for asset, text in texts.items())


def _dependence(
    sample: Mapping[str, Any], where: str, key: str = "dependence"
) -> str | None:
    """Where the dependence under `key` of a sample comes from, if stated."""
    model = _get(sample, key, Mapping, where)
    if model is None:
        return None
    kind = model.get("kind")
    if kind == "data":
        return "the data's own, the rows being the same years of the data"
    if kind == "independent":
        return "Independent(), a stated assumption"
    if kind == "comonotonic":
        return "Comonotonic(), a stated assumption"
    if kind == "empirical_copula":
        place = f"{where}.{key}"
        years = _get(model, "m", int, place)
        first = _get(model, "first_year", int, place)
        last = _get(model, "last_year", int, place)
        draws = _get(model, "k", int, place)
        row = "year" if _get(model, "pool", Mapping, place) is None else "model-year"
        return (
            f"EmpiricalCopula, a stated assumption: the ranks of {years} reference "
            f"{row}s, {first} to {last} ({_short(model.get('content_hash'))})"
            + ("" if draws is None else f", {draws} draws per {row}")
        )
    if kind == "gaussian_copula":
        correlation = model.get("correlation")
        order = ", in column order" if isinstance(correlation, list) else ""
        return (
            "GaussianCopula, a stated assumption: correlation "
            f"{json.dumps(correlation)}{order}"
        )
    return f"{kind}, a stated assumption"


_ANNUAL_MAXIMA = (
    "Two annual maxima of one year may come from different events: the rows are "
    "the same years, not the same events."
)
"""What per-year rows of several hazards are not, stated next to every number."""


def _components(sample: Mapping[str, Any], where: str) -> str | None:
    """A multi-hazard sample's components, its combined loss and its dependence."""
    entries = _objects(sample, "components", where)
    if not entries:
        return None
    described: list[str] = []
    for i, entry in enumerate(entries):
        place = f"{where}.components[{i}]"
        own = (
            "with a loss of its own"
            if _get(entry, "own_loss", bool, place)
            else "with no loss of its own"
        )
        described.append(
            f"{_get(entry, 'name', str, place)} (in {_get(entry, 'unit', str, place)}, "
            f"{_get(entry, 'tail', str, place)} tail, {own})"
        )
    text = f"Index components: {_join(described)}; the first is the sample's index."
    rule = _get(sample, "combine", Mapping, where) or {}
    statement = _get(rule, "statement", str, f"{where}.combine")
    if rule.get("kind") == "table":
        text += " Combined loss: as the table states it."
    elif statement is not None:
        text += f" Combined loss: {statement.rstrip().removesuffix('.')}."
    elif rule:
        text += f" Combined loss: {rule.get('kind')}, as stated."
    dependence = _dependence(sample, where, "hazard_dependence")
    if dependence is not None:
        text += f" Dependence between the hazards: {dependence}."
    model = _get(sample, "hazard_dependence", Mapping, where) or {}
    if model.get("kind") == "data":
        text += f" {_ANNUAL_MAXIMA}"
    if len({entry.get("tail") for entry in entries}) > 1:
        text += " With mixed tails, that dependence is between values, not severities."
    return text


def _sample_text(sample: Mapping[str, Any], where: str) -> str:
    basis_risk = _for_each_part(sample, where, _basis_risk)
    text = (
        f"{_source(sample, where)}. "
        f"Basis risk: {basis_risk}{_hazard_noise_scope(sample)}"
        f"{_mean_ratios(sample, where)}. "
        f"Loss: {_for_each_part(sample, where, _loss)}."
    )
    dependence = _dependence(sample, where)
    if dependence is not None:
        text += f" Dependence between the assets: {dependence}."
    components = _components(sample, where)
    if components is not None:
        text += f" {components}"
    return text


def _recorded(document: Mapping[str, Any], key: str, where: str) -> str | None:
    """`document[key]` as text; `None` when absent, null or empty."""
    value = _get(document, key, (str, int), where)
    if isinstance(value, bool):
        raise EvaluationError(f"{where}.{key} has the wrong type: {value!r}")
    return None if value is None or value == "" else str(value)


def _year_spans(years: Sequence[int]) -> str:
    """Years as runs, as in `1981 to 1988 and 1997 to 2020`."""
    spans: list[list[int]] = []
    for year in years:
        if spans and year == spans[-1][1] + 1:
            spans[-1][1] = year
        else:
            spans.append([year, year])
    return _join([str(a) if a == b else f"{a} to {b}" for a, b in spans])


def _trend_fit(fit: Mapping[str, Any], where: str) -> str:
    """One series' fit: its role, asset and run, its slope and intercept with units."""
    role = _get(fit, "role", str, where) or "series"
    component = _get(fit, "component", str, where)
    if component is not None:
        role += f" of component {component}"
    asset = _get(fit, "asset_id", str, where)
    model = _get(fit, "model", str, where)
    run = "" if model is None else f" ({model} {_get(fit, 'member', str, where)})"
    unit = _get(fit, "unit", str, where)
    per_year = f" {unit} per year" if unit else " per year"
    amount = f" {unit}" if unit else ""
    return (
        f"{role}{'' if asset is None else f' of {asset}'}{run} slope "
        f"{_number(fit.get('slope'), f'{where}.slope')}{per_year}, intercept "
        f"{_number(fit.get('intercept'), f'{where}.intercept')}{amount}"
    )


def _span(values: Sequence[float], say: Callable[[float], str]) -> str:
    """The range of `values` in words: `a to b`, or `a` when it is one value."""
    low, high = say(min(values)), say(max(values))
    return low if low == high else f"{low} to {high}"


def _trend_fits(fits: Sequence[Mapping[str, Any]], where: str) -> str:
    """Each series' fit in words; above `_MANY_ASSETS` assets, the ranges.

    A sample of several assets, as `Joint()` makes, has one fit per asset and
    run, so its line would grow with assets times runs. Above `_MANY_ASSETS`
    assets each role and unit states the range of its slopes and intercepts
    and the series they range over; the sample's trend record keeps every fit.
    """
    stated = [_trend_fit(fit, f"{where}.fits[{i}]") for i, fit in enumerate(fits)]
    assets = _first_seen(
        str(fit["asset_id"]) for fit in fits if fit.get("asset_id") is not None
    )
    if len(assets) <= _MANY_ASSETS:
        return "; ".join(stated)
    groups: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for fit in fits:
        key = (fit.get("role") or "series", fit.get("unit"))
        groups.setdefault(key, []).append(fit)

    def say(value: float) -> str:
        return _number(value, where)

    parts: list[str] = []
    for (role, unit), group in groups.items():
        slopes = _span([fit["slope"] for fit in group], say)
        intercepts = _span([fit["intercept"] for fit in group], say)
        owners = len({fit.get("asset_id") for fit in group})
        parts.append(
            f"{role} slope {slopes}{f' {unit}' if unit else ''} per year and "
            f"intercept {intercepts}{f' {unit}' if unit else ''}, over "
            f"{len(group)} series of {owners} assets"
        )
    return f"{'; '.join(parts)}; the sample's trend record holds each series' fit"


def _trend_text(record: Mapping[str, Any], where: str) -> str:
    """One trend record of a per-year sample in words, as far as it is known."""
    kind = record.get("kind")
    if kind == "as_recorded":
        return "the years are used as recorded; no trend is removed"
    if kind == "recent_years":
        n = _get(record, "n", int, where)
        years = _list(record, "years", where)
        span = f" ({years[0]} to {years[1]})" if len(years) == 2 else ""
        kept = f"only the last {n} years" if n is not None else "only the last years"
        return f"{kept} of the series are kept{span}, as recorded; no value changes"
    if kind != "linear_trend":
        return f"{kind}, as stated"
    removed = (
        "an additive linear trend in the mean is removed, by ordinary least "
        "squares on the year"
    )
    reference = _get(record, "reference_year", int, where)
    fit_years = _list(record, "fit_years", where)
    fits = _objects(record, "fits", where)
    if not fit_years or not fits:
        # A plan before execution: each fold's fit is not made yet.
        at = reference if reference is not None else "the last year of the series"
        return (
            f"{removed}, with reference year {at} and at least "
            f"{_get(record, 'min_years', int, where)} fit years; the fit years, "
            "slope and intercept are not known yet"
        )
    return (
        f"{removed}, fitted on {len(fit_years)} years ({_year_spans(fit_years)}), "
        f"reference year {reference}, where each intercept is the fitted value: "
        f"{_trend_fits(fits, where)}"
    )


def _trend_lines(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line for each way the per-year samples treat the trend.

    Every per-year sample has one. Its record is under `trend`; a null or
    absent one reads as the years as recorded. Samples that treat the trend
    alike share a line; lines that differ name their assets, or count them
    above `_MANY_ASSETS`.
    """
    stated: dict[str, list[str]] = {}
    for i, sample in enumerate(samples):
        where = f"setup.samples[{i}]"
        if _get(sample, "source_kind", str, where) != "annual":
            continue
        record = _get(sample, "trend", Mapping, where)
        text = _trend_text(record or {"kind": "as_recorded"}, f"{where}.trend")
        stated.setdefault(text, []).extend(
            str(asset) for asset in _list(sample, "asset_ids", where)
        )
    if len(stated) == 1:
        return [f"Trend: {next(iter(stated))}."]
    return [
        f"Trend of {_named_assets(_first_seen(assets))}: {text}."
        if assets
        else f"Trend: {text}."
        for text, assets in stated.items()
    ]


_POOL_STATEMENTS = (
    "A draw mixes model spread with year-to-year variability. The rows are not "
    "independent years: rows of one run share that model's climate and rows of one "
    "year the pathway's forcing, so the number of rows is not an effective sample "
    "size."
)
"""What a pool's rows are not, stated next to every number made from them."""


def _pool_head(pool: Mapping[str, Any], where: str) -> str:
    """A pool's runs and pathway in words; before execution, what will name them."""
    runs = _objects(pool, "runs", where)
    if not runs:
        columns = [
            _get(pool, key, str, where) or "not stated"
            for key in ("model_column", "member_column", "pathway_column")
        ]
        return (
            "model-years of several runs are pooled for each asset, every run with "
            f"the same weight; a run is named by the columns {_join(columns)}, or by "
            "its source plan; the runs, the pathway and the years are not known yet"
        )
    names = [
        f"{_get(run, 'model', str, where)} {_get(run, 'member', str, where)}"
        for run in runs
    ]
    return (
        f"{len(runs)} runs under pathway {_get(pool, 'pathway', str, where)} "
        f"({'; '.join(names)}), with the same years and the same weight each"
    )


def _pool_lines(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per pool: its runs, pathway, years and rows, and what they are not.

    The blocks of a split share a pool's line, and its years are those of all
    its blocks; each `Split` line states its own. Pools of the same runs share
    a line too, and lines that differ name their assets.
    """
    stated: dict[str, dict[str, Any]] = {}
    for i, sample in enumerate(samples):
        where = f"setup.samples[{i}]"
        pool = _get(sample, "pool", Mapping, where)
        if pool is None:
            continue
        place = f"{where}.pool"
        years = [int(year) for year in _list(pool, "years", place)]
        count = len(_objects(pool, "runs", place))
        group = stated.setdefault(
            _pool_head(pool, place),
            {"assets": [], "count": count, "whole": True, "years": set()},
        )
        group["assets"] += [str(asset) for asset in _list(sample, "asset_ids", where)]
        group["years"] |= set(years)
        rows = _get(pool, "rows", int, place)
        group["whole"] &= rows is None or rows == count * len(years)
    lines: list[str] = []
    for head, group in stated.items():
        known = sorted(group["years"])
        size = ""
        if known:
            size = f": {len(known)} years ({_year_spans(known)})"
            if group["whole"]:
                size += f", {group['count'] * len(known)} rows"
            else:
                size += ", on a selection of the rows"
        label = (
            "Pool"
            if len(stated) == 1 or not group["assets"]
            else f"Pool of {_join(_first_seen(group['assets']))}"
        )
        lines.append(f"{label}: {head}{size}. {_POOL_STATEMENTS}")
    return lines


_CMIP_COLUMNS = ("model", "member", "scenario")
"""The columns that make a per-year table's `scenario` a CMIP6 pathway."""


def _series_pathway(
    series: Mapping[str, Any], document: Mapping[str, Any], where: str
) -> str | None:
    """A per-year series' pathway: its source's, else its rows' one scenario.

    A CMIP6 plan's `explain()` document states `pathway`; on the plan route
    the source is the caller's, and the plan's pathway is recorded beside it,
    on the series. A table of CMIP6 runs, such as NEX-GDDP-CMIP6's
    `annual_extremes()`, has `model`, `member` and `scenario` columns
    instead, and the `scenario` of the series' rows is the pathway when they
    hold one. A table with only a `scenario` column states no pathway: there
    the word need not mean one.
    """
    stated = _recorded(document, "pathway", where) or _recorded(
        series, "pathway", f"{where}.series"
    )
    if stated is not None:
        return stated
    columns = _get(series, "table_columns", Mapping, f"{where}.series") or {}
    if not all(name in columns for name in _CMIP_COLUMNS):
        return None
    place = f"{where}.series.table_columns"
    scenario = _get(columns, "scenario", Mapping, place) or {}
    values = _list(scenario, "values", f"{place}.scenario")
    if len(values) != 1:
        return None
    return _recorded({"scenario": values[0]}, "scenario", place)


def _canonical_place(path: str | None) -> str:
    """A canonical dataset in words, by its path if one is recorded."""
    if path is None:
        return "a canonical dataset, its path not recorded"
    return f"the canonical dataset {path}"


def _canonical_files(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """The canonical files of a comparison's `source_records`, by their SHA-256."""
    files: dict[str, Mapping[str, Any]] = {}
    for i, record in enumerate(records):
        where = f"source_records[{i}]"
        file = _get(record, "canonical_file", Mapping, where)
        if file is None:
            continue
        checksum = _recorded(file, "file_sha256", f"{where}.canonical_file")
        if checksum is not None:
            files[checksum] = file
    return files


def _canonical_file(
    entry: Mapping[str, Any], files: Mapping[str, Mapping[str, Any]], where: str
) -> str:
    """Which canonical file a curve was read from, if a plan recorded it.

    A plan's curve names its file by `canonical_file_sha256`, and the file's
    path is in the `source_records` entry of that checksum.
    """
    checksum = _recorded(entry, "canonical_file_sha256", where)
    if checksum is None:
        return ""
    path = _recorded(files.get(checksum, {}), "path", f"{where}.canonical_file")
    digest = checksum.removeprefix("sha256:")
    return f", read from {_canonical_place(path)}, SHA-256 {digest}"


def _source_record(
    entry: Mapping[str, Any],
    where: str,
    files: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """One source record: what it is, and its provider, dataset, horizon, pathway.

    A curve's provider and dataset come from its dataset metadata's `source`
    block, and its hazard, horizon and pathway from its canonical row; a
    per-year series' provider and dataset from its provenance's `source`, and
    its pathway as `_series_pathway` says. A field the record does not hold
    is said to be not recorded. A curve that a plan read also names its
    canonical file, the path and the SHA-256 that `files` holds of it.
    """
    role = _recorded(entry, "role", where) or "source"
    row = _get(entry, "row", Mapping, where) or {}
    pathway = _recorded(row, "pathway", f"{where}.row")
    file = ""
    document: Mapping[str, Any]
    if entry.get("kind") == "curve":
        metadata = _get(entry, "metadata", Mapping, where) or {}
        document = _get(metadata, "source", Mapping, f"{where}.metadata") or {}
        head = f"the {role} curve"
        hazard = _recorded(row, "hazard_name", f"{where}.row")
        if hazard is not None:
            head += f" {hazard}"
        file = _canonical_file(entry, files or {}, where)
    elif "series" in entry:
        series = _get(entry, "series", Mapping, where) or {}
        if isinstance(series.get("source"), str):
            document = {"name": series["source"]}
        else:
            document = _get(series, "source", Mapping, f"{where}.series") or {}
        head = f"the {role} series"
        model = _recorded(entry, "model", where)
        if model is not None:
            head += f" of {model} {_recorded(entry, 'member', where)}"
        if pathway is None:
            pathway = _recorded(entry, "pathway", where) or _series_pathway(
                series, document, where
            )
    else:
        document = entry
        head = f"the {role}"
    component = _recorded(entry, "component", where)
    if component is not None:
        head += f" of component {component}"
    name = _recorded(document, "name", where)
    if name is not None:
        head += f' "{name}"'
    fields = {
        "provider": _recorded(document, "provider", where),
        "dataset": _recorded(document, "dataset", where),
        "horizon": _recorded(row, "horizon", f"{where}.row"),
        "pathway": pathway,
    }
    known = [f"{key} {value}" for key, value in fields.items() if value is not None]
    missing = [key for key, value in fields.items() if value is None]
    text = head + (f" from {_join(known)}" if known else "")
    if missing:
        text += f", with {_join(missing)} not recorded"
    return text + file


_SELECTION = (
    ("hazard_names", "hazard"),
    ("horizons", "horizon"),
    ("pathways", "pathway"),
)
"""The axes of a canonical dataset's selection, and their nouns."""


def _selected(selection: Mapping[str, Any], key: str, noun: str, where: str) -> str:
    """One axis of a dataset's selection: `hazard rx1day`, `any horizon`."""
    if selection.get(key) is None:
        return f"any {noun}"
    words: list[str] = []
    for i, value in enumerate(_list(selection, key, where)):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise EvaluationError(f"{where}.{key}[{i}] has the wrong type: {value!r}")
        words.append(str(value))
    if not words:
        return f"no {noun}"
    return f"{noun if len(words) == 1 else noun + 's'} {_join(words)}"


def _planned_source(entry: Mapping[str, Any], where: str) -> str:
    """One source of a plan's `explain()` document, as far as it is known.

    A canonical dataset of fitted curves is named by its path and the
    hazards, horizons and pathways selected from it; its provider and dataset
    are in its metadata, which only execution reads. Per-year values are
    named by the provider and dataset the plan states, and by the file,
    table or source plan they are read from. Any other record is described
    as a comparison's source records are.
    """
    kind = _get(entry, "kind", str, where)
    if kind == "canonical_dataset":
        place = _canonical_place(_recorded(entry, "path", where))
        selection = _get(entry, "selection", Mapping, where)
        if selection is None:
            chosen = "its selection not recorded"
        else:
            chosen = "for " + _join(
                [
                    _selected(selection, key, noun, f"{where}.selection")
                    for key, noun in _SELECTION
                ]
            )
        return (
            f"the hazard curves of {place}, {chosen}; their provider and dataset "
            "are not known yet"
        )
    provenance = _get(entry, "provenance", Mapping, where)
    if kind not in ("file", "table", "plan") or provenance is None:
        return _source_record(entry, where)
    text = _source_record({"role": "hazard", "series": {"source": provenance}}, where)
    if kind == "file":
        path = _recorded(entry, "path", where)
        return f"{text}, read from a file{'' if path is None else f' {path}'}"
    if kind == "table":
        rows = _get(entry, "rows", int, where)
        size = "" if rows is None else f" of {rows} row{'' if rows == 1 else 's'}"
        return f"{text}, read from a table{size}"
    name = _recorded(entry, "type", where)
    return f"{text}, read from a source plan{'' if name is None else f' {name}'}"


def _sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    planned: bool = False,
    records: Sequence[Mapping[str, Any]] = (),
) -> str:
    """The `Source` line; `records` are the comparison's `source_records`."""
    files = _canonical_files(records)
    stated = _first_seen(
        _planned_source(entry, f"sources[{i}]")
        if planned
        else _source_record(entry, f"sources[{i}]", files)
        for i, entry in enumerate(sources)
    )
    return f"Source: {'; '.join(stated) or 'not recorded'}."


_LOCATED_BY: dict[str | None, str] = {
    None: "a point, in the nearest grid cell",
    "h3_cell": "the asset's H3 cell, in the grid cell that holds it",
}
"""A nearest-cell series' `asset_location` in words; absent, it had a point."""


def _degrees(value: float) -> str:
    """Degrees as recorded, without rounding a grid cell's centre to six digits."""
    return f"{value:.10g}"


def _latitude_longitude(
    series: Mapping[str, Any], key: str, where: str
) -> tuple[float, float]:
    """`series[key]`, a mapping of a latitude and a longitude, as two numbers."""
    pair = _get(series, key, Mapping, where) or {}
    place = f"{where}.{key}"
    _number(pair.get("latitude"), f"{place}.latitude")
    _number(pair.get("longitude"), f"{place}.longitude")
    return float(pair["latitude"]), float(pair["longitude"])


def _series_owner(
    entry: Mapping[str, Any], samples: Mapping[str, Mapping[str, Any]], where: str
) -> str:
    """The asset of a `sources` entry, from the entry or its sample of one asset."""
    owner = _recorded(entry, "asset_id", where)
    if owner is not None:
        return owner
    sample = samples.get(str(entry.get("sample_hash")), {})
    assets = _list(sample, "asset_ids", f"{where}.sample")
    return str(assets[0]) if len(assets) == 1 else "an asset not stated"


def _location_lines(
    sources: Sequence[Mapping[str, Any]], samples: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """One line per way nearest-cell series were located, with the cells matched.

    A series read at the nearest grid cell records the cell's centre, the grid
    step and, when its asset was located by an H3 cell, `asset_location`. It
    records neither the asset's point nor its distance to the cell, so the
    text states neither. Series of one role, located alike on one grid step,
    share a line that names each asset and its cell or, above `_MANY_ASSETS`,
    counts the assets and cells and gives the range of the centres. An asset
    with several cells for one role, as on the different grids of a pool's
    models, has each run named. Other series record no cell and have no line,
    but a component's series read from the rows of a stated filter has one, which
    says where it is read.
    """
    groups: dict[tuple[str, str, str], dict[tuple[str, float, float], list[str]]] = {}
    cells: dict[tuple[str, str], set[tuple[float, float]]] = {}
    filtered: list[str] = []
    for i, entry in enumerate(sources):
        where = f"sources[{i}]"
        series = _get(entry, "series", Mapping, where)
        if series is None:
            continue
        place = f"{where}.series"
        role = f"{_recorded(entry, 'role', where) or 'source'} series"
        component = _recorded(entry, "component", where)
        if component is not None:
            role += f" of component {component}"
        if series.get("kind") != "nearest_cell":
            chosen = _get(series, "where", Mapping, place)
            if component is not None and chosen:
                stated = _join([f"{key} = {chosen[key]}" for key in sorted(chosen)])
                filtered.append(f"Location: {role}, the rows where {stated}.")
            continue
        located = _get(series, "asset_location", str, place)
        how = _LOCATED_BY.get(located) or f"{located}, in a matched grid cell"
        latitude, longitude = _latitude_longitude(series, "cell", place)
        steps = _latitude_longitude(series, "spacing_degrees", place)
        step = f"{_degrees(steps[0])} by {_degrees(steps[1])}"
        owner = _series_owner(entry, samples, where)
        model = _recorded(entry, "model", where)
        run = "" if model is None else f"{model} {_recorded(entry, 'member', where)}"
        group = groups.setdefault((role, how, step), {})
        group.setdefault((owner, latitude, longitude), []).append(run)
        cells.setdefault((role, owner), set()).add((latitude, longitude))
    lines: list[str] = []
    for (role, how, step), matched in groups.items():
        owners = _first_seen(owner for owner, _, _ in matched)
        if len(owners) > _MANY_ASSETS:
            places = {(latitude, longitude) for _, latitude, longitude in matched}
            latitudes = _span([latitude for latitude, _ in places], _degrees)
            longitudes = _span([longitude for _, longitude in places], _degrees)
            which = (
                f"{len(owners)} assets in {len(places)} cells, with centres at "
                f"latitudes {latitudes} and longitudes {longitudes}"
            )
        else:
            named: list[str] = []
            for (owner, latitude, longitude), runs in matched.items():
                several = all(runs) and len(cells[(role, owner)]) > 1
                name = f"{owner} of {_join(_first_seen(runs))}" if several else owner
                named.append(f"{name} at ({_degrees(latitude)}, {_degrees(longitude)})")
            which = f"{_join(named)}, the cell centres as (latitude, longitude)"
        lines.append(
            f"Location: {role}, located by {how} (grid step {step} degrees): {which}."
        )
    return lines + _first_seen(filtered)


_ROW_KEYS = ("sample_hash", "split", "fold", "design_id")
"""The columns of a comparison table that name each row's evaluation and design."""


def _kept_scenarios(
    scenarios: Mapping[str, Any], evaluations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """`setup.scenarios` for the evaluations a table's rows name, and the reference."""
    named = {entry.get("scenario") for entry in evaluations}
    hashes = {entry.get("sample_hash") for entry in evaluations}
    kept: dict[str, Any] = {}
    for label, entry in scenarios["scenarios"].items():
        if label in named or label == scenarios["reference"]:
            blocks = [
                block for block in entry["blocks"] if block["sample_hash"] in hashes
            ]
            kept[label] = {**entry, "blocks": blocks}
    return {**scenarios, "labels": list(kept), "scenarios": kept}


def _kept_frontier(
    records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The entries of `setup.frontier` with the budgets the `rows` still name.

    A row names a budget by its design id on the block the entry reads.
    """
    named = {
        (row["sample_hash"], row["split"], row["fold"], row["design_id"])
        for row in rows
    }
    kept: list[dict[str, Any]] = []
    for record in records:
        block = (record["sample_hash"], record["split"], record["fold"])
        pairs = [
            (budget, design_id)
            for budget, design_id in zip(
                record["budgets"], record["design_ids"], strict=True
            )
            if (*block, design_id) in named
        ]
        if pairs:
            kept.append(
                {
                    **record,
                    "budgets": [budget for budget, _ in pairs],
                    "design_ids": [design_id for _, design_id in pairs],
                }
            )
    return kept


def _of_rows(
    metadata: EvaluationMetadata, table: Any
) -> tuple[
    dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], str | None
]:
    """The setup, designs and sources of the evaluations a table's rows name.

    A row names its evaluation by its sample hash, split and fold, and its
    design by its design id. The samples kept are those the evaluations kept
    are evaluated or designed on, the held-out blocks of a pooled evaluation
    among them, and the source records those of the samples kept. A pooled
    row's design lists the designs of its folds, which are kept too. An
    intervals table's flagged columns are those of the rows it still holds, and
    a frontier's budgets those its rows still name. A table without those
    columns, or without rows, names no evaluation: then the setup is that of the
    whole comparison, and the last item says so.
    Raises `EvaluationError` for a row that names an evaluation or a design
    the metadata does not describe.
    """
    setup = metadata.setup.model_dump(mode="json")
    missing = [name for name in _ROW_KEYS if name not in table.column_names]
    if missing:
        columns = "column" if len(missing) == 1 else "columns"
        return (
            setup,
            metadata.designs,
            metadata.sources,
            f"Rows: not identified, as the table lacks the {columns} "
            f"{_join(missing)}; this is the setup of the whole comparison, which "
            "may include evaluations that the rows do not belong to.",
        )
    if table.num_rows == 0:
        return (
            setup,
            metadata.designs,
            metadata.sources,
            "Rows: none; this is the setup of the whole comparison that the "
            "table's metadata describes.",
        )
    described = [name for name in ("level", "scope") if name in table.column_names]
    hashed = "cover_hash" in table.column_names
    # Across scenarios a block has one evaluation per design scenario.
    by_design = (
        setup.get("scenarios") is not None and "design_scenario" in table.column_names
    )
    rows = table.select(
        [
            *_ROW_KEYS,
            *described,
            *(["cover_hash"] if hashed else []),
            *(["design_scenario"] if by_design else []),
        ]
    ).to_pylist()

    def key_of(entry: Mapping[str, Any]) -> Any:
        """The evaluation a row or a record of one names."""
        first = (entry["sample_hash"], entry["split"], entry["fold"])
        return (*first, entry.get("design_scenario")) if by_design else first

    # The levels and scopes of the rows the table still holds.
    named: dict[Any, dict[str, list[str]]] = {}
    covers: dict[Any, set[str]] = {}
    for row in rows:
        key = key_of(row)
        entry = named.setdefault(key, {name: [] for name in described})
        for name in described:
            if row[name] not in entry[name]:
                entry[name].append(row[name])
        if hashed:
            covers.setdefault(key, set()).add(row["cover_hash"])

    def of_covers(evaluation: Mapping[str, Any], key: Any) -> dict[str, Any]:
        """The row and package records of the covers the table still names."""
        if not hashed:
            return {}
        return {
            name: [
                item for item in evaluation[name] if item["cover_hash"] in covers[key]
            ]
            for name in ("rows", "packages", "premium_bases")
            if name in evaluation
        }

    evaluations = [
        {**entry, **named[key_of(entry)], **of_covers(entry, key_of(entry))}
        for entry in setup["evaluations"]
        if key_of(entry) in named
    ]
    # A drift's rows leave out the reference's, which its differences read.
    drift = setup.get("drift")
    reference = None if drift is None else drift.get("reference")
    if reference is not None:
        evaluations += [
            entry
            for entry in setup["evaluations"]
            if entry.get("scenario") == reference and key_of(entry) not in named
        ]
    unknown = set(named) - {key_of(entry) for entry in evaluations}
    if unknown:
        sample_hash, split, fold, *_ = min(unknown, key=str)
        raise EvaluationError(
            "a row of the table names an evaluation that its metadata does not "
            f"describe: sample {_short(sample_hash)} ({split}, fold {fold}); give "
            "a concatenation of tables their merged metadata "
            "(EvaluationMetadata.merge)"
        )
    wanted = {row["design_id"] for row in rows if row["design_id"] is not None}
    wanted |= {
        design_id
        for record in metadata.designs
        if record["design_id"] in wanted and record.get("kind") == "pooled"
        for design_id in record["designs"]
    }
    designs = tuple(r for r in metadata.designs if r["design_id"] in wanted)
    absent = wanted - {record["design_id"] for record in designs}
    if absent:
        raise EvaluationError(
            f"a row of the table names design {_short(min(absent, key=str))}, "
            "which its metadata does not describe"
        )
    hashes = {str(entry["sample_hash"]) for entry in evaluations}
    for i, entry in enumerate(evaluations):
        where = f"setup.evaluations[{i}]"
        hashes.update(str(value) for value in _list(entry, "design_samples", where))
        details = _get(entry, "details", Mapping, where) or {}
        hashes.update(str(value) for value in _list(details, "blocks", where))
    samples = [s for s in setup["samples"] if str(s["sample_hash"]) in hashes]
    sources = tuple(
        entry
        for entry in metadata.sources
        if entry.get("sample_hash") is None or str(entry["sample_hash"]) in hashes
    )
    intervals = setup.get("intervals")
    if intervals is not None and {"column", "short_block"} <= set(table.column_names):
        flagged = _first_seen(
            row["column"]
            for row in table.select(["column", "short_block"]).to_pylist()
            if row["short_block"]
        )
        intervals = {**intervals, "short_block_columns": flagged}
    return (
        {
            **setup,
            "samples": samples,
            "evaluations": evaluations,
            "intervals": intervals,
            **(
                {}
                if setup.get("scenarios") is None
                else {"scenarios": _kept_scenarios(setup["scenarios"], evaluations)}
            ),
            **(
                {}
                if setup.get("frontier") is None
                else {"frontier": _kept_frontier(setup["frontier"], rows)}
            ),
        },
        designs,
        sources,
        None,
    )


def _seeds(evaluation: Mapping[str, Any], where: str) -> tuple[int | None, int | None]:
    """The plan's seed and the fresh draws' seed an evaluation records."""
    details = _get(evaluation, "details", Mapping, where) or {}
    return (
        _get(details, "plan_seed", int, f"{where}.details"),
        _get(details, "fresh_draws_seed", int, f"{where}.details"),
    )


def _seeds_text(plan: int | None, fresh: int | None) -> str:
    parts = [] if plan is None else [f"the plan's seed {plan}"]
    if fresh is not None:
        parts.append(f"the fresh draws' seed {fresh}")
    return "; ".join(parts)


_COUNTED = re.compile(r"(\d+) of (\d+) draws (.+)", re.DOTALL)
"""A note that counts draws, such as the draws beyond a curve's fitted support."""

_VERB = re.compile(r"\b(is|are|was|were)\b")
"""The verb of a counted note's draws: the first `is`, `are`, `was` or `were`."""


def _agree(rest: str, *, singular: bool) -> str:
    """The rest of a counted note, its verb in the singular or the plural.

    A counted note's verb agrees with its count (`1 of 400 draws is rarer`,
    `2 of 400 draws are rarer`), and the counts of several samples together
    take the plural.
    """
    forms = {"are": "is", "were": "was"} if singular else {"is": "are", "was": "were"}
    return _VERB.sub(lambda match: forms.get(match[1], match[1]), rest, count=1)


_MANY_ASSETS = 5
"""Above this many assets, the text gives counts and ranges, not each asset's values."""


def _named_assets(assets: Sequence[str]) -> str:
    """The assets by name, or their number above `_MANY_ASSETS`."""
    return f"{len(assets)} assets" if len(assets) > _MANY_ASSETS else _join(assets)


def _owned_note(note: str, assets: Collection[str]) -> tuple[str, str] | None:
    """The asset and the text of a note written `Asset <id>: <text>`, if it is one.

    An asset id may hold a colon, so each `: ` is tried against `assets`.
    """
    prefix = "Asset "
    if not note.startswith(prefix):
        return None
    end = note.find(": ", len(prefix))
    while end != -1:
        if note[len(prefix) : end] in assets:
            return note[len(prefix) : end], note[end + 2 :]
        end = note.find(": ", end + 1)
    return None


def _counted_by_asset(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], bool]:
    """The samples, with each per-asset counted note on one entry per asset.

    A sample of several assets writes a counted note per asset, as
    `Asset <id>: <n> of <m> draws ...`. Each asset's counted notes move to an
    entry of their own, so they are counted as the notes of one-asset samples
    are. Whether any moved is the second item.
    """
    expanded: list[Mapping[str, Any]] = []
    moved = False
    for i, sample in enumerate(samples):
        where = f"setup.samples[{i}]"
        values = _list(sample, "notes", where)
        if not all(isinstance(note, str) for note in values):
            raise EvaluationError(f"{where}.notes must be a list of text")
        assets = {str(asset) for asset in _list(sample, "asset_ids", where)}
        kept: list[str] = []
        own: dict[str, list[str]] = {}
        for note in values:
            owned = _owned_note(note, assets)
            if owned is not None and _COUNTED.fullmatch(owned[1]):
                own.setdefault(owned[0], []).append(owned[1])
            else:
                kept.append(note)
        if not own:
            expanded.append(sample)
            continue
        moved = True
        key = sample.get("lineage") or sample.get("sample_hash") or f"#{i}"
        expanded.append({**sample, "lineage": key, "notes": kept})
        expanded += [
            {
                **sample,
                "asset_ids": [asset],
                "lineage": f"{key}#{asset}",
                "notes": notes,
            }
            for asset, notes in own.items()
        ]
    return expanded, moved


def _note_kind(
    members: Sequence[int],
    samples: Sequence[Mapping[str, Any]],
    roles: Sequence[Sequence[str]],
    fresh: set[str],
) -> str | None:
    """Which kind of sample a group is; `None` if its blocks play both roles."""
    kinds = set()
    for i in members:
        if "design" in roles[i]:
            kinds.add("the design sample")
        elif str(samples[i].get("sample_hash")) in fresh:
            kinds.add("the fresh draws")
        else:
            kinds.add("the evaluation sample")
    return kinds.pop() if len(kinds) == 1 else None


def _shared_asset_notes(lines: Sequence[str], assets: Collection[str]) -> list[str]:
    """Write a note that several assets share once, naming them."""
    shared: dict[str, list[str]] = {}
    owned = {line: _owned_note(line, assets) for line in lines}
    for found in owned.values():
        if found is not None:
            shared.setdefault(found[1], []).append(found[0])
    written: list[str] = []
    for line in lines:
        found = owned[line]
        if found is None or len(shared[found[1]]) == 1:
            written.append(line)
        elif found[0] == shared[found[1]][0]:
            names = shared[found[1]]
            who = (
                f"{len(names)} assets"
                if len(names) > _MANY_ASSETS
                else f"Assets {_join(names)}"
            )
            written.append(f"{who}: {found[1]}")
    return written


def _note_labels(
    groups: Mapping[str, Sequence[int]],
    samples: Sequence[Mapping[str, Any]],
    roles: Sequence[Sequence[str]],
    owners: Sequence[Sequence[str]],
    fresh: set[str],
) -> dict[str, str]:
    """Which drawn sample each group of samples is, in words.

    A group is the samples of one lineage: one drawn sample and the blocks
    split from it, which carry its notes. By its samples' roles it is the
    design sample, the fresh draws or another evaluation sample, followed by
    its asset when the setup has several. A group whose blocks play both
    roles is named by its asset alone, or is the sample split into blocks.
    Groups that would read the same also get their first sample's hash.
    """
    several = len(_first_seen(asset for assets in owners for asset in assets)) > 1
    labels: dict[str, str] = {}
    for key, members in groups.items():
        kind = _note_kind(members, samples, roles, fresh)
        assets = _first_seen(asset for i in members for asset in owners[i])
        owner = f"for {_join(assets)}" if several and assets else ""
        if kind is not None:
            labels[key] = " ".join(["in", kind, owner]).rstrip()
        else:
            labels[key] = owner or "in the sample split into blocks"
    for label in set(labels.values()):
        same = [key for key, value in labels.items() if value == label]
        if len(same) < 2:
            continue
        for key in same:
            first = groups[key][0]
            name = samples[first].get("sample_hash")
            which = _short(name) if name is not None else f"setup.samples[{first}]"
            labels[key] = f"{label} (sample {which})"
    return labels


def _counted_parts(
    entries: Sequence[tuple[str, str, str, str, str | None]],
) -> tuple[list[str], bool]:
    """The counts of one note, and whether its verb is singular.

    Each entry is a count, a total, the label of its sample, the kind of that
    sample and the asset it is for. Up to `_MANY_ASSETS` assets each has its
    own count; above that, each kind of sample states the range of the counts.
    """
    assets = _first_seen(entry[4] for entry in entries if entry[4] is not None)
    if len(assets) <= _MANY_ASSETS:
        parts = [
            f"{count} of {total}{' draws' if j == 0 else ''} {label}"
            for j, (count, total, label, _, _) in enumerate(entries)
        ]
        return parts, len(entries) == 1 and int(entries[0][0]) == 1
    ranges: dict[tuple[str, str], list[int]] = {}
    for count, total, _, kind, _ in entries:
        ranges.setdefault((total, kind), []).append(int(count))
    parts = []
    for j, ((total, kind), counts) in enumerate(ranges.items()):
        low, high = min(counts), max(counts)
        span = str(low) if low == high else f"{low} to {high}"
        place = f" in {kind}" if kind else ""
        parts.append(
            f"{span} of {total}{' draws' if j == 0 else ''}{place} for each of "
            f"{len(counts)} assets"
        )
    return parts, False


def _notes(
    samples: Sequence[Mapping[str, Any]], evaluations: Sequence[Mapping[str, Any]]
) -> list[str]:
    """The samples' notes, each written once.

    Blocks split from one drawn sample carry its notes, which are written
    once. A note that counts draws (`<n> of <m> draws ...`) and differs
    between drawn samples only in its counts, such as the draws beyond a
    curve's fitted support in the design sample and in the fresh draws, is
    written once with every sample's own counts, asset by asset and the
    design sample first. Its verb is singular for one sample's count of 1
    and plural otherwise, whichever form each sample's note has. The counted
    notes that a sample of several assets writes per asset
    (`Asset <id>: <n> of <m> draws ...`) are written the same way; above
    `_MANY_ASSETS` assets, each kind of sample states the range of their
    counts. A note that several assets share word for word is written once,
    naming them.
    """
    samples, moved = _counted_by_asset(samples)
    notes: list[list[str]] = []
    roles: list[list[str]] = []
    owners: list[list[str]] = []
    for i, sample in enumerate(samples):
        where = f"setup.samples[{i}]"
        notes.append(_list(sample, "notes", where))
        roles.append([str(role) for role in _list(sample, "roles", where)])
        owners.append([str(asset) for asset in _list(sample, "asset_ids", where)])
    groups: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        key = sample.get("lineage") or sample.get("sample_hash") or f"#{i}"
        groups.setdefault(str(key), []).append(i)
    group_of = {i: key for key, members in groups.items() for i in members}
    counted = {
        tuple(
            _first_seen(
                note for i in members for note in notes[i] if _COUNTED.fullmatch(note)
            )
        )
        for members in groups.values()
    }
    every_asset = {asset for assets in owners for asset in assets}
    if len(counted) <= 1 and not moved:
        # One drawn sample, or the same counts in each: the notes as they are.
        return _shared_asset_notes(
            _first_seen(note for values in notes for note in values), every_asset
        )

    fresh = {
        str(evaluation.get("sample_hash"))
        for evaluation in evaluations
        if evaluation.get("method") == "fresh_draws"
    }
    labels = _note_labels(groups, samples, roles, owners, fresh)
    designed = {group_of[i] for i in range(len(samples)) if "design" in roles[i]}
    first = {
        asset: n
        for n, asset in enumerate(
            _first_seen(asset for assets in owners for asset in assets)
        )
    }

    def place(i: int) -> tuple[int, bool]:
        asset = first[owners[i][0]] if owners[i] else len(first)
        return asset, group_of[i] not in designed

    kinds = {
        key: _note_kind(members, samples, roles, fresh) or ""
        for key, members in groups.items()
    }

    # A note as it is (no entries), or the rest of a counted note with each
    # drawn sample's counts, label, kind and asset.
    statements: dict[tuple[bool, str], list[tuple[str, str, str, str, str | None]]] = {}
    for i in sorted(range(len(samples)), key=place):
        for note in notes[i]:
            match = _COUNTED.fullmatch(note)
            if match is None:
                statements.setdefault((False, note), [])
                continue
            count, total, rest = match.groups()
            entries = statements.setdefault((True, _agree(rest, singular=False)), [])
            asset = owners[i][0] if len(owners[i]) == 1 else None
            entry = (count, total, labels[group_of[i]], kinds[group_of[i]], asset)
            if entry not in entries:
                entries.append(entry)
    lines: list[str] = []
    for (is_counted, text), entries in statements.items():
        if not is_counted:
            lines.append(text)
            continue
        parts, singular = _counted_parts(entries)
        lines.append(f"{_join(parts)} {_agree(text, singular=singular)}")
    return _shared_asset_notes(lines, every_asset)


def _pricing(pricing: Any, where: str) -> str:
    if pricing is None:
        return "not stated"
    if not isinstance(pricing, Mapping):
        raise EvaluationError(f"{where}.pricing must be an object")
    if pricing.get("kind") == "pure":
        return "the pure premium, P = E[payout]"
    if pricing.get("kind") == "cost_of_capital":
        rate = _number(pricing.get("rate"), f"{where}.pricing.rate")
        q = _number(pricing.get("q"), f"{where}.pricing.q")
        return (
            f"a cost-of-capital load, P = (1 - {rate}) E[payout] + {rate} "
            f"CVaR_{q}(payout)"
        )
    return str(pricing.get("kind"))


def _pooled_split(
    evaluation: Mapping[str, Any],
    evaluated: str,
    where: str,
) -> str:
    """The `Split` line of a pooled evaluation."""
    details = _get(evaluation, "details", Mapping, where) or {}
    k = _get(details, "k", int, f"{where}.details")
    designed = bool(_list(evaluation, "design_samples", where))
    how = []
    if designed:
        how.append(
            "each year is evaluated by the design made without it and pays that "
            "fold's design premium"
        )
    if details.get("given_covers"):
        how.append(
            "a cover given as is applies on every fold and is priced once on the "
            "pooled rows"
        )
    checked = (
        "the leakage check passed on every fold"
        if evaluation.get("leakage_check") == "passed"
        else "no design, so no leakage check"
    )
    return (
        f"Split: out of sample, the {k or 'k'} folds of a {k or 'k'}-fold split over "
        f"blocks of years, pooled: evaluated on {evaluated}; {'; '.join(how)}; each "
        "metric is computed once over all those rows, with the same probability on "
        f"each, and is not a mean over the folds; {checked}."
    )


_LEAKAGE_CHECKS: Mapping[Any, str] = {
    "passed": "the leakage check passed: the two share no period of one source",
    "different_sources": (
        "different sources: the two have no lineage or run in common, so no row could "
        "be compared; that does not say that the sources are independent, since two "
        "runs may share a history or a calibration, which the module can't see"
    ),
}
"""What each recorded leakage check says, out of sample."""


def _cross_row(evaluation: Mapping[str, Any], where: str) -> str:
    """Whose terms an `other_scenario` evaluation reads, and on whose block."""
    scenario = _get(evaluation, "scenario", str, where)
    designed_on = _get(evaluation, "design_scenario", str, where)
    who = (
        "covers given as is, from the reference's design block,"
        if designed_on is None
        else f"designs from scenario {designed_on}"
    )
    return f"{who} on the block of scenario {scenario}"


def _split(
    evaluation: Mapping[str, Any],
    samples: Mapping[str, Mapping[str, Any]],
    where: str,
) -> str:
    split = _get(evaluation, "split", str, where)
    fold = _get(evaluation, "fold", int, where)
    method = _get(evaluation, "method", str, where)
    details = _get(evaluation, "details", Mapping, where) or {}
    folded = "fold not stated" if fold is None else f"fold {fold}"
    sample_hash = evaluation.get("sample_hash")
    evaluated = (
        "the evaluation sample, not known yet"
        if sample_hash is None
        else _years(samples.get(str(sample_hash)), where)
    )
    if split == "in_sample":
        return (
            f"Split: in sample, {folded}: evaluated on the design sample itself, "
            f"{evaluated}. In-sample numbers show the fit, not out-of-sample "
            "performance."
        )
    if split != "out_of_sample":
        return f"Split: {split or 'not stated'}, {folded}: evaluated on {evaluated}."
    if method == "kfold_pooled":
        return _pooled_split(evaluation, evaluated, where)
    if method == "holdout":
        fraction = details.get("train_fraction")
        how = f"{folded}, a holdout" + (
            ""
            if fraction is None
            else f" with train fraction {_number(fraction, f'{where}.details')}"
        )
    elif method == "kfold":
        k = _get(details, "k", int, f"{where}.details")
        how = f"{folded} of a {k or 'k'}-fold split over blocks of years"
    elif method == "fresh_draws":
        how = f"{folded}, on fresh Monte Carlo draws"
    elif method == "separate_sample":
        how = f"{folded}, on a separate sample from another source"
    elif method == "disjoint_periods":
        how = f"{folded}, on periods of the same source the design sample lacks"
    elif method == "other_scenario":
        how = f"{folded}, on another scenario's block: {_cross_row(evaluation, where)}"
    else:
        how = f"{folded}, split method not stated"
    designs = [str(value) for value in _list(evaluation, "design_samples", where)]
    designed = _join(
        _first_seen(_years(samples.get(value), where) for value in designs)
    )
    if "design_samples" not in evaluation:
        designed = "the design sample, not known yet"
    checked = _LEAKAGE_CHECKS.get(
        evaluation.get("leakage_check"), "no leakage check is recorded"
    )
    return (
        f"Split: out of sample, {how}: evaluated on {evaluated}; designed on "
        f"{designed or 'no design sample'}; {checked}."
    )


def _objective(objective: Mapping[str, Any], where: str) -> str:
    kind = objective.get("kind")
    if kind is None:
        return "not stated"
    if kind == "mse":
        return "MSE, E[(L - payout)^2]"
    if kind == "expected_uncovered":
        return "E[U], the expected uncovered loss"
    if kind == "weighted_cvar":
        alpha = _number(objective.get("alpha"), f"{where}.objective.alpha")
        beta = _number(objective.get("beta"), f"{where}.objective.beta")
        return (
            "weighted CVaR, alpha CVaR_beta(U) + (1 - alpha) E[U] with "
            f"alpha {alpha} and beta {beta}"
        )
    if kind == "user":
        parameters = objective.get("parameters") or {}
        return f"the user objective {objective.get('name')}" + (
            f" ({json.dumps(parameters, sort_keys=True)})" if parameters else ""
        )
    return str(kind)


def _design_labels(
    designs: tuple[dict[str, Any], ...], covers: tuple[CoverEntry, ...]
) -> dict[str, str | None]:
    """Each design's label, by `design_id`.

    A design record holds the label of its cover as evaluated
    (`cover_label`). Two designs can resolve to the same cover, for example
    under a binding budget, so a label is never looked up by the cover hash
    alone: for a record without `cover_label`, the cover's label is used
    only when the metadata gives that hash one label.
    """
    by_hash: dict[str, set[str | None]] = {}
    for entry in covers:
        by_hash.setdefault(entry.hash, set()).add(entry.label)
    labels: dict[str, str | None] = {}
    for record in designs:
        design_id = str(record["design_id"])
        if "cover_label" in record:
            where = f"designs[{_short(design_id)}]"
            labels[design_id] = _get(record, "cover_label", str, where)
            continue
        found = by_hash.get(str(record.get("cover_hash")), set())
        labels[design_id] = next(iter(found)) if len(found) == 1 else None
    return labels


def _unidentified(record: Mapping[str, Any], where: str) -> list[str]:
    """The amounts no design year reaches: flat, or by asset on a joint design."""
    by_asset = record.get("unidentified")
    if not isinstance(by_asset, Mapping):
        return [str(name) for name in _list(record, "unidentified", where)]
    return [
        f"{asset} {name}"
        for asset in by_asset
        for name in _list(by_asset, asset, f"{where}.unidentified")
    ]


def _design_terms(record: Mapping[str, Any], unit: str, where: str) -> list[str]:
    """What a design is made under: its scope if joint, objective, pricing, terms."""
    objective = _get(record, "objective", Mapping, where) or {}
    scope = _get(record, "scope", Mapping, where) or {}
    joint = scope.get("kind") == "joint"
    on_total = " on the total" if joint else ""
    parts: list[str] = []
    if joint:
        assets = [str(asset) for asset in _list(record, "asset_ids", where)]
        parts.append(
            "scope: joint, one objective, one budget and one premium on the total "
            f"of {_join(assets) or 'its assets'}, with one cover per asset"
        )
    parts += [
        f"objective: {_objective(objective, where)}",
        f"pricing: {_pricing(record.get('pricing'), where)}",
    ]
    budget = _get(record, "budget", Mapping, where) or {}
    if budget.get("ratio") is not None:
        ratio = _number(budget["ratio"], f"{where}.budget.ratio")
        expected = "E[L] of the total" if joint else "E[L]"
        parts.append(
            f"budget: premium at most {ratio} x {expected} of the design sample"
        )
    elif budget.get("amount") is not None:
        amount = _number(budget["amount"], f"{where}.budget.amount")
        parts.append(f"budget: premium at most {amount} {unit}{on_total}")
    else:
        parts.append("no budget")
    limit = _get(record, "limit", Mapping, where) or {}
    if limit.get("amount") is not None:
        each = " on each cover" if joint else ""
        parts.append(
            f"limit: {_number(limit['amount'], f'{where}.limit')} {unit}{each}"
        )
    else:
        parts.append("no limit")
    return parts


def _designed_on(
    record: Mapping[str, Any],
    samples: Mapping[str, Mapping[str, Any]],
    where: str,
) -> str | None:
    """What loss a design on a sample of several hazards read, if it was on one."""
    sample = samples.get(str(record.get("sample_hash")))
    if sample is None:
        return None
    if _objects(sample, "components", where):
        return "designed on the combined loss"
    for step in reversed(_objects(sample, "selection", where)):
        if step.get("kind") == "component":
            name = step.get("name")
            if step.get("loss") == "own":
                return f"designed on the own loss of component {name}"
            return f"designed on the index of component {name}, on the combined loss"
    return None


def _design(
    record: Mapping[str, Any],
    labels: Mapping[str, str | None],
    unit: str,
    samples: Mapping[str, Mapping[str, Any]],
) -> str:
    design_id = _short(record["design_id"])
    where = f"designs[{design_id}]"
    label = labels.get(str(record["design_id"]))
    objective = _get(record, "objective", Mapping, where) or {}
    scope = _get(record, "scope", Mapping, where) or {}
    joint = scope.get("kind") == "joint"
    on_total = " on the total" if joint else ""
    parts = _design_terms(record, unit, where)
    designed_on = _designed_on(record, samples, where)
    if designed_on is not None:
        parts.append(designed_on)
    if record.get("premium") is not None:
        premium = _number(record["premium"], f"{where}.premium")
        parts.append(f"design premium: {premium} {unit}{on_total}")
    if record.get("route") is not None:
        parts.append(f"route: {record['route']}")
    unique = _get(record, "unique", bool, where)
    answer = "not checked" if unique is None else "yes" if unique else "no"
    parts.append(f"unique optimum: {answer}")
    ties = _get(record, "ties", int, where)
    parts.append("ties: not counted" if ties is None else f"ties: {ties}")
    if joint and unique is not True:
        parts.append("the amounts of each asset are one member of an optimal set")
    unidentified = _unidentified(record, where)
    if unidentified:
        parts.append(
            f"amounts no design year reaches: {_join(unidentified)}, set to the "
            f"{record.get('fill') or 'lowest'} value allowed"
        )
    flag = _get(record, "degenerate_tail", str, where)
    if flag == "expected_uncovered":
        parts.append(
            "degenerate tail: beta is at most the probability of a zero loss, so "
            "the objective is a multiple of E[U]"
        )
    elif flag == "single_outcome":
        parts.append("degenerate tail: the tail holds at most one draw")
    elif flag is not None:
        parts.append(f"degenerate tail: {flag}")
    elif objective.get("kind") == "weighted_cvar":
        parts.append("tail: not degenerate")
    name = f"{label} ({design_id})" if label else design_id
    return f"Design {name}: {'; '.join(parts)}."


def _planned_designs(
    document: Mapping[str, Any], assets: Sequence[str], unit: str
) -> list[str]:
    """One `Design` line per scope and objective of a plan's `explain()` document.

    The document states the terms of the designs to be made: `designs` (the
    objectives, the pricing, the budget, the limit and the templates) and,
    under `Joint()`, `scopes`. The line names the templates and states those
    terms as a designed line does; what only a design knows is not known yet.
    Without templates nothing is designed, so there is no line.
    """
    where = "designs"
    terms = _get(document, where, Mapping, "document")
    if terms is None:
        return []
    covers = {
        str(cover.get("hash")): cover
        for cover in _objects(document, "covers", "document")
    }
    names = _first_seen(
        _get(covers.get(str(template), {}), "label", str, "covers") or _short(template)
        for template in _list(terms, "templates", where)
    )
    if not names:
        return []
    scopes = _objects(document, "scopes", "document") or [{"kind": "individual"}]
    return [
        f"Design {_join(names)}, planned: "
        + "; ".join(
            _design_terms(
                {**terms, "objective": objective, "scope": scope, "asset_ids": assets},
                unit,
                where,
            )
        )
        + "; design premium, route and non-uniqueness report: not known yet."
        for scope in scopes
        for objective in _objects(terms, "objectives", where)
    ]


def _metrics(options: Mapping[str, Any], unit: str) -> str:
    def stated(key: str, suffix: str) -> str:
        where = f"setup.metric_options.{key}"
        if options.get(key) is None:
            return "not stated"
        if key == "payout_threshold":
            return f"{_number(options[key], where)}{suffix}"
        return f"{_numbers(options[key], where)}{suffix}"

    return (
        f"Metrics: beta {stated('betas', '')}; return periods "
        f"{stated('return_periods', ' years')}; loss thresholds "
        f"{stated('loss_thresholds', f' {unit}')}; payout threshold "
        f"{stated('payout_threshold', f' {unit}')}."
    )


_ONE_CURVE = (
    "Out of sample here means new draws from the same curve, so these rows test "
    "the design's sampling error, not the curve."
)
"""What an evaluation says when its blocks are draws from one curve."""

_ONE_WINDOW = (
    "Out of sample here means new draws from the same window of horizons, so these "
    "rows test the design's sampling error, not the curves of the window."
)
"""What an evaluation says when its blocks are draws from one window of horizons."""


def _draws_from_one_curve(
    evaluation: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
) -> bool:
    """Whether an out-of-sample evaluation's blocks are draws from one curve.

    Those are fresh draws, and a holdout or k-fold of draws, pooled or not. Before
    execution a plan's evaluations name no sample, and its samples are all of one
    kind. Where the metadata records the curves, a design block whose curve the
    evaluation block does not read is not one curve, whatever the method, and
    neither is a window of horizons beside anything but the same members. A pooled
    evaluation of covers given as is has no design block.
    """
    method = evaluation.get("method")
    if evaluation.get("split") != "out_of_sample" or method not in (
        "fresh_draws",
        "holdout",
        "kfold",
        "kfold_pooled",
    ):
        return False
    if method == "kfold_pooled" and not evaluation.get("design_samples"):
        return False
    sample_hash = evaluation.get("sample_hash")
    read = [
        sample
        for sample in samples
        if sample_hash is None or sample.get("sample_hash") == sample_hash
    ]
    if method != "fresh_draws" and (
        not read or any(sample.get("source_kind") != "monte_carlo" for sample in read)
    ):
        return False

    def curves(block: Any) -> set[str]:
        return {
            _json_text(source["curve"])
            for source in sources
            if source.get("sample_hash") == block and source.get("curve") is not None
        }

    def members(block: Any) -> Any:
        return next(
            (
                sample.get("members")
                for sample in samples
                if sample.get("sample_hash") == block
            ),
            None,
        )

    details = evaluation.get("details") or {}
    blocks = details.get("blocks") or [sample_hash]
    held = set().union(*map(curves, blocks))
    window = members(blocks[0])
    return all(
        curves(block) <= held and members(block) == window
        for block in evaluation.get("design_samples") or ()
    )


_ALLOCATED_LEVELS = {
    "asset_by_value": "in proportion to its stated value",
    "asset_by_loss": (
        "in proportion to its loss in the draw, and to its stated value in a draw "
        "without loss"
    ),
}
"""The levels of allocated rows, and how each shares out the total payout."""


def _named(
    evaluation: Mapping[str, Any], key: str, default: str, where: str
) -> list[str]:
    """An evaluation's levels or scopes, or those of a row of one asset."""
    if evaluation.get(key) is None:
        return [default]
    return [str(value) for value in _list(evaluation, key, where)]


def _levels_of(
    evaluations: Sequence[Mapping[str, Any]],
    samples: Mapping[str, Mapping[str, Any]],
    assets: Sequence[str],
) -> tuple[list[str], list[str], str | None]:
    """The rows' levels, their scopes, and the words for levels beyond assets."""
    levels = _first_seen(
        level
        for i, evaluation in enumerate(evaluations)
        for level in _named(evaluation, "level", "asset", f"setup.evaluations[{i}]")
    )
    scopes = _first_seen(
        scope
        for i, evaluation in enumerate(evaluations)
        for scope in _named(
            evaluation, "scope", "individual", f"setup.evaluations[{i}]"
        )
    )
    if set(levels) <= {"asset"} and set(scopes) <= {"individual"}:
        return levels, scopes, None
    sums: list[str] = []
    for i, evaluation in enumerate(evaluations):
        if "total" not in _named(
            evaluation, "level", "asset", f"setup.evaluations[{i}]"
        ):
            continue
        sample = samples.get(str(evaluation.get("sample_hash")))
        summed = None if sample is None else sample.get("asset_ids")
        sums.append(_join([str(asset) for asset in summed or assets]) or "its assets")
    each = _join(assets) or "the assets not stated"
    words: list[str] = []
    for level in levels:
        if level == "asset":
            words.append(f"asset, one row per asset ({each})")
        elif level == "total":
            words.append(
                f"total, the sum over {' or over '.join(_first_seen(sums))} in each "
                "draw: the owner's total loss against the total payout"
            )
        elif level == "package":
            words.append(
                f"package, the separate contracts of {each} as one row: their "
                "payouts and their premiums added up"
            )
        elif level in _ALLOCATED_LEVELS:
            words.append(
                f"{level}, one row per asset ({each}): its share of the joint "
                f"design's total payout {_ALLOCATED_LEVELS[level]}, an accounting "
                "rule and not the payout of the asset's cover, with no premium or "
                "net-loss values"
            )
        else:
            words.append(level)
    text = "; ".join(words)
    if scopes == ["joint"]:
        text += ". Scope: joint, one objective and one budget on the owner's total"
    elif scopes == ["individual"]:
        text += ". Scope: individual, each cover standing on its own"
    else:
        text += f". Scope: {_join(sorted(scopes))}, kept apart by the scope column"
    return levels, scopes, text


def _row_lines(
    evaluations: Sequence[Mapping[str, Any]],
    designs: Sequence[Mapping[str, Any]],
    samples: Mapping[str, Mapping[str, Any]],
    unit: str,
) -> list[str]:
    """One line per row on a sample of several hazards: what it reads and is under."""
    records = {str(record["design_id"]): record for record in designs}
    lines: list[str] = []
    for i, evaluation in enumerate(evaluations):
        where = f"setup.evaluations[{i}]"
        for j, row in enumerate(_objects(evaluation, "rows", where)):
            place = f"{where}.rows[{j}]"
            level = _get(row, "level", str, place)
            record = records.get(str(row.get("design_id")))
            if level == "package":
                designed = "each contract designed or given on its own"
                pricing = "each contract on its own terms"
            elif record is not None:
                designed = (
                    _designed_on(record, samples, place)
                    or "designed on a sample of one hazard"
                )
                pricing = _pricing(record.get("pricing"), place)
            elif row.get("design_id") is not None:
                designed = "designed, with no design record given here"
                pricing = "as its design states"
            else:
                designed = "given as is"
                pricing = _pricing(evaluation.get("pricing"), place)
            bound = row.get("design_budget")
            budget = (
                "no design budget"
                if bound is None
                else f"budget: premium at most {_number(bound, place)} {unit}"
            )
            reads = _join((_get(row, "components", str, place) or "").split("+"))
            lines.append(
                f"Row {_get(row, 'cover_label', str, place)} ({level}): reads {reads}; "
                f"{designed}; {budget}; pricing: {pricing}."
            )
    return _first_seen(lines)


def _contract_text(contract: Mapping[str, Any], unit: str, where: str) -> str:
    bound, limit = contract.get("budget"), contract.get("limit")
    budget = (
        "no design budget"
        if bound is None
        else f"budget at most {_number(bound, where)} {unit}"
    )
    stated = (
        "no limit stated at design"
        if limit is None
        else f"limit {_number(limit, where)} {unit}"
    )
    premium = _number(contract.get("premium"), f"{where}.premium")
    name = _get(contract, "name", str, where)
    return f"{name} ({budget}, premium {premium} {unit}, {stated})"


def _package_lines(evaluations: Sequence[Mapping[str, Any]], unit: str) -> list[str]:
    """One line per package: its contracts' terms, their sums and what they mean."""
    lines: list[str] = []
    for i, evaluation in enumerate(evaluations):
        where = f"setup.evaluations[{i}]"
        for j, package in enumerate(_objects(evaluation, "packages", where)):
            place = f"{where}.packages[{j}]"
            contracts = _objects(package, "contracts", place)
            budget = package.get("budget")
            summed = _number(package.get("premium_sum"), f"{place}.premium_sum")
            text = (
                f"Package {_get(package, 'label', str, place)}: "
                + _join(
                    [
                        _contract_text(contract, unit, f"{place}.contracts[{k}]")
                        for k, contract in enumerate(contracts)
                    ]
                )
                + (
                    "; not every contract has a design budget"
                    if budget is None
                    else f"; the budgets sum to {_number(budget, place)} {unit}"
                )
                + f"; the premiums sum to {summed} {unit}"
            )
            if package.get("premium_basis") == "stated":
                stated = _number(package.get("premium"), f"{place}.premium")
                text += f", and its row uses the stated premium {stated} {unit}"
            lines.append(
                text + ". Each cover keeps its own limit, and the package has none of "
                "its own: its largest year is at most the sum of its covers' limits. "
                "A loading is not additive across contracts, so these premiums "
                "price each contract on its own, not one contract on the sum."
            )
    return _first_seen(lines)


def _intervals(intervals: Mapping[str, Any]) -> str:
    """The `Intervals` line: how the bounds are made and what they leave out."""
    where = "setup.intervals"
    resamples = _number(intervals.get("n_bootstrap"), f"{where}.n_bootstrap")
    level = _number(intervals.get("ci_level"), f"{where}.ci_level")
    unit = _get(intervals, "unit", str, where)
    redrawn = (
        "the evaluated rows, as the reference of an EmpiricalCopula is part of the "
        "stated model"
        if unit == "row"
        else "the evaluated years, each with every row of that year"
    )
    text = (
        f"Intervals: {_get(intervals, 'method', str, where)}, level {level}, from "
        f"{resamples} resamples of {redrawn}, with replacement and with the designs "
        f"held fixed. An interval is the middle {level} of the recomputed values. It "
        "describes how much a number moves with the years drawn. It leaves out "
        "uncertainty in the hazard data, the impact function and the design, and on "
        "a curve block it is the Monte Carlo error of the draws and says nothing "
        "about the curve. Two rows' intervals are not a test of a difference between "
        "them."
    )
    short = [str(name) for name in _list(intervals, "short_block_columns", where)]
    if short:
        text += (
            f" The level of these columns reaches the block's worst row, so their "
            "bounds cannot reach beyond the block's largest rows and understate how "
            f"far the number could move: {', '.join(short)}."
        )
    return text


def _frontier_budget(budget: Mapping[str, Any], unit: str, where: str) -> str:
    """One budget of a frontier, as the `Design` lines state a budget."""
    if budget.get("ratio") is not None:
        return f"{_number(budget['ratio'], f'{where}.ratio')} x E[L]"
    return f"{_number(budget.get('amount'), f'{where}.amount')} {unit}"


def _frontier(
    record: Mapping[str, Any],
    samples: Mapping[str, Mapping[str, Any]],
    unit: str,
    where: str,
) -> str:
    """The `Frontier` line: what each row is, and that no budget is named best."""
    objective = _get(record, "objective", Mapping, where) or {}
    budgets = [
        _frontier_budget(budget, unit, f"{where}.budgets[{i}]")
        for i, budget in enumerate(_objects(record, "budgets", where))
    ]
    designed = _years(samples.get(str(record.get("design_sample"))), where)
    evaluated = _years(samples.get(str(record.get("sample_hash"))), where)
    split = str(record.get("split")).replace("_", " ")
    return (
        f"Frontier: one design per budget ({_join(budgets)}) of template "
        f"{_short(record.get('template_hash'))} for {_objective(objective, where)}, "
        f"each solved on its own on {designed}; each row is read on {evaluated} "
        f"({split}, fold {record.get('fold')}). The table names no best budget."
    )


def _rank_block_designs(
    evaluation: Mapping[str, Any],
    sample: Mapping[str, Any],
    designs: Sequence[Mapping[str, Any]],
    where: str,
) -> list[Mapping[str, Any]]:
    """The designs of an evaluation's rows that read rank blocks.

    On a sample of several hazards those are the designs its rows name. On a
    sample of several assets a set of covers has no design on its `"total"` row,
    so they are the joint designs made on the evaluation's design samples.
    """
    if sample.get("components"):
        named = {
            str(row.get("design_id")) for row in _objects(evaluation, "rows", where)
        }
        return [record for record in designs if str(record["design_id"]) in named]
    made_on = {str(value) for value in _list(evaluation, "design_samples", where)}
    return [
        record
        for record in designs
        if (record.get("scope") or {}).get("kind") == "joint"
        and str(record.get("sample_hash")) in made_on
    ]


def _rank_block_text(
    evaluation: Mapping[str, Any],
    sample: Mapping[str, Any],
    designs: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str | None],
    betas: Any,
    where: str,
) -> str | None:
    """The `Rank blocks` line of one evaluation, if a row reads rank blocks.

    A payout breakdown has no tail metric and no such columns, so it has none.
    """
    if _list(evaluation, "part_kinds", where):
        return None
    levels = _named(evaluation, "level", "asset", where)
    components = bool(sample.get("components"))
    read = {
        level: _reference_years(
            level,
            dependence=sample.get("dependence"),
            hazard_dependence=sample.get("hazard_dependence"),
            components=components,
        )
        for level in levels
    }
    filled = [level for level in levels if read[level] is not None]
    if not filled:
        return None
    years = read[filled[0]]
    assert years is not None
    key = "hazard_dependence" if components else "dependence"
    model = _get(sample, key, Mapping, where) or {}
    row = "year" if model.get("pool") is None else "model-year"
    place = "setup.metric_options.betas"
    _numbers(betas, place)
    flagged = [beta for beta in betas if _tail_in_one_block(float(beta), years)]
    text = (
        f"Rank blocks: the {_join(filled)} rows read columns in the ranks of {years} "
        f"reference {row}s (m = {years}), and inside one {row}'s block the columns are "
        "independent by assumption. Where (1 - beta) * m <= 1 the tail holds at most "
        "one block, so a tail metric at that beta lies inside the assumption. Flagged "
        f"betas of the metric options: {_numbers(flagged, place)}. Not flagged: "
        f"{_numbers([beta for beta in betas if beta not in flagged], place)}."
    )
    stated: list[str] = []
    for record in _rank_block_designs(evaluation, sample, designs, where):
        objective = record.get("objective") or {}
        if objective.get("kind") != "weighted_cvar":
            continue
        design_id = _short(record["design_id"])
        beta = _number(objective.get("beta"), f"designs[{design_id}].objective.beta")
        label = labels.get(str(record["design_id"]))
        name = f"{label} ({design_id})" if label else design_id
        answer = (
            "flagged too"
            if _tail_in_one_block(float(objective["beta"]), years)
            else "not flagged"
        )
        stated.append(f"{name}: beta {beta}, {answer}")
    if stated:
        text += f" Weighted-CVaR beta of each row's design: {'; '.join(stated)}."
    return text


def _rank_block_lines(
    evaluations: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    by_hash: Mapping[str, Mapping[str, Any]],
    designs: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str | None],
    options: Any,
) -> list[str]:
    """One `Rank blocks` line per evaluation whose rows read rank blocks.

    Before execution a plan's evaluations name no sample, and its samples are
    all of one kind.
    """
    betas = (options or {}).get("betas") or []
    records = [record for record in designs if record.get("kind") != "pooled"]
    lines: list[str] = []
    for i, evaluation in enumerate(evaluations):
        sample_hash = evaluation.get("sample_hash")
        read = samples if sample_hash is None else [by_hash.get(str(sample_hash))]
        for sample in read:
            if sample is None:
                continue
            text = _rank_block_text(
                evaluation, sample, records, labels, betas, f"setup.evaluations[{i}]"
            )
            if text is not None:
                lines.append(text)
    return _first_seen(lines)


def _window_lines(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per window of horizons among samples no set of scenarios states."""
    lines: list[str] = []
    for i, sample in enumerate(samples):
        members = _list(sample, "members", f"setup.samples[{i}]")
        if len(members) < 2:
            continue
        names = _join([f"{pathway} {horizon}" for pathway, horizon in members])
        lines.append(
            f"Window of horizons: {len(members)} curves ({names}), one pathway at "
            "several horizons, mixed with equal weight on one set of uniforms: row d "
            f"reads member d mod {len(members)}. The horizon column is the centre of "
            "the members, a horizon no row need be read at, and nothing is "
            "interpolated between them: every row is a draw from a curve the dataset "
            "holds."
        )
    return _first_seen(lines)


_CMIP6_CAVEAT = (
    "a climate model is a model world, so use results as change signals, not design "
    "values"
)
"""crc-sdk's own caveat on native CMIP6, from `docs/sources/cmip6.md`."""


def _block_made(made: Mapping[str, Any], where: str) -> str:
    """How a scenario's block was made."""
    kind = _get(made, "kind", str, where)
    if kind == "holdout":
        fraction = _number(made.get("train_fraction"), f"{where}.train_fraction")
        return f"from a holdout with train fraction {fraction}"
    if kind == "kfold":
        fold = _get(made, "fold", int, where)
        return f"from fold {fold} of a {_get(made, 'k', int, where)}-fold split"
    return "as built, with no split"


def _scenario_block(
    block: Mapping[str, Any],
    members: Sequence[Any],
    samples: Mapping[str, Mapping[str, Any]],
    where: str,
) -> str:
    """A block's rows, how it was made, and its rows per member of a window."""
    rows = _years(samples.get(str(block.get("sample_hash"))), where)
    text = f"{rows}, {_block_made(block.get('made') or {}, where)}"
    counts = _list(block, "member_rows", where)
    if len(members) > 1:
        each = ", ".join(
            f"{member[1]}: {count}"
            for member, count in zip(members, counts, strict=True)
        )
        text += f" (rows per member on this block: {each})"
    return text


def _named_or_silent(value: Any) -> str:
    return "none recorded" if value is None else str(value)


def _hazard_read(entry: Mapping[str, Any]) -> str:
    """The hazard a scenario reads, and whether the caller stated its name."""
    hazard = entry.get("hazard")
    if not isinstance(hazard, Mapping):
        return "none recorded"

    def name(record: Mapping[str, Any]) -> str:
        said = (
            "as the caller states"
            if record.get("from") == "stated"
            else "as its source names it"
        )
        return f"{record.get('name')}, {said}"

    if isinstance(hazard.get("name"), str):
        return name(hazard)
    return _join(
        [
            f"component {part}: {name(own)}"
            for part, own in hazard.items()
            if own is not None
        ]
    )


def _scenario_line(
    label: str,
    entry: Mapping[str, Any],
    reference: str,
    samples: Mapping[str, Mapping[str, Any]],
    where: str,
) -> str:
    """Statement 1: one scenario's label, role, sources, rows and block."""
    role = " (the reference)" if label == reference else ""
    members = _list(entry, "members", where)
    blocks = [
        _scenario_block(block, members, samples, f"{where}.blocks[{i}]")
        for i, block in enumerate(_objects(entry, "blocks", where))
    ]
    text = (
        f"Scenario {label}{role}: hazard {_hazard_read(entry)}; product "
        f"{_named_or_silent(entry.get('product'))}; pathway "
        f"{_named_or_silent(entry.get('pathway'))}; horizon "
        f"{_named_or_silent(entry.get('horizon'))}; rows: "
        f"{'; '.join(blocks) or 'not among these rows'}."
    )
    if len(members) > 1:
        names = _join([f"{pathway} {horizon}" for pathway, horizon in members])
        text += (
            f" A window of {len(members)} horizons: {names}, mixed with equal weight "
            "on one set of uniforms. The horizon column is their centre, a horizon no "
            "row need be read at."
        )
    elif members:
        pathway, horizon = members[0]
        pair = f"{_named_or_silent(pathway)} {_named_or_silent(horizon)}"
        text += f" A single horizon: {pair}."
    elif entry.get("horizon") is not None:
        text += " The horizon column is the centre of the series' whole window."
    return text


def _cross_rows(
    evaluations: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Each cross row's terms and block, once, in words."""
    return _first_seen(
        _cross_row(evaluation, f"setup.evaluations[{i}]")
        for i, evaluation in enumerate(evaluations)
        if evaluation.get("method") == "other_scenario"
    )


def _fixed_line(evaluations: Sequence[Mapping[str, Any]]) -> str:
    """Statement 2: what stays fixed, and what a cross row is."""
    text = (
        "Fixed across the scenarios: each cover's or design's terms, the loss model, "
        "the values, the basis-risk model, the pricing and the metric options."
    )
    crossed = _cross_rows(evaluations)
    if crossed:
        text += (
            f" A cross row is a cover or design from one scenario on another's block: "
            f"{_join(crossed)}. Its terms are read on that block as they stand, in "
            "index units, and nothing rescales them: a cross row says what the same "
            "terms would pay on that block."
        )
    return text


def _products_differ(scenarios: Mapping[str, Any], reference: str) -> list[str]:
    """Statement 4: the scenarios whose product differs from the reference's."""
    entries = _get(scenarios, "scenarios", Mapping, "setup.scenarios") or {}
    ours = (entries.get(reference) or {}).get("product")
    return [
        (
            f"Products: scenario {label} reads "
            f"{_named_or_silent(entry.get('product'))} and the reference {reference} "
            f"reads {_named_or_silent(ours)}, so a "
            "difference between them includes the products: resolution, method and "
            "pooling differ, and it can't be attributed to the scenario alone."
        )
        for label, entry in entries.items()
        if entry.get("product") != ours
    ]


def _read_at(places: Sequence[Mapping[str, Any]], where: str) -> str:
    """Where a scenario reads the index, for each asset or component."""
    if len(places) > _MANY_ASSETS:
        kinds = _first_seen(str(place.get("kind")) for place in places)
        said = {
            "h3_cell": "H3 cell",
            "nearest_cell": "nearest grid cell",
            "station": "station",
        }
        return (
            f"{len(places)} assets, each at its "
            f"{_join([said.get(kind, 'stated location') for kind in kinds])}"
        )
    parts: list[str] = []
    for i, place in enumerate(places):
        kind = place.get("kind")
        owner = _get(place, "owner", str, f"{where}[{i}]")
        if kind == "h3_cell":
            parts.append(f"the H3 cell {place.get('cell')} of {owner}")
        elif kind == "nearest_cell":
            latitude = _degrees(float(place["latitude"]))
            longitude = _degrees(float(place["longitude"]))
            parts.append(
                f"the nearest grid cell of {owner}, centred at "
                f"({latitude}, {longitude})"
            )
        elif kind == "station":
            parts.append(f"station {place.get('station')}, for {owner}")
        else:
            parts.append(f"where the source of {owner} states")
    return _join(_first_seen(parts))


def _index_line(scenarios: Mapping[str, Any]) -> str:
    """Statement 5: where each scenario reads the index."""
    entries = _get(scenarios, "scenarios", Mapping, "setup.scenarios") or {}
    parts = [
        f"scenario {label} at {_read_at(_objects(entry, 'index_read', label), label)}"
        for label, entry in entries.items()
    ]
    return f"Index: {'; '.join(parts)}. Each scenario reads its own index there."


_PREMIUM_BASES = {
    "design": "the design premium, the contract price set on the design block",
    "reference": "priced once, with the row's pricing, on the reference's design block",
    "stated": "stated with premiums=",
}
"""What each premium basis says across scenarios."""


def _premium_line(evaluations: Sequence[Mapping[str, Any]]) -> str | None:
    """Statement 6: the premium basis of every cover."""
    by_basis: dict[str, list[str]] = {}
    for i, evaluation in enumerate(evaluations):
        for j, entry in enumerate(
            _objects(evaluation, "premium_bases", f"setup.evaluations[{i}]")
        ):
            where = f"setup.evaluations[{i}].premium_bases[{j}]"
            basis = _get(entry, "premium_basis", str, where)
            if basis is None:
                continue
            names = by_basis.setdefault(basis, [])
            name = str(entry.get("cover_label"))
            if name not in names:
                names.append(name)
    if not by_basis:
        return None
    parts = [
        f"{_join(names)}: {_PREMIUM_BASES.get(basis, basis)}"
        for basis, names in by_basis.items()
    ]
    return (
        f"Premium basis: {'; '.join(parts)}. premium and the net-loss columns stay at "
        "that contract price on every block, and premium_repriced prices the same "
        "payouts on each block."
    )


def _budget_line(
    designs: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str | None],
    samples: Mapping[str, Mapping[str, Any]],
    unit: str,
) -> str | None:
    """Statement 7: whether each design's budget moves with the scenario."""
    records = [record for record in designs if record.get("kind") != "pooled"]
    parts: list[str] = []
    for record in records:
        design_id = _short(record["design_id"])
        where = f"designs[{design_id}]"
        budget = _get(record, "budget", Mapping, where)
        if budget is None:
            continue
        sample = samples.get(str(record.get("sample_hash"))) or {}
        label = labels.get(str(record["design_id"]))
        name = f"{label} ({design_id})" if label else design_id
        on = f"{name}, designed on scenario {sample.get('scenario')}"
        if budget.get("ratio") is not None:
            ratio = _number(budget["ratio"], f"{where}.budget.ratio")
            parts.append(
                f"{on}: ratio, a premium of at most {ratio} x E[L] of its design "
                "block, so the bound moves with the scenario"
            )
        elif budget.get("amount") is not None:
            amount = _number(budget["amount"], f"{where}.budget.amount")
            parts.append(
                f"{on}: amount, a premium of at most {amount} {unit}, so the bound "
                "does not move with the scenario"
            )
    return f"Budgets: {'; '.join(parts)}." if parts else None


def _tail_lines(scenarios: Mapping[str, Any], options: Any) -> list[str]:
    """Statement 9: beta is one on every scenario, and the short blocks' columns."""
    betas = (options or {}).get("betas") if isinstance(options, Mapping) else None
    lines = (
        [
            f"Tails: beta is the same on every scenario "
            f"({_numbers(betas, 'setup.metric_options.betas')}), while the loss level "
            "of its tail is not."
        ]
        if betas
        else []
    )
    by_block: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    entries = _get(scenarios, "scenarios", Mapping, "setup.scenarios") or {}
    for label, entry in entries.items():
        for j, block in enumerate(_objects(entry, "blocks", label)):
            where = f"setup.scenarios.{label}.blocks[{j}]"
            columns = tuple(
                str(name) for name in _list(block, "short_block_columns", where)
            )
            if columns:
                count = _get(block, "n_rows", int, where) or 0
                names = by_block.setdefault((count, columns), [])
                if label not in names:
                    names.append(label)
    for (rows, columns), labels in by_block.items():
        lines.append(
            f"Short blocks: on the {rows} rows of {_join(labels)}, a beta with "
            "(1 - beta) * n <= 1 reads the block's worst row and a return period above "
            f"n its largest value, in these columns: {', '.join(columns)}."
        )
    return lines


def _widths_line(scenarios: Mapping[str, Any]) -> str | None:
    """Statement 10: windows of different widths, or one beside a single horizon."""
    entries = _get(scenarios, "scenarios", Mapping, "setup.scenarios") or {}
    widths = {
        label: len(entry["members"])
        for label, entry in entries.items()
        if entry.get("members")
    }
    if len(set(widths.values())) <= 1:
        return None
    if min(widths.values()) == 1:
        what = "a window of horizons beside a single horizon"
    else:
        what = "windows of different widths"
    named = _join([f"{label} ({width})" for label, width in widths.items()])
    return (
        f"Widths: this set holds {what}, with these numbers of members: {named}. A "
        "difference between them includes the difference in width."
    )


def _scenario_lines(
    setup: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    samples: Mapping[str, Mapping[str, Any]],
    designs: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str | None],
    unit: str,
) -> list[str]:
    """The statements of a comparison across scenarios, next to its numbers."""
    scenarios = _get(setup, "scenarios", Mapping, "setup")
    if scenarios is None:
        return []
    reference = _get(scenarios, "reference", str, "setup.scenarios") or ""
    entries = _get(scenarios, "scenarios", Mapping, "setup.scenarios") or {}
    lines = [
        _scenario_line(label, entry, reference, samples, f"setup.scenarios.{label}")
        for label, entry in entries.items()
    ]
    lines.append(_fixed_line(evaluations))
    shared = scenarios.get("shared_draws")
    if isinstance(shared, Mapping):
        names = [str(label) for label in _list(shared, "labels", "setup.scenarios")]
        lines.append(
            f"Shared draws: the curve scenarios {_join(names)} share one set of "
            "evaluation draws, separate from the design's: row d holds one uniform and "
            "one noise factor on each."
        )
    lines += _products_differ(scenarios, reference)
    lines.append(_index_line(scenarios))
    for line in (
        _premium_line(evaluations),
        _budget_line(designs, labels, samples, unit),
    ):
        if line is not None:
            lines.append(line)
    native = [
        label
        for label, entry in entries.items()
        if "cmip6" in str(entry.get("product") or "").split("+")
    ]
    if native:
        lines.append(f"Native CMIP6 ({_join(native)}): {_CMIP6_CAVEAT}.")
    lines += _tail_lines(scenarios, setup.get("metric_options"))
    width = _widths_line(scenarios)
    if width is not None:
        lines.append(width)
    return lines


def _is_document(value: Any) -> TypeGuard[Mapping[str, Any]]:
    """Whether `value` is a whole document, such as `explain()`'s, not a setup."""
    return isinstance(value, Mapping) and "setup" in value and "samples" not in value


def setup_text(
    metadata: EvaluationMetadata | EvaluationSetup | Mapping[str, Any] | pa.Table,
) -> str:
    """Return the setup behind a comparison's numbers, as plain text.

    Each line is one statement, for a notebook or a report to print next to
    the numbers it describes:

    - `Rows`, first and only for a table whose rows are not identified (see
      below): that the text is the setup of the whole comparison;
    - `Level`, `Scope`, `Scenario`, `Basis`: the level of the rows and the
      assets (one row per asset; which assets a total sums; the rule of an
      allocation), the scope when it is not that of rows of single assets,
      the scenario, and the annual basis (one draw is one year);
    - `Split`, one line per evaluation: in or out of sample, the fold, how
      the split was made (a holdout and its train fraction, k-fold and k,
      fresh draws, a separate sample, another scenario's block with the two
      scenarios named), the evaluation and design samples (years, periods,
      blocks) and the leakage check; and, where the design and evaluation blocks
      are draws from one curve or from one window of horizons, that these rows
      test the design's sampling error, not the curve. A pooled evaluation has one line,
      with k, the rows and that each year is evaluated by the design made
      without it;
    - `Seeds`: the plan's seed (`plan_seed`) and the fresh draws' seed
      (`fresh_draws_seed`) that the evaluations' `details` record, if any;
    - on a table of `compare_scenarios`, in this order: one `Scenario` line each
      (its label, whether it is the reference, the hazard it reads and whether
      the caller stated that name, its product, pathway and horizon, its rows and
      how each block was made, and whether it is a single horizon or a window,
      with a window's members and each one's rows); `Fixed across the
      scenarios` (what stays fixed, and which scenario's terms a cross row reads
      on which block); `Shared draws`, for curves; `Products`, where a scenario's
      differs from the reference's; `Index` (where each reads it); `Premium
      basis`; `Budgets` (whether each design's moves with the scenario); `Native
      CMIP6` (crc-sdk's caveat, for that product); `Tails` and `Short blocks`;
      and `Widths`, where windows of different widths, or one beside a single
      horizon, stand side by side. `Drift` on a table of `to_drift`: the
      reference its differences read;
    - `Sample`: the source kind, the probability semantics and who stated
      them, the basis-risk model and its parameters, and for a model that acts
      on the hazard the mean hazard and the mean impact with it over without
      it (above five assets, their ranges), the loss model (of each
      asset, when the assets differ, and of each component on a sample of
      several hazards), the dependence between the assets, and on a sample
      with index components each component's name, unit, tail and whether it
      has a loss of its own, the rule of the combined loss or that a table
      states it, and the dependence between the hazards: the data's own, with
      the sentence that two annual maxima of one year may come from different
      events, or the stated model, called an assumption, and, with mixed
      tails, that the dependence is between values and not severities;
    - `Combined loss`, when a sample evaluated on has index components: that
      every row's metrics are on the combined loss;
    - `Source`: each source record of the samples, such as a hazard curve or
      a per-year series, named by its component on a sample of several
      hazards, with its provider, dataset, horizon and pathway, and which of
      them the record does not hold, and for a curve that a plan
      read, its canonical dataset's path and SHA-256; before execution, the
      sources that a plan's `explain()` document states (a canonical dataset's
      path and selection, or a per-year source's provider and dataset, and the
      file, table or plan it is read from);
    - `Location`, for per-year series read at the nearest grid cell: whether
      each asset was located by a point or by its H3 cell, the grid step and
      the matched cell's centre, one line per role, component, way and grid
      step; above five assets, the number of assets and cells and the range of
      the centres. The asset's point and its distance to the cell are not
      recorded. A component's series read from the rows of a stated filter has
      a line too, naming the filter;
    - `Trend`, for per-year samples: the years as recorded, the window of
      recent years, or the linear trend removed, with its fit years, each
      series' slope and intercept with their units, and the reference year;
      above five assets, the range of the slopes and of the intercepts;
    - `Pool`, for pooled model-years: the runs, the pathway, the years, the
      number of rows, that every run has the same weight, and that a draw
      mixes model spread with year-to-year variability and that the number
      of rows is not an effective sample size;
    - `Window of horizons`, outside a comparison across scenarios: a window's
      members, that they are mixed with equal weight on one set of uniforms, and
      that `horizon` is their centre, interpolating nothing;
    - `Sample notes`: what the sources recorded, such as draws beyond the
      fitted support, a pooled ensemble, dropped years, an index read from
      another series, a comonotone index and loss, or the assumption of a
      dependence model. A note that differs between drawn samples or assets
      only in its counts, such as the draws beyond the fitted support in the
      design sample and in the fresh draws, is written once, with each
      sample's or asset's own counts, its verb in the singular for one count
      of 1 and in the plural otherwise; above five assets it gives the range
      of the counts. A note that several assets share is written once, naming
      them;
    - `Pricing`: the pricing of the covers given as is; a design keeps its
      design premium, and a row's `premium_basis` says which premium it uses;
    - `Design`, one line per design: the objective, pricing, budget and
      limit, the design premium, the route, the non-uniqueness report
      (`unique`, `ties`, amounts no design year reaches) and a degenerate
      tail; for a joint design also its scope and which assets it covers,
      and for a design on a sample of several hazards the loss it read.
      Before execution, one line per objective and scope that a plan's
      `explain()` document plans, naming its templates, with the objective,
      pricing, budget and limit;
    - `Row`, on a sample of several hazards, one line per row: the components
      its cover reads, what loss it was designed on, its budget and its
      pricing;
    - `Package`, one line per package row: each contract's budget, premium and
      limit, their sums, that each cover keeps its own limit and that a loading
      is not additive across contracts;
    - `Rank blocks`, for rows that read a sum over the columns an
      `EmpiricalCopula` reordered: the number of reference years m, the betas of
      the metric options flagged by `(1 - beta) * m <= 1` and those not, and
      whether the weighted-CVaR beta of a row's design is flagged too;
    - `Payout breakdown`, for a breakdown holding a `MaxOf` cover: that its rows
      are an attribution;
    - `Frontier`, one line per call of `design_frontier`: the budgets, the
      template and the objective, that there is one design per budget on the
      stated design block and each row is read on the stated block, and that the
      table names no best budget;
    - `Intervals`, for a table of `bootstrap_intervals`: the resamples, the
      level, the unit, what an interval leaves out, that two rows' intervals
      are not a test of a difference between them, and the columns whose level
      reaches the block's worst row;
    - `Metrics`: the levels beta, the return periods and the loss and payout
      thresholds.

    Statements that every evaluation or sample shares are written once; the
    seeds too, unless the evaluations record different ones, when each
    `Split` line states its own.

    `metadata` is one of:

    - a comparison's `EvaluationMetadata`, or its JSON document as a mapping
      (`json.loads(metadata.to_json())`);
    - a comparison table (a `pyarrow.Table`, such as one `compare` returns,
      one read back from a file that `write_comparison` or a plan's
      `write_parquet` wrote, or rows filtered from either). Its
      `crc.parametric.evaluation` metadata is read as
      `EvaluationMetadata.from_table` reads it, and the text states the setup
      of its rows: the evaluations the rows name (by `sample_hash`, `split`
      and `fold`) with the levels and scopes of those rows, the samples
      those are evaluated and designed on, their sources, and the designs
      the rows name (by `design_id`). All the rows of a comparison give the
      text of its metadata. A table without those columns or without rows
      names no evaluation: its text is that of the whole comparison, after
      a `Rows` line that says so;
    - a plan's `explain(format="json")` document (a mapping with a `setup`
      and no `samples`): its setup, a `Source` line from its `sources` and a
      `Design` line per planned objective from its `designs`;
    - a setup alone: an `EvaluationSetup`, or a mapping in its layout, such
      as a setup known before anything is drawn.

    A setup alone has no design or source records, so it has no `Design`,
    `Source` or `Location` lines, and an `explain()` document has no `Location`
    lines. Keys that a setup does not know yet (sample hashes, counts) may be
    missing: the text then says what is not known. The text depends only on
    `metadata`, so the same metadata gives the same text; nothing is read or
    written.

    Raises `EvaluationError` for setup and document values of the wrong
    type, for a table without valid `crc.parametric.evaluation` metadata,
    and for a table row that names an evaluation or a design its metadata
    does not describe (as after `pyarrow.concat_tables`, which keeps the
    first table's metadata: give a concatenation the metadata of
    `EvaluationMetadata.merge`); and `TypeError` for an argument of none of
    these types.
    """
    designs: tuple[dict[str, Any], ...] = ()
    labels: dict[str, str | None] = {}
    source: str | None = None
    rows: str | None = None
    recorded: Sequence[Mapping[str, Any]] = ()
    explained: Mapping[str, Any] | None = None
    if _is_document(metadata) and "schema_version" in metadata:
        # The metadata's own JSON document, as `to_json` writes it.
        metadata = EvaluationMetadata(**metadata)
    if isinstance(metadata, pa.Table):
        whole = EvaluationMetadata.from_table(metadata)
        setup: Mapping[str, Any]
        setup, designs, recorded, rows = _of_rows(whole, metadata)
        labels = _design_labels(designs, whole.covers)
        source = _sources(recorded, records=whole.source_records)
    elif isinstance(metadata, EvaluationMetadata):
        setup = metadata.setup.model_dump(mode="json")
        designs = metadata.designs
        labels = _design_labels(designs, metadata.covers)
        recorded = metadata.sources
        source = _sources(recorded, records=metadata.source_records)
    elif isinstance(metadata, EvaluationSetup):
        setup = metadata.model_dump(mode="json")
    elif _is_document(metadata):
        # A plan's explain() document: its setup, and its sources and designs
        # as far as they are known before execution.
        explained = metadata
        setup = _get(metadata, "setup", Mapping, "document") or {}
        source = _sources(_objects(metadata, "sources", "document"), planned=True)
    elif isinstance(metadata, Mapping):
        setup = metadata
    else:
        raise TypeError(
            "setup_text takes EvaluationMetadata, a comparison table, an "
            "EvaluationSetup, a setup mapping or a plan's explain(format='json') "
            f"document, got {type(metadata).__name__}"
        )
    samples = _objects(setup, "samples", "setup")
    evaluations = _objects(setup, "evaluations", "setup")
    by_hash = {
        str(sample["sample_hash"]): sample
        for sample in samples
        if sample.get("sample_hash") is not None
    }
    assets: list[str] = []
    scenarios: list[str] = []
    units: list[str] = []
    for i, sample in enumerate(samples):
        where = f"setup.samples[{i}]"
        assets += [str(asset) for asset in _list(sample, "asset_ids", where)]
        scenarios += [_get(sample, "scenario", str, where) or "not stated"]
        units += [_get(sample, "loss_unit", str, where) or "an unstated unit"]
    assets, scenarios, units = map(_first_seen, (assets, scenarios, units))
    unit = units[0] if len(units) == 1 else "loss units"

    levels, scopes, described = _levels_of(evaluations, by_hash, assets)
    if described is not None:
        level = described
    elif not assets:
        level = "asset, the assets not stated"
    elif len(assets) == 1:
        level = f"asset ({assets[0]})"
    else:
        level = f"asset, one row per asset ({_join(assets)}); no portfolio total"
    basis = setup.get("basis")
    if basis == "annual":
        basis = "annual, one draw is one year"
    lines = [] if rows is None else [rows]
    lines.append(
        f"Level: {level}. Scenario: {_join(scenarios) or 'not stated'}. "
        f"Basis: {basis or 'not stated'}."
    )

    pricings = [
        _pricing(evaluation.get("pricing"), f"setup.evaluations[{i}]")
        for i, evaluation in enumerate(evaluations)
    ]
    shared = len(set(pricings)) <= 1
    seeds = [
        _seeds(evaluation, f"setup.evaluations[{i}]")
        for i, evaluation in enumerate(evaluations)
    ]
    plan_seeds = {plan for plan, _ in seeds if plan is not None}
    fresh_seeds = {fresh for _, fresh in seeds if fresh is not None}
    one_seed = len(plan_seeds) <= 1 and len(fresh_seeds) <= 1
    written: set[tuple[str, str]] = set()
    for i, evaluation in enumerate(evaluations):
        text = _split(evaluation, by_hash, f"setup.evaluations[{i}]")
        if _draws_from_one_curve(evaluation, samples, recorded):
            evaluated = by_hash.get(str(evaluation.get("sample_hash"))) or {}
            text += f" {_ONE_WINDOW if evaluated.get('members') else _ONE_CURVE}"
        if not one_seed and _seeds_text(*seeds[i]):
            text += f" Seeds: {_seeds_text(*seeds[i])}."
        if not shared:
            text += f" Pricing: {pricings[i]}."
        # Comparisons of one sample at different levels share their split;
        # evaluations of different samples keep a line each.
        if evaluation.get("sample_hash") is not None:
            key = (str(evaluation["sample_hash"]), text)
            if key in written:
                continue
            written.add(key)
        lines.append(text)
    if not evaluations:
        lines.append("Split: not known.")
    if one_seed and (plan_seeds or fresh_seeds):
        plan = next(iter(plan_seeds), None)
        fresh = next(iter(fresh_seeds), None)
        lines.append(f"Seeds: {_seeds_text(plan, fresh)}.")
    across = _get(setup, "scenarios", Mapping, "setup") is not None
    lines += _scenario_lines(setup, evaluations, by_hash, designs, labels, unit)
    drift = _get(setup, "drift", Mapping, "setup")
    if drift is not None:
        lines.append(
            "Drift: each row's difference and ratio are against the row of the same "
            f"cover or design on scenario {drift.get('reference')}, with no ranking."
        )

    # Every sample is checked, also those the text describes only once.
    years = [_years(sample, f"setup.samples[{i}]") for i, sample in enumerate(samples)]
    descriptions = [
        _sample_text(sample, f"setup.samples[{i}]") for i, sample in enumerate(samples)
    ]
    if not samples:
        lines.append("Sample: not known.")
    elif len(set(descriptions)) == 1:
        lines.append(f"Sample: {descriptions[0]}")
    else:
        for i, (span, text) in enumerate(zip(years, descriptions, strict=True)):
            # Design draws and fresh draws of one size read the same by their years.
            roles = _list(samples[i], "roles", f"setup.samples[{i}]")
            role = (
                f", the {' and '.join(roles)} sample" if years.count(span) > 1 else ""
            )
            lines.append(f"Sample of {span}{role if roles else ''}: {text}")
    if any(
        sample.get("components") and "evaluation" in _list(sample, "roles", "sample")
        for sample in samples
    ):
        lines.append(
            "Combined loss: every row's metrics are on the combined loss of its "
            "sample, so the row of a cover on one hazard sets that cover's payout "
            "against the loss of all the hazards."
        )

    if source is not None:
        lines.append(source)
    lines += _location_lines(recorded, by_hash)
    lines += _trend_lines(samples)
    lines += _pool_lines(samples)
    if not across:
        lines += _window_lines(samples)
    notes = _notes(samples, evaluations)
    if notes:
        lines.append(f"Sample notes: {' '.join(notes)}")

    if not pricings:
        pricing = "not stated"
    else:
        pricing = pricings[0] if shared else "as each split states"
    where_priced = (
        "once on the reference's design block" if across else "on the evaluation sample"
    )
    priced = (
        f"Pricing: {pricing}, {where_priced}, for the covers given as "
        "is; a design keeps its design premium, and premium_basis says which "
        "premium a row uses."
    )
    if "total" in levels and "individual" in scopes:
        priced += (
            " The total row of a set of covers holds the sum of their premiums, "
            "one per contract, unless a total premium is stated."
        )
    if "joint" in scopes:
        priced += " A joint design's premium is that of one contract on the total."
    if "package" in levels:
        priced += (
            " The package row holds the sum of its contracts' premiums, one per "
            "contract, unless a total premium is stated."
        )
    if any(
        evaluation.get("method") == "kfold_pooled" and evaluation.get("design_samples")
        for evaluation in evaluations
    ):
        priced += (
            " On a pooled row each year pays its own fold's design premium, and "
            "premium is their mean."
        )
    part_kinds = [
        str(kind)
        for i, evaluation in enumerate(evaluations)
        for kind in _list(evaluation, "part_kinds", f"setup.evaluations[{i}]")
    ]
    if not part_kinds:
        lines.append(priced)
    else:
        lines.append(
            "Pricing: a payout breakdown holds no premiums; a design's line states "
            "its pricing."
        )
    lines += [
        _design(record, labels, unit, by_hash)
        for record in designs
        if record.get("kind") != "pooled"
    ]
    lines += _row_lines(evaluations, designs, by_hash, unit)
    lines += _package_lines(evaluations, unit)
    options = setup.get("metric_options")
    lines += _rank_block_lines(
        evaluations,
        samples,
        by_hash,
        designs,
        labels,
        options if isinstance(options, Mapping) else None,
    )
    if "attribution" in part_kinds:
        lines.append(
            "Payout breakdown: the rows of a MaxOf cover are an attribution: each "
            "year's payout goes to the component paying most, in equal shares when "
            "several tie. One contract pays one amount."
        )
    if explained is not None:
        lines += _planned_designs(explained, assets, unit)
    lines += [
        _frontier(record, by_hash, unit, f"setup.frontier[{i}]")
        for i, record in enumerate(_objects(setup, "frontier", "setup"))
    ]
    intervals = _get(setup, "intervals", Mapping, "setup")
    if intervals is not None:
        lines.append(_intervals(intervals))
    if isinstance(options, Mapping):
        lines.append(_metrics(options, unit))
    else:
        lines.append("Metrics: the options are not stated.")
    return "\n".join(lines)
