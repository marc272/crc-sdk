"""Outcome samples: years with a probability, an index value and a loss each.

Every design and evaluation runs on an `OutcomeSample`. One row (a draw) is
one year: a cover pays once per draw, on that draw's index, against that
draw's loss. The rows are simulated years (`sample_curve`), observed or
modelled years (`sample_annual`), or the rows of a user table
(`OutcomeSample.from_table`).

- `period` labels the rows: the calendar year of a per-year sample, the draw
  number of a Monte Carlo sample or of a pool of model-years, whose rows record
  their year, model and member. It is strictly increasing.
- `probability` is each row's probability; it sums to 1.
- `index` and `loss` have one row per draw and one column per asset, in
  `asset_ids` order: `(n, n_assets)`.
- `hazard` is the value the loss model read, kept when it differs from the
  index (a basis-risk model or a separate index series); otherwise `None`. On
  a sample of several hazards it is the first component's, so on a first
  component with no impact, which no loss reads, it is the hazard at the asset
  that a basis-risk model with `apply_hazard`, such as `HazardNoise`, gave.
- `provenance` (`SampleProvenance`) records the source, the semantics, the
  loss model, the basis-risk model and the generator state.

A sample holds one asset or more, with distinct ids, and a sample of several
states where the dependence between them comes from. `OutcomeSample.asset`
takes one asset's sample out of it, and `OutcomeSample.join` puts one-asset
samples of the same years side by side. `design`, `evaluate` and `compare`
work on one asset's sample.

A sample of one asset may hold several named hazards, its `index_components`:
one `IndexComponent` per hazard, the first being what `index` holds, while
`loss` is the combined loss. Such a sample comes from the same years of several
hazards (`sample_annual_hazards`), from fitted curves under a stated dependence
(`sample_curve_hazards`) or from a user table.
`OutcomeSample.component` takes one hazard's sample out of it.

`holdout` and `year_block_kfold` split a sample into contiguous blocks of
periods, so the held-out block of a per-year sample is its latest years. A
pool of model-years is split by its distinct years, not by its rows, so every
run holds out the same years.
Monte Carlo draws are independent, so a fresh sample from another generator
state is the plainer out-of-sample test.

`OutcomeSample.to_table` returns a sample as its long table, with the units
and the provenance in the schema metadata, and
`OutcomeSample.from_written_table` reads such a table back.

Everything here works on numpy arrays and Arrow tables and does no I/O:
read a Parquet file with `pyarrow.parquet.read_table` and pass the table.
"""

from __future__ import annotations

import dataclasses
import json
import math
import operator
import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    EvaluationError,
    LeakageError,
    SampleError,
    ScenarioError,
)
from crc_sdk.parametric.index import (
    CompositeIndex,
    IndexDefinition,
    Tail,
    _component_name,
)
from crc_sdk.parametric.metadata import _series_pathway
from crc_sdk.pathways import pathways_equivalent, resolve_pathway

__all__ = [
    "SAMPLE_COMPONENTS_SCHEMA_VERSION",
    "SAMPLE_METADATA_KEY",
    "SAMPLE_SCHEMA_VERSION",
    "SAMPLE_SEMANTICS_VERSION",
    "ComponentLoss",
    "HazardFrom",
    "IndexComponent",
    "OutcomeSample",
    "SampleProvenance",
    "SemanticsSource",
    "SourceKind",
    "holdout",
    "year_block_kfold",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SAMPLE_METADATA_KEY = "crc.parametric.sample"
"""The Parquet schema metadata key of `OutcomeSample.to_table()`."""

SAMPLE_SCHEMA_VERSION = "1.0"
"""The `schema_version` of a one-hazard sample's hash document and table metadata."""

SAMPLE_COMPONENTS_SCHEMA_VERSION = "1.1"
"""The `schema_version` of the same documents of a sample with index components."""

SAMPLE_SEMANTICS_VERSION = "crc-parametric-sample/1"
"""The `semantics_version` of a sample's hash document and table metadata."""

SourceKind = Literal["monte_carlo", "annual", "table"]
"""Where a sample's rows come from: curve draws, per-year values or a table."""

SemanticsSource = Literal["dataset", "caller", "data"]
"""Who stated what a row means: the dataset, the caller, or the data itself."""

ComponentLoss = Literal["own", "combined"]
"""The loss of a component's view: its hazard's own, or the combined loss."""

HazardFrom = Literal["curve_row", "source_plan", "table_column", "stated"]
"""What names the hazard a sample's loss reads: its record, or the caller."""

_INT32_MAX = 2**31 - 1
_PROBABILITY_TOLERANCE = 1e-12
_HOLDOUT_TOLERANCE = 1e-9
_IDENTIFIER = re.compile(r"^\S(?:[\s\S]*\S)?$")
_DEFAULT_SCENARIO = "baseline"
"""The label of a sample nobody named: what a read of it refuses is left to a set."""

COMONOTONE_MESSAGE = (
    "the index and the loss are comonotone: the loss is a function of the "
    "index, so a cover can track the loss almost exactly. State a basis-risk "
    "model (HazardNoise, ImpactNoise, or an index series read elsewhere than "
    "the loss) to measure basis risk"
)

_TABLE_COMONOTONE_NOTE = (
    "The table's loss is a monotone function of its index: index and loss are "
    "comonotone."
)


# -- plain JSON documents ----------------------------------------------------


def _freeze(value: Any) -> Any:
    """Return an immutable copy: mappings as read-only proxies, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return plain dicts and lists, the inverse of `_freeze`."""
    if isinstance(value, BaseModel):
        return _thaw(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _check_json(name: str, value: Any) -> None:
    """Raise `SampleError` unless `value` is a document of plain JSON types."""
    try:
        json.dumps(_thaw(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise SampleError(f"{name} must hold plain JSON values: {error}") from error


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SampleError(
            f"{name} must be a non-empty string without surrounding whitespace, "
            f"got {value!r}"
        )
    return value


def _source_document(source: Any) -> dict[str, Any]:
    """Return a caller's source description as a plain JSON document.

    A string is a name, a pydantic model (`SourceProvenance`) is dumped, and a
    mapping (for example a plan's `explain(format="json")`) is copied.
    """
    if source is None:
        return {}
    if isinstance(source, str):
        return {"name": _identifier("source", source)}
    if isinstance(source, BaseModel):
        document: dict[str, Any] = source.model_dump(mode="json")
    elif isinstance(source, Mapping):
        document = _thaw(source)
    else:
        raise TypeError(
            "source must be a name, a SourceProvenance or a mapping, "
            f"got {type(source).__name__}"
        )
    _check_json("source", document)
    return document


def _licence_entry(role: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the licence and attribution a source document declares."""

    def text(key: str) -> str | None:
        value = document.get(key)
        return None if value is None else str(value)

    name = document.get("dataset") or document.get("name")
    return {
        "role": role,
        "source": None if name is None else str(name),
        "licence": text("licence"),
        "attribution": text("attribution"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _generator_record(rng: np.random.Generator, step: str) -> dict[str, Any]:
    """Return the bit generator's name and its state, before `step` draws."""
    bit_generator = rng.bit_generator
    return {
        "step": step,
        "bit_generator": type(bit_generator).__name__,
        "state": json.dumps(bit_generator.state, sort_keys=True, default=_json_default),
    }


def _require_generator(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    return rng


# -- Arrow tables ------------------------------------------------------------


def _arrow_table(data: Any, what: str) -> Any:
    """Return `data` as a `pyarrow.Table`, without reading any file."""
    if isinstance(data, (str, bytes, os.PathLike)):
        raise TypeError(
            f"{what} must be an in-memory table; this module does no I/O, so "
            "read the file first (for example pyarrow.parquet.read_table)"
        )
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    if isinstance(data, Mapping) or hasattr(data, "__arrow_c_stream__"):
        try:
            return pa.table(data)
        except (TypeError, ValueError, pa.ArrowException) as error:
            raise TypeError(f"{what} cannot be read as an Arrow table") from error
    if isinstance(data, Iterable):
        batches = list(data)
        if not batches or not all(isinstance(b, pa.RecordBatch) for b in batches):
            raise TypeError(f"{what} must be a table or Arrow record batches")
        return pa.Table.from_batches(batches)
    raise TypeError(f"{what} must be an Arrow table, got {type(data).__name__}")


def _column(table: Any, name: str, what: str) -> Any:
    if not isinstance(name, str) or name not in table.column_names:
        raise SampleError(
            f"{what} has no column {name!r}; its columns are {table.column_names}"
        )
    return table.column(name)


def _integer_values(column: Any, name: str) -> IntArray:
    if not (pa.types.is_integer(column.type)):
        raise SampleError(f"column {name!r} must hold integers, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def _float_values(column: Any, name: str) -> FloatArray:
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise SampleError(f"column {name!r} must be numeric, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def _string_values(column: Any, name: str) -> list[str]:
    if pa.types.is_integer(column.type):
        column = column.cast(pa.string())
    elif pa.types.is_dictionary(column.type):
        column = column.cast(pa.string())
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        raise SampleError(f"column {name!r} must hold strings, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    return [str(value) for value in column.to_pylist()]


# -- array validation --------------------------------------------------------


def _periods(value: Any) -> IntArray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise SampleError(f"period must hold integers, got dtype {array.dtype}")
    if array.ndim != 1 or array.shape[0] == 0:
        raise SampleError(f"period must be 1-D and non-empty, got shape {array.shape}")
    if array.dtype.kind == "u" and array.size and int(array.max()) > 2**63 - 1:
        raise SampleError("period values must fit in a signed 64-bit integer")
    periods = np.array(array, dtype=np.int64)
    if periods.shape[0] > 1 and not (np.diff(periods) > 0).all():
        raise SampleError("period must be strictly increasing")
    return periods


def _float_array(name: str, value: Any) -> FloatArray:
    try:
        return np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError(f"{name} must be numeric") from error


def _matrix(name: str, value: Any, n: int, n_assets: int) -> FloatArray:
    array = _float_array(name, value)
    if array.ndim == 1 and n_assets == 1:
        array = array.reshape(-1, 1)
    if array.shape != (n, n_assets):
        raise SampleError(
            f"{name} must have shape ({n}, {n_assets}), one row per period and "
            f"one column per asset; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise SampleError(f"{name} must be finite")
    return array


def _asset_note(asset: str, note: str) -> str:
    """One asset's note, in the notes of a sample of several assets."""
    return f"Asset {asset}: {note}"


def _component_note(name: str, note: str) -> str:
    """One component's note, in the notes of a sample with index components."""
    return f"Component {name}: {note}"


def _read_only(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def _hash_document(
    *,
    asset_ids: Sequence[str],
    period: IntArray,
    probability: FloatArray,
    index: FloatArray,
    loss: FloatArray,
    hazard: FloatArray | None,
    index_unit: str,
    loss_unit: str,
    semantics: str,
    scenario: str,
    components: Mapping[str, IndexComponent] | None = None,
) -> dict[str, Any]:
    """The document behind `content_hash`: the long table, sorted, and its units.

    A sample with index components adds each one's values, in component order,
    under another `schema_version`; a one-hazard sample's document holds no
    trace of them.
    """
    order = sorted(range(len(asset_ids)), key=lambda j: asset_ids[j].encode("utf-8"))
    n_assets = len(order)

    def flat(array: FloatArray) -> list[float]:
        values: list[float] = array[:, order].reshape(-1).tolist()
        return values

    document: dict[str, Any] = {
        "asset_id": [asset_ids[j] for j in order] * period.shape[0],
        "hazard": None if hazard is None else flat(hazard),
        "index": flat(index),
        "index_unit": index_unit,
        "loss": flat(loss),
        "loss_unit": loss_unit,
        "period": np.repeat(period, n_assets).tolist(),
        "probability": np.repeat(probability, n_assets).tolist(),
        "scenario": scenario,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "semantics": semantics,
        "semantics_version": SAMPLE_SEMANTICS_VERSION,
    }
    if components is not None:
        document["schema_version"] = SAMPLE_COMPONENTS_SCHEMA_VERSION

        def column(array: FloatArray | None) -> list[float] | None:
            return None if array is None else array[:, 0].tolist()

        document["components"] = [
            {
                "hazard": column(component.hazard),
                "name": name,
                "own_loss": column(component.own_loss),
                "tail": component.tail,
                "unit": component.unit,
                "values": column(component.values),
            }
            for name, component in components.items()
        ]
    return document


def _is_monotone_function(index: FloatArray, loss: FloatArray) -> bool:
    """Whether the loss is a non-constant monotone function of the index."""
    if loss.max() == loss.min():
        return False
    order = np.lexsort((loss, index))
    x, y = index[order], loss[order]
    tied = np.diff(x) == 0
    steps = np.diff(y)
    if (steps[tied] != 0).any():
        return False
    steps = steps[~tied]
    return bool((steps >= 0).all() or (steps <= 0).all())


# -- the provenance ----------------------------------------------------------


def _check_pool(pool: Mapping[str, Any]) -> None:
    """Raise `SampleError` unless `pool` names its runs and each row's run."""
    if not {"pathway", "runs", "year", "run"} <= set(pool):
        raise SampleError(
            "pool must hold a pathway, its runs, and a year and a run per row"
        )
    for name in ("runs", "year", "run"):
        if isinstance(pool[name], (str, Mapping)) or not isinstance(
            pool[name], Sequence
        ):
            raise SampleError(f"pool's {name} must be a list")
    runs, years, positions = pool["runs"], pool["year"], pool["run"]
    if not all(
        isinstance(run, Mapping) and {"model", "member", "lineage"} <= set(run)
        for run in runs
    ):
        raise SampleError("each run of a pool needs a model, a member and a lineage")
    if len(years) != len(positions):
        raise SampleError(
            f"a pool needs one year and one run per row, got {len(years)} years "
            f"and {len(positions)} runs"
        )
    if any(
        isinstance(position, bool)
        or not isinstance(position, (int, np.integer))
        or not 0 <= position < len(runs)
        for position in positions
    ):
        raise SampleError(f"the rows of a pool name runs outside its {len(runs)} runs")


def _check_hazard(hazard: Mapping[str, Any]) -> None:
    """Raise `SampleError` unless `hazard` says what names the hazard, and what."""
    if set(hazard) != {"name", "from"}:
        raise SampleError(
            f"hazard must hold a name and where it comes from, got {sorted(hazard)}"
        )
    _identifier("hazard's name", hazard["name"])
    if hazard["from"] not in get_args(HazardFrom):
        raise SampleError(
            f"hazard's from must be one of {get_args(HazardFrom)}, "
            f"got {hazard['from']!r}"
        )


@dataclass(frozen=True, eq=False)
class SampleProvenance:
    """Where a sample's rows come from, and how they were made.

    - `source_kind`: `"monte_carlo"` (draws from a curve), `"annual"`
      (per-year values) or `"table"` (a user table).
    - `sources`: one document per input, each with a `role` (`"hazard"`,
      `"index"` or `"table"`): for a curve, its dataset metadata, the curve's
      parameters and the caller's details (horizon, pathway, file checksum);
      for a per-year series, its provenance; for a table, the caller's name
      and the table's content hash.
    - `semantics_source`: who stated that a row is a year: `"dataset"` (the
      curve's metadata), `"caller"` (`assume_semantics=`, or a table's
      `semantics="annual"`) or `"data"` (per-year values are years).
    - `probability_semantics`: the curve's probability semantics, for Monte
      Carlo samples; otherwise `None`.
    - `loss_model` and `basis_risk`: their documents; `None` for a table. On a
      sample of several hazards from curves or per-year series, `loss_model` is
      `{"kind": "components", "components": [{"name": name, "loss_model":
      document}, ...]}`, in component order, for the components with an impact.
    - `rng`: for every random step, in draw order, the bit generator's name
      and its state before the step, as JSON text. `generator()` rebuilds it.
    - `licences`: the licence and attribution each input declares.
    - `details`: what the source counted: draws outside the fitted support,
      draws in a hurdle atom, dropped years, the matched grid cell, ...
    - `notes`: plain sentences for the setup text next to a number.
    - `lineage`: identifies the source of the rows. Two samples with the same
      lineage and a shared period overlap; out-of-sample evaluation on them
      raises `LeakageError`.
    - `parent` and `selection`: for a sample split or selected from another,
      the content hash of the sample it came from and the steps taken.
    - `dependence`: for a sample of several assets, where the dependence
      between them comes from; such a sample must state it. `{"kind": "data"}`
      is the data's own: the rows are the same years of the data. `None` on a
      one-asset sample.
    - `parts`: for a sample of several assets, each asset's own provenance
      document, by asset id. `None` on a one-asset sample.
    - `trend`: how a per-year sample treats the climate trend:
      `{"kind": "as_recorded"}`; `{"kind": "recent_years", "n": n, "years":
      [first, last]}`; or `{"kind": "linear_trend", "method": "ols_additive",
      "reference_year": t0, "min_years": m, "fit_years": [...], "fits": [...]}`,
      with one `{"role", "slope", "intercept", "unit"}` per series. `None` on
      Monte Carlo and table samples. The lineage leaves it out: a design on
      adjusted years, evaluated on the same years as recorded, shares their
      rows.
    - `pool`: for pooled model-years (`sample_annual_pool`), who each row is:
      `pathway`; `runs`, one document per run with its `model`, `member`,
      `pathway`, `lineage`, the provenance of its `series` and the
      `dropped_years` it had no value for; and `year` and `run`, one entry per
      row, the calendar year and the run's position in `runs`. `select`,
      `holdout` and `year_block_kfold` keep the entries of the rows they keep.
      `None` on every other sample.
    - `hazard_dependence`: for a sample with index components, where the
      dependence between the hazards comes from; such a sample must state it.
      `{"kind": "data"}` is the data's own; on a sample drawn from curves it is
      the record of the stated model. `None` on a one-hazard sample.
    - `combine`: for a sample with index components, how the combined loss was
      made: `{"kind": "table"}` when a table states it, the rule's document with
      its sentence under `statement`, or `{"kind": "one_impact", "component":
      name, "statement": ...}` when only one component has an impact and no rule
      is needed. `None` on a one-hazard sample.
    - `components`: for a sample with index components, each component's own
      provenance document, by name. `None` on a one-hazard sample.
    - `source_lineages`: for a sample whose rows read several series or streams
      of draws, each one's lineage by name: a component's name on a sample of
      several hazards from per-year series or curves, `"hazard"` and `"index"`
      on a per-year sample with an index series. It is the lineage that source
      has in a sample of its own, so a leakage check sees a year of one source
      through any sample that holds it. A component's view keeps the sources it
      reads. `None` on every other sample.
    - `hazard`: the hazard the sample's loss reads, as `{"name": name, "from":
      where}`, `where` being a `HazardFrom`: `"curve_row"` (the canonical row),
      `"source_plan"` (a source document's `recipe` or `hazard_name`),
      `"table_column"` (a one-valued `recipe` column) or `"stated"`
      (`hazard_name=`, for a source that names none). It is the series the loss
      reads, not a separate index series, and it stays out of `content_hash`,
      `design_id` and the lineage. `None` when nothing names it, and on a
      sample of several hazards, whose `components` hold one each.

    Mappings and sequences are stored read-only; `to_document()` returns
    plain JSON types, without the three fields of a sample with components and
    without `source_lineages` and `hazard` when they are `None`. Raises
    `SampleError` for unknown kinds or values that are not plain JSON.
    """

    source_kind: SourceKind
    sources: tuple[Mapping[str, Any], ...]
    semantics_source: SemanticsSource
    probability_semantics: str | None = None
    loss_model: Mapping[str, Any] | None = None
    basis_risk: Mapping[str, Any] | None = None
    rng: tuple[Mapping[str, Any], ...] = ()
    licences: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    lineage: str | None = None
    parent: str | None = None
    selection: tuple[Mapping[str, Any], ...] = ()
    dependence: Mapping[str, Any] | None = None
    parts: Mapping[str, Mapping[str, Any]] | None = None
    trend: Mapping[str, Any] | None = None
    pool: Mapping[str, Any] | None = None
    hazard_dependence: Mapping[str, Any] | None = None
    combine: Mapping[str, Any] | None = None
    components: Mapping[str, Mapping[str, Any]] | None = None
    source_lineages: Mapping[str, str] | None = None
    hazard: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in get_args(SourceKind):
            raise SampleError(
                f"source_kind must be one of {get_args(SourceKind)}, "
                f"got {self.source_kind!r}"
            )
        if self.semantics_source not in get_args(SemanticsSource):
            raise SampleError(
                f"semantics_source must be one of {get_args(SemanticsSource)}, "
                f"got {self.semantics_source!r}"
            )
        for name in ("probability_semantics", "lineage", "parent"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise SampleError(f"{name} must be a string or None, got {value!r}")
        for name in ("sources", "rng", "licences", "selection"):
            value = getattr(self, name)
            if isinstance(value, (str, Mapping)) or not isinstance(value, Sequence):
                raise SampleError(f"{name} must be a sequence of mappings")
            if not all(isinstance(item, Mapping) for item in value):
                raise SampleError(f"{name} must be a sequence of mappings")
        for record in self.rng:
            if not {"bit_generator", "state", "step"} <= set(record):
                raise SampleError(
                    "each rng record needs a step, a bit_generator and a state"
                )
        for name in (
            "loss_model",
            "basis_risk",
            "dependence",
            "parts",
            "trend",
            "pool",
            "hazard_dependence",
            "combine",
            "components",
            "source_lineages",
            "hazard",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise SampleError(f"{name} must be a mapping or None")
        for name in ("dependence", "trend", "hazard_dependence", "combine"):
            value = getattr(self, name)
            if value is not None and "kind" not in value:
                raise SampleError(f"{name} must name its kind")
        if self.hazard is not None:
            _check_hazard(self.hazard)
        if self.pool is not None:
            _check_pool(self.pool)
        if self.parts is not None and not all(
            isinstance(part, Mapping) for part in self.parts.values()
        ):
            raise SampleError("parts must map each asset id to a provenance document")
        if self.components is not None and not all(
            isinstance(part, Mapping) for part in self.components.values()
        ):
            raise SampleError(
                "components must map each component name to a provenance document"
            )
        if self.source_lineages is not None and not all(
            isinstance(name, str) and isinstance(lineage, str)
            for name, lineage in self.source_lineages.items()
        ):
            raise SampleError("source_lineages must map each source to its lineage")
        if not isinstance(self.details, Mapping):
            raise SampleError("details must be a mapping")
        if isinstance(self.notes, str) or not all(
            isinstance(note, str) for note in self.notes
        ):
            raise SampleError("notes must be a sequence of strings")
        for name in (
            "sources",
            "loss_model",
            "basis_risk",
            "rng",
            "licences",
            "details",
            "selection",
            "dependence",
            "parts",
            "trend",
            "pool",
            "hazard_dependence",
            "combine",
            "components",
            "source_lineages",
            "hazard",
        ):
            value = getattr(self, name)
            _check_json(name, value)
            object.__setattr__(self, name, _freeze(_thaw(value)))
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_document(self) -> dict[str, Any]:
        """Return every field as plain JSON types.

        The fields of a sample with index components, `source_lineages` and
        `hazard` are left out when they are `None`, so the document of a sample
        that reads one source and names no hazard does not change.
        """
        return {
            item.name: _thaw(getattr(self, item.name))
            for item in dataclasses.fields(self)
            if item.name not in _LEFT_OUT_WHEN_NONE
            or getattr(self, item.name) is not None
        }

    def generator(self, step: int = 0) -> np.random.Generator:
        """Return a generator in the recorded state before random step `step`.

        Step 0 is the state at entry of the call that built the sample, so
        calling the same function again with this generator rebuilds the
        sample exactly. Raises `SampleError` if nothing was drawn or the bit
        generator is not one of numpy's.
        """
        if not self.rng:
            raise SampleError("the sample was built without a random generator")
        record = self.rng[step]
        name = record["bit_generator"]
        bit_type = getattr(np.random, str(name), None)
        if not (
            isinstance(bit_type, type) and issubclass(bit_type, np.random.BitGenerator)
        ):
            raise SampleError(f"cannot rebuild the bit generator {name!r}")
        bit_generator = bit_type()
        bit_generator.state = json.loads(record["state"])
        return np.random.Generator(bit_generator)

    def _selected(
        self,
        *,
        parent: str,
        lineage: str,
        step: Mapping[str, Any],
        rows: NDArray[np.bool_],
    ) -> SampleProvenance:
        pool = self.pool
        if pool is not None:
            pool = {
                **pool,
                "year": np.asarray(pool["year"], dtype=np.int64)[rows].tolist(),
                "run": np.asarray(pool["run"], dtype=np.int64)[rows].tolist(),
            }
        return dataclasses.replace(
            self,
            parent=parent,
            lineage=lineage,
            selection=(*self.selection, step),
            pool=pool,
        )


_COMPONENT_FIELDS = frozenset({"hazard_dependence", "combine", "components"})
"""The provenance fields of a sample with index components."""

_LEFT_OUT_WHEN_NONE = _COMPONENT_FIELDS | {"source_lineages", "hazard"}
"""The provenance fields a document holds only when they are set."""


# -- the components of a sample ----------------------------------------------


def _component_column(name: str, value: Any) -> FloatArray:
    """Return a component's column as a finite, read-only `(n, 1)` array."""
    array = _float_array(name, value)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != 1 or array.shape[0] == 0:
        raise SampleError(f"{name} must have shape (n, 1), got {array.shape}")
    if not np.isfinite(array).all():
        raise SampleError(f"{name} must be finite")
    return _read_only(array)


@dataclass(frozen=True, eq=False)
class IndexComponent:
    """One named hazard of a sample: its index, unit, tail and own loss.

    `values` is the component's index, `(n, 1)`, finite; a 1-D array is read
    as one column. `unit` is its unit, and `tail` the severe side, `"upper"`
    or `"lower"`. `hazard` is the value the loss model read, kept when it
    differs from the index, else `None`. On a component with no impact, which
    no loss reads, it is the hazard at the asset that a basis-risk model with
    `apply_hazard`, such as `HazardNoise`, gave. `own_loss` is the loss of
    this hazard alone, finite and non-negative, in the sample's loss unit, and
    `None` for a component with no impact. The arrays are read-only copies.

    Raises `SampleError` for any invalid field.
    """

    values: FloatArray
    _: KW_ONLY
    unit: str
    tail: Tail
    hazard: FloatArray | None = None
    own_loss: FloatArray | None = None

    def __post_init__(self) -> None:
        values = _component_column("values", self.values)
        _identifier("unit", self.unit)
        if self.tail not in get_args(Tail):
            raise SampleError(f"tail must be 'upper' or 'lower', got {self.tail!r}")
        hazard, own_loss = (
            None if value is None else _component_column(name, value)
            for name, value in (("hazard", self.hazard), ("own_loss", self.own_loss))
        )
        for name, column in (("hazard", hazard), ("own_loss", own_loss)):
            if column is not None and column.shape != values.shape:
                raise SampleError(
                    f"{name} must have the shape of values, {values.shape}; "
                    f"got {column.shape}"
                )
        if own_loss is not None and (own_loss < 0).any():
            raise SampleError("own_loss must be non-negative")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "hazard", hazard)
        object.__setattr__(self, "own_loss", own_loss)

    def _subset(self, rows: NDArray[np.bool_]) -> IndexComponent:
        return dataclasses.replace(
            self,
            values=self.values[rows],
            hazard=None if self.hazard is None else self.hazard[rows],
            own_loss=None if self.own_loss is None else self.own_loss[rows],
        )


def _checked_components(
    components: Any,
    *,
    n: int,
    n_assets: int,
    index: FloatArray,
    hazard: FloatArray | None,
    index_unit: str,
    provenance: SampleProvenance,
) -> Mapping[str, IndexComponent]:
    """Check a sample's components and return them read-only, in order."""
    if not isinstance(components, Mapping):
        raise SampleError(
            "index_components must map each component name to an IndexComponent, "
            f"got {type(components).__name__}"
        )
    if len(components) < 2:
        raise SampleError(
            f"index_components holds two components or more, got {len(components)}"
        )
    for name, component in components.items():
        _component_name(name, SampleError)
        if not isinstance(component, IndexComponent):
            raise SampleError(
                f"component {name!r} must be an IndexComponent, "
                f"got {type(component).__name__}"
            )
        if component.values.shape[0] != n:
            raise SampleError(
                f"component {name!r} has {component.values.shape[0]} rows, but the "
                f"sample holds {n} periods"
            )
    if n_assets != 1:
        raise SampleError(
            f"a sample with index components holds exactly one asset, got {n_assets}: "
            "several assets with several hazards need a dependence model over both"
        )
    first_name, first = next(iter(components.items()))
    if not np.array_equal(index, first.values):
        raise SampleError(
            f"index must hold the values of the first component, {first_name!r}"
        )
    if index_unit != first.unit:
        raise SampleError(
            f"index_unit {index_unit!r} must be the unit of the first component, "
            f"{first_name!r}, which is {first.unit!r}"
        )
    same_hazard = (
        hazard is None and first.hazard is None
        if hazard is None or first.hazard is None
        else np.array_equal(hazard, first.hazard)
    )
    if not same_hazard:
        raise SampleError(
            f"hazard must hold the hazard of the first component, {first_name!r}"
        )
    if provenance.hazard_dependence is None:
        raise SampleError(
            "a sample with index components must state where the dependence between "
            "its hazards comes from: set the provenance's hazard_dependence, for "
            'example {"kind": "data"} for the same years of the data'
        )
    if provenance.components is not None and set(provenance.components) != set(
        components
    ):
        raise SampleError(
            f"the provenance's components are for {sorted(provenance.components)}, "
            f"but the sample holds {sorted(components)}"
        )
    return MappingProxyType(dict(components))


# -- the sample --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class OutcomeSample:
    """Years (or simulated years) with a probability, an index and a loss each.

    One draw is one year: a cover pays once per draw, on its index, against
    its loss. `period` is the year label or the draw number, strictly
    increasing. `probability` is positive and sums to 1 within 1e-12.
    `index` and `loss` have shape `(n, n_assets)`, one column per asset in
    `asset_ids` order; a 1-D array is read as one column. They are finite,
    and the loss is non-negative. `hazard` is the value the loss model read,
    when it differs from the index, else `None`; on a sample of several
    hazards it is the first component's, so on a first component with no
    impact it is the hazard at the asset that a basis-risk model with
    `apply_hazard`, such as `HazardNoise`, gave. `index_unit` and
    `loss_unit` name the units; the loss unit must equal a cover's payout
    unit for designs and evaluations. `semantics` is `"annual"`.

    A sample holds one asset or more, with distinct ids. A sample of several
    must state where the dependence between them comes from, in its
    provenance's `dependence`: it is never assumed. `asset` takes one asset's
    sample out of it and `join` puts one-asset samples side by side.
    The arrays are read-only copies. Samples compare by identity; compare
    `content_hash` for equal contents.

    `index_components` maps each named hazard of a multi-hazard sample to its
    `IndexComponent`, two or more, in order and read-only; it is `None` on every
    other sample. Such a sample holds one asset and states the dependence
    between its hazards in its provenance's `hazard_dependence`. Its first
    component is the primary: `index`, `index_unit` and `hazard` hold its
    values, so code that reads one index reads it, while `loss` is the
    combined loss. Each name is non-empty, with no surrounding whitespace and
    no `.`, `+`, `[` or `]`. `component` takes one hazard's sample out of it.

    Raises `SampleError` for any invalid field.
    """

    asset_ids: tuple[str, ...]
    period: IntArray
    probability: FloatArray
    index: FloatArray
    loss: FloatArray
    hazard: FloatArray | None
    index_unit: str
    loss_unit: str
    semantics: Literal["annual"]
    provenance: SampleProvenance
    scenario: str = "baseline"
    index_components: Mapping[str, IndexComponent] | None = None

    def __post_init__(self) -> None:
        ids = self.asset_ids
        if isinstance(ids, str) or not isinstance(ids, Sequence):
            raise SampleError("asset_ids must be a sequence of asset ids")
        asset_ids = tuple(_identifier("asset id", value) for value in ids)
        if not asset_ids:
            raise SampleError("a sample holds one asset or more, got none")
        if len(set(asset_ids)) != len(asset_ids):
            raise SampleError(f"asset ids must be distinct, got {list(asset_ids)}")
        period = _periods(self.period)
        n = period.shape[0]
        probability = _float_array("probability", self.probability)
        if probability.shape != (n,):
            raise SampleError(
                f"probability must have shape ({n},), got {probability.shape}"
            )
        if not np.isfinite(probability).all() or (probability <= 0).any():
            raise SampleError("probability must be finite and positive")
        total = math.fsum(probability.tolist())
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise SampleError(
                f"probability must sum to 1 within {_PROBABILITY_TOLERANCE}, "
                f"got {total!r}"
            )
        index = _matrix("index", self.index, n, len(asset_ids))
        loss = _matrix("loss", self.loss, n, len(asset_ids))
        if (loss < 0).any():
            raise SampleError("loss must be non-negative")
        hazard = (
            None
            if self.hazard is None
            else _matrix("hazard", self.hazard, n, len(asset_ids))
        )
        _identifier("index_unit", self.index_unit)
        _identifier("loss_unit", self.loss_unit)
        if self.semantics != "annual":
            raise SampleError(
                f"semantics must be 'annual' (one draw is one year), got "
                f"{self.semantics!r}; 'event' is reserved"
            )
        if not isinstance(self.provenance, SampleProvenance):
            raise SampleError("provenance must be a SampleProvenance")
        if len(asset_ids) == 1 and not (
            self.provenance.dependence is None and self.provenance.parts is None
        ):
            raise SampleError(
                "a one-asset sample has no dependence between assets and no parts"
            )
        if len(asset_ids) > 1 and self.provenance.dependence is None:
            raise SampleError(
                f"a sample of {len(asset_ids)} assets must state where the dependence "
                "between them comes from: set the provenance's dependence, for "
                'example {"kind": "data"} for the same years of the data'
            )
        if self.provenance.parts is not None and set(self.provenance.parts) != set(
            asset_ids
        ):
            raise SampleError(
                f"the provenance's parts are for {sorted(self.provenance.parts)}, "
                f"but the sample holds {sorted(asset_ids)}"
            )
        pool = self.provenance.pool
        if pool is not None and len(pool["year"]) != n:
            raise SampleError(
                f"the provenance's pool names {len(pool['year'])} rows, but the "
                f"sample holds {n}"
            )
        _identifier("scenario", self.scenario)
        components: Mapping[str, IndexComponent] | None = None
        if self.index_components is not None:
            components = _checked_components(
                self.index_components,
                n=n,
                n_assets=len(asset_ids),
                index=index,
                hazard=hazard,
                index_unit=self.index_unit,
                provenance=self.provenance,
            )
        elif any(
            getattr(self.provenance, name) is not None for name in _COMPONENT_FIELDS
        ):
            raise SampleError(
                "a sample without index components has no dependence between hazards, "
                "no rule for a combined loss and no components in its provenance"
            )
        object.__setattr__(self, "index_components", components)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "period", _read_only(period))
        object.__setattr__(self, "probability", _read_only(probability))
        object.__setattr__(self, "index", _read_only(index))
        object.__setattr__(self, "loss", _read_only(loss))
        object.__setattr__(
            self, "hazard", None if hazard is None else _read_only(hazard)
        )

    def __len__(self) -> int:
        return int(self.period.shape[0])

    def __repr__(self) -> str:
        """A short description, which prints no arrays and stays short at any size.

        It gives the shape of `index` and `loss` (periods, assets), the asset
        ids (the first two and the last above three), the hazard (the stated
        hazard's name, or the components of a sample of several hazards), the
        units, the semantics (the annual basis), the scenario and the start of
        `content_hash`.
        """

        def shortened(names: Sequence[str]) -> str:
            shown = [repr(name) for name in names]
            if len(shown) > 3:
                shown = [*shown[:2], "...", shown[-1]]
            return f"[{', '.join(shown)}]"

        if self.index_components is not None:
            hazard = f"components={shortened(tuple(self.index_components))}"
        else:
            stated = self.provenance.hazard
            hazard = f"hazard_name={None if stated is None else stated['name']!r}"
        return (
            f"OutcomeSample(shape={self.index.shape}, "
            f"assets={shortened(self.asset_ids)}, {hazard}, "
            f"index_unit={self.index_unit!r}, loss_unit={self.loss_unit!r}, "
            f"semantics={self.semantics!r}, scenario={self.scenario!r}, "
            f"content_hash='{self.content_hash[:19]}...')"
        )

    @cached_property
    def content_hash(self) -> str:
        """`"sha256:<hex>"` of the sample's canonical long table and units.

        The document holds the long table (`to_table` without the scenario
        column), sorted by period and asset id, with the units, the
        semantics and the scenario, in the canonical JSON of cover documents.
        A sample with index components adds each one's name, unit, tail,
        values, hazard and own loss. The provenance is not part of it.
        """
        return content_hash(
            _hash_document(
                asset_ids=self.asset_ids,
                period=self.period,
                probability=self.probability,
                index=self.index,
                loss=self.loss,
                hazard=self.hazard,
                index_unit=self.index_unit,
                loss_unit=self.loss_unit,
                semantics=self.semantics,
                scenario=self.scenario,
                components=self.index_components,
            )
        )

    @property
    def lineage(self) -> str:
        """The identity of the rows' source, for leakage checks.

        The provenance's `lineage`, else the content hash of the sample this
        one was selected from, else this sample's own content hash.
        """
        return self.provenance.lineage or self.provenance.parent or self.content_hash

    def select(self, periods: ArrayLike) -> OutcomeSample:
        """Return the sample of the given periods, in period order.

        The probabilities are rescaled to sum to 1 on the selection. Raises
        `SampleError` for an empty selection, repeated periods, or periods
        the sample does not hold.
        """
        wanted = np.asarray(periods)
        if wanted.dtype.kind not in "iu" or wanted.ndim != 1 or wanted.size == 0:
            raise SampleError("periods must be a non-empty 1-D sequence of integers")
        wanted = wanted.astype(np.int64)
        if np.unique(wanted).size != wanted.size:
            raise SampleError("periods must not repeat")
        unknown = np.setdiff1d(wanted, self.period)
        if unknown.size:
            raise SampleError(
                f"the sample holds no periods {unknown[:10].tolist()}"
                + (" ..." if unknown.size > 10 else "")
            )
        rows = np.isin(self.period, wanted)
        return self._subset(rows, {"kind": "select", "periods": int(rows.sum())})

    def _subset(
        self, rows: NDArray[np.bool_], step: Mapping[str, Any]
    ) -> OutcomeSample:
        probability = self.probability[rows]
        probability = probability / math.fsum(probability.tolist())
        provenance = self.provenance._selected(
            parent=self.provenance.parent or self.content_hash,
            lineage=self.lineage,
            step=step,
            rows=rows,
        )
        return OutcomeSample(
            asset_ids=self.asset_ids,
            period=self.period[rows],
            probability=probability,
            index=self.index[rows],
            loss=self.loss[rows],
            hazard=None if self.hazard is None else self.hazard[rows],
            index_unit=self.index_unit,
            loss_unit=self.loss_unit,
            semantics=self.semantics,
            provenance=provenance,
            scenario=self.scenario,
            index_components=(
                None
                if self.index_components is None
                else {
                    name: component._subset(rows)
                    for name, component in self.index_components.items()
                }
            ),
        )

    def asset(self, asset_id: str) -> OutcomeSample:
        """Return one asset's sample: the same periods and probabilities.

        The provenance is the asset's own, from the provenance's `parts`
        when it has them, and `selection` records the step after the steps
        of that asset's own sample. The lineage is kept, so a leakage check
        still sees the source: of a joined sample it is the part's, otherwise
        this sample's. A hazard equal to the asset's index is dropped, as on
        any sample, so a part of a join comes back with its content hash.

        A one-asset sample returns itself. Raises `SampleError` for an asset
        the sample does not hold.
        """
        if asset_id not in self.asset_ids:
            raise SampleError(
                f"the sample holds no asset {asset_id!r}; it holds "
                f"{list(self.asset_ids)}"
            )
        if len(self.asset_ids) == 1:
            return self
        column = self.asset_ids.index(asset_id)
        index = self.index[:, column]
        hazard = None if self.hazard is None else self.hazard[:, column]
        if hazard is not None and np.array_equal(hazard, index):
            hazard = None
        own = self.provenance
        part = None if own.parts is None else _written_provenance(own.parts[asset_id])
        earlier = () if part is None else part.selection
        # Taking an asset commutes with a split, and a split stays the last
        # step: the setup text reads the block from the last one.
        step = {"kind": "asset", "asset_id": asset_id}
        provenance = dataclasses.replace(
            own if part is None else part,
            parent=own.parent or self.content_hash,
            lineage=(None if part is None else part.lineage) or self.lineage,
            selection=(*earlier, step, *own.selection),
            dependence=None,
            parts=None,
            pool=own.pool if part is None else _part_pool(own.pool, part.pool),
        )
        return OutcomeSample(
            asset_ids=(asset_id,),
            period=self.period,
            probability=self.probability,
            index=index,
            loss=self.loss[:, column],
            hazard=hazard,
            index_unit=self.index_unit,
            loss_unit=self.loss_unit,
            semantics=self.semantics,
            provenance=provenance,
            scenario=self.scenario,
        )

    def component(self, name: str, *, loss: ComponentLoss = "own") -> OutcomeSample:
        """Return one hazard's sample: the same periods and probabilities.

        Its `index`, `index_unit` and `hazard` are the component's. With
        `loss="own"` its loss is the component's `own_loss`, so a separate
        cover can be designed on its own hazard's loss; with
        `loss="combined"` it keeps this sample's combined loss, for a cover on
        this index designed against what the owner loses in total.

        The provenance is the component's own, from the provenance's
        `components` when it has them, and `selection` records the step after
        the steps of that component's own sample. The lineage is this sample's,
        so a leakage check still sees the source. `source_lineages` keeps the
        sources the view reads: its own, and with `loss="combined"` those of
        every component with an impact, which are in that loss.

        Raises `SampleError` on a sample without index components, for a
        component the sample does not hold, an unknown `loss`, and for
        `loss="own"` on a component with no impact, which has no own loss.
        """
        components = self.index_components
        if components is None:
            raise SampleError(
                "this sample holds one hazard, so it has no components to take; "
                "use it as it is"
            )
        if name not in components:
            raise SampleError(
                f"the sample holds no component {name!r}; it holds {list(components)}"
            )
        if loss not in get_args(ComponentLoss):
            raise SampleError(
                f"loss must be one of {get_args(ComponentLoss)}, got {loss!r}"
            )
        chosen = components[name]
        if loss == "own":
            if chosen.own_loss is None:
                raise SampleError(
                    f"component {name!r} has no impact, so it has no loss of its own: "
                    f"use component({name!r}, loss='combined') for this index against "
                    "the combined loss"
                )
            values = chosen.own_loss
        else:
            values = self.loss
        own = self.provenance
        part = (
            None
            if own.components is None
            else _written_provenance(own.components[name])
        )
        earlier: tuple[Mapping[str, Any], ...] = (
            _hazard_reorder(own) if part is None else part.selection
        )
        # Taking a component commutes with a split, and a split stays the last
        # step: the setup text reads the block from the last one.
        step = {"kind": "component", "name": name, "loss": loss}
        read = {name} | (
            {key for key, each in components.items() if each.own_loss is not None}
            if loss == "combined"
            else set()
        )
        provenance = dataclasses.replace(
            own if part is None else part,
            parent=own.parent or self.content_hash,
            lineage=self.lineage,
            selection=(*earlier, step, *own.selection),
            pool=own.pool,
            hazard_dependence=None,
            combine=None,
            components=None,
            source_lineages=(
                None
                if own.source_lineages is None
                else {
                    key: lineage
                    for key, lineage in own.source_lineages.items()
                    if key in read
                }
            ),
        )
        return OutcomeSample(
            asset_ids=self.asset_ids,
            period=self.period,
            probability=self.probability,
            index=chosen.values,
            loss=values,
            hazard=chosen.hazard,
            index_unit=chosen.unit,
            loss_unit=self.loss_unit,
            semantics=self.semantics,
            provenance=provenance,
            scenario=self.scenario,
        )

    @classmethod
    def join(cls, samples: Iterable[OutcomeSample]) -> OutcomeSample:
        """Put one-asset samples of the same years side by side.

        The columns are in the order of `samples`. The periods, probabilities,
        units, semantics and scenario must be identical and the asset ids
        distinct. The parts' trends must share their kind, reference year and
        fit years, so the total sums years on one basis; the slopes differ per
        asset, and the provenance's `trend` names each fit's asset. Pools must
        hold the same (year, model, member) rows under one pathway, as draw
        numbers alone would not line them up, and the provenance's `pool`
        records them. A part without a separate `hazard` contributes its index
        when another part has one.

        The provenance records `dependence` as `{"kind": "data"}`, the data's
        own, because the rows are the same years of the data, and holds each
        part's provenance in `parts`. Its `hazard` is the one the parts name, or
        none if any is silent; two names raise `ScenarioError` if the samples
        carry a `scenario` label. Its sources, licences, generator records
        and notes are the parts', each source, record and note naming its
        asset. Its lineage is new, built from the parts' lineages, and a
        leakage check reads each part's own in `parts` too, so a year of a part
        is the same row here and in that part. The loss and basis-risk models
        are per asset, so they are in `parts` only.

        Monte Carlo samples are refused: independently drawn curve samples
        line up by draw number only, so joining them would impose independence
        without stating it. Samples with index components are refused too:
        several assets with several hazards need a dependence model over both.

        Raises `SampleError` for fewer than two samples, a sample of several
        assets, a Monte Carlo sample, a sample with index components, a repeated
        asset id, any field that differs or a pool beside a sample that is not
        one, and `TypeError` for anything but an iterable of samples.
        """
        if isinstance(samples, (str, bytes, Mapping)) or not isinstance(
            samples, Iterable
        ):
            raise TypeError(
                "samples must be an iterable of OutcomeSample, got "
                f"{type(samples).__name__}"
            )
        parts = tuple(_require_sample(part) for part in samples)
        if len(parts) < 2:
            raise SampleError(f"join needs two samples or more, got {len(parts)}")
        for part in parts:
            if len(part.asset_ids) != 1:
                raise SampleError(
                    "join takes one-asset samples, but one holds "
                    f"{len(part.asset_ids)} assets; take each with asset() first"
                )
            if part.index_components is not None:
                raise SampleError(
                    f"join refuses the sample of asset {part.asset_ids[0]!r}, which "
                    "holds index components: several assets with several hazards "
                    "need a dependence model over both"
                )
            if part.provenance.source_kind == "monte_carlo":
                raise SampleError(
                    f"join refuses the Monte Carlo sample of asset "
                    f"{part.asset_ids[0]!r}: independently drawn curve samples line "
                    "up by draw number only, so joining them would impose "
                    "independence without stating it; use sample_curves, which "
                    "states the dependence"
                )
        ids = [part.asset_ids[0] for part in parts]
        if len(set(ids)) != len(ids):
            raise SampleError(f"join needs distinct asset ids, got {ids}")
        first = parts[0]
        for name in ("period", "probability"):
            for part in parts[1:]:
                if not np.array_equal(getattr(part, name), getattr(first, name)):
                    raise SampleError(
                        f"join needs identical {name} columns, but asset "
                        f"{part.asset_ids[0]!r} differs from asset {ids[0]!r}"
                    )
        for name in ("index_unit", "loss_unit", "semantics", "scenario"):
            values = {getattr(part, name) for part in parts}
            if len(values) > 1:
                raise SampleError(
                    f"join needs one {name} in every sample, got {sorted(values)}"
                )
        _check_same_trend(parts)
        trend = _joined_trend(parts)
        _check_same_pool(parts)
        pool = _joined_pool(parts)
        kinds = {part.provenance.source_kind for part in parts}
        stated = {part.provenance.semantics_source for part in parts}
        documents = {part.asset_ids[0]: part.provenance.to_document() for part in parts}
        licences: list[dict[str, Any]] = []
        for document in documents.values():
            for licence in document["licences"]:
                if licence not in licences:
                    licences.append(licence)
        # Parts of different kinds line up because the caller says they do.
        provenance = SampleProvenance(
            source_kind=first.provenance.source_kind if len(kinds) == 1 else "table",
            sources=tuple(
                {**source, "asset_id": asset}
                for asset, document in documents.items()
                for source in document["sources"]
            ),
            semantics_source=(
                first.provenance.semantics_source if len(stated) == 1 else "caller"
            ),
            rng=tuple(
                {**record, "asset_id": asset}
                for asset, document in documents.items()
                for record in document["rng"]
            ),
            licences=tuple(licences),
            details={
                "draws": len(first),
                "periods": [int(first.period[0]), int(first.period[-1])],
            },
            notes=tuple(
                _asset_note(asset, note)
                for asset, document in documents.items()
                for note in document["notes"]
            ),
            lineage=content_hash(
                {
                    "kind": "join",
                    "lineages": {part.asset_ids[0]: part.lineage for part in parts},
                }
            ),
            dependence={"kind": "data"},
            parts={
                part.asset_ids[0]: {
                    **documents[part.asset_ids[0]],
                    "lineage": part.lineage,
                }
                for part in parts
            },
            trend=trend,
            pool=pool,
            hazard=_agreed_hazard(
                [part.provenance.hazard for part in parts], first.scenario
            ),
        )
        hazard = (
            None
            if all(part.hazard is None for part in parts)
            else np.hstack(
                [part.index if part.hazard is None else part.hazard for part in parts]
            )
        )
        return cls(
            asset_ids=tuple(ids),
            period=first.period,
            probability=first.probability,
            index=np.hstack([part.index for part in parts]),
            loss=np.hstack([part.loss for part in parts]),
            hazard=hazard,
            index_unit=first.index_unit,
            loss_unit=first.loss_unit,
            semantics=first.semantics,
            provenance=provenance,
            scenario=first.scenario,
        )

    def to_table(self) -> Any:
        """Return the long table: one row per period and asset.

        Columns `scenario`, `period`, `asset_id`, `probability`, `index`,
        `loss` and `hazard` (null when the hazard is the index), sorted by
        period and asset id. The schema metadata holds, under
        `SAMPLE_METADATA_KEY` (`crc.parametric.sample`), a JSON document with
        the units, the semantics, the scenario, the asset ids, the content
        hash and the provenance, so `pyarrow.parquet.write_table` writes a
        self-describing file.

        A sample with index components adds, per component, the columns
        `index[<name>]`, `hazard[<name>]` and `own_loss[<name>]`, null where it
        has none, and the metadata adds `components`, each one's name, unit
        and tail, in order.
        """
        order = sorted(
            range(len(self.asset_ids)), key=lambda j: self.asset_ids[j].encode("utf-8")
        )
        sorted_ids = [self.asset_ids[j] for j in order]
        n_periods = self.period.shape[0]
        rows = n_periods * len(order)

        def flat(array: FloatArray) -> Any:
            return pa.array(
                np.ascontiguousarray(array[:, order]).reshape(-1), pa.float64()
            )

        components = self.index_components
        metadata: dict[str, Any] = {
            "asset_ids": list(self.asset_ids),
            "content_hash": self.content_hash,
            "index_unit": self.index_unit,
            "loss_unit": self.loss_unit,
            "provenance": self.provenance.to_document(),
            "scenario": self.scenario,
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "semantics": self.semantics,
            "semantics_version": SAMPLE_SEMANTICS_VERSION,
        }
        if components is not None:
            metadata["schema_version"] = SAMPLE_COMPONENTS_SCHEMA_VERSION
            metadata["components"] = [
                {"name": name, "tail": component.tail, "unit": component.unit}
                for name, component in components.items()
            ]
        hazard = (
            pa.nulls(rows, pa.float64()) if self.hazard is None else flat(self.hazard)
        )
        asset_column = (
            pa.repeat(pa.scalar(sorted_ids[0], pa.string()), rows)
            if len(sorted_ids) == 1
            else pa.array(sorted_ids * n_periods, pa.string())
        )
        fields = [
            ("scenario", pa.string()),
            ("period", pa.int64()),
            ("asset_id", pa.string()),
            ("probability", pa.float64()),
            ("index", pa.float64()),
            ("loss", pa.float64()),
            ("hazard", pa.float64()),
        ]
        arrays = [
            pa.repeat(pa.scalar(self.scenario, pa.string()), rows),
            pa.array(np.repeat(self.period, len(order)), pa.int64()),
            asset_column,
            pa.array(np.repeat(self.probability, len(order)), pa.float64()),
            flat(self.index),
            flat(self.loss),
            hazard,
        ]
        for name, component in (components or {}).items():
            for column, values in (
                ("index", component.values),
                ("hazard", component.hazard),
                ("own_loss", component.own_loss),
            ):
                fields.append((f"{column}[{name}]", pa.float64()))
                arrays.append(
                    pa.nulls(rows, pa.float64()) if values is None else flat(values)
                )
        schema = pa.schema(
            fields,
            metadata={
                SAMPLE_METADATA_KEY: json.dumps(
                    metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            },
        )
        return pa.Table.from_arrays(arrays, schema=schema)

    @classmethod
    def from_written_table(cls, table: Any) -> OutcomeSample:
        """Read back a sample written by `to_table`, with its provenance.

        `table` is the table `to_table()` returned, or the one read from its
        Parquet file with `pyarrow.parquet.read_table`: an Arrow table or
        record batch whose schema metadata holds `SAMPLE_METADATA_KEY`. The
        rows, units, semantics, scenario and provenance are restored, so the
        sample keeps its lineage for out-of-sample checks. Its content hash
        must equal the one the metadata records: a table changed after it
        was written (rows filtered, sorted or edited) is rejected; build a
        sample from such a table with `from_table`.

        Raises `SampleError` if the metadata is missing, unreadable or of
        another schema or semantics version, a column is missing or holds
        values the metadata contradicts (another scenario or asset, nulls),
        a field is invalid, or the content hash differs; `TypeError` for a
        file path (this module does no I/O) or a non-table.
        """
        data = _arrow_table(table, "a written sample")
        metadata = _written_metadata(data)
        what = "the written sample"
        asset_ids = metadata["asset_ids"]
        if (
            not isinstance(asset_ids, list)
            or not asset_ids
            or not all(isinstance(item, str) for item in asset_ids)
        ):
            raise SampleError(
                f"the metadata's asset_ids must be a non-empty list of strings, "
                f"got {asset_ids!r}"
            )
        n_assets = len(asset_ids)
        rows = data.num_rows
        if rows == 0 or rows % n_assets:
            raise SampleError(
                f"the written sample has {rows} rows, not a positive multiple of "
                f"its {n_assets} assets"
            )
        n = rows // n_assets
        # Rows are sorted by period, then asset id in UTF-8 byte order.
        order = sorted(range(n_assets), key=lambda j: asset_ids[j].encode("utf-8"))
        columns = np.argsort(order)
        written_ids = [asset_ids[j] for j in order] * n
        if _string_values(_column(data, "asset_id", what), "asset_id") != written_ids:
            raise SampleError(
                f"the asset_id column does not hold the metadata's assets "
                f"{asset_ids} in the written order"
            )
        scenario = _identifier("scenario", metadata["scenario"])
        scenarios = set(_string_values(_column(data, "scenario", what), "scenario"))
        if scenarios != {scenario}:
            raise SampleError(
                f"the scenario column holds {sorted(scenarios)}, but the metadata "
                f"states {scenario!r}"
            )

        def per_period(name: str, values: NDArray[Any]) -> NDArray[Any]:
            wide = values.reshape(n, n_assets)
            if not (wide == wide[:, :1]).all():
                raise SampleError(f"column {name!r} differs between assets of a period")
            return wide[:, 0]

        def per_asset(name: str) -> FloatArray:
            values = _float_values(_column(data, name, what), name)
            return values.reshape(n, n_assets)[:, columns]

        def optional(name: str, meaning: str) -> FloatArray | None:
            nulls = _column(data, name, what).null_count
            if nulls == rows:
                return None
            if nulls == 0:
                return per_asset(name)
            raise SampleError(
                f"column {name!r} is null in {nulls} of {rows} rows; it must be "
                f"null in every row ({meaning}) or in none"
            )

        hazard = optional("hazard", "the hazard is the index")
        index_components: dict[str, IndexComponent] | None = None
        if "components" in metadata:
            index_components = {
                name: IndexComponent(
                    per_asset(f"index[{name}]"),
                    unit=unit,
                    tail=tail,
                    hazard=optional(f"hazard[{name}]", "the hazard is the index"),
                    own_loss=optional(
                        f"own_loss[{name}]", "the component has no impact"
                    ),
                )
                for name, unit, tail in _written_components(metadata["components"])
            }
        sample = cls(
            asset_ids=tuple(asset_ids),
            period=per_period(
                "period", _integer_values(_column(data, "period", what), "period")
            ),
            probability=per_period(
                "probability",
                _float_values(_column(data, "probability", what), "probability"),
            ),
            index=per_asset("index"),
            loss=per_asset("loss"),
            hazard=hazard,
            index_unit=metadata["index_unit"],
            loss_unit=metadata["loss_unit"],
            semantics=metadata["semantics"],
            provenance=_written_provenance(metadata["provenance"]),
            scenario=scenario,
            index_components=index_components,
        )
        if sample.content_hash != metadata["content_hash"]:
            raise SampleError(
                f"the table's rows do not match the content hash its metadata "
                f"records ({metadata['content_hash']}): the table was changed after "
                "it was written"
            )
        return sample

    @classmethod
    def from_table(
        cls,
        table: Any,
        *,
        period: str,
        asset_id: str,
        index: str | Mapping[str, str],
        loss: str,
        probability: str | None = None,
        hazard: str | Mapping[str, str] | None = None,
        own_loss: Mapping[str, str] | None = None,
        semantics: Literal["annual"] = "annual",
        index_unit: str | Mapping[str, str] | None = None,
        loss_unit: str | None = None,
        tails: Mapping[str, Tail] | None = None,
        source: Any = None,
        scenario: str = "baseline",
        hazard_name: str | Mapping[str, str] | None = None,
    ) -> OutcomeSample:
        """Build a sample from a user table, one row per period and asset.

        `period`, `asset_id`, `index` and `loss` name the required columns;
        `probability` and `hazard` name optional ones. Without a probability
        column every row gets `1/n`. One row is one year: the caller states
        it by passing the table, and the provenance records it as the
        caller's (`semantics_source="caller"`). `semantics` may say so too;
        `"annual"`, the default, is its only value. `table` is an Arrow table or
        record batch, or any object Arrow reads (a mapping of columns, a
        pandas or polars frame); to use a Parquet file, read it with
        `pyarrow.parquet.read_table`. `source` names the table for the
        provenance: a string, a `SourceProvenance` (with its licence) or a
        mapping.

        `index_unit` and `loss_unit` name the units of the index and loss
        columns. Left out, each is the unit the table states for its column:
        the column's Arrow field metadata `unit`, or, for the columns of a
        table that `to_table` wrote, the units its `SAMPLE_METADATA_KEY`
        record holds. A table that states none needs them stated, and a stated
        unit must equal the one the table states.

        Rows are sorted by period, and `asset_ids` by asset id. Every
        (period, asset) pair must appear exactly once. A probability column
        must agree between the assets of a period. A table of several assets
        records `dependence` as `{"kind": "data"}`, the data's own: its rows
        are what the caller states. The table carries its own basis risk; if
        an asset's loss is a monotone function of its index, this warns with
        `ComonotoneIndexWarning`.

        `scenario` is the caller's name for the rows, as in `sample_annual`.
        `hazard_name` states the hazard the loss reads, and is optional where
        `source` names one in its `recipe` or `hazard_name`, or a `recipe` column
        holds one value: then the two must agree. Only a scenario set needs a
        name, so a sample without one stays valid.

        With a mapping from component name to column as `index`, two or more,
        the table builds a multi-hazard sample of one asset, whose first
        component is the primary. `index_unit` is then a mapping by name too,
        `tails` is required and maps each name to `"upper"` or `"lower"`, and
        `hazard` and `own_loss` map any of the names to a column, as
        `hazard_name` does to a name. `loss` is the combined loss as the caller
        states it: `combine` records
        `{"kind": "table"}`, and `hazard_dependence` `{"kind": "data"}`, as
        the rows are what the caller states. If a component's index determines
        the combined loss or its own loss monotonically, this warns with
        `ComonotoneIndexWarning`.

        With several index columns, `index_unit` maps any of the component
        names to a unit, and each component left out takes its column's unit
        from the table, as above.

        Raises `SampleError` for a missing column, nulls, non-integer
        periods, repeated or missing (period, asset) pairs, or invalid
        values, for a unit that is neither stated nor stated by the table, or
        that differs from the table's, for arguments that don't suit the form
        of `index`, and for several assets with several index columns;
        `ScenarioError` for a `hazard_name` the source contradicts; and
        `TypeError` for a file path or a non-table.
        """
        if semantics != "annual":
            raise SampleError(
                f"semantics must be 'annual' (one row is one year), got "
                f"{semantics!r}; 'event' is reserved"
            )
        scenario = _identifier("scenario", scenario)
        if isinstance(index, Mapping):
            return cls._from_components_table(
                table,
                period=period,
                asset_id=asset_id,
                index=index,
                loss=loss,
                probability=probability,
                hazard=hazard,
                own_loss=own_loss,
                semantics=semantics,
                index_unit=index_unit,
                loss_unit=loss_unit,
                tails=tails,
                source=source,
                scenario=scenario,
                hazard_name=hazard_name,
            )
        _refuse_component_arguments(
            tails=tails,
            own_loss=own_loss,
            hazard=hazard,
            index_unit=index_unit,
            hazard_name=hazard_name,
        )
        assert not isinstance(hazard, Mapping) and not isinstance(index_unit, Mapping)
        data = _arrow_table(table, "a sample table")
        what = "the sample table"
        periods = _integer_values(_column(data, period, what), period)
        ids = _string_values(_column(data, asset_id, what), asset_id)
        index_values = _float_values(_column(data, index, what), index)
        loss_values = _float_values(_column(data, loss, what), loss)
        probability_values = (
            None
            if probability is None
            else _float_values(_column(data, probability, what), probability)
        )
        hazard_values = (
            None
            if hazard is None
            else _float_values(_column(data, hazard, what), hazard)
        )
        written = _written_units(data)
        index_unit = _resolved_unit(
            "index_unit", index_unit, _column_unit(data, index, written)
        )
        loss_unit = _resolved_unit(
            "loss_unit", loss_unit, _column_unit(data, loss, written)
        )
        if not ids:
            raise SampleError("the sample table has no rows")
        distinct = sorted(set(ids))
        n_assets = len(distinct)
        position = {asset: j for j, asset in enumerate(distinct)}
        columns = np.array([position[value] for value in ids], dtype=np.int64)
        all_periods, rows = np.unique(periods, return_inverse=True)
        n = all_periods.shape[0]
        counts = np.zeros((n, n_assets), dtype=np.int64)
        np.add.at(counts, (rows, columns), 1)
        for j, asset in enumerate(distinct):
            repeated = all_periods[counts[:, j] > 1]
            if repeated.size:
                raise SampleError(
                    f"periods {repeated[:10].tolist()} appear more than once for "
                    f"asset {asset!r}"
                )
            missing = all_periods[counts[:, j] == 0]
            if missing.size:
                raise SampleError(
                    f"asset {asset!r} has no row for periods "
                    f"{missing[:10].tolist()}; every (period, asset) pair must "
                    "appear in the table"
                )

        def wide(values: FloatArray) -> FloatArray:
            array = np.empty((n, n_assets), dtype=np.float64)
            array[rows, columns] = values
            return array

        index_wide = wide(index_values)
        loss_wide = wide(loss_values)
        hazard_wide = None if hazard_values is None else wide(hazard_values)
        if probability_values is None:
            per_period = np.full(n, 1.0 / n)
        else:
            probabilities = wide(probability_values)
            if n_assets > 1 and not (probabilities == probabilities[:, :1]).all():
                raise SampleError(
                    f"column {probability!r} differs between the assets of a period"
                )
            per_period = probabilities[:, 0]
        source_document = _source_document(source)
        hazard_record = _hazard_record(
            hazard_name, [_table_hazard(source_document, data)], scenario
        )
        # Validate every field first; the provenance needs the content hash. It
        # has the default label, so the lineage below leaves `scenario` out.
        bare = cls(
            asset_ids=tuple(distinct),
            period=all_periods,
            probability=per_period,
            index=index_wide,
            loss=loss_wide,
            hazard=hazard_wide,
            index_unit=index_unit,
            loss_unit=loss_unit,
            semantics=semantics,
            provenance=SampleProvenance(
                source_kind="table",
                sources=(),
                semantics_source="caller",
                dependence=None if n_assets == 1 else {"kind": "data"},
            ),
        )
        table_hash = bare.content_hash
        comonotone = [
            asset
            for j, asset in enumerate(distinct)
            if _is_monotone_function(index_wide[:, j], loss_wide[:, j])
        ]
        if comonotone:
            where = "" if n_assets == 1 else f" (assets {', '.join(comonotone)})"
            warnings.warn(
                COMONOTONE_MESSAGE + where, ComonotoneIndexWarning, stacklevel=2
            )
        identity: dict[str, Any] = (
            {"kind": "table", "asset_id": distinct[0]}
            if n_assets == 1
            else {"kind": "table", "asset_ids": distinct}
        )
        if source_document:
            identity["source"] = source_document
        else:
            identity["data"] = table_hash
        provenance = SampleProvenance(
            source_kind="table",
            sources=(
                {
                    "role": "table",
                    "source": source_document,
                    "columns": {
                        "period": period,
                        "asset_id": asset_id,
                        "index": index,
                        "loss": loss,
                        "probability": probability,
                        "hazard": hazard,
                    },
                    "content_hash": table_hash,
                },
            ),
            semantics_source="caller",
            licences=(_licence_entry("table", source_document),),
            details={
                "draws": n,
                "periods": [int(periods[0]), int(periods[-1])],
                "probability": "uniform" if probability is None else "column",
            },
            notes=(
                (_TABLE_COMONOTONE_NOTE,)
                if comonotone and n_assets == 1
                else tuple(_asset_note(a, _TABLE_COMONOTONE_NOTE) for a in comonotone)
            ),
            lineage=content_hash(identity),
            hazard=hazard_record,
        )
        if n_assets > 1:
            parts = {
                asset: dataclasses.replace(
                    provenance,
                    notes=(_TABLE_COMONOTONE_NOTE,) if asset in comonotone else (),
                ).to_document()
                for asset in distinct
            }
            provenance = dataclasses.replace(
                provenance, dependence={"kind": "data"}, parts=parts
            )
        return dataclasses.replace(bare, provenance=provenance, scenario=scenario)

    @classmethod
    def _from_components_table(
        cls,
        table: Any,
        *,
        period: str,
        asset_id: str,
        index: Mapping[str, str],
        loss: str,
        probability: str | None,
        hazard: Any,
        own_loss: Any,
        semantics: Literal["annual"],
        index_unit: Any,
        loss_unit: str | None,
        tails: Any,
        source: Any,
        scenario: str,
        hazard_name: Any,
    ) -> OutcomeSample:
        """Build a multi-hazard sample from a user table; see `from_table`."""
        names = list(index)
        if len(names) < 2:
            raise SampleError(
                "a mapping as index names two components or more, got "
                f"{len(names)}; for one hazard give the column's name"
            )
        for name in names:
            _component_name(name, SampleError)
        stated_units = _by_component("index_unit", index_unit, names, every=False)
        severe = _by_component("tails", tails, names, every=True)
        hazards = _by_component("hazard", hazard, names, every=False)
        own_losses = _by_component("own_loss", own_loss, names, every=False)
        stated_names = _by_component("hazard_name", hazard_name, names, every=False)
        data = _arrow_table(table, "a sample table")
        what = "the sample table"
        periods = _integer_values(_column(data, period, what), period)
        ids = _string_values(_column(data, asset_id, what), asset_id)
        if not ids:
            raise SampleError("the sample table has no rows")
        distinct = sorted(set(ids))
        if len(distinct) > 1:
            raise SampleError(
                "a sample with index components holds exactly one asset, but the "
                f"table holds {len(distinct)}: {distinct[:5]}"
                + (" ..." if len(distinct) > 5 else "")
            )
        asset = distinct[0]
        all_periods, rows, counts = np.unique(
            periods, return_inverse=True, return_counts=True
        )
        repeated = all_periods[counts > 1]
        if repeated.size:
            raise SampleError(
                f"periods {repeated[:10].tolist()} appear more than once for "
                f"asset {asset!r}"
            )
        n = all_periods.shape[0]

        def by_period(name: str) -> FloatArray:
            array = np.empty(n, dtype=np.float64)
            array[rows] = _float_values(_column(data, name, what), name)
            return array

        loss_values = by_period(loss)
        written = _written_units(data)
        units = {
            name: _resolved_unit(
                f"index_unit[{name!r}]",
                stated_units.get(name),
                _column_unit(data, index[name], written),
            )
            for name in names
        }
        loss_unit = _resolved_unit(
            "loss_unit", loss_unit, _column_unit(data, loss, written)
        )
        components = {
            name: IndexComponent(
                by_period(index[name]),
                unit=units[name],
                tail=severe[name],
                hazard=by_period(hazards[name]) if name in hazards else None,
                own_loss=by_period(own_losses[name]) if name in own_losses else None,
            )
            for name in names
        }
        first = components[names[0]]
        per_period = (
            np.full(n, 1.0 / n) if probability is None else by_period(probability)
        )
        source_document = _source_document(source)
        # Validate every field first; the provenance needs the content hash.
        bare = cls(
            asset_ids=(asset,),
            period=all_periods,
            probability=per_period,
            index=first.values,
            loss=loss_values,
            hazard=first.hazard,
            index_unit=first.unit,
            loss_unit=loss_unit,
            semantics=semantics,
            provenance=SampleProvenance(
                source_kind="table",
                sources=(),
                semantics_source="caller",
                hazard_dependence={"kind": "data"},
                combine={"kind": "table"},
            ),
            index_components=components,
        )
        table_hash = bare.content_hash
        comonotone = [
            name
            for name, component in components.items()
            if _is_monotone_function(component.values[:, 0], loss_values)
            or (
                component.own_loss is not None
                and _is_monotone_function(
                    component.values[:, 0], component.own_loss[:, 0]
                )
            )
        ]
        if comonotone:
            warnings.warn(
                f"{COMONOTONE_MESSAGE} (components {', '.join(comonotone)})",
                ComonotoneIndexWarning,
                stacklevel=3,
            )
        identity: dict[str, Any] = {"kind": "table", "asset_id": asset}
        if source_document:
            identity["source"] = source_document
        else:
            identity["data"] = table_hash
        licences = (_licence_entry("table", source_document),)
        details = {
            "draws": n,
            "periods": [int(all_periods[0]), int(all_periods[-1])],
            "probability": "uniform" if probability is None else "column",
        }
        shared_columns = {
            "period": period,
            "asset_id": asset_id,
            "probability": probability,
        }
        provenance = SampleProvenance(
            source_kind="table",
            sources=(
                {
                    "role": "table",
                    "source": source_document,
                    "columns": {
                        **shared_columns,
                        "index": dict(index),
                        "loss": loss,
                        "hazard": hazards or None,
                        "own_loss": own_losses or None,
                    },
                    "content_hash": table_hash,
                },
            ),
            semantics_source="caller",
            licences=licences,
            details=details,
            notes=tuple(
                _component_note(name, _TABLE_COMONOTONE_NOTE) for name in comonotone
            ),
            lineage=content_hash(identity),
            hazard_dependence={"kind": "data"},
            combine={"kind": "table"},
            components={
                name: SampleProvenance(
                    source_kind="table",
                    sources=(
                        {
                            "role": "table",
                            "component": name,
                            "source": source_document,
                            "columns": {
                                **shared_columns,
                                "index": index[name],
                                "loss": own_losses.get(name),
                                "hazard": hazards.get(name),
                            },
                            "content_hash": table_hash,
                        },
                    ),
                    semantics_source="caller",
                    licences=licences,
                    details=details,
                    notes=(_TABLE_COMONOTONE_NOTE,) if name in comonotone else (),
                    hazard=_hazard_record(stated_names.get(name), [None], scenario),
                ).to_document()
                for name in names
            },
        )
        return dataclasses.replace(bare, provenance=provenance, scenario=scenario)


def _refuse_component_arguments(
    *, tails: Any, own_loss: Any, hazard: Any, index_unit: Any, hazard_name: Any
) -> None:
    """Raise `SampleError` for arguments that go with several index columns."""
    if tails is not None or own_loss is not None:
        raise SampleError(
            "tails and own_loss go with several index columns: a mapping from "
            "component name to column as index"
        )
    if isinstance(hazard, Mapping) or isinstance(index_unit, Mapping):
        raise SampleError(
            "hazard and index_unit are mappings only with several index columns: "
            "a mapping from component name to column as index"
        )
    if isinstance(hazard_name, Mapping):
        raise SampleError(
            "hazard_name is a mapping only with several index columns: a mapping "
            "from component name to column as index"
        )


_UNIT_KEY = b"unit"
"""The Arrow field metadata key that states a column's unit."""


def _written_units(table: Any) -> dict[str, str]:
    """The units that the record of a table `to_table` wrote holds, by column.

    `index_unit` for its `index` column, `loss_unit` for `loss`, and each
    component's unit for its `index[<name>]` column. A table without a
    readable record states no unit this way.
    """
    raw = (table.schema.metadata or {}).get(SAMPLE_METADATA_KEY.encode("utf-8"))
    if raw is None:
        return {}
    try:
        document = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(document, Mapping):
        return {}
    units = {
        column: document[key]
        for column, key in (("index", "index_unit"), ("loss", "loss_unit"))
        if isinstance(document.get(key), str)
    }
    components = document.get("components")
    for entry in components if isinstance(components, list) else ():
        if (
            isinstance(entry, Mapping)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("unit"), str)
        ):
            units[f"index[{entry['name']}]"] = entry["unit"]
    return units


@dataclass(frozen=True)
class _ColumnUnit:
    """The unit a table states for one column, and where it states it."""

    column: str
    unit: str | None
    where: str | None


def _field_unit(table: Any, column: str) -> str | None:
    """The unit that the Arrow field metadata of `column` states, if any.

    `column` must be a column of `table`. Raises `SampleError` for a `unit`
    that is not non-empty UTF-8 text without surrounding whitespace.
    """
    raw = (table.schema.field(column).metadata or {}).get(_UNIT_KEY)
    if raw is None:
        return None
    what = f"the field metadata 'unit' of column {column!r}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SampleError(f"{what} is not UTF-8 text") from error
    return _identifier(what, text)


def _column_unit(table: Any, column: str, written: Mapping[str, str]) -> _ColumnUnit:
    """The unit `table` states for `column`, and where; `unit` is `None` if none.

    The column's Arrow field metadata `unit` states it, and so does the
    record of a table `to_table` wrote (`written`), for its own columns; the
    two must agree.
    """
    _column(table, column, "the sample table")
    stated = _field_unit(table, column)
    recorded = written.get(column)
    if stated is not None and recorded is not None and stated != recorded:
        raise SampleError(
            f"column {column!r} has two units: {stated!r} in its field metadata "
            f"and {recorded!r} in the table's {SAMPLE_METADATA_KEY} record"
        )
    if stated is not None:
        return _ColumnUnit(column, stated, "its field metadata")
    if recorded is not None:
        return _ColumnUnit(column, recorded, f"the {SAMPLE_METADATA_KEY} record")
    return _ColumnUnit(column, None, None)


def _resolved_unit(argument: str, stated: Any, data: _ColumnUnit) -> str:
    """`stated`, or the table's unit for the column when it is `None`.

    Raises `SampleError` when neither gives a unit, or the two differ.
    """
    if stated is None:
        if data.unit is None:
            raise SampleError(
                f"{argument} is missing, and the table states no unit for column "
                f"{data.column!r}: state {argument}, or give the column an Arrow "
                "field metadata 'unit'"
            )
        return data.unit
    unit = _identifier(argument, stated)
    if data.unit is not None and unit != data.unit:
        raise SampleError(
            f"{argument}={unit!r} differs from the unit {data.unit!r} that the "
            f"table states for column {data.column!r} in {data.where}; leave "
            f"{argument} out to use the table's"
        )
    return unit


def _by_component(
    name: str, value: Any, names: Sequence[str], *, every: bool
) -> dict[str, Any]:
    """`value` as a mapping over component names; all of them if `every`."""
    if value is None and not every:
        return {}
    if not isinstance(value, Mapping):
        raise SampleError(
            f"with several index columns, {name} must be a mapping by component "
            f"name, got {value!r}"
        )
    unknown = sorted(str(key) for key in value if key not in names)
    missing = [component for component in names if component not in value]
    if unknown or (every and missing):
        raise SampleError(
            f"{name} is keyed by the components {list(names)}: "
            f"unknown {unknown}, missing {missing if every else []}"
        )
    return dict(value)


# -- written samples ---------------------------------------------------------

_WRITTEN_KEYS = frozenset(
    {
        "asset_ids",
        "content_hash",
        "index_unit",
        "loss_unit",
        "provenance",
        "scenario",
        "schema_version",
        "semantics",
        "semantics_version",
    }
)


def _written_metadata(table: Any) -> dict[str, Any]:
    """Return the `SAMPLE_METADATA_KEY` document of a table `to_table` wrote."""
    raw = (table.schema.metadata or {}).get(SAMPLE_METADATA_KEY.encode())
    if raw is None:
        raise SampleError(
            f"the table has no {SAMPLE_METADATA_KEY!r} schema metadata, so it was "
            "not written by OutcomeSample.to_table; build a sample from it with "
            "OutcomeSample.from_table"
        )
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise SampleError(
            f"the {SAMPLE_METADATA_KEY!r} metadata is not valid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise SampleError(f"the {SAMPLE_METADATA_KEY!r} metadata must be an object")
    version = document.get("schema_version", SAMPLE_SCHEMA_VERSION)
    if version not in (SAMPLE_SCHEMA_VERSION, SAMPLE_COMPONENTS_SCHEMA_VERSION):
        raise SampleError(
            f"the written sample has schema_version {version!r}; this version of "
            f"crc-sdk reads {SAMPLE_SCHEMA_VERSION!r} and "
            f"{SAMPLE_COMPONENTS_SCHEMA_VERSION!r}"
        )
    with_components = version == SAMPLE_COMPONENTS_SCHEMA_VERSION
    expected_keys = _WRITTEN_KEYS | ({"components"} if with_components else set())
    missing = sorted(expected_keys - set(document))
    if missing:
        raise SampleError(f"the {SAMPLE_METADATA_KEY!r} metadata lacks {missing}")
    unknown = sorted(set(document) - expected_keys)
    if unknown:
        raise SampleError(
            f"the {SAMPLE_METADATA_KEY!r} metadata has unknown keys {unknown}"
        )
    if document["semantics_version"] != SAMPLE_SEMANTICS_VERSION:
        raise SampleError(
            f"the written sample has semantics_version "
            f"{document['semantics_version']!r}; this version of crc-sdk reads "
            f"{SAMPLE_SEMANTICS_VERSION!r}"
        )
    return document


def _written_components(documents: Any) -> list[tuple[str, str, Any]]:
    """Each component's (name, unit, tail) of a written sample's metadata."""
    if not isinstance(documents, list) or not all(
        isinstance(entry, dict) and set(entry) == {"name", "tail", "unit"}
        for entry in documents
    ):
        raise SampleError(
            f"the {SAMPLE_METADATA_KEY!r} metadata's components must be a list of "
            "objects with a name, a unit and a tail"
        )
    return [(entry["name"], entry["unit"], entry["tail"]) for entry in documents]


def _written_provenance(document: Any) -> SampleProvenance:
    """Return the `SampleProvenance` of its `to_document()` output."""
    if not isinstance(document, Mapping):
        raise SampleError("the written provenance must be an object")
    names = {item.name for item in dataclasses.fields(SampleProvenance)}
    unknown = sorted(set(document) - names)
    if unknown:
        raise SampleError(f"the written provenance has unknown fields {unknown}")
    try:
        return SampleProvenance(**document)
    except TypeError as error:
        raise SampleError(f"the written provenance is incomplete: {error}") from error


# -- what a sample reads -------------------------------------------------------


def _is_labelled(sample: OutcomeSample) -> bool:
    """Whether a caller named the sample's scenario, which makes a read strict."""
    return sample.scenario != _DEFAULT_SCENARIO


def _named_in_source(document: Mapping[str, Any]) -> str | None:
    """The hazard a source document names: its `recipe`, else its `hazard_name`."""
    for key in ("recipe", "hazard_name"):
        value = document.get(key)
        if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
            return value
    return None


def _table_hazard(
    document: Mapping[str, Any], table: Any
) -> tuple[str, HazardFrom] | None:
    """The hazard a table's source document or one-valued `recipe` column names."""
    named = _named_in_source(document)
    if named is not None:
        return named, "source_plan"
    if "recipe" not in table.column_names:
        return None
    values = {
        value
        for value in table.column("recipe").unique().to_pylist()
        if value is not None
    }
    if len(values) != 1:
        return None
    value = values.pop()
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
        return value, "table_column"
    return None


def _hazard_record(
    stated: Any,
    found: Sequence[tuple[str, HazardFrom] | None],
    scenario: str,
) -> dict[str, str] | None:
    """What a sample records of the hazard its loss reads.

    `found` holds what each of its sources names, `None` for a silent one, and
    `stated` is `hazard_name=`. A stated name never overrides the record: it
    must agree with every source that names one, and only stands for a source
    that is silent. Sources that name two hazards record none, and raise on a
    sample with a `scenario` label.
    """
    if stated is not None:
        stated = _identifier("hazard_name", stated)
    named = [each for each in found if each is not None]
    names = list(dict.fromkeys(name for name, _ in named))
    if stated is not None:
        for name in names:
            if name != stated:
                raise ScenarioError(
                    f"scenario {scenario!r}: hazard_name={stated!r} contradicts the "
                    f"source, which names {name!r}; a stated name never overrides "
                    "the record"
                )
    if len(names) > 1:
        if scenario != _DEFAULT_SCENARIO:
            raise ScenarioError(
                f"scenario {scenario!r}: the sources of one sample must read one "
                f"hazard, but they name {names}"
            )
        return None
    if names and len(named) == len(found):
        origins = {origin for _, origin in named}
        return {
            "name": names[0],
            "from": "stated" if "stated" in origins else named[0][1],
        }
    return None if stated is None else {"name": stated, "from": "stated"}


def _agreed_hazard(
    records: Sequence[Mapping[str, Any] | None], scenario: str
) -> dict[str, str] | None:
    """One hazard record for the assets of a sample, from each asset's own."""
    return _hazard_record(
        None,
        [
            None if record is None else (record["name"], record["from"])
            for record in records
        ],
        scenario,
    )


@dataclass(frozen=True)
class _SourceRead:
    """What one hazard source of a sample states, and whose it is."""

    pathway: str | None
    horizon: int | None
    product: str | None
    owner: str


@dataclass(frozen=True)
class _SourceColumns:
    """The `pathway`, `horizon` and `product` of a sample's comparison rows."""

    pathway: str | None = None
    horizon: int | None = None
    product: str | None = None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _year(value: Any) -> int | None:
    """`value` as a year the `int32` column holds; `None` for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    year = int(value)
    return year if abs(year) <= _INT32_MAX else None


def _centre(years: Any) -> int | None:
    """`(first + last) // 2` of two years, as `TemporalWindow.horizon` has it."""
    if not isinstance(years, Sequence) or len(years) != 2:
        return None
    first, last = _year(years[0]), _year(years[1])
    return None if first is None or last is None else (first + last) // 2


def _source_read(
    source: Mapping[str, Any], details: Mapping[str, Any], owner: str
) -> _SourceRead:
    """What one source's record says a row reads; a field it lacks is null."""
    if source.get("kind") == "curve":
        row = source.get("row") or {}
        dataset = (source.get("metadata") or {}).get("source") or {}
        return _SourceRead(
            _text(row.get("pathway")),
            _year(row.get("horizon")),
            _text(dataset.get("provider")),
            owner,
        )
    if "series" in source:
        series = source["series"] or {}
        stated = series.get("source")
        document = stated if isinstance(stated, Mapping) else {}
        # A record setup_text would refuse states no pathway here.
        try:
            pathway = _text(source.get("pathway")) or _series_pathway(
                series, document, "the series"
            )
        except EvaluationError:
            pathway = None
        return _SourceRead(
            pathway,
            _centre(details.get("years")),
            _text(document.get("provider")),
            owner,
        )
    if source.get("role") == "table":
        document = source.get("source") or {}
        return _SourceRead(
            _text(document.get("pathway")),
            None,
            _text(document.get("provider")),
            owner,
        )
    return _SourceRead(None, None, None, owner)


def _pathway_statements(source: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every place a hazard source's record states a pathway, and what it states."""
    places: list[tuple[str, Any]]
    if source.get("kind") == "curve":
        places = [
            ("the canonical row", (source.get("row") or {}).get("pathway")),
            ("details=", (source.get("details") or {}).get("pathway")),
        ]
    elif "series" in source:
        series = source["series"] or {}
        document = series.get("source")
        stated = document if isinstance(document, Mapping) else {}
        try:
            column = _series_pathway(
                {"table_columns": series.get("table_columns")}, {}, "the series"
            )
        except EvaluationError:
            column = None
        places = [
            ("the model run", source.get("pathway")),
            ("the series' source", stated.get("pathway")),
            ("the series", series.get("pathway")),
            ("the table's scenario column", column),
        ]
    else:
        places = []
    return [(place, label) for place, label in places if _text(label) is not None]


def _hazard_reads(document: Mapping[str, Any], owner: str) -> list[_SourceRead]:
    """What every source the loss reads states, the index series left out."""
    for key, kind in (("parts", "asset"), ("components", "component")):
        nested = document.get(key)
        if nested:
            return [
                read
                for name, own in nested.items()
                for read in _hazard_reads(own, f"{kind} {name!r}")
            ]
    details = document.get("details") or {}
    sources = [
        source
        for source in document.get("sources") or ()
        if source.get("role") in ("hazard", "table")
    ]
    window = details.get("window")
    if window is not None and sources:
        members = window["members"]
        first = _source_read(sources[0], details, owner)
        return [
            _SourceRead(
                _text(members[0][0]),
                _centre([members[0][1], members[-1][1]]),
                first.product,
                owner,
            )
        ]
    return [_source_read(source, details, owner) for source in sources]


def _agreed_field(
    reads: Sequence[_SourceRead],
    field_name: str,
    *,
    scenario: str,
    strict: bool,
    same: Callable[[Any, Any], bool] = operator.eq,
) -> Any:
    """The one value of `field_name` that every read states; null if they differ.

    Null equals null. On a strict read a difference raises `ScenarioError`,
    naming the two sources.
    """
    first = reads[0]
    ours = getattr(first, field_name)
    for read in reads[1:]:
        theirs = getattr(read, field_name)
        if (ours is None and theirs is None) or (
            ours is not None and theirs is not None and same(ours, theirs)
        ):
            continue
        if strict:
            raise ScenarioError(
                f"scenario {scenario!r} reads one {field_name}, but its sources "
                f"differ: {first.owner} states {ours!r} and {read.owner} states "
                f"{theirs!r}"
            )
        return None
    return ours


def _source_columns(sample: OutcomeSample, *, strict: bool) -> _SourceColumns:
    """The `pathway`, `horizon` and `product` that a sample's source records state.

    They come from the records, never from the sample's label, and each is
    null where the records state none. `pathway` keeps the source's spelling,
    compared across sources with `pathways_equivalent`. A window of curves reads
    its members' one pathway and the centre of their horizons, a per-year series
    or a pool the centre of its whole window, so a split doesn't change it. The
    sources of a sample must agree. On a sample of several hazards `product` is
    instead every provider, sorted, joined by `+`.

    On a strict read an unknown pathway and sources that disagree raise
    `ScenarioError`. Otherwise the pathway stands as spelled and a field the
    sources disagree on is null.
    """
    document = sample.provenance.to_document()
    reads = _hazard_reads(document, "the sample")
    if not reads:
        return _SourceColumns()
    scenario = sample.scenario
    if strict:
        for read in reads:
            if read.pathway is not None:
                _check_known_pathway(read.pathway, scenario)
    product: str | None
    if sample.index_components is not None:
        products = sorted({read.product for read in reads if read.product is not None})
        product = "+".join(products) if products else None
    else:
        product = _agreed_field(reads, "product", scenario=scenario, strict=strict)
    return _SourceColumns(
        pathway=_agreed_field(
            reads,
            "pathway",
            scenario=scenario,
            strict=strict,
            same=pathways_equivalent,
        ),
        horizon=_agreed_field(reads, "horizon", scenario=scenario, strict=strict),
        product=product,
    )


def _check_known_pathway(label: str, scenario: str) -> None:
    """Raise `ScenarioError` unless `crc_sdk.pathways` knows the label."""
    try:
        resolve_pathway(label, strict=True)
    except ValueError as error:
        raise ScenarioError(
            f"scenario {scenario!r} reads the pathway {label!r}, which "
            "crc_sdk.pathways does not know; add a private one with "
            "crc_sdk.pathways.register_pathway"
        ) from error


# -- splits ------------------------------------------------------------------


def _require_sample(sample: Any) -> OutcomeSample:
    if not isinstance(sample, OutcomeSample):
        raise TypeError(f"expected an OutcomeSample, got {type(sample).__name__}")
    return sample


def _trend_of(sample: OutcomeSample) -> Mapping[str, Any]:
    """A sample's trend record; none recorded means the years as they are."""
    return sample.provenance.trend or {"kind": "as_recorded"}


def _check_same_trend(parts: Sequence[OutcomeSample]) -> None:
    """Raise `SampleError` unless the parts of a join treat the trend alike."""
    first = parts[0]
    stated = _trend_of(first)
    for part in parts[1:]:
        own = _trend_of(part)
        for key, name in (
            ("kind", "kind"),
            ("reference_year", "reference year"),
            ("fit_years", "fit years"),
        ):
            if own.get(key) != stated.get(key):
                raise SampleError(
                    f"join needs one basis for the trend in every sample, so that "
                    f"the total sums years on one basis, but the trend's {name} of "
                    f"asset {part.asset_ids[0]!r} is {own.get(key)!r} and of asset "
                    f"{first.asset_ids[0]!r} is {stated.get(key)!r}"
                )


def _joined_trend(parts: Sequence[OutcomeSample]) -> dict[str, Any] | None:
    """What the parts of a join state alike, with each part's fits named."""
    trends = [part.provenance.trend for part in parts]
    stated = [trend for trend in trends if trend is not None]
    if not stated:
        return None
    joined = {
        key: _thaw(value)
        for key, value in stated[0].items()
        if key != "fits" and all(other.get(key) == value for other in stated)
    }
    if "fits" in stated[0]:
        joined["fits"] = [
            {**_thaw(fit), "asset_id": part.asset_ids[0]}
            for part in parts
            for fit in _trend_of(part)["fits"]
        ]
    return joined


def _pool_row_ids(pool: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    """Each row of a pool as (year, model, member)."""
    runs = pool["runs"]
    return [
        (int(year), str(runs[run]["model"]), str(runs[run]["member"]))
        for year, run in zip(pool["year"], pool["run"], strict=True)
    ]


def _check_same_pool(parts: Sequence[OutcomeSample]) -> None:
    """Raise `SampleError` unless the parts of a join hold the same model-years.

    Their draw numbers alone would not line them up, the reason `join` refuses
    Monte Carlo samples.
    """
    first = parts[0]
    stated = first.provenance.pool
    for part in parts[1:]:
        own = part.provenance.pool
        if stated is None and own is None:
            continue
        if stated is None or own is None:
            raise SampleError(
                "join needs model-years pooled in every sample or in none, because "
                "draw numbers alone would not line them up, but of assets "
                f"{first.asset_ids[0]!r} and {part.asset_ids[0]!r} only one is pooled"
            )
        if own["pathway"] != stated["pathway"]:
            raise SampleError(
                f"join needs one pathway in every pool, but asset "
                f"{part.asset_ids[0]!r} is under {own['pathway']!r} and asset "
                f"{first.asset_ids[0]!r} under {stated['pathway']!r}"
            )
        if _pool_row_ids(own) != _pool_row_ids(stated):
            raise SampleError(
                f"join needs pools of the same (year, model, member) rows, because "
                f"draw numbers alone would not line them up, but those of asset "
                f"{part.asset_ids[0]!r} differ from those of asset "
                f"{first.asset_ids[0]!r}"
            )


def _joined_pool(parts: Sequence[OutcomeSample]) -> dict[str, Any] | None:
    """The rows the parts of a join hold alike, each run named across the assets.

    A run's lineage is built from each asset's own, as `join` builds a sample's
    lineage, so a plain sample of the same assets counts as one run. `parts`
    holds each asset's own runs and series.
    """
    first = parts[0].provenance.pool
    if first is None:
        return None
    named: list[tuple[str, dict[tuple[str, str], Mapping[str, Any]]]] = []
    for part in parts:
        pool = part.provenance.pool
        assert pool is not None
        named.append(
            (
                part.asset_ids[0],
                {(run["model"], run["member"]): run for run in pool["runs"]},
            )
        )
    held = sorted({int(position) for position in first["run"]})
    runs: list[dict[str, Any]] = []
    for position in held:
        run = first["runs"][position]
        key = (run["model"], run["member"])
        runs.append(
            {
                "model": run["model"],
                "member": run["member"],
                "pathway": run.get("pathway"),
                "lineage": content_hash(
                    {
                        "kind": "join",
                        "lineages": {
                            asset: own[key]["lineage"] for asset, own in named
                        },
                    }
                ),
                "dropped_years": sorted(
                    {
                        int(year)
                        for _, own in named
                        for year in own[key].get("dropped_years", ())
                    }
                ),
            }
        )
    return {
        "pathway": first["pathway"],
        "runs": runs,
        "year": [int(year) for year in first["year"]],
        "run": [held.index(int(position)) for position in first["run"]],
    }


def _part_pool(
    joined: Mapping[str, Any] | None, own: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """One asset's own pool, on the rows its joined sample holds now."""
    if joined is None or own is None:
        return None
    position = {(run["model"], run["member"]): i for i, run in enumerate(own["runs"])}
    runs = [position[(run["model"], run["member"])] for run in joined["runs"]]
    return {
        **_thaw(own),
        "year": [int(year) for year in joined["year"]],
        "run": [runs[int(run)] for run in joined["run"]],
    }


def _pool_entry(pool: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A pool's record in a comparison's metadata: its runs and years.

    Without the row arrays, which the sample's own metadata holds and the
    order of its rows gives, and without each run's series, which the
    comparison's `sources` holds.
    """
    if pool is None:
        return None
    return {
        "pathway": pool["pathway"],
        "rows": len(pool["year"]),
        "runs": [
            {key: value for key, value in run.items() if key != "series"}
            for run in pool["runs"]
        ],
        "years": sorted({int(year) for year in pool["year"]}),
    }


def _row_years(sample: OutcomeSample) -> IntArray:
    """Each row's year: a pool's calendar year, otherwise the row's period."""
    pool = sample.provenance.pool
    if pool is None:
        return sample.period
    years: IntArray = np.asarray(pool["year"], dtype=np.int64)
    return years


def _sample_years(sample: OutcomeSample) -> IntArray:
    """The distinct years of a sample's rows, in order."""
    years: IntArray = np.unique(_row_years(sample))
    return years


def _named_components(
    index: IndexDefinition | CompositeIndex, sample: OutcomeSample
) -> tuple[IndexComponent, ...]:
    """The components a cover on `index` reads, in order.

    A one-index cover reads the component named its `hazard_name`, and a
    `CompositeIndex` the components of its names. Each unit and tail must be
    the component's. Raises `SampleError` otherwise, and when the sample has
    no index components.
    """
    components = sample.index_components
    if isinstance(index, CompositeIndex):
        wanted = dict(index.components)
        if components is None:
            raise SampleError(
                f"the cover reads the components {list(index.names)}, but this "
                "sample holds one index; it needs a sample with index_components"
            )
        missing = [name for name in wanted if name not in components]
        if missing:
            raise SampleError(
                f"the cover reads the components {list(index.names)}, but the "
                f"sample holds no {missing}; it holds {list(components)}"
            )
    else:
        wanted = {index.hazard_name: index}
        if components is None or index.hazard_name not in components:
            raise SampleError(
                f"the cover pays on the hazard {index.hazard_name!r}, but the sample "
                f"holds no such component; it holds "
                f"{[] if components is None else list(components)}. "
                "Take one with sample.component(name, loss='combined') to read "
                "another component's index against the combined loss"
            )
    read: list[IndexComponent] = []
    for name, definition in wanted.items():
        assert components is not None
        component = components[name]
        if definition.unit != component.unit:
            raise SampleError(
                f"the cover's index {name!r} is in {definition.unit!r}, but the "
                f"sample's component is in {component.unit!r}"
            )
        if definition.tail != component.tail:
            raise SampleError(
                f"the cover's index {name!r} has the tail {definition.tail!r}, but "
                f"the sample's component has {component.tail!r}"
            )
        read.append(component)
    return tuple(read)


def _check_cover_index(
    index: IndexDefinition | CompositeIndex, sample: OutcomeSample
) -> None:
    """Raise `SampleError` unless a cover on `index` can read `sample`."""
    if isinstance(index, IndexDefinition) and sample.index_components is None:
        if index.unit != sample.index_unit:
            raise SampleError(
                f"the cover's index is in {index.unit!r}, but the sample's "
                f"index is in {sample.index_unit!r}"
            )
        return
    _named_components(index, sample)


def _cover_index_values(
    index: IndexDefinition | CompositeIndex, sample: OutcomeSample, column: int = 0
) -> FloatArray:
    """What a cover on `index` reads of `sample`, for the asset in `column`.

    `(n,)`, or `(n, k)` for a `CompositeIndex`, one column per component.
    """
    if isinstance(index, IndexDefinition) and sample.index_components is None:
        values: FloatArray = sample.index[:, column]
        return values
    read = _named_components(index, sample)
    if isinstance(index, CompositeIndex):
        return np.hstack([component.values for component in read])
    return read[0].values[:, 0]


def _require_one_asset(sample: OutcomeSample, function: str) -> None:
    """Raise `SampleError` if `sample` holds several assets; `function` needs one."""
    count = len(sample.asset_ids)
    if count > 1:
        shown = ", ".join(sample.asset_ids[:5]) + (", ..." if count > 5 else "")
        raise SampleError(
            f"{function} takes one asset's sample, but this one holds {count} assets "
            f"({shown}): take one with sample.asset(asset_id), or use design_joint "
            "for one design over several assets"
        )


def _hazard_reorder(provenance: SampleProvenance) -> tuple[Mapping[str, Any], ...]:
    """The reorder step of an `EmpiricalCopula` across hazards, if it has one.

    A component's own document records its reorder; a sample with no such
    documents has only this to record it, so that a split is still refused.
    """
    model = provenance.hazard_dependence
    if model is None or model["kind"] != "empirical_copula":
        return ()
    return ({"kind": "reorder", "dependence": model["kind"]},)


def _refuse_reference_year_blocks(sample: OutcomeSample, function: str) -> None:
    """Raise `SampleError` if `sample`'s rows come in reference-year blocks.

    A row block of such a sample is a set of reference years, so a split
    would not hold out independent draws. One asset's sample taken from it
    has no `dependence` of its own, but its `selection` records the reordering.
    """
    provenance = sample.provenance
    models = {step.get("dependence") for step in provenance.selection}
    for model in (provenance.dependence, provenance.hazard_dependence):
        if model is not None:
            models.add(model["kind"])
    if "empirical_copula" in models:
        raise SampleError(
            f"{function} cannot split an EmpiricalCopula sample: its rows come in "
            "reference-year blocks, so a block of rows is a set of reference years; "
            "test it out of sample on fresh draws (FreshDraws on a plan, or "
            "sample_curves or sample_curve_hazards with new generators and the same "
            "reference)"
        )


def _holds_hazard_distribution(document: Mapping[str, Any]) -> bool:
    """Whether a provenance document states the held form of `HazardNoise`.

    That is `HazardNoise(..., hold="hazard_distribution")`, in the document or
    in one of its parts or components.
    """
    model = document.get("basis_risk")
    if (
        isinstance(model, Mapping)
        and model.get("kind") == "hazard_noise"
        and model.get("hold") == "hazard_distribution"
    ):
        return True
    return any(
        _holds_hazard_distribution(own)
        for key in ("parts", "components")
        for own in (document.get(key) or {}).values()
    )


_ONE_RANKING = (
    "HazardNoise(..., hold='hazard_distribution') gave each row's hazard at the "
    "asset an index value by its rank over every row the sampler built"
)


def _refuse_one_ranking(sample: OutcomeSample, function: str) -> None:
    """Raise `SampleError` if `sample`'s hazard at the asset was ranked over its rows.

    A block of such a sample would hold index values of the other block's
    rows, so a split would not hold out what the design reads.
    """
    if _holds_hazard_distribution(sample.provenance.to_document()):
        raise SampleError(
            f"{function} cannot split this sample: {_ONE_RANKING}, so one block "
            "would hold values of the other's; build the design and the evaluation "
            "samples with separate calls (fresh draws from a curve, FreshDraws on a "
            "plan, or per-year samples of separate years), or use the default "
            "hold='hazard_mean', whose factors are each row's own"
        )


def holdout(
    sample: OutcomeSample, train_fraction: float = 0.7
) -> tuple[OutcomeSample, OutcomeSample]:
    """Split a sample into a design block and a held-out block.

    The design block is the first `floor(train_fraction * n + 1e-9)` periods
    in period order (the tolerance makes `0.7 * 10` give 7), the held-out
    block the rest. For per-year samples the held-out block is therefore the
    latest years. A pool of model-years is split by its `n` distinct years
    instead of its rows, as cutting rows could cut through a year, so every
    run holds out the same latest years. Each block's probabilities are
    rescaled to sum to 1. Both blocks share the sample's lineage, so
    evaluating a design on a block that overlaps its design block raises
    `LeakageError`.

    Raises `SampleError` unless `train_fraction` is a number in (0, 1) that
    leaves at least one period in each block, for an `EmpiricalCopula`
    sample, whose rows come in reference-year blocks: use fresh draws, and for
    a sample built with `HazardNoise(..., hold="hazard_distribution")`, whose
    hazard at the asset is ranked over every row: build each block separately.
    """
    sample = _require_sample(sample)
    _refuse_reference_year_blocks(sample, "holdout")
    _refuse_one_ranking(sample, "holdout")
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, Real)
        or not math.isfinite(float(train_fraction))
        or not 0.0 < float(train_fraction) < 1.0
    ):
        raise SampleError(
            f"train_fraction must be a number in (0, 1), got {train_fraction!r}"
        )
    fraction = float(train_fraction)
    row_years = _row_years(sample)
    blocks = np.unique(row_years)
    n = blocks.size
    n_train = math.floor(fraction * n + _HOLDOUT_TOLERANCE)
    if not 1 <= n_train <= n - 1:
        noun = "period" if sample.provenance.pool is None else "year"
        raise SampleError(
            f"train_fraction {fraction} of {n} {noun}s gives {n_train} design "
            f"{noun}s; both blocks need at least one {noun}"
        )
    design_rows = np.isin(row_years, blocks[:n_train])
    step = {"kind": "holdout", "train_fraction": fraction, "fold": 0}
    return (
        sample._subset(design_rows, {**step, "block": "train"}),
        sample._subset(~design_rows, {**step, "block": "test"}),
    )


def year_block_kfold(
    sample: OutcomeSample, k: int
) -> tuple[tuple[OutcomeSample, OutcomeSample], ...]:
    """k-fold cross-validation over contiguous blocks of periods.

    The periods, in order, are split into `k` contiguous blocks with
    `numpy.array_split` (the first `n % k` blocks get one period more). Fold
    `i` holds out block `i` and designs on the others. A pool of model-years
    is split by its `n` distinct years instead of its rows, so every run holds
    out the same years in each fold, and the last fold the latest. Returns one
    `(design, held_out)` pair per fold, in fold order; each sample's
    probabilities are rescaled to sum to 1.

    Raises `SampleError` unless `k` is an integer with `2 <= k <= n`, for
    an `EmpiricalCopula` sample, whose rows come in reference-year blocks: use
    fresh draws, and for a sample built with `HazardNoise(...,
    hold="hazard_distribution")`, whose hazard at the asset is ranked over
    every row: build each block separately.
    """
    sample = _require_sample(sample)
    _refuse_reference_year_blocks(sample, "year_block_kfold")
    _refuse_one_ranking(sample, "year_block_kfold")
    row_years = _row_years(sample)
    blocks = np.unique(row_years)
    n = blocks.size
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 2 <= k <= n:
        raise SampleError(f"k must be an integer in [2, {n}], got {k!r}")
    folds: list[tuple[OutcomeSample, OutcomeSample]] = []
    for fold, block in enumerate(np.array_split(blocks, int(k))):
        held_out = np.isin(row_years, block)
        step = {"kind": "kfold", "k": int(k), "fold": fold}
        folds.append(
            (
                sample._subset(~held_out, {**step, "block": "train"}),
                sample._subset(held_out, {**step, "block": "test"}),
            )
        )
    return tuple(folds)


@dataclass(frozen=True, eq=False)
class _SourceRows:
    """The rows of a sample that one source feeds, and what to call the source."""

    rows: NDArray[np.bool_]
    name: str


def _stated_table_lineage(document: Mapping[str, Any], asset: str) -> str | None:
    """`asset`'s lineage in a table with a stated source; `None` for any other.

    `from_table` identifies a table of several assets by all of them, so each
    asset needs this to link to a table of that asset alone from the same
    source. A table with no stated source is identified by its data, which
    links it to nothing else.
    """
    sources: Sequence[Mapping[str, Any]] = document.get("sources") or ()
    if document.get("source_kind") != "table" or len(sources) != 1:
        return None
    stated = sources[0].get("source")
    if sources[0].get("role") != "table" or not stated:
        return None
    return content_hash({"kind": "table", "asset_id": asset, "source": stated})


def _sources_in(
    document: Mapping[str, Any],
    *,
    lineage: str | None,
    asset: str | None,
    owner: str,
    pool: Mapping[str, Any] | None,
    every_row: NDArray[np.bool_],
) -> dict[str, _SourceRows]:
    """The sources in a provenance document, on the rows of its sample.

    `pool` is the document's pool on those rows. A part of a joined sample
    feeds every row of it, so the parts are read the same way, each asset's
    own pool brought to the sample's current rows.
    """
    sources: dict[str, _SourceRows] = {}
    if pool is not None:
        positions = np.asarray(pool["run"], dtype=np.int64)
        for position, run in enumerate(pool["runs"]):
            sources[str(run["lineage"])] = _SourceRows(
                positions == position,
                f"run {run['model']} {run['member']} of {owner}",
            )
    elif lineage is not None:
        sources[lineage] = _SourceRows(every_row, owner)
    for name, own in (document.get("source_lineages") or {}).items():
        sources.setdefault(own, _SourceRows(every_row, f"{name!r} of {owner}"))
    if asset is not None:
        stated = _stated_table_lineage(document, asset)
        if stated is not None:
            sources.setdefault(stated, _SourceRows(every_row, owner))
    for part_asset, part in (document.get("parts") or {}).items():
        found = _sources_in(
            part,
            lineage=part.get("lineage"),
            asset=part_asset,
            owner=f"asset {part_asset!r}",
            pool=_part_pool(pool, part.get("pool")),
            every_row=every_row,
        )
        for key, each in found.items():
            sources.setdefault(key, each)
    return sources


def _rows_by_source(sample: OutcomeSample) -> dict[str, _SourceRows]:
    """The sources of a sample's rows, by lineage, with the rows each one feeds.

    A pool's rows come from its runs, and any other sample counts as one run,
    so that a model-year is the same row in a pool and in a sample of its run.
    A sample of several hazards, one with an index series and a joined or drawn
    sample of several assets also hold the sources they are made of, each by
    the lineage it has alone, so that a year of one source is the same row in
    every sample that holds it.
    """
    count = len(sample.asset_ids)
    asset = sample.asset_ids[0] if count == 1 else None
    return _sources_in(
        sample.provenance.to_document(),
        lineage=sample.lineage,
        asset=asset,
        owner=f"{count} assets" if asset is None else f"asset {asset!r}",
        pool=sample.provenance.pool,
        every_row=np.ones(len(sample), dtype=np.bool_),
    )


def _share_a_source(first: OutcomeSample, second: OutcomeSample) -> bool:
    """Whether a source of one sample's rows feeds the other's, by its own lineage."""
    return not _rows_by_source(first).keys().isdisjoint(_rows_by_source(second))


def _curves_read(document: Mapping[str, Any]) -> set[str]:
    """The curve documents a Monte Carlo sample reads, as canonical text."""
    found = {
        json.dumps(source["curve"], sort_keys=True, default=_json_default)
        for source in document.get("sources") or ()
        if source.get("role") == "hazard" and source.get("curve") is not None
    }
    for key in ("parts", "components"):
        for own in (document.get(key) or {}).values():
            found |= _curves_read(own)
    return found


def _reads_the_curves_of(design: OutcomeSample, evaluation: OutcomeSample) -> bool:
    """Whether `evaluation` reads every curve of `design`, window for window.

    A curve the record can't describe (`"kind": "other"`) reads as equal to
    another of its type, as nothing tells them apart. Draws from a window are
    from its members together, so a window and anything but the same members are
    different curves, even when one holds a curve of the other.
    """
    ours = design.provenance.to_document()
    theirs = evaluation.provenance.to_document()
    if (ours.get("details") or {}).get("window") != (theirs.get("details") or {}).get(
        "window"
    ):
        return False
    return _curves_read(ours) <= _curves_read(theirs)


def _hazard_generators(document: Mapping[str, Any]) -> set[tuple[str, str]]:
    """The generator record of every `hazard` step of a provenance document."""
    found = {
        (str(record["bit_generator"]), str(record["state"]))
        for record in document.get("rng") or ()
        if record.get("step") == "hazard"
    }
    for key in ("parts", "components"):
        for own in (document.get(key) or {}).values():
            found |= _hazard_generators(own)
    return found


def _check_generators_not_reused(
    design: OutcomeSample, evaluation: OutcomeSample
) -> None:
    """Raise `LeakageError` for Monte Carlo samples that reuse one generator state."""
    if (
        design.provenance.source_kind != "monte_carlo"
        or evaluation.provenance.source_kind != "monte_carlo"
    ):
        return
    reused = _hazard_generators(design.provenance.to_document()) & _hazard_generators(
        evaluation.provenance.to_document()
    )
    if not reused:
        return
    periods = np.intersect1d(design.period, evaluation.period)
    if periods.size:
        raise LeakageError(
            f"the evaluation sample and the design sample share {periods.size} "
            f"draw{'' if periods.size == 1 else 's'} (e.g. {periods[:5].tolist()}): a "
            "hazard step of each has one generator record, so row d of each holds "
            "one uniform, whatever the curve; build each block with a new generator "
            "from one seed sequence, as reusing one generator state repeats its draws"
        )


def _check_out_of_sample(design: OutcomeSample, evaluation: OutcomeSample) -> None:
    """Raise `LeakageError` if `evaluation` shares rows with `design`.

    Two samples share a row when one source feeds that row of each in the same
    year: they have the same lineage (the same source, or splits of one
    sample) and a common period, or one holds a source of the other. A source
    is compared by its own lineage, whichever samples hold it: a run of a pool
    (a plain per-year sample counts as one run, so a model-year cannot be on
    both sides, between pools of different runs or between a pool and one of
    its runs), a hazard or an index series of a per-year sample, a component
    of a sample of several hazards, or an asset of a joined or drawn sample.
    The error names the shared sources and years. A linear trend fitted on a
    year the evaluation sample holds, by either sample, leaks that year too.

    Two Monte Carlo samples share a draw when a `hazard` step of one has the
    generator record of a `hazard` step of the other and they have a period in
    common, whatever the curve: row d of each then holds one uniform. Two
    different curves have no source in common, so this is what shows that a
    block reuses the design's draws; the blocks of one holdout have disjoint
    periods and pass.

    Two samples cut from one sample built with `HazardNoise(...,
    hold="hazard_distribution")` share rows too, whatever their periods: each
    row's hazard at the asset is an index value that the ranks over every row
    of that sample chose, so a block holds values of the other's rows.
    `holdout` and `year_block_kfold` refuse such a sample; this catches blocks
    taken with `select`.
    """
    design = _require_sample(design)
    evaluation = _require_sample(evaluation)
    designed = _rows_by_source(design)
    held = _rows_by_source(evaluation)
    common = [lineage for lineage in held if lineage in designed]
    if not common:
        _check_generators_not_reused(design, evaluation)
        return
    design_years = _row_years(design)
    held_years = _row_years(evaluation)
    shared = np.zeros(len(evaluation), dtype=np.bool_)
    names: list[str] = []
    for lineage in common:
        rows = held[lineage].rows & np.isin(
            held_years, design_years[designed[lineage].rows]
        )
        if rows.any():
            shared |= rows
            # The longer name is the more specific, a component's over its sample's.
            names.append(max(designed[lineage].name, held[lineage].name, key=len))
    count = int(shared.sum())
    if count:
        pooled = design.provenance.pool or evaluation.provenance.pool
        noun = "period" if pooled is None else "model-year"
        noun += "" if count == 1 else "s"
        years = np.unique(held_years[shared])
        distinct = list(dict.fromkeys(names))
        sources = "; ".join(distinct[:3]) + ("; ..." if len(distinct) > 3 else "")
        raise LeakageError(
            f"the evaluation sample shares {count} {noun} with the design "
            f"sample (e.g. {years[:5].tolist()}) from the same source ({sources}); "
            "evaluate out of sample on a held-out block or on fresh draws"
        )
    common_rows = np.logical_or.reduce([held[lineage].rows for lineage in common])
    for role, sample in (("design", design), ("evaluation", evaluation)):
        fitted = _trend_of(sample).get("fit_years")
        if fitted is None:
            continue
        overlap = np.intersect1d(
            np.array(fitted, dtype=np.int64), held_years[common_rows]
        )
        if overlap.size:
            raise LeakageError(
                f"the {role} sample's linear trend was fitted on {overlap.size} "
                f"years the evaluation sample holds (e.g. {overlap[:5].tolist()}) "
                "from the same source; state the design years with "
                "LinearTrend(fit_years=...), or let a plan fit the trend per fold"
            )
    _check_not_one_ranking(design, evaluation)


def _check_not_one_ranking(design: OutcomeSample, evaluation: OutcomeSample) -> None:
    """Raise `LeakageError` for two cuts of one sample ranked over its rows."""
    roots = {
        sample.provenance.parent or sample.content_hash
        for sample in (design, evaluation)
    }
    if len(roots) > 1 or not any(
        _holds_hazard_distribution(sample.provenance.to_document())
        for sample in (design, evaluation)
    ):
        return
    raise LeakageError(
        "the design and the evaluation samples are cut from one sample, and "
        f"{_ONE_RANKING}, so each holds values of the other's rows; build them "
        "with separate calls (fresh draws from a curve, or per-year samples of "
        "separate years)"
    )
