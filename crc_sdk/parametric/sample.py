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
  index (a basis-risk model or a separate index series); otherwise `None`.
- `provenance` (`SampleProvenance`) records the source, the semantics, the
  loss model, the basis-risk model and the generator state.

A sample holds one asset or more, with distinct ids, and a sample of several
states where the dependence between them comes from. `OutcomeSample.asset`
takes one asset's sample out of it, and `OutcomeSample.join` puts one-asset
samples of the same years side by side. `design`, `evaluate` and `compare`
work on one asset's sample.

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
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.errors import ComonotoneIndexWarning, LeakageError, SampleError

__all__ = [
    "SAMPLE_METADATA_KEY",
    "SAMPLE_SCHEMA_VERSION",
    "SAMPLE_SEMANTICS_VERSION",
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
"""The `schema_version` of a sample's hash document and table metadata."""

SAMPLE_SEMANTICS_VERSION = "crc-parametric-sample/1"
"""The `semantics_version` of a sample's hash document and table metadata."""

SourceKind = Literal["monte_carlo", "annual", "table"]
"""Where a sample's rows come from: curve draws, per-year values or a table."""

SemanticsSource = Literal["dataset", "caller", "data"]
"""Who stated what a row means: the dataset, the caller, or the data itself."""

_PROBABILITY_TOLERANCE = 1e-12
_HOLDOUT_TOLERANCE = 1e-9
_IDENTIFIER = re.compile(r"^\S(?:[\s\S]*\S)?$")

COMONOTONE_MESSAGE = (
    "the index and the loss are comonotone: without a basis-risk model the "
    "loss is a function of the index, so a cover can track the loss almost "
    "exactly. State a basis-risk model (ImpactNoise, or an index series read "
    "elsewhere than the loss) to measure basis risk"
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
) -> dict[str, Any]:
    """The document behind `content_hash`: the long table, sorted, and its units."""
    order = sorted(range(len(asset_ids)), key=lambda j: asset_ids[j].encode("utf-8"))
    n_assets = len(order)

    def flat(array: FloatArray) -> list[float]:
        values: list[float] = array[:, order].reshape(-1).tolist()
        return values

    return {
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
    - `loss_model` and `basis_risk`: their documents; `None` for a table.
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

    Mappings and sequences are stored read-only; `to_document()` returns
    plain JSON types. Raises `SampleError` for unknown kinds or values that
    are not plain JSON.
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
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise SampleError(f"{name} must be a mapping or None")
        for name in ("dependence", "trend"):
            value = getattr(self, name)
            if value is not None and "kind" not in value:
                raise SampleError(f"{name} must name its kind")
        if self.pool is not None:
            _check_pool(self.pool)
        if self.parts is not None and not all(
            isinstance(part, Mapping) for part in self.parts.values()
        ):
            raise SampleError("parts must map each asset id to a provenance document")
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
        ):
            value = getattr(self, name)
            _check_json(name, value)
            object.__setattr__(self, name, _freeze(_thaw(value)))
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_document(self) -> dict[str, Any]:
        """Return every field as plain JSON types."""
        return {
            item.name: _thaw(getattr(self, item.name))
            for item in dataclasses.fields(self)
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
    when it differs from the index, else `None`. `index_unit` and
    `loss_unit` name the units; the loss unit must equal a cover's payout
    unit for designs and evaluations. `semantics` is `"annual"`.

    A sample holds one asset or more, with distinct ids. A sample of several
    must state where the dependence between them comes from, in its
    provenance's `dependence`: it is never assumed. `asset` takes one asset's
    sample out of it and `join` puts one-asset samples side by side.
    The arrays are read-only copies. Samples compare by identity; compare
    `content_hash` for equal contents.

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

    @cached_property
    def content_hash(self) -> str:
        """`"sha256:<hex>"` of the sample's canonical long table and units.

        The document holds the long table (`to_table` without the scenario
        column), sorted by period and asset id, with the units, the
        semantics and the scenario, in the canonical JSON of cover documents.
        The provenance is not part of it.
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
        part's provenance in `parts`. Its sources, licences, generator records
        and notes are the parts', each source, record and note naming its
        asset. Its lineage is new, built from the parts' lineages. The loss
        and basis-risk models are per asset, so they are in `parts` only.

        Monte Carlo samples are refused: independently drawn curve samples
        line up by draw number only, so joining them would impose independence
        without stating it.

        Raises `SampleError` for fewer than two samples, a sample of several
        assets, a Monte Carlo sample, a repeated asset id, any field that
        differs or a pool beside a sample that is not one, and `TypeError` for
        anything but an iterable of samples.
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

        metadata = {
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
        hazard = (
            pa.nulls(rows, pa.float64()) if self.hazard is None else flat(self.hazard)
        )
        asset_column = (
            pa.repeat(pa.scalar(sorted_ids[0], pa.string()), rows)
            if len(sorted_ids) == 1
            else pa.array(sorted_ids * n_periods, pa.string())
        )
        schema = pa.schema(
            [
                ("scenario", pa.string()),
                ("period", pa.int64()),
                ("asset_id", pa.string()),
                ("probability", pa.float64()),
                ("index", pa.float64()),
                ("loss", pa.float64()),
                ("hazard", pa.float64()),
            ],
            metadata={
                SAMPLE_METADATA_KEY: json.dumps(
                    metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            },
        )
        return pa.Table.from_arrays(
            [
                pa.repeat(pa.scalar(self.scenario, pa.string()), rows),
                pa.array(np.repeat(self.period, len(order)), pa.int64()),
                asset_column,
                pa.array(np.repeat(self.probability, len(order)), pa.float64()),
                flat(self.index),
                flat(self.loss),
                hazard,
            ],
            schema=schema,
        )

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

        hazard_column = _column(data, "hazard", what)
        if hazard_column.null_count == rows:
            hazard: FloatArray | None = None
        elif hazard_column.null_count == 0:
            hazard = per_asset("hazard")
        else:
            raise SampleError(
                f"column 'hazard' is null in {hazard_column.null_count} of {rows} "
                "rows; it must be null in every row (the hazard is the index) or "
                "in none"
            )
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
        index: str,
        loss: str,
        probability: str | None = None,
        hazard: str | None = None,
        semantics: Literal["annual"],
        index_unit: str,
        loss_unit: str,
        source: Any = None,
    ) -> OutcomeSample:
        """Build a sample from a user table, one row per period and asset.

        `period`, `asset_id`, `index` and `loss` name the required columns;
        `probability` and `hazard` name optional ones. Without a probability
        column every row gets `1/n`. `semantics="annual"` is required: the
        caller states that one row is one year. `table` is an Arrow table or
        record batch, or any object Arrow reads (a mapping of columns, a
        pandas or polars frame); to use a Parquet file, read it with
        `pyarrow.parquet.read_table`. `source` names the table for the
        provenance: a string, a `SourceProvenance` (with its licence) or a
        mapping.

        Rows are sorted by period, and `asset_ids` by asset id. Every
        (period, asset) pair must appear exactly once. A probability column
        must agree between the assets of a period. A table of several assets
        records `dependence` as `{"kind": "data"}`, the data's own: its rows
        are what the caller states. The table carries its own basis risk; if
        an asset's loss is a monotone function of its index, this warns with
        `ComonotoneIndexWarning`.

        Raises `SampleError` for a missing column, nulls, non-integer
        periods, repeated or missing (period, asset) pairs, or invalid
        values, and `TypeError` for a file path or a non-table.
        """
        if semantics != "annual":
            raise SampleError(
                f"semantics must be 'annual' (one row is one year), got "
                f"{semantics!r}; 'event' is reserved"
            )
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
        # Validate every field first; the provenance needs the content hash.
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
        return dataclasses.replace(bare, provenance=provenance)


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
    missing = sorted(_WRITTEN_KEYS - set(document))
    if missing:
        raise SampleError(f"the {SAMPLE_METADATA_KEY!r} metadata lacks {missing}")
    unknown = sorted(set(document) - _WRITTEN_KEYS)
    if unknown:
        raise SampleError(
            f"the {SAMPLE_METADATA_KEY!r} metadata has unknown keys {unknown}"
        )
    for key, expected in (
        ("schema_version", SAMPLE_SCHEMA_VERSION),
        ("semantics_version", SAMPLE_SEMANTICS_VERSION),
    ):
        if document[key] != expected:
            raise SampleError(
                f"the written sample has {key} {document[key]!r}; this version of "
                f"crc-sdk reads {expected!r}"
            )
    return document


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


def _refuse_reference_year_blocks(sample: OutcomeSample, function: str) -> None:
    """Raise `SampleError` if `sample`'s rows come in reference-year blocks.

    A row block of such a sample is a set of reference years, so a split
    would not hold out independent draws. One asset's sample taken from it
    has no `dependence` of its own, but its `selection` records the reordering.
    """
    provenance = sample.provenance
    models = {step.get("dependence") for step in provenance.selection}
    if provenance.dependence is not None:
        models.add(provenance.dependence["kind"])
    if "empirical_copula" in models:
        raise SampleError(
            f"{function} cannot split an EmpiricalCopula sample: its rows come in "
            "reference-year blocks, so a block of rows is a set of reference years; "
            "test it out of sample on fresh draws (FreshDraws on a plan, or "
            "sample_curves with new generators and the same reference)"
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
    leaves at least one period in each block, and for an `EmpiricalCopula`
    sample, whose rows come in reference-year blocks: use fresh draws.
    """
    sample = _require_sample(sample)
    _refuse_reference_year_blocks(sample, "holdout")
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

    Raises `SampleError` unless `k` is an integer with `2 <= k <= n`, and for
    an `EmpiricalCopula` sample, whose rows come in reference-year blocks: use
    fresh draws.
    """
    sample = _require_sample(sample)
    _refuse_reference_year_blocks(sample, "year_block_kfold")
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


def _years_by_run(sample: OutcomeSample) -> dict[str, IntArray]:
    """The source of a sample's rows, by lineage, with the years each holds.

    A pool's rows come from its runs, and any other sample counts as one run,
    so that a model-year is the same row in a pool and in a sample of its run.
    """
    pool = sample.provenance.pool
    if pool is None:
        return {sample.lineage: sample.period}
    years = np.asarray(pool["year"], dtype=np.int64)
    positions = np.asarray(pool["run"], dtype=np.int64)
    return {
        str(run["lineage"]): years[positions == position]
        for position, run in enumerate(pool["runs"])
    }


def _check_out_of_sample(design: OutcomeSample, evaluation: OutcomeSample) -> None:
    """Raise `LeakageError` if `evaluation` shares rows with `design`.

    Two samples share rows when they have the same lineage (the same source,
    or splits of one sample) and a common period. On a pool the comparison is
    of (run lineage, year) pairs, and a plain per-year sample counts as one
    run: a model-year cannot be on both sides, between pools of different runs
    or between a pool and one of its runs. A linear trend fitted on a year the
    evaluation sample holds, by either sample, leaks that year too.
    """
    design = _require_sample(design)
    evaluation = _require_sample(evaluation)
    designed = _years_by_run(design)
    held = _years_by_run(evaluation)
    common = designed.keys() & held.keys()
    if not common:
        return
    pooled = design.provenance.pool or evaluation.provenance.pool
    shared = [np.intersect1d(designed[lineage], held[lineage]) for lineage in common]
    count = sum(each.size for each in shared)
    if count:
        noun = "period" if pooled is None else "model-year"
        noun += "" if count == 1 else "s"
        years = np.unique(np.concatenate(shared))
        raise LeakageError(
            f"the evaluation sample shares {count} {noun} with the design "
            f"sample (e.g. {years[:5].tolist()}) from the same source; evaluate "
            "out of sample on a held-out block or on fresh draws"
        )
    for role, sample in (("design", design), ("evaluation", evaluation)):
        fitted = _trend_of(sample).get("fit_years")
        if fitted is None:
            continue
        fit_years = np.array(fitted, dtype=np.int64)
        overlap = np.unique(
            np.concatenate(
                [np.intersect1d(fit_years, held[lineage]) for lineage in common]
            )
        )
        if overlap.size:
            raise LeakageError(
                f"the {role} sample's linear trend was fitted on {overlap.size} "
                f"years the evaluation sample holds (e.g. {overlap[:5].tolist()}) "
                "from the same source; state the design years with "
                "LinearTrend(fit_years=...), or let a plan fit the trend per fold"
            )
