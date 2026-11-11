"""Pooled k-fold metrics and block-bootstrap intervals over years.

- `compare_pooled(entries, folds, ...)` evaluates every held-out year of a
  k-fold split with the design made without it, and computes each metric once
  over all those years.
- `bootstrap_intervals(entries, on, *, rng, ...)` redraws the evaluated years
  with replacement and recomputes each metric with the designs held fixed. It
  returns a separate long table, so a comparison table and its schema don't
  change.

**How folds are pooled.** The held-out blocks of `year_block_kfold` partition
the years. A pooled row is the metrics of `metric_values` on all those rows
together, each row's payout from its own fold's design, with probability 1/N on
each of the N rows. It is not a mean over the folds: the mean of k CVaRs is the
CVaR of no distribution. Each year pays the contract in force for it, that
fold's design premium, so the net loss of row d is L_d - payout_d + P_d, and a
row's `premium` is the mean of the P_d. A cover given as is has one contract,
priced once on the pooled rows.

**Intervals.** A replicate draws the Y years of the block with replacement and
reweights each row by how often its year was drawn, which gives the metrics of
the concatenated years. The unit is a year with every row of that year: one row
of a series or a curve sample, and every run's row of one calendar year in a
pool; rows of several assets or hazards in a year stay together. An
`EmpiricalCopula` block is resampled by row, because its reference is part of
the stated model. One set of replicates serves every entry and every metric of
a call. Nothing is solved again. A design's premium is a fixed price, and a
cover priced on the block is priced again on each replicate.

An interval is the middle `ci_level` of the recomputed values (the percentile
interval, `numpy.quantile` with linear interpolation). It describes how much a
number moves with the years drawn. It leaves out uncertainty in the hazard data,
the impact function and the design, and on a curve block it is the Monte Carlo
error of the draws and says nothing about the curve. Two rows' intervals are not
a test of a difference between them.

Both functions rank nothing. They work in memory and do no I/O. Random numbers
come only from the `rng` a call states, and its state at entry is recorded, so
the table can be rebuilt.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import NDArray

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import EvaluationError, SampleError
from crc_sdk.parametric.evaluate import (
    _NO_PREMIUM_METRICS,
    Allocation,
    CoverPackage,
    CoverSet,
    DesignedCover,
    EvaluationEntry,
    Split,
    _allocated_payout,
    _block_fields,
    _check_keys,
    _component_values,
    _component_view,
    _CoverSet,
    _entries,
    _Entry,
    _entry,
    _Item,
    _item_reads,
    _joint_reads,
    _JointEntry,
    _metadata,
    _options,
    _package_record,
    _premium,
    _prepared,
    _row_record,
    _rows,
    _Run,
    _sample,
    _schema,
    _short_blocks,
    _split_step,
    _table,
    _total_payout,
    _union_reads,
    _values_hash,
    to_long,
)
from crc_sdk.parametric.joint import JointDesignResult
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    KEY_COLUMNS,
    CoverEntry,
    EvaluationMetadata,
    EvaluationSetup,
    _is_block_column,
    _reaches_worst_row,
    _replaced,
)
from crc_sdk.parametric.metrics import (
    MetricColumn,
    MetricOptions,
    metric_columns,
    metric_values,
)
from crc_sdk.parametric.pricing import Pricing, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _cover_index_values,
    _generator_record,
    _refuse_reference_year_blocks,
    _require_generator,
    _row_years,
)

__all__ = ["PooledEntry", "bootstrap_intervals", "compare_pooled"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

PooledEntry = (
    Cover
    | CoverSet
    | Sequence[DesignedCover]
    | Sequence[JointDesignResult]
    | Sequence[CoverSet]
    | Sequence[CoverPackage]
)
"""What `compare_pooled` takes per entry: see its docstring."""

Unit = Literal["year", "row"]

_INT32_MAX = 2**31 - 1

_EQUAL_PROBABILITY_TOLERANCE = 1e-9
"""How far a block's probabilities may be from equal, relative to the largest."""


# -- what each row of a comparison reads ---------------------------------------


@dataclass(frozen=True, eq=False)
class _Arrays:
    """One row of a comparison, draw by draw.

    `contracts` holds the payout of each contract the row sums, which a cover
    given as is prices one by one.
    """

    loss: FloatArray
    payout: FloatArray
    unidentified: BoolArray | None
    contracts: tuple[FloatArray, ...]


def _asset_arrays(
    item: _Item, asset: str, run: _Run, *, component: str | None = None
) -> _Arrays:
    sample = run.sample
    column = sample.asset_ids.index(asset)
    index = (
        _cover_index_values(item.cover.index_definition, sample, column)
        if component is None
        else _component_values(sample, component)
    )
    payout = item.cover.payout_values(index)
    return _Arrays(sample.loss[:, column], payout, _item_reads(item, index), (payout,))


def _set_arrays(entry: _CoverSet, run: _Run) -> list[_Arrays]:
    sample = run.sample
    parts = [
        _asset_arrays(entry.items[asset], asset, run) for asset in sample.asset_ids
    ]
    total = _Arrays(
        sample.loss.sum(axis=1),
        _total_payout(entry.covers, sample),
        _union_reads(part.unidentified for part in parts),
        tuple(part.payout for part in parts),
    )
    return [*parts, total]


def _joint_arrays(entry: _JointEntry, run: _Run) -> list[_Arrays]:
    sample = run.sample
    payout = _total_payout(entry.design.covers, sample)
    filled = _joint_reads(entry, sample)
    arrays = [_Arrays(sample.loss.sum(axis=1), payout, filled, (payout,))]
    if run.allocation is None:
        return arrays
    assert run.values is not None
    for asset in sample.asset_ids:
        share = _allocated_payout(run.allocation, payout, sample, run.values, asset)
        loss = sample.loss[:, sample.asset_ids.index(asset)]
        arrays.append(_Arrays(loss, share, filled, (share,)))
    return arrays


def _package_arrays(entry: CoverPackage, run: _Run) -> list[_Arrays]:
    sample = run.sample
    asset = sample.asset_ids[0]
    parts = [
        _asset_arrays(item, asset, run, component=name)
        for name, item in entry._items.items()
    ]
    package = _Arrays(
        sample.loss[:, 0],
        entry.as_cover().payout_values(_cover_index_values(entry.index, sample)),
        _union_reads(part.unidentified for part in parts),
        tuple(part.payout for part in parts),
    )
    return [*parts, package]


def _arrays(entry: _Entry, run: _Run) -> list[_Arrays]:
    """Each row of `entry` draw by draw, in the order of `_rows(entry, run)`."""
    if isinstance(entry, _Item):
        return [_asset_arrays(entry, run.sample.asset_ids[0], run)]
    if isinstance(entry, _CoverSet):
        return _set_arrays(entry, run)
    if isinstance(entry, CoverPackage):
        return _package_arrays(entry, run)
    return _joint_arrays(entry, run)


@dataclass(frozen=True, eq=False)
class _Read:
    """One row of a comparison, and how a replicate sets its premium.

    `premium` is a fixed price, over the block or per draw. `pricing`, when
    set, prices each contract on the rows a replicate reads. With neither,
    the row has no premium, as an allocated row has none.
    """

    arrays: _Arrays
    premium: float | FloatArray | None
    pricing: Pricing | None


@dataclass(frozen=True, eq=False)
class _Evaluated:
    """A comparison table and what resampling its rows needs."""

    table: Any
    reads: tuple[_Read, ...]
    probability: FloatArray
    year: IntArray
    n_years: int
    unit: Unit
    options: MetricOptions


def _no_premium_names(options: MetricOptions) -> frozenset[str]:
    return frozenset(
        column.name
        for column in metric_columns(options)
        if column.metric in _NO_PREMIUM_METRICS
    )


def _metrics(
    read: _Read,
    kept: BoolArray,
    probability: FloatArray,
    options: MetricOptions,
    without: frozenset[str],
) -> dict[str, float | None]:
    """The metrics of `read` on the `kept` rows, under `probability`."""
    arrays = read.arrays
    premium: float | FloatArray
    if read.pricing is not None:
        premium = (
            math.fsum(
                read.pricing.premium(payout[kept], probability)
                for payout in arrays.contracts
            )
            + 0.0
        )
    elif read.premium is None:
        premium = 0.0
    elif isinstance(read.premium, float):
        premium = read.premium
    else:
        premium = read.premium[kept]
    values = metric_values(
        arrays.loss[kept],
        arrays.payout[kept],
        probability,
        premium=premium,
        options=options,
        unidentified=(
            None if arrays.unidentified is None else arrays.unidentified[kept]
        ),
    )
    if read.premium is None and read.pricing is None:
        values.update(dict.fromkeys(without))
    return values


# -- pooled folds --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _PooledEntry:
    """One entry of `compare_pooled`: what applies on each fold."""

    folds: tuple[_Entry, ...]
    given: bool


def _folds_hash(hashes: Iterable[str]) -> str:
    """The `sha256:` hash of the canonical JSON of per-fold hashes, in fold order."""
    return content_hash(list(hashes))


def _checked_pairs(folds: Any) -> list[tuple[OutcomeSample, OutcomeSample]]:
    if isinstance(folds, (str, bytes, OutcomeSample)) or not isinstance(
        folds, Sequence
    ):
        raise TypeError(
            "folds must be what year_block_kfold returns: one (design block, "
            f"held-out block) pair per fold, got {type(folds).__name__}"
        )
    pairs: list[tuple[OutcomeSample, OutcomeSample]] = []
    for pair in folds:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(
                "each fold must be a (design block, held-out block) pair of "
                "OutcomeSample"
            )
        pairs.append(
            (
                _sample(pair[0], "a fold's design block"),
                _sample(pair[1], "a fold's held-out block"),
            )
        )
    if len(pairs) < 2:
        raise EvaluationError(
            f"pooling needs the folds of a k-fold split, got {len(pairs)} pair; a "
            "holdout and fresh draws have one block, so they have nothing to pool"
        )
    return pairs


def _check_steps(pairs: Sequence[tuple[OutcomeSample, OutcomeSample]]) -> None:
    """Raise unless the blocks are the k folds of one split, in fold order."""
    k = len(pairs)
    for position, pair in enumerate(pairs):
        for sample, block in zip(pair, ("train", "test"), strict=True):
            step = _split_step(sample)
            if (
                step is None
                or step.get("kind") != "kfold"
                or step.get("k") != k
                or step.get("fold") != position
                or step.get("block") != block
            ):
                found = "no split step" if step is None else dict(step)
                raise EvaluationError(
                    f"position {position} of folds must be fold {position} of a "
                    f"{k}-fold split, its {block} block, as year_block_kfold "
                    f"returns them in order; found {found}"
                )
    blocks = [sample for pair in pairs for sample in pair]
    if len({sample.lineage for sample in blocks}) > 1:
        raise EvaluationError(
            "the folds must be the folds of one split: their blocks have "
            "different lineages"
        )
    scenarios = sorted({sample.scenario for sample in blocks})
    if len(scenarios) > 1:
        raise EvaluationError(
            f"the blocks of one split share one scenario label, got {scenarios}"
        )


def _check_years(pairs: Sequence[tuple[OutcomeSample, OutcomeSample]]) -> None:
    """Raise unless the held-out blocks partition the years."""
    held = np.concatenate([np.unique(_row_years(test)) for _, test in pairs])
    every = np.unique(held)
    if every.size != held.size:
        values, counts = np.unique(held, return_counts=True)
        raise EvaluationError(
            "the held-out blocks repeat years, e.g. "
            f"{values[counts > 1][:5].tolist()}: every year is held out once"
        )
    for position, (train, test) in enumerate(pairs):
        read = np.union1d(_row_years(train), _row_years(test))
        missed = np.setxor1d(read, every)
        if missed.size:
            raise EvaluationError(
                f"the held-out blocks miss years of fold {position}, e.g. "
                f"{missed[:5].tolist()}: every year is held out once"
            )


def _check_equal_probabilities(
    pairs: Sequence[tuple[OutcomeSample, OutcomeSample]],
) -> None:
    for position, pair in enumerate(pairs):
        for sample in pair:
            p = sample.probability
            if np.ptp(p) > _EQUAL_PROBABILITY_TOLERANCE * float(p.max()):
                raise EvaluationError(
                    f"a block of fold {position} has unequal probabilities: "
                    "pooling gives every held-out row the same probability"
                )


def _has_designs(entry: _Entry) -> bool:
    if isinstance(entry, _JointEntry):
        return True
    if isinstance(entry, _Item):
        return entry.design is not None
    if isinstance(entry, CoverPackage):
        return entry._designed
    return any(item.design is not None for item in entry.items.values())


def _pooled_entry(entry: Any, k: int) -> _PooledEntry:
    if isinstance(entry, (Cover, Mapping)):
        checked = _entry(entry)
        if _has_designs(checked):
            raise EvaluationError(
                "a mapping on its own is a set of covers given as is, which applies "
                f"on every fold; give designs as a sequence of {k}, one per fold"
            )
        return _PooledEntry((checked,) * k, True)
    if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
        raise TypeError(
            f"each entry is a sequence of {k} designs, joint designs, mappings of "
            "designs or packages of designs in fold order, or one resolved Cover or "
            f"mapping of covers, got {type(entry).__name__}"
        )
    if len(entry) != k:
        raise EvaluationError(
            f"an entry needs one item per fold: {k} folds, got {len(entry)} items"
        )
    per_fold = tuple(_entry(item) for item in entry)
    for fold, item in enumerate(per_fold):
        if not _has_designs(item):
            raise EvaluationError(
                f"the item of fold {fold} holds no design: a cover given as is "
                "applies on every fold, so give it alone and not in a sequence"
            )
    return _PooledEntry(per_fold, False)


def _pooled_entries(entries: Any, k: int) -> list[_PooledEntry]:
    if isinstance(entries, (str, bytes, Mapping, Cover)) or not isinstance(
        entries, Sequence
    ):
        raise TypeError(
            "entries must be a sequence with one item per entry, got "
            f"{type(entries).__name__}"
        )
    if not entries:
        raise EvaluationError("compare_pooled needs at least one entry")
    return [_pooled_entry(entry, k) for entry in entries]


def _own_design_samples(
    entry: _Entry, train: OutcomeSample
) -> list[tuple[OutcomeSample, OutcomeSample]]:
    """Each design sample of `entry`, with the view of `train` it must equal."""
    if isinstance(entry, _JointEntry):
        return [(entry.design.design_sample, train)]
    if isinstance(entry, _CoverSet):
        return [
            (item.design.design_sample, train.asset(asset))
            for asset, item in entry.items.items()
            if item.design is not None
        ]
    items = list(entry._items.values()) if isinstance(entry, CoverPackage) else [entry]
    return [
        (item.design.design_sample, _component_view(train, item.design.design_sample))
        for item in items
        if item.design is not None
    ]


def _check_fold_designs(entry: _Entry, train: OutcomeSample, fold: int) -> None:
    for own, view in _own_design_samples(entry, train):
        if own.content_hash != view.content_hash:
            raise EvaluationError(
                f"a design of fold {fold} was made on {own.content_hash}, which is "
                f"not that fold's design block ({view.content_hash})"
            )


def _given_label(row: Mapping[str, Any]) -> str | None:
    """The label a row's cover carries; `None` where the row shows its hash."""
    label: str = row["cover_label"]
    return None if label == row["cover_hash"].removeprefix("sha256:")[:12] else label


def _joined(parts: Sequence[_Arrays]) -> _Arrays:
    """One row's draws over all the folds, in fold order."""
    filled = [part.unidentified for part in parts if part.unidentified is not None]
    return _Arrays(
        np.concatenate([part.loss for part in parts]),
        np.concatenate([part.payout for part in parts]),
        np.concatenate(filled) if len(filled) == len(parts) else None,
        tuple(
            np.concatenate(contract)
            for contract in zip(*(part.contracts for part in parts), strict=True)
        ),
    )


def _agreed_flag(values: Sequence[str | None]) -> str | None:
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else "mixed"


def _all_or_none_true(values: Sequence[bool | None]) -> bool | None:
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def _pooled_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    given: bool,
    sizes: Sequence[int],
    sample_hash: str,
) -> dict[str, Any]:
    """The columns before the metrics of one pooled row."""
    first = rows[0]
    digest = (
        str(first["cover_hash"])
        if given
        else _folds_hash(row["cover_hash"] for row in rows)
    )
    labels = {_given_label(row) for row in rows}
    shared = labels.pop() if len(labels) == 1 else None
    label = (
        str(first["cover_label"])
        if given
        else (digest.removeprefix("sha256:")[:12] if shared is None else shared)
    )
    design_ids = [row["design_id"] for row in rows]
    ties = [row["ties"] for row in rows]
    bounds = [row["design_budget"] for row in rows]
    return {
        "cover_label": label,
        "cover_hash": digest,
        "design_id": None if None in design_ids else _folds_hash(design_ids),
        "asset_id": first["asset_id"],
        "level": first["level"],
        "scope": first["scope"],
        "scenario": first["scenario"],
        "design_scenario": first["design_scenario"],
        "pathway": first["pathway"],
        "horizon": first["horizon"],
        "product": first["product"],
        "split": "out_of_sample",
        "fold": None,
        "sample_hash": sample_hash,
        "n_rows": sum(row["n_rows"] for row in rows),
        "premium_basis": first["premium_basis"],
        "unique": _all_or_none_true([row["unique"] for row in rows]),
        "ties": None if None in ties else sum(ties),
        "degenerate_tail": _agreed_flag([row["degenerate_tail"] for row in rows]),
        "components": first["components"],
        "design_budget": (
            None
            if None in bounds
            else math.fsum(
                bound * size for bound, size in zip(bounds, sizes, strict=True)
            )
            / sum(sizes)
            + 0.0
        ),
        **{name: value for name, value in first.items() if _is_block_column(name)},
    }


def _pooled_read(
    rows: Sequence[Mapping[str, Any]],
    arrays: _Arrays,
    *,
    given: bool,
    sizes: Sequence[int],
    pricing: Pricing,
) -> _Read:
    if given:
        return _Read(arrays, None, pricing)
    if rows[0]["premium"] is None:
        return _Read(arrays, None, None)
    per_draw = np.concatenate(
        [
            np.full(size, row["premium"], dtype=np.float64)
            for row, size in zip(rows, sizes, strict=True)
        ]
    )
    return _Read(arrays, per_draw, None)


def _shape(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (row["level"], row["asset_id"], row["scope"], row["components"]) for row in rows
    ]


def _pooled_sample_entry(
    merged: EvaluationMetadata,
    tests: Sequence[OutcomeSample],
    sample_hash: str,
) -> dict[str, Any]:
    """The entry of `setup.samples` for the pooled rows."""
    blocks = {str(entry["sample_hash"]): entry for entry in merged.setup.samples}
    first = blocks[tests[0].content_hash]
    entry: dict[str, Any] = {
        **first,
        "blocks": [test.content_hash for test in tests],
        "draws": sum(len(test) for test in tests),
        "periods": [
            min(int(test.period[0]) for test in tests),
            max(int(test.period[-1]) for test in tests),
        ],
        "roles": ["evaluation"],
        "sample_hash": sample_hash,
        "selection": [{"k": len(tests), "kind": "kfold_pooled"}],
    }
    if first["pool"] is not None:
        entry["pool"] = {
            **first["pool"],
            "rows": entry["draws"],
            "years": sorted(
                {year for test in tests for year in _row_years(test).tolist()}
            ),
        }
    return entry


@dataclass(frozen=True, eq=False)
class _PooledRows:
    """The pooled rows of a call, and what the metadata says of them."""

    rows: list[dict[str, Any]]
    reads: list[_Read]
    covers: dict[tuple[str, str | None], CoverEntry]
    designs: dict[str, dict[str, Any]]
    row_records: list[dict[str, Any]]
    package_records: list[dict[str, Any]]


def _record_pooled(
    fold_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    collected: _PooledRows,
) -> None:
    """Add the entries that say what a pooled row's hashes are made of."""
    labels = {_given_label(row) for row in fold_rows}
    shared = labels.pop() if len(labels) == 1 else None
    entry = CoverEntry(
        label=shared,
        hash=identity["cover_hash"],
        document={
            "kind": "pooled",
            "covers": [row["cover_hash"] for row in fold_rows],
        },
    )
    collected.covers.setdefault((entry.hash, shared), entry)
    if identity["design_id"] is not None:
        collected.designs.setdefault(
            identity["design_id"],
            {
                "cover_label": shared,
                "design_id": identity["design_id"],
                "designs": [row["design_id"] for row in fold_rows],
                "kind": "pooled",
            },
        )


def _record_folds(
    entry: _PooledEntry,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    identities: Sequence[Mapping[str, Any]],
    collected: _PooledRows,
) -> None:
    """Add what the setup text says of each fold's rows and packages."""
    for fold, group in enumerate(groups[:1] if entry.given else groups):
        suffix = "" if entry.given else f", fold {fold}"
        collected.row_records.extend(
            {
                **_row_record(row),
                "cover_hash": identity["cover_hash"],
                "cover_label": f"{identity['cover_label']}{suffix}",
            }
            for row, identity in zip(group, identities, strict=True)
        )
        item = entry.folds[fold]
        if isinstance(item, CoverPackage):
            collected.package_records.append(
                {
                    **_package_record(item, group),
                    "cover_hash": identities[-1]["cover_hash"],
                    "label": f"{identities[-1]['cover_label']}{suffix}",
                }
            )


def _pool_entry_rows(
    entry: _PooledEntry,
    runs: Sequence[_Run],
    *,
    sample_hash: str,
    probability: FloatArray,
    collected: _PooledRows,
) -> None:
    """Add the pooled rows of one entry."""
    groups = [_rows(item, run) for item, run in zip(entry.folds, runs, strict=True)]
    drawn = [_arrays(item, run) for item, run in zip(entry.folds, runs, strict=True)]
    if any(_shape(group) != _shape(groups[0]) for group in groups):
        raise EvaluationError(
            "an entry's items must give the same rows on every fold: the same "
            "levels, assets, scopes and components"
        )
    sizes = [len(run.sample) for run in runs]
    every_row = np.ones(probability.shape[0], dtype=np.bool_)
    options = runs[0].options
    without = _no_premium_names(options)
    identities: list[dict[str, Any]] = []
    for position in range(len(groups[0])):
        fold_rows = [group[position] for group in groups]
        identity = _pooled_identity(
            fold_rows, given=entry.given, sizes=sizes, sample_hash=sample_hash
        )
        read = _pooled_read(
            fold_rows,
            _joined([each[position] for each in drawn]),
            given=entry.given,
            sizes=sizes,
            pricing=runs[0].pricing,
        )
        metrics = _metrics(read, every_row, probability, options, without)
        # A pooled row reads every held-out row, so its flags are on all of them.
        flags = _short_blocks(options, identity["n_rows"])
        collected.rows.append({**identity, **flags, **metrics})
        collected.reads.append(read)
        identities.append(identity)
        if not entry.given:
            _record_pooled(fold_rows, identity, collected)
    _record_folds(entry, groups, identities, collected)


def _pooled_metadata(
    per_fold: Sequence[Sequence[_Entry]],
    pairs: Sequence[tuple[OutcomeSample, OutcomeSample]],
    runs: Sequence[_Run],
    *,
    sample_hash: str,
    given: bool,
    collected: _PooledRows,
    columns: Sequence[MetricColumn],
) -> EvaluationMetadata:
    """The metadata of a pooled table: every block, and the pooled rows' entries."""
    values_sha256 = None if runs[0].values is None else _values_hash(runs[0].values)
    parts: list[EvaluationMetadata] = []
    design_hashes: dict[str, None] = {}
    for fold, (train, test) in enumerate(pairs):
        own = {
            sample.content_hash: sample
            for item in per_fold[fold]
            for sample, _ in _own_design_samples(item, train)
        }
        design_hashes.update(dict.fromkeys(own))
        parts.append(
            _metadata(
                per_fold[fold],
                test,
                evaluation={
                    "fold": fold,
                    "sample_hash": test.content_hash,
                    "split": "out_of_sample",
                },
                design_samples=tuple(own.values()),
                options=runs[0].options,
                columns=columns,
                values_sha256=values_sha256,
            )
        )
    merged = EvaluationMetadata.merge(*parts)
    tests = [test for _, test in pairs]
    evaluation: dict[str, Any] = {
        "allocation": runs[0].allocation,
        "design_samples": list(design_hashes),
        "details": {
            "blocks": [test.content_hash for test in tests],
            "given_covers": given,
            "k": len(pairs),
        },
        "fold": None,
        "leakage_check": "passed" if design_hashes else "not_applicable",
        "level": list(dict.fromkeys(row["level"] for row in collected.rows)),
        "method": "kfold_pooled",
        "pricing": runs[0].pricing.to_document(),
        "sample_hash": sample_hash,
        "scope": list(dict.fromkeys(row["scope"] for row in collected.rows)),
        "split": "out_of_sample",
    }
    if tests[0].index_components is not None:
        evaluation["rows"] = collected.row_records
    if collected.package_records:
        evaluation["packages"] = collected.package_records
    metadata: EvaluationMetadata = _replaced(
        merged,
        setup=EvaluationSetup(
            basis=merged.setup.basis,
            metric_options=merged.setup.metric_options,
            samples=(
                *merged.setup.samples,
                _pooled_sample_entry(merged, tests, sample_hash),
            ),
            evaluations=(evaluation,),
        ),
        covers=(*merged.covers, *collected.covers.values()),
        designs=(*merged.designs, *collected.designs.values()),
    )
    return metadata


def _pooled(
    entries: Any,
    folds: Any,
    *,
    pricing: Any,
    options: Any,
    allocation: Any,
    values: Any,
) -> _Evaluated:
    pairs = _checked_pairs(folds)
    _check_steps(pairs)
    _check_years(pairs)
    _check_equal_probabilities(pairs)
    k = len(pairs)
    pooled = _pooled_entries(entries, k)
    for fold, (train, _) in enumerate(pairs):
        for entry in pooled:
            _check_fold_designs(entry.folds[fold], train, fold)
    per_fold = [[entry.folds[fold] for entry in pooled] for fold in range(k)]
    # The design block stands in as the design sample of a cover given as is
    # for the checks of one fold; the metadata lists only the designs' own.
    runs = [
        _prepared(
            per_fold[fold],
            test,
            split="out_of_sample",
            fold=fold,
            pricing=pricing,
            premium=None,
            options=options,
            design_sample=train,
            allocation=allocation,
            values=values,
        )[0]
        for fold, (train, test) in enumerate(pairs)
    ]
    tests = [test for _, test in pairs]
    n_rows = sum(len(test) for test in tests)
    probability = np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    sample_hash = _folds_hash(test.content_hash for test in tests)
    collected = _PooledRows([], [], {}, {}, [], [])
    for entry in pooled:
        _pool_entry_rows(
            entry,
            runs,
            sample_hash=sample_hash,
            probability=probability,
            collected=collected,
        )
    _check_keys(collected.rows, KEY_COLUMNS)
    columns = metric_columns(runs[0].options)
    metadata = _pooled_metadata(
        per_fold,
        pairs,
        runs,
        sample_hash=sample_hash,
        given=any(entry.given for entry in pooled),
        collected=collected,
        columns=columns,
    )
    schema = _schema(columns, metadata, extra=_block_fields(runs[0].options.betas))
    schema = schema.set(schema.get_field_index("fold"), pa.field("fold", pa.int32()))
    year = np.concatenate([_row_years(test) for test in tests])
    distinct, positions = np.unique(year, return_inverse=True)
    return _Evaluated(
        table=pa.Table.from_pylist(collected.rows, schema=schema),
        reads=tuple(collected.reads),
        probability=probability,
        year=positions.astype(np.int64),
        n_years=int(distinct.size),
        unit="year",
        options=runs[0].options,
    )


# -- one block -----------------------------------------------------------------


def _resampling_unit(sample: OutcomeSample) -> Unit:
    """`"row"` on an `EmpiricalCopula` block, `"year"` on any other."""
    try:
        _refuse_reference_year_blocks(sample, "bootstrap_intervals")
    except SampleError:
        return "row"
    return "year"


def _is_given(row: Mapping[str, Any]) -> bool:
    """Whether a row is that of one cover given as is."""
    return bool(row["level"] == "asset" and row["design_id"] is None)


def _stated_premium(value: Any, label: str) -> float:
    amount = _premium(value)
    if amount is None:
        raise EvaluationError(f"premiums= states no number for {label!r}")
    return amount


def _apply_premiums(
    premiums: Any, groups: Sequence[list[dict[str, Any]]]
) -> list[list[dict[str, Any]]]:
    """The rows with the premiums `premiums` states for covers given as is."""
    if premiums is None:
        return list(groups)
    if not isinstance(premiums, Mapping):
        raise TypeError(
            "premiums must map a cover_label to a stated premium, got "
            f"{type(premiums).__name__}"
        )
    used: set[str] = set()
    updated: list[list[dict[str, Any]]] = []
    for group in groups:
        given = [row["cover_label"] for row in group if _is_given(row)]
        named = [label for label in given if label in premiums]
        if not named:
            updated.append(group)
            continue
        if len(named) != len(given):
            raise EvaluationError(
                "premiums= states every cover of a set or a package, or none of them"
            )
        used.update(named)
        total = math.fsum(_stated_premium(premiums[label], label) for label in named)
        rows: list[dict[str, Any]] = []
        for row in group:
            if _is_given(row):
                label = row["cover_label"]
                amount = _stated_premium(premiums[label], label)
                rows.append({**row, "premium": amount, "premium_basis": "stated"})
            elif row["level"] in ("total", "package"):
                rows.append({**row, "premium": total + 0.0, "premium_basis": "stated"})
            else:
                rows.append(row)
        updated.append(rows)
    unknown = sorted(str(key) for key in premiums if key not in used)
    if unknown:
        raise EvaluationError(
            f"premiums= names {unknown}, which match no cover given as is"
        )
    return updated


def _block_read(row: Mapping[str, Any], arrays: _Arrays, pricing: Pricing) -> _Read:
    if row["premium_basis"] == "evaluation":
        return _Read(arrays, None, pricing)
    premium = row["premium"]
    return _Read(arrays, None if premium is None else float(premium), None)


def _block(
    entries: Any,
    sample: OutcomeSample,
    *,
    split: Any,
    fold: Any,
    pricing: Any,
    premiums: Any,
    options: Any,
    design_sample: Any,
    allocation: Any,
    values: Any,
) -> _Evaluated:
    if split is None:
        raise EvaluationError(
            "split= is required on one block, as in compare: 'in_sample' or "
            "'out_of_sample'"
        )
    checked: list[_Entry] = _entries(entries)
    arguments: dict[str, Any] = {
        "split": split,
        "fold": fold,
        "pricing": pricing,
        "premium": None,
        "options": options,
        "design_sample": design_sample,
        "allocation": allocation,
        "values": values,
    }
    table = _table(checked, sample, **arguments)
    run, _, _ = _prepared(checked, sample, **arguments)
    groups = _apply_premiums(premiums, [_rows(entry, run) for entry in checked])
    rows = [row for group in groups for row in group]
    drawn = [each for entry in checked for each in _arrays(entry, run)]
    reads = tuple(
        _block_read(row, arrays, run.pricing)
        for row, arrays in zip(rows, drawn, strict=True)
    )
    probability = sample.probability / math.fsum(sample.probability.tolist())
    if premiums is not None:
        every_row = np.ones(len(sample), dtype=np.bool_)
        without = _no_premium_names(run.options)
        for row, read in zip(rows, reads, strict=True):
            row.update(_metrics(read, every_row, probability, run.options, without))
        table = pa.Table.from_pylist(rows, schema=table.schema)
    distinct, positions = np.unique(_row_years(sample), return_inverse=True)
    return _Evaluated(
        table=table,
        reads=reads,
        probability=probability,
        year=positions.astype(np.int64),
        n_years=int(distinct.size),
        unit=_resampling_unit(sample),
        options=run.options,
    )


# -- intervals -----------------------------------------------------------------


def _year_counts(draws: IntArray, n_years: int) -> IntArray:
    """How often each year is drawn: `(B, Y)` from the year indices `(B, Y)`."""
    n_replicates = draws.shape[0]
    flat = (
        draws.astype(np.int64)
        + np.arange(n_replicates, dtype=np.int64)[:, None] * n_years
    ).ravel()
    counts: IntArray = np.bincount(flat, minlength=n_replicates * n_years).reshape(
        n_replicates, n_years
    )
    return counts


def _replicates(evaluated: _Evaluated, counts: IntArray) -> FloatArray:
    """The metrics under each row of year counts, `(B, rows, columns)`.

    A replicate gives each row its probability times how often its year was
    drawn, rescaled to sum to 1, and leaves out the rows of years not drawn.
    A metric undefined in a replicate is NaN.
    """
    options = evaluated.options
    names = [column.name for column in metric_columns(options)]
    without = _no_premium_names(options)
    result = np.full(
        (counts.shape[0], len(evaluated.reads), len(names)), np.nan, dtype=np.float64
    )
    for b, count in enumerate(counts):
        weight = evaluated.probability * count[evaluated.year]
        kept = weight > 0.0
        probability = weight[kept] / math.fsum(weight[kept].tolist())
        for r, read in enumerate(evaluated.reads):
            values = _metrics(read, kept, probability, options, without)
            result[b, r] = np.array([values[name] for name in names], dtype=np.float64)
    return result


def _short_block(column: MetricColumn, n_rows: int) -> bool | None:
    """Whether the column's level reaches the block's worst row; `None` if no level."""
    return _reaches_worst_row(column.parameter, n_rows)


def _n_bootstrap(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= _INT32_MAX
    ):
        raise EvaluationError(f"n_bootstrap must be a positive integer, got {value!r}")
    return int(value)


def _ci_level(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.floating))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise EvaluationError(f"ci_level must lie in (0, 1), got {value!r}")
    return float(value)


def _interval_table(
    evaluated: _Evaluated,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_level: float,
) -> Any:
    wide = evaluated.table
    columns = metric_columns(evaluated.options)
    n_rows = int(evaluated.probability.shape[0])
    record = _generator_record(rng, "bootstrap")
    draws = rng.integers(
        0, evaluated.n_years, size=(n_bootstrap, evaluated.n_years), dtype=np.int64
    )
    replicates = _replicates(evaluated, _year_counts(draws, evaluated.n_years))
    low, high = np.quantile(
        replicates, [(1.0 - ci_level) / 2.0, (1.0 + ci_level) / 2.0], axis=0
    )
    undefined = np.isnan(replicates).sum(axis=0)
    point_null = np.column_stack(
        [np.asarray(wide.column(column.name).is_null()) for column in columns]
    )
    # to_long orders its rows by metric column, then by the wide rows.
    null = (np.isnan(low) | point_null).ravel(order="F")
    flags = [_short_block(column, n_rows) for column in columns]

    long = to_long(wide)
    n_long = long.num_rows
    additions = (
        (
            pa.field("ci_low", pa.float64()),
            pa.array(low.ravel(order="F"), mask=null, type=pa.float64()),
        ),
        (
            pa.field("ci_high", pa.float64()),
            pa.array(high.ravel(order="F"), mask=null, type=pa.float64()),
        ),
        (
            pa.field("ci_level", pa.float64(), nullable=False),
            pa.array([ci_level] * n_long, pa.float64()),
        ),
        (
            pa.field("n_bootstrap", pa.int32(), nullable=False),
            pa.array([n_bootstrap] * n_long, pa.int32()),
        ),
        (
            pa.field("n_undefined", pa.int32(), nullable=False),
            pa.array(undefined.ravel(order="F"), pa.int32()),
        ),
        (
            pa.field("short_block", pa.bool_()),
            pa.array(
                [flag for flag in flags for _ in range(wide.num_rows)], pa.bool_()
            ),
        ),
    )
    for field, values in additions:
        long = long.append_column(field, values)
    sample_hashes = wide.column("sample_hash").unique().to_pylist()
    metadata = EvaluationMetadata.from_table(wide)
    metadata = _replaced(
        metadata,
        setup=_replaced(
            metadata.setup,
            intervals={
                "ci_level": ci_level,
                "method": "percentile",
                "n_bootstrap": n_bootstrap,
                "short_block_columns": [
                    column.name
                    for column, flag in zip(columns, flags, strict=True)
                    if flag
                ],
                "unit": evaluated.unit,
            },
        ),
        rng=(*metadata.rng, {**record, "sample_hash": sample_hashes[0]}),
    )
    existing = dict(long.schema.metadata or {})
    existing[EVALUATION_METADATA_KEY.encode("utf-8")] = metadata.to_json().encode(
        "utf-8"
    )
    return long.replace_schema_metadata(existing)


# -- the public functions ------------------------------------------------------


def compare_pooled(
    entries: Sequence[PooledEntry],
    folds: Sequence[tuple[OutcomeSample, OutcomeSample]],
    *,
    pricing: Pricing = PurePremium(),
    options: MetricOptions = MetricOptions(),
    allocation: Allocation | None = None,
    values: Mapping[str, float] | None = None,
) -> Any:
    """Compute each metric once over all the held-out years of a k-fold split.

    `folds` is what `year_block_kfold` returns, in fold order. Each item of
    `entries` is one of:

    - a sequence of k of what `compare` takes that hold designs, in fold order:
      designs, `JointDesignResult`s, mappings of designs or packages of
      designs, each made on its fold's design block;
    - one resolved `Cover` or mapping of covers, which applies on every fold.

    The table has the columns of `compare`, one row per entry at the levels
    `compare` gives it, and every row reads all the held-out years, each with
    probability 1/N and its payout from its own fold's design. On a pooled row:

    - `split` is `"out_of_sample"` and `fold` is null;
    - `cover_hash`, `design_id` and `sample_hash` are the `sha256:` hash of the
      canonical JSON of the per-fold hashes, in fold order; a cover given as is
      keeps its own `cover_hash`, and `design_id` is null where `compare`'s is;
    - `cover_label` is the covers' label if they share one, else the first 12
      hex digits of the hash;
    - `premium` is the probability-weighted mean of each year's premium: for
      designs, that fold's design premium, `premium_basis` `"design"`; a cover
      given as is is priced once on the pooled rows with `pricing`,
      `premium_basis` `"evaluation"`;
    - the net loss of a year adds its own premium;
    - `unique` is true if every fold's is, false if any fold's is, else null;
      `ties` is the sum of the folds', and `degenerate_tail` the folds' flag if
      they all agree, else `"mixed"`;
    - `unidentified_share` reads each row by its own fold's design, and
      `design_budget` is the mean over the rows of each fold's bound (null if
      any has none).

    `allocation` and `values` are those of `compare`. The table's metadata
    holds every block, one entry for the pooled rows, and one
    `setup.evaluations` entry with `fold` null and `method` `"kfold_pooled"`;
    `setup_text` states it and `write_comparison` writes it. The table ranks
    nothing.

    Raises `EvaluationError` (a `ValueError`) for pairs that are not the k folds
    of one split in fold order, held-out blocks that repeat or miss a year, a
    design whose design sample is not its fold's design block, a block with
    unequal probabilities, blocks of different `scenario` labels, and items that
    give different rows on different folds; otherwise as `compare` does, and
    `TypeError` for arguments of the wrong type.
    """
    return _pooled(
        entries,
        folds,
        pricing=pricing,
        options=options,
        allocation=allocation,
        values=values,
    ).table


def bootstrap_intervals(
    entries: Sequence[EvaluationEntry] | Sequence[PooledEntry],
    on: OutcomeSample | Sequence[tuple[OutcomeSample, OutcomeSample]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    ci_level: float = 0.9,
    split: Split | None = None,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    premiums: Mapping[str, float] | None = None,
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
    allocation: Allocation | None = None,
    values: Mapping[str, float] | None = None,
) -> Any:
    """Return block-bootstrap intervals of a comparison's metrics, as a long table.

    `on` is one block, read as `compare` reads it, or the folds of a k-fold
    split, read as `compare_pooled` reads them, with `entries` as each of them
    takes. `split` is required on one block, and `fold`, `design_sample` and
    `premiums` are for one block too: `premiums` maps a `cover_label` to a stated
    premium for the covers given as is, every cover of a set or package or none.

    `rng` is required and is the only source of randomness. One call of
    `rng.integers(0, Y, size=(n_bootstrap, Y))` draws the block's Y years with
    replacement, and its state at entry goes into the metadata's `rng`, step
    `"bootstrap"`. The same generator state gives the same table. One set of
    replicates serves every entry and every metric, and the designs are held
    fixed.

    The table has `to_long`'s columns and, for each wide row and metric column:

    - `value`: the wide table's number, which resampling never changes;
    - `ci_low`, `ci_high`: the (1 - ci_level)/2 and (1 + ci_level)/2 quantiles
      of the replicates, null when the metric is undefined in a replicate or
      null in the wide table;
    - `ci_level`, `n_bootstrap`: as stated;
    - `n_undefined`: the replicates in which the metric is undefined;
    - `short_block`: true when the column's level reaches the block's worst row,
      (1 - beta) * n <= 1 or T > n with n the rows the metric reads, and null
      for a column with no level. Such bounds cannot reach beyond the block's
      largest rows, so they understate how far the number could move.

    An interval describes how much a number moves with the years drawn. It
    leaves out uncertainty in the hazard data, the impact function and the
    design, and on a curve block it is the Monte Carlo error of the draws and
    says nothing about the curve. Two rows' intervals are not a test of a
    difference between them.

    The table holds an `EvaluationMetadata` under the same key as a
    comparison's, with `setup.intervals`, and `write_comparison` writes it.

    Raises as `compare` or `compare_pooled` does; `EvaluationError` for an
    invalid `n_bootstrap` or `ci_level`, for `split` missing on one block, for
    `split`, `fold`, `design_sample` or `premiums` with folds, and for a
    `premiums` key that matches no cover given as is; `TypeError` for an `rng`
    that is not a `numpy.random.Generator`.
    """
    generator = _require_generator(rng)
    count = _n_bootstrap(n_bootstrap)
    level = _ci_level(ci_level)
    checked_options = _options(options)
    if isinstance(on, OutcomeSample):
        evaluated = _block(
            entries,
            on,
            split=split,
            fold=fold,
            pricing=pricing,
            premiums=premiums,
            options=checked_options,
            design_sample=design_sample,
            allocation=allocation,
            values=values,
        )
    else:
        given = [
            name
            for name, value in (
                ("split", split),
                ("design_sample", design_sample),
                ("premiums", premiums),
            )
            if value is not None
        ]
        if fold != 0:
            given.append("fold")
        if given:
            raise EvaluationError(
                f"these arguments are for one block, and with folds every row reads "
                f"all the held-out years: {', '.join(given)}"
            )
        evaluated = _pooled(
            entries,
            on,
            pricing=pricing,
            options=checked_options,
            allocation=allocation,
            values=values,
        )
    return _interval_table(evaluated, generator, count, level)
