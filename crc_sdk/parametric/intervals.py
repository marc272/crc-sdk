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

**Rows built together.** Resampling rows treats them as independent draws. The
rows of a block whose columns `EmpiricalCopula`, `Comonotonic()` or
`GaussianCopula` moved, and of a block built with `HazardNoise(...,
hold="hazard_distribution")`, are not: each asset's or hazard's draws, or each
row's hazard at the asset, were placed by their rank over every row. So on
those rows each replicate builds the rows again as the block built them:

- **Re-dealt.** A row that reads two columns or more of such a block (an
  asset of several, or a hazard of several): each replicate redraws each
  column's draws on its own, with replacement, and deals them to the rows
  again by the model's rule. Under an `EmpiricalCopula` the draws sorted by
  their index fill the ranks each reference year held in that column, and a
  year's rows take its draws in the order they were drawn. Under
  `Comonotonic()` the draws of equal rank in every column share a row, and
  under `GaussianCopula` each row takes the draw of the rank its latent normal
  score held in the column, which the sample keeps (`latent_ranks`); copies of
  one draw are ranked in the order drawn.
- **Re-ranked.** A row that reads a column built with the held form: each
  replicate resamples by the unit above, and ranks the hazard at the asset
  again over the resampled rows, as the held form ranks it over the rows of a
  call. Each row takes the loss of its new hazard at the asset, and on a
  sample of several hazards its damage, which the block's own rows give.

A row of both is re-dealt and re-ranked. A replicate holds the rows rebuilt,
not reweighted: a row drawn twice can take two places in the reference, or two
hazards at the asset. Across hazards the combined loss is made again by the
sample's rule, from each hazard's damage, which the sample keeps beside its own
loss, times the asset's value: the one number whose product with each damage
is that hazard's own loss, as the loss model made it. So a sample with one
impact, and a built-in rule (`CappedSum`, `CompoundDamage`) whatever the
value, can be rebuilt. A sample without damages, written before samples kept
them, is rebuilt under a built-in rule on damage ratios (a value of 1) only.

**Rows that can't be built again keep today's method**, with today's draws and
numbers, and `setup.intervals.kept_today` lists each kind of them with the
reason, which the setup text states: a sample of several hazards whose
combined loss can't be made again (a rule of the caller's, a table's loss, or
a stated value or a value column on a sample without damages), a one-hazard
view whose combined loss reads hazards it doesn't hold, a selection of a held
sample, whose hazards at the asset came from rows it doesn't hold, a record of
an `EmpiricalCopula` without its draws per reference year, an `EmpiricalCopula`
block whose record says a basis-risk model of the caller's changed an index,
which then may not order the draws as the copula dealt them, a `GaussianCopula`
sample without its latent ranks (written before samples kept them), a
`Comonotonic()` block whose indices don't rise together (a basis-risk model of
the caller's moved one), a block that several models moved, and pooled rows of
an `EmpiricalCopula` or a held block, which come only from folds built by
hand. A cut of a `Comonotonic()` or `GaussianCopula` sample, as a holdout's
block, one fold's block or a selection of periods, is kept too: the model
dealt its columns over rows the cut doesn't hold. Pooled rows of such folds
are re-dealt over the joined held-out blocks, the one sample they are cuts
of, and each replicate is cut into its folds again.
`docs/parametric.md` gives a synthetic coverage study of these methods, and
lists the rows built together that it didn't measure. Resampling whole
reference years is not offered: it made most intervals of a copula block too
wide.

A model of the caller's also places each column's draws by a rule over every
row, so a row that reads several of its columns is built together too. Nothing
says how its reorder places the draws, so those rows keep the unit above, and
their coverage is unmeasured. The column `built_together` flags all those rows
by how the block was built, and the setup text says how each was resampled.

**Clusters.** Under `Clustered`, a `"cluster"` row reads its cluster's assets,
whose index its cover averages and whose losses it sums, and so does each of
those assets' allocated rows, a share of that cluster's payout; the `"total"`
row reads every asset. So the rows of a cluster of two assets or more are
built together as a `"total"` row is, and re-dealt or re-ranked by the same
rule, and those of a one-asset cluster are an asset's own rows.

Both functions rank nothing. They work in memory and do no I/O. Random numbers
come only from the `rng` a call states, and its state at entry is recorded, so
the table can be rebuilt.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import NDArray

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.combine import (
    CappedSum,
    CombineRule,
    CompoundDamage,
    _combined_impact,
)
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
    _any_entries,
    _any_entry,
    _AnyEntry,
    _block_fields,
    _check_keys,
    _cluster_payouts,
    _cluster_share,
    _ClusteredEntry,
    _component_values,
    _component_view,
    _CoverSet,
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
from crc_sdk.parametric.index import CompositeIndex, IndexDefinition
from crc_sdk.parametric.joint import ClusteredDesignResult, JointDesignResult
from crc_sdk.parametric.metadata import (
    _BUILT_TOGETHER,
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
    IndexComponent,
    OutcomeSample,
    SampleProvenance,
    _cover_index_values,
    _generator_record,
    _holds_hazard_distribution,
    _refuse_reference_year_blocks,
    _reorder_kinds,
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
    | Sequence[ClusteredDesignResult]
    | Sequence[CoverSet]
    | Sequence[CoverPackage]
)
"""What `compare_pooled` takes per entry: see its docstring."""

Unit = Literal["year", "row"]
"""How the rows of today's method are resampled: by year, or by row."""

RecordedUnit = Literal["year", "row", "as_built"]
"""What `setup.intervals.unit` records: `"as_built"` when a row was rebuilt."""

_Method = Literal["today", "rerank", "redeal"]
"""How one wide row is resampled."""

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


def _cluster_of_asset(entry: _ClusteredEntry) -> dict[str, str]:
    """The cluster of each asset of a design under `Clustered`."""
    return {
        asset: cluster
        for cluster, members in entry.design.members.items()
        for asset in members
    }


def _cluster_arrays(entry: _ClusteredEntry, run: _Run) -> list[_Arrays]:
    """A design's rows under `Clustered`, draw by draw, in the order of its rows.

    Under `joint=False` each cluster's row, then the total, whose contracts are
    the clusters' covers; under `joint=True` the total alone, one contract on
    the total. Then the allocated rows, each asset's share of its cluster's
    payout.
    """
    sample = run.sample
    design = entry.design
    paid = _cluster_payouts(entry, sample)
    arrays = [
        _Arrays(
            paid.loss[cluster],
            paid.payout[cluster],
            paid.filled[cluster],
            (paid.payout[cluster],),
        )
        for cluster in design.clusters or {}
    ]
    contracts = (
        (paid.total,) if design.clusters is None else tuple(paid.payout.values())
    )
    arrays.append(
        _Arrays(sample.loss.sum(axis=1), paid.total, paid.total_filled, contracts)
    )
    if run.allocation is None:
        return arrays
    assert run.values is not None
    members = design.members
    cluster_of = _cluster_of_asset(entry)
    for asset in sample.asset_ids:
        cluster = cluster_of[asset]
        share = _cluster_share(
            run.allocation, paid, cluster, members[cluster], sample, run.values, asset
        )
        loss = sample.loss[:, sample.asset_ids.index(asset)]
        arrays.append(_Arrays(loss, share, paid.filled[cluster], (share,)))
    return arrays


def _arrays(entry: _AnyEntry, run: _Run) -> list[_Arrays]:
    """Each row of `entry` draw by draw, in the order of `_rows(entry, run)`."""
    if isinstance(entry, _Item):
        return [_asset_arrays(entry, run.sample.asset_ids[0], run)]
    if isinstance(entry, _CoverSet):
        return _set_arrays(entry, run)
    if isinstance(entry, CoverPackage):
        return _package_arrays(entry, run)
    if isinstance(entry, _ClusteredEntry):
        return _cluster_arrays(entry, run)
    return _joint_arrays(entry, run)


# -- rows built together -------------------------------------------------------


_MOVED = frozenset({"comonotonic", "empirical_copula", "gaussian_copula"})
"""The built-in models that move a column's draws, each its own kind of rows."""

_CALLERS_MODEL = "dependence_model"
"""The kind of rows that a dependence model of the caller's moved."""

_HELD = "hazard_distribution"
"""`HazardNoise(..., hold="hazard_distribution")` ranked each row's hazard."""

_UNRECORDED = "the combined loss"
"""A stand-in column for a combined loss whose hazards a view doesn't record."""


def _view_columns(sample: OutcomeSample) -> frozenset[str]:
    """The columns that every row of a one-asset, one-hazard sample reads.

    That one column, unless the sample is a `component(name, loss="combined")`
    view: its loss reads the hazards with an impact too, which its
    `source_lineages` keeps. A view that doesn't keep them counts as reading
    several.
    """
    for step in reversed(sample.provenance.selection):
        if step.get("kind") != "component":
            continue
        if step.get("loss") != "combined":
            break
        read = sample.provenance.source_lineages
        return frozenset({str(step.get("name")), *(read or (_UNRECORDED,))})
    return frozenset(sample.asset_ids)


def _impact_columns(sample: OutcomeSample) -> frozenset[str]:
    """The hazards that the combined loss of a sample of several hazards reads."""
    components = sample.index_components
    assert components is not None
    return frozenset(
        name for name, component in components.items() if component.own_loss is not None
    )


def _index_columns(index: IndexDefinition | CompositeIndex) -> frozenset[str]:
    if isinstance(index, CompositeIndex):
        return frozenset(index.components)
    return frozenset({index.hazard_name})


def _cover_columns(cover: Cover, sample: OutcomeSample, asset: str) -> frozenset[str]:
    """The columns that the row of `cover` at `asset` reads: its index and loss."""
    if sample.index_components is not None:
        return _index_columns(cover.index_definition) | _impact_columns(sample)
    if len(sample.asset_ids) > 1:
        return frozenset({asset})
    return _view_columns(sample)


def _cluster_reads(entry: _ClusteredEntry, run: _Run) -> list[frozenset[str]]:
    """The assets each row of a design under `Clustered` reads.

    A cluster's row reads its own assets, whose index its cover averages and
    whose losses it sums, and so does each of their allocated rows, a share of
    that cluster's payout; the `"total"` row reads every asset.
    """
    members = {
        cluster: frozenset(assets) for cluster, assets in entry.design.members.items()
    }
    reads = [members[cluster] for cluster in entry.design.clusters or {}]
    reads.append(frozenset(run.sample.asset_ids))
    if run.allocation is not None:
        cluster_of = _cluster_of_asset(entry)
        reads += [members[cluster_of[asset]] for asset in run.sample.asset_ids]
    return reads


def _reads(entry: _AnyEntry, run: _Run) -> list[frozenset[str]]:
    """The columns each row of `entry` reads, in the order of `_arrays`.

    A column is an asset of a sample of several assets, a hazard of a sample
    of several hazards, and the one column of any other sample. A `"total"`
    row reads its parts' columns, and so do a joint design's rows, whose
    allocated rows share the total payout. Under `Clustered` a cluster's rows
    read that cluster's assets.
    """
    sample = run.sample
    if isinstance(entry, _ClusteredEntry):
        return _cluster_reads(entry, run)
    if isinstance(entry, _Item):
        return [_cover_columns(entry.cover, sample, sample.asset_ids[0])]
    if isinstance(entry, _CoverSet):
        parts = [
            _cover_columns(entry.items[asset].cover, sample, asset)
            for asset in sample.asset_ids
        ]
        return [*parts, frozenset().union(*parts)]
    if isinstance(entry, CoverPackage):
        impacts = _impact_columns(sample)
        parts = [frozenset({name}) | impacts for name in entry._items]
        return [*parts, frozenset().union(*parts)]
    every = frozenset().union(
        *(
            _cover_columns(entry.design.covers[asset], sample, asset)
            for asset in sample.asset_ids
        )
    )
    return [every] * (1 if run.allocation is None else 1 + len(sample.asset_ids))


def _held_columns(sample: OutcomeSample, names: frozenset[str]) -> frozenset[str]:
    """The columns among `names` whose hazard at the asset was ranked over the rows.

    The sample's own basis-risk model applies to every column; on a join, each
    part's applies to its asset.
    """
    document = sample.provenance.to_document()
    if not _holds_hazard_distribution(document):
        return frozenset()
    own = document.get("parts") or document.get("components")
    if own is None or _holds_hazard_distribution(
        {"basis_risk": document.get("basis_risk")}
    ):
        return names
    return frozenset(
        name for name in names if name in own and _holds_hazard_distribution(own[name])
    )


def _moved_by(sample: OutcomeSample) -> frozenset[str]:
    """The dependence models that moved the draws of `sample`'s columns.

    Each by its `kind`, `"empirical_copula"`, `"comonotonic"` or
    `"gaussian_copula"`, and `"dependence_model"` for a model of the caller's.
    Empty when every column is in draw order, as under `Independent()`.
    """
    return frozenset(
        kind if kind in _MOVED else _CALLERS_MODEL
        for kind in _reorder_kinds(sample.provenance.to_document())
    )


def _built_together(entry: _AnyEntry, run: _Run) -> list[frozenset[str]]:
    """How each row of `entry` reads rows built together, in the order of `_arrays`.

    The models that moved the block's columns (`_moved_by`), when the row
    reads two columns or more: each column's draws were placed by their ranks
    over every row, so its rows sit beside the other columns' by rule and not
    as independent draws. One column's draws are the same whatever their
    order, so a row of one column has none. `"hazard_distribution"` when a
    column the row reads holds the held form of `HazardNoise`. Empty when the
    row reads neither.
    """
    reads = _reads(entry, run)
    moved = _moved_by(run.sample)
    held = _held_columns(run.sample, frozenset().union(*reads))
    return [
        (moved if len(read) > 1 else frozenset())
        | (frozenset({_HELD}) if read & held else frozenset())
        for read in reads
    ]


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
class _Fold:
    """One fold of pooled rows: its held-out block's evaluation and its entries."""

    run: _Run
    entries: tuple[_AnyEntry, ...]


@dataclass(frozen=True, eq=False)
class _Evaluated:
    """A comparison table and what resampling its rows needs.

    `together` holds, for each of `reads`, how its rows were built together
    (`_built_together`). On one block, `run` is its evaluation, `entries` its
    entries, and `positions` says for each of `reads` which entry it is a row of
    and where in `_arrays(entry, run)`, so that a rebuilt replicate can read the
    row again. Pooled folds have no `run`: `folds` holds each fold's, and
    `positions` says the same of each fold's entries.
    """

    table: Any
    reads: tuple[_Read, ...]
    together: tuple[frozenset[str], ...]
    probability: FloatArray
    year: IntArray
    n_years: int
    unit: Unit
    options: MetricOptions
    run: _Run | None = None
    entries: tuple[_AnyEntry, ...] = ()
    positions: tuple[tuple[int, int], ...] = ()
    folds: tuple[_Fold, ...] = ()


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

    folds: tuple[_AnyEntry, ...]
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


def _has_designs(entry: _AnyEntry) -> bool:
    if isinstance(entry, (_JointEntry, _ClusteredEntry)):
        return True
    if isinstance(entry, _Item):
        return entry.design is not None
    if isinstance(entry, CoverPackage):
        return entry._designed
    return any(item.design is not None for item in entry.items.values())


def _pooled_entry(entry: Any, k: int) -> _PooledEntry:
    if isinstance(entry, (Cover, Mapping)):
        checked = _any_entry(entry)
        if _has_designs(checked):
            raise EvaluationError(
                "a mapping on its own is a set of covers given as is, which applies "
                f"on every fold; give designs as a sequence of {k}, one per fold"
            )
        return _PooledEntry((checked,) * k, True)
    if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
        raise TypeError(
            f"each entry is a sequence of {k} designs, joint designs, designs under "
            "Clustered, mappings of designs or packages of designs in fold order, or "
            f"one resolved Cover or mapping of covers, got {type(entry).__name__}"
        )
    if len(entry) != k:
        raise EvaluationError(
            f"an entry needs one item per fold: {k} folds, got {len(entry)} items"
        )
    per_fold = tuple(_any_entry(item) for item in entry)
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
    entry: _AnyEntry, train: OutcomeSample
) -> list[tuple[OutcomeSample, OutcomeSample]]:
    """Each design sample of `entry`, with the view of `train` it must equal."""
    if isinstance(entry, (_JointEntry, _ClusteredEntry)):
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


def _check_fold_designs(entry: _AnyEntry, train: OutcomeSample, fold: int) -> None:
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
        "cluster_id": first["cluster_id"],
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
        (
            row["level"],
            row["asset_id"],
            row["scope"],
            row["components"],
            row["cluster_id"],
        )
        for row in rows
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
    together: list[frozenset[str]]


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
    built = [
        _built_together(item, run) for item, run in zip(entry.folds, runs, strict=True)
    ]
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
        # A pooled row reads every fold's rows, so it reads any fold's built together.
        collected.together.append(
            frozenset().union(*(each[position] for each in built))
        )
        identities.append(identity)
        if not entry.given:
            _record_pooled(fold_rows, identity, collected)
    _record_folds(entry, groups, identities, collected)


def _pooled_metadata(
    per_fold: Sequence[Sequence[_AnyEntry]],
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
    collected = _PooledRows([], [], {}, {}, [], [], [])
    positions: list[tuple[int, int]] = []
    for position, entry in enumerate(pooled):
        before = len(collected.rows)
        _pool_entry_rows(
            entry,
            runs,
            sample_hash=sample_hash,
            probability=probability,
            collected=collected,
        )
        positions.extend(
            (position, place) for place in range(len(collected.rows) - before)
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
    distinct, of_year = np.unique(year, return_inverse=True)
    return _Evaluated(
        table=pa.Table.from_pylist(collected.rows, schema=schema),
        reads=tuple(collected.reads),
        together=tuple(collected.together),
        probability=probability,
        year=of_year.astype(np.int64),
        n_years=int(distinct.size),
        unit="year",
        options=runs[0].options,
        positions=tuple(positions),
        folds=tuple(_Fold(run, tuple(per_fold[fold])) for fold, run in enumerate(runs)),
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
    checked: list[_AnyEntry] = _any_entries(entries)
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
    drawn: list[_Arrays] = []
    places: list[tuple[int, int]] = []
    for position, entry in enumerate(checked):
        arrays = _arrays(entry, run)
        drawn.extend(arrays)
        places.extend((position, place) for place in range(len(arrays)))
    together = tuple(
        kinds for entry in checked for kinds in _built_together(entry, run)
    )
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
        together=together,
        probability=probability,
        year=positions.astype(np.int64),
        n_years=int(distinct.size),
        unit=_resampling_unit(sample),
        options=run.options,
        run=run,
        entries=tuple(checked),
        positions=tuple(places),
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


def _replicates(
    evaluated: _Evaluated, counts: IntArray, rows: Sequence[int] | None = None
) -> FloatArray:
    """The metrics under each row of year counts, `(B, rows, columns)`.

    A replicate gives each row its probability times how often its year was
    drawn, rescaled to sum to 1, and leaves out the rows of years not drawn.
    A metric undefined in a replicate is NaN. With `rows`, only those wide rows
    are computed, and the others stay NaN.
    """
    options = evaluated.options
    names = [column.name for column in metric_columns(options)]
    without = _no_premium_names(options)
    result = np.full(
        (counts.shape[0], len(evaluated.reads), len(names)), np.nan, dtype=np.float64
    )
    computed = range(len(evaluated.reads)) if rows is None else rows
    if not computed:
        return result
    for b, count in enumerate(counts):
        weight = evaluated.probability * count[evaluated.year]
        kept = weight > 0.0
        probability = weight[kept] / math.fsum(weight[kept].tolist())
        for r in computed:
            values = _metrics(evaluated.reads[r], kept, probability, options, without)
            result[b, r] = np.array([values[name] for name in names], dtype=np.float64)
    return result


# -- rebuilding rows built together --------------------------------------------


_REDEALT = "empirical_copula"
"""The kind of rows that each replicate deals again by the copula's reference."""

_COMONOTONIC = "comonotonic"
_GAUSSIAN = "gaussian_copula"

_DEALT = (_REDEALT, _COMONOTONIC, _GAUSSIAN)
"""The kinds of rows that each replicate deals again, by the model's own rule."""

_ONE_IMPACT = "one_impact"
"""The `combine` kind of a sample whose combined loss is one hazard's own loss."""

_BUILT_IN_RULES: Mapping[str, Callable[[Mapping[str, Any]], CombineRule]] = {
    "capped_sum": lambda record: CappedSum(record["cap"]),
    "compound_damage": lambda record: CompoundDamage(),
}
"""The built-in rules for a combined loss, made again from what a sample records."""


_Combine = Callable[[Mapping[str, FloatArray], Mapping[str, FloatArray]], FloatArray]
"""Makes a row's combined loss from its hazards' own losses and their damages."""


def _method(kinds: frozenset[str]) -> _Method:
    """How a row of these kinds is resampled: re-dealt, re-ranked, or as today."""
    if any(kind in kinds for kind in _DEALT):
        return "redeal"
    if _HELD in kinds:
        return "rerank"
    return "today"


def _kept_kind(kinds: frozenset[str]) -> str:
    """The kind a row kept on today's method would have been built again for."""
    return next((kind for kind in _DEALT if kind in kinds), _HELD)


@dataclass(frozen=True, eq=False)
class _Column:
    """One column of a block, as its rows hold it: an asset or a hazard.

    `hazard` is the hazard at the asset, `None` where it is the index. `loss`
    is the column's own loss: an asset's loss, a hazard's own loss, or `None`
    for a hazard with no impact. `damage` is a hazard's damage before the rule
    for the combined loss, `None` for an asset and where the sample keeps
    none. `by_hazard` holds, on a held column with a loss, the block's hazards
    at the asset sorted, and the loss and the damage of each.
    """

    name: str
    index: FloatArray
    hazard: FloatArray | None
    loss: FloatArray | None
    damage: FloatArray | None
    held: bool
    by_hazard: tuple[FloatArray, FloatArray, FloatArray | None] | None


@dataclass(frozen=True, eq=False)
class _Builder:
    """What rebuilding the rows of one block needs.

    `deal` is the model a re-deal follows, the kind of rows it deals, or
    `None` when no row is re-dealt. For an `EmpiricalCopula`, `members` holds
    the block's rows of each member of a window (one member for a curve), in
    row order, with the reference year of each row. For `Comonotonic()` and
    `GaussianCopula`, `members` holds each member's rows with their positions,
    and `ranks` each column's rank of each row's draw, by which the draws are
    dealt again. `year_rows` holds the rows of each year, for re-ranking, and
    `year_row` the one row of each year where every year has one. `combine`
    makes a sample of several hazards' combined loss from its hazards' own
    losses and damages.
    """

    sample: OutcomeSample
    columns: tuple[_Column, ...]
    members: tuple[tuple[IntArray, IntArray], ...]
    year_rows: tuple[IntArray, ...]
    year_row: IntArray | None
    combine: _Combine | None
    provenance: SampleProvenance
    deal: str | None = None
    ranks: tuple[IntArray, ...] = ()

    def drawn_rows(self, years: IntArray) -> IntArray:
        """The rows of the years drawn, in the order drawn."""
        if self.year_row is not None:
            single: IntArray = self.year_row[years]
            return single
        return np.concatenate([self.year_rows[year] for year in years])


class _Unbuildable(Exception):
    """A block's rows can't be built again; `reason` is what the record says."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_POOLED_BY_HAND = "pooled_by_hand"
_COMBINED_VIEW = "combined_view"
_COMBINED_LOSS_NEEDS_VALUE = "combined_loss_needs_value"
_CALLER_RULE = "caller_rule"
_TABLE_LOSS = "table_loss"
_COMBINED_LOSS_NOT_REPRODUCED = "combined_loss_not_reproduced"
_VALUE_NOT_FOUND = "value_not_found"
_HELD_SELECT = "held_select"
_HELD_LOSS_NOT_ONE = "held_loss_not_one"
_COPULA_WITHOUT_K = "copula_without_k"
_INDEX_MOVED = "index_moved"
_GAUSSIAN_WITHOUT_RANKS = "gaussian_without_ranks"
_INDEX_NOT_COMONOTONE = "index_not_comonotone"
_SEVERAL_MODELS = "several_models"
_CUT_OF_DEAL = "cut_of_deal"
"""Why the rows of a block were not built again, as `kept_today` records it."""

_CUTS = frozenset({"select", "holdout", "kfold"})
"""The kinds of selection steps that cut rows out of a sample."""


def _block_columns(sample: OutcomeSample) -> list[_Column]:
    """The columns of `sample`, without the losses by hazard."""
    components = sample.index_components
    if components is None:
        held = _held_columns(sample, frozenset(sample.asset_ids))
        return [
            _Column(
                asset,
                sample.index[:, j],
                None if sample.hazard is None else sample.hazard[:, j],
                sample.loss[:, j],
                None,
                asset in held,
                None,
            )
            for j, asset in enumerate(sample.asset_ids)
        ]
    held = _held_columns(sample, frozenset(components))
    return [
        _Column(
            name,
            component.values[:, 0],
            None if component.hazard is None else component.hazard[:, 0],
            None if component.own_loss is None else component.own_loss[:, 0],
            None if component.damage is None else component.damage[:, 0],
            name in held,
            None,
        )
        for name, component in components.items()
    ]


def _by_hazard(column: _Column) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """A held column's hazards at the asset, sorted, and the loss and damage of each.

    The held form gives each row one of the index values of the rows the call
    built, so on a block of all those rows every index value is a row's hazard
    at the asset, and that row's loss and damage are those of the value. Raises
    unless every index value is one, and unless rows of one hazard have one
    loss and one damage.
    """
    assert column.loss is not None
    hazard = column.index if column.hazard is None else column.hazard
    order = np.argsort(hazard, kind="stable")
    hazards, losses = hazard[order], column.loss[order]
    damages = None if column.damage is None else column.damage[order]
    same = hazards[1:] == hazards[:-1]
    for values in (losses, damages):
        if values is not None and (values[1:][same] != values[:-1][same]).any():
            # The loss of a new hazard can't be read from these rows.
            raise _Unbuildable(_HELD_LOSS_NOT_ONE)
    found = np.minimum(np.searchsorted(hazards, column.index), hazards.size - 1)
    if not (hazards[found] == column.index).all():
        # The held form gives each row one of the index values of its call's
        # rows, and this block doesn't hold them all, as a selection doesn't.
        raise _Unbuildable(_HELD_SELECT)
    return hazards, losses, damages


def _loss_of(
    column: _Column, hazard: FloatArray
) -> tuple[FloatArray, FloatArray | None]:
    """The loss and damage of each hazard at the asset, from the block's rows."""
    assert column.by_hazard is not None
    hazards, losses, damages = column.by_hazard
    found = np.searchsorted(hazards, hazard)
    loss: FloatArray = losses[found]
    return loss, None if damages is None else damages[found]


def _values_are_one(sample: OutcomeSample) -> bool:
    """Whether every loss model of a sample of several hazards has a value of 1.

    A loss model without a value gives the loss as the damage ratio, so its
    record's value source is `"none"`; a stated value or a value column is
    never recorded.
    """
    document = sample.provenance.loss_model or {}
    listed = document.get("components")
    if document.get("kind") != "components" or not isinstance(listed, Sequence):
        return False
    sources = [
        ((entry.get("loss_model") or {}).get("value") or {}).get("source")
        for entry in listed
        if isinstance(entry, Mapping)
    ]
    return len(sources) == len(listed) and all(source == "none" for source in sources)


def _own_loss_of(name: str) -> _Combine:
    """The combined loss of a sample with one impact: that hazard's own loss."""

    def combined(
        own: Mapping[str, FloatArray], damage: Mapping[str, FloatArray]
    ) -> FloatArray:
        return own[name]

    return combined


def _rule_on_damage_ratios(
    rule: CombineRule, kind: str, impacted: Sequence[str]
) -> _Combine:
    """A rule's combined loss where the value is 1, so each own loss is an impact.

    For a sample that keeps no damage, as one written before samples kept it.
    """

    def combined(
        own: Mapping[str, FloatArray], damage: Mapping[str, FloatArray]
    ) -> FloatArray:
        return _combined_impact(rule, kind, {name: own[name] for name in impacted})

    return combined


def _rule_times_value(
    rule: CombineRule, kind: str, impacted: Sequence[str], value: float
) -> _Combine:
    """A rule's result on the hazards' damages, times the asset's value.

    The product is the loss model's, `impact * value`, so on the block's own
    rows it gives the sample's combined loss exactly.
    """

    def combined(
        own: Mapping[str, FloatArray], damage: Mapping[str, FloatArray]
    ) -> FloatArray:
        result = _combined_impact(rule, kind, {name: damage[name] for name in impacted})
        loss: FloatArray = result * value
        return loss

    return combined


_CANDIDATE_VALUES = 3
"""How many of the most frequent quotients `_value_of` tries, with their neighbours."""


def _value_of(
    pairs: Sequence[tuple[FloatArray, FloatArray]],
    combined: tuple[FloatArray, FloatArray],
) -> float:
    """The asset's value: the one number whose product with each damage is its loss.

    `pairs` holds each hazard's damage and own loss, and `combined` the rule's
    result on the block and the sample's combined loss. A loss model multiplies
    each damage by the value, so the value is read back from the quotients of
    the rows, each tried with its floating-point neighbours: the first that
    gives every own loss exactly, and the combined loss too, is it. A block
    whose damages are all 0 has losses of 0 whatever the value, and so does
    every replicate of it, so its value is taken as 0.

    Raises `_Unbuildable` unless a value gives every own loss exactly. One that
    gives every own loss but not the combined loss is returned, and the caller's
    check of the combined loss then refuses it.
    """
    damage = np.concatenate([each for each, _ in pairs])
    loss = np.concatenate([each for _, each in pairs])
    positive = damage > 0.0
    if not positive.any():
        if loss.any():
            raise _Unbuildable(_VALUE_NOT_FOUND)
        return 0.0
    quotients, counts = np.unique(loss[positive] / damage[positive], return_counts=True)
    frequent = quotients[np.argsort(-counts, kind="stable")[:_CANDIDATE_VALUES]]
    found: float | None = None
    for quotient in frequent:
        for value in (
            quotient,
            np.nextafter(quotient, -np.inf),
            np.nextafter(quotient, np.inf),
        ):
            if not np.array_equal(damage * value, loss):
                continue
            if np.array_equal(combined[0] * value, combined[1]):
                return float(value)
            found = float(value) if found is None else found
    if found is None:
        raise _Unbuildable(_VALUE_NOT_FOUND)
    return found


def _combine(sample: OutcomeSample, columns: Sequence[_Column]) -> _Combine:
    """The combined loss of a sample of several hazards, from its hazards' rows.

    A sample with one impact has that hazard's own loss. A built-in rule takes
    the hazards' damages, and its result is multiplied by the asset's value,
    read from the block as the one number whose product with each damage is
    that hazard's own loss. A sample that keeps no damage is made again on
    damage ratios only, where the value is 1 and each own loss is an impact.
    Raises `_Unbuildable` for any other sample, and unless the rule gives the
    sample's own combined loss on its rows.
    """
    record = sample.provenance.combine or {}
    kind = str(record.get("kind"))
    impacted = [column.name for column in columns if column.loss is not None]
    own = {column.name: column.loss for column in columns if column.loss is not None}
    damage = {
        column.name: column.damage for column in columns if column.damage is not None
    }
    rule: _Combine
    if kind == _ONE_IMPACT:
        only = str(record.get("component"))
        rule = _own_loss_of(only)
    elif kind in _BUILT_IN_RULES:
        built = _BUILT_IN_RULES[kind](record)
        if impacted and set(damage) == set(impacted):
            value = _value_of(
                [(damage[name], own[name]) for name in impacted],
                (
                    _combined_impact(built, kind, {n: damage[n] for n in impacted}),
                    sample.loss[:, 0],
                ),
            )
            rule = _rule_times_value(built, kind, impacted, value)
        elif _values_are_one(sample):
            rule = _rule_on_damage_ratios(built, kind, impacted)
        else:
            # Without the damages, the value can't be read from the rows, and a
            # record never holds it.
            raise _Unbuildable(_COMBINED_LOSS_NEEDS_VALUE)
    else:
        raise _Unbuildable(_TABLE_LOSS if kind == "table" else _CALLER_RULE)
    if not np.array_equal(rule(own, damage), sample.loss[:, 0]):
        raise _Unbuildable(_COMBINED_LOSS_NOT_REPRODUCED)
    return rule


def _index_moved(sample: OutcomeSample) -> bool:
    """Whether the record says a basis-risk model changed a column's index.

    Each column's own provenance, in `parts` or `components`, holds its
    basis-risk model's summary, where the sampler recorded whether the index
    the model returned differs from the one it was given (`index_changed`).
    No built-in model changes it, so there the index is the curve's draw.
    """
    own = sample.provenance.parts or sample.provenance.components or {}
    for document in own.values():
        details = document.get("details") if isinstance(document, Mapping) else None
        summary = details.get("basis_risk") if isinstance(details, Mapping) else None
        if isinstance(summary, Mapping) and summary.get("index_changed") is True:
            return True
    return False


def _reference_years(sample: OutcomeSample) -> tuple[tuple[IntArray, IntArray], ...]:
    """Each member's rows of an `EmpiricalCopula` block, with each row's year.

    A row's period is its draw number in the call that built the sample, and
    the copula deals within each member of a window (row `d` reads member
    `d mod m`), `k` rows to each reference year, in reference-year order.

    The re-deal reads each row's rank in its column from the index. The
    copula ranked the draws by their uniforms, which the sample doesn't keep,
    and the curve's draw rises with its uniform, so the index gives that order
    wherever it is the draw. Raises `_Unbuildable` when the record doesn't
    state `k`, and when it says a basis-risk model of the caller's changed an
    index: that index may not give the copula's order, and nothing in a block
    of one row per reference year would show it.
    """
    record = (
        sample.provenance.dependence
        if sample.index_components is None
        else sample.provenance.hazard_dependence
    ) or {}
    k = record.get("k")
    if (
        record.get("kind") != _REDEALT
        or isinstance(k, bool)
        or not isinstance(k, int)
        or k < 1
    ):
        raise _Unbuildable(_COPULA_WITHOUT_K)
    if _index_moved(sample):
        raise _Unbuildable(_INDEX_MOVED)
    window = sample.provenance.details.get("window")
    m = 1 if window is None else len(window["members"])
    period = sample.period.astype(np.int64)
    members: list[tuple[IntArray, IntArray]] = []
    for member in range(m):
        rows = np.flatnonzero(period % m == member).astype(np.int64)
        if rows.size:
            members.append((rows, (period[rows] // m) // k))
    return tuple(members)


def _window_members(sample: OutcomeSample) -> int:
    """How many members the sample's window of horizons has: 1 for a curve."""
    window = sample.provenance.details.get("window")
    return 1 if window is None else len(window["members"])


def _ranked_members(
    sample: OutcomeSample, columns: Sequence[_Column], deal: str
) -> tuple[tuple[tuple[IntArray, IntArray], ...], tuple[IntArray, ...]]:
    """Each member's rows, and each column's rank of each row's draw.

    A model reorders within each member of a window (row `d` reads member `d
    mod m`). Under `GaussianCopula` the ranks are those the sample keeps, its
    latent normal scores' ranks. Under `Comonotonic()` the draws of equal rank
    share a row, so one rank serves every column: the order of the rows by
    the first column's index, ties by the next column's and then by row, along
    which every column's index must rise. Raises `_Unbuildable` when the
    sample doesn't keep the ranks, or when the indices don't rise together,
    as where a basis-risk model of the caller's moved an index.
    """
    m = _window_members(sample)
    period = sample.period.astype(np.int64)
    members: list[tuple[IntArray, IntArray]] = []
    for member in range(m):
        rows = np.flatnonzero(period % m == member).astype(np.int64)
        if rows.size:
            members.append((rows, np.arange(rows.size, dtype=np.int64)))
    if deal == _GAUSSIAN:
        kept = sample.latent_ranks
        if kept is None:
            raise _Unbuildable(_GAUSSIAN_WITHOUT_RANKS)
        ranks = tuple(kept[:, j] for j in range(kept.shape[1]))
        for rows, _ in members:
            for rank in ranks:
                if np.unique(rank[rows]).size != rows.size:
                    # Ties leave a row's place among its column's draws open.
                    raise _Unbuildable(_GAUSSIAN_WITHOUT_RANKS)
        return tuple(members), ranks
    common = np.empty(len(sample), dtype=np.int64)
    for rows, positions in members:
        keys = [column.index[rows] for column in reversed(columns)]
        order = np.lexsort((positions, *keys))
        for column in columns:
            if (np.diff(column.index[rows][order]) < 0).any():
                raise _Unbuildable(_INDEX_NOT_COMONOTONE)
        common[rows[order]] = positions
    return tuple(members), (common,) * len(columns)


def _deal_of(sample: OutcomeSample) -> str:
    """The rule a re-deal of `sample`'s rows follows: the kind of its model.

    An `EmpiricalCopula` keeps its own rule wherever it is among the models
    that moved the columns. `Comonotonic()` and `GaussianCopula` each need to
    be the one model. Raises `_Unbuildable` for several.
    """
    moved = _moved_by(sample)
    if _REDEALT in moved:
        return _REDEALT
    if len(moved) == 1 and next(iter(moved)) in _DEALT:
        return next(iter(moved))
    raise _Unbuildable(_SEVERAL_MODELS)


@dataclass(frozen=True, eq=False)
class _Plan:
    """How each row of a call is resampled, and what rebuilding rows needs.

    `methods` holds each row's method once the rows that can't be built again
    are taken back to today's. `kept` holds, for each of those rows, the
    reason. `builder` is `None` when no row is built again. For pooled rows
    re-dealt, it is the builder of the joined held-out blocks, and `folds`
    holds each fold's rows in that sample, in fold order; it is empty otherwise.
    """

    methods: tuple[_Method, ...]
    kept: Mapping[int, str]
    builder: _Builder | None
    folds: tuple[IntArray, ...] = ()


def _reasons(
    evaluated: _Evaluated, methods: Sequence[_Method]
) -> tuple[dict[_Method, str], _Builder | None]:
    """Why each method can't build the block's rows again, and what it needs."""
    run = evaluated.run
    assert run is not None
    return _block_reasons(run.sample, evaluated.year, evaluated.n_years, methods)


def _block_reasons(
    sample: OutcomeSample, year: IntArray, n_years: int, methods: Sequence[_Method]
) -> tuple[dict[_Method, str], _Builder | None]:
    """Why each method can't build the rows of `sample` again, and what it needs.

    `year` is each row's year among the block's `n_years`, for re-ranking.
    """
    if (
        sample.index_components is None
        and len(sample.asset_ids) == 1
        and len(_view_columns(sample)) > 1
    ):
        # The view's loss reads hazards whose draws it doesn't hold.
        return {"rerank": _COMBINED_VIEW, "redeal": _COMBINED_VIEW}, None
    columns = _block_columns(sample)
    held_reason: str | None = None
    try:
        columns = [
            dataclasses.replace(column, by_hazard=_by_hazard(column))
            if column.held and column.loss is not None
            else column
            for column in columns
        ]
    except _Unbuildable as unbuildable:
        held_reason = unbuildable.reason
    redeal = "redeal" in methods
    held_loss = any(column.held and column.loss is not None for column in columns)
    several = sample.index_components is not None
    combine = None
    combine_reason: str | None = None
    if several and (redeal or held_loss):
        try:
            combine = _combine(sample, columns)
        except _Unbuildable as unbuildable:
            combine_reason = unbuildable.reason
    members: tuple[tuple[IntArray, IntArray], ...] = ()
    ranks: tuple[IntArray, ...] = ()
    deal: str | None = None
    copula_reason: str | None = None
    if redeal:
        try:
            deal = _deal_of(sample)
            if deal == _REDEALT:
                members = _reference_years(sample)
            elif any(step.get("kind") in _CUTS for step in sample.provenance.selection):
                # The model dealt each column over rows the cut doesn't hold.
                raise _Unbuildable(_CUT_OF_DEAL)
            else:
                members, ranks = _ranked_members(sample, columns, deal)
        except _Unbuildable as unbuildable:
            copula_reason = unbuildable.reason
    reasons: dict[_Method, str] = {}
    rerank_reason = held_reason or (combine_reason if held_loss else None)
    if rerank_reason is not None:
        reasons["rerank"] = rerank_reason
    held = any(column.held for column in columns)
    redeal_reason = copula_reason or combine_reason or (held_reason if held else None)
    if redeal_reason is not None:
        reasons["redeal"] = redeal_reason
    year_rows = tuple(
        np.flatnonzero(year == each).astype(np.int64) for each in range(n_years)
    )
    provenance = sample.provenance
    return reasons, _Builder(
        sample=sample,
        columns=tuple(columns),
        members=members,
        year_rows=year_rows,
        year_row=(
            np.array([rows[0] for rows in year_rows], dtype=np.int64)
            if all(rows.size == 1 for rows in year_rows)
            else None
        ),
        combine=combine,
        # Only the arrays are read again, so the rebuilt rows need no record.
        provenance=SampleProvenance(
            source_kind=provenance.source_kind,
            sources=(),
            semantics_source=provenance.semantics_source,
            dependence=None if len(sample.asset_ids) == 1 else {"kind": "as_built"},
            hazard_dependence=(
                None if sample.index_components is None else {"kind": "as_built"}
            ),
        ),
        deal=deal if copula_reason is None else None,
        ranks=ranks,
    )


def _plan(evaluated: _Evaluated) -> _Plan:
    """Each row's method; rows that can't be built again keep today's method.

    A row that can't be built again, as the block doesn't hold what that needs,
    is resampled as before, with today's draws and numbers, and `kept` says why.
    """
    wanted = [_method(kinds) for kinds in evaluated.together]
    if all(method == "today" for method in wanted):
        return _Plan(tuple(wanted), {}, None)
    if evaluated.run is None:
        return _pooled_plan(evaluated, wanted)
    reasons, builder = _reasons(evaluated, wanted)
    kept = {r: reasons[method] for r, method in enumerate(wanted) if method in reasons}
    methods = tuple("today" if r in kept else method for r, method in enumerate(wanted))
    rebuilt = any(method != "today" for method in methods)
    return _Plan(methods, kept, builder if rebuilt else None)


def _joined_blocks(
    tests: Sequence[OutcomeSample],
) -> tuple[OutcomeSample, tuple[IntArray, ...]]:
    """The held-out blocks as the one sample they are cuts of, and each one's rows.

    The blocks partition the years of one sample (`_check_steps`,
    `_check_years`), so their rows, in period order, are that sample's, and
    its record is a block's without the fold's step.
    """
    first = tests[0]
    order = np.argsort(np.concatenate([test.period for test in tests]), kind="stable")

    def rows(parts: Sequence[Any]) -> Any:
        return None if parts[0] is None else np.concatenate(parts)[order]

    provenance = first.provenance
    step = _split_step(first)
    pools = [test.provenance.pool or {} for test in tests]
    held = [test.index_components or {} for test in tests]
    sample = dataclasses.replace(
        first,
        period=rows([test.period for test in tests]),
        probability=np.full(order.size, 1.0 / order.size, dtype=np.float64),
        index=rows([test.index for test in tests]),
        loss=rows([test.loss for test in tests]),
        hazard=rows([test.hazard for test in tests]),
        provenance=dataclasses.replace(
            provenance,
            selection=tuple(s for s in provenance.selection if s is not step),
            pool=provenance.pool
            and {
                **provenance.pool,
                **{
                    key: rows([np.asarray(pool[key]) for pool in pools]).tolist()
                    for key in ("year", "run")
                },
            },
        ),
        index_components=first.index_components
        and {
            name: dataclasses.replace(
                component,
                **{
                    field: rows([getattr(each[name], field) for each in held])
                    for field in ("values", "hazard", "own_loss", "damage")
                },
            )
            for name, component in first.index_components.items()
        },
        latent_ranks=rows([test.latent_ranks for test in tests]),
    )
    object.__setattr__(sample, "_view_of", first._view_of)
    place = np.argsort(order, kind="stable").astype(np.int64)
    ends = np.cumsum([len(test) for test in tests])[:-1]
    return sample, tuple(np.split(place, ends))


def _pooled_plan(evaluated: _Evaluated, wanted: Sequence[_Method]) -> _Plan:
    """Each pooled row's method: re-dealt over the joined blocks, or kept as today.

    A pooled row reads every fold's held-out block, each with its own fold's
    designs. The blocks are cuts of one sample, whose columns `Comonotonic()`
    or `GaussianCopula` dealt over every row of it, so a row of several such
    columns is re-dealt on the joined blocks, as a row of that sample is, and
    each replicate is cut into its folds again. `year_block_kfold` refuses
    blocks of an `EmpiricalCopula` and of a held `HazardNoise`, so pooled rows
    of them come only from folds built by hand, and keep today's method. A row
    whose joined blocks can't be dealt again keeps it too, with the reason.
    """
    kept: dict[int, str] = {}
    for r, (method, kinds) in enumerate(zip(wanted, evaluated.together, strict=True)):
        if method != "today" and (_REDEALT in kinds or _HELD in kinds):
            kept[r] = _POOLED_BY_HAND
    dealt = [
        r for r, method in enumerate(wanted) if method == "redeal" and r not in kept
    ]
    builder: _Builder | None = None
    folds: tuple[IntArray, ...] = ()
    if dealt:
        sample, folds = _joined_blocks([fold.run.sample for fold in evaluated.folds])
        distinct, year = np.unique(_row_years(sample), return_inverse=True)
        reasons, builder = _block_reasons(
            sample, year.astype(np.int64), int(distinct.size), ["redeal"]
        )
        if "redeal" in reasons:
            kept.update(dict.fromkeys(dealt, reasons["redeal"]))
            builder, folds = None, ()
    methods = tuple("today" if r in kept else method for r, method in enumerate(wanted))
    return _Plan(methods, kept, builder, folds)


def _kept_record(
    evaluated: _Evaluated, kept: Mapping[int, str]
) -> list[dict[str, str]]:
    """`kept_today`: each kind of rows kept as today, with its reason, once each."""
    found = {(_kept_kind(evaluated.together[r]), reason) for r, reason in kept.items()}
    order = list(_BUILT_TOGETHER)
    return [
        {"kind": kind, "reason": reason}
        for kind, reason in sorted(
            found, key=lambda pair: (order.index(pair[0]), pair[1])
        )
    ]


def _redeal_draws(
    builder: _Builder, rng: np.random.Generator, n_bootstrap: int
) -> list[IntArray]:
    """Each column's draws dealt again, as rows of the block: `(B, n)` per column.

    For each column in the block's order, and each member in order, one call
    `rng.integers(0, n_m, size=(B, n_m))` redraws the member's `n_m` draws with
    replacement. They are dealt by the model's rule: an `EmpiricalCopula`'s
    (`_reference_draws`), or the ranks of `Comonotonic()` and `GaussianCopula`
    (`_ranked_draws`).
    """
    if builder.deal == _REDEALT:
        return _reference_draws(builder, rng, n_bootstrap)
    return _ranked_draws(builder, rng, n_bootstrap)


def _reference_draws(
    builder: _Builder, rng: np.random.Generator, n_bootstrap: int
) -> list[IntArray]:
    """An `EmpiricalCopula`'s draws dealt again: `(B, n)` source rows per column.

    The redrawn draws, sorted by their index (ties in the order drawn), fill
    the ranks that each reference year's rows held in the column, and a year's
    rows take its draws in the order they were drawn, as the copula deals them.
    """
    n = len(builder.sample)
    sources: list[IntArray] = []
    for column in builder.columns:
        source = np.empty((n_bootstrap, n), dtype=np.int64)
        for rows, years in builder.members:
            size = rows.size
            drawn = rng.integers(0, size, size=(n_bootstrap, size), dtype=np.int64)
            index = column.index[rows]
            rank = np.empty(size, dtype=np.int64)
            rank[np.argsort(index, kind="stable")] = np.arange(size, dtype=np.int64)
            ranked = np.argsort(index[drawn], axis=1, kind="stable")
            held = ranked[:, rank]
            dealt = np.lexsort((held, np.broadcast_to(years, held.shape)), axis=-1)
            taken = np.take_along_axis(held, dealt, axis=1)
            source[:, rows] = rows[np.take_along_axis(drawn, taken, axis=1)]
        sources.append(source)
    return sources


def _ranked_draws(
    builder: _Builder, rng: np.random.Generator, n_bootstrap: int
) -> list[IntArray]:
    """Draws dealt again by the model's ranks: `(B, n)` source rows per column.

    The redrawn draws of a column are put in the order of their ranks there,
    copies of one draw in the order drawn, and each row takes the draw of the
    rank it held: under `Comonotonic()` the same in every column, so the draws
    of equal rank share a row, and under `GaussianCopula` that of its latent
    normal score in the column, which the sample keeps.
    """
    n = len(builder.sample)
    sources: list[IntArray] = []
    for rank in builder.ranks:
        source = np.empty((n_bootstrap, n), dtype=np.int64)
        for rows, positions in builder.members:
            size = rows.size
            drawn = rng.integers(0, size, size=(n_bootstrap, size), dtype=np.int64)
            held = rank[rows]
            place = np.empty(size, dtype=np.int64)
            place[np.argsort(held, kind="stable")] = positions
            ranked = np.argsort(held[drawn], axis=1, kind="stable")
            source[:, rows] = rows[np.take_along_axis(drawn, ranked[:, place], axis=1)]
        sources.append(source)
    return sources


def _reranked(
    column: _Column, rows: IntArray
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    """A held column's hazard at the asset ranked again over `rows`, and its loss.

    The rows are ranked by their hazard at the asset, which holds the rank the
    held form gave them, ties by the index and then by row, and the row of rank
    r takes the r-th smallest index value among them. Returns the hazards at
    the asset, and the loss and the damage of each, `None` where the column has
    none.
    """
    index = column.index[rows]
    hazard = index if column.hazard is None else column.hazard[rows]
    order = np.lexsort((np.arange(rows.size), index, hazard))
    placed = np.empty_like(index)
    placed[order] = np.sort(index)
    if column.loss is None:
        return placed, None, None
    return placed, *_loss_of(column, placed)


def _rebuilt_sample(
    builder: _Builder, sources: Sequence[IntArray], probability: FloatArray
) -> OutcomeSample:
    """The block's columns with each column's rows taken from `sources`.

    A held column's hazard at the asset is ranked again over its rows, and a
    sample of several hazards takes its combined loss again from its rule. A
    component's view stays that component's view, as a block of it does.
    """
    sample = builder.sample
    index: list[FloatArray] = []
    hazard: list[FloatArray] = []
    loss: list[FloatArray | None] = []
    damage: list[FloatArray | None] = []
    for column, rows in zip(builder.columns, sources, strict=True):
        index.append(column.index[rows])
        if column.held:
            placed, own, harm = _reranked(column, rows)
        else:
            placed = index[-1] if column.hazard is None else column.hazard[rows]
            own = None if column.loss is None else column.loss[rows]
            harm = None if column.damage is None else column.damage[rows]
        hazard.append(placed)
        loss.append(own)
        damage.append(harm)
    n = probability.shape[0]
    components = sample.index_components
    if components is None:
        # Every asset has a loss.
        losses = [value for value in loss if value is not None]
        one_hazard = OutcomeSample(
            asset_ids=sample.asset_ids,
            period=np.arange(n, dtype=np.int64),
            probability=probability,
            index=np.column_stack(index),
            loss=np.column_stack(losses),
            hazard=None if sample.hazard is None else np.column_stack(hazard),
            index_unit=sample.index_unit,
            loss_unit=sample.loss_unit,
            semantics=sample.semantics,
            provenance=builder.provenance,
            scenario=sample.scenario,
        )
        object.__setattr__(one_hazard, "_view_of", sample._view_of)
        return one_hazard
    rebuilt = {
        column.name: IndexComponent(
            index[j],
            unit=components[column.name].unit,
            tail=components[column.name].tail,
            hazard=None if column.hazard is None else hazard[j],
            own_loss=loss[j],
            damage=damage[j],
        )
        for j, column in enumerate(builder.columns)
    }
    own_losses = {
        name: value
        for name, value in zip(rebuilt, loss, strict=True)
        if value is not None
    }
    damages = {
        name: value
        for name, value in zip(rebuilt, damage, strict=True)
        if value is not None
    }
    combined = (
        sample.loss[sources[0], 0]
        if builder.combine is None
        else builder.combine(own_losses, damages)
    )
    first = next(iter(rebuilt.values()))
    return OutcomeSample(
        asset_ids=sample.asset_ids,
        period=np.arange(n, dtype=np.int64),
        probability=probability,
        index=first.values,
        loss=combined,
        hazard=first.hazard,
        index_unit=sample.index_unit,
        loss_unit=sample.loss_unit,
        semantics=sample.semantics,
        provenance=builder.provenance,
        scenario=sample.scenario,
        index_components=rebuilt,
    )


def _rebuilt_replicates(
    evaluated: _Evaluated,
    builder: _Builder,
    methods: Sequence[_Method],
    draws: IntArray,
    redealt: Sequence[IntArray],
    result: FloatArray,
) -> None:
    """Fill `result` for the rows re-dealt or re-ranked, replicate by replicate.

    A re-ranked replicate holds the rows of the years drawn, in the order
    drawn, each as often as its year was drawn. A re-dealt replicate holds the
    block's rows, each column's from its own redrawn draws. Each replicate
    builds one sample per method, and every entry's rows of that method read it.
    """
    run = evaluated.run
    assert run is not None
    options = evaluated.options
    names = [column.name for column in metric_columns(options)]
    without = _no_premium_names(options)
    by_method = {
        method: [r for r, each in enumerate(methods) if each == method]
        for method in ("rerank", "redeal")
    }
    for b in range(draws.shape[0]):
        for method, rows in by_method.items():
            if not rows:
                continue
            if method == "rerank":
                read = builder.drawn_rows(draws[b])
                sources = [read] * len(builder.columns)
                weight = evaluated.probability[read]
                probability = weight / math.fsum(weight.tolist())
            else:
                sources = [source[b] for source in redealt]
                probability = evaluated.probability
            rebuilt = dataclasses.replace(
                run, sample=_rebuilt_sample(builder, sources, probability)
            )
            every_row = np.ones(probability.shape[0], dtype=np.bool_)
            arrays: dict[int, list[_Arrays]] = {}
            for r in rows:
                entry, place = evaluated.positions[r]
                if entry not in arrays:
                    arrays[entry] = _arrays(evaluated.entries[entry], rebuilt)
                read_again = dataclasses.replace(
                    evaluated.reads[r], arrays=arrays[entry][place]
                )
                values = _metrics(read_again, every_row, probability, options, without)
                result[b, r] = np.array(
                    [values[name] for name in names], dtype=np.float64
                )


def _pooled_rebuilt_replicates(
    evaluated: _Evaluated,
    plan: _Plan,
    redealt: Sequence[IntArray],
    result: FloatArray,
) -> None:
    """Fill `result` for the pooled rows re-dealt, replicate by replicate.

    A replicate deals the joined held-out blocks again, as the one sample they
    are, cuts it into the folds' blocks, reads each re-dealt row on every
    block with that fold's entries, and joins the blocks' rows in fold order,
    each paying its own fold's contract, as `compare_pooled` joins them.
    """
    builder = plan.builder
    assert builder is not None
    options = evaluated.options
    names = [column.name for column in metric_columns(options)]
    without = _no_premium_names(options)
    rows = [r for r, method in enumerate(plan.methods) if method == "redeal"]
    every_row = np.ones(evaluated.probability.shape[0], dtype=np.bool_)
    for b in range(result.shape[0]):
        rebuilt = [
            dataclasses.replace(
                fold.run,
                sample=_rebuilt_sample(
                    builder,
                    [source[b][rows_of] for source in redealt],
                    fold.run.sample.probability,
                ),
            )
            for fold, rows_of in zip(evaluated.folds, plan.folds, strict=True)
        ]
        arrays: list[dict[int, list[_Arrays]]] = [{} for _ in rebuilt]
        for r in rows:
            entry, place = evaluated.positions[r]
            parts: list[_Arrays] = []
            for fold, run, read in zip(evaluated.folds, rebuilt, arrays, strict=True):
                if entry not in read:
                    read[entry] = _arrays(fold.entries[entry], run)
                parts.append(read[entry][place])
            read_again = dataclasses.replace(evaluated.reads[r], arrays=_joined(parts))
            values = _metrics(
                read_again, every_row, evaluated.probability, options, without
            )
            result[b, r] = np.array([values[name] for name in names], dtype=np.float64)


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
    plan = _plan(evaluated)
    methods, builder = plan.methods, plan.builder
    today = [r for r, method in enumerate(methods) if method == "today"]
    record = _generator_record(rng, "bootstrap")
    draws = rng.integers(
        0, evaluated.n_years, size=(n_bootstrap, evaluated.n_years), dtype=np.int64
    )
    replicates = _replicates(
        evaluated, _year_counts(draws, evaluated.n_years), rows=today
    )
    unit: RecordedUnit = evaluated.unit
    if builder is not None:
        redealt = (
            _redeal_draws(builder, rng, n_bootstrap) if "redeal" in methods else []
        )
        if plan.folds:
            _pooled_rebuilt_replicates(evaluated, plan, redealt, replicates)
        else:
            _rebuilt_replicates(evaluated, builder, methods, draws, redealt, replicates)
        unit = "as_built"
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
    together = [bool(kinds) for kinds in evaluated.together]

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
        (
            pa.field("built_together", pa.bool_()),
            pa.array([flag for _ in columns for flag in together], pa.bool_()),
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
                "unit": unit,
                "built_together": [
                    kind
                    for kind in _BUILT_TOGETHER
                    if any(kind in kinds for kinds in evaluated.together)
                ],
                # Left out when every row was built as its block built it, so a
                # call with nothing kept writes the document it wrote before.
                **(
                    {"kept_today": _kept_record(evaluated, plan.kept)}
                    if plan.kept
                    else {}
                ),
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
      designs, `JointDesignResult`s, `ClusteredDesignResult`s, mappings of
      designs or packages of designs, each made on its fold's design block;
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

    A design under `Clustered` gives, as in `compare`, its `"cluster"` rows
    under `joint=False`, with `cluster_id`, and its `"total"` row; each pooled
    row reads every fold's design of that cluster or total. `allocation` and
    `values` are those of `compare`. The table's metadata
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

    Rows built together are built again on each replicate, as the block built
    them (the module docstring has the rules). A row that reads two columns or
    more of an `EmpiricalCopula` block is re-dealt: after the call above, for
    each column of the block in order and each member of a window in order, one
    call `rng.integers(0, n_m, size=(n_bootstrap, n_m))` redraws the member's
    n_m draws, which are dealt to the reference years again by the copula's
    rule. A row that reads two columns or more of a block that `Comonotonic()`
    or `GaussianCopula` moved is re-dealt the same way, with the same layout of
    the draws, by the model's ranks: the identity of ranks, or the ranks of the
    latent normal scores the sample keeps; on folds, over the joined held-out
    blocks. A row that reads a column built with `HazardNoise(...,
    hold="hazard_distribution")` is re-ranked over the years drawn by the call
    above. Every other row reads that call as before. `setup.intervals.unit` is
    `"as_built"` when a row was rebuilt, and `"row"` (an `EmpiricalCopula`
    block) or `"year"` otherwise. A row built together that can't be built
    again, as the block doesn't hold what that needs, reads that call as
    before, and `setup.intervals.kept_today` lists `{"kind", "reason"}` for
    each kind of such rows: `"combined_loss_needs_value"` (a sample without
    damages, with a stated value or a value column), `"caller_rule"`,
    `"table_loss"`, `"combined_loss_not_reproduced"`, `"value_not_found"`,
    `"combined_view"`, `"held_select"`, `"held_loss_not_one"`,
    `"copula_without_k"`, `"index_moved"`, `"gaussian_without_ranks"`,
    `"index_not_comonotone"`, `"several_models"`, `"pooled_by_hand"` or
    `"cut_of_deal"` (one cut of a `Comonotonic()` or `GaussianCopula` sample:
    a holdout's block, one fold's block, a selection). The key is left out
    when there is none.

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
      largest rows, so they understate how far the number could move;
    - `built_together`: true when the wide row reads rows built together: on a
      block whose columns a dependence model moved (`EmpiricalCopula`,
      `Comonotonic()`, `GaussianCopula` or a model of the caller's), a row that
      reads two of its columns or more (an asset or a hazard each; a `"total"`
      row, a joint design's rows, under `Clustered` the rows of a cluster of
      two assets or more, and on a sample of several hazards a row whose index
      and combined loss read two hazards or more), and any row that
      reads a column built with `HazardNoise(...,
      hold="hazard_distribution")`. A pooled row is flagged when a fold's row
      is. `setup.intervals.built_together` lists how the flagged rows were
      built: `"empirical_copula"`, `"comonotonic"`, `"gaussian_copula"`,
      `"dependence_model"` and `"hazard_distribution"`.

    An interval describes how much a number moves with the years drawn. It
    leaves out uncertainty in the hazard data, the impact function and the
    design, and on a curve block it is the Monte Carlo error of the draws and
    says nothing about the curve. Two rows' intervals are not a test of a
    difference between them. On the rows with `built_together` true that a
    model of the caller's moved, and on those kept on today's method, an
    interval can be narrower or wider than the Monte Carlo error of the draws,
    and the setup text says so; it states the method of the rows rebuilt and
    the reason of the rows kept.

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
