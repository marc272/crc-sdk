"""Evaluate covers on an outcome sample and compare them in one table.

- `evaluate(cover, sample, *, split, ...)` returns a one-row comparison
  table: the cover's metrics (`crc_sdk.parametric.metrics`) on the sample.
- `compare(covers, sample, *, split, ...)` returns one row per cover, with
  covers as rows and metrics as columns, the layout of crc-sdk's portfolio
  output.
- `write_comparison(table, path)` writes a comparison table to one Parquet
  file atomically, with its metadata under `crc.parametric.evaluation`.
- `to_long(table)` gives the tidy form, one row per cover and metric.

A cover is a resolved `Cover`, or a design (`DesignedCover`, what `design`
returns), which brings its design sample and its premium.

**A sample of several assets.** Each entry is then a `JointDesignResult` or a
mapping from every asset id of the sample to that asset's cover, or to its
design. Every such evaluation gets a row at the level of the owner's total
(`level="total"`), which holds the metrics of the sum of the assets' losses
against the sum of their payouts, draw by draw; the `scope` column says
`"joint"` or `"individual"`.

- A set of individual covers gets one `"asset"` row per asset and one `"total"`
  row. Each cover is its own contract, so the total's premium is the sum of the
  asset rows' premiums.
- A joint design gets one `"total"` row, with the premium of one contract on
  the total. It gets no `"asset"` row, because the split of the total payout
  across the assets is the solver's choice and is often not identified.

Per-asset rows of a joint design exist only under a stated `allocation`, with
`values=` giving every asset's value: `level="asset_by_value"` shares the total
payout of each draw in proportion to the values, and `level="asset_by_loss"` in
proportion to the assets' losses in that draw (by value in a draw without
loss). An allocation is an accounting statement about the total payout, not
the payout of an asset's cover; a pricing prices the total payout, so these
rows have no premium and no net-loss columns.

**A sample of several hazards** has `index_components`. Its one asset's rows
read the loss of the sample, the combined loss, for every cover: a cover with
a `CompositeIndex` pays on the components of its names, and a cover on one
index on the component named its `hazard_name`, so both kinds share one table.

**In and out of sample.** In sample means the design sample. Out of sample
means a held-out block of years (`holdout`, `year_block_kfold`), fresh Monte
Carlo draws, or any other sample that shares no year with the design sample.
Every row says which, in `split` and `fold`, and the metadata says how the
split was made. An out-of-sample evaluation needs the design sample (a
design's own, or `design_sample=` for a cover given as is) and raises
`LeakageError` when the two share a period of the same source.

**The premium used** is the `premium=` argument if given (`premium_basis`
`"stated"`); otherwise, for a design, the design premium, the contract price
priced on the design sample (`"design"`), so out-of-sample rows use the price
agreed at design; otherwise the premium priced on the evaluation sample with
`pricing` (`"evaluation"`).

Invalid arguments raise `EvaluationError` (a `ValueError`), and arguments of
the wrong type `TypeError`; problems of the cover or the sample raise
`UnresolvedParameterError`, `SampleError` or `LeakageError`.

Everything except `write_comparison` works in memory and does no I/O.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, get_args, runtime_checkable
from uuid import uuid4

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from crc_sdk._version import framework_version, platform_tag, sdk_version
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import (
    EvaluationError,
    SampleError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.joint import JointDesignResult, covers_hash
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    KEY_COLUMNS,
    ColumnEntry,
    CoverEntry,
    EvaluationMetadata,
    EvaluationSetup,
    _identity_columns,
)
from crc_sdk.parametric.metrics import MetricOptions, metric_columns, metric_values
from crc_sdk.parametric.pricing import CostOfCapitalLoad, Pricing, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _check_cover_index,
    _check_out_of_sample,
    _cover_index_values,
    _pool_entry,
    _require_one_asset,
)

__all__ = [
    "Allocation",
    "CoverSet",
    "DesignedCover",
    "EvaluationEntry",
    "PremiumBasis",
    "Split",
    "compare",
    "evaluate",
    "to_long",
    "write_comparison",
]

Split = Literal["in_sample", "out_of_sample"]
"""Whether a row is on the design sample or on a sample held out from it."""

PremiumBasis = Literal["stated", "design", "evaluation"]
"""Which premium a row uses: the stated one, the design's, or one priced on it."""

Allocation = Literal["value", "loss"]
"""How a joint design's total payout is shared out over its assets."""

FloatArray = NDArray[np.float64]

_INT32_MAX = 2**31 - 1

_ALLOCATED_LEVEL: Mapping[Allocation, str] = {
    "value": "asset_by_value",
    "loss": "asset_by_loss",
}
"""The `level` of an allocated row, under each rule."""

_NO_PREMIUM_METRICS = frozenset(
    {"premium", "var_net_loss", "cvar_net_loss", "net_loss_rp"}
)
"""The metrics that need a premium, so an allocated row leaves them null."""


@runtime_checkable
class DesignedCover(Protocol):
    """A designed cover with its record, as `design` returns it.

    `evaluate` and `compare` take one in place of a `Cover`:

    - `cover`: the resolved cover;
    - `design_id`: the design's id (`"sha256:<hex>"`);
    - `premium`: the design premium, the contract price, priced on the
      design sample;
    - `design_sample`: the sample the cover was designed on, for the
      in-sample check and the leakage check out of sample;
    - `unique`, `ties`, `degenerate_tail`: the design's non-uniqueness report
      and degenerate-tail flag, copied into every row;
    - `to_record()`: the design's record as plain JSON types, written to the
      metadata's `designs` with the label of the cover as evaluated
      (`cover_label`); a `"solver"` entry with the solver's `name` and
      `version` is listed in its `solvers`.
    """

    @property
    def cover(self) -> Cover: ...

    @property
    def design_id(self) -> str: ...

    @property
    def premium(self) -> float: ...

    @property
    def design_sample(self) -> OutcomeSample: ...

    @property
    def unique(self) -> bool | None: ...

    @property
    def ties(self) -> int | None: ...

    @property
    def degenerate_tail(self) -> str | None: ...

    def to_record(self) -> Mapping[str, Any]: ...


CoverSet = Mapping[str, Cover | DesignedCover]
"""Covers that each stand on their own: every asset id to its cover or design."""

EvaluationEntry = Cover | DesignedCover | JointDesignResult | CoverSet
"""What `evaluate` and `compare` take: see their docstrings."""


@dataclass(frozen=True)
class _Item:
    """One cover to evaluate, with its design if it has one."""

    cover: Cover
    design: DesignedCover | None
    record: dict[str, Any] | None


@dataclass(frozen=True)
class _CoverSet:
    """Covers that each stand on their own, by asset id."""

    items: Mapping[str, _Item]

    @property
    def covers(self) -> dict[str, Cover]:
        return {asset: item.cover for asset, item in self.items.items()}


@dataclass(frozen=True)
class _JointEntry:
    """A joint design: one total, one cover per asset."""

    design: JointDesignResult
    record: dict[str, Any]


_Entry = _Item | _CoverSet | _JointEntry


@dataclass(frozen=True)
class _Run:
    """What every row of one evaluation shares."""

    sample: OutcomeSample
    split: Split
    fold: int
    pricing: Pricing
    premium: float | None
    options: MetricOptions
    allocation: Allocation | None
    values: Mapping[str, float] | None


# -- argument checks ---------------------------------------------------------


def _split(split: Any) -> Split:
    if split not in get_args(Split):
        raise EvaluationError(f"split must be one of {get_args(Split)}, got {split!r}")
    checked: Split = split
    return checked


def _fold(fold: Any) -> int:
    if (
        isinstance(fold, bool)
        or not isinstance(fold, (int, np.integer))
        or not 0 <= int(fold) <= _INT32_MAX
    ):
        raise EvaluationError(f"fold must be a non-negative integer, got {fold!r}")
    return int(fold)


def _pricing(pricing: Any) -> Pricing:
    if not isinstance(pricing, (PurePremium, CostOfCapitalLoad)):
        raise TypeError(
            "pricing must be PurePremium() or CostOfCapitalLoad(rate, q), "
            f"got {type(pricing).__name__}"
        )
    return pricing


def _options(options: Any) -> MetricOptions:
    if not isinstance(options, MetricOptions):
        raise TypeError(f"options must be MetricOptions, got {type(options).__name__}")
    return options


def _sample(sample: Any, name: str) -> OutcomeSample:
    if not isinstance(sample, OutcomeSample):
        raise TypeError(f"{name} must be an OutcomeSample, got {type(sample).__name__}")
    return sample


def _premium(premium: Any) -> float | None:
    if premium is None:
        return None
    if isinstance(premium, bool) or not isinstance(premium, (int, float, np.number)):
        raise EvaluationError(f"premium must be a number, got {premium!r}")
    value = float(premium)
    if not math.isfinite(value) or value < 0.0:
        raise EvaluationError(
            f"premium must be finite and non-negative, got {premium!r}"
        )
    return value + 0.0


def _design_record(design: DesignedCover) -> dict[str, Any]:
    design_id = design.design_id
    if not isinstance(design_id, str) or not design_id:
        raise EvaluationError(f"a design needs a design_id, got {design_id!r}")
    record = design.to_record()
    if not isinstance(record, Mapping):
        raise TypeError("a design's to_record() must return a mapping")
    record = dict(record)
    if record.setdefault("design_id", design_id) != design_id:
        raise EvaluationError(
            f"the design's record names the design_id {record['design_id']!r}, "
            f"not {design_id!r}"
        )
    premium = design.premium
    if isinstance(premium, bool) or not isinstance(premium, (int, float, np.number)):
        raise EvaluationError(f"the design premium must be a number, got {premium!r}")
    if not math.isfinite(float(premium)) or float(premium) < 0.0:
        raise EvaluationError(
            f"the design premium must be finite and >= 0, got {premium!r}"
        )
    _sample(design.design_sample, "a design's design_sample")
    unique, ties, flag = design.unique, design.ties, design.degenerate_tail
    if unique is not None and not isinstance(unique, bool):
        raise EvaluationError(
            f"a design's unique must be a bool or None, got {unique!r}"
        )
    if ties is not None and (
        isinstance(ties, bool) or not isinstance(ties, (int, np.integer)) or ties < 0
    ):
        raise EvaluationError(f"a design's ties must be an integer >= 0, got {ties!r}")
    if flag is not None and not isinstance(flag, str):
        raise EvaluationError(
            f"a design's degenerate_tail must be a string, got {flag!r}"
        )
    return record


def _item(entry: Any) -> _Item:
    if isinstance(entry, Cover):
        return _Item(entry, None, None)
    if isinstance(entry, DesignedCover):
        cover = entry.cover
        if not isinstance(cover, Cover):
            raise TypeError(
                f"a design's cover must be a Cover, got {type(cover).__name__}"
            )
        return _Item(cover, entry, _design_record(entry))
    raise TypeError(f"expected a Cover or a designed cover, got {type(entry).__name__}")


def _cover_set(entry: Mapping[Any, Any]) -> _CoverSet:
    items: dict[str, _Item] = {}
    for asset, value in entry.items():
        if not isinstance(asset, str):
            raise TypeError(f"a set of covers is keyed by asset id, got {asset!r}")
        items[asset] = _item(value)
    if len({item.design is None for item in items.values()}) > 1:
        raise EvaluationError(
            "a set of covers holds covers or designs, not a mix of the two"
        )
    return _CoverSet(items)


def _entry(entry: Any) -> _Entry:
    if isinstance(entry, JointDesignResult):
        return _JointEntry(entry, entry.to_record())
    if isinstance(entry, Mapping):
        return _cover_set(entry)
    return _item(entry)


def _entries(covers: Any) -> list[_Entry]:
    if isinstance(covers, (str, bytes, Cover)) or not isinstance(covers, Sequence):
        raise TypeError("covers must be a sequence of covers or designs")
    if not covers:
        raise EvaluationError("compare needs at least one cover")
    return [_entry(entry) for entry in covers]


def _allocation(allocation: Any) -> Allocation | None:
    if allocation is None:
        return None
    if allocation not in get_args(Allocation):
        raise EvaluationError(
            f"allocation must be None or one of {get_args(Allocation)}, "
            f"got {allocation!r}"
        )
    checked: Allocation = allocation
    return checked


def _values(
    values: Any, allocation: Allocation | None, sample: OutcomeSample
) -> dict[str, float] | None:
    """Check `values=`; a sample holds no asset values, so a rule needs them."""
    if allocation is None:
        if values is not None:
            raise EvaluationError(
                "values= is for an allocation: it was given without allocation="
            )
        return None
    if values is None:
        raise EvaluationError(
            f"allocation={allocation!r} needs values=, every asset's value in the "
            "loss unit: a sample holds no asset values"
        )
    if not isinstance(values, Mapping):
        raise TypeError(
            f"values must map each asset id to its value, got {type(values).__name__}"
        )
    assets = sample.asset_ids
    if set(values) != set(assets):
        raise EvaluationError(
            f"values= must give every asset of the sample {list(assets)}, and no "
            f"other: it gives {sorted(map(str, values))}"
        )
    checked: dict[str, float] = {}
    for asset in assets:
        value = values[asset]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise EvaluationError(
                f"the value of asset {asset!r} must be a finite number >= 0, "
                f"got {value!r}"
            )
        checked[asset] = float(value) + 0.0
    if not any(checked.values()):
        raise EvaluationError("values= are all zero, so they share nothing out")
    return checked


def _values_hash(values: Mapping[str, float]) -> str:
    """The hash of the (asset id, value) pairs; the values are never written."""
    pairs = sorted(
        ([asset, value] for asset, value in values.items()),
        key=lambda pair: str(pair[0]).encode("utf-8"),
    )
    return content_hash({"asset_values": pairs, "value_column": None})


def _check_cover(cover: Cover, sample: OutcomeSample, asset: str) -> None:
    free = cover.free_parameters()
    if free:
        raise UnresolvedParameterError(
            f"the cover is a template; resolve its free parameters before "
            f"evaluating it: {', '.join(free)}"
        )
    if cover.payout_unit != sample.loss_unit:
        raise SampleError(
            f"the cover pays in {cover.payout_unit!r}, but the sample's loss is "
            f"in {sample.loss_unit!r}"
        )
    _check_cover_index(cover.index_definition, sample)
    if cover.asset_id is not None and cover.asset_id != asset:
        raise SampleError(
            f"the cover is written for asset {cover.asset_id!r}, but the sample "
            f"holds asset {asset!r}"
        )


def _check_own_sample(own: OutcomeSample, given: OutcomeSample | None) -> None:
    if given is not None and own.content_hash != given.content_hash:
        raise EvaluationError(
            "design_sample= differs from the design's own design sample; "
            "leave it out for designed covers"
        )


def _component_view(sample: OutcomeSample, own: OutcomeSample) -> OutcomeSample:
    """`sample` as a design sample read it: that component's view, if it is one.

    A cover designed on `sample.component(name)` has a one-hazard design
    sample, so it is compared with the same component's view of a multi-hazard
    sample. Any other design sample is compared with `sample` itself.
    """
    components = sample.index_components
    if components is None or own.index_components is not None:
        return sample
    for step in reversed(own.provenance.selection):
        if step.get("kind") != "component":
            continue
        chosen = components.get(step["name"])
        if chosen is None or (step["loss"] == "own" and chosen.own_loss is None):
            return sample
        return sample.component(step["name"], loss=step["loss"])
    return sample


def _check_item(
    item: _Item, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    count = len(sample.asset_ids)
    if count > 1:
        shown = ", ".join(sample.asset_ids[:5]) + (", ..." if count > 5 else "")
        raise SampleError(
            f"one cover takes one asset's sample, but this one holds {count} assets "
            f"({shown}): take one with sample.asset(asset_id), give a mapping from "
            "every asset id to its cover, or give a design from design_joint"
        )
    _check_cover(item.cover, sample, sample.asset_ids[0])
    if item.design is None:
        return
    own = item.design.design_sample
    if own.asset_ids != sample.asset_ids:
        raise SampleError(
            "the design sample and the evaluation sample hold different assets"
        )
    _check_own_sample(own, None if given is None else _component_view(given, own))


def _check_set(
    entry: _CoverSet, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    assets = sample.asset_ids
    if len(assets) == 1:
        raise SampleError(
            f"a mapping of covers is for a sample of several assets, but this one "
            f"holds one, {assets[0]!r}: give its cover"
        )
    if set(entry.items) != set(assets):
        raise SampleError(
            f"a set of covers needs one cover for every asset of the sample "
            f"{list(assets)}, and no other: it has {sorted(entry.items)}"
        )
    for asset, item in entry.items.items():
        _check_cover(item.cover, sample, asset)
        if item.design is None:
            continue
        own = item.design.design_sample
        if own.asset_ids != (asset,):
            raise SampleError(
                f"the design for asset {asset!r} was made on a sample of "
                f"{list(own.asset_ids)}: a set of covers takes each asset's own "
                "design"
            )
        _check_own_sample(own, None if given is None else given.asset(asset))


def _check_joint(
    entry: _JointEntry, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    design = entry.design
    if set(design.asset_ids) != set(sample.asset_ids):
        raise SampleError(
            f"the joint design is for the assets {list(design.asset_ids)}, but the "
            f"sample holds {list(sample.asset_ids)}"
        )
    for asset, cover in design.covers.items():
        _check_cover(cover, sample, asset)
    _check_own_sample(design.design_sample, given)


def _check_entry(
    entry: _Entry, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    if isinstance(entry, _Item):
        _check_item(entry, sample, given)
    elif isinstance(entry, _CoverSet):
        _check_set(entry, sample, given)
    else:
        _check_joint(entry, sample, given)


# -- splits ------------------------------------------------------------------


def _split_step(sample: OutcomeSample) -> Mapping[str, Any] | None:
    """The last holdout or k-fold step that made `sample`, if any."""
    step: Mapping[str, Any]
    for step in reversed(sample.provenance.selection):
        if step.get("kind") in ("holdout", "kfold"):
            return step
    return None


def _check_split(
    own: OutcomeSample | None,
    sample: OutcomeSample,
    split: Split,
    design_sample: OutcomeSample | None,
) -> OutcomeSample | None:
    """Check the split for one cover; return its design sample, if known."""
    known = own if own is not None else design_sample
    if split == "in_sample":
        if (
            known is not None
            and known.content_hash != _component_view(sample, known).content_hash
        ):
            raise SampleError(
                "in sample means on the design sample, but the sample differs "
                "from the cover's design sample; evaluate with "
                "split='out_of_sample'"
            )
        return known
    if known is None:
        raise SampleError(
            "an out-of-sample evaluation needs the design sample to check that "
            "the two share no period: pass a design, or design_sample= with "
            "the sample the cover's terms were chosen on"
        )
    _check_out_of_sample(known, sample)
    return known


def _design_pairs(
    entry: _Entry,
    sample: OutcomeSample,
    split: Split,
    given: OutcomeSample | None,
) -> list[tuple[OutcomeSample, OutcomeSample]]:
    """Check the split for one entry; return (design sample, evaluated on) pairs.

    A set of designs holds one design sample per asset, each of that asset
    alone, so each is checked against the asset's view of `sample`, which
    keeps the lineage a leakage check reads.
    """
    if isinstance(entry, _CoverSet) and any(
        item.design is not None for item in entry.items.values()
    ):
        pairs: list[tuple[OutcomeSample, OutcomeSample]] = []
        for asset, item in entry.items.items():
            assert item.design is not None
            view = sample.asset(asset)
            known = _check_split(item.design.design_sample, view, split, None)
            assert known is not None
            pairs.append((known, view))
        return pairs
    own: OutcomeSample | None = None
    if isinstance(entry, _Item) and entry.design is not None:
        own = entry.design.design_sample
    elif isinstance(entry, _JointEntry):
        own = entry.design.design_sample
    known = _check_split(own, sample, split, given)
    return [] if known is None else [(known, sample)]


def _split_document(
    sample: OutcomeSample,
    split: Split,
    fold: int,
    pairs: Sequence[tuple[OutcomeSample, OutcomeSample]],
) -> dict[str, Any]:
    """How the split was made, for the metadata's `setup.evaluations`."""
    design_samples = [design for design, _ in pairs]
    step = _split_step(sample)
    if step is not None and "fold" in step and int(step["fold"]) != fold:
        raise EvaluationError(
            f"the sample is block {step.get('block')!r} of fold {step['fold']} of "
            f"a {step['kind']} split, but fold={fold} was given"
        )
    details: dict[str, Any] = {}
    if split == "in_sample":
        method = "in_sample"
    elif step is not None and step.get("block") == "test":
        method = str(step["kind"])
        details = {key: step[key] for key in ("train_fraction", "k") if key in step}
    elif all(design.lineage != view.lineage for design, view in pairs):
        both_drawn = sample.provenance.source_kind == "monte_carlo" and all(
            design.provenance.source_kind == "monte_carlo" for design in design_samples
        )
        method = "fresh_draws" if both_drawn else "separate_sample"
    else:
        method = "disjoint_periods"
    hashes = sorted({design.content_hash for design in design_samples})
    return {
        "design_samples": hashes,
        "details": details,
        "fold": fold,
        "leakage_check": "passed" if split == "out_of_sample" else "not_applicable",
        "method": method,
        "sample_hash": sample.content_hash,
        "split": split,
    }


# -- rows --------------------------------------------------------------------


def _label(cover: Cover) -> str:
    if cover.label is not None:
        return cover.label
    return cover.definition_hash.removeprefix("sha256:")[:12]


def _shared_label(covers: Mapping[str, Cover]) -> str | None:
    """The label every cover of a set carries, if they all share one."""
    labels = {cover.label for cover in covers.values()}
    return labels.pop() if len(labels) == 1 else None


def _set_label(covers: Mapping[str, Cover], digest: str) -> str:
    shared = _shared_label(covers)
    return digest.removeprefix("sha256:")[:12] if shared is None else shared


def _identity(
    run: _Run,
    *,
    label: str,
    cover_hash: str,
    design: DesignedCover | JointDesignResult | None,
    asset_id: str | None,
    level: str,
    scope: str,
    basis: PremiumBasis | None,
) -> dict[str, Any]:
    """The columns of a row before its metrics."""
    return {
        "cover_label": label,
        "cover_hash": cover_hash,
        "design_id": None if design is None else design.design_id,
        "asset_id": asset_id,
        "level": level,
        "scope": scope,
        "scenario": run.sample.scenario,
        "split": run.split,
        "fold": run.fold,
        "sample_hash": run.sample.content_hash,
        "premium_basis": basis,
        "unique": None if design is None else design.unique,
        "ties": None if design is None or design.ties is None else int(design.ties),
        "degenerate_tail": None if design is None else design.degenerate_tail,
    }


def _asset_row(
    item: _Item, asset: str, run: _Run, *, premium: float | None
) -> dict[str, Any]:
    """The row of one asset's own cover, on that asset's column of the sample."""
    sample = run.sample
    probability = sample.probability
    column = sample.asset_ids.index(asset)
    loss = sample.loss[:, column]
    payout = item.cover.payout_values(
        _cover_index_values(item.cover.index_definition, sample, column)
    )
    basis: PremiumBasis
    if premium is not None:
        basis, used = "stated", premium
    elif item.design is not None:
        basis, used = "design", float(item.design.premium) + 0.0
    else:
        basis, used = "evaluation", run.pricing.premium(payout, probability)
    row = _identity(
        run,
        label=_label(item.cover),
        cover_hash=item.cover.definition_hash,
        design=item.design,
        asset_id=asset,
        level="asset",
        scope="individual",
        basis=basis,
    )
    row.update(
        metric_values(loss, payout, probability, premium=used, options=run.options)
    )
    return row


def _total_payout(covers: Mapping[str, Cover], sample: OutcomeSample) -> FloatArray:
    """The sum over the assets of each draw's payout."""
    total = np.zeros(len(sample), dtype=np.float64)
    for asset, cover in covers.items():
        column = sample.asset_ids.index(asset)
        total += cover.payout_values(
            _cover_index_values(cover.index_definition, sample, column)
        )
    return total


def _set_rows(entry: _CoverSet, run: _Run) -> list[dict[str, Any]]:
    """One row per asset, then the total of the separate contracts."""
    sample = run.sample
    rows = [
        _asset_row(entry.items[asset], asset, run, premium=None)
        for asset in sample.asset_ids
    ]
    basis: PremiumBasis
    if run.premium is not None:
        basis, used = "stated", run.premium
    else:
        basis = rows[0]["premium_basis"]
        used = math.fsum(row["premium"] for row in rows) + 0.0
    covers = entry.covers
    digest = covers_hash(covers)
    total = _identity(
        run,
        label=_set_label(covers, digest),
        cover_hash=digest,
        design=None,
        asset_id=None,
        level="total",
        scope="individual",
        basis=basis,
    )
    total.update(
        metric_values(
            sample.loss.sum(axis=1),
            _total_payout(covers, sample),
            sample.probability,
            premium=used,
            options=run.options,
        )
    )
    return [*rows, total]


def _allocated_payout(
    rule: Allocation,
    payout: FloatArray,
    sample: OutcomeSample,
    values: Mapping[str, float],
    asset: str,
) -> FloatArray:
    """An asset's share of the total payout, draw by draw, under `rule`."""
    share = values[asset] / math.fsum(values.values())
    if rule == "value":
        return payout * share
    column = sample.asset_ids.index(asset)
    total_loss = sample.loss.sum(axis=1)
    shares = np.divide(
        sample.loss[:, column],
        total_loss,
        out=np.full(len(sample), share, dtype=np.float64),
        where=total_loss > 0.0,
    )
    return payout * shares


def _joint_rows(entry: _JointEntry, run: _Run) -> list[dict[str, Any]]:
    """The total of a joint design, then its allocated rows, if a rule is set."""
    design = entry.design
    sample = run.sample
    probability = sample.probability
    covers = design.covers
    label = _set_label(covers, design.covers_hash)
    payout = _total_payout(covers, sample)
    basis: PremiumBasis
    if run.premium is not None:
        basis, used = "stated", run.premium
    else:
        basis, used = "design", float(design.premium) + 0.0
    total = _identity(
        run,
        label=label,
        cover_hash=design.covers_hash,
        design=design,
        asset_id=None,
        level="total",
        scope="joint",
        basis=basis,
    )
    total.update(
        metric_values(
            sample.loss.sum(axis=1),
            payout,
            probability,
            premium=used,
            options=run.options,
        )
    )
    rows = [total]
    if run.allocation is None:
        return rows
    assert run.values is not None
    without = [
        column.name
        for column in metric_columns(run.options)
        if column.metric in _NO_PREMIUM_METRICS
    ]
    for asset in sample.asset_ids:
        row = _identity(
            run,
            label=label,
            cover_hash=design.covers_hash,
            design=design,
            asset_id=asset,
            level=_ALLOCATED_LEVEL[run.allocation],
            scope="joint",
            basis=None,
        )
        row.update(
            metric_values(
                sample.loss[:, sample.asset_ids.index(asset)],
                _allocated_payout(run.allocation, payout, sample, run.values, asset),
                probability,
                premium=0.0,
                options=run.options,
            )
        )
        row.update(dict.fromkeys(without))
        rows.append(row)
    return rows


def _rows(entry: _Entry, run: _Run) -> list[dict[str, Any]]:
    if isinstance(entry, _Item):
        return [_asset_row(entry, run.sample.asset_ids[0], run, premium=run.premium)]
    if isinstance(entry, _CoverSet):
        return _set_rows(entry, run)
    return _joint_rows(entry, run)


def _schema(options: MetricOptions, metadata: EvaluationMetadata) -> Any:
    identity = [
        pa.field("cover_label", pa.string(), nullable=False),
        pa.field("cover_hash", pa.string(), nullable=False),
        pa.field("design_id", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("level", pa.string(), nullable=False),
        pa.field("scope", pa.string(), nullable=False),
        pa.field("scenario", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("fold", pa.int32(), nullable=False),
        pa.field("sample_hash", pa.string(), nullable=False),
        pa.field("premium_basis", pa.string()),
        pa.field("unique", pa.bool_()),
        pa.field("ties", pa.int32()),
        pa.field("degenerate_tail", pa.string()),
    ]
    metrics = [
        pa.field(column.name, pa.float64()) for column in metric_columns(options)
    ]
    return pa.schema(
        identity + metrics,
        metadata={EVALUATION_METADATA_KEY: metadata.to_json()},
    )


# -- metadata ----------------------------------------------------------------


def _parts_entry(provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    """Each asset's loss and basis-risk models, on a sample of several assets."""
    parts = provenance["parts"]
    if parts is None:
        return None
    return {
        asset: {key: part[key] for key in ("basis_risk", "loss_model") if key in part}
        for asset, part in parts.items()
    }


def _sample_entry(sample: OutcomeSample, roles: set[str]) -> dict[str, Any]:
    provenance = sample.provenance.to_document()
    entry: dict[str, Any] = {
        "asset_ids": list(sample.asset_ids),
        "basis_risk": provenance["basis_risk"],
        "dependence": provenance["dependence"],
        "details": provenance["details"],
        "draws": len(sample),
        "index_unit": sample.index_unit,
        "lineage": sample.lineage,
        "loss_model": provenance["loss_model"],
        "loss_unit": sample.loss_unit,
        "notes": provenance["notes"],
        "parent": provenance["parent"],
        "parts": _parts_entry(provenance),
        "periods": [int(sample.period[0]), int(sample.period[-1])],
        "pool": _pool_entry(provenance["pool"]),
        "probability_semantics": provenance["probability_semantics"],
        "roles": sorted(roles),
        "sample_hash": sample.content_hash,
        "scenario": sample.scenario,
        "selection": provenance["selection"],
        "semantics": sample.semantics,
        "semantics_source": provenance["semantics_source"],
        "source_kind": provenance["source_kind"],
        "trend": provenance["trend"],
    }
    if sample.index_components is not None:
        entry["components"] = [
            {
                "name": name,
                "own_loss": component.own_loss is not None,
                "tail": component.tail,
                "unit": component.unit,
            }
            for name, component in sample.index_components.items()
        ]
        entry["combine"] = provenance.get("combine")
        entry["hazard_dependence"] = provenance["hazard_dependence"]
    return entry


def _cover_entry(cover: Cover) -> CoverEntry:
    return CoverEntry(
        label=cover.label, hash=cover.definition_hash, document=cover.to_document()
    )


def _set_entry(covers: Mapping[str, Cover]) -> CoverEntry:
    """The entry of a set of covers: the hash a total row names, and its parts."""
    return CoverEntry(
        label=_shared_label(covers),
        hash=covers_hash(covers),
        document={
            "kind": "cover_set",
            "covers": {asset: cover.definition_hash for asset, cover in covers.items()},
        },
    )


def _cover_entries(entry: _Entry) -> list[CoverEntry]:
    if isinstance(entry, _Item):
        return [_cover_entry(entry.cover)]
    covers = entry.covers if isinstance(entry, _CoverSet) else entry.design.covers
    return [*map(_cover_entry, covers.values()), _set_entry(covers)]


def _records(entry: _Entry) -> list[dict[str, Any]]:
    """The design records of an entry, each with the label as evaluated."""
    if isinstance(entry, _JointEntry):
        return [{**entry.record, "cover_label": _shared_label(entry.design.covers)}]
    items = list(entry.items.values()) if isinstance(entry, _CoverSet) else [entry]
    # The label of the cover as evaluated, so each design keeps its own even
    # when two designs resolve to the same cover.
    return [
        {**item.record, "cover_label": item.cover.label}
        for item in items
        if item.record is not None
    ]


def _metadata(
    entries: Sequence[_Entry],
    sample: OutcomeSample,
    *,
    evaluation: dict[str, Any],
    design_samples: Sequence[OutcomeSample],
    options: MetricOptions,
    values_sha256: str | None,
) -> EvaluationMetadata:
    roles: dict[str, tuple[OutcomeSample, set[str]]] = {
        sample.content_hash: (sample, {"evaluation"})
    }
    for design_sample in design_samples:
        entry = roles.setdefault(design_sample.content_hash, (design_sample, set()))
        entry[1].add("design")
    samples = [_sample_entry(each, kinds) for each, kinds in roles.values()]
    sources: list[dict[str, Any]] = []
    licences: list[dict[str, Any]] = []
    rng: list[dict[str, Any]] = []
    for each, _ in roles.values():
        document = each.provenance.to_document()
        sources += [
            {**source, "sample_hash": each.content_hash}
            for source in document["sources"]
        ]
        licences += document["licences"]
        rng += [
            {**record, "sample_hash": each.content_hash} for record in document["rng"]
        ]
    covers: dict[tuple[str, str | None], CoverEntry] = {}
    records: dict[str, dict[str, Any]] = {}
    for evaluated in entries:
        for cover_entry in _cover_entries(evaluated):
            covers.setdefault((cover_entry.hash, cover_entry.label), cover_entry)
        for record in _records(evaluated):
            records.setdefault(record["design_id"], record)
    solvers: list[dict[str, Any]] = []
    for record in records.values():
        solver = record.get("solver")
        if isinstance(solver, Mapping) and solver not in solvers:
            solvers.append(dict(solver))
    unique_licences: list[dict[str, Any]] = []
    for licence in licences:
        if licence not in unique_licences:
            unique_licences.append(licence)
    return EvaluationMetadata(
        creation_version=sdk_version(),
        crc_framework_version=framework_version(),
        numpy_version=np.__version__,
        platform=platform_tag(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        solvers=tuple(solvers),
        setup=EvaluationSetup(
            basis="annual",
            metric_options=options.to_document(),
            samples=tuple(samples),
            evaluations=(evaluation,),
        ),
        columns={
            column.name: ColumnEntry(**column.to_document(sample.loss_unit))
            for column in metric_columns(options)
        },
        covers=tuple(covers.values()),
        designs=tuple(records.values()),
        sources=tuple(sources),
        assets={
            "count": len(sample.asset_ids),
            "id_column": None,
            "values_sha256": values_sha256,
        },
        licences=tuple(unique_licences),
        rng=tuple(rng),
    )


def _table(
    entries: Sequence[_Entry],
    sample: OutcomeSample,
    *,
    split: Any,
    fold: Any,
    pricing: Any,
    premium: Any,
    options: Any,
    design_sample: Any,
    allocation: Any,
    values: Any,
) -> Any:
    sample = _sample(sample, "sample")
    checked_split = _split(split)
    checked_fold = _fold(fold)
    checked_pricing = _pricing(pricing)
    checked_premium = _premium(premium)
    checked_options = _options(options)
    given = None if design_sample is None else _sample(design_sample, "design_sample")
    rule = _allocation(allocation)
    checked_values = _values(values, rule, sample)
    if rule is not None and not any(isinstance(e, _JointEntry) for e in entries):
        raise EvaluationError(
            "allocation= shares out a joint design's total payout, but no entry is "
            "a joint design"
        )
    if given is not None:
        if len(sample.asset_ids) == 1:
            _require_one_asset(given, "evaluate and compare")
        elif set(given.asset_ids) != set(sample.asset_ids):
            raise SampleError(
                f"design_sample= holds the assets {list(given.asset_ids)}, but the "
                f"sample holds {list(sample.asset_ids)}"
            )
    for entry in entries:
        _check_entry(entry, sample, given)
    pairs = [
        pair
        for entry in entries
        for pair in _design_pairs(entry, sample, checked_split, given)
    ]
    design_samples = {design.content_hash: design for design, _ in pairs}
    evaluation = _split_document(sample, checked_split, checked_fold, pairs)
    evaluation["pricing"] = checked_pricing.to_document()
    run = _Run(
        sample=sample,
        split=checked_split,
        fold=checked_fold,
        pricing=checked_pricing,
        premium=checked_premium,
        options=checked_options,
        allocation=rule,
        values=checked_values,
    )
    rows = [row for entry in entries for row in _rows(entry, run)]
    evaluation["level"] = list(dict.fromkeys(row["level"] for row in rows))
    evaluation["scope"] = list(dict.fromkeys(row["scope"] for row in rows))
    evaluation["allocation"] = rule
    keys = [tuple(row[name] for name in KEY_COLUMNS) for row in rows]
    repeated = sorted({key[0] for key in keys if keys.count(key) > 1})
    if repeated:
        raise EvaluationError(
            f"covers repeat a row key (the same cover and design twice): {repeated}"
        )
    metadata = _metadata(
        entries,
        sample,
        evaluation=evaluation,
        design_samples=tuple(design_samples.values()),
        options=checked_options,
        values_sha256=None if checked_values is None else _values_hash(checked_values),
    )
    return pa.Table.from_pylist(rows, schema=_schema(checked_options, metadata))


# -- the public functions ----------------------------------------------------


def evaluate(
    cover: EvaluationEntry,
    sample: OutcomeSample,
    *,
    split: Split,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    premium: float | None = None,
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
    allocation: Allocation | None = None,
    values: Mapping[str, float] | None = None,
) -> Any:
    """Evaluate one cover on a sample; return a comparison table.

    On a sample of one asset, `cover` is a resolved `Cover` or a design
    (`DesignedCover`), and the table has one row. On a sample of several
    assets, `cover` is a `JointDesignResult`, or a mapping from every asset id
    of the sample to that asset's `Cover` or design (all covers or all
    designs, each designed on that asset's own sample). On a sample of one
    asset with index components, a cover reads its components and the row's
    loss is the combined loss. The rows are then
    those of the module docstring: a mapping gives one `"asset"` row per
    asset and one `"total"` row, and a joint design one `"total"` row and, under
    `allocation`, one allocated row per asset. `split` is `"in_sample"` (on the
    design sample) or `"out_of_sample"` (on a sample that shares no period
    with it), and `fold` the fold of a k-fold split (0 for a holdout and for
    fresh draws). `options` states the metrics' levels and thresholds.

    The premium used is `premium` if given, else a design's premium, else the
    premium priced on `sample` with `pricing`; the row's `premium_basis` says
    which. A joint design's premium is that of one contract on the total. A
    set of covers pays one premium per contract: its asset rows use their
    own premiums, and its `"total"` row their sum, or `premium` if given, which
    states the total and leaves the asset rows as they are.

    `design_sample` is the sample a cover given as is was chosen on. Out of
    sample, a design sample is required (a design brings its own), and the
    evaluation raises `LeakageError` if the two samples share a period of the
    same source; in sample, a known design sample must be `sample` itself.
    With several assets it is a sample of those assets.

    `allocation` (`"value"` or `"loss"`, default none) asks for a joint
    design's per-asset rows and needs `values`, a mapping from every asset id
    to its value in the loss unit. Only the hash of the values is written.

    The table has the identity columns (`cover_label`, `cover_hash`,
    `design_id`, `asset_id`, `level`, `scope`, `scenario`, `split`, `fold`,
    `sample_hash`, `premium_basis`, `unique`, `ties`, `degenerate_tail`)
    and one float64 column per metric, null where the metric is undefined.
    `asset_id` is null on `"total"` rows. `premium_basis`, `premium` and the
    net-loss columns are null on allocated rows. On `"total"` and allocated
    rows, `cover_hash` is the hash of the set of covers and `cover_label` their
    shared label, or else the first 12 hex digits of that hash. Its schema
    metadata holds the `EvaluationMetadata` under
    `crc.parametric.evaluation`.

    Raises `UnresolvedParameterError` for a template, `SampleError` if one
    cover is given for a sample of several assets (take one with
    `sample.asset(asset_id)`), if a set of covers or a joint design doesn't
    hold exactly the sample's assets, if a cover's units or asset differ from
    the sample's or a design sample is missing or wrong, `LeakageError` for a
    shared period, `EvaluationError` (a `ValueError`) for invalid arguments,
    such as an unknown split or allocation, a negative fold or premium, a
    fold that contradicts the sample's k-fold step, an invalid design record,
    a set mixing covers and designs, an allocation without a joint design or
    without valid `values`, or `values` without an allocation, and `TypeError`
    for arguments of the wrong type.
    """
    return _table(
        _entries([cover]),
        sample,
        split=split,
        fold=fold,
        pricing=pricing,
        premium=premium,
        options=options,
        design_sample=design_sample,
        allocation=allocation,
        values=values,
    )


def compare(
    covers: Sequence[EvaluationEntry],
    sample: OutcomeSample,
    *,
    split: Split,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
    allocation: Allocation | None = None,
    values: Mapping[str, float] | None = None,
) -> Any:
    """Evaluate several covers on one sample; return their rows in one table.

    The arguments are those of `evaluate`, without `premium`: a design's
    rows use its design premium, and a cover given as is the premium priced
    on `sample` with `pricing`. `design_sample` is for the covers given as
    is; each design brings its own, and a `design_sample` that differs from
    it raises `EvaluationError`. Each entry of `covers` is one that `evaluate`
    takes; `allocation` applies to the joint designs among them, and raises
    `EvaluationError` if there is none.

    The table has covers as rows and metrics as columns, in the order of
    `covers`. Its key (`cover_hash`, `design_id`, `asset_id`, `level`,
    `scenario`, `split`, `fold`) is unique: the same cover twice raises
    `EvaluationError`. To put rows of several samples, splits or folds in one
    file, concatenate the tables with `pyarrow.concat_tables` and pass
    `EvaluationMetadata.merge` of their metadata to `write_comparison`.

    Raises as `evaluate` does, and `EvaluationError` for an empty `covers`.
    """
    return _table(
        _entries(covers),
        sample,
        split=split,
        fold=fold,
        pricing=pricing,
        premium=None,
        options=options,
        design_sample=design_sample,
        allocation=allocation,
        values=values,
    )


def write_comparison(
    table: Any,
    path: str | os.PathLike[str],
    *,
    metadata: EvaluationMetadata | None = None,
) -> None:
    """Write a comparison table to one local Parquet file, atomically.

    `metadata` defaults to the table's own (`EvaluationMetadata.from_table`);
    pass `EvaluationMetadata.merge(...)` for a concatenation of several
    tables. It is checked against the table (`EvaluationMetadata.check_table`)
    and written under `crc.parametric.evaluation`. The file is written to a
    temporary sibling and moved into place with `os.replace`, so a failed
    write leaves no partial file and any previous file intact.

    Raises `EvaluationError` if the metadata does not describe the table or
    the path is a URL, and `TypeError` for a table that is not a
    `pyarrow.Table` or metadata that is not `EvaluationMetadata`.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(f"table must be a pyarrow.Table, got {type(table).__name__}")
    chosen = EvaluationMetadata.from_table(table) if metadata is None else metadata
    if not isinstance(chosen, EvaluationMetadata):
        raise TypeError(
            f"metadata must be EvaluationMetadata, got {type(chosen).__name__}"
        )
    chosen.check_table(table)
    text = os.fspath(path)
    if not isinstance(text, str) or "://" in text:
        raise EvaluationError(f"write_comparison writes local files, got {path!r}")
    target = Path(text)
    existing = dict(table.schema.metadata or {})
    existing[EVALUATION_METADATA_KEY.encode("utf-8")] = chosen.to_json().encode("utf-8")
    output = table.replace_schema_metadata(existing)
    staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        pq.write_table(output, staged)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def to_long(table: Any) -> Any:
    """Return the tidy form of a comparison table: one row per metric.

    Each row holds the identity columns of the wide row, then `column` (the
    wide column's name), `metric`, `parameter` and `parameter_value` (for
    example `"beta"` and 0.99; null for metrics without one), `unit` and
    `value`. Rows are ordered by metric column, then by the wide rows'
    order. The table's metadata is kept.

    Raises `EvaluationError` for a table without comparison metadata or
    with metadata that does not describe it, and `TypeError` for a table
    that is not a `pyarrow.Table`.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(f"table must be a pyarrow.Table, got {type(table).__name__}")
    metadata = EvaluationMetadata.from_table(table)
    metadata.check_table(table)
    columns = _identity_columns(table.column_names)
    identity = table.select(columns)
    n = table.num_rows
    parts = []
    for name in table.column_names:
        if name in columns:
            continue
        entry = metadata.columns[name]
        parameter: str | None = None
        value: float | None = None
        if entry.parameter:
            parameter, value = next(iter(entry.parameter.items()))
        parts.append(
            identity.append_column("column", pa.array([name] * n, pa.string()))
            .append_column("metric", pa.array([entry.metric] * n, pa.string()))
            .append_column("parameter", pa.array([parameter] * n, pa.string()))
            .append_column("parameter_value", pa.array([value] * n, pa.float64()))
            .append_column("unit", pa.array([entry.unit] * n, pa.string()))
            .append_column("value", table.column(name).cast(pa.float64()))
        )
    long = pa.concat_tables(parts)
    return long.replace_schema_metadata(table.schema.metadata)
