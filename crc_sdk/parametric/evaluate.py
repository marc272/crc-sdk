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
The column `components` says what each row's cover reads, and `design_budget`
the design's budget bound; a row of a cover given as is has none.

**A package of separate contracts** (`CoverPackage`) sets one-index covers,
each designed on its own, beside a cover on several components. It gets one
`"asset"` row per contract and one `"package"` row, with the metrics of the
year's total payout against the combined loss and the sum of the contracts'
premiums. `split_budget` splits one budget between the contracts, by stated
shares. Each row is a measurement of a stated setup, and `compare` ranks nothing.

**What a row read.** `scenario` is the caller's label of the sample the row is
evaluated on, and `design_scenario` that of the design's own design sample, null
for a cover given as is. `pathway`, `horizon` and `product` come from that
sample's source records, never from its label, and are null where they state
none. `n_rows` is the number of rows of the sample. Nothing in these columns ranks
a pathway, a horizon or a product.

**In and out of sample.** In sample means the design sample. Out of sample
means a held-out block of years (`holdout`, `year_block_kfold`), fresh Monte
Carlo draws, or any other sample that shares no year with the design sample.
Every row says which, in `split` and `fold`, and the metadata says how the
split was made. An out-of-sample evaluation needs the design sample (a
design's own, or `design_sample=` for a cover given as is) and raises
`LeakageError` when the two share a period of the same source, or when two
Monte Carlo samples of any curves reuse one generator state on a common period.

**The premium used** is the `premium=` argument if given (`premium_basis`
`"stated"`); otherwise, for a design, the design premium, the contract price
priced on the design sample (`"design"`), so out-of-sample rows use the price
agreed at design; otherwise the premium priced on the evaluation sample with
`pricing` (`"evaluation"`). `compare_scenarios` prices a cover given as is once,
on the reference's design block (`"reference"`).

**Amounts no design year reached.** `unidentified_share` is the probability of
the evaluation block's draws whose payout reads an amount the design's fill rule
set: any non-zero basis entry of such an amount counts, in full. On the design
block it is 0 by construction, and above 0 only on a block that reaches where
the design had no year. On a `"total"`, `"package"` or allocated row it is the
union over the contracts, and it is null if any of them has no report: for a
cover given as is, a design record without the key and a design on the numpy
grid, where no fill rule ran.

**Rank blocks.** Inside one reference year's block an `EmpiricalCopula` treats
the columns as independent. When `(1 - beta) * m <= 1`, for the reference's m
years, the tail at beta holds at most one such block, so a tail metric of a sum
over the reordered columns lies inside that assumption. Beside the identity
columns each row records `copula_reference_years`, m, and one boolean
`copula_block_tail_b<beta>` per beta of the metric options, with the tolerance of
`degenerate_tail` (1e-9). They are filled on the `"total"` and allocated rows of a
sample of several assets under that model, and on every row of a sample of
several hazards under it, where each row reads the combined loss. Any other row
reads null there: one asset's draws are the same under every model, and other
dependence models, per-year samples and a `component(name)` view have no rank
blocks. A flag says where a number lies inside an assumption, nothing more.

**The breakdown.** `payout_breakdown(covers, sample, *, split, ...)` returns a
separate table with one row per entry and part: the part's expected payout, its
share of the entry's and how often it pays. The rows of one entry add up to its
E[payout]. A `MaxOf`'s rows are an attribution, because one contract pays one
amount, and a `StepGrid`'s are regions, because a grid has no payout per hazard.

Invalid arguments raise `EvaluationError` (a `ValueError`), and arguments of
the wrong type `TypeError`; problems of the cover or the sample raise
`UnresolvedParameterError`, `SampleError` or `LeakageError`.

Everything except `write_comparison` works in memory and does no I/O.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, get_args, runtime_checkable
from uuid import uuid4

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from crc_sdk._version import framework_version, platform_tag, sdk_version
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric._pairing import _check_reads
from crc_sdk.parametric.constraints import Budget
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import (
    EvaluationError,
    SampleError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.index import CompositeIndex, _component_name
from crc_sdk.parametric.joint import JointDesignResult, covers_hash
from crc_sdk.parametric.metadata import (
    _ADDED_METRIC_COLUMNS,
    _ADDED_SCENARIO_COLUMNS,
    _BLOCK_TAIL_PREFIX,
    _REFERENCE_YEARS_COLUMN,
    EVALUATION_METADATA_KEY,
    IDENTITY_COLUMNS,
    KEY_COLUMNS,
    ColumnEntry,
    CoverEntry,
    EvaluationMetadata,
    EvaluationSetup,
    _identity_columns,
    _reference_years,
    _tail_in_one_block,
)
from crc_sdk.parametric.metrics import (
    MetricColumn,
    MetricOptions,
    breakdown_columns,
    breakdown_values,
    metric_columns,
    metric_values,
)
from crc_sdk.parametric.metrics import (
    _label as _metric_label,
)
from crc_sdk.parametric.payouts import LinearInAmounts, MaxOf, StepGrid, SumOf
from crc_sdk.parametric.pricing import CostOfCapitalLoad, Pricing, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _check_out_of_sample,
    _cover_index_values,
    _is_labelled,
    _pool_entry,
    _reads_the_curves_of,
    _require_one_asset,
    _share_a_source,
    _source_columns,
    _SourceColumns,
)

__all__ = [
    "Allocation",
    "CoverPackage",
    "CoverSet",
    "DesignedCover",
    "EvaluationEntry",
    "PremiumBasis",
    "Split",
    "compare",
    "evaluate",
    "payout_breakdown",
    "split_budget",
    "to_long",
    "write_comparison",
]

Split = Literal["in_sample", "out_of_sample"]
"""Whether a row is on the design sample or on a sample held out from it."""

PremiumBasis = Literal["stated", "design", "evaluation", "reference"]
"""Which premium a row uses: stated, the design's, priced on it or on the reference."""

Allocation = Literal["value", "loss"]
"""How a joint design's total payout is shared out over its assets."""

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

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

_PARTS_TOLERANCE = 1e-9
"""How far the parts of a payout may be from it, relative to its largest year."""

_SHARES_TOLERANCE = 1e-9
"""How far the stated shares of a budget may sum from 1."""

_PART_KINDS: Mapping[type[Any], str] = {
    SumOf: "component",
    MaxOf: "attribution",
    StepGrid: "region",
}
"""The `part_kind` of each built-in family with a `breakdown`."""


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
      `version` is listed in its `solvers`; an `"unidentified"` entry, the
      names of the amounts its fill rule set, feeds `unidentified_share`,
      which is null without one or when `"amounts"` is `"grid"`.
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
    read: _SourceColumns
    price_on: OutcomeSample | None = None
    """Where a cover given as is is priced, if not on the sample evaluated."""


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


@dataclass(frozen=True, eq=False, init=False)
class CoverPackage:
    """Separate contracts on one asset's hazards, as one row of a comparison.

    `covers` maps each component name to the one-index cover or the design of
    the contract on that component: two or more, all covers or all designs,
    with one `payout_unit` and one `asset_id` (or none). Each key is the
    component the contract reads, so it need not equal the cover's
    `hazard_name`. `label` is a display name.

    A package holds contracts that each keep their own terms, record and
    premium:

    - `covers`: the covers, by component name;
    - `index`: the `CompositeIndex` of their indices;
    - `payout`: the `SumOf` of the covers as designed, every amount fixed;
    - `as_cover()`: the same payout as one contract;
    - `package_hash`: `covers_hash` of `covers`, so it cannot collide with the
      `definition_hash` of a one-contract `SumOf` of the same terms.

    `evaluate` and `compare` take a package as an entry and give one row per
    contract and one `"package"` row: the premium there is the sum of the
    contracts', each on its own terms, so a loading applies per contract. Each
    cover keeps its own limit, and the package has none of its own.

    Raises `EvaluationError` for fewer than two covers, a mix of covers and
    designs, a cover on several components, or different units or assets;
    `UnresolvedParameterError` for a template; `TypeError` for arguments of
    the wrong type.
    """

    label: str | None
    index: CompositeIndex
    payout: SumOf
    package_hash: str
    _items: Mapping[str, _Item] = field(repr=False)

    def __init__(
        self, covers: Mapping[str, Cover | DesignedCover], *, label: str | None = None
    ) -> None:
        if not isinstance(covers, Mapping):
            raise TypeError(
                "covers must map each component name to its cover or design, got "
                f"{type(covers).__name__}"
            )
        if len(covers) < 2:
            raise EvaluationError(
                f"a CoverPackage holds two covers or more, got {len(covers)}"
            )
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise EvaluationError(f"label must be a non-empty string, got {label!r}")
        items: dict[str, _Item] = {}
        for name, value in covers.items():
            items[_component_name(name, EvaluationError)] = _item(value)
        if len({item.design is None for item in items.values()}) > 1:
            raise EvaluationError(
                "a CoverPackage holds covers or designs, not a mix of the two"
            )
        for name, item in items.items():
            if isinstance(item.cover.index_definition, CompositeIndex):
                raise EvaluationError(
                    f"a CoverPackage holds covers on one index, but the cover of "
                    f"{name!r} pays on several components"
                )
            free = item.cover.free_parameters()
            if free:
                raise UnresolvedParameterError(
                    f"the cover of {name!r} is a template; resolve its free "
                    f"parameters first: {', '.join(free)}"
                )
        resolved = {name: item.cover for name, item in items.items()}
        for what, seen in (
            ("payout_unit", {cover.payout_unit for cover in resolved.values()}),
            ("asset_id", {cover.asset_id for cover in resolved.values()}),
        ):
            if len(seen) > 1:
                raise EvaluationError(
                    f"the covers of a CoverPackage share one {what}, got "
                    f"{sorted(map(repr, seen))}"
                )
        index = CompositeIndex({name: cover.index for name, cover in resolved.items()})
        payout = SumOf({name: cover.payout for name, cover in resolved.items()})
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "payout", payout)
        object.__setattr__(self, "package_hash", covers_hash(resolved))
        object.__setattr__(self, "_items", items)

    @property
    def covers(self) -> Mapping[str, Cover]:
        """The covers, by component name, in order."""
        return MappingProxyType(
            {name: item.cover for name, item in self._items.items()}
        )

    def as_cover(self) -> Cover:
        """Return the same payout as one contract: a `Cover` with the `SumOf`.

        It has one document and one premium, so under a loading it is priced
        on the sum and not per contract.
        """
        first = next(iter(self._items.values())).cover
        return Cover(
            self.index,
            self.payout,
            payout_unit=first.payout_unit,
            asset_id=first.asset_id,
            label=self.label,
        )

    @property
    def _designed(self) -> bool:
        return all(item.design is not None for item in self._items.values())


EvaluationEntry = Cover | DesignedCover | JointDesignResult | CoverSet | CoverPackage
"""What `evaluate` and `compare` take: see their docstrings."""

_Entry = _Item | _CoverSet | _JointEntry | CoverPackage


def _entry(entry: Any) -> _Entry:
    if isinstance(entry, CoverPackage):
        return entry
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


def _check_cover(
    cover: Cover, sample: OutcomeSample, asset: str, *, read_index: bool = True
) -> None:
    """Raise unless `cover` can be evaluated on `sample` for `asset`.

    A package's contracts read their components by key, so `_check_package`
    checks the index and passes `read_index=False`.
    """
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
    if read_index:
        _check_reads(cover.index_definition, sample, asset)
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
    _check_item_design(item, sample, given)


def _check_item_design(
    item: _Item, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    if item.design is None:
        return
    own = item.design.design_sample
    if own.asset_ids != sample.asset_ids:
        raise SampleError(
            "the design sample and the evaluation sample hold different assets"
        )
    _check_own_sample(own, None if given is None else _component_view(given, own))


def _check_package(
    entry: CoverPackage, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    asset = sample.asset_ids[0]
    _check_reads(entry.index, sample, asset)
    for item in entry._items.values():
        _check_cover(item.cover, sample, asset, read_index=False)
        _check_item_design(item, sample, given)


def _check_set(
    entry: _CoverSet, sample: OutcomeSample, given: OutcomeSample | None
) -> None:
    assets = sample.asset_ids
    if len(assets) == 1:
        hint = (
            "give a CoverPackage for separate contracts on its hazards"
            if sample.index_components is not None
            else "give its cover"
        )
        raise SampleError(
            f"a mapping of covers is for a sample of several assets, but this one "
            f"holds one, {assets[0]!r}: {hint}"
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
    elif isinstance(entry, CoverPackage):
        _check_package(entry, sample, given)
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
    if isinstance(entry, CoverPackage) and entry._designed:
        return _package_pairs(entry, sample, split)
    own: OutcomeSample | None = None
    if isinstance(entry, _Item) and entry.design is not None:
        own = entry.design.design_sample
    elif isinstance(entry, _JointEntry):
        own = entry.design.design_sample
    known = _check_split(own, sample, split, given)
    return [] if known is None else [(known, sample)]


def _package_pairs(
    entry: CoverPackage, sample: OutcomeSample, split: Split
) -> list[tuple[OutcomeSample, OutcomeSample]]:
    """Check each contract's design sample against its view of `sample`."""
    pairs: list[tuple[OutcomeSample, OutcomeSample]] = []
    for item in entry._items.values():
        assert item.design is not None
        own = item.design.design_sample
        view = _component_view(sample, own)
        known = _check_split(own, view, split, None)
        assert known is not None
        pairs.append((known, view))
    return pairs


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
    elif not any(_share_a_source(design, view) for design, view in pairs):
        # A source is compared by its own lineage, whichever samples hold it, so
        # years of one series seen through another kind of sample are of one source.
        both_drawn = sample.provenance.source_kind == "monte_carlo" and all(
            design.provenance.source_kind == "monte_carlo"
            and _reads_the_curves_of(design, view)
            for design, view in pairs
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
    components: str | None = None,
    budget: float | None = None,
    design_scenario: str | None = None,
) -> dict[str, Any]:
    """The columns of a row before its metrics.

    `design_scenario` is for a row with no design of its own, the sum of
    separate contracts; a row with a design takes its design sample's label.
    """
    return {
        "cover_label": label,
        "cover_hash": cover_hash,
        "design_id": None if design is None else design.design_id,
        "asset_id": asset_id,
        "level": level,
        "scope": scope,
        "scenario": run.sample.scenario,
        "design_scenario": (
            design_scenario if design is None else design.design_sample.scenario
        ),
        "pathway": run.read.pathway,
        "horizon": run.read.horizon,
        "product": run.read.product,
        "split": run.split,
        "fold": run.fold,
        "sample_hash": run.sample.content_hash,
        "n_rows": len(run.sample),
        "premium_basis": basis,
        "unique": None if design is None else design.unique,
        "ties": None if design is None or design.ties is None else int(design.ties),
        "degenerate_tail": None if design is None else design.degenerate_tail,
        "components": components,
        "design_budget": budget,
    }


def _shared_design_scenario(items: Iterable[_Item]) -> str | None:
    """The label the design samples of separate contracts share, if they do.

    Null for covers given as is, and for designs made on different labels.
    """
    labels = {
        None if item.design is None else item.design.design_sample.scenario
        for item in items
    }
    return labels.pop() if len(labels) == 1 else None


def _reads(cover: Cover, sample: OutcomeSample) -> str | None:
    """What `cover` reads of a sample with index components, joined by `+`."""
    if sample.index_components is None:
        return None
    index = cover.index_definition
    return (
        "+".join(index.names)
        if isinstance(index, CompositeIndex)
        else (index.hazard_name)
    )


def _budget_bound(record: Mapping[str, Any] | None) -> float | None:
    """A design's budget bound in payout units; `None` if it has no budget."""
    bound = None if record is None else record.get("budget_bound")
    if bound is None:
        return None
    if (
        isinstance(bound, bool)
        or not isinstance(bound, (int, float, np.number))
        or not math.isfinite(float(bound))
    ):
        raise EvaluationError(
            f"a design's budget_bound must be a number, got {bound!r}"
        )
    return float(bound) + 0.0


def _summed_budget(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """The sum of the rows' budget bounds; `None` if any row has none."""
    bounds = [row["design_budget"] for row in rows]
    return None if None in bounds else math.fsum(bounds) + 0.0


def _component_values(sample: OutcomeSample, name: str) -> FloatArray:
    """The index values of a component of `sample`, `(n,)`."""
    components = sample.index_components
    assert components is not None
    values: FloatArray = components[name].values[:, 0]
    return values


def _filled_amounts(
    record: Mapping[str, Any] | None, asset: str | None = None
) -> list[str] | None:
    """The amounts a design's fill rule set; `None` if no fill rule is on record.

    A cover given as is has no record, and on the numpy grid (`amounts` is
    `"grid"`) no fill rule ran. A joint design's record lists them by asset, so
    it takes `asset`.
    """
    if (
        record is None
        or "unidentified" not in record
        or record.get("amounts") == "grid"
    ):
        return None
    names = record["unidentified"]
    if asset is not None and isinstance(names, Mapping):
        names = names.get(asset)
    if (
        isinstance(names, (str, bytes))
        or not isinstance(names, Sequence)
        or not all(isinstance(name, str) for name in names)
    ):
        raise EvaluationError(
            f"a design's unidentified must list the names of amounts, got {names!r}"
        )
    return list(names)


def _amount_columns(
    payout: Any, index: FloatArray
) -> tuple[tuple[str, ...], FloatArray]:
    """A resolved payout's amounts and their basis columns on `index`.

    A resolved `MaxOf` has a fixed scale, so it lists no amounts. The scale is
    the one amount a design can have set there, and its column is the payout at
    a scale of 1.
    """
    if isinstance(payout, MaxOf):
        largest = MaxOf(payout.components, scale=1.0).evaluate(index)
        return ("scale",), largest.reshape(-1, 1)
    if not isinstance(payout, LinearInAmounts):
        return (), np.zeros((index.shape[0], 0), dtype=np.float64)
    names = tuple(payout.amount_names())
    return names, np.asarray(payout.basis(index), dtype=np.float64)


def _reads_filled(
    cover: Cover, names: Sequence[str], index: FloatArray
) -> BoolArray | None:
    """Which draws of `index` read one of `names`: any non-zero basis entry.

    `None` if the cover has no column for one of `names`, so that a record of
    a user's design that names amounts in its own way reads as null.
    """
    if not names:
        return np.zeros(index.shape[0], dtype=np.bool_)
    amounts, basis = _amount_columns(cover.payout, index)
    if basis.shape != (index.shape[0], len(amounts)) or not np.isfinite(basis).all():
        raise EvaluationError(
            f"the basis of payout {cover.payout.kind!r} must be finite, with one "
            f"row per draw and one column per amount; got shape {basis.shape}"
        )
    if not set(names) <= set(amounts):
        return None
    columns = [amounts.index(name) for name in names]
    reads: BoolArray = (basis[:, columns] != 0.0).any(axis=1)
    return reads


def _item_reads(item: _Item, index: FloatArray) -> BoolArray | None:
    """Which draws of a cover's `index` read a filled amount, if that is known."""
    names = _filled_amounts(item.record)
    return None if names is None else _reads_filled(item.cover, names, index)


def _union_reads(parts: Iterable[BoolArray | None]) -> BoolArray | None:
    """The draws in which any contract reads a filled amount, if all are known."""
    listed = list(parts)
    known = [part for part in listed if part is not None]
    return None if len(known) < len(listed) else np.logical_or.reduce(known)


def _item_identity(
    item: _Item,
    asset: str,
    run: _Run,
    *,
    basis: PremiumBasis,
    component: str | None,
) -> dict[str, Any]:
    """The columns before the metrics of one cover's row."""
    return _identity(
        run,
        label=_label(item.cover),
        cover_hash=item.cover.definition_hash,
        design=item.design,
        asset_id=asset,
        level="asset",
        scope="individual",
        basis=basis,
        components=component or _reads(item.cover, run.sample),
        budget=_budget_bound(item.record),
    )


def _reference_price(
    item: _Item, asset: str, run: _Run, component: str | None
) -> float:
    """`run.pricing` on the payouts of `item` over the block a price is made on."""
    block = run.price_on
    assert block is not None
    index = (
        _cover_index_values(
            item.cover.index_definition, block, block.asset_ids.index(asset)
        )
        if component is None
        else _component_values(block, component)
    )
    return run.pricing.premium(item.cover.payout_values(index), block.probability)


def _asset_row(
    item: _Item,
    asset: str,
    run: _Run,
    *,
    premium: float | None,
    component: str | None = None,
) -> dict[str, Any]:
    """The row of one asset's own cover, on that asset's column of the sample.

    `component` names what a package's contract reads, its key in the package.
    """
    sample = run.sample
    probability = sample.probability
    column = sample.asset_ids.index(asset)
    loss = sample.loss[:, column]
    index = (
        _cover_index_values(item.cover.index_definition, sample, column)
        if component is None
        else _component_values(sample, component)
    )
    payout = item.cover.payout_values(index)
    basis: PremiumBasis
    if premium is not None:
        basis, used = "stated", premium
    elif item.design is not None:
        basis, used = "design", float(item.design.premium) + 0.0
    elif run.price_on is not None:
        basis, used = "reference", _reference_price(item, asset, run, component)
    else:
        basis, used = "evaluation", run.pricing.premium(payout, probability)
    row = _item_identity(item, asset, run, basis=basis, component=component)
    row.update(
        metric_values(
            loss,
            payout,
            probability,
            premium=used,
            options=run.options,
            unidentified=_item_reads(item, index),
        )
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
        budget=_summed_budget(rows),
        design_scenario=_shared_design_scenario(entry.items.values()),
    )
    total.update(
        metric_values(
            sample.loss.sum(axis=1),
            _total_payout(covers, sample),
            sample.probability,
            premium=used,
            options=run.options,
            unidentified=_union_reads(
                _item_reads(
                    item,
                    _cover_index_values(
                        item.cover.index_definition,
                        sample,
                        sample.asset_ids.index(asset),
                    ),
                )
                for asset, item in entry.items.items()
            ),
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


def _joint_reads(entry: _JointEntry, sample: OutcomeSample) -> BoolArray | None:
    """The draws in which any asset's payout reads a filled amount."""
    parts: list[BoolArray | None] = []
    for asset, cover in entry.design.covers.items():
        names = _filled_amounts(entry.record, asset)
        column = sample.asset_ids.index(asset)
        parts.append(
            None
            if names is None
            else _reads_filled(
                cover,
                names,
                _cover_index_values(cover.index_definition, sample, column),
            )
        )
    return _union_reads(parts)


def _joint_rows(entry: _JointEntry, run: _Run) -> list[dict[str, Any]]:
    """The total of a joint design, then its allocated rows, if a rule is set."""
    design = entry.design
    sample = run.sample
    probability = sample.probability
    covers = design.covers
    label = _set_label(covers, design.covers_hash)
    payout = _total_payout(covers, sample)
    filled = _joint_reads(entry, sample)
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
        budget=_budget_bound(entry.record),
    )
    total.update(
        metric_values(
            sample.loss.sum(axis=1),
            payout,
            probability,
            premium=used,
            options=run.options,
            unidentified=filled,
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
                unidentified=filled,
            )
        )
        row.update(dict.fromkeys(without))
        rows.append(row)
    return rows


def _package_label(entry: CoverPackage) -> str:
    if entry.label is not None:
        return entry.label
    return entry.package_hash.removeprefix("sha256:")[:12]


def _package_identity(
    entry: CoverPackage,
    run: _Run,
    *,
    basis: PremiumBasis,
    budget: float | None,
) -> dict[str, Any]:
    """The columns before the metrics of a package's row."""
    return _identity(
        run,
        label=_package_label(entry),
        cover_hash=entry.package_hash,
        design=None,
        asset_id=run.sample.asset_ids[0],
        level="package",
        scope="individual",
        basis=basis,
        components="+".join(entry.index.names),
        budget=budget,
        design_scenario=_shared_design_scenario(entry._items.values()),
    )


def _package_rows(entry: CoverPackage, run: _Run) -> list[dict[str, Any]]:
    """One row per contract, then the package: their sum on the combined loss."""
    sample = run.sample
    asset = sample.asset_ids[0]
    rows = [
        _asset_row(item, asset, run, premium=None, component=name)
        for name, item in entry._items.items()
    ]
    basis: PremiumBasis
    if run.premium is not None:
        basis, used = "stated", run.premium
    else:
        basis = rows[0]["premium_basis"]
        used = math.fsum(row["premium"] for row in rows) + 0.0
    package = _package_identity(entry, run, basis=basis, budget=_summed_budget(rows))
    package.update(
        metric_values(
            sample.loss[:, 0],
            entry.as_cover().payout_values(_cover_index_values(entry.index, sample)),
            sample.probability,
            premium=used,
            options=run.options,
            unidentified=_union_reads(
                _item_reads(item, _component_values(sample, name))
                for name, item in entry._items.items()
            ),
        )
    )
    return [*rows, package]


def _entry_rows(entry: _Entry, run: _Run) -> list[dict[str, Any]]:
    if isinstance(entry, _Item):
        return [_asset_row(entry, run.sample.asset_ids[0], run, premium=run.premium)]
    if isinstance(entry, _CoverSet):
        return _set_rows(entry, run)
    if isinstance(entry, CoverPackage):
        return _package_rows(entry, run)
    return _joint_rows(entry, run)


def _block_tail_name(beta: float) -> str:
    """The rank-block flag's column at `beta`, with the metrics' label of beta."""
    return f"{_BLOCK_TAIL_PREFIX}{_metric_label(beta)}"


def _block_values(run: _Run, level: str) -> dict[str, Any]:
    """The rank-block columns of a row at `level`, all null without rank blocks."""
    provenance = run.sample.provenance
    years = _reference_years(
        level,
        dependence=provenance.dependence,
        hazard_dependence=provenance.hazard_dependence,
        components=run.sample.index_components is not None,
    )
    return {
        _REFERENCE_YEARS_COLUMN: years,
        **{
            _block_tail_name(beta): (
                None if years is None else _tail_in_one_block(beta, years)
            )
            for beta in run.options.betas
        },
    }


def _rows(entry: _Entry, run: _Run) -> list[dict[str, Any]]:
    """The rows of one entry, with their rank-block columns."""
    return [
        {**row, **_block_values(run, row["level"])} for row in _entry_rows(entry, run)
    ]


def _block_fields(betas: Iterable[float]) -> list[Any]:
    """The fields of the rank-block columns, for `betas`."""
    return [
        pa.field(_REFERENCE_YEARS_COLUMN, pa.int32()),
        *(pa.field(_block_tail_name(beta), pa.bool_()) for beta in betas),
    ]


def _schema(
    columns: Sequence[MetricColumn],
    metadata: EvaluationMetadata,
    *,
    extra: Sequence[Any] = (),
) -> Any:
    identity = [
        pa.field("cover_label", pa.string(), nullable=False),
        pa.field("cover_hash", pa.string(), nullable=False),
        pa.field("design_id", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("level", pa.string(), nullable=False),
        pa.field("scope", pa.string(), nullable=False),
        pa.field("scenario", pa.string(), nullable=False),
        pa.field("design_scenario", pa.string()),
        pa.field("pathway", pa.string()),
        pa.field("horizon", pa.int32()),
        pa.field("product", pa.string()),
        pa.field("split", pa.string(), nullable=False),
        pa.field("fold", pa.int32(), nullable=False),
        pa.field("sample_hash", pa.string(), nullable=False),
        pa.field("n_rows", pa.int32(), nullable=False),
        pa.field("premium_basis", pa.string()),
        pa.field("unique", pa.bool_()),
        pa.field("ties", pa.int32()),
        pa.field("degenerate_tail", pa.string()),
        pa.field("components", pa.string()),
        pa.field("design_budget", pa.float64()),
    ]
    metrics = [pa.field(column.name, pa.float64()) for column in columns]
    return pa.schema(
        [*identity, *extra, *metrics],
        metadata={EVALUATION_METADATA_KEY: metadata.to_json()},
    )


# -- metadata ----------------------------------------------------------------


def _parts_entry(provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    """Each asset's loss and basis-risk models, on a sample of several assets.

    A model that acts on the hazard adds that asset's two mean ratios, which the
    setup text states. Nothing else of a part's details is kept.
    """
    parts = provenance["parts"]
    if parts is None:
        return None
    entries: dict[str, Any] = {}
    for asset, part in parts.items():
        entry = {key: part[key] for key in ("basis_risk", "loss_model") if key in part}
        summary = part["details"].get("basis_risk")
        if summary and "hazard_mean_ratio" in summary:
            entry["details"] = {"basis_risk": summary}
        entries[asset] = entry
    return entries


def _sample_entry(sample: OutcomeSample, roles: set[str]) -> dict[str, Any]:
    provenance = sample.provenance.to_document()
    read = _source_columns(sample, strict=False)
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
    if read != _SourceColumns():
        entry |= {
            "horizon": read.horizon,
            "pathway": read.pathway,
            "product": read.product,
        }
    window = provenance["details"].get("window")
    if window is not None:
        entry["members"] = window["members"]
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


def _package_entry(entry: CoverPackage) -> CoverEntry:
    """The entry of a package: the hash its row names, and its contracts."""
    return CoverEntry(
        label=entry.label,
        hash=entry.package_hash,
        document={
            "kind": "package",
            "covers": {
                name: cover.definition_hash for name, cover in entry.covers.items()
            },
        },
    )


def _cover_entries(entry: _Entry) -> list[CoverEntry]:
    if isinstance(entry, _Item):
        return [_cover_entry(entry.cover)]
    if isinstance(entry, CoverPackage):
        return [*map(_cover_entry, entry.covers.values()), _package_entry(entry)]
    covers = entry.covers if isinstance(entry, _CoverSet) else entry.design.covers
    return [*map(_cover_entry, covers.values()), _set_entry(covers)]


def _records(entry: _Entry) -> list[dict[str, Any]]:
    """The design records of an entry, each with the label as evaluated."""
    if isinstance(entry, _JointEntry):
        return [{**entry.record, "cover_label": _shared_label(entry.design.covers)}]
    items: list[_Item]
    if isinstance(entry, _CoverSet):
        items = list(entry.items.values())
    elif isinstance(entry, CoverPackage):
        items = list(entry._items.values())
    else:
        items = [entry]
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
    columns: Sequence[MetricColumn],
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
            for column in columns
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


def _prepared(
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
    price_on: OutcomeSample | None = None,
) -> tuple[_Run, dict[str, Any], tuple[OutcomeSample, ...]]:
    """Check an evaluation; return its run, its record and its design samples.

    `price_on` is where a cover given as is is priced, for `compare_scenarios`:
    on the reference's design block and not on the block evaluated.
    """
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
    if price_on is not None:
        # A cover given as is is priced there, so it must read that block too.
        for entry in entries:
            _check_entry(entry, price_on, None)
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
        read=_source_columns(sample, strict=_is_labelled(sample)),
        price_on=price_on,
    )
    return run, evaluation, tuple(design_samples.values())


def _check_keys(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Raise `EvaluationError` if two rows share a key."""
    keys = [tuple(row[name] for name in columns) for row in rows]
    repeated = sorted({key[0] for key in keys if keys.count(key) > 1})
    if repeated:
        raise EvaluationError(
            f"covers repeat a row key (the same cover and design twice): {repeated}"
        )


def _row_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """What the setup text says of a row on a sample of several hazards."""
    return {
        key: row[key]
        for key in (
            "components",
            "cover_hash",
            "cover_label",
            "design_budget",
            "design_id",
            "level",
        )
    }


def _package_record(
    entry: CoverPackage, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Each contract's terms and the package's sums, for the setup text."""
    *contracts, package = rows

    def limit(item: _Item) -> float | None:
        stated = None if item.record is None else item.record.get("limit")
        amount = stated.get("amount") if isinstance(stated, Mapping) else None
        return None if amount is None else float(amount) + 0.0

    return {
        "budget": package["design_budget"],
        "contracts": [
            {
                "budget": row["design_budget"],
                "limit": limit(item),
                "name": name,
                "premium": row["premium"],
            }
            for (name, item), row in zip(entry._items.items(), contracts, strict=True)
        ],
        "cover_hash": entry.package_hash,
        "label": package["cover_label"],
        "premium": package["premium"],
        "premium_basis": package["premium_basis"],
        "premium_sum": math.fsum(row["premium"] for row in contracts) + 0.0,
    }


def _describe_rows(
    evaluation: dict[str, Any],
    run: _Run,
    entries: Sequence[_Entry],
    by_entry: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    """Add to `evaluation` what the setup text says of the rows of `entries`."""
    rows = [row for group in by_entry for row in group]
    evaluation["level"] = list(dict.fromkeys(row["level"] for row in rows))
    evaluation["scope"] = list(dict.fromkeys(row["scope"] for row in rows))
    evaluation["allocation"] = run.allocation
    if run.sample.index_components is not None:
        evaluation["rows"] = [_row_record(row) for row in rows]
    packages = [
        _package_record(entry, group)
        for entry, group in zip(entries, by_entry, strict=True)
        if isinstance(entry, CoverPackage)
    ]
    if packages:
        evaluation["packages"] = packages


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
    run, evaluation, design_samples = _prepared(
        entries,
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
    by_entry = [_rows(entry, run) for entry in entries]
    rows = [row for group in by_entry for row in group]
    _describe_rows(evaluation, run, entries, by_entry)
    _check_keys(rows, KEY_COLUMNS)
    columns = metric_columns(run.options)
    metadata = _metadata(
        entries,
        run.sample,
        evaluation=evaluation,
        design_samples=design_samples,
        options=run.options,
        columns=columns,
        values_sha256=None if run.values is None else _values_hash(run.values),
    )
    schema = _schema(columns, metadata, extra=_block_fields(run.options.betas))
    return pa.Table.from_pylist(rows, schema=schema)


# -- the breakdown -----------------------------------------------------------


def _checked_parts(
    payout: Any, parts: Any, n_draws: int
) -> list[tuple[str, FloatArray]]:
    """A `breakdown`'s columns; raise `EvaluationError` if they are invalid."""
    if not isinstance(parts, Mapping) or not parts:
        raise EvaluationError(
            f"the breakdown of payout {payout.kind!r} must be a non-empty mapping "
            "from a part's name to its payouts"
        )
    checked: list[tuple[str, FloatArray]] = []
    for name, values in parts.items():
        if not isinstance(name, str) or not name:
            raise EvaluationError(
                f"the breakdown of payout {payout.kind!r} names a part {name!r}"
            )
        part = np.asarray(values, dtype=np.float64)
        if (
            part.shape != (n_draws,)
            or not np.isfinite(part).all()
            or (part < 0.0).any()
        ):
            raise EvaluationError(
                f"part {name!r} of payout {payout.kind!r} must be finite and "
                f"non-negative, with one value per draw; got shape {part.shape}"
            )
        checked.append((name, part))
    return checked


def _cover_parts(
    cover: Cover, sample: OutcomeSample
) -> tuple[FloatArray, list[tuple[str, str, FloatArray]]]:
    """A cover's payout and its parts: each one's name, kind and values."""
    index = cover.index_definition
    values = _cover_index_values(index, sample)
    payout_values = cover.payout_values(values)
    if not isinstance(index, CompositeIndex):
        return payout_values, [(index.hazard_name, "component", payout_values)]
    payout = cover.payout
    breakdown = getattr(payout, "breakdown", None)
    if not callable(breakdown):
        return payout_values, [("whole", "whole", payout_values)]
    kind = _PART_KINDS.get(type(payout), "user")
    parts = _checked_parts(payout, breakdown(values), len(sample))
    return payout_values, [(name, kind, part) for name, part in parts]


def _check_parts(
    label: str, payout: FloatArray, parts: Sequence[tuple[str, str, FloatArray]]
) -> None:
    """Raise `EvaluationError` unless the parts add up to the payout, year by year."""
    summed = np.sum([part for _, _, part in parts], axis=0)
    allowed = _PARTS_TOLERANCE * float(payout.max(initial=0.0))
    gap = float(np.abs(summed - payout).max(initial=0.0))
    if gap > allowed:
        raise EvaluationError(
            f"the parts of {label!r} differ from its payout by {gap:.3g} in a year"
        )


def _breakdown_rows(entry: _Entry, run: _Run) -> list[dict[str, Any]]:
    """One row per part of one entry."""
    sample = run.sample
    asset = sample.asset_ids[0]
    identity: dict[str, Any]
    parts: list[tuple[str, str, FloatArray]]
    if isinstance(entry, _Item):
        payout, parts = _cover_parts(entry.cover, sample)
        identity = _item_identity(
            entry,
            asset,
            run,
            basis="evaluation" if entry.design is None else "design",
            component=None,
        )
    elif isinstance(entry, CoverPackage):
        parts = [
            (
                name,
                "component",
                item.cover.payout_values(_component_values(sample, name)),
            )
            for name, item in entry._items.items()
        ]
        payout = entry.as_cover().payout_values(
            _cover_index_values(entry.index, sample)
        )
        identity = _package_identity(
            entry,
            run,
            basis="design" if entry._designed else "evaluation",
            budget=_summed_budget(
                [
                    {"design_budget": _budget_bound(item.record)}
                    for item in entry._items.values()
                ]
            ),
        )
    else:
        raise EvaluationError(
            "payout_breakdown takes covers, designs and packages on one asset's "
            "sample, not a set of covers or a joint design"
        )
    _check_parts(identity["cover_label"], payout, parts)
    probability = sample.probability
    expected = math.fsum((probability * payout).tolist()) + 0.0
    return [
        {
            **identity,
            "part": name,
            "part_kind": kind,
            **breakdown_values(
                part, probability, expected_payout=expected, options=run.options
            ),
        }
        for name, kind, part in parts
    ]


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
    (`DesignedCover`), and the table has one row. With index components, a
    cover reads its components and the row's loss is the combined loss. On a
    sample of several assets, `cover` is a `JointDesignResult`, or a mapping
    from every asset id of the sample to that asset's `Cover` or design (all
    covers or all designs, each designed on that asset's own sample). The rows
    are then those of the module docstring: a mapping gives one `"asset"` row per
    asset and one `"total"` row, and a joint design one `"total"` row and, under
    `allocation`, one allocated row per asset. On a sample with index components,
    `cover` may be a `CoverPackage`, which gives one `"asset"` row per contract
    and one `"package"` row. `split` is `"in_sample"` (on the
    design sample) or `"out_of_sample"` (on a sample that shares no period
    with it), and `fold` the fold of a k-fold split (0 for a holdout and for
    fresh draws). `options` states the metrics' levels and thresholds.

    The premium used is `premium` if given, else a design's premium, else the
    premium priced on `sample` with `pricing`; the row's `premium_basis` says
    which. A joint design's premium is that of one contract on the total. A
    set of covers pays one premium per contract: its asset rows use their
    own premiums, and its `"total"` row their sum, or `premium` if given, which
    states the total and leaves the asset rows as they are. A package does the
    same with its contracts and its `"package"` row.

    `design_sample` is the sample a cover given as is was chosen on. Out of
    sample, a design sample is required (a design brings its own), and the
    evaluation raises `LeakageError` if the two samples share a period of the
    same source; in sample, a known design sample must be `sample` itself.
    With several assets it is a sample of those assets.

    `allocation` (`"value"` or `"loss"`, default none) asks for a joint
    design's per-asset rows and needs `values`, a mapping from every asset id
    to its value in the loss unit. Only the hash of the values is written.

    The table has the identity columns (`cover_label`, `cover_hash`,
    `design_id`, `asset_id`, `level`, `scope`, `scenario`, `design_scenario`,
    `pathway`, `horizon`, `product`, `split`, `fold`, `sample_hash`, `n_rows`,
    `premium_basis`, `unique`, `ties`, `degenerate_tail`, `components`,
    `design_budget`) and one float64 column per metric, null where the metric
    is undefined. `design_scenario` is the label of the design's own design
    sample: null for a cover given as is, and on a `"total"` or `"package"` row
    whose designs don't share one. `pathway`, `horizon` and `product` are what
    the sample's source records state, never its label, and null where they
    state none: the centre of a window of horizons or of a per-year series'
    years, and on a sample of several hazards every provider, joined by `+`.
    `n_rows` is the number of rows of the sample. `components` is what the row's
    cover reads,
    joined by `+`, and null on a sample of one hazard. `design_budget` is the
    design's budget bound in payout units, the sum on a package or a total of
    separate covers, and null for a cover given as is. `payout_frequency` is
    the probability that the cover pays, and `unidentified_share` the
    probability of the draws whose payout reads an amount that the design's
    fill rule set (see the module docstring).
    `asset_id` is null on `"total"` rows. `premium_basis`, `premium` and the
    net-loss columns are null on allocated rows. On `"total"` and allocated
    rows, `cover_hash` is the hash of the set of covers and `cover_label` their
    shared label, or else the first 12 hex digits of that hash; on a
    `"package"` row, `cover_hash` is the `package_hash` and `cover_label` the
    package's label, or else the first 12 hex digits. Its schema metadata holds
    the `EvaluationMetadata` under `crc.parametric.evaluation`.

    Raises `UnresolvedParameterError` for a template, `SampleError` if one
    cover is given for a sample of several assets (take one with
    `sample.asset(asset_id)`), if a set of covers or a joint design doesn't
    hold exactly the sample's assets, if a cover's units or asset differ from
    the sample's or a design sample is missing or wrong, `LeakageError` for a
    shared period, `ScenarioError` on a sample a caller labelled with
    `scenario=` whose records state a pathway `crc_sdk.pathways` doesn't know
    or whose sources disagree on `pathway`, `horizon` or `product`,
    `EvaluationError` (a `ValueError`) for invalid arguments,
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
    `scenario`, `design_scenario`, `split`, `fold`) is unique: the same cover
    twice raises `EvaluationError`. To put rows of several samples, splits or
    folds in one file, concatenate the tables with `pyarrow.concat_tables` and
    pass `EvaluationMetadata.merge` of their metadata to `write_comparison`.

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


def _own_loss_shares(
    components: Mapping[str, Any], probability: FloatArray
) -> dict[str, float]:
    """Each component's share of the summed expected own losses."""
    missing = [name for name, part in components.items() if part.own_loss is None]
    if missing:
        raise SampleError(
            f"shares='own_loss' needs every component's own loss, but {missing} "
            "have none; state the shares as a mapping"
        )
    expected = {
        name: math.fsum((probability * part.own_loss[:, 0]).tolist())
        for name, part in components.items()
    }
    total = math.fsum(expected.values())
    if total <= 0.0:
        raise SampleError(
            "shares='own_loss' divides by the summed expected own losses, which "
            "are zero; state the shares as a mapping"
        )
    return {name: value / total for name, value in expected.items()}


def _stated_shares(
    shares: Mapping[Any, Any], components: Mapping[str, Any]
) -> dict[str, float]:
    """The stated shares; raise `EvaluationError` if they are invalid."""
    unknown = [name for name in shares if name not in components]
    if unknown:
        raise EvaluationError(
            f"shares names {unknown}, which the sample does not hold; it holds "
            f"{list(components)}"
        )
    checked: dict[str, float] = {}
    for name, share in shares.items():
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float, np.number))
            or not math.isfinite(float(share))
            or float(share) < 0.0
        ):
            raise EvaluationError(
                f"the share of {name!r} must be a finite number >= 0, got {share!r}"
            )
        checked[name] = float(share) + 0.0
    total = math.fsum(checked.values())
    if abs(total - 1.0) > _SHARES_TOLERANCE:
        raise EvaluationError(f"the shares must sum to 1, got {total}")
    return checked


def split_budget(
    budget: Budget,
    sample: OutcomeSample,
    *,
    shares: Literal["own_loss"] | Mapping[str, float],
) -> Mapping[str, Budget]:
    """Split one budget between the contracts on a sample's components.

    Returns a `Budget(amount=...)` per component: the bound of `budget` on
    `sample` (for a ratio, of the expected combined loss) times the
    component's share, so the bounds sum to the budget's bound on `sample`.

    `shares` has no default, because how to split a budget between hazards is
    an assumption. `"own_loss"` takes each component's share of the summed
    expected own losses, over every component of the sample. A mapping states
    the shares of the components named, which must sum to 1.

    Raises `SampleError` for a sample without index components and for
    `"own_loss"` on a component with no own loss or on own losses that are all
    zero; `EvaluationError` for invalid shares; `TypeError` for arguments of
    the wrong type.
    """
    if not isinstance(budget, Budget):
        raise TypeError(f"budget must be a Budget, got {type(budget).__name__}")
    checked = _sample(sample, "sample")
    components = checked.index_components
    if components is None:
        raise SampleError(
            "split_budget splits a budget between the components of a sample of "
            "several hazards, and this sample has none"
        )
    probability = checked.probability
    fractions: dict[str, float]
    if isinstance(shares, str):
        if shares != "own_loss":
            raise EvaluationError(
                f"shares must be 'own_loss' or a mapping, got {shares!r}"
            )
        fractions = _own_loss_shares(components, probability)
    elif isinstance(shares, Mapping):
        fractions = _stated_shares(shares, components)
    else:
        raise TypeError(
            "shares must be 'own_loss' or a mapping from component name to share, "
            f"got {type(shares).__name__}"
        )
    bound = budget.bound(math.fsum((probability * checked.loss[:, 0]).tolist()))
    return MappingProxyType(
        {name: Budget(amount=bound * share) for name, share in fractions.items()}
    )


def payout_breakdown(
    covers: Sequence[EvaluationEntry],
    sample: OutcomeSample,
    *,
    split: Split,
    fold: int = 0,
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
) -> Any:
    """Break each entry's payout down into its parts; return one table.

    `covers` is what `compare` takes, on one asset's `sample`: covers, designs
    and packages. The table has one row per entry and part, with the identity
    columns of `compare` and:

    - `part` and `part_kind`: a one-index cover's component (`"component"`);
      each component of a `SumOf` or a package (`"component"`); each component
      of a `MaxOf` (`"attribution"`, each year's payout going to the component
      paying most, in equal shares when several tie, because one contract
      pays one amount); `only:<first>`, `only:<second>` and `both` for a
      `StepGrid` (`"region"`: the years where one index is past its first
      threshold and the other is not, and those where both are, because a
      grid has no payout per hazard); the keys of a user family's
      `breakdown(index)` (`"user"`); or `"whole"` for a user family with none
      (`"whole"`);
    - `expected_payout`, the part's expected payout;
    - `share`, its share of the entry's E[payout], null when that is 0;
    - `payout_frequency`, P(the part pays more than `options.payout_threshold`).

    The rows of one entry add up to its E[payout]: the parts are checked
    against the payout year by year. The table holds an `EvaluationMetadata`
    under the same key as a comparison's, and `write_comparison` writes it.
    `split`, `fold` and `design_sample` are those of `evaluate`.

    Raises as `compare` does, `SampleError` for a sample of several assets, and
    `EvaluationError` when a user family's parts are invalid or do not add up
    to its payout.
    """
    entries = _entries(covers)
    _require_one_asset(_sample(sample, "sample"), "payout_breakdown")
    run, evaluation, design_samples = _prepared(
        entries,
        sample,
        split=split,
        fold=fold,
        pricing=PurePremium(),
        premium=None,
        options=options,
        design_sample=design_sample,
        allocation=None,
        values=None,
    )
    rows = [row for entry in entries for row in _breakdown_rows(entry, run)]
    _check_keys(rows, (*KEY_COLUMNS, "part"))
    evaluation["pricing"] = None
    evaluation["level"] = list(dict.fromkeys(row["level"] for row in rows))
    evaluation["scope"] = list(dict.fromkeys(row["scope"] for row in rows))
    evaluation["allocation"] = None
    evaluation["part_kinds"] = list(dict.fromkeys(row["part_kind"] for row in rows))
    columns = breakdown_columns(run.options)
    metadata = _metadata(
        entries,
        run.sample,
        evaluation=evaluation,
        design_samples=design_samples,
        options=run.options,
        columns=columns,
        values_sha256=None,
    )
    extra = (
        pa.field("part", pa.string(), nullable=False),
        pa.field("part_kind", pa.string(), nullable=False),
    )
    return pa.Table.from_pylist(rows, schema=_schema(columns, metadata, extra=extra))


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


def _added_entries() -> dict[str, ColumnEntry]:
    """The entries of the metric columns that a table written earlier lacks."""
    return {
        column.name: ColumnEntry(**column.to_document("1"))
        for column in metric_columns(MetricOptions())
        if column.name in _ADDED_METRIC_COLUMNS
    }


def _with_added_metrics(table: Any) -> Any:
    """`table` with the metric columns it was written without, all null.

    A payout breakdown has columns of its own and no `payout_to_loss_ratio`, so
    it is returned as it is.
    """
    names = table.column_names
    if "payout_to_loss_ratio" not in names:
        return table
    position = names.index("payout_to_loss_ratio") + 1
    for name in _ADDED_METRIC_COLUMNS:
        if name not in names:
            table = table.add_column(
                position,
                pa.field(name, pa.float64()),
                pa.nulls(table.num_rows, pa.float64()),
            )
        position += 1
    return table


_SCENARIO_COLUMN_TYPES: Mapping[str, Any] = {
    "design_scenario": pa.string(),
    "pathway": pa.string(),
    "horizon": pa.int32(),
    "product": pa.string(),
    "n_rows": pa.int32(),
}
"""The types of the identity columns that read as null in an earlier table."""


def _with_added_identity(table: Any) -> Any:
    """`table` with the scenario columns it was written without, all null.

    Each goes after the identity column that comes before it, which it has or
    has just been given.
    """
    for name in _ADDED_SCENARIO_COLUMNS:
        if name in table.column_names:
            continue
        previous = IDENTITY_COLUMNS[IDENTITY_COLUMNS.index(name) - 1]
        kind = _SCENARIO_COLUMN_TYPES[name]
        table = table.add_column(
            table.column_names.index(previous) + 1,
            pa.field(name, kind),
            pa.nulls(table.num_rows, kind),
        )
    return table


def _with_added_blocks(table: Any, betas: Iterable[float]) -> Any:
    """`table` with the rank-block columns it was written without, all null.

    A payout breakdown has none, so it is returned as it is.
    """
    if "payout_to_loss_ratio" not in table.column_names:
        return table
    for column in _block_fields(betas):
        if column.name not in table.column_names:
            table = table.append_column(column, pa.nulls(table.num_rows, column.type))
    return table


def to_long(table: Any) -> Any:
    """Return the tidy form of a comparison table: one row per metric.

    Each row holds the identity columns of the wide row, then `column` (the
    wide column's name), `metric`, `parameter` and `parameter_value` (for
    example `"beta"` and 0.99; null for metrics without one), `unit` and
    `value`. Rows are ordered by metric column, then by the wide rows'
    order. The table's metadata is kept. A comparison written before
    `payout_frequency` and `unidentified_share` existed reads as null: its
    long form holds them with a null `value`. So does one written before
    `design_scenario`, `pathway`, `horizon`, `product` and `n_rows` existed, or
    before the rank-block columns existed: its long form holds those columns,
    null.

    Raises `EvaluationError` for a table without comparison metadata or
    with metadata that does not describe it, or for a table of
    `bootstrap_intervals`, which is in the long form already, and `TypeError`
    for a table that is not a `pyarrow.Table`.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(f"table must be a pyarrow.Table, got {type(table).__name__}")
    if "ci_low" in table.column_names:
        raise EvaluationError(
            "to_long takes a wide comparison table, but this is a table of "
            "bootstrap_intervals, already one row per wide row and metric column"
        )
    metadata = EvaluationMetadata.from_table(table)
    metadata.check_table(table)
    schema_metadata = table.schema.metadata
    table = _with_added_blocks(
        _with_added_identity(_with_added_metrics(table)),
        metadata.setup.metric_options.get("betas") or (),
    )
    described = {**_added_entries(), **metadata.columns}
    columns = _identity_columns(table.column_names)
    identity = table.select(columns)
    n = table.num_rows
    parts = []
    for name in table.column_names:
        if name in columns:
            continue
        entry = described[name]
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
    return long.replace_schema_metadata(schema_metadata)
