"""The parametric plan: from hazard curves or per-year values to compared covers.

A plan is lazy and immutable, as every crc-sdk source is. Builders return a
new plan and do no I/O, and neither does `explain()`. It has two entry
points:

```text
ParametricPlan.for_evaluation(request, value_column=, value_unit=)   # fitted curves
  .monte_carlo(n_draws, seed=)          # required there: simulated years
ParametricPlan.for_annual_extremes(source, assets, impact=, unit=, provenance=)
  .index(source, location=, unit=, provenance=)   # optional: read elsewhere
# then, on either plan:
  .basis_risk(model)                    # required: how index and loss differ
  .covers(*covers)                      # resolved covers and templates
  .objective(*objectives, pricing=, budget=, limit=, options=)
  .pricing(PurePremium() | CostOfCapitalLoad(rate, q))     # optional
  .split(FreshDraws(n, seed=) | Holdout(0.7) | KFold(5))   # optional
  .metrics(MetricOptions(...))          # optional: levels and thresholds
  .scopes(Individual() | Joint())       # optional: asset by asset, or on the total
  .dependence(Comonotonic() | ...)      # with Joint() on curves: required
  .allocation("value" | "loss")         # optional: per-asset rows of a joint design
  .explain()                            # the setup, no I/O
  .samples()                            # execution: one sample per asset
  .joint_sample()                       # execution: Joint() only
  .designs()                            # execution: the designed covers
  .write_parquet("comparison.parquet")  # execution: written atomically
```

**Fitted curves** (`for_evaluation`). `request` is the `PortfolioEvaluation`
built with `HazardDataset.local(path).for_assets(assets).select(...)
.impact(...)`; the plan reuses its dataset, assets, selection and impact.
At execution, `request.asset_curves()` gives each asset's canonical curve
row (one hazard, horizon and pathway per asset), and `sample_curve` draws
`n_draws` simulated years from it, each with probability 1/n. The
probability-semantics rules of `sample_curve` apply: a curve whose
`probability_semantics` is unspecified is refused unless `.monte_carlo(...,
assume_semantics=...)` states one, and draws beyond the fitted support and a
pooled ensemble are noted in each sample. The impact gets the asset's
matched H3 cell and its context columns, as `request.write_parquet` gives
them.

**Per-year values** (`for_annual_extremes`). `source` is a crc-sdk plan with
an `annual_extremes()` execution point (ERA5 and CMIP6 canonicalization
plans, NEX-GDDP-CMIP6, station plans, NOAA water levels), an Arrow table in
one of their layouts, or a local Parquet file of one. Each asset's series is
the grid cell whose centre is nearest the asset (`NearestCell`, the default,
for per-cell tables; an asset located by its H3 cell needs the whole cell
inside one grid cell), or the station an asset column names
(`StationColumn`). `unit` is the unit of the table's values and
`provenance` (a `SourceProvenance`) the source's licence, attribution and
version: annual-extremes tables carry no licence of their own. Each asset
gets one `OutcomeSample` from `sample_annual`: its years are the rows, each
with probability 1/n. `trend=` states how the climate trend is treated,
`scenario=` names the rows, and `cell=` gives point assets the H3 cell that
some registry impacts need in their context: the name of an asset column, or
`H3Cell(resolution=r)` for the cell of the asset's point.

**Pooled model-years** (`pool=`, per-year values only). `pool=ModelPool()` pools
the years of several climate-model runs per asset: `source` is one table, file
or plan whose rows carry the columns `ModelPool` names, as NEX-GDDP-CMIP6's
batches do, or a sequence of sources, one per run, as CMIP6's plans are. Each
asset gets one sample from `sample_annual_pool`, its rows the runs' model-years
at `1/N` each and `period` their draw number. Each run is located on its own
grid, and a run named by no column is named by its source plan's `explain()`.

**An index read elsewhere** (`.index`, per-year values only). The cover pays on
a series from a second source, at a station an asset column names or in the
grid cell nearest the asset or another point, while the loss still reads the
hazard. Each asset's sample is that of `sample_annual(hazard, index=series)`.
The index is read on exactly the hazard's years, under the plan's one
`missing=`. A cover stated with `AtAsset()` is bound at execution to each
asset's resolved `AtStation` or `AtPoint` (the cell centre, never the asset's
own point), so the location is in its document and its hashes.

On both routes the loss is `impact(hazard) × value`, with the value from an
asset column or 1 (the damage ratio), and the basis-risk model is required:
`NoBasisRisk()` must be written out.

**Seeding.** The plan holds an integer `seed` (`monte_carlo`'s, or
`for_annual_extremes`'). At execution it builds
`numpy.random.SeedSequence(seed)` and spawns one child generator per asset,
in asset order; each draws the hazard uniforms first, then the basis-risk
model. Fresh draws use `SeedSequence(FreshDraws.seed)` the same way. On the
per-year route only the basis-risk model draws, so `ImpactNoise` needs a
seed there.

**Covers and designs.** A cover written for no asset (`asset_id=None`)
applies to every asset, as a copy written for that asset; one written for an
asset applies to that asset only. Templates, covers with `Free` parameters,
are designed under each objective of `.objective(...)` on each design block;
resolved covers are evaluated as they are. A designed cover's label is the
template's label (or the first 12 hex digits of its hash) followed by the
objective, for example `"step (mse)"`.

**Pricing.** One pricing prices every premium of the plan: the designs' on
their design block and the resolved covers' on each evaluation block. It is
`.objective(..., pricing=...)`'s or `.pricing(...)`'s, which must agree when
both are stated, and `PurePremium()` when neither is.

**Splits.** Without `.split(...)`, every cover is designed and evaluated on
the whole sample, so every row is in sample. `FreshDraws(n_draws, seed=)`
(curves only) designs on the plan's draws and evaluates on `n_draws` fresh
draws from the same curve. `Holdout(train_fraction)` designs on the first
periods (years or draws) and evaluates on the rest; `KFold(k)` holds out
each of `k` contiguous blocks in turn. Every block gets in-sample rows (on
the design block) and out-of-sample rows (on the held-out block or the fresh
draws). Resolved covers are evaluated on the same blocks, with the design
block as their design sample, so their terms are taken as chosen without
the held-out data.

**The climate trend under a split.** `LinearTrend` is fitted on the design
years only: with `Holdout` or `KFold` the plan builds each asset's sample once
per fold, fitted on that fold's design years, and takes the fold's two blocks
from it, so a held-out value changes neither the fit nor the design block. A
basis-risk model that draws starts every fold from the asset's same generator
state, so a year's factor does not depend on the fold. `RecentYears` is not
rebuilt: the window comes first and the split divides it. `.samples()` and
`ParametricResult.samples` hold all the years, fitted on all of them. On a pool
each run has its own fit, on the same design years with the same reference
year, and a split divides the pool's years, not its rows.

**Scopes.** By default, `Individual()`, each asset is designed and evaluated
on its own sample: nothing sums over the assets and no dependence between them
is imposed. `Joint()` needs at least two assets and designs each template once
per objective with `design_joint`, as one objective and one budget on the
owner's total, on one sample of all the assets:

- on fitted curves, `sample_curves` under the model that `.dependence(...)`
  must state, with the plan's generators, one per asset;
- on per-year values, `OutcomeSample.join` of the assets' samples, whose
  dependence is the data's own, so `.dependence(...)` raises there. With
  `LinearTrend` and a holdout or k-fold the assets' samples are joined per fold,
  each fitted on that fold's design years.

Under `Joint()` a cover must be written for no asset, because each cover applies
to every asset; different terms per asset go through `design_joint`. A resolved
cover is evaluated as a set, one contract per asset. `.samples()` returns each
asset's view of the sample and `.joint_sample()` the sample itself. `.scopes`
may hold both scopes: each asset is then also designed on its own view, and the
`scope` column keeps the rows of each scope apart. `.allocation(rule)` adds a
joint design's per-asset rows, with the values the plan's loss model read.
`docs/parametric.md` describes the rows.

**Output.** `write_parquet(path)` writes one comparison table, as `compare`
builds it, through a temporary sibling and `os.replace`. Its metadata under
`crc.parametric.evaluation` (`EvaluationMetadata`) holds the setup of every
number: the samples with their source, loss and basis-risk models, the
splits, the pricing, the designs and the covers, plus the source's record
once, in `source_records` (the canonical file's path and SHA-256, or the
annual table's checksum and the source plan's `explain()`, and the index
table's likewise), the licences,
the generator states, and the plan's seed in every evaluation's `details`
(`"plan_seed"`, with `"fresh_draws_seed"` on fresh-draw rows). Asset values
are never written; the metadata holds a SHA-256 of the (asset id, value)
pairs. The returned `ParametricResult` holds the samples, the fresh draws
and the designs and, under `Joint()`, the sample of all the assets.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from crc_framework import (
    CallableImpact,
    ClimateImpact,
    ImpactFunction,
    TransformContext,
    impact_registry_manifest,
)
from crc_framework import ImpactExpression as FrameworkImpactExpression
from numpy.typing import NDArray

from crc_sdk.impacts.calibration import units_equivalent
from crc_sdk.impacts.expression import ImpactExpression
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.dependence import (
    Comonotonic,
    Dependence,
    EmpiricalCopula,
    Independent,
)
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import ProbabilitySemanticsError, SampleError
from crc_sdk.parametric.evaluate import Allocation, Split, compare, write_comparison
from crc_sdk.parametric.index import AtAsset, AtPoint, AtStation, IndexDefinition
from crc_sdk.parametric.joint import Individual, Joint, JointDesignResult, design_joint
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY, EvaluationMetadata
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    Objective,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.pricing import CostOfCapitalLoad, Pricing, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _arrow_table,
    _identifier,
    _sample_years,
    _string_values,
    _thaw,
    holdout,
    year_block_kfold,
)
from crc_sdk.parametric.sources import (
    AnnualSemantics,
    AnnualSeries,
    MissingYears,
    ModelRun,
    OutsideSupport,
    _annual_rows,
    _coordinate,
    _pool_rows,
    _source_plan_document,
    _warn_comonotone,
    sample_curve,
    sample_curves,
)
from crc_sdk.parametric.trend import AsRecorded, LinearTrend, Trend, _require_trend
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows.distributions import CURVE_COLUMNS
from crc_sdk.workflows.portfolio import (
    AssetPortfolio,
    CellColumn,
    ImpactContextColumns,
    ImpactSpec,
    PointColumns,
    PortfolioEvaluation,
    _ExpressionImpact,
)

__all__ = [
    "FreshDraws",
    "H3Cell",
    "Holdout",
    "KFold",
    "ModelPool",
    "NearestCell",
    "ParametricPlan",
    "ParametricResult",
    "StationColumn",
]

_CURVES_READ_AT_THE_ASSET = (
    "an index read elsewhere than a fitted curve's asset needs a joint "
    "distribution of two curves, which the curves do not give"
)
_EXECUTION = (
    "the source is read, the samples are built and the covers are designed and "
    "evaluated only at samples(), designs() and write_parquet()"
)


# -- value objects -----------------------------------------------------------


@dataclass(frozen=True)
class Holdout:
    """Design on the first share of the years; evaluate on the rest.

    The design block is the first `floor(train_fraction * n)` years of each
    asset's sample (`holdout`), so the held-out block holds its latest
    years. Raises `ValueError` unless `train_fraction` lies in (0, 1).
    """

    train_fraction: float = 0.7

    def __post_init__(self) -> None:
        value = self.train_fraction
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(
                f"train_fraction must be a number in (0, 1), got {value!r}"
            )
        object.__setattr__(self, "train_fraction", float(value))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "holdout", "train_fraction": f}`."""
        return {"kind": "holdout", "train_fraction": self.train_fraction}


@dataclass(frozen=True)
class KFold:
    """Hold out each of `k` contiguous blocks of years in turn.

    Fold `i` designs on the other blocks and evaluates on block `i`
    (`year_block_kfold`), so every year is held out once. Raises
    `ValueError` unless `k` is an integer of at least 2; a sample with fewer
    than `k` years raises `SampleError` at execution.
    """

    k: int

    def __post_init__(self) -> None:
        value = self.k
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"k must be an integer of at least 2, got {value!r}")
        if int(value) < 2:
            raise ValueError(f"k must be an integer of at least 2, got {value!r}")
        object.__setattr__(self, "k", int(value))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "kfold", "k": k}`."""
        return {"kind": "kfold", "k": self.k}


def _positive_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _seed_value(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class FreshDraws:
    """Evaluate on fresh Monte Carlo draws from each asset's curve.

    For plans on fitted curves (`for_evaluation`). The designs are made on
    the plan's draws (`monte_carlo`), and every cover is evaluated out of
    sample on `n_draws` fresh draws from the same curve, drawn with one
    child generator of `numpy.random.SeedSequence(seed)` per asset, in asset
    order. Draws are independent simulated years, so fresh draws are the
    plain out-of-sample test. `seed` must differ from the plan's seed; it is
    recorded in every fresh-draw evaluation's `details` as
    `"fresh_draws_seed"`. Raises `ValueError` unless `n_draws` is a positive
    integer and `seed` a non-negative one.
    """

    n_draws: int
    seed: int = field(kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_draws", _positive_count("n_draws", self.n_draws))
        object.__setattr__(self, "seed", _seed_value("seed", self.seed))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "fresh_draws", "n_draws": n, "seed": seed}`."""
        return {"kind": "fresh_draws", "n_draws": self.n_draws, "seed": self.seed}


@dataclass(frozen=True)
class _MonteCarlo:
    """What `.monte_carlo(...)` states: the draws per asset and the curve rules."""

    n_draws: int
    assume_semantics: AnnualSemantics | None
    outside_support: OutsideSupport

    def to_document(self) -> dict[str, Any]:
        return {
            "assume_semantics": self.assume_semantics,
            "n_draws": self.n_draws,
            "outside_support": self.outside_support,
        }


def _column_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty column name, got {value!r}")
    return value


@dataclass(frozen=True)
class NearestCell:
    """Read each asset's series from the grid cell whose centre is nearest it.

    For per-cell annual tables such as ERA5's and CMIP6's `annual_extremes()`
    (`year`, `latitude`, `longitude`, `value`). The cell is accepted only if
    its centre lies within half a grid step of the asset in latitude and in
    longitude, as `AnnualSeries.nearest_cell` decides; otherwise the plan
    raises at execution. `longitude` and `latitude` name the table's
    coordinate columns; `spacing` is the grid step `(latitude, longitude)` in
    degrees, by default the smallest step between the table's coordinates.

    - Assets located by a point (`PointColumns`) are located at the point.
    - Assets located by an H3 cell (`CellColumn`) are located at the cell:
      the grid cell nearest the H3 cell's centre is accepted only if the
      whole H3 cell lies within it, every vertex within half a grid step of
      its centre. An H3 cell that straddles two grid cells does not say
      which cell's series is the asset's, so the plan raises `SampleError`
      at execution; give such an asset a point instead. The series records
      `"asset_location": "h3_cell"`. This needs the `geometry` extra
      (`crc-sdk[geometry]`).
    """

    longitude: str = "longitude"
    latitude: str = "latitude"
    spacing: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _column_name("longitude", self.longitude)
        _column_name("latitude", self.latitude)
        if self.spacing is not None:
            spacing = self.spacing
            if (
                isinstance(spacing, (str, bytes))
                or not isinstance(spacing, Sequence)
                or len(spacing) != 2
                or not all(
                    isinstance(step, Real)
                    and not isinstance(step, bool)
                    and math.isfinite(float(step))
                    and float(step) > 0.0
                    for step in spacing
                )
            ):
                raise ValueError(
                    "spacing must be a (latitude, longitude) pair of positive "
                    f"degrees, got {spacing!r}"
                )
            object.__setattr__(self, "spacing", (float(spacing[0]), float(spacing[1])))

    def to_document(self) -> dict[str, Any]:
        """Return the locator's kind, columns and spacing."""
        return {
            "kind": "nearest_cell",
            "latitude_column": self.latitude,
            "longitude_column": self.longitude,
            "spacing": None if self.spacing is None else list(self.spacing),
        }


@dataclass(frozen=True)
class StationColumn:
    """Read each asset's series from the station an asset column names.

    For per-station annual tables such as the station plans'
    `annual_extremes()` (`station`, `year`, ..., `value`). `name` is the
    asset column holding each asset's station id, and `table_column` the
    table column it is matched against (`"station"`; NOAA water levels use
    `"station_id"`).
    """

    name: str
    table_column: str = "station"

    def __post_init__(self) -> None:
        _column_name("name", self.name)
        _column_name("table_column", self.table_column)

    def to_document(self) -> dict[str, Any]:
        """Return the locator's kind and columns."""
        return {
            "kind": "station",
            "asset_column": self.name,
            "table_column": self.table_column,
        }


@dataclass(frozen=True)
class H3Cell:
    """Give each asset the H3 cell of its point, at a stated resolution.

    For `for_annual_extremes(..., cell=H3Cell(resolution=r))`: the impact's
    context gets the cell at resolution `r` of each asset's point
    (`PointColumns`). The cells are computed at execution, with
    `crc_sdk.geometry.h3.point_to_cell`, which needs the `geometry` extra
    (`crc-sdk[geometry]`); stating one needs nothing. There is no default
    resolution: which one a registry impact expects is the caller's to state.
    The cell is recorded in each sample's provenance and the point is not, so
    a fine resolution records the asset's place to within that cell.

    Raises `ValueError` unless `resolution` is an integer from 0 to 15.
    """

    resolution: int

    def __post_init__(self) -> None:
        value = self.resolution
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= 15
        ):
            raise ValueError(
                f"resolution must be an integer from 0 to 15, got {value!r}"
            )
        object.__setattr__(self, "resolution", int(value))

    def to_document(self) -> dict[str, Any]:
        """Return `{"kind": "h3_cell", "resolution": r}`."""
        return {"kind": "h3_cell", "resolution": self.resolution}


@dataclass(frozen=True)
class ModelPool:
    """Pool the years of several climate-model runs, per asset.

    For `for_annual_extremes(..., pool=ModelPool())`. `model`, `member` and
    `pathway` name the columns that say whose each row of the source is, as
    NEX-GDDP-CMIP6's batches carry them. A source without them, such as a CMIP6
    plan, is named by its `explain()` document, where its `model`, `member` and
    `pathway` keys state the run. Raises `ValueError` for an empty column name.
    """

    model: str = "model"
    member: str = "member"
    pathway: str = "scenario"

    def __post_init__(self) -> None:
        _column_name("model", self.model)
        _column_name("member", self.member)
        _column_name("pathway", self.pathway)

    def to_document(self) -> dict[str, Any]:
        """Return the kind and the columns that name a run."""
        return {
            "kind": "model_pool",
            "model_column": self.model,
            "member_column": self.member,
            "pathway_column": self.pathway,
        }


@dataclass(frozen=True, eq=False)
class _SeriesRead:
    """What a plan reads for each asset: its hazard or its `.index(...)` series.

    `years` are the years the plan states for the hazard. The index takes the
    hazard's years at execution, so its `years` are `None`, and `point` names
    the asset columns of another point for `NearestCell` to read at.
    """

    role: Literal["hazard", "index"]
    source: Any
    location: NearestCell | StationColumn
    unit: str
    provenance: SourceProvenance
    year_column: str
    value_column: str
    years: tuple[int, ...] | None = None
    point: PointColumns | None = None


@dataclass(frozen=True, eq=False)
class _OpenedRead:
    """A `_SeriesRead` at execution: its table and what is recorded of it."""

    read: _SeriesRead
    table: Any
    record: dict[str, Any]
    source: dict[str, Any]
    pathway: str | None


@dataclass(frozen=True, eq=False)
class _OpenedRun:
    """One run of a pool at execution: its own rows, and whose they are."""

    opened: _OpenedRead
    model: str
    member: str
    pathway: str


@dataclass(frozen=True)
class _DesignSpec:
    """What `.objective(...)` states: the objectives and the design terms.

    `pricing` is `None` when `.objective(...)` states none; the plan's
    pricing then applies (`ParametricPlan._pricing`).
    """

    objectives: tuple[Objective, ...]
    pricing: Pricing | None
    budget: Budget | None
    limit: Limit | None
    options: DesignOptions

    def to_document(self, pricing: Pricing) -> dict[str, Any]:
        """The design terms, with `pricing`, the plan's pricing."""
        return {
            "budget": None if self.budget is None else self.budget.to_document(),
            "limit": None if self.limit is None else self.limit.to_document(),
            "objectives": [objective.to_document() for objective in self.objectives],
            "options": self.options.to_document(),
            "pricing": pricing.to_document(),
        }


_PRICINGS = (PurePremium, CostOfCapitalLoad)


def _check_pricing(pricing: Any) -> Pricing:
    if not isinstance(pricing, _PRICINGS):
        raise TypeError(
            "pricing must be PurePremium() or CostOfCapitalLoad(rate, q), "
            f"got {type(pricing).__name__}"
        )
    return pricing


def _same_pricing(first: Pricing, second: Pricing) -> bool:
    return json.dumps(first.to_document(), sort_keys=True) == json.dumps(
        second.to_document(), sort_keys=True
    )


def _pricing_conflict(by_pricing: Pricing, by_objective: Pricing) -> ValueError:
    return ValueError(
        f".pricing(...) states {by_pricing.to_document()}, but "
        f".objective(..., pricing=...) states {by_objective.to_document()}; one "
        "pricing prices every premium of the plan, so state it once, or state "
        "the same pricing in both"
    )


@dataclass(frozen=True, eq=False)
class ParametricResult:
    """What `ParametricPlan.write_parquet` wrote, and what it was built from.

    - `output`: the Parquet file;
    - `table`: the comparison table as written, with its metadata;
    - `metadata`: its `EvaluationMetadata`, under `crc.parametric.evaluation`;
    - `samples`: one `OutcomeSample` per asset, in asset order (all years
      or draws, before any split; with `LinearTrend` and a holdout or k-fold,
      fitted on all the years, while the rows come from one sample per fold).
      Under `Joint()`, each asset's view of `joint_sample`;
    - `designs`: every `DesignResult`, in asset, fold, cover and objective
      order. Under `Joint()`, every `JointDesignResult`, in fold, cover and
      objective order, each followed by its template's per-asset
      `DesignResult`s when the plan also holds `Individual()`;
    - `fresh_samples`: with `FreshDraws`, one `OutcomeSample` per asset, in
      asset order: the fresh draws the out-of-sample rows are evaluated on.
      `None` without fresh draws. Under `Joint()`, each asset's view of
      `fresh_joint_sample`;
    - `joint_sample`: under `Joint()`, the sample of all the assets, with all
      its years or draws, before any split; otherwise `None`;
    - `fresh_joint_sample`: under `Joint()` with `FreshDraws`, the fresh draws
      of all the assets, whose `content_hash` is the `sample_hash` of the
      out-of-sample rows; otherwise `None`.
    """

    output: Path
    table: Any
    metadata: EvaluationMetadata
    samples: Mapping[str, OutcomeSample]
    # `DesignResult`, or `JointDesignResult` under `Joint()`: it depends on the
    # plan's scopes, so existing callers of a plan of one scope keep their types.
    designs: tuple[Any, ...]
    fresh_samples: Mapping[str, OutcomeSample] | None = None
    joint_sample: OutcomeSample | None = None
    fresh_joint_sample: OutcomeSample | None = None

    @property
    def row_count(self) -> int:
        """The number of rows written."""
        return int(self.table.num_rows)


@dataclass(frozen=True, eq=False)
class _Run:
    samples: Mapping[str, OutcomeSample]
    fresh: Mapping[str, OutcomeSample] | None
    designs: tuple[Any, ...]
    tables: tuple[Any, ...]
    source_key: str
    source: dict[str, Any]
    id_column: str
    values_sha256: str | None
    index_source: dict[str, Any] | None = None
    joint: OutcomeSample | None = None
    fresh_joint: OutcomeSample | None = None
    run_records: Mapping[tuple[str, str], dict[str, Any]] | None = None


@dataclass(frozen=True, eq=False)
class _AnnualAsset:
    """What one asset's per-year samples are built from: a series, or a pool's runs."""

    asset_id: str
    series: AnnualSeries | tuple[ModelRun, ...]
    index_series: AnnualSeries | None
    loss_model: LossModel
    cell: int | None
    location: AtPoint | AtStation | None
    unused: list[int]


_Block = tuple[int, OutcomeSample, OutcomeSample | None]
"""A fold, its design block, and its evaluation block if it has one."""


@dataclass(frozen=True, eq=False)
class _Samples:
    """What execution builds before any design: the samples and the source.

    `folds` holds each asset's sample of each fold, when each fold has a sample
    of its own. Under `Joint()`, `joint` and `fresh_joint` are the samples of
    all the assets, and `samples` and `fresh` each asset's view of them. A pool
    read from a sequence of sources has each run's record in `run_records`,
    by model and member.
    """

    portfolio: AssetPortfolio
    samples: dict[str, OutcomeSample]
    fresh: dict[str, OutcomeSample] | None
    record_key: str
    record: dict[str, Any]
    index_record: dict[str, Any] | None = None
    locations: Mapping[str, AtPoint | AtStation] | None = None
    folds: Mapping[str, tuple[OutcomeSample, ...]] | None = None
    joint: OutcomeSample | None = None
    fresh_joint: OutcomeSample | None = None
    run_records: Mapping[tuple[str, str], dict[str, Any]] | None = None


# -- helpers -----------------------------------------------------------------


def _require_several_assets(asset_ids: Sequence[str]) -> None:
    if len(asset_ids) < 2:
        raise ValueError(
            "Joint() designs several assets together, but the plan holds "
            f"{list(asset_ids)}; use Individual() for one asset"
        )


def _copula_split_error(rule: Holdout | KFold) -> SampleError:
    return SampleError(
        f"{type(rule).__name__} cannot split a sample under EmpiricalCopula: its "
        "rows come in reference-year blocks, so a block of rows is a set of "
        "reference years; test it out of sample with .split(FreshDraws(n_draws, "
        "seed=...)), whose n_draws must be a multiple of the reference's years"
    )


def _json_safe(value: Any) -> Any:
    """Return `value` as plain JSON types; unknown objects become their text."""
    return json.loads(json.dumps(value, default=str))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _table_sha256(table: Any) -> str:
    """SHA-256 of the table's Arrow IPC stream, in one chunk, without metadata."""
    data = table.replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, data.schema) as writer:
        writer.write_table(data)
    return f"sha256:{hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()}"


def _stated_pathway(explained: Any) -> str | None:
    """The pathway a source plan's `explain()` document states, if any."""
    pathway = explained.get("pathway") if isinstance(explained, dict) else None
    return pathway if isinstance(pathway, str) and pathway else None


def _source_kind(source: Any) -> Literal["table", "file", "plan"]:
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        return "table"
    if isinstance(source, (str, os.PathLike)):
        return "file"
    if callable(getattr(source, "annual_extremes", None)):
        return "plan"
    raise TypeError(
        "source must be a plan with an annual_extremes() execution point, a "
        "pyarrow.Table in an annual-extremes layout, or a local Parquet file; "
        f"got {type(source).__name__}"
    )


def _pool_source(source: Any) -> Any:
    """A pool's `source`: a sequence becomes a tuple, each of its sources checked."""
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if not source:
            raise ValueError("source must name at least one source of the pool")
        for each in source:
            _source_kind(each)
        return tuple(source)
    _source_kind(source)
    return source


def _run_names(table: Any, name: str) -> Any:
    """A column that names runs, as text; `SampleError` if it cannot."""
    column = table.column(name)
    if pa.types.is_dictionary(column.type):
        column = column.cast(column.type.value_type)
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        raise SampleError(f"column {name!r} must hold strings, not {column.type}")
    if column.null_count:
        raise SampleError(
            f"column {name!r} has {column.null_count} null values, so it does not "
            "name every row's run"
        )
    return column.cast(pa.string())


def _assets(assets: Any) -> AssetPortfolio | Path:
    """Return the assets as an `AssetPortfolio`, or a file read at execution."""
    if isinstance(assets, AssetPortfolio):
        return assets
    if isinstance(assets, (str, os.PathLike)):
        return Path(assets)
    return AssetPortfolio(assets)


def _table_columns(assets: AssetPortfolio | Path) -> tuple[str, ...] | None:
    """The asset table's columns, when it is in memory; else `None`."""
    if isinstance(assets, AssetPortfolio) and isinstance(assets.data, pa.Table):
        return tuple(assets.data.column_names)
    return None


def _registry_needs_cell(
    impact: Any, context: Any, assets: AssetPortfolio | Path, cell: Any
) -> str | None:
    """The registry factor that needs an H3 cell the plan cannot give, if any.

    `cell=` gives one. `LossModel.for_asset` fills the cell from an asset's
    `CellColumn` when context columns are given; otherwise only a
    `TransformContext` gives one.
    """
    if cell is not None or not isinstance(impact, ClimateImpact):
        return None
    if (
        isinstance(assets, AssetPortfolio)
        and isinstance(assets.location, CellColumn)
        and isinstance(context, ImpactContextColumns)
        and context.items()
    ):
        return None
    factor = str(getattr(impact, "factor", ""))
    entry = next(
        (
            item
            for item in impact_registry_manifest()["factors"]
            if item["factor"] == factor
        ),
        None,
    )
    if entry is None or "cell" not in tuple(entry.get("required_context") or ()):
        return None
    for given in (getattr(impact, "context", None), context):
        if isinstance(given, TransformContext) and given.cell is not None:
            return None
    return factor


def _check_cell(cell: Any, impact: Any, context: Any) -> None:
    """Check `cell=`: what it states, and that the impact can take the cell."""
    if isinstance(cell, str):
        _column_name("cell", cell)
    elif not isinstance(cell, H3Cell):
        raise TypeError(
            "cell must be the name of an asset column or H3Cell(resolution=r), "
            f"got {type(cell).__name__}"
        )
    if not _takes_context(impact):
        raise SampleError(
            "cell= fills the impact's context, and impact expressions and "
            "callables take none"
        )
    for given in (getattr(impact, "context", None), context):
        if isinstance(given, TransformContext) and given.cell is not None:
            raise ValueError(
                "cell= states each asset's H3 cell, and the impact's context "
                "already holds one; state one of them"
            )


def _cell_origin(cell: str | H3Cell) -> str:
    """Where `cell=` takes an asset's H3 cell from, in words."""
    if isinstance(cell, H3Cell):
        return f"the cell at resolution {cell.resolution} of the asset's point"
    return f"the asset column {cell}"


def _noted(sample: OutcomeSample, note: str) -> OutcomeSample:
    """`sample` with one more note in its provenance."""
    provenance = dataclasses.replace(
        sample.provenance, notes=(*sample.provenance.notes, note)
    )
    return dataclasses.replace(sample, provenance=provenance)


def _listed(values: Sequence[Any] | None) -> list[Any] | None:
    return None if values is None else list(values)


def _views(joint: OutcomeSample) -> dict[str, OutcomeSample]:
    """Each asset's own sample of a sample of several assets, in asset order."""
    return {asset_id: joint.asset(asset_id) for asset_id in joint.asset_ids}


def _objective_tag(objective: Objective) -> str:
    if isinstance(objective, WeightedCVaR):
        return f"weighted_cvar alpha={objective.alpha:g} beta={objective.beta:g}"
    if isinstance(objective, UserObjective):
        return f"user {objective.name}"
    return str(objective.to_document()["kind"])


def _index_origin(read: _SeriesRead) -> str:
    """Where `.index(...)` takes an asset's index series from, in words."""
    if isinstance(read.location, StationColumn):
        return f"the station the asset column {read.location.name} names"
    if read.point is None:
        return "the grid cell of the index source nearest the asset"
    return (
        "the grid cell of the index source nearest the point in the asset "
        f"columns {read.point.longitude} and {read.point.latitude}"
    )


def _wrapped_longitude(longitude: float) -> float:
    """`longitude` within [-180, 180], as `AtPoint` takes it; a grid may run to 360."""
    if -180.0 <= longitude <= 180.0:
        return longitude
    return (longitude + 180.0) % 360.0 - 180.0


def _resolved_location(
    read: _SeriesRead, series: AnnualSeries, row: Mapping[str, Any]
) -> AtPoint | AtStation:
    """Where an asset's index series was read: its station, or the cell centre.

    The matched cell's centre, as the series records it, and not the point it
    was matched from: the asset's own point stays out of the cover's document.
    """
    if isinstance(read.location, StationColumn):
        return AtStation(str(row[read.location.name]))
    cell = series.provenance["cell"]
    return AtPoint(
        _wrapped_longitude(float(cell["longitude"])), float(cell["latitude"])
    )


def _on_years(
    series: AnnualSeries, years: NDArray[np.int64]
) -> tuple[AnnualSeries, list[int]]:
    """`series` on exactly `years`, and the years with a value that it left out.

    A year of `years` that `series` does not hold has no value, so that the
    plan's `missing=` decides what becomes of it, as for a null year.
    """
    values = np.full(years.shape[0], np.nan)
    _, on_years, in_series = np.intersect1d(
        years, series.years, assume_unique=True, return_indices=True
    )
    values[on_years] = series.values[in_series]
    held = series.years[~np.isnan(series.values)]
    unused = [int(year) for year in np.setdiff1d(held, years)]
    return AnnualSeries(years, values, series.unit, _thaw(series.provenance)), unused


def _index_recorded(
    sample: OutcomeSample,
    read: _SeriesRead,
    location: AtPoint | AtStation,
    unused: Sequence[int],
) -> OutcomeSample:
    """`sample` with the index years not used and where the index was read."""
    asset_id = sample.asset_ids[0]
    place = (
        f"station {location.station_id}"
        if isinstance(location, AtStation)
        else (
            f"the grid cell centred at latitude {location.latitude:g}, "
            f"longitude {location.longitude:g}"
        )
    )
    notes = [
        f"The index of asset {asset_id} is read at {place}: {_index_origin(read)}."
    ]
    if unused:
        noun, verb = ("year", "is") if len(unused) == 1 else ("years", "are")
        notes.append(
            f"{len(unused)} index {noun} outside the hazard's years {verb} not "
            f"used: {', '.join(str(year) for year in unused[:10])}"
            f"{', ...' if len(unused) > 10 else ''}."
        )
    provenance = dataclasses.replace(
        sample.provenance,
        details={
            **_thaw(sample.provenance.details),
            "unused_index_years": list(unused),
        },
        notes=(*sample.provenance.notes, *notes),
    )
    return dataclasses.replace(sample, provenance=provenance)


def _same_location(first: Any, second: Any) -> bool:
    """Whether two stated locations name one station, or one point on the globe."""
    if isinstance(first, AtStation) and isinstance(second, AtStation):
        return first.station_id == second.station_id
    if isinstance(first, AtPoint) and isinstance(second, AtPoint):
        d_lon = (first.longitude - second.longitude + 180.0) % 360.0 - 180.0
        return math.isclose(
            first.latitude, second.latitude, abs_tol=1e-9
        ) and math.isclose(d_lon, 0.0, abs_tol=1e-9)
    return False


def _index_at(
    cover: Cover, asset_id: str, location: AtPoint | AtStation
) -> IndexDefinition:
    """`cover`'s index, for an asset whose index series is read at `location`.

    A cover stated with `AtAsset()` is bound to it; one that already names a
    location keeps its own, which must be that one.
    """
    index = cover.index
    if isinstance(index.location, AtAsset):
        return IndexDefinition(
            index.hazard_name,
            index.unit,
            tail=index.tail,
            location=location,
            aggregation=index.aggregation,
            source=index.source,
        )
    if _same_location(index.location, location):
        return index
    raise SampleError(
        f"cover {cover.label or cover.definition_hash!r} names the location "
        f"{index.location!r}, but the plan reads the index of asset {asset_id!r} "
        f"at {location!r}; name that one, or state AtAsset()"
    )


def _bound(
    cover: Cover,
    asset_id: str,
    label: str | None = None,
    *,
    location: AtPoint | AtStation | None = None,
) -> Cover:
    """`cover` written for `asset_id`, with `label` and `location` if given."""
    name = cover.label if label is None else label
    index = cover.index if location is None else _index_at(cover, asset_id, location)
    if cover.asset_id == asset_id and name == cover.label and index is cover.index:
        return cover
    return Cover(
        index,
        cover.payout,
        payout_unit=cover.payout_unit,
        asset_id=asset_id,
        label=name,
    )


def _design_label(template: Cover, objective: Objective) -> str:
    base = template.label or template.definition_hash.removeprefix("sha256:")[:12]
    return f"{base} ({_objective_tag(objective)})"


_CONTEXT_FIELDS = ("cell", "country", "continent", "building_type", "historic_mean")
_CURVE_ROW_COLUMNS = (
    "cell_index",
    "hazard_name",
    "horizon",
    "pathway",
    "source_id",
    *CURVE_COLUMNS,
)
"""The columns of an `asset_curves()` row that make its canonical curve row."""


def _request_impact(spec: ImpactSpec) -> tuple[Any, ImpactContextColumns | None]:
    """The impact a request's `ImpactSpec` holds, as given, and its context.

    An expression comes back as the `ImpactExpression` itself, so its
    `definition_hash` is kept, and a callable as the callable.
    """
    function: Any = spec.function
    if isinstance(function, _ExpressionImpact):
        function = function.expression
    elif isinstance(function, CallableImpact):
        function = function.function
    context = spec.context_columns if spec.context_columns.items() else None
    return function, context


def _takes_context(impact: Any) -> bool:
    """Whether `LossModel` passes a framework context to `impact`."""
    return not isinstance(
        impact, (ImpactExpression, FrameworkImpactExpression)
    ) and isinstance(impact, ImpactFunction)


def _cell_context(
    impact: Any,
    cell: Any,
    asset: Mapping[str, Any],
    context: ImpactContextColumns | TransformContext | None,
) -> TransformContext:
    """The context an impact gets for one asset with an H3 cell.

    As `PortfolioEvaluation` gives it for a matched row: the impact's own
    context, or a fixed `context` in its place, with the H3 cell and the
    asset's context columns on top; null asset values keep the impact's own.
    """
    base = (
        context
        if isinstance(context, TransformContext)
        else getattr(impact, "context", None)
    )
    fields: dict[str, Any] = {
        name: getattr(base, name) if isinstance(base, TransformContext) else None
        for name in _CONTEXT_FIELDS
    }
    if cell is not None:
        fields["cell"] = int(cell)
    for field_name, column_name in (
        context.items() if isinstance(context, ImpactContextColumns) else ()
    ):
        if column_name not in asset:
            raise SampleError(f"the asset table has no column {column_name!r}")
        value = asset[column_name]
        if value is None:
            continue
        fields[field_name] = (
            float(value) if field_name == "historic_mean" else str(value)
        )
    return TransformContext(**fields)


def _h3_index(asset_id: str, value: Any, column: str) -> int:
    """An asset's H3 cell as an integer; `SampleError` if it is not one."""
    if value is None:
        raise SampleError(f"asset {asset_id!r} has no H3 cell in column {column!r}")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) < 2**64
    ):
        raise SampleError(
            f"asset {asset_id!r} has {value!r} in column {column!r}, which is not "
            "an unsigned 64-bit H3 cell index"
        )
    return int(value)


def _point_cell(longitude: Any, latitude: Any, resolution: int) -> int:
    """The H3 cell of a point; `ImportError` without the `geometry` extra."""
    latitude = _coordinate("latitude", latitude, 90.0)
    longitude = _coordinate("longitude", longitude, 360.0)
    try:
        # Here, so that importing this module does not need the extra.
        from crc_sdk.geometry.h3 import point_to_cell

        return point_to_cell(longitude, latitude, resolution)
    except ImportError as error:
        raise ImportError(
            "H3Cell computes each asset's H3 cell from its point, which requires "
            "`pip install crc-sdk[geometry]`"
        ) from error


def _h3_cell(
    asset_id: str, value: Any, column: str
) -> tuple[str, tuple[float, float], tuple[tuple[float, float], ...]]:
    """An asset's H3 cell: its text, its centre and its boundary vertices.

    The centre and the vertices are `(latitude, longitude)` pairs in degrees.
    Raises `SampleError` for a missing or invalid cell, and `ImportError`
    without the `geometry` extra.
    """
    index = _h3_index(asset_id, value, column)
    try:
        import h3  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "NearestCell locates assets by their H3 cells (CellColumn) with the "
            "cells' geometry, which requires `pip install crc-sdk[geometry]`"
        ) from error
    text = str(h3.int_to_str(index))
    if not h3.is_valid_cell(text):
        raise SampleError(
            f"asset {asset_id!r} has {index} in column {column!r}, which is "
            "not a valid H3 cell"
        )
    latitude, longitude = h3.cell_to_latlng(text)
    vertices = tuple((float(lat), float(lon)) for lat, lon in h3.cell_to_boundary(text))
    return text, (float(latitude), float(longitude)), vertices


def _within_grid_cell(
    vertices: Sequence[tuple[float, float]],
    cell: Mapping[str, Any],
    spacing: Mapping[str, Any],
) -> bool:
    """Whether every vertex lies within half a grid step of the cell's centre.

    The test of `AnnualSeries.nearest_cell`, with its slack; longitudes are
    compared modulo 360.
    """
    slack = 1.0 + 1e-9
    half_lat = 0.5 * float(spacing["latitude"]) * slack
    half_lon = 0.5 * float(spacing["longitude"]) * slack
    centre_lat, centre_lon = float(cell["latitude"]), float(cell["longitude"])
    for latitude, longitude in vertices:
        d_lon = (longitude - centre_lon + 180.0) % 360.0 - 180.0
        if abs(latitude - centre_lat) > half_lat or abs(d_lon) > half_lon:
            return False
    return True


# -- the plan ----------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class ParametricPlan:
    """A lazy plan from hazard curves or per-year values to compared covers.

    Start one with `ParametricPlan.for_evaluation(request)` (fitted curves)
    or `ParametricPlan.for_annual_extremes(...)` (per-year values); the
    builders (`monte_carlo`, `index`, `basis_risk`, `covers`, `objective`,
    `pricing`, `split`, `metrics`, `scopes`, `dependence`, `allocation`) return
    new plans, and neither they nor `explain()` do any I/O. The execution
    points `samples()`, `joint_sample()`, `designs()` and `write_parquet(path)`
    read the source and the assets. The module docstring describes both routes.

    `entry` names the entry point. On the curve route `source` is the
    request, `unit` and `provenance` are `None` (the dataset's metadata
    states them at execution) and `draws` holds what `monte_carlo` states.
    `pricing_rule` holds what `.pricing(...)` states, `cell`, `trend` and
    `scenario` what `for_annual_extremes(...)` states, `index_read` what
    `.index(...)` states, and `scope_list`, `dependence_model` and
    `allocation_rule` what `.scopes(...)`, `.dependence(...)` and
    `.allocation(...)` state. `pool` holds what `for_annual_extremes(...)`
    states of a pool of model-years, and `source` is then a tuple of its sources
    when several were given.
    """

    source: Any
    assets: AssetPortfolio | Path
    loss_model: LossModel
    unit: str | None
    provenance: SourceProvenance | None
    location: NearestCell | StationColumn = field(default_factory=NearestCell)
    year_column: str = "year"
    hazard_column: str = "value"
    years: tuple[int, ...] | None = None
    missing: MissingYears = "raise"
    seed: int | None = None
    basis_risk_model: BasisRisk | None = None
    cover_list: tuple[Cover, ...] = ()
    design_spec: _DesignSpec | None = None
    split_rule: Holdout | KFold | FreshDraws | None = None
    metric_options: MetricOptions = field(default_factory=MetricOptions)
    entry: Literal["for_annual_extremes", "for_evaluation"] = "for_annual_extremes"
    draws: _MonteCarlo | None = None
    pricing_rule: Pricing | None = None
    cell: str | H3Cell | None = None
    index_read: _SeriesRead | None = None
    trend: Trend = field(default_factory=AsRecorded)
    scenario: str = "baseline"
    scope_list: tuple[Individual | Joint, ...] = (Individual(),)
    dependence_model: Dependence | None = None
    allocation_rule: Allocation | None = None
    pool: ModelPool | None = None

    # -- the entry points -----------------------------------------------------

    @classmethod
    def for_evaluation(
        cls,
        request: PortfolioEvaluation,
        *,
        value_column: str | None = None,
        value_unit: str = "1",
    ) -> ParametricPlan:
        """Plan covers on each asset's fitted hazard curve.

        `request` is the `PortfolioEvaluation` built with
        `HazardDataset.local(path).for_assets(assets).select(...)
        .impact(...)`; the plan reuses its dataset, assets, selection and
        impact, and changes nothing in it. Each asset needs one curve: the
        selection must leave one hazard, horizon and pathway. At execution,
        `request.asset_curves()` gives the asset's canonical curve row and
        `sample_curve` draws the simulated years that `.monte_carlo(n_draws,
        seed=...)` states, so `.monte_carlo(...)` is required.

        - The loss is `impact(hazard) × value`: the impact is the request's
          (an `ImpactExpression` keeps its `definition_hash`), the value the
          asset column `value_column` in `value_unit`, or 1 (the damage
          ratio, `value_unit="1"`). A framework `ImpactFunction` gets the
          context `request.write_parquet` gives it: its own, with the asset's
          matched H3 cell and the request's context columns on top.
        - The index is the curve's hazard at the asset, in the dataset's
          `value_unit`, so every cover's index must be in that unit
          (checked at execution).
        - The assets are an in-memory table or a local Parquet file (read at
          execution); the asset order is the table's row order.

        Raises `TypeError` for a request of another type (for a remote
        source, materialize it first and build the request on the
        `HazardDataset` it returns) or for SQL assets; `ValueError` for a
        request without an impact or a selection of several hazards,
        horizons or pathways; and `SampleError` for a missing asset column
        or an invalid loss model.
        """
        if not isinstance(request, PortfolioEvaluation):
            raise TypeError(
                "request must be a PortfolioEvaluation, as HazardDataset.local("
                "path).for_assets(assets) builds; for a remote source, "
                "materialize it first (plan.materialize(path) returns the "
                f"HazardDataset), got {type(request).__name__}"
            )
        spec = request.impact_spec
        if spec is None:
            raise ValueError(
                "the request has no impact; add .impact(function, name=..., "
                "value_unit=..., value_semantics=...) to state the loss"
            )
        selection = request.selection
        for name, values in (
            ("hazard_names", selection.hazard_names),
            ("horizons", selection.horizons),
            ("pathways", selection.pathways),
        ):
            if values is not None and len(values) > 1:
                raise ValueError(
                    f"the plan samples one curve per asset, but the request "
                    f"selects {len(values)} {name}; select one with "
                    f".select({name}=[...])"
                )
        portfolio = request.portfolio
        if isinstance(portfolio.data, str):
            raise TypeError(
                "the plan reads each asset's row at execution; pass the assets "
                "as an Arrow table or a Parquet file path, not SQL"
            )
        impact, context = _request_impact(spec)
        loss_model = LossModel(
            impact,
            value_column=value_column,
            value_unit=value_unit,
            context=context,
            name=spec.name,
        )
        columns = _table_columns(portfolio)
        if columns is not None:
            needed = [value_column] if value_column is not None else []
            if context is not None:
                needed += [name for _, name in context.items()]
            absent = [name for name in needed if name not in columns]
            if absent:
                raise SampleError(f"the asset table has no columns {absent}")
        return cls(
            source=request,
            assets=portfolio,
            loss_model=loss_model,
            unit=None,
            provenance=None,
            entry="for_evaluation",
        )

    @classmethod
    def for_annual_extremes(
        cls,
        source: Any,
        assets: Any,
        *,
        impact: Any,
        unit: str,
        provenance: SourceProvenance,
        value_column: str | None = None,
        value_unit: str = "1",
        context: ImpactContextColumns | TransformContext | None = None,
        location: NearestCell | StationColumn = NearestCell(),
        year_column: str = "year",
        hazard_column: str = "value",
        years: Sequence[int] | None = None,
        missing: MissingYears = "raise",
        seed: int | None = None,
        trend: Trend = AsRecorded(),
        scenario: str = "baseline",
        cell: str | H3Cell | None = None,
        pool: ModelPool | None = None,
    ) -> ParametricPlan:
        """Plan covers on per-year hazard values at each asset.

        `source` is a plan with an `annual_extremes()` execution point (for
        example `HazardDataset.era5(...)...canonicalize()`), an Arrow table
        in an annual-extremes layout, or a local Parquet file of one; it is
        read only at an execution point. `assets` is an `AssetPortfolio`, an
        in-memory table `AssetPortfolio` accepts, or a local Parquet file
        read at execution; the asset order is the table's row order.

        - `impact`, `value_column`, `value_unit` and `context` make the loss
          model, `LossModel(impact, value_column=..., value_unit=...,
          context=...)`: loss = `impact(hazard) × value`, with the value from
          the asset column, or 1 (the damage ratio, `value_unit="1"`).
        - `unit` is the unit of the table's values. It is checked against
          the impact's calibration here, against a `unit` column of the table
          and against the source plan's `explain()` at execution.
        - `provenance` states the source's licence, attribution and version
          (annual-extremes tables carry none); it identifies the series in
          every sample, so one source has one `provenance`. A pathway that
          the source plan's `explain()` states, as a CMIP6 plan's does, is
          recorded beside it at execution, so that the setup text states it.
        - `location` finds each asset's series: `NearestCell()` (the
          default), for assets located by a point or by an H3 cell that lies
          within one grid cell, or `StationColumn(name)`. `year_column` and
          `hazard_column` name the table's year and value columns.
        - `years` states the years every series must cover (by default the
          years the table holds); `missing="drop_year"` drops a year without
          a value and lists it, instead of raising.
        - `seed` seeds the basis-risk draws: one child of
          `numpy.random.SeedSequence(seed)` per asset, in asset order.
        - `cell` puts an H3 cell for each asset in the impact's context, on
          top of `context`: the name of an asset column of unsigned 64-bit
          H3 cells, or `H3Cell(resolution=r)` for the cell of the asset's
          point, which needs the `geometry` extra at execution. The series
          is still located by `location`. Each sample records the cell, not
          the point. `H3Cell` needs a point, so it cannot serve assets
          located by `CellColumn`.
        - `trend` states how the climate trend is treated: `AsRecorded()`,
          the default, `RecentYears(n)` or `LinearTrend(reference_year=None,
          min_years=10)` (`crc_sdk.parametric.trend`). `years` comes first
          and `RecentYears` takes the last n of them. With `.index(...)`, the
          hazard and the index each have their own fit, on the same years
          with the same reference year. Under a split, `LinearTrend` is
          fitted per fold: the module docstring says how.
        - `scenario` is your name for the rows, in the samples' tables and
          hashes and in the comparison's `scenario` column. It is not derived
          from the source: the `scenario` column of a CMIP6 table is its
          pathway, which the setup text states as well.
        - `pool` (`ModelPool()`) pools the years of several model runs, per
          asset, as `sample_annual_pool` does. `source` is then one table,
          file or plan whose rows carry the columns it names, or a sequence
          of sources, one per run. A run is named by those columns or, where
          a source has none, by its plan's `explain()`. Each run is located on
          its own grid, so the matched cell and the grid step are recorded per
          run. `years`, `missing` and `trend` apply to every run, and every
          run must then hold the same years. `.samples()` returns one pooled
          sample per asset. `.index(...)` and `StationColumn` are refused.

        Raises `TypeError` for arguments of the wrong type, `SampleError`
        for a unit the impact is not calibrated for, a missing asset column,
        an invalid `scenario` or a `cell` for an impact that takes no
        context, and `ValueError` for other invalid arguments, among them a
        `LinearTrend` with `fit_years`, which the plan sets per fold, a
        `cell` where the impact's context already holds one, and `pool` with
        `StationColumn`, since models have no stations. A registry
        impact that needs an H3 cell in its context (such as `rx1day`)
        raises `SampleError` unless `cell` states each asset's, the impact
        or `context` is a
        `TransformContext` with a cell, or the assets are located by their
        H3 cells (`CellColumn`) and `context` names context columns: without
        one of these, a point asset has no cell on this route.
        """
        if pool is None:
            _source_kind(source)
        elif not isinstance(pool, ModelPool):
            raise TypeError(f"pool must be a ModelPool, got {type(pool).__name__}")
        else:
            source = _pool_source(source)
        portfolio = _assets(assets)
        unit = _identifier("unit", unit)
        if not isinstance(provenance, SourceProvenance):
            raise TypeError(
                "provenance must be a SourceProvenance stating the source's "
                f"licence and attribution, got {type(provenance).__name__}"
            )
        loss_model = LossModel(
            impact, value_column=value_column, value_unit=value_unit, context=context
        )
        loss_model.check_hazard_unit(unit)
        if not isinstance(location, (NearestCell, StationColumn)):
            raise TypeError(
                "location must be NearestCell() or StationColumn(name), got "
                f"{type(location).__name__}"
            )
        if pool is not None and isinstance(location, StationColumn):
            raise ValueError(
                "pool= pools climate-model runs, and models have no stations; "
                "locate each asset with NearestCell()"
            )
        if cell is not None:
            _check_cell(cell, impact, context)
        columns = _table_columns(portfolio)
        if columns is not None:
            needed = [value_column] if value_column is not None else []
            if isinstance(location, StationColumn):
                needed.append(location.name)
            if isinstance(cell, str):
                needed.append(cell)
            if isinstance(context, ImpactContextColumns):
                needed += [name for _, name in context.items()]
            absent = [name for name in needed if name not in columns]
            if absent:
                raise SampleError(f"the asset table has no columns {absent}")
        factor = _registry_needs_cell(impact, context, portfolio, cell)
        if factor is not None:
            raise SampleError(
                f"impact {factor!r} needs an H3 cell in its context, and this "
                "route reads none for an asset; state each asset's cell with "
                "cell= (an asset column, or H3Cell(resolution=r)), give the "
                "impact a context with the cell (crc_framework.impacts."
                "for_factor(..., context=TransformContext(cell=...))), or pass "
                "context= a TransformContext with it"
            )
        _column_name("year_column", year_column)
        _column_name("hazard_column", hazard_column)
        window: tuple[int, ...] | None = None
        if years is not None:
            if isinstance(years, (str, bytes)) or not isinstance(years, Sequence):
                raise TypeError("years must be a sequence of integer years")
            if not years or not all(
                isinstance(year, (int, np.integer)) and not isinstance(year, bool)
                for year in years
            ):
                raise ValueError("years must be a non-empty sequence of integer years")
            window = tuple(sorted(int(year) for year in years))
            if len(set(window)) != len(window):
                raise ValueError("years must not repeat")
        if missing not in get_args(MissingYears):
            raise ValueError(
                f"missing must be one of {get_args(MissingYears)}, got {missing!r}"
            )
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
        ):
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        trend = _require_trend(trend)
        if isinstance(trend, LinearTrend) and trend.fit_years is not None:
            raise ValueError(
                "LinearTrend(fit_years=...) states the fit years of one sample; a "
                "plan sets them per fold, so leave them out"
            )
        scenario = _identifier("scenario", scenario)
        return cls(
            source=source,
            assets=portfolio,
            loss_model=loss_model,
            unit=unit,
            provenance=provenance,
            location=location,
            year_column=year_column,
            hazard_column=hazard_column,
            years=window,
            missing=missing,
            seed=None if seed is None else int(seed),
            cell=cell,
            trend=trend,
            scenario=scenario,
            pool=pool,
        )

    # -- builders ---------------------------------------------------------------

    @property
    def _on_curves(self) -> bool:
        return self.entry == "for_evaluation"

    def monte_carlo(
        self,
        n_draws: int,
        *,
        seed: int,
        assume_semantics: AnnualSemantics | None = None,
        outside_support: OutsideSupport = "raise",
    ) -> ParametricPlan:
        """State the simulated years: `n_draws` draws from each asset's curve.

        Required on plans on fitted curves (`for_evaluation`). Each draw is
        one year with probability `1/n_draws`, as `sample_curve` draws it.
        `seed` is the plan's seed: one child of
        `numpy.random.SeedSequence(seed)` per asset, in asset order, draws
        the hazard and then the basis-risk model. It is recorded in every
        evaluation's `details` as `"plan_seed"`.

        A draw is a year only if the curve describes one year's value, so
        the dataset's `probability_semantics` must be `annual_exceedance` or
        `annual_value_distribution`. A dataset that declares none is refused
        at execution unless `assume_semantics` states one of the two; a
        stated value that contradicts the declared one raises.
        `outside_support` is what a tabulated curve does with draws outside
        its knots: `"raise"` (the default), `"extrapolate"` or `"clamp"`
        (`sample_curve`). Replaces any draws stated before.

        Raises `ValueError` on a plan on per-year values, for invalid counts,
        seeds or `outside_support`, or a seed equal to the fresh draws';
        `ProbabilitySemanticsError` for another `assume_semantics`.
        """
        if not self._on_curves:
            raise ValueError(
                "monte_carlo draws simulated years from fitted curves; this plan "
                "reads per-year values (for_annual_extremes)"
            )
        count = _positive_count("n_draws", n_draws)
        plan_seed = _seed_value("seed", seed)
        if assume_semantics is not None and assume_semantics not in get_args(
            AnnualSemantics
        ):
            raise ProbabilitySemanticsError(
                f"assume_semantics must be one of {get_args(AnnualSemantics)}, "
                f"got {assume_semantics!r}"
            )
        if outside_support not in get_args(OutsideSupport):
            raise ValueError(
                f"outside_support must be one of {get_args(OutsideSupport)}, "
                f"got {outside_support!r}"
            )
        rule = self.split_rule
        if isinstance(rule, FreshDraws) and rule.seed == plan_seed:
            raise ValueError(
                f"seed {plan_seed} is also the fresh draws' seed; fresh draws need "
                "another seed, or they repeat the design draws"
            )
        draws = _MonteCarlo(
            n_draws=count,
            assume_semantics=assume_semantics,
            outside_support=outside_support,
        )
        return dataclasses.replace(self, seed=plan_seed, draws=draws)

    def basis_risk(self, model: BasisRisk) -> ParametricPlan:
        """State the basis-risk model every sample is built with.

        `NoBasisRisk()` (the index is the hazard at the asset; samples warn
        with `ComonotoneIndexWarning`, except with `.index(...)`, where the
        index is another series), `ImpactNoise(sigma, upper=...)` or a
        user model that implements `BasisRisk`. Required before execution.
        Raises `TypeError` for another type, and `ValueError` for
        `ImpactNoise` on a plan on per-year values without a seed (on fitted
        curves, `monte_carlo` states the seed).
        """
        if not isinstance(model, BasisRisk):
            raise TypeError(
                "model must be a BasisRisk model such as NoBasisRisk() or "
                f"ImpactNoise(sigma), got {type(model).__name__}"
            )
        if isinstance(model, ImpactNoise) and self.seed is None and not self._on_curves:
            raise ValueError(
                "ImpactNoise draws one factor per year; give the plan a seed "
                "(for_annual_extremes(..., seed=...))"
            )
        return dataclasses.replace(self, basis_risk_model=model)

    def covers(self, *covers: Cover) -> ParametricPlan:
        """Set the covers to compare: resolved covers and templates.

        A template (a cover with `Free` parameters) is designed under each
        objective of `.objective(...)`; a resolved cover is evaluated as it
        is. Each cover pays in the loss unit on an index read at the asset
        (`AtAsset()`), in the table's unit on per-year values; on fitted
        curves the index unit is checked against the dataset's `value_unit`
        at execution. With `.index(...)`, stated first, the index unit is
        checked against its `unit`, and a cover may also name the location
        that the index is read at (`AtStation` or `AtPoint`, the kind that
        `.index`'s `location` resolves to), which is checked at execution.
        Under `Joint()` a cover must be written for no asset, as it applies to
        every asset. Replaces any covers set before.

        Raises `TypeError` for a non-cover, `SampleError` for a cover in
        other units or on an index read elsewhere, and `ValueError` for no
        covers or the same cover twice.
        """
        if not covers:
            raise ValueError("covers needs at least one cover")
        seen: set[tuple[str, str | None]] = set()
        for cover in covers:
            if not isinstance(cover, Cover):
                raise TypeError(f"expected a Cover, got {type(cover).__name__}")
            self._check_cover(cover)
            key = (cover.definition_hash, cover.asset_id)
            if key in seen:
                raise ValueError(
                    f"cover {cover.definition_hash} is given twice; each cover "
                    "is compared once"
                )
            seen.add(key)
        return dataclasses.replace(self, cover_list=tuple(covers))

    def _check_cover(self, cover: Cover) -> None:
        """Check a cover against the loss unit and the index the plan reads."""
        name = cover.label or cover.definition_hash
        if cover.payout_unit != self.loss_model.value_unit:
            raise SampleError(
                f"cover {name!r} pays in {cover.payout_unit!r}, but the loss is in "
                f"{self.loss_model.value_unit!r}"
            )
        read = self.index_read
        unit = self.unit if read is None else read.unit
        if unit is not None and cover.index.unit != unit:
            values = "table's" if read is None else "index series'"
            raise SampleError(
                f"cover {name!r} pays on an index in {cover.index.unit!r}, but the "
                f"{values} values are in {unit!r}"
            )
        location = cover.index.location
        if isinstance(location, AtAsset):
            return
        if read is None:
            raise SampleError(
                "this plan reads the index at each asset (AtAsset()); "
                + (
                    _CURVES_READ_AT_THE_ASSET
                    if self._on_curves
                    else "to read it elsewhere, state where with .index(source, "
                    "location=..., unit=..., provenance=...) before .covers(...)"
                )
            )
        expected = AtStation if isinstance(read.location, StationColumn) else AtPoint
        if not isinstance(location, expected):
            raise SampleError(
                f"cover {name!r} names the location {location!r}, but .index(...) "
                f"resolves each asset's to an {expected.__name__}; name that one, "
                "or state AtAsset()"
            )

    def _index_refusal(self) -> str | None:
        """Why this plan cannot read its index elsewhere; `None` if it can."""
        if self._on_curves:
            return f".index(...) reads per-year values; {_CURVES_READ_AT_THE_ASSET}"
        if self.pool is not None:
            return (
                ".index(...) cannot read an index for a pool of model-years: an "
                "observed station and a free-running model share no weather, so "
                "pairing them by year would invent a dependence"
            )
        return None

    def index(
        self,
        source: Any,
        *,
        location: NearestCell | StationColumn,
        unit: str,
        provenance: SourceProvenance,
        point: PointColumns | None = None,
        year_column: str = "year",
        index_column: str = "value",
    ) -> ParametricPlan:
        """Read the index elsewhere: the series each cover pays on.

        For plans on per-year values. The covers pay on a series read from
        `source` at `location`; the loss still reads the hazard of
        `for_annual_extremes`. Each asset's sample is that of
        `sample_annual(hazard, index=series, ...)`, so `NoBasisRisk()` does
        not warn, and `.basis_risk(...)` is required all the same.

        - `source` takes what `for_annual_extremes` takes: a plan with an
          `annual_extremes()` execution point, an Arrow table or a local
          Parquet file; it is read only at an execution point.
        - `location` is required: `StationColumn(name)`, the station an asset
          column names, or `NearestCell(...)`, the nearest cell of the index
          source's own grid, accepted as for the hazard. `NearestCell` reads
          at the asset's own location by default; `point=PointColumns(longitude,
          latitude)` names asset columns holding another point, such as a
          reference site. Only the matched cell is recorded, not the point,
          so an index grid finer than the hazard's records an asset's place
          to within its own cell.
        - `unit` is the index series' unit. It is checked against a `unit`
          column of the table and the source plan's `explain()`, and not
          against the impact, because the loss does not read it. A cover's
          index unit is checked against it, so state `.index(...)` before
          `.covers(...)` when the two units differ or a cover names a
          location; stated after, it checks the covers again.
        - `provenance` states the index source's licence, attribution and
          version, as for the hazard. `year_column` and `index_column` name
          the table's year and value columns.

        **The years.** The index is read on exactly the hazard's years. A
        hazard year the index lacks or holds as null is a missing year,
        under the plan's one `missing=`: it raises and is listed, or with
        `missing="drop_year"` it is dropped from the sample and listed in
        `dropped_years`. An index year the hazard does not have is not used,
        and `unused_index_years` lists it.

        **The covers.** At execution, a cover stated with `AtAsset()` is
        bound to each asset's resolved location: `AtStation` with the asset's
        station id, or `AtPoint` at the matched cell's centre, its longitude
        wrapped to [-180, 180]. The location is thus in its document and in
        `definition_hash` and `design_id`, and the asset's own point is not.
        A cover that already names a location must name that one, or
        execution raises `SampleError`. `explain()` shows the covers as
        stated and the location rule; the written metadata holds the bound
        documents. Replaces any index stated before.

        Raises `ValueError` on a plan on fitted curves, whose index at
        another location would need a joint distribution of two curves, on a
        plan that pools model-years, for `point=` with `StationColumn` and
        for empty column names; `TypeError`
        for arguments of the wrong type; and `SampleError` for a missing
        asset column, or a cover the new index does not fit.
        """
        refusal = self._index_refusal()
        if refusal is not None:
            raise ValueError(refusal)
        _source_kind(source)
        if not isinstance(location, (NearestCell, StationColumn)):
            raise TypeError(
                "location must be NearestCell() or StationColumn(name), got "
                f"{type(location).__name__}"
            )
        if point is not None:
            if not isinstance(point, PointColumns):
                raise TypeError(
                    "point must be PointColumns(longitude, latitude) naming asset "
                    f"columns, got {type(point).__name__}"
                )
            if isinstance(location, StationColumn):
                raise ValueError(
                    "point= says where NearestCell reads; StationColumn reads "
                    "the station the asset column names"
                )
            _column_name("point.longitude", point.longitude)
            _column_name("point.latitude", point.latitude)
        if not isinstance(provenance, SourceProvenance):
            raise TypeError(
                "provenance must be a SourceProvenance stating the index source's "
                f"licence and attribution, got {type(provenance).__name__}"
            )
        _column_name("year_column", year_column)
        _column_name("index_column", index_column)
        columns = _table_columns(self.assets)
        if columns is not None:
            needed = (
                [location.name]
                if isinstance(location, StationColumn)
                else ([] if point is None else [point.longitude, point.latitude])
            )
            absent = [name for name in needed if name not in columns]
            if absent:
                raise SampleError(f"the asset table has no columns {absent}")
        read = _SeriesRead(
            role="index",
            source=source,
            location=location,
            unit=_identifier("unit", unit),
            provenance=provenance,
            year_column=year_column,
            value_column=index_column,
            point=point,
        )
        plan = dataclasses.replace(self, index_read=read)
        for cover in self.cover_list:
            plan._check_cover(cover)
        return plan

    def objective(
        self,
        *objectives: Objective,
        pricing: Pricing | None = None,
        budget: Budget | None = None,
        limit: Limit | None = None,
        options: DesignOptions = DesignOptions(),
    ) -> ParametricPlan:
        """Design the templates: under each objective, with these terms.

        `objectives` are `MSE()`, `ExpectedUncovered()`,
        `WeightedCVaR(alpha, beta)` or `UserObjective`s; each template is
        designed under each, on each design block. `pricing` prices the
        premium of the designs on their design block, and of the resolved
        covers on each evaluation block. It is the plan's pricing, which
        `.pricing(...)` states too: without it here, `.pricing(...)`'s
        applies, and without both `PurePremium()`. `budget`, `limit` and
        `options` are those of `design`. Without `.objective(...)` templates
        cannot be evaluated. Replaces any objective set before.

        Raises `TypeError` for arguments of the wrong type, and `ValueError`
        for no objective, the same objective twice, or a pricing that
        differs from the one `.pricing(...)` states.
        """
        if not objectives:
            raise ValueError("objective needs at least one objective")
        kinds = (MSE, ExpectedUncovered, WeightedCVaR, UserObjective)
        documents: list[str] = []
        for objective in objectives:
            if not isinstance(objective, kinds):
                raise TypeError(
                    "objectives must be MSE(), ExpectedUncovered(), "
                    "WeightedCVaR(alpha, beta) or UserObjective, got "
                    f"{type(objective).__name__}"
                )
            text = json.dumps(objective.to_document(), sort_keys=True)
            if text in documents:
                raise ValueError(f"objective {text} is given twice")
            documents.append(text)
        if pricing is not None:
            pricing = _check_pricing(pricing)
            stated = self.pricing_rule
            if stated is not None and not _same_pricing(stated, pricing):
                raise _pricing_conflict(stated, pricing)
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError(f"budget must be a Budget, got {type(budget).__name__}")
        if limit is not None and not isinstance(limit, Limit):
            raise TypeError(f"limit must be a Limit, got {type(limit).__name__}")
        if not isinstance(options, DesignOptions):
            raise TypeError(
                f"options must be DesignOptions, got {type(options).__name__}"
            )
        spec = _DesignSpec(
            objectives=tuple(objectives),
            pricing=pricing,
            budget=budget,
            limit=limit,
            options=options,
        )
        return dataclasses.replace(self, design_spec=spec)

    def pricing(self, pricing: Pricing) -> ParametricPlan:
        """Price the premiums: `PurePremium()` or `CostOfCapitalLoad(rate, q)`.

        One pricing prices every premium of the plan: the resolved covers'
        on each evaluation block and the designs' on their design block, so
        that every row of the comparison is priced alike. This builder
        states it for a plan of resolved covers, which needs no
        `.objective(...)`; `.objective(..., pricing=...)` states it too, and
        both may be given when they state the same pricing. Without either,
        the premiums are pure (`PurePremium()`). `explain()` shows the
        pricing in every planned evaluation and in the design terms.
        Replaces any pricing set before with `.pricing(...)`.

        Raises `TypeError` for another type, and `ValueError` for a pricing
        that differs from the one `.objective(..., pricing=...)` states.
        """
        chosen = _check_pricing(pricing)
        spec = self.design_spec
        if (
            spec is not None
            and spec.pricing is not None
            and not _same_pricing(chosen, spec.pricing)
        ):
            raise _pricing_conflict(chosen, spec.pricing)
        return dataclasses.replace(self, pricing_rule=chosen)

    def split(self, rule: Holdout | KFold | FreshDraws) -> ParametricPlan:
        """Hold out data: `FreshDraws(n_draws, seed=)`, `Holdout(f)` or `KFold(k)`.

        Every design block gets in-sample rows and every held-out block (or
        the fresh draws) out-of-sample rows. `FreshDraws` needs a plan on
        fitted curves; `Holdout` and `KFold` split per-year samples by year
        and Monte Carlo samples by draw; a `LinearTrend` is then fitted per
        fold, on the design years only. Raises `TypeError` for another rule
        or fresh draws on per-year values, `ValueError` for fresh draws with
        the plan's seed, and `SampleError` for `Holdout` or `KFold` with
        `.dependence(EmpiricalCopula(...))`, which only fresh draws can test.
        """
        if not isinstance(rule, (Holdout, KFold, FreshDraws)):
            raise TypeError(
                "rule must be FreshDraws(n_draws, seed=...), Holdout(train_fraction) "
                f"or KFold(k), got {type(rule).__name__}"
            )
        if isinstance(rule, FreshDraws):
            if not self._on_curves:
                raise TypeError(
                    "fresh draws need a Monte Carlo sample from a curve "
                    "(for_evaluation); split per-year values with Holdout or KFold"
                )
            if rule.seed == self.seed:
                raise ValueError(
                    f"seed {rule.seed} is also the plan's seed; fresh draws need "
                    "another seed, or they repeat the design draws"
                )
        elif isinstance(self.dependence_model, EmpiricalCopula):
            raise _copula_split_error(rule)
        return dataclasses.replace(self, split_rule=rule)

    def metrics(self, options: MetricOptions) -> ParametricPlan:
        """Set the metrics' levels and thresholds (`MetricOptions`).

        Raises `TypeError` for another type.
        """
        if not isinstance(options, MetricOptions):
            raise TypeError(
                f"options must be MetricOptions, got {type(options).__name__}"
            )
        return dataclasses.replace(self, metric_options=options)

    def scopes(self, *scopes: Individual | Joint) -> ParametricPlan:
        """State how the assets are designed: `Individual()` and/or `Joint()`.

        `Individual()`, the default, designs and evaluates each asset on its
        own sample. `Joint()` designs each template once per objective for one
        objective and one budget on the asset owner's total (`design_joint`),
        one cover per asset, on one sample of all the assets, and every
        evaluation gets a row at the level of that total. A plan may hold both:
        the `scope` column keeps the rows of each scope apart.

        `Joint()` needs at least two assets; on fitted curves it also needs
        `.dependence(...)`, and each cover must be written for no asset, as it
        applies to every asset. Those are checked by `explain()` and at every
        execution point, since builders may come in any order. Replaces any
        scopes stated before.

        Raises `TypeError` for another type, and `ValueError` for no scope, a
        scope given twice, or `Joint()` on fewer than two assets (when the
        asset table is in memory; otherwise at execution).
        """
        if not scopes:
            raise ValueError("scopes needs at least one scope")
        for scope in scopes:
            if not isinstance(scope, (Individual, Joint)):
                raise TypeError(
                    "scopes must be Individual() or Joint(), got "
                    f"{type(scope).__name__}"
                )
        kinds = [scope.to_document()["kind"] for scope in scopes]
        repeated = sorted({kind for kind in kinds if kinds.count(kind) > 1})
        if repeated:
            raise ValueError(f"scope {repeated[0]!r} is given twice")
        plan = dataclasses.replace(self, scope_list=tuple(scopes))
        ids = self._known_asset_ids()
        if ids is not None and plan._holds_joint:
            _require_several_assets(ids)
        return plan

    def dependence(self, model: Dependence) -> ParametricPlan:
        """State the dependence between the assets' curve draws, under `Joint()`.

        For plans on fitted curves, where it is required with `Joint()`: a
        fitted curve says nothing about how two places move together, so
        `Independent()`, `Comonotonic()` or `EmpiricalCopula(reference)` is an
        assumption the plan states, in its metadata and its setup text. The
        draws are those of `sample_curves`, with the plan's generators. Without
        `Joint()` nothing puts the assets side by side, so `explain()` and every
        execution point raise. Replaces any dependence stated before.

        Raises `ValueError` on a plan on per-year values, whose dependence is
        the data's own: the rows are the same years of the data. Raises
        `TypeError` for another type, and `SampleError` for
        `EmpiricalCopula(...)` under `Holdout` or `KFold`.
        """
        if not self._on_curves:
            raise ValueError(
                "a plan on per-year values takes no dependence model: its "
                "dependence is the data's own, the rows being the same years of "
                "the data"
            )
        if not isinstance(model, (Independent, Comonotonic, EmpiricalCopula)):
            raise TypeError(
                "model must be Independent(), Comonotonic() or "
                f"EmpiricalCopula(reference), got {type(model).__name__}"
            )
        rule = self.split_rule
        if isinstance(model, EmpiricalCopula) and isinstance(rule, (Holdout, KFold)):
            raise _copula_split_error(rule)
        return dataclasses.replace(self, dependence_model=model)

    def allocation(self, rule: Allocation) -> ParametricPlan:
        """Add a joint design's per-asset rows, under `rule`: `"value"` or `"loss"`.

        By default a joint design has no per-asset rows, because the split of
        the total payout across the assets is the solver's choice. A rule is an
        accounting statement about that total payout, not the payout of an
        asset's cover: `"value"` shares each draw's total payout in proportion
        to the assets' values and `"loss"` in proportion to their losses in
        that draw. The rows have `level` `"asset_by_value"` or
        `"asset_by_loss"` and no premium or net-loss columns. The values are
        those the plan's loss model reads (the asset column `value_column`, or
        1 for a damage ratio); only their hash is written. It applies to joint
        designs only, so `explain()` and every execution point raise without
        `Joint()`. Replaces any allocation stated before.

        Raises `ValueError` for another rule.
        """
        if rule not in get_args(Allocation):
            raise ValueError(
                f"rule must be one of {get_args(Allocation)}, got {rule!r}"
            )
        return dataclasses.replace(self, allocation_rule=rule)

    # -- scopes -----------------------------------------------------------------

    @property
    def _holds_joint(self) -> bool:
        return any(isinstance(scope, Joint) for scope in self.scope_list)

    @property
    def _holds_individual(self) -> bool:
        return any(isinstance(scope, Individual) for scope in self.scope_list)

    def _check_scopes(self) -> None:
        """Check what no single builder can, as the builders come in any order."""
        model = self.dependence_model
        if not self._holds_joint:
            if model is not None:
                raise ValueError(
                    ".dependence(...) says how the assets are put side by side, "
                    "and only Joint() does that; state .scopes(Joint())"
                )
            if self.allocation_rule is not None:
                raise ValueError(
                    ".allocation(...) shares out a joint design's total payout; "
                    "state .scopes(Joint())"
                )
            return
        if model is None and self._on_curves:
            raise ValueError(
                "Joint() on fitted curves needs a stated dependence between the "
                "assets, because a fitted curve says nothing about how two places "
                "move together: state .dependence(Independent()), "
                ".dependence(Comonotonic()) or .dependence(EmpiricalCopula(reference))"
            )
        for cover in self.cover_list:
            if cover.asset_id is not None:
                raise ValueError(
                    f"cover {cover.label or cover.definition_hash!r} is written for "
                    f"asset {cover.asset_id!r}, but under Joint() each cover applies "
                    "to every asset; write it for no asset, or give different terms "
                    "per asset to design_joint"
                )
        if isinstance(model, EmpiricalCopula):
            ids = self._known_asset_ids()
            columns = model.reference.asset_ids if ids is None else tuple(ids)
            if self.draws is not None:
                model.record(self.draws.n_draws, columns)
            if isinstance(self.split_rule, FreshDraws):
                model.record(self.split_rule.n_draws, columns)

    def _dependence_document(self) -> dict[str, Any]:
        """Where the dependence under `Joint()` comes from, as far as it is known."""
        model = self.dependence_model
        if model is None:
            return {"kind": "data"}
        if self.draws is None:
            return model.to_document()
        ids = self._known_asset_ids()
        columns: tuple[str, ...] = () if ids is None else tuple(ids)
        if isinstance(model, EmpiricalCopula) and ids is None:
            columns = model.reference.asset_ids
        return model.record(self.draws.n_draws, columns)

    def _planned_rows(self) -> dict[str, Any]:
        """The levels, scopes and allocation of the rows under `Joint()`."""
        designed = any(cover.free_parameters() for cover in self.cover_list)
        fixed = any(not cover.free_parameters() for cover in self.cover_list)
        individual = fixed or (designed and self._holds_individual)
        joint = designed or not self.cover_list
        rule = self.allocation_rule
        levels = [
            *(["asset"] if individual else []),
            "total",
            *([f"asset_by_{rule}"] if joint and rule is not None else []),
        ]
        scopes = [
            *(["individual"] if individual else []),
            *(["joint"] if joint else []),
        ]
        return {"level": levels, "scope": scopes, "allocation": rule}

    # -- explain ----------------------------------------------------------------

    def _pricing(self) -> Pricing:
        """The plan's pricing: `.objective`'s, else `.pricing`'s, else pure."""
        spec = self.design_spec
        if spec is not None and spec.pricing is not None:
            return spec.pricing
        return PurePremium() if self.pricing_rule is None else self.pricing_rule

    def _known_asset_ids(self) -> list[str] | None:
        assets = self.assets
        if not isinstance(assets, AssetPortfolio) or not isinstance(
            assets.data, pa.Table
        ):
            return None
        if assets.id_column not in assets.data.column_names:
            return None
        column = assets.data.column(assets.id_column)
        if column.null_count:
            return None
        return [str(value) for value in column.to_pylist()]

    def _seed_details(self) -> dict[str, Any]:
        """`{"plan_seed": seed}` for a plan with a seed, else `{}`."""
        return {} if self.seed is None else {"plan_seed": self.seed}

    def _evaluations(self) -> list[dict[str, Any]]:
        pricing = self._pricing().to_document()
        rule = self.split_rule
        if rule is None:
            folds, method, details = 1, None, {}
        elif isinstance(rule, Holdout):
            folds, method = 1, "holdout"
            details = {"train_fraction": rule.train_fraction}
        elif isinstance(rule, FreshDraws):
            folds, method = 1, "fresh_draws"
            details = {"fresh_draws_seed": rule.seed}
        else:
            folds, method, details = rule.k, "kfold", {"k": rule.k}
        rows = self._planned_rows() if self._holds_joint else {}
        evaluations: list[dict[str, Any]] = []
        for fold in range(folds):
            evaluations.append(
                {
                    **rows,
                    "details": self._seed_details(),
                    "fold": fold,
                    "method": "in_sample",
                    "pricing": pricing,
                    "split": "in_sample",
                }
            )
            if method is not None:
                evaluations.append(
                    {
                        **rows,
                        "details": {**details, **self._seed_details()},
                        "fold": fold,
                        "method": method,
                        "pricing": pricing,
                        "split": "out_of_sample",
                    }
                )
        return evaluations

    def _source_descriptions(self) -> list[dict[str, Any]]:
        """The hazard's sources, as far as they are known: one entry per source."""
        if self.pool is None:
            return [self._source_description(self.source)]
        pooled = self.pool.to_document()
        return [
            {**self._source_description(source), "pool": pooled}
            for source in self._pool_sources()
        ]

    def _source_description(self, source: Any) -> dict[str, Any]:
        if self._on_curves:
            request: PortfolioEvaluation = source
            selection = request.selection
            spec = request.impact_spec
            assert spec is not None
            return {
                "impact": {
                    "name": spec.name,
                    "value_semantics": spec.value_semantics,
                    "value_unit": spec.value_unit,
                },
                "join": "PortfolioEvaluation.asset_curves()",
                "kind": "canonical_dataset",
                "metadata": None,
                "path": os.fspath(request.dataset.provider.source),
                "selection": {
                    "hazard_names": _listed(selection.hazard_names),
                    "horizons": _listed(selection.horizons),
                    "pathways": _listed(selection.pathways),
                },
                "type": type(request.dataset).__name__,
            }
        kind = _source_kind(source)
        assert self.provenance is not None
        description: dict[str, Any] = {
            "columns": {"value": self.hazard_column, "year": self.year_column},
            "kind": kind,
            "location": self.location.to_document(),
            "missing": self.missing,
            "provenance": self.provenance.model_dump(mode="json"),
            "role": "hazard",
            "type": type(source).__name__,
            "unit": self.unit,
            "years": None if self.years is None else list(self.years),
        }
        if kind == "file":
            description["path"] = os.fspath(source)
        elif kind == "table":
            description["rows"] = int(source.num_rows)
        return description

    def _index_description(self, read: _SeriesRead) -> dict[str, Any]:
        """`.index(...)`'s source, in the form of a written `sources` entry."""
        kind = _source_kind(read.source)
        description: dict[str, Any] = {
            "columns": {"value": read.value_column, "year": read.year_column},
            "location": read.location.to_document(),
            "point": (
                None
                if read.point is None
                else {
                    "latitude_column": read.point.latitude,
                    "longitude_column": read.point.longitude,
                }
            ),
            "role": "index",
            "series": {"source": read.provenance.model_dump(mode="json")},
            "source_kind": kind,
            "type": type(read.source).__name__,
            "unit": read.unit,
        }
        if kind == "file":
            description["path"] = os.fspath(read.source)
        elif kind == "table":
            description["rows"] = int(read.source.num_rows)
        return description

    def _sample_description(self) -> dict[str, Any]:
        """One asset's sample, as far as it is known before execution."""
        read = self.index_read
        notes: list[str] = []
        if self.cell is not None:
            notes.append(
                f"The impact reads each asset's H3 cell: {_cell_origin(self.cell)}."
            )
        if read is not None:
            notes.append(f"The index is read elsewhere: {_index_origin(read)}.")
        sample: dict[str, Any] = {
            "index_unit": self.unit if read is None else read.unit,
            "loss_model": self.loss_model.to_document(),
            "loss_unit": self.loss_model.value_unit,
            "notes": notes,
            "scenario": self.scenario,
            "semantics": "annual",
            "semantics_source": "data",
            "source_kind": "annual",
        }
        if self._on_curves:
            draws = self.draws
            assumed = None if draws is None else draws.assume_semantics
            sample.update(
                {
                    "draws": None if draws is None else draws.n_draws,
                    "probability_semantics": assumed,
                    "semantics_source": "dataset" if assumed is None else "caller",
                    "source_kind": "monte_carlo",
                }
            )
        else:
            sample["trend"] = self.trend.to_document()
            if self.pool is not None:
                sample["pool"] = self.pool.to_document()
        if self.basis_risk_model is not None:
            sample["basis_risk"] = self.basis_risk_model.to_document()
        if self._holds_joint:
            sample["dependence"] = self._dependence_document()
            if self.dependence_model is not None and self.draws is not None:
                notes.append(
                    self.dependence_model.statement(self.draws.n_draws, "assets")
                )
        return sample

    def _rng_description(self) -> dict[str, Any]:
        generators = (
            "one child of numpy.random.SeedSequence(seed) per asset, in asset order"
        )
        if not self._on_curves:
            return {
                "generators": generators,
                "seed": self.seed,
                "steps": ["basis_risk"],
            }
        rule = self.split_rule
        return {
            "fresh_draws_seed": rule.seed if isinstance(rule, FreshDraws) else None,
            "generators": generators
            + "; fresh draws: one child of SeedSequence(fresh_draws_seed) per asset",
            "seed": self.seed,
            "steps": ["hazard", "basis_risk"],
        }

    def _explain_document(self) -> dict[str, Any]:
        ids = self._known_asset_ids()
        sample = self._sample_description()
        if ids is None:
            samples = [{**sample, "asset_ids": []}]
        elif self._holds_joint:
            samples = [{**sample, "asset_ids": ids}]
        else:
            samples = [{**sample, "asset_ids": [asset_id]} for asset_id in ids]
        spec = self.design_spec
        designs = None
        if spec is not None:
            designs = {
                **spec.to_document(self._pricing()),
                "templates": [
                    cover.definition_hash
                    for cover in self.cover_list
                    if cover.free_parameters()
                ],
            }
        assets = self.assets
        if isinstance(assets, AssetPortfolio):
            location: Any = repr(assets.location)
            id_column: str | None = assets.id_column
        else:
            location, id_column = None, None
        if isinstance(assets, Path):
            path: str | None = os.fspath(assets)
        elif isinstance(assets.data, os.PathLike):
            path = os.fspath(assets.data)
        else:
            path = None
        provenance = self.provenance
        licences: list[dict[str, Any]] | None = (
            None
            if provenance is None
            else [
                {
                    "attribution": provenance.attribution,
                    "licence": provenance.licence,
                    "role": "hazard",
                    "source": provenance.dataset,
                }
            ]
        )
        read = self.index_read
        if licences is not None and read is not None:
            licences.append(
                {
                    "attribution": read.provenance.attribution,
                    "licence": read.provenance.licence,
                    "role": "index",
                    "source": read.provenance.dataset,
                }
            )
        document: dict[str, Any] = {
            "assets": {
                "count": None if ids is None else len(ids),
                "id_column": id_column,
                "location": location,
                "path": path,
                "value_column": self.loss_model.value_column,
            },
            "covers": [
                {
                    "asset_id": cover.asset_id,
                    "document": cover.to_document(),
                    "free_parameters": list(cover.free_parameters()),
                    "hash": cover.definition_hash,
                    "label": cover.label,
                }
                for cover in self.cover_list
            ],
            "designs": designs,
            "entry": self.entry,
            "execution": _EXECUTION,
            "licences": licences,
            "producer": "crc_sdk.parametric",
            "rng": self._rng_description(),
            "setup": {
                "basis": "annual",
                "evaluations": self._evaluations(),
                "metric_options": self.metric_options.to_document(),
                "samples": samples,
            },
            "sources": [
                *self._source_descriptions(),
                *([] if read is None else [self._index_description(read)]),
            ],
        }
        if self._holds_joint:
            model = self.dependence_model
            document["scopes"] = [scope.to_document() for scope in self.scope_list]
            document["dependence"] = None if model is None else model.to_document()
            document["allocation"] = self.allocation_rule
        if isinstance(self.cell, H3Cell):
            document["assets"]["cell"] = self.cell.to_document()
        elif self.cell is not None:
            document["assets"]["cell"] = {"kind": "column", "column": self.cell}
        if self._on_curves:
            rule = self.split_rule
            document["monte_carlo"] = {
                **(
                    {"assume_semantics": None, "n_draws": None, "outside_support": None}
                    if self.draws is None
                    else self.draws.to_document()
                ),
                "fresh_draws": (
                    rule.to_document() if isinstance(rule, FreshDraws) else None
                ),
                "seed": self.seed,
            }
        return document

    def explain(
        self, *, format: Literal["text", "json"] = "text"
    ) -> str | dict[str, Any]:
        """Describe the plan: its setup as far as it is known, without I/O.

        The document mirrors the metadata the plan writes
        (`EvaluationMetadata`): `setup` (the basis, the metric options, one
        sample per asset with its source, loss and basis-risk models and, on
        per-year values, its `trend` option and any `pool` columns, and the
        planned evaluations with
        their split, fold and pricing), `covers`, `designs` (the objectives
        and design terms), `sources`, `assets`, `licences`, `rng` (the seeds)
        and `entry`; with
        `.index(...)`, `sources` has an entry with role `"index"` that states
        the location rule, and `covers` holds the covers as stated; a plan on
        fitted curves adds `monte_carlo` (the draws, the semantics stated by
        the caller, `outside_support` and the fresh draws), and a plan with
        `cell=` adds what it states to `assets` as `cell`. Its `setup` has
        the layout of `EvaluationSetup`, so `setup_text(plan.explain(
        format="json")["setup"])` states it before anything is read; the
        plan's seed is in every evaluation's `details` under `"plan_seed"`,
        and the fresh draws' under `"fresh_draws_seed"`. What only execution
        knows is in the written metadata: the years and the matched cells or
        curves, the checksums, the hashes of the samples and designs and, on
        fitted curves, the dataset's metadata (its units, semantics and
        licence, so `licences` is `None` here); with `cell=`, the cells are
        known only there too, and with `.index(...)` the bound covers.
        A plan that holds `Joint()` adds `scopes`, `dependence` (`None` on
        per-year values, whose dependence is the data's own) and `allocation`;
        `setup.samples` then has one sample of all the assets, and each planned
        evaluation its `level`, `scope` and `allocation`. `format="json"` returns a
        dict, `"text"` its JSON text.

        Raises `ValueError` for `Joint()` on fitted curves without
        `.dependence(...)`, for a dependence or an allocation without
        `Joint()`, and for a cover written for one asset under `Joint()`;
        `SampleError` for a draw count or assets that `EmpiricalCopula`'s
        reference does not fit.
        """
        self._check_scopes()
        document = self._explain_document()
        if format == "json":
            return document
        if format != "text":
            raise ValueError("explain format must be 'text' or 'json'")
        return json.dumps(document, indent=2, sort_keys=True)

    def __repr__(self) -> str:
        ids = self._known_asset_ids()
        assets = "?" if ids is None else str(len(ids))
        objectives = 0 if self.design_spec is None else len(self.design_spec.objectives)
        split = "none" if self.split_rule is None else repr(self.split_rule)
        if self._on_curves:
            source = Path(self.source.dataset.provider.source).name
            draws = "?" if self.draws is None else str(self.draws.n_draws)
            return (
                f"ParametricPlan.for_evaluation({source}, assets={assets}, "
                f"draws={draws}, covers={len(self.cover_list)}, "
                f"objectives={objectives}, split={split})"
            )
        return (
            f"ParametricPlan.for_annual_extremes({type(self.source).__name__}, "
            f"assets={assets}, covers={len(self.cover_list)}, "
            f"objectives={objectives}, split={split})"
        )

    # -- execution --------------------------------------------------------------

    @property
    def _hazard_read(self) -> _SeriesRead:
        """Where each asset's hazard series is read (per-year route only)."""
        assert self.unit is not None and self.provenance is not None
        return _SeriesRead(
            role="hazard",
            source=self.source,
            location=self.location,
            unit=self.unit,
            provenance=self.provenance,
            year_column=self.year_column,
            value_column=self.hazard_column,
            years=self.years,
        )

    def _require_basis_risk(self) -> BasisRisk:
        if self.basis_risk_model is None:
            raise ValueError(
                "state a basis-risk model with .basis_risk(...); NoBasisRisk() "
                "states that the index is the hazard at the asset"
            )
        return self.basis_risk_model

    def _portfolio(self) -> AssetPortfolio:
        """The assets as an in-memory `AssetPortfolio` (reads a file)."""
        assets = self.assets
        if isinstance(assets, Path):
            return AssetPortfolio(pq.read_table(assets))
        data = assets.data
        if isinstance(data, (str, os.PathLike)):
            table = pq.read_table(Path(data))
        else:
            table = _arrow_table(data, "the asset table")
        return AssetPortfolio(
            table,
            id_column=assets.id_column,
            location=assets.location,
            passthrough_columns=assets.passthrough_columns,
        )

    def _asset_rows(
        self, portfolio: AssetPortfolio
    ) -> list[tuple[str, dict[str, Any]]]:
        table = portfolio.data
        if portfolio.id_column not in table.column_names:
            raise SampleError(
                f"the asset table has no id column {portfolio.id_column!r}"
            )
        ids = _string_values(table.column(portfolio.id_column), portfolio.id_column)
        if not ids:
            raise SampleError("the asset table has no assets")
        repeated = sorted({asset_id for asset_id in ids if ids.count(asset_id) > 1})
        if repeated:
            raise SampleError(f"assets {repeated[:10]} appear more than once")
        return list(zip(ids, table.to_pylist(), strict=True))

    def _read_source(self, read: _SeriesRead) -> tuple[Any, dict[str, Any]]:
        """Read one annual table; return it with its record for the metadata."""
        source = read.source
        name = "annual" if read.role == "hazard" else "index"
        kind = _source_kind(source)
        record: dict[str, Any] = {"kind": kind, "type": type(source).__name__}
        if kind == "table":
            table = _arrow_table(source, f"the {name} table")
        elif kind == "file":
            path = Path(source)
            table = pq.read_table(path)
            record["path"] = os.fspath(path)
            record["file_sha256"] = _file_sha256(path)
        else:
            who = "source" if read.role == "hazard" else "index source"
            table = _arrow_table(
                source.annual_extremes(), f"the {who}'s annual_extremes()"
            )
            explained = (
                _json_safe(_source_plan_document(source))
                if callable(getattr(source, "explain", None))
                else None
            )
            record["explain"] = explained
            stated = explained.get("unit") if isinstance(explained, dict) else None
            if isinstance(stated, str) and not units_equivalent(stated, read.unit):
                raise SampleError(
                    f"the {who}'s values are in {stated!r}, but the plan states "
                    f"unit={read.unit!r}"
                )
        record["columns"] = list(table.column_names)
        record["rows"] = int(table.num_rows)
        record["table_sha256"] = _table_sha256(table)
        return table, record

    def _series_source(self, read: _SeriesRead) -> dict[str, Any]:
        """The series' `source`: what identifies the data, stable across runs."""
        document: dict[str, Any] = read.provenance.model_dump(mode="json")
        kind = _source_kind(read.source)
        if kind == "plan":
            document["source_type"] = type(read.source).__name__
        elif kind == "file":
            document["file_sha256"] = _file_sha256(Path(read.source))
        return document

    def _open(self, read: _SeriesRead) -> _OpenedRead:
        """Read a source once, for all the assets' series."""
        table, record = self._read_source(read)
        return _OpenedRead(
            read,
            table,
            record,
            self._series_source(read),
            _stated_pathway(record.get("explain")),
        )

    def _pool_sources(self) -> tuple[Any, ...]:
        """The sources of a pool: several, or the one that holds every run."""
        return self.source if isinstance(self.source, tuple) else (self.source,)

    def _runs_of(self, opened: _OpenedRead, *, alone: bool) -> list[_OpenedRun]:
        """The runs one opened source holds, each with its own rows."""
        pool = self.pool
        assert pool is not None
        table = opened.table
        if table.num_rows == 0:
            raise SampleError("a source of the pool has no rows")
        columns = {
            key: name
            for key, name in (
                ("model", pool.model),
                ("member", pool.member),
                ("pathway", pool.pathway),
            )
            if name in table.column_names
        }
        explained = opened.record.get("explain")
        stated: dict[str, str] = {}
        for key in ("model", "member", "pathway"):
            if key in columns:
                continue
            value = explained.get(key) if isinstance(explained, dict) else None
            if not isinstance(value, str) or not value:
                raise SampleError(
                    f"a source of the pool states no {key}: it has no column "
                    f"{getattr(pool, key)!r} and no explain() document that states "
                    "one; a run must be named by its columns or its source plan"
                )
            stated[key] = value
        if not columns:
            return [_OpenedRun(opened, **stated)]
        names = {key: _run_names(table, name) for key, name in columns.items()}
        found = sorted(
            pa.table(names).group_by(list(names)).aggregate([]).to_pylist(),
            key=lambda row: [row[key].encode("utf-8") for key in names],
        )
        if alone and len(found) > 1:
            raise SampleError(
                f"each source of a sequence is one run, but one holds {len(found)}: "
                f"{found}; pass them as one source, or one source per run"
            )
        runs: list[_OpenedRun] = []
        for run in found:
            mask: Any = None
            for key, value in run.items():
                matches = pc.equal(names[key], pa.scalar(value, pa.string()))
                mask = matches if mask is None else pc.and_(mask, matches)
            rows = dataclasses.replace(opened, table=table.filter(mask))
            runs.append(_OpenedRun(rows, **stated, **run))
        return runs

    def _open_runs(self) -> list[_OpenedRun]:
        """Read the sources of a pool once, for all the assets, and find its runs."""
        sources = self._pool_sources()
        runs: list[_OpenedRun] = []
        for source in sources:
            read = dataclasses.replace(self._hazard_read, source=source)
            runs += self._runs_of(self._open(read), alone=len(sources) > 1)
        recipes: set[str] = set()
        for run in runs:
            explained = run.opened.record.get("explain")
            recipe = explained.get("recipe") if isinstance(explained, dict) else None
            if isinstance(recipe, str) and recipe:
                recipes.add(recipe)
        if len(recipes) > 1:
            raise SampleError(
                f"a pool needs one recipe in every run, got {sorted(recipes)}"
            )
        return runs

    def _asset_cell(
        self, portfolio: AssetPortfolio, asset_id: str, row: Mapping[str, Any]
    ) -> int | None:
        """The H3 cell `cell=` states for one asset; `None` without `cell=`."""
        stated = self.cell
        if stated is None:
            return None
        if isinstance(stated, str):
            if stated not in row:
                raise SampleError(f"the asset table has no column {stated!r}")
            return _h3_index(asset_id, row[stated], stated)
        located = portfolio.location
        if not isinstance(located, PointColumns):
            raise SampleError(
                "H3Cell computes an asset's H3 cell from its point, and these "
                f"assets have none ({type(located).__name__}); name the column "
                "that holds their cells with cell="
            )
        longitude, latitude = row.get(located.longitude), row.get(located.latitude)
        if longitude is None or latitude is None:
            raise SampleError(f"asset {asset_id!r} has no longitude or latitude")
        return _point_cell(longitude, latitude, stated.resolution)

    def _series(
        self,
        opened: _OpenedRead,
        row: Mapping[str, Any],
        asset_id: str,
        portfolio: AssetPortfolio,
    ) -> AnnualSeries:
        """One asset's series of an opened source, as read at its location."""
        series = self._located_series(opened, row, asset_id, portfolio)
        if opened.pathway is None:
            return series
        # Beside the caller's source: the plan does not rewrite it.
        return AnnualSeries(
            series.years,
            series.values,
            series.unit,
            {**_thaw(series.provenance), "pathway": opened.pathway},
        )

    def _located_series(
        self,
        opened: _OpenedRead,
        row: Mapping[str, Any],
        asset_id: str,
        portfolio: AssetPortfolio,
    ) -> AnnualSeries:
        read, table, source = opened.read, opened.table, opened.source
        location = read.location
        if isinstance(location, NearestCell):
            located = portfolio.location if read.point is None else read.point
            vertices: tuple[tuple[float, float], ...] | None = None
            if isinstance(located, PointColumns):
                longitude = row.get(located.longitude)
                latitude = row.get(located.latitude)
                if longitude is None or latitude is None:
                    raise SampleError(
                        f"asset {asset_id!r} has no longitude or latitude"
                    )
            elif isinstance(located, CellColumn):
                cell, (latitude, longitude), vertices = _h3_cell(
                    asset_id, row.get(located.name), located.name
                )
            else:
                raise SampleError(
                    "NearestCell locates assets by a point (PointColumns) or an "
                    f"H3 cell (CellColumn), got {type(located).__name__}"
                )
            series = AnnualSeries.nearest_cell(
                table,
                longitude=longitude,
                latitude=latitude,
                unit=read.unit,
                year=read.year_column,
                value=read.value_column,
                longitude_column=location.longitude,
                latitude_column=location.latitude,
                spacing=location.spacing,
                years=read.years,
                source=source,
            )
            if vertices is None:
                return series
            matched = series.provenance["cell"]
            spacing = series.provenance["spacing_degrees"]
            if not _within_grid_cell(vertices, matched, spacing):
                raise SampleError(
                    f"asset {asset_id!r} is located by its H3 cell {cell}, which "
                    "straddles cells of the grid (step "
                    f"{spacing['latitude']:g}, {spacing['longitude']:g} degrees): "
                    "its centre is nearest the grid cell at "
                    f"({matched['latitude']:g}, {matched['longitude']:g}), but "
                    "part of it lies in another, so the cell does not say which "
                    "series is the asset's; give the asset a point "
                    "(PointColumns) instead"
                )
            return AnnualSeries(
                series.years,
                series.values,
                series.unit,
                {**_thaw(series.provenance), "asset_location": "h3_cell"},
            )
        if location.name not in row:
            raise SampleError(f"the asset table has no column {location.name!r}")
        station = row[location.name]
        if station is None:
            raise SampleError(
                f"asset {asset_id!r} names no station in column {location.name!r}"
            )
        return AnnualSeries.from_table(
            table,
            year=read.year_column,
            value=read.value_column,
            where={location.table_column: station},
            unit=read.unit,
            years=read.years,
            source=source,
        )

    def _generators(self, count: int) -> list[np.random.Generator | None]:
        if self.seed is None:
            return [None] * count
        children = np.random.SeedSequence(self.seed).spawn(count)
        return [np.random.default_rng(child) for child in children]

    def _covers_for(
        self,
        asset_ids: Sequence[str],
        locations: Mapping[str, AtPoint | AtStation] | None = None,
    ) -> dict[str, list[Cover]]:
        """Each asset's covers, written for it and bound to its index location."""
        known = set(asset_ids)
        unknown = sorted(
            {
                str(cover.asset_id)
                for cover in self.cover_list
                if cover.asset_id is not None and cover.asset_id not in known
            }
        )
        if unknown:
            raise ValueError(
                f"covers are written for assets {unknown}, which the plan's "
                "assets do not hold"
            )
        covers: dict[str, list[Cover]] = {}
        for asset_id in asset_ids:
            location = None if locations is None else locations[asset_id]
            chosen = [
                _bound(cover, asset_id, location=location)
                for cover in self.cover_list
                if cover.asset_id in (None, asset_id)
            ]
            if not chosen:
                raise ValueError(
                    f"asset {asset_id!r} has no cover; give it one, or leave it "
                    "out of the assets"
                )
            hashes = [cover.definition_hash for cover in chosen]
            repeated = sorted({h for h in hashes if hashes.count(h) > 1})
            if repeated:
                raise ValueError(
                    f"cover {repeated[0]} applies to asset {asset_id!r} twice: "
                    + (
                        "once written for no asset and once for this one"
                        if locations is None
                        else "two covers have the same terms once bound to this "
                        "asset and the location of its index"
                    )
                )
            covers[asset_id] = chosen
        return covers

    def _blocks(
        self, sample: OutcomeSample, fresh: OutcomeSample | None
    ) -> list[tuple[int, OutcomeSample, OutcomeSample | None]]:
        rule = self.split_rule
        if rule is None:
            return [(0, sample, None)]
        if isinstance(rule, FreshDraws):
            return [(0, sample, fresh)]
        if isinstance(rule, Holdout):
            train, test = holdout(sample, rule.train_fraction)
            return [(0, train, test)]
        return [
            (fold, train, test)
            for fold, (train, test) in enumerate(year_block_kfold(sample, rule.k))
        ]

    def _annual_sample(
        self,
        asset: _AnnualAsset,
        trend: Trend,
        rng: np.random.Generator | None,
    ) -> OutcomeSample:
        """One asset's per-year sample under `trend`, with its notes."""
        if isinstance(asset.series, AnnualSeries):
            sample = _annual_rows(
                asset.series,
                loss_model=asset.loss_model,
                basis_risk=self._require_basis_risk(),
                asset_id=asset.asset_id,
                index=asset.index_series,
                rng=rng,
                missing=self.missing,
                trend=trend,
                scenario=self.scenario,
            )
        else:
            sample = _pool_rows(
                asset.series,
                loss_model=asset.loss_model,
                basis_risk=self._require_basis_risk(),
                asset_id=asset.asset_id,
                rng=rng,
                missing=self.missing,
                trend=trend,
                scenario=self.scenario,
            )
        if asset.cell is not None:
            assert self.cell is not None
            sample = _noted(
                sample,
                f"The impact reads H3 cell {asset.cell:x} for asset "
                f"{asset.asset_id}: {_cell_origin(self.cell)}.",
            )
        if self.index_read is not None:
            assert asset.location is not None
            sample = _index_recorded(
                sample, self.index_read, asset.location, asset.unused
            )
        return sample

    def _fold_blocks(self, samples: Sequence[OutcomeSample]) -> list[_Block]:
        """Each fold's two blocks, each from that fold's own sample."""
        return [self._blocks(sample, None)[fold] for fold, sample in enumerate(samples)]

    def _asset_blocks(self, built: _Samples, asset_id: str) -> list[_Block]:
        """The blocks one asset is designed and evaluated on, on its own."""
        if built.folds is not None:
            return self._fold_blocks(built.folds[asset_id])
        fresh = None if built.fresh is None else built.fresh[asset_id]
        return self._blocks(built.samples[asset_id], fresh)

    def _joint_blocks(self, built: _Samples) -> list[_Block]:
        """The blocks the assets are designed and evaluated on together.

        With a sample per fold, the assets' samples are joined fold by fold, so
        a split step is recorded on each joined sample and every asset's trend
        is fitted on that fold's design years.
        """
        joint = built.joint
        assert joint is not None
        if built.folds is None:
            return self._blocks(joint, built.fresh_joint)
        parts = (built.folds[asset_id] for asset_id in joint.asset_ids)
        return self._fold_blocks(
            [OutcomeSample.join(fold) for fold in zip(*parts, strict=True)]
        )

    def _refitted_samples(
        self,
        asset: _AnnualAsset,
        sample: OutcomeSample,
        generator: np.random.Generator | None,
    ) -> tuple[OutcomeSample, ...]:
        """One sample per fold of all the years, fitted on that fold's design years.

        The fit years are the years the split puts in the design block of
        `sample`, so the two cannot disagree. Every fold starts from the
        asset's same generator state, in `generator`.
        """
        trend = self.trend
        assert isinstance(trend, LinearTrend)
        refits: list[OutcomeSample] = []
        for fold, design_block, _ in self._blocks(sample, None):
            design_years = _sample_years(design_block)
            count = design_years.shape[0]
            if count < trend.min_years:
                raise SampleError(
                    f"asset {asset.asset_id!r}, fold {fold}: LinearTrend needs at "
                    f"least {trend.min_years} fit years (min_years), and the design "
                    f"block holds {count}; no fold is dropped"
                )
            fitted = dataclasses.replace(
                trend, fit_years=tuple(int(year) for year in design_years)
            )
            refits.append(self._annual_sample(asset, fitted, copy.deepcopy(generator)))
        return tuple(refits)

    def _build_samples(self, *, fresh: bool = False, folds: bool = False) -> _Samples:
        """The samples of every asset; with `fresh`, also their fresh draws.

        With `folds`, also each asset's sample per fold, where a linear trend
        is fitted per fold. Under `Joint()`, also the sample of all the assets,
        and `samples` holds each asset's view of it.
        """
        self._check_scopes()
        if self._on_curves:
            return self._curve_samples(fresh=fresh)
        basis_risk = self._require_basis_risk()
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        if self._holds_joint:
            _require_several_assets([asset_id for asset_id, _ in rows])
        # Before the source is read, which may take long or download data.
        cells = [self._asset_cell(portfolio, asset_id, row) for asset_id, row in rows]
        pool_runs = None if self.pool is None else self._open_runs()
        hazard = (
            self._open(self._hazard_read) if pool_runs is None else pool_runs[0].opened
        )
        index = None if self.index_read is None else self._open(self.index_read)
        generators = self._generators(len(rows))
        per_fold = (
            folds
            and isinstance(self.trend, LinearTrend)
            and isinstance(self.split_rule, (Holdout, KFold))
        )
        samples: dict[str, OutcomeSample] = {}
        locations: dict[str, AtPoint | AtStation] = {}
        refitted: dict[str, tuple[OutcomeSample, ...]] = {}
        for (asset_id, row), rng, cell in zip(rows, generators, cells, strict=True):
            series: AnnualSeries | tuple[ModelRun, ...]
            if pool_runs is None:
                series = self._series(hazard, row, asset_id, portfolio)
            else:
                series = tuple(
                    ModelRun(
                        self._series(run.opened, row, asset_id, portfolio),
                        model=run.model,
                        member=run.member,
                        pathway=run.pathway,
                    )
                    for run in pool_runs
                )
            index_series: AnnualSeries | None = None
            unused: list[int] = []
            location: AtPoint | AtStation | None = None
            if index is not None:
                # .index(...) is refused on a pool, so the hazard is one series.
                assert isinstance(series, AnnualSeries)
                read_series = self._series(index, row, asset_id, portfolio)
                location = _resolved_location(index.read, read_series, row)
                locations[asset_id] = location
                index_series, unused = _on_years(read_series, series.years)
            loss_model = (
                self.loss_model.for_asset(portfolio, asset_id)
                if cell is None
                else self._cell_loss_model(portfolio, asset_id, row, cell)
            )
            asset = _AnnualAsset(
                asset_id, series, index_series, loss_model, cell, location, unused
            )
            first_state = copy.deepcopy(rng)
            samples[asset_id] = self._annual_sample(asset, self.trend, rng)
            _warn_comonotone(basis_risk, separate_index=index is not None)
            if per_fold:
                refitted[asset_id] = self._refitted_samples(
                    asset, samples[asset_id], first_state
                )
        joint = OutcomeSample.join(samples.values()) if self._holds_joint else None
        return _Samples(
            portfolio,
            samples if joint is None else _views(joint),
            None,
            "annual_table",
            hazard.record,
            index_record=None if index is None else index.record,
            locations=None if index is None else locations,
            folds=refitted if per_fold else None,
            joint=joint,
            run_records=(
                None
                if pool_runs is None or not isinstance(self.source, tuple)
                else {(run.model, run.member): run.opened.record for run in pool_runs}
            ),
        )

    # -- the curve route ---------------------------------------------------------

    def _require_draws(self) -> _MonteCarlo:
        if self.draws is None:
            raise ValueError(
                "state the simulated years with .monte_carlo(n_draws, seed=...); "
                "the plan draws them from each asset's curve"
            )
        return self.draws

    def _curve_rows(
        self, curves: Any, id_column: str, asset_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """Each asset's one matched curve row of `asset_curves()`."""
        rows: dict[str, list[dict[str, Any]]] = {}
        ids = _string_values(curves.column(id_column), id_column)
        for asset_id, row in zip(ids, curves.to_pylist(), strict=True):
            rows.setdefault(asset_id, []).append(row)
        matched: dict[str, dict[str, Any]] = {}
        for asset_id in asset_ids:
            found = rows.get(asset_id, [])
            if not found:
                raise SampleError(
                    f"asset {asset_id!r} has no curve in the request's selection"
                )
            if len(found) > 1:
                keys = [
                    (row["hazard_name"], row["horizon"], row["pathway"])
                    for row in found
                ]
                raise SampleError(
                    f"asset {asset_id!r} has {len(found)} curves (hazard, horizon, "
                    f"pathway): {keys}; the plan samples one curve per asset, so "
                    "select one with request.select(hazard_names=[...], "
                    "horizons=[...], pathways=[...])"
                )
            matched[asset_id] = found[0]
        return matched

    def _cell_loss_model(
        self,
        portfolio: AssetPortfolio,
        asset_id: str,
        asset: Mapping[str, Any],
        cell: Any,
    ) -> LossModel:
        """The loss model for one asset, with its H3 cell in the impact's context."""
        model = self.loss_model
        if _takes_context(model.impact):
            model = LossModel(
                model.impact,
                value_column=model.value_column,
                value_unit=model.value_unit,
                context=_cell_context(model.impact, cell, asset, model.context),
                name=model.name,
            )
        return model.for_asset(portfolio, asset_id)

    def _curve_samples(self, *, fresh: bool) -> _Samples:
        basis_risk = self._require_basis_risk()
        draws = self._require_draws()
        request: PortfolioEvaluation = self.source
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        if self._holds_joint:
            _require_several_assets([asset_id for asset_id, _ in rows])
        metadata = request.dataset.metadata()
        for cover in self.cover_list:
            if cover.index.unit != metadata.value_unit:
                raise SampleError(
                    f"cover {cover.label or cover.definition_hash!r} pays on an "
                    f"index in {cover.index.unit!r}, but the dataset's values are "
                    f"in {metadata.value_unit!r}"
                )
        path = Path(request.dataset.provider.source)
        file_sha256 = _file_sha256(path)
        curves = dataclasses.replace(request, portfolio=portfolio).asset_curves()
        matched = self._curve_rows(
            curves, portfolio.id_column, [asset_id for asset_id, _ in rows]
        )
        rule = self.split_rule if fresh else None
        fresh_seed = rule.seed if isinstance(rule, FreshDraws) else None
        generators = self._generators(len(rows))
        fresh_generators: list[np.random.Generator | None] = (
            [None] * len(rows)
            if fresh_seed is None
            else [
                np.random.default_rng(child)
                for child in np.random.SeedSequence(fresh_seed).spawn(len(rows))
            ]
        )
        asset_ids = [asset_id for asset_id, _ in rows]
        loss_models = {
            asset_id: self._cell_loss_model(
                portfolio, asset_id, row, matched[asset_id].get("cell_index")
            )
            for asset_id, row in rows
        }
        canonical = {
            asset_id: {
                name: curve[name] for name in _CURVE_ROW_COLUMNS if name in curve
            }
            for asset_id, curve in matched.items()
        }

        def draw_joint(
            count: int, drawn: Sequence[np.random.Generator | None]
        ) -> OutcomeSample:
            rngs: dict[str, np.random.Generator] = {}
            for asset_id, generator in zip(asset_ids, drawn, strict=True):
                assert generator is not None
                rngs[asset_id] = generator
            assert self.dependence_model is not None
            return sample_curves(
                canonical,
                dependence=self.dependence_model,
                metadata=dict.fromkeys(asset_ids, metadata),
                n_draws=count,
                rngs=rngs,
                loss_models=loss_models,
                basis_risk=basis_risk,
                assume_semantics=draws.assume_semantics,
                outside_support=draws.outside_support,
                # One document for every asset, so each match is named by its asset.
                details={
                    "file_sha256": file_sha256,
                    "spatial_match_by_asset": {
                        asset_id: matched[asset_id]["spatial_match"]
                        for asset_id in asset_ids
                    },
                },
            )

        def draw(
            asset_id: str, count: int, generator: np.random.Generator | None
        ) -> OutcomeSample:
            assert generator is not None
            return sample_curve(
                canonical[asset_id],
                metadata=metadata,
                n_draws=count,
                rng=generator,
                loss_model=loss_models[asset_id],
                basis_risk=basis_risk,
                asset_id=asset_id,
                assume_semantics=draws.assume_semantics,
                outside_support=draws.outside_support,
                details={
                    "file_sha256": file_sha256,
                    "spatial_match": matched[asset_id]["spatial_match"],
                },
            )

        samples: dict[str, OutcomeSample]
        fresh_samples: dict[str, OutcomeSample] = {}
        joint: OutcomeSample | None = None
        fresh_joint: OutcomeSample | None = None
        if self._holds_joint:
            joint = draw_joint(draws.n_draws, generators)
            samples = _views(joint)
            if isinstance(rule, FreshDraws):
                fresh_joint = draw_joint(rule.n_draws, fresh_generators)
                fresh_samples = _views(fresh_joint)
        else:
            samples = {
                asset_id: draw(asset_id, draws.n_draws, rng)
                for asset_id, rng in zip(asset_ids, generators, strict=True)
            }
            if isinstance(rule, FreshDraws):
                fresh_samples = {
                    asset_id: draw(asset_id, rule.n_draws, rng)
                    for asset_id, rng in zip(asset_ids, fresh_generators, strict=True)
                }
        record = {
            "curves": int(curves.num_rows),
            "file_sha256": file_sha256,
            "kind": "canonical_dataset",
            "path": os.fspath(path),
        }
        return _Samples(
            portfolio,
            samples,
            fresh_samples if isinstance(rule, FreshDraws) else None,
            "canonical_file",
            record,
            joint=joint,
            fresh_joint=fresh_joint,
        )

    def _values_sha256(
        self, portfolio: AssetPortfolio, asset_ids: Sequence[str]
    ) -> str | None:
        column = self.loss_model.value_column
        if column is None:
            return None
        values = dict(
            zip(
                _string_values(
                    portfolio.data.column(portfolio.id_column), portfolio.id_column
                ),
                portfolio.data.column(column).to_pylist(),
                strict=True,
            )
        )
        pairs = sorted(
            ([asset_id, float(values[asset_id])] for asset_id in asset_ids),
            key=lambda pair: str(pair[0]).encode("utf-8"),
        )
        return content_hash({"asset_values": pairs, "value_column": column})

    def _asset_values(
        self, portfolio: AssetPortfolio, asset_ids: Sequence[str]
    ) -> dict[str, float]:
        """Each asset's value, as the plan's loss model reads it; 1 for a ratio."""
        values: dict[str, float] = {}
        for asset_id in asset_ids:
            read = self.loss_model.for_asset(portfolio, asset_id)._value
            values[asset_id] = 1.0 if read is None else read
        return values

    def _block_tables(
        self,
        entries: Sequence[Any],
        block: _Block,
        *,
        values: Mapping[str, float] | None = None,
    ) -> list[Any]:
        """The rows of one block: on the design block, and on any held-out one."""
        fold, train, test = block
        evaluated: list[tuple[Split, OutcomeSample]] = [("in_sample", train)]
        if test is not None:
            evaluated.append(("out_of_sample", test))
        return [
            compare(
                entries,
                sample,
                split=split,
                fold=fold,
                pricing=self._pricing(),
                options=self.metric_options,
                design_sample=train,
                allocation=self.allocation_rule,
                values=values,
            )
            for split, sample in evaluated
        ]

    def _individual_design(
        self,
        cover: Cover,
        asset_id: str,
        train: OutcomeSample,
        objective: Objective,
        label: str,
    ) -> DesignResult:
        """One template's design on one asset's design block, labelled `label`."""
        spec = self.design_spec
        assert spec is not None
        result = design(
            cover,
            train,
            objective,
            pricing=self._pricing(),
            budget=spec.budget,
            limit=spec.limit,
            options=spec.options,
        )
        return dataclasses.replace(result, cover=_bound(result.cover, asset_id, label))

    def _joint_design(
        self,
        templates: Mapping[str, Cover],
        train: OutcomeSample,
        objective: Objective,
        label: str,
    ) -> JointDesignResult:
        """One template's design on the owner's total, its covers labelled `label`."""
        spec = self.design_spec
        assert spec is not None
        result = design_joint(
            [templates[asset_id] for asset_id in train.asset_ids],
            train,
            objective,
            pricing=self._pricing(),
            budget=spec.budget,
            limit=spec.limit,
            options=spec.options,
        )
        return dataclasses.replace(
            result,
            covers={
                asset_id: _bound(cover, asset_id, label)
                for asset_id, cover in result.covers.items()
            },
        )

    def _joint_entries(
        self, train: OutcomeSample, covers: Mapping[str, Sequence[Cover]]
    ) -> tuple[list[Any], list[Any]]:
        """The entries to compare on one block of the joint sample, and the designs.

        A resolved cover is one contract per asset. A template is designed under
        each objective on the owner's total and, if the plan holds
        `Individual()` too, on each asset's own view of the block.
        """
        assets = train.asset_ids
        entries: list[Any] = []
        designs: list[Any] = []
        for position, stated in enumerate(self.cover_list):
            own = {asset_id: covers[asset_id][position] for asset_id in assets}
            if not stated.free_parameters():
                entries.append(own)
                continue
            assert self.design_spec is not None
            for objective in self.design_spec.objectives:
                label = _design_label(stated, objective)
                joint = self._joint_design(own, train, objective, label)
                designs.append(joint)
                entries.append(joint)
                if self._holds_individual:
                    each = {
                        asset_id: self._individual_design(
                            own[asset_id],
                            asset_id,
                            train.asset(asset_id),
                            objective,
                            label,
                        )
                        for asset_id in assets
                    }
                    designs += each.values()
                    entries.append(each)
        return entries, designs

    def _run(self, *, evaluate: bool) -> _Run:
        if not self.cover_list:
            raise ValueError("the plan has no covers; set them with .covers(...)")
        spec = self.design_spec
        templates = [cover for cover in self.cover_list if cover.free_parameters()]
        if templates and spec is None:
            names = ", ".join(
                repr(cover.label or cover.definition_hash) for cover in templates
            )
            raise ValueError(
                f"covers {names} have free parameters; design them with "
                ".objective(...), or resolve them with Cover.with_values"
            )
        built = self._build_samples(fresh=evaluate, folds=True)
        portfolio, samples = built.portfolio, built.samples
        asset_ids = list(samples)
        covers = self._covers_for(asset_ids, built.locations)
        designs: list[Any] = []
        tables: list[Any] = []
        if built.joint is not None:
            values = (
                None
                if self.allocation_rule is None
                else self._asset_values(portfolio, asset_ids)
            )
            for block in self._joint_blocks(built):
                entries, made = self._joint_entries(block[1], covers)
                designs += made
                if evaluate:
                    tables += self._block_tables(entries, block, values=values)
        else:
            for asset_id in asset_ids:
                for block in self._asset_blocks(built, asset_id):
                    train = block[1]
                    items: list[Any] = []
                    for cover in covers[asset_id]:
                        if not cover.free_parameters():
                            items.append(cover)
                            continue
                        assert spec is not None
                        for objective in spec.objectives:
                            result = self._individual_design(
                                cover,
                                asset_id,
                                train,
                                objective,
                                _design_label(cover, objective),
                            )
                            designs.append(result)
                            items.append(result)
                    if evaluate:
                        tables += self._block_tables(items, block)
        return _Run(
            samples=MappingProxyType(samples),
            fresh=None if built.fresh is None else MappingProxyType(built.fresh),
            designs=tuple(designs),
            tables=tuple(tables),
            source_key=built.record_key,
            source=built.record,
            id_column=portfolio.id_column,
            values_sha256=self._values_sha256(portfolio, asset_ids),
            index_source=built.index_record,
            joint=built.joint,
            fresh_joint=built.fresh_joint,
            run_records=built.run_records,
        )

    def samples(self) -> Mapping[str, OutcomeSample]:
        """Execution: build one outcome sample per asset, in asset order.

        Reads the source and the assets. Each sample holds all of the
        asset's draws or years, before any split (fresh draws are drawn by
        `write_parquet()` only, which returns them in `fresh_samples`), so a
        `LinearTrend` here is fitted on all the years, and its record says so.
        Under `Joint()`, each is that asset's view of `joint_sample()`.
        Raises `ValueError` without a basis-risk model or, on fitted curves,
        without `monte_carlo`, and as `explain()` does for the scopes;
        `SampleError` for an asset that cannot be
        located (an H3 cell that straddles grid cells included), a curve that
        cannot be sampled (`sample_curve`), several curves for one asset, a
        cover whose index unit is not the dataset's, or a series with a
        missing year, an index year included (unless `missing="drop_year"`),
        or a trend the series cannot take;
        `ProbabilitySemanticsError` for curves whose semantics are not those
        of one year; and `ImportError` without the `geometry` extra for
        assets located by their H3 cells on per-year values, or for
        `cell=H3Cell(...)`, whose cells are computed before the source is
        read.
        """
        return MappingProxyType(self._build_samples().samples)

    def joint_sample(self) -> OutcomeSample:
        """Execution: the sample of all the assets, side by side, under `Joint()`.

        Reads the source and the assets. On fitted curves it is what
        `sample_curves` draws under the plan's `.dependence(...)`, and on
        per-year values `OutcomeSample.join` of each asset's sample. It holds
        all the draws or years, before any split, so a `LinearTrend` here is
        fitted on all the years, like `.samples()`, while the rows of a holdout
        or k-fold come from one sample per fold.

        Raises `ValueError` for a plan that does not hold `Joint()`, and as
        `samples` does; `SampleError` when the assets' years differ, as
        `OutcomeSample.join` raises, which a stated `years=` resolves.
        """
        if not self._holds_joint:
            raise ValueError(
                "joint_sample() puts the assets side by side under Joint(); state "
                ".scopes(Joint())"
            )
        joint = self._build_samples().joint
        assert joint is not None
        return joint

    def designs(self) -> tuple[Any, ...]:
        """Execution: design every template, in asset, fold, cover, objective order.

        Each template is designed on each design block (the whole sample
        without a split or with fresh draws; the design block of a holdout;
        each fold's design block) under each objective. Under `Joint()` the
        designs are `JointDesignResult`s, in fold, cover and objective order,
        each followed by that template's `DesignResult`s, in asset order, when
        the plan also holds `Individual()`. Raises `ValueError`
        without covers, an objective or a basis-risk model, the design errors
        of `design` and `design_joint`, and `SolverUnavailableError` without
        `crc-sdk[parametric]` on the LP and QP routes.
        """
        if self.design_spec is None:
            raise ValueError(
                "the plan has no objective; state one with .objective(...)"
            )
        return self._run(evaluate=False).designs

    def write_parquet(self, path: str | os.PathLike[str]) -> ParametricResult:
        """Execution: design, evaluate and compare; write the comparison.

        Builds the samples, designs the templates, evaluates every cover and
        design on each block (`compare`) and writes the rows of every asset,
        split and fold to one local Parquet file, through a temporary
        sibling and `os.replace`; the parent directory is created. The
        metadata (`EvaluationMetadata`, under `crc.parametric.evaluation`)
        merges that of every block, and adds the source's record once, in
        `source_records`: on fitted curves `{"canonical_file": {...}}` (the
        dataset's path, its SHA-256 and the number of matched curves); on
        per-year values `{"annual_table": {...}}` (the table's rows, columns
        and SHA-256, the source plan's `explain()` and, for a file, its path
        and SHA-256; a pool has one hazard source per run, and a sequence of
        sources one such record each). Each hazard source names its record by
        that checksum:
        `canonical_file_sha256` (the file's) or `annual_table_sha256` (the
        table's). With `.index(...)`, `source_records` also holds
        `{"index_table": {...}}`, of the same form, each source with role
        `"index"` names it by `index_table_sha256`, `licences` holds the
        index source's terms and `covers` the bound documents. It adds the
        plan's seed to every evaluation's `details`
        (`"plan_seed"`) and the fresh draws' seed to every fresh-draw
        evaluation's (`"fresh_draws_seed"`), and the asset id column and a
        SHA-256 of the (asset id, value) pairs to `assets`.

        Under `Joint()` the rows are one `"total"` row per joint design and
        block (`scope="joint"`) and, under `.allocation(...)`, one allocated
        row per asset; a resolved cover has one `"asset"` row per asset and a
        `"total"` row (`scope="individual"`), and so does a template's
        per-asset design if the plan also holds `Individual()`. Only the rows
        of a plan that holds `Joint()` include a `"total"` row.

        Returns a `ParametricResult`, with the fresh draws in
        `fresh_samples` when the split is `FreshDraws`. Raises `ValueError`
        without covers, a basis-risk model, or an objective for templates;
        `SampleError` for a cover that names another location than the index
        is read at, or a fold with fewer design years than `LinearTrend`'s
        `min_years`; and as `samples`, `design` and `compare` raise.
        """
        run = self._run(evaluate=True)
        merged = EvaluationMetadata.merge(
            *(EvaluationMetadata.from_table(table) for table in run.tables)
        )
        document = merged.model_dump(mode="json")
        checksum = run.source[
            "file_sha256" if run.source_key == "canonical_file" else "table_sha256"
        ]
        checksums = {"hazard": {f"{run.source_key}_sha256": checksum}}
        records: list[dict[str, Any]] = [{run.source_key: run.source}]
        by_run = run.run_records
        if by_run is not None:
            records = [{run.source_key: record} for record in by_run.values()]
        if run.index_source is not None:
            checksums["index"] = {
                "index_table_sha256": run.index_source["table_sha256"]
            }
            records.append({"index_table": run.index_source})
        document["sources"] = [
            {
                **entry,
                **checksums.get(str(entry.get("role")), {}),
                **(
                    {}
                    if by_run is None
                    else {
                        f"{run.source_key}_sha256": by_run[
                            (entry["model"], entry["member"])
                        ]["table_sha256"]
                    }
                ),
            }
            for entry in document["sources"]
        ]
        document["source_records"] = [*document.get("source_records", ()), *records]
        seeded = self._seed_details()
        rule = self.split_rule
        fresh = {"fresh_draws_seed": rule.seed} if isinstance(rule, FreshDraws) else {}
        document["setup"]["evaluations"] = [
            {
                **entry,
                "details": {
                    **entry.get("details", {}),
                    **(fresh if entry.get("method") == "fresh_draws" else {}),
                    **seeded,
                },
            }
            for entry in document["setup"]["evaluations"]
        ]
        document["assets"] = {
            "count": len(run.samples),
            "id_column": run.id_column,
            # Without a value column, an allocation's values are the plan's 1s.
            "values_sha256": run.values_sha256
            or document["assets"].get("values_sha256"),
        }
        metadata = EvaluationMetadata.model_validate(document)
        table = pa.concat_tables(run.tables)
        existing = dict(table.schema.metadata or {})
        existing[EVALUATION_METADATA_KEY.encode("utf-8")] = metadata.to_json().encode(
            "utf-8"
        )
        table = table.replace_schema_metadata(existing)
        text = os.fspath(path)
        if not isinstance(text, str) or "://" in text:
            raise ValueError(f"write_parquet writes local files, got {path!r}")
        target = Path(text)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_comparison(table, target, metadata=metadata)
        return ParametricResult(
            output=target,
            table=table,
            metadata=metadata,
            samples=run.samples,
            designs=run.designs,
            fresh_samples=run.fresh,
            joint_sample=run.joint,
            fresh_joint_sample=run.fresh_joint,
        )
