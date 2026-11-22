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
  .pricing(pricing)                     # optional: one pricing for every premium
  .split(FreshDraws(n, seed=) | Holdout(0.7) | KFold(5))   # optional
  .metrics(MetricOptions(...))          # optional: levels and thresholds
  .scopes(Individual() | Joint() | Clustered(...))  # optional: asset, total, cluster
  .dependence(Comonotonic() | ...)      # with Joint() or Clustered on curves: required
  .allocation("value" | "loss")         # optional: a joint design's allocated rows
  .scenarios(ScenarioSet(...))          # optional: several scenarios, out of sample
  .hazard(name, source, ...)            # optional: another hazard at each asset
  .combine(rule)                        # with two impacts or more: required
  .package(covers, label=...)           # optional: separate contracts as one row
  .explain()                            # the setup, no I/O
  .samples()                            # execution: one sample per asset
  .joint_sample()                       # execution: Joint() or Clustered only
  .scenario_samples()                   # execution: every scenario's samples
  .designs()                            # execution: the designed covers
  .payout_breakdown()                   # execution: each entry's payout in parts
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
per-year route only the basis-risk model draws, so `HazardNoise` and
`ImpactNoise` need a seed there.

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
  must state, with the plan's generators, one per asset. A joint design's rows
  are on the owner's total, which reads every asset, so a holdout or k-fold
  takes `Independent()` only: under any other model `explain()` and every
  execution point raise `LeakageError`, naming `FreshDraws`;
- on per-year values, `OutcomeSample.join` of the assets' samples, whose
  dependence is the data's own, so `.dependence(...)` raises there. With
  `LinearTrend` and a holdout or k-fold the assets' samples are joined per fold,
  each fitted on that fold's design years.

Under `Joint()` a cover must be written for no asset, because each cover applies
to every asset; different terms per asset go through `design_joint`. A resolved
cover is evaluated as a set, one contract per asset. `.samples()` returns each
asset's view of the sample and `.joint_sample()` the sample itself. `.scopes`
may hold several scopes: each asset is then also designed on its own view, and
the `scope` column keeps the rows of each scope apart. `.allocation(rule)` adds a
joint design's allocated rows, with the values the plan's loss model read, in a
table and a file of their own (`allocate`), so they never sit beside the rows
of covers that stand on their own. `docs/parametric.md` describes the rows.

`Clustered(assignment, values=..., joint=...)` designs on the same sample of all
the assets, as `Joint()` does, and under the same rules: each template once per
objective with `design_joint` under that scope, one cover per cluster on the
value-weighted mean of its assets' indices, with the rows `compare` gives it.
Its weights are the values the plan's loss model reads (the asset column
`value_column`, or 1 for a damage ratio), so its `values` must be those, and
execution raises otherwise; its assignment must hold exactly the plan's assets.
Its allocated rows share each cluster's payout over that cluster's assets.
`.index(...)` is refused beside it: a cluster's cover pays on one index
definition over its assets, so it can't name each asset's station or point.

**Scenarios.** `.scenarios(ScenarioSet(...))` reads several scenarios: on fitted
curves each selected (pathway, horizon), or a window of horizons under
`labels=`, and a `CurveScenario` for another canonical dataset; on per-year
values the plan's own source and each `AnnualScenario`. Each scenario's samples
are built as the plan builds its own, from the same generators, and the refusals
of `ScenarioSet` run on them. Every template is designed on each scenario's
design block (`design_on="each"`) or the reference's (`"reference"`), and every
cover and design is evaluated on every scenario's evaluation block with
`compare_scenarios`, one fold and asset at a time, or once for all the assets
under `Joint()` or `Clustered`, so the rows are out of sample only and
`.split(...)` is required. Under `.allocation(rule)` a joint or cluster design's
allocated rows on every scenario's block go to their own file
(`allocate_scenarios`), and `.payout_breakdown()` gives each scenario's
breakdown (`payout_breakdown_scenarios`). A request that selects several horizons or
pathways needs `.scenarios(...)`: without it `explain()` and every execution
point raise `ValueError`.

**Several hazards.** `.hazard(name, source, ...)` adds a hazard read at each
asset, an index component of one sample per asset, as `sample_annual_hazards`
(per-year values) or `sample_curve_hazards` (fitted curves) builds it. On
per-year values `source` takes what `for_annual_extremes` takes, and
`.hazard(name, tail=...)` without one names the plan's own hazard, which comes
first. On fitted curves `source` is a second request for the plan's assets, and
the plan's own request is the first component, named by the hazard it selects;
`.dependence(...)` must state how the hazards' draws are put side by side.
`.combine(rule)` states the rule of the combined loss whenever two components or
more have an impact. A cover on a `CompositeIndex` is designed on the combined
loss, and a cover on one index on its own hazard's loss (`sample.component`).
`.package(covers, label=...)` makes separate contracts, named by their labels,
one row of the comparison (`CoverPackage`); with a budget, `shares=` splits it
between them (`split_budget`). Every row's metrics are on the combined loss.
`Joint()`, `Clustered`, `pool=` and `.index(...)` raise on such a plan.

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
pairs. Under `.allocation(rule)`, a joint design's allocated rows go to a
second file beside the first, `<stem>.allocated<suffix>`, with metadata of the
same form. The returned `ParametricResult` holds the samples, the fresh draws
and the designs and, under `Joint()` or `Clustered`, the sample of all the assets
and any allocated rows.
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
from crc_sdk.parametric.basis_risk import BasisRisk, HazardNoise, ImpactNoise
from crc_sdk.parametric.combine import CombineRule, _rule_record
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.dependence import (
    Comonotonic,
    Dependence,
    DependenceModel,
    EmpiricalCopula,
    GaussianCopula,
    Independent,
    _model_record,
)
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import (
    EvaluationError,
    LeakageError,
    ProbabilitySemanticsError,
    SampleError,
    ScenarioError,
)
from crc_sdk.parametric.evaluate import (
    Allocation,
    CoverPackage,
    Split,
    _premium,
    _stated_shares,
    allocate,
    compare,
    payout_breakdown,
    split_budget,
    write_comparison,
)
from crc_sdk.parametric.index import (
    AtAsset,
    AtPoint,
    AtStation,
    CompositeIndex,
    IndexDefinition,
    Tail,
    _component_name,
)
from crc_sdk.parametric.joint import (
    Clustered,
    ClusteredDesignResult,
    Individual,
    Joint,
    JointDesignResult,
    design_joint,
)
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.metadata import (
    _ACROSS_YEARS,
    EVALUATION_METADATA_KEY,
    EvaluationMetadata,
)
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    Objective,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.pricing import Pricing, PurePremium, _check_pricing
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _arrow_table,
    _check_known_pathway,
    _identifier,
    _named_in_source,
    _sample_years,
    _source_columns,
    _string_values,
    _thaw,
    holdout,
    year_block_kfold,
)
from crc_sdk.parametric.scenarios import (
    AnnualScenario,
    CurveScenario,
    ScenarioSet,
    _check_pairs_held,
    _check_scenarios,
    _same_pair,
    allocate_scenarios,
    compare_scenarios,
    payout_breakdown_scenarios,
)
from crc_sdk.parametric.sources import (
    AnnualSemantics,
    AnnualSeries,
    MissingYears,
    ModelRun,
    OutsideSupport,
    _annual_hazard_rows,
    _annual_rows,
    _combine_record,
    _coordinate,
    _pool_rows,
    _source_plan_document,
    _warn_comonotone,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric.trend import AsRecorded, LinearTrend, Trend, _require_trend
from crc_sdk.pathways import pathways_equivalent
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows.distributions import CURVE_COLUMNS
from crc_sdk.workflows.portfolio import (
    AssetPortfolio,
    CellColumn,
    HazardDataset,
    HazardSelection,
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
_HAZARDS_REFUSE_JOINT = (
    "Joint() is refused on a plan with .hazard(...): a sample of several hazards "
    "holds one asset, and several assets with several hazards would need a "
    "dependence over both axes, which no model states; design each asset on its "
    "own, with Individual()"
)
_HAZARDS_REFUSE_CLUSTERED = _HAZARDS_REFUSE_JOINT.replace("Joint()", "Clustered(...)")
_INDEX_REFUSES_CLUSTERED = (
    "Clustered(...) is refused on a plan with .index(...): a cluster's cover pays "
    "on the weighted mean of one IndexDefinition over its assets, so it can't "
    "name the station or point each asset's index series is read at, as a cover "
    "bound under .index(...) does"
)
_HAZARDS_REFUSE_POOL = (
    "pool= is refused on a plan with .hazard(...): a pool across hazards would "
    "need the same (year, model, member) rows for every hazard, which no rule "
    "states; read one series per hazard"
)
_HAZARDS_REFUSE_INDEX = (
    ".index(...) is refused on a plan with .hazard(...): an index read elsewhere "
    "has no form per component yet; each component's index is its hazard at the "
    "asset"
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
    `"station_id"`). Each series records `table_column` as its station column
    (`AnnualSeries.from_table(station=)`), whatever it is called, so a
    scenario set's place checks and a cover bound with `AtStation` read the
    station from it.
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


@dataclass(frozen=True, eq=False)
class _Hazard:
    """One index component of a plan with several hazards.

    On per-year values `source` is `None` for the plan's own hazard, whose read
    is the plan's, and otherwise what `for_annual_extremes` takes, read at
    `location` (the plan's when `None`) with the plan's years, `missing=` and
    trend. On fitted curves `source` is the component's `PortfolioEvaluation`,
    `unit` and `tail` are `None` (its dataset states them at execution) and the
    other read fields keep their defaults. `loss_model` is the component's
    impact on the plan's value, `None` for a component with no impact.
    """

    name: str
    source: Any = None
    loss_model: LossModel | None = None
    unit: str | None = None
    tail: Tail | None = None
    provenance: SourceProvenance | None = None
    location: NearestCell | StationColumn | None = None
    year_column: str = "year"
    hazard_column: str = "value"
    hazard_name: str | None = None

    def to_document(self, *, own: bool, on_curves: bool) -> dict[str, Any]:
        """What `explain()` lists of the component: where it is read, and what."""
        document: dict[str, Any] = {
            "impact": None
            if self.loss_model is None
            else self.loss_model.to_document(),
            "name": self.name,
            "own": own,
        }
        if on_curves:
            request: PortfolioEvaluation = self.source
            document["hazard_names"] = _listed(request.selection.hazard_names)
            document["path"] = os.fspath(request.dataset.provider.source)
            return document
        document |= {
            "hazard_name": self.hazard_name,
            "tail": self.tail,
            "unit": self.unit,
        }
        return document


@dataclass(frozen=True, eq=False)
class _Package:
    """What `.package(...)` states: the separate contracts by component, and how.

    `covers` holds (component, `cover_label`) pairs in order, and `shares` the
    shares of `split_budget`: `"own_loss"`, (component, share) pairs or `None`.
    """

    covers: tuple[tuple[str, str], ...]
    label: str
    shares: Literal["own_loss"] | tuple[tuple[str, float], ...] | None

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.covers)

    def split_shares(self) -> Literal["own_loss"] | dict[str, float] | None:
        """The shares as `split_budget` takes them."""
        if self.shares is None or isinstance(self.shares, str):
            return self.shares
        return dict(self.shares)

    def to_document(self) -> dict[str, Any]:
        shares = self.split_shares()
        return {
            "covers": dict(self.covers),
            "label": self.label,
            "shares": shares,
        }


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
      Under `Joint()` or `Clustered`, each asset's view of `joint_sample`;
    - `designs`: every `DesignResult`, in asset, fold, cover and objective
      order. Under `Joint()`, every `JointDesignResult`, in fold, cover and
      objective order, each followed by its template's `ClusteredDesignResult`
      under `Clustered` and per-asset `DesignResult`s when the plan also holds
      `Individual()`; under `Clustered` alone, every `ClusteredDesignResult` in
      that order;
    - `fresh_samples`: with `FreshDraws`, one `OutcomeSample` per asset, in
      asset order: the fresh draws the out-of-sample rows are evaluated on.
      `None` without fresh draws. Under `Joint()` or `Clustered`, each asset's
      view of `fresh_joint_sample`;
    - `joint_sample`: under `Joint()` or `Clustered`, the sample of all the
      assets, with all its years or draws, before any split; otherwise `None`;
    - `fresh_joint_sample`: under `Joint()` or `Clustered` with `FreshDraws`,
      the fresh draws of all the assets, whose `content_hash` is the
      `sample_hash` of the out-of-sample rows; otherwise `None`;
    - `allocated`: under `.allocation(...)`, the joint or cluster designs'
      allocated rows as written, in a table of their own with its metadata,
      as `allocate` builds them; otherwise `None`;
    - `allocated_output`: the file of `allocated`, `<stem>.allocated<suffix>`
      beside `output`; otherwise `None`;
    - `scenario_samples`: with `.scenarios(...)`, every scenario's samples, by
      label and then by asset, the reference first; `samples`, `fresh_samples`
      and `joint_sample` are then the reference's. Otherwise `None`;
    - `scenario_fresh_samples`: with `.scenarios(...)` and `FreshDraws`, every
      curve scenario's fresh draws, by label and then by asset, on one set of
      uniforms; their `content_hash` is the `sample_hash` of each scenario's
      rows. Otherwise `None`.
    """

    output: Path
    table: Any
    metadata: EvaluationMetadata
    samples: Mapping[str, OutcomeSample]
    # `DesignResult`, `JointDesignResult` or `ClusteredDesignResult`: it depends on
    # the plan's scopes, so existing callers of a plan of one scope keep their types.
    designs: tuple[Any, ...]
    fresh_samples: Mapping[str, OutcomeSample] | None = None
    joint_sample: OutcomeSample | None = None
    fresh_joint_sample: OutcomeSample | None = None
    allocated: Any = None
    allocated_output: Path | None = None
    scenario_samples: Mapping[str, Mapping[str, OutcomeSample]] | None = None
    scenario_fresh_samples: Mapping[str, Mapping[str, OutcomeSample]] | None = None

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
    allocated: tuple[Any, ...]
    source_key: str
    source: dict[str, Any]
    id_column: str
    values_sha256: str | None
    index_source: dict[str, Any] | None = None
    joint: OutcomeSample | None = None
    fresh_joint: OutcomeSample | None = None
    run_records: Mapping[tuple[str, str], dict[str, Any]] | None = None
    scenarios: Mapping[str, _Samples] | None = None
    component_records: Mapping[str, dict[str, Any]] | None = None


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


@dataclass(frozen=True, eq=False)
class _HazardAsset:
    """What one asset's per-year sample of several hazards is built from."""

    asset_id: str
    series: dict[str, AnnualSeries]
    loss_models: dict[str, LossModel]
    cell: int | None


_Block = tuple[int, OutcomeSample, OutcomeSample | None]
"""A fold, its design block, and its evaluation block if it has one."""

DesignOn = Literal["each", "reference"]
"""Where a plan with scenarios designs: on every scenario, or the reference's."""

_Pairs = tuple[tuple[str, int], ...]
"""A curve scenario's (pathway, horizon) pairs, in horizon order."""


@dataclass(frozen=True, eq=False)
class _Planned:
    """One scenario of a plan, as far as it is known before execution.

    `kind` is `"selection"` for pairs of the request's selection, `"dataset"`
    for a `CurveScenario`, `"own"` for a per-year plan's own source and
    `"annual"` for an `AnnualScenario`. `members` are a curve scenario's pairs,
    `None` where they are known only at execution or the scenario reads years.
    """

    label: str
    kind: Literal["selection", "dataset", "own", "annual"]
    members: _Pairs | None = None
    described: AnnualScenario | CurveScenario | None = None


@dataclass(frozen=True, eq=False)
class _CurveRead:
    """One canonical dataset at execution: its metadata, its record and its rows.

    `rows` holds each asset's matched canonical rows, one per (pathway, horizon).
    """

    metadata: HazardDatasetMetadata
    record: dict[str, Any]
    rows: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True, eq=False)
class _Samples:
    """What execution builds before any design: the samples and the source.

    `folds` holds each asset's sample of each fold, when each fold has a sample
    of its own. Under `Joint()` or `Clustered`, `joint` and `fresh_joint` are the
    samples of all the assets, and `samples` and `fresh` each asset's view of
    them. A pool
    read from a sequence of sources has each run's record in `run_records`,
    by model and member. A plan with several hazards has each component's
    record in `component_records`, by name, and `record` is the first's.
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
    component_records: Mapping[str, dict[str, Any]] | None = None


# -- helpers -----------------------------------------------------------------


def _require_several_assets(asset_ids: Sequence[str], scope: str = "Joint()") -> None:
    if len(asset_ids) < 2:
        raise ValueError(
            f"{scope} designs several assets together, but the plan holds "
            f"{list(asset_ids)}; use Individual() for one asset"
        )


def _check_assigned(scope: Clustered, asset_ids: Sequence[str]) -> None:
    """Raise `SampleError` unless `scope` assigns exactly the plan's assets."""
    left_out = [asset for asset in asset_ids if asset not in scope.assignment]
    unknown = [asset for asset in scope.assignment if asset not in asset_ids]
    if not left_out and not unknown:
        return
    empty = [
        cluster
        for cluster, members in scope.members.items()
        if all(asset in unknown for asset in members)
    ]
    message = (
        "Clustered assigns every asset of the plan to one cluster, but the "
        f"assignment leaves out {left_out} and names {unknown}, which the plan's "
        f"assets {list(asset_ids)} don't hold"
    )
    if empty:
        message += f"; so the clusters {empty} hold no asset of the plan"
    raise SampleError(message)


def _copula_split_error(rule: Holdout | KFold) -> SampleError:
    return SampleError(
        f"{type(rule).__name__} cannot split a sample under EmpiricalCopula: its "
        "rows come in reference-year blocks, so a block of rows is a set of "
        "reference years; test it out of sample with .split(FreshDraws(n_draws, "
        "seed=...)), whose n_draws must be a multiple of the reference's years"
    )


def _moved_split_error(
    rule: Holdout | KFold,
    model: Dependence | DependenceModel,
    *,
    columns: Literal["hazards", "assets"] = "hazards",
) -> LeakageError:
    """The refusal of a split by rows of draws a model put side by side.

    `columns` is what the model put side by side: the hazards of `.hazard(...)`,
    whose rows are on the combined loss, or the assets of `Joint()`, whose rows
    are on the owner's total.
    """
    named = type(model).__name__
    one = columns[:-1]
    how = (
        f"{named} put each {one}'s draws in the rows by their ranks over every "
        f"row, so each block's values of one {one} depend on the other block's "
        "values of the others"
        if isinstance(model, (Comonotonic, GaussianCopula))
        else f"{named} reordered the {columns}' draws, and nothing says how its "
        "reorder treats the rows"
    )
    rows = (
        "every row is on the combined loss"
        if columns == "hazards"
        else "a joint design's rows are on the owner's total, which reads every asset"
    )
    return LeakageError(
        f"{type(rule).__name__} cuts one sample of curve draws into blocks, and "
        f"{how}: {rows}, so its out-of-sample rows would raise LeakageError once "
        "the covers were designed. Test the designs on fresh draws under the same "
        "model: .split(FreshDraws(n_draws, seed=...))"
    )


def _spanning(asset_id: str, sample: OutcomeSample) -> str:
    """The asset and the years its series spans, as the merge's refusal names them."""
    years = sample.provenance.details.get("years")
    span = (
        f", whose series spans {years[0]} to {years[1]}"
        if isinstance(years, (list, tuple)) and len(years) == 2
        else ""
    )
    return f" (asset {asset_id!r}{span})"


def _check_scenario_horizons(built: Mapping[str, _Samples]) -> None:
    """Raise `EvaluationError` if two assets read a per-year scenario at two horizons.

    A per-year scenario's horizon is the centre of each asset's series' whole
    window, `(first + last) // 2`, and `write_parquet` merges the tables of
    every asset, which must state one set of scenarios: assets whose series
    span different years, as `missing="drop_year"` can leave them, would be
    refused there (`EvaluationMetadata.merge`), after the designs. This raises
    the same error, word for word, before anything is designed: each asset is
    compared with the first, in asset order, and the scenarios in the set's
    order, as the merge compares the tables.
    """
    labels = list(built)
    first, *others = list(built[labels[0]].samples)
    for asset_id in others:
        for label in labels:
            samples = built[label].samples
            ours = _source_columns(samples[first], strict=False).horizon
            theirs = _source_columns(samples[asset_id], strict=False).horizon
            if ours != theirs:
                raise EvaluationError(
                    "the metadata differ in their scenarios: only the blocks of one "
                    "set of scenarios may differ between tables, but scenario "
                    f"{label!r} has the horizon {ours!r} in one table"
                    f"{_spanning(first, samples[first])} and {theirs!r} in another"
                    f"{_spanning(asset_id, samples[asset_id])}{_ACROSS_YEARS}"
                )


def _contract(label: str, packaged: bool) -> str:
    """A cover, as an error names it: a package's contract, or a cover."""
    return f"package contract {label!r}" if packaged else f"cover {label!r}"


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


def _same_assets(first: AssetPortfolio, second: AssetPortfolio) -> bool:
    """Whether two portfolios hold the same assets, in the same order and places."""
    if first is second:
        return True
    if (first.id_column, first.location) != (second.id_column, second.location):
        return False
    ours, theirs = first.data, second.data
    if isinstance(ours, (pa.Table, pa.RecordBatch)) and type(ours) is type(theirs):
        return bool(ours.equals(theirs))
    if isinstance(ours, (str, os.PathLike)) and isinstance(theirs, (str, os.PathLike)):
        return os.fspath(ours) == os.fspath(theirs)
    return ours is theirs


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
    index: IndexDefinition | CompositeIndex = (
        cover.index_definition
        if location is None
        else _index_at(cover, asset_id, location)
    )
    if (
        cover.asset_id == asset_id
        and name == cover.label
        and index is cover.index_definition
    ):
        return cover
    return Cover(
        index,
        cover.payout,
        payout_unit=cover.payout_unit,
        asset_id=asset_id,
        label=name,
    )


def _relabelled(cover: Cover, label: str) -> Cover:
    """A cluster's `cover` with `label`; its index and asset stay the cluster's."""
    if cover.label == label:
        return cover
    index = cover.cluster_index
    assert index is not None
    return Cover(
        index,
        cover.payout,
        payout_unit=cover.payout_unit,
        asset_id=cover.asset_id,
        label=label,
    )


def _cover_label(cover: Cover) -> str:
    """A cover's `cover_label` in a comparison: its label, or its hash's start."""
    if cover.label is not None:
        return cover.label
    return cover.definition_hash.removeprefix("sha256:")[:12]


def _checked_premiums(premiums: Any) -> tuple[tuple[str, float], ...] | None:
    """`premiums=` of `.scenarios(...)`, as (cover_label, premium) pairs."""
    if premiums is None:
        return None
    if not isinstance(premiums, Mapping):
        raise TypeError(
            "premiums must map a cover_label to a stated premium, got "
            f"{type(premiums).__name__}"
        )
    pairs: list[tuple[str, float]] = []
    for key, value in premiums.items():
        if not isinstance(key, str):
            raise TypeError(f"premiums must be keyed by cover_label, got {key!r}")
        amount = _premium(value)
        if amount is None:
            raise EvaluationError(f"premiums= states no number for {key!r}")
        pairs.append((key, amount))
    return tuple(pairs)


def _designed(entry: Any) -> bool:
    """Whether a plan's entry was designed: a design, or a package of designs."""
    if isinstance(entry, Cover):
        return False
    if isinstance(entry, CoverPackage):
        return entry._designed
    return True


def _given(entry: Any) -> list[Cover]:
    """The covers given as is that an entry holds: itself, or a package's contracts."""
    if isinstance(entry, Cover):
        return [entry]
    if isinstance(entry, CoverPackage):
        return [] if entry._designed else list(entry.covers.values())
    return []


def _source_components(document: Mapping[str, Any]) -> list[str | None]:
    """The component each of a document's `sources` reads, where it reads one.

    A source of a sample of several hazards names its component; a source of
    one hazard's view, `sample.component(name)`, is named by the view's step.
    """
    viewed: dict[str, str] = {}
    for sample in document["setup"]["samples"]:
        steps = [
            step
            for step in sample.get("selection") or ()
            if step.get("kind") == "component"
        ]
        if steps:
            viewed[str(sample["sample_hash"])] = str(steps[-1]["name"])
    return [
        entry.get("component") or viewed.get(str(entry.get("sample_hash")))
        for entry in document["sources"]
    ]


def _checksum_key(record_key: str) -> str:
    """The key of a source record's checksum: a canonical file's, or a table's."""
    return "file_sha256" if record_key == "canonical_file" else "table_sha256"


def _component_parts(records: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The components' source records, each once, with the components that read it."""
    parts: list[dict[str, Any]] = []
    texts: list[str] = []
    for name, record in records.items():
        text = json.dumps(record, sort_keys=True, default=str)
        if text in texts:
            parts[texts.index(text)]["components"].append(name)
            continue
        texts.append(text)
        parts.append({**record, "components": [name]})
    return parts


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


def _several_selected(name: str, count: int) -> ValueError:
    """The refusal of a selection of several hazards, horizons or pathways."""
    return ValueError(
        f"the plan samples one curve per asset, but the request selects {count} "
        f"{name}; select one with .select({name}=[...])"
        + (
            ""
            if name == "hazard_names"
            else ", or state the scenarios to read with .scenarios(ScenarioSet(...))"
        )
    )


def _source_pathway(source: Any) -> str | None:
    """The pathway a source plan's `explain()` states; `None` for other sources."""
    if _source_kind(source) != "plan" or not callable(getattr(source, "explain", None)):
        return None
    return _stated_pathway(_source_plan_document(source))


def _source_hazard(source: Any) -> str | None:
    """The hazard a source plan's `explain()` names; `None` for other sources."""
    if _source_kind(source) != "plan" or not callable(getattr(source, "explain", None)):
        return None
    explained = _source_plan_document(source)
    return _named_in_source(explained) if isinstance(explained, Mapping) else None


def _window_count(label: str, what: str, n: int, m: int, step: int = 1) -> None:
    """Raise `ScenarioError` unless `n` rows give every member of `m` as many."""
    unit = m * step
    if n % unit:
        lower = max(n // unit, 1) * unit
        whose = "" if step == 1 else f", its members times the reference's {step} rows"
        raise ScenarioError(
            f"scenario {label!r} is a window of {m} horizons, so {what} must be a "
            f"whole multiple of {unit}{whose}, got {n}; the nearest valid counts "
            f"are {lower} and {lower + unit}"
        )


# -- the plan ----------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class ParametricPlan:
    """A lazy plan from hazard curves or per-year values to compared covers.

    Start one with `ParametricPlan.for_evaluation(request)` (fitted curves)
    or `ParametricPlan.for_annual_extremes(...)` (per-year values); the
    builders (`monte_carlo`, `index`, `basis_risk`, `covers`, `objective`,
    `pricing`, `split`, `metrics`, `scopes`, `dependence`, `allocation`,
    `scenarios`, `hazard`, `combine`, `package`) return new plans, and neither
    they nor `explain()` do any I/O. The execution points `samples()`,
    `joint_sample()`, `scenario_samples()`, `designs()`, `payout_breakdown()`
    and `write_parquet(path)` read the source and the assets. The module
    docstring describes both routes.

    `entry` names the entry point. On the curve route `source` is the
    request, `unit` and `provenance` are `None` (the dataset's metadata
    states them at execution) and `draws` holds what `monte_carlo` states.
    `pricing_rule` holds what `.pricing(...)` states, `cell`, `trend` and
    `scenario` what `for_annual_extremes(...)` states, `index_read` what
    `.index(...)` states, and `scope_list`, `dependence_model` and
    `allocation_rule` what `.scopes(...)`, `.dependence(...)` and
    `.allocation(...)` state. `pool` holds what `for_annual_extremes(...)`
    states of a pool of model-years, and `source` is then a tuple of its sources
    when several were given. `hazard_name` is the name `for_annual_extremes`
    states, and `scenario_set`, `design_on` and `scenario_premiums` what
    `.scenarios(...)` states. `stated_pathway` and `names_hazard` are set on the
    plan a per-year scenario is read with: the pathway an `AnnualScenario`
    states, and that each series records the hazard its source plan names.
    `hazard_list`, `combine_rule` and `package_list` hold what `.hazard(...)`,
    `.combine(...)` and `.package(...)` state.
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
    scope_list: tuple[Individual | Joint | Clustered, ...] = (Individual(),)
    dependence_model: Dependence | DependenceModel | None = None
    allocation_rule: Allocation | None = None
    pool: ModelPool | None = None
    hazard_name: str | None = None
    scenario_set: ScenarioSet | None = None
    design_on: DesignOn = "each"
    scenario_premiums: tuple[tuple[str, float], ...] | None = None
    stated_pathway: str | None = None
    names_hazard: bool = False
    hazard_list: tuple[_Hazard, ...] = ()
    combine_rule: CombineRule | None = None
    package_list: tuple[_Package, ...] = ()

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
        seed=...)` states, so `.monte_carlo(...)` is required. A selection of
        several horizons or pathways is for `.scenarios(...)`, which reads each
        selected (pathway, horizon), or a window of them, as one scenario;
        without it, `explain()` and every execution point raise `ValueError`.

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
        request without an impact or a selection of several hazards; and
        `SampleError` for a missing asset column or an invalid loss model.
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
        if selection.hazard_names is not None and len(selection.hazard_names) > 1:
            raise _several_selected("hazard_names", len(selection.hazard_names))
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
        hazard_name: str | None = None,
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
        - `hazard_name` states the hazard the loss reads. It is optional where
          the source names one (a source plan's `recipe` or `hazard_name`, or
          a one-valued `recipe` column of the table), and must then agree, which
          `explain()` checks for a source plan and execution for a table. A
          scenario set needs every scenario's name
          (`.scenarios(...)`), so it is required there for a source that is
          silent, such as a plain table, a file or NOAA water levels. Each
          sample records the name in its provenance's `hazard`, and whether the
          caller stated it.

        Raises `TypeError` for arguments of the wrong type, `SampleError`
        for a unit the impact is not calibrated for, a missing asset column,
        an invalid `scenario` or a `cell` for an impact that takes no
        context, `ScenarioError` for a `hazard_name` a source plan
        contradicts, and `ValueError` for other invalid arguments, among them a
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
        if hazard_name is not None:
            hazard_name = _identifier("hazard_name", hazard_name)
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
            hazard_name=hazard_name,
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
        index is another series), `HazardNoise(sigma)`, `ImpactNoise(sigma,
        upper=...)` or a user model that implements `BasisRisk`. Required
        before execution. Raises `TypeError` for another type, and `ValueError`
        for `HazardNoise` or `ImpactNoise`, at every sigma, on a plan on
        per-year values without a seed (on fitted curves, `monte_carlo` states
        the seed).
        """
        if not isinstance(model, BasisRisk):
            raise TypeError(
                "model must be a BasisRisk model such as NoBasisRisk() or "
                f"ImpactNoise(sigma), got {type(model).__name__}"
            )
        if (
            isinstance(model, (HazardNoise, ImpactNoise))
            and self.seed is None
            and not self._on_curves
        ):
            raise ValueError(
                f"{type(model).__name__} draws one factor per year; give the plan "
                "a seed (for_annual_extremes(..., seed=...))"
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
        Under `Joint()` and `Clustered` a cover must be written for no asset, as
        it applies to every asset or cluster. With `.hazard(...)`, a cover may
        pay on a `CompositeIndex`,
        and each cover's index is checked against the components it reads, by
        name, unit and tail, at `explain()` and at every execution point, so
        only its payout unit is checked here. Replaces any covers set before.

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
        if self.hazard_list or isinstance(cover.index_definition, CompositeIndex):
            # Checked against the components by explain() and at execution, as
            # .hazard(...) may come after .covers(...).
            return
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
        if self.hazard_list:
            raise ValueError(_HAZARDS_REFUSE_INDEX)
        if self._clustered is not None:
            raise ValueError(_INDEX_REFUSES_CLUSTERED)
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
        """State the pricing of every premium of the plan.

        `pricing` is `PurePremium()`, `CostOfCapitalLoad(rate, q)` or
        `ExpectedPayoutMultiple(multiple, expense)`. One pricing prices every
        premium of the plan: the resolved covers' on each evaluation block
        and the designs' on their design block, so that every row of the
        comparison is priced alike. This builder states it for a plan of
        resolved covers, which needs no
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
        On fitted curves under `Joint()`, `Clustered` or `.hazard(...)`, whose
        rows read every column the dependence model placed, `Holdout` and `KFold` take
        `Independent()` only: under any other model `explain()` and every
        execution point raise `LeakageError`, naming `FreshDraws`.
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

    def scopes(self, *scopes: Individual | Joint | Clustered) -> ParametricPlan:
        """State how the assets are designed: `Individual()`, `Joint()`, `Clustered`.

        `Individual()`, the default, designs and evaluates each asset on its
        own sample. `Joint()` designs each template once per objective for one
        objective and one budget on the asset owner's total (`design_joint`),
        one cover per asset, on one sample of all the assets, and every
        evaluation gets a row at the level of that total.
        `Clustered(assignment, values=..., joint=...)` designs each template
        once per objective with `design_joint` under that scope, one cover per
        cluster on the value-weighted mean of its assets' indices, on the same
        sample of all the assets: under `joint=True` one objective and one
        budget on the total, under `joint=False` one design problem per
        cluster, each with its own budget. Its `values` must be those the
        plan's loss model reads (the asset column `value_column`, or 1 for a
        damage ratio), so that the weights are the plan's values; execution
        raises otherwise. A plan may hold several scopes, each once: the
        `scope` column keeps the rows of each scope apart.

        `Joint()` and `Clustered` need at least two assets; on fitted curves
        they also need `.dependence(...)`, and with `Holdout` or `KFold` that
        model must be `Independent()` (`LeakageError` otherwise, naming
        `FreshDraws`); and each cover must be written for no asset, as it
        applies to every asset or cluster. `Clustered`'s assignment must hold
        exactly the plan's assets. Those are checked by `explain()` and at every
        execution point, since builders may come in any order. Replaces any
        scopes stated before.

        Raises `TypeError` for another type; `ValueError` for no scope, a
        scope given twice, `Joint()` or `Clustered` on fewer than two assets
        (when the asset table is in memory; otherwise at execution), either
        beside `.hazard(...)`, and `Clustered` beside `.index(...)`; and
        `SampleError` for an assignment that leaves out an asset of the plan or
        names one it doesn't hold (when the asset table is in memory).
        """
        if not scopes:
            raise ValueError("scopes needs at least one scope")
        for scope in scopes:
            if not isinstance(scope, (Individual, Joint, Clustered)):
                raise TypeError(
                    "scopes must be Individual(), Joint() or Clustered(...), got "
                    f"{type(scope).__name__}"
                )
        kinds = [scope.to_document()["kind"] for scope in scopes]
        repeated = sorted({kind for kind in kinds if kinds.count(kind) > 1})
        if repeated:
            raise ValueError(f"scope {repeated[0]!r} is given twice")
        plan = dataclasses.replace(self, scope_list=tuple(scopes))
        if plan._together and self.hazard_list:
            raise ValueError(plan._hazards_refusal())
        clustered = plan._clustered
        if clustered is not None and self.index_read is not None:
            raise ValueError(_INDEX_REFUSES_CLUSTERED)
        ids = self._known_asset_ids()
        if ids is not None and plan._together:
            _require_several_assets(ids, plan._together_name())
            if clustered is not None:
                _check_assigned(clustered, ids)
        return plan

    def dependence(self, model: Dependence | DependenceModel) -> ParametricPlan:
        """State the dependence between curve draws: of the assets, or the hazards.

        For plans on fitted curves, where it is required with `Joint()`,
        `Clustered` and `.hazard(...)`: a fitted curve says nothing about how
        two places, or two
        hazards, move together, so `Independent()`, `Comonotonic()`,
        `EmpiricalCopula(reference)`, `GaussianCopula(correlation)` or a
        `DependenceModel` of your own is an assumption the plan states, in its
        metadata and its setup text. Under `Joint()` and `Clustered` the draws
        are those of `sample_curves`, with the plan's generators, the columns being the
        assets; with `.hazard(...)` those of `sample_curve_hazards`, the columns
        being the hazards, and an `EmpiricalCopula`'s reference holds exactly the
        plan's components. Without either nothing puts draws side by side, so
        `explain()` and every execution point raise. Replaces any dependence
        stated before.

        Raises `ValueError` on a plan on per-year values, whose dependence is
        the data's own: the rows are the same years of the data. Raises
        `TypeError` for another type, and `SampleError` for
        `EmpiricalCopula(...)` under `Holdout` or `KFold`; a model that doesn't
        pass the gate of `DependenceModel` raises `SampleError` at `explain()`
        and at every execution point.
        """
        if not self._on_curves:
            raise ValueError(
                "a plan on per-year values takes no dependence model: its "
                "dependence is the data's own, the rows being the same years of "
                "the data"
            )
        if isinstance(model, type) or not isinstance(model, DependenceModel):
            raise TypeError(
                "model must be Independent(), Comonotonic(), "
                "EmpiricalCopula(reference), GaussianCopula(correlation) or another "
                f"DependenceModel, got {type(model).__name__}"
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
        `"asset_by_loss"` and no premium or net-loss values, and they go in a
        table of their own, as `allocate` builds it (`allocate_scenarios` with
        `.scenarios(...)`): `write_parquet` writes them to a second file and
        returns them in `ParametricResult.allocated`, so they never sit beside
        an `"asset"` or `"total"` row. The values are
        those the plan's loss model reads (the asset column `value_column`, or
        1 for a damage ratio); only their hash is written. It applies to joint
        designs only, so `explain()` and every execution point raise without
        `Joint()` or `Clustered`. Under `Clustered` each cluster's payout is
        shared over that cluster's own assets, by the same rules. Replaces any
        allocation stated before.

        Raises `ValueError` for another rule.
        """
        if rule not in get_args(Allocation):
            raise ValueError(
                f"rule must be one of {get_args(Allocation)}, got {rule!r}"
            )
        return dataclasses.replace(self, allocation_rule=rule)

    def scenarios(
        self,
        scenarios: ScenarioSet,
        *,
        design_on: DesignOn = "each",
        premiums: Mapping[str, float] | None = None,
    ) -> ParametricPlan:
        """Read several scenarios, and put every cover or design on each one's block.

        `scenarios` is a `ScenarioSet` that describes what the plan reads, with a
        `reference` you name: the module chooses none. Each scenario's samples
        are built as the plan builds its own, every template is designed on the
        design blocks, and every cover and design is evaluated on every
        scenario's evaluation block with `compare_scenarios`, out of sample only,
        so the rows are the function route's. `scenario` says where a row is
        evaluated and `design_scenario` where it was designed.

        **On fitted curves** the request's `select(...)` must state exactly one
        hazard in `hazard_names`, and `horizons` and `pathways`, so that
        `explain()` lists the scenarios without reading anything. Without
        `labels=`, every selected (pathway, horizon) is one scenario, labelled
        `<pathway>_<horizon>`. `labels=` maps a label to its pairs instead and
        lists exactly the pairs read, each of them selected: several pairs of one
        pathway are a window of horizons, an equal-weight mixture of their curves
        on one stream of uniforms, and one pair is a single horizon. So
        `ScenarioSet(reference="2030s", labels={"2030s": [("ssp585", 2025),
        ("ssp585", 2030), ("ssp585", 2035)], ...})` is the whole statement. A
        `CurveScenario` among the set's entries reads another local canonical
        dataset. One `.monte_carlo(n_draws, seed=...)` serves every curve
        scenario: each asset's generators are rebuilt per scenario, from the
        plan's seed and from `FreshDraws.seed`, so the curve scenarios share one
        set of design draws and one of evaluation draws. `n_draws` and
        `FreshDraws(n)` must be whole multiples of every window's members.

        **On per-year values** the plan's own source is one scenario, under its
        `scenario=` label, and each `AnnualScenario` among the set's entries is
        another, such as another pathway's years. `.index(...)` applies to every
        scenario, on that scenario's hazard years.

        - `design_on="each"`, the default, designs every template under every
          objective on every scenario's design block, so that how a fixed cover
          drifts and a design per scenario are slices of one table;
          `"reference"` designs on the reference's only.
        - `premiums` maps a resolved cover's `cover_label` to a stated premium,
          its `premium_basis` `"stated"`. Another resolved cover is priced once,
          with the plan's pricing, on the reference's design block
          (`"reference"`). A design keeps its design premium. `premium_repriced`
          prices the same payouts on each block.
        - `.split(...)` is required, because the rows are out of sample only:
          `FreshDraws` on curves, `Holdout` or `KFold` on either route.

        The refusals of a `ScenarioSet` run on the samples at execution, through
        the function `ScenarioSet` itself runs. On per-year values every asset
        must read a scenario at one horizon, the centre of its series' whole
        window, so assets whose series span different years, as
        `missing="drop_year"` can leave them, raise `EvaluationError` at every
        execution point, once the samples are built and before anything is
        designed: state `years=` that every asset's series holds, or run those
        assets apart. `explain()` lists the scenarios, their sources and
        members, the reference and `design_on`, without I/O. Replaces any
        scenarios stated before.

        Raises `TypeError` for arguments of the wrong type and for a set of built
        samples, which `compare_scenarios` takes; `ValueError` for an
        `AnnualScenario` on fitted curves, a `CurveScenario` or `labels=` on
        per-year values, a `design_on` other than `"each"` or `"reference"`, and
        a selection that doesn't state one hazard, its horizons and its
        pathways; `ScenarioError` for a label used twice, a reference that is
        none of the scenarios, a pair of `labels=` the request doesn't select
        and a selected pathway `crc_sdk.pathways` doesn't know;
        `EvaluationError` for a premium that is not a finite non-negative
        number; and `SampleError` for an `AnnualScenario` in a unit the impact
        is not calibrated for.
        """
        if not isinstance(scenarios, ScenarioSet):
            raise TypeError(
                f"scenarios must be a ScenarioSet, got {type(scenarios).__name__}"
            )
        if not scenarios._describes:
            raise TypeError(
                "this set holds built samples, which compare_scenarios takes; a plan "
                "reads its scenarios, so describe them: ScenarioSet(reference=..., "
                "labels=...) on fitted curves, or AnnualScenario and CurveScenario "
                "entries"
            )
        if design_on not in get_args(DesignOn):
            raise ValueError(
                f"design_on must be one of {get_args(DesignOn)}, got {design_on!r}"
            )
        described = scenarios._described
        if self._on_curves:
            annual = [entry for entry in described if isinstance(entry, AnnualScenario)]
            if annual:
                raise ValueError(
                    f"AnnualScenario {annual[0].label!r} reads per-year values, and "
                    "this plan reads fitted curves (for_evaluation); state a "
                    "CurveScenario for another canonical dataset"
                )
            self._check_scenario_selection()
        else:
            curves = [entry for entry in described if isinstance(entry, CurveScenario)]
            if curves:
                raise ValueError(
                    f"CurveScenario {curves[0].label!r} reads fitted curves, and this "
                    "plan reads per-year values (for_annual_extremes); state an "
                    "AnnualScenario for another per-year source"
                )
            if scenarios._pairs is not None:
                raise ValueError(
                    "labels= names (pathway, horizon) pairs of the selection of a "
                    "plan on fitted curves; a per-year plan's scenarios are its own "
                    "source, under its scenario= label, and AnnualScenario entries"
                )
        plan = dataclasses.replace(
            self,
            scenario_set=scenarios,
            design_on=design_on,
            scenario_premiums=_checked_premiums(premiums),
        )
        for planned in plan._planned_scenarios():
            if isinstance(planned.described, AnnualScenario):
                plan._check_annual_units(planned.described)
        return plan

    # -- several hazards --------------------------------------------------------

    def hazard(
        self,
        name: str,
        source: Any = None,
        *,
        impact: Any = None,
        unit: str | None = None,
        tail: Tail | None = None,
        provenance: SourceProvenance | None = None,
        location: NearestCell | StationColumn | None = None,
        year_column: str | None = None,
        hazard_column: str | None = None,
        hazard_name: str | None = None,
    ) -> ParametricPlan:
        """Add a hazard read at each asset: an index component of its sample.

        A plan with `.hazard(...)` builds one sample of several hazards per
        asset, as `sample_annual_hazards` or `sample_curve_hazards` builds it,
        each component with its index, unit, tail and the loss of that hazard
        alone, and one combined loss per year under `.combine(rule)`. `name` is
        the component's name: non-empty, without surrounding whitespace and
        without `.`, `+`, `[` or `]`, and not taken for a hazard name. A cover on
        one index reads the component named its `hazard_name`, and a cover on a
        `CompositeIndex` the components of its names.

        **On per-year values** (`for_annual_extremes`), `.hazard(name,
        tail=...)` without a `source` names the plan's own hazard, whose tail
        (`"upper"` or `"lower"`) it states; it is the first component, whatever
        the order of the calls, and a plan with another component needs it.
        With a `source`, which takes what `for_annual_extremes` takes (a plan
        with an `annual_extremes()` execution point, an Arrow table or a local
        Parquet file; not a pool's runs), the component's series is read at
        each asset as the plan's own is:

        - `unit` (required) is the unit of its values, checked against its
          impact's calibration here, and against a `unit` column of the table
          and the source plan's `explain()` at execution;
        - `tail` (required) states its severe side;
        - `provenance` (required) its licence, attribution and version;
        - `impact` its impact, on the plan's value, `value_unit` and context, so
          the loss models have one value; without one the component is an index
          only, with no loss of its own, and needs no rule;
        - `location` finds its series, the plan's `location=` by default;
          `year_column` and `hazard_column` name its columns (`"year"` and
          `"value"` by default); `hazard_name` states the hazard it reads, as
          `for_annual_extremes(hazard_name=)` does for the plan's own.

        The plan's `years=`, `missing=` and `trend=` apply to every series, so
        the series must hold the same years: a row is one year of every hazard,
        and the setup text says that two annual maxima of one year may come from
        different events. One basis-risk model draws for every component, from
        the asset's generator, in component order.

        **On fitted curves** (`for_evaluation`), `source` is a second
        `PortfolioEvaluation` for the plan's assets, built as the plan's own
        request is, that selects at most one hazard, horizon and pathway, or
        the plan's horizons and pathways under `.scenarios(...)`; its
        `.impact(...)`, if it has one, is the component's impact, on the plan's
        value. The plan's own request is the first component, named by the one
        hazard it must select in `hazard_names`. A component's unit is its
        dataset's `value_unit` and its tail the dataset's `return_period_tail`,
        so only `name` and `source` are taken. `.dependence(...)` is required:
        it states how the hazards' draws are put side by side. The first
        component draws from the asset's child generator, so it draws what the
        plan with one hazard draws, and each other component from a generator
        spawned from that child, in component order.

        A cover on a `CompositeIndex` is designed on the combined loss; a cover
        on one index on its own hazard's loss, `sample.component(name)`, so a
        template on a component with no impact raises: design it on the
        function route. State `.hazard(...)` before `.covers(...)`: the covers'
        units and tails are then checked against the components by `explain()`
        and at every execution point. `Joint()`, `Clustered`, `pool=` and
        `.index(...)` raise on a plan with `.hazard(...)`. `explain()` lists
        the components without I/O, and `.samples()` returns one sample of
        several hazards per asset.

        Raises `TypeError` for arguments of the wrong type; `ValueError` for an
        invalid or repeated name, a keyword the route doesn't take, a missing
        `unit`, `tail` or `provenance`, a request for other assets or that
        selects several hazards, and for a plan that pools model-years, holds
        `Joint()` or `Clustered`, or reads its index elsewhere; and
        `SampleError` for a unit the
        impact is not calibrated for, a missing asset column or an impact that
        needs an H3 cell the plan doesn't give.
        """
        name = _component_name(name, ValueError)
        if self.pool is not None:
            raise ValueError(_HAZARDS_REFUSE_POOL)
        if self._together:
            raise ValueError(self._hazards_refusal())
        if self.index_read is not None:
            raise ValueError(_HAZARDS_REFUSE_INDEX)
        taken = [each.name for each in self.hazard_list]
        if self._on_curves:
            taken.append(self._primary_name())
        if name in taken:
            raise ValueError(f"the component {name!r} is given twice")
        if self._on_curves:
            stated = {
                "impact": impact,
                "unit": unit,
                "tail": tail,
                "provenance": provenance,
                "location": location,
                "year_column": year_column,
                "hazard_column": hazard_column,
                "hazard_name": hazard_name,
            }
            given = sorted(key for key, value in stated.items() if value is not None)
            if given:
                raise ValueError(
                    f"on fitted curves a component takes its impact from its "
                    "request's .impact(...), and its unit and tail from its "
                    f"dataset's metadata, so {given} are not taken; pass the request"
                )
            added = _Hazard(name, source, self._curve_component_model(source))
            return dataclasses.replace(self, hazard_list=(*self.hazard_list, added))
        if tail != "upper" and tail != "lower":
            raise ValueError(
                f"tail must state the component's severe side, 'upper' or 'lower', "
                f"got {tail!r}"
            )
        if source is None:
            stated_own = {
                "impact": impact,
                "unit": unit,
                "provenance": provenance,
                "location": location,
                "year_column": year_column,
                "hazard_column": hazard_column,
                "hazard_name": hazard_name,
            }
            extra = sorted(
                key for key, value in stated_own.items() if value is not None
            )
            if extra:
                raise ValueError(
                    f".hazard(name, tail=...) without a source names the plan's own "
                    f"hazard, whose {extra} are for_annual_extremes(...)'s; state "
                    "them there"
                )
            if any(each.source is None for each in self.hazard_list):
                raise ValueError("the plan's own hazard is named twice")
            own = _Hazard(name, None, None, tail=tail)
            return dataclasses.replace(self, hazard_list=(*self.hazard_list, own))
        added = self._annual_component(
            name,
            source,
            impact=impact,
            unit=unit,
            tail=tail,
            provenance=provenance,
            location=location,
            year_column="year" if year_column is None else year_column,
            hazard_column="value" if hazard_column is None else hazard_column,
            hazard_name=hazard_name,
        )
        return dataclasses.replace(self, hazard_list=(*self.hazard_list, added))

    def _annual_component(
        self,
        name: str,
        source: Any,
        *,
        impact: Any,
        unit: str | None,
        tail: Tail,
        provenance: SourceProvenance | None,
        location: NearestCell | StationColumn | None,
        year_column: str,
        hazard_column: str,
        hazard_name: str | None,
    ) -> _Hazard:
        """A per-year component read from another source, after its checks."""
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            raise TypeError(
                "a component reads one series per asset, so source must be one "
                "plan, table or file; pools are refused with .hazard(...)"
            )
        _source_kind(source)
        if unit is None:
            raise ValueError(f"component {name!r} needs unit=, the unit of its values")
        unit = _identifier("unit", unit)
        if not isinstance(provenance, SourceProvenance):
            raise TypeError(
                "provenance must be a SourceProvenance stating the source's licence "
                f"and attribution, got {type(provenance).__name__}"
            )
        if location is not None and not isinstance(
            location, (NearestCell, StationColumn)
        ):
            raise TypeError(
                "location must be NearestCell() or StationColumn(name), got "
                f"{type(location).__name__}"
            )
        _column_name("year_column", year_column)
        _column_name("hazard_column", hazard_column)
        if hazard_name is not None:
            hazard_name = _identifier("hazard_name", hazard_name)
        model: LossModel | None = None
        if impact is not None:
            plan_model = self.loss_model
            model = LossModel(
                impact,
                value_column=plan_model.value_column,
                value_unit=plan_model.value_unit,
                context=plan_model.context,
            )
            model.check_hazard_unit(unit)
            factor = _registry_needs_cell(
                impact, plan_model.context, self.assets, self.cell
            )
            if factor is not None:
                raise SampleError(
                    f"component {name!r}: impact {factor!r} needs an H3 cell in its "
                    "context, and this route reads none for an asset; state each "
                    "asset's cell with for_annual_extremes(..., cell=...), or give "
                    "the impact a context with the cell"
                )
        read_at = self.location if location is None else location
        columns = _table_columns(self.assets)
        if columns is not None and isinstance(read_at, StationColumn):
            if read_at.name not in columns:
                raise SampleError(f"the asset table has no columns {[read_at.name]}")
        return _Hazard(
            name,
            source,
            model,
            unit=unit,
            tail=tail,
            provenance=provenance,
            location=location,
            year_column=year_column,
            hazard_column=hazard_column,
            hazard_name=hazard_name,
        )

    def _curve_component_model(self, source: Any) -> LossModel | None:
        """A curve component's loss model: its request's impact on the plan's value.

        Raises unless `source` is a request for the plan's assets that selects at
        most one hazard.
        """
        if not isinstance(source, PortfolioEvaluation):
            raise TypeError(
                "on fitted curves a component's source is a PortfolioEvaluation for "
                "the plan's assets, as HazardDataset.local(path).for_assets(assets) "
                f"builds; got {type(source).__name__}"
            )
        names = source.selection.hazard_names
        if names is not None and len(names) > 1:
            raise _several_selected("hazard_names", len(names))
        own: PortfolioEvaluation = self.source
        if not _same_assets(source.portfolio, own.portfolio):
            raise ValueError(
                "a component's request must be for the plan's assets: build it on "
                "the same assets, with .for_assets(...) of the plan's request"
            )
        spec = source.impact_spec
        if spec is None:
            return None
        impact, context = _request_impact(spec)
        plan_model = self.loss_model
        columns = _table_columns(source.portfolio)
        if columns is not None and context is not None:
            absent = [name for _, name in context.items() if name not in columns]
            if absent:
                raise SampleError(f"the asset table has no columns {absent}")
        return LossModel(
            impact,
            value_column=plan_model.value_column,
            value_unit=plan_model.value_unit,
            context=context,
            name=spec.name,
        )

    def combine(self, rule: CombineRule) -> ParametricPlan:
        """State the rule of the combined loss of a plan with several hazards.

        Required whenever two components or more have an impact, with no
        default, because how the hazards' impacts make one year's loss is a
        statement about the asset: `CappedSum(cap)`, `CompoundDamage()` or a
        `CombineRule` of your own, which passes the gate of `sample_annual_hazards`.
        Each component's impact comes from its loss model, after the basis-risk
        model; the rule takes the impacts, and the combined loss is its result
        times the value. With fewer than two impacts it raises at `explain()` and
        at every execution point, rather than being ignored. Replaces any rule
        stated before.

        Raises `TypeError` for anything but a `CombineRule`, a bare callable
        included, and `SampleError` for a document or sentence that doesn't pass
        the gate.
        """
        _rule_record(rule)
        return dataclasses.replace(self, combine_rule=rule)

    def package(
        self,
        covers: Mapping[str, str],
        *,
        label: str,
        shares: Literal["own_loss"] | Mapping[str, float] | None = None,
    ) -> ParametricPlan:
        """Make separate contracts one row of the comparison: a `CoverPackage`.

        `covers` maps each component to the `cover_label` of a cover on one
        index in `.covers(...)` (its label, or the first 12 hex digits of its
        hash): two or more, all templates or all resolved, each in one package
        at most. The key is the component the contract reads, so it need not be
        the cover's `hazard_name`. On each design block the package holds the
        contracts as designed there, each on its own hazard's loss
        (`sample.component(name)`), one package per objective, labelled `label`
        followed by the objective, or the resolved covers as they are, labelled
        `label`. Its rows are `CoverPackage`'s: one `"asset"` row per contract and
        one `"package"` row, whose metrics are on the summed payout against the
        combined loss and whose premium is the sum of the contracts', each on
        its own terms. Its contracts are not compared again on their own. The
        package sits where its first contract sits in `.covers(...)`.

        **The budget.** With a budget on `.objective(...)` and a package of
        templates, `shares` is required: the budget is that of one contract on
        every hazard, and each contract takes `split_budget(budget, block,
        shares=shares)[component]` on each design block, so the bounds sum to the
        one bound. `"own_loss"` splits it by each component's share of the summed
        expected own losses, and a mapping by component states the shares, which
        must name every contract and sum to 1. How to split a budget between
        hazards is an assumption, so there is no default. A template named in no
        package takes the budget as stated. Without a budget, or for a package of
        resolved covers, `shares` raises.

        Checked against the covers and the components by `explain()` and at
        every execution point, as builders come in any order. Replaces any
        package stated before under the same `label`.

        Raises `TypeError` for arguments of the wrong type, and `ValueError` for
        fewer than two contracts, an invalid component name or label; at
        `explain()` and at execution, `ValueError` for a label that names no
        cover, or several, a cover on several components, a mix of templates and
        resolved covers, a cover in two packages, a component the plan doesn't
        hold, `shares` where it is refused, a missing `shares`, or a package
        without `.hazard(...)`.
        """
        if not isinstance(covers, Mapping):
            raise TypeError(
                "covers must map each component to the cover_label of its contract, "
                f"got {type(covers).__name__}"
            )
        if len(covers) < 2:
            raise ValueError(
                f"a package holds two contracts or more, got {len(covers)}"
            )
        pairs: list[tuple[str, str]] = []
        for component, cover_label in covers.items():
            component = _component_name(component, ValueError)
            if not isinstance(cover_label, str) or not cover_label:
                raise TypeError(
                    f"covers[{component!r}] must be the cover_label of a cover in "
                    f".covers(...), got {cover_label!r}"
                )
            pairs.append((component, cover_label))
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"label must be a non-empty string, got {label!r}")
        held: Literal["own_loss"] | tuple[tuple[str, float], ...] | None
        if shares is None or shares == "own_loss":
            held = shares
        elif isinstance(shares, Mapping):
            held = tuple((str(key), value) for key, value in shares.items())
        else:
            raise TypeError(
                "shares must be 'own_loss' or a mapping from component to share, "
                f"got {shares!r}"
            )
        stated = _Package(tuple(pairs), label, held)
        others = tuple(each for each in self.package_list if each.label != label)
        return dataclasses.replace(self, package_list=(*others, stated))

    # -- what a plan with several hazards reads ------------------------------------

    def _primary_name(self) -> str:
        """The first component's name: on curves, the hazard the request selects."""
        if not self._on_curves:
            own = [each for each in self.hazard_list if each.source is None]
            if not own:
                raise ValueError(
                    "a per-year plan with .hazard(name, source, ...) names its own "
                    "hazard too, the first component: state .hazard(name, "
                    "tail=...) without a source"
                )
            return own[0].name
        names = self.source.selection.hazard_names
        if names is None or len(names) != 1:
            raise ValueError(
                "with .hazard(...), the plan's request must select exactly one "
                "hazard in hazard_names, which names its component, the first"
            )
        return _component_name(str(names[0]), ValueError)

    def _components(self) -> list[_Hazard]:
        """Every component, the plan's own first, each with its read filled in."""
        if not self.hazard_list:
            return []
        if self._on_curves:
            own = _Hazard(self._primary_name(), self.source, self.loss_model)
            return [own, *self.hazard_list]
        first = self._primary_name()
        named = next(each for each in self.hazard_list if each.name == first)
        own = dataclasses.replace(
            named,
            loss_model=self.loss_model,
            unit=self.unit,
            provenance=self.provenance,
            location=self.location,
            year_column=self.year_column,
            hazard_column=self.hazard_column,
            hazard_name=self.hazard_name,
        )
        others = [each for each in self.hazard_list if each.source is not None]
        return [own, *others]

    def _check_annual_units(self, entry: AnnualScenario) -> None:
        """Check an `AnnualScenario`'s units against the impacts that read them.

        A scenario of several hazards is checked once the components are known,
        at `explain()` and at execution.
        """
        if isinstance(entry.unit, str):
            self.loss_model.check_hazard_unit(entry.unit)
            return
        if not isinstance(entry.unit, Mapping) or not self._complete_hazards():
            return
        for component in self._components():
            unit = entry.unit.get(component.name)
            if unit is not None and component.loss_model is not None:
                component.loss_model.check_hazard_unit(unit)

    def _complete_hazards(self) -> bool:
        """Whether every component is known: on per-year values, the own one named."""
        if not self.hazard_list:
            return False
        return self._on_curves or any(each.source is None for each in self.hazard_list)

    def _check_scenario_selection(self) -> None:
        """Raise unless the request states what `explain()` lists the scenarios by."""
        selection = self.source.selection
        names = selection.hazard_names
        if names is None or len(names) != 1:
            raise ValueError(
                "with .scenarios(...), the request's select(...) must state exactly "
                "one hazard in hazard_names, so that explain() names the hazard every "
                "scenario reads without reading anything"
            )
        for name in ("horizons", "pathways"):
            if getattr(selection, name) is None:
                raise ValueError(
                    f"with .scenarios(...), the request's select(...) must state "
                    f"{name}, so that explain() lists the scenarios without reading "
                    "anything"
                )

    def _selected_pairs(self) -> list[tuple[str, int]]:
        """Every (pathway, horizon) the request selects, pathway by pathway."""
        selection = self.source.selection
        return [
            (str(pathway), int(horizon))
            for pathway in selection.pathways or ()
            for horizon in selection.horizons or ()
        ]

    def _planned_scenarios(self) -> list[_Planned]:
        """The plan's scenarios, the reference first, then as given.

        On curves the selection's come first, then each `CurveScenario`; on
        per-year values the plan's own source, then each `AnnualScenario`.
        """
        stated = self.scenario_set
        assert stated is not None
        planned: list[_Planned] = []
        if self._on_curves:
            selected = self._selected_pairs()
            pairs = stated._pairs
            if pairs is None:
                for pathway, horizon in selected:
                    label = _identifier("scenario", f"{pathway}_{horizon}")
                    _check_known_pathway(pathway, label)
                    planned.append(_Planned(label, "selection", ((pathway, horizon),)))
            else:
                for label, members in pairs.items():
                    for pair in members:
                        if not any(_same_pair(pair, each) for each in selected):
                            raise ScenarioError(
                                f"scenario {label!r} lists the (pathway, horizon) pair "
                                f"{pair}, which the request doesn't select; it selects "
                                f"{selected}, and labels= lists the pairs read"
                            )
                    planned.append(_Planned(label, "selection", tuple(members)))
            for entry in stated._described:
                assert isinstance(entry, CurveScenario)
                horizons = entry.horizons
                stated_pairs = (
                    None
                    if entry.pathway is None or horizons is None
                    else tuple((entry.pathway, horizon) for horizon in horizons)
                )
                planned.append(_Planned(entry.label, "dataset", stated_pairs, entry))
        else:
            planned.append(_Planned(self.scenario, "own"))
            for entry in stated._described:
                assert isinstance(entry, AnnualScenario)
                planned.append(_Planned(entry.label, "annual", None, entry))
        labels = [each.label for each in planned]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ScenarioError(
                f"two scenarios are labelled {repeated[0]!r}: each scenario has its "
                "own label"
                + (
                    ""
                    if self._on_curves
                    else ", and the plan's own source is labelled by its scenario="
                )
            )
        if stated.reference not in labels:
            raise ScenarioError(
                f"the reference {stated.reference!r} must be one of the plan's "
                f"scenarios, {labels}"
            )
        return sorted(planned, key=lambda each: each.label != stated.reference)

    def _scenario_plan(self, planned: _Planned) -> ParametricPlan:
        """The plan one per-year scenario is read with: the plan's, or its own source.

        The loss model, the cell, the seed, the covers and the rest are the plan's.
        Every scenario's series record the hazard their source plan names.
        """
        base = dataclasses.replace(
            self, scenario_set=None, scenario_premiums=None, names_hazard=True
        )
        entry = planned.described
        if not isinstance(entry, AnnualScenario):
            return base
        if isinstance(entry.source, Mapping):
            return self._hazard_scenario_plan(base, entry)
        pool = self.pool if entry.pool is None else (entry.pool or None)
        if pool is None and isinstance(entry.source, tuple):
            raise ValueError(
                f"scenario {entry.label!r} gives a sequence of sources, which are a "
                "pool's runs; state pool=ModelPool() on it or on the plan"
            )
        # One source, so AnnualScenario holds no mapping in these fields.
        provenance, unit, location = entry.provenance, entry.unit, entry.location
        year_column, hazard_column = entry.year_column, entry.hazard_column
        hazard_name = entry.hazard_name
        assert isinstance(provenance, SourceProvenance)
        assert not isinstance(unit, Mapping) and not isinstance(location, Mapping)
        assert not isinstance(year_column, Mapping)
        assert not isinstance(hazard_column, Mapping)
        assert not isinstance(hazard_name, Mapping)
        return dataclasses.replace(
            base,
            source=entry.source,
            provenance=provenance,
            unit=self.unit if unit is None else unit,
            location=self.location if location is None else location,
            year_column=self.year_column if year_column is None else year_column,
            hazard_column=(
                self.hazard_column if hazard_column is None else hazard_column
            ),
            years=self.years if entry.years is None else tuple(entry.years),
            missing=self.missing if entry.missing is None else entry.missing,
            trend=self.trend if entry.trend is None else entry.trend,
            pool=pool,
            scenario=entry.label,
            hazard_name=hazard_name,
            stated_pathway=entry.pathway,
        )

    def _hazard_scenario_plan(
        self, base: ParametricPlan, entry: AnnualScenario
    ) -> ParametricPlan:
        """The plan a per-year scenario of several hazards is read with.

        Each component reads its own source of `entry`. A per-source field left
        out takes that component's value on the plan, and `hazard_name` is
        never the plan's: a scenario states each name, where its source is
        silent.
        """
        sources: Mapping[str, Any] = entry.source

        def stated(field_name: str, component: str) -> Any:
            value = getattr(entry, field_name)
            return value.get(component) if isinstance(value, Mapping) else None

        provenance: Mapping[str, SourceProvenance] = entry.provenance  # type: ignore[assignment]
        replaced: list[_Hazard] = []
        for component in self.hazard_list:
            if component.source is None:
                replaced.append(component)
                continue
            name = component.name
            own_location = (
                self.location if component.location is None else component.location
            )
            replaced.append(
                dataclasses.replace(
                    component,
                    source=sources[name],
                    provenance=provenance[name],
                    unit=stated("unit", name) or component.unit,
                    location=stated("location", name) or own_location,
                    year_column=stated("year_column", name) or component.year_column,
                    hazard_column=(
                        stated("hazard_column", name) or component.hazard_column
                    ),
                    hazard_name=stated("hazard_name", name),
                )
            )
        primary = self._primary_name()
        return dataclasses.replace(
            base,
            source=sources[primary],
            provenance=provenance[primary],
            unit=stated("unit", primary) or self.unit,
            location=stated("location", primary) or self.location,
            year_column=stated("year_column", primary) or self.year_column,
            hazard_column=stated("hazard_column", primary) or self.hazard_column,
            years=self.years if entry.years is None else tuple(entry.years),
            missing=self.missing if entry.missing is None else entry.missing,
            trend=self.trend if entry.trend is None else entry.trend,
            pool=None,
            scenario=entry.label,
            hazard_name=stated("hazard_name", primary),
            stated_pathway=entry.pathway,
            hazard_list=tuple(replaced),
        )

    def _check_scenario_plan(self) -> None:
        """Check what no single builder can, for a plan with or without scenarios.

        Without scenarios, a plan on curves reads one horizon and one pathway.
        With them, the rows are out of sample, so a split is required; every
        window's members divide the draws; and the premiums name resolved covers.
        """
        stated = self.scenario_set
        if stated is None:
            if self._on_curves:
                selection = self.source.selection
                for name in ("horizons", "pathways"):
                    values = getattr(selection, name)
                    if values is not None and len(values) > 1:
                        raise _several_selected(name, len(values))
            self._check_hazard_name()
            return
        if self.split_rule is None:
            raise ValueError(
                "a plan with scenarios needs .split(...): its rows are out of sample "
                "only, each scenario's design block apart from its evaluation block "
                "(FreshDraws on fitted curves, Holdout or KFold on either route)"
            )
        planned = self._planned_scenarios()
        draws = self.draws
        rule = self.split_rule
        moved = self._together or bool(self.hazard_list)
        model = self.dependence_model if moved else None
        step = len(model.reference) if isinstance(model, EmpiricalCopula) else 1
        for each in planned:
            self._check_described(each)
            if each.members is not None and len(each.members) > 1:
                m = len(each.members)
                if draws is not None:
                    _window_count(each.label, "n_draws", draws.n_draws, m, step)
                if isinstance(rule, FreshDraws):
                    _window_count(each.label, "FreshDraws(n)", rule.n_draws, m, step)
            if each.kind in ("own", "annual"):
                variant = self._scenario_plan(each)
                if variant.pool is not None and self.index_read is not None:
                    refusal = variant._index_refusal()
                    assert refusal is not None
                    raise ValueError(f"scenario {each.label!r}: {refusal}")
                if variant.pool is not None and isinstance(
                    variant.location, StationColumn
                ):
                    raise ValueError(
                        f"scenario {each.label!r} pools climate-model runs, and "
                        "models have no stations; locate each asset with "
                        "NearestCell()"
                    )
                variant._check_hazard_name()
        given = {
            _cover_label(cover)
            for cover in self.cover_list
            if not cover.free_parameters()
        }
        unknown = sorted(
            key for key, _ in self.scenario_premiums or () if key not in given
        )
        if unknown:
            raise EvaluationError(
                f"premiums= names {unknown}, which match no cover given as is; it "
                f"maps the cover_label of a resolved cover, here {sorted(given)}"
            )

    def _check_described(self, planned: _Planned) -> None:
        """A described scenario's sources: one, or one per component.

        On a plan with several hazards an `AnnualScenario`'s `source` and a
        `CurveScenario`'s `dataset` map each component to its own; on any other
        plan they are one source.
        """
        entry = planned.described
        if entry is None:
            return
        given: Any = (
            entry.source if isinstance(entry, AnnualScenario) else entry.dataset
        )
        several = isinstance(given, Mapping)
        if not self.hazard_list:
            if several:
                raise ValueError(
                    f"scenario {entry.label!r} maps its sources by component, and "
                    "this plan reads one hazard per asset; give one source, or state "
                    "the hazards with .hazard(...)"
                )
            return
        names = [each.name for each in self._components()]
        if not several:
            raise ValueError(
                f"scenario {entry.label!r} gives one source, and this plan reads the "
                f"hazards {names}: map each component to its own"
            )
        if sorted(given) != sorted(names):
            raise ValueError(
                f"scenario {entry.label!r} maps the components {list(given)}, and the "
                f"plan's are {names}"
            )
        if isinstance(entry, AnnualScenario):
            self._check_annual_units(entry)

    def _check_hazard_name(self) -> None:
        """Raise `ScenarioError` for a stated hazard name a source plan contradicts.

        The plan's own, and each per-year component's.
        """
        if self._on_curves:
            return
        stated = self.hazard_name
        sources = self._pool_sources() if self.pool is not None else (self.source,)
        for source in sources if stated is not None else ():
            named = _source_hazard(source)
            if named is not None and named != stated:
                raise ScenarioError(
                    f"scenario {self.scenario!r}: hazard_name={stated!r} contradicts "
                    f"the source plan, which names {named!r}; a stated name never "
                    "overrides the record"
                )
        for component in self.hazard_list:
            if component.source is None or component.hazard_name is None:
                continue
            named = _source_hazard(component.source)
            if named is not None and named != component.hazard_name:
                raise ScenarioError(
                    f"scenario {self.scenario!r}, component {component.name!r}: "
                    f"hazard_name={component.hazard_name!r} contradicts the source "
                    f"plan, which names {named!r}; a stated name never overrides the "
                    "record"
                )

    def _check_hazard_plan(self) -> None:
        """Check a plan with several hazards, or that a plan without them needs none.

        Run by `explain()` and every execution point, as `.hazard(...)` may come
        after the builders it constrains.
        """
        if not self.hazard_list:
            if self.combine_rule is not None:
                raise ValueError(
                    ".combine(...) states the rule of the combined loss of several "
                    "hazards; state the hazards with .hazard(...)"
                )
            if self.package_list:
                raise ValueError(
                    ".package(...) makes separate contracts on several hazards one "
                    "row; state the hazards with .hazard(...)"
                )
            for cover in self.cover_list:
                index = cover.index_definition
                if isinstance(index, CompositeIndex):
                    raise SampleError(
                        f"cover {cover.label or cover.definition_hash!r} pays on the "
                        f"components {list(index.names)}, and this plan reads one "
                        "hazard per asset; state the others with .hazard(...)"
                    )
            return
        if self._together:
            raise ValueError(self._hazards_refusal())
        if self.pool is not None:
            raise ValueError(_HAZARDS_REFUSE_POOL)
        if self.index_read is not None:
            raise ValueError(_HAZARDS_REFUSE_INDEX)
        components = self._components()
        if len(components) < 2:
            raise ValueError(
                "a plan with .hazard(...) reads two hazards or more at each asset; "
                "add another with .hazard(name, source, ...)"
            )
        _combine_record(
            self.combine_rule,
            [each.name for each in components if each.loss_model is not None],
        )
        if self._on_curves:
            self._check_curve_hazards(components)
        by_name = {each.name: each for each in components}
        self._check_hazard_covers(by_name)
        self._check_packages(by_name, self._package_members())

    def _check_curve_hazards(self, components: Sequence[_Hazard]) -> None:
        """The dependence across the hazards, and each component's selection."""
        model = self.dependence_model
        if model is None:
            raise ValueError(
                "several hazards on fitted curves need a stated dependence between "
                "them, because fitted curves say nothing about how two hazards move "
                "together: state .dependence(Independent()), "
                ".dependence(Comonotonic()), .dependence(GaussianCopula(correlation)) "
                "or .dependence(EmpiricalCopula(reference))"
            )
        names = [each.name for each in components]
        rule = self.split_rule
        if isinstance(rule, (Holdout, KFold)) and not isinstance(
            model, (Independent, EmpiricalCopula)
        ):
            raise _moved_split_error(rule, model)
        for count in (
            None if self.draws is None else self.draws.n_draws,
            rule.n_draws if isinstance(rule, FreshDraws) else None,
        ):
            if count is not None:
                _model_record(model, count, names, "hazards")
        own: PortfolioEvaluation = self.source
        for component in components[1:]:
            request: PortfolioEvaluation = component.source
            selection = request.selection
            if self.scenario_set is None:
                for axis in ("horizons", "pathways"):
                    values = getattr(selection, axis)
                    if values is not None and len(values) > 1:
                        raise ValueError(
                            f"component {component.name!r}: "
                            + str(_several_selected(axis, len(values)))
                        )
                continue
            names_selected = selection.hazard_names
            if names_selected is None or len(names_selected) != 1:
                raise ValueError(
                    f"with .scenarios(...), component {component.name!r}'s request "
                    "must select exactly one hazard in hazard_names, so that "
                    "explain() names the hazard it reads without reading anything"
                )
            for axis in ("horizons", "pathways"):
                stated = getattr(selection, axis)
                planned = getattr(own.selection, axis)
                if stated is not None and sorted(map(str, stated)) != sorted(
                    map(str, planned or ())
                ):
                    raise ValueError(
                        f"with .scenarios(...), component {component.name!r}'s "
                        f"request selects the {axis} {list(stated)}, but the plan's "
                        f"selects {_listed(planned)}: a scenario reads one (pathway, "
                        "horizon) of every hazard, so select the plan's, or none"
                    )

    def _package_members(self) -> list[dict[str, int]]:
        """Each package's contracts, by component: their positions in `.covers`.

        Raises `ValueError` for a label that names no cover or several, a cover
        on several components, a mix of templates and resolved covers, covers
        written for different assets, and a cover in two packages.
        """
        positions: dict[str, list[int]] = {}
        for position, cover in enumerate(self.cover_list):
            positions.setdefault(_cover_label(cover), []).append(position)
        packaged: dict[int, str] = {}
        members: list[dict[str, int]] = []
        for package in self.package_list:
            found: dict[str, int] = {}
            for component, cover_label in package.covers:
                held = positions.get(cover_label, [])
                if len(held) != 1:
                    raise ValueError(
                        f"package {package.label!r} names the cover_label "
                        f"{cover_label!r}, which "
                        + ("no cover" if not held else f"{len(held)} covers")
                        + " in .covers(...) has; it names one cover each, by its "
                        f"label or the first 12 hex digits of its hash, of "
                        f"{sorted(positions)}"
                    )
                position = held[0]
                cover = self.cover_list[position]
                if isinstance(cover.index_definition, CompositeIndex):
                    raise ValueError(
                        f"package {package.label!r}: cover {cover_label!r} pays on "
                        "several components, and a package holds contracts on one "
                        "index each"
                    )
                if position in packaged:
                    raise ValueError(
                        f"cover {cover_label!r} is a contract of the packages "
                        f"{packaged[position]!r} and {package.label!r}; a cover is "
                        "in one package at most, as its rows would repeat"
                    )
                packaged[position] = package.label
                found[component] = position
            covers = [self.cover_list[position] for position in found.values()]
            if len({not cover.free_parameters() for cover in covers}) > 1:
                raise ValueError(
                    f"package {package.label!r} holds templates and resolved covers; "
                    "a package holds contracts designed alike: all templates, or "
                    "all resolved"
                )
            if len({cover.asset_id for cover in covers}) > 1:
                raise ValueError(
                    f"package {package.label!r} holds covers written for different "
                    "assets; a package is one asset's contracts"
                )
            members.append(found)
        return members

    def _check_hazard_covers(self, by_name: Mapping[str, _Hazard]) -> None:
        """Each cover against the components it reads: their names, units, tails.

        A contract of a package reads the component its package names; any
        other cover on one index the component named its `hazard_name`.
        """
        names = list(by_name)
        templates = {
            cover.label or cover.definition_hash
            for cover in self.cover_list
            if cover.free_parameters()
            and not isinstance(cover.index_definition, CompositeIndex)
        }
        for label, component_name, index, packaged in self._cover_reads():
            if not isinstance(index.location, AtAsset):
                raise SampleError(
                    f"cover {label!r} reads its index at {index.location!r}, and a "
                    "plan with .hazard(...) reads each component's index at the "
                    "asset (AtAsset())"
                )
            component = by_name.get(component_name)
            if component is None:
                raise SampleError(
                    f"cover {label!r} reads the component {component_name!r}, and "
                    f"the plan's components are {names}: a cover on one index reads "
                    "the component named its hazard_name, or the one its "
                    ".package(...) names"
                )
            if component.unit is not None and index.unit != component.unit:
                raise SampleError(
                    f"cover {label!r} pays on an index in {index.unit!r}, but "
                    f"component {component_name!r} is in {component.unit!r}"
                )
            if component.tail is not None and index.tail != component.tail:
                raise SampleError(
                    f"{_contract(label, packaged)} has the tail {index.tail!r} on "
                    f"component {component_name!r}, whose tail is "
                    f"{component.tail!r}"
                )
            if label in templates and component.loss_model is None:
                raise SampleError(
                    f"template {label!r} on one index is designed on the own loss of "
                    f"component {component_name!r}, which has no impact and so no "
                    "loss of its own; design it on the combined loss on the function "
                    f"route: design(template, sample.component({component_name!r}, "
                    "loss='combined'), ...)"
                )

    def _check_packages(
        self, by_name: Mapping[str, _Hazard], members: Sequence[Mapping[str, int]]
    ) -> None:
        """Each package's components, and its shares against the budget."""
        spec = self.design_spec
        budget = None if spec is None else spec.budget
        for package, found in zip(self.package_list, members, strict=True):
            unknown = [name for name in package.components if name not in by_name]
            if unknown:
                raise ValueError(
                    f"package {package.label!r} names the components {unknown}, and "
                    f"the plan's are {list(by_name)}"
                )
            templates = any(
                self.cover_list[position].free_parameters()
                for position in found.values()
            )
            shares = package.split_shares()
            if not templates:
                if shares is not None:
                    raise ValueError(
                        f"package {package.label!r} holds resolved covers, which have "
                        "no budget to split: leave shares= out"
                    )
                continue
            if budget is None:
                if shares is not None:
                    raise ValueError(
                        f"package {package.label!r} states shares=, but "
                        ".objective(...) states no budget to split; leave shares= out, "
                        "or state the budget"
                    )
                continue
            if shares is None:
                raise ValueError(
                    f"package {package.label!r} holds templates, and .objective(...) "
                    "states a budget, the budget of one contract on every hazard: "
                    "state how to split it between the contracts with shares="
                    "'own_loss' or a mapping by component; there is no default"
                )
            if isinstance(shares, dict):
                if sorted(shares) != sorted(package.components):
                    raise ValueError(
                        f"package {package.label!r}: shares= names {sorted(shares)}, "
                        f"and must name its contracts' components, "
                        f"{sorted(package.components)}"
                    )
                _stated_shares(shares, dict.fromkeys(package.components))

    # -- scopes -----------------------------------------------------------------

    @property
    def _holds_joint(self) -> bool:
        """Whether the plan holds `Joint()`."""
        return any(isinstance(scope, Joint) for scope in self.scope_list)

    @property
    def _clustered(self) -> Clustered | None:
        """The plan's `Clustered` scope, if it holds one."""
        for scope in self.scope_list:
            if isinstance(scope, Clustered):
                return scope
        return None

    @property
    def _together(self) -> bool:
        """Whether the plan designs assets together, on one sample of all of them.

        Under `Joint()` or `Clustered`: either builds the sample of all the
        assets, and its designs read several assets.
        """
        return self._holds_joint or self._clustered is not None

    def _together_name(self) -> str:
        """The scopes that design assets together, as a message names them."""
        names = [
            *(["Joint()"] if self._holds_joint else []),
            *(["Clustered(...)"] if self._clustered is not None else []),
        ]
        return " and ".join(names) or "Joint()"

    def _hazards_refusal(self) -> str:
        """The refusal of `.hazard(...)` beside a scope that designs assets together."""
        return _HAZARDS_REFUSE_JOINT if self._holds_joint else _HAZARDS_REFUSE_CLUSTERED

    @property
    def _holds_individual(self) -> bool:
        return any(isinstance(scope, Individual) for scope in self.scope_list)

    def _check_scopes(self) -> None:
        """Check what no single builder can, as the builders come in any order."""
        model = self.dependence_model
        if not self._together:
            if model is not None and not self.hazard_list:
                raise ValueError(
                    ".dependence(...) says how the assets are put side by side, "
                    "and only Joint() or Clustered(...) does that; state "
                    ".scopes(Joint()), or the hazards it puts side by side with "
                    ".hazard(...)"
                )
            if self.allocation_rule is not None:
                raise ValueError(
                    ".allocation(...) shares out a joint design's total payout; "
                    "state .scopes(Joint()) or .scopes(Clustered(...))"
                )
            return
        clustered = self._clustered
        if clustered is not None:
            if self.index_read is not None:
                raise ValueError(_INDEX_REFUSES_CLUSTERED)
            ids = self._known_asset_ids()
            if ids is not None:
                _check_assigned(clustered, ids)
        if model is None and self._on_curves:
            raise ValueError(
                f"{self._together_name()} on fitted curves needs a stated dependence "
                "between the assets, because a fitted curve says nothing about how "
                "two places move together: state .dependence(Independent()), "
                ".dependence(Comonotonic()) or .dependence(EmpiricalCopula(reference))"
            )
        rule = self.split_rule
        # The joint rows read every asset, so a block of rows that a model placed
        # by the ranks over both blocks is not out of sample. EmpiricalCopula's
        # refusal is the builders' (`_copula_split_error`).
        if (
            model is not None
            and isinstance(rule, (Holdout, KFold))
            and not isinstance(model, (Independent, EmpiricalCopula))
        ):
            raise _moved_split_error(rule, model, columns="assets")
        for cover in self.cover_list:
            if cover.asset_id is not None:
                raise ValueError(
                    f"cover {cover.label or cover.definition_hash!r} is written for "
                    f"asset {cover.asset_id!r}, but under {self._together_name()} "
                    "each cover applies to every asset; write it for no asset, or "
                    "give different terms per asset to design_joint"
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
        """The levels, scopes and allocation of the rows of the scopes together."""
        designed = any(cover.free_parameters() for cover in self.cover_list)
        fixed = any(not cover.free_parameters() for cover in self.cover_list)
        individual = fixed or (designed and self._holds_individual)
        together = designed or not self.cover_list
        clustered = self._clustered
        cluster = together and clustered is not None
        rule = self.allocation_rule
        levels = [
            *(["asset"] if individual else []),
            *(["cluster"] if cluster and clustered and not clustered.joint else []),
            "total",
            *([f"asset_by_{rule}"] if together and rule is not None else []),
        ]
        scopes = [
            *(["individual"] if individual else []),
            *(["joint"] if together and self._holds_joint else []),
            *(["cluster"] if cluster else []),
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
        rows = self._planned_rows() if self._together else {}
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
        """The hazard's sources, as far as they are known: one entry per source.

        With several hazards, one entry per component, which names it.
        """
        if self.hazard_list:
            return [
                {
                    **(
                        self._source_description(component.source)
                        if self._on_curves
                        else self._annual_description(component)
                    ),
                    "component": component.name,
                }
                for component in self._components()
            ]
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
            return {
                "impact": (
                    None
                    if spec is None
                    else {
                        "name": spec.name,
                        "value_semantics": spec.value_semantics,
                        "value_unit": spec.value_unit,
                    }
                ),
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
        assert self.provenance is not None and self.unit is not None
        return self._annual_description(
            _Hazard(
                "",
                source,
                unit=self.unit,
                provenance=self.provenance,
                location=self.location,
                year_column=self.year_column,
                hazard_column=self.hazard_column,
            )
        )

    def _annual_description(self, component: _Hazard) -> dict[str, Any]:
        """A per-year source, as `component` reads it: the plan's own for its first."""
        source = self.source if component.source is None else component.source
        location = self.location if component.location is None else component.location
        kind = _source_kind(source)
        assert component.provenance is not None
        description: dict[str, Any] = {
            "columns": {
                "value": component.hazard_column,
                "year": component.year_column,
            },
            "kind": kind,
            "location": location.to_document(),
            "missing": self.missing,
            "provenance": component.provenance.model_dump(mode="json"),
            "role": "hazard",
            "type": type(source).__name__,
            "unit": component.unit,
            "years": None if self.years is None else list(self.years),
        }
        if kind == "file":
            description["path"] = os.fspath(source)
        elif kind == "table":
            description["rows"] = int(source.num_rows)
        return description

    def _hazard_licences(self) -> list[dict[str, Any]] | None:
        """The hazard sources' terms, each once; `None` on curves (read later)."""
        if self.provenance is None:
            return None
        stated = (
            [component.provenance for component in self._components()]
            if self.hazard_list
            else [self.provenance]
        )
        licences: list[dict[str, Any]] = []
        for provenance in stated:
            assert provenance is not None
            entry = {
                "attribution": provenance.attribution,
                "licence": provenance.licence,
                "role": "hazard",
                "source": provenance.dataset,
            }
            if entry not in licences:
                licences.append(entry)
        return licences

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
        if self._together:
            sample["dependence"] = self._dependence_document()
            if self.dependence_model is not None and self.draws is not None:
                notes.append(
                    self.dependence_model.statement(self.draws.n_draws, "assets")
                )
        if self.hazard_list:
            self._describe_components(sample, notes)
        return sample

    def _describe_components(self, sample: dict[str, Any], notes: list[str]) -> None:
        """A sample of several hazards, as far as it is known before execution.

        Its components (a curve's unit and tail are its dataset's, known only at
        execution), each one's loss model, the rule and the dependence between
        the hazards, in the keys the written `setup.samples` entries have.
        """
        components = self._components()
        sample["components"] = [
            {
                "name": each.name,
                "own_loss": each.loss_model is not None,
                "tail": each.tail,
                "unit": each.unit,
            }
            for each in components
        ]
        sample["loss_model"] = {
            "kind": "components",
            "components": [
                {"name": each.name, "loss_model": each.loss_model.to_document()}
                for each in components
                if each.loss_model is not None
            ],
        }
        sample["combine"] = _combine_record(
            self.combine_rule,
            [each.name for each in components if each.loss_model is not None],
        )
        model = self.dependence_model
        if not self._on_curves:
            sample["hazard_dependence"] = {"kind": "data"}
            return
        if model is None:
            sample["hazard_dependence"] = None
            return
        if self.draws is None:
            sample["hazard_dependence"] = model.to_document()
            return
        record, sentence = _model_record(
            model, self.draws.n_draws, [each.name for each in components], "hazards"
        )
        sample["hazard_dependence"] = record
        notes.append(sentence)

    def _rng_description(self) -> dict[str, Any]:
        generators = (
            "one child of numpy.random.SeedSequence(seed) per asset, in asset order"
        )
        if not self._on_curves:
            if self.hazard_list:
                generators += (
                    "; the basis-risk model draws for every component from it, in "
                    "component order"
                )
            return {
                "generators": generators,
                "seed": self.seed,
                "steps": ["basis_risk"],
            }
        rule = self.split_rule
        if self.hazard_list:
            generators += (
                "; it draws the first component, and each other component draws "
                "from a child it spawns, in component order"
            )
        return {
            "fresh_draws_seed": rule.seed if isinstance(rule, FreshDraws) else None,
            "generators": generators
            + "; fresh draws: one child of SeedSequence(fresh_draws_seed) per asset",
            "seed": self.seed,
            "steps": ["hazard", "basis_risk"],
        }

    def _by_asset(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """One sample description per asset, or one of all of them under `Joint()`."""
        ids = self._known_asset_ids()
        if ids is None:
            return [{**sample, "asset_ids": []}]
        if self._together:
            return [{**sample, "asset_ids": ids}]
        return [{**sample, "asset_ids": [asset_id]} for asset_id in ids]

    def _explain_document(self) -> dict[str, Any]:
        ids = self._known_asset_ids()
        samples = self._by_asset(self._sample_description())
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
        licences = self._hazard_licences()
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
        if self._together:
            model = self.dependence_model
            document["scopes"] = [scope.to_document() for scope in self.scope_list]
            document["dependence"] = None if model is None else model.to_document()
            document["allocation"] = self.allocation_rule
        if self.hazard_list:
            components = self._components()
            document["hazards"] = [
                component.to_document(own=i == 0, on_curves=self._on_curves)
                for i, component in enumerate(components)
            ]
            document["combine"] = samples[0]["combine"]
            document["hazard_dependence"] = samples[0]["hazard_dependence"]
            document["packages"] = [
                package.to_document() for package in self.package_list
            ]
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

    def _scenario_entry(
        self, planned: _Planned, variant: ParametricPlan | None
    ) -> dict[str, Any]:
        """One scenario's entry of `setup.scenarios`, as far as it is known."""
        members = (
            None
            if planned.members is None
            else [[pathway, year] for pathway, year in planned.members]
        )
        pathway: str | None = None
        horizon: int | None = None
        product: str | None = None
        hazard: dict[str, Any] | None = None
        if variant is None:
            names = self.source.selection.hazard_names
            hazard = {"name": str(names[0]), "from": "curve_row"}
            if self.hazard_list:
                hazard = {
                    component.name: {
                        "name": str(component.source.selection.hazard_names[0]),
                        "from": "curve_row",
                    }
                    for component in self._components()
                    if component.loss_model is not None
                }
            if planned.members is not None:
                pathway = planned.members[0][0]
                horizons = [year for _, year in planned.members]
                horizon = (horizons[0] + horizons[-1]) // 2
        elif variant.hazard_list:
            pathway, product, hazard = variant._component_reads()
        else:
            assert variant.provenance is not None
            product = variant.provenance.provider
            sources = variant._pool_sources() if variant.pool else (variant.source,)
            named = [_source_hazard(source) for source in sources]
            pathway = variant.stated_pathway or next(
                (each for each in map(_source_pathway, sources) if each), None
            )
            if any(named):
                hazard = {
                    "name": str(next(n for n in named if n)),
                    "from": "source_plan",
                }
            elif variant.hazard_name is not None:
                hazard = {"name": variant.hazard_name, "from": "stated"}
        if variant is not None:
            horizon = variant._stated_horizon()
        return {
            "blocks": [],
            "hazard": hazard,
            "horizon": horizon,
            "index_read": [],
            "members": members,
            "n_rows": None,
            "pathway": pathway,
            "product": product,
            "source_kind": "monte_carlo" if self._on_curves else "annual",
        }

    def _stated_horizon(self) -> int | None:
        """A per-year scenario's horizon, where `years=` states it; else `None`.

        A series' horizon is the centre of its whole window, `(first + last) //
        2`. A stated `years=` is that window when every year must hold a value;
        under `missing="drop_year"` a year may drop out at either end, so the
        window is known only at execution.
        """
        years = self.years
        if years is None or self.missing == "drop_year":
            return None
        return (years[0] + years[-1]) // 2

    def _component_reads(
        self,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """A per-year plan's pathway, product and hazards, read by component.

        The product joins the components' distinct providers, sorted, with `+`,
        and the hazards are those of the components with an impact, as a sample
        of several hazards records them.
        """
        components = self._components()
        sources = [
            self.source if component.source is None else component.source
            for component in components
        ]
        providers = sorted(
            {
                str(component.provenance.provider)
                for component in components
                if component.provenance is not None and component.provenance.provider
            }
        )
        pathway = self.stated_pathway or next(
            (each for each in map(_source_pathway, sources) if each), None
        )
        hazard: dict[str, Any] = {}
        for component, source in zip(components, sources, strict=True):
            if component.loss_model is None:
                continue
            named = _source_hazard(source)
            if named is not None:
                hazard[component.name] = {"name": named, "from": "source_plan"}
            elif component.hazard_name is not None:
                hazard[component.name] = {
                    "name": component.hazard_name,
                    "from": "stated",
                }
            else:
                hazard[component.name] = None
        return pathway, "+".join(providers) or None, hazard

    def _scenario_evaluations(self, labels: Sequence[str]) -> list[dict[str, Any]]:
        """The planned evaluations: every design scenario on every scenario's block."""
        stated = self.scenario_set
        assert stated is not None
        reference = stated.reference
        templates = any(cover.free_parameters() for cover in self.cover_list)
        given = any(not cover.free_parameters() for cover in self.cover_list)
        design_scenarios: list[str | None] = [
            *(self._design_labels(labels) if templates else []),
            *([None] if given else []),
        ]
        rows = self._planned_rows() if self._together else {}
        pricing = self._pricing().to_document()
        rule = self.split_rule
        evaluations: list[dict[str, Any]] = []
        for entry in self._evaluations():
            if entry["split"] != "out_of_sample":
                continue
            made: dict[str, Any] = {"kind": "as_built"}
            if isinstance(rule, Holdout):
                made = {"kind": "holdout", "train_fraction": rule.train_fraction}
            elif isinstance(rule, KFold):
                made = {"kind": "kfold", "k": rule.k, "fold": entry["fold"]}
            for label in labels:
                for design_scenario in design_scenarios:
                    designed_on = (
                        reference if design_scenario is None else design_scenario
                    )
                    own = designed_on == label
                    details = (
                        dict(entry["details"])
                        if own
                        else {"block": made, **self._seed_details()}
                    )
                    evaluations.append(
                        {
                            **rows,
                            "design_scenario": design_scenario,
                            "details": details,
                            "fold": entry["fold"],
                            "method": entry["method"] if own else "other_scenario",
                            "pricing": pricing,
                            "scenario": label,
                            "split": "out_of_sample",
                        }
                    )
        return evaluations

    def _design_labels(self, labels: Sequence[str]) -> list[str]:
        """The scenarios the templates are designed on, under `design_on`."""
        stated = self.scenario_set
        assert stated is not None
        return list(labels) if self.design_on == "each" else [stated.reference]

    def _explain_scenarios(self, document: dict[str, Any]) -> None:
        """Add the scenarios to `explain()`'s document: what each reads, and where."""
        stated = self.scenario_set
        assert stated is not None
        planned = self._planned_scenarios()
        labels = [each.label for each in planned]
        entries: dict[str, Any] = {}
        described: dict[str, Any] = {}
        samples: list[dict[str, Any]] = []
        own_sources = document["sources"]
        index = [entry for entry in own_sources if entry.get("role") == "index"]
        sources: list[dict[str, Any]] = []
        licences: list[dict[str, Any]] = []
        selection = [each.label for each in planned if each.kind == "selection"]
        for each in planned:
            variant = None if self._on_curves else self._scenario_plan(each)
            entry = self._scenario_entry(each, variant)
            entries[each.label] = entry
            own = variant if variant is not None else self
            sample = {
                **own._sample_description(),
                "scenario": each.label,
                **{key: entry[key] for key in ("horizon", "pathway", "product")},
                "members": entry["members"],
            }
            samples += self._by_asset(sample)
            description: dict[str, Any] = {
                "kind": each.kind,
                "members": entry["members"],
            }
            if each.kind == "dataset":
                assert isinstance(each.described, CurveScenario)
                read = self._curve_scenario_descriptions(each.described)
                sources += read
                paths = {str(part.get("component")): part["path"] for part in read}
                description |= {
                    "path": paths
                    if isinstance(each.described.dataset, Mapping)
                    else read[0]["path"]
                }
            elif variant is not None:
                read = variant._source_descriptions()
                sources += [{**part, "scenarios": [each.label]} for part in read]
                for licence in variant._hazard_licences() or ():
                    if licence not in licences:
                        licences.append(licence)
                description |= {
                    "hazard_name": (
                        {
                            component.name: component.hazard_name
                            for component in variant._components()
                        }
                        if variant.hazard_list
                        else variant.hazard_name
                    ),
                    "pathway": variant.stated_pathway,
                    "pool": None
                    if variant.pool is None
                    else variant.pool.to_document(),
                    "trend": variant.trend.to_document(),
                }
            described[each.label] = description
        if self._on_curves:
            dataset = [entry for entry in own_sources if entry.get("role") != "index"]
            sources = [{**entry, "scenarios": selection} for entry in dataset] + sources
        if document["licences"] is not None:
            licences += [
                entry for entry in document["licences"] if entry["role"] != "hazard"
            ]
            document["licences"] = licences
        document["sources"] = [*sources, *index]
        curves = list(labels) if self._on_curves else None
        document["setup"]["scenarios"] = {
            "design_on": self.design_on,
            "labels": labels,
            "mixed_products": stated.mixed_products,
            "reference": stated.reference,
            "scenarios": entries,
            "shared_draws": None if curves is None else {"labels": curves, "rng": {}},
        }
        document["setup"]["samples"] = samples
        document["setup"]["evaluations"] = self._scenario_evaluations(labels)
        document["scenarios"] = {
            "design_on": self.design_on,
            "labels": labels,
            "mixed_products": stated.mixed_products,
            "premiums": (
                None if self.scenario_premiums is None else dict(self.scenario_premiums)
            ),
            "reference": stated.reference,
            "scenarios": described,
        }

    def _curve_scenario_descriptions(
        self, entry: CurveScenario
    ) -> list[dict[str, Any]]:
        """A `CurveScenario`'s datasets, in the form of the plan's own source entries.

        One, or one per component, each reading that component's hazard.
        """
        datasets: dict[str | None, HazardDataset] = (
            dict(entry.dataset.items())
            if isinstance(entry.dataset, Mapping)
            else {None: entry.dataset}
        )
        requests: dict[str | None, Any] = {
            component.name: component.source for component in self._components()
        } or {None: self.source}
        described: list[dict[str, Any]] = []
        for component, dataset in datasets.items():
            request: PortfolioEvaluation = requests[component]
            own = self._source_description(request)
            names = request.selection.hazard_names
            described.append(
                {
                    **own,
                    **({} if component is None else {"component": component}),
                    "path": os.fspath(dataset.provider.source),
                    "scenarios": [entry.label],
                    "selection": {
                        "hazard_names": _listed(names),
                        "horizons": _listed(entry.horizons),
                        "pathways": None if entry.pathway is None else [entry.pathway],
                    },
                    "type": type(dataset).__name__,
                }
            )
        return described

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
        the layout of `EvaluationSetup`, so
        `setup_text(plan.explain(format="json"))` states the whole document
        before anything is read: the setup, the sources and the planned
        designs (`setup_text` of its `["setup"]` states the setup alone). The
        plan's seed is in every evaluation's `details` under `"plan_seed"`,
        and the fresh draws' under `"fresh_draws_seed"`. What only execution
        knows is in the written metadata: the years and the matched cells or
        curves, the checksums, the hashes of the samples and designs and, on
        fitted curves, the dataset's metadata (its units, semantics and
        licence, so `licences` is `None` here); with `cell=`, the cells are
        known only there too, and with `.index(...)` the bound covers.
        A plan that holds `Joint()` or `Clustered` adds `scopes` (a
        `Clustered`'s document holds its assignment and setting, not its
        values), `dependence` (`None` on per-year values, whose dependence is
        the data's own) and `allocation`;
        `setup.samples` then has one sample of all the assets, and each planned
        evaluation its `level`, `scope` and `allocation`. `format="json"` returns a
        dict, `"text"` its JSON text.

        With `.scenarios(...)`, the document adds `scenarios` (the reference,
        the labels, `design_on`, `mixed_products`, `premiums` and each
        scenario's kind and members), `setup.scenarios` in the form the
        written metadata has, one sample per scenario and asset, the planned
        evaluations of every design scenario on every scenario's block, and
        one entry of `sources` per scenario source, with the labels that read
        it.

        Raises `ValueError` for `Joint()` or `Clustered` on fitted curves
        without `.dependence(...)`, for a dependence or an allocation without
        either, for a cover written for one asset under either, and for
        `Clustered` beside `.index(...)`; `SampleError` for an assignment of
        `Clustered` that doesn't hold exactly the plan's assets, when they are
        in memory; `LeakageError` for `Joint()` or `Clustered` on fitted curves
        with `Holdout` or `KFold`
        under a dependence model other than `Independent()`, naming
        `FreshDraws`;
        for a request that selects several horizons or pathways without
        `.scenarios(...)`; with `.scenarios(...)`, without `.split(...)` or
        with `.allocation(...)`; `SampleError` for a draw count or assets that
        `EmpiricalCopula`'s reference does not fit; `ScenarioError` for a
        window whose members don't divide `n_draws` or `FreshDraws(n)`, and
        for a stated `hazard_name` a source plan contradicts; and
        `EvaluationError` for `premiums=` that name no cover given as is.

        With `.hazard(...)`, the document adds `hazards` (each component: its
        name, whether it is the plan's own, its impact and where it is read),
        `combine` (the rule's record), `hazard_dependence` (the data's own on
        per-year values, the stated model's record on fitted curves) and
        `packages`; `sources` has one entry per component, which names it, and
        each sample its `components`, `combine` and `hazard_dependence`. It
        raises as "Several hazards" in `.hazard(...)` says: `ValueError` for
        `Joint()`, `pool=` or `.index(...)` beside it, a per-year plan that
        doesn't name its own hazard, several hazards on curves without
        `.dependence(...)`, and packages that don't fit the covers;
        `SampleError` for a missing or needless rule, and a cover that doesn't
        fit the components it reads.
        """
        self._check_hazard_plan()
        self._check_scopes()
        self._check_scenario_plan()
        document = self._explain_document()
        if self.scenario_set is not None:
            self._explain_scenarios(document)
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

    def _open(self, read: _SeriesRead, *, named: bool | None = None) -> _OpenedRead:
        """Read a source once, for all the assets' series.

        On a plan with scenarios, or with a stated `hazard_name` (`named`, for
        a component of a plan with several hazards), the series' source records
        the hazard its source plan's `explain()` names, so that the sample's
        `hazard` record says the source names it.
        """
        table, record = self._read_source(read)
        source = self._series_source(read)
        explained = record.get("explain")
        stated = self.hazard_name is not None if named is None else named
        if (
            read.role == "hazard"
            and (self.names_hazard or stated)
            and isinstance(explained, Mapping)
        ):
            for key in ("recipe", "hazard_name"):
                value = explained.get(key)
                if isinstance(value, str) and value:
                    source[key] = value
        return _OpenedRead(read, table, record, source, _stated_pathway(explained))

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
            if key == "pathway" and not value and self.stated_pathway is not None:
                # An AnnualScenario's pathway fills the record where none states it.
                value = self.stated_pathway
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
        """One asset's series of an opened source, as read at its location.

        The pathway a source plan states is recorded beside the caller's source,
        and so is an `AnnualScenario`'s stated pathway where the source states
        none; one that the source plan contradicts raises `ScenarioError`.
        """
        series = self._located_series(opened, row, asset_id, portfolio)
        pathway = opened.pathway
        stated = self.stated_pathway if opened.read.role == "hazard" else None
        if stated is not None:
            if pathway is not None and not pathways_equivalent(pathway, stated):
                raise ScenarioError(
                    f"scenario {self.scenario!r}: pathway={stated!r} contradicts the "
                    f"source plan, which states {pathway!r}; a stated pathway must "
                    "agree with its source"
                )
            pathway = stated if pathway is None else pathway
        if pathway is None:
            return series
        # Beside the caller's source: the plan does not rewrite it.
        return AnnualSeries(
            series.years,
            series.values,
            series.unit,
            {**_thaw(series.provenance), "pathway": pathway},
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
            station=location.table_column,
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
        asset: _AnnualAsset | _HazardAsset,
        trend: Trend,
        rng: np.random.Generator | None,
    ) -> OutcomeSample:
        """One asset's per-year sample under `trend`, with its notes."""
        if isinstance(asset, _HazardAsset):
            return self._annual_hazard_sample(asset, trend, rng)
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
                hazard_name=self.hazard_name,
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
                hazard_name=self.hazard_name,
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

    def _annual_hazard_sample(
        self,
        asset: _HazardAsset,
        trend: Trend,
        rng: np.random.Generator | None,
    ) -> OutcomeSample:
        """One asset's per-year sample of several hazards, as the function builds it.

        `sample_annual_hazards`, with the components' tails and hazard names,
        the plan's rule, basis-risk model, `missing=` and scenario label.
        """
        components = self._components()
        names = {
            component.name: component.hazard_name
            for component in components
            if component.hazard_name is not None
        }
        tails: dict[str, Tail] = {}
        for component in components:
            assert component.tail is not None
            tails[component.name] = component.tail
        sample = _annual_hazard_rows(
            asset.series,
            tails=tails,
            loss_models=asset.loss_models,
            combine=self.combine_rule,
            basis_risk=self._require_basis_risk(),
            asset_id=asset.asset_id,
            rng=rng,
            missing=self.missing,
            trend=trend,
            scenario=self.scenario,
            hazard_names=names or None,
        )
        if asset.cell is not None:
            assert self.cell is not None
            sample = _noted(
                sample,
                f"The impact reads H3 cell {asset.cell:x} for asset "
                f"{asset.asset_id}: {_cell_origin(self.cell)}.",
            )
        return sample

    def _component_read(self, component: _Hazard) -> _SeriesRead:
        """Where each asset's series of a per-year component is read."""
        if component.source is None:
            return self._hazard_read
        assert component.unit is not None and component.provenance is not None
        return _SeriesRead(
            role="hazard",
            source=component.source,
            location=self.location
            if component.location is None
            else component.location,
            unit=component.unit,
            provenance=component.provenance,
            year_column=component.year_column,
            value_column=component.hazard_column,
            years=self.years,
        )

    def _annual_hazard_samples(self, *, folds: bool) -> _Samples:
        """Each asset's per-year sample of several hazards; per fold under a trend.

        Every source is read once for all the assets. Each asset's generator
        draws the basis-risk model for every component, in component order.
        """
        basis_risk = self._require_basis_risk()
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        # Before the sources are read, which may take long or download data.
        cells = [self._asset_cell(portfolio, asset_id, row) for asset_id, row in rows]
        components = self._components()
        opened = {
            component.name: self._open(
                self._component_read(component),
                named=None
                if component.source is None
                else component.hazard_name is not None,
            )
            for component in components
        }
        generators = self._generators(len(rows))
        per_fold = (
            folds
            and isinstance(self.trend, LinearTrend)
            and isinstance(self.split_rule, (Holdout, KFold))
        )
        samples: dict[str, OutcomeSample] = {}
        refitted: dict[str, tuple[OutcomeSample, ...]] = {}
        for (asset_id, row), rng, cell in zip(rows, generators, cells, strict=True):
            series = {
                name: self._series(each, row, asset_id, portfolio)
                for name, each in opened.items()
            }
            models = {
                component.name: (
                    component.loss_model.for_asset(portfolio, asset_id)
                    if cell is None
                    else self._cell_loss_model(
                        portfolio, asset_id, row, cell, model=component.loss_model
                    )
                )
                for component in components
                if component.loss_model is not None
            }
            asset = _HazardAsset(asset_id, series, models, cell)
            first_state = copy.deepcopy(rng)
            samples[asset_id] = self._annual_hazard_sample(asset, self.trend, rng)
            _warn_comonotone(basis_risk, separate_index=False)
            if per_fold:
                refitted[asset_id] = self._refitted_samples(
                    asset, samples[asset_id], first_state
                )
        first = components[0].name
        return _Samples(
            portfolio,
            samples,
            None,
            "annual_table",
            opened[first].record,
            folds=refitted if per_fold else None,
            component_records={name: each.record for name, each in opened.items()},
        )

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
        asset: _AnnualAsset | _HazardAsset,
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
        self._check_hazard_plan()
        self._check_scopes()
        self._check_scenario_plan()
        assert self.scenario_set is None, "a plan with scenarios builds each one's"
        if self._on_curves:
            return self._curve_samples(fresh=fresh)
        if self.hazard_list:
            return self._annual_hazard_samples(folds=folds)
        basis_risk = self._require_basis_risk()
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        if self._together:
            self._check_together(portfolio, [asset_id for asset_id, _ in rows])
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
        joint = OutcomeSample.join(samples.values()) if self._together else None
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
        model: LossModel | None = None,
    ) -> LossModel:
        """The loss model for one asset, with its H3 cell in the impact's context.

        `model` is a component's, by default the plan's.
        """
        model = self.loss_model if model is None else model
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
        if self.hazard_list:
            return self._curve_hazard_samples(fresh=fresh)
        self._require_basis_risk()
        self._require_draws()
        request: PortfolioEvaluation = self.source
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        if self._together:
            self._check_together(portfolio, [asset_id for asset_id, _ in rows])
        metadata = request.dataset.metadata()
        self._check_cover_units(metadata)
        path = Path(request.dataset.provider.source)
        file_sha256 = _file_sha256(path)
        curves = dataclasses.replace(request, portfolio=portfolio).asset_curves()
        matched = self._curve_rows(
            curves, portfolio.id_column, [asset_id for asset_id, _ in rows]
        )
        record = {
            "curves": int(curves.num_rows),
            "file_sha256": file_sha256,
            "kind": "canonical_dataset",
            "path": os.fspath(path),
        }
        read = _CurveRead(
            metadata, record, {asset_id: [row] for asset_id, row in matched.items()}
        )
        return self._drawn_curves(portfolio, rows, read, fresh=fresh, label=None)

    def _check_cover_units(self, metadata: HazardDatasetMetadata) -> None:
        """Raise `SampleError` for a cover on an index in another unit than curves'."""
        for cover in self.cover_list:
            if cover.index.unit != metadata.value_unit:
                raise SampleError(
                    f"cover {cover.label or cover.definition_hash!r} pays on an "
                    f"index in {cover.index.unit!r}, but the dataset's values are "
                    f"in {metadata.value_unit!r}"
                )

    def _drawn_curves(
        self,
        portfolio: AssetPortfolio,
        rows: Sequence[tuple[str, dict[str, Any]]],
        read: _CurveRead,
        *,
        fresh: bool,
        label: str | None,
    ) -> _Samples:
        """Each asset's draws from its curves in `read`, under `label`.

        An asset's curves are one canonical row, or several, a window of
        horizons, which `sample_curve` and `sample_curves` mix on one stream of
        uniforms. `label=None` leaves the sample's `scenario` at its default.
        """
        basis_risk = self._require_basis_risk()
        draws = self._require_draws()
        metadata, file_sha256 = read.metadata, read.record["file_sha256"]
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
        first = {asset_id: read.rows[asset_id][0] for asset_id in asset_ids}
        loss_models = {
            asset_id: self._cell_loss_model(
                portfolio, asset_id, row, first[asset_id].get("cell_index")
            )
            for asset_id, row in rows
        }
        canonical: dict[str, Any] = {}
        for asset_id in asset_ids:
            members = [
                {name: curve[name] for name in _CURVE_ROW_COLUMNS if name in curve}
                for curve in read.rows[asset_id]
            ]
            canonical[asset_id] = members[0] if len(members) == 1 else members
        named: dict[str, Any] = {} if label is None else {"scenario": label}

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
                        asset_id: first[asset_id]["spatial_match"]
                        for asset_id in asset_ids
                    },
                },
                **named,
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
                    "spatial_match": first[asset_id]["spatial_match"],
                },
                **named,
            )

        samples: dict[str, OutcomeSample]
        fresh_samples: dict[str, OutcomeSample] = {}
        joint: OutcomeSample | None = None
        fresh_joint: OutcomeSample | None = None
        if self._together:
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
        return _Samples(
            portfolio,
            samples,
            fresh_samples if isinstance(rule, FreshDraws) else None,
            "canonical_file",
            read.record,
            joint=joint,
            fresh_joint=fresh_joint,
        )

    # -- several hazards on curves -------------------------------------------------

    def _component_curves(
        self,
        component: _Hazard,
        portfolio: AssetPortfolio,
        asset_ids: Sequence[str],
        *,
        one: bool,
    ) -> _CurveRead:
        """A component's matched curves: one per asset, or every selected pair."""
        request: PortfolioEvaluation = dataclasses.replace(
            component.source, portfolio=portfolio
        )
        metadata = request.dataset.metadata()
        self._check_component_units(component.name, metadata)
        if not one:
            return self._read_curves(request, asset_ids, check=False)
        path = Path(request.dataset.provider.source)
        file_sha256 = _file_sha256(path)
        curves = request.asset_curves()
        matched = self._curve_rows(curves, portfolio.id_column, asset_ids)
        record = {
            "curves": int(curves.num_rows),
            "file_sha256": file_sha256,
            "kind": "canonical_dataset",
            "path": os.fspath(path),
        }
        return _CurveRead(
            metadata, record, {asset_id: [row] for asset_id, row in matched.items()}
        )

    def _check_component_units(
        self, name: str, metadata: HazardDatasetMetadata
    ) -> None:
        """Raise `SampleError` for a cover on component `name` in another unit.

        Or with another tail: a component's tail is its dataset's
        `return_period_tail`, so a package's contracts are checked here too,
        before anything is designed.
        """
        for label, component, index, packaged in self._cover_reads():
            if component != name:
                continue
            if index.unit != metadata.value_unit:
                raise SampleError(
                    f"cover {label!r} pays on an index in {index.unit!r}, but the "
                    f"dataset of component {name!r} holds values in "
                    f"{metadata.value_unit!r}"
                )
            if index.tail != metadata.return_period_tail:
                raise SampleError(
                    f"{_contract(label, packaged)} has the tail {index.tail!r} on "
                    f"component {name!r}, whose dataset's return_period_tail is "
                    f"{metadata.return_period_tail!r}"
                )

    def _cover_reads(self) -> list[tuple[str, str, IndexDefinition, bool]]:
        """What each cover reads: its label, the component, its index, if packaged."""
        packaged = {
            position: component
            for found in self._package_members()
            for component, position in found.items()
        }
        reads: list[tuple[str, str, IndexDefinition, bool]] = []
        for position, cover in enumerate(self.cover_list):
            label = cover.label or cover.definition_hash
            definition = cover.index_definition
            if isinstance(definition, CompositeIndex):
                reads += [
                    (label, name, index, False)
                    for name, index in definition.components.items()
                ]
                continue
            component = packaged.get(position, definition.hazard_name)
            reads.append((label, component, definition, position in packaged))
        return reads

    def _curve_hazard_samples(self, *, fresh: bool) -> _Samples:
        """Each asset's draws of several hazards, under the stated dependence."""
        self._require_basis_risk()
        self._require_draws()
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        asset_ids = [asset_id for asset_id, _ in rows]
        reads = {
            component.name: self._component_curves(
                component, portfolio, asset_ids, one=True
            )
            for component in self._components()
        }
        return self._drawn_hazards(portfolio, rows, reads, fresh=fresh, label=None)

    def _component_generators(
        self, seed: int, count: int, names: Sequence[str]
    ) -> list[dict[str, np.random.Generator]]:
        """Each asset's generators, by component.

        The asset's child of `SeedSequence(seed)` draws the first component, as
        it draws the hazard of a plan with one, and each other component draws
        from a child it spawns, in component order.
        """
        made: list[dict[str, np.random.Generator]] = []
        for child in np.random.SeedSequence(seed).spawn(count):
            generators = {names[0]: np.random.default_rng(child)}
            for name, spawned in zip(
                names[1:], child.spawn(len(names) - 1), strict=True
            ):
                generators[name] = np.random.default_rng(spawned)
            made.append(generators)
        return made

    def _drawn_hazards(
        self,
        portfolio: AssetPortfolio,
        rows: Sequence[tuple[str, dict[str, Any]]],
        reads: Mapping[str, _CurveRead],
        *,
        fresh: bool,
        label: str | None,
    ) -> _Samples:
        """Each asset's draws of several hazards from their curves in `reads`.

        `sample_curve_hazards`, with each asset's generators by component, the
        plan's dependence model, rule and basis-risk model. A component's curves
        are one canonical row, or a window of horizons. `label=None` leaves the
        sample's `scenario` at its default.
        """
        basis_risk = self._require_basis_risk()
        draws = self._require_draws()
        model = self.dependence_model
        assert model is not None and self.seed is not None
        components = self._components()
        names = [component.name for component in components]
        rule = self.split_rule if fresh else None
        generators = self._component_generators(self.seed, len(rows), names)
        fresh_generators = (
            None
            if not isinstance(rule, FreshDraws)
            else self._component_generators(rule.seed, len(rows), names)
        )
        named: dict[str, Any] = {} if label is None else {"scenario": label}

        def draw(
            asset_id: str,
            row: Mapping[str, Any],
            count: int,
            drawn: Mapping[str, np.random.Generator],
        ) -> OutcomeSample:
            first = {name: reads[name].rows[asset_id][0] for name in names}
            curves: dict[str, Any] = {}
            for name in names:
                members = [
                    {key: curve[key] for key in _CURVE_ROW_COLUMNS if key in curve}
                    for curve in reads[name].rows[asset_id]
                ]
                curves[name] = members[0] if len(members) == 1 else members
            loss_models = {
                component.name: self._cell_loss_model(
                    portfolio,
                    asset_id,
                    row,
                    first[component.name].get("cell_index"),
                    model=component.loss_model,
                )
                for component in components
                if component.loss_model is not None
            }
            return sample_curve_hazards(
                curves,
                dependence=model,
                metadata={name: reads[name].metadata for name in names},
                n_draws=count,
                rngs=drawn,
                loss_models=loss_models,
                combine=self.combine_rule,
                basis_risk=basis_risk,
                asset_id=asset_id,
                assume_semantics=draws.assume_semantics,
                outside_support=draws.outside_support,
                # One document for every component, so each is named by its own.
                details={
                    "file_sha256_by_component": {
                        name: reads[name].record["file_sha256"] for name in names
                    },
                    "spatial_match_by_component": {
                        name: first[name]["spatial_match"] for name in names
                    },
                },
                **named,
            )

        samples: dict[str, OutcomeSample] = {}
        fresh_samples: dict[str, OutcomeSample] = {}
        for i, (asset_id, row) in enumerate(rows):
            samples[asset_id] = draw(asset_id, row, draws.n_draws, generators[i])
            if fresh_generators is not None:
                assert isinstance(rule, FreshDraws)
                fresh_samples[asset_id] = draw(
                    asset_id, row, rule.n_draws, fresh_generators[i]
                )
        return _Samples(
            portfolio,
            samples,
            fresh_samples if fresh_generators is not None else None,
            "canonical_file",
            reads[names[0]].record,
            component_records={name: reads[name].record for name in names},
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

    def _check_together(
        self, portfolio: AssetPortfolio, asset_ids: Sequence[str]
    ) -> None:
        """Check the assets of `Joint()` or `Clustered`, before any source is read.

        Both need several assets. `Clustered`'s assignment must hold exactly the
        plan's assets, and its values must be those the plan's loss model reads,
        so that each cluster's weights are the plan's values, as on the function
        route with `values=` those values.
        """
        _require_several_assets(asset_ids, self._together_name())
        clustered = self._clustered
        if clustered is None:
            return
        _check_assigned(clustered, asset_ids)
        read = self._asset_values(portfolio, asset_ids)
        differ = [
            asset for asset in asset_ids if clustered.values[asset] != read[asset]
        ]
        if differ:
            column = self.loss_model.value_column
            where = (
                "1 for each asset, as the plan's losses are damage ratios"
                if column is None
                else f"the asset column {column!r}"
            )
            raise ValueError(
                "Clustered weights each cluster's index by the assets' values, and "
                f"on a plan those are the values its loss model reads: {where}. "
                f"Its values= differ from them for the assets {differ}: state "
                "those values in Clustered(..., values=...)"
            )

    @staticmethod
    def _evaluated(block: _Block) -> list[tuple[Split, OutcomeSample]]:
        """Where a block's rows are read: its design block, and any held-out one."""
        _, train, test = block
        evaluated: list[tuple[Split, OutcomeSample]] = [("in_sample", train)]
        if test is not None:
            evaluated.append(("out_of_sample", test))
        return evaluated

    def _block_tables(self, entries: Sequence[Any], block: _Block) -> list[Any]:
        """The rows of one block: on the design block, and on any held-out one."""
        fold, train, _ = block
        return [
            compare(
                entries,
                sample,
                split=split,
                fold=fold,
                pricing=self._pricing(),
                options=self.metric_options,
                design_sample=train,
            )
            for split, sample in self._evaluated(block)
        ]

    def _breakdown_tables(self, entries: Sequence[Any], block: _Block) -> list[Any]:
        """The parts of one block's payouts: on the design block, and any held-out."""
        fold, train, _ = block
        return [
            payout_breakdown(
                entries,
                sample,
                split=split,
                fold=fold,
                options=self.metric_options,
                design_sample=train,
            )
            for split, sample in self._evaluated(block)
        ]

    def _allocated_tables(
        self,
        designs: Sequence[JointDesignResult | ClusteredDesignResult],
        block: _Block,
        values: Mapping[str, float] | None,
    ) -> list[Any]:
        """The allocated rows of one block's joint designs, in tables of their own."""
        rule = self.allocation_rule
        if rule is None or values is None:
            return []
        fold = block[0]
        return [
            allocate(
                designs,
                sample,
                split=split,
                fold=fold,
                allocation=rule,
                values=values,
                options=self.metric_options,
            )
            for split, sample in self._evaluated(block)
        ]

    def _individual_design(
        self,
        cover: Cover,
        asset_id: str,
        train: OutcomeSample,
        objective: Objective,
        label: str,
        budget: Budget | None | Literal["stated"] = "stated",
    ) -> DesignResult:
        """One template's design on one asset's design block, labelled `label`.

        `budget` is a package contract's share; `"stated"` takes the plan's.
        """
        spec = self.design_spec
        assert spec is not None
        result = design(
            cover,
            train,
            objective,
            pricing=self._pricing(),
            budget=spec.budget if budget == "stated" else budget,
            limit=spec.limit,
            options=spec.options,
        )
        return dataclasses.replace(result, cover=_bound(result.cover, asset_id, label))

    def _design_block(self, cover: Cover, train: OutcomeSample) -> OutcomeSample:
        """What a template is designed on: the block, or its hazard's own loss.

        On a plan with several hazards a cover on one index is designed on the
        component named its `hazard_name`, with that hazard's own loss; a cover
        on a `CompositeIndex` on the combined loss.
        """
        definition = cover.index_definition
        if not self.hazard_list or isinstance(definition, CompositeIndex):
            return train
        return train.component(definition.hazard_name)

    def _asset_entries(
        self, asset_id: str, bound: Sequence[Cover], train: OutcomeSample
    ) -> tuple[list[list[Any]], list[DesignResult]]:
        """One asset's entries on one design block, slot by slot, and the designs.

        A slot is a cover of `.covers(...)` that applies to the asset, or a
        package, at the place of its first contract; the contracts of a package
        have no slot of their own. A template's slot holds one design per
        objective, and a package of templates one package per objective.
        `bound` holds the asset's covers, in the order of `.covers(...)`.
        """
        positions = [
            position
            for position, cover in enumerate(self.cover_list)
            if cover.asset_id in (None, asset_id)
        ]
        own = dict(zip(positions, bound, strict=True))
        members = self._package_members() if self.package_list else []
        first_of: dict[int, int] = {}
        packaged: set[int] = set()
        for number, found in enumerate(members):
            # A package's covers are written for one asset, or for none.
            if all(position in own for position in found.values()):
                first_of[min(found.values())] = number
                packaged.update(found.values())
        slots: list[list[Any]] = []
        designs: list[DesignResult] = []
        for position in positions:
            if position in first_of:
                number = first_of[position]
                contracts = {
                    component: own[held] for component, held in members[number].items()
                }
                slots.append(
                    self._package_entries(
                        self.package_list[number], contracts, asset_id, train, designs
                    )
                )
            elif position not in packaged:
                slots.append(
                    self._cover_entries(own[position], asset_id, train, designs)
                )
        return slots, designs

    def _cover_entries(
        self,
        cover: Cover,
        asset_id: str,
        train: OutcomeSample,
        designs: list[DesignResult],
    ) -> list[Any]:
        """A cover's entries on one design block: itself, or a design per objective."""
        if not cover.free_parameters():
            return [cover]
        spec = self.design_spec
        assert spec is not None
        block = self._design_block(cover, train)
        made: list[Any] = []
        for objective in spec.objectives:
            result = self._individual_design(
                cover, asset_id, block, objective, _design_label(cover, objective)
            )
            designs.append(result)
            made.append(result)
        return made

    def _package_entries(
        self,
        package: _Package,
        contracts: Mapping[str, Cover],
        asset_id: str,
        train: OutcomeSample,
        designs: list[DesignResult],
    ) -> list[Any]:
        """A package's entries on one design block: its covers, or its designs.

        Each template is designed on its component's own loss, under its share
        of the plan's budget (`split_budget` on `train`), one package per
        objective.
        """
        if not any(cover.free_parameters() for cover in contracts.values()):
            return [CoverPackage(dict(contracts), label=package.label)]
        spec = self.design_spec
        assert spec is not None
        budgets = (
            None
            if spec.budget is None
            else split_budget(spec.budget, train, shares=package.split_shares())  # type: ignore[arg-type]
        )
        made: list[Any] = []
        for objective in spec.objectives:
            designed: dict[str, DesignResult] = {}
            for component, cover in contracts.items():
                result = self._individual_design(
                    cover,
                    asset_id,
                    train.component(component),
                    objective,
                    _design_label(cover, objective),
                    budget=None if budgets is None else budgets[component],
                )
                designs.append(result)
                designed[component] = result
            made.append(
                CoverPackage(
                    designed, label=f"{package.label} ({_objective_tag(objective)})"
                )
            )
        return made

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

    def _cluster_design(
        self,
        template: Cover,
        scope: Clustered,
        train: OutcomeSample,
        objective: Objective,
        label: str,
    ) -> ClusteredDesignResult:
        """One template's design under `Clustered`, once per cluster, labelled `label`.

        `_check_together` checked that the scope's values are the plan's.
        """
        spec = self.design_spec
        assert spec is not None
        result = design_joint(
            [template] * len(scope.cluster_ids),
            train,
            objective,
            pricing=self._pricing(),
            budget=spec.budget,
            limit=spec.limit,
            options=spec.options,
            scope=scope,
        )
        return dataclasses.replace(
            result,
            covers={
                cluster: _relabelled(cover, label)
                for cluster, cover in result.covers.items()
            },
        )

    def _joint_entries(
        self, train: OutcomeSample, covers: Mapping[str, Sequence[Cover]]
    ) -> tuple[list[Any], list[Any]]:
        """The entries to compare on one block of the joint sample, and the designs.

        A resolved cover is one contract per asset. A template is designed under
        each objective on the owner's total under `Joint()`, once per cluster
        under `Clustered`, and, if the plan holds `Individual()` too, on each
        asset's own view of the block.
        """
        positions, designs = self._joint_positions(train, covers)
        return [entry for entries in positions for entry in entries], designs

    def _joint_positions(
        self, train: OutcomeSample, covers: Mapping[str, Sequence[Cover]]
    ) -> tuple[list[list[Any]], list[Any]]:
        """`_joint_entries`, each cover's entries apart, in the order of the covers."""
        assets = train.asset_ids
        positions: list[list[Any]] = []
        designs: list[Any] = []
        for position, stated in enumerate(self.cover_list):
            own = {asset_id: covers[asset_id][position] for asset_id in assets}
            entries: list[Any] = []
            positions.append(entries)
            if not stated.free_parameters():
                entries.append(own)
                continue
            assert self.design_spec is not None
            clustered = self._clustered
            for objective in self.design_spec.objectives:
                label = _design_label(stated, objective)
                if self._holds_joint:
                    joint = self._joint_design(own, train, objective, label)
                    designs.append(joint)
                    entries.append(joint)
                if clustered is not None:
                    by_cluster = self._cluster_design(
                        stated, clustered, train, objective, label
                    )
                    designs.append(by_cluster)
                    entries.append(by_cluster)
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
        return positions, designs

    def _run(self, *, evaluate: bool, breakdown: bool = False) -> _Run:
        """Build, design and evaluate; with `breakdown`, the payouts' parts instead.

        `breakdown` is for a plan of one scope, `Individual()`.
        """
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
        if self.scenario_set is not None:
            return self._scenario_run(evaluate=evaluate, breakdown=breakdown)
        built = self._build_samples(fresh=evaluate, folds=True)
        portfolio, samples = built.portfolio, built.samples
        asset_ids = list(samples)
        covers = self._covers_for(asset_ids, built.locations)
        designs: list[Any] = []
        tables: list[Any] = []
        allocated: list[Any] = []
        if built.joint is not None:
            values = (
                None
                if self.allocation_rule is None
                else self._asset_values(portfolio, asset_ids)
            )
            for block in self._joint_blocks(built):
                entries, made = self._joint_entries(block[1], covers)
                designs += made
                if not evaluate:
                    continue
                joint = [
                    each
                    for each in made
                    if isinstance(each, (JointDesignResult, ClusteredDesignResult))
                ]
                if values is not None and not joint:
                    raise EvaluationError(
                        ".allocation(...) shares out a joint design's total payout, "
                        "but no entry is a joint design: the plan's covers are all "
                        f"resolved, and {self._together_name()} evaluates each as a "
                        "set of contracts"
                    )
                tables += self._block_tables(entries, block)
                allocated += self._allocated_tables(joint, block, values)
        else:
            for asset_id in asset_ids:
                for block in self._asset_blocks(built, asset_id):
                    slots, made = self._asset_entries(
                        asset_id, covers[asset_id], block[1]
                    )
                    designs += made
                    if evaluate:
                        items = [entry for slot in slots for entry in slot]
                        tables += (
                            self._breakdown_tables(items, block)
                            if breakdown
                            else self._block_tables(items, block)
                        )
        return _Run(
            samples=MappingProxyType(samples),
            fresh=None if built.fresh is None else MappingProxyType(built.fresh),
            designs=tuple(designs),
            tables=tuple(tables),
            allocated=tuple(allocated),
            source_key=built.record_key,
            source=built.record,
            id_column=portfolio.id_column,
            values_sha256=self._values_sha256(portfolio, asset_ids),
            index_source=built.index_record,
            joint=built.joint,
            fresh_joint=built.fresh_joint,
            run_records=built.run_records,
            component_records=built.component_records,
        )

    # -- scenarios ----------------------------------------------------------------

    def _read_curves(
        self,
        request: PortfolioEvaluation,
        asset_ids: Sequence[str],
        *,
        check: bool = True,
    ) -> _CurveRead:
        """One canonical dataset's matched curves, each asset's in its own list.

        `check=False` leaves the covers' units to the caller, a component's.
        """
        metadata = request.dataset.metadata()
        if check:
            self._check_cover_units(metadata)
        path = Path(request.dataset.provider.source)
        file_sha256 = _file_sha256(path)
        curves = request.asset_curves()
        id_column = request.portfolio.id_column
        held: dict[str, list[dict[str, Any]]] = {asset_id: [] for asset_id in asset_ids}
        ids = _string_values(curves.column(id_column), id_column)
        for asset_id, row in zip(ids, curves.to_pylist(), strict=True):
            if asset_id in held:
                held[asset_id].append(row)
        record = {
            "curves": int(curves.num_rows),
            "file_sha256": file_sha256,
            "kind": "canonical_dataset",
            "path": os.fspath(path),
        }
        return _CurveRead(metadata, record, held)

    def _dataset_request(
        self,
        entry: CurveScenario,
        portfolio: AssetPortfolio,
        component: _Hazard | None = None,
    ) -> PortfolioEvaluation:
        """The request a `CurveScenario` reads: the plan's, on its own dataset.

        For a component of a plan with several hazards, that component's request
        on the dataset `entry` maps it to.
        """
        request: PortfolioEvaluation = (
            self.source if component is None else component.source
        )
        stated = entry.dataset
        if isinstance(stated, Mapping):
            assert component is not None
            dataset = stated[component.name]
        else:
            dataset = stated
        return dataclasses.replace(
            request,
            dataset=dataset,
            portfolio=portfolio,
            selection=HazardSelection(
                hazard_names=request.selection.hazard_names,
                horizons=entry.horizons,
                pathways=None if entry.pathway is None else (entry.pathway,),
            ),
        )

    def _dataset_members(
        self, entry: CurveScenario, read: _CurveRead, request: PortfolioEvaluation
    ) -> _Pairs:
        """The pairs a `CurveScenario` reads: stated, or the one its dataset holds.

        Raises `ScenarioError` for a dataset with no curve of the plan's hazard,
        naming the hazards it holds, and where it holds several pathways or
        horizons that `pathway=` and `horizon=` don't choose between.
        """
        names = request.selection.hazard_names
        assert names is not None
        place = request.dataset.provider.source
        found = list(
            dict.fromkeys(
                (str(row["pathway"]), int(row["horizon"]))
                for rows in read.rows.values()
                for row in rows
            )
        )
        horizons = entry.horizons
        stated = (
            None
            if entry.pathway is None or horizons is None
            else tuple((entry.pathway, horizon) for horizon in horizons)
        )
        if not found:
            every = dataclasses.replace(request, selection=HazardSelection())
            hazards = sorted(
                {str(row["hazard_name"]) for row in every.asset_curves().to_pylist()}
            )
            if names[0] not in hazards:
                raise ScenarioError(
                    f"scenario {entry.label!r} reads the plan's hazard {names[0]!r}, "
                    f"but its dataset {place} holds no curve "
                    f"of it for the plan's assets; it holds the hazards {hazards}"
                )
            if stated is None:
                raise ScenarioError(
                    f"scenario {entry.label!r}: its dataset holds no curve of "
                    f"{names[0]!r} for the plan's assets at the stated pathway or "
                    "horizon"
                )
        if stated is not None:
            # A pair the dataset doesn't hold is the pairs check's to name.
            return stated
        pathways = list(dict.fromkeys(pathway for pathway, _ in found))
        if len(pathways) > 1:
            raise ScenarioError(
                f"scenario {entry.label!r} reads a dataset that holds the pathways "
                f"{pathways} for the plan's hazard; state pathway="
            )
        if horizons is not None:
            return tuple((pathways[0], horizon) for horizon in horizons)
        years = sorted({horizon for _, horizon in found})
        if len(years) > 1:
            raise ScenarioError(
                f"scenario {entry.label!r} reads a dataset that holds the horizons "
                f"{years} for the plan's hazard; state horizon=, one year or "
                "several for a window"
            )
        return ((pathways[0], years[0]),)

    def _curve_scenarios(self, *, fresh: bool) -> dict[str, _Samples]:
        """Every curve scenario's samples, the reference first, on one set of draws.

        Each scenario's draws come from the asset's same child generators, so the
        curve scenarios share their uniforms row by row.
        """
        if self.hazard_list:
            return self._curve_hazard_scenarios(fresh=fresh)
        self._require_basis_risk()
        self._require_draws()
        request: PortfolioEvaluation = self.source
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        asset_ids = [asset_id for asset_id, _ in rows]
        if self._together:
            self._check_together(portfolio, asset_ids)
        own = self._read_curves(
            dataclasses.replace(request, portfolio=portfolio), asset_ids
        )
        built: dict[str, _Samples] = {}
        for each in self._planned_scenarios():
            read, members = own, each.members
            if isinstance(each.described, CurveScenario):
                asked = self._dataset_request(each.described, portfolio)
                read = self._read_curves(asked, asset_ids)
                members = self._dataset_members(each.described, read, asked)
            assert members is not None
            pairs = {
                asset_id: [
                    (row["pathway"], row["horizon"]) for row in read.rows[asset_id]
                ]
                for asset_id in asset_ids
            }
            _check_pairs_held(
                each.label,
                members,
                {f"asset {asset_id!r}": pairs[asset_id] for asset_id in asset_ids},
            )
            chosen = {
                asset_id: [
                    next(
                        row
                        for row, held in zip(
                            read.rows[asset_id], pairs[asset_id], strict=True
                        )
                        if _same_pair(member, held)
                    )
                    for member in members
                ]
                for asset_id in asset_ids
            }
            built[each.label] = self._drawn_curves(
                portfolio,
                rows,
                _CurveRead(read.metadata, read.record, chosen),
                fresh=fresh,
                label=each.label,
            )
        return built

    def _curve_hazard_scenarios(self, *, fresh: bool) -> dict[str, _Samples]:
        """Every curve scenario's draws of several hazards, the reference first.

        Each component reads every scenario's (pathway, horizon) pairs from its
        own dataset, the selection's from its request's and a `CurveScenario`'s
        from the dataset it maps the component to. The draws come from each
        asset's same generators, so the scenarios share their uniforms.
        """
        self._require_basis_risk()
        self._require_draws()
        portfolio = self._portfolio()
        rows = self._asset_rows(portfolio)
        asset_ids = [asset_id for asset_id, _ in rows]
        components = self._components()
        own = {
            component.name: self._component_curves(
                component, portfolio, asset_ids, one=False
            )
            for component in components
        }
        built: dict[str, _Samples] = {}
        for each in self._planned_scenarios():
            reads, members = own, each.members
            if isinstance(each.described, CurveScenario):
                reads = {}
                for component in components:
                    asked = self._dataset_request(each.described, portfolio, component)
                    read = self._read_curves(asked, asset_ids, check=False)
                    self._check_component_units(component.name, read.metadata)
                    reads[component.name] = read
                    if members is None:
                        members = self._dataset_members(each.described, read, asked)
            assert members is not None
            chosen: dict[str, _CurveRead] = {}
            for component in components:
                read = reads[component.name]
                pairs = {
                    asset_id: [
                        (row["pathway"], row["horizon"]) for row in read.rows[asset_id]
                    ]
                    for asset_id in asset_ids
                }
                _check_pairs_held(
                    each.label,
                    members,
                    {
                        f"component {component.name!r} of asset {asset_id!r}": pairs[
                            asset_id
                        ]
                        for asset_id in asset_ids
                    },
                )
                chosen[component.name] = _CurveRead(
                    read.metadata,
                    read.record,
                    {
                        asset_id: [
                            next(
                                row
                                for row, held in zip(
                                    read.rows[asset_id], pairs[asset_id], strict=True
                                )
                                if _same_pair(member, held)
                            )
                            for member in members
                        ]
                        for asset_id in asset_ids
                    },
                )
            built[each.label] = self._drawn_hazards(
                portfolio, rows, chosen, fresh=fresh, label=each.label
            )
        return built

    def _built_scenarios(self, *, fresh: bool, folds: bool) -> dict[str, _Samples]:
        """Every scenario's samples, the reference first, after the set's refusals.

        The refusals of a `ScenarioSet` run on each asset's samples, or on the
        samples of all the assets under `Joint()` or `Clustered`, through the
        function the set runs at construction. On per-year values one asset at
        a time, the assets must read each scenario at one horizon, which the
        merge of their tables requires (`_check_scenario_horizons`), before
        anything is designed.
        """
        self._check_hazard_plan()
        self._check_scopes()
        self._check_scenario_plan()
        stated = self.scenario_set
        assert stated is not None
        if self._on_curves:
            built = self._curve_scenarios(fresh=fresh)
        else:
            built = {
                each.label: self._scenario_plan(each)._build_samples(
                    fresh=fresh, folds=folds
                )
                for each in self._planned_scenarios()
            }
        groups: list[list[OutcomeSample]]
        if self._together:
            groups = [[]]
            for each in built.values():
                assert each.joint is not None
                groups[0].append(each.joint)
        else:
            reference = built[stated.reference]
            groups = [
                [each.samples[asset_id] for each in built.values()]
                for asset_id in reference.samples
            ]
        for group in groups:
            _check_scenarios(
                group, reference=stated.reference, mixed_products=stated.mixed_products
            )
        if not self._on_curves and not self._together:
            _check_scenario_horizons(built)
        return built

    def _scenario_blocks(self, blocks: Mapping[str, _Block]) -> ScenarioSet:
        """One fold's evaluation blocks of every scenario, as a `ScenarioSet`."""
        stated = self.scenario_set
        assert stated is not None
        evaluated: list[OutcomeSample] = []
        for block in blocks.values():
            assert block[2] is not None
            evaluated.append(block[2])
        return ScenarioSet(
            evaluated, reference=stated.reference, mixed_products=stated.mixed_products
        )

    def _scenario_table(
        self,
        entries: Sequence[Any],
        blocks: Mapping[str, _Block],
        given: Sequence[Cover],
        used: set[str],
        *,
        breakdown: bool = False,
        scenarios: ScenarioSet | None = None,
    ) -> Any:
        """One fold's rows, for one asset or for all of them: `compare_scenarios`.

        With `breakdown`, the parts of each entry's payout instead, as
        `payout_breakdown_scenarios` gives them, which hold no premium.
        `scenarios` is the fold's evaluation blocks, if already built.
        """
        stated = self.scenario_set
        assert stated is not None
        reference = blocks[stated.reference]
        labels = {_cover_label(cover) for cover in given}
        premiums = {
            key: value for key, value in self.scenario_premiums or () if key in labels
        }
        used.update(premiums)
        if scenarios is None:
            scenarios = self._scenario_blocks(blocks)
        if breakdown:
            return payout_breakdown_scenarios(
                entries,
                scenarios,
                fold=reference[0],
                options=self.metric_options,
                design_sample=reference[1] if given else None,
            )
        return compare_scenarios(
            entries,
            scenarios,
            fold=reference[0],
            pricing=self._pricing(),
            premiums=premiums or None,
            options=self.metric_options,
            design_sample=reference[1] if given else None,
        )

    def _allocated_scenarios(
        self,
        entries: Sequence[Any],
        scenarios: ScenarioSet,
        fold: int,
        values: Mapping[str, float] | None,
    ) -> list[Any]:
        """One fold's allocated rows of the joint designs, in a table of their own."""
        rule = self.allocation_rule
        if rule is None or values is None:
            return []
        joint = [
            entry
            for entry in entries
            if isinstance(entry, (JointDesignResult, ClusteredDesignResult))
        ]
        if not joint:
            raise EvaluationError(
                ".allocation(...) shares out a joint design's total payout, "
                "but no entry is a joint design: the plan's covers are all "
                f"resolved, and {self._together_name()} evaluates each as a set of "
                "contracts"
            )
        return [
            allocate_scenarios(
                joint,
                scenarios,
                allocation=rule,
                values=values,
                fold=fold,
                options=self.metric_options,
            )
        ]

    def _scenario_run(self, *, evaluate: bool, breakdown: bool = False) -> _Run:
        """`_run` for a plan with scenarios: designs per scenario, one set per fold.

        `breakdown` is for a plan of one scope, `Individual()`: the parts of each
        entry's payout instead of the comparison.
        """
        built = self._built_scenarios(fresh=evaluate, folds=True)
        stated = self.scenario_set
        assert stated is not None
        labels = list(built)
        designed_on = self._design_labels(labels)
        reference = built[stated.reference]
        portfolio = reference.portfolio
        asset_ids = list(reference.samples)
        covers = {
            label: self._covers_for(asset_ids, built[label].locations)
            for label in labels
        }
        designs: list[Any] = []
        tables: list[Any] = []
        allocated: list[Any] = []
        used: set[str] = set()
        if self._together:
            values = (
                None
                if self.allocation_rule is None
                else self._asset_values(portfolio, asset_ids)
            )
            blocks = {label: self._joint_blocks(built[label]) for label in labels}
            made: dict[tuple[str, int], list[list[Any]]] = {}
            for label in designed_on:
                for fold, block in enumerate(blocks[label]):
                    positions, drawn_up = self._joint_positions(block[1], covers[label])
                    made[(label, fold)] = positions
                    designs += drawn_up
            for fold in range(len(blocks[stated.reference]) if evaluate else 0):
                entries: list[Any] = []
                given: list[Cover] = []
                for position, cover in enumerate(self.cover_list):
                    if not cover.free_parameters():
                        own = {
                            asset_id: covers[stated.reference][asset_id][position]
                            for asset_id in asset_ids
                        }
                        entries.append(own)
                        given += own.values()
                        continue
                    for label in designed_on:
                        entries += made[(label, fold)][position]
                fold_blocks = {label: blocks[label][fold] for label in labels}
                evaluated = self._scenario_blocks(fold_blocks)
                tables.append(
                    self._scenario_table(
                        entries, fold_blocks, given, used, scenarios=evaluated
                    )
                )
                allocated += self._allocated_scenarios(
                    entries, evaluated, fold_blocks[stated.reference][0], values
                )
        else:
            per_asset = {
                (label, asset_id): self._asset_blocks(built[label], asset_id)
                for label in labels
                for asset_id in asset_ids
            }
            slots: dict[tuple[str, str, int], list[list[Any]]] = {}
            for label in designed_on:
                for asset_id in asset_ids:
                    for fold, block in enumerate(per_asset[(label, asset_id)]):
                        made_here, drawn_up = self._asset_entries(
                            asset_id, covers[label][asset_id], block[1]
                        )
                        slots[(label, asset_id, fold)] = made_here
                        designs += drawn_up
            for asset_id in asset_ids if evaluate else ():
                count = len(per_asset[(stated.reference, asset_id)])
                for fold in range(count):
                    entries = []
                    given = []
                    reference_slots = slots[(stated.reference, asset_id, fold)]
                    for i, slot in enumerate(reference_slots):
                        if not any(_designed(entry) for entry in slot):
                            entries += slot
                            given += [
                                cover for entry in slot for cover in _given(entry)
                            ]
                            continue
                        for label in designed_on:
                            entries += slots[(label, asset_id, fold)][i]
                    fold_blocks = {
                        label: per_asset[(label, asset_id)][fold] for label in labels
                    }
                    tables.append(
                        self._scenario_table(
                            entries, fold_blocks, given, used, breakdown=breakdown
                        )
                    )
        unused = sorted(
            key for key, _ in self.scenario_premiums or () if key not in used
        )
        if evaluate and unused:
            raise EvaluationError(
                f"premiums= names {unused}, which match no cover given as is"
            )
        return _Run(
            samples=MappingProxyType(reference.samples),
            fresh=None
            if reference.fresh is None
            else MappingProxyType(reference.fresh),
            designs=tuple(designs),
            tables=tuple(tables),
            allocated=tuple(allocated),
            source_key=reference.record_key,
            source=reference.record,
            id_column=portfolio.id_column,
            values_sha256=self._values_sha256(portfolio, asset_ids),
            index_source=reference.index_record,
            joint=reference.joint,
            fresh_joint=reference.fresh_joint,
            run_records=reference.run_records,
            scenarios=MappingProxyType(built),
            component_records=reference.component_records,
        )

    def samples(self) -> Mapping[str, OutcomeSample]:
        """Execution: build one outcome sample per asset, in asset order.

        Reads the source and the assets. Each sample holds all of the
        asset's draws or years, before any split (fresh draws are drawn by
        `write_parquet()` only, which returns them in `fresh_samples`), so a
        `LinearTrend` here is fitted on all the years, and its record says so.
        Under `Joint()` or `Clustered`, each is that asset's view of
        `joint_sample()`.
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
        read. With `.scenarios(...)`, these are the reference's samples, after
        the set's refusals (`ScenarioError`), and `.scenario_samples()` holds
        every scenario's. With `.hazard(...)`, each is the asset's sample of
        several hazards, as `sample_annual_hazards` or `sample_curve_hazards`
        builds it, its loss the combined loss.
        """
        if self.scenario_set is not None:
            reference = self.scenario_set.reference
            built = self._built_scenarios(fresh=False, folds=False)
            return MappingProxyType(built[reference].samples)
        return MappingProxyType(self._build_samples().samples)

    def scenario_samples(self) -> Mapping[str, Mapping[str, OutcomeSample]]:
        """Execution: every scenario's samples, by label and then by asset.

        The reference first, then as the plan states them: on fitted curves the
        selection's scenarios, then each `CurveScenario`; on per-year values the
        plan's own source, then each `AnnualScenario`. Each holds all its draws
        or years, before any split, as `.samples()` does, and the curve
        scenarios share one set of draws. The set's refusals run on them, so
        this raises `ScenarioError` as `ScenarioSet` does on built samples.
        Raises `ValueError` for a plan without `.scenarios(...)`, and as
        `samples` does.
        """
        if self.scenario_set is None:
            raise ValueError(
                "scenario_samples() holds every scenario's samples; state "
                ".scenarios(ScenarioSet(...))"
            )
        built = self._built_scenarios(fresh=False, folds=False)
        return MappingProxyType(
            {label: MappingProxyType(each.samples) for label, each in built.items()}
        )

    def joint_sample(self) -> OutcomeSample:
        """Execution: the sample of all the assets, side by side, under `Joint()`.

        And under `Clustered`, which designs on the same sample.

        Reads the source and the assets. On fitted curves it is what
        `sample_curves` draws under the plan's `.dependence(...)`, and on
        per-year values `OutcomeSample.join` of each asset's sample. It holds
        all the draws or years, before any split, so a `LinearTrend` here is
        fitted on all the years, like `.samples()`, while the rows of a holdout
        or k-fold come from one sample per fold.

        Raises `ValueError` for a plan that holds neither, and as `samples`
        does; `SampleError` when the assets' years differ, as
        `OutcomeSample.join` raises, which a stated `years=` resolves. With
        `.scenarios(...)`, it is the reference's.
        """
        if not self._together:
            raise ValueError(
                "joint_sample() puts the assets side by side under Joint() or "
                "Clustered(...); state .scopes(Joint())"
            )
        if self.scenario_set is not None:
            reference = self.scenario_set.reference
            joint = self._built_scenarios(fresh=False, folds=False)[reference].joint
        else:
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
        the plan also holds `Individual()`. Under `Clustered` each template's
        `ClusteredDesignResult` comes after its `JointDesignResult` and before
        its `DesignResult`s. With `.scenarios(...)`, they are in scenario,
        asset, fold, cover and objective order (scenario, fold, cover and
        objective under `Joint()` or `Clustered`), on every scenario's design
        blocks under
        `design_on="each"` and on the reference's under `"reference"`. With
        `.package(...)`, a package's contracts are designed where the package
        sits, its first contract's place, under each objective in turn and in
        component order. Raises
        `ValueError` without covers, an objective or a basis-risk model, the
        design errors of `design` and `design_joint`, and
        `SolverUnavailableError` without `crc-sdk[parametric]` on the LP and QP
        routes.
        """
        if self.design_spec is None:
            raise ValueError(
                "the plan has no objective; state one with .objective(...)"
            )
        return self._run(evaluate=False).designs

    def payout_breakdown(self) -> Any:
        """Execution: each entry's expected payout in its parts, on every block.

        `payout_breakdown` of the entries `write_parquet` compares, on each
        asset's design block and on any held-out block or fresh draws, in the
        same asset, fold and split order, as one table: one row per entry and
        part, with `compare`'s identity columns, `part`, `part_kind`,
        `expected_payout`, `share` and `payout_frequency`. A cover on one index
        has one part, its component; a `SumOf` or a package one per component;
        a `MaxOf` an attribution to the component paying most; a `StepGrid` its
        regions. The parts of one entry add up to its expected payout. The
        table's metadata is that of the comparison's evaluations, with the
        plan's source records; nothing is written.

        With `.scenarios(...)`, the rows are `payout_breakdown_scenarios`' on each
        fold's evaluation blocks of every scenario, one asset at a time, as
        `write_parquet`'s are `compare_scenarios`', all out of sample: each
        scenario's breakdown, with `scenario` and `design_scenario`. The metadata
        records the set under `setup.scenarios`, with `design_on`.

        Raises `ValueError` for a plan that holds `Joint()`, whose designs are on
        the owner's total, or `Clustered`, whose covers read several assets.
        Otherwise raises as `write_parquet` does.
        """
        if self._together:
            on = "the owner's total" if self._clustered is None else "several assets"
            raise ValueError(
                "payout_breakdown() breaks down one asset's covers, designs and "
                f"packages, and {self._together_name()} designs on {on}; take "
                "payout_breakdown on each asset's own design on the function route"
            )
        run = self._run(evaluate=True, breakdown=True)
        table, _ = self._written(run, run.tables)
        return table

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
        block (`scope="joint"`); a resolved cover has one `"asset"` row per
        asset and a `"total"` row (`scope="individual"`), and so does a
        template's per-asset design if the plan also holds `Individual()`. Under
        `Clustered` they are `compare`'s rows of a `ClusteredDesignResult`
        (`scope="cluster"`): a `"cluster"` row per cluster and the `"total"`
        row under `joint=False`, the `"total"` row alone under `joint=True`.
        Only the rows of a plan that holds `Joint()` or `Clustered` include a
        `"total"` row.

        Under `.allocation(rule)`, each joint or cluster design's allocated
        rows, one per asset and block, are `allocate`'s, and they go in a table
        of their own,
        never beside an `"asset"` or `"total"` row: the second file
        `<stem>.allocated<suffix>` beside `path` (for `"comparison.parquet"`,
        `"comparison.allocated.parquet"`), written atomically in the same way,
        with its own metadata, which records the rule and the hash of the asset
        values. The comparison's own evaluations then record no allocation.

        With `.scenarios(...)`, the rows are `compare_scenarios`' on each fold's
        blocks of every scenario, one asset at a time, or once for all the
        assets under `Joint()` or `Clustered`, all out of sample, in one file.
        The metadata records the set under `setup.scenarios`, with
        `design_on`; each dataset or per-year source is in `source_records`
        once, with the labels of the scenarios that read it under `scenarios`,
        and each hazard source names its `scenario`. The result's `samples` are the
        reference's, and `scenario_samples` and `scenario_fresh_samples` hold
        every scenario's. Under `.allocation(rule)` the second file holds each
        joint design's allocated rows on every scenario's block, one fold at a
        time, as `allocate_scenarios` gives them, with the same records.

        With `.hazard(...)`, every row is on the combined loss of its block: a
        cover on a `CompositeIndex` designed on the combined loss, a cover on one
        index on its own hazard's loss, and a package as one `"package"` row
        beside its contracts' `"asset"` rows, as `compare` gives them, with the
        `components` and `design_budget` columns. `source_records` holds each
        component's record once, with the components that read it under
        `components`, and each hazard source, a component's view's included,
        names its record by its checksum.

        Returns a `ParametricResult`, with the fresh draws in
        `fresh_samples` when the split is `FreshDraws`, and the allocated rows
        in `allocated` and their file in `allocated_output` under
        `.allocation(...)`. Raises `ValueError`
        without covers, a basis-risk model, or an objective for templates;
        `SampleError` for a cover that names another location than the index
        is read at, or a fold with fewer design years than `LinearTrend`'s
        `min_years`; `EvaluationError` (a `ValueError`) for `.allocation(...)`
        on a plan whose covers are all resolved, which has no joint design; and
        as `samples`, `design`, `compare` and `allocate` raise.
        """
        run = self._run(evaluate=True)
        table, metadata = self._written(run, run.tables)
        allocated = self._written(run, run.allocated) if run.allocated else None
        text = os.fspath(path)
        if not isinstance(text, str) or "://" in text:
            raise ValueError(f"write_parquet writes local files, got {path!r}")
        target = Path(text)
        beside = target.with_name(f"{target.stem}.allocated{target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_comparison(table, target, metadata=metadata)
        if allocated is not None:
            write_comparison(allocated[0], beside, metadata=allocated[1])
        return ParametricResult(
            output=target,
            table=table,
            metadata=metadata,
            samples=run.samples,
            designs=run.designs,
            fresh_samples=run.fresh,
            joint_sample=run.joint,
            fresh_joint_sample=run.fresh_joint,
            allocated=None if allocated is None else allocated[0],
            allocated_output=None if allocated is None else beside,
            scenario_samples=(
                None
                if run.scenarios is None
                else MappingProxyType(
                    {
                        label: MappingProxyType(each.samples)
                        for label, each in run.scenarios.items()
                    }
                )
            ),
            scenario_fresh_samples=(
                None
                if run.scenarios is None or run.fresh is None
                else MappingProxyType(
                    {
                        label: MappingProxyType(each.fresh or {})
                        for label, each in run.scenarios.items()
                    }
                )
            ),
        )

    @staticmethod
    def _scenario_sources(run: _Run, document: dict[str, Any]) -> None:
        """Each scenario's source records, once each, and each source's checksum.

        A record read by several scenarios, as the selection's dataset is, is
        kept once with all their labels under `scenarios`. Each hazard source
        names its record by its checksum and its scenario by `scenario`.
        """
        assert run.scenarios is not None
        records: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for label, built in run.scenarios.items():
            parts = (
                _component_parts(built.component_records)
                if built.component_records is not None
                else [built.record]
                if built.run_records is None
                else list(built.run_records.values())
            )
            for part in parts:
                text = json.dumps({built.record_key: part}, sort_keys=True, default=str)
                if text in seen:
                    records[seen[text]][built.record_key]["scenarios"].append(label)
                    continue
                seen[text] = len(records)
                records.append({built.record_key: {**part, "scenarios": [label]}})
        if run.index_source is not None:
            records.append({"index_table": run.index_source})
        label_of = {
            str(sample["sample_hash"]): str(sample["scenario"])
            for sample in document["setup"]["samples"]
        }
        sources: list[dict[str, Any]] = []
        components = _source_components(document)
        for entry, component in zip(document["sources"], components, strict=True):
            role = entry.get("role")
            extra: dict[str, Any] = {}
            if role == "index" and run.index_source is not None:
                extra = {"index_table_sha256": run.index_source["table_sha256"]}
            scenario = label_of.get(str(entry.get("sample_hash")))
            if role == "hazard" and scenario is not None and scenario in run.scenarios:
                built = run.scenarios[scenario]
                key = built.record_key
                if built.component_records is not None:
                    part = built.component_records[str(component)]
                    extra = {f"{key}_sha256": part[_checksum_key(key)]}
                elif built.run_records is not None:
                    part = built.run_records[(entry["model"], entry["member"])]
                    extra = {f"{key}_sha256": part["table_sha256"]}
                else:
                    checksum = (
                        "file_sha256" if key == "canonical_file" else "table_sha256"
                    )
                    extra = {f"{key}_sha256": built.record[checksum]}
            if scenario is not None:
                extra["scenario"] = scenario
            sources.append({**entry, **extra})
        document["sources"] = sources
        document["source_records"] = [*document.get("source_records", ()), *records]

    def _written(
        self, run: _Run, tables: Sequence[Any]
    ) -> tuple[Any, EvaluationMetadata]:
        """The tables of a run as one, with the plan's records in its metadata."""
        merged = EvaluationMetadata.merge(
            *(EvaluationMetadata.from_table(table) for table in tables)
        )
        document = merged.model_dump(mode="json")
        if run.scenarios is not None:
            self._scenario_sources(run, document)
        else:
            checksum = run.source[
                "file_sha256" if run.source_key == "canonical_file" else "table_sha256"
            ]
            checksums = {"hazard": {f"{run.source_key}_sha256": checksum}}
            records: list[dict[str, Any]] = [{run.source_key: run.source}]
            by_run = run.run_records
            if by_run is not None:
                records = [{run.source_key: record} for record in by_run.values()]
            by_component = run.component_records
            if by_component is not None:
                records = [
                    {run.source_key: part} for part in _component_parts(by_component)
                ]
            if run.index_source is not None:
                checksums["index"] = {
                    "index_table_sha256": run.index_source["table_sha256"]
                }
                records.append({"index_table": run.index_source})
            components = _source_components(document)
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
                    **(
                        {
                            f"{run.source_key}_sha256": by_component[str(component)][
                                _checksum_key(run.source_key)
                            ]
                        }
                        if by_component is not None
                        and entry.get("role") == "hazard"
                        and component in by_component
                        else {}
                    ),
                }
                for entry, component in zip(
                    document["sources"], components, strict=True
                )
            ]
            document["source_records"] = [
                *document.get("source_records", ()),
                *records,
            ]
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
        if run.scenarios is not None and document["setup"].get("scenarios"):
            document["setup"]["scenarios"]["design_on"] = self.design_on
        metadata = EvaluationMetadata.model_validate(document)
        table = pa.concat_tables(tables)
        existing = dict(table.schema.metadata or {})
        existing[EVALUATION_METADATA_KEY.encode("utf-8")] = metadata.to_json().encode(
            "utf-8"
        )
        table = table.replace_schema_metadata(existing)
        return table, metadata
