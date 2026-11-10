# Parametric covers

[Docs index](README.md) · [Project README](../README.md)

`crc_sdk.parametric` defines, designs, evaluates and compares parametric
insurance covers on hazard indices. A parametric cover pays an amount fixed in
advance whenever a measured index, such as a year's maximum daily rainfall,
reaches stated thresholds, whatever the loss turns out to be. The gap between
payout and loss is the basis risk.

There are two ways to use the module:

- **The plan.** `ParametricPlan.for_evaluation(request)` takes the portfolio
  evaluation you build with `HazardDataset.local(path).for_assets(assets)
  .select(...).impact(...)`, draws simulated years from each asset's fitted
  hazard curve, and chains the steps to a written comparison. It is a lazy
  crc-sdk plan: builders and `explain()` do no I/O, and the data is read only
  at the execution points `samples()`, `designs()` and `write_parquet()`. The
  same plan also runs on per-year values, such as ERA5's annual extremes
  ([next step](#next-step-per-year-values)).
- **The functions.** Every step is also a function on numpy arrays and Arrow
  tables that does no I/O (`sample_curve`, `sample_annual`, `design`,
  `compare`, ...), for your own tables, fitted curves and chains. Only
  `write_comparison` writes a file.

## Concepts

| Concept | Name | What it is |
| --- | --- | --- |
| Cover | `Cover` | The contract terms: an index, a payout function with its parameters, the payout unit and the asset |
| Index | `IndexDefinition` | What the cover pays on: hazard name, unit, severe tail, location, aggregation |
| Payout function | `LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or your own | Maps a year's index value to its payout |
| Template | a `Cover` with `Free` parameters | Terms a design chooses within stated bounds |
| Outcome sample | `OutcomeSample` | Years (or simulated years), each with a probability, an index value and a loss |
| Loss model | `LossModel` | loss = impact(hazard) × asset value |
| Basis-risk model | `HazardNoise`, `ImpactNoise`, `NoBasisRisk`, or an index series read elsewhere | How the index and the loss differ |
| Objective | `MSE`, `ExpectedUncovered`, `WeightedCVaR`, `UserObjective` | What a design minimises |
| Pricing and constraints | `PurePremium`, `CostOfCapitalLoad`, `Budget`, `Limit` | The premium of a payout, and what a design must respect |
| Design | `design`, `DesignResult` | A template's resolved cover and its design record |
| Scope | `Individual()`, `Joint()` | Whether each asset's cover is designed on its own or several together, for one objective on the owner's total ([joint design](#joint-design-over-several-assets)) |
| Comparison | `compare`, a `pyarrow.Table` | Covers as rows, metrics as columns, in and out of sample |
| Pooled folds and intervals | `compare_pooled`, `bootstrap_intervals` | Metrics computed once over all the held-out years of a k-fold split, and how much a number moves with the years drawn ([pooled folds and intervals](#pooled-folds-and-intervals)) |
| Plan | `ParametricPlan`, `ParametricResult` | The lazy workflow from a portfolio evaluation's curves (or per-year values) to a written comparison |

"Impact" is crc-sdk's impact value, usually a damage ratio; "loss" is impact ×
value, in the payout's unit; "hazard" is the value the loss model reads at the
asset; "index" is the value the cover pays on. Without a basis-risk model,
index and hazard are the same value. "Portfolio" keeps crc-sdk's meaning, a
set of assets.

## Quick start

Design a step cover on 5,000 years simulated from a fitted rainfall curve,
under three objectives, and compare the designs with a fixed linear cover on
5,000 fresh draws. The canonical dataset (one Gumbel curve in the asset's H3
cell) and the asset are an illustration, not data. The designs need the
optional extra `crc-sdk[parametric]` (the HiGHS solver), and `point_to_cell`
needs `crc-sdk[geometry]`.

```python
import pyarrow as pa
from crc_framework import PiecewiseLinearImpact
from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    MSE, Budget, Cover, ExpectedUncovered, Free, FreshDraws, HazardNoise,
    ImpactNoise, IndexDefinition, LinearPayout, MetricOptions, ParametricPlan,
    StepPayout, WeightedCVaR, setup_text,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import HazardDataset

# A synthetic canonical dataset of annual maximum daily rainfall. Real curves
# come from a source plan: HazardDataset.crc_open(...)...materialize(path).
metadata = HazardDatasetMetadata(
    h3_resolution=7, value_unit="mm/day",
    value_semantics="annual maximum daily rainfall",
    probability_semantics="annual_exceedance",   # the curve of an annual maximum
    return_period_support=(2.0, 500.0),          # the fitted range
    producer="example", creation_version="1",
    source=SourceProvenance(provider="example", dataset="synthetic rx1day",
                            licence="CC0-1.0", attribution="Synthetic data"),
)
curve = {
    "cell_index": point_to_cell(6.96, 50.94, 7), "source_id": "example",
    "source_geometry": None, "hazard_name": "rx1day", "horizon": 2050,
    "pathway": "ssp245", "curve_kind": "fitted", "curve_type": "gumbel_r",
    "curve_shape": None, "curve_location": 50.0, "curve_scale": 12.0,
    "curve_atom_probability": None, "curve_atom_location": None,
}
write_hazard_dataset(pa.Table.from_pylist([curve], schema=hazard_arrow_schema()),
                     "rx1day.parquet", metadata)

assets = pa.table({"asset_id": ["warehouse-a"], "longitude": [6.96],
                   "latitude": [50.94], "value_eur": [2e6]})
request = (
    HazardDataset.local("rx1day.parquet")
    .for_assets(assets)
    .select(hazard_names=["rx1day"], horizons=[2050], pathways=["ssp245"])
    .impact(PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),   # damage ratio
            name="rx1day_damage", value_unit="1", value_semantics="damage ratio")
)

index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
step = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
             payout_unit="EUR", label="step")
linear = Cover(index, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR",
               label="linear")

plan = (
    ParametricPlan.for_evaluation(request, value_column="value_eur",
                                  value_unit="EUR")
    .monte_carlo(5_000, seed=42)               # simulated years from the curve
    .basis_risk(HazardNoise(0.3))              # stated, never assumed
    .covers(step, linear)                      # a template and a fixed cover
    .objective(MSE(), ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9),
               budget=Budget(ratio=0.8))
    .split(FreshDraws(5_000, seed=7))          # evaluate on fresh draws
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
)
print(setup_text(plan.explain(format="json")))   # the setup; no I/O

result = plan.write_parquet("comparison.parquet")   # execution; atomic
```

`plan.explain()` gives the whole plan as JSON text; `setup_text` states its
setup and the dataset it reads in plain sentences. `result.table` is the comparison as written: one row
per cover (the three designs, labelled `"step (mse)"`,
`"step (expected_uncovered)"` and `"step (weighted_cvar alpha=0.9 beta=0.9)"`,
and the linear cover) and split. The out-of-sample rows, on the fresh draws,
are the ones to report, each with its setup:

```python
import pyarrow.compute as pc

table = result.table
out_of_sample = table.filter(pc.equal(table["split"], "out_of_sample"))
out_of_sample.select(["cover_label", "premium_basis", "premium",
                      "expected_uncovered", "coverage_ratio", "tail_coverage_b0_9",
                      "unique"])
print(setup_text(out_of_sample))   # the setup of these rows, to print with them
[design.unique for design in result.designs]   # is each design the only optimum?
```

The designs' rows use the design premium (`premium_basis` `"design"`), the
linear cover's rows the premium of its payouts on the fresh draws
(`"evaluation"`). The setup text also states what the curve's fitted range
means for these draws: with a shortest fitted return period of 2 years, about
half of the simulated years are more frequent than it, where the curve is
extrapolated. Any
number read from this table describes this synthetic curve under this setup
and nothing more.

## The parametric plan

```text
ParametricPlan.for_evaluation(request, *, value_column=None, value_unit="1")
  .monte_carlo(n_draws, *, seed, assume_semantics=None, outside_support="raise")
ParametricPlan.for_annual_extremes(source, assets, *, impact, unit, provenance,
                                   trend=AsRecorded(), scenario="baseline",
                                   pool=None, ...)
  .index(source, *, location, unit, provenance, ...)   # optional: read elsewhere
# then, on either plan:
  .basis_risk(model)                          # required
  .covers(*covers)                            # resolved covers and templates
  .objective(*objectives, pricing=None, budget=None, limit=None,
             options=DesignOptions())         # to design the templates
  .pricing(PurePremium() | CostOfCapitalLoad(rate, q))   # optional
  .split(FreshDraws(n_draws, seed=...) | Holdout(train_fraction) | KFold(k))
  .metrics(MetricOptions(...))                # optional
  .scopes(Individual() | Joint())             # optional; default Individual()
  .dependence(Independent() | Comonotonic() | EmpiricalCopula(reference))
                                              # fitted curves; required with Joint()
  .allocation("value" | "loss")               # optional: a joint design's asset rows
  .explain(format="text")                     # no I/O
  .samples() / .joint_sample() / .designs() / .write_parquet(path)   # execution
```

### Fitted curves

- **`request`** is the `PortfolioEvaluation` built with
  `HazardDataset.local(path).for_assets(assets).select(...).impact(...)`, the
  request whose `write_parquet` evaluates the assets at return periods. The
  plan reuses its dataset, assets, selection and impact and changes nothing in
  it. Materialize a remote source first: a source plan's `materialize(path)`
  returns the `HazardDataset` to build the request on.
- **The curves.** At execution, `request.asset_curves()` joins each asset to
  its canonical curve as `write_parquet` does (the asset's H3 cell, refined by
  exact geometry where the source has it); an asset without a curve, or with
  several for one scenario, raises. The plan samples one curve per asset, so
  the selection must leave one hazard, horizon and pathway; several raise
  `SampleError`.
- **`.monte_carlo(n_draws, *, seed, assume_semantics=None,
  outside_support="raise")`** is required: each asset's sample is `n_draws`
  simulated years drawn from its curve by `sample_curve`, each with
  probability `1/n_draws`. The rules of [outcome samples](#outcome-samples)
  apply. The dataset's `probability_semantics` must be `annual_exceedance` or
  `annual_value_distribution`; a dataset that declares none is refused unless
  `assume_semantics=` states one of the two, and the setup then says that the
  caller stated it. A `no_data` curve raises. Draws beyond the fitted support
  are counted and named in the setup, and so is a pooled ensemble.
  `outside_support` is what a tabulated curve does with draws outside its
  knots.
- **The loss** is `impact(hazard) × value`: the request's impact (an
  `ImpactExpression` keeps its `definition_hash`) times the asset column
  `value_column` in `value_unit`, or 1 with `value_unit="1"` (the damage
  ratio). A framework impact gets the context `write_parquet` gives it: its
  own, with the asset's matched H3 cell and the request's context columns on
  top, so a registry impact such as `rx1day` needs no cell of its own.
- **The index** is the curve's hazard at the asset, in the dataset's
  `value_unit`: each cover's index must be in that unit, which is checked at
  execution.
- **The assets** are an in-memory table or a local Parquet file, read at
  execution, in row order. Asset values are read for the loss and never
  written.

### Builders

| Builder | What it states |
| --- | --- |
| `.monte_carlo(n_draws, *, seed, ...)` | Fitted curves only, and required there: the simulated years per asset, the plan's seed, the semantics the caller states for a dataset that declares none, and `outside_support` |
| `.basis_risk(model)` | How index and loss differ: `HazardNoise(sigma)`, `ImpactNoise(sigma, upper=...)`, `NoBasisRisk()` (warns with `ComonotoneIndexWarning`) or a `BasisRisk` of your own ([what each can change](#basis-risk-models)). Required: no model is assumed |
| `.covers(*covers)` | The covers to compare. A template is designed; a resolved cover is evaluated as it is. A cover written for no asset applies to every asset, as a copy written for that asset; one written for an asset applies to it only. Each cover pays in the loss unit, on an index read at the asset (`AtAsset()`), in the hazard's unit, or as [`.index(...)`](#an-index-read-elsewhere) states |
| `.objective(*objectives, pricing, budget, limit, options)` | Designs every template under each objective, with `design`'s terms. `pricing` is the plan's pricing, below |
| `.pricing(pricing)` | The plan's pricing, `PurePremium()` or `CostOfCapitalLoad(rate, q)`, for a plan of resolved covers that has no `.objective(...)` |
| `.split(rule)` | `FreshDraws(n_draws, seed=...)` (fitted curves), `Holdout(train_fraction)` or `KFold(k)`, below |
| `.metrics(options)` | The `MetricOptions` of every row |
| `.scopes(*scopes)` | `Individual()`, the default, designs each asset on its own; `Joint()` designs each template on the owner's total ([joint design](#joint-design-over-several-assets)) |
| `.dependence(model)` | Fitted curves only, and required there with `Joint()`: how the assets' draws are put side by side (`Independent()`, `Comonotonic()` or `EmpiricalCopula(reference)`). A plan on per-year values raises: its dependence is the data's own |
| `.allocation(rule)` | With `Joint()`: adds a joint design's per-asset rows, under `"value"` or `"loss"` |

A designed cover keeps the template's label (or the first 12 hex digits of its
hash) followed by the objective, for example `"step (mse)"`; its hash and
design id identify it.

**Pricing.** One pricing prices every premium of a plan, so that its rows are
priced alike: each design's on its design block, and each resolved cover's on
each evaluation block. `.objective(..., pricing=...)` states it, and so does
`.pricing(...)`, which a plan of resolved covers needs to price them without an
objective. Both may be given if they state the same pricing; two different
pricings raise `ValueError`, whichever comes first. Without either, the
premiums are pure (`PurePremium()`). `explain()` shows the pricing in every
planned evaluation, and in the design terms.

```python
from crc_sdk.parametric import CostOfCapitalLoad

priced = (plan.covers(linear)                   # the quick start's fixed cover
              .pricing(CostOfCapitalLoad(0.06, 0.99)))
priced_result = priced.write_parquet("priced.parquet")
```

### Splits

- **No split:** every cover is designed and evaluated on the whole sample,
  and every row is in sample.
- **`FreshDraws(n_draws, seed=...)`** (fitted curves): designs on the plan's
  draws, evaluation on `n_draws` fresh draws from the same curve. Draws are
  independent simulated years, so this is the plain out-of-sample test. Its
  seed must differ from the plan's.
- **`Holdout(train_fraction)`:** each asset's sample is split by `holdout`:
  designs on the first `floor(train_fraction · n)` periods (years, or draws),
  evaluation on the rest, so a per-year sample is tested on its latest years.
- **`KFold(k)`:** `year_block_kfold` holds out each of `k` contiguous blocks of
  periods in turn; every fold has its own designs.

A pool of model-years is split by its distinct years, not its rows: see [pooled
model-years](#pooled-model-years).

With `LinearTrend`, `Holdout` and `KFold` build each asset's sample once per
fold, fitted on that fold's design years: see [Climate
trends](#climate-trends).

Every design block gets in-sample rows and every held-out block, or the fresh
draws, out-of-sample rows (`split`, `fold`). Resolved covers are evaluated on
the same blocks, with the design block as their design sample: their terms are
taken as chosen without the held-out data. The leakage checks of `compare` run
on every block.

### Seeding

The plan holds an integer seed: `monte_carlo`'s on fitted curves,
`for_annual_extremes(..., seed=...)`'s on per-year values. At execution it
builds `numpy.random.SeedSequence(seed)` and spawns one child generator per
asset, in asset order, so adding an asset at the end leaves the others'
samples as they were. Each child draws the hazard first, then the basis-risk
model. Fresh draws spawn their children from `SeedSequence(FreshDraws.seed)`
in the same way. On per-year values only the basis-risk model draws, so
`HazardNoise` and `ImpactNoise` need a seed there, at every sigma. Each
sample's provenance records its
generator's state before each draw. With `Joint()` on fitted curves,
`sample_curves` takes those same children, one per asset, and its dependence
model reorders the rows without any further random number.

### Explain, execute, write

`explain()` returns the plan's setup as far as it is known before execution,
as JSON text (`format="json"` gives a dict), without I/O. It mirrors the
metadata the plan writes: `setup` (the basis, the metric options, one sample per
asset with its loss and basis-risk models, the planned evaluations with their
split, fold and pricing), `covers`, `designs`, `sources`, `assets`, `licences`,
`rng` and `entry`. On fitted curves, `monte_carlo` lists the draws, the
semantics the caller states and the fresh draws; the dataset's own metadata
(its units, semantics and licence) is read only at execution, so `licences` is
`None` there.

| Execution point | Returns |
| --- | --- |
| `.samples()` | One `OutcomeSample` per asset, all draws or years, in asset order; with `Joint()`, each asset's view of `.joint_sample()` |
| `.joint_sample()` | With `Joint()` only: the `OutcomeSample` of all the assets, all draws or years |
| `.designs()` | Every `DesignResult`, in asset, fold, cover and objective order; with `Joint()`, every `JointDesignResult`, in fold, cover and objective order |
| `.write_parquet(path)` | A `ParametricResult`: `output`, `table`, `metadata`, `samples`, `designs`, `fresh_samples`, `joint_sample`, `fresh_joint_sample` and `row_count` |

`write_parquet` writes the rows of every asset, split and fold to one local
Parquet file through a temporary sibling and `os.replace`, and creates the
parent directory. Its metadata, under `crc.parametric.evaluation`, merges that
of every block (below) and adds the source's record once, in
`source_records`: on fitted curves `{"canonical_file": {...}}` (the dataset's
path, its SHA-256 and the number of matched curves; each sample also records
its curve row's key and spatial match), on per-year values
`{"annual_table": {...}}` (its rows, columns and SHA-256, the source plan's
`explain()`, and for a file its path and SHA-256). Each hazard source names
its record by that checksum, `canonical_file_sha256` or `annual_table_sha256`,
so merged comparisons keep each record once and tell their samples apart.
Every evaluation's `details` holds the plan's seed (`"plan_seed"`), every
fresh-draw evaluation's also the fresh draws' seed (`"fresh_draws_seed"`), and
`assets` the count, the id column and a SHA-256 of the (asset id, value) pairs.
Metadata that plans wrote before `source_records`, with the record repeated in
every hazard source, reads as before.

The `ParametricResult` holds what the rows are evaluated on: `samples`, one
sample per asset with all its draws or years, and, with `FreshDraws`,
`fresh_samples`, one sample per asset of the fresh draws, whose
`content_hash` is the `sample_hash` of the out-of-sample rows (`None` without
fresh draws). With `LinearTrend` and a holdout or k-fold split, `samples` are
fitted on all the years, and the rows come from one sample per fold. With
`Joint()`, `joint_sample` and `fresh_joint_sample` hold the samples of all the
assets, and `samples` and `fresh_samples` each asset's view of them.

## Next step: per-year values

The same plan runs on per-year values: `ParametricPlan.for_annual_extremes`
reads the years of an annual-extremes table (ERA5, CMIP6, NEX-GDDP-CMIP6, a
station, NOAA water levels) at each asset. Each row is a recorded or modelled
year with probability `1/n`, so no curve and no probability semantics are
assumed, and by default the years are used as recorded ([Climate
trends](#climate-trends)). Here the quick start's covers and asset are
designed on 28 years of a synthetic table and compared on the 12 years held
out:

```python
import numpy as np
from crc_sdk.parametric import Holdout

# The layout of ERA5's and CMIP6's annual_extremes(): one row per year and cell.
rng = np.random.default_rng(1)
years = np.arange(1981, 2021)
annual = pa.table({
    "year": np.repeat(years, 4).astype("int32"),
    "latitude": np.tile([50.875, 50.875, 51.125, 51.125], len(years)),
    "longitude": np.tile([6.875, 7.125, 6.875, 7.125], len(years)),
    "value": rng.gumbel(50.0, 12.0, 4 * len(years)),       # mm/day
})
provenance = SourceProvenance(provider="example", dataset="synthetic rx1day",
                              licence="CC0-1.0", attribution="Synthetic data")

annual_plan = (
    ParametricPlan.for_annual_extremes(
        annual, assets,
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),   # damage ratio
        unit="mm/day", value_column="value_eur", value_unit="EUR",
        provenance=provenance, seed=42,
    )
    .basis_risk(ImpactNoise(0.3, upper=1.0))
    .covers(step, linear)
    .objective(MSE(), ExpectedUncovered(), WeightedCVaR(alpha=0.9, beta=0.9),
               budget=Budget(ratio=0.8))
    .split(Holdout(0.7))                       # design on 1981-2008, test on 2009-2020
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
)
annual_result = annual_plan.write_parquet("annual-comparison.parquet")
```

The builders, splits, seeding and outputs are those above.

### Source and assets

- **`source`** is a plan with an `annual_extremes()` execution point (the ERA5
  and CMIP6 canonicalization plans, NEX-GDDP-CMIP6, the station plans, NOAA
  water levels), an Arrow table in one of their layouts, or a local Parquet
  file of one (for example from a station plan's `write_annual_extremes`). It
  is read once per execution.
- **`assets`** is an `AssetPortfolio`, an in-memory table it accepts, or a
  local Parquet file read at execution. The asset order is the table's row
  order. Asset values are read for the loss and never written.
- **`location`** finds each asset's series. `NearestCell()` reads per-cell
  tables (`year`, `latitude`, `longitude`, `value`): the cell whose centre is
  nearest the asset, accepted only within half a grid step in latitude and
  longitude (`AnnualSeries.nearest_cell`). An asset with a point
  (`PointColumns`) is located at its point. An asset with an H3 cell
  (`CellColumn`) is located at its cell, with the `geometry` extra: the grid
  cell nearest the H3 cell's centre is accepted only if the whole H3 cell lies
  within it. An H3 cell that straddles two grid cells does not say which
  series is the asset's, so it raises `SampleError`; give that asset a point.
  Such a series records `"asset_location": "h3_cell"`.
  `StationColumn("station_id")` reads per-station tables: the asset column
  names each asset's station, matched against the table's `station` column
  (`table_column="station_id"` for NOAA water levels). An asset that cannot be
  located raises; none is dropped.
- **`unit`** is the unit of the table's values. It is checked against the
  impact's calibration (an `ImpactExpression` or a registry impact), against a
  `unit` column of the table, and against the source plan's `explain()`.
- **`provenance`** is a `SourceProvenance` with the source's licence,
  attribution and version, which annual-extremes tables do not carry (for a
  plan that also materializes curves, `HazardDataset.local(path).provenance()`
  of its canonical file). It identifies the series of every sample, so give
  one source the same provenance each time: the leakage checks then link its
  blocks. A pathway that a source plan's `explain()` states, as a CMIP6 plan's
  does, is recorded beside it at execution, and the setup text states it;
  before execution the plan does not read its source, so the text cannot.
- **The loss model** is `LossModel(impact, value_column=..., value_unit=...,
  context=...)`: loss = `impact(hazard) × value`, the value from an asset
  column, or 1 with `value_unit="1"` (the damage ratio). A registry impact
  that needs an H3 cell in its context (such as `rx1day`) raises unless
  `cell=` states each asset's, the impact, or `context`, is a
  `TransformContext` with the cell, or the assets have H3 cells (`CellColumn`)
  and `context` names context columns (`ImpactContextColumns`), which give
  each asset's impact its own cell.
- **`cell`** gives each asset an H3 cell for the impact's context, as point
  assets need for such an impact: `cell="h3_cell"` reads it from an asset
  column of unsigned 64-bit H3 cells, and `cell=H3Cell(resolution=r)` computes
  the cell of the asset's point at resolution `r`, at execution, which needs
  the `geometry` extra (`crc-sdk[geometry]`). There is no default resolution;
  state the one the impact expects. The series is still located by `location`.
  Each sample's provenance and the setup text record the cell, not the point.
- **`years`** states the years every series must cover (by default the years
  the table holds); a year without a value raises, unless `missing="drop_year"`
  drops it and lists it.

### Climate trends

The years of a record carry the climate's trend over them, so using them as
they are is a choice like any other, and the plan states it. `trend=` takes one
of three options, on `for_annual_extremes`, `sample_annual` and
`sample_annual_pool`. Which one suits a hazard is your choice, and the module
says nothing about it.

| Option | What it does | What the sample records in `provenance.trend` |
| --- | --- | --- |
| `AsRecorded()`, the default | Uses the years as they are | `{"kind": "as_recorded"}` |
| `RecentYears(n)` | Keeps the last `n` entries of the series' years. No value changes | `n` and the window's first and last year |
| `LinearTrend(reference_year=None, *, min_years=10)` | Removes an additive linear trend in the mean, by ordinary least squares on the year | `method` (`"ols_additive"`), `reference_year`, `min_years`, `fit_years`, and in `fits` one `role`, `slope`, `intercept` and `unit` per series |

```python
from crc_sdk.parametric import LinearTrend

trended_result = (
    ParametricPlan.for_annual_extremes(
        annual, assets,
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
        unit="mm/day", value_column="value_eur", value_unit="EUR",
        provenance=provenance, seed=42,
        trend=LinearTrend(),                  # fitted per fold, on the design years
        scenario="1981-2020, moved to 2020",  # your name for the rows
    )
    .basis_risk(ImpactNoise(0.3, upper=1.0))
    .covers(step, linear)
    .objective(ExpectedUncovered(), budget=Budget(ratio=0.8))
    .split(Holdout(0.7))
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
    .write_parquet("trended-comparison.parquet")
)
print(setup_text(trended_result.metadata))
```

**`RecentYears(n)`** comes after `years=` and before `missing=`: a gap before
the window does not matter, and a gap inside it is a missing year. An `n` above
the series' length raises `SampleError`. The window comes first and a split
divides it, so the held-out block is the latest years inside the window.

**`LinearTrend`** fits `x_t = a + b·(t − t₀) + e_t` on the fit years, with `t`
the calendar year and `t₀` the reference year, and replaces every year's value
by `x_t − b·(t − t₀)`. So `a`, the recorded intercept, is the fitted value at
the reference year, and the adjusted fit years have mean `a`. The spread around
the line is not touched, so a change in the variance or the tail stays in the
data.

- **The reference year** is by default the last entry of the series' years,
  whether or not it has a value and whichever block it falls in, so it is the
  same in every fold. A stated year may lie outside the record.
- **One fit per series.** Each asset's hazard series has its own fit. With
  `.index(...)` the index series has its own, on the same years with the same
  reference year.
- **Too few years.** Fewer than `min_years` fit years raise `SampleError`,
  which names the count and, on a plan, the fold; no fold is dropped. The
  default of 10 is a floor against a slope that is nearly all noise, not a
  statement that ten years are enough. `min_years` below 3, which would leave
  no residual, raises at construction.
- **No clipping.** A non-finite result, or a value that was non-negative as
  recorded and is negative after the adjustment, raises `SampleError`. A value
  that is negative as recorded (a temperature) may stay negative.
- **Under a split**, the fit sees the design years only. With `Holdout` or
  `KFold`, the plan builds each asset's sample once per fold, fitted on the
  years the split puts in that fold's design block, and takes the fold's two
  blocks from it: the held-out years are adjusted with the design years' slope,
  and a changed held-out value changes neither the fit nor any row of the
  design block. A basis-risk model that draws starts every fold from the
  asset's same generator state, so a year's factor does not depend on the fold.
  `.samples()` and `ParametricResult.samples` hold all the years before any
  split, so their fit uses all the years, and their record says so.
- **`fit_years`** is for one sample built by hand ([from per-year
  values](#from-per-year-values)). On a plan it raises `ValueError`, because the
  plan sets it per fold.

**The order** is `years=`; the window of `RecentYears`; `missing=`; the fit and
the adjustment of the hazard and of the index; the impact on the adjusted
hazard; the basis-risk model. A trend draws no random numbers. The sample's
`index` and `hazard` hold the adjusted values.

**In the metadata and the text.** `setup_text` writes one `Trend` line for
every per-year sample, the default included: the option and, for `LinearTrend`,
the fit years, each series' slope with its unit, its intercept and the
reference year. In a comparison's metadata each `setup.samples` entry holds the
record under `trend`, null on Monte Carlo and table samples, and a document
without the key reads as the years as recorded. Before execution,
`plan.explain(format="json")` states the option in each `setup.samples` entry.

**`scenario=`** names the rows: it sets `sample.scenario`, so it reaches
`to_table`, `content_hash` and the comparison's `scenario` column. It is yours,
and is not derived from the source. The `scenario` column of a CMIP6 table is
its pathway, another thing, and the setup text states both, so state a label
whenever you use a pathway: without one, rows under `ssp245` are labelled
`"baseline"`. The leakage check sees through a relabelling.

**`hazard_name=`** names the hazard the loss reads. Every sample builder takes
it, as `hazard_names=` by component on the builders of several hazards, and a
table with several index columns takes a mapping. It is optional where the source
names a hazard: a canonical row's `hazard_name`, a source document's `recipe` or
`hazard_name`, or a one-valued `recipe` column. Then the two must agree, or
`ScenarioError` names both. `sample.provenance.hazard` records the name and what
states it, and the hashes and the lineage leave it out.

### An index read elsewhere

The covers can pay on a series read elsewhere than the loss, such as a station
or a grid cell that is not the asset's own. `.index(...)` states that series;
the loss still reads the hazard of `for_annual_extremes`. It is for plans on
per-year values:

```python
from crc_sdk.parametric import NoBasisRisk, StationColumn

# The layout of a station plan's annual_extremes(): one row per year and station.
stations = pa.table({
    "station": ["ST-1"] * len(years),
    "year": years.astype("int32"),
    "value": rng.gumbel(45.0, 11.0, len(years)),       # mm/day
})
gauges = SourceProvenance(provider="example", dataset="synthetic gauges",
                          licence="CC0-1.0", attribution="Synthetic data")

station_plan = (
    ParametricPlan.for_annual_extremes(
        annual, assets.append_column("station_id", pa.array(["ST-1"])),
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
        unit="mm/day", value_column="value_eur", value_unit="EUR",
        provenance=provenance, seed=42,
    )
    .index(stations, location=StationColumn("station_id"), unit="mm/day",
           provenance=gauges)
    .basis_risk(NoBasisRisk())                 # required, as on every plan
    .covers(step, linear)
    .objective(ExpectedUncovered(), budget=Budget(ratio=0.8))
    .split(Holdout(0.7))
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
)
station_result = station_plan.write_parquet("station-comparison.parquet")
```

```text
.index(source, *, location, unit, provenance, point=None, year_column="year",
       index_column="value")
```

Each asset's sample is that of `sample_annual(hazard, index=series, ...)` built
by hand ([from per-year values](#from-per-year-values)), so the dependence
between index and loss is the data's own.

- **`source`** takes what `for_annual_extremes` takes, and is read once per
  execution. `year_column` and `index_column` name its columns.
- **`location`** is required. `StationColumn(name)` reads the station an asset
  column names. `NearestCell(...)` reads the nearest cell of the index
  source's own grid, under the rule above: at the asset's location by default,
  or, with `point=PointColumns(longitude, latitude)`, at the point that those
  asset columns hold, such as a reference site. Only the matched cell is
  recorded, not the point, so an index grid finer than the hazard's records an
  asset's place to within its own cell. `point=` with `StationColumn` raises
  `ValueError`.
- **`unit`** is the index series' unit. It is checked against a `unit` column
  of the table and against the source plan's `explain()`, and not against the
  impact, because the loss does not read it.
- **`provenance`** states the index source's licence, attribution and version,
  as for the hazard.
- **The years.** The index is read on exactly the hazard's years. A hazard year
  that the index lacks or holds as null is a missing year under the plan's one
  `missing=`: it raises and is listed or, with `missing="drop_year"`, it is
  dropped from the sample and listed in `dropped_years`. An index year the
  hazard does not have is not used, and `unused_index_years` lists it.
- **The covers.** At execution, a cover stated with `AtAsset()` is bound to each
  asset's resolved location, as the plan already binds it to the asset:
  `AtStation(<the asset's station id>)`, or `AtPoint(<the matched cell's
  centre>)` with its longitude wrapped to [−180, 180]. So the location is in the
  cover's document, its `definition_hash`, the `cover_hash` column and the
  `design_id`, and the asset's own point is not. A cover that already names a
  location must name that one: another kind of location raises `SampleError`
  at `.covers(...)`, and another station or point at execution.
- **Units and order.** `.covers(...)` checks a cover's index unit against
  `.index`'s `unit`, not the hazard's, so state `.index(...)` first when the
  two differ or when a cover names a location. Stated after `.covers(...)`, it
  checks the covers again and raises on a mismatch.
- **Basis risk.** `.basis_risk(model)` stays required. `NoBasisRisk()` does not
  warn with a separate index series, as in `sample_annual`, and `ImpactNoise`
  may be added on top. `HazardNoise` raises `SampleError` at execution: the
  index and the hazard already differ, and the hazard at the asset is then data.
- **What is recorded.** `explain()` shows the covers as stated and the location
  rule, in the `sources` entry with role `"index"`. In the written metadata,
  `covers` holds the bound documents; `sources` has an entry with role
  `"index"`, the series with its matched cell or station; `source_records` has
  an `{"index_table": ...}` record of the form of `annual_table`, which each
  index entry names by `index_table_sha256`; and `licences` holds the index
  source's terms. Each sample's notes, and so the setup text, state where its
  index is read.
- **Refused.** On a plan on fitted curves `.index(...)` raises `ValueError`: an
  index at another location needs a joint distribution of two curves.

### Pooled model-years

A single model run has few years. `pool=` pools the years of several model runs
under one pathway into one sample per asset, on `for_annual_extremes`, and
`sample_annual_pool` does so by hand. Every model-year is one row. Whether a
pool suits a question is your choice, and the module says nothing about it.

```python
from crc_sdk.parametric import ModelPool

# The layout of NEX-GDDP-CMIP6's annual_extremes(): one row per year and cell,
# whose model, member and scenario columns say whose it is.
future = np.arange(2015, 2045)
runs = pa.concat_tables(
    pa.table({
        "year": np.repeat(future, 4).astype("int32"),
        "latitude": np.tile([50.875, 50.875, 51.125, 51.125], len(future)),
        "longitude": np.tile([6.875, 7.125, 6.875, 7.125], len(future)),
        "value": rng.gumbel(50.0, 12.0, 4 * len(future)),       # mm/day
        "model": [model] * (4 * len(future)),
        "member": ["r1i1p1f1"] * (4 * len(future)),
        "scenario": ["ssp245"] * (4 * len(future)),
    })
    for model in ("model-a", "model-b")
)
archive = SourceProvenance(provider="example", dataset="synthetic pooled rx1day",
                           licence="CC0-1.0", attribution="Synthetic data")

pooled_result = (
    ParametricPlan.for_annual_extremes(
        runs, assets,
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
        unit="mm/day", value_column="value_eur", value_unit="EUR",
        provenance=archive, seed=42,
        pool=ModelPool(),                 # the columns that name a run
        scenario="ssp245, two models",    # your name for the rows
    )
    .basis_risk(ImpactNoise(0.3, upper=1.0))
    .covers(step, linear)
    .objective(ExpectedUncovered(), budget=Budget(ratio=0.8))
    .split(Holdout(0.7))                  # design on 2015-2035, test on 2036-2044
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
    .write_parquet("pooled-comparison.parquet")
)
print(setup_text(pooled_result.metadata))
```

**The rows.** 3 models of `n` years give `3n` rows at `1/(3n)` each, in (year,
model, member) order, with models and members in UTF-8 byte order. `period` is
the draw number, `0` to `N − 1`, and `source_kind` stays `"annual"`. Each row's
identity is in `provenance.pool`: one document per run in `runs` (`model`,
`member`, `pathway`, `lineage`, `series` and the `dropped_years` it had no value
for), and one entry per row in `year` and `run`. `select`, `holdout` and
`year_block_kfold` keep the entries of the rows they keep. In a comparison's
metadata, `setup.samples[i].periods` therefore holds draw numbers, the first
and the last, and the calendar years are in `setup.samples[i].pool.years`.

**What a pool is not.** A draw mixes model spread with year-to-year variability.
The rows are not independent years: rows of one run share that model's climate
and rows of one year the pathway's forcing, so the number of rows is not an
effective sample size. Every number from a pool has these two statements next to
it, in the `Pool` line of `setup_text`.

**One pathway, one weight per model.** These raise `SampleError`: runs under
different pathways (`historical` is one, so a pool can't span NEX-GDDP-CMIP6's
historical and SSP experiments); different units; different recipes, where the
runs state one; the same model twice, whatever the member, so every model has
the same weight; fewer than two runs. A table with several models at one
location still raises without `pool=`.

**The same years.** After `years=`, the window of `RecentYears` and `missing=`,
every run must hold the same years, or `SampleError` lists each run's years; a
stated `years=` resolves it. With `missing="drop_year"`, a year without a value
in any run is dropped from every run and listed with the run that lacked it.

**Splits** work on the pool's distinct years, not on its rows. `Holdout` puts
the first `floor(train_fraction · n)` of the `n` years in the design block and
`KFold(k)` divides the `n` years, so `k` is at most `n`. Every run holds out the
same years: the latest ones under `Holdout`, and one block per fold under
`KFold`, the last fold holding out the latest. `FreshDraws` raises `TypeError`,
as on every per-year plan.

**Leakage.** The check compares (run lineage, year) pairs instead of periods,
and a plain per-year sample counts as one run. So a model-year can't be on both
sides, between pools of different runs or between a pool and one of its runs.

**Trend and basis risk.** Each run has its own fit, on the same design years
with the same reference year, the last of the pool's years. `RecentYears(n)`
takes the last `n` years of each run. `NoBasisRisk()` warns with
`ComonotoneIndexWarning`, as for one series: pooling puts no difference between
index and loss. `HazardNoise` and `ImpactNoise` draw one factor per row, in row
order, from the asset's generator.

**The source.** With `pool=`, `source` is one table, file or plan whose rows
carry the columns `ModelPool(model="model", member="member",
pathway="scenario")` names, as NEX-GDDP-CMIP6's batches do, or a sequence of
sources, one per run, as CMIP6's plans are (one model each). A source without
those columns is named by its plan's `explain(format="json")` (`model`, `member`,
`pathway`), and a source that states neither raises `SampleError`. Each run is
located on its own grid, so the matched cell and the grid step are recorded per
run. `provenance` states the archive's licence and attribution once; each run's
own columns are recorded with it. `.samples()` returns one pooled sample per
asset. `cell=` is allowed, as the cell belongs to the asset, not to a run.

**Under `Joint()`** the assets' pools are joined by `OutcomeSample.join`, which
needs the same (year, model, member) rows under one pathway, because draw
numbers alone would not line them up. The dependence is the data's own, the rows
being the same model-years. With `LinearTrend` and a split the plan joins per
fold.

**Refused.** `.index(...)` raises `ValueError` on a pool: an observed station
and a free-running model share no weather, so pairing them by year would invent
a dependence. `pool=` with `StationColumn` raises `ValueError`, as models have no
stations.

**By hand**, the run's series and its names go in a `ModelRun`:

```text
ModelRun(series, *, model, member, pathway)
sample_annual_pool(runs, *, loss_model, basis_risk, asset_id, rng=None,
                   missing="raise", trend=AsRecorded(), scenario="baseline",
                   hazard_name=None)
```

To hold out a block under `LinearTrend`, state the design years with
`LinearTrend(fit_years=...)`, as for one series.

**In the metadata and the text.** Each `setup.samples` entry holds `pool`, with
the runs and their `years` and `rows` but without the row arrays, which are in
the sample's own metadata (`crc.parametric.sample`); it is null on every other
sample, and a document without the key reads as null. `sources` has one hazard
entry per run. A sequence of sources has one `annual_table` record each in
`source_records`, and each entry names its own by `annual_table_sha256`. Before
execution, `plan.explain(format="json")` states the pool's columns in
`setup.samples` and in each entry of `sources`. A `scenario=` label is the name
of the rows: without one, rows under `ssp245` are labelled `"baseline"`, and the
`Source` line states the pathway.

## The function route

The plan's steps are functions you can chain by hand, for a table, a curve or
an index of your own.

### From a fitted curve

The quick start's plan, by hand, for its one asset: `request.asset_curves()`
gives the asset's canonical curve row, `sample_curve` draws simulated years
from it, and fresh draws from another generator are the out-of-sample test.
With the plan's generators for its first asset, the samples are the plan's:

```python
import numpy as np
from crc_sdk.parametric import (LossModel, compare, design, sample_curve,
                                write_comparison)

row = request.asset_curves().to_pylist()[0]   # one row per asset and scenario
metadata = request.dataset.metadata()          # units, tail, semantics, licence
loss_model = LossModel(PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
                       value=2e6, value_unit="EUR")

def draws(seed: int):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return sample_curve(row, metadata=metadata, n_draws=5_000, rng=rng,
                        loss_model=loss_model, basis_risk=HazardNoise(0.3),
                        asset_id="warehouse-a")

design_draws, fresh_draws = draws(42), draws(7)
designed = design(step, design_draws, ExpectedUncovered(), budget=Budget(ratio=0.8))
table = compare([designed, linear], fresh_draws, split="out_of_sample",
                design_sample=design_draws)   # the fixed cover needs its design sample
write_comparison(table, "by-hand.parquet")   # atomic
```

`curve` may also be a crc-framework distribution, for example
`distribution_from_hazard_row(row)`, or `CurveParameters`. A draw is a year
only if the curve describes one year's value: its `probability_semantics`
must be `annual_exceedance` or `annual_value_distribution` ([outcome
samples](#outcome-samples)).

### From per-year values

The per-year plan, by hand, for its one asset. With the plan's generator for
its first asset, the sample is the plan's:

```python
from crc_sdk.parametric import AnnualSeries, holdout, sample_annual

series = AnnualSeries.nearest_cell(annual, longitude=6.96, latitude=50.94,
                                   unit="mm/day", source=provenance)
rng = np.random.default_rng(np.random.SeedSequence(42).spawn(1)[0])
sample = sample_annual(series, loss_model=loss_model,
                       basis_risk=ImpactNoise(0.3, upper=1.0),
                       asset_id="warehouse-a", rng=rng)
design_years, held_out = holdout(sample, train_fraction=0.7)
designed = design(step, design_years, ExpectedUncovered(), budget=Budget(ratio=0.8))
table = compare([designed, linear], held_out, split="out_of_sample",
                design_sample=design_years)
write_comparison(table, "by-hand-annual.parquet")   # atomic
```

`AnnualSeries.from_table(table, where={"station": ...}, unit=...)` reads a
station instead, and `sample_annual(..., index=other_series)` reads the index
at another cell, station or hazard than the loss, as the plan's
[`.index(...)`](#an-index-read-elsewhere) does.

`trend=` and `scenario=` are those of the plan ([Climate
trends](#climate-trends)). By hand, a `LinearTrend` is fitted on every kept
year unless `fit_years=` states others, so to hold out a block, state the
design years:

```python
from crc_sdk.parametric import LinearTrend

fitted = sample_annual(series, loss_model=loss_model,
                       basis_risk=ImpactNoise(0.3, upper=1.0),
                       asset_id="warehouse-a", rng=np.random.default_rng(7),
                       trend=LinearTrend(fit_years=series.years[:28]))
design_years, held_out = holdout(fitted, train_fraction=0.7)
```

`holdout(sample_annual(..., trend=LinearTrend()))`, fitted on all the years, is
refused at evaluation with `LeakageError`, which names `fit_years`: the fit saw
the held-out years.

## Reporting numbers

- Report out-of-sample rows (`split == "out_of_sample"`): held-out years, or
  fresh draws from a curve. In-sample rows measure the fit on the design years
  and show the gap.
- State the setup next to every number: the asset, or which assets a total
  sums and the dependence between them, the basis (one draw is one year), the
  basis-risk model, the pricing and budget, and the split (below).
- A design is one member of a set of optimal designs unless `unique` is true
  ([non-unique optima](#non-unique-optima)).
- A comparison holds for its stated setup. The module makes no claim that one
  objective, payout family, cover or scope performs better in general.
- An interval from [`bootstrap_intervals`](#pooled-folds-and-intervals) says how
  much a number moves with the years drawn. Two rows' intervals are not a test of
  a difference between them.

### Stating the setup

Every number in a comparison rests on a setup: the asset, the basis, the
split, the samples' source and basis-risk model, the pricing and, for a design,
its objective and constraints. `setup_text(result.metadata)` states that setup
in plain text, one statement per line (`Level`, `Split`, `Seeds`, `Sample`,
`Source`, `Location`, `Trend`, `Sample notes`, `Pricing`, `Design`, `Intervals`,
`Metrics`), for a notebook or a report to print next to the numbers. The text
reads only the metadata, so the same metadata always gives the same text, and
statements that every split or sample shares are written once.

- `Split` says, on rows whose design block and evaluation block are draws from
  one curve (fresh draws, or a holdout or k-fold of draws), that out of sample
  there means new draws from the same curve, so these rows test the design's
  sampling error, not the curve. A pooled evaluation has one `Split` line, with
  k, the rows and that each year is evaluated by the design made without it.
- `Seeds` states the plan's seed, which every evaluation's `details` holds
  under `"plan_seed"`, and the fresh draws' seed, which every fresh-draw
  evaluation's holds under `"fresh_draws_seed"`. When merged evaluations
  record different seeds, each `Split` line states its own.
- `Source` names each source record of the samples: for a fitted curve, its
  hazard, the provider and dataset of its dataset's metadata, and the horizon
  and pathway of its canonical row; for per-year values, the provider and
  dataset of the series. A field that a record does not hold is stated as not
  recorded, never guessed. A curve that a plan read also names the canonical
  dataset's path and SHA-256, which `source_records` holds under
  `canonical_file`.
- `Location` states how a per-year series read at the nearest grid cell found
  its asset: by a point or by the asset's H3 cell, the grid step, and each
  asset's matched cell as the centre's (latitude, longitude). There is one line
  per role (hazard, index), way and grid step. Above five assets the line
  counts the assets and cells and gives the range of the centres. The asset's
  point and its distance to the cell are not recorded, so they are not stated;
  a report that prints the line shows each asset's grid cell, as the result
  file does.
- `Trend` states how a per-year sample treats the climate trend ([Climate
  trends](#climate-trends)). Samples that treat it alike share a line, and
  lines that differ name their assets. Above five assets, as under `Joint()`,
  where a `LinearTrend` has one fit per asset and run, the line gives the range
  of the slopes and of the intercepts, and counts the assets; the sample's
  `trend` record holds each series' fit.
- `Intervals`, for a table of `bootstrap_intervals`, states the resamples, the
  level and the unit, what an interval leaves out, that two rows' intervals are
  not a test of a difference between them, and the columns whose level reaches
  the block's worst row.
- `Sample notes` writes a note that counts draws once, with each sample's own
  counts. In the quick start, the draws beyond the curve's fitted support
  read "2527 of 5000 draws in the design sample and 2464 of 5000 in the fresh
  draws are more frequent than the shortest fitted return period (2 years)",
  and the same for the longest. A sample's count of 1 takes the singular ("1
  of 1500 draws in the design sample is rarer than ..."); counts of several
  samples together take the plural. Blocks split from one sample share its
  notes, which are written once.

`setup_text(table)` states the setup of a comparison table's own rows: a table
that `compare` returns, one read back with `pyarrow.parquet.read_table`, or
rows filtered from either, such as the quick start's out-of-sample rows. It
reads the table's `crc.parametric.evaluation` metadata, which rows and columns
taken from the table keep, and keeps the evaluations that the rows name (by
`sample_hash`, `split` and `fold`), their samples and sources, and the designs
the rows name (by `design_id`). So the out-of-sample rows of the quick start
give the text of `result.metadata` without the in-sample `Split` line, and the
rows of one asset give that asset's setup alone. All of a table's rows give the
text of its metadata.

- A table without those four columns, such as a selection of a few columns,
  or without rows, gives the text of the whole comparison, after a first line
  that says so: `Rows: not identified, as the table lacks the column fold;
  this is the setup of the whole comparison, ...`.
- A table without that metadata raises `EvaluationError`, and so does a row
  that names an evaluation or a design the metadata does not describe, as
  after `pyarrow.concat_tables` of two comparisons, which keeps only the first
  table's metadata. Give a concatenation the metadata of
  `EvaluationMetadata.merge`.

Before execution, `setup_text(plan.explain(format="json"))` gives the same
lines, and says what only execution knows (the years of each block, the leakage
check, the dataset's own metadata) is not known yet. In place of the design
records it has one `Design` line per objective and scope the plan states: the
templates, the objective, the pricing, the budget and the limit, and under
`Joint()` the scope and the assets of the total, with the design premium, the
route and the non-uniqueness report not known yet. Without templates nothing is
designed, so there is no such line. Its `Source` line names what the plan
states: on fitted curves the canonical dataset's path and the hazards, horizons
and pathways selected ("Source: the hazard curves of the canonical dataset rx1day.parquet, for
hazard rx1day, horizon 2050 and pathway ssp245; their provider and dataset are
not known yet." in the quick start), and on per-year values the provider and
dataset of the plan's `provenance` and the file, table or source plan the
values are read from. It has no `Location` line, as only execution matches the
cells. `setup_text(plan.explain(format="json")["setup"])`, or
`setup_text(metadata.setup)`, gives the setup alone, without `Source` or
`Design` lines. Out-of-sample rows' `Split` line says how the split was made and
that the leakage check passed.

## Define a cover and evaluate its payouts

```python
from crc_sdk.parametric import Cover, IndexDefinition, StepPayout

index = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
cover = Cover(
    index,
    StepPayout(breakpoints=[60.0, 80.0, 100.0], levels=[1e3, 5e3, 2e4]),
    payout_unit="EUR",
    asset_id="warehouse-a",
)
cover.payout_values([45.0, 60.0, 85.0, 140.0])   # array([0., 1000., 5000., 20000.])
```

- **Index:** `IndexDefinition(hazard_name, unit, *, tail="upper", location=AtAsset(), aggregation, source=None)`.
  `tail="upper"` means larger values are more severe (rainfall, wind);
  `tail="lower"` means smaller values are (a drought index), as a dataset's
  `return_period_tail` says. `location` is `AtAsset()`, `AtPoint(longitude,
  latitude)`, `AtStation(station_id)` or `AtCell(cell_index)` (an H3 cell).
- **Cover:** `Cover(index, payout, *, payout_unit, asset_id=None, label=None)`.
  The payout's `tail` must equal the index's. `label` is a display name only.

## Payout families

Each family pays nothing below its first threshold and more as the index moves
to the severe side. Thresholds are in index units and listed in order of
increasing severity: increasing for `tail="upper"`, decreasing for
`tail="lower"`. A value equal to a threshold belongs to the more severe side.
Amounts are in payout units and non-negative.

| Family | Parameters | Payout at severity `s` |
| --- | --- | --- |
| `LinearPayout(attachment, exhaustion, limit)` | thresholds a < e; amount `limit` | `limit · clip((s − a)/(e − a), 0, 1)` |
| `StepPayout(breakpoints, levels)` | thresholds b_0 < … < b_{n−1}; one level each | `levels[k]` for b_k ≤ s < b_{k+1}; the last level from b_{n−1} on |
| `PiecewiseLinearPayout(knots, levels)` | thresholds c_0 < … < c_{m−1}, m ≥ 2; one level each | the points (c_k, `levels[k]`) joined linearly; the last level from c_{m−1} on |
| `BinaryPayout(trigger, amount)` | threshold t; `amount` | `amount` for s ≥ t |

`StepPayout` and `PiecewiseLinearPayout` take `monotone=True` by default, which
requires non-decreasing levels; `monotone=False` is allowed. Every built-in
family is linear in its amounts (`LinearInAmounts`): `basis(index) @ amounts`
gives the payout, and `amount_constraints(limit)` states the amounts' bounds.

## Fixed and free parameters

Each parameter is a float (a fixed term) or `Free(lower=0.0, upper=None, *,
grid=None)`, a value a design chooses within bounds. A cover with a free
parameter is a template: it has a document and a hash, but
`payout_values` raises `UnresolvedParameterError` until it is resolved.

```python
from crc_sdk.parametric import Free

template = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                 payout_unit="EUR", asset_id="warehouse-a")
template.free_parameters()   # ('levels[0]', 'levels[1]', 'levels[2]')
resolved = template.with_values({"levels[0]": 1e3, "levels[1]": 5e3, "levels[2]": 2e4})
```

- Parameter names are flat: `attachment`, `exhaustion`, `limit`,
  `breakpoints[0]`, `levels[2]`, `knots[1]`, `trigger`, `amount`.
- A free amount needs `lower >= 0`; `upper=None` leaves it unbounded.
- A free threshold needs a finite `upper` and a `grid`: a count of evenly
  spaced values, or the values themselves. A count is stored as its values.
- Invalid parameters raise `CoverDefinitionError` (a `ValueError`); invalid
  index values raise `SampleError`.

## The cover document and its hash

`cover.to_json()` returns the cover document as canonical JSON: the index, the
payout with every parameter, `payout_unit`, `asset_id`, `schema_version` and
`semantics_version`. `cover.definition_hash` is `sha256:` plus the SHA-256 of
that text, in the canonical spelling crc-framework uses for
`ImpactExpression.definition_hash`. The label is not part of the document, so
two covers with the same terms have the same hash; `asset_id` is, so the same
terms on another asset are another contract. `Cover.from_json(text)` reads a
cover back, and covers compare, hash and pickle by their document.

## User-defined payouts

A payout family is any class that implements the `PayoutFunction` protocol
(`kind`, `tail`, `index_names`, `parameters()`, `with_values()`, `evaluate()`,
`to_document()` and `from_document()`). Register it with `@register_payout`
so `Cover.from_json` can read covers that use it; an unregistered `kind`
raises `UnknownPayoutKindError`. A family that is linear in its amounts can
also implement `LinearInAmounts`.

## Outcome samples

Covers are designed and evaluated on an `OutcomeSample`: rows (draws) with a
probability each, and for each row an index value and a loss. One draw is one
year: a cover pays once per draw, on that draw's index, against that draw's
loss. `period` is the year, or the draw number of a simulated sample. `index`
and `loss` have one row per draw and one column per asset, and `hazard` keeps
the value the loss model read when it differs from the index.

A sample can hold several assets. `OutcomeSample.join` puts one-asset samples of
the same years side by side and `OutcomeSample.from_table` reads a table of
several assets; `sample.asset(asset_id)` takes one asset's sample back out, and
keeps its lineage. A sample of several assets states where the dependence
between them comes from: for `join` and `from_table` it is the data's own, the
same years of the data. `join` refuses Monte Carlo samples, because independent
draws line up by draw number only: [`sample_curves`](#several-assets-from-fitted-curves)
draws several assets under a dependence you state. `design` takes one asset's
sample and raises `SampleError` on a sample of several; `design_joint` takes a
sample of several ([joint design](#joint-design-over-several-assets)), and
`evaluate` and `compare` take either.

| Source | Function | One draw is | Probability |
| --- | --- | --- | --- |
| A fitted curve | `sample_curve(curve, *, metadata, n_draws, rng, loss_model, basis_risk, asset_id, scenario="baseline", hazard_name=None)` | a simulated year | `1/n` |
| Fitted curves of several assets | `sample_curves(curves, *, dependence, metadata, n_draws, rngs, loss_models, basis_risk, scenario="baseline", hazard_name=None)` | a simulated year | `1/n` |
| Per-year values | `sample_annual(AnnualSeries, *, loss_model, basis_risk, asset_id, index=None, trend=AsRecorded(), scenario="baseline", hazard_name=None)` | an observed or modelled year | `1/n` |
| Pooled model-years | `sample_annual_pool(runs, *, loss_model, basis_risk, asset_id, trend=AsRecorded(), scenario="baseline", hazard_name=None)` | one run's year | `1/N` |
| A user table | `OutcomeSample.from_table(table, *, period, asset_id, index, loss, semantics="annual", ..., scenario="baseline", hazard_name=None)` | a table row | its `probability` column, or `1/n` |

A curve may also be a **window of horizons**: a sequence of two or more
canonical rows of one hazard, cell, pathway and dataset, which differ in `horizon`
alone. It is an equal-weight mixture of those curves. One stream of `n_draws`
uniforms is drawn, as for one curve, and with `m` members in horizon order, row
`d` reads member `d mod m` at its uniform, so `n_draws` must be a whole multiple
of `m`. Every row is a draw from a curve the dataset holds, and nothing is
computed between two horizons. The provenance holds one `hazard` source per
member and the members in `details.window`, and the `horizon` column names their
centre, `(first + last) // 2`, where no row need be read.

```python
import numpy as np
from crc_sdk.parametric import HazardNoise, LossModel, sample_curve
from crc_sdk.impacts import PiecewiseLinearImpact
from crc_sdk.workflows import distribution_from_hazard_row

# row: one canonical curve row; metadata: its dataset's HazardDatasetMetadata
loss_model = LossModel(PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
                       value=2e6, value_unit="EUR")
sample = sample_curve(distribution_from_hazard_row(row), metadata=metadata,
                      n_draws=10_000, rng=np.random.default_rng(42),
                      loss_model=loss_model, basis_risk=HazardNoise(0.3),
                      asset_id="warehouse-a")
```

**Curves.** Each draw takes `u = rng.random(n_draws)` and `distribution.quantiles(u)`,
crc-framework's own inverse-CDF route. A draw is a year only if the curve
describes one year's value:

- `probability_semantics` must be `annual_exceedance` or
  `annual_value_distribution`; any other value raises
  `ProbabilitySemanticsError`;
- unspecified semantics raise too, unless the caller states one of the two
  with `assume_semantics=`; a stated value that contradicts the declared one
  raises;
- a hurdle curve keeps its atom (a dry year gets the atom's value), and a
  `no_data` row raises instead of becoming zero;
- draws outside the fitted support are counted: rarer than the longest fitted
  return period warns with `ReturnPeriodExtrapolationWarning`; more frequent
  than the shortest is recorded in the provenance notes. A tabulated curve
  raises outside its knots unless `outside_support="extrapolate"` or
  `"clamp"` is stated.

**Per-year values.** `AnnualSeries.from_table(table, *, year, value, where,
unit, source=None)` reads one location of an `annual_extremes()` table (a
station with `where={"station": ...}`), and `AnnualSeries.nearest_cell(table,
*, longitude, latitude, unit, source=None)` the grid cell whose centre lies
within half a grid step of a point. A year without a value raises in
`sample_annual`, unless `missing="drop_year"`, which lists the dropped years.
`trend=` states how the climate trend is treated, by default as recorded
([Climate trends](#climate-trends)).

`source` describes the table in the series' provenance: a name, a
`SourceProvenance`, a mapping such as a plan's `explain(format="json")`, or
the source plan itself, of which only `explain()` is called, so no data is
fetched. An ERA5 or CMIP6 plan's document names no provider or dataset, so
they are named from what it states; an ERA5 document states no terms either,
so the series records crc-sdk's `ERA5_LICENCE` and `ERA5_ATTRIBUTION`, without
the year of retrieval that fitted curves add, which `explain()` does not know.
A mapping's own names and terms win.

The leakage checks link two series through their source. What a source plan's
document says of its cache and its run is left out of that link, and so are
the years it asks for and, where the series' cell or filter names its rows,
the area: each fetch and each window of one source is linked, and two series
that share rows and a source raise. A `source` of your own is kept whole. A
table's `retrieved_at` column is left out too.

A nearest-cell series records the cell's centre and the grid step, not the
point itself: it says which grid cell holds the asset, not where in the cell it
lies. A result file can say more of an asset's place: `cell=H3Cell(r)` records
the asset's H3 cell, so the caller's resolution sets how much is disclosed; a
source plan's `explain()` records its `area` whole; and `.index(...,
location=NearestCell(...))` on a grid finer than or offset from the hazard's
narrows the place further, because a contract names where its index is read.

**Loss model.** `LossModel(impact, *, value_column=None, value=None,
value_unit="1", context=None, name=None)` computes `impact(hazard) × value`.
`impact` is an `ImpactExpression`, a framework `ImpactFunction` (such as
`PiecewiseLinearImpact` or a registry impact) or a vectorised callable, as
`PortfolioEvaluation.impact(...)` takes, and `name` labels it in the provenance.
Without a value the loss is the damage ratio. A value column, and
`ImpactContextColumns` for registry impacts, are read from the asset's row with
`loss_model.for_asset(assets, asset_id)`; a
missing or repeated asset raises. An `ImpactExpression` declares the hazard
unit it is calibrated for (`loss_model.exposure_unit`). Before drawing,
`sample_curve` and `sample_annual` check the curve's `value_unit` or the hazard
series' unit against it and raise `SampleError` when they differ (compared as
`check_exposure` compares units, through the framework's aliases);
`loss_model.check_hazard_unit(unit)`, or `hazard_unit=` in `evaluate`, is the
same check by hand. Other impacts declare no unit and are not checked.

**Basis risk** is a required argument: every sample states how its index and
its loss differ. [Basis-risk models](#basis-risk-models) says what each model
can and can't change.

**Splits.** `holdout(sample, train_fraction=0.7)` designs on the first
`floor(train_fraction · n)` periods and holds out the rest, so a per-year
sample is tested on its latest years. `year_block_kfold(sample, k)` holds out
each of `k` contiguous blocks in turn. On a [pool of
model-years](#pooled-model-years), `n` is its number of distinct years. Monte
Carlo draws are independent, so a fresh sample from another generator is the
plainer out-of-sample test.

**Provenance.** `sample.provenance` (`SampleProvenance`) records the source
(dataset metadata with licence, attribution, version and checksum; the series
and its matched cell or station; or the table's name), the semantics and who
stated them, the loss model (impact type, name, `definition_hash`, value
column and unit, not the value itself), the basis-risk model, the generator's
state before each random step (`provenance.generator()` rebuilds it), what was
counted (draws outside the support, atom draws, dropped years), a per-year
sample's treatment of the trend (`provenance.trend`), a pool's rows
(`provenance.pool`) and notes for the setup text. `sample.content_hash` is `sha256:` plus the SHA-256 of the
canonical long table with its units, semantics and scenario.
`sample.to_table()` returns that long table, with the metadata under the
Parquet key `crc.parametric.sample`, and `OutcomeSample.from_written_table(table)`
reads it back, in memory or from `pyarrow.parquet.read_table`, with its
provenance and lineage; a table whose rows no longer match the recorded
content hash raises `SampleError`.

### Basis-risk models

The basis-risk model is a required argument: every sample states how its index
and its loss differ. The module does not say which model suits an asset.

| Model | Index | Loss | What it can and can't change |
| --- | --- | --- | --- |
| `HazardNoise(sigma)` | the hazard, unchanged | `impact(h) × value`, where the hazard at the asset is `h = index · exp(sigma·Z − sigma²/2)`, one `Z` per year and asset | A year can pay with no loss and a loss can occur with no payout, so the false-negative and false-positive rates respond to `sigma`. The index does not change |
| `ImpactNoise(sigma, *, upper=None)` | the hazard, unchanged | each year's impact times `exp(sigma·Z − sigma²/2)`, capped at `upper` | The size of a loss the index has already fixed. It can't change whether a year pays or whether a loss occurs, so `false_negative_rate_l0` and `false_positive_rate_l0` don't move with `sigma`. With a loss threshold above zero they can, because the factor can carry a loss across the threshold |
| `NoBasisRisk()` | the hazard at the asset | `impact(hazard) × value`; warns with `ComonotoneIndexWarning` | Nothing: the loss is a function of the index, so a cover can track it almost exactly |
| `index=` series in `sample_annual` | another cell, station or hazard | from the hazard series | The dependence between index and loss is the data's own, not a stated factor |

**`HazardNoise`.**

- The cover pays on the index and the loss model reads `h`, which the sample's
  `hazard` holds. `sigma` is an assumption you state, and the module does not
  estimate it.
- It suits a hazard on a ratio scale, where zero means none (rain, wind, depth).
  A negative hazard raises `SampleError`, and a zero stays zero. The module can't
  tell that a temperature in °C is not on such a scale.
- The mean is on the hazard, not on the loss. With a damage function that is
  convex or has a threshold, the mean impact moves with `sigma`. The provenance's
  `details["basis_risk"]` records `hazard_mean_ratio` and `impact_mean_ratio`,
  each the mean with the factor over the mean without it, and the setup text
  gives both.
- Its `sigma` and the `sigma` of `ImpactNoise` act at different places, one on
  the hazard and one on the impact, and are not the same physical quantity.
- It draws at every `sigma`, so a sweep over `sigma` from one generator state
  reads one set of draws. It needs a generator at every `sigma`.
- With `sigma=0` the factor is exactly 1: the index, the loss and `content_hash`
  equal those of `NoBasisRisk()`, `hazard` is `None`, and it warns as
  `NoBasisRisk()` does.
- A separate index series (`index=`, or the plan's `.index(...)`) raises
  `SampleError`, because the hazard at the asset is then data.
- Under `LinearTrend` the factor multiplies the adjusted hazard. In
  `sample_curves` each asset draws its factors from its own generator, so they
  are independent between assets, and a dependence model moves a row's index,
  hazard and loss together. On a sample of several hazards one model serves
  every component, each with factors of its own. A component with no impact
  still draws its factors, so the others' draws don't depend on it, and a
  negative value on it raises as on any other.

**A model of your own** implements the `BasisRisk` protocol (`name`,
`to_document()`, `apply(index, hazard, impact, rng)`). To act before the impact is
computed it also implements `HazardBasisRisk`, which adds
`apply_hazard(index, hazard, rng)` returning the index and the hazard at the
asset, so the model never holds the damage function. It may have both methods.

With one generator per call the order is fixed:

1. the hazard: a curve's uniforms and quantiles, or a series' years with its
   trend;
2. `apply_hazard(index, hazard, rng)`, if the model has it, which gives the index
   and the hazard at the asset;
3. `loss_model.impact_values(hazard at the asset)`, which draws nothing;
4. `apply(index, hazard, impact, rng)`, which gives the index and the impact;
5. the loss, impact × value.

A model without `apply_hazard` skips step 2. `apply_hazard` must return a tuple of
two arrays of the input's shape, finite, or the sampler raises `SampleError`.

### Several assets from fitted curves

A fitted curve describes one place and says nothing about how two places move
together. So a sample of several assets from curves needs a dependence that you
state: `sample_curves` takes it as a required argument with no default, and the
sample records it as an assumption.

```python
import numpy as np
from crc_sdk.parametric import Comonotonic, ImpactNoise, sample_curves

assets = ("warehouse-a", "warehouse-b")
children = np.random.SeedSequence(42).spawn(len(assets))
sample = sample_curves(
    {"warehouse-a": row_a, "warehouse-b": row_b},   # the order sets asset_ids
    dependence=Comonotonic(),
    metadata={asset: metadata for asset in assets},
    n_draws=10_000,
    rngs={a: np.random.default_rng(c) for a, c in zip(assets, children)},
    loss_models={asset: loss_model for asset in assets},
    basis_risk=ImpactNoise(0.3, upper=1.0),         # one model for every asset
)
```

`curves`, `metadata`, `rngs` and `loss_models` are mappings by asset id. Each
asset is drawn exactly as `sample_curve` draws it from that asset's generator:
the uniforms first, then the basis-risk model. Each model only reorders: a row's
hazard, index and loss stay together, no value changes and no further random
number is used. So an asset's rows are the same under every model, and each
model says which assets' draws share a row. Ranks are those of the uniforms, so
a hurdle curve's atom adds no ties. `period` is the row number, 0 to
`n_draws - 1`.

| Model | What shares a row | What it assumes |
| --- | --- | --- |
| `Independent()` | every asset's draws, in draw order | the assets are independent |
| `Comonotonic()` | every asset's draw of rank r; the first asset stays in draw order | the assets' uniforms have rank correlation 1 |
| `EmpiricalCopula(reference)` | for each reference year, the block of each asset's draws at that year's rank there | the reference's ranks at block level, and independence between the assets inside a block |

**`EmpiricalCopula(reference)`** takes the ranks of a stated per-year sample.
`reference` is an `OutcomeSample` of `m` observed or modelled years with equal
probabilities, holding exactly the curves' assets, for example
`OutcomeSample.join` of `sample_annual` samples; a Monte Carlo sample raises
`SampleError`, as it has no dependence of its own to lend. Ranks are read from
its `hazard`, or from its `index` when it has no separate hazard. Only the ranks
are used, so its unit may differ from the curves'. A larger reference value
takes a larger draw; negate a reference that moves the other way. A reference
may be a [pool of model-years](#pooled-model-years): `m` is then its number of
rows, each model-year taking a block of its own, and `first_year` and
`last_year` in its record are the pool's calendar years.

- `n_draws` must be a whole multiple `k·m` of the reference's years. Any other
  count raises `SampleError` and names the two nearest valid counts.
- Each asset's draws, sorted by their uniforms, make `m` consecutive blocks of
  `k`. A reference year of rank r at that asset takes block r, and inside a
  block the draws keep the order in which they were drawn. The rows are in
  reference-year order: the first `k` rows are the first reference year.
- With `k = 1` the sample's ranks are the reference's. With `k > 1` the block of
  every row's draw equals the reference's rank of that row's year.
- Years with equal values at one asset pool their blocks, dealt in draw order,
  `k` each, in year order, so a tie adds no dependence between assets.
- Inside a block the reference says nothing, so the assets are independent
  there. Dependence finer than one year in `m` is assumed, not observed.
- The rows come in reference-year blocks, so `holdout` and `year_block_kfold`
  raise `SampleError` on such a sample and on one asset's sample taken from it.
  The out-of-sample test is a fresh sample: call `sample_curves` again with new
  generators and the same reference. The reference is part of the stated model,
  so the two samples share it.

**The record.** `sample.provenance.dependence` holds the model's document:
`{"kind": "independent"}`, `{"kind": "comonotonic"}`, or for `EmpiricalCopula`
the reference's `content_hash`, `sources`, `licences`, `first_year`, `last_year`,
`m` and `k`. The reference's licences are listed in `provenance.licences` too.
`provenance.notes` start with a sentence that names the model and calls the
dependence an assumption. `sample.asset(asset_id)` gives one asset's sample,
with its own provenance; a reordering is a step of its `selection`.

The dependence model is what you assume about the assets, and the module does
not choose it. The same generators give the same draws under every model, so
the lineage doesn't depend on the model: evaluating out of sample needs other
generators, whichever models the two samples state. On a plan,
`.dependence(model)` states it and `.joint_sample()` returns the sample.

## Evaluate and compare covers

`evaluate(cover, sample, *, split, ...)` returns a one-row table of a cover's
metrics on a sample; `compare(covers, sample, *, split, ...)` returns one row
per cover, with covers as rows and metrics as columns. A cover is a resolved
`Cover` (a template raises `UnresolvedParameterError`) or a design, which
brings its design sample and its premium. The cover's payout unit, index unit
and asset must be the sample's. On a sample of several assets an entry is a
`JointDesignResult`, or a mapping from every asset id to that asset's cover or
design ([the rows](#the-rows-of-a-joint-design)).

```python
from crc_sdk.parametric import (CostOfCapitalLoad, MetricOptions, compare,
                                holdout, to_long, write_comparison)

design_years, held_out = holdout(sample, train_fraction=0.7)
# cover_a and cover_b were chosen on design_years, so name it
table = compare([cover_a, cover_b], held_out, split="out_of_sample",
                design_sample=design_years, pricing=CostOfCapitalLoad(0.06, 0.99),
                options=MetricOptions(betas=(0.9, 0.99), loss_thresholds=(0.0, 1e5)))
write_comparison(table, "comparison.parquet")   # atomic
tidy = to_long(table)                           # one row per cover and metric
```

### Metrics

On an evaluation sample with draw probabilities p, loss L, payout π, the
uncovered loss U = max(L − π, 0), the residual R = L − π, the premium used P and
the net loss N = L − π + P (the owner's cost of the year with the cover):

| Column | Definition | Null when |
| --- | --- | --- |
| `expected_loss`, `expected_payout`, `expected_uncovered` | E[L], E[π], E[U] | — |
| `premium` | P, the premium used (below) | — |
| `coverage_ratio` | E[min(π, L)] / E[L]: the share of the expected loss that payouts cover; payouts above the loss don't count | E[L] = 0 |
| `payout_to_loss_ratio` | E[π] / E[L]: payouts above the loss count too | E[L] = 0 |
| `payout_frequency` | P(π > ε): the probability that the cover pays | — |
| `unidentified_share` | Σ p · 1{B_j(x) ≠ 0 for some j in A}, with B the basis of the resolved cover on this block's index and A the amounts the fill rule set: the block's probability whose payout reads an amount that no design year reached. Any non-zero basis entry counts in full. It is 0 on the design block, and above 0 only on a block that reaches where the design had no year. On a `"total"`, `"package"` or allocated row it is the union over the contracts | A cover given as is; a design on the numpy grid, where no fill rule ran; a `"total"` or `"package"` row with such a contract |
| `tail_coverage_b<β>` | Σ w min(π, L) / Σ w L, with w the CVaR_β weights of L: the share of the loss covered in the worst 1 − β of loss years | Σ w L = 0 |
| `cvar_uncovered_to_loss_b<β>` | CVaR_β(U) / CVaR_β(L): the tail of the uncovered loss against the tail of the uninsured loss | CVaR_β(L) = 0 |
| `false_negative_rate_l<ℓ>` | P(π ≤ ε, L > ℓ) / P(L > ℓ): years with a loss above ℓ and no payout | P(L > ℓ) = 0 |
| `false_positive_rate_l<ℓ>` | P(π > ε, L ≤ ℓ) / P(L ≤ ℓ): years with a payout and a loss of at most ℓ | P(L ≤ ℓ) = 0 |
| `rmse` | sqrt(E[R²]) | — |
| `correlation` | the p-weighted Pearson correlation of L and π | either is constant |
| `var_uncovered_b<β>`, `cvar_uncovered_b<β>` | VaR_β(U), CVaR_β(U) | — |
| `var_net_loss_b<β>`, `cvar_net_loss_b<β>` | VaR_β(N), CVaR_β(N) | — |
| `var_loss_b<β>`, `cvar_loss_b<β>` | VaR_β(L), CVaR_β(L), the same for every cover on a sample | — |
| `loss_rp<T>`, `payout_rp<T>`, `uncovered_rp<T>`, `net_loss_rp<T>` | the value exceeded with probability 1/T: the lower (1 − 1/T)-quantile. T is in years, because a draw is a year | — |

`MetricOptions(betas=(0.9, 0.95, 0.99), return_periods=(2, 5, 10, 25, 50, 100,
250), loss_thresholds=(0.0,), payout_threshold=0.0)` states β, T, ℓ and ε; no
threshold is hidden. ℓ is in loss units; ε is the payout at or below which a
year counts as "no payout". Labels write the number with the point as an
underscore, as crc-sdk's return-period columns do: β = 0.99 gives
`cvar_uncovered_b0_99`, T = 100 gives `loss_rp100`, ℓ = 0 gives `l0`. Repeated
values raise `EvaluationError`, a `ValueError`.

**VaR and CVaR.** `VaR_β(X)` is the smallest x with Σ_{X ≤ x} p ≥ β, where the
comparison allows 1e-10 for the rounding of the summed probabilities, so a β
exactly on a step of the distribution gives the lower value on every platform.
`CVaR_β(X) = VaR_β(X) + Σ p (X − VaR_β(X))⁺ / (1 − β)`, the mean of the worst
1 − β of the probability; it does not depend on which value the VaR picks.

**crc-framework's CVaR is a different quantity.** Its `compute_risk` gives VaR
and CVaR over the 2ⁿ branches of independent binary factor outcomes, each with
a downside probability and impact. This module's VaR and CVaR are over the
years of an outcome sample, for one continuous quantity of one cover
(uncovered loss, net loss or loss). The random variable, the outcomes and the
dependence assumptions all differ. The two never share a column and are never
compared.

### Pricing and the premium used

`PurePremium()` prices P = E[π]. `CostOfCapitalLoad(rate, q)` prices
P = (1 − rate) · E[π] + rate · CVaR_q(π), with rate in [0, 1] and q in (0, 1).
The premium a row uses is, in this order:

- the `premium=` argument of `evaluate` (`premium_basis` `"stated"`);
- for a design, the design premium: the contract price, priced on the design
  sample (`"design"`), so out-of-sample rows use the price agreed at design;
- otherwise the premium priced on the evaluation sample with `pricing`
  (`"evaluation"`).

### In and out of sample

In sample means the design sample; out of sample means held-out year blocks
(`holdout`, `year_block_kfold`), fresh Monte Carlo draws, or another sample
that shares no year with the design sample. Every row says which in `split`
(`"in_sample"` or `"out_of_sample"`) and `fold` (0 for a holdout and fresh
draws; the fold for k-fold, which must match the sample's own split).

- An out-of-sample evaluation needs the design sample: a design brings its
  own; for a cover given as is, pass `design_sample=` with the sample its
  terms were chosen on. Without one it raises `SampleError`. A design sample
  that shares a period with the evaluation sample from the same source raises
  `LeakageError`; on a [pool](#pooled-model-years), a (run, year) pair
  does. So does a linear trend of either sample fitted on a year the
  evaluation sample holds from the same source; the error names `fit_years`.
  A trend or a `scenario` label does not make one source's rows another's.
- In sample, a known design sample must be the evaluation sample itself.
- The metadata records how the split was made: `holdout` (with the train
  fraction), `kfold` (with k), `fresh_draws` (Monte Carlo samples from
  different generator states), `separate_sample` or `disjoint_periods`.

Published numbers come from out-of-sample rows; in-sample rows show the gap
between the two.

### The comparison table

| Column | Type | Meaning |
| --- | --- | --- |
| `cover_label` | string | The cover's label, or the first 12 hex digits of its hash. On `"total"` and allocated rows, the covers' label if they all share one, else the first 12 hex digits of `cover_hash` |
| `cover_hash` | string | `Cover.definition_hash`. On `"total"` and allocated rows, the hash of the set of covers, one per asset |
| `design_id` | string, nullable | The design's id; null for a cover given as is, and on the `"total"` row of a set of covers |
| `asset_id` | string, nullable | The asset; null exactly on `"total"` rows |
| `level` | string | `"asset"`: an asset's own cover. `"total"`: the sum over the assets of one owner, draw by draw. `"asset_by_value"`, `"asset_by_loss"`: an asset's share of a joint design's total payout, under the rule in the name. `"package"`: the sum of one asset's separate contracts (`CoverPackage`), draw by draw, against the combined loss, with the sum of their premiums; `cover_hash` is the package's hash and `design_id` is null |
| `scope` | string | `"individual"`, each cover standing on its own, or `"joint"`, one objective and one budget on the owner's total |
| `scenario` | string | The sample's scenario, `"baseline"` by default |
| `design_scenario` | string, nullable | The `scenario` of the design's own design sample. Null for a cover given as is, and on a `"total"` or `"package"` row whose designs don't share one |
| `pathway`, `horizon`, `product` | string, int32, string; nullable | What the sample's source records state, never its label, and null where they state none: the canonical row's `pathway` and `horizon` and the dataset's provider; for a window of horizons, the members' pathway and the centre of their horizons; for per-year rows, the pathway of the series or the pool, the centre `(first + last) // 2` of its years, and the provider of its source; for a table, its source's `pathway` and `provider`. On a sample of several hazards `product` holds every provider, joined by `+` |
| `split` | string | `"in_sample"` or `"out_of_sample"` |
| `fold` | int32 | 0 for a holdout and fresh draws; the fold for k-fold. Null on a pooled row of `compare_pooled`, and only there |
| `sample_hash` | string | The evaluation sample's `content_hash` |
| `n_rows` | int32 | The number of rows of the evaluation sample |
| `premium_basis` | string, nullable | `"stated"`, `"design"` or `"evaluation"`; null exactly on allocated rows |
| `unique`, `ties`, `degenerate_tail` | bool, int32, string; nullable | The design's non-uniqueness report; null for a cover given as is |
| `components` | string, nullable | What the row's cover reads of a sample with index components, joined by `+`; null on a sample of one hazard |
| `design_budget` | float64, nullable | The design's budget bound in payout units. On a `"package"` row, or the `"total"` row of a set of covers, the sum of the covers' bounds, null if any has none. Null for a cover given as is |
| one column per metric | float64, nullable | As defined above; null where undefined |

A comparison on one asset has `level` `"asset"` and `scope` `"individual"`
on every row, apart from a package's `"package"` row. [Joint design](#the-rows-of-a-joint-design) says what the other
values hold. The key (`cover_hash`, `design_id`, `asset_id`, `level`,
`scenario`, `design_scenario`, `split`, `fold`) is unique; `compare` raises
`EvaluationError` for the same cover twice. A table written without
`design_scenario`, `pathway`, `horizon`, `product` or `n_rows` reads as null
there. A sample with a `scenario=` label of yours raises `ScenarioError` if its
records state a pathway that `crc_sdk.pathways` doesn't know
(`register_pathway` adds a private one), or if its sources disagree on
`pathway`, `horizon` or `product`. On an unlabelled sample the pathway stands as
spelled and a field the sources disagree on is null. Invalid evaluation arguments, metric options and comparison metadata
raise `EvaluationError`, a `ValueError`; arguments of the wrong type raise
`TypeError`.
`to_long(table)` gives the tidy form: the identity columns, then `column`,
`metric`, `parameter`, `parameter_value`, `unit` and `value`.

**Metadata.** The table's schema metadata holds one JSON document under the
key `crc.parametric.evaluation`, validated by `EvaluationMetadata` and written
with sorted keys: the crc-sdk, crc-framework and numpy versions, the platform
and the time; the setup (basis, metric options, every sample with its source,
semantics, basis-risk and loss models and dependence, and every evaluation with
its split, pricing, levels, scopes and allocation); every metric column's metric, parameter, unit and definition; every
cover's document; the designs' records, each with its cover's label as
evaluated (`cover_label`), so two designs that resolve to the same cover keep
their own labels; the samples' sources, licences and generator states; and,
in a plan's comparison, what the plan read, once (`source_records`). Asset
values are never written. A comparison is derived
data under its inputs' licences, which the metadata lists.

`write_comparison(table, path, *, metadata=None)` checks that the metadata
describes the table, then writes one local Parquet file through a temporary
sibling and `os.replace`, so a failed write leaves no partial file. To write
rows of several calls (in and out of sample, several folds) to one file,
concatenate the tables with `pyarrow.concat_tables` and pass
`metadata=EvaluationMetadata.merge(*(EvaluationMetadata.from_table(t) for t in
tables))`.

### Pooled folds and intervals

```python
from crc_sdk.parametric import bootstrap_intervals, compare_pooled

folds = year_block_kfold(sample, 5)
designs = [design(template, train, objective, budget=budget) for train, _ in folds]
table = compare_pooled([designs], folds)
intervals = bootstrap_intervals([designs], folds, rng=numpy.random.default_rng(7))
```

`compare_pooled` evaluates every held-out year with the design made without it,
and computes each metric once over all those years. The held-out blocks of a
k-fold split partition the years, so every year is evaluated once, each row with
probability 1/N and its payout from its own fold's design. A pooled row is not
the mean of the folds' numbers: the mean of k CVaRs is the CVaR of no
distribution.

Each entry is a sequence of k of what `compare` takes that hold designs, in fold
order (designs, `JointDesignResult`s, mappings of designs or packages of
designs), or one resolved `Cover` or mapping of covers, which applies on every
fold. The rows have the levels `compare` gives them and these values:

| Column | On a pooled row |
| --- | --- |
| `split`, `fold` | `"out_of_sample"`, and null |
| `cover_hash` | The `sha256:` hash of the canonical JSON of the per-fold cover hashes, in fold order. For a cover given as is, its own hash |
| `cover_label` | The covers' label if they share one, else the first 12 hex digits of the hash |
| `design_id` | The same hash of the design ids; null where `compare`'s is |
| `sample_hash` | The same hash of the held-out blocks' `content_hash` |
| `premium`, `premium_basis` | The probability-weighted mean of each row's premium. Each year pays the contract in force for it, that fold's design premium (`"design"`). A cover given as is has one contract, priced once on the pooled rows (`"evaluation"`) |
| The net-loss columns | L_d - payout_d + P_d, with P_d the premium of row d's own fold |
| `unique` | True if every fold's is true, false if any fold's is false, otherwise null |
| `ties`, `degenerate_tail` | The sum of the folds', and the folds' flag if they all agree, else `"mixed"` |
| `unidentified_share` | Each row by its own fold's design |
| `design_budget` | The mean over the rows of each fold's bound, null if any has none |
| `n_rows` | The sum of the blocks' rows |
| `design_scenario`, `pathway`, `horizon`, `product` | The blocks' one value |

`compare_pooled` raises `EvaluationError` for pairs that aren't the k folds of
one split in fold order, held-out blocks that repeat or miss a year, a design
whose design sample isn't its fold's design block, a block with unequal
probabilities (which only a stated `probability` column can give), and blocks of
different `scenario` labels. A holdout and fresh draws have one block, so they
have nothing to pool.

`bootstrap_intervals(entries, on, *, rng, n_bootstrap=1000, ci_level=0.9, ...)`
redraws the evaluated years with replacement and recomputes each metric with the
designs held fixed. `on` is one block, read as `compare` reads it and with a
required `split`, or the folds, read as `compare_pooled` reads them. An interval
is the middle `ci_level` of those values, the percentile interval
(`numpy.quantile`, linear interpolation). It describes how much a number moves
with the years drawn. It leaves out uncertainty in the hazard data, the impact
function and the design, and on a curve block it is the Monte Carlo error of the
draws and says nothing about the curve. Two rows' intervals are not a test of a
difference between them.

- **The unit** is a year with every row of that year: one row of a series or a
  curve sample, and every run's row of one calendar year in a pool. Rows of
  several assets or hazards in a year stay together. An `EmpiricalCopula` block
  is resampled by row, because its reference is part of the stated model.
- **Replicates.** One call `rng.integers(0, Y, size=(n_bootstrap, Y))` draws the
  block's Y years. A replicate reweights the rows by how often their year was
  drawn, which gives the metrics of the concatenated years. One set of replicates
  serves every entry and every metric of a call. Nothing is solved again.
- **The premium.** A design's premium is a fixed price, so on one block its
  interval is the value itself. On pooled folds, the mean of the years' premiums
  moves with the years drawn. A cover priced on the block is priced again on each
  replicate. `premiums=` maps a `cover_label` to a stated premium, on one block.
- **The generator** is a required keyword, and no global state is read. Its state
  at entry is in the metadata's `rng`, step `"bootstrap"`, so the table can be
  rebuilt.

The result is a separate long table with `to_long`'s columns, one row per wide
row and metric column, and these. `value` is the wide table's number: resampling
never changes a point estimate.

| Column | Type | Meaning |
| --- | --- | --- |
| `ci_low`, `ci_high` | float64, nullable | The bounds. Null when the metric is undefined in a replicate, or null in the wide table |
| `ci_level`, `n_bootstrap` | float64, int32 | As stated |
| `n_undefined` | int32 | The replicates in which the metric is undefined, such as those with no expected loss |
| `short_block` | bool, nullable | True when the column's level reaches the block's worst row: (1 - beta) n <= 1 for a beta column, T > n for a return-period column, with n the rows the metric reads. Null for a column with no level |

The bounds of a `short_block` row are computed like any other. They cannot reach
beyond the block's largest rows, so they understate how far the number could
move, and the setup text names those columns. The table holds the metadata under
the same key as a comparison's, with `setup.intervals`, and `write_comparison`
writes it. Neither function ranks anything.

## Design a cover

`design(template, sample, objective, *, pricing=PurePremium(), budget=None,
limit=None, options=DesignOptions())` chooses a template's `Free` parameters
on a design sample and returns a `DesignResult`: the resolved cover and its
design record. The template must be written for the sample's asset (or for
none) and pay in the sample's loss unit on an index in its index unit.

```python
from crc_sdk.parametric import (Budget, CostOfCapitalLoad, DesignOptions, Free,
                                StepPayout, WeightedCVaR, design, evaluate, holdout)

design_years, held_out = holdout(sample, train_fraction=0.7)
template = Cover(index, StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                 payout_unit="EUR", asset_id="warehouse-a")
result = design(template, design_years, WeightedCVaR(alpha=0.9, beta=0.95),
                pricing=CostOfCapitalLoad(0.06, 0.99), budget=Budget(ratio=0.8),
                options=DesignOptions(ranges=True))
result.cover, result.parameters, result.objective_value, result.premium
evaluate(result, held_out, split="out_of_sample")   # priced at the design premium
```

The design needs the optional extra `crc-sdk[parametric]` (the HiGHS solver,
`highspy`) on its linear and quadratic routes; without it they raise
`SolverUnavailableError`, an `ImportError` that names the extra. Importing
`crc_sdk.parametric` loads no solver.

### Objectives

On the design sample, with probabilities p, loss L, payout π and the uncovered
loss U = max(L − π, 0):

| Objective | Minimises | Solved as |
| --- | --- | --- |
| `MSE()` | Σ p (L − π)²: paying more than the loss counts as much as paying less | a QP in the amounts |
| `ExpectedUncovered()` | E[U] = Σ p U: paying more than the loss earns nothing | an LP |
| `WeightedCVaR(alpha, beta)` | α · CVaR_β(U) + (1 − α) · E[U], α ∈ [0, 1], β ∈ (0, 1) | an LP, the CVaR in the Rockafellar–Uryasev form min_z z + Σ p (U − z)⁺ / (1 − β) |
| `UserObjective(function, *, name, parameters=None)` | `function(loss, payout, probability)`, a float | a grid search, evaluated in numpy |

The objectives weigh the years differently: MSE penalises payouts that miss
the loss in either direction, E[U] counts every year's shortfall alike, and
the weighted CVaR adds weight on the worst 1 − β of the shortfall. Which suits
a cover depends on what its owner needs; none is better in general. A user
objective's `name` and `parameters` go into the record and the design id; its
function does not, so give each function its own name.

### Pricing and constraints

- The premium is priced on the design sample with `pricing`: `PurePremium()`
  or `CostOfCapitalLoad(rate, q)`. The cost-of-capital premium is convex in
  the payout, so the design stays an LP or a QP; its CVaR enters in the
  Rockafellar–Uryasev form too.
- `Budget(ratio=k)` means premium ≤ k · E[L] on the design sample;
  `Budget(amount=B)` means premium ≤ B, in payout units. Exactly one of the two.
- `Limit(amount)`: no payout exceeds `amount`. For the built-in families it
  bounds every amount (their payout is at most their largest amount).
- The payout family's own constraints hold: amounts ≥ 0, non-decreasing levels
  under `monotone=True`, and every `Free` parameter's bounds.
- E[U] and the weighted CVaR need a budget or bounded amounts (a `Free` upper
  bound or a `Limit`): without either, any payout at least as large as the loss
  is optimal, and `design` raises `DesignError` before solving. MSE needs
  neither.

### Routes

| Template | Objective | Route |
| --- | --- | --- |
| amounts free, thresholds fixed (any built-in family, or a `LinearInAmounts` user family) | E[U], weighted CVaR | one LP (`route="lp"`) |
| the same | MSE | one QP in the amounts (`route="qp"`) |
| free thresholds on grids (attachment, exhaustion, breakpoints, knots, trigger) | any built-in | a grid over the thresholds, with the LP or QP at each point (`route="grid"`) |
| a `UserObjective`, or a payout not linear in its amounts | any | a grid over every free parameter, evaluated in numpy; candidates that break the budget or the limit are dropped |

- **The grid.** Every parameter the grid searches needs `Free(..., grid=...)`.
  The candidates are the Cartesian product of the grids, in lexicographic
  order; unordered thresholds and non-monotone levels are skipped.
  `DesignOptions(max_grid_points=10_000)` caps the product; a larger grid
  raises `DesignError`. On a finite sample the objective is piecewise constant
  or piecewise smooth in the thresholds, so a stated grid is used rather than
  gradients. `DesignResult.profile` is a `pyarrow.Table` with one row per
  candidate: its values, `objective` and `status` (`"optimal"` or `"feasible"`,
  `"infeasible"`, `"invalid"`, `"over_budget"`, `"over_limit"`).
- **Scaling.** Losses and payouts are divided by the largest loss of a draw (1
  if that is at most 1e-6) before solving; results are in payout units.
- **Solver settings.** HiGHS's simplex for the LPs (a vertex solution), its QP
  solver with a Hessian regularisation of 1e-12 for the QPs. The record holds
  the solver's name and version, the options, the status, the iterations and
  the solver time.

### Checks and errors

After every solve numpy recomputes the payout, the objective, the premium, the
budget, the limit and the family's constraints from the returned values, with
the relative tolerance `DesignOptions.feasibility_tolerance` (1e-7, relative to
the larger of the bound and the scale). Nothing is returned that fails them.

| Error | When |
| --- | --- |
| `InfeasibleDesignError` | no payout meets the constraints, including a fixed amount or a lower bound above the limit, and a grid without a feasible candidate |
| `UnboundedDesignError` | the solver reports an unbounded objective |
| `DesignError` | any other solver status, a failed check, a missing grid, a grid above the cap, E[U] or weighted CVaR without a budget or bounds |
| `SolverUnavailableError` | the LP or QP route without `crc-sdk[parametric]` |

The design errors are `RuntimeError`s and carry `asset_id`, `status` and
`solver_message`. Free amounts below 1e-9 of the scale are set to 0 and listed
in `DesignResult.zeroed`, so a rounding residue does not count as a payout.

### Non-unique optima

A design sample often leaves part of a design open, and the result says so
rather than presenting one solver's answer as the design.

- **Unidentified amounts.** An amount whose payout band no design year reaches
  changes nothing on the sample: not the objective, the premium or the budget.
  `DesignResult.unidentified` lists them, and `DesignOptions(fill="lowest")`,
  the default, sets each to the smallest value the constraints allow (under
  `monotone`, the previous level; else its lower bound); `fill="highest"` takes
  the largest, and raises without a bound above. Both keep the design exactly
  optimal, but out of sample a year can fall into such a band, and then the
  rule sets its payout.
- **The LP route.** With `DesignOptions(ranges=True)` every identified amount
  is ranged over the near-optimal designs, those within ε · max(1, |optimum|)
  of the optimum in scaled units, at ε = 1e-9 and ε = 1e-11 (four more LPs per
  amount, at HiGHS feasibility tolerances of 1e-10). The ranging LPs share one
  HiGHS instance, each starting from the basis of the one before it and the
  first from the design LP's. An amount is pinned when
  its narrow range is at most half its wide one, or narrower than 1e-7 of the
  scale; `unique` is true when every identified amount is pinned, and
  `DesignResult.ranges` gives the wide ranges in payout units. At a unique
  vertex a range shrinks with ε; on a set of optimal designs it keeps the set's
  width. Without `ranges=True`, `unique` is `None` on the LP route.
- **The QP route.** `unique` is true when H = Bᵀ diag(p) B, over the identified
  amounts, is positive definite (smallest eigenvalue above 1e-10 of the
  largest).
- **The grid.** `ties` counts the other candidates whose objective lies within
  `feasibility_tolerance` · max(1, |best|) of the best (in scaled units; in the
  objective's own units for a user objective); the first in lexicographic order
  wins. `ties` is 0 on the LP and QP routes. With ties, `unique` is false;
  otherwise it is the LP's or QP's report at the chosen candidate, and `None`
  for a grid evaluated in numpy.

**Degenerate tails.** For a weighted CVaR the result records `p_zero_loss` =
P(L = 0), `tail_outcomes` = (1 − β) / p_min, and `degenerate_tail`:
`"expected_uncovered"` when β ≤ `p_zero_loss` (the objective is then a multiple
of E[U]), `"single_outcome"` when the tail holds at most one draw (a minimax
over years), otherwise `None`. They are recorded, not raised.

### The design result

`DesignResult` holds the resolved `cover` and its `parameters` (the designed
value of each free parameter); `template_hash`; `design_id`, `sha256:` plus the
SHA-256 of the canonical JSON of the template hash, the objective, the pricing,
the budget, the limit, the options and the design sample's `content_hash`
(the solver and the time are in the record, not in the id); the
`objective_value`, the design `premium`, `expected_loss`, `budget_bound` and
`budget_use`; the `route`; the solver's identity, status and work; the
non-uniqueness report and the degenerate-tail flags; and the `design_sample`.

`evaluate` and `compare` take a `DesignResult` in place of a cover: its rows
use the design premium (`premium_basis` `"design"`), carry its `design_id`,
`unique`, `ties` and `degenerate_tail`, and its `to_record()` goes into the
metadata's `designs`, with the solver in `solvers`. In-sample rows of a design
measure its fit on the years it was chosen on; out-of-sample rows, on held-out
years or fresh draws, are the ones to report.

## Joint design over several assets

A risk owner with several sites may ask for one objective on the total: what
the owner loses and is paid across the sites in the same year. A joint design
states that objective. It is a method, and this page describes what it does
and what it optimises. It makes no statement about how a joint design does
against covers designed asset by asset.

### What a joint design optimises

`design_joint(templates, sample, objective, *, pricing=PurePremium(),
budget=None, limit=None, options=DesignOptions())` takes one template per asset
of a sample of several assets. On each draw d, the owner's total loss is
L_d = Σ_j L_{d,j} and the total payout π_d = Σ_j π_{d,j}.

- **The objective and the premium** of [Design a cover](#design-a-cover) are
  taken on L_d and π_d. `Budget(ratio=κ)` means P ≤ κ·E[L] on the total, and
  `Budget(amount=B)` means P ≤ B on the total.
- **Each asset keeps its own cover**, on its own index, and each cover pays by
  its own terms. `limit=` is each cover's own limit, because a limit is a term
  of one contract.
- **It is solved** as the LP or QP of `design`, on the block basis
  [B_1 | … | B_m], with the summed loss.
- **Thresholds are fixed.** Only amounts may be `Free`, so a joint design has no
  grid route: a free threshold, a `UserObjective` or a payout without
  `LinearInAmounts` raises `DesignError`. To look at other thresholds, run the
  design once per choice.
- **The solver's split between the assets is often not identified.** The covers
  enter the objective through their sum, so several splits of one total payout
  can be optimal. Unless `unique` is true, a design's per-asset amounts are one
  member of an optimal set, and `DesignOptions(ranges=True)` reports their
  ranges on the LP route ([non-unique optima](#non-unique-optima)).

`JointDesignResult` holds the record of a design, with these differences:
`covers` (asset id to resolved `Cover`) replaces `cover`, `template_hashes`
replaces `template_hash`, `covers_hash` identifies the set, and `parameters`,
`unidentified`, `zeroed` and `ranges` are keyed by asset id. `premium`,
`objective_value` and `expected_loss` are on the total.

### On the plan

`.scopes(Joint())` runs the same design from a plan: each template is designed
once per objective and block, for one objective and one budget on the total. It
needs at least two assets, and `.objective(...)` as for every template. Here two
sites on the per-year table of the [next step](#next-step-per-year-values):

```python
from crc_sdk.parametric import Joint

book = pa.table({"asset_id": ["warehouse-a", "warehouse-b"],
                 "longitude": [6.96, 7.10], "latitude": [50.94, 51.10],
                 "value_eur": [2e6, 1.5e6]})
joint_plan = (
    ParametricPlan.for_annual_extremes(
        annual, book,
        impact=PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),   # damage ratio
        unit="mm/day", value_column="value_eur", value_unit="EUR",
        provenance=provenance, seed=42,
    )
    .basis_risk(ImpactNoise(0.3, upper=1.0))
    .covers(step)                              # one cover per asset, on its own index
    .objective(WeightedCVaR(alpha=0.9, beta=0.9), budget=Budget(ratio=0.8))
    .split(Holdout(0.7))
    .metrics(MetricOptions(betas=(0.9,), return_periods=(10,)))
    .scopes(Joint())
    .allocation("value")                       # optional: the rows per asset, below
)
print(setup_text(joint_plan.explain(format="json")))   # the setup; no I/O
joint_result = joint_plan.write_parquet("joint-comparison.parquet")
```

- **The sample.** On per-year values, `OutcomeSample.join` of each asset's
  sample; the dependence is the data's own, and `.dependence(...)` raises. The
  assets must hold the same years: different windows raise, and a stated
  `years=` resolves them. On fitted curves, `sample_curves` under the model that
  `.dependence(...)` must state, and without one `explain()` and every execution
  point raise `ValueError`:

  ```text
  ParametricPlan.for_evaluation(request, value_column="value_eur", value_unit="EUR")
      .monte_carlo(5_000, seed=42)
      .scopes(Joint())
      .dependence(Comonotonic())          # Independent(), EmpiricalCopula(reference)
      .split(FreshDraws(5_000, seed=7))   # the test of an EmpiricalCopula
  ```

  `.joint_sample()` returns the sample and `.samples()` each asset's view of it.
  `EmpiricalCopula` takes `FreshDraws`: `Holdout` and `KFold` raise `SampleError`,
  as a block of its rows is a set of reference years.
- **The covers.** A cover must be written for no asset, as it applies to every
  asset (the plan raises otherwise); `design_joint` takes different terms per
  asset. A resolved cover is evaluated as one contract per asset, with
  `scope` `"individual"`.
- **A climate trend.** Under `LinearTrend` with `Holdout` or `KFold`, each
  asset's sample is fitted on the fold's design years and the assets are
  joined fold by fold, so a joint design sees no held-out year. `.joint_sample()`
  holds all the years, fitted on all of them.
- **An index read elsewhere.** `.index(...)` composes with `Joint()`: each
  asset's cover is bound to the location of its own index series.
- **`.scopes`** may hold both scopes, and the `scope` column keeps the rows of
  each scope apart. This page runs `Joint()` alone.

The designs are in `joint_result.designs`, one `JointDesignResult` per template,
objective and block, and `setup_text(joint_result.metadata)` states the setup of
the rows: which assets a total sums, the dependence, the allocation.

### The function route

The plan's steps by hand, for the same two sites. With the plan's generators,
the sample and the designs are the plan's:

```python
from crc_sdk.parametric import (AnnualSeries, LossModel, OutcomeSample, compare,
                                design_joint, holdout, sample_annual)

sites = [("warehouse-a", 6.96, 50.94, 2e6), ("warehouse-b", 7.10, 51.10, 1.5e6)]
children = np.random.SeedSequence(42).spawn(len(sites))
joined = OutcomeSample.join(
    sample_annual(
        AnnualSeries.nearest_cell(annual, longitude=lon, latitude=lat,
                                  unit="mm/day", source=provenance),
        loss_model=LossModel(PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5]),
                             value=value, value_unit="EUR"),
        basis_risk=ImpactNoise(0.3, upper=1.0),
        asset_id=asset,
        rng=np.random.default_rng(child),
    )
    for (asset, lon, lat, value), child in zip(sites, children)
)
joint_years, joint_held_out = holdout(joined, train_fraction=0.7)
joint_design = design_joint([step, step], joint_years,
                            WeightedCVaR(alpha=0.9, beta=0.9),
                            budget=Budget(ratio=0.8))
joint_table = compare([joint_design], joint_held_out, split="out_of_sample",
                      allocation="value",
                      values={"warehouse-a": 2e6, "warehouse-b": 1.5e6})
```

### The rows of a joint design

`evaluate` and `compare` take a sample of several assets and, as each entry,
a `JointDesignResult` or a mapping from every asset id to that asset's cover or
design, which are all covers or all designs. `allocation=` and `values=` ask for
a joint design's per-asset rows.

| Entry | Rows |
| --- | --- |
| A `JointDesignResult` | One `"total"` row, `scope="joint"`, and one allocated row per asset when `allocation` is set. No `"asset"` row, because the split of the total payout across the assets is the solver's choice |
| A set of individual covers | One `"asset"` row per asset and one `"total"` row, all `scope="individual"` |

- **`"total"` rows** hold every metric of [the table](#metrics) on L_d and π_d.
  A joint design's premium is its design premium, the price of one contract on
  the total. A set of individual covers pays one premium per contract, so its
  total's premium is the sum of the asset rows' premiums, and its `design_id`,
  `unique`, `ties` and `degenerate_tail` are null.
- **Allocated rows** (`level` `"asset_by_value"` or `"asset_by_loss"`) share a
  joint design's total payout out over its assets: π_{d,i} = π_d · v_i / Σ_k v_k
  by value, and π_{d,i} = π_d · L_{d,i} / L_d by loss, by value in a draw without
  loss. `premium` and the net-loss columns are null, as a pricing prices the
  total payout and not a part of it. `expected_loss` and `expected_payout` add
  up to the total row's under both rules, and `expected_uncovered` under `"loss"`
  only; the other metrics don't add up.
- **A rule is an accounting statement** about the total payout, not the payout
  of an asset's cover. The rule is in `level`, so no row can be read as the
  solver's split.
- **The values.** `values=` maps every asset id to its value in the loss unit: a
  sample holds no asset values. On a plan `.allocation(rule)` reads the values
  its loss model read. Only their hash is written.

Report the out-of-sample rows, each with its setup, as for any comparison. The
`scope` column keeps the rows of each scope apart and makes no statement about
them.

## What this version does not do

- **Clusters of assets, free thresholds in a joint design.** A joint design
  covers all the assets of its sample, as one scope, with fixed thresholds
  only. A `Limit` is each cover's own: a limit on the total isn't offered. By
  default, each asset is designed and evaluated on its own sample and a plan
  builds one sample per asset, so it assumes no dependence between them.
- **Different terms per asset on the plan.** Under `Joint()` a plan applies each
  cover to every asset. `design_joint` takes one template per asset.
- **A `GaussianCopula` between curve samples.** The dependence models are
  `Independent()`, `Comonotonic()` and `EmpiricalCopula(reference)`.
- **Several hazards in one index.** An index is one hazard at one location.
- **Several scenarios at once.** A per-year plan names one `scenario` for its
  rows, a sample on fitted curves is `"baseline"`, and a plan on fitted curves
  samples one hazard, horizon and pathway per asset.
- **Several members of one model in a pool.** A model appears once, whatever
  its member. **An index read elsewhere on a pool**, even from the same run at
  another cell, is refused.
- **Several events within a year.** One draw is one year.
- **Remote requests in the plan.** `for_evaluation` takes a request on a local
  canonical file; materialize a remote source first (its `materialize(path)`
  returns the `HazardDataset`).
- **An index read elsewhere on fitted curves.** A plan on fitted curves reads
  the index at each asset: an index elsewhere needs a joint distribution of two
  curves, which the curves do not give. On per-year values,
  [`.index(...)`](#an-index-read-elsewhere) reads it elsewhere.
- **Pooled folds and intervals on the plan.** A plan's `KFold` writes rows per
  fold; `compare_pooled` and `bootstrap_intervals` are functions. An interval on
  the difference between two rows is not offered.
