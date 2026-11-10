"""Parametric insurance covers on Climate Risk Commons hazards.

A `Cover` pays on an index, an `IndexDefinition`, through a payout function:
`LinearPayout`, `StepPayout`, `PiecewiseLinearPayout`, `BinaryPayout`, or a
user family that implements `PayoutFunction` and is registered with
`register_payout`. Each payout parameter is a float or `Free`, a value a
design chooses within bounds. Covers round-trip through canonical JSON and
carry a `definition_hash`, as `ImpactExpression` does. A cover on several
hazards pays on a `CompositeIndex`, through a family with several
`index_names`: `SumOf` and `MaxOf` combine one-index payouts, `StepGrid` is one
table over two indices, or a user family.

Covers are designed and evaluated on an `OutcomeSample`, one row per year
with an index value and a loss: `sample_curve` draws simulated years from a
fitted curve, `sample_annual` reads the years of an `AnnualSeries`, and
`OutcomeSample.from_table` takes a user table, which may hold several index
columns: a sample may hold named `IndexComponent`s at one asset, and
`OutcomeSample.component` takes one hazard's sample out of it.
`sample_annual_hazards` reads the same years of several hazards at one asset,
and `CappedSum`, `CompoundDamage` or a `CombineRule` states their combined loss.
`sample_curve_hazards` draws several hazards at one asset from fitted curves,
under a dependence the caller states (`Independent`, `Comonotonic`,
`EmpiricalCopula`, `GaussianCopula` or a `DependenceModel`), and `sample_curves`
does the same across assets.
`LossModel` turns the hazard into a loss and a `BasisRisk` model states how index
and loss differ. On per-year samples, `AsRecorded`, `RecentYears` and
`LinearTrend` state how the climate trend is treated, and `sample_annual_pool`
pools the model-years of several `ModelRun`s.

`design` chooses a template's free parameters on an outcome sample for an
objective (`MSE`, `ExpectedUncovered`, `WeightedCVaR` or a `UserObjective`)
under a `Budget` and a `Limit`; its LP and QP routes need the optional extra
`crc-sdk[parametric]`. `design_joint` designs one cover per asset of a
sample of several assets, for one objective and one budget on the asset
owner's total. `evaluate` and `compare` assess covers and designs, and
`payout_breakdown` splits each one's expected payout into its parts. A
`CoverPackage` is separately designed contracts on one asset's hazards, as one
row of a comparison, and `split_budget` splits one budget between them.
`compare_pooled` computes each metric once over all the held-out years of a
k-fold split, and `bootstrap_intervals` gives each metric an interval from
redrawing the evaluated years.
`ScenarioSet` holds one evaluation block per scenario under a named reference,
and refuses a set in which anything but the hazard differs.

`ParametricPlan` chains these steps as a lazy plan, to a written
comparison: `ParametricPlan.for_evaluation(request)` from the fitted hazard
curves of a `PortfolioEvaluation` (with `.monte_carlo(...)` and, for the
out-of-sample test, `FreshDraws`), or `ParametricPlan.for_annual_extremes(...)`
from an annual-extremes source and assets. By default each asset is designed on
its own; `.scopes(Joint())` designs them together. Its builders and `explain()`
do no I/O.

The functions work on numpy arrays and Arrow tables in memory and do no I/O,
except `write_comparison`, which writes one Parquet file atomically. The
plan reads its source and assets only at its execution points (`samples`,
`designs`, `write_parquet`), and `write_parquet` writes atomically too. See
`docs/parametric.md`.
"""

from crc_sdk.parametric.basis_risk import (
    BasisRisk,
    HazardBasisRisk,
    HazardNoise,
    ImpactNoise,
    NoBasisRisk,
)
from crc_sdk.parametric.combine import CappedSum, CombineRule, CompoundDamage
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.dependence import (
    Comonotonic,
    DependenceModel,
    EmpiricalCopula,
    GaussianCopula,
    Independent,
)
from crc_sdk.parametric.design import DesignOptions, DesignResult, design
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    CoverDefinitionError,
    DesignError,
    EvaluationError,
    InfeasibleDesignError,
    LeakageError,
    ParametricError,
    ProbabilitySemanticsError,
    SampleError,
    ScenarioError,
    SolverUnavailableError,
    UnboundedDesignError,
    UnknownPayoutKindError,
    UnresolvedParameterError,
)
from crc_sdk.parametric.evaluate import (
    CoverPackage,
    compare,
    evaluate,
    payout_breakdown,
    split_budget,
    to_long,
    write_comparison,
)
from crc_sdk.parametric.index import (
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    CompositeIndex,
    IndexDefinition,
)
from crc_sdk.parametric.intervals import bootstrap_intervals, compare_pooled
from crc_sdk.parametric.joint import (
    Individual,
    Joint,
    JointDesignResult,
    design_joint,
)
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.metadata import EvaluationMetadata, setup_text
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import (
    MSE,
    ExpectedUncovered,
    UserObjective,
    WeightedCVaR,
)
from crc_sdk.parametric.parameters import Free
from crc_sdk.parametric.payouts import (
    BinaryPayout,
    LinearConstraints,
    LinearInAmounts,
    LinearPayout,
    MaxOf,
    PayoutFunction,
    PiecewiseLinearPayout,
    StepGrid,
    StepPayout,
    SumOf,
    register_payout,
)
from crc_sdk.parametric.plan import (
    FreshDraws,
    H3Cell,
    Holdout,
    KFold,
    ModelPool,
    NearestCell,
    ParametricPlan,
    ParametricResult,
    StationColumn,
)
from crc_sdk.parametric.pricing import CostOfCapitalLoad, PurePremium
from crc_sdk.parametric.sample import (
    IndexComponent,
    OutcomeSample,
    SampleProvenance,
    holdout,
    year_block_kfold,
)
from crc_sdk.parametric.scenarios import ScenarioSet
from crc_sdk.parametric.sources import (
    AnnualSeries,
    ModelRun,
    sample_annual,
    sample_annual_hazards,
    sample_annual_pool,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric.trend import AsRecorded, LinearTrend, RecentYears

__all__ = [
    "AnnualSeries",
    "AsRecorded",
    "AtAsset",
    "AtCell",
    "AtPoint",
    "AtStation",
    "BasisRisk",
    "BinaryPayout",
    "Budget",
    "CappedSum",
    "CombineRule",
    "ComonotoneIndexWarning",
    "Comonotonic",
    "CompositeIndex",
    "CompoundDamage",
    "CostOfCapitalLoad",
    "Cover",
    "CoverDefinitionError",
    "CoverPackage",
    "DependenceModel",
    "DesignError",
    "DesignOptions",
    "DesignResult",
    "EmpiricalCopula",
    "EvaluationError",
    "EvaluationMetadata",
    "ExpectedUncovered",
    "Free",
    "FreshDraws",
    "GaussianCopula",
    "H3Cell",
    "HazardBasisRisk",
    "HazardNoise",
    "Holdout",
    "ImpactNoise",
    "Independent",
    "IndexComponent",
    "IndexDefinition",
    "Individual",
    "InfeasibleDesignError",
    "Joint",
    "JointDesignResult",
    "KFold",
    "LeakageError",
    "Limit",
    "LinearConstraints",
    "LinearInAmounts",
    "LinearPayout",
    "LinearTrend",
    "LossModel",
    "MSE",
    "MaxOf",
    "MetricOptions",
    "ModelPool",
    "ModelRun",
    "NearestCell",
    "NoBasisRisk",
    "OutcomeSample",
    "ParametricError",
    "ParametricPlan",
    "ParametricResult",
    "PayoutFunction",
    "PiecewiseLinearPayout",
    "ProbabilitySemanticsError",
    "PurePremium",
    "RecentYears",
    "SampleError",
    "SampleProvenance",
    "ScenarioError",
    "ScenarioSet",
    "SolverUnavailableError",
    "StationColumn",
    "StepGrid",
    "StepPayout",
    "SumOf",
    "UnboundedDesignError",
    "UnknownPayoutKindError",
    "UnresolvedParameterError",
    "UserObjective",
    "WeightedCVaR",
    "bootstrap_intervals",
    "compare",
    "compare_pooled",
    "design",
    "design_joint",
    "evaluate",
    "holdout",
    "payout_breakdown",
    "register_payout",
    "sample_annual",
    "sample_annual_hazards",
    "sample_annual_pool",
    "sample_curve",
    "sample_curve_hazards",
    "sample_curves",
    "setup_text",
    "split_budget",
    "to_long",
    "write_comparison",
    "year_block_kfold",
]
