"""The public names of crc_sdk.parametric and what importing it loads."""

from __future__ import annotations

import subprocess
import sys

import crc_sdk.parametric as parametric

PUBLIC = [
    "AnnualSeries",
    "AsRecorded",
    "AtAsset",
    "AtCell",
    "AtPoint",
    "AtStation",
    "BasisRisk",
    "BinaryPayout",
    "Budget",
    "ComonotoneIndexWarning",
    "Comonotonic",
    "CompositeIndex",
    "CostOfCapitalLoad",
    "Cover",
    "CoverDefinitionError",
    "CoverPackage",
    "DesignError",
    "DesignOptions",
    "DesignResult",
    "EmpiricalCopula",
    "EvaluationError",
    "EvaluationMetadata",
    "ExpectedUncovered",
    "Free",
    "FreshDraws",
    "H3Cell",
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
    "compare",
    "design",
    "design_joint",
    "evaluate",
    "holdout",
    "payout_breakdown",
    "register_payout",
    "sample_annual",
    "sample_annual_pool",
    "sample_curve",
    "sample_curves",
    "setup_text",
    "split_budget",
    "to_long",
    "write_comparison",
    "year_block_kfold",
]


def test_public_names() -> None:
    assert parametric.__all__ == PUBLIC
    for name in PUBLIC:
        assert getattr(parametric, name).__module__.startswith("crc_sdk.parametric")


def test_importing_the_module_loads_no_solver() -> None:
    # The base install has no solver; the module must import without one.
    code = (
        "import sys, crc_sdk.parametric; "
        "print(sorted(m for m in ('highspy', 'scipy', 'cvxpy') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_only_computing_an_h3_cell_needs_the_geometry_extra() -> None:
    # With the geometry extra's packages made unimportable, as in a base
    # install, the module imports and an H3Cell can be stated; a plan built
    # with one explains itself; only computing the cells raises, and says so.
    code = """
import sys

sys.modules["h3"] = None
sys.modules["shapely"] = None

import numpy as np
import pyarrow as pa

from crc_framework import PiecewiseLinearImpact
from crc_sdk.parametric import H3Cell, NoBasisRisk, ParametricPlan
from crc_sdk.types import SourceProvenance

annual = pa.table(
    {
        "year": pa.array([2000, 2001], pa.int32()),
        "latitude": [50.0, 50.0],
        "longitude": [7.0, 8.0],
        "value": [1.0, 2.0],
    }
)
assets = pa.table({"asset_id": ["a"], "longitude": [7.0], "latitude": [50.0]})
plan = ParametricPlan.for_annual_extremes(
    annual,
    assets,
    impact=PiecewiseLinearImpact([0.0, 1.0], [0.0, 1.0]),
    unit="mm/day",
    provenance=SourceProvenance(provider="p", dataset="d"),
    cell=H3Cell(resolution=7),
).basis_risk(NoBasisRisk())
plan.explain()
try:
    plan.samples()
except ImportError as error:
    print(error)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "crc-sdk[geometry]" in result.stdout
