"""The cluster scope in the functions built on `compare`.

`compare_pooled`, `bootstrap_intervals`, `compare_scenarios` and
`allocate_scenarios` take a `ClusteredDesignResult`; `payout_breakdown`,
`payout_breakdown_scenarios` and `design` refuse what they can't read. Each
row is checked against a computation written apart from the module, from the
covers' terms and the sample's arrays. Synthetic numbers, offline. Nothing here
is a finding about a design, and nothing sets a cluster design against another
scope.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AtPoint,
    Clustered,
    ClusteredDesignResult,
    ComonotoneIndexWarning,
    Comonotonic,
    CostOfCapitalLoad,
    Cover,
    DesignError,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    HazardNoise,
    Independent,
    IndexDefinition,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    SampleProvenance,
    ScenarioError,
    ScenarioSet,
    StepPayout,
    allocate,
    allocate_scenarios,
    bootstrap_intervals,
    compare,
    compare_pooled,
    compare_scenarios,
    design,
    design_joint,
    holdout,
    payout_breakdown,
    payout_breakdown_scenarios,
    sample_annual,
    sample_curves,
    setup_text,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.constraints import Budget
from crc_sdk.parametric.metrics import metric_columns, metric_values
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.ReturnPeriodExtrapolationWarning"
    ),
]

UNTYPED_INTERVALS: Any = bootstrap_intervals
UNTYPED_CURVES: Any = sample_curves
WIND = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
ASSETS = ("A", "B", "C")
ASSIGNMENT = {"A": "north", "B": "north", "C": "south"}
VALUES = {"A": 3.0e6, "B": 1.0e6, "C": 2.0e6}
FEW = MetricOptions(betas=(0.9,), return_periods=(5.0,))
NAMES = [column.name for column in metric_columns(FEW)]
NO_PREMIUM = {
    column.name
    for column in metric_columns(FEW)
    if column.metric in ("premium", "var_net_loss", "cvar_net_loss", "net_loss_rp")
}
# Which draws read an amount the fill rule set is checked where fills are, in
# the cluster scope's own tests.
UNCHECKED = frozenset({"unidentified_share"})


def _table_sample(n: int = 30) -> OutcomeSample:
    """Three assets over `n` years of a seeded synthetic wind and its losses."""
    rng = np.random.default_rng(20261122)
    index = rng.gumbel(30.0, 6.0, size=(n, 3))
    loss = np.where(index > 33.0, (index - 33.0) * 1e5, 0.0) * [1.0, 0.5, 0.8]
    return OutcomeSample(
        asset_ids=ASSETS,
        period=np.arange(1980, 1980 + n),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=loss,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "synthetic book"},),
            semantics_source="caller",
            licences=({"role": "table", "licence": "CC0-1.0"},),
            dependence={"kind": "data"},
        ),
    )


def _template() -> Cover:
    return Cover(
        WIND,
        StepPayout([33.0, 38.0, 45.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        label="step",
    )


def _scope(joint: bool, assignment: Mapping[str, str] = ASSIGNMENT) -> Clustered:
    return Clustered(assignment, values=VALUES, joint=joint)


def _design(
    joint: bool,
    sample: OutcomeSample,
    assignment: Mapping[str, str] = ASSIGNMENT,
    objective: Any = None,
) -> ClusteredDesignResult:
    scope = _scope(joint, assignment)
    return design_joint(
        [_template()] * len(scope.cluster_ids),
        sample,
        ExpectedUncovered() if objective is None else objective,
        budget=Budget(ratio=0.8),
        scope=scope,
    )


# -- each row's draws, written apart from the module ---------------------------


def _cluster_draws(
    design_: ClusteredDesignResult, sample: OutcomeSample
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each cluster's loss and payout on `sample`, from its index's weights."""
    loss: dict[str, np.ndarray] = {}
    payout: dict[str, np.ndarray] = {}
    for cluster, cover in design_.covers.items():
        index = cover.cluster_index
        assert index is not None
        columns = [sample.asset_ids.index(asset) for asset in index.weights]
        weights = np.array(list(index.weights.values()))
        loss[cluster] = sample.loss[:, columns].sum(axis=1)
        payout[cluster] = cover.payout_values(sample.index[:, columns] @ weights)
    return loss, payout


def _row_draws(
    design_: ClusteredDesignResult, sample: OutcomeSample, row: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """One row's loss and payout, draw by draw."""
    loss, payout = _cluster_draws(design_, sample)
    level = row["level"]
    if level == "cluster":
        return loss[row["cluster_id"]], payout[row["cluster_id"]]
    if level == "total":
        return sample.loss.sum(axis=1), sum(payout.values(), np.zeros(len(sample)))
    asset = row["asset_id"]
    cluster = ASSIGNMENT[asset]
    members = [name for name, own in ASSIGNMENT.items() if own == cluster]
    own_loss = sample.loss[:, sample.asset_ids.index(asset)]
    by_value = VALUES[asset] / math.fsum(VALUES[name] for name in members)
    if level == "asset_by_value":
        return own_loss, payout[cluster] * by_value
    assert level == "asset_by_loss"
    shares = np.array(
        [
            own / whole if whole > 0.0 else by_value
            for own, whole in zip(own_loss, loss[cluster], strict=True)
        ]
    )
    return own_loss, payout[cluster] * shares


def _expected(
    loss: np.ndarray,
    payout: np.ndarray,
    premium: float | np.ndarray | None,
) -> dict[str, float | None]:
    """The metrics of equally likely draws, with no premium on an allocated row."""
    values = metric_values(
        loss,
        payout,
        np.full(loss.shape[0], 1.0 / loss.shape[0]),
        premium=0.0 if premium is None else premium,
        options=FEW,
    )
    if premium is None:
        values.update(dict.fromkeys(NO_PREMIUM))
    return values


def _same(found: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for name in NAMES:
        if name in UNCHECKED:
            continue
        if expected[name] is None:
            assert found[name] is None, name
        else:
            assert found[name] == pytest.approx(expected[name], rel=1e-9, abs=1e-6), (
                name
            )


def _bounds(table: Any) -> dict[tuple[Any, ...], dict[str, tuple[Any, Any]]]:
    """Each wide row's (ci_low, ci_high) by metric column."""
    found: dict[tuple[Any, ...], dict[str, tuple[Any, Any]]] = {}
    for row in table.to_pylist():
        key = (row["cover_label"], row["level"], row["cluster_id"], row["asset_id"])
        found.setdefault(key, {})[row["column"]] = (row["ci_low"], row["ci_high"])
    return found


def _flags(table: Any) -> dict[tuple[Any, ...], bool]:
    return {
        (row["level"], row["cluster_id"], row["asset_id"]): row["built_together"]
        for row in table.to_pylist()
    }


def _intervals_record(table: Any) -> dict[str, Any]:
    intervals = EvaluationMetadata.from_table(table).setup.intervals
    assert intervals is not None
    return intervals


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["cover_label"], row["level"], row["cluster_id"], row["asset_id"])


def _shape(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [(row["level"], row["cluster_id"], row["asset_id"]) for row in rows]


def _rows_of(joint: bool, rule: str) -> list[tuple[Any, ...]]:
    """The rows of a design of `ASSIGNMENT` under an allocation, in order."""
    clusters = [] if joint else [("cluster", "north", None), ("cluster", "south", None)]
    return [
        *clusters,
        ("total", None, None),
        *((f"asset_by_{rule}", None, asset) for asset in ASSETS),
    ]


# -- pooled folds ----------------------------------------------------------------


@pytest.mark.parametrize("joint", [True, False], ids=["joint", "separate"])
def test_a_pooled_row_reads_every_held_out_year_with_its_folds_design(
    joint: bool,
) -> None:
    sample = _table_sample()
    folds = year_block_kfold(sample, 3)
    per_fold = [_design(joint, train) for train, _ in folds]
    table = compare_pooled(
        [per_fold], folds, options=FEW, allocation="value", values=VALUES
    )
    by_fold = [
        compare(
            [made],
            test,
            split="out_of_sample",
            fold=fold,
            options=FEW,
            allocation="value",
            values=VALUES,
        ).to_pylist()
        for fold, (made, (_, test)) in enumerate(zip(per_fold, folds, strict=True))
    ]
    rows = table.to_pylist()
    assert _shape(rows) == _shape(by_fold[0]) == _rows_of(joint, "value")
    for position, row in enumerate(rows):
        fold_rows = [each[position] for each in by_fold]
        assert row["split"] == "out_of_sample" and row["fold"] is None
        assert row["scope"] == "cluster"
        assert row["cover_hash"] == content_hash([r["cover_hash"] for r in fold_rows])
        assert row["design_id"] == content_hash([r["design_id"] for r in fold_rows])
        drawn = [
            _row_draws(made, test, row)
            for made, (_, test) in zip(per_fold, folds, strict=True)
        ]
        premium = (
            None
            if row["premium"] is None
            else np.concatenate(
                [
                    np.full(len(test), r["premium"])
                    for r, (_, test) in zip(fold_rows, folds, strict=True)
                ]
            )
        )
        expected = _expected(
            np.concatenate([loss for loss, _ in drawn]),
            np.concatenate([payout for _, payout in drawn]),
            premium,
        )
        _same(row, expected)
        if row["level"] == "cluster":
            # Each fold's design premium of that cluster's own contract.
            assert row["premium"] == pytest.approx(
                np.mean([r["premium"] for r in fold_rows]), rel=1e-12
            )
    assert set(table.column("cluster_id").to_pylist()) == (
        {None} if joint else {"north", "south", None}
    )
    text = setup_text(table)
    assert "Scope: cluster" in text


def test_a_fold_of_other_clusters_is_refused() -> None:
    sample = _table_sample()
    folds = year_block_kfold(sample, 3)
    mixed = [_design(fold != 1, train) for fold, (train, _) in enumerate(folds)]
    with pytest.raises(EvaluationError, match="the same rows on every fold"):
        compare_pooled([mixed], folds, options=FEW)
    # As many clusters, but other ones: their rows differ by cluster_id alone.
    renamed = {"A": "east", "B": "east", "C": "west"}
    other = [
        _design(False, train, renamed if fold == 1 else ASSIGNMENT)
        for fold, (train, _) in enumerate(folds)
    ]
    with pytest.raises(EvaluationError, match="the same rows on every fold"):
        compare_pooled([other], folds, options=FEW)


def test_a_pooled_design_must_be_its_folds() -> None:
    sample = _table_sample()
    folds = year_block_kfold(sample, 3)
    per_fold = [_design(True, train) for train, _ in folds]
    with pytest.raises(EvaluationError, match="not that fold's design block"):
        compare_pooled([per_fold[::-1]], folds, options=FEW)


# -- intervals on one block -------------------------------------------------------


@pytest.mark.parametrize("joint", [True, False], ids=["joint", "separate"])
def test_intervals_of_a_cluster_design_resample_its_years(joint: bool) -> None:
    sample = _table_sample()
    train, test = holdout(sample, 0.6)
    made = _design(joint, train)
    arguments: dict[str, Any] = {
        "split": "out_of_sample",
        "options": FEW,
        "allocation": "loss",
        "values": VALUES,
    }
    table = UNTYPED_INTERVALS(
        [made], test, rng=np.random.default_rng(4), n_bootstrap=1, **arguments
    )
    # Today's one draw of the block's years, each year one row.
    years = np.random.default_rng(4).integers(0, len(test), size=(1, len(test)))[0]
    wide = compare([made], test, **arguments).to_pylist()
    assert _shape(wide) == _rows_of(joint, "loss")
    bounds = _bounds(table)
    for row in wide:
        loss, payout = _row_draws(made, test, row)
        expected = _expected(loss[years], payout[years], row["premium"])
        found = {name: bounds[_key(row)][name][0] for name in NAMES}
        _same(found, expected)
        assert {name: bounds[_key(row)][name][1] for name in NAMES} == found
    # Per-year data: no dependence model moved a column, so no row is flagged.
    assert set(table.column("built_together").to_pylist()) == {False}
    record = _intervals_record(table)
    assert record["unit"] == "year" and record["built_together"] == []
    assert "kept_today" not in record


def test_intervals_over_folds_take_a_cluster_design() -> None:
    sample = _table_sample()
    folds = year_block_kfold(sample, 3)
    per_fold = [_design(False, train) for train, _ in folds]
    table = UNTYPED_INTERVALS(
        [per_fold], folds, rng=np.random.default_rng(2), n_bootstrap=20, options=FEW
    )
    pooled = compare_pooled([per_fold], folds, options=FEW)
    assert table.num_rows == pooled.num_rows * len(NAMES)
    assert {(row["level"], row["cluster_id"]) for row in table.to_pylist()} == {
        ("cluster", "north"),
        ("cluster", "south"),
        ("total", None),
    }
    assert _intervals_record(table)["unit"] == "year"


# -- rows built together -------------------------------------------------------


M = 3
K = 4
N = K * M
LOCATIONS = {"A": 30.0, "B": 33.0, "C": 36.0}


def _metadata() -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="m/s",
        value_semantics="annual maximum",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="synthetic",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic fixture",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
    )


RAMP = PiecewiseLinearImpact([30.0, 60.0], [0.0, 1.0])


def _loss_model() -> LossModel:
    return LossModel(RAMP, value=1e6, value_unit="EUR")


def _reference() -> OutcomeSample:
    """`M` years whose three columns move together, an EmpiricalCopula's reference."""
    rng = np.random.default_rng(5)
    common = rng.gamma(2.0, 5.0, M)
    years = np.arange(1990, 1990 + M)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        return OutcomeSample.join(
            sample_annual(
                AnnualSeries(
                    years,
                    common + rng.gamma(1.0, 2.0, M),
                    "m/s",
                    {"source": {"name": f"series of {asset}"}},
                ),
                loss_model=_loss_model(),
                basis_risk=NoBasisRisk(),
                asset_id=asset,
            )
            for asset in ASSETS
        )


def _curves(dependence: Any, basis_risk: Any = None, n: int = N) -> OutcomeSample:
    return sample_curves(
        {
            asset: FittedDistribution.from_parameters(
                "gumbel_r", location=LOCATIONS[asset], scale=6.0
            )
            for asset in ASSETS
        },
        dependence=dependence,
        metadata=dict.fromkeys(ASSETS, _metadata()),
        n_draws=n,
        rngs={asset: np.random.default_rng(31 + j) for j, asset in enumerate(ASSETS)},
        loss_models=dict.fromkeys(ASSETS, _loss_model()),
        basis_risk=HazardNoise(0.3) if basis_risk is None else basis_risk,
    )


def _expected_redeal(
    index: np.ndarray, k: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Each column's rows as one replicate deals them, written apart from the module.

    A column's redrawn draws, sorted by value with ties in the order drawn, fill
    the places each reference year's rows held in it, and a year's rows take its
    draws in the order drawn.
    """
    n, columns = index.shape
    sources = []
    for j in range(columns):
        drawn = rng.integers(0, n, size=(1, n), dtype=np.int64)[0]
        held = sorted(range(n), key=lambda row: (index[row, j], row))
        place = {row: p for p, row in enumerate(held)}
        by_value = sorted(range(n), key=lambda d: (index[drawn[d], j], d))
        source = np.empty(n, dtype=np.int64)
        for year in range(n // k):
            rows = list(range(year * k, year * k + k))
            taken = sorted(by_value[place[row]] for row in rows)
            for row, d in zip(rows, taken, strict=True):
                source[row] = drawn[d]
        sources.append(source)
    return sources


def _rebuilt(sample: OutcomeSample, sources: Sequence[np.ndarray]) -> OutcomeSample:
    """`sample` with each column's rows taken from its own source rows."""
    return OutcomeSample(
        asset_ids=sample.asset_ids,
        period=np.arange(len(sample)),
        probability=sample.probability,
        index=np.column_stack(
            [sample.index[source, j] for j, source in enumerate(sources)]
        ),
        loss=np.column_stack(
            [sample.loss[source, j] for j, source in enumerate(sources)]
        ),
        hazard=None,
        index_unit=sample.index_unit,
        loss_unit=sample.loss_unit,
        semantics=sample.semantics,
        provenance=SampleProvenance(
            source_kind="table",
            sources=(),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )


@pytest.mark.parametrize("joint", [True, False], ids=["joint", "separate"])
def test_a_cluster_design_on_a_copula_block_is_re_dealt(joint: bool) -> None:
    # A cluster of two assets reads two columns the copula dealt, as the total
    # does: re-dealt. A one-asset cluster's rows read one column: today's draw.
    sample = _curves(EmpiricalCopula(_reference()))
    made = _design(joint, sample, objective=MSE())
    arguments: dict[str, Any] = {
        "split": "in_sample",
        "options": FEW,
        "allocation": "loss",
        "values": VALUES,
    }
    table = UNTYPED_INTERVALS(
        [made], sample, rng=np.random.default_rng(6), n_bootstrap=1, **arguments
    )
    copy = np.random.default_rng(6)
    today = copy.integers(0, N, size=(1, N), dtype=np.int64)[0]
    sources = _expected_redeal(sample.index, K, copy)
    rebuilt = _rebuilt(sample, sources)
    bounds = _bounds(table)
    flags = _flags(table)
    # North's assets, A and B, and every asset for the total.
    together: set[tuple[Any, ...]] = {
        ("cluster", "north", None),
        ("total", None, None),
        ("asset_by_loss", None, "A"),
        ("asset_by_loss", None, "B"),
    }
    for row in compare([made], sample, **arguments).to_pylist():
        key = (row["level"], row["cluster_id"], row["asset_id"])
        assert flags[key] is (key in together), key
        if key in together:
            loss, payout = _row_draws(made, rebuilt, row)
        else:
            loss, payout = (each[today] for each in _row_draws(made, sample, row))
        found = {name: bounds[_key(row)][name][0] for name in NAMES}
        _same(found, _expected(loss, payout, row["premium"]))
    record = _intervals_record(table)
    assert record["unit"] == "as_built"
    assert record["built_together"] == ["empirical_copula"]
    assert "kept_today" not in record
    line = next(
        text for text in setup_text(table).splitlines() if text.startswith("Intervals")
    )
    assert "deals them to the reference years again by the copula's rule" in line


def test_the_rows_of_one_asset_clusters_keep_todays_numbers() -> None:
    # Every cluster of one asset: only the total reads two columns or more.
    sample = _curves(EmpiricalCopula(_reference()))
    alone = {"A": "a", "B": "b", "C": "c"}
    made = _design(False, sample, alone, objective=MSE())
    table = _intervals_on(made, sample)
    flags = _flags(table)
    assert flags == {
        ("cluster", "a", None): False,
        ("cluster", "b", None): False,
        ("cluster", "c", None): False,
        ("total", None, None): True,
    }


def _intervals_on(made: Any, sample: OutcomeSample, n_bootstrap: int = 5) -> Any:
    return UNTYPED_INTERVALS(
        [made],
        sample,
        split="in_sample",
        rng=np.random.default_rng(3),
        n_bootstrap=n_bootstrap,
        options=FEW,
    )


def test_a_held_block_of_a_cluster_design_is_re_ranked() -> None:
    # Every column holds the held form, so every row of the design reads one.
    sample = _curves(Independent(), HazardNoise(0.3, hold="hazard_distribution"))
    made = _design(False, sample, objective=MSE())
    table = _intervals_on(made, sample)
    assert set(_flags(table).values()) == {True}
    record = _intervals_record(table)
    assert record["unit"] == "as_built"
    assert record["built_together"] == ["hazard_distribution"]


def test_a_comonotonic_block_keeps_todays_method_on_its_clusters() -> None:
    sample = _curves(Comonotonic())
    made = _design(False, sample, objective=MSE())
    table = _intervals_on(made, sample)
    assert _flags(table) == {
        ("cluster", "north", None): True,
        ("cluster", "south", None): False,
        ("total", None, None): True,
    }
    record = _intervals_record(table)
    assert record["unit"] == "year"
    assert record["built_together"] == ["comonotonic"]


# -- scenarios -------------------------------------------------------------------


CELLS = {asset: point_to_cell(6.9 + 0.2 * j, 50.9, 7) for j, asset in enumerate(ASSETS)}


def _row(asset: str, horizon: int, shift: float) -> dict[str, Any]:
    """One canonical row of an asset's fitted curve of the wind, at `horizon`."""
    return {
        "cell_index": CELLS[asset],
        "source_id": f"fixture-{asset}",
        "source_geometry": None,
        "hazard_name": "wind",
        "horizon": horizon,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": LOCATIONS[asset] + shift,
        "curve_scale": 6.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def _book(label: str, seed: int, shift: float = 0.0) -> OutcomeSample:
    """Each scenario on the same seeds, so the scenarios share their draws."""
    horizon = 2050 if shift == 0.0 else 2080
    sample: OutcomeSample = UNTYPED_CURVES(
        {asset: _row(asset, horizon, shift) for asset in ASSETS},
        dependence=Independent(),
        metadata=dict.fromkeys(ASSETS, _metadata()),
        n_draws=60,
        rngs={asset: np.random.default_rng(seed + j) for j, asset in enumerate(ASSETS)},
        loss_models=dict.fromkeys(ASSETS, _loss_model()),
        basis_risk=HazardNoise(0.3),
        scenario=label,
    )
    return sample


def _scenarios() -> ScenarioSet:
    return ScenarioSet(
        [_book("early", 11), _book("late", 11, shift=4.0)], reference="early"
    )


@pytest.mark.parametrize("joint", [True, False], ids=["joint", "separate"])
def test_compare_scenarios_gives_a_cluster_designs_rows_on_every_block(
    joint: bool,
) -> None:
    made = _design(joint, _book("early", 1), objective=MSE())
    scenarios = _scenarios()
    pricing = CostOfCapitalLoad(0.2, 0.9)
    table = compare_scenarios([made], scenarios, pricing=pricing, options=FEW)
    for label in scenarios.labels:
        block = scenarios.sample(label)
        mine = [row for row in table.to_pylist() if row["scenario"] == label]
        plain = compare(
            [made], block, split="out_of_sample", pricing=pricing, options=FEW
        ).to_pylist()
        assert [
            {key: value for key, value in row.items() if key != "premium_repriced"}
            for row in mine
        ] == plain
        loss, payout = _cluster_draws(made, block)
        for row in mine:
            assert row["design_scenario"] == "early" and row["scope"] == "cluster"
            if row["level"] == "cluster":
                repriced = pricing.premium(payout[row["cluster_id"]], block.probability)
            elif joint:
                total = sum(payout.values(), np.zeros(len(block)))
                repriced = pricing.premium(total, block.probability)
            else:
                # Each cluster is its own contract, priced on its own payout.
                repriced = math.fsum(
                    pricing.premium(each, block.probability) for each in payout.values()
                )
            assert row["premium_repriced"] == pytest.approx(repriced, rel=1e-12)
    levels = [row["level"] for row in table.to_pylist()]
    assert levels == (["total"] * 2 if joint else ["cluster", "cluster", "total"] * 2)


@pytest.mark.parametrize("rule", ["value", "loss"])
def test_allocate_scenarios_gives_a_cluster_designs_allocated_rows(rule: Any) -> None:
    made = _design(True, _book("early", 1), objective=MSE())
    later = _design(False, _book("late", 3, shift=4.0), objective=MSE())
    scenarios = _scenarios()
    table = allocate_scenarios(
        [made, later], scenarios, allocation=rule, values=VALUES, options=FEW
    )
    beside = compare_scenarios(
        [made, later], scenarios, allocation=rule, values=VALUES, options=FEW
    )
    level = f"asset_by_{rule}"
    assert table.to_pylist() == [
        row for row in beside.to_pylist() if row["level"] == level
    ]
    # 2 designs x 2 scenarios x 3 assets, each a share of its own cluster's payout.
    assert table.num_rows == 12
    for row in table.to_pylist():
        assert row["scope"] == "cluster" and row["premium"] is None
        design_ = made if row["design_scenario"] == "early" else later
        block = scenarios.sample(row["scenario"])
        loss, payout = _row_draws(design_, block, row)
        _same(row, {**_expected(loss, payout, None), "premium_repriced": None})
    assert {
        e["allocation"] for e in EvaluationMetadata.from_table(table).setup.evaluations
    } == {rule}


def test_allocate_scenarios_on_one_scenario_is_allocate(tmp_path: Path) -> None:
    made = _design(False, _book("early", 1), objective=MSE())
    block = _book("early", 11)
    table = allocate_scenarios(
        [made],
        ScenarioSet([block], reference="early"),
        allocation="value",
        values=VALUES,
        options=FEW,
    )
    plain = allocate(
        [made],
        block,
        split="out_of_sample",
        allocation="value",
        values=VALUES,
        options=FEW,
    )
    assert table.drop_columns(["premium_repriced"]).to_pylist() == plain.to_pylist()
    assert table.column("premium_repriced").null_count == table.num_rows
    path = tmp_path / "allocated.parquet"
    write_comparison(table, path)
    assert pq.read_table(path).equals(table)


def test_a_cluster_cover_bound_elsewhere_is_refused_on_a_scenario() -> None:
    # A cluster's cover is read at each of its assets, on the index they read.
    made = _design(False, _book("early", 1), objective=MSE())
    elsewhere = IndexDefinition(
        "wind", "m/s", aggregation="annual_maximum", location=AtPoint(7.0, 50.0)
    )
    template = Cover(
        elsewhere,
        StepPayout([33.0, 38.0, 45.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        label="step",
    )
    scope = _scope(False)
    bound = design_joint(
        [template, template],
        _book("early", 1),
        MSE(),
        budget=Budget(ratio=0.8),
        scope=scope,
    )
    compare_scenarios([made], _scenarios(), options=FEW)
    with pytest.raises(ScenarioError, match="differ in location"):
        compare_scenarios([bound], _scenarios(), options=FEW)


# -- what is refused -------------------------------------------------------------


def test_a_payout_breakdown_refuses_a_cluster_design() -> None:
    sample = _table_sample()
    made = _design(True, sample)
    message = "parts one asset's payout over its hazards"
    with pytest.raises(EvaluationError, match=message):
        payout_breakdown([made], sample, split="in_sample")
    with pytest.raises(EvaluationError, match=message):
        payout_breakdown_scenarios(
            [_design(True, _book("early", 1), objective=MSE())], _scenarios()
        )


def test_design_refuses_a_clusters_template_naming_design_joint() -> None:
    sample = _table_sample()
    cover = _design(True, sample).covers["north"]
    index = cover.cluster_index
    assert index is not None
    template = Cover(
        index, StepPayout([33.0], [Free()]), payout_unit="EUR", asset_id="north"
    )
    for on in (sample.asset("A"), sample):
        with pytest.raises(
            DesignError, match=r"design_joint\(.*scope=Clustered"
        ) as raised:
            design(template, on, MSE())
        assert raised.value.asset_id == "north"
        assert "['A', 'B']" in str(raised.value)
