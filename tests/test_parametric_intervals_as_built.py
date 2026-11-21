"""`bootstrap_intervals` on rows built together: each replicate builds them again.

On a row that reads two columns or more of an `EmpiricalCopula` block, each
replicate redraws each column's draws on its own and deals them to the
reference years again by the copula's rule. On a row that reads a column built
with `HazardNoise(..., hold="hazard_distribution")`, each replicate ranks the
hazard at the asset again over the resampled rows. These tests pin both methods
on small blocks against a separate computation from the same generator, check
the layout of the draws, that every other row keeps today's draws and numbers,
what is refused and what the metadata and the setup text say. Every number is
synthetic; the coverage of these intervals is measured in a separate study, not
here.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    CappedSum,
    ComonotoneIndexWarning,
    Comonotonic,
    CompoundDamage,
    Cover,
    CoverPackage,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    Free,
    GaussianCopula,
    HazardNoise,
    Independent,
    IndexDefinition,
    LinearPayout,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    PurePremium,
    SampleProvenance,
    StepPayout,
    bootstrap_intervals,
    compare,
    design_joint,
    sample_annual,
    sample_annual_hazards,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric.intervals import (
    _block,
    _block_columns,
    _by_hazard,
    _plan,
    _pooled,
    _redeal_draws,
    _replicates,
    _reranked,
    _year_counts,
)
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

ASSETS = ("site-a", "site-b")
HAZARDS = ("rain", "wind")
UNITS = {"rain": "mm/day", "wind": "m/s"}
LOCATIONS = {"site-a": 50.0, "site-b": 60.0, "rain": 50.0, "wind": 60.0}
RAMP = PiecewiseLinearImpact([30.0, 120.0], [0.0, 1.0])
M = 3
K = 2
N = K * M
FEW = MetricOptions(betas=(0.9,), return_periods=(5.0,))
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
INDICES = {
    "rain": IndexDefinition("rain", "mm/day", aggregation="annual_maximum"),
    "wind": IndexDefinition("wind", "m/s", aggregation="annual_maximum"),
}
HELD = HazardNoise(0.3, hold="hazard_distribution")
NAMES = [column.name for column in metric_columns(FEW)]

REBUILT_TEXT = "Each resample builds those rows again as the block built them"
REDEALT_TEXT = "deals them to the reference years again by the copula's rule"
RERANKED_TEXT = "each resample ranks it again over the resampled rows"
UNMEASURED_TEXT = "is not the sampling error of the number"


def _metadata(unit: str = "mm/day") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit=unit,
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


def _curve(name: str) -> FittedDistribution:
    return FittedDistribution.from_parameters(
        "gumbel_r", location=LOCATIONS[name], scale=10.0
    )


def _loss(name: str) -> LossModel:
    return LossModel(RAMP, value=100.0, value_unit="EUR", name=f"loss of {name}")


def _ratio(name: str) -> LossModel:
    """A loss model without a value: the loss is the damage ratio, in unit "1"."""
    return LossModel(RAMP, name=f"damage ratio of {name}")


def _cover(
    asset: str | None = None,
    *,
    index: IndexDefinition = INDEX,
    label: str = "c",
    limit: float = 60.0,
    unit: str = "EUR",
) -> Cover:
    return Cover(
        index,
        LinearPayout(55.0, 90.0, limit),
        payout_unit=unit,
        asset_id=asset,
        label=label,
    )


def _covers() -> dict[str, Cover]:
    return {asset: _cover(asset, label=asset) for asset in ASSETS}


def _reference(columns: Sequence[str], m: int = M) -> OutcomeSample:
    """`m` years whose columns move together, as an EmpiricalCopula's reference."""
    rng = np.random.default_rng(5)
    common = rng.gamma(2.0, 20.0, m)
    years = np.arange(1990, 1990 + m)
    series = {
        name: AnnualSeries(
            years,
            common + rng.gamma(1.0, 5.0, m),
            UNITS.get(name, "mm/day"),
            {"source": {"name": f"series of {name}"}},
        )
        for name in columns
    }
    if set(columns) == set(HAZARDS):
        return sample_annual_hazards(
            series,
            tails=dict.fromkeys(HAZARDS, "upper"),
            loss_models={"rain": _ratio("rain")},
            basis_risk=NoBasisRisk(),
            asset_id="site",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        parts = [
            sample_annual(
                series[name],
                loss_model=_loss(name),
                basis_risk=NoBasisRisk(),
                asset_id=name,
            )
            for name in columns
        ]
    return OutcomeSample.join(parts)


def _assets(
    dependence: Any, basis_risk: Any = None, *, n: int = N, seed: int = 11
) -> OutcomeSample:
    return sample_curves(
        {asset: _curve(asset) for asset in ASSETS},
        dependence=dependence,
        metadata={asset: _metadata() for asset in ASSETS},
        n_draws=n,
        rngs={asset: np.random.default_rng(seed + j) for j, asset in enumerate(ASSETS)},
        loss_models={asset: _loss(asset) for asset in ASSETS},
        basis_risk=HazardNoise(0.3) if basis_risk is None else basis_risk,
    )


def _copula(m: int = M) -> EmpiricalCopula:
    return EmpiricalCopula(_reference(ASSETS, m))


def _hazards(
    *,
    loss_models: dict[str, LossModel] | None = None,
    combine: Any = CappedSum(1.0),
    basis_risk: Any = None,
    dependence: Any = None,
) -> OutcomeSample:
    """Two hazards at one asset, their losses damage ratios unless stated."""
    models = (
        {name: _ratio(name) for name in HAZARDS} if loss_models is None else loss_models
    )
    return sample_curve_hazards(
        {name: _curve(name) for name in HAZARDS},
        dependence=(
            EmpiricalCopula(_reference(HAZARDS)) if dependence is None else dependence
        ),
        metadata={name: _metadata(UNITS[name]) for name in HAZARDS},
        n_draws=N,
        rngs={name: np.random.default_rng(21 + j) for j, name in enumerate(HAZARDS)},
        loss_models=models,
        combine=combine if len(models) > 1 else None,
        basis_risk=HazardNoise(0.3) if basis_risk is None else basis_risk,
        asset_id="site",
    )


def _package(unit: str = "1") -> CoverPackage:
    return CoverPackage(
        {
            name: _cover(index=INDICES[name], label=f"on {name}", limit=0.6, unit=unit)
            for name in HAZARDS
        },
        label="package",
    )


def _curve_sample(basis_risk: Any = HELD, n: int = 12) -> OutcomeSample:
    return sample_curve(
        _curve("site-a"),
        metadata=_metadata(),
        n_draws=n,
        rng=np.random.default_rng(4),
        loss_model=_loss("site-a"),
        basis_risk=basis_risk,
        asset_id="a",
    )


def _intervals(
    entries: list[Any], sample: OutcomeSample, *, seed: int = 3, n_bootstrap: int = 4
) -> Any:
    return UNTYPED_INTERVALS(
        entries,
        sample,
        split="in_sample",
        rng=np.random.default_rng(seed),
        n_bootstrap=n_bootstrap,
        options=FEW,
    )


def _intervals_record(table: Any) -> dict[str, Any]:
    intervals = EvaluationMetadata.from_table(table).setup.intervals
    assert intervals is not None
    return intervals


def _intervals_line(table: Any) -> str:
    return next(
        line for line in setup_text(table).splitlines() if line.startswith("Intervals:")
    )


def _bounds(table: Any, level: str, label: str | None = None) -> dict[str, Any]:
    """One wide row's (ci_low, ci_high) by metric column."""
    return {
        row["column"]: (row["ci_low"], row["ci_high"])
        for row in table.to_pylist()
        if row["level"] == level and (label is None or row["cover_label"] == label)
    }


def _check_single_replicate(
    found: dict[str, Any],
    expected: dict[str, float | None],
    without: frozenset[str] = frozenset(),
) -> None:
    """With one replicate, both bounds are that replicate's value."""
    for name in NAMES:
        if name in without:
            continue
        low, high = found[name]
        if expected[name] is None:
            assert low is None and high is None, name
        else:
            assert low == pytest.approx(expected[name], rel=1e-12, abs=1e-12), name
            assert high == pytest.approx(expected[name], rel=1e-12, abs=1e-12), name


def _premium(payouts: Sequence[np.ndarray], probability: np.ndarray) -> float:
    """The sum of each contract's pure premium, as a replicate prices covers."""
    return math.fsum(PurePremium().premium(payout, probability) for payout in payouts)


# -- an independent computation of the two methods -----------------------------


def _expected_redeal(
    index: np.ndarray, k: int, rng: np.random.Generator, n_bootstrap: int
) -> list[np.ndarray]:
    """Each column's rows as a replicate deals them: `(B, n)` source rows per column.

    Written apart from the module: the draws of a column sorted by value, ties in
    the order drawn, fill the places each reference year's rows held in it, and
    a year's rows take its draws in the order drawn.
    """
    n, columns = index.shape
    sources = []
    for j in range(columns):
        drawn = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int64)
        held = sorted(range(n), key=lambda row: (index[row, j], row))
        place = {row: p for p, row in enumerate(held)}
        source = np.empty((n_bootstrap, n), dtype=np.int64)
        for b in range(n_bootstrap):
            by_value = sorted(range(n), key=lambda d: (index[drawn[b, d], j], d))
            for year in range(n // k):
                rows = list(range(year * k, year * k + k))
                taken = sorted(by_value[place[row]] for row in rows)
                for row, d in zip(rows, taken, strict=True):
                    source[b, row] = drawn[b, d]
        sources.append(source)
    return sources


def _expected_rerank(index: np.ndarray, hazard: np.ndarray) -> np.ndarray:
    """The hazard at the asset of rows ranked again: by hazard, index, then row."""
    order = sorted(range(index.size), key=lambda r: (hazard[r], index[r], r))
    placed = np.empty_like(index)
    for rank, row in enumerate(order):
        placed[row] = np.sort(index)[rank]
    return placed


# -- re-dealing across assets --------------------------------------------------


def test_a_copula_total_is_re_dealt_by_the_reference() -> None:
    sample = _assets(_copula())
    rng = np.random.default_rng(8)
    copy = np.random.default_rng(8)
    table = UNTYPED_INTERVALS(
        [_covers()], sample, split="in_sample", rng=rng, n_bootstrap=1, options=FEW
    )
    copy.integers(0, N, size=(1, N), dtype=np.int64)  # today's draw, by row
    sources = _expected_redeal(sample.index, K, copy, 1)
    probability = np.full(N, 1.0 / N)
    loss = sum(sample.loss[sources[j][0], j] for j in range(2))
    payouts = [
        _covers()[asset].payout_values(sample.index[sources[j][0], j])
        for j, asset in enumerate(ASSETS)
    ]
    expected = metric_values(
        loss,
        payouts[0] + payouts[1],
        probability,
        premium=_premium(payouts, probability),
        options=FEW,
    )
    _check_single_replicate(_bounds(table, "total"), expected)
    # Both draws are then taken: the generator is where the copy is.
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)


def test_a_re_dealt_replicate_keeps_each_years_ranks() -> None:
    # Each column's draws of one reference year hold one band of its ranks, the
    # band the year's rows held in the block; a column's draws keep their rows.
    sample = _assets(_copula(4), n=4 * 3)
    rng = np.random.default_rng(2)
    rng.integers(0, 12, size=(20, 12), dtype=np.int64)
    sources = _expected_redeal(sample.index, 3, rng, 20)
    for j in range(2):
        held = np.argsort(np.argsort(sample.index[:, j], kind="stable"), kind="stable")
        for b in range(20):
            redrawn = np.sort(sample.index[sources[j][b], j])
            for year in range(4):
                rows = slice(3 * year, 3 * year + 3)
                assert sorted(sample.index[sources[j][b][rows], j]) == sorted(
                    redrawn[np.sort(held[rows])]
                )


def test_an_assets_own_row_keeps_todays_numbers() -> None:
    # Today's draw stays the first, so an asset's own row reads it as before:
    # the same intervals as on that asset's own view, resampled by row.
    sample = _assets(_copula())
    table = _intervals([_covers()], sample, n_bootstrap=50)
    for asset in ASSETS:
        alone = _intervals(
            [_cover(asset, label=asset)], sample.asset(asset), n_bootstrap=50
        )
        assert _bounds(table, "asset", asset) == _bounds(alone, "asset", asset)
    assert _intervals_record(alone)["unit"] == "row"


def test_a_joint_design_on_a_copula_block_is_re_dealt() -> None:
    # Every row of a joint design reads every column, its allocated rows too.
    sample = _assets(_copula(), n=12)
    steps = [
        Cover(INDEX, StepPayout([60.0], [Free()]), payout_unit="EUR", asset_id=asset)
        for asset in ASSETS
    ]
    joint = design_joint(steps, sample, MSE())
    table = UNTYPED_INTERVALS(
        [joint],
        sample,
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=1,
        options=FEW,
        allocation="loss",
        values={"site-a": 2.0, "site-b": 1.0},
    )
    copy = np.random.default_rng(1)
    copy.integers(0, 12, size=(1, 12), dtype=np.int64)
    sources = _expected_redeal(sample.index, 12 // M, copy, 1)
    losses = np.column_stack([sample.loss[sources[j][0], j] for j in range(2)])
    payout = sum(
        joint.covers[asset].payout_values(sample.index[sources[j][0], j])
        for j, asset in enumerate(ASSETS)
    )
    wide = compare([joint], sample, split="in_sample", options=FEW)
    premium = next(
        row["premium"] for row in wide.to_pylist() if row["level"] == "total"
    )
    total = metric_values(
        losses.sum(axis=1),
        payout,
        np.full(12, 1.0 / 12.0),
        premium=float(premium),
        options=FEW,
    )
    # The share of draws reading a filled amount is checked where fills are.
    _check_single_replicate(
        _bounds(table, "total"), total, frozenset({"unidentified_share"})
    )
    assert _intervals_record(table)["unit"] == "as_built"


def _horizon(horizon: int, location: float) -> dict[str, Any]:
    """One canonical row of a fitted curve at one horizon."""
    return {
        "cell_index": 613196570331971583,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": horizon,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": location,
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def test_a_window_is_re_dealt_within_each_member() -> None:
    # Row d reads member d mod 2 in every column, and the copula deals within
    # each member's rows, so a replicate keeps every column's draws in their
    # member and each reference year's draws in its band of that member.
    window = sample_curves(
        {
            "site-a": [_horizon(2045, 50.0), _horizon(2055, 60.0)],
            "site-b": [_horizon(2045, 55.0), _horizon(2055, 70.0)],
        },
        dependence=_copula(),
        metadata={asset: _metadata() for asset in ASSETS},
        n_draws=2 * M * K,
        rngs={asset: np.random.default_rng(31 + j) for j, asset in enumerate(ASSETS)},
        loss_models={asset: _loss(asset) for asset in ASSETS},
        basis_risk=HazardNoise(0.3),
    )
    evaluated = _block(
        [_covers()],
        window,
        split="in_sample",
        fold=0,
        pricing=PurePremium(),
        premiums=None,
        options=FEW,
        design_sample=None,
        allocation=None,
        values=None,
    )
    builder = _plan(evaluated).builder
    assert builder is not None
    assert [rows.tolist() for rows, _ in builder.members] == [
        [0, 2, 4, 6, 8, 10],
        [1, 3, 5, 7, 9, 11],
    ]
    assert [years.tolist() for _, years in builder.members] == [[0, 0, 1, 1, 2, 2]] * 2
    sources = _redeal_draws(builder, np.random.default_rng(0), 30)
    for j in range(2):
        assert (sources[j] % 2 == np.arange(2 * M * K) % 2).all()
        for rows, years in builder.members:
            index = window.index[rows, j]
            held = np.argsort(np.argsort(index, kind="stable"), kind="stable")
            for b in range(30):
                redrawn = np.sort(window.index[sources[j][b][rows], j])
                for year in range(M):
                    mine = years == year
                    assert sorted(window.index[sources[j][b][rows[mine]], j]) == (
                        sorted(redrawn[np.sort(held[mine])])
                    )
    table = _intervals([_covers()], window)
    assert _intervals_record(table)["unit"] == "as_built"


def test_an_independent_block_draws_nothing_more() -> None:
    sample = _assets(Independent())
    rng = np.random.default_rng(6)
    table = UNTYPED_INTERVALS(
        [_covers()], sample, split="in_sample", rng=rng, n_bootstrap=5, options=FEW
    )
    copy = np.random.default_rng(6)
    copy.integers(0, N, size=(5, N), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    assert _intervals_record(table)["unit"] == "year"


@pytest.mark.parametrize(
    "model", [Comonotonic, lambda: GaussianCopula(0.6)], ids=["comonotonic", "gaussian"]
)
def test_rows_other_models_moved_keep_todays_method(model: Any) -> None:
    sample = _assets(model())
    rng = np.random.default_rng(6)
    table = UNTYPED_INTERVALS(
        [_covers()], sample, split="in_sample", rng=rng, n_bootstrap=5, options=FEW
    )
    copy = np.random.default_rng(6)
    copy.integers(0, N, size=(5, N), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    record = _intervals_record(table)
    assert record["unit"] == "year"
    assert UNMEASURED_TEXT in _intervals_line(table)
    assert REBUILT_TEXT not in _intervals_line(table)


# -- re-ranking a held block ---------------------------------------------------


def _held_table_sample() -> OutcomeSample:
    """Eight years whose hazard at the asset is the index's values reordered."""
    index = np.array([12.0, 35.0, 50.0, 7.0, 64.0, 41.0, 23.0, 58.0])
    hazard = index[[2, 0, 4, 6, 1, 7, 3, 5]]
    return OutcomeSample(
        asset_ids=("a",),
        period=np.arange(2000, 2008),
        probability=np.full(8, 1.0 / 8.0),
        index=index,
        loss=3.0 * hazard,
        hazard=hazard,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test table", "licence": "CC0-1.0"},),
            semantics_source="caller",
            basis_risk=HELD.to_document(),
        ),
    )


def test_a_held_block_is_re_ranked_over_the_resampled_rows() -> None:
    sample = _held_table_sample()
    cover = _cover(limit=100.0)
    table = UNTYPED_INTERVALS(
        [cover],
        sample,
        split="in_sample",
        rng=np.random.default_rng(13),
        n_bootstrap=1,
        options=FEW,
    )
    rows = np.random.default_rng(13).integers(0, 8, size=(1, 8), dtype=np.int64)[0]
    assert np.unique(rows).size < rows.size  # a year drawn twice
    index, hazard = sample.index[rows, 0], sample.hazard[rows, 0]  # type: ignore[index]
    placed = _expected_rerank(index, hazard)
    # A row drawn twice takes two hazards at the asset here.
    twice = next(row for row in rows if (rows == row).sum() > 1)
    assert np.unique(placed[rows == twice]).size > 1
    probability = np.full(8, 1.0 / 8.0)
    payout = cover.payout_values(index)
    expected = metric_values(
        3.0 * placed,
        payout,
        probability,
        premium=_premium([payout], probability),
        options=FEW,
    )
    _check_single_replicate(_bounds(table, "asset"), expected)
    assert _intervals_record(table)["unit"] == "as_built"


def test_re_ranking_reads_todays_draws_and_draws_nothing_more() -> None:
    sample = _curve_sample()
    rng = np.random.default_rng(9)
    table = UNTYPED_INTERVALS(
        [_cover()], sample, split="in_sample", rng=rng, n_bootstrap=3, options=FEW
    )
    copy = np.random.default_rng(9)
    copy.integers(0, 12, size=(3, 12), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    assert _intervals_record(table)["unit"] == "as_built"
    assert RERANKED_TEXT in _intervals_line(table)


def test_a_re_ranked_replicate_holds_the_drawn_rows_index_values() -> None:
    # The held form gives each row one of the index values of its call's rows,
    # so a replicate's hazards at the asset are its own rows' index values, and
    # each row's loss is the loss the block's rows give that hazard.
    sample = _curve_sample(n=40)
    assert sample.hazard is not None
    column = _block_columns(sample)[0]
    column = dataclasses.replace(column, by_hazard=_by_hazard(column))
    rows = np.random.default_rng(1).integers(0, 40, size=40)
    placed, loss = _reranked(column, rows)
    assert loss is not None
    assert sorted(placed) == sorted(sample.index[rows, 0])
    for value, own in zip(placed, loss, strict=True):
        given = sample.loss[sample.hazard[:, 0] == value, 0]
        assert given.size and (given == own).all()


def test_a_held_copula_block_is_re_dealt_and_re_ranked() -> None:
    sample = _assets(_copula(), HELD)
    table = _intervals([_covers()], sample)
    record = _intervals_record(table)
    assert record["unit"] == "as_built"
    assert record["built_together"] == ["empirical_copula", "hazard_distribution"]
    line = _intervals_line(table)
    assert REDEALT_TEXT in line and RERANKED_TEXT in line
    assert UNMEASURED_TEXT not in line


def test_a_held_block_of_several_hazards_is_re_ranked_with_its_rule() -> None:
    sample = _hazards(basis_risk=HELD, dependence=Independent())
    table = _intervals([_package()], sample)
    assert _intervals_record(table)["unit"] == "as_built"
    assert set(table.column("built_together").to_pylist()) == {True}


# -- re-dealing across hazards -------------------------------------------------


def test_a_copula_across_hazards_is_re_dealt_with_its_rule() -> None:
    sample = _hazards()
    package = _package()
    table = UNTYPED_INTERVALS(
        [package],
        sample,
        split="in_sample",
        rng=np.random.default_rng(17),
        n_bootstrap=1,
        options=FEW,
    )
    copy = np.random.default_rng(17)
    copy.integers(0, N, size=(1, N), dtype=np.int64)
    components = sample.index_components
    assert components is not None
    values = np.column_stack([components[name].values[:, 0] for name in HAZARDS])
    sources = _expected_redeal(values, K, copy, 1)
    own = {
        name: components[name].own_loss[sources[j][0], 0]  # type: ignore[index]
        for j, name in enumerate(HAZARDS)
    }
    combined = np.minimum(own["rain"] + own["wind"], 1.0)
    payouts = [
        package._items[name].cover.payout_values(values[sources[j][0], j])
        for j, name in enumerate(HAZARDS)
    ]
    probability = np.full(N, 1.0 / N)
    expected = metric_values(
        combined,
        payouts[0] + payouts[1],
        probability,
        premium=_premium(payouts, probability),
        options=FEW,
    )
    _check_single_replicate(_bounds(table, "package"), expected)
    # A part reads its own hazard's payout against the combined loss.
    rain = metric_values(
        combined,
        payouts[0],
        probability,
        premium=_premium(payouts[:1], probability),
        options=FEW,
    )
    _check_single_replicate(_bounds(table, "asset", "on rain"), rain)


def test_compound_damage_on_damage_ratios_is_re_dealt() -> None:
    table = _intervals([_package()], _hazards(combine=CompoundDamage()))
    assert _intervals_record(table)["unit"] == "as_built"


def test_one_impact_across_hazards_is_re_dealt_with_that_hazards_loss() -> None:
    # The combined loss is the one impacted hazard's own loss, stated in EUR.
    sample = _hazards(loss_models={"rain": _loss("rain")})
    table = _intervals([_package("EUR")], sample)
    flags = {row["cover_label"]: row["built_together"] for row in table.to_pylist()}
    assert flags == {"on rain": False, "on wind": True, "package": True}
    assert _intervals_record(table)["unit"] == "as_built"


# -- what is kept on today's method --------------------------------------------


def _todays_bounds(
    entries: list[Any], on: Any, seed: int, n_bootstrap: int, **arguments: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Every row's bounds as today's method gives them, in the long table's order.

    Written from today's steps: one draw of the years, each replicate
    reweighting the rows, the percentile bounds.
    """
    evaluated = (
        _block(
            entries,
            on,
            split="in_sample",
            fold=0,
            pricing=PurePremium(),
            premiums=None,
            options=FEW,
            design_sample=None,
            allocation=None,
            values=None,
            **arguments,
        )
        if isinstance(on, OutcomeSample)
        else _pooled(
            entries,
            on,
            pricing=PurePremium(),
            options=FEW,
            allocation=None,
            values=None,
        )
    )
    years = evaluated.n_years
    draws = np.random.default_rng(seed).integers(
        0, years, size=(n_bootstrap, years), dtype=np.int64
    )
    replicates = _replicates(evaluated, _year_counts(draws, years))
    low, high = np.quantile(replicates, [(1.0 - 0.9) / 2.0, (1.0 + 0.9) / 2.0], axis=0)
    return low.ravel(order="F"), high.ravel(order="F")


def _kept(
    entries: list[Any], on: Any, kept: list[dict[str, str]], unit: str = "row"
) -> Any:
    """The call's intervals, checked to be today's on every row, with the record."""
    rng = np.random.default_rng(5)
    arguments: dict[str, Any] = (
        {"split": "in_sample"} if isinstance(on, OutcomeSample) else {}
    )
    table = UNTYPED_INTERVALS(
        entries, on, rng=rng, n_bootstrap=7, options=FEW, **arguments
    )
    low, high = _todays_bounds(entries, on, 5, 7)
    found_low = np.array(
        [np.nan if v is None else v for v in table.column("ci_low").to_pylist()]
    )
    found_high = np.array(
        [np.nan if v is None else v for v in table.column("ci_high").to_pylist()]
    )
    defined = ~np.isnan(found_low)
    assert np.array_equal(found_low[defined], low[defined])
    assert np.array_equal(found_high[defined], high[defined])
    # Today's one draw, and nothing more.
    copy = np.random.default_rng(5)
    years = _todays_years(entries, on)
    copy.integers(0, years, size=(7, years), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    record = _intervals_record(table)
    assert record["unit"] == unit
    assert record["kept_today"] == kept
    assert any(table.column("built_together").to_pylist())
    return table


def _todays_years(entries: list[Any], on: Any) -> int:
    if isinstance(on, OutcomeSample):
        return int(np.unique(on.period).size)
    return int(sum(len(test) for _, test in on))


def test_a_stated_value_across_hazards_keeps_todays_method() -> None:
    sample = _hazards(
        loss_models={name: _loss(name) for name in HAZARDS}, combine=CappedSum(100.0)
    )
    table = _kept(
        [_package("EUR")],
        sample,
        [{"kind": "empirical_copula", "reason": "combined_loss_needs_value"}],
    )
    line = _intervals_line(table)
    assert UNMEASURED_TEXT in line
    assert REBUILT_TEXT not in line
    assert "couldn't be built again on each replicate" in line
    assert "which a record never holds" in line


class _Larger:
    """A rule of the caller's: the larger impact."""

    def to_document(self) -> dict[str, Any]:
        return {"kind": "larger"}

    def statement(self) -> str:
        return "The larger impact, times the value."

    def combine(self, impacts: Any) -> Any:
        return np.maximum(*impacts.values())


def test_a_rule_of_the_callers_across_hazards_keeps_todays_method() -> None:
    _kept(
        [_package()],
        _hazards(combine=_Larger()),
        [{"kind": "empirical_copula", "reason": "caller_rule"}],
    )


def test_a_held_block_of_several_hazards_with_a_value_keeps_todays_method() -> None:
    # Re-ranking moves each hazard's own loss, so the combined loss is needed.
    sample = _hazards(
        loss_models={name: _loss(name) for name in HAZARDS},
        combine=CappedSum(100.0),
        basis_risk=HELD,
        dependence=Independent(),
    )
    _kept(
        [_package("EUR")],
        sample,
        [{"kind": "hazard_distribution", "reason": "combined_loss_needs_value"}],
        unit="year",
    )


def test_a_view_whose_combined_loss_reads_other_hazards_keeps_todays_method() -> None:
    view = _hazards().component("rain", loss="combined")
    _kept(
        [_cover(index=INDICES["rain"], limit=0.6, unit="1")],
        view,
        [{"kind": "empirical_copula", "reason": "combined_view"}],
    )


def test_an_own_loss_view_of_a_copula_sample_keeps_todays_method() -> None:
    view = _hazards().component("rain")
    table = _intervals([_cover(index=INDICES["rain"], limit=0.6, unit="1")], view)
    assert not any(table.column("built_together").to_pylist())
    assert _intervals_record(table)["unit"] == "row"
    assert "kept_today" not in _intervals_record(table)


def test_a_selection_of_a_held_sample_keeps_todays_method() -> None:
    sample = _curve_sample(n=20)
    _kept(
        [_cover()],
        sample.select(sample.period[:10]),
        [{"kind": "hazard_distribution", "reason": "held_select"}],
        unit="year",
    )


def test_pooled_folds_of_held_blocks_keep_todays_method() -> None:
    # year_block_kfold refuses a held sample, so these folds are made by hand:
    # the blocks of a plain per-year sample, each recording the held model.
    years = np.arange(1991, 2015)
    series = AnnualSeries(
        years,
        np.random.default_rng(1).gumbel(50.0, 10.0, years.size),
        "mm/day",
        {"source": {"name": "series of a"}},
    )
    sample = sample_annual(
        series,
        loss_model=_loss("a"),
        basis_risk=HazardNoise(0.3),
        asset_id="a",
        rng=np.random.default_rng(101),
    )

    def held(block: OutcomeSample) -> OutcomeSample:
        provenance = dataclasses.replace(
            block.provenance, basis_risk=HELD.to_document(), parent=None
        )
        return dataclasses.replace(block, provenance=provenance)

    folds = tuple(
        (held(train), held(test)) for train, test in year_block_kfold(sample, 3)
    )
    _kept(
        [_cover("a")],
        folds,
        [{"kind": "hazard_distribution", "reason": "pooled_by_hand"}],
        unit="year",
    )


def _without_k(sample: OutcomeSample) -> OutcomeSample:
    provenance = sample.provenance
    assert provenance.dependence is not None
    document = dict(provenance.dependence)
    del document["k"]
    return dataclasses.replace(
        sample, provenance=dataclasses.replace(provenance, dependence=document)
    )


def test_a_copula_record_without_its_draws_per_year_keeps_todays_method() -> None:
    _kept(
        [_covers()],
        _without_k(_assets(_copula())),
        [{"kind": "empirical_copula", "reason": "copula_without_k"}],
    )


def test_rows_kept_beside_rows_rebuilt_are_stated_apart() -> None:
    # Without k the total can't be dealt again; each asset's held row is still
    # re-ranked, so the call is "as_built" and states both.
    sample = _without_k(_assets(_copula(), HELD))
    rng = np.random.default_rng(5)
    table = UNTYPED_INTERVALS(
        [_covers()], sample, split="in_sample", rng=rng, n_bootstrap=7, options=FEW
    )
    record = _intervals_record(table)
    assert record["unit"] == "as_built"
    assert record["kept_today"] == [
        {"kind": "empirical_copula", "reason": "copula_without_k"}
    ]
    low, _ = _todays_bounds([_covers()], sample, 5, 7)
    rows = table.to_pylist()
    total = [i for i, row in enumerate(rows) if row["level"] == "total"]
    assert total
    for i in total:
        assert rows[i]["ci_low"] == (None if np.isnan(low[i]) else low[i])
    line = _intervals_line(table)
    assert RERANKED_TEXT in line and REDEALT_TEXT not in line
    assert "The rows that couldn't be built again were resampled as before" in line
    assert "draws per reference year" in line


def test_kept_rows_merge_and_filter_away() -> None:
    kept = _intervals(
        [_package("EUR")],
        _hazards(
            loss_models={name: _loss(name) for name in HAZARDS},
            combine=CappedSum(100.0),
        ),
    )
    view = _intervals(
        [_cover(index=INDICES["rain"])],
        _hazards(
            loss_models={name: _loss(name) for name in HAZARDS},
            combine=CappedSum(100.0),
        ).component("rain", loss="combined"),
    )
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(kept), EvaluationMetadata.from_table(view)
    )
    assert merged.setup.intervals is not None
    assert merged.setup.intervals["kept_today"] == [
        {"kind": "empirical_copula", "reason": "combined_loss_needs_value"},
        {"kind": "empirical_copula", "reason": "combined_view"},
    ]
    plain = EvaluationMetadata.from_table(
        _intervals(
            [_cover("site-b", label="site-b")], _assets(_copula()).asset("site-b")
        )
    )
    assert "kept_today" not in plain.setup.intervals  # type: ignore[operator]
    again = EvaluationMetadata.merge(EvaluationMetadata.from_table(kept), plain)
    assert again.setup.intervals["kept_today"] == [  # type: ignore[index]
        {"kind": "empirical_copula", "reason": "combined_loss_needs_value"}
    ]
    unflagged = _intervals([_covers()], _without_k(_assets(_copula())))
    rows = unflagged.filter(
        pa.array([not flag for flag in unflagged.column("built_together").to_pylist()])
    )
    assert "kept_today" not in _intervals_line(rows)
    assert "couldn't be built again" not in _intervals_line(rows)


def test_an_unknown_reason_is_refused_by_the_setup_text() -> None:
    table = _intervals([_covers()], _without_k(_assets(_copula())))
    document = json.loads(EvaluationMetadata.from_table(table).to_json())
    document["setup"]["intervals"]["kept_today"] = [
        {"kind": "empirical_copula", "reason": "tired"}
    ]
    with pytest.raises(EvaluationError, match="kept_today names a kind or a reason"):
        setup_text(EvaluationMetadata.from_json(json.dumps(document)))


def test_resampling_by_reference_year_is_not_offered() -> None:
    parameters = set(inspect.signature(bootstrap_intervals).parameters)
    assert parameters == {
        "entries",
        "on",
        "rng",
        "n_bootstrap",
        "ci_level",
        "split",
        "fold",
        "pricing",
        "premiums",
        "options",
        "design_sample",
        "allocation",
        "values",
    }


# -- the record and the setup text ----------------------------------------------


def test_the_setup_text_names_the_method_of_re_dealt_rows() -> None:
    line = _intervals_line(_intervals([_covers()], _assets(_copula())))
    assert "resamples of the evaluated rows" in line
    assert REBUILT_TEXT in line and REDEALT_TEXT in line
    assert RERANKED_TEXT not in line
    assert UNMEASURED_TEXT not in line
    assert '"Rows built together"' in line
    assert 'measured these methods only on the "total" rows of two assets' in line


def test_the_setup_text_of_a_held_block_names_re_ranking() -> None:
    line = _intervals_line(_intervals([_cover()], _curve_sample()))
    assert "resamples of the evaluated years" in line
    assert RERANKED_TEXT in line and REDEALT_TEXT not in line


def test_a_comonotonic_held_block_states_both_methods() -> None:
    # The held hazard is re-ranked; the draws Comonotonic() placed keep their rows.
    table = _intervals([_covers()], _assets(Comonotonic(), HELD))
    record = _intervals_record(table)
    assert record["unit"] == "as_built"
    assert record["built_together"] == ["comonotonic", "hazard_distribution"]
    line = _intervals_line(table)
    assert RERANKED_TEXT in line
    assert "On the rows whose draws Comonotonic() placed" in line


def test_rows_filtered_to_todays_method_read_as_todays() -> None:
    table = _intervals([_covers()], _assets(_copula()))
    kept = table.filter(
        pa.array([not flag for flag in table.column("built_together").to_pylist()])
    )
    assert kept.num_rows and _intervals_record(kept)["unit"] == "as_built"
    line = _intervals_line(kept)
    # The rows left were resampled by row, as a copula block's own rows are.
    assert "resamples of the evaluated rows" in line
    assert REBUILT_TEXT not in line
    assert "built_together" not in line


def test_tables_of_different_units_do_not_merge() -> None:
    rebuilt = EvaluationMetadata.from_table(_intervals([_covers()], _assets(_copula())))
    today = EvaluationMetadata.from_table(
        _intervals([_covers()], _assets(Independent()))
    )
    with pytest.raises(EvaluationError, match="their method, unit"):
        EvaluationMetadata.merge(rebuilt, today)
    merged = EvaluationMetadata.merge(rebuilt, rebuilt)
    assert merged.setup.intervals is not None
    assert merged.setup.intervals["unit"] == "as_built"


def test_a_table_written_before_reads_as_before() -> None:
    # A record of today's unit, with the copula's kind, keeps erratum 34's sentence.
    table = _intervals([_covers()], _assets(_copula()))
    document = json.loads(EvaluationMetadata.from_table(table).to_json())
    document["setup"]["intervals"]["unit"] = "row"
    line = setup_text(EvaluationMetadata.from_json(json.dumps(document)))
    assert UNMEASURED_TEXT in line
    assert REBUILT_TEXT not in line
