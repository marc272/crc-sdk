"""`bootstrap_intervals` re-deals the rows `Comonotonic()` and `GaussianCopula` built.

On a row that reads two columns or more of a block one of these models moved,
each replicate redraws each column's draws on its own and deals them to the
rows again by the model's ranks: the identity of ranks under `Comonotonic()`,
and under `GaussianCopula` the ranks of its latent normal scores, which the
sample keeps. These tests pin the re-deal on small blocks against a separate
computation from the same generator, across assets and hazards, on a window,
on pooled folds and with the held form, check the layout of the draws, that
every other row keeps today's draws and numbers, what keeps today's method
and what the record and the setup text say. Every number is synthetic; the
coverage of these intervals is measured in a separate study, not here.
"""

from __future__ import annotations

import dataclasses
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    CappedSum,
    Comonotonic,
    Cover,
    CoverPackage,
    EvaluationMetadata,
    Free,
    GaussianCopula,
    HazardNoise,
    IndexDefinition,
    LinearPayout,
    LossModel,
    MetricOptions,
    OutcomeSample,
    PurePremium,
    StepPayout,
    bootstrap_intervals,
    compare,
    design,
    design_joint,
    holdout,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric.intervals import _block, _plan, _redeal_draws
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
N = 12
FEW = MetricOptions(betas=(0.9,), return_periods=(5.0,))
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
INDICES = {
    "rain": IndexDefinition("rain", "mm/day", aggregation="annual_maximum"),
    "wind": IndexDefinition("wind", "m/s", aggregation="annual_maximum"),
}
HELD = HazardNoise(0.3, hold="hazard_distribution")
NAMES = [column.name for column in metric_columns(FEW)]
RHO = 0.6

REBUILT_TEXT = "Each resample builds those rows again as the block built them"
COMONOTONIC_TEXT = "deals them to the rows again rank by rank"
GAUSSIAN_TEXT = "again by the ranks of the copula's latent normal scores"
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


SEEDS = (11, 12)


def _assets(
    dependence: Any,
    basis_risk: Any = None,
    *,
    n: int = N,
    seeds: tuple[int, int] = SEEDS,
    curves: Any = None,
) -> OutcomeSample:
    return sample_curves(
        {asset: _curve(asset) for asset in ASSETS} if curves is None else curves,
        dependence=dependence,
        metadata={asset: _metadata() for asset in ASSETS},
        n_draws=n,
        rngs={a: np.random.default_rng(seed) for a, seed in zip(ASSETS, seeds)},
        loss_models={asset: _loss(asset) for asset in ASSETS},
        basis_risk=HazardNoise(0.3) if basis_risk is None else basis_risk,
    )


def _hazards(dependence: Any, basis_risk: Any = None) -> OutcomeSample:
    """Two hazards at one asset, their losses damage ratios, under CappedSum(1)."""
    return sample_curve_hazards(
        {name: _curve(name) for name in HAZARDS},
        dependence=dependence,
        metadata={name: _metadata(UNITS[name]) for name in HAZARDS},
        n_draws=N,
        rngs={name: np.random.default_rng(21 + j) for j, name in enumerate(HAZARDS)},
        loss_models={
            name: LossModel(RAMP, name=f"ratio of {name}") for name in HAZARDS
        },
        combine=CappedSum(1.0),
        basis_risk=HazardNoise(0.3) if basis_risk is None else basis_risk,
        asset_id="site",
    )


def _package() -> CoverPackage:
    return CoverPackage(
        {
            name: _cover(index=INDICES[name], label=f"on {name}", limit=0.6, unit="1")
            for name in HAZARDS
        },
        label="package",
    )


def _intervals(
    entries: list[Any], on: Any, *, seed: int = 3, n_bootstrap: int = 4, **kw: Any
) -> Any:
    arguments: dict[str, Any] = (
        {"split": "in_sample"} if isinstance(on, OutcomeSample) else {}
    )
    return UNTYPED_INTERVALS(
        entries,
        on,
        rng=np.random.default_rng(seed),
        n_bootstrap=n_bootstrap,
        options=FEW,
        **{**arguments, **kw},
    )


def _record(table: Any) -> dict[str, Any]:
    intervals = EvaluationMetadata.from_table(table).setup.intervals
    assert intervals is not None
    return intervals


def _line(table: Any) -> str:
    return next(
        line for line in setup_text(table).splitlines() if line.startswith("Intervals:")
    )


def _bounds(table: Any, level: str, label: str | None = None) -> dict[str, Any]:
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


def _premium(payouts: list[np.ndarray], probability: np.ndarray) -> float:
    return math.fsum(PurePremium().premium(payout, probability) for payout in payouts)


# -- a separate computation of the re-deal -------------------------------------


def _latent_ranks(seeds: tuple[int, ...], rho: float, n: int) -> np.ndarray:
    """The ranks of y = z L^T, from each column's own uniforms, row by row."""
    quantile = NormalDist().inv_cdf
    tiny = 2.0**-53
    z = np.column_stack(
        [
            [
                quantile(min(max(u, tiny), 1.0 - tiny))
                for u in np.random.default_rng(seed).random(n)
            ]
            for seed in seeds
        ]
    )
    y = z @ np.array([[1.0, 0.0], [rho, math.sqrt(1.0 - rho * rho)]]).T
    return np.argsort(np.argsort(y, axis=0, kind="stable"), axis=0, kind="stable")


def _expected_sources(
    keys: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> list[np.ndarray]:
    """Each column's source rows as a replicate deals them, written apart.

    `keys[:, j]` orders column j's draws: their rank there. A column's redrawn
    draws are sorted by it, copies of one draw in the order drawn, and row i
    takes the draw of the rank `keys[i, j]` holds among the column's keys.
    """
    n, columns = keys.shape
    sources = []
    for j in range(columns):
        drawn = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int64)
        place = {
            row: p for p, row in enumerate(sorted(range(n), key=lambda r: keys[r, j]))
        }
        source = np.empty((n_bootstrap, n), dtype=np.int64)
        for b in range(n_bootstrap):
            ranked = sorted(range(n), key=lambda d: (keys[drawn[b, d], j], d))
            for row in range(n):
                source[b, row] = drawn[b, ranked[place[row]]]
        sources.append(source)
    return sources


def _comonotonic_keys(sample: OutcomeSample) -> np.ndarray:
    """Under Comonotonic() every column's rank of a row is the first column's."""
    rank = np.argsort(np.argsort(sample.index[:, 0], kind="stable"), kind="stable")
    return np.column_stack([rank] * len(sample.asset_ids))


def _total_on(sample: OutcomeSample, sources: list[np.ndarray]) -> dict[str, Any]:
    """The metrics of the `"total"` row of `_covers()` on one re-dealt replicate."""
    probability = np.full(len(sample), 1.0 / len(sample))
    loss = sum(sample.loss[sources[j][0], j] for j in range(2))
    payouts = [
        _covers()[asset].payout_values(sample.index[sources[j][0], j])
        for j, asset in enumerate(ASSETS)
    ]
    return metric_values(
        np.asarray(loss),
        payouts[0] + payouts[1],
        probability,
        premium=_premium(payouts, probability),
        options=FEW,
    )


# -- re-dealing across assets --------------------------------------------------


def test_a_comonotonic_total_is_re_dealt_rank_by_rank() -> None:
    sample = _assets(Comonotonic())
    rng, copy = np.random.default_rng(8), np.random.default_rng(8)
    table = _intervals([_covers()], sample, n_bootstrap=1, seed=8)
    copy.integers(0, N, size=(1, N), dtype=np.int64)  # today's draw, by year
    sources = _expected_sources(_comonotonic_keys(sample), copy, 1)
    # The draws of equal rank in every column share a row: the replicate's
    # rows, ordered by the first column, are ordered by the second too.
    first, second = (sample.index[sources[j][0], j] for j in range(2))
    order = np.lexsort((second, first))
    assert (np.diff(second[order]) >= 0).all()
    _check_single_replicate(_bounds(table, "total"), _total_on(sample, sources))
    # Today's draw, then one per column: the generator is where the copy is.
    rng.integers(0, N, size=(1, N), dtype=np.int64)
    for _ in ASSETS:
        rng.integers(0, N, size=(1, N), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    record = _record(table)
    assert record["unit"] == "as_built"
    assert record["built_together"] == ["comonotonic"]
    assert "kept_today" not in record


@pytest.mark.parametrize("rho", [RHO, -0.8])
def test_a_gaussian_total_is_re_dealt_by_its_latent_ranks(rho: float) -> None:
    sample = _assets(GaussianCopula(rho))
    keys = _latent_ranks(SEEDS, rho, N)
    assert np.array_equal(sample.latent_ranks, keys)  # type: ignore[arg-type]
    table = _intervals([_covers()], sample, n_bootstrap=1, seed=8)
    copy = np.random.default_rng(8)
    copy.integers(0, N, size=(1, N), dtype=np.int64)
    sources = _expected_sources(keys, copy, 1)
    # Each row's draw holds the rank of its latent score in the column.
    for j in range(2):
        redrawn = np.sort(sample.index[sources[j][0], j])
        assert np.array_equal(sample.index[sources[j][0], j], redrawn[keys[:, j]])
    _check_single_replicate(_bounds(table, "total"), _total_on(sample, sources))
    assert _record(table)["built_together"] == ["gaussian_copula"]
    assert _record(table)["unit"] == "as_built"


@pytest.mark.parametrize(
    "model", [Comonotonic, lambda: GaussianCopula(RHO)], ids=["comonotonic", "gaussian"]
)
def test_a_replicate_that_draws_each_draw_once_is_the_block(model: Any) -> None:
    # Dealt again by the ranks the block holds, each draw drawn once in order
    # goes back to its own row: the re-deal repeats the step that built them.
    sample = _assets(model(), n=40)
    builder = _plan(_evaluated([_covers()], sample)).builder
    assert builder is not None

    class _InOrder:
        def integers(self, low: int, high: int, size: Any, dtype: Any) -> np.ndarray:
            return np.tile(np.arange(high, dtype=dtype), (size[0], 1))

    sources = _redeal_draws(builder, _InOrder(), 2)  # type: ignore[arg-type]
    for source in sources:
        assert (source == np.arange(40)).all()


def _hurdle(curve_kind: str = "hurdle") -> dict[str, Any]:
    """A curve row whose draws below 0.4 are its atom, 50 mm/day: they tie."""
    return {
        **_horizon(2045, 60.0),
        "curve_kind": curve_kind,
        "curve_atom_probability": 0.4 if curve_kind == "hurdle" else None,
        "curve_atom_location": 50.0 if curve_kind == "hurdle" else None,
    }


@pytest.mark.parametrize(
    "model", [Comonotonic, lambda: GaussianCopula(RHO)], ids=["comonotonic", "gaussian"]
)
def test_at_an_atom_the_re_deal_keeps_the_places_the_model_gave(model: Any) -> None:
    # Site a's draws tie at the atom, each with its own loss (HazardNoise moves
    # each draw's hazard at the asset). The model placed them by their uniforms:
    # under GaussianCopula the sample keeps those ranks, which its index doesn't
    # give, and under Comonotonic() site b's index orders them. A replicate that
    # draws each draw once in order still rebuilds the block row for row.
    sample = _assets(
        model(), n=40, curves={"site-a": _hurdle(), "site-b": _hurdle("fitted")}
    )
    tied = sample.index[:, 0] == 50.0
    assert tied.sum() > 10 and np.unique(sample.loss[tied, 0]).size > 10
    if sample.latent_ranks is not None:
        assert np.array_equal(sample.latent_ranks, _latent_ranks(SEEDS, RHO, 40))
        by_index = np.argsort(np.argsort(sample.index[:, 0], kind="stable"))
        assert not np.array_equal(by_index, sample.latent_ranks[:, 0])
    builder = _plan(_evaluated([_covers()], sample)).builder
    assert builder is not None

    class _InOrder:
        def integers(self, low: int, high: int, size: Any, dtype: Any) -> np.ndarray:
            return np.tile(np.arange(high, dtype=dtype), (size[0], 1))

    for source in _redeal_draws(builder, _InOrder(), 1):  # type: ignore[arg-type]
        assert (source == np.arange(40)).all()
    assert _record(_intervals([_covers()], sample))["unit"] == "as_built"


def _evaluated(entries: list[Any], sample: OutcomeSample, **kw: Any) -> Any:
    arguments: dict[str, Any] = {
        "split": "in_sample",
        "fold": 0,
        "pricing": PurePremium(),
        "premiums": None,
        "options": FEW,
        "design_sample": None,
        "allocation": None,
        "values": None,
    }
    return _block(entries, sample, **{**arguments, **kw})


def test_an_assets_own_row_keeps_todays_numbers() -> None:
    # Today's draw stays the first, so an asset's own row reads it as before:
    # the same intervals as on that asset's own view, resampled by year.
    for model in (Comonotonic(), GaussianCopula(RHO)):
        sample = _assets(model, n=30)
        table = _intervals([_covers()], sample, n_bootstrap=50)
        for asset in ASSETS:
            alone = _intervals(
                [_cover(asset, label=asset)], sample.asset(asset), n_bootstrap=50
            )
            assert _bounds(table, "asset", asset) == _bounds(alone, "asset", asset)
            assert _record(alone)["unit"] == "year"


def test_a_joint_design_is_re_dealt_with_its_allocated_rows() -> None:
    sample = _assets(Comonotonic())
    steps = [
        Cover(INDEX, StepPayout([60.0], [Free()]), payout_unit="EUR", asset_id=asset)
        for asset in ASSETS
    ]
    joint = design_joint(steps, sample, MSE())
    table = _intervals(
        [joint],
        sample,
        seed=1,
        n_bootstrap=1,
        allocation="loss",
        values={"site-a": 2.0, "site-b": 1.0},
    )
    copy = np.random.default_rng(1)
    copy.integers(0, N, size=(1, N), dtype=np.int64)
    sources = _expected_sources(_comonotonic_keys(sample), copy, 1)
    losses = np.column_stack([sample.loss[sources[j][0], j] for j in range(2)])
    payout = sum(
        joint.covers[asset].payout_values(sample.index[sources[j][0], j])
        for j, asset in enumerate(ASSETS)
    )
    wide = compare([joint], sample, split="in_sample", options=FEW)
    premium = next(
        row["premium"] for row in wide.to_pylist() if row["level"] == "total"
    )
    expected = metric_values(
        losses.sum(axis=1),
        np.asarray(payout),
        np.full(N, 1.0 / N),
        premium=float(premium),
        options=FEW,
    )
    _check_single_replicate(
        _bounds(table, "total"), expected, frozenset({"unidentified_share"})
    )
    flagged = {row["level"] for row in table.to_pylist() if row["built_together"]}
    assert flagged == {"total", "asset_by_loss"}
    assert _record(table)["unit"] == "as_built"


def _horizon(horizon: int, location: float) -> dict[str, Any]:
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


@pytest.mark.parametrize(
    "model", [Comonotonic, lambda: GaussianCopula(RHO)], ids=["comonotonic", "gaussian"]
)
def test_a_window_is_re_dealt_within_each_member(model: Any) -> None:
    # Row d reads member d mod 2 in every column, and the model reorders within
    # each member's rows, so a replicate keeps every column's draws in their
    # member, and each row the rank it held among its member's rows.
    window = _assets(
        model(),
        n=2 * N,
        curves={
            "site-a": [_horizon(2045, 50.0), _horizon(2055, 60.0)],
            "site-b": [_horizon(2045, 55.0), _horizon(2055, 70.0)],
        },
    )
    builder = _plan(_evaluated([_covers()], window)).builder
    assert builder is not None
    assert [rows.tolist() for rows, _ in builder.members] == [
        list(range(0, 2 * N, 2)),
        list(range(1, 2 * N, 2)),
    ]
    sources = _redeal_draws(builder, np.random.default_rng(0), 30)
    for j in range(2):
        assert (sources[j] % 2 == np.arange(2 * N) % 2).all()
        for rows, _ in builder.members:
            held = np.argsort(np.argsort(window.index[rows, j]))
            for b in range(30):
                taken = window.index[sources[j][b][rows], j]
                assert np.array_equal(taken, np.sort(taken)[held])
    assert _record(_intervals([_covers()], window))["unit"] == "as_built"


# -- re-dealing across hazards -------------------------------------------------


def test_a_gaussian_package_across_hazards_is_re_dealt_with_its_rule() -> None:
    sample = _hazards(GaussianCopula(-0.5))
    components = sample.index_components
    assert components is not None
    keys = _latent_ranks((21, 22), -0.5, N)
    assert np.array_equal(sample.latent_ranks, keys)  # type: ignore[arg-type]
    table = _intervals([_package()], sample, seed=4, n_bootstrap=1)
    copy = np.random.default_rng(4)
    copy.integers(0, N, size=(1, N), dtype=np.int64)
    sources = _expected_sources(keys, copy, 1)
    damage = {
        name: components[name].damage[sources[j][0], 0]  # type: ignore[index]
        for j, name in enumerate(HAZARDS)
    }
    combined = np.minimum(damage["rain"] + damage["wind"], 1.0)
    payouts = [
        _package()
        ._items[name]
        .cover.payout_values(components[name].values[sources[j][0], 0])
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
    on_rain = metric_values(
        combined,
        payouts[0],
        probability,
        premium=_premium([payouts[0]], probability),
        options=FEW,
    )
    _check_single_replicate(_bounds(table, "asset", "on rain"), on_rain)
    assert _record(table)["unit"] == "as_built"


def test_a_comonotonic_package_across_hazards_is_re_dealt() -> None:
    table = _intervals([_package()], _hazards(Comonotonic()))
    record = _record(table)
    assert record["unit"] == "as_built"
    assert record["built_together"] == ["comonotonic"]
    assert all(table.column("built_together").to_pylist())


def test_a_held_block_is_re_dealt_and_re_ranked() -> None:
    for model, kind in ((Comonotonic(), "comonotonic"), (GaussianCopula(RHO), None)):
        table = _intervals([_covers()], _assets(model, HELD))
        record = _record(table)
        assert record["unit"] == "as_built"
        assert record["built_together"][-1] == "hazard_distribution"
        assert "kept_today" not in record
        line = _line(table)
        assert RERANKED_TEXT in line
        assert (COMONOTONIC_TEXT if kind else GAUSSIAN_TEXT) in line
        assert UNMEASURED_TEXT not in line


# -- pooled folds ----------------------------------------------------------------


def _fold_designs(folds: Any, payout: Any = None) -> list[dict[str, Any]]:
    """Each fold's designs, one per asset on its design block's asset."""
    return [
        {
            asset: design(
                Cover(
                    INDEX,
                    StepPayout([60.0], [Free()]) if payout is None else payout,
                    payout_unit="EUR",
                    asset_id=asset,
                    label=asset,
                ),
                train.asset(asset),
                MSE(),
            )
            for asset in ASSETS
        }
        for train, _ in folds
    ]


@pytest.mark.parametrize(
    "model", [Comonotonic, lambda: GaussianCopula(RHO)], ids=["comonotonic", "gaussian"]
)
def test_pooled_rows_are_re_dealt_over_the_joined_blocks(model: Any) -> None:
    # The held-out blocks are cuts of one deal over every row, so they are
    # re-dealt together, as the sample they were cut from, and each row of a
    # replicate pays its own fold's contract.
    sample = _assets(model(), n=24)
    folds = year_block_kfold(sample, 3)
    designs = _fold_designs(folds)
    rng = np.random.default_rng(6)
    table = UNTYPED_INTERVALS([designs], folds, rng=rng, n_bootstrap=1, options=FEW)
    copy = np.random.default_rng(6)
    copy.integers(0, 24, size=(1, 24), dtype=np.int64)
    ranks = sample.latent_ranks
    keys = _comonotonic_keys(sample) if ranks is None else np.asarray(ranks)
    sources = _expected_sources(keys, copy, 1)
    index = [sample.index[sources[j][0], j] for j in range(2)]
    payout, premium = [], []
    for fold, ((_, test), made) in enumerate(zip(folds, designs, strict=True)):
        held_out = np.isin(sample.period, test.period)
        payout.append(
            sum(
                made[asset].cover.payout_values(index[j][held_out])
                for j, asset in enumerate(ASSETS)
            )
        )
        rows = compare(
            [made], test, split="out_of_sample", fold=fold, options=FEW
        ).to_pylist()
        total = next(row for row in rows if row["level"] == "total")
        premium.append(np.full(len(test), total["premium"]))
    expected = metric_values(
        np.asarray(sum(sample.loss[sources[j][0], j] for j in range(2))),
        np.concatenate(payout),
        np.full(24, 1.0 / 24.0),
        premium=np.concatenate(premium),
        options=FEW,
    )
    # The share of draws reading a filled amount is checked where fills are.
    _check_single_replicate(
        _bounds(table, "total"), expected, frozenset({"unidentified_share"})
    )
    # Today's draw, then one per column of the joined blocks.
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    record = _record(table)
    assert record["unit"] == "as_built"
    assert "kept_today" not in record


@pytest.mark.parametrize(
    "model",
    [Comonotonic, lambda: GaussianCopula(-0.8)],
    ids=["comonotonic", "gaussian"],
)
def test_pooled_counts_are_the_one_block_re_deal_when_the_covers_agree(
    model: Any,
) -> None:
    # With the same covers on every fold a pooled row's counts read nothing of
    # a fold, so they are those of the joined blocks re-dealt as one block.
    sample = _assets(model(), n=24)
    folds = year_block_kfold(sample, 3)
    designs = _fold_designs(folds, LinearPayout(55.0, 90.0, Free(60.0, 60.0)))
    pooled = _bounds(_intervals([designs], folds, n_bootstrap=30), "total")
    block = _bounds(_intervals([_covers()], sample, n_bootstrap=30), "total")
    counts = [
        name
        for name in NAMES
        if name.startswith(("payout_frequency", "false_negative_rate"))
    ]
    assert len(counts) == 2
    for name in counts:
        assert pooled[name] == block[name] and None not in pooled[name], name
        assert pooled[name][0] < pooled[name][1], name


def test_pooled_rows_whose_joined_blocks_cant_be_dealt_keep_todays_method() -> None:
    # The folds of a selection join to that selection, itself a cut of a deal.
    sample = _assets(Comonotonic(), n=24)
    folds = year_block_kfold(sample.select(sample.period[:18]), 3)
    rng, copy = np.random.default_rng(6), np.random.default_rng(6)
    table = UNTYPED_INTERVALS(
        [_fold_designs(folds)], folds, rng=rng, n_bootstrap=3, options=FEW
    )
    copy.integers(0, 18, size=(3, 18), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    record = _record(table)
    assert record["unit"] == "year"
    assert record["kept_today"] == [{"kind": "comonotonic", "reason": "cut_of_deal"}]


def test_pooled_asset_rows_keep_todays_numbers() -> None:
    # An asset's own pooled row reads today's draw, as on that asset's folds.
    sample = _assets(GaussianCopula(RHO), n=24)
    folds = year_block_kfold(sample, 3)
    designs = _fold_designs(folds)
    table = UNTYPED_INTERVALS(
        [designs], folds, rng=np.random.default_rng(2), n_bootstrap=30, options=FEW
    )
    assert _record(table)["unit"] == "as_built"
    for asset in ASSETS:
        own = [made[asset] for made in designs]
        alone = tuple((train.asset(asset), test.asset(asset)) for train, test in folds)
        found = UNTYPED_INTERVALS(
            [own], alone, rng=np.random.default_rng(2), n_bootstrap=30, options=FEW
        )
        assert _record(found)["unit"] == "year"
        assert _bounds(table, "asset", asset) == _bounds(found, "asset")


# -- what keeps today's method ---------------------------------------------------


def _todays_bounds(entries: list[Any], sample: OutcomeSample, seed: int, b: int) -> Any:
    """Every row's bounds by year, as today's method gives them: one draw."""
    from crc_sdk.parametric.intervals import _replicates, _year_counts

    evaluated = _evaluated(entries, sample)
    years = evaluated.n_years
    draws = np.random.default_rng(seed).integers(0, years, size=(b, years))
    replicates = _replicates(evaluated, _year_counts(draws, years))
    low, high = np.quantile(replicates, [(1.0 - 0.9) / 2.0, (1.0 + 0.9) / 2.0], axis=0)
    return low.ravel(order="F"), high.ravel(order="F")


def _kept(entries: list[Any], sample: OutcomeSample, kept: list[dict[str, str]]) -> Any:
    rng = np.random.default_rng(5)
    table = UNTYPED_INTERVALS(
        entries, sample, split="in_sample", rng=rng, n_bootstrap=7, options=FEW
    )
    low, high = _todays_bounds(entries, sample, 5, 7)
    found = np.array(
        [np.nan if v is None else v for v in table.column("ci_low").to_pylist()]
    )
    defined = ~np.isnan(found)
    assert np.array_equal(found[defined], low[defined])
    copy = np.random.default_rng(5)
    copy.integers(0, len(sample), size=(7, len(sample)), dtype=np.int64)
    assert rng.integers(0, 2**62) == copy.integers(0, 2**62)
    record = _record(table)
    assert record["unit"] == "year"
    assert record["kept_today"] == kept
    return table


def test_a_gaussian_sample_without_its_ranks_keeps_todays_method() -> None:
    # As a sample written before samples kept the ranks of the latent scores.
    sample = dataclasses.replace(_assets(GaussianCopula(RHO)), latent_ranks=None)
    table = _kept(
        [_covers()],
        sample,
        [{"kind": "gaussian_copula", "reason": "gaussian_without_ranks"}],
    )
    line = _line(table)
    assert UNMEASURED_TEXT in line and REBUILT_TEXT not in line
    assert "doesn't keep the ranks of the GaussianCopula's latent normal" in line


def test_a_gaussian_sample_with_tied_ranks_keeps_todays_method() -> None:
    sample = _assets(GaussianCopula(RHO))
    tied = np.array(sample.latent_ranks)
    tied[1, 1] = tied[0, 1]
    _kept(
        [_covers()],
        dataclasses.replace(sample, latent_ranks=tied),
        [{"kind": "gaussian_copula", "reason": "gaussian_without_ranks"}],
    )


class _MovedIndex:
    """A basis-risk model of the caller's that moves the index, not the hazard."""

    name = "moved index"

    def to_document(self) -> dict[str, Any]:
        return {"kind": "moved_index"}

    def apply(self, index: Any, hazard: Any, impact: Any, rng: Any) -> Any:
        del hazard
        return index + rng.normal(0.0, 20.0, index.shape), impact


def test_comonotonic_indices_that_dont_rise_together_keep_todays_method() -> None:
    # The model moved the draws by their uniforms; the index no longer gives
    # that order, so the ranks Comonotonic() aligned can't be read.
    sample = _assets(Comonotonic(), _MovedIndex())
    table = _kept(
        [_covers()],
        sample,
        [{"kind": "comonotonic", "reason": "index_not_comonotone"}],
    )
    assert "don't rise together over the rows" in _line(table)


def test_a_gaussian_sample_with_a_moved_index_is_still_re_dealt() -> None:
    # Its ranks are kept, so no index is read for them.
    table = _intervals([_covers()], _assets(GaussianCopula(RHO), _MovedIndex()))
    assert _record(table)["unit"] == "as_built"


def test_a_block_moved_by_two_models_keeps_todays_method() -> None:
    sample = _assets(Comonotonic())
    parts = sample.provenance.parts
    assert parts is not None
    moved = {
        asset: {
            **document,
            "selection": [{"kind": "reorder", "dependence": "gaussian_copula"}],
        }
        if asset == "site-a"
        else dict(document)
        for asset, document in parts.items()
    }
    odd = dataclasses.replace(
        sample, provenance=dataclasses.replace(sample.provenance, parts=moved)
    )
    _kept(
        [_covers()],
        odd,
        [{"kind": "comonotonic", "reason": "several_models"}],
    )


@pytest.mark.parametrize(
    ("model", "kind"),
    [(Comonotonic, "comonotonic"), (lambda: GaussianCopula(RHO), "gaussian_copula")],
    ids=["comonotonic", "gaussian"],
)
def test_a_cut_of_a_deal_is_resampled_by_year(model: Any, kind: str) -> None:
    # The model dealt each column over every row of the sample, so the rows of
    # a cut, dealt again alone, would not repeat that deal.
    sample = _assets(model(), n=24)
    for cut in (
        holdout(sample)[1],
        year_block_kfold(sample, 3)[0][1],
        sample.select(sample.period[4:]),
    ):
        table = _kept([_covers()], cut, [{"kind": kind, "reason": "cut_of_deal"}])
        line = _line(table)
        assert UNMEASURED_TEXT in line and REBUILT_TEXT not in line
        assert "the block is a cut of a sample, as a holdout's block" in line


# -- the record and the setup text ------------------------------------------------


@pytest.mark.parametrize(
    ("model", "text"),
    [(Comonotonic, COMONOTONIC_TEXT), (lambda: GaussianCopula(RHO), GAUSSIAN_TEXT)],
    ids=["comonotonic", "gaussian"],
)
def test_the_setup_text_names_the_re_deal(model: Any, text: str) -> None:
    line = _line(_intervals([_covers()], _assets(model())))
    assert "resamples of the evaluated years" in line
    assert REBUILT_TEXT in line and text in line
    assert UNMEASURED_TEXT not in line
    assert (
        'and on the "total" rows of two assets under Comonotonic() and under a '
        "GaussianCopula, on one block and pooled over the folds of year_block_kfold "
        "(re-dealt)"
    ) in line


def test_rows_filtered_to_todays_method_read_as_by_year() -> None:
    table = _intervals([_covers()], _assets(Comonotonic()))
    kept = table.filter(
        np.array([not flag for flag in table.column("built_together").to_pylist()])
    )
    assert _record(kept)["unit"] == "as_built"
    line = _line(kept)
    assert "resamples of the evaluated years" in line
    assert REBUILT_TEXT not in line
