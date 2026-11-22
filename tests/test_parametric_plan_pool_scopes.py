"""A pool on the plan under `Joint()` and under `Clustered`.

`pool=` with `.scopes(Joint())` or `.scopes(Clustered(...))`: the assets' pools
are joined by `OutcomeSample.join` on the same (year, model, member) rows, and
pools whose rows differ are refused. The tests build tables only to check their
rows and columns, and that the plan gives what the functions give; nothing here
is a finding about a design. Every number is synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    Budget,
    Clustered,
    ClusteredDesignResult,
    Cover,
    Free,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    Individual,
    Joint,
    JointDesignResult,
    LinearPayout,
    LossModel,
    MetricOptions,
    ModelPool,
    ModelRun,
    OutcomeSample,
    ParametricPlan,
    SampleError,
    StepPayout,
    compare,
    design_joint,
    holdout,
    sample_annual_pool,
)
from crc_sdk.types import SourceProvenance

UNTYPED_PLAN: Any = ParametricPlan
SEED = 11
YEARS = np.arange(2015, 2035)
# Out of byte order in the table: the pool orders them.
MODELS = ("model-b", "model-a")
MEMBER = "r1i1p1f1"
CELLS = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
# Each asset's point, the cell it is matched to and its value in EUR.
ASSETS = {
    "a": ((6.90, 50.90), CELLS[0], 3e6),
    "b": ((7.10, 50.90), CELLS[1], 1e6),
    "c": ((6.90, 51.10), CELLS[2], 2e6),
}
VALUES = {asset: value for asset, (_, _, value) in ASSETS.items()}
REGIONS = {"a": "north", "b": "north", "c": "south"}
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
INDEX = IndexDefinition("rx1day", "mm/day")
STEP = Cover(
    INDEX, StepPayout([60.0, 80.0], [Free(), Free()]), payout_unit="EUR", label="step"
)
LINEAR = Cover(INDEX, LinearPayout(60.0, 120.0, 5e5), payout_unit="EUR", label="linear")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
ARCHIVE = SourceProvenance(
    provider="example",
    dataset="synthetic pooled rx1day",
    licence="CC0-1.0",
    attribution="Synthetic annual maxima",
)
SCOPES = {
    "joint": Joint(),
    "cluster-joint": Clustered(REGIONS, joint=True),
    "cluster-separate": Clustered(REGIONS, joint=False),
}


def _table(gaps: Sequence[tuple[str, int, str]] = ()) -> Any:
    """The layout of NEX-GDDP-CMIP6's `annual_extremes()`, in one table.

    Each gap `(model, year, asset)` is a missing value of that model's run in
    that year, at the cell the asset is matched to.
    """
    tables: list[Any] = []
    cell_years = np.repeat(YEARS, len(CELLS))
    latitudes = np.tile([lat for lat, _ in CELLS], len(YEARS))
    longitudes = np.tile([lon for _, lon in CELLS], len(YEARS))
    for position, model in enumerate(MODELS):
        count = cell_years.shape[0]
        values = np.random.default_rng([7, position]).gumbel(60.0, 12.0, count)
        for gap_model, year, asset in gaps:
            latitude, longitude = ASSETS[asset][1]
            if gap_model == model:
                values[
                    (cell_years == year)
                    & (latitudes == latitude)
                    & (longitudes == longitude)
                ] = np.nan
        tables.append(
            pa.table(
                {
                    "year": pa.array(cell_years, pa.int32()),
                    "latitude": latitudes,
                    "longitude": longitudes,
                    "value": values,
                    "model": [model] * count,
                    "member": [MEMBER] * count,
                    "scenario": ["ssp245"] * count,
                }
            )
        )
    return pa.concat_tables(tables)


def _assets() -> Any:
    return pa.table(
        {
            "asset_id": list(ASSETS),
            "longitude": [point[0] for point, _, _ in ASSETS.values()],
            "latitude": [point[1] for point, _, _ in ASSETS.values()],
            "value_eur": [value for _, _, value in ASSETS.values()],
        }
    )


def _plan(scope: Any, source: Any = None, **options: Any) -> Any:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "value_column": "value_eur",
        "value_unit": "EUR",
        "provenance": ARCHIVE,
        "seed": SEED,
        "pool": ModelPool(),
    }
    arguments.update(options)
    return (
        UNTYPED_PLAN.for_annual_extremes(
            _table() if source is None else source, _assets(), **arguments
        )
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(STEP, LINEAR)
        .objective(MSE(), budget=Budget(ratio=0.8))
        .split(Holdout(0.75))
        .metrics(OPTIONS)
        .scopes(scope)
    )


def _pools_by_hand() -> list[OutcomeSample]:
    """Each asset's pool, built as the plan builds it, with the plan's generators."""
    table = _table()
    children = np.random.SeedSequence(SEED).spawn(len(ASSETS))
    pools: list[OutcomeSample] = []
    for (asset, (point, _, value)), child in zip(ASSETS.items(), children):
        runs = [
            ModelRun(
                AnnualSeries.nearest_cell(
                    table.filter(pc.equal(table.column("model"), model)),
                    longitude=point[0],
                    latitude=point[1],
                    unit="mm/day",
                    source=ARCHIVE,
                ),
                model=model,
                member=MEMBER,
                pathway="ssp245",
            )
            for model in MODELS
        ]
        pools.append(
            sample_annual_pool(
                runs,
                loss_model=LossModel(RAMP, value=value, value_unit="EUR"),
                basis_risk=ImpactNoise(0.3, upper=1.0),
                asset_id=asset,
                rng=np.random.default_rng(child),
            )
        )
    return pools


def _pool_of(sample: OutcomeSample) -> dict[str, Any]:
    pool = sample.provenance.pool
    assert pool is not None
    return dict(pool)


def _model_years(sample: OutcomeSample) -> list[tuple[int, str, str]]:
    """The (year, model, member) of each row of a pooled sample, in row order."""
    pool = _pool_of(sample)
    runs = pool["runs"]
    return [
        (int(year), runs[run]["model"], runs[run]["member"])
        for year, run in zip(pool["year"], pool["run"], strict=True)
    ]


def _expected_loss(block: OutcomeSample, assets: Sequence[str]) -> float:
    """Σ_j E[L_j] over `assets` on `block`, draw by draw."""
    columns = [block.asset_ids.index(asset) for asset in assets]
    return float(block.probability @ block.loss[:, columns].sum(axis=1))


def _written_for_each_asset(cover: Cover) -> dict[str, Cover]:
    """`cover` written for each asset, as the plan binds a cover to every asset."""
    return {
        asset: Cover(
            INDEX,
            cover.payout,
            payout_unit=cover.payout_unit,
            asset_id=asset,
            label=cover.label,
        )
        for asset in ASSETS
    }


def _by_hand(made: Any, joined: OutcomeSample) -> Any:
    """`compare`'s rows of a design and of the linear cover, on both blocks."""
    train, test = holdout(joined, 0.75)
    entries = [made, _written_for_each_asset(LINEAR)]
    return pa.concat_tables(
        [
            compare(
                entries, train, split="in_sample", design_sample=train, options=OPTIONS
            ),
            compare(
                entries,
                test,
                split="out_of_sample",
                design_sample=train,
                options=OPTIONS,
            ),
        ]
    )


def _same_rows(planned: Any, by_hand: Any) -> None:
    """The plan's rows are `compare`'s, but for the design's label."""
    columns = [name for name in by_hand.column_names if name != "cover_label"]
    assert planned.select(columns).to_pylist() == by_hand.select(columns).to_pylist()


def _rows(table: Any, label: str, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in table.to_pylist()
        if row["cover_label"] == label and row["split"] == split
    ]


# -- the samples ---------------------------------------------------------------


@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_the_joint_sample_is_the_join_of_each_asset_s_pool(scope: Any) -> None:
    plan = _plan(scope)
    joined = plan.joint_sample()
    by_hand = OutcomeSample.join(_pools_by_hand())
    assert joined.content_hash == by_hand.content_hash
    assert joined.lineage == by_hand.lineage
    assert joined.asset_ids == ("a", "b", "c")
    # Two models of 20 years: 40 rows at 1/40 each, numbered from 0.
    assert len(joined) == 2 * len(YEARS) == 40
    assert np.array_equal(joined.period, np.arange(40))
    assert np.array_equal(joined.probability, np.full(40, 1.0 / 40))
    assert joined.provenance.source_kind == "annual"
    assert joined.provenance.to_document()["dependence"] == {"kind": "data"}
    pool = _pool_of(joined)
    assert pool["pathway"] == "ssp245"
    assert [(run["model"], run["member"]) for run in pool["runs"]] == [
        ("model-a", MEMBER),
        ("model-b", MEMBER),
    ]


@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_samples_are_each_asset_s_view_of_the_joint_sample(scope: Any) -> None:
    plan = _plan(scope)
    samples = plan.samples()
    joined = plan.joint_sample()
    assert list(samples) == ["a", "b", "c"]
    for own, (asset, sample) in zip(_pools_by_hand(), samples.items(), strict=True):
        assert sample.asset_ids == (asset,)
        assert sample.content_hash == joined.asset(asset).content_hash
        assert sample.content_hash == own.content_hash
        assert _model_years(sample) == _model_years(joined)


@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_the_joined_rows_are_one_model_year_at_every_asset(scope: Any) -> None:
    joined = _plan(scope).joint_sample()
    rows = _model_years(joined)
    # In (year, model, member) order, the models in byte order, each once.
    assert rows == sorted(rows)
    assert rows == [
        (int(year), model, MEMBER) for year in YEARS for model in sorted(MODELS)
    ]
    # Row k of every asset's column is the same model-year: the value the
    # model's run holds that year at the cell the asset is matched to.
    table = _table()
    recorded = {
        (row["model"], row["year"], row["latitude"], row["longitude"]): row["value"]
        for row in table.to_pylist()
    }
    for column, (_, cell, _) in enumerate(ASSETS.values()):
        expected = [recorded[(model, year, *cell)] for year, model, _ in rows]
        assert np.array_equal(joined.index[:, column], expected)


# -- Joint() -------------------------------------------------------------------


def test_joint_designs_on_a_pool_are_the_function_route_s() -> None:
    plan = _plan(Joint())
    joined = plan.joint_sample()
    train, _ = holdout(joined, 0.75)
    (planned,) = plan.designs()
    assert isinstance(planned, JointDesignResult)
    assert planned.design_sample.content_hash == train.content_hash
    # Holdout cuts years, not rows: 15 of the 20 years, of every model.
    assert sorted({year for year, _, _ in _model_years(train)}) == list(YEARS[:15])
    made = design_joint(
        list(_written_for_each_asset(STEP).values()),
        train,
        MSE(),
        budget=Budget(ratio=0.8),
    )
    assert planned.design_id == made.design_id
    assert planned.parameters == made.parameters
    assert planned.premium == made.premium
    assert list(planned.covers) == ["a", "b", "c"]


def test_the_total_rows_under_joint_are_the_function_route_s(tmp_path: Path) -> None:
    plan = _plan(Joint())
    result = plan.write_parquet(tmp_path / "joint.parquet")
    joined = plan.joint_sample()
    assert result.joint_sample is not None
    assert result.joint_sample.content_hash == joined.content_hash
    (planned,) = result.designs
    made = design_joint(
        list(_written_for_each_asset(STEP).values()),
        holdout(joined, 0.75)[0],
        MSE(),
        budget=Budget(ratio=0.8),
    )
    _same_rows(result.table, _by_hand(made, joined))
    blocks = dict(zip(("in_sample", "out_of_sample"), holdout(joined, 0.75)))
    for split, block in blocks.items():
        # The joint design: one "total" row, and no "asset" row.
        (total,) = _rows(result.table, "step (mse)", split)
        assert (total["level"], total["scope"]) == ("total", "joint")
        assert total["design_id"] == planned.design_id
        assert total["premium"] == planned.premium
        assert total["sample_hash"] == block.content_hash
        assert total["n_rows"] == len(block) == (30 if split == "in_sample" else 10)
        assert total["expected_loss"] == pytest.approx(
            _expected_loss(block, list(ASSETS)), rel=1e-12
        )
        # The resolved cover: one contract per asset, and their total.
        rows = _rows(result.table, "linear", split)
        assert [(row["level"], row["scope"], row["asset_id"]) for row in rows] == [
            ("asset", "individual", "a"),
            ("asset", "individual", "b"),
            ("asset", "individual", "c"),
            ("total", "individual", None),
        ]
        *assets, linear_total = rows
        assert linear_total["design_id"] is None
        assert linear_total["premium"] == pytest.approx(
            sum(row["premium"] for row in assets), rel=1e-12
        )
        assert linear_total["expected_loss"] == pytest.approx(
            total["expected_loss"], rel=1e-12
        )


# -- Clustered -----------------------------------------------------------------


@pytest.mark.parametrize("joint", [True, False], ids=["joint", "separate"])
def test_cluster_designs_on_a_pool_are_the_function_route_s(joint: bool) -> None:
    plan = _plan(Clustered(REGIONS, joint=joint))
    train, _ = holdout(plan.joint_sample(), 0.75)
    (planned,) = plan.designs()
    assert isinstance(planned, ClusteredDesignResult)
    assert planned.design_sample.content_hash == train.content_hash
    scope = Clustered(REGIONS, values=VALUES, joint=joint)
    made = design_joint(
        [STEP, STEP], train, MSE(), budget=Budget(ratio=0.8), scope=scope
    )
    assert planned.design_id == made.design_id
    assert planned.parameters == made.parameters
    assert list(planned.covers) == ["north", "south"]
    north = planned.covers["north"].cluster_index
    south = planned.covers["south"].cluster_index
    assert north is not None and dict(north.weights) == {"a": 0.75, "b": 0.25}
    assert south is not None and dict(south.weights) == {"c": 1.0}


@pytest.mark.parametrize("joint", [True, False], ids=["joint", "separate"])
def test_the_cluster_and_total_rows_on_a_pool(tmp_path: Path, joint: bool) -> None:
    plan = _plan(Clustered(REGIONS, joint=joint))
    result = plan.write_parquet(tmp_path / "cluster.parquet")
    joined = plan.joint_sample()
    assert result.joint_sample is not None
    assert result.joint_sample.content_hash == joined.content_hash
    (planned,) = result.designs
    scope = Clustered(REGIONS, values=VALUES, joint=joint)
    made = design_joint(
        [STEP, STEP],
        holdout(joined, 0.75)[0],
        MSE(),
        budget=Budget(ratio=0.8),
        scope=scope,
    )
    _same_rows(result.table, _by_hand(made, joined))
    blocks = dict(zip(("in_sample", "out_of_sample"), holdout(joined, 0.75)))
    for split, block in blocks.items():
        rows = _rows(result.table, "step (mse)", split)
        assert {row["scope"] for row in rows} == {"cluster"}
        assert all(row["asset_id"] is None for row in rows)
        assert all(row["sample_hash"] == block.content_hash for row in rows)
        *clusters, total = rows
        assert (total["level"], total["cluster_id"]) == ("total", None)
        assert total["design_id"] == planned.design_id
        assert total["expected_loss"] == pytest.approx(
            _expected_loss(block, list(ASSETS)), rel=1e-12
        )
        if joint:
            # One design problem on the owner's total: no row per cluster.
            assert clusters == []
            assert total["premium"] == planned.premium
            continue
        # One design problem per cluster: its own row, its own premium.
        assert [(row["level"], row["cluster_id"]) for row in clusters] == [
            ("cluster", "north"),
            ("cluster", "south"),
        ]
        members = {"north": ["a", "b"], "south": ["c"]}
        for row in clusters:
            own = planned.clusters[row["cluster_id"]]
            assert row["premium"] == own.premium
            assert row["expected_loss"] == pytest.approx(
                _expected_loss(block, members[row["cluster_id"]]), rel=1e-12
            )
        assert total["premium"] == pytest.approx(
            sum(row["premium"] for row in clusters), rel=1e-12
        )
    # A resolved cover is one contract per asset, as under Joint().
    linear = _rows(result.table, "linear", "out_of_sample")
    assert [(row["level"], row["scope"]) for row in linear] == [
        ("asset", "individual"),
        ("asset", "individual"),
        ("asset", "individual"),
        ("total", "individual"),
    ]


# -- pools whose rows differ ---------------------------------------------------

# Under missing="drop_year" each asset's pool drops the years one of its runs
# lacks at the asset's cell, so the assets' pools may hold different rows.
# Asset b lacks model-b's 2021 and the others model-a's 2020: 38 rows each,
# but not the same model-years.
OTHER_YEARS = (("model-a", 2020, "a"), ("model-b", 2021, "b"), ("model-a", 2020, "c"))
# Asset a lacks model-a's 2020, and the others hold every year: 38 rows and 40.
FEWER_YEARS = (("model-a", 2020, "a"),)
EXECUTIONS = {
    "samples": lambda plan, path: plan.samples(),
    "joint_sample": lambda plan, path: plan.joint_sample(),
    "designs": lambda plan, path: plan.designs(),
    "write_parquet": lambda plan, path: plan.write_parquet(path),
}


@pytest.mark.parametrize("execute", list(EXECUTIONS.values()), ids=list(EXECUTIONS))
@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_pools_of_other_model_years_are_refused(
    tmp_path: Path, scope: Any, execute: Any
) -> None:
    plan = _plan(scope, _table(OTHER_YEARS), missing="drop_year")
    plan.explain()  # reads nothing, so the rows are not known yet
    path = tmp_path / "refused.parquet"
    with pytest.raises(
        SampleError,
        match=r"join needs pools of the same \(year, model, member\) rows, because "
        r"draw numbers alone would not line them up, but those of asset 'b' differ "
        r"from those of asset 'a'",
    ):
        execute(plan, path)
    assert not path.exists()


@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_pools_of_other_lengths_are_refused(tmp_path: Path, scope: Any) -> None:
    plan = _plan(scope, _table(FEWER_YEARS), missing="drop_year")
    path = tmp_path / "refused.parquet"
    with pytest.raises(
        SampleError,
        match=r"join needs identical period columns, but asset 'b' differs from "
        r"asset 'a'",
    ):
        plan.write_parquet(path)
    assert not path.exists()


@pytest.mark.parametrize("gaps", [OTHER_YEARS, FEWER_YEARS], ids=["other", "fewer"])
def test_the_same_pools_are_designed_asset_by_asset(gaps: Any) -> None:
    # The refusal is the join's: on its own, each asset's pool is built.
    plan = _plan(Individual(), _table(gaps), missing="drop_year")
    samples = plan.samples()
    dropped = {asset: set() for asset in ASSETS} | {
        asset: {year} for _, year, asset in gaps
    }
    for asset, sample in samples.items():
        assert len(sample) == 2 * (len(YEARS) - len(dropped[asset]))
        years = {year for year, _, _ in _model_years(sample)}
        assert not years & dropped[asset]
    assert len(plan.designs()) == len(ASSETS)


@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_pools_that_drop_the_same_years_join(scope: Any) -> None:
    gaps = (("model-a", 2020, "a"), ("model-b", 2020, "b"), ("model-a", 2020, "c"))
    plan = _plan(scope, _table(gaps), missing="drop_year")
    joined = plan.joint_sample()
    assert len(joined) == 2 * (len(YEARS) - 1)
    assert 2020 not in {year for year, _, _ in _model_years(joined)}
    assert len(plan.designs()) == 1


@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_a_stated_years_resolves_pools_of_other_model_years(scope: Any) -> None:
    years = list(range(2022, 2035))
    plan = _plan(scope, _table(OTHER_YEARS), missing="drop_year", years=years)
    joined = plan.joint_sample()
    assert sorted({year for year, _, _ in _model_years(joined)}) == years
    assert len(joined) == 2 * len(years)
    assert len(plan.designs()) == 1
