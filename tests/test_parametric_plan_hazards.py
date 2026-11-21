"""ParametricPlan.hazard: several hazards at each asset, end to end.

Every number here is synthetic: small made-up tables of years and curves. The
tests check that the plan composes the function route and records what it read;
nothing here is a finding about a hazard, a rule, a copula or a cover form.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact, TransformContext

from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    MSE,
    AnnualScenario,
    AnnualSeries,
    Budget,
    CappedSum,
    ComonotoneIndexWarning,
    CompositeIndex,
    CompoundDamage,
    Cover,
    CoverPackage,
    CurveScenario,
    EmpiricalCopula,
    EvaluationError,
    ExpectedUncovered,
    Free,
    FreshDraws,
    GaussianCopula,
    Holdout,
    ImpactNoise,
    Independent,
    IndexDefinition,
    Joint,
    KFold,
    LeakageError,
    LinearPayout,
    LinearTrend,
    LossModel,
    MaxOf,
    MetricOptions,
    ModelPool,
    NearestCell,
    OutcomeSample,
    ParametricPlan,
    SampleError,
    ScenarioSet,
    StationColumn,
    StepGrid,
    StepPayout,
    compare,
    compare_scenarios,
    design,
    holdout,
    payout_breakdown,
    sample_annual_hazards,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    split_budget,
    year_block_kfold,
)
from crc_sdk.parametric.sample import _thaw
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import (
    AssetPortfolio,
    HazardDataset,
    ReturnPeriodExtrapolationWarning,
)

UNTYPED_PLAN: Any = ParametricPlan
UNTYPED_COMPARE: Any = compare
UNTYPED_CURVE_SCENARIO: Any = CurveScenario
RAIN = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
HEAT = PiecewiseLinearImpact([32.0, 42.0], [0.0, 0.3])
RAIN_INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
HEAT_INDEX = IndexDefinition("txx", "degC", aggregation="annual_maximum")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
NOISE = ImpactNoise(0.3, upper=1.0)
STATIONS = {"warehouse-a": "ST-1", "depot-b": "ST-2"}
POINTS = {"warehouse-a": (6.96, 50.94), "depot-b": (7.50, 51.00)}
CELLS = {name: point_to_cell(lon, lat, 7) for name, (lon, lat) in POINTS.items()}
YEARS = list(range(1981, 2017))


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        yield


def _provenance(provider: str = "example", dataset: str = "synthetic") -> Any:
    return SourceProvenance(
        provider=provider,
        dataset=dataset,
        version="1",
        licence="CC0-1.0",
        attribution="Synthetic data",
    )


def _assets() -> Any:
    return pa.table(
        {
            "asset_id": list(STATIONS),
            "longitude": [POINTS[name][0] for name in STATIONS],
            "latitude": [POINTS[name][1] for name in STATIONS],
            "station": list(STATIONS.values()),
            "value_eur": [2e6, 3e6],
        }
    )


def _weather(shift: float = 0.0, seed: int = 3) -> Any:
    """Each station's yearly maximum daily rain and temperature, made up."""
    rng = np.random.default_rng(seed)
    n = len(YEARS)
    return pa.table(
        {
            "station": ["ST-1"] * n + ["ST-2"] * n,
            "year": np.array(YEARS * 2, dtype=np.int32),
            "rx1day": rng.gumbel(50.0 + shift, 12.0, 2 * n),
            "txx": rng.gumbel(33.0 + shift / 10.0, 2.0, 2 * n),
        }
    )


WEATHER = _weather()


def _grid() -> Cover:
    return Cover(
        CompositeIndex({"rx1day": RAIN_INDEX, "txx": HEAT_INDEX}),
        StepGrid({"rx1day": [70.0], "txx": [36.0]}, [[0.0, Free()], [Free(), Free()]]),
        payout_unit="EUR",
        label="grid",
    )


def _rain() -> Cover:
    return Cover(
        RAIN_INDEX, StepPayout([70.0], [Free()]), payout_unit="EUR", label="rain"
    )


def _heat() -> Cover:
    return Cover(
        HEAT_INDEX, StepPayout([36.0], [Free()]), payout_unit="EUR", label="heat"
    )


def _either() -> Cover:
    """A resolved cover on both components: the larger of two linear payouts."""
    return Cover(
        CompositeIndex({"rx1day": RAIN_INDEX, "txx": HEAT_INDEX}),
        MaxOf(
            {
                "rx1day": LinearPayout(60.0, 120.0, 4e5),
                "txx": LinearPayout(35.0, 42.0, 4e5),
            }
        ),
        payout_unit="EUR",
        label="either",
    )


def _annual_plan(**overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "impact": RAIN,
        "unit": "mm/day",
        "provenance": _provenance(),
        "value_column": "value_eur",
        "value_unit": "EUR",
        "location": StationColumn("station"),
        "hazard_column": "rx1day",
        "seed": 7,
    }
    arguments.update(overrides)
    source = arguments.pop("source", WEATHER)
    return UNTYPED_PLAN.for_annual_extremes(source, _assets(), **arguments)


def _two_hazards(
    plan: Any, source: Any = WEATHER, hazard_name: str | None = None
) -> Any:
    return (
        plan.hazard("rx1day", tail="upper")
        .hazard(
            "txx",
            source,
            impact=HEAT,
            unit="degC",
            tail="upper",
            provenance=_provenance(),
            hazard_column="txx",
            hazard_name=hazard_name,
        )
        .combine(CompoundDamage())
    )


def _designed(plan: Any, **objective: Any) -> Any:
    terms: dict[str, Any] = {"budget": Budget(ratio=0.8)}
    terms.update(objective)
    return (
        plan.basis_risk(NOISE)
        .covers(_grid(), _rain(), _heat(), _either())
        .package({"rx1day": "rain", "txx": "heat"}, label="separate", shares="own_loss")
        .objective(ExpectedUncovered(), **terms)
        .metrics(OPTIONS)
    )


def _rows_of(table: Any, **match: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in table.to_pylist()
        if all(row[key] == value for key, value in match.items())
    ]


# -- the function route, by hand -------------------------------------------------------


def _series(
    asset_id: str, name: str, unit: str, table: Any = WEATHER, **source: Any
) -> AnnualSeries:
    """One asset's series, read as the plan reads a station's rows."""
    return AnnualSeries.from_table(
        table,
        value=name,
        where={"station": STATIONS[asset_id]},
        station="station",
        unit=unit,
        source={**_provenance().model_dump(mode="json"), **source},
    )


def _loss(impact: Any, asset_id: str, cell: int | None = None) -> LossModel:
    context = None if cell is None else TransformContext(cell=cell)
    return LossModel(
        impact, value_column="value_eur", value_unit="EUR", context=context
    ).for_asset(AssetPortfolio(_assets()), asset_id)


def _annual_by_hand(
    asset_id: str,
    *,
    table: Any = WEATHER,
    scenario: str = "baseline",
    pathway: str | None = None,
    **extra: Any,
) -> OutcomeSample:
    """`sample_annual_hazards` with the asset's child generator of the plan's seed.

    A stated `pathway` is recorded beside each series' source, as the plan does.
    """
    children = np.random.SeedSequence(7).spawn(len(STATIONS))
    child = children[list(STATIONS).index(asset_id)]
    series = {
        "rx1day": _series(asset_id, "rx1day", "mm/day", table),
        "txx": _series(asset_id, "txx", "degC", table),
    }
    if pathway is not None:
        series = {
            name: AnnualSeries(
                each.years,
                each.values,
                each.unit,
                {**_thaw(each.provenance), "pathway": pathway},
            )
            for name, each in series.items()
        }
    sample: OutcomeSample = sample_annual_hazards(
        series,
        tails={"rx1day": "upper", "txx": "upper"},
        loss_models={"rx1day": _loss(RAIN, asset_id), "txx": _loss(HEAT, asset_id)},
        combine=CompoundDamage(),
        basis_risk=NOISE,
        asset_id=asset_id,
        rng=np.random.default_rng(child),
        scenario=scenario,
        **extra,
    )
    return sample


def _entries_by_hand(
    train: OutcomeSample, asset_id: str, budget: Budget | None = Budget(ratio=0.8)
) -> list[Any]:
    """What the plan compares on one design block: the grid, the package, the cover.

    Each template is written for the asset, as the plan writes a cover stated for
    no asset, so the design ids are the plan's.
    """
    shares = None if budget is None else split_budget(budget, train, shares="own_loss")
    grid = design(_for(_grid(), asset_id), train, ExpectedUncovered(), budget=budget)
    rain = design(
        _for(_rain(), asset_id),
        train.component("rx1day"),
        ExpectedUncovered(),
        budget=None if shares is None else shares["rx1day"],
    )
    heat = design(
        _for(_heat(), asset_id),
        train.component("txx"),
        ExpectedUncovered(),
        budget=None if shares is None else shares["txx"],
    )
    return [
        _relabelled(grid, asset_id, "grid (expected_uncovered)"),
        CoverPackage(
            {
                "rx1day": _relabelled(rain, asset_id, "rain (expected_uncovered)"),
                "txx": _relabelled(heat, asset_id, "heat (expected_uncovered)"),
            },
            label="separate (expected_uncovered)",
        ),
        Cover(
            _either().index_definition,
            _either().payout,
            payout_unit="EUR",
            asset_id=asset_id,
            label="either",
        ),
    ]


def _for(cover: Cover, asset_id: str) -> Cover:
    """`cover` written for `asset_id`."""
    return Cover(
        cover.index_definition,
        cover.payout,
        payout_unit=cover.payout_unit,
        asset_id=asset_id,
        label=cover.label,
    )


def _relabelled(made: Any, asset_id: str, label: str) -> Any:
    """A design whose cover is written for `asset_id` under `label`, as in the plan."""
    cover = made.cover
    return dataclasses.replace(
        made,
        cover=Cover(
            cover.index_definition,
            cover.payout,
            payout_unit=cover.payout_unit,
            asset_id=asset_id,
            label=label,
        ),
    )


# -- one hazard: as before -------------------------------------------------------------


def test_a_plan_with_one_hazard_states_none_of_the_new_keys() -> None:
    plan = _annual_plan().basis_risk(NOISE).covers(_rain()).objective(MSE())
    document = plan.explain(format="json")
    for key in ("hazards", "combine", "hazard_dependence", "packages"):
        assert key not in document
    (sample, _) = document["setup"]["samples"]
    for key in ("components", "combine", "hazard_dependence"):
        assert key not in sample
    assert all("component" not in source for source in document["sources"])


def test_one_hazard_refuses_what_several_need() -> None:
    plan = _annual_plan().basis_risk(NOISE)
    with pytest.raises(ValueError, match=r"\.combine\(\.\.\.\) states the rule"):
        plan.covers(_rain()).combine(CompoundDamage()).explain()
    with pytest.raises(ValueError, match=r"\.package\(\.\.\.\) makes separate"):
        twin = Cover(
            RAIN_INDEX, StepPayout([80.0], [Free()]), payout_unit="EUR", label="twin"
        )
        plan.covers(_rain(), twin).package(
            {"a": "rain", "b": "twin"}, label="separate"
        ).explain()
    with pytest.raises(SampleError, match="reads one hazard per asset"):
        plan.covers(_grid()).explain()


# -- per-year values -------------------------------------------------------------------


def test_per_year_hazards_give_the_function_routes_rows(tmp_path: Path) -> None:
    plan = _designed(_two_hazards(_annual_plan())).split(Holdout(0.7))
    result = plan.write_parquet(tmp_path / "hazards.parquet")
    for asset_id in STATIONS:
        by_hand = _annual_by_hand(asset_id)
        sample = result.samples[asset_id]
        assert sample.content_hash == by_hand.content_hash
        assert sample.provenance.to_document() == by_hand.provenance.to_document()
        assert sample.index_components is not None
        assert list(sample.index_components) == ["rx1day", "txx"]
        train, test = holdout(by_hand, 0.7)
        entries = _entries_by_hand(train, asset_id)
        for split, block in (("in_sample", train), ("out_of_sample", test)):
            table = UNTYPED_COMPARE(
                entries,
                block,
                split=split,
                design_sample=train,
                options=OPTIONS,
            )
            assert (
                _rows_of(result.table, asset_id=asset_id, split=split)
                == table.to_pylist()
            )
    # Designs in asset, fold, slot, objective order: the grid, then the package's.
    assert [made.cover.label for made in result.designs] == [
        "grid (expected_uncovered)",
        "rain (expected_uncovered)",
        "heat (expected_uncovered)",
    ] * 2
    assert [made.cover.label for made in plan.designs()] == [
        made.cover.label for made in result.designs
    ]


def test_one_comparison_holds_the_grid_each_single_cover_and_their_package(
    tmp_path: Path,
) -> None:
    plan = _designed(_two_hazards(_annual_plan())).split(Holdout(0.7))
    table = plan.write_parquet(tmp_path / "hazards.parquet").table
    rows = _rows_of(table, asset_id="warehouse-a", split="out_of_sample")
    assert [(row["cover_label"], row["level"], row["components"]) for row in rows] == [
        ("grid (expected_uncovered)", "asset", "rx1day+txx"),
        ("rain (expected_uncovered)", "asset", "rx1day"),
        ("heat (expected_uncovered)", "asset", "txx"),
        ("separate (expected_uncovered)", "package", "rx1day+txx"),
        ("either", "asset", "rx1day+txx"),
    ]
    # The same draws and the same combined loss: one evaluation block for all.
    assert len({row["sample_hash"] for row in rows}) == 1
    # The split budget sums to the one bound, and the package's premium to its two.
    grid, rain, heat, package, _ = rows
    assert package["design_budget"] == pytest.approx(grid["design_budget"])
    assert rain["design_budget"] + heat["design_budget"] == pytest.approx(
        package["design_budget"]
    )
    assert package["premium"] == pytest.approx(rain["premium"] + heat["premium"])
    text = setup_text(pa.Table.from_pylist(rows, schema=table.schema))
    assert (
        "Row grid (expected_uncovered) (asset): reads rx1day and txx; designed on the "
        "combined loss; budget: premium at most" in text
    )
    assert (
        "Row rain (expected_uncovered) (asset): reads rx1day; designed on the own "
        "loss of component rx1day; budget: premium at most" in text
    )
    assert (
        "Row separate (expected_uncovered) (package): reads rx1day and txx; each "
        "contract designed or given on its own; budget: premium at most" in text
    )
    assert "Row either (asset): reads rx1day and txx; given as is" in text
    assert "pricing: the pure premium" in text
    assert "Package separate (expected_uncovered): rx1day (budget at most" in text
    assert "every row's metrics are on the combined loss" in text


def test_the_breakdown_is_the_function_routes(tmp_path: Path) -> None:
    plan = _designed(_two_hazards(_annual_plan())).split(Holdout(0.7))
    parts = plan.payout_breakdown()
    asset_id = "warehouse-a"
    train, test = holdout(_annual_by_hand(asset_id), 0.7)
    by_hand = payout_breakdown(
        _entries_by_hand(train, asset_id),
        test,
        split="out_of_sample",
        design_sample=train,
        options=OPTIONS,
    )
    assert _rows_of(parts, asset_id=asset_id, split="out_of_sample") == (
        by_hand.to_pylist()
    )
    kinds = {
        (row["cover_label"], row["part_kind"])
        for row in _rows_of(parts, asset_id=asset_id)
    }
    assert ("grid (expected_uncovered)", "region") in kinds
    assert ("separate (expected_uncovered)", "component") in kinds
    assert ("either", "attribution") in kinds
    # Nothing is written.
    assert list(tmp_path.iterdir()) == []


def test_a_linear_trend_is_fitted_per_fold_on_every_series() -> None:
    plan = _designed(_two_hazards(_annual_plan(trend=LinearTrend(min_years=5))))
    plan = plan.split(KFold(3))
    designs = plan.designs()
    asset_id = "warehouse-a"
    whole = _annual_by_hand(asset_id, trend=LinearTrend(min_years=5))
    for fold, (train, _) in enumerate(year_block_kfold(whole, 3)):
        fit_years = tuple(int(year) for year in train.period)
        refitted = _annual_by_hand(
            asset_id, trend=LinearTrend(min_years=5, fit_years=fit_years)
        )
        design_block = year_block_kfold(refitted, 3)[fold][0]
        grid = design(
            _grid(), design_block, ExpectedUncovered(), budget=Budget(ratio=0.8)
        )
        made = [d for d in designs if d.cover.asset_id == asset_id][3 * fold]
        assert made.parameters == grid.parameters
        assert made.design_sample.content_hash == design_block.content_hash


def test_explain_lists_the_hazards_without_io(tmp_path: Path) -> None:
    absent = tmp_path / "nowhere" / "txx.parquet"
    plan = _designed(_two_hazards(_annual_plan(), source=absent)).split(Holdout(0.7))
    document = plan.explain(format="json")
    assert not absent.parent.exists()
    assert [entry["name"] for entry in document["hazards"]] == ["rx1day", "txx"]
    assert [entry["own"] for entry in document["hazards"]] == [True, False]
    assert document["hazards"][1]["unit"] == "degC"
    assert document["combine"]["kind"] == "compound_damage"
    assert document["hazard_dependence"] == {"kind": "data"}
    assert document["packages"] == [
        {
            "covers": {"rx1day": "rain", "txx": "heat"},
            "label": "separate",
            "shares": "own_loss",
        }
    ]
    assert [source["component"] for source in document["sources"]] == [
        "rx1day",
        "txx",
    ]
    assert document["sources"][1]["path"] == str(absent)
    (sample, _) = document["setup"]["samples"]
    assert [entry["name"] for entry in sample["components"]] == ["rx1day", "txx"]
    assert sample["loss_model"]["kind"] == "components"
    text = setup_text(document)
    assert "Index components: rx1day (in mm/day, upper tail" in text
    assert "the hazard series of component txx" in text
    assert "read from a file" in text


# -- fitted curves ---------------------------------------------------------------------


def _metadata(hazard: str) -> HazardDatasetMetadata:
    unit, what = (
        ("mm/day", "annual maximum daily rainfall")
        if hazard == "rx1day"
        else ("degC", "annual maximum daily temperature")
    )
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit=unit,
        value_semantics=what,
        producer="tests",
        source=_provenance(),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 500.0),
    )


def _curve(asset: str, hazard: str, horizon: int, pathway: str) -> dict[str, Any]:
    location, scale = (48.0, 12.0) if hazard == "rx1day" else (33.0, 2.0)
    return {
        "cell_index": CELLS[asset],
        "source_id": f"fixture-{asset}",
        "source_geometry": None,
        "hazard_name": hazard,
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": location
        + scale * 0.02 * (horizon - 2030)
        + (0.0 if asset == "warehouse-a" else scale / 4.0)
        + (0.0 if pathway == "ssp585" else scale / 10.0),
        "curve_scale": scale,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def _curves(
    path: Path,
    hazard: str,
    horizons: Sequence[int] = (2030,),
    pathway: str = "ssp585",
) -> Path:
    stated = _metadata(hazard)
    rows = [
        _curve(asset, hazard, horizon, pathway)
        for asset in POINTS
        for horizon in horizons
    ]
    table = pa.Table.from_pylist(rows, schema=hazard_arrow_schema(stated))
    write_hazard_dataset(table, path, stated, max_workers=1)
    return path


def _request(path: Path, hazard: str, impact: Any, **select: Any) -> Any:
    selection: dict[str, Any] = {"hazard_names": [hazard]}
    selection.update(select)
    request = HazardDataset.local(path).for_assets(_assets()).select(**selection)
    if impact is None:
        return request
    return request.impact(
        impact, name=f"{hazard}-ramp", value_unit="1", value_semantics="damage ratio"
    )


@pytest.fixture
def datasets(tmp_path: Path) -> dict[str, Path]:
    return {
        "rx1day": _curves(tmp_path / "rx1day.parquet", "rx1day"),
        "txx": _curves(tmp_path / "txx.parquet", "txx"),
    }


def _curve_plan(datasets: Mapping[str, Path], **select: Any) -> Any:
    return (
        UNTYPED_PLAN.for_evaluation(
            _request(datasets["rx1day"], "rx1day", RAIN, **select),
            value_column="value_eur",
            value_unit="EUR",
        )
        .monte_carlo(400, seed=42)
        .hazard("txx", _request(datasets["txx"], "txx", HEAT, **select))
        .combine(CappedSum(1.0))
    )


def _curve_rows(path: Path, hazard: str) -> dict[tuple[str, int], dict[str, Any]]:
    request = _request(path, hazard, None)
    return {
        (row["asset_id"], row["horizon"]): row
        for row in request.asset_curves().to_pylist()
    }


def _drawn_by_hand(
    datasets: Mapping[str, Path],
    seed: int,
    n: int,
    dependence: Any,
    horizons: Sequence[int] = (2030,),
    scenario: str = "baseline",
) -> dict[str, OutcomeSample]:
    """`sample_curve_hazards`: the asset's child draws rain, a child it spawns heat."""
    rows = {name: _curve_rows(path, name) for name, path in datasets.items()}
    metadata = {
        name: HazardDataset.local(path).metadata() for name, path in datasets.items()
    }
    drawn: dict[str, OutcomeSample] = {}
    for asset_id, child in zip(
        POINTS, np.random.SeedSequence(seed).spawn(len(POINTS)), strict=True
    ):
        rngs = {"rx1day": np.random.default_rng(child)}
        rngs["txx"] = np.random.default_rng(child.spawn(1)[0])
        curves = {}
        for name in datasets:
            members = [rows[name][(asset_id, horizon)] for horizon in horizons]
            curves[name] = members if len(members) > 1 else members[0]
        drawn[asset_id] = sample_curve_hazards(
            curves,
            dependence=dependence,
            metadata=metadata,
            n_draws=n,
            rngs=rngs,
            loss_models={
                "rx1day": _loss(RAIN, asset_id, CELLS[asset_id]),
                "txx": _loss(HEAT, asset_id, CELLS[asset_id]),
            },
            combine=CappedSum(1.0),
            basis_risk=NOISE,
            asset_id=asset_id,
            scenario=scenario,
        )
    return drawn


def test_curve_hazards_give_the_function_routes_rows(
    datasets: dict[str, Path], tmp_path: Path
) -> None:
    plan = _designed(_curve_plan(datasets).dependence(GaussianCopula(0.3)))
    plan = plan.split(FreshDraws(400, seed=8))
    result = plan.write_parquet(tmp_path / "curves.parquet")
    drawn = _drawn_by_hand(datasets, 42, 400, GaussianCopula(0.3))
    fresh = _drawn_by_hand(datasets, 8, 400, GaussianCopula(0.3))
    assert result.fresh_samples is not None
    for asset_id in POINTS:
        assert result.samples[asset_id].content_hash == drawn[asset_id].content_hash
        assert (
            result.fresh_samples[asset_id].content_hash == fresh[asset_id].content_hash
        )
        entries = _entries_by_hand(drawn[asset_id], asset_id)
        for split, block in (
            ("in_sample", drawn[asset_id]),
            ("out_of_sample", fresh[asset_id]),
        ):
            table = UNTYPED_COMPARE(
                entries,
                block,
                split=split,
                design_sample=drawn[asset_id],
                options=OPTIONS,
            )
            assert (
                _rows_of(result.table, asset_id=asset_id, split=split)
                == table.to_pylist()
            )
    # The first component draws what the plan with one hazard draws.
    one = (
        UNTYPED_PLAN.for_evaluation(
            _request(datasets["rx1day"], "rx1day", RAIN),
            value_column="value_eur",
            value_unit="EUR",
        )
        .monte_carlo(400, seed=42)
        .basis_risk(NOISE)
    )
    alone = _drawn_by_hand(datasets, 42, 400, Independent())
    for asset_id, sample in one.samples().items():
        assert alone[asset_id].component("rx1day").content_hash == sample.content_hash
    # Each component's canonical file is recorded once, and its source names it.
    document = result.metadata.model_dump(mode="json")
    files = [record["canonical_file"] for record in document["source_records"]]
    assert [record["components"] for record in files] == [["rx1day"], ["txx"]]
    hazards = [s for s in document["sources"] if s["role"] == "hazard"]
    checksums = {record["components"][0]: record["file_sha256"] for record in files}
    # Every curve read names its own file, a component's view's included.
    for source in hazards:
        read = "rx1day" if source["curve"]["scale"] == 12.0 else "txx"
        assert source["canonical_file_sha256"] == checksums[read]
        assert source.get("component", read) == read
    text = setup_text(result.metadata)
    assert "Dependence between the hazards: GaussianCopula, a stated assumption" in text


def test_several_hazards_on_curves_need_a_dependence(
    datasets: dict[str, Path], tmp_path: Path
) -> None:
    plan = _designed(_curve_plan(datasets)).split(FreshDraws(400, seed=8))
    for call in (
        lambda: plan.explain(),
        plan.samples,
        plan.designs,
        plan.payout_breakdown,
        lambda: plan.write_parquet(tmp_path / "out.parquet"),
    ):
        with pytest.raises(ValueError, match="need a stated dependence between them"):
            call()


def test_a_component_with_no_impact_is_an_index_only(
    datasets: dict[str, Path],
) -> None:
    plan = (
        UNTYPED_PLAN.for_evaluation(
            _request(datasets["rx1day"], "rx1day", RAIN),
            value_column="value_eur",
            value_unit="EUR",
        )
        .monte_carlo(400, seed=42)
        .hazard("txx", _request(datasets["txx"], "txx", None))
        .dependence(Independent())
        .basis_risk(NOISE)
    )
    sample = plan.samples()["warehouse-a"]
    assert sample.index_components is not None
    assert sample.index_components["txx"].own_loss is None
    # No rule with one impact, and a template on txx can't be designed on its own loss.
    with pytest.raises(SampleError, match="only component 'rx1day' has one"):
        plan.combine(CappedSum(1.0)).explain()
    with pytest.raises(SampleError, match="which has no impact"):
        plan.covers(_heat()).objective(MSE()).explain()


def test_the_hazards_dependence_is_any_model_on_the_plan(
    datasets: dict[str, Path],
) -> None:
    for model in (Independent(), GaussianCopula(0.6)):
        plan = _curve_plan(datasets).dependence(model).basis_risk(NOISE)
        drawn = _drawn_by_hand(datasets, 42, 400, model)
        for asset_id, sample in plan.samples().items():
            assert sample.content_hash == drawn[asset_id].content_hash
    untyped: Any = _curve_plan(datasets)
    with pytest.raises(TypeError, match="GaussianCopula"):
        untyped.dependence("gaussian")


def test_joint_takes_a_gaussian_copula_too(tmp_path: Path) -> None:
    path = _curves(tmp_path / "rx1day.parquet", "rx1day")
    plan = (
        UNTYPED_PLAN.for_evaluation(
            _request(path, "rx1day", RAIN), value_column="value_eur", value_unit="EUR"
        )
        .monte_carlo(400, seed=42)
        .basis_risk(NOISE)
        .scopes(Joint())
        .dependence(GaussianCopula(0.5))
    )
    rows = _curve_rows(path, "rx1day")
    children = np.random.SeedSequence(42).spawn(len(POINTS))
    by_hand = sample_curves(
        {asset_id: rows[(asset_id, 2030)] for asset_id in POINTS},
        dependence=GaussianCopula(0.5),
        metadata=dict.fromkeys(POINTS, HazardDataset.local(path).metadata()),
        n_draws=400,
        rngs={
            asset_id: np.random.default_rng(child)
            for asset_id, child in zip(POINTS, children, strict=True)
        },
        loss_models={
            asset_id: _loss(RAIN, asset_id, CELLS[asset_id]) for asset_id in POINTS
        },
        basis_risk=NOISE,
    )
    assert plan.joint_sample().content_hash == by_hand.content_hash


# -- refusals --------------------------------------------------------------------------


def test_joint_pool_and_index_are_refused_with_several_hazards(
    datasets: dict[str, Path],
) -> None:
    hazards = _two_hazards(_annual_plan()).basis_risk(NOISE).covers(_rain())
    with pytest.raises(ValueError, match=r"Joint\(\) is refused"):
        hazards.scopes(Joint())
    with pytest.raises(ValueError, match=r"Joint\(\) is refused"):
        _annual_plan().scopes(Joint()).hazard("rx1day", tail="upper")
    with pytest.raises(ValueError, match=r"pool= is refused"):
        _annual_plan(location=NearestCell(), pool=ModelPool()).hazard(
            "rx1day", tail="upper"
        )
    index = {
        "location": StationColumn("station"),
        "unit": "mm/day",
        "provenance": _provenance(),
        "index_column": "rx1day",
    }
    with pytest.raises(ValueError, match=r"\.index\(\.\.\.\) is refused"):
        hazards.index(WEATHER, **index)
    with pytest.raises(ValueError, match=r"\.index\(\.\.\.\) is refused"):
        _annual_plan().index(WEATHER, **index).hazard("rx1day", tail="upper")
    # On curves too, and at explain() whatever the order of the builders.
    curves = _curve_plan(datasets).dependence(Independent())
    with pytest.raises(ValueError, match=r"Joint\(\) is refused"):
        curves.scopes(Joint())
    stated = dataclass_replace(curves, scope_list=(Joint(),))
    with pytest.raises(ValueError, match=r"Joint\(\) is refused"):
        stated.explain()


def dataclass_replace(plan: Any, **fields: Any) -> Any:
    return dataclasses.replace(plan, **fields)


def test_the_builders_check_what_they_state(datasets: dict[str, Path]) -> None:
    plan = _annual_plan()
    with pytest.raises(ValueError, match="'upper' or 'lower'"):
        plan.hazard("rx1day")
    with pytest.raises(ValueError, match="component name"):
        plan.hazard("rx1day.max", tail="upper")
    with pytest.raises(ValueError, match="for_annual_extremes"):
        plan.hazard("rx1day", tail="upper", unit="mm/day")
    with pytest.raises(ValueError, match="named twice"):
        plan.hazard("a", tail="upper").hazard("b", tail="upper")
    with pytest.raises(ValueError, match="given twice"):
        plan.hazard("a", tail="upper").hazard("a", tail="upper")
    with pytest.raises(ValueError, match="needs unit="):
        plan.hazard("txx", WEATHER, tail="upper", provenance=_provenance())
    with pytest.raises(TypeError, match="SourceProvenance"):
        plan.hazard("txx", WEATHER, unit="degC", tail="upper")
    with pytest.raises(TypeError, match="pools are refused"):
        plan.hazard(
            "txx",
            [WEATHER, WEATHER],
            unit="degC",
            tail="upper",
            provenance=_provenance(),
        )
    with pytest.raises(TypeError, match="CombineRule"):
        plan.combine(lambda impacts: impacts["rx1day"])
    # A per-year plan with another component names its own hazard too.
    unnamed = plan.hazard(
        "txx", WEATHER, unit="degC", tail="upper", provenance=_provenance()
    )
    with pytest.raises(ValueError, match="names its own hazard too"):
        unnamed.explain()
    with pytest.raises(ValueError, match="two hazards or more"):
        plan.hazard("rx1day", tail="upper").explain()
    # Curves take the request alone.
    curves = _curve_plan(datasets)
    with pytest.raises(ValueError, match=r"\['tail', 'unit'\] are not taken"):
        curves.hazard(
            "other", _request(datasets["txx"], "txx", HEAT), unit="degC", tail="upper"
        )
    other = (
        HazardDataset.local(datasets["txx"])
        .for_assets(_assets().slice(0, 1))
        .select(hazard_names=["txx"])
    )
    with pytest.raises(ValueError, match="for the plan's assets"):
        curves.hazard("other", other)
    several = _request(datasets["txx"], "txx", HEAT, hazard_names=["txx", "rx1day"])
    with pytest.raises(ValueError, match="selects 2 hazard_names"):
        curves.hazard("other", several)


def test_the_rule_is_required_with_two_impacts() -> None:
    plan = (
        _annual_plan()
        .hazard("rx1day", tail="upper")
        .hazard(
            "txx",
            WEATHER,
            impact=HEAT,
            unit="degC",
            tail="upper",
            provenance=_provenance(),
            hazard_column="txx",
        )
        .basis_risk(NOISE)
    )
    with pytest.raises(SampleError, match="none is assumed"):
        plan.explain()
    with pytest.raises(SampleError, match="none is assumed"):
        plan.samples()


def test_covers_are_checked_against_the_components() -> None:
    plan = _two_hazards(_annual_plan()).basis_risk(NOISE)
    wrong_unit = Cover(
        IndexDefinition("txx", "K", aggregation="annual_maximum"),
        StepPayout([300.0], [1e5]),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match="component 'txx' is in 'degC'"):
        plan.covers(wrong_unit).explain()
    elsewhere = Cover(
        IndexDefinition("wind", "m/s", aggregation="annual_maximum"),
        StepPayout([30.0], [1e5]),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match="the plan's components are"):
        plan.covers(elsewhere).explain()
    lower = Cover(
        IndexDefinition("txx", "degC", tail="lower", aggregation="annual_maximum"),
        StepPayout([20.0], [1e5], tail="lower"),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match="whose tail is 'upper'"):
        plan.covers(lower).explain()


def test_packages_are_checked_against_the_covers_and_the_budget() -> None:
    plan = (
        _two_hazards(_annual_plan()).basis_risk(NOISE).covers(_grid(), _rain(), _heat())
    )
    budgeted = plan.objective(MSE(), budget=Budget(ratio=0.8))
    pairs = {"rx1day": "rain", "txx": "heat"}
    with pytest.raises(ValueError, match="state how to split it"):
        budgeted.package(pairs, label="separate").explain()
    with pytest.raises(ValueError, match="states no budget to split"):
        plan.objective(MSE()).package(pairs, label="p", shares="own_loss").explain()
    with pytest.raises(ValueError, match="no cover in"):
        budgeted.package({"rx1day": "rain", "txx": "nothing"}, label="p").explain()
    with pytest.raises(ValueError, match="pays on several components"):
        budgeted.package({"rx1day": "grid", "txx": "heat"}, label="p").explain()
    with pytest.raises(ValueError, match="in one package at most"):
        budgeted.package(pairs, label="p", shares="own_loss").package(
            {"rx1day": "rain", "txx": "heat"}, label="q", shares="own_loss"
        ).explain()
    with pytest.raises(ValueError, match="must name its contracts' components"):
        budgeted.package(pairs, label="p", shares={"rx1day": 1.0}).explain()
    with pytest.raises(EvaluationError, match="sum to 1"):
        budgeted.package(pairs, label="p", shares={"rx1day": 0.5, "txx": 0.4}).explain()
    with pytest.raises(ValueError, match="two contracts or more"):
        budgeted.package({"rx1day": "rain"}, label="p")
    resolved = plan.covers(
        _rain().with_values({"levels[0]": 1e5}), _heat().with_values({"levels[0]": 1e5})
    )
    with pytest.raises(ValueError, match="no budget to split"):
        resolved.package(
            {"rx1day": "rain", "txx": "heat"}, label="p", shares="own_loss"
        ).explain()


def test_a_package_of_resolved_covers_is_one_row(tmp_path: Path) -> None:
    rain = _rain().with_values({"levels[0]": 1e5})
    heat = _heat().with_values({"levels[0]": 2e5})
    plan = (
        _two_hazards(_annual_plan())
        .basis_risk(NOISE)
        .covers(rain, heat)
        .package({"rx1day": "rain", "txx": "heat"}, label="separate")
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    table = plan.write_parquet(tmp_path / "out.parquet").table
    rows = _rows_of(table, asset_id="depot-b", split="out_of_sample")
    assert [(row["cover_label"], row["level"]) for row in rows] == [
        ("rain", "asset"),
        ("heat", "asset"),
        ("separate", "package"),
    ]
    assert rows[2]["premium"] == pytest.approx(rows[0]["premium"] + rows[1]["premium"])


# -- scenarios -------------------------------------------------------------------------


def test_per_year_scenarios_of_several_hazards_give_the_function_routes_rows(
    tmp_path: Path,
) -> None:
    later = _weather(shift=6.0, seed=5)
    scenario = AnnualScenario(
        "later",
        {"rx1day": later, "txx": later},
        provenance={"rx1day": _provenance(), "txx": _provenance()},
        hazard_name={"rx1day": "rx1day", "txx": "txx"},
        pathway="ssp585",
    )
    plan = _designed(
        _two_hazards(
            _annual_plan(hazard_name="rx1day", scenario="early"), hazard_name="txx"
        )
    ).split(Holdout(0.7))
    plan = plan.scenarios(ScenarioSet([scenario], reference="early"))
    result = plan.write_parquet(tmp_path / "scenarios.parquet")
    assert result.scenario_samples is not None
    names = {"rx1day": "rx1day", "txx": "txx"}
    for asset_id in STATIONS:
        early = _annual_by_hand(asset_id, scenario="early", hazard_names=names)
        late = _annual_by_hand(
            asset_id,
            table=later,
            scenario="later",
            pathway="ssp585",
            hazard_names=names,
        )
        assert result.scenario_samples["early"][asset_id].content_hash == (
            early.content_hash
        )
        assert result.scenario_samples["later"][asset_id].content_hash == (
            late.content_hash
        )
        early_train, early_test = holdout(early, 0.7)
        late_train, late_test = holdout(late, 0.7)
        entries = [
            *_entries_by_hand(early_train, asset_id)[:2],
            *_entries_by_hand(late_train, asset_id)[:2],
        ]
        # As the plan orders them: each slot's designs by design scenario.
        ordered = [entries[0], entries[2], entries[1], entries[3]]
        either = _entries_by_hand(early_train, asset_id)[2]
        table = compare_scenarios(
            [*ordered, either],
            ScenarioSet([early_test, late_test], reference="early"),
            design_sample=early_train,
            options=OPTIONS,
        )
        assert _rows_of(result.table, asset_id=asset_id) == table.to_pylist()
    text = setup_text(result.metadata)
    assert "Scenario later: hazard component rx1day: rx1day, as the caller states" in (
        text
    )


def test_curve_scenarios_of_several_hazards_share_their_draws(
    tmp_path: Path,
) -> None:
    horizons = (2030, 2050)
    datasets = {
        "rx1day": _curves(tmp_path / "rx1day.parquet", "rx1day", horizons),
        "txx": _curves(tmp_path / "txx.parquet", "txx", horizons),
    }
    select = {"horizons": list(horizons), "pathways": ["ssp585"]}
    plan = _designed(
        _curve_plan(datasets, **select).dependence(GaussianCopula(0.3))
    ).split(FreshDraws(400, seed=8))
    plan = plan.scenarios(ScenarioSet(reference="ssp585_2030"))
    samples = plan.scenario_samples()
    for label, year in (("ssp585_2030", 2030), ("ssp585_2050", 2050)):
        drawn = _drawn_by_hand(
            datasets, 42, 400, GaussianCopula(0.3), (year,), scenario=label
        )
        for asset_id in POINTS:
            assert samples[label][asset_id].content_hash == drawn[asset_id].content_hash
    result = plan.write_parquet(tmp_path / "out.parquet")
    rows = result.table.to_pylist()
    assert {row["scenario"] for row in rows} == {"ssp585_2030", "ssp585_2050"}
    assert {row["level"] for row in rows} == {"asset", "package"}
    document = plan.explain(format="json")
    entry = document["setup"]["scenarios"]["scenarios"]["ssp585_2050"]
    assert entry["hazard"] == {
        "rx1day": {"name": "rx1day", "from": "curve_row"},
        "txx": {"name": "txx", "from": "curve_row"},
    }
    # Another dataset per component is a scenario too.
    other = {
        "rx1day": _curves(tmp_path / "rx1day-b.parquet", "rx1day", (2050,), "ssp245"),
        "txx": _curves(tmp_path / "txx-b.parquet", "txx", (2050,), "ssp245"),
    }
    stated = plan.scenarios(
        ScenarioSet([CurveScenario("ssp245", other)], reference="ssp585_2030")
    )
    built = stated.scenario_samples()
    assert list(built) == ["ssp585_2030", "ssp585_2050", "ssp245"]
    components = built["ssp245"]["depot-b"].index_components
    assert components is not None and list(components) == ["rx1day", "txx"]


def test_scenario_sources_are_mapped_by_component() -> None:
    later = _weather(shift=6.0, seed=5)
    one = AnnualScenario("later", later, provenance=_provenance())
    plan = _designed(_two_hazards(_annual_plan())).split(Holdout(0.7))
    with pytest.raises(ValueError, match="map each component to its own"):
        plan.scenarios(ScenarioSet([one], reference="baseline")).explain()
    wrong = AnnualScenario(
        "later",
        {"rx1day": later, "wind": later},
        provenance={"rx1day": _provenance(), "wind": _provenance()},
    )
    with pytest.raises(ValueError, match="the plan's are"):
        plan.scenarios(ScenarioSet([wrong], reference="baseline")).explain()
    mapped = AnnualScenario(
        "later",
        {"rx1day": later, "txx": later},
        provenance={"rx1day": _provenance(), "txx": _provenance()},
    )
    single = _annual_plan().basis_risk(NOISE).covers(_rain()).split(Holdout(0.7))
    with pytest.raises(ValueError, match="maps its sources by component"):
        single.scenarios(ScenarioSet([mapped], reference="baseline")).explain()


def test_scenario_descriptions_check_their_mappings(tmp_path: Path) -> None:
    later = _weather(shift=6.0)
    with pytest.raises(TypeError, match="map each of them"):
        AnnualScenario("x", {"rx1day": later, "txx": later}, provenance=_provenance())
    with pytest.raises(ValueError, match="which source doesn't map"):
        AnnualScenario(
            "x",
            {"rx1day": later, "txx": later},
            provenance={"rx1day": _provenance(), "txx": _provenance()},
            unit={"wind": "m/s"},
        )
    with pytest.raises(TypeError, match="maps components to values too"):
        AnnualScenario(
            "x",
            {"rx1day": later, "txx": later},
            provenance={"rx1day": _provenance(), "txx": _provenance()},
            unit="mm/day",
        )
    with pytest.raises(TypeError, match="map each component to its source"):
        AnnualScenario("x", later, provenance=_provenance(), unit={"rx1day": "mm/day"})
    with pytest.raises(ValueError, match="reads no pool"):
        AnnualScenario(
            "x",
            {"rx1day": later, "txx": later},
            provenance={"rx1day": _provenance(), "txx": _provenance()},
            pool=ModelPool(),
        )
    with pytest.raises(ValueError, match="two or more"):
        AnnualScenario("x", {"rx1day": later}, provenance={"rx1day": _provenance()})
    path = _curves(tmp_path / "rx1day.parquet", "rx1day")
    with pytest.raises(ValueError, match="two or more"):
        CurveScenario("x", {"rx1day": path})
    with pytest.raises(TypeError, match="HazardDataset"):
        UNTYPED_CURVE_SCENARIO("x", {"rx1day": path, "txx": 3})
    stated = CurveScenario("x", {"rx1day": path, "txx": path})
    assert "rx1day.parquet" in repr(stated)


def test_an_empirical_copula_across_the_hazards(datasets: dict[str, Path]) -> None:
    # The reference: per-year rows of the same two components, without impacts.
    reference = sample_annual_hazards(
        {
            "rx1day": _series("warehouse-a", "rx1day", "mm/day"),
            "txx": _series("warehouse-a", "txx", "degC"),
        },
        tails={"rx1day": "upper", "txx": "upper"},
        loss_models={"rx1day": _loss(RAIN, "warehouse-a")},
        basis_risk=NOISE,
        asset_id="warehouse-a",
        rng=np.random.default_rng(1),
    )
    model = EmpiricalCopula(reference)
    years = len(YEARS)
    plan = (
        UNTYPED_PLAN.for_evaluation(
            _request(datasets["rx1day"], "rx1day", RAIN),
            value_column="value_eur",
            value_unit="EUR",
        )
        .monte_carlo(4 * years, seed=42)
        .hazard("txx", _request(datasets["txx"], "txx", HEAT))
        .combine(CappedSum(1.0))
        .dependence(model)
        .basis_risk(NOISE)
    )
    drawn = _drawn_by_hand(datasets, 42, 4 * years, model)
    for asset_id, sample in plan.samples().items():
        assert sample.content_hash == drawn[asset_id].content_hash
    assert plan.explain(format="json")["hazard_dependence"]["kind"] == (
        "empirical_copula"
    )
    # A draw count the reference's years don't divide, and a split by rows.
    with pytest.raises(SampleError, match="multiple"):
        dataclass_replace(
            plan, draws=dataclasses.replace(plan.draws, n_draws=101)
        ).explain()
    with pytest.raises(SampleError, match="EmpiricalCopula"):
        plan.split(Holdout(0.5))


def test_curve_explain_lists_the_components(datasets: dict[str, Path]) -> None:
    plan = _designed(_curve_plan(datasets).dependence(GaussianCopula(0.3)))
    document = plan.explain(format="json")
    assert [entry["name"] for entry in document["hazards"]] == ["rx1day", "txx"]
    assert document["hazards"][1]["path"] == str(datasets["txx"])
    assert document["hazards"][1]["hazard_names"] == ["txx"]
    assert document["hazard_dependence"] == {
        "correlation": 0.3,
        "kind": "gaussian_copula",
    }
    assert [source["component"] for source in document["sources"]] == [
        "rx1day",
        "txx",
    ]
    assert document["licences"] is None
    (sample, _) = document["setup"]["samples"]
    assert [entry["unit"] for entry in sample["components"]] == [None, None]
    assert "a child it spawns" in document["rng"]["generators"]
    text = setup_text(document)
    assert "rx1day (in a unit not known yet" in text
    assert "GaussianCopula, a stated assumption: correlation 0.3" in text


def test_a_component_reads_the_plans_scenarios(tmp_path: Path) -> None:
    horizons = (2030, 2050)
    datasets = {
        "rx1day": _curves(tmp_path / "rx1day.parquet", "rx1day", horizons),
        "txx": _curves(tmp_path / "txx.parquet", "txx", horizons),
    }
    select = {"horizons": list(horizons), "pathways": ["ssp585"]}
    plan = (
        UNTYPED_PLAN.for_evaluation(
            _request(datasets["rx1day"], "rx1day", RAIN, **select),
            value_column="value_eur",
            value_unit="EUR",
        )
        .monte_carlo(400, seed=42)
        .combine(CappedSum(1.0))
        .dependence(Independent())
        .basis_risk(NOISE)
        .covers(_rain().with_values({"levels[0]": 1e5}))
        .split(FreshDraws(400, seed=8))
        .scenarios(ScenarioSet(reference="ssp585_2030"))
    )
    other = plan.hazard("txx", _request(datasets["txx"], "txx", HEAT, horizons=[2030]))
    with pytest.raises(ValueError, match="select the plan's, or none"):
        other.explain()
    unnamed = plan.hazard(
        "txx", _request(datasets["txx"], "txx", HEAT, hazard_names=None)
    )
    with pytest.raises(ValueError, match="exactly one hazard"):
        unnamed.explain()
    # Without scenarios, a component reads one horizon.
    several = (
        _curve_plan(datasets, **select).dependence(Independent()).basis_risk(NOISE)
    )
    with pytest.raises(ValueError, match="component 'txx'"):
        several.explain()


def test_curve_scenario_rows_are_compare_scenarios(tmp_path: Path) -> None:
    horizons = (2030, 2050)
    datasets = {
        "rx1day": _curves(tmp_path / "rx1day.parquet", "rx1day", horizons),
        "txx": _curves(tmp_path / "txx.parquet", "txx", horizons),
    }
    select = {"horizons": list(horizons), "pathways": ["ssp585"]}
    plan = _designed(
        _curve_plan(datasets, **select).dependence(GaussianCopula(0.3))
    ).split(FreshDraws(400, seed=8))
    plan = plan.scenarios(ScenarioSet(reference="ssp585_2030"), design_on="reference")
    result = plan.write_parquet(tmp_path / "out.parquet")
    labels = ("ssp585_2030", "ssp585_2050")
    designs = {
        label: _drawn_by_hand(
            datasets, 42, 400, GaussianCopula(0.3), (year,), scenario=label
        )
        for label, year in zip(labels, horizons, strict=True)
    }
    fresh = {
        label: _drawn_by_hand(
            datasets, 8, 400, GaussianCopula(0.3), (year,), scenario=label
        )
        for label, year in zip(labels, horizons, strict=True)
    }
    for asset_id in POINTS:
        reference = designs["ssp585_2030"][asset_id]
        table = compare_scenarios(
            _entries_by_hand(reference, asset_id),
            ScenarioSet(
                [fresh[label][asset_id] for label in labels], reference="ssp585_2030"
            ),
            design_sample=reference,
            options=OPTIONS,
        )
        assert _rows_of(result.table, asset_id=asset_id) == table.to_pylist()
    shared = result.metadata.model_dump(mode="json")["setup"]["scenarios"][
        "shared_draws"
    ]
    assert sorted(shared["rng"]) == [
        "component 'rx1day' of asset 'depot-b'",
        "component 'rx1day' of asset 'warehouse-a'",
        "component 'txx' of asset 'depot-b'",
        "component 'txx' of asset 'warehouse-a'",
    ]


def test_a_window_of_horizons_reads_every_component(tmp_path: Path) -> None:
    horizons = (2030, 2050)
    datasets = {
        "rx1day": _curves(tmp_path / "rx1day.parquet", "rx1day", horizons),
        "txx": _curves(tmp_path / "txx.parquet", "txx", horizons),
    }
    select = {"horizons": list(horizons), "pathways": ["ssp585"]}
    window = {"window": [("ssp585", year) for year in horizons]}
    plan = (
        _curve_plan(datasets, **select)
        .dependence(GaussianCopula(0.3))
        .basis_risk(NOISE)
        .covers(_rain().with_values({"levels[0]": 1e5}))
        .split(FreshDraws(400, seed=8))
        .scenarios(ScenarioSet(reference="window", labels=window))
    )
    sample = plan.scenario_samples()["window"]["warehouse-a"]
    drawn = _drawn_by_hand(
        datasets, 42, 400, GaussianCopula(0.3), horizons, scenario="window"
    )
    assert sample.content_hash == drawn["warehouse-a"].content_hash
    entry = plan.explain(format="json")["setup"]["scenarios"]["scenarios"]["window"]
    assert entry["members"] == [["ssp585", 2030], ["ssp585", 2050]]


def test_a_holdout_of_curve_draws_under_a_stated_dependence(
    datasets: dict[str, Path], tmp_path: Path
) -> None:
    independent = _designed(_curve_plan(datasets).dependence(Independent()))
    result = independent.split(Holdout(0.5)).write_parquet(tmp_path / "a.parquet")
    assert {row["split"] for row in result.table.to_pylist()} == {
        "in_sample",
        "out_of_sample",
    }
    # A copula reorders the draws, so two blocks of one sample share their rows'
    # draws; compare refuses it out of sample, and fresh draws are the test.
    copula = _designed(_curve_plan(datasets).dependence(GaussianCopula(0.3)))
    with pytest.raises(LeakageError):
        copula.split(Holdout(0.5)).write_parquet(tmp_path / "b.parquet")
