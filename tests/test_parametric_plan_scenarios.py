"""ParametricPlan.scenarios: every cover or design on every scenario's block.

Every number here is synthetic: small made-up curves and tables of years.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact, TransformContext

from crc_sdk.connectors import hazard_arrow_schema, write_hazard_dataset
from crc_sdk.geometry import point_to_cell
from crc_sdk.parametric import (
    MSE,
    AnnualScenario,
    AnnualSeries,
    Budget,
    ComonotoneIndexWarning,
    Cover,
    CurveScenario,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    FreshDraws,
    Holdout,
    ImpactNoise,
    Independent,
    IndexDefinition,
    Individual,
    Joint,
    KFold,
    LinearPayout,
    LinearTrend,
    LossModel,
    MetricOptions,
    ModelPool,
    NearestCell,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    SampleError,
    ScenarioError,
    ScenarioSet,
    StationColumn,
    StepPayout,
    allocate_scenarios,
    compare_scenarios,
    design,
    design_joint,
    holdout,
    payout_breakdown_scenarios,
    sample_annual,
    sample_curve,
    sample_curves,
    setup_text,
    to_drift,
    year_block_kfold,
)
from crc_sdk.parametric import plan as plan_module
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import (
    AssetPortfolio,
    HazardDataset,
    ReturnPeriodExtrapolationWarning,
)

UNTYPED_PLAN: Any = ParametricPlan
UNTYPED_SET: Any = ScenarioSet
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
POINTS = {"warehouse-a": (6.96, 50.94), "depot-b": (7.50, 51.00)}
CELLS = {name: point_to_cell(lon, lat, 7) for name, (lon, lat) in POINTS.items()}
EARLY = (2025, 2030, 2035)
MIDDLE = (2045, 2050, 2055)
LATE = (2075, 2080, 2085)
PATHWAYS = ("ssp245", "ssp585")
WINDOWS = {
    "2030s": [("ssp585", year) for year in EARLY],
    "2050s": [("ssp585", year) for year in MIDDLE],
}


def _provenance(provider: str = "example", dataset: str = "synthetic") -> Any:
    return SourceProvenance(
        provider=provider,
        dataset=dataset,
        version="1",
        licence="CC0-1.0",
        attribution="Synthetic data",
    )


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        yield


# -- fitted curves -----------------------------------------------------------------


def _metadata(provider: str = "example") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=_provenance(provider),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 500.0),
    )


def _curve(
    asset: str, horizon: int, pathway: str = "ssp585", **fields: Any
) -> dict[str, Any]:
    row = {
        "cell_index": CELLS[asset],
        "source_id": f"fixture-{asset}",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 48.0
        + 0.2 * (horizon - 2025)
        + (0.0 if asset == "warehouse-a" else 4.0),
        "curve_scale": 12.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    row.update(fields)
    return row


def _dataset(
    path: Path,
    horizons: Sequence[int] = (*EARLY, *MIDDLE),
    *,
    provider: str = "example",
    pathway: str = "ssp585",
    hazard: str = "rx1day",
) -> Path:
    stated = _metadata(provider)
    rows = [
        _curve(asset, horizon, pathway, hazard_name=hazard)
        for asset in POINTS
        for horizon in horizons
    ]
    table = pa.Table.from_pylist(rows, schema=hazard_arrow_schema(stated))
    write_hazard_dataset(table, path, stated, max_workers=1)
    return path


def _assets(*ids: str) -> Any:
    ids = ids or tuple(POINTS)
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [POINTS[i][0] for i in ids],
            "latitude": [POINTS[i][1] for i in ids],
            "value_eur": [2e6 + 1e6 * k for k in range(len(ids))],
        }
    )


def _request(path: Path, **select: Any) -> Any:
    selection: dict[str, Any] = {
        "hazard_names": ["rx1day"],
        "horizons": [*EARLY, *MIDDLE],
        "pathways": ["ssp585"],
    }
    selection.update(select)
    return (
        HazardDataset.local(path)
        .for_assets(_assets())
        .select(**selection)
        .impact(RAMP, name="ramp", value_unit="1", value_semantics="damage ratio")
    )


def _step() -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        label="step",
    )


def _linear(asset_id: str | None = None) -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        label="linear",
        asset_id=asset_id,
    )


def _curve_plan(path: Path, **select: Any) -> Any:
    return (
        UNTYPED_PLAN.for_evaluation(
            _request(path, **select), value_column="value_eur", value_unit="EUR"
        )
        .monte_carlo(600, seed=42)
        .basis_risk(ImpactNoise(0.3))
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=0.8))
        .metrics(OPTIONS)
    )


@pytest.fixture
def curves(tmp_path: Path) -> Path:
    return _dataset(tmp_path / "rx1day.parquet")


def _loss(asset_id: str) -> LossModel:
    return LossModel(
        RAMP,
        value_column="value_eur",
        value_unit="EUR",
        context=TransformContext(cell=CELLS[asset_id]),
    ).for_asset(AssetPortfolio(_assets()), asset_id)


def _rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Each asset's canonical rows, by asset and horizon."""
    request = _request(path, horizons=None, pathways=None)
    return {
        (row["asset_id"], row["horizon"]): row
        for row in request.asset_curves().to_pylist()
    }


def _drawn(
    path: Path, horizons: Sequence[int], label: str, n: int, seed: int
) -> dict[str, OutcomeSample]:
    """The function route: each asset's draws from a window, a child generator each."""
    rows = _rows(path)
    metadata = HazardDataset.local(path).metadata()
    children = np.random.SeedSequence(seed).spawn(len(POINTS))
    drawn: dict[str, OutcomeSample] = {}
    for asset_id, child in zip(POINTS, children, strict=True):
        members = [rows[(asset_id, horizon)] for horizon in horizons]
        drawn[asset_id] = sample_curve(
            members if len(members) > 1 else members[0],
            metadata=metadata,
            n_draws=n,
            rng=np.random.default_rng(child),
            loss_model=_loss(asset_id),
            basis_risk=ImpactNoise(0.3),
            asset_id=asset_id,
            scenario=label,
        )
    return drawn


def _plan_rows(table: Any, **match: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in table.to_pylist()
        if all(row[key] == value for key, value in match.items())
    ]


def test_windows_on_curves_give_the_function_routes_rows(
    curves: Path, tmp_path: Path
) -> None:
    plan = _curve_plan(curves).split(FreshDraws(600, seed=7))
    plan = plan.scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    result = plan.write_parquet(tmp_path / "out" / "scenarios.parquet")
    # One file, written atomically: nothing else beside it.
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["scenarios.parquet"]
    assert result.scenario_samples is not None
    assert result.scenario_fresh_samples is not None
    assert list(result.scenario_samples) == ["2030s", "2050s"]
    # The samples are sample_curve's on each window, from the plan's generators.
    for label, horizons in (("2030s", EARLY), ("2050s", MIDDLE)):
        drawn = _drawn(curves, horizons, label, 600, 42)
        fresh = _drawn(curves, horizons, label, 600, 7)
        for asset_id in POINTS:
            assert (
                result.scenario_samples[label][asset_id].content_hash
                == drawn[asset_id].content_hash
            )
            assert (
                result.scenario_fresh_samples[label][asset_id].content_hash
                == fresh[asset_id].content_hash
            )
    # .samples() and the result's samples are the reference's.
    assert {s.content_hash for s in result.samples.values()} == {
        s.content_hash for s in result.scenario_samples["2030s"].values()
    }
    assert [s.content_hash for s in plan.samples().values()] == [
        s.content_hash for s in result.samples.values()
    ]
    # The designs: scenario, asset, fold, cover and objective order.
    assert [(d.design_sample.scenario, d.cover.asset_id) for d in result.designs] == [
        ("2030s", "warehouse-a"),
        ("2030s", "depot-b"),
        ("2050s", "warehouse-a"),
        ("2050s", "depot-b"),
    ]
    for planned in result.designs:
        sample = planned.design_sample
        by_hand = design(
            _step(), sample, MSE(), budget=Budget(ratio=0.8)
        )  # written for no asset, the same terms
        assert planned.parameters == by_hand.parameters
    # The rows: compare_scenarios on each asset's blocks, row for row.
    for asset_id in POINTS:
        designs = [d for d in result.designs if d.cover.asset_id == asset_id]
        table = compare_scenarios(
            [*designs, _linear(asset_id)],
            ScenarioSet(
                [result.scenario_fresh_samples[label][asset_id] for label in WINDOWS],
                reference="2030s",
            ),
            design_sample=result.scenario_samples["2030s"][asset_id],
            options=OPTIONS,
        )
        assert _plan_rows(result.table, asset_id=asset_id) == table.to_pylist()
    rows = result.table.to_pylist()
    assert {row["split"] for row in rows} == {"out_of_sample"}
    assert {(row["scenario"], row["horizon"]) for row in rows} == {
        ("2030s", 2030),
        ("2050s", 2050),
    }
    assert {row["premium_basis"] for row in rows if row["cover_label"] == "linear"} == {
        "reference"
    }
    # Drift works on the plan's table, from the set's reference.
    drift = to_drift(result.table)
    assert set(drift.column("reference_scenario").to_pylist()) == {"2030s"}


def test_the_metadata_records_the_scenarios_and_their_sources(
    curves: Path, tmp_path: Path
) -> None:
    plan = _curve_plan(curves).split(FreshDraws(600, seed=7))
    plan = plan.scenarios(
        ScenarioSet(reference="2030s", labels=WINDOWS), design_on="reference"
    )
    result = plan.write_parquet(tmp_path / "scenarios.parquet")
    document = result.metadata.model_dump(mode="json")
    scenarios = document["setup"]["scenarios"]
    assert scenarios["design_on"] == "reference"
    assert scenarios["reference"] == "2030s"
    assert scenarios["labels"] == ["2030s", "2050s"]
    assert scenarios["scenarios"]["2050s"]["members"] == [
        ["ssp585", year] for year in MIDDLE
    ]
    # Each asset's block, from both assets' tables.
    assert len(scenarios["scenarios"]["2030s"]["blocks"]) == 2
    assert set(scenarios["shared_draws"]["rng"]) == {
        "asset 'warehouse-a'",
        "asset 'depot-b'",
    }
    # One record of the dataset, read by both scenarios; each source names its own.
    (record,) = document["source_records"]
    assert record["canonical_file"]["scenarios"] == ["2030s", "2050s"]
    hazards = [s for s in document["sources"] if s["role"] == "hazard"]
    assert {s["scenario"] for s in hazards} == {"2030s", "2050s"}
    assert {s["canonical_file_sha256"] for s in hazards} == {
        record["canonical_file"]["file_sha256"]
    }
    # design_on="reference": every design is the reference's.
    assert {d.design_sample.scenario for d in result.designs} == {"2030s"}
    assert {
        row["design_scenario"]
        for row in result.table.to_pylist()
        if row["design_id"] is not None
    } == {"2030s"}
    text = setup_text(result.metadata)
    assert "Scenario 2030s (the reference)" in text
    assert "asset warehouse-a: 600 simulated years" in text
    # The written file reads back as written.
    assert EvaluationMetadata.from_table(result.table) == result.metadata


def test_a_single_horizon_and_a_window_under_labels(curves: Path) -> None:
    labels = {"2030s": WINDOWS["2030s"], "mid-century": [("ssp585", 2050)]}
    plan = _curve_plan(curves).split(Holdout(0.5))
    plan = plan.scenarios(ScenarioSet(reference="2030s", labels=labels))
    samples = plan.scenario_samples()
    window = samples["2030s"]["warehouse-a"].provenance.details["window"]
    assert [list(pair) for pair in window["members"]] == [
        ["ssp585", year] for year in EARLY
    ]
    assert "window" not in samples["mid-century"]["warehouse-a"].provenance.details
    # A single horizon is a plain sample_curve of that curve.
    single = _drawn(curves, (2050,), "mid-century", 600, 42)
    assert (
        samples["mid-century"]["warehouse-a"].content_hash
        == single["warehouse-a"].content_hash
    )
    document: Any = plan.explain(format="json")
    entries = document["setup"]["scenarios"]["scenarios"]
    assert entries["2030s"]["members"] == [["ssp585", year] for year in EARLY]
    assert entries["2030s"]["horizon"] == 2030
    assert entries["mid-century"]["members"] == [["ssp585", 2050]]
    assert entries["mid-century"]["horizon"] == 2050
    text = setup_text(document)
    assert "A window of 3 horizons: ssp585 2025, ssp585 2030 and ssp585 2035" in text
    assert "A single horizon: ssp585 2050." in text


def test_without_labels_each_selected_pair_is_a_scenario(curves: Path) -> None:
    plan = _curve_plan(curves, horizons=[2030, 2050]).split(Holdout(0.5))
    plan = plan.scenarios(ScenarioSet(reference="ssp585_2030"))
    document: Any = plan.explain(format="json")
    assert document["scenarios"]["labels"] == ["ssp585_2030", "ssp585_2050"]
    assert list(plan.scenario_samples()) == ["ssp585_2030", "ssp585_2050"]


def test_several_pathways_are_scenarios_too(tmp_path: Path) -> None:
    stated = _metadata()
    rows = [
        _curve(asset, 2050, pathway, curve_location=50.0 + 3.0 * k)
        for asset in POINTS
        for k, pathway in enumerate(PATHWAYS)
    ]
    path = tmp_path / "pathways.parquet"
    table = pa.Table.from_pylist(rows, schema=hazard_arrow_schema(stated))
    write_hazard_dataset(table, path, stated, max_workers=1)
    plan = _curve_plan(path, horizons=[2050], pathways=list(PATHWAYS))
    with pytest.raises(ValueError, match="selects 2 pathways; select one"):
        plan.explain()
    plan = plan.split(FreshDraws(600, seed=7)).scenarios(
        ScenarioSet(reference="ssp245_2050")
    )
    result = plan.write_parquet(tmp_path / "pathways-out.parquet")
    assert {(r["scenario"], r["pathway"]) for r in result.table.to_pylist()} == {
        ("ssp245_2050", "ssp245"),
        ("ssp585_2050", "ssp585"),
    }


def test_joint_designs_on_curves_give_the_function_routes_rows(
    curves: Path, tmp_path: Path
) -> None:
    plan = (
        _curve_plan(curves)
        .scopes(Joint())
        .dependence(Independent())
        .split(Holdout(0.5))
        .scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    )
    result = plan.write_parquet(tmp_path / "joint.parquet")
    rows = _drawn_joint(curves)
    metadata = HazardDataset.local(curves).metadata()
    assert result.scenario_samples is not None
    blocks: dict[str, tuple[OutcomeSample, OutcomeSample]] = {}
    for label, horizons in (("2030s", EARLY), ("2050s", MIDDLE)):
        children = np.random.SeedSequence(42).spawn(2)
        joint = sample_curves(
            {asset_id: [rows[(asset_id, h)] for h in horizons] for asset_id in POINTS},
            dependence=Independent(),
            metadata=dict.fromkeys(POINTS, metadata),
            n_draws=600,
            rngs={
                asset_id: np.random.default_rng(child)
                for asset_id, child in zip(POINTS, children, strict=True)
            },
            loss_models={asset_id: _loss(asset_id) for asset_id in POINTS},
            basis_risk=ImpactNoise(0.3),
            scenario=label,
        )
        for asset_id in POINTS:
            assert (
                joint.asset(asset_id).content_hash
                == result.scenario_samples[label][asset_id].content_hash
            )
        blocks[label] = holdout(joint, 0.5)
    # Joint designs, in scenario, fold, cover and objective order.
    assert [d.design_sample.scenario for d in result.designs] == ["2030s", "2050s"]
    for planned in result.designs:
        train = blocks[planned.design_sample.scenario][0]
        by_hand = design_joint(
            [_step(), _step()], train, MSE(), budget=Budget(ratio=0.8)
        )
        assert planned.parameters == by_hand.parameters
    linear = {asset_id: _linear(asset_id) for asset_id in POINTS}
    table = compare_scenarios(
        [*result.designs, linear],
        ScenarioSet([blocks[label][1] for label in WINDOWS], reference="2030s"),
        design_sample=blocks["2030s"][0],
        options=OPTIONS,
    )
    assert result.table.to_pylist() == table.to_pylist()
    assert {row["level"] for row in result.table.to_pylist()} == {"asset", "total"}


def _drawn_joint(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return _rows(path)


def test_another_canonical_dataset_is_a_curve_scenario(
    curves: Path, tmp_path: Path
) -> None:
    later = _dataset(tmp_path / "later.parquet", LATE)
    entry = CurveScenario("2080s", later, pathway="ssp585", horizon=LATE)
    plan = _curve_plan(curves).split(FreshDraws(600, seed=7))
    plan = plan.scenarios(
        ScenarioSet([entry], reference="2030s", labels={"2030s": WINDOWS["2030s"]})
    )
    document: Any = plan.explain(format="json")
    assert document["scenarios"]["scenarios"]["2080s"] == {
        "kind": "dataset",
        "members": [["ssp585", year] for year in LATE],
        "path": str(later),
    }
    paths = [(s["path"], s["scenarios"]) for s in document["sources"]]
    assert paths == [(str(curves), ["2030s"]), (str(later), ["2080s"])]
    result = plan.write_parquet(tmp_path / "two.parquet")
    assert result.scenario_samples is not None
    drawn = _drawn(later, LATE, "2080s", 600, 42)
    assert (
        result.scenario_samples["2080s"]["depot-b"].content_hash
        == drawn["depot-b"].content_hash
    )
    records = result.metadata.model_dump(mode="json")["source_records"]
    assert [r["canonical_file"]["scenarios"] for r in records] == [
        ["2030s"],
        ["2080s"],
    ]
    # A dataset of one pair needs no pathway or horizon.
    single = _dataset(tmp_path / "single.parquet", (2080,))
    alone = (
        _curve_plan(curves)
        .split(FreshDraws(600, seed=7))
        .scenarios(
            ScenarioSet(
                [CurveScenario("2080", single)],
                reference="2030s",
                labels={"2030s": WINDOWS["2030s"]},
            )
        )
    )
    assert (
        "window" not in alone.scenario_samples()["2080"]["depot-b"].provenance.details
    )


def test_a_curve_scenarios_refusals(curves: Path, tmp_path: Path) -> None:
    base = _curve_plan(curves).split(FreshDraws(600, seed=7))
    labels = {"2030s": WINDOWS["2030s"]}

    def run(entry: Any, **options: Any) -> Any:
        stated = ScenarioSet([entry], reference="2030s", labels=labels, **options)
        return base.scenarios(stated).scenario_samples()

    later = _dataset(tmp_path / "later.parquet", LATE)
    with pytest.raises(ScenarioError, match="holds the horizons"):
        run(CurveScenario("2080s", later))
    with pytest.raises(ScenarioError, match="doesn't hold"):
        run(CurveScenario("2080s", later, pathway="ssp585", horizon=(2075, 2090)))
    other = _dataset(tmp_path / "heat.parquet", (2080,), hazard="txx")
    with pytest.raises(ScenarioError, match=r"holds the hazards \['txx'\]"):
        run(CurveScenario("heat", other))
    # Another product needs mixed_products=True, the reference included.
    product = _dataset(tmp_path / "product.parquet", (2080,), provider="other")
    with pytest.raises(ScenarioError, match="product"):
        run(CurveScenario("other", product))
    assert run(CurveScenario("other", product), mixed_products=True)
    # One pathway, horizon and product twice.
    same = _dataset(tmp_path / "same.parquet", EARLY)
    with pytest.raises(ScenarioError, match="one pathway, horizon and product"):
        run(CurveScenario("again", same, pathway="ssp585", horizon=EARLY))


def test_several_horizons_need_scenarios_and_a_split(curves: Path) -> None:
    plan = _curve_plan(curves)
    with pytest.raises(ValueError, match="selects 6 horizons"):
        plan.explain()
    with pytest.raises(ValueError, match="selects 6 horizons"):
        plan.designs()
    stated = plan.scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    for run in (stated.explain, stated.samples, stated.scenario_samples):
        with pytest.raises(ValueError, match="needs .split"):
            run()


def test_explain_lists_the_scenarios_without_io(tmp_path: Path) -> None:
    absent = tmp_path / "absent.parquet"
    elsewhere = tmp_path / "elsewhere.parquet"
    plan = (
        _curve_plan(absent)
        .split(FreshDraws(900, seed=7))
        .scenarios(
            ScenarioSet(
                [CurveScenario("2080s", elsewhere, pathway="ssp585", horizon=LATE)],
                reference="2030s",
                labels=WINDOWS,
            ),
            premiums={"linear": 3e4},
        )
    )
    document: Any = plan.explain(format="json")
    plan.explain()
    assert sorted(p.name for p in tmp_path.iterdir()) == []
    scenarios = document["scenarios"]
    assert scenarios["reference"] == "2030s"
    assert scenarios["design_on"] == "each"
    assert scenarios["premiums"] == {"linear": 3e4}
    assert scenarios["labels"] == ["2030s", "2050s", "2080s"]
    assert scenarios["scenarios"]["2050s"]["members"] == [
        ["ssp585", year] for year in MIDDLE
    ]
    setup = document["setup"]
    assert setup["scenarios"]["shared_draws"]["labels"] == ["2030s", "2050s", "2080s"]
    assert {s["scenario"] for s in setup["samples"]} == {"2030s", "2050s", "2080s"}
    evaluations = setup["evaluations"]
    assert {(e["scenario"], e["design_scenario"]) for e in evaluations} == {
        (label, design)
        for label in ("2030s", "2050s", "2080s")
        for design in ("2030s", "2050s", "2080s", None)
    }
    assert {
        e["method"] for e in evaluations if e["design_scenario"] == e["scenario"]
    } == {"fresh_draws"}
    text = setup_text(document)
    assert "Scenario 2080s: hazard rx1day, as its source names it" in text
    assert "rows: not known yet" in text
    assert repr(plan).startswith("ParametricPlan.for_evaluation(absent.parquet")
    assert not absent.exists() and not elsewhere.exists()


def test_explain_checks_every_windows_draws(curves: Path) -> None:
    stated = ScenarioSet(reference="2030s", labels=WINDOWS)
    plan = _curve_plan(curves).scenarios(stated)
    odd = plan.monte_carlo(601, seed=42).split(FreshDraws(600, seed=7))
    with pytest.raises(ScenarioError, match="n_draws must be a whole multiple of 3"):
        odd.explain()
    with pytest.raises(ScenarioError, match="nearest valid counts are 600 and 603"):
        odd.samples()
    fresh = plan.split(FreshDraws(500, seed=7))
    with pytest.raises(
        ScenarioError, match=r"FreshDraws\(n\) must be a whole multiple"
    ):
        fresh.explain()
    # A single horizon has no members to divide by.
    single = plan.scenarios(
        ScenarioSet(reference="one", labels={"one": [("ssp585", 2030)]})
    )
    single.monte_carlo(601, seed=42).split(FreshDraws(601, seed=7)).explain()


def test_the_scenarios_builder_refuses_what_it_can(curves: Path) -> None:
    plan = _curve_plan(curves).split(FreshDraws(600, seed=7))
    sample = _drawn(curves, (2030,), "built", 30, 1)["warehouse-a"]
    with pytest.raises(TypeError, match="built samples"):
        plan.scenarios(ScenarioSet([sample], reference="built"))
    with pytest.raises(TypeError, match="ScenarioSet"):
        plan.scenarios(WINDOWS)
    with pytest.raises(ValueError, match="design_on"):
        plan.scenarios(ScenarioSet(reference="2030s", labels=WINDOWS), design_on="all")
    years = AnnualScenario("future", _years(2041), provenance=_provenance())
    with pytest.raises(ValueError, match="AnnualScenario 'future' reads per-year"):
        plan.scenarios(ScenarioSet([years], reference="future"))
    with pytest.raises(ScenarioError, match="the request doesn't select"):
        plan.scenarios(
            ScenarioSet(reference="late", labels={"late": [("ssp585", 2080)]})
        )
    with pytest.raises(ScenarioError, match="the reference 'nowhere'"):
        plan.scenarios(ScenarioSet(reference="nowhere"))
    with pytest.raises(ScenarioError, match="labelled '2030s'"):
        plan.scenarios(
            ScenarioSet(
                [CurveScenario("2030s", curves)], reference="2030s", labels=WINDOWS
            )
        )
    unnamed = UNTYPED_PLAN.for_evaluation(
        _request(curves, hazard_names=None), value_column="value_eur", value_unit="EUR"
    )
    with pytest.raises(ValueError, match="exactly one hazard"):
        unnamed.scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    open_horizons = UNTYPED_PLAN.for_evaluation(
        _request(curves, horizons=None), value_column="value_eur", value_unit="EUR"
    )
    with pytest.raises(ValueError, match="must state horizons"):
        open_horizons.scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    stated = plan.scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    with pytest.raises(EvaluationError, match=r"premiums= names \['step'\]"):
        stated.scenarios(
            ScenarioSet(reference="2030s", labels=WINDOWS), premiums={"step": 1.0}
        ).explain()
    with pytest.raises(EvaluationError, match="finite and non-negative"):
        plan.scenarios(
            ScenarioSet(reference="2030s", labels=WINDOWS), premiums={"linear": -1.0}
        )
    # Allocated rows across scenarios are offered: they go to a file of their own.
    joint = stated.scopes(Joint()).dependence(Independent()).allocation("value")
    assert joint.explain(format="json")["allocation"] == "value"
    with pytest.raises(ValueError, match="holds every scenario's samples"):
        _curve_plan(curves, horizons=[2030]).scenario_samples()
    # There is no per-year option on the split: the rule makes every scenario's blocks.
    assert list(inspect.signature(ParametricPlan.split).parameters) == ["self", "rule"]


def test_premiums_state_a_given_covers_price_on_every_scenario(
    curves: Path, tmp_path: Path
) -> None:
    plan = (
        _curve_plan(curves)
        .split(FreshDraws(600, seed=7))
        .scenarios(
            ScenarioSet(reference="2030s", labels=WINDOWS), premiums={"linear": 3e4}
        )
    )
    rows = plan.write_parquet(tmp_path / "stated.parquet").table.to_pylist()
    given = [row for row in rows if row["cover_label"] == "linear"]
    assert {(row["premium_basis"], row["premium"]) for row in given} == {
        ("stated", 3e4)
    }


# -- per-year values ----------------------------------------------------------------


def _years(
    first: int,
    count: int = 30,
    *,
    location: float = 50.0,
    seed: int = 1,
    **columns: Any,
) -> Any:
    years = list(range(first, first + count))
    rows = {
        "year": pa.array(years * 2, pa.int32()),
        "latitude": [50.94] * count + [51.00] * count,
        "longitude": [6.96] * count + [7.50] * count,
        "value": np.random.default_rng(seed).gumbel(location, 12.0, 2 * count),
    }
    for name, value in columns.items():
        rows[name] = [value] * (2 * count)
    return pa.table(rows)


GRID = NearestCell(spacing=(0.06, 0.54))


def _annual_plan(source: Any = None, **options: Any) -> Any:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "provenance": _provenance(dataset="historical years"),
        "value_column": "value_eur",
        "value_unit": "EUR",
        "location": GRID,
        "scenario": "historical",
        "hazard_name": "rx1day",
        "seed": 3,
    }
    arguments.update(options)
    return (
        UNTYPED_PLAN.for_annual_extremes(
            _years(1985) if source is None else source, _assets(), **arguments
        )
        .basis_risk(ImpactNoise(0.3))
        .covers(_step(), _linear())
        .objective(ExpectedUncovered(), budget=Budget(ratio=0.8))
        .metrics(OPTIONS)
    )


def _future(**options: Any) -> AnnualScenario:
    arguments: dict[str, Any] = {
        "provenance": _provenance(dataset="future years"),
        "pathway": "ssp585",
        "hazard_name": "rx1day",
    }
    arguments.update(options)
    return AnnualScenario("ssp585", _years(2041, location=58.0, seed=2), **arguments)


def _series(table: Any, asset_id: str, dataset: str, **provenance: Any) -> Any:
    lon, lat = POINTS[asset_id]
    series = AnnualSeries.nearest_cell(
        table,
        longitude=lon,
        latitude=lat,
        unit="mm/day",
        spacing=GRID.spacing,
        source=_provenance(dataset=dataset).model_dump(mode="json"),
    )
    if not provenance:
        return series
    return AnnualSeries(
        series.years, series.values, series.unit, {**series.provenance, **provenance}
    )


def _annual_samples(
    table: Any, label: str, dataset: str, **provenance: Any
) -> dict[str, OutcomeSample]:
    children = np.random.SeedSequence(3).spawn(len(POINTS))
    return {
        asset_id: sample_annual(
            _series(table, asset_id, dataset, **provenance),
            loss_model=LossModel(
                RAMP, value_column="value_eur", value_unit="EUR"
            ).for_asset(AssetPortfolio(_assets()), asset_id),
            basis_risk=ImpactNoise(0.3),
            asset_id=asset_id,
            rng=np.random.default_rng(child),
            scenario=label,
            hazard_name="rx1day",
        )
        for asset_id, child in zip(POINTS, children, strict=True)
    }


def test_per_year_scenarios_give_the_function_routes_rows(tmp_path: Path) -> None:
    plan = (
        _annual_plan()
        .split(Holdout(0.7))
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    result = plan.write_parquet(tmp_path / "years.parquet")
    assert result.scenario_samples is not None
    assert result.scenario_fresh_samples is None
    own = _annual_samples(_years(1985), "historical", "historical years")
    future = _annual_samples(
        _years(2041, location=58.0, seed=2), "ssp585", "future years", pathway="ssp585"
    )
    for asset_id in POINTS:
        assert (
            result.scenario_samples["historical"][asset_id].content_hash
            == own[asset_id].content_hash
        )
        assert (
            result.scenario_samples["ssp585"][asset_id].content_hash
            == future[asset_id].content_hash
        )
        blocks = {
            label: holdout(result.scenario_samples[label][asset_id], 0.7)
            for label in ("historical", "ssp585")
        }
        designs = [d for d in result.designs if d.cover.asset_id == asset_id]
        assert [d.design_sample.content_hash for d in designs] == [
            blocks[label][0].content_hash for label in ("historical", "ssp585")
        ]
        table = compare_scenarios(
            [*designs, _linear(asset_id)],
            ScenarioSet(
                [blocks[label][1] for label in ("historical", "ssp585")],
                reference="historical",
            ),
            design_sample=blocks["historical"][0],
            options=OPTIONS,
        )
        assert _plan_rows(result.table, asset_id=asset_id) == table.to_pylist()
    rows = result.table.to_pylist()
    assert {(r["scenario"], r["pathway"], r["horizon"]) for r in rows} == {
        ("historical", None, 1999),
        ("ssp585", "ssp585", 2055),
    }
    document = result.metadata.model_dump(mode="json")
    records = [
        record["annual_table"]["scenarios"] for record in document["source_records"]
    ]
    assert records == [["historical"], ["ssp585"]]
    for source in document["sources"]:
        assert source["annual_table_sha256"] == next(
            r["annual_table"]["table_sha256"]
            for r in document["source_records"]
            if r["annual_table"]["scenarios"] == [source["scenario"]]
        )
    hazards = {
        label: samples["warehouse-a"].provenance.hazard
        for label, samples in result.scenario_samples.items()
    }
    assert hazards == {
        "historical": {"name": "rx1day", "from": "stated"},
        "ssp585": {"name": "rx1day", "from": "stated"},
    }
    text = setup_text(result.metadata)
    assert "different sources" in text
    assert "beta is the same on every scenario (0.9)" in text


def test_joint_designs_on_years_give_the_function_routes_rows(tmp_path: Path) -> None:
    plan = (
        _annual_plan()
        .scopes(Joint())
        .split(KFold(3))
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    result = plan.write_parquet(tmp_path / "joint.parquet")
    assert result.scenario_samples is not None
    joined = {
        label: OutcomeSample.join(result.scenario_samples[label].values())
        for label in ("historical", "ssp585")
    }
    folds = {label: year_block_kfold(joined[label], 3) for label in joined}
    assert [
        (d.design_sample.scenario, d.design_sample.content_hash) for d in result.designs
    ] == [
        (label, folds[label][fold][0].content_hash)
        for label in ("historical", "ssp585")
        for fold in range(3)
    ]
    linear = {asset_id: _linear(asset_id) for asset_id in POINTS}
    expected: list[dict[str, Any]] = []
    for fold in range(3):
        designs = [
            d
            for d in result.designs
            if d.design_sample.content_hash
            in {folds[label][fold][0].content_hash for label in folds}
        ]
        expected += compare_scenarios(
            [*designs, linear],
            ScenarioSet(
                [folds[label][fold][1] for label in ("historical", "ssp585")],
                reference="historical",
            ),
            fold=fold,
            design_sample=folds["historical"][fold][0],
            options=OPTIONS,
        ).to_pylist()
    assert result.table.to_pylist() == expected


def test_an_annual_scenario_takes_the_plans_fields(tmp_path: Path) -> None:
    trend = _annual_plan(trend=LinearTrend(min_years=10)).split(Holdout(0.7))
    stated = trend.scenarios(ScenarioSet([_future()], reference="historical"))
    samples = stated.scenario_samples()
    future = samples["ssp585"]["warehouse-a"]
    assert future.provenance.trend is not None
    assert future.provenance.trend["kind"] == "linear_trend"
    # A field left None takes the plan's; its years are its own.
    assert list(future.provenance.details["years"]) == [2041, 2070]
    pathways = future.provenance.sources[0]["series"]["pathway"]
    assert pathways == "ssp585"
    # .index(...) reads every scenario's index, on its own years: a station index
    # with no future years is refused, under the plan's missing=.
    stations = pa.table(
        {
            "station": ["S1"] * 30,
            "year": pa.array(range(1985, 2015), pa.int32()),
            "value": np.linspace(40.0, 90.0, 30),
        }
    )
    assets = _assets().append_column("gauge", pa.array(["S1", "S1"]))
    plan = (
        UNTYPED_PLAN.for_annual_extremes(
            _years(1985),
            assets,
            impact=RAMP,
            unit="mm/day",
            provenance=_provenance(dataset="historical years"),
            value_column="value_eur",
            value_unit="EUR",
            location=GRID,
            scenario="historical",
            hazard_name="rx1day",
            seed=3,
        )
        .index(
            stations,
            location=StationColumn("gauge"),
            unit="mm/day",
            provenance=_provenance(dataset="gauge"),
        )
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .split(Holdout(0.7))
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    with pytest.raises(SampleError, match="no value"):
        plan.scenario_samples()


def test_per_year_scenarios_refuse_what_differs(tmp_path: Path) -> None:
    base = _annual_plan().split(Holdout(0.7))

    def run(entry: AnnualScenario, **options: Any) -> Any:
        stated = ScenarioSet([entry], reference="historical", **options)
        return base.scenarios(stated).scenario_samples()

    # A silent source needs a stated hazard name; the plan's isn't a default.
    with pytest.raises(ScenarioError, match="names no hazard_name"):
        run(_future(hazard_name=None))
    with pytest.raises(ScenarioError, match="hazard_name"):
        run(_future(hazard_name="rx5day"))
    # A stated pathway the table's scenario column contradicts.
    contradicted = AnnualScenario(
        "ssp585",
        _years(2041, seed=2, model="m1", member="r1i1p1f1", scenario="ssp245"),
        provenance=_provenance(dataset="future years"),
        pathway="ssp585",
        hazard_name="rx1day",
        pool=False,
    )
    with pytest.raises(ScenarioError, match="contradicts itself on the pathway"):
        run(contradicted)
    # Another product needs mixed_products=True.
    other = _future(provenance=_provenance(provider="other"))
    with pytest.raises(ScenarioError, match="product"):
        run(other)
    assert run(other, mixed_products=True)
    # Another trend option.
    with pytest.raises(ScenarioError, match="trend"):
        run(_future(trend=LinearTrend(min_years=10)))
    # The plan's own label is a scenario's: another may not take it.
    with pytest.raises(ScenarioError, match="labelled 'historical'"):
        base.scenarios(
            ScenarioSet(
                [AnnualScenario("historical", _years(2041), provenance=_provenance())],
                reference="historical",
            )
        )
    with pytest.raises(ValueError, match="CurveScenario 'curves' reads fitted"):
        base.scenarios(
            ScenarioSet(
                [CurveScenario("curves", tmp_path / "c.parquet")],
                reference="historical",
            )
        )
    with pytest.raises(ValueError, match="labels= names"):
        base.scenarios(ScenarioSet(reference="x", labels={"x": [("ssp585", 2050)]}))
    # A unit of its own is the scenario's, and the set compares the units.
    with pytest.raises(ScenarioError, match="index_unit"):
        run(_future(unit="mm"))


def _stations(first: int, stations: Sequence[str], seed: int) -> Any:
    years = list(range(first, first + 30))
    return pa.table(
        {
            "gauge": [s for s in stations for _ in years],
            "year": pa.array(years * len(stations), pa.int32()),
            "value": np.random.default_rng(seed).gumbel(55.0, 12.0, 30 * len(stations)),
        }
    )


def test_stations_under_a_renamed_column_are_compared_by_place(tmp_path: Path) -> None:
    assets = _assets().append_column("gauge_id", pa.array(["S1", "S2"]))
    assets = assets.append_column("other_gauge", pa.array(["S3", "S4"]))
    plan = (
        UNTYPED_PLAN.for_annual_extremes(
            _stations(1985, ["S1", "S2", "S3", "S4"], 1),
            assets,
            impact=RAMP,
            unit="mm/day",
            provenance=_provenance(dataset="gauges"),
            value_column="value_eur",
            value_unit="EUR",
            location=StationColumn("gauge_id", table_column="gauge"),
            scenario="historical",
            hazard_name="rx1day",
            seed=3,
        )
        .basis_risk(ImpactNoise(0.3))
        .covers(_linear())
        .metrics(OPTIONS)
        .split(Holdout(0.7))
    )
    later = _stations(2041, ["S1", "S2", "S3", "S4"], 2)
    same = AnnualScenario(
        "ssp585",
        later,
        provenance=_provenance(dataset="later gauges"),
        hazard_name="rx1day",
    )
    # The plan's own stations: one place in both scenarios.
    result = plan.scenarios(ScenarioSet([same], reference="historical")).write_parquet(
        tmp_path / "gauges.parquet"
    )
    series = result.samples["warehouse-a"].provenance.sources[0]["series"]
    assert series["columns"]["station"] == "gauge"
    # A scenario that reads another station under the same asset id is refused.
    elsewhere = AnnualScenario(
        "ssp585",
        later,
        provenance=_provenance(dataset="later gauges"),
        hazard_name="rx1day",
        location=StationColumn("other_gauge", table_column="gauge"),
    )
    with pytest.raises(ScenarioError, match="station"):
        plan.scenarios(ScenarioSet([elsewhere], reference="historical")).samples()


def test_pools_are_scenarios_of_their_own(tmp_path: Path) -> None:
    def pool(first: int, pathway: str, seed: int) -> Any:
        tables = [
            _years(
                first, seed=seed + k, model=model, member="r1i1p1f1", scenario=pathway
            )
            for k, model in enumerate(("m1", "m2"))
        ]
        return pa.concat_tables(tables)

    plan = _annual_plan(pool(1985, "historical", 1), pool=ModelPool()).split(
        Holdout(0.7)
    )
    future = AnnualScenario(
        "ssp585",
        pool(2041, "ssp585", 5),
        provenance=_provenance(dataset="future years"),
        hazard_name="rx1day",
    )
    stated = plan.scenarios(ScenarioSet([future], reference="historical"))
    samples = stated.scenario_samples()
    assert samples["ssp585"]["depot-b"].provenance.pool is not None
    result = stated.write_parquet(tmp_path / "pools.parquet")
    assert {r["pathway"] for r in result.table.to_pylist()} == {"historical", "ssp585"}
    # A series beside pools needs mixed_products=True, and pool=False states one.
    series = AnnualScenario(
        "ssp585",
        _years(2041, seed=2),
        provenance=_provenance(dataset="future years"),
        hazard_name="rx1day",
        pathway="ssp585",
        pool=False,
    )
    with pytest.raises(ScenarioError, match="pool"):
        plan.scenarios(ScenarioSet([series], reference="historical")).samples()


class _SourcePlan:
    """A stand-in for a crc-sdk plan with an annual_extremes() execution point."""

    def __init__(self, table: Any, recipe: str = "rx1day") -> None:
        self.table = table
        self.recipe = recipe
        self.reads = 0

    def annual_extremes(self) -> Any:
        self.reads += 1
        return self.table

    def explain(self, *, format: str = "text") -> Any:
        return {"recipe": self.recipe, "unit": "mm/day", "pathway": "ssp585"}


def test_a_source_plan_names_the_hazard(tmp_path: Path) -> None:
    source = _SourcePlan(_years(2041, location=58.0, seed=2))
    entry = AnnualScenario(
        "ssp585", source, provenance=_provenance(dataset="future years")
    )
    plan = _annual_plan().split(Holdout(0.7))
    samples = plan.scenarios(ScenarioSet([entry], reference="historical"))
    samples.explain()
    assert source.reads == 0  # explain() reads no data, the source plan's included
    built = samples.scenario_samples()["ssp585"]["warehouse-a"]
    assert source.reads == 1
    assert built.provenance.hazard == {"name": "rx1day", "from": "source_plan"}
    document: Any = samples.explain(format="json")
    entries = document["setup"]["scenarios"]["scenarios"]
    assert entries["ssp585"]["hazard"] == {"name": "rx1day", "from": "source_plan"}
    assert entries["ssp585"]["pathway"] == "ssp585"
    # A stated name the source plan contradicts raises before anything is read.
    named = _annual_plan(_SourcePlan(_years(1985)), hazard_name="rx5day")
    with pytest.raises(ScenarioError, match="contradicts the source plan"):
        named.explain()
    # And a stated pathway it contradicts, at execution.
    wrong = AnnualScenario(
        "ssp245",
        source,
        provenance=_provenance(dataset="future years"),
        pathway="ssp245",
    )
    with pytest.raises(ScenarioError, match="pathway='ssp245' contradicts"):
        plan.scenarios(ScenarioSet([wrong], reference="historical")).samples()


def test_a_plan_without_scenarios_records_none(tmp_path: Path) -> None:
    result = _annual_plan(hazard_name=None).write_parquet(tmp_path / "plain.parquet")
    document = result.metadata.model_dump(mode="json")
    assert document["setup"].get("scenarios") is None
    assert result.scenario_samples is None and result.scenario_fresh_samples is None
    assert all("scenario" not in source for source in document["sources"])
    assert all(
        "scenarios" not in record["annual_table"]
        for record in document["source_records"]
    )
    # No hazard is recorded without a name or scenarios, as before.
    assert all(sample.provenance.hazard is None for sample in result.samples.values())


def test_a_stated_hazard_name_is_recorded_without_scenarios(tmp_path: Path) -> None:
    samples = _annual_plan().samples()
    assert {s.provenance.hazard["name"] for s in samples.values()} == {"rx1day"}


def test_the_descriptions_check_their_arguments(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="SourceProvenance"):
        AnnualScenario("x", _years(2041), provenance="example")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pool must be"):
        AnnualScenario("x", _years(2041), provenance=_provenance(), pool=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fit_years"):
        AnnualScenario(
            "x",
            _years(2041),
            provenance=_provenance(),
            trend=LinearTrend(fit_years=tuple(range(2041, 2060))),
        )
    with pytest.raises(ScenarioError, match="crc_sdk.pathways does not know"):
        AnnualScenario("x", _years(2041), provenance=_provenance(), pathway="ssp999")
    with pytest.raises(ValueError, match="models have no stations"):
        AnnualScenario(
            "x",
            _years(2041),
            provenance=_provenance(),
            pool=ModelPool(),
            location=StationColumn("gauge"),
        )
    with pytest.raises(TypeError, match="HazardDataset"):
        CurveScenario("x", 3)  # type: ignore[arg-type]
    with pytest.raises(ScenarioError, match="repeat"):
        CurveScenario("x", tmp_path / "c.parquet", horizon=(2050, 2050))
    entry = CurveScenario("x", tmp_path / "c.parquet", horizon=(2055, 2045, 2050))
    assert entry.horizon == (2045, 2050, 2055)
    assert CurveScenario("y", tmp_path / "c.parquet", horizon=[2050]).horizon == 2050
    sample = _drawn_sample()
    with pytest.raises(TypeError, match="never both"):
        UNTYPED_SET([sample, entry], reference="x")
    with pytest.raises(ScenarioError, match="labelled 'x'"):
        UNTYPED_SET([entry, entry], reference="x")
    with pytest.raises(ScenarioError, match="labelled 'x'"):
        UNTYPED_SET([entry], reference="x", labels={"x": [("ssp585", 2050)]})
    # A set of descriptions holds no samples, and compare_scenarios refuses it.
    described = ScenarioSet([entry], reference="x")
    assert "scenarios=['x']" in repr(described)
    with pytest.raises(TypeError, match="describes the scenarios a plan reads"):
        compare_scenarios([_linear()], described)
    assert repr(entry).startswith("CurveScenario('x', c.parquet")
    assert repr(AnnualScenario("z", _years(2041), provenance=_provenance())).startswith(
        "AnnualScenario('z', Table"
    )


def _drawn_sample() -> OutcomeSample:
    table = pa.table(
        {
            "asset": ["warehouse-a"] * 2,
            "year": [2000, 2001],
            "index": [1.0, 2.0],
            "loss": [0.0, 1.0],
        }
    )
    return OutcomeSample.from_table(
        table,
        period="year",
        asset_id="asset",
        index="index",
        loss="loss",
        index_unit="mm/day",
        loss_unit="EUR",
        scenario="built",
    )


def test_compare_on_one_scenario_is_unchanged(curves: Path, tmp_path: Path) -> None:
    """The plan's own compare rows and its rows on the reference alone agree."""
    plan = _curve_plan(curves, horizons=[2030]).split(FreshDraws(600, seed=7))
    alone = plan.write_parquet(tmp_path / "alone.parquet")
    stated = plan.scenarios(ScenarioSet(reference="ssp585_2030"))
    across = stated.write_parquet(tmp_path / "across.parquet")
    plain = [
        row
        for row in alone.table.to_pylist()
        if row["split"] == "out_of_sample" and row["cover_label"] != "linear"
    ]
    scenario = [
        row for row in across.table.to_pylist() if row["cover_label"] != "linear"
    ]
    keys = ("expected_payout", "expected_uncovered", "coverage_ratio")
    assert [[row[k] for k in keys] for row in plain] == [
        [row[k] for k in keys] for row in scenario
    ]


def test_folds_and_both_scopes_run_through_the_scenarios(
    curves: Path, tmp_path: Path
) -> None:
    stated = ScenarioSet(reference="2030s", labels=WINDOWS)
    folded = _curve_plan(curves).split(KFold(3)).scenarios(stated)
    result = folded.write_parquet(tmp_path / "folds.parquet")
    # 2 assets x 3 folds x 2 scenarios x (2 designs + 1 cover)
    assert result.row_count == 2 * 3 * 2 * 3
    assert [d.parameters for d in folded.designs()] == [
        d.parameters for d in result.designs
    ]
    blocks = result.metadata.setup.scenarios
    assert blocks is not None
    assert len(blocks["scenarios"]["2050s"]["blocks"]) == 2 * 3
    both = (
        _curve_plan(curves)
        .scopes(Individual(), Joint())
        .dependence(Independent())
        .split(FreshDraws(600, seed=7))
        .scenarios(stated)
    )
    rows = both.write_parquet(tmp_path / "both.parquet").table.to_pylist()
    designed = {(row["scope"], row["level"]) for row in rows if row["design_id"]}
    assert designed == {("joint", "total"), ("individual", "asset")}
    assert {row["design_scenario"] for row in rows} == {"2030s", "2050s", None}


def test_merge_combines_the_assets_of_one_set(curves: Path) -> None:
    def table(asset_id: str, seed: int) -> Any:
        evaluated = {
            label: _drawn(curves, horizons, label, 300, seed)[asset_id]
            for label, horizons in (("2030s", EARLY), ("2050s", MIDDLE))
        }
        designed = _drawn(curves, EARLY, "2030s", 300, 1)[asset_id]
        return compare_scenarios(
            [_linear(asset_id)],
            ScenarioSet(list(evaluated.values()), reference="2030s"),
            design_sample=designed,
            options=OPTIONS,
        )

    first, second = table("warehouse-a", 2), table("depot-b", 2)
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(first), EvaluationMetadata.from_table(second)
    )
    assert merged.setup.scenarios is not None
    entry = merged.setup.scenarios["scenarios"]["2030s"]
    assert len(entry["blocks"]) == 2
    assert entry["n_rows"] == 600
    assert {place["owner"] for place in entry["index_read"]} == {
        "asset 'warehouse-a'",
        "asset 'depot-b'",
    }
    # One asset's draws from two generators are two sets, not one.
    with pytest.raises(EvaluationError, match="two generator records"):
        EvaluationMetadata.merge(
            EvaluationMetadata.from_table(first),
            EvaluationMetadata.from_table(table("warehouse-a", 3)),
        )


def test_a_window_under_an_empirical_copula_divides_by_the_reference(
    curves: Path,
) -> None:
    years = np.arange(2000, 2006)
    reference = OutcomeSample.join(
        OutcomeSample.from_table(
            pa.table(
                {
                    "asset": [asset_id] * 6,
                    "year": years,
                    "index": np.linspace(40.0, 90.0, 6)[:: 1 if k else -1],
                    "loss": np.zeros(6),
                }
            ),
            period="year",
            asset_id="asset",
            index="index",
            loss="loss",
            index_unit="mm/day",
            loss_unit="EUR",
        )
        for k, asset_id in enumerate(POINTS)
    )
    plan = (
        _curve_plan(curves)
        .monte_carlo(24, seed=42)
        .scopes(Joint())
        .dependence(EmpiricalCopula(reference))
        .split(FreshDraws(36, seed=7))
        .scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    )
    with pytest.raises(ScenarioError, match="its members times the reference's 6"):
        plan.explain()
    plan.monte_carlo(36, seed=42).explain()


# -- allocated rows and breakdowns across scenarios ------------------------------------

VALUES = {"warehouse-a": 2e6, "depot-b": 3e6}
"""The assets' values, as the plan's loss model reads them from `value_eur`."""


def _joint_by_hand(curves: Path, fraction: float = 0.5) -> dict[str, Any]:
    """Each window's sample of both assets, from the plan's generators, held out."""
    rows = _rows(curves)
    metadata = HazardDataset.local(curves).metadata()
    blocks: dict[str, Any] = {}
    for label, horizons in (("2030s", EARLY), ("2050s", MIDDLE)):
        children = np.random.SeedSequence(42).spawn(2)
        joint = sample_curves(
            {asset_id: [rows[(asset_id, h)] for h in horizons] for asset_id in POINTS},
            dependence=Independent(),
            metadata=dict.fromkeys(POINTS, metadata),
            n_draws=600,
            rngs={
                asset_id: np.random.default_rng(child)
                for asset_id, child in zip(POINTS, children, strict=True)
            },
            loss_models={asset_id: _loss(asset_id) for asset_id in POINTS},
            basis_risk=ImpactNoise(0.3),
            scenario=label,
        )
        blocks[label] = holdout(joint, fraction)
    return blocks


def test_allocated_rows_across_scenarios_go_to_a_file_of_their_own(
    curves: Path, tmp_path: Path
) -> None:
    base = (
        _curve_plan(curves)
        .scopes(Joint())
        .dependence(Independent())
        .split(Holdout(0.5))
        .scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    )
    out = tmp_path / "out"
    result = base.allocation("value").write_parquet(out / "joint.parquet")
    plain = base.write_parquet(tmp_path / "plain" / "joint.parquet")
    # The comparison is the plan's without an allocation, row for row, and its
    # evaluations record none.
    assert result.table.to_pylist() == plain.table.to_pylist()
    assert {e["allocation"] for e in result.metadata.setup.evaluations} == {None}
    assert result.allocated_output == out / "joint.allocated.parquet"
    assert sorted(p.name for p in out.iterdir()) == [
        "joint.allocated.parquet",
        "joint.parquet",
    ]
    # The allocated rows: allocate_scenarios on the blocks of the function route.
    blocks = _joint_by_hand(curves)
    for planned in result.designs:
        assert (
            planned.design_sample.content_hash
            == blocks[planned.design_sample.scenario][0].content_hash
        )
    evaluated = ScenarioSet([blocks[label][1] for label in WINDOWS], reference="2030s")
    by_hand = allocate_scenarios(
        list(result.designs),
        evaluated,
        allocation="value",
        values=VALUES,
        options=OPTIONS,
    )
    assert result.allocated is not None
    rows = result.allocated.to_pylist()
    assert rows == by_hand.to_pylist()
    # 2 designs (one per window) x 2 scenarios x 2 assets, all allocated.
    assert len(rows) == 8
    assert {row["level"] for row in rows} == {"asset_by_value"}
    assert {(row["design_scenario"], row["scenario"]) for row in rows} == {
        (designed, evaluated_on) for designed in WINDOWS for evaluated_on in WINDOWS
    }
    # They are compare_scenarios' allocated rows, without the totals beside them.
    beside = compare_scenarios(
        list(result.designs),
        evaluated,
        allocation="value",
        values=VALUES,
        options=OPTIONS,
    )
    assert [r for r in beside.to_pylist() if r["level"] == "asset_by_value"] == rows
    # The second file: the rows, with the set and the plan's records.
    written = pq.read_table(result.allocated_output)
    assert written.equals(result.allocated)
    stored = EvaluationMetadata.from_table(written)
    assert stored.setup.scenarios is not None
    assert stored.setup.scenarios["design_on"] == "each"
    assert stored.setup.scenarios["labels"] == ["2030s", "2050s"]
    assert {e["allocation"] for e in stored.setup.evaluations} == {"value"}
    assert {e["pricing"] is None for e in stored.setup.evaluations} == {True}
    assert "other_scenario" in {e["method"] for e in stored.setup.evaluations}
    (record,) = stored.source_records
    assert record["canonical_file"]["scenarios"] == ["2030s", "2050s"]
    assert stored.assets["values_sha256"] is not None
    text = setup_text(stored)
    assert "Pricing: allocated rows hold no premium" in text
    assert "Scenario 2030s (the reference)" in text


def test_allocated_rows_across_scenarios_on_years_fold_by_fold(
    tmp_path: Path,
) -> None:
    plan = (
        _annual_plan()
        .scopes(Joint())
        .split(KFold(3))
        .allocation("loss")
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    result = plan.write_parquet(tmp_path / "joint.parquet")
    assert result.scenario_samples is not None and result.allocated is not None
    labels = ("historical", "ssp585")
    joined = {
        label: OutcomeSample.join(result.scenario_samples[label].values())
        for label in labels
    }
    folds = {label: year_block_kfold(joined[label], 3) for label in labels}
    expected: list[dict[str, Any]] = []
    for fold in range(3):
        trains = {folds[label][fold][0].content_hash for label in labels}
        designs = [d for d in result.designs if d.design_sample.content_hash in trains]
        expected += allocate_scenarios(
            designs,
            ScenarioSet(
                [folds[label][fold][1] for label in labels], reference="historical"
            ),
            allocation="loss",
            values=VALUES,
            fold=fold,
            options=OPTIONS,
        ).to_pylist()
    assert result.allocated.to_pylist() == expected
    assert {row["level"] for row in expected} == {"asset_by_loss"}
    assert {row["fold"] for row in expected} == {0, 1, 2}


def test_an_allocation_across_scenarios_needs_a_joint_design(tmp_path: Path) -> None:
    plan = (
        _annual_plan()
        .covers(_linear())
        .scopes(Joint())
        .split(Holdout(0.7))
        .allocation("value")
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    with pytest.raises(EvaluationError, match="no entry is a joint design"):
        plan.write_parquet(tmp_path / "joint.parquet")


def test_a_breakdown_across_scenarios_is_payout_breakdown_scenarios(
    tmp_path: Path,
) -> None:
    plan = (
        _annual_plan()
        .split(Holdout(0.7))
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    table = plan.payout_breakdown()
    result = plan.write_parquet(tmp_path / "years.parquet")
    # The breakdown writes nothing.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["years.parquet"]
    assert result.scenario_samples is not None
    labels = ("historical", "ssp585")
    for asset_id in POINTS:
        blocks = {
            label: holdout(result.scenario_samples[label][asset_id], 0.7)
            for label in labels
        }
        designs = [d for d in result.designs if d.cover.asset_id == asset_id]
        by_hand = payout_breakdown_scenarios(
            [*designs, _linear(asset_id)],
            ScenarioSet([blocks[label][1] for label in labels], reference="historical"),
            design_sample=blocks["historical"][0],
            options=OPTIONS,
        )
        assert _plan_rows(table, asset_id=asset_id) == by_hand.to_pylist()
    # Each entry's parts add up to its expected payout on each scenario's block.
    rows = table.to_pylist()
    assert {row["scenario"] for row in rows} == set(labels)
    assert {row["split"] for row in rows} == {"out_of_sample"}
    for row in result.table.to_pylist():
        parts = [
            part["expected_payout"]
            for part in rows
            if all(
                part[key] == row[key]
                for key in ("cover_hash", "design_id", "asset_id", "scenario")
            )
        ]
        assert sum(parts) == pytest.approx(row["expected_payout"])
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.setup.scenarios is not None
    assert metadata.setup.scenarios["design_on"] == "each"
    assert [r["annual_table"]["scenarios"] for r in metadata.source_records] == [
        ["historical"],
        ["ssp585"],
    ]
    assert "Pricing: a payout breakdown holds no premiums" in setup_text(metadata)


def test_a_breakdown_across_windows_of_curves(curves: Path, tmp_path: Path) -> None:
    plan = (
        _curve_plan(curves)
        .split(FreshDraws(600, seed=7))
        .scenarios(ScenarioSet(reference="2030s", labels=WINDOWS))
    )
    table = plan.payout_breakdown()
    result = plan.write_parquet(tmp_path / "scenarios.parquet")
    assert result.scenario_samples is not None
    assert result.scenario_fresh_samples is not None
    for asset_id in POINTS:
        designs = [d for d in result.designs if d.cover.asset_id == asset_id]
        by_hand = payout_breakdown_scenarios(
            [*designs, _linear(asset_id)],
            ScenarioSet(
                [result.scenario_fresh_samples[label][asset_id] for label in WINDOWS],
                reference="2030s",
            ),
            design_sample=result.scenario_samples["2030s"][asset_id],
            options=OPTIONS,
        )
        assert _plan_rows(table, asset_id=asset_id) == by_hand.to_pylist()


def test_an_asset_whose_kept_years_differ_is_named_at_the_merge(
    tmp_path: Path,
) -> None:
    future = _years(2041, location=58.0, seed=2)
    # depot-b's series has no value in 2041, so it spans 2042 to 2070.
    future = future.filter(
        pc.invert(
            pc.and_(pc.equal(future["year"], 2041), pc.equal(future["latitude"], 51.0))
        )
    )
    entry = _future()
    shorter = AnnualScenario(
        "ssp585",
        future,
        provenance=entry.provenance,
        pathway="ssp585",
        hazard_name="rx1day",
    )
    plan = (
        _annual_plan(missing="drop_year")
        .split(Holdout(0.7))
        .scenarios(ScenarioSet([shorter], reference="historical"))
    )
    with pytest.raises(EvaluationError) as raised:
        plan.write_parquet(tmp_path / "years.parquet")
    message = str(raised.value)
    assert "scenario 'ssp585' has the horizon 2055 in one table" in message
    assert "asset 'warehouse-a', whose series spans 2041 to 2070" in message
    assert "2056 in another (asset 'depot-b', whose series spans 2042 to 2070)" in (
        message
    )
    assert "state years= that every asset's series holds" in message
    # Years every asset's series holds read one horizon, and the plan runs.
    plan.scenarios(
        ScenarioSet(
            [
                AnnualScenario(
                    "ssp585",
                    future,
                    provenance=entry.provenance,
                    pathway="ssp585",
                    hazard_name="rx1day",
                    years=list(range(2042, 2071)),
                )
            ],
            reference="historical",
        )
    ).write_parquet(tmp_path / "years.parquet")


def _without(table: Any, year: int, latitude: float) -> Any:
    """`table` without one year of one asset's cell."""
    return table.filter(
        pc.invert(
            pc.and_(
                pc.equal(table["year"], year), pc.equal(table["latitude"], latitude)
            )
        )
    )


def _shorter_future(table: Any) -> AnnualScenario:
    entry = _future()
    return AnnualScenario(
        "ssp585",
        table,
        provenance=entry.provenance,
        pathway="ssp585",
        hazard_name="rx1day",
    )


@pytest.mark.parametrize("rule", [Holdout(0.7), KFold(3)], ids=["holdout", "kfold"])
def test_an_asset_whose_kept_years_differ_is_refused_before_any_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule: Any
) -> None:
    # depot-b's series has no value in 2041, so it spans 2042 to 2070.
    future = _without(_years(2041, location=58.0, seed=2), 2041, 51.0)
    plan = (
        _annual_plan(missing="drop_year")
        .split(rule)
        .scenarios(ScenarioSet([_shorter_future(future)], reference="historical"))
    )
    # explain() reads nothing: under drop_year a year may drop out at either end,
    # so it states no horizon, and it can't see the years each asset keeps.
    document: Any = plan.explain(format="json")
    assert document["setup"]["scenarios"]["scenarios"]["ssp585"]["horizon"] is None
    # The merge refused the plan's tables after the designs; it is the same error.
    with monkeypatch.context() as patched:
        patched.setattr(plan_module, "_check_scenario_horizons", lambda built: None)
        with pytest.raises(EvaluationError) as merged:
            plan.write_parquet(tmp_path / "merged.parquet")
    message = str(merged.value)
    assert "scenario 'ssp585' has the horizon 2055 in one table" in message
    assert "2056 in another (asset 'depot-b', whose series spans 2042 to 2070)" in (
        message
    )

    def designed(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the refusal comes before any design")

    monkeypatch.setattr(plan_module, "design", designed)
    out = tmp_path / "years.parquet"
    for run in (
        plan.samples,
        plan.scenario_samples,
        plan.designs,
        plan.payout_breakdown,
        lambda: plan.write_parquet(out),
    ):
        with pytest.raises(EvaluationError) as raised:
            run()
        assert str(raised.value) == message
    assert not out.exists()


def test_an_asset_whose_kept_years_keep_the_horizon_is_not_refused(
    tmp_path: Path,
) -> None:
    # A year dropped inside the window leaves its centre, so both assets read
    # one horizon and the tables merge.
    future = _without(_years(2041, location=58.0, seed=2), 2050, 51.0)
    result = (
        _annual_plan(missing="drop_year")
        .split(Holdout(0.7))
        .scenarios(ScenarioSet([_shorter_future(future)], reference="historical"))
        .write_parquet(tmp_path / "years.parquet")
    )
    stated = result.metadata.setup.scenarios
    assert stated is not None
    assert stated["scenarios"]["ssp585"]["horizon"] == 2055


def test_explain_states_a_per_year_horizon_where_years_states_it(
    tmp_path: Path,
) -> None:
    stated = (
        _annual_plan(years=list(range(1985, 2015)))
        .split(Holdout(0.7))
        .scenarios(
            ScenarioSet(
                [_future(years=list(range(2041, 2071)))], reference="historical"
            )
        )
    )
    document: Any = stated.explain(format="json")
    entries = document["setup"]["scenarios"]["scenarios"]
    assert {label: entry["horizon"] for label, entry in entries.items()} == {
        "historical": 1999,
        "ssp585": 2055,
    }
    assert "horizon 1999" in setup_text(document)
    # The horizons explain() states are those the plan writes.
    written = stated.write_parquet(tmp_path / "years.parquet").metadata.setup.scenarios
    assert written is not None
    assert {
        label: entry["horizon"] for label, entry in written["scenarios"].items()
    } == {"historical": 1999, "ssp585": 2055}
    # A scenario's own years= wins over the plan's, and without years= the window
    # is known only at execution, as under missing="drop_year".
    own = (
        _annual_plan()
        .split(Holdout(0.7))
        .scenarios(
            ScenarioSet(
                [_future(years=list(range(2045, 2061)))], reference="historical"
            )
        )
    )
    entries = own.explain(format="json")["setup"]["scenarios"]["scenarios"]
    assert {label: entry["horizon"] for label, entry in entries.items()} == {
        "historical": None,
        "ssp585": 2052,
    }
    dropped = (
        _annual_plan(years=list(range(1985, 2015)), missing="drop_year")
        .split(Holdout(0.7))
        .scenarios(ScenarioSet([_future()], reference="historical"))
    )
    entries = dropped.explain(format="json")["setup"]["scenarios"]["scenarios"]
    assert {entry["horizon"] for entry in entries.values()} == {None}
