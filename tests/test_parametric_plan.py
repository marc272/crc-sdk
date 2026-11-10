"""ParametricPlan: builders, explain(), seeding, splits and the written output."""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import h3  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact, TransformContext
from crc_framework import impacts as framework_impacts

from crc_sdk.geometry import point_to_cell
from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AtStation,
    Budget,
    ComonotoneIndexWarning,
    CostOfCapitalLoad,
    Cover,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    H3Cell,
    HazardNoise,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    KFold,
    Limit,
    LinearPayout,
    LossModel,
    MetricOptions,
    NearestCell,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    PurePremium,
    SampleError,
    StationColumn,
    StepPayout,
    WeightedCVaR,
    compare,
    design,
    evaluate,
    holdout,
    sample_annual,
    setup_text,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY
from crc_sdk.parametric.sources import _great_circle_km
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows import AssetPortfolio, CellColumn, ImpactContextColumns
from crc_sdk.workflows.noaa_water_levels import NOAAExtremeWaterLevelPlan

UNTYPED_PLAN: Any = ParametricPlan
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
PROVENANCE = SourceProvenance(
    provider="example",
    dataset="synthetic rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic annual maxima",
)
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
CELLS = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
YEARS = np.arange(1981, 2021)
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))


def _annual(seed: int = 1, years: Any = YEARS) -> Any:
    """A per-cell annual-extremes table in the layout of ERA5's and CMIP6's."""
    rng = np.random.default_rng(seed)
    n = len(years) * len(CELLS)
    return pa.table(
        {
            "year": pa.array(np.repeat(years, len(CELLS)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in CELLS], len(years)),
            "longitude": np.tile([lon for _, lon in CELLS], len(years)),
            "value": rng.gumbel(50.0, 12.0, n),
        }
    )


def _assets(*ids: str) -> Any:
    ids = ids or ("warehouse-a", "depot-b")
    points = {"warehouse-a": (6.96, 50.94), "depot-b": (7.10, 51.10)}
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [points.get(i, (6.9, 50.9))[0] for i in ids],
            "latitude": [points.get(i, (6.9, 50.9))[1] for i in ids],
            "value_eur": [2e6 + 1e5 * k for k in range(len(ids))],
        }
    )


def _plan(source: Any = None, assets: Any = None, **options: Any) -> ParametricPlan:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "provenance": PROVENANCE,
        "value_column": "value_eur",
        "value_unit": "EUR",
        "seed": 42,
    }
    arguments.update(options)
    plan: ParametricPlan = UNTYPED_PLAN.for_annual_extremes(
        _annual() if source is None else source,
        _assets() if assets is None else assets,
        **arguments,
    )
    return plan


def _step(label: str | None = "step", asset_id: str | None = None) -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        asset_id=asset_id,
        label=label,
    )


def _label(cover: Cover) -> str:
    return cover.definition_hash.removeprefix("sha256:")[:12]


def _linear(asset_id: str | None = None) -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        asset_id=asset_id,
        label="linear",
    )


class _SourcePlan:
    """A stand-in for a crc-sdk plan with an annual_extremes() execution point."""

    def __init__(self, table: Any, unit: str = "mm/day") -> None:
        self.table = table
        self.unit = unit
        self.calls: list[str] = []

    def annual_extremes(self) -> Iterator[Any]:
        self.calls.append("annual_extremes")
        return iter(self.table.to_batches(max_chunksize=50))

    def explain(self, *, format: str = "text") -> Any:
        self.calls.append("explain")
        return {"recipe": "rx1day", "unit": self.unit, "years": [1981, 2020]}


SERIES = (
    "the hazard series from provider example and dataset synthetic rx1day, with "
    "horizon and pathway not recorded"
)


def _lines(text: str) -> dict[str, list[str]]:
    """The lines of a setup text by their label (`Split`, `Source`, ...)."""
    lines: dict[str, list[str]] = {}
    for line in text.splitlines():
        lines.setdefault(line.split(":")[0], []).append(line)
    return lines


@pytest.fixture(autouse=True)
def _quiet_comonotone() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        yield


# -- builders and explain ------------------------------------------------------


def test_builders_return_new_plans_and_explain_does_no_io(tmp_path: Path) -> None:
    source = _SourcePlan(_annual())
    base = _plan(source)
    plan = (
        base.basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    assert base.basis_risk_model is None and base.cover_list == ()
    assert plan.split_rule == Holdout(0.7)
    plan.explain()
    plan.explain(format="json")
    repr(plan)
    assert source.calls == []
    # Files are not opened by builders or explain() either.
    missing = tmp_path / "absent.parquet"
    on_files = _plan(missing, tmp_path / "assets.parquet").basis_risk(NoBasisRisk())
    document: Any = on_files.covers(_linear()).explain(format="json")
    assert document["sources"][0]["kind"] == "file"
    assert document["sources"][0]["path"] == str(missing)
    assert _lines(setup_text(document))["Source"] == [
        f"Source: {SERIES}, read from a file {missing}."
    ]
    assert document["assets"]["count"] is None
    assert not missing.exists()


def test_explain_mirrors_the_written_metadata() -> None:
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(MSE(), WeightedCVaR(alpha=0.9, beta=0.9), budget=Budget(ratio=1.0))
        .split(KFold(4))
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    assert set(document) == {
        "assets",
        "covers",
        "designs",
        "entry",
        "execution",
        "licences",
        "producer",
        "rng",
        "setup",
        "sources",
    }
    setup = document["setup"]
    assert setup["basis"] == "annual"
    assert setup["metric_options"] == OPTIONS.to_document()
    assert [sample["asset_ids"] for sample in setup["samples"]] == [
        ["warehouse-a"],
        ["depot-b"],
    ]
    sample = setup["samples"][0]
    assert sample["basis_risk"] == {"kind": "impact_noise", "sigma": 0.3, "upper": 1.0}
    assert sample["loss_model"]["value"] == {
        "asset_value_hash": None,
        "column": "value_eur",
        "source": "column",
        "unit": "EUR",
    }
    assert [(e["split"], e["fold"], e["method"]) for e in setup["evaluations"]] == [
        (split, fold, method)
        for fold in range(4)
        for split, method in (("in_sample", "in_sample"), ("out_of_sample", "kfold"))
    ]
    assert setup["evaluations"][1]["details"] == {"k": 4, "plan_seed": 42}
    assert setup["evaluations"][0]["details"] == {"plan_seed": 42}
    assert document["designs"]["objectives"] == [
        {"kind": "mse"},
        {"alpha": 0.9, "beta": 0.9, "kind": "weighted_cvar"},
    ]
    assert document["designs"]["budget"] == {"ratio": 1.0}
    assert document["designs"]["templates"] == [_step().definition_hash]
    assert [cover["free_parameters"] for cover in document["covers"]] == [
        ["levels[0]", "levels[1]", "levels[2]"],
        [],
    ]
    assert document["licences"] == [
        {
            "attribution": "Synthetic annual maxima",
            "licence": "CC0-1.0",
            "role": "hazard",
            "source": "synthetic rx1day",
        }
    ]
    assert document["rng"]["seed"] == 42
    assert document["sources"][0]["provenance"] == PROVENANCE.model_dump(mode="json")
    assert document["sources"][0]["rows"] == len(YEARS) * len(CELLS)
    assert json.loads(str(plan.explain())) == document
    with pytest.raises(ValueError, match="text"):
        UNTYPED_PLAN.explain(plan, format="yaml")


def test_setup_text_states_the_plan_before_execution() -> None:
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    text = setup_text(document["setup"])
    assert "Level: asset, one row per asset (warehouse-a and depot-b)" in text
    assert "Basis: annual, one draw is one year." in text
    assert "Split: out of sample, fold 0, a holdout with train fraction 0.7" in text
    assert "not known yet" in text
    assert "impact noise, a mean-one lognormal factor with sigma 0.3" in text
    assert "times the asset value in column value_eur, in EUR" in text
    unset: Any = _plan().explain(format="json")
    assert "Basis risk: not stated." in setup_text(unset["setup"])
    # The whole document names the source too: the provider and dataset the plan
    # states, and the table or source plan the values are read from.
    rows = len(YEARS) * len(CELLS)
    source = f"Source: {SERIES}, read from a table of {rows} rows."
    lines = text.splitlines()
    after = max(i for i, line in enumerate(lines) if line.startswith("Sample"))
    design = (
        "Design step, planned: objective: MSE, E[(L - payout)^2]; pricing: the pure "
        "premium, P = E[payout]; budget: premium at most 1 x E[L] of the design "
        "sample; no limit; design premium, route and non-uniqueness report: not "
        "known yet."
    )
    assert setup_text(document).splitlines() == [
        *lines[: after + 1],
        source,
        *lines[after + 1 : -1],
        design,
        lines[-1],
    ]
    planned: Any = _plan(_SourcePlan(_annual())).explain(format="json")
    assert _lines(setup_text(planned))["Source"] == [
        f"Source: {SERIES}, read from a source plan _SourcePlan."
    ]


def test_the_docstring_of_explain_names_the_whole_document() -> None:
    stated = " ".join((ParametricPlan.explain.__doc__ or "").split())
    assert 'setup_text(plan.explain(format="json"))' in stated


def test_setup_text_states_one_design_per_planned_objective() -> None:
    plan = (
        _plan()
        .basis_risk(NoBasisRisk())
        .covers(
            _step(),
            Cover(
                INDEX,
                StepPayout([70.0, 90.0], [Free(), Free()]),
                payout_unit="EUR",
                label="other",
            ),
            _linear(),
        )
        .objective(
            MSE(),
            WeightedCVaR(alpha=0.9, beta=0.95),
            budget=Budget(amount=2e5),
            limit=Limit(1e6),
        )
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    lines = _lines(setup_text(document))["Design step and other, planned"]
    terms = (
        "pricing: the pure premium, P = E[payout]; budget: premium at most 200,000 "
        "EUR; limit: 1,000,000 EUR; design premium, route and non-uniqueness report: "
        "not known yet."
    )
    assert lines == [
        "Design step and other, planned: objective: MSE, E[(L - payout)^2]; " + terms,
        "Design step and other, planned: objective: weighted CVaR, alpha CVaR_beta(U) "
        "+ (1 - alpha) E[U] with alpha 0.9 and beta 0.95; " + terms,
    ]
    # A setup alone holds no design terms.
    alone = _lines(setup_text(document["setup"]))
    assert not any(label.startswith("Design") for label in alone)


def test_setup_text_states_no_design_without_templates() -> None:
    plan = (
        _plan()
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .objective(MSE())
        .metrics(OPTIONS)
    )
    document: Any = plan.explain(format="json")
    assert document["designs"]["templates"] == []
    assert not any(k.startswith("Design") for k in _lines(setup_text(document)))


def test_setup_text_states_how_a_point_found_each_asset(tmp_path: Path) -> None:
    result = (
        _plan()
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    # Every block and evaluation of an asset reads one cell: one line says so.
    assert _lines(setup_text(result.metadata))["Location"] == [
        "Location: hazard series, located by a point, in the nearest grid cell (grid "
        "step 0.25 by 0.25 degrees): warehouse-a at (50.875, 6.875) and depot-b at "
        "(51.125, 7.125), the cell centres as (latitude, longitude)."
    ]
    # A selection of rows states its own assets' cells only.
    rows = result.table.filter(pc.equal(result.table.column("asset_id"), "depot-b"))
    assert _lines(setup_text(rows))["Location"] == [
        "Location: hazard series, located by a point, in the nearest grid cell (grid "
        "step 0.25 by 0.25 degrees): depot-b at (51.125, 7.125), the cell centres as "
        "(latitude, longitude)."
    ]
    # Neither the asset's own point nor a distance is stated.
    line = _lines(setup_text(result.metadata))["Location"][0]
    assert "50.94" not in line and "6.96" not in line and "km" not in line
    # Before execution only the rule is known, and it is in `sources`.
    planned: Any = _plan().basis_risk(NoBasisRisk()).explain(format="json")
    assert "Location" not in _lines(setup_text(planned))


def test_setup_text_states_an_asset_located_by_its_h3_cell(tmp_path: Path) -> None:
    cell = point_to_cell(6.96, 50.94, 7)
    result = (
        _plan(assets=_cell_assets(cell))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    assert _lines(setup_text(result.metadata))["Location"] == [
        "Location: hazard series, located by the asset's H3 cell, in the grid cell "
        "that holds it (grid step 0.25 by 0.25 degrees): warehouse-a at (50.875, "
        "6.875), the cell centres as (latitude, longitude)."
    ]


def test_setup_text_states_no_location_for_a_station(tmp_path: Path) -> None:
    stations = pa.table(
        {
            "year": pa.array(YEARS, pa.int32()),
            "station": ["S1"] * len(YEARS),
            "value": np.random.default_rng(5).gumbel(60.0, 12.0, len(YEARS)),
        }
    )
    assets = _assets("warehouse-a").append_column("station_id", pa.array(["S1"]))
    result = (
        _plan(stations, assets, location=StationColumn("station_id"))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    assert "Location" not in _lines(setup_text(result.metadata))


def test_the_callers_provenance_names_a_plans_source_not_the_source_plan() -> None:
    class CMIP6Shaped(_SourcePlan):
        def explain(self, *, format: str = "text") -> Any:
            return {
                **super().explain(format=format),
                "archive": "aws",
                "catalogue": "https://example.org/pangeo-cmip6.csv",
                "pathway": "ssp245",
                "model": "M",
                "member": "r1i1p1f1",
            }

    sample = (
        _plan(CMIP6Shaped(_annual()), _assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .samples()["warehouse-a"]
    )
    source = sample.provenance.sources[0]["series"]["source"]
    assert (source["provider"], source["dataset"]) == ("example", "synthetic rx1day")
    assert "pathway" not in source


def test_explain_without_a_split_lists_in_sample_rows_only() -> None:
    document: Any = _plan().basis_risk(NoBasisRisk()).explain(format="json")
    assert [(e["split"], e["fold"]) for e in document["setup"]["evaluations"]] == [
        ("in_sample", 0)
    ]
    assert document["designs"] is None
    assert document["setup"]["evaluations"][0]["pricing"] == {"kind": "pure"}


def test_the_entry_point_checks_its_arguments() -> None:
    with pytest.raises(TypeError, match="annual_extremes"):
        _plan(source=[1, 2, 3])
    with pytest.raises(TypeError, match="SourceProvenance"):
        _plan(provenance={"licence": "CC0-1.0"})
    calibration = ImpactCalibration(
        exposure_variable="flood_depth",
        exposure_unit="m",
        impact_metric="damage_ratio",
        impact_unit="1",
    )
    depth = ImpactExpression.tabulated(
        [(0.0, 0.0), (2.0, 1.0)], calibration=calibration
    )
    with pytest.raises(SampleError, match="calibrated"):
        _plan(impact=depth)
    with pytest.raises(SampleError, match="no columns"):
        _plan(value_column="replacement_value")
    with pytest.raises(SampleError, match="no columns"):
        _plan(location=StationColumn("station_id"))
    with pytest.raises(TypeError, match="NearestCell"):
        _plan(location="nearest")
    with pytest.raises(ValueError, match="years"):
        _plan(years=[2000, 2000])
    with pytest.raises(ValueError, match="missing"):
        _plan(missing="zero")
    with pytest.raises(ValueError, match="seed"):
        _plan(seed=-1)
    with pytest.raises(ValueError, match="spacing"):
        NearestCell(spacing=(0.25, 0.0))
    with pytest.raises(ValueError, match="column"):
        StationColumn("")


def _cell_assets(*cells: Any, ids: Sequence[str] = ("warehouse-a",)) -> Any:
    """Assets located by their H3 cells (`CellColumn`), with values."""
    return AssetPortfolio(
        pa.table(
            {
                "asset_id": list(ids),
                "cell_index": pa.array(list(cells), pa.uint64()),
                "value_eur": [2e6 + 1e5 * k for k in range(len(ids))],
            }
        ),
        location=CellColumn(),
    )


def test_nearest_cell_locates_assets_by_their_h3_cells() -> None:
    # The H3 cell of warehouse-a's point lies inside the grid cell
    # (50.875, 6.875), the cell its point is nearest.
    cell = point_to_cell(6.96, 50.94, 7)
    by_cell = _plan(assets=_cell_assets(cell)).basis_risk(ImpactNoise(0.3, upper=1.0))
    by_point = _plan(assets=_assets("warehouse-a")).basis_risk(
        ImpactNoise(0.3, upper=1.0)
    )
    sample = by_cell.samples()["warehouse-a"]
    assert sample.content_hash == by_point.samples()["warehouse-a"].content_hash
    series = sample.provenance.sources[0]["series"]
    assert series["asset_location"] == "h3_cell"
    assert series["cell"] == {"latitude": 50.875, "longitude": 6.875}
    # The function route at the H3 cell's centre, with the plan's generator.
    latitude, longitude = h3.cell_to_latlng(h3.int_to_str(cell))
    by_hand = AnnualSeries.nearest_cell(
        _annual(),
        longitude=longitude,
        latitude=latitude,
        unit="mm/day",
        source=PROVENANCE.model_dump(mode="json"),
    )
    assert series == {**by_hand.provenance, "asset_location": "h3_cell"}
    expected = sample_annual(
        by_hand,
        loss_model=LossModel(
            RAMP, value_column="value_eur", value_unit="EUR"
        ).for_asset(_cell_assets(cell), "warehouse-a"),
        basis_risk=ImpactNoise(0.3, upper=1.0),
        asset_id="warehouse-a",
        rng=np.random.default_rng(np.random.SeedSequence(42).spawn(1)[0]),
    )
    assert sample.content_hash == expected.content_hash
    # Builders and explain() read no geometry.
    document: Any = by_cell.covers(_linear()).explain(format="json")
    assert document["assets"]["location"] == "CellColumn(name='cell_index')"


def test_an_h3_cell_that_straddles_grid_cells_raises() -> None:
    # A point on the boundary between two grid cells (longitude 7.0): its H3
    # cell has vertices on both sides, so it does not say which series is
    # the asset's.
    straddling = point_to_cell(7.0, 50.9, 7)
    vertices = h3.cell_to_boundary(h3.int_to_str(straddling))
    assert min(lon for _, lon in vertices) < 7.0 < max(lon for _, lon in vertices)
    plan = _plan(assets=_cell_assets(straddling)).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="straddles cells of the grid"):
        plan.samples()
    # A coarser grid holds the whole H3 cell: the 0.5-degree step puts the
    # boundary at 6.75 and 7.25, so the cell lies in one grid cell.
    coarse = pa.table(
        {
            "year": pa.array(YEARS, pa.int32()),
            "latitude": [51.0] * len(YEARS),
            "longitude": [7.0] * len(YEARS),
            "value": np.linspace(30.0, 150.0, len(YEARS)),
        }
    )
    on_coarse = _plan(
        coarse, _cell_assets(straddling), location=NearestCell(spacing=(0.5, 0.5))
    ).basis_risk(NoBasisRisk())
    sample = on_coarse.samples()["warehouse-a"]
    assert sample.provenance.sources[0]["series"]["cell"] == {
        "latitude": 51.0,
        "longitude": 7.0,
    }


def test_h3_cell_assets_that_cannot_be_located_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    far = point_to_cell(12.0, 40.0, 7)
    with pytest.raises(SampleError, match="half a grid step"):
        _plan(assets=_cell_assets(far)).basis_risk(NoBasisRisk()).samples()
    with pytest.raises(SampleError, match="not a valid H3 cell"):
        _plan(assets=_cell_assets(12345)).basis_risk(NoBasisRisk()).samples()
    with pytest.raises(SampleError, match="has no H3 cell"):
        _plan(assets=_cell_assets(None)).basis_risk(NoBasisRisk()).samples()
    # Without the geometry extra, the error names it.
    cell = point_to_cell(6.96, 50.94, 7)
    monkeypatch.setitem(sys.modules, "h3", None)
    with pytest.raises(ImportError, match=r"crc-sdk\[geometry\]"):
        _plan(assets=_cell_assets(cell)).basis_risk(NoBasisRisk()).samples()


def test_a_registry_impact_that_needs_a_cell_is_refused_without_one() -> None:
    rx1day = framework_impacts.for_factor("rx1day")
    with pytest.raises(SampleError, match="H3 cell"):
        _plan(impact=rx1day, value_column=None, value_unit="1")
    cell = point_to_cell(6.96, 50.94, 7)
    with_cell = framework_impacts.for_factor(
        "rx1day", context=TransformContext(cell=cell)
    )
    plan = _plan(impact=with_cell, value_column=None, value_unit="1")
    samples = plan.basis_risk(NoBasisRisk()).samples()
    assert all(float(sample.loss.max()) > 0.0 for sample in samples.values())
    fixed = _plan(
        impact=rx1day,
        value_column=None,
        value_unit="1",
        context=TransformContext(cell=cell),
    )
    assert fixed.loss_model.context == TransformContext(cell=cell)
    # Assets with an H3 cell column fill the cell when context columns are given.
    stations = pa.table(
        {
            "station": ["S1"] * 30,
            "year": pa.array(range(1991, 2021), pa.int32()),
            "value": np.linspace(30.0, 150.0, 30),
        }
    )
    located = AssetPortfolio(
        pa.table(
            {
                "asset_id": ["a"],
                "cell_index": pa.array([cell], pa.uint64()),
                "station_id": ["S1"],
                "country": ["DEU"],
            }
        ),
        location=CellColumn(),
    )
    by_cell = _plan(
        stations,
        located,
        impact=rx1day,
        value_column=None,
        value_unit="1",
        context=ImpactContextColumns(country="country"),
        location=StationColumn("station_id"),
    )
    sample = by_cell.basis_risk(NoBasisRisk()).samples()["a"]
    expected = with_cell.evaluate(np.linspace(30.0, 150.0, 30))
    np.testing.assert_allclose(sample.loss[:, 0], expected)
    # So do assets located on a grid by their H3 cells (NearestCell).
    on_grid = _plan(
        assets=AssetPortfolio(
            pa.table(
                {
                    "asset_id": ["a"],
                    "cell_index": pa.array([cell], pa.uint64()),
                    "country": ["DEU"],
                }
            ),
            location=CellColumn(),
        ),
        impact=rx1day,
        value_column=None,
        value_unit="1",
        context=ImpactContextColumns(country="country"),
    )
    gridded = on_grid.basis_risk(NoBasisRisk()).samples()["a"]
    np.testing.assert_allclose(
        gridded.loss[:, 0], with_cell.evaluate(gridded.index[:, 0])
    )


RX1DAY = framework_impacts.for_factor("rx1day")
A_CELL = point_to_cell(6.96, 50.94, 7)


def _with_cells(assets: Any, *cells: Any, name: str = "h3") -> Any:
    return assets.append_column(name, pa.array(list(cells), pa.uint64()))


def _no_noise(plan: ParametricPlan) -> Any:
    return plan.basis_risk(NoBasisRisk()).samples()


def _fixed_context(sample: OutcomeSample) -> Any:
    return sample.provenance.to_document()["loss_model"]["context"]["fixed"]


def test_cell_gives_each_point_asset_its_own_cell() -> None:
    # A point on another continent, where rx1day has other coefficients: the
    # same hazard values must give another loss through its own cell.
    points = {"cologne": (6.96, 50.94), "accra": (-0.19, 5.60)}
    cells = {name: point_to_cell(*point, 7) for name, point in points.items()}
    values = np.linspace(30.0, 150.0, 30)
    stations = pa.table(
        {
            "station": ["S1"] * 30 + ["S2"] * 30,
            "year": pa.array([*range(1991, 2021)] * 2, pa.int32()),
            "value": np.concatenate([values, values]),
        }
    )
    assets = pa.table(
        {
            "asset_id": list(points),
            "longitude": [point[0] for point in points.values()],
            "latitude": [point[1] for point in points.values()],
            "station_id": ["S1", "S2"],
        }
    )
    arguments: dict[str, Any] = {
        "impact": RX1DAY,
        "value_column": None,
        "value_unit": "1",
        "location": StationColumn("station_id"),
    }
    computed = _no_noise(
        _plan(stations, assets, **arguments, cell=H3Cell(resolution=7))
    )
    read = _no_noise(
        _plan(stations, _with_cells(assets, *cells.values()), **arguments, cell="h3")
    )
    losses = {}
    for name, cell in cells.items():
        by_hand = framework_impacts.for_factor(
            "rx1day", context=TransformContext(cell=cell)
        )
        for sample in (computed[name], read[name]):
            np.testing.assert_allclose(sample.loss[:, 0], by_hand.evaluate(values))
            assert _fixed_context(sample)["cell"] == format(cell, "x")
        assert computed[name].content_hash == read[name].content_hash
        losses[name] = computed[name].loss[:, 0]
    assert not np.allclose(losses["cologne"], losses["accra"])


def test_cell_works_for_assets_on_a_grid_and_in_a_file(tmp_path: Path) -> None:
    # The hazard is still read at the point; only the impact gets the cell.
    by_cell = _no_noise(
        _plan(assets=_assets("warehouse-a"), impact=RX1DAY, cell=H3Cell(7))
    )["warehouse-a"]
    by_point = _no_noise(_plan(assets=_assets("warehouse-a")))["warehouse-a"]
    np.testing.assert_array_equal(by_cell.index, by_point.index)
    assert by_cell.provenance.sources[0]["series"]["cell"] == {
        "latitude": 50.875,
        "longitude": 6.875,
    }
    # Assets in a file are read at execution, as ever.
    file = tmp_path / "assets.parquet"
    plan = _plan(assets=file, impact=RX1DAY, cell=H3Cell(7)).basis_risk(NoBasisRisk())
    document: Any = plan.explain(format="json")
    assert document["assets"]["cell"] == {"kind": "h3_cell", "resolution": 7}
    assert not file.exists()
    pq.write_table(_assets("warehouse-a"), file)
    assert plan.samples()["warehouse-a"].content_hash == by_cell.content_hash


def test_the_cell_is_recorded_and_the_point_is_not(tmp_path: Path) -> None:
    longitude, latitude = 6.9613, 50.9427
    cell = format(point_to_cell(longitude, latitude, 7), "x")
    assets = pa.table(
        {
            "asset_id": ["a"],
            "longitude": [longitude],
            "latitude": [latitude],
            "value_eur": [2e6],
        }
    )
    plan = (
        _plan(assets=assets, impact=RX1DAY, cell=H3Cell(resolution=7))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    planned: Any = plan.explain(format="json")
    assert planned["assets"]["cell"] == {"kind": "h3_cell", "resolution": 7}
    origin = "the cell at resolution 7 of the asset's point."
    assert f"The impact reads each asset's H3 cell: {origin}" in setup_text(planned)

    result = plan.write_parquet(tmp_path / "cell.parquet")
    sample = result.samples["a"]
    assert _fixed_context(sample)["cell"] == cell
    note = f"The impact reads H3 cell {cell} for asset a: {origin}"
    assert note in sample.provenance.notes
    schema = pq.read_schema(result.output).metadata
    records = {
        "provenance": json.dumps(sample.provenance.to_document()),
        "metadata": result.metadata.to_json(),
        "file": schema[EVALUATION_METADATA_KEY.encode("utf-8")].decode("utf-8"),
        "setup text": setup_text(result.metadata),
        "setup text of the rows": setup_text(result.table),
    }
    for name, record in records.items():
        assert cell in record, name
    records["explain"] = json.dumps(planned)
    records["setup text of the plan"] = setup_text(planned)
    for name, record in records.items():
        assert str(longitude) not in record, name
        assert str(latitude) not in record, name


def _assert_no_distance_or_point(
    records: dict[str, str],
    matched: Any,
    longitude: float,
    latitude: float,
    *others: Any,
) -> None:
    """No record states the distance to the matched cell, or where the asset is."""
    distance = float(
        _great_circle_km(
            latitude,
            longitude,
            np.array([matched["latitude"]]),
            np.array([matched["longitude"]]),
        )[0]
    )
    assert distance > 0.0
    for name in ("provenance", "metadata"):
        assert '"cell"' in records[name], name  # these records carry the series
    for name, record in records.items():
        assert "distance_km" not in record, name
        for value in (distance, longitude, latitude, *others):
            assert str(value) not in record, name


def test_no_nearest_cell_record_of_the_function_route_locates_the_asset() -> None:
    longitude, latitude = 6.9613, 50.9427
    series = AnnualSeries.nearest_cell(
        _annual(),
        longitude=longitude,
        latitude=latitude,
        unit="mm/day",
        source=PROVENANCE.model_dump(mode="json"),
    )
    sample = sample_annual(
        series,
        loss_model=LossModel(RAMP, value=2e6, value_unit="EUR"),
        basis_risk=ImpactNoise(0.3, upper=1.0),
        asset_id="a",
        rng=np.random.default_rng(0),
    )
    table = evaluate(_linear(), sample, split="in_sample")
    records = {
        "series": str(series.provenance),
        "provenance": json.dumps(sample.provenance.to_document()),
        "metadata": EvaluationMetadata.from_table(table).to_json(),
        "setup text": setup_text(table),
    }
    _assert_no_distance_or_point(
        records, series.provenance["cell"], longitude, latitude
    )


@pytest.mark.parametrize("located_by", ["point", "h3_cell"])
def test_no_nearest_cell_record_of_the_plan_route_locates_the_asset(
    located_by: str, tmp_path: Path
) -> None:
    longitude, latitude = 6.9613, 50.9427
    others: tuple[Any, ...] = ()
    assets: Any = pa.table(
        {
            "asset_id": ["a"],
            "longitude": [longitude],
            "latitude": [latitude],
            "value_eur": [2e6],
        }
    )
    if located_by == "h3_cell":
        cell = point_to_cell(longitude, latitude, 7)
        assets = _cell_assets(cell, ids=("a",))
        # The point of an asset located by its H3 cell is that cell's centre.
        latitude, longitude = h3.cell_to_latlng(h3.int_to_str(cell))
        others = (cell, format(cell, "x"))
    plan = (
        _plan(assets=assets)
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_linear())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    planned: Any = plan.explain(format="json")
    result = plan.write_parquet(tmp_path / "nearest.parquet")
    sample = result.samples["a"]
    schema = pq.read_schema(result.output).metadata
    records = {
        "provenance": json.dumps(sample.provenance.to_document()),
        "metadata": result.metadata.to_json(),
        "file": schema[EVALUATION_METADATA_KEY.encode("utf-8")].decode("utf-8"),
        "setup text": setup_text(result.metadata),
        "setup text of the rows": setup_text(result.table),
        "explain": json.dumps(planned),
        "setup text of the plan": setup_text(planned),
    }
    series = sample.provenance.sources[0]["series"]
    assert ("asset_location" in series) == (located_by == "h3_cell")
    _assert_no_distance_or_point(records, series["cell"], longitude, latitude, *others)


def test_a_cell_column_is_named_in_the_setup_text() -> None:
    assets = _with_cells(_assets("warehouse-a"), A_CELL)
    plan = _plan(assets=assets, impact=RX1DAY, cell="h3").basis_risk(NoBasisRisk())
    planned: Any = plan.explain(format="json")
    assert planned["assets"]["cell"] == {"kind": "column", "column": "h3"}
    assert "H3 cell: the asset column h3." in setup_text(planned)
    sample = plan.samples()["warehouse-a"]
    assert (
        f"The impact reads H3 cell {A_CELL:x} for asset warehouse-a: the asset "
        "column h3." in sample.provenance.notes
    )


def test_only_computing_a_cell_needs_the_geometry_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _SourcePlan(_annual())
    assets = _assets("warehouse-a")
    monkeypatch.setitem(sys.modules, "h3", None)
    plan = _plan(source, assets, impact=RX1DAY, cell=H3Cell(resolution=7))
    plan.explain()
    with pytest.raises(ImportError, match=r"crc-sdk\[geometry\]"):
        plan.basis_risk(NoBasisRisk()).samples()
    # Refused before the source is read.
    assert source.calls == []
    # Reading the cell from a column computes nothing.
    read = _plan(assets=_with_cells(assets, A_CELL), impact=RX1DAY, cell="h3")
    assert _fixed_context(_no_noise(read)["warehouse-a"])["cell"] == f"{A_CELL:x}"


def test_a_plan_without_cell_is_as_before() -> None:
    plan = _plan().basis_risk(NoBasisRisk())
    assert plan.cell is None
    document: Any = plan.explain(format="json")
    assert "cell" not in document["assets"]
    assert {str(s["notes"]) for s in document["setup"]["samples"]} == {"[]"}
    sample = plan.samples()["warehouse-a"]
    assert sample.provenance.to_document()["loss_model"]["context"] is None
    assert not any("H3" in note for note in sample.provenance.notes)
    with pytest.raises(SampleError, match=r"cell= \(an asset column"):
        _plan(impact=RX1DAY, value_column=None, value_unit="1")
    # Assets with an H3 cell are refused too without context columns or cell=.
    with pytest.raises(SampleError, match="H3 cell"):
        _plan(assets=_cell_assets(A_CELL), impact=RX1DAY)


def test_cell_goes_on_top_of_the_context() -> None:
    assets = _assets("warehouse-a").append_column("country", pa.array(["DEU"]))
    fixed = {
        "building_type": None,
        "cell": f"{A_CELL:x}",
        "continent": None,
        "country": "DEU",
        "historic_mean": None,
    }
    for options in (
        {"context": ImpactContextColumns(country="country")},
        {"context": TransformContext(country="DEU")},
        {
            "impact": framework_impacts.for_factor(
                "rx1day", context=TransformContext(country="DEU")
            )
        },
    ):
        arguments: dict[str, Any] = {"impact": RX1DAY, **options}
        sample = _no_noise(_plan(assets=assets, cell=H3Cell(7), **arguments))
        assert _fixed_context(sample["warehouse-a"]) == fixed
    # Assets located by their H3 cells may name the column, and may not ask
    # for a cell computed from a point they do not have.
    located = _plan(assets=_cell_assets(A_CELL), impact=RX1DAY, cell="cell_index")
    assert _fixed_context(_no_noise(located)["warehouse-a"])["cell"] == f"{A_CELL:x}"
    plan = _plan(assets=_cell_assets(A_CELL), impact=RX1DAY, cell=H3Cell(7))
    with pytest.raises(SampleError, match="from its point"):
        _no_noise(plan)


def test_cell_is_checked_when_the_plan_is_built() -> None:
    wrong: tuple[Any, ...] = (-1, 16, True, 7.0, "7")
    for resolution in wrong:
        with pytest.raises(ValueError, match="resolution"):
            H3Cell(resolution)
    numpy_integer: Any = np.int64(4)
    assert H3Cell(numpy_integer).resolution == 4
    assert H3Cell(7).to_document() == {"kind": "h3_cell", "resolution": 7}
    with pytest.raises(TypeError, match="H3Cell"):
        _plan(cell=7)
    with pytest.raises(ValueError, match="column"):
        _plan(cell="")
    with pytest.raises(SampleError, match="no columns"):
        _plan(cell="h3")
    # An impact that takes no context cannot take a cell.
    calibration = ImpactCalibration(
        exposure_variable="rx1day",
        exposure_unit="mm/day",
        impact_metric="damage_ratio",
        impact_unit="1",
    )
    expression = ImpactExpression.tabulated(
        [(0.0, 0.0), (200.0, 1.0)], calibration=calibration
    )
    for impact in (expression, lambda values: values / 200.0):
        with pytest.raises(SampleError, match="take none"):
            _plan(impact=impact, cell=H3Cell(7))
    # Two statements of the cell are not resolved silently.
    held = TransformContext(cell=A_CELL)
    with pytest.raises(ValueError, match="state one"):
        _plan(impact=RX1DAY, context=held, cell=H3Cell(7))
    with pytest.raises(ValueError, match="state one"):
        _plan(impact=framework_impacts.for_factor("rx1day", context=held), cell="h3")


def test_assets_without_a_usable_cell_raise(tmp_path: Path) -> None:
    def plan(assets: Any, cell: Any, **options: Any) -> ParametricPlan:
        return _plan(assets=assets, impact=RX1DAY, cell=cell, **options)

    assets = _assets("warehouse-a")
    with pytest.raises(SampleError, match="no longitude or latitude"):
        _no_noise(
            plan(
                assets.set_column(1, "longitude", pa.array([None], pa.float64())),
                H3Cell(7),
            )
        )
    stations = pa.table(
        {
            "station": ["S1"] * 3,
            "year": pa.array([2000, 2001, 2002], pa.int32()),
            "value": [50.0, 60.0, 70.0],
        }
    )
    outside = assets.set_column(2, "latitude", pa.array([95.0])).append_column(
        "station_id", pa.array(["S1"])
    )
    with pytest.raises(SampleError, match="latitude"):
        _no_noise(
            _plan(
                stations,
                outside,
                impact=RX1DAY,
                cell=H3Cell(7),
                location=StationColumn("station_id"),
            )
        )
    with pytest.raises(SampleError, match="has no H3 cell"):
        _no_noise(plan(_with_cells(assets, None), "h3"))
    text = assets.append_column("h3", pa.array([f"{A_CELL:x}"]))
    with pytest.raises(SampleError, match="unsigned 64-bit"):
        _no_noise(plan(text, "h3"))
    file = tmp_path / "assets.parquet"
    pq.write_table(assets, file)
    with pytest.raises(SampleError, match="no column 'h3'"):
        _no_noise(plan(file, "h3"))
    # The registry impact refuses what is not a cell, or too coarse a cell.
    with pytest.raises(ValueError, match="H3"):
        _no_noise(plan(_with_cells(assets, 12345), "h3"))
    with pytest.raises(ValueError, match="H3"):
        _no_noise(plan(assets, H3Cell(3)))


def test_hazard_noise_runs_through_a_per_year_plan() -> None:
    plan = (
        _plan()
        .basis_risk(HazardNoise(0.3))
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    samples = plan.samples()
    for sample in samples.values():
        assert sample.hazard is not None
        assert sample.provenance.basis_risk == {"kind": "hazard_noise", "sigma": 0.3}
    assert not np.array_equal(
        samples["warehouse-a"].hazard, samples["depot-b"].hazard
    ), "each asset draws from its own generator"
    text = setup_text(plan.explain(format="json"))
    assert "hazard noise, the hazard at the asset being the index times" in text


def test_basis_risk_is_required_and_noise_needs_a_seed() -> None:
    plan = _plan().covers(_linear())
    with pytest.raises(ValueError, match="basis-risk"):
        plan.samples()
    with pytest.raises(ValueError, match="basis-risk"):
        plan.write_parquet("never.parquet")
    with pytest.raises(ValueError, match="seed"):
        _plan(seed=None).basis_risk(ImpactNoise(0.3))
    for sigma in (0.0, 0.3):
        with pytest.raises(ValueError, match="HazardNoise.*seed"):
            _plan(seed=None).basis_risk(HazardNoise(sigma))
    with pytest.raises(TypeError, match="BasisRisk"):
        UNTYPED_PLAN.basis_risk(plan, "none")


def test_covers_are_checked_against_the_plans_units_and_location() -> None:
    plan = _plan()
    with pytest.raises(SampleError, match="pays in"):
        plan.covers(Cover(INDEX, LinearPayout(60.0, 120.0, 1.0), payout_unit="USD"))
    with pytest.raises(SampleError, match="index in"):
        plan.covers(
            Cover(
                IndexDefinition("rx1day", "mm", aggregation="annual_maximum"),
                LinearPayout(60.0, 120.0, 1.0),
                payout_unit="EUR",
            )
        )
    at_station = IndexDefinition(
        "rx1day", "mm/day", location=AtStation("X1"), aggregation="annual_maximum"
    )
    with pytest.raises(SampleError, match="AtAsset"):
        plan.covers(
            Cover(at_station, LinearPayout(60.0, 120.0, 1.0), payout_unit="EUR")
        )
    with pytest.raises(ValueError, match="twice"):
        plan.covers(_linear(), _linear())
    with pytest.raises(ValueError, match="at least one"):
        plan.covers()
    with pytest.raises(TypeError, match="Cover"):
        UNTYPED_PLAN.covers(plan, "cover")


def test_objectives_and_splits_are_checked() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="twice"):
        plan.objective(MSE(), MSE())
    with pytest.raises(ValueError, match="at least one"):
        plan.objective()
    with pytest.raises(TypeError, match="objectives"):
        UNTYPED_PLAN.objective(plan, "mse")
    with pytest.raises(TypeError, match="pricing"):
        UNTYPED_PLAN.objective(plan, MSE(), pricing="pure")
    with pytest.raises(TypeError, match="Holdout"):
        UNTYPED_PLAN.split(plan, 0.7)
    with pytest.raises(TypeError, match="MetricOptions"):
        UNTYPED_PLAN.metrics(plan, {"betas": (0.9,)})
    with pytest.raises(ValueError, match="train_fraction"):
        Holdout(1.0)
    with pytest.raises(ValueError, match="k must"):
        KFold(1)


COC = CostOfCapitalLoad(0.06, 0.99)


def test_pricing_prices_a_plan_of_resolved_covers(tmp_path: Path) -> None:
    base = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    plan = base.pricing(COC)
    assert base.pricing_rule is None and plan.pricing_rule == COC
    # explain() states the pricing of every planned evaluation, without I/O.
    document: Any = plan.explain(format="json")
    assert document["designs"] is None
    assert [e["pricing"] for e in document["setup"]["evaluations"]] == [
        COC.to_document()
    ] * 2
    loaded = "Pricing: a cost-of-capital load, P = (1 - 0.06) E[payout] + 0.06 "
    assert loaded in setup_text(document["setup"])
    unpriced: Any = base.explain(format="json")
    assert {e["pricing"]["kind"] for e in unpriced["setup"]["evaluations"]} == {"pure"}
    # Every row is priced with it on its evaluation block, as compare prices.
    result = plan.write_parquet(tmp_path / "priced.parquet")
    pure = base.write_parquet(tmp_path / "pure.parquet")
    train, test = holdout(result.samples["warehouse-a"], 0.7)
    blocks: tuple[tuple[Literal["in_sample", "out_of_sample"], Any], ...] = (
        ("in_sample", train),
        ("out_of_sample", test),
    )
    for split, block in blocks:
        by_hand = compare(
            [_linear("warehouse-a")],
            block,
            split=split,
            pricing=COC,
            design_sample=train,
            options=OPTIONS,
        ).to_pylist()[0]
        row = next(r for r in result.table.to_pylist() if r["split"] == split)
        assert row["premium"] == by_hand["premium"]
        assert row["cvar_net_loss_b0_9"] == by_hand["cvar_net_loss_b0_9"]
        unloaded = next(r for r in pure.table.to_pylist() if r["split"] == split)
        assert row["premium"] > unloaded["premium"]
    assert {
        json.dumps(e["pricing"], sort_keys=True)
        for e in result.metadata.setup.evaluations
    } == {json.dumps(COC.to_document(), sort_keys=True)}
    assert loaded in setup_text(result.metadata)
    with pytest.raises(TypeError, match="pricing"):
        UNTYPED_PLAN.pricing(base, "cost of capital")


def test_pricing_and_the_objectives_pricing_must_agree() -> None:
    plan = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_step(), _linear())
        .split(Holdout(0.7))
    )
    by_pricing = plan.pricing(COC).objective(MSE(), budget=Budget(ratio=1.0))
    by_objective = plan.objective(MSE(), pricing=COC, budget=Budget(ratio=1.0))
    stated_twice = plan.objective(
        MSE(), pricing=CostOfCapitalLoad(0.06, 0.99), budget=Budget(ratio=1.0)
    ).pricing(COC)
    documents: list[Any] = [
        p.explain(format="json") for p in (by_pricing, by_objective, stated_twice)
    ]
    assert documents[0] == documents[1] == documents[2]
    assert documents[0]["designs"]["pricing"] == COC.to_document()
    # The designs are priced with the plan's pricing too.
    designed = by_pricing.designs()
    assert [d.design_id for d in designed] == [
        d.design_id for d in by_objective.designs()
    ]
    train, _ = holdout(by_pricing.samples()["warehouse-a"], 0.7)
    bound = Cover(
        INDEX, _step().payout, payout_unit="EUR", asset_id="warehouse-a", label="step"
    )
    expected = design(bound, train, MSE(), pricing=COC, budget=Budget(ratio=1.0))
    assert designed[0].design_id == expected.design_id
    # Two different pricings raise, in either order.
    with pytest.raises(ValueError, match="state it once"):
        plan.pricing(COC).objective(MSE(), pricing=PurePremium())
    with pytest.raises(ValueError, match="state it once"):
        plan.objective(MSE(), pricing=PurePremium()).pricing(COC)
    # Without either, the premiums are pure, as before.
    pure: Any = plan.objective(MSE()).explain(format="json")
    assert pure["designs"]["pricing"] == {"kind": "pure"}
    # An objective restated without a pricing keeps .pricing(...)'s.
    restated: Any = (
        plan.pricing(COC).objective(MSE()).objective(MSE()).explain(format="json")
    )
    assert restated["designs"]["pricing"] == COC.to_document()


# -- samples and seeding -------------------------------------------------------


def test_samples_match_the_function_route() -> None:
    table = _annual()
    plan = _plan(table).basis_risk(ImpactNoise(0.3, upper=1.0))
    samples = plan.samples()
    assert list(samples) == ["warehouse-a", "depot-b"]
    children = np.random.SeedSequence(42).spawn(2)
    assets = AssetPortfolio(_assets())
    for (asset_id, sample), child, (longitude, latitude) in zip(
        samples.items(), children, [(6.96, 50.94), (7.10, 51.10)], strict=True
    ):
        series = AnnualSeries.nearest_cell(
            table,
            longitude=longitude,
            latitude=latitude,
            unit="mm/day",
            source=PROVENANCE,
        )
        loss_model = LossModel(RAMP, value_column="value_eur", value_unit="EUR")
        expected = sample_annual(
            series,
            loss_model=loss_model.for_asset(assets, asset_id),
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id=asset_id,
            rng=np.random.default_rng(child),
        )
        assert sample.content_hash == expected.content_hash
        assert sample.period.tolist() == YEARS.tolist()
        assert sample.provenance.sources[0]["series"]["source"] == (
            PROVENANCE.model_dump(mode="json")
        )
        assert sample.provenance.licences[0]["licence"] == "CC0-1.0"
    again = plan.samples()
    assert [s.content_hash for s in again.values()] == [
        s.content_hash for s in samples.values()
    ]


def test_each_asset_draws_from_its_own_child_generator() -> None:
    plan = _plan().basis_risk(ImpactNoise(0.3, upper=1.0))
    both = plan.samples()
    first = _plan(assets=_assets("warehouse-a"))
    first_only = first.basis_risk(ImpactNoise(0.3, upper=1.0)).samples()
    # Another asset after it changes nothing for the first asset's draws.
    assert first_only["warehouse-a"].content_hash == both["warehouse-a"].content_hash
    other_seed = _plan(seed=7).basis_risk(ImpactNoise(0.3, upper=1.0)).samples()
    assert other_seed["warehouse-a"].content_hash != both["warehouse-a"].content_hash
    without_seed = _plan(seed=None).basis_risk(NoBasisRisk()).samples()
    assert without_seed["warehouse-a"].provenance.rng == ()


def test_stations_are_matched_by_an_asset_column() -> None:
    rng = np.random.default_rng(3)
    stations = pa.table(
        {
            "station": ["S1"] * 30 + ["S2"] * 30,
            "year": pa.array(list(range(1991, 2021)) * 2, pa.int32()),
            "value": rng.gumbel(50.0, 12.0, 60),
        }
    )
    assets = pa.table(
        {
            "asset_id": ["a", "b"],
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "station_id": ["S2", "S1"],
        }
    )
    plan = _plan(
        stations,
        assets,
        value_column=None,
        value_unit="1",
        location=StationColumn("station_id"),
    ).basis_risk(NoBasisRisk())
    samples = plan.samples()
    expected = AnnualSeries.from_table(stations, where={"station": "S2"}, unit="mm/day")
    np.testing.assert_array_equal(samples["a"].index[:, 0], expected.values)
    series = samples["b"].provenance.sources[0]["series"]
    assert series["where"] == {"station": "S1"}


def test_missing_years_raise_unless_dropped() -> None:
    table = _annual()
    keep = np.ones(table.num_rows, dtype=bool)
    keep[np.flatnonzero(table.column("year").to_numpy() == 1990)[:1]] = False
    gappy = table.filter(pa.array(keep))
    plan = _plan(gappy, _assets("warehouse-a")).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="drop_year"):
        plan.samples()
    dropped = _plan(gappy, _assets("warehouse-a"), missing="drop_year")
    sample = dropped.basis_risk(NoBasisRisk()).samples()["warehouse-a"]
    assert 1990 not in sample.period.tolist() and len(sample) == len(YEARS) - 1
    assert list(sample.provenance.details["dropped_years"]) == [1990]


def test_an_asset_outside_the_grid_raises() -> None:
    far = pa.table(
        {
            "asset_id": ["far"],
            "longitude": [12.0],
            "latitude": [40.0],
            "value_eur": [1.0],
        }
    )
    with pytest.raises(SampleError, match="half a grid step"):
        _plan(assets=far).basis_risk(NoBasisRisk()).samples()


def test_repeated_assets_raise() -> None:
    assets = pa.concat_tables([_assets("warehouse-a"), _assets("warehouse-a")])
    with pytest.raises(SampleError, match="more than once"):
        _plan(assets=assets).basis_risk(NoBasisRisk()).samples()


# -- covers, designs and the comparison ---------------------------------------


def test_designs_follow_the_function_route() -> None:
    table = _annual()
    plan = (
        _plan(table)
        .basis_risk(NoBasisRisk())
        .covers(_step(), _linear())
        .objective(MSE(), ExpectedUncovered(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
    )
    designs = plan.designs()
    assert [(d.cover.asset_id, d.cover.label) for d in designs] == [
        ("warehouse-a", "step (mse)"),
        ("warehouse-a", "step (expected_uncovered)"),
        ("depot-b", "step (mse)"),
        ("depot-b", "step (expected_uncovered)"),
    ]
    sample = plan.samples()["warehouse-a"]
    train, _ = holdout(sample, 0.7)
    bound = Cover(
        INDEX,
        _step().payout,
        payout_unit="EUR",
        asset_id="warehouse-a",
        label="step",
    )
    expected = design(bound, train, MSE(), budget=Budget(ratio=1.0))
    assert designs[0].design_id == expected.design_id
    assert designs[0].cover.definition_hash == expected.cover.definition_hash
    assert designs[0].design_sample.content_hash == train.content_hash


def test_designs_need_an_objective_and_covers() -> None:
    plan = _plan().basis_risk(NoBasisRisk()).covers(_step())
    with pytest.raises(ValueError, match="objective"):
        plan.designs()
    with pytest.raises(ValueError, match="no covers"):
        _plan().basis_risk(NoBasisRisk()).objective(MSE()).designs()
    with pytest.raises(ValueError, match="free parameters"):
        plan.write_parquet("never.parquet")


def test_covers_are_bound_to_their_assets() -> None:
    twice = _plan().basis_risk(NoBasisRisk()).covers(_linear(), _linear("depot-b"))
    with pytest.raises(ValueError, match="applies to asset 'depot-b' twice"):
        # The cover written for no asset, bound to depot-b, is the second one.
        twice.write_parquet("never.parquet")
    unknown = _plan().basis_risk(NoBasisRisk()).covers(_linear(asset_id="elsewhere"))
    with pytest.raises(ValueError, match="do not hold"):
        unknown.write_parquet("never.parquet")
    only_b = _plan().basis_risk(NoBasisRisk()).covers(_linear("depot-b"))
    with pytest.raises(ValueError, match="has no cover"):
        only_b.write_parquet("never.parquet")


def test_write_parquet_writes_every_split_with_its_setup(tmp_path: Path) -> None:
    other = Cover(
        INDEX, LinearPayout(70.0, 130.0, 4e5), payout_unit="EUR", asset_id="depot-b"
    )
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear(), other)
        .objective(MSE(), WeightedCVaR(alpha=0.9, beta=0.9), budget=Budget(ratio=1.0))
        .split(KFold(4))
        .metrics(OPTIONS)
    )
    output = tmp_path / "nested" / "comparison.parquet"
    result = plan.write_parquet(output)
    assert result.output == output and output.is_file()
    assert [path.name for path in output.parent.iterdir()] == ["comparison.parquet"]
    table = pq.read_table(output)
    assert table.num_rows == result.row_count
    # Per fold and split: warehouse-a has 2 designs and 1 cover, depot-b 2 and 2.
    assert result.row_count == 4 * 2 * (3 + 4)
    rows = table.select(["asset_id", "cover_label", "split", "fold"]).to_pylist()
    labels = {(row["asset_id"], row["cover_label"]) for row in rows}
    assert ("warehouse-a", "linear") in labels
    assert ("depot-b", "linear") in labels
    assert {r["asset_id"] for r in rows if r["cover_label"] == _label(other)} == {
        "depot-b"
    }
    out = table.filter(pa.compute.equal(table.column("split"), "out_of_sample"))
    designed = out.filter(pa.compute.is_valid(out.column("design_id")))
    assert set(designed.column("premium_basis").to_pylist()) == {"design"}
    fixed = out.filter(pa.compute.is_null(out.column("design_id")))
    assert set(fixed.column("premium_basis").to_pylist()) == {"evaluation"}
    assert len(result.designs) == 4 * 2 * 2
    assert set(result.samples) == {"warehouse-a", "depot-b"}

    metadata = EvaluationMetadata.from_table(table)
    assert metadata == result.metadata
    assert metadata.assets["count"] == 2
    assert metadata.assets["id_column"] == "asset_id"
    assert str(metadata.assets["values_sha256"]).startswith("sha256:")
    assert "2000000.0" not in metadata.to_json()  # asset values are not written
    methods = {(e["split"], e["method"]) for e in metadata.setup.evaluations}
    assert methods == {("in_sample", "in_sample"), ("out_of_sample", "kfold")}
    assert all(e["details"]["plan_seed"] == 42 for e in metadata.setup.evaluations)
    text = setup_text(metadata)
    assert "fold 3 of a 4-fold split over blocks of years" in text
    assert "the leakage check passed" in text
    assert "Design step (mse) (" in text
    # The annual table's record is written once; every hazard source names it.
    hazard = [entry for entry in metadata.sources if entry["role"] == "hazard"]
    assert len(hazard) == len(metadata.setup.samples)
    (record,) = metadata.source_records
    assert set(record) == {"annual_table"}
    table_record = record["annual_table"]
    assert table_record["kind"] == "table"
    assert table_record["table_sha256"].startswith("sha256:")
    assert table_record["rows"] == len(YEARS) * len(CELLS)
    assert all("annual_table" not in entry for entry in hazard)
    assert {entry["annual_table_sha256"] for entry in hazard} == {
        table_record["table_sha256"]
    }
    assert metadata.to_json().count(table_record["table_sha256"]) == 1 + len(hazard)
    assert metadata.licences[0]["licence"] == "CC0-1.0"
    assert len({entry["sample_hash"] for entry in metadata.rng}) == len(
        metadata.setup.samples
    )
    assert len(metadata.designs) == len(result.designs)


def test_without_a_split_every_row_is_in_sample(tmp_path: Path) -> None:
    result = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_linear(), _step())
        .objective(ExpectedUncovered(), budget=Budget(ratio=1.0))
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "in-sample.parquet")
    )
    assert set(result.table.column("split").to_pylist()) == {"in_sample"}
    details = [e["details"] for e in result.metadata.setup.evaluations]
    assert details == [{"plan_seed": 42}]
    assert result.row_count == 2
    sample = result.samples["warehouse-a"]
    assert result.designs[0].design_sample.content_hash == sample.content_hash
    assert result.fresh_samples is None


def test_a_written_file_is_replaced_atomically(tmp_path: Path) -> None:
    output = tmp_path / "comparison.parquet"
    output.write_bytes(b"previous")
    plan = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    plan.write_parquet(output)
    assert pq.read_table(output).num_rows == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["comparison.parquet"]
    with pytest.raises(ValueError, match="local"):
        plan.write_parquet("s3://bucket/comparison.parquet")
    unseeded = _plan(assets=_assets("warehouse-a"), seed=None)
    written = (
        unseeded.basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(output)
    )
    assert [e["details"] for e in written.metadata.setup.evaluations] == [{}]


def test_file_sources_and_assets_are_read_at_execution(tmp_path: Path) -> None:
    source, assets = tmp_path / "annual.parquet", tmp_path / "assets.parquet"
    plan = (
        _plan(source, assets)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    pq.write_table(_annual(), source)
    pq.write_table(_assets(), assets)
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    entry = next(e for e in result.metadata.sources if e["role"] == "hazard")
    (record,) = result.metadata.source_records
    assert record["annual_table"]["path"] == str(source)
    assert record["annual_table"]["file_sha256"].startswith("sha256:")
    series_source = entry["series"]["source"]
    assert series_source["file_sha256"] == record["annual_table"]["file_sha256"]
    assert series_source["licence"] == "CC0-1.0"
    assert result.row_count == 2


def test_a_source_plan_is_executed_once_and_explained(tmp_path: Path) -> None:
    source = _SourcePlan(_annual())
    plan = _plan(source).basis_risk(NoBasisRisk()).covers(_linear()).metrics(OPTIONS)
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    assert source.calls == ["annual_extremes", "explain"]
    entry = next(e for e in result.metadata.sources if e["role"] == "hazard")
    (record,) = result.metadata.source_records
    assert record["annual_table"]["explain"]["recipe"] == "rx1day"
    assert entry["series"]["source"]["source_type"] == "_SourcePlan"
    with pytest.raises(SampleError, match="unit"):
        _plan(_SourcePlan(_annual(), unit="mm")).basis_risk(NoBasisRisk()).samples()


def _earlier_layout(metadata: EvaluationMetadata) -> dict[str, Any]:
    """The metadata as plans wrote it before `source_records`.

    The annual table's record was then repeated in every hazard source, under
    `annual_table`, and the metadata had no `source_records`.
    """
    document: dict[str, Any] = json.loads(metadata.to_json())
    (record,) = document.pop("source_records")
    document["sources"] = [
        {
            **{k: v for k, v in entry.items() if k != "annual_table_sha256"},
            **record,
        }
        if entry["role"] == "hazard"
        else entry
        for entry in document["sources"]
    ]
    return document


def test_metadata_in_the_earlier_layout_still_reads(tmp_path: Path) -> None:
    plan = (
        _plan()
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    earlier = _earlier_layout(result.metadata)
    assert "source_records" not in earlier
    assert all(
        entry["annual_table"]["table_sha256"].startswith("sha256:")
        for entry in earlier["sources"]
        if entry["role"] == "hazard"
    )
    stored = EvaluationMetadata.from_json(json.dumps(earlier))
    assert stored.source_records == ()
    # setup_text states the same setup from either layout, and from a table.
    assert setup_text(stored) == setup_text(result.metadata)
    text = json.dumps(earlier, sort_keys=True).encode("utf-8")
    table = result.table.replace_schema_metadata(
        {EVALUATION_METADATA_KEY.encode("utf-8"): text}
    )
    pq.write_table(table, tmp_path / "earlier.parquet")
    read = pq.read_table(tmp_path / "earlier.parquet")
    assert EvaluationMetadata.from_table(read) == stored
    assert setup_text(read) == setup_text(result.table)
    out = read.filter(pa.compute.equal(read.column("split"), "out_of_sample"))
    assert setup_text(out) == setup_text(
        result.table.filter(
            pa.compute.equal(result.table.column("split"), "out_of_sample")
        )
    )


def test_merged_comparisons_keep_each_source_record(tmp_path: Path) -> None:
    def written(table: Any, name: str) -> Any:
        return (
            _plan(table)
            .basis_risk(NoBasisRisk())
            .covers(_linear())
            .metrics(OPTIONS)
            .write_parquet(tmp_path / name)
        )

    first, second = written(_annual(1), "first.parquet"), written(_annual(2), "two")
    merged = EvaluationMetadata.merge(first.metadata, second.metadata)
    records = [entry["annual_table"] for entry in merged.source_records]
    assert [r["table_sha256"] for r in records] == [
        first.metadata.source_records[0]["annual_table"]["table_sha256"],
        second.metadata.source_records[0]["annual_table"]["table_sha256"],
    ]
    # Each hazard source names the record of the table its sample was read from.
    by_sample = {
        entry["sample_hash"]: entry["annual_table_sha256"]
        for entry in merged.sources
        if entry["role"] == "hazard"
    }
    for result, record in zip((first, second), records, strict=True):
        for sample in result.samples.values():
            assert by_sample[sample.content_hash] == record["table_sha256"]
    again = EvaluationMetadata.merge(first.metadata, first.metadata)
    assert again.source_records == first.metadata.source_records


def test_out_of_sample_rows_match_compare_on_the_held_out_years(
    tmp_path: Path,
) -> None:
    plan = (
        _plan(assets=_assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_step(), _linear())
        .objective(MSE(), budget=Budget(ratio=1.0))
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    sample = result.samples["warehouse-a"]
    train, test = holdout(sample, 0.7)
    by_hand = compare(
        [result.designs[0], _linear("warehouse-a")],
        test,
        split="out_of_sample",
        design_sample=train,
        options=OPTIONS,
    )
    written = result.table.filter(
        pa.compute.equal(result.table.column("split"), "out_of_sample")
    )
    for name in ("cover_hash", "sample_hash", "expected_uncovered", "premium", "rmse"):
        assert written.column(name).to_pylist() == by_hand.column(name).to_pylist()


def test_the_repr_is_short() -> None:
    plan = _plan().covers(_step()).objective(MSE()).split(Holdout(0.7))
    text = repr(plan)
    assert text.startswith("ParametricPlan.for_annual_extremes(Table, assets=2")
    assert "objectives=1" in text and len(text) < 200


class _CMIP6Plan(_SourcePlan):
    """A source plan whose `explain()` states a CMIP6 pathway."""

    def __init__(self, table: Any, pathway: str = "ssp245") -> None:
        super().__init__(table)
        self.pathway = pathway

    def explain(self, *, format: str = "text") -> Any:
        return {
            **super().explain(format=format),
            "archive": "aws",
            "catalogue": "https://example.org/pangeo-cmip6.csv",
            "pathway": self.pathway,
            "model": "M",
            "member": "r1i1p1f1",
        }


def test_a_cmip6_plans_pathway_is_stated_beside_the_callers_source(
    tmp_path: Path,
) -> None:
    source = _CMIP6Plan(_annual())
    result = (
        _plan(source, _assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    assert source.calls == ["annual_extremes", "explain"]
    assert _lines(setup_text(result.metadata))["Source"] == [
        "Source: the hazard series from provider example, dataset synthetic rx1day "
        "and pathway ssp245, with horizon not recorded."
    ]
    series = result.samples["warehouse-a"].provenance.sources[0]["series"]
    assert series["pathway"] == "ssp245"
    # The caller's own document is left as it is.
    assert "pathway" not in series["source"]
    assert series["source"]["licence"] == "CC0-1.0"


def test_a_plans_pathway_tells_its_samples_apart() -> None:
    def lineage(pathway: str) -> str | None:
        plan = _plan(_CMIP6Plan(_annual(), pathway), _assets("warehouse-a"))
        sample = plan.basis_risk(NoBasisRisk()).samples()["warehouse-a"]
        return sample.lineage

    assert lineage("ssp245") == lineage("ssp245")
    assert lineage("ssp245") != lineage("ssp585")


def test_a_plan_without_a_pathway_records_none() -> None:
    sample = (
        _plan(_SourcePlan(_annual()), _assets("warehouse-a"))
        .basis_risk(NoBasisRisk())
        .samples()["warehouse-a"]
    )
    assert "pathway" not in sample.provenance.sources[0]["series"]


def test_a_noaa_plan_works_on_the_per_year_route(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)

    class WithoutTheFetch(NOAAExtremeWaterLevelPlan):
        def annual_extremes(self) -> Any:
            return pa.table(
                {
                    "station_id": ["8518750"] * 30,
                    "year": pa.array(range(1991, 2021), pa.int32()),
                    "detrended_extreme_m": rng.gumbel(70.0, 12.0, 30),
                }
            )

    assets = pa.table(
        {
            "asset_id": ["quay"],
            "longitude": [-74.0],
            "latitude": [40.7],
            "station_id": ["8518750"],
            "value_eur": [2e6],
        }
    )
    result = (
        _plan(
            WithoutTheFetch.for_station("8518750"),
            assets,
            unit="m",
            location=StationColumn("station_id", table_column="station_id"),
            hazard_column="detrended_extreme_m",
        )
        .basis_risk(NoBasisRisk())
        .covers(
            Cover(
                IndexDefinition("water_level", "m", aggregation="annual_maximum"),
                LinearPayout(60.0, 120.0, 5e5),
                payout_unit="EUR",
            )
        )
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    (record,) = result.metadata.source_records
    assert record["annual_table"]["explain"]["station_id"] == "8518750"
    assert result.row_count == 1
