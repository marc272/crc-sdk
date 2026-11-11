"""ParametricPlan.index: an index read elsewhere than at the asset."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h3  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.geometry import point_to_cell
from crc_sdk.impacts import ImpactCalibration, ImpactExpression
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    ComonotoneIndexWarning,
    Cover,
    Free,
    HazardNoise,
    Holdout,
    ImpactNoise,
    IndexDefinition,
    KFold,
    LinearPayout,
    LossModel,
    MetricOptions,
    NearestCell,
    NoBasisRisk,
    ParametricPlan,
    SampleError,
    ScenarioError,
    StationColumn,
    StepPayout,
    compare,
    holdout,
    sample_annual,
    setup_text,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY
from crc_sdk.parametric.scenarios import _check_bound_cover
from crc_sdk.types import SourceProvenance
from crc_sdk.workflows import AssetPortfolio, CellColumn, PointColumns

UNTYPED_PLAN: Any = ParametricPlan
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
HAZARD = SourceProvenance(
    provider="example",
    dataset="synthetic rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic annual maxima",
)
GAUGES = SourceProvenance(
    provider="example",
    dataset="synthetic gauges",
    version="2",
    licence="CC-BY-4.0",
    attribution="Synthetic gauge network",
)
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
HAZARD_CELLS = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
INDEX_CELLS = ((50.75, 6.75), (50.75, 7.25), (51.25, 6.75), (51.25, 7.25))
YEARS = np.arange(1981, 2021)
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
POINTS = {"a": (6.96, 50.94), "b": (7.10, 51.10)}
STATIONS = {"a": "S2", "b": "S1"}
# The cells of INDEX_CELLS that hold each asset's point, as (latitude, longitude).
NEAREST = {"a": (50.75, 6.75), "b": (51.25, 7.25)}


def _grid(cells: Any, seed: int = 1, years: Any = YEARS) -> Any:
    """A per-cell annual table in the layout of ERA5's and CMIP6's."""
    rng = np.random.default_rng(seed)
    n = len(years) * len(cells)
    return pa.table(
        {
            "year": pa.array(np.repeat(years, len(cells)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in cells], len(years)),
            "longitude": np.tile([lon for _, lon in cells], len(years)),
            "value": rng.gumbel(50.0, 12.0, n),
        }
    )


def _hazard(years: Any = YEARS) -> Any:
    return _grid(HAZARD_CELLS, years=years)


def _index_grid() -> Any:
    return _grid(INDEX_CELLS, seed=5)


def _stations(years: Any = YEARS, seed: int = 3, names: Any = ("S1", "S2")) -> Any:
    """A per-station annual table, as a station plan's `annual_extremes()`."""
    rng = np.random.default_rng(seed)
    return pa.table(
        {
            "station": [name for name in names for _ in years],
            "year": pa.array(np.tile(years, len(names)), pa.int32()),
            "value": rng.gumbel(45.0, 11.0, len(names) * len(years)),
        }
    )


def _assets(*ids: str) -> Any:
    ids = ids or ("a", "b")
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [POINTS[i][0] for i in ids],
            "latitude": [POINTS[i][1] for i in ids],
            "value_eur": [2e6 + 1e5 * k for k in range(len(ids))],
            "station_id": [STATIONS[i] for i in ids],
            "reference_lon": [7.2 for _ in ids],
            "reference_lat": [50.8 for _ in ids],
        }
    )


def _plan(source: Any = None, assets: Any = None, **options: Any) -> ParametricPlan:
    arguments: dict[str, Any] = {
        "impact": RAMP,
        "unit": "mm/day",
        "provenance": HAZARD,
        "value_column": "value_eur",
        "value_unit": "EUR",
        "seed": 42,
    }
    arguments.update(options)
    plan: ParametricPlan = UNTYPED_PLAN.for_annual_extremes(
        _hazard() if source is None else source,
        _assets() if assets is None else assets,
        **arguments,
    )
    return plan


def _at_stations(
    plan: ParametricPlan, source: Any = None, **options: Any
) -> ParametricPlan:
    arguments: dict[str, Any] = {
        "location": StationColumn("station_id"),
        "unit": "mm/day",
        "provenance": GAUGES,
    }
    arguments.update(options)
    return plan.index(_stations() if source is None else source, **arguments)


def _nearest(
    plan: ParametricPlan, source: Any = None, **options: Any
) -> ParametricPlan:
    arguments: dict[str, Any] = {
        "location": NearestCell(),
        "unit": "mm/day",
        "provenance": GAUGES,
    }
    arguments.update(options)
    return plan.index(_index_grid() if source is None else source, **arguments)


def _linear(
    location: Any = AtAsset(), asset_id: str | None = None, label: str = "linear"
) -> Cover:
    index = IndexDefinition(
        "rx1day", "mm/day", location=location, aggregation="annual_maximum"
    )
    return Cover(
        index,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        asset_id=asset_id,
        label=label,
    )


def _step() -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        label="step",
    )


def _loss(asset_id: str) -> LossModel:
    model = LossModel(RAMP, value_column="value_eur", value_unit="EUR")
    return model.for_asset(AssetPortfolio(_assets()), asset_id)


class _SourcePlan:
    """A stand-in for a crc-sdk plan with an annual_extremes() execution point."""

    def __init__(
        self, table: Any, unit: str = "mm/day", pathway: str | None = None
    ) -> None:
        self.table = table
        self.unit = unit
        self.pathway = pathway
        self.calls: list[str] = []

    def annual_extremes(self) -> Iterator[Any]:
        self.calls.append("annual_extremes")
        return iter(self.table.to_batches(max_chunksize=50))

    def explain(self, *, format: str = "text") -> Any:
        self.calls.append("explain")
        document: dict[str, Any] = {"recipe": "rx1day", "unit": self.unit}
        if self.pathway is not None:
            document["pathway"] = self.pathway
        return document


@pytest.fixture(autouse=True)
def _quiet_comonotone() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        yield


# -- the builder ---------------------------------------------------------------


def test_the_builder_returns_a_new_plan_and_does_no_io(tmp_path: Path) -> None:
    source, assets = tmp_path / "annual.parquet", tmp_path / "assets.parquet"
    index = _SourcePlan(_stations())
    base = _plan(source, assets)
    plan = _at_stations(base, index)
    assert base.index_read is None and plan.index_read is not None
    planned: Any = (
        plan.covers(_linear()).basis_risk(NoBasisRisk()).explain(format="json")
    )
    assert planned["sources"][1]["role"] == "index"
    assert index.calls == [] and not list(tmp_path.iterdir())
    for path_like in (tmp_path / "index.parquet", str(tmp_path / "index.parquet")):
        _at_stations(_plan(), path_like)
    assert not list(tmp_path.iterdir())


def test_a_later_index_replaces_an_earlier_one() -> None:
    plan = _nearest(_at_stations(_plan()))
    assert plan.index_read is not None
    assert isinstance(plan.index_read.location, NearestCell)


def test_the_builder_checks_its_arguments() -> None:
    plan: Any = _plan()
    stations = _stations()
    location = StationColumn("station_id")
    with pytest.raises(TypeError, match="source"):
        plan.index(1.0, location=location, unit="mm/day", provenance=GAUGES)
    with pytest.raises(TypeError, match="location"):
        plan.index(stations, location="station", unit="mm/day", provenance=GAUGES)
    with pytest.raises(TypeError, match="provenance"):
        plan.index(stations, location=location, unit="mm/day", provenance="gauges")
    with pytest.raises(SampleError, match="unit"):
        plan.index(stations, location=location, unit=" ", provenance=GAUGES)
    with pytest.raises(TypeError, match="point"):
        plan.index(
            stations,
            location=NearestCell(),
            unit="mm/day",
            provenance=GAUGES,
            point=("lon", "lat"),
        )
    with pytest.raises(ValueError, match="year_column"):
        plan.index(
            stations,
            location=location,
            unit="mm/day",
            provenance=GAUGES,
            year_column="",
        )
    with pytest.raises(ValueError, match="index_column"):
        plan.index(
            stations,
            location=location,
            unit="mm/day",
            provenance=GAUGES,
            index_column=" value",
        )
    with pytest.raises(TypeError, match="location"):
        plan.index(stations, unit="mm/day", provenance=GAUGES)
    with pytest.raises(SampleError, match="no columns"):
        plan.index(
            stations,
            location=StationColumn("absent"),
            unit="mm/day",
            provenance=GAUGES,
        )


def test_point_goes_with_nearest_cell_only() -> None:
    with pytest.raises(ValueError, match="StationColumn"):
        _at_stations(_plan(), point=PointColumns("reference_lon", "reference_lat"))
    with pytest.raises(SampleError, match="no columns"):
        _nearest(_plan(), point=PointColumns("absent", "reference_lat"))
    with pytest.raises(ValueError, match="point"):
        _nearest(_plan(), point=PointColumns("", "reference_lat"))


def test_without_index_a_cover_read_elsewhere_stays_refused() -> None:
    with pytest.raises(SampleError, match=r"AtAsset\(\).*\.index\("):
        _plan().covers(_linear(AtStation("S2")))


# -- the samples ---------------------------------------------------------------


@pytest.mark.parametrize("model", [NoBasisRisk(), ImpactNoise(0.3, upper=1.0)])
def test_samples_equal_a_station_index_built_by_hand(model: Any) -> None:
    hazard, stations = _hazard(), _stations()
    samples = _at_stations(_plan(hazard), stations).basis_risk(model).samples()
    children = np.random.SeedSequence(42).spawn(2)
    for asset_id, child in zip(("a", "b"), children, strict=True):
        longitude, latitude = POINTS[asset_id]
        series = AnnualSeries.nearest_cell(
            hazard, longitude=longitude, latitude=latitude, unit="mm/day", source=HAZARD
        )
        station_series = AnnualSeries.from_table(
            stations,
            where={"station": STATIONS[asset_id]},
            station="station",
            unit="mm/day",
            years=series.years.tolist(),
            source=GAUGES,
        )
        expected = sample_annual(
            series,
            index=station_series,
            loss_model=_loss(asset_id),
            basis_risk=model,
            asset_id=asset_id,
            rng=np.random.default_rng(child),
        )
        sample = samples[asset_id]
        assert sample.content_hash == expected.content_hash
        assert sample.lineage == expected.lineage
        assert sample.to_table().equals(expected.to_table())
        assert sample.provenance.sources == expected.provenance.sources
        assert sample.provenance.licences == expected.provenance.licences
        np.testing.assert_array_equal(sample.index[:, 0], station_series.values)
        assert sample.hazard is not None
        np.testing.assert_array_equal(sample.hazard[:, 0], series.values)
        assert sample.provenance.sources[1]["series"]["where"] == {
            "station": STATIONS[asset_id]
        }


def test_a_station_column_of_any_name_is_the_series_station() -> None:
    # A table whose station column is not called `station`, as HYDAT's.
    gauges = _stations().rename_columns(["station_number", "year", "value"])
    location = StationColumn("station_id", table_column="station_number")
    samples = (
        _at_stations(_plan(), gauges, location=location)
        .basis_risk(NoBasisRisk())
        .samples()
    )
    for asset_id, station in STATIONS.items():
        sample = samples[asset_id]
        series = sample.provenance.sources[1]["series"]
        assert series["columns"]["station"] == "station_number"
        assert series["where"] == {"station_number": station}
        _check_bound_cover(_linear(AtStation(station), asset_id=asset_id), sample)
        other = "S1" if station == "S2" else "S2"
        with pytest.raises(ScenarioError, match="location"):
            _check_bound_cover(_linear(AtStation(other), asset_id=asset_id), sample)
    # The hazard read at a station records its column too.
    hazard = _plan(gauges, location=location).basis_risk(NoBasisRisk()).samples()
    for asset_id, station in STATIONS.items():
        series = hazard[asset_id].provenance.sources[0]["series"]
        assert series["columns"]["station"] == "station_number"
        assert series["where"] == {"station_number": station}


def test_samples_equal_a_nearest_cell_index_built_by_hand() -> None:
    hazard, grid = _hazard(), _index_grid()
    samples = _nearest(_plan(hazard), grid).basis_risk(NoBasisRisk()).samples()
    for asset_id in ("a", "b"):
        longitude, latitude = POINTS[asset_id]
        series = AnnualSeries.nearest_cell(
            hazard, longitude=longitude, latitude=latitude, unit="mm/day", source=HAZARD
        )
        cell_series = AnnualSeries.nearest_cell(
            grid,
            longitude=longitude,
            latitude=latitude,
            unit="mm/day",
            years=series.years.tolist(),
            source=GAUGES,
        )
        expected = sample_annual(
            series,
            index=cell_series,
            loss_model=_loss(asset_id),
            basis_risk=NoBasisRisk(),
            asset_id=asset_id,
        )
        assert samples[asset_id].content_hash == expected.content_hash
        assert samples[asset_id].lineage == expected.lineage
        latitude, longitude = NEAREST[asset_id]
        cell = samples[asset_id].provenance.sources[1]["series"]["cell"]
        assert cell == {"latitude": latitude, "longitude": longitude}


def test_nearest_cell_reads_at_another_point_when_stated() -> None:
    grid = _index_grid()
    plan = _nearest(
        _plan(), grid, point=PointColumns("reference_lon", "reference_lat")
    ).basis_risk(NoBasisRisk())
    samples = plan.samples()
    expected = AnnualSeries.nearest_cell(
        grid, longitude=7.2, latitude=50.8, unit="mm/day", source=GAUGES
    )
    for sample in samples.values():
        np.testing.assert_array_equal(sample.index[:, 0], expected.values)
        assert sample.provenance.sources[1]["series"]["cell"] == {
            "latitude": 50.75,
            "longitude": 7.25,
        }


def test_named_columns_and_a_stated_spacing_are_used() -> None:
    renamed = _index_grid().rename_columns(["ano", "lat", "lon", "rain"])
    samples = (
        _plan()
        .index(
            renamed,
            location=NearestCell(longitude="lon", latitude="lat", spacing=(0.5, 0.5)),
            unit="mm/day",
            provenance=GAUGES,
            year_column="ano",
            index_column="rain",
        )
        .basis_risk(NoBasisRisk())
        .samples()
    )
    plain = _nearest(_plan()).basis_risk(NoBasisRisk()).samples()
    for asset_id, sample in samples.items():
        np.testing.assert_array_equal(sample.index, plain[asset_id].index)
        columns = sample.provenance.sources[1]["series"]["columns"]
        assert columns == {
            "year": "ano",
            "value": "rain",
            "latitude": "lat",
            "longitude": "lon",
        }


def test_no_comonotone_warning_with_a_separate_index() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComonotoneIndexWarning)
        for sample in (
            _at_stations(_plan()).basis_risk(NoBasisRisk()).samples().values()
        ):
            assert sample.provenance.details["index_from"] == "series"
    with pytest.warns(ComonotoneIndexWarning):
        _plan().basis_risk(NoBasisRisk()).samples()


def test_impact_noise_may_be_added_on_top() -> None:
    plain = _at_stations(_plan()).basis_risk(NoBasisRisk()).samples()
    noisy = _at_stations(_plan()).basis_risk(ImpactNoise(0.3, upper=1.0)).samples()
    for asset_id, sample in noisy.items():
        np.testing.assert_array_equal(sample.index, plain[asset_id].index)
        assert not np.array_equal(sample.loss, plain[asset_id].loss)
        assert sample.provenance.rng != ()


@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_hazard_noise_is_refused_with_a_separate_index(sigma: float) -> None:
    plan = _at_stations(_plan()).basis_risk(HazardNoise(sigma))
    with pytest.raises(SampleError, match="separate index"):
        plan.samples()


def test_the_index_unit_is_not_checked_against_the_impact() -> None:
    expression = ImpactExpression.tabulated(
        [(0.0, 0.0), (200.0, 1.0)],
        calibration=ImpactCalibration(
            exposure_variable="rainfall",
            exposure_unit="mm/day",
            impact_metric="damage_ratio",
            impact_unit="1",
        ),
    )
    plan = _at_stations(_plan(impact=expression), unit="m/s").basis_risk(NoBasisRisk())
    assert plan.samples()["a"].index_unit == "m/s"
    with pytest.raises(SampleError):
        _plan(impact=expression, unit="m/s")


# -- binding the covers --------------------------------------------------------


def _documents(result: Any) -> dict[str, dict[str, Any]]:
    return {entry.hash: entry.document for entry in result.metadata.covers}


def test_covers_are_bound_to_the_resolved_station(tmp_path: Path) -> None:
    plan = (
        _at_stations(_plan())
        .basis_risk(NoBasisRisk())
        .covers(_linear(), _step())
        .objective(MSE())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    documents = _documents(result)
    rows = result.table.to_pylist()
    assert rows
    for row in rows:
        document = documents[row["cover_hash"]]
        assert document["asset_id"] == row["asset_id"]
        assert document["index"]["location"] == {
            "kind": "station",
            "station_id": STATIONS[row["asset_id"]],
        }
    for asset_id, station in STATIONS.items():
        expected = _linear(AtStation(station), asset_id=asset_id)
        linear_rows = [
            row
            for row in rows
            if row["asset_id"] == asset_id and row["cover_label"] == "linear"
        ]
        assert {row["cover_hash"] for row in linear_rows} == {expected.definition_hash}
    for design_result in result.designs:
        location = design_result.cover.index.location
        owner = design_result.cover.asset_id
        assert isinstance(location, AtStation) and owner is not None
        assert location.station_id == STATIONS[owner]


def test_the_location_is_in_the_hashes_and_the_design_id() -> None:
    def designed(name: str) -> Any:
        # The same numbers under another station name: only the location differs.
        table = _stations(names=(name,))
        assets = _assets("a").set_column(4, "station_id", pa.array([name]))
        plan = _at_stations(_plan(assets=assets), table).basis_risk(NoBasisRisk())
        (result,) = plan.covers(_step()).objective(MSE()).designs()
        return plan.samples()["a"], result

    (first_sample, first), (second_sample, second) = designed("S1"), designed("S3")
    assert first_sample.content_hash == second_sample.content_hash
    assert first.cover.payout.to_document() == second.cover.payout.to_document()
    assert first.cover.definition_hash != second.cover.definition_hash
    assert first.design_id != second.design_id


def test_covers_are_bound_to_the_cell_centre(tmp_path: Path) -> None:
    plan = (
        _nearest(_plan()).basis_risk(NoBasisRisk()).covers(_linear()).metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    documents = _documents(result)
    for row in result.table.to_pylist():
        latitude, longitude = NEAREST[row["asset_id"]]
        assert documents[row["cover_hash"]]["index"]["location"] == {
            "kind": "point",
            "latitude": latitude,
            "longitude": longitude,
            "name": None,
        }


def _greenwich() -> tuple[Any, Any, NearestCell]:
    """A hazard table, assets and the locator for a grid from 0 to 360 degrees."""
    hazard = _grid(((50.875, -0.125), (50.875, 0.125), (51.125, -0.125)))
    assets = pa.table(
        {
            "asset_id": ["a"],
            "longitude": [-0.1],
            "latitude": [50.9],
            "value_eur": [1e6],
        }
    )
    return hazard, assets, NearestCell(spacing=(0.5, 0.5))


def test_the_cell_centres_longitude_is_wrapped(tmp_path: Path) -> None:
    hazard, assets, location = _greenwich()
    index = _grid(((50.75, 359.75), (50.75, 0.25), (51.25, 359.75)), seed=5)
    plan = (
        _nearest(_plan(hazard, assets), index, location=location)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    ((_, document),) = _documents(result).items()
    assert document["index"]["location"] == {
        "kind": "point",
        "latitude": 50.75,
        "longitude": -0.25,
        "name": None,
    }
    # The series records the grid's own numbers.
    sample = result.samples["a"]
    assert sample.provenance.sources[1]["series"]["cell"]["longitude"] == 359.75


def test_a_cover_may_name_the_wrapped_centre_and_keeps_its_own_document(
    tmp_path: Path,
) -> None:
    hazard, assets, location = _greenwich()
    index = _grid(((50.75, 359.75), (50.75, 0.25), (51.25, 359.75)), seed=5)
    named = _linear(AtPoint(-0.25, 50.75, name="grid cell"))
    plan = (
        _nearest(_plan(hazard, assets), index, location=location)
        .basis_risk(NoBasisRisk())
        .covers(named)
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    ((_, document),) = _documents(result).items()
    assert document["index"]["location"]["name"] == "grid cell"


def test_a_cover_that_names_a_location_must_name_that_one(tmp_path: Path) -> None:
    right = _linear(AtStation("S2"), asset_id="a")
    plan = _at_stations(_plan(assets=_assets("a"))).basis_risk(NoBasisRisk())
    result = plan.covers(right).metrics(OPTIONS).write_parquet(tmp_path / "c.parquet")
    assert list(_documents(result)) == [right.definition_hash]
    wrong = _linear(AtStation("S1"), asset_id="a")
    with pytest.raises(SampleError, match="names the location"):
        plan.covers(wrong).metrics(OPTIONS).write_parquet(tmp_path / "d.parquet")
    # A cover written for no asset must fit every asset it applies to.
    both = _at_stations(_plan()).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="names the location"):
        both.covers(_linear(AtStation("S2"))).metrics(OPTIONS).write_parquet(
            tmp_path / "e.parquet"
        )
    grid = _nearest(_plan(assets=_assets("a"))).basis_risk(NoBasisRisk())
    grid.covers(_linear(AtPoint(6.75, 50.75))).metrics(OPTIONS).write_parquet(
        tmp_path / "f.parquet"
    )
    with pytest.raises(SampleError, match="names the location"):
        grid.covers(_linear(AtPoint(7.25, 50.75))).metrics(OPTIONS).write_parquet(
            tmp_path / "g.parquet"
        )


def test_a_cover_on_another_kind_of_location_is_refused_at_the_builder() -> None:
    stations = _at_stations(_plan())
    grid = _nearest(_plan())
    with pytest.raises(SampleError, match="AtStation"):
        stations.covers(_linear(AtPoint(6.75, 50.75)))
    with pytest.raises(SampleError, match="AtPoint"):
        grid.covers(_linear(AtStation("S2")))
    with pytest.raises(SampleError, match="AtStation"):
        stations.covers(_linear(AtCell("872a1072bffffff")))


def test_two_covers_that_bind_to_the_same_terms_raise(tmp_path: Path) -> None:
    plan = _at_stations(_plan(assets=_assets("a"))).basis_risk(NoBasisRisk())
    twins = plan.covers(_linear(), _linear(AtStation("S2"))).metrics(OPTIONS)
    with pytest.raises(ValueError, match="same terms once bound"):
        twins.write_parquet(tmp_path / "comparison.parquet")


def test_integer_station_ids_bind_as_text(tmp_path: Path) -> None:
    stations = pa.table(
        {
            "station": [725180] * len(YEARS),
            "year": pa.array(YEARS, pa.int32()),
            "value": np.linspace(30.0, 150.0, len(YEARS)),
        }
    )
    assets = _assets("a").set_column(4, "station_id", pa.array([725180]))
    plan = (
        _at_stations(_plan(assets=assets), stations)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    ((_, document),) = _documents(result).items()
    assert document["index"]["location"]["station_id"] == "725180"


# -- units and order -----------------------------------------------------------


def test_the_index_comes_first_when_the_units_differ() -> None:
    in_mm = Cover(
        IndexDefinition("rx1day", "mm", aggregation="annual_maximum"),
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
    )
    with pytest.raises(SampleError, match="the table's values are in"):
        _plan().covers(in_mm)
    plan = _at_stations(_plan(), unit="mm").covers(in_mm)
    assert plan.cover_list == (in_mm,)
    with pytest.raises(SampleError, match="the index series' values are in"):
        _at_stations(_plan(), unit="mm").covers(_linear())


def test_an_index_after_the_covers_checks_them_again() -> None:
    with pytest.raises(SampleError, match="the index series' values are in"):
        _at_stations(_plan().covers(_linear()), unit="mm")
    plan = _at_stations(_plan().covers(_linear()))
    assert len(plan.cover_list) == 1
    with_location = _at_stations(_plan()).covers(_linear(AtStation("S2")))
    with pytest.raises(SampleError, match="AtPoint"):
        _nearest(with_location)


# -- the years -----------------------------------------------------------------


def _gappy(*years: int) -> Any:
    """Station S2 and S1 without a row in `years`[0], and null in `years`[1]."""
    gone, null = years
    table = _stations()
    keep = pc.not_equal(table["year"], gone)
    table = table.filter(keep)
    values = [
        None if year == null else value
        for year, value in zip(
            table["year"].to_pylist(), table["value"].to_pylist(), strict=True
        )
    ]
    return table.set_column(2, "value", pa.array(values, pa.float64()))


def test_a_hazard_year_the_index_lacks_raises_and_is_listed() -> None:
    plan = _at_stations(_plan(), _gappy(1990, 2005)).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match=r"index \[1990, 2005\]"):
        plan.samples()


def test_a_hazard_year_the_index_lacks_is_dropped_and_listed() -> None:
    hazard, stations = _hazard(), _gappy(1990, 2005)
    plan = _at_stations(_plan(hazard, missing="drop_year"), stations).basis_risk(
        NoBasisRisk()
    )
    samples = plan.samples()
    for asset_id, sample in samples.items():
        assert 1990 not in sample.period and 2005 not in sample.period
        assert len(sample) == len(YEARS) - 2
        assert sample.provenance.details["dropped_years"] == (1990, 2005)
        assert any("dropped: 1990, 2005" in note for note in sample.provenance.notes)
        longitude, latitude = POINTS[asset_id]
        series = AnnualSeries.nearest_cell(
            hazard, longitude=longitude, latitude=latitude, unit="mm/day", source=HAZARD
        )
        expected = sample_annual(
            series,
            index=AnnualSeries.from_table(
                stations,
                where={"station": STATIONS[asset_id]},
                unit="mm/day",
                years=series.years.tolist(),
                source=GAUGES,
            ),
            loss_model=_loss(asset_id),
            basis_risk=NoBasisRisk(),
            asset_id=asset_id,
            missing="drop_year",
        )
        assert sample.content_hash == expected.content_hash


def test_a_station_record_shorter_than_the_hazard_is_a_missing_year() -> None:
    short = _stations(YEARS[10:])
    with pytest.raises(SampleError, match="1981"):
        _at_stations(_plan(), short).basis_risk(NoBasisRisk()).samples()
    samples = (
        _at_stations(_plan(missing="drop_year"), short)
        .basis_risk(NoBasisRisk())
        .samples()
    )
    assert samples["a"].provenance.details["dropped_years"] == tuple(
        YEARS[:10].tolist()
    )
    assert samples["a"].period.tolist() == YEARS[10:].tolist()


def test_no_hazard_year_with_an_index_value_raises() -> None:
    elsewhere = _stations(np.arange(1950, 1970))
    plan = _at_stations(_plan(missing="drop_year"), elsewhere).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="no year has a value"):
        plan.samples()


def test_an_index_year_the_hazard_does_not_have_is_listed_and_not_used() -> None:
    hazard, stations = _hazard(), _stations(np.arange(1975, 2021))
    samples = _at_stations(_plan(hazard), stations).basis_risk(NoBasisRisk()).samples()
    for asset_id, sample in samples.items():
        longitude, latitude = POINTS[asset_id]
        series = AnnualSeries.nearest_cell(
            hazard, longitude=longitude, latitude=latitude, unit="mm/day", source=HAZARD
        )
        expected = sample_annual(
            series,
            index=AnnualSeries.from_table(
                stations,
                where={"station": STATIONS[asset_id]},
                unit="mm/day",
                years=series.years.tolist(),
                source=GAUGES,
            ),
            loss_model=_loss(asset_id),
            basis_risk=NoBasisRisk(),
            asset_id=asset_id,
        )
        assert sample.content_hash == expected.content_hash
        assert sample.lineage == expected.lineage
        assert sample.period.tolist() == YEARS.tolist()
        assert sample.provenance.details["unused_index_years"] == tuple(
            range(1975, 1981)
        )
        assert (
            "6 index years outside the hazard's years are not used: "
            "1975, 1976, 1977, 1978, 1979, 1980."
        ) in sample.provenance.notes


def test_the_plans_years_window_the_hazard_and_so_the_index() -> None:
    window = tuple(range(1990, 2011))
    samples = _at_stations(_plan(years=window)).basis_risk(NoBasisRisk()).samples()
    sample = samples["a"]
    assert sample.period.tolist() == list(window)
    assert sample.provenance.details["unused_index_years"] == (
        *range(1981, 1990),
        *range(2011, 2021),
    )


def test_an_index_with_the_hazards_years_lists_none() -> None:
    sample = _at_stations(_plan()).basis_risk(NoBasisRisk()).samples()["a"]
    assert sample.provenance.details["unused_index_years"] == ()
    assert not any("not used" in note for note in sample.provenance.notes)


def test_a_single_unused_year_reads_in_the_singular() -> None:
    stations = _stations(np.arange(1980, 2021))
    sample = _at_stations(_plan(), stations).basis_risk(NoBasisRisk()).samples()["a"]
    assert (
        "1 index year outside the hazard's years is not used: 1980."
        in sample.provenance.notes
    )


# -- assets that cannot be read ------------------------------------------------


def test_an_asset_without_a_station_raises() -> None:
    assets = _assets("a").set_column(4, "station_id", pa.array([None], pa.string()))
    plan = _at_stations(_plan(assets=assets)).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="names no station"):
        plan.samples()


def test_a_station_the_index_table_does_not_hold_raises() -> None:
    assets = _assets("a").set_column(4, "station_id", pa.array(["S9"]))
    plan = _at_stations(_plan(assets=assets)).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="no rows"):
        plan.samples()


def test_an_asset_outside_the_index_grid_raises() -> None:
    grid = _grid(((10.0, 10.0), (10.5, 10.0), (10.0, 10.5)), seed=5)
    plan = _nearest(_plan(), grid).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="half a grid step"):
        plan.samples()


def test_an_asset_without_the_stated_point_raises() -> None:
    assets = _assets("a").set_column(5, "reference_lon", pa.array([None], pa.float64()))
    plan = _nearest(
        _plan(assets=assets), point=PointColumns("reference_lon", "reference_lat")
    ).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="no longitude or latitude"):
        plan.samples()


def test_h3_located_assets_read_the_index_at_their_cell() -> None:
    cell = point_to_cell(6.96, 50.94, 7)
    latitude, longitude = h3.cell_to_latlng(h3.int_to_str(cell))
    assets = AssetPortfolio(
        pa.table(
            {
                "asset_id": ["a"],
                "cell_index": pa.array([cell], pa.uint64()),
                "value_eur": [2e6],
            }
        ),
        location=CellColumn(),
    )
    grid = _index_grid()
    sample = (
        _nearest(_plan(assets=assets), grid).basis_risk(NoBasisRisk()).samples()["a"]
    )
    expected = AnnualSeries.nearest_cell(
        grid, longitude=longitude, latitude=latitude, unit="mm/day", source=GAUGES
    )
    np.testing.assert_array_equal(sample.index[:, 0], expected.values)
    assert sample.provenance.sources[1]["series"]["asset_location"] == "h3_cell"


# -- what is recorded ----------------------------------------------------------


def test_the_index_source_is_recorded_once(tmp_path: Path) -> None:
    stations = _stations()
    result = (
        _at_stations(_plan(), stations)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .split(KFold(2))
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    assert [set(record) for record in result.metadata.source_records] == [
        {"annual_table"},
        {"index_table"},
    ]
    index_record = result.metadata.source_records[1]["index_table"]
    assert index_record["kind"] == "table"
    assert index_record["rows"] == stations.num_rows
    assert index_record["columns"] == stations.column_names
    entries = [e for e in result.metadata.sources if e["role"] == "index"]
    assert entries
    for entry in entries:
        assert entry["index_table_sha256"] == index_record["table_sha256"]
        assert "annual_table_sha256" not in entry
    hazard = [e for e in result.metadata.sources if e["role"] == "hazard"]
    assert all("index_table_sha256" not in entry for entry in hazard)
    assert {
        "role": "index",
        "source": "synthetic gauges",
        "licence": "CC-BY-4.0",
        "attribution": "Synthetic gauge network",
    } in result.metadata.licences


def test_a_file_as_the_index_source(tmp_path: Path) -> None:
    source = tmp_path / "stations.parquet"
    pq.write_table(_stations(), source)
    result = (
        _at_stations(_plan(), source)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    record = result.metadata.source_records[1]["index_table"]
    assert record["path"] == str(source)
    assert record["file_sha256"].startswith("sha256:")
    entry = next(e for e in result.metadata.sources if e["role"] == "index")
    assert entry["series"]["source"]["file_sha256"] == record["file_sha256"]


def test_a_plan_as_the_index_source(tmp_path: Path) -> None:
    source = _SourcePlan(_stations(), pathway="ssp245")
    result = (
        _at_stations(_plan(), source)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    assert source.calls == ["annual_extremes", "explain"]
    record = result.metadata.source_records[1]["index_table"]
    assert record["explain"]["recipe"] == "rx1day"
    entry = next(e for e in result.metadata.sources if e["role"] == "index")
    assert entry["series"]["source"]["source_type"] == "_SourcePlan"
    assert entry["series"]["pathway"] == "ssp245"
    assert entry["series"]["source"]["dataset"] == "synthetic gauges"
    with pytest.raises(SampleError, match="index source's values are in"):
        _at_stations(_plan(), _SourcePlan(_stations(), unit="mm")).basis_risk(
            NoBasisRisk()
        ).samples()


def test_a_unit_column_of_the_index_table_is_checked() -> None:
    stations = _stations().append_column("unit", pa.array(["mm"] * 80))
    plan = _at_stations(_plan(), stations).basis_risk(NoBasisRisk())
    with pytest.raises(SampleError, match="unit"):
        plan.samples()


def _assert_points_absent(records: dict[str, str], points: tuple[float, ...]) -> None:
    for name, record in records.items():
        assert "distance_km" not in record, name
        for value in points:
            assert str(value) not in record, name


@pytest.mark.parametrize("reference", [False, True])
def test_no_record_holds_the_assets_point_or_the_reference_point(
    reference: bool, tmp_path: Path
) -> None:
    point, stated = (6.9613, 50.9427), (7.2013, 50.8127)
    assets = pa.table(
        {
            "asset_id": ["a"],
            "longitude": [point[0]],
            "latitude": [point[1]],
            "value_eur": [2e6],
            "reference_lon": [stated[0]],
            "reference_lat": [stated[1]],
        }
    )
    columns = PointColumns("reference_lon", "reference_lat") if reference else None
    plan = (
        _nearest(_plan(assets=assets), point=columns)
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_linear())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    planned: Any = plan.explain(format="json")
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    schema = pq.read_schema(result.output).metadata
    records = {
        "provenance": json.dumps(result.samples["a"].provenance.to_document()),
        "metadata": result.metadata.to_json(),
        "file": schema[EVALUATION_METADATA_KEY.encode("utf-8")].decode("utf-8"),
        "setup text": setup_text(result.metadata),
        "setup text of the rows": setup_text(result.table),
        "explain": json.dumps(planned),
        "setup text of the plan": setup_text(planned),
    }
    _assert_points_absent(records, (*point, *stated))
    assert '"cell"' in records["metadata"]


# -- explain and the setup text ------------------------------------------------


def test_explain_shows_the_covers_as_stated_and_the_location_rule() -> None:
    plan = (
        _at_stations(_plan())
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    planned: Any = plan.explain(format="json")
    assert planned["covers"][0]["document"]["index"]["location"] == {"kind": "asset"}
    hazard, index = planned["sources"]
    assert hazard["role"] == "hazard"
    assert index["role"] == "index"
    assert index["location"] == {
        "kind": "station",
        "asset_column": "station_id",
        "table_column": "station",
    }
    assert index["point"] is None
    assert index["unit"] == "mm/day"
    assert index["source_kind"] == "table"
    assert index["series"]["source"]["dataset"] == "synthetic gauges"
    assert [entry["role"] for entry in planned["licences"]] == ["hazard", "index"]
    assert all(s["index_unit"] == "mm/day" for s in planned["setup"]["samples"])


def test_explain_states_another_point_by_its_columns_only() -> None:
    plan = _nearest(
        _plan(), point=PointColumns("reference_lon", "reference_lat")
    ).basis_risk(NoBasisRisk())
    planned: Any = plan.explain(format="json")
    assert planned["sources"][1]["point"] == {
        "latitude_column": "reference_lat",
        "longitude_column": "reference_lon",
    }
    assert any(
        "nearest the point in the asset columns reference_lon and reference_lat" in note
        for note in planned["setup"]["samples"][0]["notes"]
    )


def test_a_plan_without_an_index_has_no_index_source() -> None:
    planned: Any = _plan().basis_risk(NoBasisRisk()).explain(format="json")
    assert [entry.get("role") for entry in planned["sources"]] == ["hazard"]
    assert [entry["role"] for entry in planned["licences"]] == ["hazard"]


def test_the_setup_text_states_where_the_index_is_read(tmp_path: Path) -> None:
    plan = (
        _at_stations(_plan(assets=_assets("a")))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    before = setup_text(plan.explain(format="json"))
    assert (
        "the index series from provider example and dataset synthetic gauges" in before
    )
    assert (
        "The index is read elsewhere: the station the asset column station_id names."
        in before
    )
    after = setup_text(plan.write_parquet(tmp_path / "c.parquet").metadata)
    assert (
        "the index series from provider example and dataset synthetic gauges" in after
    )
    assert "The index is read from another series than the loss." in after
    assert (
        "The index of asset a is read at station S2: the station the asset column "
        "station_id names."
    ) in after


def test_the_setup_text_states_the_matched_cell(tmp_path: Path) -> None:
    plan = (
        _nearest(_plan(assets=_assets("a")))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
    )
    text = setup_text(plan.write_parquet(tmp_path / "c.parquet").metadata)
    assert (
        "The index of asset a is read at the grid cell centred at latitude 50.75, "
        "longitude 6.75: the grid cell of the index source nearest the asset."
    ) in text


# -- splits, end to end --------------------------------------------------------


def test_out_of_sample_rows_match_compare_on_the_bound_covers(tmp_path: Path) -> None:
    plan = (
        _at_stations(_plan(assets=_assets("a")))
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .split(Holdout(0.7))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    train, test = holdout(result.samples["a"], 0.7)
    bound = _linear(AtStation("S2"), asset_id="a")
    expected = compare(
        [bound],
        test,
        split="out_of_sample",
        options=OPTIONS,
        design_sample=train,
    )
    rows = result.table.filter(pc.equal(result.table["split"], "out_of_sample"))
    assert rows.column("cover_hash").to_pylist() == [bound.definition_hash]
    assert rows.column("sample_hash").to_pylist() == [test.content_hash]
    assert rows.column("premium").to_pylist() == expected.column("premium").to_pylist()
    assert (
        rows.column("expected_uncovered").to_pylist()
        == expected.column("expected_uncovered").to_pylist()
    )


def test_designs_on_templates_pay_on_the_index_series(tmp_path: Path) -> None:
    plan = (
        _at_stations(_plan(assets=_assets("a")))
        .basis_risk(NoBasisRisk())
        .covers(_step())
        .objective(MSE())
        .split(KFold(2))
        .metrics(OPTIONS)
    )
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    assert len(result.designs) == 2
    sample = result.samples["a"]
    assert sample.hazard is not None
    assert not np.array_equal(sample.index, sample.hazard)
    documents = _documents(result)
    assert all(
        document["index"]["location"] == {"kind": "station", "station_id": "S2"}
        for document in documents.values()
    )


def test_setup_text_states_where_the_hazard_and_the_index_were_matched(
    tmp_path: Path,
) -> None:
    result = (
        _nearest(_plan())
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .write_parquet(tmp_path / "comparison.parquet")
    )
    text = setup_text(result.metadata)
    assert [line for line in text.splitlines() if line.startswith("Location")] == [
        "Location: hazard series, located by a point, in the nearest grid cell (grid "
        "step 0.25 by 0.25 degrees): a at (50.875, 6.875) and b at (51.125, 7.125), "
        "the cell centres as (latitude, longitude).",
        "Location: index series, located by a point, in the nearest grid cell (grid "
        "step 0.5 by 0.5 degrees): a at (50.75, 6.75) and b at (51.25, 7.25), the "
        "cell centres as (latitude, longitude).",
    ]
