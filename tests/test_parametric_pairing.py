"""A cover is read only on a sample of its own hazard, on every route.

Every number here is synthetic: small made-up tables, series and curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    Budget,
    CompositeIndex,
    Cover,
    CoverPackage,
    ExpectedUncovered,
    Free,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LinearPayout,
    LossModel,
    ModelRun,
    OutcomeSample,
    SampleError,
    ScenarioSet,
    StepPayout,
    compare,
    compare_scenarios,
    design,
    design_joint,
    evaluate,
    holdout,
    payout_breakdown,
    sample_annual,
    sample_annual_pool,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric._pairing import _check_reads
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.distributions.ReturnPeriodExtrapolationWarning"
    ),
]

UNTYPED_TABLE: Any = OutcomeSample.from_table

RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
CELL = 613196570331971583
YEARS = list(range(2000, 2030))


def _index(name: str, unit: str = "mm/day") -> IndexDefinition:
    return IndexDefinition(name, unit)


def _cover(name: str, **options: Any) -> Cover:
    return Cover(
        _index(name), LinearPayout(60.0, 120.0, 1e5), payout_unit="EUR", **options
    )


def _template(name: str, **options: Any) -> Cover:
    return Cover(
        _index(name), StepPayout([60.0], [Free()]), payout_unit="EUR", **options
    )


def _table(
    hazard_name: str | None = "rx1day",
    *,
    asset: str = "a",
    scenario: str = "baseline",
    shift: float = 0.0,
    pathway: str | None = None,
) -> OutcomeSample:
    rng = np.random.default_rng(5)
    index = rng.gumbel(60.0, 15.0, len(YEARS)) + shift
    table = pa.table(
        {
            "year": YEARS,
            "asset": [asset] * len(YEARS),
            "index": index,
            "loss": np.maximum(index - 50.0, 0.0) * 1e4,
        }
    )
    sample: OutcomeSample = UNTYPED_TABLE(
        table,
        period="year",
        asset_id="asset",
        index="index",
        loss="loss",
        index_unit="mm/day",
        loss_unit="EUR",
        hazard_name=hazard_name,
        scenario=scenario,
        source=None if pathway is None else {"provider": "fixture", "pathway": pathway},
    )
    return sample


def _dataset(unit: str = "mm/day") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=8,
        value_unit=unit,
        value_semantics="annual maximum",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="synthetic",
            licence="CC0-1.0",
            attribution="Synthetic fixture",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 200.0),
    )


def _row(hazard_name: str, location: float = 60.0) -> dict[str, Any]:
    return {
        "cell_index": CELL,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": hazard_name,
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": location,
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def _book(names: dict[str, str]) -> OutcomeSample:
    """A sample of several assets from curves, each asset of its named hazard."""
    curves = {asset: _row(name) for asset, name in names.items()}
    sample: OutcomeSample = sample_curves(
        curves,
        dependence=Independent(),
        metadata=dict.fromkeys(curves, _dataset()),
        n_draws=200,
        rngs={
            asset: np.random.default_rng(seed) for seed, asset in enumerate(curves, 1)
        },
        loss_models=dict.fromkeys(curves, LOSS),
        basis_risk=ImpactNoise(0.2),
    )
    return sample


def _hazards() -> OutcomeSample:
    """One asset, two components named after the hazards their rows record."""
    curves = {"rx1day": _row("rx1day"), "txx": _row("txx", 35.0)}
    sample: OutcomeSample = sample_curve_hazards(
        curves,
        dependence=Independent(),
        metadata={"rx1day": _dataset(), "txx": _dataset("degC")},
        n_draws=200,
        rngs={"rx1day": np.random.default_rng(1), "txx": np.random.default_rng(2)},
        loss_models={"rx1day": LOSS},
        basis_risk=ImpactNoise(0.2),
        asset_id="a",
    )
    return sample


def _series(source: Any, *, seed: int = 3) -> AnnualSeries:
    values = np.random.default_rng(seed).gumbel(60.0, 12.0, len(YEARS))
    return AnnualSeries(
        np.array(YEARS), values, "mm/day", {"kind": "table", "source": source}
    )


def _pool(*sources: Any, asset: str = "a") -> OutcomeSample:
    """A pool of one run per source, models M1, M2, ..., for `asset`."""
    runs = [
        ModelRun(
            _series(source, seed=seed),
            model=f"M{seed}",
            member="r1i1p1f1",
            pathway="ssp245",
        )
        for seed, source in enumerate(sources, start=1)
    ]
    sample: OutcomeSample = sample_annual_pool(
        runs,
        loss_model=LOSS,
        basis_risk=ImpactNoise(0.2),
        asset_id=asset,
        rng=np.random.default_rng(0),
    )
    return sample


# -- the one check -----------------------------------------------------------------


def test_a_cover_on_another_hazard_is_refused_naming_both() -> None:
    with pytest.raises(SampleError, match="'wind'.*'rx1day'.*hazard_name="):
        _check_reads(_index("wind"), _table("rx1day"))


def test_a_cover_on_the_recorded_hazard_passes() -> None:
    _check_reads(_index("rx1day"), _table("rx1day"))


def test_a_sample_that_records_no_hazard_is_read_as_before() -> None:
    # Nothing names the hazard, so only the units are compared.
    sample = _table(None)
    assert sample.provenance.hazard is None
    _check_reads(_index("wind"), sample)
    with pytest.raises(SampleError, match="'degC'.*'mm/day'"):
        _check_reads(_index("wind", "degC"), sample)


def test_the_name_is_compared_exactly() -> None:
    with pytest.raises(SampleError, match="'Rx1day'.*'rx1day'"):
        _check_reads(_index("Rx1day"), _table("rx1day"))


def test_a_curve_row_names_the_hazard() -> None:
    sample = _book({"a": "rx1day", "b": "rx1day"}).asset("a")
    with pytest.raises(SampleError, match="'wind'.*'rx1day'.*canonical row"):
        _check_reads(_index("wind"), sample)


# -- the routes --------------------------------------------------------------------


def test_evaluate_compare_and_the_breakdown_refuse_another_hazard() -> None:
    sample = _table("rx1day")
    wind = _cover("wind")
    with pytest.raises(SampleError, match="'wind'.*'rx1day'"):
        evaluate(wind, sample, split="in_sample")
    with pytest.raises(SampleError, match="'wind'.*'rx1day'"):
        compare([_cover("rx1day"), wind], sample, split="in_sample")
    with pytest.raises(SampleError, match="'wind'.*'rx1day'"):
        payout_breakdown([wind], sample, split="in_sample")
    assert evaluate(_cover("rx1day"), sample, split="in_sample").num_rows == 1


def test_design_refuses_another_hazard_before_any_solve() -> None:
    sample = _table("rx1day")
    with pytest.raises(SampleError, match="'wind'.*'rx1day'"):
        design(_template("wind"), sample, ExpectedUncovered(), budget=Budget(ratio=1))
    designed = design(
        _template("rx1day"), sample, ExpectedUncovered(), budget=Budget(ratio=1.0)
    )
    assert designed.cover.index.hazard_name == "rx1day"


def test_a_design_is_refused_on_a_held_out_block_of_another_hazard() -> None:
    train, _ = holdout(_table("rx1day"), 0.7)
    _, other = holdout(_table("rx5day", shift=1.0), 0.7)
    designed = design(_template("rx1day"), train, MSE())
    with pytest.raises(SampleError, match="'rx1day'.*'rx5day'"):
        evaluate(designed, other, split="out_of_sample")


def test_each_asset_of_a_book_is_compared_with_its_own_record() -> None:
    book = _book({"a": "rx1day", "b": "wind"})
    # The assets record two hazards, so the sample records none of its own.
    assert book.provenance.hazard is None
    _check_reads(_index("rx1day"), book, "a")
    _check_reads(_index("wind"), book, "b")
    with pytest.raises(SampleError, match="'rx1day'.*asset 'b' reads 'wind'"):
        _check_reads(_index("rx1day"), book, "b")
    with pytest.raises(SampleError, match="asset 'b'"):
        _check_reads(_index("rx1day"), book)


def test_a_set_of_covers_and_a_joint_design_check_each_asset() -> None:
    book = _book({"a": "rx1day", "b": "rx1day"})
    covers = {"a": _cover("rx1day"), "b": _cover("wind")}
    with pytest.raises(SampleError, match="'wind'.*asset 'b' reads 'rx1day'"):
        evaluate(covers, book, split="in_sample")
    templates = [_template("rx1day"), _template("wind")]
    with pytest.raises(SampleError, match="'wind'.*asset 'b' reads 'rx1day'"):
        design_joint(templates, book, MSE())
    result = design_joint([_template("rx1day"), _template("rx1day")], book, MSE())
    assert evaluate(result, book, split="in_sample").num_rows == 1


def test_an_index_series_is_compared_with_its_own_source() -> None:
    hazard = _series({"provider": "fixture", "recipe": "rx1day"})
    index = _series({"provider": "fixture", "recipe": "wind"}, seed=4)
    sample = sample_annual(
        hazard,
        index=index,
        loss_model=LOSS,
        basis_risk=ImpactNoise(0.2),
        asset_id="a",
        rng=np.random.default_rng(0),
    )
    # The loss reads rx1day and the index wind, on purpose: the cover is wind's.
    assert sample.provenance.hazard == {"name": "rx1day", "from": "source_plan"}
    _check_reads(_index("wind"), sample)
    with pytest.raises(SampleError, match="'rx1day'.*index series.*'wind'"):
        evaluate(_cover("rx1day"), sample, split="in_sample")


def test_an_index_series_that_names_nothing_is_read_as_before() -> None:
    hazard = _series({"provider": "fixture", "recipe": "rx1day"})
    index = _series({"provider": "fixture"}, seed=4)
    sample = sample_annual(
        hazard,
        index=index,
        loss_model=LOSS,
        basis_risk=ImpactNoise(0.2),
        asset_id="a",
        rng=np.random.default_rng(0),
    )
    _check_reads(_index("anything"), sample)


def test_an_index_series_named_by_its_recipe_column() -> None:
    table = pa.table(
        {
            "year": YEARS,
            "value": np.linspace(30.0, 90.0, len(YEARS)),
            "recipe": ["wind"] * len(YEARS),
        }
    )
    index = AnnualSeries.from_table(table, unit="mm/day", source="gauges")
    sample = sample_annual(
        _series({"provider": "fixture", "recipe": "rx1day"}),
        index=index,
        loss_model=LOSS,
        basis_risk=ImpactNoise(0.2),
        asset_id="a",
        rng=np.random.default_rng(0),
    )
    with pytest.raises(SampleError, match="'rx1day'.*index series.*'wind'"):
        _check_reads(_index("rx1day"), sample)


def test_each_component_is_compared_with_its_own_record() -> None:
    sample = _hazards()
    rain, heat = _index("rx1day"), _index("txx", "degC")
    _check_reads(CompositeIndex({"rx1day": rain, "txx": heat}), sample)
    # The component is read by its key; its record names another hazard.
    renamed = CompositeIndex({"rx1day": _index("rain"), "txx": heat})
    with pytest.raises(SampleError, match="'rain'.*component 'rx1day'.*'rx1day'"):
        _check_reads(renamed, sample)
    # A component's view keeps the component's record.
    view = sample.component("rx1day")
    with pytest.raises(SampleError, match="'rain'.*'rx1day'"):
        _check_reads(_index("rain"), view)


def test_a_package_reads_each_component_by_its_key() -> None:
    sample = _hazards()
    package = CoverPackage(
        {
            "rx1day": _cover("rx1day"),
            "txx": Cover(
                _index("heat", "degC"),
                LinearPayout(36.0, 42.0, 1e5),
                payout_unit="EUR",
            ),
        }
    )
    with pytest.raises(SampleError, match="'heat'.*component 'txx'.*'txx'"):
        evaluate(package, sample, split="in_sample")


def test_compare_scenarios_refuses_a_design_block_of_another_hazard() -> None:
    blocks = [
        _table("rx1day", scenario="a", pathway="ssp245"),
        _table("rx1day", scenario="b", shift=5.0, pathway="ssp585"),
    ]
    scenarios = ScenarioSet([holdout(block, 0.5)[1] for block in blocks], reference="a")
    elsewhere, _ = holdout(
        _table("rx5day", scenario="a", shift=1.0, pathway="ssp245"), 0.5
    )
    # A cover given as is is priced on the reference's design block.
    with pytest.raises(SampleError, match="'rx1day'.*'rx5day'"):
        compare_scenarios([_cover("rx1day")], scenarios, design_sample=elsewhere)
    train, _ = holdout(blocks[0], 0.5)
    table = compare_scenarios([_cover("rx1day")], scenarios, design_sample=train)
    assert table.num_rows == 2


# -- pools ---------------------------------------------------------------------------


def test_a_pool_whose_runs_name_its_hazard_records_and_compares_it() -> None:
    named = {"provider": "fixture", "recipe": "rx1day"}
    sample = _pool(named, named)
    assert sample.provenance.hazard == {"name": "rx1day", "from": "source_plan"}
    _check_reads(_index("rx1day"), sample)
    with pytest.raises(SampleError, match="'wind'.*the sample reads 'rx1day'"):
        _check_reads(_index("wind"), sample)


def test_a_pool_with_a_silent_run_compares_the_runs_that_name_one() -> None:
    # A run that names none leaves the pool without a record of its own.
    sample = _pool({"provider": "fixture", "hazard_name": "rx1day"}, "a table")
    assert sample.provenance.hazard is None
    _check_reads(_index("rx1day"), sample)
    with pytest.raises(
        SampleError, match="'wind'.*run M1 r1i1p1f1 of the sample reads 'rx1day'"
    ):
        _check_reads(_index("wind"), sample)
    with pytest.raises(SampleError, match="'wind'.*run M1"):
        evaluate(_cover("wind"), sample, split="in_sample")
    with pytest.raises(SampleError, match="'wind'.*run M1"):
        design(_template("wind"), sample, MSE())


def test_a_pool_whose_runs_name_nothing_is_read_as_before() -> None:
    sample = _pool("a table", "another table")
    assert sample.provenance.hazard is None
    _check_reads(_index("anything"), sample)


def test_each_asset_of_joined_pools_compares_its_own_runs() -> None:
    joined = OutcomeSample.join(
        [
            _pool({"provider": "fixture", "recipe": "rx1day"}, "a table", asset="a"),
            _pool("a table", "another table", asset="b"),
        ]
    )
    _check_reads(_index("wind"), joined, "b")
    with pytest.raises(SampleError, match="'wind'.*run M1 r1i1p1f1 of asset 'a'"):
        _check_reads(_index("wind"), joined)
