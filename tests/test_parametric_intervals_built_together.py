"""The `built_together` flag of `bootstrap_intervals`: which rows it marks.

Resampling rows treats them as independent draws. An `EmpiricalCopula` deals
each column's draws to its reference years by rank, and `HazardNoise(...,
hold="hazard_distribution")` ranks each row's hazard at the asset over every
row, so their rows are not. A synthetic coverage study found intervals
miscalibrated on the rows of an `EmpiricalCopula` block that read several
columns, and on every row of a held `HazardNoise` block, and calibrated on an
asset's own row of a copula block and on independent rows. These tests check
that the flag marks exactly those rows, by how the block was built, and that
the setup text says so. Every number is synthetic.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    CappedSum,
    ComonotoneIndexWarning,
    Comonotonic,
    Cover,
    CoverPackage,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    Free,
    GaussianCopula,
    HazardNoise,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LinearPayout,
    LossModel,
    MetricOptions,
    NoBasisRisk,
    OutcomeSample,
    StepPayout,
    bootstrap_intervals,
    design_joint,
    sample_annual,
    sample_annual_hazards,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import EVALUATION_METADATA_KEY
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
HAZARDS = ("rain", "heat")
UNITS = {"rain": "mm/day", "heat": "degC"}
LOCATIONS = {"site-a": 50.0, "site-b": 60.0, "rain": 50.0, "heat": 60.0}
RAMP = PiecewiseLinearImpact([30.0, 120.0], [0.0, 1.0])
M = 10
K = 2
N = K * M
FEW = MetricOptions(betas=(0.9,), return_periods=(5.0,))
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
HELD = HazardNoise(0.3, hold="hazard_distribution")
YEARS = np.arange(1991, 1991 + 24)

COPULA_TEXT = "an EmpiricalCopula dealt each asset's or hazard's draws"
HELD_TEXT = "HazardNoise(..., hold='hazard_distribution') gave each row's hazard"
SENTENCE = "The rows with built_together true read draws that were built together"


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
    index: Any = INDEX,
    label: str = "c",
    limit: float = 60.0,
) -> Cover:
    return Cover(
        index,
        LinearPayout(55.0, 90.0, limit),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _covers() -> dict[str, Cover]:
    return {asset: _cover(asset, label=asset) for asset in ASSETS}


def _assets_reference() -> OutcomeSample:
    """M years of the two assets, whose columns move together."""
    rng = np.random.default_rng(5)
    common = rng.gamma(2.0, 20.0, M)
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        for asset in ASSETS:
            values = common + rng.gamma(1.0, 5.0, M)
            parts.append(
                sample_annual(
                    AnnualSeries(
                        np.arange(1990, 1990 + M),
                        values,
                        "mm/day",
                        {"source": {"name": f"series of {asset}"}},
                    ),
                    loss_model=_loss(asset),
                    basis_risk=NoBasisRisk(),
                    asset_id=asset,
                )
            )
    return OutcomeSample.join(parts)


def _hazard_reference() -> OutcomeSample:
    rng = np.random.default_rng(6)
    series = {
        name: AnnualSeries(
            np.arange(1990, 1990 + M),
            rng.gamma(2.0, 20.0, M),
            UNITS[name],
            {"source": {"name": f"series of {name}"}},
        )
        for name in HAZARDS
    }
    return sample_annual_hazards(
        series,
        tails=dict.fromkeys(HAZARDS, "upper"),
        loss_models={"rain": _loss("rain")},
        basis_risk=NoBasisRisk(),
        asset_id="site",
    )


def _assets(dependence: Any, basis_risk: Any = None, seed: int = 11) -> OutcomeSample:
    return sample_curves(
        {asset: _curve(asset) for asset in ASSETS},
        dependence=dependence,
        metadata={asset: _metadata() for asset in ASSETS},
        n_draws=N,
        rngs={asset: np.random.default_rng(seed + j) for j, asset in enumerate(ASSETS)},
        loss_models={asset: _loss(asset) for asset in ASSETS},
        basis_risk=ImpactNoise(0.2, upper=1.0) if basis_risk is None else basis_risk,
    )


def _copula() -> EmpiricalCopula:
    return EmpiricalCopula(_assets_reference())


def _hazards(
    impacted: tuple[str, ...] = HAZARDS, dependence: Any = None
) -> OutcomeSample:
    return sample_curve_hazards(
        {name: _curve(name) for name in HAZARDS},
        dependence=(
            EmpiricalCopula(_hazard_reference()) if dependence is None else dependence
        ),
        metadata={name: _metadata(UNITS[name]) for name in HAZARDS},
        n_draws=N,
        rngs={name: np.random.default_rng(21 + j) for j, name in enumerate(HAZARDS)},
        loss_models={name: _loss(name) for name in impacted},
        combine=CappedSum(100.0) if len(impacted) > 1 else None,
        basis_risk=ImpactNoise(0.2, upper=1.0),
        asset_id="site",
    )


def _series(seed: int, name: str) -> AnnualSeries:
    values = np.random.default_rng(seed).gumbel(50.0, 10.0, YEARS.size)
    return AnnualSeries(YEARS, values, "mm/day", {"source": {"name": name}})


def _annual(asset: str, basis_risk: Any, seed: int = 1) -> OutcomeSample:
    return sample_annual(
        _series(seed, f"series of {asset}"),
        loss_model=_loss(asset),
        basis_risk=basis_risk,
        asset_id=asset,
        rng=np.random.default_rng(seed + 100),
    )


def _intervals(entries: list[Any], sample: OutcomeSample, **arguments: Any) -> Any:
    return UNTYPED_INTERVALS(
        entries,
        sample,
        split="in_sample",
        rng=np.random.default_rng(3),
        n_bootstrap=4,
        options=FEW,
        **arguments,
    )


def _flags(table: Any) -> list[tuple[Any, Any, bool]]:
    """Each wide row's (level, asset_id or cover label, flag); one flag per row."""
    by_row: dict[tuple[Any, Any, Any], set[bool]] = {}
    for row in table.to_pylist():
        key = (row["level"], row["asset_id"], row["cover_label"])
        by_row.setdefault(key, set()).add(row["built_together"])
    for key, flags in by_row.items():
        assert len(flags) == 1, f"{key} has different flags on different columns"
    return [(*key[:2], flags.pop()) for key, flags in by_row.items()]


def _record(table: Any) -> list[str]:
    intervals = EvaluationMetadata.from_table(table).setup.intervals
    assert intervals is not None
    kinds: list[str] = intervals["built_together"]
    return kinds


def _intervals_line(table: Any) -> str:
    return next(
        line for line in setup_text(table).splitlines() if line.startswith("Intervals:")
    )


# -- an EmpiricalCopula across assets ----------------------------------------------


def test_a_copula_flags_the_total_and_not_an_assets_own_row() -> None:
    table = _intervals([_covers()], _assets(_copula()))
    flags = _flags(table)
    assert flags == [
        ("asset", "site-a", False),
        ("asset", "site-b", False),
        ("total", None, True),
    ]
    assert _record(table) == ["empirical_copula"]


def test_a_copula_flags_every_row_of_a_joint_design() -> None:
    sample = _assets(_copula())
    steps = [
        Cover(INDEX, StepPayout([60.0], [Free()]), payout_unit="EUR", asset_id=asset)
        for asset in ASSETS
    ]
    joint = design_joint(steps, sample, MSE())
    table = _intervals(
        [joint], sample, allocation="loss", values={"site-a": 2.0, "site-b": 1.0}
    )
    levels = {(level, asset): flag for level, asset, flag in _flags(table)}
    assert levels == {
        ("total", None): True,
        ("asset_by_loss", "site-a"): True,
        ("asset_by_loss", "site-b"): True,
    }


@pytest.mark.parametrize(
    "dependence",
    [Independent(), Comonotonic(), GaussianCopula(np.array([[1.0, 0.8], [0.8, 1.0]]))],
    ids=["independent", "comonotonic", "gaussian"],
)
def test_the_other_dependence_models_are_not_flagged(dependence: Any) -> None:
    # The flag marks what the study measured: independent rows were calibrated,
    # and these two models were not part of it.
    table = _intervals([_covers()], _assets(dependence))
    assert not any(table.column("built_together").to_pylist())
    assert _record(table) == []


def test_an_assets_own_view_of_a_copula_sample_is_not_flagged() -> None:
    view = _assets(_copula()).asset("site-b")
    table = _intervals([_cover("site-b")], view)
    assert EvaluationMetadata.from_table(table).setup.intervals["unit"] == "row"  # type: ignore[index]
    assert not any(table.column("built_together").to_pylist())


def test_a_written_copula_sample_keeps_its_flags() -> None:
    read = OutcomeSample.from_written_table(_assets(_copula()).to_table())
    table = _intervals([_covers()], read)
    assert [flag for *_, flag in _flags(table)] == [False, False, True]


# -- an EmpiricalCopula across hazards ---------------------------------------------


def _package() -> CoverPackage:
    return CoverPackage(
        {"rain": _cover(index=RAIN, label="r"), "heat": _cover(index=HEAT, label="h")},
        label="package",
    )


def test_with_two_impacts_every_row_reads_several_hazards() -> None:
    entries = [
        _package(),
        _cover(index=RAIN, label="on rain", limit=70.0),
        _cover(index=HEAT, label="on heat", limit=70.0),
    ]
    table = _intervals(entries, _hazards())
    assert set(table.column("built_together").to_pylist()) == {True}
    assert _record(table) == ["empirical_copula"]


def test_with_one_impact_a_row_on_that_hazard_alone_is_not_flagged() -> None:
    sample = _hazards(impacted=("rain",))
    table = _intervals(
        [
            _package(),
            _cover(index=RAIN, label="on rain", limit=70.0),
            _cover(index=HEAT, label="on heat", limit=70.0),
        ],
        sample,
    )
    flags = {
        (row["level"], row["cover_label"]): row["built_together"]
        for row in table.to_pylist()
    }
    assert flags == {
        ("asset", "r"): False,
        ("asset", "h"): True,
        ("package", "package"): True,
        ("asset", "on rain"): False,
        ("asset", "on heat"): True,
    }


@pytest.mark.parametrize(
    ("impacted", "name", "loss", "flagged"),
    [
        (HAZARDS, "rain", "own", False),
        (HAZARDS, "heat", "own", False),
        (HAZARDS, "rain", "combined", True),
        (("rain",), "heat", "combined", True),
        (("rain",), "rain", "combined", False),
    ],
)
def test_a_hazards_view_is_flagged_when_its_loss_reads_another_hazard(
    impacted: tuple[str, ...], name: str, loss: Any, flagged: bool
) -> None:
    view = _hazards(impacted).component(name, loss=loss)
    index = RAIN if name == "rain" else HEAT
    table = _intervals([_cover(index=index)], view)
    assert set(table.column("built_together").to_pylist()) == {flagged}


def test_the_other_models_across_hazards_are_not_flagged() -> None:
    table = _intervals([_package()], _hazards(dependence=Independent()))
    assert not any(table.column("built_together").to_pylist())


# -- the held form of HazardNoise ----------------------------------------------------


def _curve_sample(basis_risk: Any) -> OutcomeSample:
    return sample_curve(
        _curve("site-a"),
        metadata=_metadata(),
        n_draws=40,
        rng=np.random.default_rng(4),
        loss_model=_loss("site-a"),
        basis_risk=basis_risk,
        asset_id="a",
    )


def test_every_row_of_a_held_block_is_flagged() -> None:
    table = _intervals([_cover()], _curve_sample(HELD))
    assert set(table.column("built_together").to_pylist()) == {True}
    assert _record(table) == ["hazard_distribution"]
    assert EvaluationMetadata.from_table(table).setup.intervals["unit"] == "year"  # type: ignore[index]


def test_the_default_hold_is_not_flagged() -> None:
    table = _intervals([_cover()], _curve_sample(HazardNoise(0.3)))
    assert not any(table.column("built_together").to_pylist())
    assert _record(table) == []


def test_every_row_of_a_held_block_of_several_assets_is_flagged() -> None:
    table = _intervals([_covers()], _assets(Independent(), HELD))
    assert [flag for *_, flag in _flags(table)] == [True, True, True]


def test_a_held_block_of_per_year_values_is_flagged() -> None:
    table = _intervals([_cover("a")], _annual("a", HELD))
    assert set(table.column("built_together").to_pylist()) == {True}


def test_a_join_flags_the_rows_that_read_its_held_part() -> None:
    joined = OutcomeSample.join(
        [_annual("site-a", HELD, 1), _annual("site-b", HazardNoise(0.3), 2)]
    )
    table = _intervals([_covers()], joined)
    assert _flags(table) == [
        ("asset", "site-a", True),
        ("asset", "site-b", False),
        ("total", None, True),
    ]


def test_a_held_copula_block_records_both_kinds() -> None:
    table = _intervals([_covers()], _assets(_copula(), HELD))
    assert [flag for *_, flag in _flags(table)] == [True, True, True]
    assert _record(table) == ["empirical_copula", "hazard_distribution"]
    line = _intervals_line(table)
    assert COPULA_TEXT in line
    assert HELD_TEXT in line


def test_pooled_folds_of_held_blocks_are_flagged() -> None:
    # year_block_kfold refuses to split a held sample, and the leakage check
    # refuses two cuts of one, so these folds are made by hand: the blocks of a
    # plain sample, each recording the held model and no parent. A pooled row
    # reads every fold's rows, so one fold's rows built together flag it.
    def held(sample: OutcomeSample) -> OutcomeSample:
        provenance = dataclasses.replace(
            sample.provenance, basis_risk=HELD.to_document(), parent=None
        )
        return dataclasses.replace(sample, provenance=provenance)

    folds = year_block_kfold(_annual("a", HazardNoise(0.3)), 3)
    plain = UNTYPED_INTERVALS(
        [_cover("a")], folds, rng=np.random.default_rng(1), n_bootstrap=3
    )
    assert not any(plain.column("built_together").to_pylist())
    marked = tuple((held(train), held(test)) for train, test in folds)
    table = UNTYPED_INTERVALS(
        [_cover("a")], marked, rng=np.random.default_rng(1), n_bootstrap=3
    )
    assert set(table.column("built_together").to_pylist()) == {True}


# -- the table, the metadata and the setup text ----------------------------------------


def test_the_flag_follows_short_block_and_changes_no_number() -> None:
    sample = _assets(_copula())
    table = _intervals([_covers()], sample)
    assert table.column_names[-2:] == ["short_block", "built_together"]
    assert table.schema.field("built_together").type == pa.bool_()
    again = _intervals([_covers()], sample)
    assert table.drop_columns(["built_together"]).equals(
        again.drop_columns(["built_together"])
    )


def test_the_setup_text_states_the_rows_built_together() -> None:
    line = _intervals_line(_intervals([_covers()], _assets(_copula())))
    assert SENTENCE in line
    assert COPULA_TEXT in line
    assert HELD_TEXT not in line
    assert "is not the sampling error of the number" in line
    assert '"Rows built together"' in line
    held = _intervals_line(_intervals([_cover()], _curve_sample(HELD)))
    assert HELD_TEXT in held
    assert COPULA_TEXT not in held


def test_the_setup_text_says_nothing_without_a_flagged_row() -> None:
    table = _intervals([_covers()], _assets(Independent()))
    assert "built_together" not in setup_text(table)


def test_rows_filtered_to_unflagged_ones_drop_the_sentence() -> None:
    table = _intervals([_covers()], _assets(_copula()))
    assert SENTENCE in _intervals_line(table)
    kept = table.filter(
        pa.array([row["level"] == "asset" for row in table.to_pylist()])
    )
    assert SENTENCE not in _intervals_line(kept)
    total = table.filter(table.column("built_together"))
    assert SENTENCE in _intervals_line(total)


def test_a_selection_without_the_column_keeps_the_record() -> None:
    table = _intervals([_covers()], _assets(_copula()))
    assert _intervals_line(table.drop_columns(["built_together"])) == _intervals_line(
        table
    )


def test_a_table_written_before_the_flag_reads_as_before(tmp_path: Path) -> None:
    table = _intervals([_covers()], _assets(_copula()))
    document = json.loads(EvaluationMetadata.from_table(table).to_json())
    del document["setup"]["intervals"]["built_together"]
    older = table.drop_columns(["built_together"])
    older = older.replace_schema_metadata(
        {
            **(older.schema.metadata or {}),
            EVALUATION_METADATA_KEY.encode(): json.dumps(document).encode(),
        }
    )
    metadata = EvaluationMetadata.from_table(older)
    metadata.check_table(older)
    assert "built_together" not in setup_text(older)
    path = tmp_path / "older.parquet"
    write_comparison(older, path)
    assert "built_together" not in pq.read_table(path).column_names


def test_intervals_are_written_and_read_back_with_the_flag(tmp_path: Path) -> None:
    table = _intervals([_covers()], _assets(_copula()))
    path = tmp_path / "intervals.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.column("built_together").equals(table.column("built_together"))
    assert setup_text(read) == setup_text(table)


def test_merging_combines_the_kinds() -> None:
    copula = EvaluationMetadata.from_table(_intervals([_covers()], _assets(_copula())))
    held = EvaluationMetadata.from_table(
        _intervals([_covers()], _assets(_copula(), HELD))
    )
    plain = EvaluationMetadata.from_table(_intervals([_covers()], _assets(_copula())))
    document = json.loads(plain.to_json())
    del document["setup"]["intervals"]["built_together"]
    older = EvaluationMetadata.from_json(json.dumps(document))
    merged = EvaluationMetadata.merge(copula, held, older)
    assert merged.setup.intervals is not None
    assert merged.setup.intervals["built_together"] == [
        "empirical_copula",
        "hazard_distribution",
    ]
    assert EvaluationMetadata.merge(older, older).setup.intervals == (
        older.setup.intervals
    )


def test_an_unknown_kind_is_refused_by_the_setup_text() -> None:
    table = _intervals([_covers()], _assets(_copula()))
    document = json.loads(EvaluationMetadata.from_table(table).to_json())
    document["setup"]["intervals"]["built_together"] = ["sorted_by_hand"]
    with pytest.raises(EvaluationError, match="built_together names kinds"):
        setup_text(EvaluationMetadata.from_json(json.dumps(document)))
