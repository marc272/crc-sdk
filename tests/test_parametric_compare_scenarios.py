"""Every cover or design on every scenario's block, and drift from a reference.

Every number here is synthetic: small made-up curves, series and tables.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import PiecewiseLinearImpact

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    AtCell,
    BinaryPayout,
    Budget,
    CostOfCapitalLoad,
    Cover,
    CoverPackage,
    EmpiricalCopula,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    ImpactNoise,
    Independent,
    IndexDefinition,
    LeakageError,
    LossModel,
    MetricOptions,
    ModelRun,
    NoBasisRisk,
    OutcomeSample,
    PurePremium,
    SampleError,
    SampleProvenance,
    ScenarioError,
    ScenarioSet,
    StepPayout,
    compare,
    compare_scenarios,
    design,
    design_joint,
    evaluate,
    holdout,
    sample_annual,
    sample_annual_pool,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
    setup_text,
    to_drift,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import _read_at
from crc_sdk.parametric.metrics import metric_values
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.workflows.distributions.ReturnPeriodExtrapolationWarning"
    ),
]

UNTYPED_CURVE: Any = sample_curve
UNTYPED_CURVES: Any = sample_curves
UNTYPED_HAZARDS: Any = sample_curve_hazards
UNTYPED_SET: Any = ScenarioSet
UNTYPED_COMPARE: Any = compare_scenarios
UNTYPED_DRIFT: Any = to_drift

CELL = 613196570331971583
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
LOSS = LossModel(RAMP, value=1e6, value_unit="EUR")
NOISE = ImpactNoise(0.3)
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "mm/day", aggregation="annual_maximum")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10.0,))
WINDOW = (2045, 2050, 2055)
LATER = (2075, 2080, 2085)


def _dataset(provider: str = "fixture") -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=SourceProvenance(
            provider=provider,
            dataset="rx1day",
            version="1",
            licence="CC0-1.0",
            attribution="Synthetic fixture",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 200.0),
    )


def _location(horizon: int) -> float:
    return 50.0 + (horizon - 2020) / 4.0


def _row(horizon: int = 2050, pathway: str = "ssp245", **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell_index": CELL,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": horizon,
        "pathway": pathway,
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": _location(horizon),
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }
    row.update(fields)
    return row


def _draws(
    label: str,
    curve: Any = None,
    *,
    seed: int = 2,
    n_draws: int = 120,
    provider: str = "fixture",
    **options: Any,
) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "metadata": _dataset(provider),
        "n_draws": n_draws,
        "rng": np.random.default_rng(seed),
        "loss_model": LOSS,
        "basis_risk": NOISE,
        "asset_id": "a",
        "scenario": label,
        **options,
    }
    sample: OutcomeSample = UNTYPED_CURVE(
        _row() if curve is None else curve, **arguments
    )
    return sample


def _window(label: str, horizons: Sequence[int] = WINDOW, **options: Any) -> Any:
    return _draws(label, [_row(horizon) for horizon in horizons], **options)


def _step(*amounts: Any, label: str = "step", **fields: Any) -> Cover:
    return Cover(
        INDEX,
        StepPayout([60.0, 75.0], list(amounts or (Free(), Free()))),
        payout_unit="EUR",
        label=label,
        **fields,
    )


GIVEN = _step(2e5, 6e5, label="given")


def _designed(sample: OutcomeSample, **options: Any) -> Any:
    return design(_step(label="designed"), sample, MSE(), **options)


def _series(pathway: str, first: int, *, count: int = 40, seed: int = 3) -> Any:
    years = np.arange(first, first + count, dtype=np.int64)
    values = np.random.default_rng(seed).gumbel(60.0, 12.0, count)
    return AnnualSeries(
        years,
        values,
        "mm/day",
        {
            "kind": "table",
            "source": {"provider": "fixture", "recipe": "rx1day", "pathway": pathway},
        },
    )


def _years(label: str, pathway: str, first: int, **options: Any) -> Any:
    return sample_annual(
        _series(pathway, first, seed=first),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario=label,
        **options,
    )


def _run(model: str, pathway: str, first: int = 2015) -> ModelRun:
    seed = sum(model.encode("utf-8")) + first
    series = _series(pathway, first, seed=seed)
    return ModelRun(series, model=model, member="r1i1p1f1", pathway=pathway)


def _pool(label: str, pathway: str, first: int = 2015) -> Any:
    return sample_annual_pool(
        [_run("M1", pathway, first), _run("M2", pathway, first)],
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario=label,
    )


def _book(
    label: str, seeds: tuple[int, int], *, horizon: int = 2050, location: float = 80.0
) -> Any:
    rows = {"a": _row(horizon), "b": _row(horizon, curve_location=location)}
    sample: OutcomeSample = UNTYPED_CURVES(
        rows,
        metadata=dict.fromkeys(rows, _dataset()),
        n_draws=120,
        rngs={name: np.random.default_rng(seeds[i]) for i, name in enumerate(rows)},
        loss_models=dict.fromkeys(rows, LOSS),
        basis_risk=NOISE,
        dependence=Independent(),
        scenario=label,
    )
    return sample


def _book_covers(*, free: bool = False) -> dict[str, Cover]:
    amounts: tuple[Any, ...] = (Free(), Free()) if free else (2e5, 6e5)
    return {
        asset: _step(*amounts, label=f"step-{asset}", asset_id=asset)
        for asset in ("a", "b")
    }


def _rows(table: Any, **fields: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in table.to_pylist()
        if all(row[name] == value for name, value in fields.items())
    ]


def _row_of(table: Any, **fields: Any) -> dict[str, Any]:
    (row,) = _rows(table, **fields)
    return row


def _scenarios_record(table: Any) -> dict[str, Any]:
    record = EvaluationMetadata.from_table(table).setup.scenarios
    assert record is not None
    return record


def _drift_record(table: Any) -> dict[str, Any]:
    record = EvaluationMetadata.from_table(table).setup.drift
    assert record is not None
    return record


def _evaluations(table: Any) -> tuple[dict[str, Any], ...]:
    return EvaluationMetadata.from_table(table).setup.evaluations


def _evaluation(
    table: Any, scenario: str, design_scenario: str | None
) -> dict[str, Any]:
    (entry,) = [
        each
        for each in _evaluations(table)
        if each["scenario"] == scenario and each["design_scenario"] == design_scenario
    ]
    return entry


def _pair_set() -> tuple[OutcomeSample, ScenarioSet, Any]:
    """A design on the reference, and two curve scenarios on shared draws."""
    train = _window("early", seed=1)
    blocks = [_window("early"), _window("late", LATER)]
    return train, ScenarioSet(blocks, reference="early"), _designed(train)


# -- one scenario --------------------------------------------------------------


def _same_rows(ours: Any, theirs: Any, *, skip: Sequence[str] = ()) -> None:
    assert ours.num_rows == theirs.num_rows
    for name in theirs.column_names:
        if name not in skip:
            assert ours.column(name).to_pylist() == theirs.column(name).to_pylist(), (
                name
            )


def test_one_scenario_gives_compares_rows_for_a_design() -> None:
    train = _draws("only", seed=1)
    block = _draws("only")
    designed = _designed(train)
    table = compare_scenarios([designed], ScenarioSet([block], reference="only"))
    plain = compare([designed], block, split="out_of_sample")
    assert table.column_names.count("premium_repriced") == 1
    assert "premium_repriced" not in plain.column_names
    _same_rows(table, plain)


def test_one_scenario_on_years_gives_compares_rows() -> None:
    train, test = holdout(_years("only", "ssp245", 2000))
    designed = _designed(train)
    table = compare_scenarios([designed], ScenarioSet([test], reference="only"))
    _same_rows(table, compare([designed], test, split="out_of_sample"))
    assert _evaluation(table, "only", "only")["method"] == "holdout"


def test_one_scenario_gives_compares_rows_apart_from_a_given_covers_premium() -> None:
    train = _draws("only", seed=1)
    block = _draws("only")
    table = compare_scenarios(
        [GIVEN], ScenarioSet([block], reference="only"), design_sample=train
    )
    plain = compare([GIVEN], block, split="out_of_sample", design_sample=train)
    net = [name for name in plain.column_names if "net_loss" in name]
    _same_rows(table, plain, skip=("premium", "premium_basis", *net))
    row = table.to_pylist()[0]
    assert row["premium_basis"] == "reference"
    assert plain.to_pylist()[0]["premium_basis"] == "evaluation"
    payout = GIVEN.payout_values(train.index[:, 0])
    assert row["premium"] == PurePremium().premium(payout, train.probability)


# -- the table -----------------------------------------------------------------


def test_every_entry_is_on_every_scenario_the_reference_first() -> None:
    train, scenarios, designed = _pair_set()
    late = scenarios.sample("late")
    held = ScenarioSet([late, scenarios.sample("early")], reference="early")
    table = compare_scenarios([designed, GIVEN], held, design_sample=train)
    rows = table.select(["scenario", "cover_label"]).to_pylist()
    assert [(row["scenario"], row["cover_label"]) for row in rows] == [
        ("early", "designed"),
        ("early", "given"),
        ("late", "designed"),
        ("late", "given"),
    ]
    assert set(table.column("split").to_pylist()) == {"out_of_sample"}


def test_scenario_says_where_a_row_is_and_design_scenario_where_it_was_made() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed, GIVEN], scenarios, design_sample=train)
    assert {row["design_scenario"] for row in _rows(table, cover_label="given")} == {
        None
    }
    for row in _rows(table, cover_label="designed"):
        assert row["design_scenario"] == "early"
    late = _row_of(table, cover_label="designed", scenario="late")
    assert (late["pathway"], late["horizon"]) == ("ssp245", 2080)
    assert late["n_rows"] == 120


def test_a_design_per_scenario_and_a_drifting_cover_are_slices_of_one_table() -> None:
    blocks = [_window("early"), _window("late", LATER)]
    scenarios = ScenarioSet(blocks, reference="early")
    designs = [
        _designed(_window("early", seed=1)),
        _designed(_window("late", LATER, seed=1)),
    ]
    table = compare_scenarios(designs, scenarios)
    assert table.num_rows == 4
    on_own = [
        row for row in table.to_pylist() if row["design_scenario"] == row["scenario"]
    ]
    assert {row["scenario"] for row in on_own} == {"early", "late"}


def test_two_sets_designed_on_two_scenarios_keep_their_keys() -> None:
    early = _book("early", (11, 12))
    late = _book("late", (11, 12), horizon=2080, location=95.0)
    train_early = _book("early", (1, 2))
    train_late = _book("late", (1, 2), horizon=2080, location=95.0)
    entries = [
        {
            asset: design(cover, train.asset(asset), MSE())
            for asset, cover in _book_covers(free=True).items()
        }
        for train in (train_early, train_late)
    ]
    table = compare_scenarios(entries, ScenarioSet([early, late], reference="early"))
    totals = _rows(table, level="total")
    assert {row["design_scenario"] for row in totals} == {"early", "late"}
    assert len(totals) == 4


def test_a_table_round_trips_through_a_parquet_file(tmp_path: Path) -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed, GIVEN], scenarios, design_sample=train)
    path = tmp_path / "scenarios.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.column("premium_repriced").to_pylist() == (
        table.column("premium_repriced").to_pylist()
    )
    assert setup_text(read) == setup_text(table)
    assert to_long(read).num_rows == to_long(table).num_rows


def test_rows_of_one_scenario_state_only_its_setup() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios)
    late = table.filter(pc.equal(table.column("scenario"), "late"))
    text = setup_text(late)
    assert "Scenario late" in text
    assert "Scenario early (the reference)" in text
    assert text.count("Split:") == 1
    assert "not among these rows" in text


# -- what is shared ------------------------------------------------------------


def _uniform(sample: OutcomeSample, locations: Sequence[float]) -> Any:
    """Each row's standardised Gumbel variate, which its uniform alone sets."""
    members = np.asarray(locations, dtype=np.float64)
    at = members[np.mod(sample.period, len(locations))]
    return (sample.index[:, 0] - at) / 10.0


def test_evaluation_draws_share_their_uniforms_across_curves() -> None:
    first = _draws("a", _row(2030))
    second = _draws("b", _row(2080))
    ScenarioSet([first, second], reference="a")
    np.testing.assert_allclose(
        _uniform(first, [_location(2030)]), _uniform(second, [_location(2080)])
    )


def test_evaluation_draws_share_their_uniforms_across_windows() -> None:
    first = _window("a")
    second = _window("b", LATER)
    third = _draws("c", _row(2030))
    ScenarioSet([first, second, third], reference="a")
    of_first = _uniform(first, [_location(year) for year in WINDOW])
    of_second = _uniform(second, [_location(year) for year in LATER])
    of_third = _uniform(third, [_location(2030)])
    np.testing.assert_allclose(of_first, of_second)
    np.testing.assert_allclose(of_first, of_third)


def test_the_metadata_records_who_shares_draws_and_the_generator_record() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios)
    shared = _scenarios_record(table)["shared_draws"]
    assert shared["labels"] == ["early", "late"]
    document = scenarios.sample("early").provenance.to_document()
    (record,) = [item for item in document["rng"] if item["step"] == "hazard"]
    assert shared["rng"] == {"asset 'a'": [["PCG64", record["state"]]]}


def test_reusing_the_designs_generator_on_another_curve_raises() -> None:
    train = _draws("b", _row(2030), seed=1)
    block = _draws("b", _row(2080), seed=1)
    with pytest.raises(LeakageError, match="generator record"):
        _check_out_of_sample(train, block)
    designed = _designed(train)
    with pytest.raises(LeakageError, match="generator record"):
        evaluate(designed, block, split="out_of_sample")
    with pytest.raises(LeakageError, match="generator record"):
        compare([designed], block, split="out_of_sample")
    with pytest.raises(LeakageError, match="generator record"):
        compare_scenarios([designed], ScenarioSet([block], reference="b"))
    with pytest.raises(LeakageError, match="generator record"):
        compare_scenarios(
            [GIVEN],
            ScenarioSet([_draws("b", _row(2030), seed=1)], reference="b"),
            design_sample=_draws("b", _row(2045), seed=1),
        )


def test_reusing_one_generator_in_a_hand_built_loop_raises() -> None:
    generator_seed = 5
    train = _draws("a", _row(2030), seed=generator_seed)
    for horizon in (2050, 2080):
        block = _draws("b", _row(horizon), seed=generator_seed)
        with pytest.raises(LeakageError, match="one uniform"):
            evaluate(GIVEN, block, split="out_of_sample", design_sample=train)


def test_reusing_a_windows_generator_raises_and_another_curves_passes() -> None:
    train = _window("a", seed=1)
    with pytest.raises(LeakageError, match="generator record"):
        evaluate(
            GIVEN,
            _window("b", LATER, seed=1),
            split="out_of_sample",
            design_sample=train,
        )
    evaluate(
        GIVEN, _window("b", LATER, seed=2), split="out_of_sample", design_sample=train
    )


def test_a_multi_asset_block_reusing_a_generator_raises() -> None:
    with pytest.raises(LeakageError, match="generator record"):
        _check_out_of_sample(
            _book("a", (1, 2)), _book("b", (3, 2), horizon=2080, location=95.0)
        )
    _check_out_of_sample(
        _book("a", (1, 2)), _book("b", (3, 4), horizon=2080, location=95.0)
    )


def test_blocks_of_one_holdout_pass_on_other_curves_with_disjoint_periods() -> None:
    train, _ = holdout(_draws("a", _row(2030), seed=1, n_draws=120))
    _, test = holdout(_draws("b", _row(2080), seed=1, n_draws=120))
    assert not set(train.period.tolist()) & set(test.period.tolist())
    _check_out_of_sample(train, test)
    table = compare_scenarios(
        [GIVEN],
        ScenarioSet([test], reference="b"),
        design_sample=_draws("b", _row(2080), seed=7),
    )
    assert table.num_rows == 1


def test_different_seeds_on_one_curve_still_pass() -> None:
    _check_out_of_sample(_draws("a", seed=1), _draws("a", seed=2))


def test_a_shared_source_keeps_the_error_that_names_it() -> None:
    sample = _draws("a", seed=1)
    with pytest.raises(LeakageError, match="from the same source"):
        evaluate(GIVEN, sample, split="out_of_sample", design_sample=sample)


# -- the split methods ---------------------------------------------------------


def test_cross_rows_have_the_method_other_scenario() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed, GIVEN], scenarios, design_sample=train)
    assert _evaluation(table, "early", "early")["method"] == "fresh_draws"
    assert _evaluation(table, "early", None)["method"] == "fresh_draws"
    cross = _evaluation(table, "late", "early")
    assert cross["method"] == "other_scenario"
    assert cross["leakage_check"] == "passed"
    assert cross["details"] == {"block": {"kind": "as_built"}}
    assert _evaluation(table, "late", None)["method"] == "other_scenario"


def test_other_scenario_takes_precedence_over_a_blocks_own_split() -> None:
    train, test = holdout(_window("early", n_draws=120, seed=9))
    _, late = holdout(_window("late", LATER, n_draws=120, seed=9))
    scenarios = ScenarioSet([test, late], reference="early")
    table = compare_scenarios([_designed(train)], scenarios)
    assert _evaluation(table, "early", "early")["method"] == "holdout"
    cross = _evaluation(table, "late", "early")
    assert cross["method"] == "other_scenario"
    assert cross["details"]["block"] == {
        "kind": "holdout",
        "block": "test",
        "train_fraction": 0.7,
        "fold": 0,
    }


def test_a_design_on_a_later_scenario_is_a_cross_row_on_the_reference() -> None:
    train, scenarios, _ = _pair_set()
    late_design = _designed(_window("late", LATER, seed=1))
    table = compare_scenarios([late_design], scenarios)
    assert _evaluation(table, "early", "late")["method"] == "other_scenario"
    assert _evaluation(table, "late", "late")["method"] == "fresh_draws"


def test_one_blocks_evaluations_each_name_their_design_scenario() -> None:
    train, scenarios, designed = _pair_set()
    late_design = _designed(_window("late", LATER, seed=1))
    table = compare_scenarios(
        [designed, late_design, GIVEN], scenarios, design_sample=train
    )
    named = [(e["scenario"], e["design_scenario"]) for e in _evaluations(table)]
    assert sorted(named, key=str) == sorted(
        [
            ("early", "early"),
            ("early", "late"),
            ("early", None),
            ("late", "early"),
            ("late", "late"),
            ("late", None),
        ],
        key=str,
    )


def test_an_unlabelled_pair_of_different_curves_is_a_separate_sample() -> None:
    train = _draws("baseline", _row(2030), seed=1)
    block = _draws("baseline", _row(2080), seed=2)
    table = compare([GIVEN], block, split="out_of_sample", design_sample=train)
    assert _evaluations(table)[0]["method"] == "separate_sample"
    assert "from the same curve" not in setup_text(table)


def test_fresh_draws_of_one_curve_stay_fresh_draws() -> None:
    train = _draws("baseline", seed=1)
    block = _draws("baseline", seed=2)
    table = compare([GIVEN], block, split="out_of_sample", design_sample=train)
    assert _evaluations(table)[0]["method"] == "fresh_draws"
    assert "from the same curve" in setup_text(table)


def test_a_window_beside_one_of_its_members_is_a_separate_sample() -> None:
    train = _draws("baseline", _row(2050), seed=1)
    block = _window("baseline", seed=2)
    table = compare([GIVEN], block, split="out_of_sample", design_sample=train)
    assert _evaluations(table)[0]["method"] == "separate_sample"
    assert "from the same" not in setup_text(table)


def test_disjoint_years_of_one_series_through_another_kind_are_disjoint_periods() -> (
    None
):
    run = _run("M1", "ssp245")
    alone = sample_annual(
        run.series, loss_model=LOSS, basis_risk=NoBasisRisk(), asset_id="a"
    )
    pool = sample_annual_pool(
        [run, _run("M2", "ssp245")],
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
    )
    early = alone.select(alone.period[:20])
    assert pool.provenance.pool is not None
    held_years = pool.provenance.pool["year"]
    late_rows = [
        int(period)
        for period, year in zip(pool.period, held_years, strict=True)
        if year >= 2035
    ]
    late = pool.select(late_rows)
    table = compare([GIVEN], late, split="out_of_sample", design_sample=early)
    assert _evaluations(table)[0]["method"] == "disjoint_periods"


def test_disjoint_draws_of_one_curve_through_another_kind_are_disjoint_periods() -> (
    None
):
    rain = _row(2050, hazard_name="rain")
    heat = _row(2050, hazard_name="heat", curve_location=30.0)
    alone = _draws("baseline", rain, seed=11)
    sample: OutcomeSample = UNTYPED_HAZARDS(
        {"rain": rain, "heat": heat},
        metadata={"rain": _dataset(), "heat": _dataset()},
        n_draws=120,
        rngs={"rain": np.random.default_rng(11), "heat": np.random.default_rng(12)},
        loss_models={"rain": LOSS},
        basis_risk=NOISE,
        dependence=Independent(),
        asset_id="a",
    )
    cover = Cover(RAIN, StepPayout([60.0, 75.0], [2e5, 6e5]), payout_unit="EUR")
    table = compare(
        [cover],
        sample.select(np.arange(60, 120)),
        split="out_of_sample",
        design_sample=alone.select(np.arange(0, 60)),
    )
    assert _evaluations(table)[0]["method"] == "disjoint_periods"


# -- different sources ---------------------------------------------------------


def test_pools_with_no_run_in_common_record_different_sources() -> None:
    historical = _pool("historical", "historical", 1985)
    future = _pool("ssp585", "ssp585", 2041)
    train, test = holdout(historical)
    _, late = holdout(future)
    scenarios = ScenarioSet([test, late], reference="historical")
    table = compare_scenarios([_designed(train)], scenarios)
    assert _evaluation(table, "historical", "historical")["leakage_check"] == "passed"
    assert _evaluation(table, "ssp585", "historical")["leakage_check"] == (
        "different_sources"
    )
    text = setup_text(table)
    assert "no lineage or run in common" in text
    assert "does not say that the sources are independent" in text


def test_series_with_no_lineage_in_common_record_different_sources() -> None:
    train, test = holdout(_years("early", "ssp245", 2000))
    _, late = holdout(_years("late", "ssp585", 2060))
    table = compare_scenarios(
        [_designed(train)], ScenarioSet([test, late], reference="early")
    )
    assert _evaluation(table, "late", "early")["leakage_check"] == "different_sources"


def test_a_run_in_common_is_not_different_sources() -> None:
    run = _run("M1", "ssp245")
    plain = sample_annual(
        run.series,
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario="one",
    )
    early = plain.select(plain.period[:20])
    late = plain.select(plain.period[20:])
    other = _years("two", "ssp245", 2100)
    other_late = other.select(other.period[20:])
    design_late = sample_annual(
        run.series,
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario="two",
    )
    design_late = design_late.select(design_late.period[:20])
    table = compare_scenarios(
        [_designed(design_late)], ScenarioSet([late, other_late], reference="one")
    )
    assert early.lineage == late.lineage
    entry = _evaluation(table, "one", "two")
    assert entry["method"] == "other_scenario"
    assert entry["leakage_check"] == "passed"


def test_different_sources_never_appears_on_a_plain_compare_or_on_curves() -> None:
    train, test = holdout(_years("early", "ssp245", 2000))
    _, late = holdout(_years("late", "ssp245", 2060))
    plain = compare([_designed(train)], late, split="out_of_sample")
    assert _evaluations(plain)[0]["leakage_check"] == "passed"
    _, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios)
    assert {e["leakage_check"] for e in _evaluations(table)} == {"passed"}


# -- the premium ---------------------------------------------------------------


def test_a_design_keeps_its_premium_and_is_repriced_on_each_block() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios)
    for row in table.to_pylist():
        assert row["premium_basis"] == "design"
        assert row["premium"] == designed.premium
        assert row["premium_repriced"] == row["expected_payout"]


def test_premium_repriced_equals_expected_payout_under_pure_premium() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed, GIVEN], scenarios, design_sample=train)
    for row in table.to_pylist():
        assert row["premium_repriced"] == row["expected_payout"]


def test_premium_repriced_follows_the_pricing_on_this_block() -> None:
    train, scenarios, designed = _pair_set()
    pricing = CostOfCapitalLoad(0.1, 0.9)
    table = compare_scenarios([GIVEN], scenarios, design_sample=train, pricing=pricing)
    block = scenarios.sample("late")
    payout = GIVEN.payout_values(block.index[:, 0])
    row = _row_of(table, scenario="late")
    assert row["premium_repriced"] == pricing.premium(payout, block.probability)
    reference = GIVEN.payout_values(train.index[:, 0])
    assert row["premium"] == pricing.premium(reference, train.probability)


def test_net_loss_columns_use_the_contract_price() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([GIVEN], scenarios, design_sample=train, options=OPTIONS)
    row = _row_of(table, scenario="late")
    block = scenarios.sample("late")
    expected = metric_values(
        block.loss[:, 0],
        GIVEN.payout_values(block.index[:, 0]),
        block.probability,
        premium=row["premium"],
        options=OPTIONS,
    )
    assert row["premium"] != row["premium_repriced"]
    assert row["cvar_net_loss_b0_9"] == expected["cvar_net_loss_b0_9"]
    assert row["net_loss_rp10"] == expected["net_loss_rp10"]


def test_a_stated_premium_is_stated_on_every_scenario() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios(
        [GIVEN],
        scenarios,
        design_sample=train,
        premiums={"given": 1234.0},
        options=OPTIONS,
    )
    for row in table.to_pylist():
        assert row["premium_basis"] == "stated"
        assert row["premium"] == 1234.0
        assert row["premium_repriced"] == row["expected_payout"]
    row = _row_of(table, scenario="late")
    block = scenarios.sample("late")
    expected = metric_values(
        block.loss[:, 0],
        GIVEN.payout_values(block.index[:, 0]),
        block.probability,
        premium=1234.0,
        options=OPTIONS,
    )
    assert row["cvar_net_loss_b0_9"] == expected["cvar_net_loss_b0_9"]


def test_premiums_with_a_key_that_matches_no_given_cover_raise() -> None:
    train, scenarios, designed = _pair_set()
    with pytest.raises(EvaluationError, match="match no cover given as is"):
        compare_scenarios(
            [designed, GIVEN], scenarios, design_sample=train, premiums={"nope": 1.0}
        )
    with pytest.raises(TypeError, match="premiums"):
        UNTYPED_COMPARE([GIVEN], scenarios, design_sample=train, premiums=[1.0])


def test_a_set_of_covers_is_stated_whole_or_not_at_all() -> None:
    train = _book("early", (1, 2))
    blocks = [
        _book("early", (11, 12)),
        _book("late", (11, 12), horizon=2080, location=95.0),
    ]
    scenarios = ScenarioSet(blocks, reference="early")
    covers = _book_covers()
    with pytest.raises(EvaluationError, match="every cover of a set"):
        compare_scenarios(
            [covers], scenarios, design_sample=train, premiums={"step-a": 10.0}
        )
    table = compare_scenarios(
        [covers],
        scenarios,
        design_sample=train,
        premiums={"step-a": 10.0, "step-b": 20.0},
    )
    total = _row_of(table, scenario="late", level="total")
    assert total["premium"] == 30.0
    assert total["premium_basis"] == "stated"


def test_a_set_of_covers_sums_its_premiums_and_its_repriced_premiums() -> None:
    train = _book("early", (1, 2))
    blocks = [
        _book("early", (11, 12)),
        _book("late", (11, 12), horizon=2080, location=95.0),
    ]
    scenarios = ScenarioSet(blocks, reference="early")
    table = compare_scenarios([_book_covers()], scenarios, design_sample=train)
    for label in ("early", "late"):
        parts = _rows(table, scenario=label, level="asset")
        total = _row_of(table, scenario=label, level="total")
        assert total["premium_basis"] == "reference"
        assert parts[0]["premium_basis"] == "reference"
        assert total["premium"] == pytest.approx(sum(row["premium"] for row in parts))
        assert total["premium_repriced"] == pytest.approx(
            sum(row["premium_repriced"] for row in parts)
        )
        assert total["design_scenario"] is None
        assert total["level"] == "total" and total["scope"] == "individual"


# -- joint designs -------------------------------------------------------------


def _joint_setup() -> tuple[Any, ScenarioSet]:
    train = _book("early", (1, 2))
    templates = list(_book_covers(free=True).values())
    designed = design_joint(templates, train, MSE())
    blocks = [
        _book("early", (11, 12)),
        _book("late", (11, 12), horizon=2080, location=95.0),
    ]
    return designed, ScenarioSet(blocks, reference="early")


def test_a_joint_design_has_owner_total_rows_per_scenario() -> None:
    designed, scenarios = _joint_setup()
    table = compare_scenarios([designed], scenarios)
    assert table.num_rows == 2
    for row in table.to_pylist():
        assert row["level"] == "total"
        assert row["scope"] == "joint"
        assert row["asset_id"] is None
        assert row["premium_basis"] == "design"
        assert row["premium"] == designed.premium
        assert row["design_scenario"] == "early"
        assert row["premium_repriced"] == row["expected_payout"]
    assert _evaluation(table, "late", "early")["method"] == "other_scenario"


def test_a_joint_designs_total_is_repriced_on_the_total_payout() -> None:
    designed, scenarios = _joint_setup()
    pricing = CostOfCapitalLoad(0.2, 0.9)
    table = compare_scenarios([designed], scenarios, pricing=pricing)
    block = scenarios.sample("late")
    payout = sum(
        designed.covers[asset].payout_values(block.index[:, column])
        for column, asset in enumerate(block.asset_ids)
    )
    row = _row_of(table, scenario="late")
    assert row["premium_repriced"] == pricing.premium(payout, block.probability)


def test_allocated_rows_have_no_premium_and_no_repriced_premium() -> None:
    designed, scenarios = _joint_setup()
    table = compare_scenarios(
        [designed], scenarios, allocation="value", values={"a": 1e6, "b": 1e6}
    )
    allocated = _rows(table, level="asset_by_value")
    assert len(allocated) == 4
    for row in allocated:
        assert row["premium"] is None
        assert row["premium_basis"] is None
        assert row["premium_repriced"] is None
        assert row["cvar_net_loss_b0_9"] is None


def test_an_allocation_is_refused_without_a_joint_design() -> None:
    train, scenarios, designed = _pair_set()
    with pytest.raises(EvaluationError, match="no entry is a joint design"):
        compare_scenarios([designed], scenarios, allocation="value", values={"a": 1.0})
    with pytest.raises(EvaluationError, match="without allocation"):
        compare_scenarios([designed], scenarios, values={"a": 1.0})


def test_a_joint_designs_setup_text_stays_method_only() -> None:
    designed, scenarios = _joint_setup()
    text = setup_text(compare_scenarios([designed], scenarios))
    assert "Scope: joint" in text
    for word in ("better", "outperform", "best"):
        assert word not in text


# -- packages on several hazards -----------------------------------------------


def _hazards(
    label: str, *, seeds: tuple[int, int] = (11, 12), shift: float = 0.0
) -> Any:
    curves = {
        "rain": _row(2050 + int(shift), hazard_name="rain"),
        "heat": _row(
            2050 + int(shift), hazard_name="heat", curve_location=30.0 + shift
        ),
    }
    sample: OutcomeSample = UNTYPED_HAZARDS(
        curves,
        metadata=dict.fromkeys(curves, _dataset()),
        n_draws=120,
        rngs={name: np.random.default_rng(seeds[i]) for i, name in enumerate(curves)},
        loss_models={"rain": LOSS},
        basis_risk=NOISE,
        dependence=Independent(),
        asset_id="a",
        scenario=label,
    )
    return sample


def _contract(index: IndexDefinition, label: str, *amounts: Any) -> Cover:
    return Cover(
        index,
        BinaryPayout(60.0 if index is RAIN else 40.0, *amounts),
        payout_unit="EUR",
        label=label,
    )


def test_a_package_given_as_is_sums_its_premiums_and_is_repriced_per_contract() -> None:
    package = CoverPackage(
        {"rain": _contract(RAIN, "rain", 2e5), "heat": _contract(HEAT, "heat", 1e5)},
        label="two",
    )
    train = _hazards("early", seeds=(1, 2))
    blocks = [_hazards("early"), _hazards("late", shift=20.0)]
    scenarios = ScenarioSet(blocks, reference="early")
    table = compare_scenarios([package], scenarios, design_sample=train)
    assert table.num_rows == 6
    for label in ("early", "late"):
        parts = _rows(table, scenario=label, level="asset")
        row = _row_of(table, scenario=label, level="package")
        assert row["premium_basis"] == "reference"
        assert row["premium"] == pytest.approx(sum(part["premium"] for part in parts))
        assert row["premium_repriced"] == pytest.approx(
            sum(part["premium_repriced"] for part in parts)
        )
        assert row["premium_repriced"] == pytest.approx(row["expected_payout"])
    assert setup_text(table).count("Package two") >= 1


# -- what the setup text says --------------------------------------------------


def test_the_setup_text_states_every_scenario() -> None:
    train, scenarios, designed = _pair_set()
    text = setup_text(compare_scenarios([designed], scenarios))
    assert (
        "Scenario early (the reference): hazard rx1day, as its source names it" in text
    )
    assert "Scenario late:" in text
    assert "product fixture; pathway ssp245; horizon 2050" in text
    assert "120 simulated years (draws 0 to 119)" in text
    assert "as built, with no split" in text


def test_the_setup_text_says_a_held_out_block_was_made_by_a_holdout() -> None:
    train, test = holdout(_window("early", n_draws=120, seed=9))
    table = compare_scenarios(
        [_designed(train)], ScenarioSet([test], reference="early")
    )
    text = setup_text(table)
    assert "the held-out block, from a holdout with train fraction 0.7" in text


def test_the_setup_text_states_a_stated_hazard_name() -> None:
    train, test = holdout(
        sample_annual(
            AnnualSeries(
                np.arange(2000, 2040, dtype=np.int64),
                np.random.default_rng(4).gumbel(60.0, 12.0, 40),
                "mm/day",
                {"kind": "table", "source": {"provider": "fixture"}},
            ),
            loss_model=LOSS,
            basis_risk=NoBasisRisk(),
            asset_id="a",
            scenario="stated",
            hazard_name="rx1day",
        )
    )
    table = compare_scenarios(
        [GIVEN], ScenarioSet([test], reference="stated"), design_sample=train
    )
    assert "hazard rx1day, as the caller states" in setup_text(table)


def test_the_setup_text_says_single_horizon_or_window() -> None:
    train = _window("early", seed=1)
    table = compare_scenarios(
        [GIVEN],
        ScenarioSet(
            [_window("early"), _draws("single", _row(2080))], reference="early"
        ),
        design_sample=train,
    )
    text = setup_text(table)
    assert "A window of 3 horizons: ssp245 2045, ssp245 2050 and ssp245 2055" in text
    assert "mixed with equal weight on one set of uniforms" in text
    assert "The horizon column is their centre" in text
    assert "A single horizon: ssp245 2080." in text


def test_the_setup_text_states_each_members_rows_on_the_block() -> None:
    train, test = holdout(_window("early", n_draws=120, seed=9))
    _, other = holdout(_window("late", LATER, n_draws=120, seed=9))
    table = compare_scenarios(
        [_designed(train)], ScenarioSet([test, other], reference="early")
    )
    text = setup_text(table)
    counts = np.bincount(np.mod(test.period, 3), minlength=3)
    expected = ", ".join(
        f"{year}: {count}" for year, count in zip(WINDOW, counts.tolist(), strict=True)
    )
    assert f"rows per member on this block: {expected}" in text


def test_the_setup_text_says_a_difference_includes_the_width() -> None:
    train = _window("early", seed=1)
    three = _window("early")
    two = _window("late", (2075, 2080))
    table = compare_scenarios(
        [GIVEN], ScenarioSet([three, two], reference="early"), design_sample=train
    )
    assert "windows of different widths" in setup_text(table)
    assert "A difference between them includes the difference in width." in setup_text(
        table
    )


def test_a_window_beside_a_single_horizon_says_the_width_too() -> None:
    train = _window("early", seed=1)
    table = compare_scenarios(
        [GIVEN],
        ScenarioSet(
            [_window("early"), _draws("single", _row(2080))], reference="early"
        ),
        design_sample=train,
    )
    text = setup_text(table)
    assert "a window of horizons beside a single horizon" in text
    assert "includes the difference in width" in text


def test_windows_of_one_width_say_nothing_of_the_width() -> None:
    train, scenarios, designed = _pair_set()
    assert "width" not in setup_text(compare_scenarios([designed], scenarios))


def test_the_setup_text_names_both_scenarios_of_a_cross_row() -> None:
    train, scenarios, designed = _pair_set()
    text = setup_text(
        compare_scenarios([designed, GIVEN], scenarios, design_sample=train)
    )
    assert "designs from scenario early on the block of scenario late" in text
    assert "covers given as is, from the reference's design block" in text
    assert "A cross row is a cover or design from one scenario" in text
    assert "nothing rescales them" in text
    assert "Fixed across the scenarios" in text


def test_the_setup_text_says_curves_share_their_draws() -> None:
    train, scenarios, designed = _pair_set()
    text = setup_text(compare_scenarios([designed], scenarios))
    assert "Shared draws: the curve scenarios early and late share one set" in text
    assert "separate from the design's" in text


def test_per_year_scenarios_say_nothing_of_shared_draws() -> None:
    train, test = holdout(_years("early", "ssp245", 2000))
    _, late = holdout(_years("late", "ssp245", 2060))
    table = compare_scenarios(
        [_designed(train)], ScenarioSet([test, late], reference="early")
    )
    assert "Shared draws" not in setup_text(table)


def test_the_setup_text_says_a_budget_moves_with_a_ratio() -> None:
    train, scenarios, _ = _pair_set()
    designed = _designed(train, budget=Budget(ratio=0.5))
    text = setup_text(compare_scenarios([designed], scenarios))
    assert "ratio, a premium of at most 0.5 x E[L] of its design block" in text
    assert "so the bound moves with the scenario" in text


def test_the_setup_text_says_an_amount_does_not_move() -> None:
    train, scenarios, _ = _pair_set()
    designed = _designed(train, budget=Budget(amount=2e5))
    text = setup_text(compare_scenarios([designed], scenarios))
    assert "amount, a premium of at most 200,000 EUR" in text
    assert "so the bound does not move with the scenario" in text


def test_the_setup_text_repeats_the_caveat_on_native_cmip6() -> None:
    train = _draws("a", _row(2030), seed=1, provider="cmip6")
    blocks = [
        _draws("a", _row(2030), provider="cmip6"),
        _draws("b", _row(2080), provider="cmip6"),
    ]
    table = compare_scenarios(
        [GIVEN], ScenarioSet(blocks, reference="a"), design_sample=train
    )
    text = setup_text(table)
    assert "Native CMIP6 (a and b)" in text
    assert "use results as change signals, not design values" in text


def test_no_caveat_without_native_cmip6() -> None:
    train, scenarios, designed = _pair_set()
    assert "CMIP6" not in setup_text(compare_scenarios([designed], scenarios))


def test_the_setup_text_says_the_difference_includes_the_products() -> None:
    train = _draws("a", _row(2030), seed=1)
    blocks = [_draws("a", _row(2030)), _draws("b", _row(2080), provider="other")]
    scenarios = ScenarioSet(blocks, reference="a", mixed_products=True)
    text = setup_text(compare_scenarios([GIVEN], scenarios, design_sample=train))
    assert "Products: scenario b reads other and the reference a reads fixture" in text
    assert "resolution, method and pooling differ" in text


def test_one_product_says_nothing_of_products() -> None:
    train, scenarios, designed = _pair_set()
    assert "Products:" not in setup_text(compare_scenarios([designed], scenarios))


def test_the_setup_text_says_where_each_scenario_reads_the_index() -> None:
    train, scenarios, designed = _pair_set()
    text = setup_text(compare_scenarios([designed], scenarios))
    assert f"H3 cell {CELL:x} of asset 'a'" in text
    assert "Index: scenario early at" in text


def test_the_setup_text_states_each_premium_basis() -> None:
    train, scenarios, designed = _pair_set()
    other = _step(1e5, 3e5, label="stated")
    text = setup_text(
        compare_scenarios(
            [designed, GIVEN, other],
            scenarios,
            design_sample=train,
            premiums={"stated": 5.0},
        )
    )
    assert "Premium basis:" in text
    assert "designed: the design premium" in text
    assert "given: priced once, with the row's pricing, on the reference's" in text
    assert "stated: stated with premiums=" in text
    assert "premium_repriced prices the same payouts on each block" in text


def test_the_setup_text_says_beta_is_the_same_and_names_short_block_columns() -> None:
    train, test = holdout(_years("early", "ssp245", 2000))
    _, late = holdout(_years("late", "ssp245", 2060))
    table = compare_scenarios(
        [_designed(train)],
        ScenarioSet([test, late], reference="early"),
        options=MetricOptions(betas=(0.5, 0.95), return_periods=(5.0, 100.0)),
    )
    text = setup_text(table)
    assert "Tails: beta is the same on every scenario (0.5, 0.95)" in text
    assert "the loss level of its tail is not" in text
    assert "Short blocks: on the 12 rows of early and late" in text
    assert "cvar_uncovered_b0_95" in text
    assert "loss_rp100" in text
    assert "cvar_uncovered_b0_5," not in text


def test_a_window_outside_a_set_says_what_it_is() -> None:
    train = _window("baseline", seed=1)
    block = _window("baseline", seed=2)
    text = setup_text(
        compare([GIVEN], block, split="out_of_sample", design_sample=train)
    )
    assert (
        "Window of horizons: 3 curves (ssp245 2045, ssp245 2050 and ssp245 2055)"
        in text
    )
    assert "new draws from the same window of horizons" in text
    assert "from the same curve" not in text


def test_a_holdout_of_a_window_says_the_same_window() -> None:
    train, test = holdout(_window("baseline", seed=1))
    text = setup_text(
        compare([GIVEN], test, split="out_of_sample", design_sample=train)
    )
    assert "new draws from the same window of horizons" in text


def test_a_set_says_a_window_once_and_not_as_a_window_line() -> None:
    train, scenarios, designed = _pair_set()
    assert "Window of horizons:" not in setup_text(
        compare_scenarios([designed], scenarios)
    )


def test_the_pricing_line_says_a_given_cover_is_priced_on_the_reference() -> None:
    train, scenarios, designed = _pair_set()
    text = setup_text(compare_scenarios([GIVEN], scenarios, design_sample=train))
    assert "once on the reference's design block, for the covers given as is" in text


def test_a_plain_compare_keeps_its_pricing_line() -> None:
    train, scenarios, designed = _pair_set()
    block = scenarios.sample("early")
    text = setup_text(
        compare([GIVEN], block, split="out_of_sample", design_sample=train)
    )
    assert "on the evaluation sample, for the covers given as is" in text


# -- the metadata --------------------------------------------------------------


def test_the_metadata_records_the_set() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios)
    document = _scenarios_record(table)
    assert document["reference"] == "early"
    assert document["labels"] == ["early", "late"]
    assert document["mixed_products"] is False
    assert document["design_on"] is None
    entry = document["scenarios"]["late"]
    assert entry["source_kind"] == "monte_carlo"
    assert entry["hazard"] == {"name": "rx1day", "from": "curve_row"}
    assert entry["n_rows"] == 120
    assert entry["pathway"] == "ssp245"
    assert entry["horizon"] == 2080
    assert entry["product"] == "fixture"
    assert entry["members"] == [["ssp245", year] for year in LATER]
    assert entry["blocks"][0]["member_rows"] == [40, 40, 40]


def test_the_metadata_records_null_members_for_a_per_year_scenario() -> None:
    train, test = holdout(_years("early", "ssp245", 2000))
    table = compare_scenarios(
        [_designed(train)], ScenarioSet([test], reference="early")
    )
    document = _scenarios_record(table)
    entry = document["scenarios"]["early"]
    assert entry["members"] is None
    assert document["shared_draws"] is None


def test_the_columns_describe_premium_repriced() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios)
    assert table.column_names.index("premium_repriced") == (
        table.column_names.index("premium") + 1
    )
    entry = EvaluationMetadata.from_table(table).columns["premium_repriced"]
    assert entry.metric == "premium_repriced"
    assert entry.unit == "EUR"


def test_compare_writes_no_scenario_records() -> None:
    train, scenarios, _ = _pair_set()
    table = compare(
        [GIVEN], scenarios.sample("early"), split="out_of_sample", design_sample=train
    )
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.setup.scenarios is None
    assert metadata.setup.drift is None
    document = json.loads(metadata.to_json())
    assert "scenarios" not in document["setup"]
    assert "drift" not in document["setup"]
    assert "premium_repriced" not in table.column_names
    assert "scenario" not in metadata.setup.evaluations[0]
    assert "premium_bases" not in metadata.setup.evaluations[0]


def test_a_document_without_the_keys_reads_as_before() -> None:
    train, scenarios, _ = _pair_set()
    table = compare(
        [GIVEN], scenarios.sample("early"), split="out_of_sample", design_sample=train
    )
    metadata = EvaluationMetadata.from_table(table)
    text = metadata.to_json()
    assert EvaluationMetadata.from_json(text).to_json() == text


def _fold_tables() -> tuple[Any, Any]:
    early = year_block_kfold(_window("early", n_draws=120, seed=9), 4)
    late = year_block_kfold(_window("late", LATER, n_draws=120, seed=9), 4)
    tables = []
    for fold in (0, 1):
        train = early[fold][0]
        scenarios = ScenarioSet([early[fold][1], late[fold][1]], reference="early")
        tables.append(
            compare_scenarios([GIVEN], scenarios, fold=fold, design_sample=train)
        )
    return tables[0], tables[1]


def test_folds_merge_under_one_set_of_scenarios() -> None:
    first, second = _fold_tables()
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(first), EvaluationMetadata.from_table(second)
    )
    assert merged.setup.scenarios is not None
    entry = merged.setup.scenarios["scenarios"]["early"]
    assert [block["fold"] for block in entry["blocks"]] == [0, 1]
    assert entry["n_rows"] == 60
    combined = pa.concat_tables([first, second])
    merged.check_table(combined)
    text = setup_text(merged)
    assert text.count("from fold 0 of a 4-fold split") >= 1
    assert text.count("from fold 1 of a 4-fold split") >= 1


def test_merge_requires_one_set_of_scenarios() -> None:
    first, _ = _fold_tables()
    train = _window("other", seed=1)
    other = compare_scenarios(
        [GIVEN], ScenarioSet([_window("other")], reference="other"), design_sample=train
    )
    with pytest.raises(EvaluationError, match="differ in their scenarios"):
        EvaluationMetadata.merge(
            EvaluationMetadata.from_table(first), EvaluationMetadata.from_table(other)
        )
    plain = compare(
        [GIVEN], _window("other"), split="out_of_sample", design_sample=train
    )
    with pytest.raises(EvaluationError, match="differ in their scenarios"):
        EvaluationMetadata.merge(
            EvaluationMetadata.from_table(first), EvaluationMetadata.from_table(plain)
        )


def test_a_kfold_of_a_set_keeps_its_method_on_the_fold() -> None:
    first, _ = _fold_tables()
    assert _evaluation(first, "early", None)["method"] == "kfold"
    assert _evaluation(first, "late", None)["method"] == "other_scenario"
    assert _evaluation(first, "late", None)["details"]["block"]["k"] == 4


def test_a_fold_that_contradicts_the_block_raises() -> None:
    early = year_block_kfold(_window("early", n_draws=120, seed=9), 4)
    scenarios = ScenarioSet([early[1][1]], reference="early")
    with pytest.raises(EvaluationError, match="fold=0 was given"):
        compare_scenarios([GIVEN], scenarios, design_sample=early[1][0])


# -- refusals ------------------------------------------------------------------


def test_scenarios_must_be_a_set() -> None:
    with pytest.raises(TypeError, match="ScenarioSet"):
        UNTYPED_COMPARE([GIVEN], [_window("a")])


def test_an_empty_set_is_refused() -> None:
    with pytest.raises(ScenarioError, match="holds no built samples"):
        compare_scenarios([GIVEN], ScenarioSet(reference="a"))


def test_a_plans_description_is_refused() -> None:
    described = ScenarioSet(reference="a", labels={"a": [("ssp245", 2050)]})
    with pytest.raises(TypeError, match="describes the scenarios a plan reads"):
        compare_scenarios([GIVEN], described)


def test_a_cover_given_as_is_needs_the_references_design_block() -> None:
    train, scenarios, designed = _pair_set()
    with pytest.raises(SampleError, match="design_sample="):
        compare_scenarios([GIVEN], scenarios)


def test_design_sample_must_be_labelled_as_the_reference() -> None:
    train, scenarios, designed = _pair_set()
    with pytest.raises(ScenarioError, match="must be 'early'"):
        compare_scenarios([GIVEN], scenarios, design_sample=_window("late", seed=1))


def test_a_design_sample_labelled_outside_the_set_is_refused() -> None:
    _, scenarios, _ = _pair_set()
    designed = _designed(_window("elsewhere", seed=1))
    with pytest.raises(ScenarioError, match="is not one of the set's"):
        compare_scenarios([designed], scenarios)


def test_a_design_sample_argument_must_be_a_sample() -> None:
    _, scenarios, designed = _pair_set()
    with pytest.raises(TypeError, match="design_sample"):
        UNTYPED_COMPARE([designed], scenarios, design_sample=[1])


def test_the_designs_of_one_mapping_must_share_a_label() -> None:
    blocks = [
        _book("early", (11, 12)),
        _book("late", (11, 12), horizon=2080, location=95.0),
    ]
    scenarios = ScenarioSet(blocks, reference="early")
    mixed = {
        "a": design(
            _book_covers(free=True)["a"], _book("early", (1, 2)).asset("a"), MSE()
        ),
        "b": design(
            _book_covers(free=True)["b"],
            _book("late", (1, 2), horizon=2080, location=95.0).asset("b"),
            MSE(),
        ),
    }
    with pytest.raises(ScenarioError, match="must be made on one scenario"):
        compare_scenarios([mixed], scenarios)


def test_a_cover_bound_to_a_location_a_scenario_doesnt_read_is_refused() -> None:
    bound_index = IndexDefinition(
        "rx1day",
        "mm/day",
        aggregation="annual_maximum",
        location=AtCell(format(CELL + 1, "x")),
    )
    bound = Cover(
        bound_index, StepPayout([60.0, 75.0], [2e5, 6e5]), payout_unit="EUR", label="b"
    )
    train, scenarios, _ = _pair_set()
    with pytest.raises(ScenarioError, match="differ in location"):
        compare_scenarios([bound], scenarios, design_sample=train)


def test_the_same_entry_twice_is_refused() -> None:
    train, scenarios, designed = _pair_set()
    with pytest.raises(EvaluationError, match="repeat a row key"):
        compare_scenarios([designed, designed], scenarios)


def test_an_empty_list_of_covers_is_refused() -> None:
    _, scenarios, _ = _pair_set()
    with pytest.raises(EvaluationError):
        compare_scenarios([], scenarios)


def test_a_template_is_refused() -> None:
    train, scenarios, _ = _pair_set()
    from crc_sdk.parametric import UnresolvedParameterError

    with pytest.raises(UnresolvedParameterError):
        compare_scenarios([_step()], scenarios, design_sample=train)


def test_a_design_on_the_reference_with_another_design_sample_is_refused() -> None:
    train, scenarios, designed = _pair_set()
    with pytest.raises(EvaluationError, match="differs from the design's own"):
        compare_scenarios(
            [designed, GIVEN], scenarios, design_sample=_window("early", seed=8)
        )


# -- drift ---------------------------------------------------------------------


def _drift_table(**options: Any) -> tuple[Any, Any]:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios(
        [designed, GIVEN], scenarios, design_sample=train, options=OPTIONS, **options
    )
    return table, to_drift(table)


def test_drift_is_the_long_form_with_differences_and_ratios() -> None:
    table, drift = _drift_table()
    long = to_long(table)
    assert drift.column_names == [
        *long.column_names,
        "reference_scenario",
        "reference_product",
        "reference_n_rows",
        "reference_value",
        "difference",
        "ratio",
    ]
    late = {
        row["cover_label"]: row
        for row in table.to_pylist()
        if row["scenario"] == "late"
    }
    early = {
        row["cover_label"]: row
        for row in table.to_pylist()
        if row["scenario"] == "early"
    }
    for row in drift.to_pylist():
        name = row["cover_label"]
        assert row["scenario"] == "late"
        column = row["column"]
        value, base = late[name][column], early[name][column]
        assert row["value"] == value
        assert row["reference_value"] == base
        assert row["reference_scenario"] == "early"
        assert row["reference_product"] == "fixture"
        assert row["reference_n_rows"] == 120
        if value is None or base is None:
            assert row["difference"] is None and row["ratio"] is None
            continue
        assert row["difference"] == pytest.approx(value - base)
        if base != 0.0:
            assert row["ratio"] == pytest.approx(value / base)
        else:
            assert row["ratio"] is None


def test_drift_has_no_ranking_column_and_no_reference_rows() -> None:
    _, drift = _drift_table()
    for name in drift.column_names:
        for word in ("rank", "best", "order", "worst"):
            assert word not in name
    assert set(drift.column("scenario").to_pylist()) == {"late"}


def test_drift_leaves_rows_in_the_long_forms_order() -> None:
    table, drift = _drift_table()
    long = [
        (row["column"], row["cover_label"], row["scenario"])
        for row in to_long(table).to_pylist()
        if row["scenario"] != "early"
    ]
    assert [
        (row["column"], row["cover_label"], row["scenario"])
        for row in drift.to_pylist()
    ] == long


def test_drift_pairs_a_design_with_itself_only() -> None:
    train, scenarios, designed = _pair_set()
    late_design = _designed(_window("late", LATER, seed=1))
    table = compare_scenarios([designed, late_design], scenarios, options=OPTIONS)
    drift = to_drift(table)
    for row in drift.select(
        ["design_id", "design_scenario", "column", "reference_value", "scenario"]
    ).to_pylist():
        base = _row_of(table, design_id=row["design_id"], scenario="early")
        assert row["reference_value"] == base[row["column"]]


def test_drift_takes_any_label_as_the_reference() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([designed], scenarios, options=OPTIONS)
    drift = to_drift(table, reference="late")
    assert set(drift.column("scenario").to_pylist()) == {"early"}
    assert set(drift.column("reference_scenario").to_pylist()) == {"late"}
    assert _drift_record(drift)["reference"] == "late"
    assert _scenarios_record(drift)["reference"] == "early"


def test_drift_takes_the_sets_reference_from_the_metadata() -> None:
    _, drift = _drift_table()
    assert _drift_record(drift)["reference"] == "early"


def test_drift_states_the_reference_in_its_setup_text() -> None:
    _, drift = _drift_table()
    text = setup_text(drift)
    assert "Drift: each row's difference and ratio" in text
    assert "on scenario early" in text
    assert "with no ranking" in text
    assert "Scenario early (the reference)" in text
    assert "not among these rows" not in text


def test_drift_is_null_where_a_value_or_the_reference_is_null() -> None:
    table, drift = _drift_table()
    nulls = [
        row
        for row in drift.to_pylist()
        if row["value"] is None or row["reference_value"] is None
    ]
    assert nulls
    for row in nulls:
        assert row["difference"] is None
        assert row["ratio"] is None


def test_a_ratio_is_null_where_the_reference_value_is_zero() -> None:
    train, scenarios, _ = _pair_set()
    never = _step(0.0, 0.0, label="never")
    table = compare_scenarios([never], scenarios, design_sample=train, options=OPTIONS)
    drift = to_drift(table)
    zero = [row for row in drift.to_pylist() if row["reference_value"] == 0.0]
    assert zero
    for row in zero:
        assert row["ratio"] is None
        assert row["difference"] == row["value"]


def test_drift_carries_payout_frequency_and_unidentified_share() -> None:
    _, drift = _drift_table()
    columns = set(drift.column("column").to_pylist())
    assert {"payout_frequency", "unidentified_share", "premium_repriced"} <= columns


def _reference_book(years: int = 40) -> OutcomeSample:
    index = np.random.default_rng(3).gamma(2.0, 20.0, (years, 2))
    return OutcomeSample(
        asset_ids=("a", "b"),
        period=np.arange(1990, 1990 + years, dtype=np.int64),
        probability=np.full(years, 1.0 / years),
        index=index,
        loss=index,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "hazard", "name": "series"},),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )


def test_drift_takes_rank_block_columns_as_identity_columns() -> None:
    def block(label: str, seeds: tuple[int, int], horizon: int) -> OutcomeSample:
        chosen = {
            "a": _row(horizon),
            "b": _row(horizon, curve_location=80.0 + (horizon - 2050) / 4.0),
        }
        sample: OutcomeSample = UNTYPED_CURVES(
            chosen,
            metadata=dict.fromkeys(chosen, _dataset()),
            n_draws=120,
            rngs={
                name: np.random.default_rng(seed)
                for name, seed in zip(chosen, seeds, strict=True)
            },
            loss_models=dict.fromkeys(chosen, LOSS),
            basis_risk=NOISE,
            dependence=EmpiricalCopula(_reference_book()),
            scenario=label,
        )
        return sample

    scenarios = ScenarioSet(
        [block("early", (11, 12), 2050), block("late", (11, 12), 2080)],
        reference="early",
    )
    train = block("early", (1, 2), 2050)
    table = compare_scenarios(
        [_book_covers()], scenarios, design_sample=train, options=OPTIONS
    )
    assert "copula_reference_years" in table.column_names
    drift = to_drift(table)
    assert "copula_reference_years" not in drift.column("column").unique().to_pylist()
    assert "copula_reference_years" in drift.column_names
    total = _rows(drift, level="total", column="expected_loss")
    assert total[0]["copula_reference_years"] == 40


def test_a_row_without_a_reference_row_raises() -> None:
    train, scenarios, designed = _pair_set()
    other = _designed(_window("late", LATER, seed=1))
    table = compare_scenarios([designed, other], scenarios, options=OPTIONS)
    kept = table.filter(
        pc.invert(
            pc.and_(
                pc.equal(table.column("scenario"), "early"),
                pc.equal(table.column("design_scenario"), "late"),
            )
        )
    )
    with pytest.raises(EvaluationError, match="has no row on the reference"):
        to_drift(kept)


def test_drift_needs_a_label_in_the_table() -> None:
    table, _ = _drift_table()
    with pytest.raises(EvaluationError, match="holds no scenario 'nope'"):
        to_drift(table, reference="nope")
    with pytest.raises(TypeError, match="reference must be"):
        UNTYPED_DRIFT(table, reference=1)


def test_drift_of_a_table_with_no_set_needs_a_reference() -> None:
    train, scenarios, _ = _pair_set()
    plain = compare(
        [GIVEN], scenarios.sample("early"), split="out_of_sample", design_sample=train
    )
    with pytest.raises(EvaluationError, match="states no set's reference"):
        to_drift(plain)
    drift = to_drift(plain, reference="early")
    assert drift.num_rows == 0
    assert "difference" in drift.column_names


def test_drift_refuses_what_to_long_refuses() -> None:
    with pytest.raises(TypeError):
        UNTYPED_DRIFT("not a table")


def test_drift_on_the_joint_total_pairs_the_total_with_itself() -> None:
    designed, scenarios = _joint_setup()
    table = compare_scenarios([designed], scenarios, options=OPTIONS)
    drift = to_drift(table)
    assert set(drift.column("level").to_pylist()) == {"total"}
    assert set(drift.column("scope").to_pylist()) == {"joint"}


def test_drift_rows_have_the_expected_value_for_a_simple_metric() -> None:
    train, scenarios, designed = _pair_set()
    table = compare_scenarios([GIVEN], scenarios, design_sample=train, options=OPTIONS)
    drift = to_drift(table)
    (row,) = [item for item in drift.to_pylist() if item["column"] == "expected_loss"]
    early = _row_of(table, scenario="early")["expected_loss"]
    late = _row_of(table, scenario="late")["expected_loss"]
    assert row["difference"] == pytest.approx(late - early)
    assert row["ratio"] == pytest.approx(late / early)
    assert row["unit"] == "EUR"


def test_a_mapping_of_unknown_type_still_reads_as_compare_reads_it() -> None:
    _, scenarios, _ = _pair_set()
    entry: Mapping[str, Any] = {"nope": 1.0}
    with pytest.raises(TypeError):
        compare_scenarios([entry], scenarios)


def test_a_budget_free_design_has_no_budget_line_but_the_rest_stands() -> None:
    train, scenarios, designed = _pair_set()
    assert "Budgets:" not in setup_text(compare_scenarios([designed], scenarios))


def test_expected_uncovered_designs_are_paired_by_design_id_across_objectives() -> None:
    train = _window("early", seed=1)
    templates = (_step(label="one"), _step(label="two"))
    designs = [
        design(templates[0], train, MSE()),
        design(templates[1], train, ExpectedUncovered(), budget=Budget(ratio=0.5)),
    ]
    scenarios = ScenarioSet(
        [_window("early"), _window("late", LATER)], reference="early"
    )
    drift = to_drift(compare_scenarios(designs, scenarios, options=OPTIONS))
    assert set(drift.column("cover_label").to_pylist()) == {"one", "two"}


# -- where the index is read, and malformed records -----------------------------


def _per_year_at(label: str, series_fields: Mapping[str, Any], *, first: int) -> Any:
    years = np.arange(first, first + 40, dtype=np.int64)
    values = np.random.default_rng(first).gumbel(60.0, 12.0, 40)
    provenance = {
        "kind": "table",
        "source": {"provider": "fixture", "recipe": "rx1day", "pathway": "ssp245"},
        **series_fields,
    }
    series = AnnualSeries(years, values, "mm/day", provenance)
    return sample_annual(
        series,
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id="a",
        scenario=label,
    )


def test_the_setup_text_names_a_station() -> None:
    train, test = holdout(_per_year_at("one", {"where": {"station": "S1"}}, first=2000))
    table = compare_scenarios(
        [GIVEN], ScenarioSet([test], reference="one"), design_sample=train
    )
    assert "scenario one at station S1, for asset 'a'" in setup_text(table)


def test_the_setup_text_names_the_nearest_cell() -> None:
    fields = {
        "kind": "nearest_cell",
        "cell": {"latitude": 43.5, "longitude": -79.25},
        "spacing_degrees": {"latitude": 0.25, "longitude": 0.25},
        "asset_location": "point",
    }
    train, test = holdout(_per_year_at("one", fields, first=2000))
    table = compare_scenarios(
        [GIVEN], ScenarioSet([test], reference="one"), design_sample=train
    )
    assert (
        "the nearest grid cell of asset 'a', centred at (43.5, -79.25)"
        in setup_text(table)
    )


def test_many_assets_are_counted_where_the_index_is_read() -> None:
    places = [
        {"owner": f"asset 'a{i}'", "kind": "h3_cell", "cell": "8a"} for i in range(6)
    ]
    assert _read_at(places, "where") == "6 assets, each at its H3 cell"


def test_a_scenarios_record_without_its_blocks_is_refused() -> None:
    train, scenarios, designed = _pair_set()
    metadata = EvaluationMetadata.from_table(compare_scenarios([designed], scenarios))
    document = json.loads(metadata.to_json())
    document["setup"]["scenarios"]["scenarios"]["early"].pop("blocks")
    with pytest.raises(EvaluationError, match="needs its blocks"):
        EvaluationMetadata(**document)


def test_a_scenarios_record_must_describe_its_reference() -> None:
    train, scenarios, designed = _pair_set()
    metadata = EvaluationMetadata.from_table(compare_scenarios([designed], scenarios))
    document = json.loads(metadata.to_json())
    document["setup"]["scenarios"]["reference"] = "nope"
    with pytest.raises(EvaluationError, match="describe its reference"):
        EvaluationMetadata(**document)


def test_setup_text_refuses_a_malformed_scenarios_mapping() -> None:
    train, scenarios, designed = _pair_set()
    metadata = EvaluationMetadata.from_table(compare_scenarios([designed], scenarios))
    setup = json.loads(metadata.to_json())["setup"]
    setup["scenarios"] = "not a mapping"
    with pytest.raises(EvaluationError, match="scenarios has the wrong type"):
        setup_text(setup)


def test_setup_text_reads_a_setup_alone() -> None:
    train, scenarios, designed = _pair_set()
    metadata = EvaluationMetadata.from_table(compare_scenarios([designed], scenarios))
    text = setup_text(metadata.setup)
    assert "Scenario early (the reference)" in text
    assert "Budgets:" not in text


def test_merging_two_drift_tables_needs_one_reference() -> None:
    table, _ = _drift_table()
    first = EvaluationMetadata.from_table(to_drift(table, reference="early"))
    second = EvaluationMetadata.from_table(to_drift(table, reference="late"))
    with pytest.raises(EvaluationError, match="differ in their drift"):
        EvaluationMetadata.merge(first, second)
    merged = EvaluationMetadata.merge(first, first)
    assert merged.setup.drift == first.setup.drift


def test_the_docs_example_runs_on_synthetic_windows() -> None:
    rows_2030s = [_row(horizon) for horizon in (2025, 2030, 2035)]
    rows_2050s = [_row(horizon) for horizon in (2045, 2050, 2055)]

    def block(label: str, rows: Any, seed: int) -> OutcomeSample:
        return _draws(label, rows, seed=seed, n_draws=300)

    designed = design(
        _step(), block("2030s", rows_2030s, 11), MSE(), budget=Budget(ratio=0.5)
    )
    scenarios = ScenarioSet(
        [block("2030s", rows_2030s, 22), block("2050s", rows_2050s, 22)],
        reference="2030s",
    )
    table = compare_scenarios([designed], scenarios)
    drift = to_drift(table)
    assert table.num_rows == 2
    assert drift.num_rows > 0
    assert "Scenario 2030s (the reference)" in setup_text(table)
