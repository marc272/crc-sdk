"""ParametricPlan scopes: `.scopes`, `.dependence`, `.allocation`, `.joint_sample`.

The tests build tables only to check their rows and columns, and that the plan
gives what the functions give; nothing here is a finding about a design.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
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
    AnnualSeries,
    AsRecorded,
    AtPoint,
    AtStation,
    Budget,
    ComonotoneIndexWarning,
    Comonotonic,
    Cover,
    DesignResult,
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
    JointDesignResult,
    KFold,
    LeakageError,
    LinearPayout,
    LinearTrend,
    LossModel,
    MetricOptions,
    NearestCell,
    NoBasisRisk,
    OutcomeSample,
    ParametricPlan,
    RecentYears,
    SampleError,
    StationColumn,
    StepPayout,
    allocate,
    compare,
    design_joint,
    holdout,
    sample_annual,
    sample_curves,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import (
    AssetPortfolio,
    HazardDataset,
    ReturnPeriodExtrapolationWarning,
)

UNTYPED_PLAN: Any = ParametricPlan
RAMP = PiecewiseLinearImpact([40.0, 120.0], [0.0, 0.5])
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
OPTIONS = MetricOptions(betas=(0.9,), return_periods=(10,))
PROVENANCE = SourceProvenance(
    provider="example",
    dataset="synthetic rx1day",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic data",
)
STATIONS = SourceProvenance(
    provider="example",
    dataset="synthetic stations",
    version="1",
    licence="CC0-1.0",
    attribution="Synthetic stations",
)
YEARS = np.arange(1981, 2021)
GRID = ((50.875, 6.875), (50.875, 7.125), (51.125, 6.875), (51.125, 7.125))
SITES = {
    "warehouse-a": (6.96, 50.94),
    "depot-b": (7.10, 51.10),
    "store-c": (7.10, 50.90),
}
VALUES = {"warehouse-a": 2e6, "depot-b": 3e6, "store-c": 1e6}
CELLS = {name: point_to_cell(lon, lat, 7) for name, (lon, lat) in SITES.items()}
ALL = tuple(SITES)
TWO = ALL[:2]


# -- fixtures ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        yield


def _asset_table(ids: Sequence[str] = ALL) -> Any:
    return pa.table(
        {
            "asset_id": list(ids),
            "longitude": [SITES[i][0] for i in ids],
            "latitude": [SITES[i][1] for i in ids],
            "value_eur": [VALUES[i] for i in ids],
            "station_id": [f"S{k}" for k in range(len(ids))],
        }
    )


def _step(asset: str | None = None, index: IndexDefinition = INDEX) -> Cover:
    return Cover(
        index,
        StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        asset_id=asset,
        label="step",
    )


def _linear(asset: str | None = None) -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 120.0, 5e5),
        payout_unit="EUR",
        asset_id=asset,
        label="linear",
    )


def _comparable(table: Any) -> Any:
    """A comparison without its labels and metadata, to compare two routes."""
    return table.drop_columns(["cover_label"]).replace_schema_metadata(None)


# -- per-year values -----------------------------------------------------------


def _grid(seed: int = 1, rise: float = 0.0) -> Any:
    rng = np.random.default_rng(seed)
    count = len(YEARS) * len(GRID)
    return pa.table(
        {
            "year": pa.array(np.repeat(YEARS, len(GRID)), pa.int32()),
            "latitude": np.tile([lat for lat, _ in GRID], len(YEARS)),
            "longitude": np.tile([lon for _, lon in GRID], len(YEARS)),
            "value": rng.gumbel(60.0, 12.0, count)
            + rise * (np.repeat(YEARS, len(GRID)) - 1981.0),
        }
    )


def _per_year(source: Any = None, assets: Any = None, **options: Any) -> ParametricPlan:
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
        _grid() if source is None else source,
        _asset_table() if assets is None else assets,
        **arguments,
    )
    return plan


def _designed_per_year(plan: ParametricPlan) -> ParametricPlan:
    return (
        plan.basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step())
        .objective(MSE(), budget=Budget(ratio=0.8))
        .metrics(OPTIONS)
    )


def _joined(
    source: Any,
    *,
    trend: Any = AsRecorded(),
    basis_risk: Any = None,
    index: Mapping[str, AnnualSeries] | None = None,
    ids: Sequence[str] = ALL,
) -> OutcomeSample:
    """The joined sample by hand, with the plan's generators."""
    model = ImpactNoise(0.3, upper=1.0) if basis_risk is None else basis_risk
    children = np.random.SeedSequence(42).spawn(len(ids))
    return OutcomeSample.join(
        sample_annual(
            AnnualSeries.nearest_cell(
                source,
                longitude=SITES[asset][0],
                latitude=SITES[asset][1],
                unit="mm/day",
                source=PROVENANCE,
            ),
            loss_model=LossModel(RAMP, value=VALUES[asset], value_unit="EUR"),
            basis_risk=model,
            asset_id=asset,
            rng=np.random.default_rng(child),
            index=None if index is None else index[asset],
            trend=trend,
        )
        for asset, child in zip(ids, children, strict=True)
    )


# -- fitted curves ---------------------------------------------------------------


def _hazard_metadata() -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=PROVENANCE,
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 500.0),
    )


def _curve(asset: str, shift: float) -> dict[str, Any]:
    return {
        "cell_index": CELLS[asset],
        "source_id": f"fixture-{asset}",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 50.0 + shift,
        "curve_scale": 12.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "rx1day.parquet"
    metadata = _hazard_metadata()
    table = pa.Table.from_pylist(
        [_curve(asset, 5.0 * k) for k, asset in enumerate(ALL)],
        schema=hazard_arrow_schema(metadata),
    )
    write_hazard_dataset(table, path, metadata, max_workers=1)
    return path


def _request(path: Path, ids: Sequence[str] = ALL) -> Any:
    return (
        HazardDataset.local(path)
        .for_assets(_asset_table(ids))
        .select(horizons=[2050], pathways=["ssp245"])
        .impact(RAMP, name="ramp", value_unit="1", value_semantics="damage ratio")
    )


def _on_curves(path: Path, ids: Sequence[str] = ALL) -> ParametricPlan:
    plan: ParametricPlan = UNTYPED_PLAN.for_evaluation(
        _request(path, ids), value_column="value_eur", value_unit="EUR"
    )
    return plan


def _designed_on_curves(path: Path, n_draws: int = 400) -> ParametricPlan:
    return (
        _on_curves(path)
        .monte_carlo(n_draws, seed=42)
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(_step())
        .objective(MSE(), budget=Budget(ratio=0.8))
        .metrics(OPTIONS)
    )


def _curve_loss_models() -> dict[str, LossModel]:
    assets = AssetPortfolio(_asset_table())
    return {
        asset: LossModel(
            RAMP,
            value_column="value_eur",
            value_unit="EUR",
            context=TransformContext(cell=CELLS[asset]),
        ).for_asset(assets, asset)
        for asset in ALL
    }


def _drawn(
    path: Path, dependence: Any, seed: int = 42, n_draws: int = 400
) -> OutcomeSample:
    """The joint sample by hand, with the plan's generators."""
    request = _request(path)
    rows = {row["asset_id"]: row for row in request.asset_curves().to_pylist()}
    children = np.random.SeedSequence(seed).spawn(len(ALL))
    metadata = request.dataset.metadata()
    return sample_curves(
        {asset: rows[asset] for asset in ALL},
        dependence=dependence,
        metadata=dict.fromkeys(ALL, metadata),
        n_draws=n_draws,
        rngs={
            asset: np.random.default_rng(child)
            for asset, child in zip(ALL, children, strict=True)
        },
        loss_models=_curve_loss_models(),
        basis_risk=ImpactNoise(0.3, upper=1.0),
    )


def _reference() -> OutcomeSample:
    """A sample of per-year rows to take the ranks from."""
    return _joined(_grid(5), basis_risk=NoBasisRisk())


# -- the default ---------------------------------------------------------------


def test_the_default_is_individual_and_behaviour_is_unchanged(tmp_path: Path) -> None:
    named = _designed_per_year(_per_year()).split(Holdout(0.7))
    stated = named.scopes(Individual())
    default_result = named.write_parquet(tmp_path / "default.parquet")
    stated_result = stated.write_parquet(tmp_path / "stated.parquet")
    assert _comparable(default_result.table).equals(_comparable(stated_result.table))
    levels = set(default_result.table.column("level").to_pylist())
    assert levels == {"asset"}
    assert set(default_result.table.column("scope").to_pylist()) == {"individual"}
    assert all(isinstance(d, DesignResult) for d in default_result.designs)
    assert all(len(s.asset_ids) == 1 for s in default_result.samples.values())
    assert default_result.joint_sample is None
    assert default_result.fresh_joint_sample is None


def test_an_individual_plan_states_no_scope_keys() -> None:
    default: Any = _designed_per_year(_per_year()).explain(format="json")
    stated: Any = (
        _designed_per_year(_per_year()).scopes(Individual()).explain(format="json")
    )
    assert default == stated
    assert not {"scopes", "dependence", "allocation"} & set(default)


# -- builders and refusals -----------------------------------------------------


@pytest.mark.parametrize(
    ("scopes", "error", "match"),
    [
        ((), ValueError, "at least one"),
        ((Individual(), Individual()), ValueError, "given twice"),
        ((Joint(), Joint()), ValueError, "given twice"),
        (("joint",), TypeError, "Individual"),
    ],
)
def test_scopes_refuses(scopes: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        _per_year().scopes(*scopes)


def test_scopes_replaces_the_ones_before() -> None:
    plan = _per_year().scopes(Joint()).scopes(Individual())
    assert plan.scope_list == (Individual(),)


def test_joint_needs_two_assets_when_they_are_known(dataset: Path) -> None:
    with pytest.raises(ValueError, match="several assets"):
        _per_year(assets=_asset_table(ALL[:1])).scopes(Joint())
    with pytest.raises(ValueError, match="several assets"):
        _on_curves(dataset, ALL[:1]).scopes(Joint())


def test_joint_needs_two_assets_at_execution(tmp_path: Path) -> None:
    path = tmp_path / "one.parquet"
    pq.write_table(_asset_table(ALL[:1]), path)
    plan = _per_year(assets=path).basis_risk(NoBasisRisk()).scopes(Joint())
    with pytest.raises(ValueError, match="several assets"):
        plan.samples()
    with pytest.raises(ValueError, match="several assets"):
        plan.joint_sample()


def _no_dependence(path: Path) -> ParametricPlan:
    return (
        _on_curves(path)
        .monte_carlo(100, seed=1)
        .basis_risk(NoBasisRisk())
        .covers(_step())
        .objective(MSE())
        .scopes(Joint())
    )


@pytest.mark.parametrize(
    "execute",
    [
        lambda plan, _: plan.explain(),
        lambda plan, _: plan.explain(format="json"),
        lambda plan, _: plan.samples(),
        lambda plan, _: plan.joint_sample(),
        lambda plan, _: plan.designs(),
        lambda plan, path: plan.write_parquet(path.parent / "out.parquet"),
    ],
)
def test_joint_on_curves_without_a_dependence_model_raises(
    dataset: Path, execute: Any
) -> None:
    with pytest.raises(ValueError, match=r"\.dependence\(Independent\(\)\)"):
        execute(_no_dependence(dataset), dataset)


def test_joint_needs_no_dependence_on_per_year_values() -> None:
    joint = _designed_per_year(_per_year()).scopes(Joint()).joint_sample()
    assert joint.provenance.dependence == {"kind": "data"}


@pytest.mark.parametrize(
    "model", [Independent(), Comonotonic()], ids=["independent", "comonotonic"]
)
def test_dependence_on_a_per_year_plan_raises(model: Any) -> None:
    with pytest.raises(ValueError, match="the data's own"):
        _per_year().dependence(model)
    indexed = _per_year().index(
        _stations(),
        location=StationColumn("station_id"),
        unit="mm/day",
        provenance=STATIONS,
    )
    with pytest.raises(ValueError, match="the data's own"):
        indexed.dependence(model)


def test_dependence_refuses_another_type(dataset: Path) -> None:
    untyped: Any = _on_curves(dataset)
    with pytest.raises(TypeError, match="EmpiricalCopula"):
        untyped.dependence("comonotonic")


def test_dependence_without_joint_raises_when_the_plan_runs(dataset: Path) -> None:
    plan = (
        _on_curves(dataset)
        .monte_carlo(100, seed=1)
        .basis_risk(NoBasisRisk())
        .dependence(Comonotonic())
    )
    for run in (plan.explain, plan.samples):
        with pytest.raises(ValueError, match="only Joint"):
            run()


def test_allocation_without_joint_raises_when_the_plan_runs() -> None:
    plan = _per_year().basis_risk(NoBasisRisk()).allocation("value")
    for run in (plan.explain, plan.samples):
        with pytest.raises(ValueError, match=r"scopes\(Joint\(\)\)"):
            run()


def test_allocation_refuses_another_rule() -> None:
    untyped: Any = _per_year()
    with pytest.raises(ValueError, match="value"):
        untyped.allocation("equal")


def test_a_cover_written_for_one_asset_raises_under_joint() -> None:
    plan = _per_year().basis_risk(NoBasisRisk()).covers(_step(ALL[0])).scopes(Joint())
    with pytest.raises(ValueError, match="design_joint"):
        plan.explain()
    with pytest.raises(ValueError, match="design_joint"):
        plan.objective(MSE()).designs()


def test_joint_sample_needs_joint() -> None:
    with pytest.raises(ValueError, match=r"scopes\(Joint\(\)\)"):
        _per_year().basis_risk(NoBasisRisk()).joint_sample()


@pytest.mark.parametrize("rule", [Holdout(0.7), KFold(4)], ids=["holdout", "kfold"])
def test_an_empirical_copula_cannot_be_split_by_rows(dataset: Path, rule: Any) -> None:
    copula = EmpiricalCopula(_reference())
    plan = _on_curves(dataset).monte_carlo(80, seed=1).scopes(Joint())
    with pytest.raises(SampleError, match="FreshDraws") as first:
        plan.dependence(copula).split(rule)
    with pytest.raises(SampleError, match="FreshDraws") as second:
        plan.split(rule).dependence(copula)
    assert "reference-year blocks" in str(first.value)
    assert str(first.value) == str(second.value)


def test_an_empirical_copula_takes_fresh_draws(dataset: Path) -> None:
    plan = (
        _on_curves(dataset)
        .monte_carlo(80, seed=1)
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .scopes(Joint())
        .dependence(EmpiricalCopula(_reference()))
        .split(FreshDraws(120, seed=2))
    )
    result = plan.write_parquet(dataset.parent / "copula.parquet")
    assert result.joint_sample is not None
    assert result.joint_sample.provenance.dependence is not None
    assert result.joint_sample.provenance.dependence["k"] == 2
    assert result.fresh_joint_sample is not None
    assert result.fresh_joint_sample.provenance.dependence is not None
    assert result.fresh_joint_sample.provenance.dependence["k"] == 3
    assert "EmpiricalCopula, a stated assumption" in setup_text(result.metadata)


def test_an_empirical_copula_checks_the_draw_counts_before_reading(
    dataset: Path,
) -> None:
    copula = EmpiricalCopula(_reference())
    plan = _on_curves(dataset).scopes(Joint()).dependence(copula)
    with pytest.raises(SampleError, match="nearest valid counts are 80 and 120"):
        plan.monte_carlo(100, seed=1).explain()
    fresh = plan.monte_carlo(80, seed=1).split(FreshDraws(50, seed=2))
    with pytest.raises(SampleError, match="nearest valid counts are 40 and 80"):
        fresh.explain()


def test_an_empirical_copula_must_hold_the_plan_s_assets(dataset: Path) -> None:
    plan = (
        _on_curves(dataset, TWO)
        .monte_carlo(80, seed=1)
        .scopes(Joint())
        .dependence(EmpiricalCopula(_reference()))
    )
    with pytest.raises(SampleError, match="must be the same"):
        plan.explain()


# -- explain ---------------------------------------------------------------------


class _Unreadable:
    """A source plan that fails the moment anything reads it."""

    def annual_extremes(self) -> Any:
        raise AssertionError("explain() read the source")


def test_explain_states_the_scopes_dependence_and_allocation_without_io() -> None:
    plan = (
        _designed_per_year(_per_year(_Unreadable()))
        .scopes(Joint())
        .allocation("loss")
        .split(Holdout(0.7))
    )
    document: Any = plan.explain(format="json")
    assert document["scopes"] == [{"kind": "joint"}]
    assert document["dependence"] is None
    assert document["allocation"] == "loss"
    assert [s["asset_ids"] for s in document["setup"]["samples"]] == [list(ALL)]
    assert document["setup"]["samples"][0]["dependence"] == {"kind": "data"}
    for entry in document["setup"]["evaluations"]:
        assert entry["level"] == ["total", "asset_by_loss"]
        assert entry["scope"] == ["joint"]
        assert entry["allocation"] == "loss"
    first = setup_text(document).splitlines()[0]
    assert first.startswith(
        "Level: total, the sum over warehouse-a, depot-b and store-c in each draw"
    )
    assert "Scope: joint, one objective and one budget on the owner's total" in first
    assert "Dependence between the assets: the data's own" in setup_text(document)


def test_explain_states_a_dependence_model_on_curves(dataset: Path) -> None:
    plan = _designed_on_curves(dataset).scopes(Joint()).dependence(Comonotonic())
    document: Any = plan.explain(format="json")
    assert document["dependence"] == {"kind": "comonotonic"}
    sample = document["setup"]["samples"][0]
    assert sample["dependence"] == {"kind": "comonotonic"}
    assert sample["notes"] == [
        "The dependence between the assets is an assumption: they are comonotonic, "
        "so draws of equal rank share a row."
    ]
    text = setup_text(document)
    assert "Dependence between the assets: Comonotonic(), a stated assumption" in text


def test_explain_states_the_draws_per_reference_year(dataset: Path) -> None:
    plan = (
        _on_curves(dataset)
        .monte_carlo(120, seed=1)
        .scopes(Joint())
        .dependence(EmpiricalCopula(_reference()))
    )
    document: Any = plan.explain(format="json")
    assert document["setup"]["samples"][0]["dependence"]["k"] == 3
    assert ", 3 draws per year" in setup_text(document)


_PLANNED_MSE = (
    "objective: MSE, E[(L - payout)^2]; pricing: the pure premium, P = E[payout]; "
    "budget: premium at most 0.8 x E[L] of the {base}; no limit; design premium, "
    "route and non-uniqueness report: not known yet."
)
_PLANNED_JOINT = (
    "Design step, planned: scope: joint, one objective, one budget and one premium "
    "on the total of warehouse-a, depot-b and store-c, with one cover per asset; "
    + _PLANNED_MSE.format(base="total of the design sample")
)
_PLANNED_INDIVIDUAL = "Design step, planned: " + _PLANNED_MSE.format(
    base="design sample"
)


def _designs(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("Design")]


def test_setup_text_states_a_planned_joint_design(dataset: Path) -> None:
    per_year: Any = (
        _designed_per_year(_per_year()).scopes(Joint()).explain(format="json")
    )
    assert _designs(setup_text(per_year)) == [_PLANNED_JOINT]
    on_curves: Any = (
        _designed_on_curves(dataset)
        .scopes(Joint())
        .dependence(Comonotonic())
        .explain(format="json")
    )
    assert _designs(setup_text(on_curves)) == [_PLANNED_JOINT]


def test_setup_text_states_each_scope_a_plan_designs_in() -> None:
    plan: Any = _designed_per_year(_per_year()).scopes(Individual(), Joint())
    assert _designs(setup_text(plan.explain(format="json"))) == [
        _PLANNED_INDIVIDUAL,
        _PLANNED_JOINT,
    ]


def test_a_planned_joint_design_says_the_assets_are_not_known_yet(
    tmp_path: Path,
) -> None:
    path = tmp_path / "assets.parquet"
    pq.write_table(_asset_table(), path)
    plan: Any = _designed_per_year(_per_year(assets=path)).scopes(Joint())
    document = plan.explain(format="json")
    assert document["setup"]["samples"][0]["asset_ids"] == []
    assert "on the total of its assets, with one cover per asset" in setup_text(
        document
    )


def test_a_planned_and_a_written_joint_design_state_the_same_terms(
    tmp_path: Path,
) -> None:
    plan = _designed_per_year(_per_year()).scopes(Joint())
    planned: Any = plan.explain(format="json")
    result = plan.write_parquet(tmp_path / "comparison.parquet")
    written = _designs(setup_text(result.metadata))[0]
    terms = _PLANNED_JOINT.split(": ", 1)[1].split("; design premium, route")[0]
    assert terms in written
    assert terms in _designs(setup_text(planned))[0]


def test_explain_states_both_scopes_when_the_plan_holds_both() -> None:
    plan = _designed_per_year(_per_year()).scopes(Individual(), Joint())
    document: Any = plan.explain(format="json")
    assert document["scopes"] == [{"kind": "individual"}, {"kind": "joint"}]
    entry = document["setup"]["evaluations"][0]
    assert entry["level"] == ["asset", "total"]
    assert entry["scope"] == ["individual", "joint"]


def test_explain_with_assets_in_a_file_states_one_sample(tmp_path: Path) -> None:
    path = tmp_path / "assets.parquet"
    pq.write_table(_asset_table(), path)
    document: Any = (
        _per_year(assets=path).basis_risk(NoBasisRisk()).scopes(Joint())
    ).explain(format="json")
    assert [s["asset_ids"] for s in document["setup"]["samples"]] == [[]]


# -- the function route: per-year values -----------------------------------------


def test_a_joint_plan_on_per_year_values_equals_the_function_route(
    tmp_path: Path,
) -> None:
    source = _grid()
    budget = Budget(ratio=0.8)
    plan = (
        _designed_per_year(_per_year(source))
        .covers(_step(), _linear())
        .objective(MSE(), ExpectedUncovered(), budget=budget)
        .split(Holdout(0.7))
        .scopes(Joint())
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    joined = _joined(source)
    assert result.joint_sample is not None
    assert result.joint_sample.content_hash == joined.content_hash
    assert result.joint_sample.provenance.dependence == {"kind": "data"}
    train, test = holdout(joined, 0.7)
    templates = [_step(asset) for asset in ALL]
    entries: list[Any] = [
        design_joint(templates, train, objective, budget=budget)
        for objective in (MSE(), ExpectedUncovered())
    ]
    entries.append({asset: _linear(asset) for asset in ALL})
    by_hand = pa.concat_tables(
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
    assert _comparable(result.table).equals(_comparable(by_hand))
    for planned, made in zip(result.designs, entries[:2], strict=True):
        assert isinstance(planned, JointDesignResult)
        assert planned.design_id == made.design_id
        assert planned.covers_hash == made.covers_hash
        assert planned.parameters == made.parameters
    labels = result.table.column("cover_label").to_pylist()
    assert labels[0] == "step (mse)"
    assert "step (expected_uncovered)" in labels
    assert "linear" in labels


def test_a_joint_plan_s_designs_equal_write_parquet_s() -> None:
    plan = _designed_per_year(_per_year()).split(KFold(3)).scopes(Joint())
    designs = plan.designs()
    assert [type(d) for d in designs] == [JointDesignResult] * 3
    assert all(d.design_sample.asset_ids == ALL for d in designs)
    assert all(isinstance(d.covers["depot-b"], Cover) for d in designs)
    assert designs[0].covers[ALL[0]].label == "step (mse)"


def test_each_asset_s_view_is_its_own_column_of_the_joint_sample() -> None:
    plan = _designed_per_year(_per_year()).scopes(Joint())
    views = plan.samples()
    assert list(views) == list(ALL)
    individual = _designed_per_year(_per_year()).samples()
    joint = plan.joint_sample()
    for asset in ALL:
        assert views[asset].asset_ids == (asset,)
        assert views[asset].content_hash == joint.asset(asset).content_hash
        assert views[asset].content_hash == individual[asset].content_hash
    # The view says where it came from.
    assert views[ALL[0]].provenance.selection[-1]["kind"] == "asset"


def test_different_years_raise_through_join() -> None:
    source = _grid()
    years = source.column("year").to_numpy()
    longitudes = source.column("longitude").to_numpy()
    latitudes = source.column("latitude").to_numpy()
    # warehouse-a's cell has no value for 2000.
    gap = (years == 2000) & (longitudes == 6.875) & (latitudes == 50.875)
    values = np.ma.array(source.column("value").to_numpy(), mask=gap)
    holed = source.set_column(
        source.column_names.index("value"),
        "value",
        pa.array(values, pa.float64()),
    )
    plan = (
        _per_year(holed, missing="drop_year").basis_risk(NoBasisRisk()).scopes(Joint())
    )
    with pytest.raises(SampleError, match="identical period"):
        plan.joint_sample()


# -- the function route: fitted curves -------------------------------------------


@pytest.mark.parametrize(
    "dependence", [Independent(), Comonotonic()], ids=["independent", "comonotonic"]
)
def test_a_joint_plan_on_curves_equals_the_function_route(
    dataset: Path, dependence: Any
) -> None:
    budget = Budget(ratio=0.8)
    plan = (
        _designed_on_curves(dataset)
        .split(FreshDraws(300, seed=7))
        .scopes(Joint())
        .dependence(dependence)
    )
    result = plan.write_parquet(dataset.parent / "out.parquet")
    drawn = _drawn(dataset, dependence)
    fresh = _drawn(dataset, dependence, seed=7, n_draws=300)
    assert result.joint_sample is not None and result.fresh_joint_sample is not None
    assert result.joint_sample.content_hash == drawn.content_hash
    assert result.fresh_joint_sample.content_hash == fresh.content_hash
    assert result.joint_sample.provenance.dependence == dependence.to_document()
    made = design_joint([_step(a) for a in ALL], drawn, MSE(), budget=budget)
    planned = result.designs[0]
    assert isinstance(planned, JointDesignResult)
    assert planned.design_id == made.design_id
    assert planned.covers_hash == made.covers_hash
    assert planned.parameters == made.parameters
    by_hand = pa.concat_tables(
        [
            compare(
                [made], drawn, split="in_sample", design_sample=drawn, options=OPTIONS
            ),
            compare([made], fresh, split="out_of_sample", options=OPTIONS),
        ]
    )
    assert _comparable(result.table).equals(_comparable(by_hand))
    methods = {e["method"] for e in result.metadata.setup.evaluations}
    assert methods == {"in_sample", "fresh_draws"}
    assert result.samples[ALL[1]].content_hash == drawn.asset(ALL[1]).content_hash
    assert result.fresh_samples is not None
    assert result.fresh_samples[ALL[1]].content_hash == fresh.asset(ALL[1]).content_hash


def test_an_empirical_copula_plan_equals_the_function_route(dataset: Path) -> None:
    reference = _reference()
    copula = EmpiricalCopula(reference)
    n_draws = 3 * len(reference)
    plan = (
        _designed_on_curves(dataset, n_draws)
        .split(FreshDraws(n_draws, seed=7))
        .scopes(Joint())
        .dependence(copula)
    )
    result = plan.write_parquet(dataset.parent / "out.parquet")
    drawn = _drawn(dataset, copula, n_draws=n_draws)
    assert result.joint_sample is not None
    assert result.joint_sample.content_hash == drawn.content_hash
    made = design_joint([_step(a) for a in ALL], drawn, MSE(), budget=Budget(ratio=0.8))
    assert result.designs[0].parameters == made.parameters


def test_a_joint_holdout_of_a_comonotonic_sample_is_refused(dataset: Path) -> None:
    # Comonotonic() placed each moved asset's draws by the ranks of every row,
    # so the design block's rows depend on the held-out block's draws.
    plan = (
        _designed_on_curves(dataset)
        .split(Holdout(0.75))
        .scopes(Joint())
        .dependence(Comonotonic())
    )
    with pytest.raises(LeakageError, match="depend on the other block"):
        plan.write_parquet(dataset.parent / "out.parquet")
    train, test = holdout(_drawn(dataset, Comonotonic()), 0.75)
    with pytest.raises(LeakageError, match="depend on the other block"):
        _check_out_of_sample(train, test)
    # One asset's blocks still are out of sample.
    _check_out_of_sample(train.asset(ALL[1]), test.asset(ALL[1]))


def test_each_curve_records_its_spatial_match_under_its_asset(dataset: Path) -> None:
    plan = _designed_on_curves(dataset).scopes(Joint()).dependence(Independent())
    joint = plan.joint_sample()
    details = joint.provenance.sources[0]["details"]
    assert dict(details["spatial_match_by_asset"]) == dict.fromkeys(ALL, "h3_cell")
    assert details["file_sha256"].startswith("sha256:")


# -- trends ----------------------------------------------------------------------


def test_a_joint_design_under_linear_trend_fits_on_each_fold_s_design_years(
    tmp_path: Path,
) -> None:
    source = _grid(rise=0.4)
    plan = (
        _designed_per_year(_per_year(source, trend=LinearTrend()))
        .split(KFold(3))
        .scopes(Joint())
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    blocks = np.array_split(YEARS, 3)
    by_hand: list[Any] = []
    for fold, (block, planned) in enumerate(zip(blocks, result.designs, strict=True)):
        fit_years = [int(year) for year in YEARS if year not in block]
        trend = planned.design_sample.provenance.to_document()["trend"]
        assert trend["kind"] == "linear_trend"
        assert trend["fit_years"] == fit_years
        assert trend["reference_year"] == 2020
        assert sorted({fit["asset_id"] for fit in trend["fits"]}) == sorted(ALL)
        train, test = year_block_kfold(
            _joined(source, trend=LinearTrend(fit_years=fit_years)), 3
        )[fold]
        assert planned.design_sample.content_hash == train.content_hash
        made = design_joint(
            [_step(a) for a in ALL], train, MSE(), budget=Budget(ratio=0.8)
        )
        assert planned.design_id == made.design_id
        assert planned.parameters == made.parameters
        by_hand += [
            compare(
                [made],
                train,
                split="in_sample",
                fold=fold,
                design_sample=train,
                options=OPTIONS,
            ),
            compare([made], test, split="out_of_sample", fold=fold, options=OPTIONS),
        ]
    assert _comparable(result.table).equals(_comparable(pa.concat_tables(by_hand)))
    entries = result.metadata.setup.evaluations
    held_out = [e for e in entries if e["split"] == "out_of_sample"]
    assert {e["method"] for e in held_out} == {"kfold"}
    assert {e["details"]["k"] for e in held_out} == {3}
    assert {e["leakage_check"] for e in held_out} == {"passed"}


def test_the_joint_sample_stays_all_years_under_a_split() -> None:
    source = _grid(rise=0.4)
    plan = (
        _designed_per_year(_per_year(source, trend=LinearTrend()))
        .split(Holdout(0.7))
        .scopes(Joint())
    )
    joint = plan.joint_sample()
    assert joint.content_hash == _joined(source, trend=LinearTrend()).content_hash
    assert joint.provenance.to_document()["trend"]["fit_years"] == [
        int(year) for year in YEARS
    ]


def test_a_changed_held_out_year_leaves_the_joint_design_unchanged() -> None:
    source = _grid(rise=0.4)
    values = source.column("value").to_numpy().copy()
    values[source.column("year").to_numpy() == 2020] += 80.0
    changed = source.set_column(
        source.column_names.index("value"), "value", pa.array(values, pa.float64())
    )

    def designs(table: Any) -> tuple[Any, ...]:
        return (
            _designed_per_year(_per_year(table, trend=LinearTrend()))
            .split(Holdout(0.7))
            .scopes(Joint())
            .designs()
        )

    (first,), (second,) = designs(source), designs(changed)
    assert first.design_id == second.design_id
    assert first.parameters == second.parameters


def test_recent_years_are_joined_once_and_then_split() -> None:
    source = _grid()
    plan = (
        _designed_per_year(_per_year(source, trend=RecentYears(20)))
        .split(Holdout(0.5))
        .scopes(Joint())
    )
    design = plan.designs()[0]
    train, _ = holdout(_joined(source, trend=RecentYears(20)), 0.5)
    assert design.design_sample.content_hash == train.content_hash


# -- an index read elsewhere -----------------------------------------------------


def _stations() -> Any:
    rng = np.random.default_rng(9)
    names = [f"S{k}" for k in range(len(ALL))]
    return pa.table(
        {
            "station": np.repeat(names, len(YEARS)),
            "year": pa.array(np.tile(YEARS, len(names)), pa.int32()),
            "value": rng.gumbel(65.0, 10.0, len(YEARS) * len(names)),
        }
    )


def test_joint_composes_with_an_index_read_at_a_station(tmp_path: Path) -> None:
    source = _grid()
    stations = _stations()
    plan = (
        _designed_per_year(
            _per_year(source).index(
                stations,
                location=StationColumn("station_id"),
                unit="mm/day",
                provenance=STATIONS,
            )
        )
        .split(Holdout(0.7))
        .scopes(Joint())
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    planned = result.designs[0]
    assert isinstance(planned, JointDesignResult)
    locations = {a: planned.covers[a].index.location for a in ALL}
    assert locations == {a: AtStation(f"S{k}") for k, a in enumerate(ALL)}
    series = {
        asset: AnnualSeries.from_table(
            stations,
            where={"station": f"S{k}"},
            unit="mm/day",
            source=STATIONS,
        )
        for k, asset in enumerate(ALL)
    }
    joined = _joined(source, index=series, basis_risk=ImpactNoise(0.3, upper=1.0))
    train, _ = holdout(joined, 0.7)
    assert planned.design_sample.content_hash == train.content_hash
    templates = [
        _step(
            asset,
            IndexDefinition(
                "rx1day",
                "mm/day",
                location=AtStation(f"S{k}"),
                aggregation="annual_maximum",
            ),
        )
        for k, asset in enumerate(ALL)
    ]
    made = design_joint(templates, train, MSE(), budget=Budget(ratio=0.8))
    assert planned.design_id == made.design_id
    assert planned.covers_hash == made.covers_hash
    assert planned.parameters == made.parameters
    documents = {entry.hash for entry in result.metadata.covers}
    assert planned.covers[ALL[0]].definition_hash in documents
    roles = {entry["role"] for entry in result.metadata.sources}
    assert roles == {"hazard", "index"}


def test_joint_binds_each_asset_to_its_own_cell() -> None:
    plan = _designed_per_year(_per_year()).index(
        _grid(3), location=NearestCell(), unit="mm/day", provenance=STATIONS
    )
    design = plan.scopes(Joint()).designs()[0]
    locations = {a: design.covers[a].index.location for a in ALL}
    assert all(isinstance(location, AtPoint) for location in locations.values())
    assert len({(p.longitude, p.latitude) for p in locations.values()}) == 3


# -- rows, allocation, both scopes -----------------------------------------------


def test_joint_rows_are_the_total_only_without_an_allocation(tmp_path: Path) -> None:
    result = (
        _designed_per_year(_per_year())
        .split(Holdout(0.7))
        .scopes(Joint())
        .write_parquet(tmp_path / "out.parquet")
    )
    rows = result.table.to_pylist()
    assert {row["level"] for row in rows} == {"total"}
    assert {row["scope"] for row in rows} == {"joint"}
    assert {row["asset_id"] for row in rows} == {None}
    assert len(rows) == 2
    design = result.designs[0]
    assert all(row["cover_hash"] == design.covers_hash for row in rows)
    assert all(row["design_id"] == design.design_id for row in rows)
    assert all(row["premium_basis"] == "design" for row in rows)
    stored = EvaluationMetadata.from_table(pq.read_table(result.output))
    assert stored == result.metadata
    assert stored.assets["count"] == len(ALL)
    # No rule, no allocated rows and no second file.
    assert result.allocated is None and result.allocated_output is None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.parquet"]


@pytest.mark.parametrize("rule", ["value", "loss"])
def test_an_allocation_reads_the_loss_model_s_values(tmp_path: Path, rule: Any) -> None:
    source = _grid()
    plan = (
        _designed_per_year(_per_year(source))
        .split(Holdout(0.7))
        .scopes(Joint())
        .allocation(rule)
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    level = f"asset_by_{rule}"
    # The comparison holds the owner's total; the allocated rows a table of their own.
    assert {row["level"] for row in result.table.to_pylist()} == {"total"}
    assert result.metadata.setup.evaluations[0]["allocation"] is None
    allocated = result.allocated
    assert result.allocated_output == tmp_path / "out.allocated.parquet"
    rows = allocated.to_pylist()
    assert {row["level"] for row in rows} == {level}
    assert [row["asset_id"] for row in rows[:3]] == list(ALL)
    train, _ = holdout(_joined(source), 0.7)
    made = design_joint([_step(a) for a in ALL], train, MSE(), budget=Budget(ratio=0.8))
    by_hand = allocate(
        [made],
        train,
        split="in_sample",
        options=OPTIONS,
        allocation=rule,
        values=VALUES,
    )
    in_sample = allocated.filter(pc.equal(allocated["split"], "in_sample"))
    assert _comparable(in_sample).equals(_comparable(by_hand))
    stored = EvaluationMetadata.from_table(pq.read_table(result.allocated_output))
    assert stored == EvaluationMetadata.from_table(allocated)
    written = stored.setup.evaluations[0]
    assert written["allocation"] == rule
    assert written["level"] == [level]
    assert stored.assets["values_sha256"] is not None
    assert pq.read_table(result.allocated_output).equals(allocated)


def test_an_allocation_without_a_value_column_shares_by_the_damage_ratios(
    tmp_path: Path,
) -> None:
    source = _grid()
    plan = (
        _per_year(source, value_column=None, value_unit="1")
        .basis_risk(ImpactNoise(0.3, upper=1.0))
        .covers(
            Cover(
                INDEX,
                StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                payout_unit="1",
            )
        )
        .objective(MSE())
        .metrics(OPTIONS)
        .scopes(Joint())
        .allocation("value")
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    children = np.random.SeedSequence(42).spawn(len(ALL))
    sample = OutcomeSample.join(
        sample_annual(
            AnnualSeries.nearest_cell(
                source,
                longitude=SITES[asset][0],
                latitude=SITES[asset][1],
                unit="mm/day",
                source=PROVENANCE,
            ),
            loss_model=LossModel(RAMP, value_unit="1"),
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id=asset,
            rng=np.random.default_rng(child),
        )
        for asset, child in zip(ALL, children, strict=True)
    )
    made = design_joint(
        [
            Cover(
                INDEX,
                StepPayout([60.0, 80.0, 100.0], [Free(), Free(), Free()]),
                payout_unit="1",
                asset_id=asset,
            )
            for asset in ALL
        ],
        sample,
        MSE(),
    )
    by_hand = allocate(
        [made],
        sample,
        split="in_sample",
        options=OPTIONS,
        allocation="value",
        values=dict.fromkeys(ALL, 1.0),
    )
    assert _comparable(result.allocated).equals(_comparable(by_hand))
    allocated = EvaluationMetadata.from_table(result.allocated)
    assert allocated.assets["values_sha256"] is not None
    assert result.metadata.assets["values_sha256"] is None


def test_an_allocation_needs_a_joint_design(tmp_path: Path) -> None:
    plan = (
        _per_year()
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .scopes(Joint())
        .allocation("value")
    )
    with pytest.raises(EvaluationError, match="no entry is a joint design"):
        plan.write_parquet(tmp_path / "out.parquet")


def test_a_resolved_cover_under_joint_is_a_set_of_contracts(tmp_path: Path) -> None:
    result = (
        _per_year()
        .basis_risk(NoBasisRisk())
        .covers(_linear())
        .metrics(OPTIONS)
        .scopes(Joint())
        .write_parquet(tmp_path / "out.parquet")
    )
    rows = result.table.to_pylist()
    assert [(row["asset_id"], row["level"]) for row in rows] == [
        *[(asset, "asset") for asset in ALL],
        (None, "total"),
    ]
    assert {row["scope"] for row in rows} == {"individual"}
    assert rows[-1]["premium"] == pytest.approx(sum(r["premium"] for r in rows[:-1]))


def test_a_plan_holding_both_scopes_keeps_their_rows_apart(tmp_path: Path) -> None:
    plan = (
        _designed_per_year(_per_year())
        .split(Holdout(0.7))
        .scopes(Individual(), Joint())
        .allocation("value")
    )
    result = plan.write_parquet(tmp_path / "out.parquet")
    kinds = [type(d) for d in result.designs]
    assert kinds == [JointDesignResult, *[DesignResult] * len(ALL)]
    rows = result.table.to_pylist()
    joint = {(row["level"], row["asset_id"]) for row in rows if row["scope"] == "joint"}
    individual = {
        (row["level"], row["asset_id"]) for row in rows if row["scope"] == "individual"
    }
    assert joint == {("total", None)}
    assert individual == {("total", None), *{("asset", a) for a in ALL}}
    # The allocated rows are in a table of their own, beside no asset row.
    allocated = result.allocated.to_pylist()
    assert {(row["level"], row["asset_id"]) for row in allocated} == {
        ("asset_by_value", a) for a in ALL
    }
    assert {row["scope"] for row in allocated} == {"joint"}
    joint_design = result.designs[0]
    train = joint_design.design_sample
    for asset, each in zip(ALL, result.designs[1:], strict=True):
        assert each.design_sample.content_hash == train.asset(asset).content_hash
        assert each.cover.label == "step (mse)"
    # A set of designs is no design: only the joint design has an id at the total.
    assert {row["design_id"] for row in rows if row["level"] == "total"} == {
        joint_design.design_id,
        None,
    }


# -- the setup text, on real output ----------------------------------------------


def _note_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("Sample notes:"))


def _curves_of(ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {asset: _curve_row(asset, 2.0 * k) for k, asset in enumerate(ids)}


def _curve_row(asset: str, shift: float) -> dict[str, Any]:
    return {
        "cell_index": point_to_cell(7.0 + 0.3 * shift, 51.0, 7),
        "source_id": f"fixture-{asset}",
        "hazard_name": "rx1day",
        "horizon": 2050,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": 50.0 + shift,
        "curve_scale": 12.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def _sampled(ids: Sequence[str], n_draws: int = 600) -> OutcomeSample:
    children = np.random.SeedSequence(11).spawn(len(ids))
    return sample_curves(
        _curves_of(ids),
        dependence=Comonotonic(),
        metadata=dict.fromkeys(ids, _hazard_metadata()),
        n_draws=n_draws,
        rngs={a: np.random.default_rng(c) for a, c in zip(ids, children, strict=True)},
        loss_models={a: LossModel(RAMP, value=1e6, value_unit="EUR") for a in ids},
        basis_risk=NoBasisRisk(),
    )


def _designed_table(sample: OutcomeSample) -> Any:
    made = design_joint(
        [_step(asset) for asset in sample.asset_ids],
        sample,
        MSE(),
        budget=Budget(ratio=0.8),
    )
    return compare(
        [made], sample, split="in_sample", design_sample=sample, options=OPTIONS
    )


def _counts(sample: OutcomeSample, side: str) -> dict[str, int]:
    parts = sample.provenance.parts
    assert parts is not None
    return {
        asset: int(parts[asset]["details"]["outside_support"][side])
        for asset in sample.asset_ids
    }


def test_sample_curves_notes_collapse_in_the_setup_text() -> None:
    sample = _sampled(("a", "b", "c"))
    line = _note_line(setup_text(_designed_table(sample)))
    below = _counts(sample, "below")
    assert (
        f"{below['a']} of 600 draws in the design sample for a, "
        f"{below['b']} of 600 in the design sample for b and "
        f"{below['c']} of 600 in the design sample for c are more frequent than "
        "the shortest fitted return period (2 years): the curve is extrapolated "
        "there."
    ) in line
    assert line.count("more frequent than") == 1
    assert "Asset a: " not in line
    # The note every asset shares, word for word, is written once, naming them.
    assert (
        "Assets a, b and c: No basis-risk model: the index and the loss are comonotone."
    ) in line
    assert line.count("comonotone.") == 1
    # The statement of the model comes first, once.
    statement = (
        "The dependence between the assets is an assumption: they are comonotonic"
    )
    assert line.count(statement) == 1
    assert line.index(statement) == len("Sample notes: ")


def test_counted_notes_of_many_assets_give_the_range_of_the_counts() -> None:
    ids = tuple(f"a{k}" for k in range(6))
    sample = _sampled(ids)
    line = _note_line(setup_text(_designed_table(sample)))
    below = _counts(sample, "below")
    low, high = min(below.values()), max(below.values())
    span = str(low) if low == high else f"{low} to {high}"
    assert (
        f"{span} of 600 draws in the design sample for each of 6 assets are more "
        "frequent than the shortest fitted return period (2 years)"
    ) in line
    assert "6 assets: " in line
    assert "Asset a0: " not in line


def test_the_plan_s_setup_text_collapses_its_notes_on_real_output(
    dataset: Path,
) -> None:
    plan = (
        _designed_on_curves(dataset)
        .split(FreshDraws(300, seed=7))
        .scopes(Joint())
        .dependence(Comonotonic())
        .allocation("value")
    )
    result = plan.write_parquet(dataset.parent / "out.parquet")
    text = setup_text(result.metadata)
    lines = text.splitlines()
    line = _note_line(text)
    assert result.joint_sample is not None
    assert result.fresh_joint_sample is not None
    design_below = _counts(result.joint_sample, "below")
    fresh_below = _counts(result.fresh_joint_sample, "below")
    first = ALL[0]
    assert (
        f"{design_below[first]} of 400 draws in the design sample for {first}, "
        f"{fresh_below[first]} of 300 in the fresh draws for {first}, "
    ) in line
    assert line.count("more frequent than the shortest") == 1
    assert "Asset warehouse-a: " not in line
    assert lines[0].startswith("Level: total, the sum over warehouse-a, depot-b and")
    assert "asset_by_value" not in lines[0]
    assert "Scope: joint" in lines[0]
    # The allocated rows' own table states their rule.
    allocated = setup_text(result.allocated).splitlines()[0]
    assert allocated.startswith("Level: asset_by_value, one row per asset")
    sample = next(item for item in lines if item.startswith("Sample: "))
    assert "Dependence between the assets: Comonotonic(), a stated assumption" in sample
    assert "A joint design's premium is that of one contract on the total." in text
    design = next(item for item in lines if item.startswith("Design "))
    assert "scope: joint" in design
    # The rows of a table state the same setup as the metadata.
    assert setup_text(result.table) == text
