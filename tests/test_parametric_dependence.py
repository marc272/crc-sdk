"""Stated dependence between the assets of a curve sample, offline."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from crc_framework import (
    FittedDistribution,
    HurdleDistribution,
    PiecewiseLinearImpact,
)
from numpy.typing import NDArray

import crc_sdk.parametric.dependence as dependence_module
from crc_sdk.parametric import (
    ComonotoneIndexWarning,
    Comonotonic,
    EmpiricalCopula,
    ImpactNoise,
    Independent,
    LeakageError,
    LossModel,
    NoBasisRisk,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    holdout,
    sample_annual,
    sample_curve,
    sample_curves,
    year_block_kfold,
)
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.parametric.sources import AnnualSeries
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance
from crc_sdk.workflows import ReturnPeriodExtrapolationWarning

UNTYPED_CURVES: Any = sample_curves  # for calls with deliberately invalid arguments
UNTYPED_COPULA: Any = EmpiricalCopula
ASSETS = ("site-b", "site-c", "site-a")
LOCATIONS = {"site-b": 50.0, "site-c": 80.0, "site-a": 30.0}
SEEDS = {"site-b": 11, "site-c": 22, "site-a": 33}
RAMP = PiecewiseLinearImpact([20.0, 140.0], [0.0, 1.0])
NOISE = ImpactNoise(0.3, upper=1.0)


def _metadata(**fields: Any) -> HazardDatasetMetadata:
    base: dict[str, Any] = {
        "h3_resolution": 7,
        "value_unit": "mm/day",
        "value_semantics": "annual maximum daily rainfall",
        "producer": "tests",
        "source": SourceProvenance(
            provider="fixture",
            dataset="rx1day",
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        "creation_version": "1",
        "probability_semantics": "annual_exceedance",
    }
    base.update(fields)
    return HazardDatasetMetadata(**base)


def _curve(asset: str) -> FittedDistribution:
    return FittedDistribution.from_parameters(
        "gumbel_r", location=LOCATIONS[asset], scale=10.0
    )


def _loss(asset: str) -> LossModel:
    return LossModel(RAMP, value=1e6, value_unit="EUR", name=f"loss of {asset}")


def _rngs(shift: int = 0) -> dict[str, np.random.Generator]:
    return {a: np.random.default_rng(SEEDS[a] + shift) for a in ASSETS}


def _draws(dependence: Any, n: int = 300, **options: Any) -> OutcomeSample:
    sample: OutcomeSample = UNTYPED_CURVES(
        options.pop("curves", {a: _curve(a) for a in ASSETS}),
        dependence=dependence,
        metadata=options.pop("metadata", {a: _metadata() for a in ASSETS}),
        n_draws=n,
        rngs=options.pop("rngs", _rngs()),
        loss_models=options.pop("loss_models", {a: _loss(a) for a in ASSETS}),
        basis_risk=options.pop("basis_risk", NOISE),
        **options,
    )
    return sample


def _single(asset: str, n: int = 300, seed: int | None = None) -> OutcomeSample:
    return sample_curve(
        _curve(asset),
        metadata=_metadata(),
        n_draws=n,
        rng=np.random.default_rng(SEEDS[asset] if seed is None else seed),
        loss_model=_loss(asset),
        basis_risk=NOISE,
        asset_id=asset,
    )


def _uniforms(asset: str, n: int = 300) -> NDArray[np.float64]:
    return np.random.default_rng(SEEDS[asset]).random(n)


def _reference(
    m: int = 30,
    *,
    seed: int = 5,
    assets: tuple[str, ...] = ASSETS,
    **overrides: Any,
) -> OutcomeSample:
    """A per-year reference whose columns have their own ranks."""
    rng = np.random.default_rng(seed)
    fields: dict[str, Any] = {
        "asset_ids": assets,
        "period": np.arange(1990, 1990 + m),
        "probability": np.full(m, 1.0 / m),
        "index": rng.gamma(2.0, 20.0, (m, len(assets))),
        "loss": rng.gamma(2.0, 5.0, (m, len(assets))),
        "hazard": None,
        "index_unit": "degC",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": SampleProvenance(
            source_kind="annual",
            sources=({"role": "hazard", "name": "era5 series"},),
            semantics_source="data",
            licences=(
                {
                    "role": "hazard",
                    "source": "era5",
                    "licence": "CC-BY-4.0",
                    "attribution": "ECMWF",
                },
            ),
            dependence={"kind": "data"},
        ),
    }
    fields.update(overrides)
    return OutcomeSample(**fields)


def _ranks(values: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")


def _rows(sample: OutcomeSample, column: int) -> list[tuple[float, float, float]]:
    """A column's (hazard, index, loss) rows, sorted to compare as multisets."""
    hazard = sample.index if sample.hazard is None else sample.hazard
    return sorted(
        zip(
            hazard[:, column].tolist(),
            sample.index[:, column].tolist(),
            sample.loss[:, column].tolist(),
            strict=True,
        )
    )


def _models() -> list[Any]:
    """Every model, each fit for the default 300 draws."""
    return [Independent(), Comonotonic(), EmpiricalCopula(_reference())]


# -- no default --------------------------------------------------------------


def test_dependence_has_no_default() -> None:
    parameter = inspect.signature(sample_curves).parameters["dependence"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="dependence"):
        UNTYPED_CURVES(
            {a: _curve(a) for a in ASSETS},
            metadata={a: _metadata() for a in ASSETS},
            n_draws=10,
            rngs=_rngs(),
            loss_models={a: _loss(a) for a in ASSETS},
            basis_risk=NOISE,
        )


@pytest.mark.parametrize("value", [None, "independent", object()])
def test_dependence_must_be_a_stated_model(value: Any) -> None:
    with pytest.raises(TypeError, match=r"Independent\(\), Comonotonic\(\) or"):
        _draws(value)


# -- every model only reorders -----------------------------------------------


def test_independent_equals_sample_curve_column_by_column() -> None:
    sample = _draws(Independent())
    assert sample.asset_ids == ASSETS
    assert sample.period.tolist() == list(range(300))
    assert sample.hazard is None
    for j, asset in enumerate(ASSETS):
        single = _single(asset)
        assert np.array_equal(sample.index[:, j], single.index[:, 0])
        assert np.array_equal(sample.loss[:, j], single.loss[:, 0])
        assert np.array_equal(sample.probability, single.probability)
        assert sample.asset(asset).content_hash == single.content_hash


@pytest.mark.parametrize("position", [0, 1, 2])
def test_every_model_holds_each_asset_rows_in_another_order(position: int) -> None:
    asset = ASSETS[position]
    single = _rows(_single(asset), 0)
    for model in _models():
        assert _rows(_draws(model), position) == single


class ShiftsHighCurves:
    """Shifts the index of a curve whose draws are high; leaves the others alone."""

    name = "shifts_high_curves"

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.name}

    def apply(
        self,
        index: NDArray[np.float64],
        hazard: NDArray[np.float64],
        impact: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        del rng
        return index + 3.0 * float(hazard.mean() > 70.0), impact


def test_the_reordering_models_do_move_rows() -> None:
    for model in (Comonotonic(), EmpiricalCopula(_reference())):
        sample = _draws(model)
        for j, asset in enumerate(ASSETS):
            if isinstance(model, EmpiricalCopula) or j > 0:
                assert not np.array_equal(
                    sample.index[:, j], _single(asset).index[:, 0]
                )


def test_a_rows_hazard_index_and_loss_stay_together() -> None:
    model: Any = ShiftsHighCurves()
    sample = _draws(Comonotonic(), basis_risk=model)
    assert sample.hazard is not None
    # Only site-c's curve is high: the other columns' hazard is their index.
    assert np.array_equal(sample.hazard[:, 0], sample.index[:, 0])
    assert np.array_equal(sample.hazard[:, 2], sample.index[:, 2])
    np.testing.assert_allclose(sample.index[:, 1] - sample.hazard[:, 1], 3.0)
    for j, asset in enumerate(ASSETS):
        loss_model = _loss(asset)
        expected = loss_model.loss_values(loss_model.impact_values(sample.hazard[:, j]))
        np.testing.assert_allclose(sample.loss[:, j], expected)


def test_no_separate_hazard_when_no_asset_has_one() -> None:
    assert _draws(EmpiricalCopula(_reference())).hazard is None


def test_the_model_uses_no_random_numbers_beyond_sample_curve() -> None:
    for model in _models():
        rngs = _rngs()
        _draws(model, rngs=rngs)
        for asset in ASSETS:
            own = np.random.default_rng(SEEDS[asset])
            sample_curve(
                _curve(asset),
                metadata=_metadata(),
                n_draws=300,
                rng=own,
                loss_model=_loss(asset),
                basis_risk=NOISE,
                asset_id=asset,
            )
            assert rngs[asset].bit_generator.state == own.bit_generator.state


def test_a_sample_can_be_rebuilt_from_its_records() -> None:
    reference = _reference()
    sample = _draws(EmpiricalCopula(reference))
    rngs = {asset: sample.asset(asset).provenance.generator(0) for asset in ASSETS}
    again = _draws(EmpiricalCopula(reference), rngs=rngs)
    assert again.content_hash == sample.content_hash


# -- comonotonic -------------------------------------------------------------


def test_comonotonic_keeps_the_first_asset_in_draw_order() -> None:
    sample = _draws(Comonotonic())
    single = _single(ASSETS[0])
    assert np.array_equal(sample.index[:, 0], single.index[:, 0])
    assert np.array_equal(sample.loss[:, 0], single.loss[:, 0])
    assert sample.asset(ASSETS[0]).content_hash == single.content_hash


def test_comonotonic_gives_rank_correlation_one() -> None:
    sample = _draws(Comonotonic())
    first = _ranks(sample.index[:, 0])
    for j in range(1, 3):
        assert np.array_equal(_ranks(sample.index[:, j]), first)
    assert float(np.corrcoef(_ranks(sample.index[:, 1]), first)[0, 1]) == pytest.approx(
        1.0
    )


def test_comonotonic_ranks_are_those_of_the_uniforms_so_an_atom_adds_no_ties() -> None:
    hurdle = HurdleDistribution(
        FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0),
        atom_probability=0.4,
        atom_location=0.0,
    )
    curves = {a: _curve(a) for a in ASSETS} | {"site-c": hurdle}
    uniforms = np.column_stack([_uniforms(a) for a in ASSETS])
    order = Comonotonic().reorder(uniforms, ASSETS)
    moved = np.column_stack(
        [uniforms[order[:, j], j] for j in range(uniforms.shape[1])]
    )
    for j in range(1, 3):
        assert np.array_equal(_ranks(moved[:, j]), _ranks(moved[:, 0]))

    sample = _draws(Comonotonic(), curves=curves)
    dry = sample.index[:, 1] == 0.0
    assert dry.any() and not dry.all()
    # The atom's draws share their rows with the first asset's lowest ranks.
    assert sample.index[dry, 0].max() <= sample.index[~dry, 0].min()


# -- the empirical copula ----------------------------------------------------


def test_copula_with_one_draw_per_year_has_the_references_ranks() -> None:
    reference = _reference()
    sample = _draws(EmpiricalCopula(reference), n=30)
    for j in range(3):
        assert np.array_equal(_ranks(sample.index[:, j]), _ranks(reference.index[:, j]))


@pytest.mark.parametrize("k", [1, 3, 100])
def test_copula_blocks_equal_the_references_ranks(k: int) -> None:
    reference = _reference()
    m = len(reference)
    n = k * m
    uniforms = np.column_stack([_uniforms(a, n) for a in ASSETS])
    order = EmpiricalCopula(reference).reorder(uniforms, ASSETS)
    for j in range(3):
        assert sorted(order[:, j].tolist()) == list(range(n))
        rank_of_draw = np.empty(n, dtype=np.int64)
        rank_of_draw[np.argsort(uniforms[:, j], kind="stable")] = np.arange(n)
        blocks = rank_of_draw[order[:, j]] // k
        expected = np.repeat(_ranks(reference.index[:, j]), k)
        assert np.array_equal(blocks, expected)
        # Inside a block the draws keep the order in which they were drawn.
        assert (np.diff(order[:, j].reshape(m, k), axis=1) > 0).all()


def test_copula_rows_are_in_reference_year_order() -> None:
    reference = _reference()
    sample = _draws(EmpiricalCopula(reference), n=90)
    # 3 draws to a year: a year's rows all lie below the next rank's.
    index = sample.index[:, 0].reshape(30, 3)
    ranks = _ranks(reference.index[:, 0])
    by_rank = index[np.argsort(ranks, kind="stable")]
    assert (by_rank[:-1].max(axis=1) <= by_rank[1:].min(axis=1)).all()


def test_copula_ties_pool_their_blocks_in_draw_order() -> None:
    uniforms = np.array([[0.9, 0.1, 0.5, 0.7, 0.3, 0.2]]).T
    uniforms = np.hstack([uniforms, uniforms[::-1]])
    reference = _reference(
        3,
        assets=("x", "y"),
        index=np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]),
    )
    order = EmpiricalCopula(reference).reorder(uniforms, ("x", "y"))
    # Sorted by uniform the draws are 1, 5 | 4, 2 | 3, 0. Year 2 is lowest and
    # takes the first block; years 0 and 1 tie and pool the other two.
    assert order[:, 0].tolist() == [0, 2, 3, 4, 1, 5]
    # y's uniforms sort to 4, 0 | 1, 3 | 2, 5, and it has no tie: its years take
    # those blocks in turn, each in draw order.
    assert order[:, 1].tolist() == [0, 4, 1, 3, 2, 5]


def test_copula_tie_with_one_draw_per_year_deals_draws_in_year_order() -> None:
    uniforms = np.array([[0.4, 0.9, 0.2], [0.3, 0.1, 0.6]])
    reference = _reference(
        3,
        assets=("x", "y"),
        index=np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]),
    )
    order = EmpiricalCopula(reference).reorder(uniforms.T, ("x", "y"))
    assert order[:, 0].tolist() == [0, 1, 2]


def test_copula_reads_the_reference_by_column_name() -> None:
    reference = _reference()
    uniforms = np.column_stack([_uniforms(a, 30) for a in ASSETS])
    order = EmpiricalCopula(reference).reorder(uniforms, ASSETS)
    swapped = EmpiricalCopula(reference).reorder(
        uniforms[:, [1, 0, 2]], (ASSETS[1], ASSETS[0], ASSETS[2])
    )
    assert np.array_equal(order[:, [1, 0, 2]], swapped)


def test_copula_ranks_the_hazard_when_the_reference_has_one() -> None:
    m = 30
    index = np.random.default_rng(1).gamma(2.0, 20.0, (m, 3))
    hazard = np.random.default_rng(2).gamma(2.0, 20.0, (m, 3))
    with_hazard = _reference(m, index=index, hazard=hazard)
    sample = _draws(EmpiricalCopula(with_hazard), n=30)
    for j in range(3):
        assert np.array_equal(_ranks(sample.index[:, j]), _ranks(hazard[:, j]))


def test_copula_ignores_the_references_unit_and_location() -> None:
    reference = _reference()
    assert reference.index_unit != "mm/day"
    rescaled = _reference(index=reference.index * 1000.0 + 7.0)
    a = _draws(EmpiricalCopula(reference))
    b = _draws(EmpiricalCopula(rescaled))
    assert np.array_equal(a.index, b.index)
    assert np.array_equal(a.loss, b.loss)


def test_copula_a_larger_reference_value_takes_a_larger_draw() -> None:
    reference = _reference()
    negated = _reference(index=-reference.index)
    a = _draws(EmpiricalCopula(reference), n=30)
    b = _draws(EmpiricalCopula(negated), n=30)
    for j in range(3):
        assert np.array_equal(_ranks(a.index[:, j]), 29 - _ranks(b.index[:, j]))


def test_copula_records_its_reference() -> None:
    reference = _reference(40)
    sample = _draws(EmpiricalCopula(reference), n=200)
    document = sample.provenance.to_document()["dependence"]
    assert document["kind"] == "empirical_copula"
    assert document["content_hash"] == reference.content_hash
    assert document["sources"] == [{"role": "hazard", "name": "era5 series"}]
    assert document["first_year"] == 1990
    assert document["last_year"] == 2029
    assert document["m"] == 40
    assert document["k"] == 5
    assert EmpiricalCopula(reference).to_document().items() <= document.items()
    assert "k" not in EmpiricalCopula(reference).to_document()


def test_copula_lists_the_references_licence() -> None:
    sample = _draws(EmpiricalCopula(_reference()))
    roles = {entry["role"]: entry for entry in sample.provenance.licences}
    assert roles["dependence"]["licence"] == "CC-BY-4.0"
    assert roles["dependence"]["attribution"] == "ECMWF"
    assert "hazard" in roles


def test_copula_takes_a_joined_annual_reference() -> None:
    series = {
        a: AnnualSeries(
            years=np.arange(2000, 2030),
            values=np.random.default_rng(i).gamma(2.0, 20.0, 30),
            unit="mm/day",
            provenance={"kind": "table", "source": {"name": f"era5-{a}"}},
        )
        for i, a in enumerate(ASSETS)
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        parts = [
            sample_annual(
                series[a], loss_model=_loss(a), basis_risk=NoBasisRisk(), asset_id=a
            )
            for a in ASSETS
        ]
    sample = _draws(EmpiricalCopula(OutcomeSample.join(parts)), n=60)
    assert sample.provenance.dependence is not None
    assert sample.provenance.dependence["first_year"] == 2000
    assert sample.provenance.dependence["last_year"] == 2029
    for j, asset in enumerate(ASSETS):
        by_year = sample.index[:, j].reshape(30, 2).mean(axis=1)
        assert np.array_equal(_ranks(by_year), _ranks(series[asset].values))


def test_copula_takes_a_table_reference() -> None:
    values = np.random.default_rng(3).gamma(2.0, 20.0, (10, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        reference = OutcomeSample.from_table(
            pa.table(
                {
                    "year": [2000 + i for i in range(10) for _ in ASSETS],
                    "site": [a for _ in range(10) for a in ASSETS],
                    "x": values.reshape(-1).tolist(),
                    "y": values.reshape(-1).tolist(),
                }
            ),
            period="year",
            asset_id="site",
            index="x",
            loss="y",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
            source="a table of mine",
        )
    sample = _draws(EmpiricalCopula(reference), n=10)
    for j, asset in enumerate(ASSETS):
        column = reference.asset_ids.index(asset)
        assert np.array_equal(
            _ranks(sample.index[:, j]), _ranks(reference.index[:, column])
        )


def test_copula_refuses_a_monte_carlo_reference() -> None:
    with pytest.raises(SampleError, match="no dependence"):
        EmpiricalCopula(_draws(Independent(), n=30))
    provenance = SampleProvenance(
        source_kind="monte_carlo",
        sources=(),
        semantics_source="caller",
        dependence={"kind": "data"},
    )
    with pytest.raises(SampleError, match="Monte Carlo"):
        EmpiricalCopula(_reference(provenance=provenance))


def test_copula_refuses_a_one_asset_reference() -> None:
    provenance = SampleProvenance(
        source_kind="annual", sources=(), semantics_source="data"
    )
    with pytest.raises(SampleError, match="two columns"):
        EmpiricalCopula(_reference(assets=("site-a",), provenance=provenance))


def test_copula_refuses_unequal_probabilities() -> None:
    probability = np.full(30, 1.0 / 30)
    probability[0] += 0.01
    probability[1] -= 0.01
    with pytest.raises(SampleError, match="equal probabilities"):
        EmpiricalCopula(_reference(probability=probability))


def test_copula_takes_nearly_equal_probabilities() -> None:
    probability = np.full(30, 1.0 / 30)
    probability[0] += 1e-15
    EmpiricalCopula(_reference(probability=probability))


def test_copula_refuses_anything_but_a_sample() -> None:
    with pytest.raises(TypeError, match="OutcomeSample"):
        UNTYPED_COPULA("era5")


@pytest.mark.parametrize(
    ("n", "lower", "upper"),
    [(31, 30, 60), (29, 30, 60), (1, 30, 60), (95, 90, 120)],
)
def test_copula_refuses_a_count_that_is_no_multiple(
    n: int, lower: int, upper: int
) -> None:
    rngs = _rngs()
    before = {a: rngs[a].bit_generator.state for a in ASSETS}
    with pytest.raises(SampleError, match=f"{lower} and {upper}"):
        _draws(EmpiricalCopula(_reference()), n=n, rngs=rngs)
    assert {a: rngs[a].bit_generator.state for a in ASSETS} == before


def test_copula_refuses_a_reference_of_other_assets() -> None:
    reference = _reference(assets=("site-a", "site-b", "site-x"))
    rngs = _rngs()
    before = rngs["site-a"].bit_generator.state
    with pytest.raises(SampleError, match="site-x"):
        _draws(EmpiricalCopula(reference), rngs=rngs)
    assert rngs["site-a"].bit_generator.state == before


# -- the models as objects ---------------------------------------------------


def test_models_are_small_frozen_and_documented() -> None:
    reference = _reference()
    copula = EmpiricalCopula(reference)
    assert Independent().to_document() == {"kind": "independent"}
    assert Comonotonic().to_document() == {"kind": "comonotonic"}
    assert copula.to_document()["kind"] == "empirical_copula"
    json.dumps(copula.to_document())
    assert Independent() == Independent()
    assert Comonotonic() == Comonotonic()
    assert copula == EmpiricalCopula(reference)
    assert copula != EmpiricalCopula(_reference(seed=9))
    with pytest.raises(dataclasses.FrozenInstanceError):
        copula.reference = reference  # type: ignore[misc]


def test_the_module_needs_numpy_and_the_standard_library_only() -> None:
    source = Path(dependence_module.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            roots.add(node.module.split(".")[0])
    assert roots == {
        "__future__",
        "collections",
        "crc_sdk",
        "dataclasses",
        "json",
        "math",
        "numbers",
        "numpy",
        "statistics",
        "typing",
    }


def test_the_reorder_needs_no_assets() -> None:
    uniforms = np.random.default_rng(0).random((12, 4))
    names = ("rain", "wind", "heat", "flood")
    assert np.array_equal(
        Independent().reorder(uniforms, names)[:, 3], np.arange(12, dtype=np.int64)
    )
    order = Comonotonic().reorder(uniforms, names)
    assert np.array_equal(order[:, 0], np.arange(12, dtype=np.int64))
    assert sorted(order[:, 3].tolist()) == list(range(12))


# -- the record --------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "kind"),
    [(Independent(), "independent"), (Comonotonic(), "comonotonic")],
)
def test_the_provenance_states_the_model(model: Any, kind: str) -> None:
    sample = _draws(model)
    assert sample.provenance.dependence is not None
    assert dict(sample.provenance.dependence) == {"kind": kind}


def test_the_notes_call_the_dependence_an_assumption() -> None:
    for model, word in (
        (Independent(), "independent"),
        (Comonotonic(), "comonotonic"),
        (EmpiricalCopula(_reference()), "ranks of the stated reference"),
    ):
        first = _draws(model).provenance.notes[0]
        assert first.startswith("The dependence between the assets is an assumption")
        assert word in first


def test_the_copula_note_states_the_limit_of_the_reference() -> None:
    note = _draws(EmpiricalCopula(_reference()), n=90).provenance.notes[0]
    assert "30 years from 1990 to 2019" in note
    assert "3 draws per year" in note
    assert "dependence finer than one year in 30 is assumed, not observed" in note
    assert "a design and its fresh draws share it" in note


def test_no_note_says_which_model_is_better() -> None:
    for model in _models():
        statement = model.statement(300, "assets").lower()
        for word in ("better", "worse", "outperform", "superior", "should"):
            assert word not in statement


def test_the_provenance_keeps_each_assets_own_record() -> None:
    sample = _draws(Comonotonic())
    provenance = sample.provenance
    assert provenance.source_kind == "monte_carlo"
    assert provenance.probability_semantics == "annual_exceedance"
    assert provenance.semantics_source == "dataset"
    assert provenance.loss_model is None
    assert provenance.basis_risk is not None
    assert provenance.basis_risk["kind"] == "impact_noise"
    assert [source["asset_id"] for source in provenance.sources] == list(ASSETS)
    assert sorted(record["asset_id"] for record in provenance.rng) == sorted(ASSETS * 2)
    assert provenance.details["draws"] == 300
    assert provenance.details["periods"] == (0, 299)
    assert provenance.parts is not None
    assert set(provenance.parts) == set(ASSETS)
    for asset in ASSETS:
        own = _single(asset).provenance.to_document()
        part = sample.asset(asset).provenance.to_document()
        for field in ("sources", "rng", "loss_model", "basis_risk", "details"):
            assert part[field] == own[field], field
        assert part["lineage"] == own["lineage"]
    json.dumps(provenance.to_document())


def test_a_part_records_that_its_rows_were_moved() -> None:
    sample = _draws(Comonotonic())
    assert sample.provenance.parts is not None
    assert list(sample.provenance.parts["site-b"]["selection"]) == []
    assert [dict(step) for step in sample.provenance.parts["site-c"]["selection"]] == [
        {"kind": "reorder", "dependence": "comonotonic"}
    ]
    steps = sample.asset("site-c").provenance.selection
    assert [step["kind"] for step in steps] == ["reorder", "asset"]
    untouched = _draws(Independent())
    assert [
        step["kind"] for step in untouched.asset("site-c").provenance.selection
    ] == ["asset"]


def test_the_sample_round_trips_through_its_table() -> None:
    sample = _draws(EmpiricalCopula(_reference()))
    back = OutcomeSample.from_written_table(sample.to_table())
    assert back.content_hash == sample.content_hash
    assert back.provenance.to_document() == sample.provenance.to_document()


def test_mixed_semantics_are_stated_side_by_side() -> None:
    metadata = {a: _metadata() for a in ASSETS}
    metadata["site-c"] = _metadata(probability_semantics="annual_value_distribution")
    provenance = _draws(Independent(), metadata=metadata).provenance
    assert (
        provenance.probability_semantics
        == "annual_exceedance, annual_value_distribution"
    )


def test_a_mixed_statement_of_the_semantics_is_the_callers() -> None:
    metadata = {a: _metadata() for a in ASSETS}
    metadata["site-c"] = _metadata(probability_semantics=None)
    provenance = _draws(
        Independent(), metadata=metadata, assume_semantics="annual_exceedance"
    ).provenance
    assert provenance.semantics_source == "caller"
    assert provenance.probability_semantics == "annual_exceedance"


def test_details_and_options_reach_every_asset() -> None:
    sample = _draws(Independent(), details={"file_sha256": "abc"})
    assert sample.provenance.parts is not None
    for asset in ASSETS:
        assert sample.provenance.parts[asset]["sources"][0]["details"] == {
            "file_sha256": "abc"
        }


# -- splits ------------------------------------------------------------------


def _holdout(sample: OutcomeSample) -> Any:
    return holdout(sample)


def _kfold(sample: OutcomeSample) -> Any:
    return year_block_kfold(sample, 3)


SPLITS = [(_holdout, "holdout"), (_kfold, "year_block_kfold")]
OTHER_MODELS: list[Any] = [Independent(), Comonotonic()]


@pytest.mark.parametrize(("split", "name"), SPLITS)
def test_a_split_of_a_copula_sample_is_refused_naming_fresh_draws(
    split: Any, name: str
) -> None:
    sample = _draws(EmpiricalCopula(_reference()))
    with pytest.raises(SampleError, match=rf"{name} cannot split .*FreshDraws"):
        split(sample)


@pytest.mark.parametrize(("split", "name"), SPLITS)
def test_a_split_of_a_written_copula_sample_is_refused(split: Any, name: str) -> None:
    table = _draws(EmpiricalCopula(_reference())).to_table()
    with pytest.raises(SampleError, match=name):
        split(OutcomeSample.from_written_table(table))


@pytest.mark.parametrize(("split", "name"), SPLITS)
def test_a_split_of_selected_copula_rows_is_refused(split: Any, name: str) -> None:
    selected = _draws(EmpiricalCopula(_reference())).select(np.arange(60))
    with pytest.raises(SampleError, match=name):
        split(selected)


@pytest.mark.parametrize(("split", "name"), SPLITS)
def test_a_split_of_one_assets_copula_rows_is_refused(split: Any, name: str) -> None:
    # The rows of one asset's view still come in reference-year blocks.
    view = _draws(EmpiricalCopula(_reference())).asset("site-c")
    assert view.provenance.dependence is None
    with pytest.raises(SampleError, match=name):
        split(view)


def test_selecting_periods_of_a_copula_sample_is_not_refused() -> None:
    sample = _draws(EmpiricalCopula(_reference()))
    selected = sample.select(np.arange(30))
    assert len(selected) == 30
    assert selected.lineage == sample.lineage


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_holdout_splits_the_other_models_on_every_asset(model: Any) -> None:
    sample = _draws(model)
    design, held_out = holdout(sample, train_fraction=0.7)
    assert len(design) == 210
    assert len(held_out) == 90
    assert design.asset_ids == held_out.asset_ids == ASSETS
    assert np.array_equal(design.index, sample.index[:210])
    assert np.array_equal(design.loss, sample.loss[:210])
    assert np.array_equal(held_out.index, sample.index[210:])
    assert np.array_equal(held_out.loss, sample.loss[210:])
    assert design.lineage == held_out.lineage == sample.lineage
    assert design.provenance.dependence is not None
    assert dict(design.provenance.dependence) == model.to_document()
    _check_out_of_sample(design, held_out)
    with pytest.raises(LeakageError):
        _check_out_of_sample(design, sample)


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_year_block_kfold_splits_the_other_models_on_every_asset(model: Any) -> None:
    sample = _draws(model)
    folds = year_block_kfold(sample, 3)
    assert len(folds) == 3
    held = np.concatenate([held_out.period for _, held_out in folds])
    assert held.tolist() == list(range(300))
    for design, held_out in folds:
        assert design.asset_ids == held_out.asset_ids == ASSETS
        assert np.array_equal(design.index, sample.index[design.period])
        assert np.array_equal(held_out.loss, sample.loss[held_out.period])
        _check_out_of_sample(design, held_out)


@pytest.mark.parametrize("model", OTHER_MODELS)
def test_one_assets_rows_of_the_other_models_split(model: Any) -> None:
    view = _draws(model).asset("site-c")
    design, held_out = holdout(view)
    assert (len(design), len(held_out)) == (210, 90)
    assert len(year_block_kfold(view, 3)) == 3


def test_fresh_draws_share_the_reference_and_need_a_multiple_too() -> None:
    copula = EmpiricalCopula(_reference())
    design = _draws(copula)
    fresh = _draws(copula, n=600, rngs=_rngs(100))
    assert design.provenance.dependence is not None
    assert fresh.provenance.dependence is not None
    assert (
        design.provenance.dependence["content_hash"]
        == fresh.provenance.dependence["content_hash"]
    )
    assert fresh.provenance.dependence["k"] == 20
    _check_out_of_sample(design, fresh)
    with pytest.raises(SampleError, match="600 and 630"):
        _draws(copula, n=601, rngs=_rngs(100))


# -- leakage -----------------------------------------------------------------


def test_the_lineage_links_the_same_draws_under_every_model() -> None:
    lineages = {_draws(model).lineage for model in _models()}
    assert len(lineages) == 1
    assert _draws(Independent(), rngs=_rngs(100)).lineage not in lineages


def test_draws_shared_between_models_are_a_leak() -> None:
    with pytest.raises(LeakageError):
        _check_out_of_sample(_draws(Comonotonic()), _draws(Independent()))
    _check_out_of_sample(_draws(Comonotonic()), _draws(Comonotonic(), rngs=_rngs(100)))


def test_an_assets_view_keeps_the_lineage_of_its_own_draws() -> None:
    sample = _draws(Comonotonic())
    with pytest.raises(LeakageError):
        _check_out_of_sample(sample.asset("site-c"), _single("site-c"))
    _check_out_of_sample(sample.asset("site-c"), _single("site-c", seed=999))


# -- arguments ---------------------------------------------------------------


def test_one_asset_is_refused_naming_sample_curve() -> None:
    with pytest.raises(SampleError, match="sample_curve for one"):
        UNTYPED_CURVES(
            {"site-a": _curve("site-a")},
            dependence=Independent(),
            metadata={"site-a": _metadata()},
            n_draws=10,
            rngs={"site-a": np.random.default_rng(1)},
            loss_models={"site-a": _loss("site-a")},
            basis_risk=NOISE,
        )


@pytest.mark.parametrize("name", ["metadata", "rngs", "loss_models"])
def test_a_mapping_must_name_the_curves_assets(name: str) -> None:
    built: dict[str, Any] = {
        "metadata": {a: _metadata() for a in ASSETS},
        "rngs": _rngs(),
        "loss_models": {a: _loss(a) for a in ASSETS},
    }
    built[name] = {**built[name], "site-x": built[name]["site-a"]}
    with pytest.raises(SampleError, match=name):
        _draws(Independent(), **built)


@pytest.mark.parametrize("name", ["metadata", "rngs", "loss_models", "curves"])
def test_a_mapping_must_be_a_mapping(name: str) -> None:
    options: dict[str, Any] = {name: ["not a mapping"]}
    with pytest.raises(TypeError, match=name):
        _draws(Independent(), **options)


def test_wrong_value_types_are_refused() -> None:
    with pytest.raises(TypeError, match="HazardDatasetMetadata"):
        _draws(Independent(), metadata={a: "x" for a in ASSETS})
    with pytest.raises(TypeError, match="Generator"):
        _draws(Independent(), rngs={a: "x" for a in ASSETS})
    with pytest.raises(TypeError, match="LossModel"):
        _draws(Independent(), loss_models={a: "x" for a in ASSETS})


def test_a_basis_risk_model_is_required() -> None:
    with pytest.raises(TypeError, match="basis_risk"):
        _draws(Independent(), basis_risk=None)


def test_invalid_counts_are_refused() -> None:
    with pytest.raises(SampleError, match="n_draws"):
        _draws(Independent(), n=0)
    with pytest.raises(SampleError, match="n_draws"):
        _draws(Independent(), n=True)


def test_a_loss_model_bound_to_another_asset_is_refused() -> None:
    loss_models = {a: _loss(a) for a in ASSETS}
    book = pa.table({"asset_id": ["site-b"], "value": [1.0]})
    loss_models["site-a"] = _loss("site-a").for_asset(book, "site-b")
    with pytest.raises(SampleError, match="bound to asset"):
        _draws(Independent(), loss_models=loss_models)


def test_one_shared_generator_is_refused() -> None:
    shared = np.random.default_rng(5)
    before = shared.bit_generator.state
    with pytest.raises(SampleError, match="own generator"):
        _draws(Independent(), rngs={a: shared for a in ASSETS})
    assert shared.bit_generator.state == before


def test_generators_in_the_same_state_are_refused() -> None:
    rngs = _rngs()
    rngs["site-c"] = np.random.default_rng(SEEDS["site-b"])
    with pytest.raises(SampleError, match="own generator"):
        _draws(Comonotonic(), rngs=rngs)


def test_units_must_agree_before_any_draw() -> None:
    rngs = _rngs()
    before = rngs["site-a"].bit_generator.state
    metadata = {a: _metadata() for a in ASSETS}
    metadata["site-c"] = _metadata(value_unit="mm/h")
    with pytest.raises(SampleError, match="index unit"):
        _draws(Independent(), metadata=metadata, rngs=rngs)
    assert rngs["site-a"].bit_generator.state == before
    loss_models = {a: _loss(a) for a in ASSETS}
    loss_models["site-c"] = LossModel(RAMP, value=1.0, value_unit="USD")
    with pytest.raises(SampleError, match="loss unit"):
        _draws(Independent(), loss_models=loss_models, rngs=rngs)
    assert rngs["site-a"].bit_generator.state == before


# -- warnings ----------------------------------------------------------------


def test_warnings_of_sample_curve_reach_the_caller() -> None:
    with pytest.warns(ComonotoneIndexWarning):
        _draws(Independent(), basis_risk=NoBasisRisk())


def test_draws_beyond_the_support_warn() -> None:
    metadata = {a: _metadata(return_period_support=(2.0, 10.0)) for a in ASSETS}
    with pytest.warns(ReturnPeriodExtrapolationWarning):
        _draws(Independent(), metadata=metadata)


def test_an_asset_note_names_its_asset() -> None:
    metadata = {a: _metadata(return_period_support=(2.0, 10.0)) for a in ASSETS}
    with pytest.warns(ReturnPeriodExtrapolationWarning):
        notes = _draws(Independent(), metadata=metadata).provenance.notes
    assert any(note.startswith("Asset site-c: ") for note in notes)


def test_a_join_still_refuses_curve_samples_and_names_this_function() -> None:
    with pytest.raises(SampleError, match="sample_curves"):
        OutcomeSample.join([_single(a) for a in ASSETS])
