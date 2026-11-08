"""Outcome samples of several assets: the object, `asset`, `join` and the refusals."""

from __future__ import annotations

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
    ComonotoneIndexWarning,
    Cover,
    EvaluationMetadata,
    Free,
    ImpactNoise,
    IndexDefinition,
    LeakageError,
    LinearPayout,
    LossModel,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    compare,
    design,
    evaluate,
    holdout,
    sample_curve,
    setup_text,
    year_block_kfold,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.sample import (
    SAMPLE_METADATA_KEY,
    _check_out_of_sample,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

UNTYPED_SAMPLE: Any = OutcomeSample  # for calls with deliberately invalid arguments
UNTYPED_PROVENANCE: Any = SampleProvenance
UNTYPED_JOIN: Any = OutcomeSample.join
ASSETS = ("site-b", "site-c", "site-a")
INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")


def _provenance(**fields: Any) -> SampleProvenance:
    base: dict[str, Any] = {
        "source_kind": "table",
        "sources": ({"role": "table", "name": "book"},),
        "semantics_source": "caller",
        "licences": ({"role": "table", "source": "book", "licence": "CC0-1.0"},),
    }
    return SampleProvenance(**{**base, **fields})


def _three(n: int = 12, *, seed: int = 3, **overrides: Any) -> OutcomeSample:
    """An `(n, 3)` sample whose columns differ."""
    rng = np.random.default_rng(seed)
    fields: dict[str, Any] = {
        "asset_ids": ASSETS,
        "period": np.arange(2000, 2000 + n),
        "probability": np.full(n, 1.0 / n),
        "index": rng.gamma(2.0, 20.0, (n, 3)),
        "loss": rng.gamma(2.0, 5.0, (n, 3)),
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": _provenance(dependence={"kind": "data"}),
    }
    fields.update(overrides)
    sample: OutcomeSample = UNTYPED_SAMPLE(**fields)
    return sample


def _part(asset: str, n: int = 10, *, seed: int = 1, **overrides: Any) -> OutcomeSample:
    """A one-asset per-year sample with its own source."""
    rng = np.random.default_rng(seed)
    fields: dict[str, Any] = {
        "asset_ids": (asset,),
        "period": np.arange(2000, 2000 + n),
        "probability": np.full(n, 1.0 / n),
        "index": rng.gamma(2.0, 20.0, n),
        "loss": rng.gamma(2.0, 5.0, n),
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": SampleProvenance(
            source_kind="annual",
            sources=({"role": "hazard", "name": f"series of {asset}"},),
            semantics_source="data",
            licences=({"role": "hazard", "source": "era5", "licence": "CC-BY-4.0"},),
            notes=("One note.",),
            loss_model={"kind": "impact", "asset_id": asset},
            basis_risk={"kind": "none"},
            lineage=f"lineage-of-{asset}",
        ),
    }
    fields.update(overrides)
    sample: OutcomeSample = UNTYPED_SAMPLE(**fields)
    return sample


def _curve_sample(asset: str = "site-a", seed: int = 42) -> OutcomeSample:
    metadata = HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics="annual maximum daily rainfall",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset="rx1day",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
        return_period_support=(2.0, 200.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample_curve(
            FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0),
            metadata=metadata,
            n_draws=10,
            rng=np.random.default_rng(seed),
            loss_model=LossModel(
                PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
                value=1e6,
                value_unit="EUR",
            ),
            basis_risk=ImpactNoise(0.3, upper=1.0),
            asset_id=asset,
        )


def _rows() -> dict[str, Any]:
    """A long table of two assets over three years, out of order."""
    return {
        "year": pa.array([2002, 2000, 2001, 2000, 2002, 2001], pa.int64()),
        "asset": ["b", "a", "a", "b", "a", "b"],
        "rain": [30.0, 10.0, 25.0, 50.0, 15.0, 40.0],
        "loss": [1.0, 4.0, 2.0, 0.5, 5.0, 2.5],
    }


def _from_rows(rows: dict[str, Any], **arguments: Any) -> OutcomeSample:
    return OutcomeSample.from_table(
        pa.table(rows),
        period="year",
        asset_id="asset",
        index="rain",
        loss="loss",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
        **arguments,
    )


def _linear(asset: str | None = None) -> Cover:
    return Cover(
        INDEX, LinearPayout(20.0, 80.0, 10.0), payout_unit="EUR", asset_id=asset
    )


# -- the object --------------------------------------------------------------


def test_a_sample_of_three_assets_keeps_one_column_per_asset() -> None:
    sample = _three(n=12)
    assert sample.asset_ids == ASSETS
    assert sample.index.shape == (12, 3) and sample.loss.shape == (12, 3)
    assert len(sample) == 12
    for array in (sample.index, sample.loss):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"asset_ids": ()}, "one asset or more"),
        ({"asset_ids": ("a", "b", "a")}, "distinct"),
        ({"index": np.ones(12)}, "index must have shape"),
        ({"index": np.ones((12, 2))}, "index must have shape"),
        ({"loss": np.ones((3, 12))}, "loss must have shape"),
        ({"hazard": np.ones((12, 2))}, "hazard must have shape"),
        (
            {
                "provenance": _provenance(
                    dependence={"kind": "data"}, parts={"site-a": {}}
                )
            },
            "parts are for",
        ),
    ],
)
def test_invalid_samples_of_several_assets_raise(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _three(**overrides)


@pytest.mark.parametrize("source_kind", ["monte_carlo", "annual", "table"])
def test_a_sample_of_several_assets_must_state_its_dependence(
    source_kind: str,
) -> None:
    with pytest.raises(SampleError, match="must state where the dependence"):
        _three(provenance=_provenance(source_kind=source_kind))


@pytest.mark.parametrize(
    "fields",
    [
        {"dependence": {"kind": "data"}},
        {"parts": {"site-a": {}}},
    ],
)
def test_a_one_asset_sample_has_neither_dependence_nor_parts(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(SampleError, match="one-asset sample has no dependence"):
        _part("site-a", provenance=_provenance(**fields))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"dependence": "data"}, "dependence must be a mapping"),
        ({"dependence": {"model": "x"}}, "name its kind"),
        ({"parts": "x"}, "parts must be a mapping"),
        ({"parts": {"a": 1}}, "parts must map"),
        ({"parts": {"a": {"x": float("nan")}}}, "plain JSON"),
    ],
)
def test_invalid_dependence_or_parts_raise(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(SampleError, match=message):
        _provenance(**fields)


def test_dependence_and_parts_are_read_only_and_plain_json() -> None:
    provenance = _provenance(
        dependence={"kind": "data"}, parts={"a": {"notes": ["one"]}}
    )
    with pytest.raises(TypeError):
        provenance.parts["b"] = {}  # type: ignore[index]
    document = provenance.to_document()
    assert document["dependence"] == {"kind": "data"}
    assert document["parts"] == {"a": {"notes": ["one"]}}
    json.dumps(document)


def test_a_one_asset_sample_keeps_its_hash_and_has_no_dependence() -> None:
    sample = _part("warehouse-a", n=3, index=np.array([1.0, 2.0, 3.0]))
    expected = content_hash(
        {
            "asset_id": ["warehouse-a"] * 3,
            "hazard": None,
            "index": [1.0, 2.0, 3.0],
            "index_unit": "mm/day",
            "loss": sample.loss[:, 0].tolist(),
            "loss_unit": "EUR",
            "period": [2000, 2001, 2002],
            "probability": sample.probability.tolist(),
            "scenario": "baseline",
            "schema_version": "1.0",
            "semantics": "annual",
            "semantics_version": "crc-parametric-sample/1",
        }
    )
    assert sample.content_hash == expected
    assert sample.provenance.dependence is None and sample.provenance.parts is None
    document = sample.provenance.to_document()
    assert document["dependence"] is None and document["parts"] is None


def test_a_table_written_before_these_fields_reads_back() -> None:
    table = _part("warehouse-a").to_table()
    key = SAMPLE_METADATA_KEY.encode()
    written = json.loads(table.schema.metadata[key])
    del written["provenance"]["dependence"], written["provenance"]["parts"]
    older = table.replace_schema_metadata({key: json.dumps(written)})
    read = OutcomeSample.from_written_table(older)
    assert read.provenance.dependence is None and read.provenance.parts is None


# -- tables and hashes -------------------------------------------------------


@pytest.mark.parametrize("with_hazard", [False, True])
def test_a_sample_of_three_assets_round_trips_its_table_and_hash(
    with_hazard: bool, tmp_path: Path
) -> None:
    sample = _three(hazard=np.full((12, 3), 7.0) if with_hazard else None)
    table = sample.to_table()
    assert table.num_rows == 36
    # Sorted by period, then asset id, whatever the order of the columns.
    assert table.column("asset_id").to_pylist()[:3] == [
        "site-a",
        "site-b",
        "site-c",
    ]
    assert table.column("period").to_pylist()[:4] == [2000, 2000, 2000, 2001]
    assert table.column("index").to_pylist()[:3] == [
        float(sample.index[0, 2]),
        float(sample.index[0, 0]),
        float(sample.index[0, 1]),
    ]
    path = tmp_path / "three.parquet"
    pq.write_table(table, path)
    for written in (table, table.to_batches()[0], pq.read_table(path)):
        read = OutcomeSample.from_written_table(written)
        assert read.content_hash == sample.content_hash
        assert read.asset_ids == sample.asset_ids
        for name in ("period", "probability", "index", "loss"):
            np.testing.assert_array_equal(getattr(read, name), getattr(sample, name))
        assert read.provenance.to_document() == sample.provenance.to_document()
        assert (read.hazard is None) is (not with_hazard)


@pytest.mark.parametrize(
    "dependence",
    [{"kind": "data"}, {"kind": "independent"}],
    ids=["the data's own", "a model"],
)
def test_a_written_table_of_several_assets_round_trips_its_dependence(
    dependence: dict[str, Any], tmp_path: Path
) -> None:
    sample = _three(provenance=_provenance(dependence=dependence))
    path = tmp_path / "three.parquet"
    pq.write_table(sample.to_table(), path)
    read = OutcomeSample.from_written_table(pq.read_table(path))
    assert read.provenance.to_document()["dependence"] == dependence


def test_a_written_table_of_several_assets_without_a_dependence_is_refused() -> None:
    table = _three().to_table()
    key = SAMPLE_METADATA_KEY.encode()
    written = json.loads(table.schema.metadata[key])
    written["provenance"]["dependence"] = None
    edited = table.replace_schema_metadata({key: json.dumps(written)})
    with pytest.raises(SampleError, match="must state where the dependence"):
        OutcomeSample.from_written_table(edited)


def test_the_hash_does_not_depend_on_the_order_of_the_columns() -> None:
    sample = _three()
    order = [2, 0, 1]
    moved = _three(
        asset_ids=tuple(ASSETS[j] for j in order),
        index=sample.index[:, order],
        loss=sample.loss[:, order],
    )
    assert moved.content_hash == sample.content_hash
    changed = _three(index=sample.index + np.array([[0.0, 0.0, 1.0]]))
    assert changed.content_hash != sample.content_hash


def test_a_written_table_of_three_assets_rejects_changed_rows() -> None:
    table = _three().to_table()
    position = table.schema.get_field_index("loss")
    edited = table.set_column(
        position, table.schema.field(position), pa.array([9.0] * table.num_rows)
    )
    with pytest.raises(SampleError, match="content hash"):
        OutcomeSample.from_written_table(edited)


def test_select_and_splits_act_on_every_asset_at_once() -> None:
    sample = _three(n=10)
    selected = sample.select([2001, 2004])
    np.testing.assert_array_equal(selected.loss, sample.loss[[1, 4], :])
    assert selected.lineage == sample.lineage
    assert selected.provenance.dependence == {"kind": "data"}
    design_block, held_out = holdout(sample, train_fraction=0.7)
    assert design_block.index.shape == (7, 3) and held_out.index.shape == (3, 3)
    assert held_out.period.tolist() == [2007, 2008, 2009]
    np.testing.assert_array_equal(held_out.index, sample.index[7:, :])
    _check_out_of_sample(design_block, held_out)
    with pytest.raises(LeakageError):
        _check_out_of_sample(sample, held_out)
    folds = year_block_kfold(sample, 5)
    assert all(block.index.shape == (2, 3) for _, block in folds)


# -- asset() -----------------------------------------------------------------


def test_asset_takes_one_column_and_keeps_the_lineage() -> None:
    sample = _three(hazard=np.arange(36.0).reshape(12, 3))
    view = sample.asset("site-c")
    assert view.asset_ids == ("site-c",)
    np.testing.assert_array_equal(view.period, sample.period)
    np.testing.assert_array_equal(view.probability, sample.probability)
    np.testing.assert_array_equal(view.index[:, 0], sample.index[:, 1])
    np.testing.assert_array_equal(view.loss[:, 0], sample.loss[:, 1])
    assert view.hazard is not None
    np.testing.assert_array_equal(view.hazard[:, 0], sample.hazard[:, 1])  # type: ignore[index]
    assert view.lineage == sample.lineage
    assert view.provenance.parent == sample.content_hash
    assert view.provenance.selection[-1] == {"kind": "asset", "asset_id": "site-c"}
    assert view.provenance.dependence is None and view.provenance.parts is None


def test_asset_of_a_one_asset_sample_is_the_sample() -> None:
    sample = _part("site-a")
    assert sample.asset("site-a") is sample


def test_asset_refuses_an_asset_the_sample_does_not_hold() -> None:
    with pytest.raises(SampleError, match="holds no asset 'nowhere'"):
        _three().asset("nowhere")
    with pytest.raises(SampleError, match="holds no asset 'nowhere'"):
        _part("site-a").asset("nowhere")


def test_asset_of_a_joined_sample_is_the_part() -> None:
    parts = [_part("a", seed=1), _part("b", seed=2, hazard=np.full(10, 9.0))]
    joined = OutcomeSample.join(parts)
    for part in parts:
        view = joined.asset(part.asset_ids[0])
        assert view.content_hash == part.content_hash
        assert view.lineage == part.lineage != joined.lineage
        assert view.provenance.sources == part.provenance.sources
        assert view.provenance.loss_model == part.provenance.loss_model
        assert view.provenance.notes == ("One note.",)
        assert view.provenance.parent == joined.content_hash


def test_asset_keeps_the_part_s_lineage_for_the_leakage_check() -> None:
    a, b = _part("a", seed=1), _part("b", seed=2)
    early_a, late_a = holdout(a, train_fraction=0.6)
    early_b, _ = holdout(b, train_fraction=0.6)
    joined = OutcomeSample.join([early_a, early_b])
    view = joined.asset("a")
    assert view.lineage == a.lineage
    assert [step["kind"] for step in view.provenance.selection] == ["holdout", "asset"]
    _check_out_of_sample(view, late_a)
    with pytest.raises(LeakageError):
        _check_out_of_sample(view, a)


def test_asset_survives_a_written_table() -> None:
    joined = OutcomeSample.join([_part("a", seed=1), _part("b", seed=2)])
    read = OutcomeSample.from_written_table(joined.to_table())
    assert read.asset("b").lineage == "lineage-of-b"
    assert read.asset("b").content_hash == joined.asset("b").content_hash


def test_a_block_of_a_joined_sample_stays_a_block_of_each_asset() -> None:
    joined = OutcomeSample.join([_part("a", seed=1), _part("b", seed=2)])
    design_block, held_out = holdout(joined, train_fraction=0.6)
    view, design_view = held_out.asset("a"), design_block.asset("a")
    assert view.provenance.selection[-1]["block"] == "test"
    table = evaluate(
        _linear("a"), view, split="out_of_sample", design_sample=design_view
    )
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.setup.evaluations[0]["method"] == "holdout"
    assert "the held-out block" in setup_text(table)


def test_a_design_runs_on_one_asset_of_a_joined_sample() -> None:
    joined = OutcomeSample.join([_part("a", seed=1), _part("b", seed=2)])
    view = joined.asset("a")
    template = Cover(
        INDEX, StepPayout([20.0, 60.0], [Free(), Free()]), payout_unit="EUR"
    )
    assert design(template, view, MSE()).design_sample is view


# -- join() ------------------------------------------------------------------


def test_join_puts_parts_side_by_side_and_records_the_data_s_dependence() -> None:
    a, b = _part("b", seed=1), _part("a", seed=2)
    joined = OutcomeSample.join([a, b])
    assert joined.asset_ids == ("b", "a")
    np.testing.assert_array_equal(joined.index[:, 0], a.index[:, 0])
    np.testing.assert_array_equal(joined.loss[:, 1], b.loss[:, 0])
    assert joined.hazard is None
    provenance = joined.provenance
    assert provenance.dependence == {"kind": "data"}
    assert provenance.source_kind == "annual"
    assert provenance.semantics_source == "data"
    assert set(provenance.parts or {}) == {"a", "b"}
    assert (provenance.parts or {})["b"]["lineage"] == "lineage-of-b"
    assert [source["asset_id"] for source in provenance.sources] == ["b", "a"]
    assert provenance.notes == ("Asset b: One note.", "Asset a: One note.")
    assert len(provenance.licences) == 1
    assert provenance.loss_model is None and provenance.basis_risk is None
    assert provenance.details == {"draws": 10, "periods": (2000, 2009)}


def test_join_builds_a_new_lineage_from_the_parts() -> None:
    a, b = _part("a", seed=1), _part("b", seed=2)
    joined = OutcomeSample.join([a, b])
    assert joined.lineage not in (a.lineage, b.lineage)
    assert OutcomeSample.join([b, a]).lineage == joined.lineage
    other = _part("b", seed=2, provenance=_provenance(lineage="elsewhere"))
    assert OutcomeSample.join([a, other]).lineage != joined.lineage


def test_join_resolves_a_part_without_a_stated_lineage() -> None:
    plain = _part("a", provenance=_provenance())
    joined = OutcomeSample.join([plain, _part("b", seed=2)])
    assert joined.asset("a").lineage == plain.lineage == plain.content_hash


def test_a_part_without_a_hazard_contributes_its_index() -> None:
    a, b = _part("a", seed=1), _part("b", seed=2, hazard=np.full(10, 9.0))
    hazard = OutcomeSample.join([a, b]).hazard
    assert hazard is not None
    np.testing.assert_array_equal(hazard[:, 0], a.index[:, 0])
    np.testing.assert_array_equal(hazard[:, 1], np.full(10, 9.0))


def test_join_records_the_generators_by_asset() -> None:
    record = {"step": "basis_risk", "bit_generator": "PCG64", "state": "{}"}
    a = _part(
        "a",
        provenance=SampleProvenance(
            source_kind="annual",
            sources=(),
            semantics_source="data",
            rng=(record,),
        ),
    )
    joined = OutcomeSample.join([a, _part("b", seed=2)])
    assert [dict(item) for item in joined.provenance.rng] == [
        {**record, "asset_id": "a"}
    ]


def test_parts_of_different_kinds_join_as_the_caller_s_statement() -> None:
    table = _part("b", seed=2, provenance=_provenance())
    joined = OutcomeSample.join([_part("a", seed=1), table])
    assert joined.provenance.source_kind == "table"
    assert joined.provenance.semantics_source == "caller"
    assert joined.provenance.dependence == {"kind": "data"}


def _with_other_semantics(sample: OutcomeSample) -> OutcomeSample:
    # Nothing can build another semantics yet: the rule guards the reserved one.
    object.__setattr__(sample, "semantics", "event")
    return sample


@pytest.mark.parametrize(
    ("other", "message"),
    [
        (lambda: _part("b", n=11), "period"),
        (
            lambda: _part("b", period=np.arange(2001, 2011)),
            "identical period columns, but asset 'b'",
        ),
        (
            lambda: _part("b", probability=np.r_[0.2, np.full(9, 0.8 / 9)]),
            "identical probability columns",
        ),
        (lambda: _part("b", index_unit="m"), "one index_unit"),
        (lambda: _part("b", loss_unit="USD"), "one loss_unit"),
        (lambda: _part("b", scenario="ssp585"), "one scenario"),
        (lambda: _with_other_semantics(_part("b")), "one semantics"),
        (lambda: _part("a"), "distinct asset ids"),
    ],
)
def test_join_requires_identical_rows_and_units(other: Any, message: str) -> None:
    with pytest.raises(SampleError, match=message):
        OutcomeSample.join([_part("a"), other()])


def test_join_refuses_monte_carlo_samples_naming_sample_curves() -> None:
    drawn = _curve_sample("site-b")
    with pytest.raises(SampleError, match="sample_curves"):
        OutcomeSample.join([_curve_sample("site-a", seed=1), drawn])
    with pytest.raises(SampleError, match=r"Monte Carlo sample of asset 'site-b'"):
        OutcomeSample.join([_part("site-a"), drawn])


def test_join_takes_one_asset_samples_only() -> None:
    with pytest.raises(SampleError, match=r"one-asset samples.*asset\(\)"):
        OutcomeSample.join([_three(n=10), _part("z")])


@pytest.mark.parametrize("count", [0, 1])
def test_join_needs_two_samples(count: int) -> None:
    with pytest.raises(SampleError, match="two samples or more"):
        OutcomeSample.join([_part("a")] * count)


def test_join_takes_any_iterable_of_samples() -> None:
    joined = OutcomeSample.join(part for part in (_part("a"), _part("b", seed=2)))
    assert joined.asset_ids == ("a", "b")


@pytest.mark.parametrize(
    "samples",
    ["ab", {"a": 1}, 3, [3, 4]],
    ids=["a string", "a mapping", "a number", "not samples"],
)
def test_join_refuses_anything_else(samples: Any) -> None:
    with pytest.raises(TypeError):
        UNTYPED_JOIN(samples)


def test_join_refuses_a_lone_sample() -> None:
    with pytest.raises(TypeError, match="iterable of OutcomeSample"):
        UNTYPED_JOIN(_part("a"))


# -- user tables -------------------------------------------------------------


def test_from_table_reads_several_assets() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample = _from_rows(_rows())
    assert sample.asset_ids == ("a", "b")
    assert sample.period.tolist() == [2000, 2001, 2002]
    np.testing.assert_array_equal(
        sample.index, [[10.0, 50.0], [25.0, 40.0], [15.0, 30.0]]
    )
    np.testing.assert_array_equal(sample.loss, [[4.0, 0.5], [2.0, 2.5], [5.0, 1.0]])
    np.testing.assert_allclose(sample.probability, 1 / 3)
    assert sample.provenance.dependence == {"kind": "data"}
    assert sample.provenance.source_kind == "table"
    assert sample.provenance.details["draws"] == 3


def test_a_table_of_several_assets_round_trips() -> None:
    sample = _from_rows(_rows())
    read = OutcomeSample.from_written_table(sample.to_table())
    assert read.content_hash == sample.content_hash
    assert read.asset("a").lineage == sample.lineage
    assert read.provenance.to_document()["dependence"] == {"kind": "data"}


def test_a_table_s_asset_keeps_the_table_s_lineage_and_sources() -> None:
    sample = _from_rows(_rows(), source="claims")
    view = sample.asset("b")
    assert view.lineage == sample.lineage
    assert view.provenance.source_kind == "table"
    assert view.provenance.sources == sample.provenance.sources
    assert view.provenance.selection == ({"kind": "asset", "asset_id": "b"},)
    assert _from_rows(_rows(), source="claims").lineage == sample.lineage
    assert _from_rows(_rows(), source="other").lineage != sample.lineage


def test_a_one_asset_table_has_no_dependence_and_keeps_its_lineage_rule() -> None:
    rows = {
        "year": pa.array([2001, 2000, 2002], pa.int64()),
        "asset": ["a", "a", "a"],
        "rain": [10.0, 20.0, 15.0],
        "loss": [3.0, 1.0, 9.0],
    }
    sample = _from_rows(rows, source="claims")
    assert sample.provenance.dependence is None and sample.provenance.parts is None
    assert sample.lineage == content_hash(
        {"kind": "table", "asset_id": "a", "source": {"name": "claims"}}
    )


def test_a_probability_column_must_agree_between_assets() -> None:
    rows = _rows()
    rows["p"] = [0.3, 0.3, 0.4, 0.3, 0.3, 0.4]
    sample = OutcomeSample.from_table(
        pa.table(rows),
        period="year",
        asset_id="asset",
        index="rain",
        loss="loss",
        probability="p",
        semantics="annual",
        index_unit="mm/day",
        loss_unit="EUR",
    )
    assert sample.probability.tolist() == [0.3, 0.4, 0.3]
    rows["p"] = [0.3, 0.3, 0.4, 0.4, 0.3, 0.3]
    with pytest.raises(SampleError, match="differs between the assets"):
        OutcomeSample.from_table(
            pa.table(rows),
            period="year",
            asset_id="asset",
            index="rain",
            loss="loss",
            probability="p",
            semantics="annual",
            index_unit="mm/day",
            loss_unit="EUR",
        )


def test_every_pair_of_a_table_must_appear_exactly_once() -> None:
    table = pa.table(_rows())
    rows: dict[str, Any] = table.slice(0, 5).to_pydict()
    rows["year"] = pa.array(rows["year"], pa.int64())
    with pytest.raises(SampleError, match=r"asset 'b' has no row for periods \[2001\]"):
        _from_rows(rows)
    repeated = {**_rows(), "year": pa.array([2002, 2000, 2001, 2000, 2002, 2000])}
    with pytest.raises(SampleError, match=r"\[2000\] appear more than once .* 'b'"):
        _from_rows(repeated)


def test_a_table_warns_for_each_asset_whose_loss_follows_its_index() -> None:
    rows = _rows()
    rows["loss"] = [30.0, 4.0, 2.0, 50.0, 5.0, 40.0]
    with pytest.warns(ComonotoneIndexWarning, match=r"assets b\)"):
        sample = _from_rows(rows)
    assert any(note.startswith("Asset b:") for note in sample.provenance.notes)
    assert any("comonotone" in note for note in sample.asset("b").provenance.notes)
    assert sample.asset("a").provenance.notes == ()


# -- the refusals ------------------------------------------------------------


def _assert_names_the_way_out(function: Any) -> None:
    with pytest.raises(SampleError) as raised:
        function()
    message = str(raised.value)
    assert "asset(" in message and "design_joint" in message


def test_design_refuses_a_sample_of_several_assets() -> None:
    template = Cover(INDEX, StepPayout([20.0], [Free()]), payout_unit="EUR")
    _assert_names_the_way_out(lambda: design(template, _three(), MSE()))
    joined = OutcomeSample.join([_part("a"), _part("b", seed=2)])
    _assert_names_the_way_out(lambda: design(template, joined, MSE()))


def test_evaluate_refuses_a_sample_of_several_assets() -> None:
    _assert_names_the_way_out(lambda: evaluate(_linear(), _three(), split="in_sample"))


def test_compare_refuses_a_sample_of_several_assets() -> None:
    _assert_names_the_way_out(lambda: compare([_linear()], _three(), split="in_sample"))


def test_a_design_sample_of_several_assets_is_refused_too() -> None:
    _assert_names_the_way_out(
        lambda: evaluate(
            _linear(),
            _part("a", period=np.arange(3000, 3010)),
            split="out_of_sample",
            design_sample=_three(),
        )
    )


def test_a_cover_is_evaluated_on_one_asset_of_a_table() -> None:
    sample = _from_rows(_rows())
    table = evaluate(_linear("a"), sample.asset("a"), split="in_sample")
    assert table.column("asset_id").to_pylist() == ["a"]
    assert table.column("level").to_pylist() == ["asset"]
