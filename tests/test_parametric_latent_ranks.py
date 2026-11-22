"""A `GaussianCopula` sample keeps the ranks of its latent normal scores.

Re-dealing a block's draws by the model's ranks, as `bootstrap_intervals` does
on rows built together, needs each row's rank in each column: the index ties at
a curve's atom, and a basis-risk model of the caller's may move it. These tests
pin what the samplers keep, against a separate computation of the copula's
latent scores, what a sample checks, its blocks and views, its content hash and
its table. Every number is synthetic.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from crc_framework import FittedDistribution, PiecewiseLinearImpact

from crc_sdk.parametric import (
    AnnualSeries,
    CappedSum,
    Comonotonic,
    EmpiricalCopula,
    GaussianCopula,
    HazardNoise,
    Independent,
    LossModel,
    NoBasisRisk,
    OutcomeSample,
    SampleError,
    holdout,
    sample_annual,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric import sample as sample_module
from crc_sdk.parametric.sample import (
    SAMPLE_METADATA_KEY,
    SAMPLE_RANKS_SCHEMA_VERSION,
    _hash_document,
)
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
    ),
]

RAMP = PiecewiseLinearImpact([30.0, 120.0], [0.0, 1.0])
ASSETS = ("site-a", "site-b")
HAZARDS = ("rain", "wind")
UNITS = {"rain": "mm/day", "wind": "m/s"}
LOCATIONS = {"site-a": 50.0, "site-b": 60.0, "rain": 50.0, "wind": 60.0}
N = 40

_BEFORE = {
    "assets gaussian": (
        "sha256:d1f3456b6c53ce5bc81d0d4ec523bce33d13224a8bb61d20505c57922b5bb4c4"
    ),
    "assets comonotonic": (
        "sha256:1396597fd47dceef5f07285f7e175f198aaf568d37cf09cc9d557f7860a404d4"
    ),
    "hazards gaussian": (
        "sha256:19317c143000592bcb0829feb82744e6785b47c2f6cb606efc539760a8f4ed20"
    ),
    "hazards comonotonic": (
        "sha256:25c2650b6a7ddc35e95d65c38fed4745ba383c271b72cff12dc71e157c0db216"
    ),
}
"""What these samples hashed to before samples kept their latent ranks."""

_KEPT = {
    "assets gaussian": (
        "sha256:2df1508daf63100ba9baa514f350f36adfb363596852b2772c8c0cb74ea85fc0"
    ),
    "hazards gaussian": (
        "sha256:b17c48129332ae2ab36a3cdcc9a388c3797153ae20965a404c296e8b2f03c04c"
    ),
}
"""What the `GaussianCopula` samples hash to now that they keep them."""


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


def _assets(model: Any, n: int = N, curves: Any = None) -> OutcomeSample:
    return sample_curves(
        {a: _curve(a) for a in ASSETS} if curves is None else curves,
        dependence=model,
        metadata={a: _metadata() for a in ASSETS},
        n_draws=n,
        rngs={a: np.random.default_rng(11 + j) for j, a in enumerate(ASSETS)},
        loss_models={
            a: LossModel(RAMP, value=100.0, value_unit="EUR", name=f"loss of {a}")
            for a in ASSETS
        },
        basis_risk=HazardNoise(0.3),
    )


def _hazards(model: Any, n: int = N) -> OutcomeSample:
    return sample_curve_hazards(
        {h: _curve(h) for h in HAZARDS},
        dependence=model,
        metadata={h: _metadata(UNITS[h]) for h in HAZARDS},
        n_draws=n,
        rngs={h: np.random.default_rng(21 + j) for j, h in enumerate(HAZARDS)},
        loss_models={h: LossModel(RAMP, name=f"damage ratio of {h}") for h in HAZARDS},
        combine=CappedSum(1.0),
        basis_risk=HazardNoise(0.3),
        asset_id="site",
    )


def _latent_ranks(seeds: list[int], rho: float, n: int) -> np.ndarray:
    """The ranks of y = z L^T, written apart from the module, row by row.

    Column j's draws come from `default_rng(seeds[j]).random(n)`, first; z_j
    are their normal scores in draw order, and the first column stays in draw
    order, so row i of y is the latent score of row i of the sample.
    """
    quantile = NormalDist().inv_cdf
    tiny = 2.0**-53
    uniforms = [np.random.default_rng(seed).random(n) for seed in seeds]
    z = np.column_stack(
        [
            [quantile(min(max(u, tiny), 1.0 - tiny)) for u in column]
            for column in uniforms
        ]
    )
    factor = np.array([[1.0, 0.0], [rho, math.sqrt(1.0 - rho * rho)]])
    y = z @ factor.T
    return np.argsort(np.argsort(y, axis=0, kind="stable"), axis=0, kind="stable")


def _without_ranks(sample: OutcomeSample) -> OutcomeSample:
    """The sample as one written before samples kept their latent ranks."""
    return dataclasses.replace(sample, latent_ranks=None)


# -- what the samplers keep ----------------------------------------------------


def test_a_gaussian_sample_of_assets_keeps_its_latent_ranks() -> None:
    sample = _assets(GaussianCopula(0.6))
    assert sample.latent_ranks is not None
    assert sample.latent_ranks.dtype == np.int64
    assert sample.latent_ranks.shape == (N, 2)
    assert np.array_equal(sample.latent_ranks, _latent_ranks([11, 12], 0.6, N))
    # Read-only, and each column a permutation of the rows.
    with pytest.raises(ValueError):
        sample.latent_ranks[0, 0] = 1
    for j in range(2):
        assert sorted(sample.latent_ranks[:, j]) == list(range(N))
        # Without an atom the index rises with the rank.
        order = np.argsort(sample.latent_ranks[:, j])
        assert (np.diff(sample.index[order, j]) > 0).all()


def test_a_gaussian_sample_of_hazards_keeps_its_latent_ranks() -> None:
    sample = _hazards(GaussianCopula(-0.5))
    assert sample.latent_ranks is not None
    # One column per component, in component order.
    assert sample.latent_ranks.shape == (N, 2)
    assert np.array_equal(sample.latent_ranks, _latent_ranks([21, 22], -0.5, N))


def _horizon(horizon: int, location: float) -> dict[str, Any]:
    return {
        "cell_index": 613196570331971583,
        "source_id": "fixture-1",
        "source_geometry": None,
        "hazard_name": "rx1day",
        "horizon": horizon,
        "pathway": "ssp245",
        "curve_kind": "fitted",
        "curve_type": "gumbel_r",
        "curve_shape": None,
        "curve_location": location,
        "curve_scale": 10.0,
        "curve_atom_probability": None,
        "curve_atom_location": None,
    }


def test_a_window_keeps_each_members_ranks() -> None:
    # The copula reorders within each member's rows (row d reads member d mod 2).
    window = _assets(
        GaussianCopula(0.6),
        curves={
            "site-a": [_horizon(2045, 50.0), _horizon(2055, 60.0)],
            "site-b": [_horizon(2045, 55.0), _horizon(2055, 70.0)],
        },
    )
    assert window.latent_ranks is not None
    for member in range(2):
        rows = np.arange(member, N, 2)
        for j in range(2):
            ranks = window.latent_ranks[rows, j]
            assert sorted(ranks) == list(range(N // 2))
            order = np.argsort(ranks)
            assert (np.diff(window.index[rows[order], j]) > 0).all()


def _reference() -> OutcomeSample:
    years = np.arange(1991, 2001)
    values = np.random.default_rng(5).gumbel(50.0, 10.0, (10, 2))
    parts = [
        sample_annual(
            AnnualSeries(
                years, values[:, j], "mm/day", {"source": {"name": f"series {a}"}}
            ),
            loss_model=LossModel(RAMP, value=100.0, value_unit="EUR"),
            basis_risk=NoBasisRisk(),
            asset_id=a,
        )
        for j, a in enumerate(ASSETS)
    ]
    return OutcomeSample.join(parts)


@pytest.mark.parametrize(
    "model",
    [Independent, Comonotonic, lambda: EmpiricalCopula(_reference())],
    ids=["independent", "comonotonic", "copula"],
)
def test_every_other_model_keeps_none(model: Any) -> None:
    assert _assets(model()).latent_ranks is None


@pytest.mark.parametrize("model", [Independent, Comonotonic])
def test_every_other_model_across_hazards_keeps_none(model: Any) -> None:
    assert _hazards(model()).latent_ranks is None


# -- what a sample checks ------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ranks: ranks.astype(np.float64), "must hold integers"),
        (lambda ranks: ranks[:, :1], "one column per asset"),
        (lambda ranks: ranks[:-1], "one column per asset"),
        (lambda ranks: ranks - 1, "non-negative"),
    ],
    ids=["floats", "one column", "rows", "negative"],
)
def test_invalid_latent_ranks_raise(change: Any, message: str) -> None:
    sample = _assets(GaussianCopula(0.6))
    assert sample.latent_ranks is not None
    with pytest.raises(SampleError, match=message):
        dataclasses.replace(sample, latent_ranks=change(sample.latent_ranks))


def test_latent_ranks_need_the_record_of_a_gaussian_copula() -> None:
    ranks = _assets(GaussianCopula(0.6)).latent_ranks
    with pytest.raises(SampleError, match="this one records 'comonotonic'"):
        dataclasses.replace(_assets(Comonotonic()), latent_ranks=ranks)
    hazards = _hazards(GaussianCopula(-0.5))
    with pytest.raises(SampleError, match="one column per component"):
        dataclasses.replace(hazards, latent_ranks=hazards.latent_ranks[:, :1])  # type: ignore[index]


def test_a_sample_of_one_column_keeps_no_ranks() -> None:
    sample = _assets(GaussianCopula(0.6))
    one = sample.asset("site-a")
    with pytest.raises(SampleError, match="one column per asset of a sample of"):
        dataclasses.replace(one, latent_ranks=np.zeros((N, 1), dtype=np.int64))


# -- blocks and views ------------------------------------------------------------


def test_a_block_keeps_its_rows_ranks_and_a_view_keeps_none() -> None:
    sample = _assets(GaussianCopula(0.6))
    assert sample.latent_ranks is not None
    block = sample.select(sample.period[5:25])
    assert np.array_equal(block.latent_ranks, sample.latent_ranks[5:25])  # type: ignore[arg-type]
    train, test = holdout(sample)
    assert np.array_equal(
        np.vstack([train.latent_ranks, test.latent_ranks]),  # type: ignore[list-item]
        sample.latent_ranks,
    )
    assert sample.asset("site-b").latent_ranks is None
    hazards = _hazards(GaussianCopula(-0.5))
    assert hazards.component("rain").latent_ranks is None
    assert hazards.component("rain", loss="combined").latent_ranks is None


# -- the content hash --------------------------------------------------------------


def test_the_gaussian_samples_hashes_change_and_the_others_stay() -> None:
    samples = {
        "assets gaussian": _assets(GaussianCopula(0.6)),
        "assets comonotonic": _assets(Comonotonic()),
        "hazards gaussian": _hazards(GaussianCopula(-0.5)),
        "hazards comonotonic": _hazards(Comonotonic()),
    }
    for name, sample in samples.items():
        if name in _KEPT:
            assert sample.content_hash == _KEPT[name], name
            # Without its ranks it hashes as it did before they were kept, so a
            # table written then reads back with its hash.
            assert _without_ranks(sample).content_hash == _BEFORE[name], name
        else:
            assert sample.content_hash == _BEFORE[name], name


def test_the_ranks_are_part_of_the_hash() -> None:
    sample = _assets(GaussianCopula(0.6))
    assert sample.latent_ranks is not None
    swapped = sample.latent_ranks.copy()
    swapped[[0, 1], 1] = swapped[[1, 0], 1]
    assert dataclasses.replace(sample, latent_ranks=swapped).content_hash != (
        sample.content_hash
    )


def test_the_hash_document_holds_the_ranks_under_its_own_version() -> None:
    sample = _assets(GaussianCopula(0.6))
    fields = {
        name: getattr(sample, name)
        for name in (
            "asset_ids",
            "period",
            "probability",
            "index",
            "loss",
            "hazard",
            "index_unit",
            "loss_unit",
            "semantics",
            "scenario",
        )
    }
    document = _hash_document(**fields, latent_ranks=sample.latent_ranks)
    assert document["schema_version"] == SAMPLE_RANKS_SCHEMA_VERSION == "1.3"
    assert document["latent_rank"] == sample.latent_ranks.reshape(-1).tolist()  # type: ignore[union-attr]
    plain = _hash_document(**fields)
    assert "latent_rank" not in plain and plain["schema_version"] == "1.0"
    hazards = _hazards(GaussianCopula(-0.5))
    entries = _hash_document(
        **{**fields, **{name: getattr(hazards, name) for name in fields}},
        components=hazards.index_components,
        latent_ranks=hazards.latent_ranks,
    )["components"]
    assert [entry["latent_rank"] for entry in entries] == (
        hazards.latent_ranks.T.tolist()  # type: ignore[union-attr]
    )
    assert all("damage" in entry for entry in entries)


def test_the_version_is_exported_from_the_sample_module() -> None:
    assert "SAMPLE_RANKS_SCHEMA_VERSION" in sample_module.__all__


# -- the table -------------------------------------------------------------------


def _version(table: Any) -> str:
    return str(
        json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])[
            "schema_version"
        ]
    )


def test_a_sample_of_assets_writes_and_reads_its_ranks(tmp_path: Path) -> None:
    sample = _assets(GaussianCopula(0.6))
    table = sample.to_table()
    assert table.schema.names[-1] == "latent_rank"
    assert table.schema.field("latent_rank").type == pa.int64()
    assert table.column("latent_rank").null_count == 0
    assert _version(table) == "1.3"
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    for written in (table, pq.read_table(path)):
        read = OutcomeSample.from_written_table(written)
        assert read.content_hash == sample.content_hash
        assert np.array_equal(read.latent_ranks, sample.latent_ranks)  # type: ignore[arg-type]
        assert read.to_table().equals(table)


def test_a_sample_of_hazards_writes_and_reads_its_ranks() -> None:
    sample = _hazards(GaussianCopula(-0.5))
    table = sample.to_table()
    names = table.schema.names
    for name in HAZARDS:
        place = names.index(f"latent_rank[{name}]")
        assert names[place - 1] == f"damage[{name}]"
        assert table.schema.field(place).type == pa.int64()
    assert _version(table) == "1.3"
    read = OutcomeSample.from_written_table(table)
    assert read.content_hash == sample.content_hash
    assert np.array_equal(read.latent_ranks, sample.latent_ranks)  # type: ignore[arg-type]
    assert read.index_components is not None
    assert all(each.damage is not None for each in read.index_components.values())
    # Without their damages, the components read back without them too.
    stripped = dataclasses.replace(
        sample,
        index_components={
            name: dataclasses.replace(each, damage=None)
            for name, each in sample.index_components.items()  # type: ignore[union-attr]
        },
    )
    again = OutcomeSample.from_written_table(stripped.to_table())
    assert again.content_hash == stripped.content_hash
    assert again.index_components is not None
    assert all(each.damage is None for each in again.index_components.values())


def _relabelled(table: Any, version: str) -> Any:
    document = json.loads(table.schema.metadata[SAMPLE_METADATA_KEY.encode()])
    document["schema_version"] = version
    return table.replace_schema_metadata(
        {SAMPLE_METADATA_KEY: json.dumps(document, sort_keys=True)}
    )


def test_a_written_sample_with_ranks_needs_them() -> None:
    table = _assets(GaussianCopula(0.6)).to_table()
    with pytest.raises(SampleError, match="latent_rank"):
        OutcomeSample.from_written_table(table.drop_columns(["latent_rank"]))
    edited = table.set_column(
        table.schema.get_field_index("latent_rank"),
        "latent_rank",
        pa.array(table.column("latent_rank").to_numpy()[::-1].copy(), pa.int64()),
    )
    with pytest.raises(SampleError, match="content hash"):
        OutcomeSample.from_written_table(edited)
    # A reader of the earlier version would drop the ranks; the hash says so.
    with pytest.raises(SampleError, match="content hash"):
        OutcomeSample.from_written_table(_relabelled(table, "1.0"))
    with pytest.raises(SampleError, match="'1.3'"):
        OutcomeSample.from_written_table(_relabelled(table, "1.4"))


def test_a_written_sample_of_hazards_holds_every_damage_or_none() -> None:
    table = _hazards(GaussianCopula(-0.5)).to_table()
    with pytest.raises(SampleError, match=r"but not of \['wind'\]"):
        OutcomeSample.from_written_table(table.drop_columns(["damage[wind]"]))
    with pytest.raises(SampleError, match=r"latent_rank\[rain\]"):
        OutcomeSample.from_written_table(table.drop_columns(["latent_rank[rain]"]))


def test_a_gaussian_sample_written_before_reads_back_without_ranks() -> None:
    # Its table holds no ranks, under its earlier version, and its hash is the
    # one it was written with.
    for sample, version in (
        (_assets(GaussianCopula(0.6)), "1.0"),
        (_hazards(GaussianCopula(-0.5)), "1.2"),
    ):
        before = _without_ranks(sample)
        table = before.to_table()
        assert _version(table) == version
        assert not any(name.startswith("latent_rank") for name in table.schema.names)
        read = OutcomeSample.from_written_table(table)
        assert read.latent_ranks is None
        assert read.content_hash == before.content_hash
