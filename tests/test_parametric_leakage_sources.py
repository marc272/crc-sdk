"""The leakage check across sample kinds: a year of one source, in any sample.

The numbers are synthetic. The tests check which rows the check counts as
shared and what the error says; nothing here is a finding about a hazard or a
cover.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from crc_framework import FittedDistribution

from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    CappedSum,
    Comonotonic,
    Cover,
    Free,
    GaussianCopula,
    Independent,
    IndexDefinition,
    LeakageError,
    LinearPayout,
    LinearTrend,
    LossModel,
    ModelRun,
    NoBasisRisk,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    compare,
    design,
    evaluate,
    holdout,
    sample_annual,
    sample_annual_hazards,
    sample_annual_pool,
    sample_curve,
    sample_curve_hazards,
    sample_curves,
)
from crc_sdk.parametric.sample import _check_out_of_sample
from crc_sdk.types import HazardDatasetMetadata, SourceProvenance

pytestmark = pytest.mark.filterwarnings(
    "ignore::crc_sdk.parametric.errors.ComonotoneIndexWarning"
)

UNTYPED_RUN: Any = ModelRun
ASSET = "site"
YEARS = np.arange(2000, 2012)
FIRST_YEARS = "[2000, 2001, 2002, 2003, 2004]"
LOSS = LossModel(
    lambda hazard: hazard / 200.0, value=1000.0, value_unit="EUR", name="ramp"
)
RAIN_INDEX = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
FIXED = Cover(RAIN_INDEX, LinearPayout(10.0, 90.0, 500.0), payout_unit="EUR")


def _series(name: str, *, years: Any = YEARS) -> AnnualSeries:
    """Whole numbers, so that a pinned hash is the same on every platform."""
    shift = sum(name.encode("utf-8"))
    return AnnualSeries(
        np.asarray(years),
        np.array([float((year * 7 + shift) % 89 + 1) for year in years]),
        "mm/day",
        {"source": {"name": name}},
    )


RAIN, HEAT, WIND = _series("rain"), _series("heat"), _series("wind")


def _plain(series: AnnualSeries = RAIN, **fields: Any) -> OutcomeSample:
    arguments: dict[str, Any] = {
        "loss_model": LOSS,
        "basis_risk": NoBasisRisk(),
        "asset_id": ASSET,
        **fields,
    }
    return sample_annual(series, **arguments)


def _hazards(
    series: dict[str, AnnualSeries] | None = None, **fields: Any
) -> OutcomeSample:
    chosen = {"rain": RAIN, "heat": HEAT} if series is None else series
    arguments: dict[str, Any] = {
        "tails": dict.fromkeys(chosen, "upper"),
        "loss_models": dict.fromkeys(chosen, LOSS),
        "combine": CappedSum(1.0),
        "basis_risk": NoBasisRisk(),
        "asset_id": ASSET,
        **fields,
    }
    return sample_annual_hazards(chosen, **arguments)


def _join(*series: tuple[str, AnnualSeries]) -> OutcomeSample:
    return OutcomeSample.join([_plain(each, asset_id=asset) for asset, each in series])


def _run(series: AnnualSeries, model: str) -> Any:
    return UNTYPED_RUN(series, model=model, member="r1", pathway="ssp245")


def _pool(*runs: tuple[str, AnnualSeries], asset: str = ASSET) -> OutcomeSample:
    return sample_annual_pool(
        [_run(series, model) for model, series in runs],
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id=asset,
    )


def _table(
    years: Any, assets: list[str], source: Any = "the table", **fields: Any
) -> OutcomeSample:
    rows = [int(year) for year in years]
    columns: dict[str, Any] = {
        "year": [year for year in rows for _ in assets],
        "site": [asset for _ in rows for asset in assets],
        "x": [float(year - 1999) for year in rows for _ in assets],
        "y": [float(year - 1990) for year in rows for _ in assets],
        "loss": [float(year * 7 % 11) for year in rows for _ in assets],
    }
    arguments: dict[str, Any] = {
        "period": "year",
        "asset_id": "site",
        "index": "x",
        "loss": "loss",
        "semantics": "annual",
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "source": source,
        **fields,
    }
    return OutcomeSample.from_table(columns, **arguments)


def _metadata(name: str) -> HazardDatasetMetadata:
    return HazardDatasetMetadata(
        h3_resolution=7,
        value_unit="mm/day",
        value_semantics=f"annual maximum {name}",
        producer="tests",
        source=SourceProvenance(
            provider="fixture",
            dataset=name,
            version="2026.1",
            licence="CC-BY-4.0",
            attribution="Synthetic fixture",
            checksum="sha256:" + "0" * 64,
        ),
        creation_version="1",
        probability_semantics="annual_exceedance",
    )


def _curve() -> FittedDistribution:
    return FittedDistribution.from_parameters("gumbel_r", location=50.0, scale=10.0)


def _one_curve(name: str = "rain", *, seed: int = 11, asset: str = ASSET) -> Any:
    return sample_curve(
        _curve(),
        metadata=_metadata(name),
        n_draws=40,
        rng=np.random.default_rng(seed),
        loss_model=LOSS,
        basis_risk=NoBasisRisk(),
        asset_id=asset,
    )


def _curve_hazards(*, seeds: tuple[int, int] = (11, 22), dependence: Any = None) -> Any:
    names = ("rain", "heat")
    return sample_curve_hazards(
        dict.fromkeys(names, _curve()),
        dependence=Independent() if dependence is None else dependence,
        metadata={name: _metadata(name) for name in names},
        n_draws=40,
        rngs={
            name: np.random.default_rng(seed)
            for name, seed in zip(names, seeds, strict=True)
        },
        loss_models=dict.fromkeys(names, LOSS),
        combine=CappedSum(1.0),
        basis_risk=NoBasisRisk(),
        asset_id=ASSET,
    )


def _curve_assets(*, seeds: tuple[int, int] = (11, 22)) -> Any:
    names = ("a", "b")
    return sample_curves(
        dict.fromkeys(names, _curve()),
        dependence=Independent(),
        metadata={"a": _metadata("rain"), "b": _metadata("heat")},
        n_draws=40,
        rngs={
            name: np.random.default_rng(seed)
            for name, seed in zip(names, seeds, strict=True)
        },
        loss_models=dict.fromkeys(names, LOSS),
        basis_risk=NoBasisRisk(),
    )


def _shares(design_sample: OutcomeSample, evaluation: OutcomeSample) -> str:
    with pytest.raises(LeakageError) as caught:
        _check_out_of_sample(design_sample, evaluation)
    return str(caught.value)


# -- one series and a sample of several hazards ---------------------------------


def test_a_design_on_one_series_is_refused_on_several_hazards_of_its_years() -> None:
    message = _shares(_plain(), _hazards())
    assert message.startswith("the evaluation sample shares 12 periods with")
    assert FIRST_YEARS in message
    assert "from the same source ('rain' of asset 'site')" in message


def test_a_design_on_several_hazards_is_refused_on_one_of_its_series() -> None:
    message = _shares(_hazards(), _plain(HEAT))
    assert message.startswith("the evaluation sample shares 12 periods with")
    assert "'heat' of asset 'site'" in message


def test_the_leak_is_refused_where_a_user_evaluates() -> None:
    template = Cover(RAIN_INDEX, StepPayout([50.0], [Free()]), payout_unit="EUR")
    early, _ = holdout(_plain(), 0.5)
    made = design(template, early, MSE())
    with pytest.raises(LeakageError, match="'rain' of asset 'site'"):
        compare([made], _hazards(), split="out_of_sample")
    with pytest.raises(LeakageError, match="'rain' of asset 'site'"):
        evaluate(made, _hazards(), split="out_of_sample")
    _, late = holdout(_hazards(), 0.5)
    assert compare([made], late, split="out_of_sample").num_rows == 1


def test_only_the_years_held_by_both_are_counted() -> None:
    message = _shares(_plain(), _hazards().select(np.array([2005, 2006, 2011])))
    assert message.startswith("the evaluation sample shares 3 periods with")
    assert "[2005, 2006, 2011]" in message


def test_one_shared_year_is_named_in_the_singular() -> None:
    message = _shares(_plain().select([2003]), _hazards())
    assert message.startswith("the evaluation sample shares 1 period with")
    assert "[2003]" in message


def test_disjoint_years_of_one_series_pass_through_every_kind() -> None:
    early_plain, late_plain = holdout(_plain(), 0.5)
    early_hazards, late_hazards = holdout(_hazards(), 0.5)
    _check_out_of_sample(early_plain, late_hazards)
    _check_out_of_sample(early_hazards, late_plain)
    _check_out_of_sample(early_plain, late_hazards.component("rain"))
    early_join, late_join = holdout(_join((ASSET, RAIN), ("b", HEAT)), 0.5)
    _check_out_of_sample(early_plain, late_join)
    _check_out_of_sample(early_join, late_plain)


def test_windows_of_one_series_with_a_stated_source_pass_when_disjoint() -> None:
    def window(first: int, last: int) -> AnnualSeries:
        wanted = np.arange(first, last + 1)
        keep = np.isin(RAIN.years, wanted)
        return AnnualSeries(wanted, RAIN.values[keep], RAIN.unit, RAIN.provenance)

    early = _plain(window(2000, 2005))
    late = _hazards(
        {"rain": window(2006, 2011), "heat": _series("heat", years=range(2006, 2012))}
    )
    _check_out_of_sample(early, late)
    overlapping = _hazards(
        {"rain": window(2005, 2011), "heat": _series("heat", years=range(2005, 2012))}
    )
    assert "shares 1 period with" in _shares(early, overlapping)


def test_other_series_are_no_leak() -> None:
    _check_out_of_sample(_plain(WIND), _hazards())
    _check_out_of_sample(_hazards(), _plain(WIND))
    _check_out_of_sample(_hazards({"rain": WIND, "heat": _series("gust")}), _hazards())


def test_two_samples_of_several_hazards_that_share_one_series_raise() -> None:
    message = _shares(_hazards(), _hazards({"rain": RAIN, "wind": WIND}))
    assert "'rain' of asset 'site'" in message
    assert "'wind'" not in message and "'heat'" not in message


def test_the_same_series_at_another_asset_is_another_source() -> None:
    _check_out_of_sample(_plain(asset_id="elsewhere"), _hazards())


# -- the views of a sample of several hazards -----------------------------------


def test_a_components_view_links_to_its_own_series() -> None:
    view = _hazards().component("rain")
    assert "'rain' of asset 'site'" in _shares(_plain(), view)
    assert "'rain' of asset 'site'" in _shares(view, _plain())


def test_a_view_of_its_own_loss_does_not_read_the_other_series() -> None:
    _check_out_of_sample(_plain(HEAT), _hazards().component("rain"))
    _check_out_of_sample(_hazards().component("rain"), _plain(HEAT))


def test_a_view_of_the_combined_loss_reads_every_series_in_that_loss() -> None:
    view = _hazards().component("rain", loss="combined")
    assert "'heat' of asset 'site'" in _shares(_plain(HEAT), view)
    assert "'rain' of asset 'site'" in _shares(_plain(), view)


def test_a_view_of_the_combined_loss_does_not_read_a_component_without_impact() -> None:
    sample = _hazards(
        {"rain": RAIN, "heat": HEAT, "wind": WIND},
        loss_models={"rain": LOSS, "heat": LOSS},
        tails={"rain": "upper", "heat": "upper", "wind": "upper"},
    )
    view = sample.component("rain", loss="combined")
    _check_out_of_sample(_plain(WIND), view)
    assert "'heat' of asset 'site'" in _shares(_plain(HEAT), view)
    # The whole sample holds that series all the same.
    assert "'wind' of asset 'site'" in _shares(_plain(WIND), sample)


def test_the_views_of_one_sample_stay_one_source() -> None:
    sample = _hazards()
    train, test = holdout(sample, 0.5)
    _check_out_of_sample(train.component("rain"), test.component("heat"))
    message = _shares(sample.component("rain"), sample.component("heat"))
    assert message.startswith("the evaluation sample shares 12 periods with")


def test_splits_keep_the_sources_of_a_sample() -> None:
    train, test = holdout(_hazards(), 0.5)
    assert train.provenance.source_lineages == _hazards().provenance.source_lineages
    assert test.period.tolist() == YEARS[6:].tolist()
    assert "shares 6 periods" in _shares(_plain(), test)
    assert "shares 6 periods" in _shares(_plain(), test.component("rain"))


def test_a_written_sample_keeps_its_sources() -> None:
    sample = _hazards()
    read = OutcomeSample.from_written_table(sample.to_table())
    assert read.provenance.source_lineages == sample.provenance.source_lineages
    assert read.content_hash == sample.content_hash
    assert "'rain' of asset 'site'" in _shares(_plain(), read)


def test_a_trend_fitted_on_a_held_out_year_is_seen_through_several_hazards() -> None:
    early, _ = holdout(_plain(), 0.5)
    _, late = holdout(_hazards(trend=LinearTrend(min_years=3)), 0.5)
    with pytest.raises(LeakageError, match="linear trend was fitted on 6 years"):
        _check_out_of_sample(early, late)
    stated = LinearTrend(min_years=3, fit_years=YEARS[:6].tolist())
    _, fair = holdout(_hazards(trend=stated), 0.5)
    _check_out_of_sample(early, fair)


# -- a per-year sample with an index series -------------------------------------


def test_a_sample_with_an_index_series_holds_its_hazard_and_its_index() -> None:
    sample = _plain(index=HEAT)
    assert "'hazard' of asset 'site'" in _shares(_plain(), sample)
    assert "'index' of asset 'site'" in _shares(_plain(HEAT), sample)
    assert "'index' of asset 'site'" in _shares(sample, _plain(HEAT))
    _check_out_of_sample(_plain(WIND), sample)


# -- pools ----------------------------------------------------------------------


def test_a_run_of_a_pool_is_the_series_in_a_sample_of_several_hazards() -> None:
    pool = _pool(("M", RAIN), ("N", WIND))
    message = _shares(pool, _hazards())
    assert message.startswith("the evaluation sample shares 12 model-years with")
    assert "run M r1 of asset 'site'" in message
    message = _shares(_hazards(), pool)
    assert message.startswith("the evaluation sample shares 12 model-years with")
    _check_out_of_sample(_pool(("M", WIND), ("N", _series("gust"))), _hazards())


# -- join -----------------------------------------------------------------------


def test_a_design_on_one_series_is_refused_on_a_join_that_holds_it() -> None:
    joined = _join((ASSET, RAIN), ("b", HEAT))
    message = _shares(_plain(), joined)
    assert message.startswith("the evaluation sample shares 12 periods with")
    assert FIRST_YEARS in message
    assert "from the same source (asset 'site')" in message
    assert "asset 'site'" in _shares(joined, _plain())


def test_a_part_shared_between_joins_is_refused_where_a_user_evaluates() -> None:
    covers = {ASSET: FIXED, "b": FIXED}
    given = _join((ASSET, RAIN), ("b", WIND))
    joined = _join((ASSET, RAIN), ("b", HEAT))
    assert given.lineage != joined.lineage
    with pytest.raises(LeakageError, match="asset 'site'"):
        evaluate(covers, joined, split="out_of_sample", design_sample=given)
    train, _ = holdout(given, 0.5)
    _, late = holdout(joined, 0.5)
    table = evaluate(covers, late, split="out_of_sample", design_sample=train)
    assert table.num_rows


def test_a_shared_row_of_a_join_is_counted_once() -> None:
    joined = _join((ASSET, RAIN), ("b", HEAT))
    other = _join((ASSET, RAIN), ("b", HEAT))
    message = _shares(joined, other)
    assert message.startswith("the evaluation sample shares 12 periods with")
    assert "shares 12 periods" in _shares(_join((ASSET, RAIN), ("b", WIND)), joined)


def test_a_join_and_its_part_of_another_asset_are_told_apart() -> None:
    _check_out_of_sample(_join(("a", RAIN), ("b", HEAT)), _plain(RAIN, asset_id="c"))


def test_a_part_with_an_index_series_is_seen_through_the_join() -> None:
    joined = OutcomeSample.join([_plain(index=HEAT), _plain(WIND, asset_id="b")])
    assert "'index' of asset 'site'" in _shares(_plain(HEAT), joined)


def test_the_view_of_a_join_of_samples_keeps_its_part() -> None:
    joined = _join((ASSET, RAIN), ("b", HEAT))
    assert "asset 'site'" in _shares(joined.asset(ASSET), _hazards())


def test_a_run_of_a_pool_is_seen_through_a_join_of_pools() -> None:
    joined = OutcomeSample.join(
        [
            _pool(("M", RAIN), ("N", HEAT)),
            _pool(("M", WIND), ("N", _series("gust")), asset="b"),
        ]
    )
    message = _shares(_plain(), joined)
    assert message.startswith("the evaluation sample shares 12 model-years with")
    assert "run M r1 of asset 'site'" in message
    # Both the joined run and the part's are the same rows.
    message = _shares(joined, _pool(("M", RAIN), ("X", WIND)))
    assert message.startswith("the evaluation sample shares 12 model-years with")
    _check_out_of_sample(joined, _plain(WIND))


def test_a_block_of_a_join_of_pools_reads_the_runs_on_its_own_rows() -> None:
    joined = OutcomeSample.join(
        [
            _pool(("M", RAIN), ("N", HEAT)),
            _pool(("M", WIND), ("N", _series("gust")), asset="b"),
        ]
    )
    train, test = holdout(joined, 0.5)
    _, late = holdout(_plain(), 0.5)
    _check_out_of_sample(train, late)
    message = _shares(_plain(), test)
    assert message.startswith("the evaluation sample shares 6 model-years with")


# -- curves ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dependence", [Independent(), Comonotonic(), GaussianCopula(0.5)]
)
def test_one_components_draws_are_seen_through_several_hazards(dependence: Any) -> None:
    sample = _curve_hazards(dependence=dependence)
    message = _shares(_one_curve(), sample)
    assert message.startswith("the evaluation sample shares 40 periods with")
    assert "'rain' of asset 'site'" in message
    assert "'heat' of asset 'site'" in _shares(sample, _one_curve("heat", seed=22))


def test_a_view_of_several_hazards_on_curves_links_to_its_draws() -> None:
    view = _curve_hazards().component("rain")
    assert "'rain' of asset 'site'" in _shares(_one_curve(), view)
    _check_out_of_sample(_one_curve("heat", seed=22), view)


def test_fresh_draws_of_a_component_pass() -> None:
    _check_out_of_sample(_one_curve(seed=99), _curve_hazards())
    _check_out_of_sample(_curve_hazards(seeds=(77, 88)), _curve_hazards())
    _check_out_of_sample(_curve_hazards(), _one_curve(seed=99))


def test_samples_of_several_hazards_sharing_one_generator_raise() -> None:
    message = _shares(_curve_hazards(), _curve_hazards(seeds=(11, 88)))
    assert "'rain' of asset 'site'" in message
    assert "'heat'" not in message


def test_blocks_of_one_components_draws_pass_when_disjoint() -> None:
    plain = _one_curve()
    selected = plain.select(np.arange(0, 20))
    sample = _curve_hazards().select(np.arange(20, 40))
    _check_out_of_sample(selected, sample)
    assert "shares 20 periods" in _shares(plain, sample)


def test_one_assets_draws_are_seen_through_several_assets() -> None:
    sample = _curve_assets()
    single = _one_curve(asset="a")
    assert "asset 'a'" in _shares(single, sample)
    assert "asset 'a'" in _shares(sample, single)
    _check_out_of_sample(_one_curve(seed=99, asset="a"), sample)


# -- user tables ----------------------------------------------------------------


def test_a_table_of_one_asset_links_to_a_table_of_several_from_one_source() -> None:
    message = _shares(_table(YEARS, ["a"]), _table(YEARS, ["a", "b"]))
    assert message.startswith("the evaluation sample shares 12 periods with")
    assert FIRST_YEARS in message
    assert "asset 'a'" in message
    assert "asset 'b'" in _shares(_table(YEARS, ["a", "b"]), _table(YEARS, ["b"]))


def test_an_assets_view_of_a_table_links_to_its_own_table() -> None:
    view = _table(YEARS, ["a", "b"]).asset("a")
    assert "asset 'a'" in _shares(_table(YEARS, ["a"]), view)


def test_disjoint_years_of_one_table_source_pass() -> None:
    _check_out_of_sample(_table(YEARS[:6], ["a"]), _table(YEARS[6:], ["a", "b"]))
    _check_out_of_sample(_table(YEARS, ["a"]), _table(YEARS, ["b", "c"]))


def test_another_stated_source_is_another_table() -> None:
    _check_out_of_sample(
        _table(YEARS, ["a"]), _table(YEARS, ["a", "b"], source="another table")
    )


def test_tables_with_no_stated_source_are_told_apart_by_their_data() -> None:
    _check_out_of_sample(_table(YEARS, ["a"], None), _table(YEARS, ["a", "b"], None))


def test_a_table_of_several_hazards_is_its_own_table_of_one() -> None:
    several = _table(
        YEARS,
        ["a"],
        index={"x": "x", "y": "y"},
        index_unit={"x": "mm/day", "y": "degC"},
        tails={"x": "upper", "y": "upper"},
    )
    message = _shares(_table(YEARS, ["a"]), several)
    assert message.startswith("the evaluation sample shares 12 periods with")


def test_a_join_of_tables_holds_each_tables_source() -> None:
    joined = OutcomeSample.join(
        [_table(YEARS, ["a"]), _table(YEARS, ["b"], "another table")]
    )
    assert "asset 'a'" in _shares(_table(YEARS, ["a"]), joined)
    assert "asset 'b'" in _shares(joined, _table(YEARS, ["b"], "another table"))
    _check_out_of_sample(joined, _table(YEARS, ["b"]))


# -- the record -----------------------------------------------------------------


def test_each_source_has_the_lineage_it_has_alone() -> None:
    sources = _hazards().provenance.source_lineages
    assert sources == {"rain": _plain().lineage, "heat": _plain(HEAT).lineage}
    with_index = _plain(index=HEAT).provenance.source_lineages
    assert with_index == {"hazard": _plain().lineage, "index": _plain(HEAT).lineage}
    drawn = _curve_hazards().provenance.source_lineages
    assert drawn == {
        "rain": _one_curve().lineage,
        "heat": _one_curve("heat", seed=22).lineage,
    }


def test_a_views_record_names_what_it_reads() -> None:
    sample = _hazards()
    assert sample.component("rain").provenance.source_lineages == {
        "rain": _plain().lineage
    }
    assert (
        sample.component("rain", loss="combined").provenance.source_lineages
        == sample.provenance.source_lineages
    )


def test_a_sample_of_one_source_records_no_source_lineages() -> None:
    samples = [
        _plain(),
        _pool(("M", RAIN), ("N", HEAT)),
        _join(("a", RAIN), ("b", HEAT)),
        _one_curve(),
        _curve_assets(),
        _table(YEARS, ["a"]),
    ]
    for sample in samples:
        assert sample.provenance.source_lineages is None
        assert "source_lineages" not in sample.provenance.to_document()


def test_a_table_of_several_hazards_needs_no_source_lineages() -> None:
    several = _table(
        YEARS,
        ["a"],
        index={"x": "x", "y": "y"},
        index_unit={"x": "mm/day", "y": "degC"},
        tails={"x": "upper", "y": "upper"},
    )
    assert several.provenance.source_lineages is None
    assert several.component("x", loss="combined").provenance.source_lineages is None


@pytest.mark.parametrize(
    "value", [["rain"], {"rain": 1}, {1: "lineage"}], ids=["list", "value", "key"]
)
def test_source_lineages_must_map_names_to_lineages(value: Any) -> None:
    with pytest.raises(SampleError, match="source_lineages"):
        SampleProvenance(
            source_kind="table",
            sources=(),
            semantics_source="caller",
            source_lineages=value,
        )


def test_source_lineages_are_stored_read_only() -> None:
    sources = _hazards().provenance.source_lineages
    assert sources is not None
    with pytest.raises(TypeError):
        sources["rain"] = "other"  # type: ignore[index]


def test_a_written_document_of_a_sample_with_sources_is_plain_json() -> None:
    document = _hazards().provenance.to_document()
    assert document["source_lineages"] == {
        "rain": _plain().lineage,
        "heat": _plain(HEAT).lineage,
    }
    assert json.loads(json.dumps(document)) == document


# -- what must not change --------------------------------------------------------

_KINDS: dict[str, Any] = {
    "per-year": _plain,
    "per-year with an index": lambda: _plain(index=HEAT),
    "several hazards": _hazards,
    "pool": lambda: _pool(("M", RAIN), ("N", HEAT)),
    "join": lambda: _join((ASSET, RAIN), ("b", HEAT)),
    "curve": _one_curve,
    "several assets from curves": _curve_assets,
    "several hazards from curves": _curve_hazards,
    "table": lambda: _table(YEARS, ["a"]),
    "table of several assets": lambda: _table(YEARS, ["a", "b"]),
}

_PINNED: dict[str, tuple[str, str]] = {
    "per-year": (
        "sha256:8c8b7479ad968fce4ecbc5f67263696c63af4cff5b83377dfb7c35e60d122c37",
        "sha256:90e1bc8dabea1b7ac818dbff2726d460d3cf37567c73d2e993d4f1711da93dce",
    ),
    "per-year with an index": (
        "sha256:fd9775eed052cdb29b4c8945901e8e268741cc7dd9f7b69f2360268a8deb7fc6",
        "sha256:5058cd6e64e2a916c1cfae45ad94d488527113dd919a45e7db7f49cb4469a7df",
    ),
    "several hazards": (
        "sha256:5e4d488978f940909d04aaae79cc35fe8906fc21d6cf4a05a1fcb1526c3165a8",
        "sha256:342be33478ae474ed56b5064a88c713937e4c88016486a1fae7cb7e5aaefd4fc",
    ),
    "pool": (
        "sha256:a44b4f865f11632054718d5676d166afe4b19d662e76167535e3701407de05a7",
        "sha256:03e286e2c72be8a57e7600dea61373e0905a6c0dc52308777adb7c1de796d3c6",
    ),
    "join": (
        "sha256:ab4deb71f3623ded0af2927aa3b1d7b45755fce5a03d36df9fac638a3ff3db25",
        "sha256:b4423daee4c831e002aa74c829c992645f20b646b7d23be1ceca0821fb3c33c7",
    ),
    "curve": (
        "sha256:f2bf7b4a8ebceeeedb97c4251416bacea34959455a135fca65fefdafbaf3df8f",
        "sha256:5fd1f0e72105d5ce30159502d19b383715ba34acac757a2f413d26e9163893f0",
    ),
    "several assets from curves": (
        "sha256:7cf553401e23c66727fa961cbfc0fb9c28ab29db7325d03669e09874a355f8fd",
        "sha256:fe4d51f2b42760eb93e1228d15c892c33017c2ca26e631f9bbb69b1dbee9a92a",
    ),
    "several hazards from curves": (
        "sha256:e5eb2ac3de5615512f1972131a8f6e06211d9524ba5909c0cebefaf899772eee",
        "sha256:cd76cc4a26067d25ad96a65974dc66a4e735dbc186f5c46f8a5d5c812f041f7b",
    ),
    "table": (
        "sha256:fc04225333216c8e8a85cd6979111367f27da02946a8eef23de5e5772c16e993",
        "sha256:4e9769e56fcadc9e84b21672648c84b0effb67f6209b850c3971c4e41d923799",
    ),
    "table of several assets": (
        "sha256:c6bf8369026acb89f71ddfc35332b8991ffd5e323cbf0520f326a952619bf155",
        "sha256:a29b6b4f40375873cd6d2114a89d71b4022f5feba1c7bc93af059bd06481a15e",
    ),
}
"""What these samples hashed to before sources were compared one by one."""


@pytest.mark.parametrize("kind", list(_KINDS))
def test_hashes_and_lineages_stay_as_they_were(kind: str) -> None:
    sample = _KINDS[kind]()
    assert (sample.content_hash, sample.lineage) == _PINNED[kind]
