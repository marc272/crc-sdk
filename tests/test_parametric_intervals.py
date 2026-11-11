"""`compare_pooled` and `bootstrap_intervals`.

Every number is synthetic and chosen so that it can be checked by hand or against
a stated rule. The tests check what the module computes and records; nothing here
is a finding about a cover, a design or a split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.impacts import PiecewiseLinearImpact
from crc_sdk.parametric import (
    MSE,
    AnnualSeries,
    BinaryPayout,
    CostOfCapitalLoad,
    Cover,
    CoverPackage,
    EvaluationError,
    EvaluationMetadata,
    Free,
    ImpactNoise,
    IndexComponent,
    IndexDefinition,
    JointDesignResult,
    LinearPayout,
    LinearTrend,
    LossModel,
    MetricOptions,
    ModelRun,
    OutcomeSample,
    PurePremium,
    SampleProvenance,
    StepPayout,
    UnresolvedParameterError,
    bootstrap_intervals,
    compare,
    compare_pooled,
    design_joint,
    holdout,
    sample_annual,
    sample_annual_pool,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.intervals import (
    _Arrays,
    _block,
    _Evaluated,
    _pooled,
    _Read,
    _replicates,
    _year_counts,
)
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    IDENTITY_COLUMNS,
)
from crc_sdk.parametric.metrics import metric_columns, metric_values

UNTYPED_POOLED: Any = compare_pooled
UNTYPED_INTERVALS: Any = bootstrap_intervals

INDEX = IndexDefinition("rx1day", "mm/day", aggregation="annual_maximum")
RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
FEW = MetricOptions(betas=(0.9,), return_periods=(5.0, 50.0))
TWO_HAZARD_YEARS = 12


def _provenance(**fields: Any) -> SampleProvenance:
    base: dict[str, Any] = {
        "source_kind": "table",
        "sources": ({"role": "table", "name": "test table", "licence": "CC0-1.0"},),
        "semantics_source": "caller",
        "licences": ({"role": "table", "source": "test table", "licence": "CC0-1.0"},),
    }
    return SampleProvenance(**{**base, **fields})


def _sample(
    n: int = 23,
    *,
    asset: str = "a",
    shift: float = 0.0,
    provenance: SampleProvenance | None = None,
    **overrides: Any,
) -> OutcomeSample:
    # A wavy index, so that a block of neighbouring years holds high and low ones.
    years = np.arange(n, dtype=np.float64)
    index = 60.0 + shift + 40.0 * np.sin(years * 1.7) + years
    fields: dict[str, Any] = {
        "asset_ids": (asset,),
        "period": np.arange(2000, 2000 + n),
        "probability": np.full(n, 1.0 / n),
        "index": index,
        "loss": np.where(index > 60.0, (index - 60.0) * 10.0, 0.0),
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": provenance or _provenance(),
    }
    fields.update(overrides)
    built: OutcomeSample = OutcomeSample(**fields)
    return built


def _cover(
    limit: float = 800.0, *, label: str | None = "linear", asset: str = "a"
) -> Cover:
    return Cover(
        INDEX,
        LinearPayout(60.0, 140.0, limit),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


@dataclass(frozen=True)
class _Design:
    """A designed cover, as a design would return it."""

    cover: Cover
    design_sample: OutcomeSample
    premium: float = 120.0
    design_id: str = "sha256:" + "1" * 64
    unique: bool | None = True
    ties: int | None = 0
    degenerate_tail: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "route": "lp",
            "solver": {"name": "stub", "version": "0"},
            **self.record,
        }


def _id(fold: int, asset: str = "") -> str:
    return content_hash({"asset": asset, "stub design": fold})


def _designs(
    folds: Sequence[tuple[OutcomeSample, OutcomeSample]], **fields: Any
) -> list[_Design]:
    return [
        _Design(
            _cover(800.0 + 100.0 * fold),
            train,
            premium=100.0 + 10.0 * fold,
            design_id=_id(fold),
            **fields,
        )
        for fold, (train, _) in enumerate(folds)
    ]


def _folds(n: int = 23, k: int = 5) -> tuple[tuple[OutcomeSample, OutcomeSample], ...]:
    return year_block_kfold(_sample(n), k)


def _pooled_arrays(
    designs: Sequence[_Design],
    folds: Sequence[tuple[OutcomeSample, OutcomeSample]],
) -> tuple[Any, Any, Any, Any]:
    tests = [test for _, test in folds]
    n = sum(len(test) for test in tests)
    return (
        np.concatenate([test.loss[:, 0] for test in tests]),
        np.concatenate(
            [
                design.cover.payout_values(test.index[:, 0])
                for design, test in zip(designs, tests, strict=True)
            ]
        ),
        np.concatenate(
            [
                np.full(len(test), design.premium)
                for design, test in zip(designs, tests, strict=True)
            ]
        ),
        np.full(n, 1.0 / n),
    )


def _row(table: Any) -> dict[str, Any]:
    assert table.num_rows == 1
    row: dict[str, Any] = table.to_pylist()[0]
    return row


def _metadata(table: Any) -> EvaluationMetadata:
    return EvaluationMetadata.from_table(table)


def _interval(table: Any, column: str, row: int = 0) -> dict[str, Any]:
    rows = [item for item in table.to_pylist() if item["column"] == column]
    found: dict[str, Any] = rows[row]
    return found


def _book(n: int = 21) -> OutcomeSample:
    return OutcomeSample.join([_sample(n, asset="a"), _sample(n, asset="b", shift=9.0)])


def _hazards() -> OutcomeSample:
    n = TWO_HAZARD_YEARS
    rain = np.array(
        [10.0, 60.0, 20.0, 80.0, 30.0, 15.0, 70.0, 25.0, 55.0, 12.0, 65.0, 22.0]
    )
    heat = np.array(
        [30.0, 25.0, 40.0, 20.0, 35.0, 42.0, 45.0, 38.0, 20.0, 44.0, 41.0, 18.0]
    )
    rain_loss = np.array(
        [0.0, 10.0, 0.0, 6.0, 0.0, 0.0, 8.0, 0.0, 5.0, 0.0, 7.0, 0.0],
    )
    heat_loss = np.array(
        [0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 6.0, 6.0, 0.0, 3.0, 5.0, 0.0],
    )
    return OutcomeSample(
        asset_ids=("site",),
        period=np.arange(2000, 2000 + n),
        probability=np.full(n, 1.0 / n),
        index=rain,
        loss=np.minimum(rain_loss + heat_loss, 12.0),
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=(),
            semantics_source="caller",
            hazard_dependence={"kind": "data"},
            combine={"kind": "table"},
        ),
        index_components={
            "rain": IndexComponent(
                rain, unit="mm/day", tail="upper", hazard=None, own_loss=rain_loss
            ),
            "heat": IndexComponent(
                heat, unit="degC", tail="upper", hazard=None, own_loss=heat_loss
            ),
        },
    )


def _package(train: OutcomeSample, fold: int) -> CoverPackage:
    return CoverPackage(
        {
            "rain": _Design(
                Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR", label="r"),
                train.component("rain"),
                premium=2.0 + fold,
                design_id=_id(fold, "rain"),
            ),
            "heat": _Design(
                Cover(HEAT, BinaryPayout(38.0, 5.0), payout_unit="EUR", label="h"),
                train.component("heat"),
                premium=1.0 + fold,
                design_id=_id(fold, "heat"),
            ),
        },
        label="package",
    )


def _set_of_designs(train: OutcomeSample, fold: int) -> dict[str, _Design]:
    return {
        asset: _Design(
            _cover(700.0 + 50.0 * fold, asset=asset),
            train.asset(asset),
            premium=90.0 + fold,
            design_id=_id(fold, asset),
        )
        for asset in train.asset_ids
    }


def _step(asset: str) -> Cover:
    return Cover(INDEX, StepPayout([70.0], [Free()]), payout_unit="EUR", asset_id=asset)


def _joint(train: OutcomeSample) -> JointDesignResult:
    return design_joint([_step(asset) for asset in train.asset_ids], train, MSE())


# -- one number: a premium per draw is tested in test_parametric_metrics -------


def test_a_pooled_row_is_metric_values_on_the_concatenated_rows() -> None:
    folds = _folds()
    designs = _designs(folds)
    row = _row(UNTYPED_POOLED([designs], folds))
    loss, payout, premium, probability = _pooled_arrays(designs, folds)
    expected = metric_values(
        loss, payout, probability, premium=premium, options=MetricOptions()
    )
    for name, value in expected.items():
        assert row[name] == value, name


def test_a_closed_form_at_two_folds() -> None:
    # Five years, so fold 0 holds out three and fold 1 two. By hand:
    #   L = (0, 10, 20, 30, 40); payout = (0, 12, 12, 0, 50); P = (7, 7, 7, 20, 20);
    #   U = (0, 0, 8, 30, 0); net loss = L - payout + P = (7, 5, 15, 50, 10).
    index = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    sample = _sample(5, index=index, loss=np.array([0.0, 10.0, 20.0, 30.0, 40.0]))
    folds = year_block_kfold(sample, 2)
    designs = [
        _Design(
            Cover(INDEX, BinaryPayout(15.0, 12.0), payout_unit="EUR", asset_id="a"),
            folds[0][0],
            premium=7.0,
            design_id=_id(0),
        ),
        _Design(
            Cover(INDEX, BinaryPayout(45.0, 50.0), payout_unit="EUR", asset_id="a"),
            folds[1][0],
            premium=20.0,
            design_id=_id(1),
        ),
    ]
    options = MetricOptions(betas=(0.7,), return_periods=(2.0,))
    row = _row(UNTYPED_POOLED([designs], folds, options=options))
    assert row["expected_loss"] == pytest.approx(20.0)
    assert row["expected_payout"] == pytest.approx(14.8)
    assert row["expected_uncovered"] == pytest.approx(7.6)
    assert row["coverage_ratio"] == pytest.approx(0.62)
    assert row["payout_frequency"] == pytest.approx(0.6)
    assert row["false_negative_rate_l0"] == pytest.approx(0.25)
    assert row["false_positive_rate_l0"] == pytest.approx(0.0)
    assert row["rmse"] == pytest.approx(np.sqrt(1068.0 / 5.0))
    # Weighted by the rows: (3 * 7 + 2 * 20) / 5, not the mean of 7 and 20.
    assert row["premium"] == pytest.approx(12.2)
    # On the net loss, 15 is the 0.7 quantile, and the tail above it is
    # (0.1 * 15 + 0.2 * 50) / 0.3. The folds' own CVaRs are 15 and 50, whose mean,
    # 32.5, is a different number.
    assert row["var_net_loss_b0_7"] == pytest.approx(15.0)
    assert row["cvar_net_loss_b0_7"] == pytest.approx(11.5 / 0.3)
    assert row["cvar_net_loss_b0_7"] != pytest.approx(32.5)
    assert row["net_loss_rp2"] == pytest.approx(10.0)


def test_the_pooled_row_identifies_its_folds() -> None:
    folds = _folds()
    designs = _designs(folds)
    row = _row(UNTYPED_POOLED([designs], folds))
    assert (row["split"], row["fold"]) == ("out_of_sample", None)
    assert row["cover_hash"] == content_hash(
        [design.cover.definition_hash for design in designs]
    )
    assert row["design_id"] == content_hash([design.design_id for design in designs])
    assert row["sample_hash"] == content_hash([test.content_hash for _, test in folds])
    assert row["cover_label"] == "linear"
    assert row["premium_basis"] == "design"


def test_the_label_falls_back_on_the_folds_hash() -> None:
    folds = _folds()
    designs = [
        _Design(_cover(700.0 + i, label=f"fold {i}"), train, design_id=_id(i))
        for i, (train, _) in enumerate(folds)
    ]
    row = _row(UNTYPED_POOLED([designs], folds))
    assert row["cover_label"] == row["cover_hash"].removeprefix("sha256:")[:12]
    unlabelled = [
        _Design(_cover(700.0 + i, label=None), train, design_id=_id(i))
        for i, (train, _) in enumerate(folds)
    ]
    row = _row(UNTYPED_POOLED([unlabelled], folds))
    assert row["cover_label"] == row["cover_hash"].removeprefix("sha256:")[:12]


@pytest.mark.parametrize(
    ("flags", "unique"),
    [
        ((True, True), True),
        ((True, False), False),
        ((None, False), False),
        ((True, None), None),
        ((None, None), None),
    ],
)
def test_unique_aggregates_over_the_folds(flags: Any, unique: Any) -> None:
    folds = _folds(k=2)
    designs = [
        _Design(_cover(), train, design_id=_id(i), unique=flags[i])
        for i, (train, _) in enumerate(folds)
    ]
    assert _row(UNTYPED_POOLED([designs], folds))["unique"] is unique


@pytest.mark.parametrize(
    ("ties", "flags", "total", "flag"),
    [
        ((1, 2), ("single_outcome", "single_outcome"), 3, "single_outcome"),
        ((0, 0), (None, None), 0, None),
        ((1, None), ("single_outcome", None), None, "mixed"),
    ],
)
def test_ties_and_degenerate_tails_aggregate(
    ties: Any, flags: Any, total: Any, flag: Any
) -> None:
    folds = _folds(k=2)
    designs = [
        _Design(
            _cover(), train, design_id=_id(i), ties=ties[i], degenerate_tail=flags[i]
        )
        for i, (train, _) in enumerate(folds)
    ]
    row = _row(UNTYPED_POOLED([designs], folds))
    assert (row["ties"], row["degenerate_tail"]) == (total, flag)


def test_the_design_budget_is_the_mean_over_the_rows() -> None:
    folds = _folds(5, 2)
    designs = [
        _Design(
            _cover(),
            train,
            design_id=_id(i),
            record={"budget_bound": bound},
        )
        for i, ((train, _), bound) in enumerate(zip(folds, (10.0, 20.0), strict=True))
    ]
    row = _row(UNTYPED_POOLED([designs], folds))
    assert row["design_budget"] == pytest.approx((3 * 10.0 + 2 * 20.0) / 5)
    no_bound = [*designs[:1], _Design(_cover(), folds[1][0], design_id=_id(1))]
    assert _row(UNTYPED_POOLED([no_bound], folds))["design_budget"] is None


def test_unidentified_share_reads_each_row_by_its_own_folds_design() -> None:
    folds = year_block_kfold(_sample(6, index=np.arange(10.0, 70.0, 10.0)), 2)
    step = Cover(
        INDEX, StepPayout([15.0, 25.0], [5.0, 9.0]), payout_unit="EUR", asset_id="a"
    )
    # Fold 0's design filled its second amount, and fold 1's filled none.
    filled = step.payout.amount_names()[1]  # type: ignore[attr-defined]
    designs = [
        _Design(step, folds[0][0], design_id=_id(0), record={"unidentified": [filled]}),
        _Design(step, folds[1][0], design_id=_id(1), record={"unidentified": []}),
    ]
    row = _row(UNTYPED_POOLED([designs], folds))
    # Fold 0 holds out 10, 20, 30: only 30 reads the second step, 1 row of 6.
    assert row["unidentified_share"] == pytest.approx(1.0 / 6.0)
    unknown = [designs[0], _Design(step, folds[1][0], design_id=_id(1))]
    assert _row(UNTYPED_POOLED([unknown], folds))["unidentified_share"] is None


def test_a_cover_given_as_is_is_priced_once_on_the_pooled_rows() -> None:
    folds = _folds()
    cover = _cover()
    pricing = CostOfCapitalLoad(0.2, 0.9)
    row = _row(UNTYPED_POOLED([cover], folds, pricing=pricing))
    tests = [test for _, test in folds]
    index = np.concatenate([test.index[:, 0] for test in tests])
    payout = cover.payout_values(index)
    n = index.shape[0]
    assert row["premium"] == pricing.premium(payout, np.full(n, 1.0 / n))
    assert row["premium_basis"] == "evaluation"
    assert row["cover_hash"] == cover.definition_hash
    assert row["cover_label"] == "linear"
    assert row["design_id"] is None
    assert row["unique"] is None and row["ties"] is None
    assert row["unidentified_share"] is None
    evaluation = _metadata(UNTYPED_POOLED([cover], folds)).setup.evaluations[0]
    assert evaluation["design_samples"] == []
    assert evaluation["leakage_check"] == "not_applicable"


def test_a_set_of_covers_given_as_is_sums_its_contracts() -> None:
    folds = year_block_kfold(_book(), 3)
    covers = {asset: _cover(asset=asset) for asset in ("a", "b")}
    table = UNTYPED_POOLED([covers], folds, pricing=CostOfCapitalLoad(0.2, 0.9))
    rows = table.to_pylist()
    assert [row["level"] for row in rows] == ["asset", "asset", "total"]
    assert rows[2]["premium"] == pytest.approx(rows[0]["premium"] + rows[1]["premium"])
    assert rows[2]["premium_basis"] == "evaluation"


def test_a_set_of_designs_keeps_each_folds_premiums() -> None:
    folds = year_block_kfold(_book(), 3)
    per_fold = [_set_of_designs(train, i) for i, (train, _) in enumerate(folds)]
    rows = UNTYPED_POOLED([per_fold], folds).to_pylist()
    assert [row["level"] for row in rows] == ["asset", "asset", "total"]
    sizes = [len(test) for _, test in folds]
    expected = sum(size * (90.0 + i) for i, size in enumerate(sizes)) / sum(sizes)
    assert rows[0]["premium"] == pytest.approx(expected)
    assert rows[2]["premium"] == pytest.approx(2.0 * expected)
    assert rows[2]["design_id"] is None
    assert rows[0]["design_id"] == content_hash(
        [each["a"].design_id for each in per_fold]
    )


def test_a_joint_design_with_an_allocation() -> None:
    folds = year_block_kfold(_book(), 3)
    joint = [_joint(train) for train, _ in folds]
    table = UNTYPED_POOLED(
        [joint], folds, allocation="value", values={"a": 3.0, "b": 1.0}
    )
    rows = table.to_pylist()
    assert [row["level"] for row in rows] == [
        "total",
        "asset_by_value",
        "asset_by_value",
    ]
    assert rows[0]["scope"] == "joint"
    assert rows[1]["premium"] is None and rows[1]["premium_basis"] is None
    assert rows[1]["cvar_net_loss_b0_9"] is None
    assert rows[1]["expected_payout"] == pytest.approx(
        rows[0]["expected_payout"] * 0.75
    )


def test_a_package_of_designs() -> None:
    sample = _hazards()
    folds = year_block_kfold(sample, 3)
    packages = [_package(train, i) for i, (train, _) in enumerate(folds)]
    rows = UNTYPED_POOLED([packages], folds).to_pylist()
    assert [row["level"] for row in rows] == ["asset", "asset", "package"]
    assert rows[2]["premium"] == pytest.approx(rows[0]["premium"] + rows[1]["premium"])
    assert rows[2]["premium_basis"] == "design"
    assert rows[2]["components"] == "rain+heat"
    assert rows[2]["cover_label"] == "package"


# -- what compare_pooled refuses -----------------------------------------------


def _mixed_lineage() -> Any:
    one = year_block_kfold(_sample(10), 2)
    other = year_block_kfold(_sample(10, shift=1.0), 2)
    return (one[0], other[1])


def _repeats_years() -> Any:
    sample = _sample(10)
    small = sample.select(np.arange(2000, 2006))
    return (year_block_kfold(sample, 2)[0], year_block_kfold(small, 2)[1])


def _misses_years() -> Any:
    sample = _sample(10)
    small = sample.select(np.arange(2000, 2006))
    return (year_block_kfold(small, 2)[0], year_block_kfold(sample, 2)[1])


def _unequal_probabilities() -> Any:
    n = 10
    p = np.arange(1.0, n + 1.0)
    return year_block_kfold(_sample(n, probability=p / p.sum()), 2)


def _scenarios() -> Any:
    lineage = _provenance(lineage="one source")
    first = year_block_kfold(_sample(10, provenance=lineage, scenario="low"), 2)
    second = year_block_kfold(_sample(10, provenance=lineage, scenario="high"), 2)
    return (first[0], second[1])


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: (holdout(_sample()),), "nothing to pool"),
        (lambda: (holdout(_sample()), holdout(_sample())), "2-fold split"),
        (lambda: tuple(reversed(_folds(k=2))), "position 0 of folds"),
        (lambda: _folds(k=3)[:2], "2-fold split"),
        (lambda: (_folds(k=2)[0], (_folds(k=2)[1][1], _folds(k=2)[1][0])), "train"),
        (_mixed_lineage, "lineages"),
        (_repeats_years, "repeat years"),
        (_misses_years, "miss years"),
        (_unequal_probabilities, "unequal probabilities"),
        (_scenarios, "scenario"),
    ],
)
def test_folds_are_refused(build: Any, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        UNTYPED_POOLED([_cover()], build())


def test_a_design_on_another_block_is_refused() -> None:
    folds = _folds()
    designs = _designs(folds)
    designs[1] = _Design(_cover(), folds[2][0], design_id=_id(1))
    with pytest.raises(EvaluationError, match="fold 1 was made on"):
        UNTYPED_POOLED([designs], folds)


@pytest.mark.parametrize(
    ("entry", "error", "message"),
    [
        (lambda f: _designs(f)[:2], EvaluationError, "one item per fold"),
        (lambda f: [_cover()] * len(f), EvaluationError, "holds no design"),
        (lambda f: _designs(f)[0], TypeError, "each entry is a sequence"),
        (
            lambda f: {"a": _designs(f)[0]},
            EvaluationError,
            "a mapping on its own",
        ),
    ],
)
def test_entries_are_refused(entry: Any, error: Any, message: str) -> None:
    folds = _folds()
    with pytest.raises(error, match=message):
        UNTYPED_POOLED([entry(folds)], folds)


def test_the_folds_must_be_given_as_pairs() -> None:
    with pytest.raises(TypeError, match="pair per fold"):
        UNTYPED_POOLED([_cover()], _sample())
    with pytest.raises(TypeError, match="pair of"):
        UNTYPED_POOLED([_cover()], [_sample(), _sample()])
    with pytest.raises(TypeError, match="OutcomeSample"):
        UNTYPED_POOLED([_cover()], [(1, 2), (3, 4)])


def test_entries_must_be_a_list_of_entries() -> None:
    folds = _folds()
    with pytest.raises(TypeError, match="one item per entry"):
        UNTYPED_POOLED(_cover(), folds)
    with pytest.raises(EvaluationError, match="at least one"):
        UNTYPED_POOLED([], folds)


def test_a_template_is_refused() -> None:
    template = Cover(
        INDEX, LinearPayout(60.0, 140.0, Free()), payout_unit="EUR", asset_id="a"
    )
    with pytest.raises(UnresolvedParameterError):
        UNTYPED_POOLED([template], _folds())


def test_items_with_different_rows_are_refused() -> None:
    folds = year_block_kfold(_book(), 2)
    per_fold: list[Any] = [
        _set_of_designs(folds[0][0], 0),
        _joint(folds[1][0]),
    ]
    with pytest.raises(EvaluationError, match="same rows on every fold"):
        UNTYPED_POOLED([per_fold], folds)


def test_the_same_entry_twice_is_refused() -> None:
    folds = _folds()
    designs = _designs(folds)
    with pytest.raises(EvaluationError, match="repeat a row key"):
        UNTYPED_POOLED([designs, designs], folds)


# -- the pooled metadata -------------------------------------------------------


def test_the_metadata_of_a_pooled_table() -> None:
    folds = _folds()
    designs = _designs(folds)
    table = UNTYPED_POOLED([designs], folds)
    metadata = _metadata(table)
    metadata.check_table(table)
    assert len(metadata.setup.evaluations) == 1
    evaluation = metadata.setup.evaluations[0]
    assert evaluation["fold"] is None
    assert evaluation["method"] == "kfold_pooled"
    assert evaluation["details"]["k"] == 5
    assert evaluation["details"]["blocks"] == [test.content_hash for _, test in folds]
    assert evaluation["design_samples"] == [train.content_hash for train, _ in folds]
    assert evaluation["leakage_check"] == "passed"
    by_hash = {entry["sample_hash"]: entry for entry in metadata.setup.samples}
    assert len(by_hash) == 11
    pooled = by_hash[evaluation["sample_hash"]]
    assert pooled["roles"] == ["evaluation"]
    assert pooled["draws"] == 23
    assert pooled["periods"] == [2000, 2022]
    assert pooled["blocks"] == evaluation["details"]["blocks"]
    row = _row(table)
    records = {record["design_id"]: record for record in metadata.designs}
    assert records[row["design_id"]]["designs"] == [d.design_id for d in designs]
    assert records[row["design_id"]]["kind"] == "pooled"
    assert all(d.design_id in records for d in designs)
    covers = {entry.hash: entry for entry in metadata.covers}
    assert covers[row["cover_hash"]].document == {
        "kind": "pooled",
        "covers": [d.cover.definition_hash for d in designs],
    }


def test_a_pooled_table_is_written_and_read_as_a_comparison(tmp_path: Path) -> None:
    folds = _folds()
    table = UNTYPED_POOLED([_designs(folds)], folds)
    path = tmp_path / "pooled.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.schema.field("fold").nullable
    assert read.column("fold").to_pylist() == [None]
    assert _metadata(read) == _metadata(table)
    long = to_long(table)
    assert long.column("fold").null_count == long.num_rows


def test_compare_keeps_fold_non_null() -> None:
    sample = _sample()
    table = compare([_cover()], sample, split="in_sample")
    assert not table.schema.field("fold").nullable


def test_two_pooled_tables_can_be_merged() -> None:
    folds = _folds()
    first = UNTYPED_POOLED([_designs(folds)], folds)
    second = UNTYPED_POOLED([_cover()], folds)
    merged = EvaluationMetadata.merge(_metadata(first), _metadata(second))
    merged.check_table(pa.concat_tables([first, second]))
    assert len(merged.setup.evaluations) == 2


def test_the_setup_text_of_a_pooled_table() -> None:
    folds = _folds()
    text = setup_text(UNTYPED_POOLED([_designs(folds)], folds))
    assert (
        "Split: out of sample, the 5 folds of a 5-fold split over blocks of years, "
        "pooled: evaluated on 23 years (periods 2000 to 2022); each year is "
        "evaluated by the design made without it and pays that fold's design "
        "premium; each metric is computed once over all those rows, with the same "
        "probability on each, and is not a mean over the folds; the leakage check "
        "passed on every fold."
    ) in text
    assert sum(line.startswith("Design ") for line in text.splitlines()) == 5
    assert "On a pooled row each year pays its own fold's design premium" in text
    assert "Sample of" not in text


def test_the_setup_text_says_a_given_cover_is_priced_on_the_pooled_rows() -> None:
    text = setup_text(UNTYPED_POOLED([_cover()], _folds()))
    assert "a cover given as is applies on every fold" in text
    assert "no design, so no leakage check" in text
    assert "On a pooled row each year pays" not in text


def test_the_setup_text_of_a_package_names_each_fold() -> None:
    folds = year_block_kfold(_hazards(), 3)
    packages = [_package(train, i) for i, (train, _) in enumerate(folds)]
    text = setup_text(UNTYPED_POOLED([packages], folds))
    assert "Package package, fold 0:" in text
    assert "Package package, fold 2:" in text
    assert "Row r, fold 1 (asset)" in text


def _draws(n: int = 40) -> OutcomeSample:
    curve = {"curve": {"name": "a curve"}}
    return _sample(
        n,
        provenance=SampleProvenance(
            source_kind="monte_carlo",
            sources=({"role": "hazard", **curve},),
            semantics_source="caller",
        ),
    )


def test_the_setup_text_says_what_folds_of_draws_test() -> None:
    folds = year_block_kfold(_draws(), 4)
    sentence = (
        "Out of sample here means new draws from the same curve, so these rows test "
        "the design's sampling error, not the curve."
    )
    assert sentence in setup_text(UNTYPED_POOLED([_designs(folds)], folds))
    assert sentence not in setup_text(UNTYPED_POOLED([_cover()], folds))
    assert sentence not in setup_text(UNTYPED_POOLED([_designs(_folds())], _folds()))


def test_the_rows_keep_their_fold_designs_and_blocks_in_the_setup_text() -> None:
    folds = _folds()
    table = UNTYPED_POOLED([_designs(folds)], folds)
    assert setup_text(table) == setup_text(table.slice(0, 1))
    assert sum(line.startswith("Design ") for line in setup_text(table).splitlines())


# -- replicates ----------------------------------------------------------------


def _evaluated(kind: str) -> _Evaluated:
    options = FEW
    arguments: dict[str, Any] = {"pricing": PurePremium(), "options": options}
    one: dict[str, Any] = {
        **arguments,
        "fold": 0,
        "premiums": None,
        "design_sample": None,
        "allocation": None,
        "values": None,
    }
    if kind == "cover on a block":
        return _block([_cover()], _sample(), split="in_sample", **one)
    if kind == "loaded cover on a block":
        loaded = {**one, "pricing": CostOfCapitalLoad(0.2, 0.9)}
        return _block([_cover()], _sample(), split="in_sample", **loaded)
    if kind == "design on a block":
        train, test = holdout(_sample())
        return _block([_Design(_cover(), train)], test, split="out_of_sample", **one)
    if kind == "set of covers on a block":
        covers = {asset: _cover(asset=asset) for asset in ("a", "b")}
        return _block([covers], _book(), split="in_sample", **one)
    if kind == "joint design on a block":
        sample = _book()
        return _block(
            [_joint(sample)],
            sample,
            split="in_sample",
            **{**one, "allocation": "loss", "values": {"a": 2.0, "b": 1.0}},
        )
    if kind == "package on a block":
        sample = _hazards()
        given = CoverPackage(
            {
                "rain": Cover(RAIN, BinaryPayout(50.0, 8.0), payout_unit="EUR"),
                "heat": Cover(HEAT, BinaryPayout(38.0, 5.0), payout_unit="EUR"),
            }
        )
        return _block([given], sample, split="in_sample", **one)
    pooled: dict[str, Any] = {**arguments, "allocation": None, "values": None}
    if kind == "designs on folds":
        folds = _folds()
        return _pooled([_designs(folds)], folds, **pooled)
    if kind == "cover on folds":
        return _pooled([_cover()], _folds(), **pooled)
    if kind == "set of designs on folds":
        folds = year_block_kfold(_book(), 3)
        per_fold = [_set_of_designs(train, i) for i, (train, _) in enumerate(folds)]
        return _pooled([per_fold], folds, **pooled)
    assert kind == "packages on folds"
    folds = year_block_kfold(_hazards(), 3)
    packages = [_package(train, i) for i, (train, _) in enumerate(folds)]
    return _pooled([packages], folds, **pooled)


KINDS = [
    "cover on a block",
    "loaded cover on a block",
    "design on a block",
    "set of covers on a block",
    "joint design on a block",
    "package on a block",
    "designs on folds",
    "cover on folds",
    "set of designs on folds",
    "packages on folds",
]


@pytest.mark.parametrize("kind", KINDS)
def test_all_ones_counts_give_the_point_estimate(kind: str) -> None:
    evaluated = _evaluated(kind)
    counts = np.ones((1, evaluated.n_years), dtype=np.int64)
    replicate = _replicates(evaluated, counts)[0]
    names = [column.name for column in metric_columns(evaluated.options)]
    rows = evaluated.table.to_pylist()
    for r, row in enumerate(rows):
        for c, name in enumerate(names):
            if row[name] is None:
                assert np.isnan(replicate[r, c]), name
            else:
                assert replicate[r, c] == pytest.approx(row[name], rel=1e-12), name


def test_a_replicate_gives_the_metrics_of_the_concatenated_years() -> None:
    folds = _folds()
    designs = _designs(folds)
    evaluated = _pooled(
        [designs],
        folds,
        pricing=PurePremium(),
        options=FEW,
        allocation=None,
        values=None,
    )
    drawn = np.array([0, 0, 0, 3, 5, 5, 7, 11, 11, 11, 11, 12, 13, 14, 14, 16, 17, 18,
                      19, 20, 21, 22, 22])  # fmt: skip
    counts = np.bincount(drawn, minlength=23).reshape(1, 23)
    replicate = _replicates(evaluated, counts)[0, 0]
    loss, payout, premium, _ = _pooled_arrays(designs, folds)
    expected = metric_values(
        loss[drawn],
        payout[drawn],
        np.full(23, 1.0 / 23.0),
        premium=premium[drawn],
        options=FEW,
    )
    names = [column.name for column in metric_columns(FEW)]
    for c, name in enumerate(names):
        value = expected[name]
        if value is None:
            assert np.isnan(replicate[c]), name
        else:
            assert replicate[c] == pytest.approx(value, rel=1e-12), name


def test_replicates_against_literal_values_of_the_reference_bootstrap() -> None:
    # The expected values are the reference implementation's `kpi_replicates`
    # on these arrays, with an annual measure of 8 rows at 1/8 and these year
    # draws, beta = 0.75, run once outside this repository and copied here.
    loss = np.array([0.0, 10.0, 25.0, 0.0, 40.0, 5.0, 60.0, 15.0])
    payout = np.array([0.0, 0.0, 30.0, 5.0, 20.0, 0.0, 70.0, 10.0])
    draws = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7], [6, 6, 2, 2, 4, 1, 7, 0], [3, 4, 4, 0, 5, 7, 7, 1]]
    )
    options = MetricOptions(betas=(0.75,))
    evaluated = _Evaluated(
        table=None,
        reads=(_Read(_Arrays(loss, payout, None, (payout,)), 0.0, None),),
        together=(frozenset(),),
        probability=np.full(8, 1.0 / 8.0),
        year=np.arange(8),
        n_years=8,
        unit="year",
        options=options,
    )
    names = [column.name for column in metric_columns(options)]
    replicates = _replicates(evaluated, _year_counts(draws, 8))[:, 0]
    expected = {
        "expected_loss": [19.375, 29.375, 15.625],
        "expected_payout": [16.875, 28.75, 8.125],
        "expected_uncovered": [5.0, 4.375, 8.125],
        "var_uncovered_b0_75": [5.0, 5.0, 10.0],
        "cvar_uncovered_b0_75": [15.0, 15.0, 20.0],
        "rmse": [9.3541434669, 9.8425098425, 11.1803398875],
        "payout_to_loss_ratio": [0.8709677419, 0.9787234043, 0.52],
        "correlation": [0.9167159985, 0.9378902269, 0.9295089931],
    }
    for name, values in expected.items():
        assert replicates[:, names.index(name)].tolist() == pytest.approx(
            values, abs=1e-9
        ), name


def test_a_cover_given_as_is_is_priced_again_on_each_replicate() -> None:
    pricing = CostOfCapitalLoad(0.2, 0.9)
    sample = _sample()
    evaluated = _block(
        [_cover()],
        sample,
        split="in_sample",
        fold=0,
        pricing=pricing,
        premiums=None,
        options=FEW,
        design_sample=None,
        allocation=None,
        values=None,
    )
    drawn = np.array([0, 2, 2, 3, 3, 3, 8, 9, 9, 10, 11, 12, 12, 14, 15, 15, 15, 18, 19,
                      20, 21, 21, 22])  # fmt: skip
    counts = np.bincount(drawn, minlength=23).reshape(1, 23)
    names = [column.name for column in metric_columns(FEW)]
    premium = _replicates(evaluated, counts)[0, 0, names.index("premium")]
    payout = _cover().payout_values(sample.index[:, 0])[drawn]
    assert premium == pytest.approx(
        pricing.premium(payout, np.full(23, 1.0 / 23.0)), rel=1e-12
    )


# -- the table of intervals ----------------------------------------------------


def test_one_generator_state_gives_one_table() -> None:
    folds = _folds()
    designs = _designs(folds)

    def build(seed: int) -> Any:
        return UNTYPED_INTERVALS(
            [designs], folds, rng=np.random.default_rng(seed), n_bootstrap=60
        )

    first, second = build(7), build(7)
    assert first.equals(second)
    assert _metadata(first).setup == _metadata(second).setup
    assert _metadata(first).rng == _metadata(second).rng
    assert not first.equals(build(8))


def test_no_global_random_state_is_read_or_written() -> None:
    np.random.seed(1)
    random.seed(1)
    numpy_state = np.random.get_state()[1].copy()
    python_state = random.getstate()
    table = UNTYPED_INTERVALS(
        [_cover()],
        _sample(),
        split="in_sample",
        rng=np.random.default_rng(3),
        n_bootstrap=10,
    )
    assert (np.random.get_state()[1] == numpy_state).all()
    assert random.getstate() == python_state
    np.random.seed(99)
    again = UNTYPED_INTERVALS(
        [_cover()],
        _sample(),
        split="in_sample",
        rng=np.random.default_rng(3),
        n_bootstrap=10,
    )
    assert table.equals(again)


def test_the_table_has_to_longs_columns_and_the_bounds() -> None:
    sample = _sample()
    wide = compare([_cover()], sample, split="in_sample", options=FEW)
    table = UNTYPED_INTERVALS(
        [_cover()],
        sample,
        split="in_sample",
        rng=np.random.default_rng(2),
        n_bootstrap=40,
        options=FEW,
    )
    long = to_long(wide)
    assert table.column_names == [
        *long.column_names,
        "ci_low",
        "ci_high",
        "ci_level",
        "n_bootstrap",
        "n_undefined",
        "short_block",
        "built_together",
    ]
    assert tuple(table.column_names[: len(IDENTITY_COLUMNS)]) == IDENTITY_COLUMNS
    assert table.select(long.column_names).to_pylist() == long.to_pylist()
    schema = table.schema
    assert schema.field("ci_low").type == pa.float64()
    assert schema.field("ci_low").nullable
    assert schema.field("n_bootstrap").type == pa.int32()
    assert schema.field("n_undefined").type == pa.int32()
    assert schema.field("short_block").type == pa.bool_()
    assert schema.field("built_together").type == pa.bool_()
    assert table.column("built_together").to_pylist() == [False] * table.num_rows
    assert set(table.column("ci_level").to_pylist()) == {0.9}
    assert set(table.column("n_bootstrap").to_pylist()) == {40}
    assert EVALUATION_METADATA_KEY.encode() in schema.metadata


def test_the_bounds_are_the_percentiles_of_the_replicates() -> None:
    folds = _folds()
    designs = _designs(folds)
    table = UNTYPED_INTERVALS(
        [designs],
        folds,
        rng=np.random.default_rng(11),
        n_bootstrap=80,
        ci_level=0.8,
        options=FEW,
    )
    evaluated = _pooled(
        [designs],
        folds,
        pricing=PurePremium(),
        options=FEW,
        allocation=None,
        values=None,
    )
    draws = np.random.default_rng(11).integers(0, 23, size=(80, 23))
    replicates = _replicates(evaluated, _year_counts(draws, 23))
    names = [column.name for column in metric_columns(FEW)]
    for name in ("expected_payout", "cvar_net_loss_b0_9", "premium"):
        row = _interval(table, name)
        values = replicates[:, 0, names.index(name)]
        low, high = np.quantile(values, [0.1, 0.9])
        assert (row["ci_low"], row["ci_high"]) == (low, high)
        assert row["n_undefined"] == 0


def test_a_design_premium_on_one_block_has_itself_for_an_interval() -> None:
    train, test = holdout(_sample())
    table = UNTYPED_INTERVALS(
        [_Design(_cover(), train, premium=130.0)],
        test,
        split="out_of_sample",
        rng=np.random.default_rng(5),
        n_bootstrap=50,
        options=FEW,
    )
    row = _interval(table, "premium")
    assert row["value"] == row["ci_low"] == row["ci_high"] == 130.0


def test_a_replicate_with_no_loss_leaves_a_ratio_undefined() -> None:
    # One loss year in six: many replicates draw none, and there E[L] is zero.
    sample = _sample(
        6,
        index=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 100.0]),
        loss=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
    )
    table = UNTYPED_INTERVALS(
        [_cover()],
        sample,
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=200,
    )
    row = _interval(table, "coverage_ratio")
    assert row["value"] is not None
    assert row["ci_low"] is None and row["ci_high"] is None
    assert 0 < row["n_undefined"] < 200
    assert _interval(table, "expected_loss")["n_undefined"] == 0


def test_a_null_in_the_wide_table_has_null_bounds() -> None:
    sample = _sample()
    table = UNTYPED_INTERVALS(
        [_cover()],
        sample,
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=30,
    )
    row = _interval(table, "unidentified_share")
    assert row["value"] is None and row["ci_low"] is None
    assert row["n_undefined"] == 30


def test_an_allocated_row_has_no_premium_interval() -> None:
    sample = _book()
    table = UNTYPED_INTERVALS(
        [_joint(sample)],
        sample,
        split="in_sample",
        allocation="value",
        values={"a": 1.0, "b": 1.0},
        rng=np.random.default_rng(1),
        n_bootstrap=20,
        options=FEW,
    )
    allocated = _interval(table, "premium", 1)
    assert allocated["level"] == "asset_by_value"
    assert allocated["value"] is None and allocated["ci_low"] is None
    assert allocated["n_undefined"] == 20
    assert _interval(table, "expected_payout", 1)["ci_low"] is not None


@pytest.mark.parametrize(
    ("n", "column", "flag"),
    [
        (20, "cvar_uncovered_b0_95", True),
        (19, "cvar_uncovered_b0_95", True),
        (21, "cvar_uncovered_b0_95", False),
        (20, "var_loss_b0_9", False),
        (10, "var_loss_b0_9", True),
        (20, "loss_rp20", False),
        (20, "loss_rp25", True),
        (26, "loss_rp25", False),
        (20, "expected_loss", None),
        (20, "false_negative_rate_l0", None),
    ],
)
def test_short_block(n: int, column: str, flag: bool | None) -> None:
    options = MetricOptions(betas=(0.9, 0.95), return_periods=(20.0, 25.0))
    table = UNTYPED_INTERVALS(
        [_cover()],
        _sample(n),
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=2,
        options=options,
    )
    assert _interval(table, column)["short_block"] is flag


def test_short_block_counts_the_pooled_rows() -> None:
    folds = _folds(20, 4)
    options = MetricOptions(betas=(0.95,), return_periods=(25.0,))
    table = UNTYPED_INTERVALS(
        [_cover()], folds, rng=np.random.default_rng(1), n_bootstrap=2, options=options
    )
    assert _interval(table, "cvar_loss_b0_95")["short_block"] is True
    assert _interval(table, "loss_rp25")["short_block"] is True
    metadata = _metadata(table)
    assert metadata.setup.intervals is not None
    assert "loss_rp25" in metadata.setup.intervals["short_block_columns"]
    assert "expected_loss" not in metadata.setup.intervals["short_block_columns"]


def _pool() -> OutcomeSample:
    years = np.arange(2000, 2012)
    runs = [
        ModelRun(
            AnnualSeries(
                years,
                50.0 + shift + 30.0 * np.sin(years * 1.3),
                "mm/day",
                {"source": {"name": f"synthetic {model}"}},
            ),
            model=model,
            member="r1i1p1f1",
            pathway="ssp245",
        )
        for model, shift in (("M1", 0.0), ("M2", 4.0), ("M3", 9.0))
    ]
    return sample_annual_pool(
        runs,
        loss_model=LossModel(
            PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
            value=1000.0,
            value_unit="EUR",
        ),
        basis_risk=ImpactNoise(0.2),
        asset_id="a",
        rng=np.random.default_rng(4),
    )


def test_a_pool_is_resampled_by_calendar_year() -> None:
    sample = _pool()
    assert len(sample) == 36
    evaluated = _block(
        [_cover()],
        sample,
        split="in_sample",
        fold=0,
        pricing=PurePremium(),
        premiums=None,
        options=FEW,
        design_sample=None,
        allocation=None,
        values=None,
    )
    assert evaluated.n_years == 12
    assert evaluated.unit == "year"
    # Only the first and the fourth year: their rows of every run, 3 rows each.
    counts = np.zeros((1, 12), dtype=np.int64)
    counts[0, [0, 3]] = [8, 4]
    names = [column.name for column in metric_columns(FEW)]
    value = _replicates(evaluated, counts)[0, 0, names.index("expected_loss")]
    years = np.asarray(sample.provenance.pool["year"])  # type: ignore[index]
    first, fourth = years == 2000, years == 2003
    expected = (
        8.0 * sample.loss[first, 0].sum() + 4.0 * sample.loss[fourth, 0].sum()
    ) / (12.0 * 3.0)
    assert value == pytest.approx(expected, rel=1e-12)


def test_pooled_folds_of_a_pool() -> None:
    folds = year_block_kfold(_pool(), 3)
    table = UNTYPED_INTERVALS(
        [_cover()], folds, rng=np.random.default_rng(1), n_bootstrap=20, options=FEW
    )
    assert _interval(table, "expected_loss")["ci_low"] is not None
    text = setup_text(table)
    assert "36 pooled model-years" in text
    assert text.count("Pool:") == 1


def test_folds_with_their_own_trend_fits_read_each_years_own_loss() -> None:
    years = np.arange(2000, 2020)
    series = AnnualSeries(
        years,
        50.0 + 1.5 * (years - 2000.0) + 20.0 * np.sin(years * 1.3),
        "mm/day",
        {"source": {"name": "synthetic"}},
    )

    def fold(position: int) -> tuple[OutcomeSample, OutcomeSample]:
        held_out = np.array_split(years, 4)[position]
        sample = sample_annual(
            series,
            loss_model=LossModel(
                PiecewiseLinearImpact([40.0, 100.0], [0.0, 1.0]),
                value=1000.0,
                value_unit="EUR",
            ),
            basis_risk=ImpactNoise(0.2),
            asset_id="a",
            rng=np.random.default_rng(4),
            trend=LinearTrend(fit_years=np.setdiff1d(years, held_out)),
        )
        return year_block_kfold(sample, 4)[position]

    folds = tuple(fold(position) for position in range(4))
    designs = _designs(folds)
    table = UNTYPED_POOLED([designs], folds)
    row = _row(table)
    loss, _, _, _ = _pooled_arrays(designs, folds)
    assert row["expected_loss"] == pytest.approx(loss.mean())
    # Each fold states its own values, so no one fit gives these rows.
    assert len({test.loss[0, 0] for _, test in folds}) == 4
    assert "the leakage check passed on every fold" in setup_text(table)


def test_an_empirical_copula_block_is_resampled_by_row() -> None:
    joined = _book(12)
    sample = OutcomeSample(
        asset_ids=joined.asset_ids,
        period=joined.period,
        probability=joined.probability,
        index=joined.index,
        loss=joined.loss,
        hazard=None,
        index_unit="mm/day",
        loss_unit="EUR",
        semantics="annual",
        provenance=_provenance(dependence={"kind": "empirical_copula"}),
    )
    covers = {asset: _cover(asset=asset) for asset in ("a", "b")}
    table = UNTYPED_INTERVALS(
        [covers],
        sample,
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=5,
        options=FEW,
    )
    intervals = _metadata(table).setup.intervals
    assert intervals is not None and intervals["unit"] == "row"
    assert "resamples of the evaluated rows" in setup_text(table)


def test_a_call_serves_every_entry_with_one_set_of_replicates() -> None:
    sample = _sample()
    covers = [_cover(800.0, label="low"), _cover(900.0, label="high")]
    both = UNTYPED_INTERVALS(
        covers,
        sample,
        split="in_sample",
        rng=np.random.default_rng(6),
        n_bootstrap=30,
        options=FEW,
    )
    for position, cover in enumerate(covers):
        alone = UNTYPED_INTERVALS(
            [cover],
            sample,
            split="in_sample",
            rng=np.random.default_rng(6),
            n_bootstrap=30,
            options=FEW,
        )
        assert (
            _interval(both, "cvar_uncovered_b0_9", position)["ci_low"]
            == (_interval(alone, "cvar_uncovered_b0_9")["ci_low"])
        )


# -- stated premiums -----------------------------------------------------------


def test_a_stated_premium_is_fixed() -> None:
    sample = _sample()
    table = UNTYPED_INTERVALS(
        [_cover()],
        sample,
        split="in_sample",
        premiums={"linear": 75.0},
        rng=np.random.default_rng(1),
        n_bootstrap=25,
        options=FEW,
    )
    row = _interval(table, "premium")
    assert row["premium_basis"] == "stated"
    assert row["value"] == row["ci_low"] == row["ci_high"] == 75.0
    net = metric_values(
        sample.loss[:, 0],
        _cover().payout_values(sample.index[:, 0]),
        sample.probability,
        premium=75.0,
        options=FEW,
    )
    assert _interval(table, "cvar_net_loss_b0_9")["value"] == pytest.approx(
        net["cvar_net_loss_b0_9"]
    )


def test_stated_premiums_of_a_set_sum_on_its_total() -> None:
    covers = {
        "a": _cover(asset="a", label="first"),
        "b": _cover(asset="b", label="second"),
    }
    table = UNTYPED_INTERVALS(
        [covers],
        _book(),
        split="in_sample",
        premiums={"first": 10.0, "second": 20.0},
        rng=np.random.default_rng(1),
        n_bootstrap=5,
        options=FEW,
    )
    rows = [row for row in table.to_pylist() if row["column"] == "premium"]
    assert [row["value"] for row in rows] == [10.0, 20.0, 30.0]
    assert {row["premium_basis"] for row in rows} == {"stated"}


@pytest.mark.parametrize(
    ("premiums", "message"),
    [
        ({"nobody": 1.0}, "match no cover given as is"),
        ({"linear": "cheap"}, "premium must be a number"),
        ({"linear": None}, "states no number"),
    ],
)
def test_premiums_are_checked(premiums: Any, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        UNTYPED_INTERVALS(
            [_cover()],
            _sample(),
            split="in_sample",
            premiums=premiums,
            rng=np.random.default_rng(1),
            n_bootstrap=2,
        )


def test_a_premium_for_a_design_matches_no_cover_given_as_is() -> None:
    train, test = holdout(_sample())
    with pytest.raises(EvaluationError, match="match no cover given as is"):
        UNTYPED_INTERVALS(
            [_Design(_cover(), train)],
            test,
            split="out_of_sample",
            premiums={"linear": 1.0},
            rng=np.random.default_rng(1),
            n_bootstrap=2,
        )


def test_premiums_state_every_cover_of_a_set_or_none() -> None:
    covers = {
        "a": _cover(asset="a", label="first"),
        "b": _cover(asset="b", label="second"),
    }
    with pytest.raises(EvaluationError, match="every cover of a set"):
        UNTYPED_INTERVALS(
            [covers],
            _book(),
            split="in_sample",
            premiums={"first": 1.0},
            rng=np.random.default_rng(1),
            n_bootstrap=2,
        )


def test_premiums_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="premiums must map"):
        UNTYPED_INTERVALS(
            [_cover()],
            _sample(),
            split="in_sample",
            premiums=[1.0],
            rng=np.random.default_rng(1),
            n_bootstrap=2,
        )


# -- arguments -----------------------------------------------------------------


@pytest.mark.parametrize("rng", [None, 7, np.random.RandomState(1)])
def test_a_generator_is_required(rng: Any) -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        UNTYPED_INTERVALS([_cover()], _sample(), split="in_sample", rng=rng)


def test_rng_has_no_default() -> None:
    with pytest.raises(TypeError, match="rng"):
        UNTYPED_INTERVALS([_cover()], _sample(), split="in_sample")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"n_bootstrap": 0}, "n_bootstrap"),
        ({"n_bootstrap": 1.5}, "n_bootstrap"),
        ({"n_bootstrap": True}, "n_bootstrap"),
        ({"ci_level": 0.0}, "ci_level"),
        ({"ci_level": 1.0}, "ci_level"),
        ({"ci_level": float("nan")}, "ci_level"),
    ],
)
def test_resampling_arguments_are_checked(arguments: Any, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        UNTYPED_INTERVALS(
            [_cover()],
            _sample(),
            split="in_sample",
            rng=np.random.default_rng(1),
            **arguments,
        )


def test_one_block_needs_a_split() -> None:
    with pytest.raises(EvaluationError, match="split= is required"):
        UNTYPED_INTERVALS([_cover()], _sample(), rng=np.random.default_rng(1))


@pytest.mark.parametrize(
    ("arguments", "named"),
    [
        ({"split": "out_of_sample"}, "split"),
        ({"design_sample": "anything"}, "design_sample"),
        ({"premiums": {"linear": 1.0}}, "premiums"),
        ({"fold": 2}, "fold"),
    ],
)
def test_block_arguments_are_refused_with_folds(arguments: Any, named: str) -> None:
    with pytest.raises(EvaluationError, match=f"for one block.*{named}"):
        UNTYPED_INTERVALS(
            [_cover()], _folds(), rng=np.random.default_rng(1), **arguments
        )


# -- the metadata of intervals -------------------------------------------------


def test_the_metadata_records_the_intervals_and_the_generator() -> None:
    rng = np.random.default_rng(13)
    state = json.dumps(rng.bit_generator.state, sort_keys=True)
    sample = _sample()
    table = UNTYPED_INTERVALS(
        [_cover()], sample, split="in_sample", rng=rng, n_bootstrap=15, ci_level=0.8
    )
    metadata = _metadata(table)
    metadata.check_table(table)
    intervals = metadata.setup.intervals
    assert intervals is not None
    assert {
        key: intervals[key] for key in intervals if key != "short_block_columns"
    } == {
        "built_together": [],
        "ci_level": 0.8,
        "method": "percentile",
        "n_bootstrap": 15,
        "unit": "year",
    }
    records = [record for record in metadata.rng if record["step"] == "bootstrap"]
    assert len(records) == 1
    assert records[0]["bit_generator"] == "PCG64"
    assert records[0]["sample_hash"] == sample.content_hash
    assert json.loads(records[0]["state"]) == json.loads(state)
    rebuilt = np.random.Generator(np.random.PCG64())
    rebuilt.bit_generator.state = json.loads(records[0]["state"])
    again = UNTYPED_INTERVALS(
        [_cover()], sample, split="in_sample", rng=rebuilt, n_bootstrap=15, ci_level=0.8
    )
    assert table.column("ci_high").equals(again.column("ci_high"))


def test_the_setup_text_of_intervals() -> None:
    folds = _folds()
    table = UNTYPED_INTERVALS(
        [_designs(folds)], folds, rng=np.random.default_rng(1), n_bootstrap=20
    )
    lines = setup_text(table).splitlines()
    line = next(item for item in lines if item.startswith("Intervals:"))
    assert line.startswith(
        "Intervals: percentile, level 0.9, from 20 resamples of the evaluated years, "
        "each with every row of that year, with replacement and with the designs "
        "held fixed. An interval is the middle 0.9 of the recomputed values. It "
        "describes how much a number moves with the years drawn. It leaves out "
        "uncertainty in the hazard data, the impact function and the design, and on "
        "a curve block it is the Monte Carlo error of the draws and says nothing "
        "about the curve. Two rows' intervals are not a test of a difference between "
        "them."
    )
    assert "cvar_loss_b0_99" in line
    assert "expected_loss," not in line
    assert lines[-1].startswith("Metrics:")


def test_the_setup_text_names_the_flagged_columns_of_the_rows_held() -> None:
    table = UNTYPED_INTERVALS(
        [_cover()],
        _sample(),
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=5,
        options=FEW,
    )
    mask = pa.array(
        [name == "expected_loss" for name in table.column("column").to_pylist()]
    )
    line = next(
        item
        for item in setup_text(table.filter(mask)).splitlines()
        if item.startswith("Intervals:")
    )
    assert "reaches the block's worst row" not in line


def test_intervals_are_written_and_read_back(tmp_path: Path) -> None:
    folds = _folds()
    table = UNTYPED_INTERVALS(
        [_designs(folds)], folds, rng=np.random.default_rng(1), n_bootstrap=5
    )
    path = tmp_path / "intervals.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.num_rows == table.num_rows
    assert _metadata(read).setup.intervals == _metadata(table).setup.intervals
    assert setup_text(read) == setup_text(table)


def test_the_rows_of_one_metric_can_be_written(tmp_path: Path) -> None:
    table = _intervals_of(_sample())
    kept = table.filter(
        pa.array(
            [name == "expected_loss" for name in table.column("column").to_pylist()]
        )
    )
    path = tmp_path / "one.parquet"
    write_comparison(kept, path)
    assert pq.read_table(path).num_rows == 1


def test_a_selection_without_the_column_names_keeps_the_flagged_columns() -> None:
    table = _intervals_of(_sample())
    kept = table.drop_columns(["column"])
    assert setup_text(kept) == setup_text(table)


def test_an_intervals_table_is_checked_against_its_metadata() -> None:
    table = UNTYPED_INTERVALS(
        [_cover()],
        _sample(),
        split="in_sample",
        rng=np.random.default_rng(1),
        n_bootstrap=2,
    )
    metadata = _metadata(table)
    with pytest.raises(EvaluationError, match="lacks the columns"):
        metadata.check_table(table.drop_columns(["n_undefined"]))
    with pytest.raises(EvaluationError, match="repeats a row key"):
        metadata.check_table(pa.concat_tables([table, table]))


def _intervals_of(sample: OutcomeSample, **arguments: Any) -> Any:
    return UNTYPED_INTERVALS(
        [_cover()],
        sample,
        split="in_sample",
        rng=np.random.default_rng(1),
        **{"n_bootstrap": 3, **arguments},
    )


def test_merging_intervals_combines_the_flagged_columns() -> None:
    short, long = _intervals_of(_sample(10)), _intervals_of(_sample(30))
    merged = EvaluationMetadata.merge(_metadata(short), _metadata(long))
    assert merged.setup.intervals is not None
    assert set(merged.setup.intervals["short_block_columns"]) >= set(
        _metadata(short).setup.intervals["short_block_columns"]  # type: ignore[index]
    )


def test_merging_different_intervals_raises() -> None:
    one = _metadata(_intervals_of(_sample()))
    other = _metadata(_intervals_of(_sample(), n_bootstrap=4))
    with pytest.raises(EvaluationError, match="differ in their intervals"):
        EvaluationMetadata.merge(one, other)


def test_merging_with_a_comparison_keeps_the_intervals() -> None:
    sample = _sample()
    wide = _metadata(compare([_cover()], sample, split="in_sample"))
    merged = EvaluationMetadata.merge(wide, _metadata(_intervals_of(sample)))
    assert merged.setup.intervals is not None


# -- a table without them ------------------------------------------------------


def test_a_comparison_has_no_intervals() -> None:
    table = compare([_cover()], _sample(), split="in_sample")
    metadata = _metadata(table)
    assert metadata.setup.intervals is None
    assert "Intervals:" not in setup_text(table)
    assert "bootstrap" not in {record["step"] for record in metadata.rng}


def test_a_document_without_the_key_reads_as_before() -> None:
    table = compare([_cover()], _sample(), split="in_sample")
    document = json.loads(_metadata(table).to_json())
    del document["setup"]["intervals"]
    read = EvaluationMetadata.from_json(json.dumps(document))
    assert read.setup.intervals is None
    assert setup_text(read) == setup_text(_metadata(table))
