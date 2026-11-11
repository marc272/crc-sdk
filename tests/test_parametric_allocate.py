"""A joint design's allocated rows, in a table of their own.

`allocate` returns them alone, and `allocation=` takes joint designs only, so
allocated rows never share a table with the rows of covers that stand on their
own. The samples are synthetic; nothing here is a finding about a design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    Budget,
    Cover,
    EvaluationError,
    EvaluationMetadata,
    Free,
    IndexDefinition,
    JointDesignResult,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    allocate,
    bootstrap_intervals,
    compare,
    compare_pooled,
    design_joint,
    evaluate,
    holdout,
    setup_text,
    to_long,
    write_comparison,
    year_block_kfold,
)
from crc_sdk.parametric.metadata import _identity_columns

UNTYPED_ALLOCATE: Any = allocate
UNTYPED_COMPARE: Any = compare
UNTYPED_INTERVALS: Any = bootstrap_intervals

INDEX = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
VALUES = {"A": 300.0, "B": 100.0}


def _sample(n: int = 24) -> OutcomeSample:
    years = np.arange(n, dtype=np.float64)
    index = np.column_stack(
        [2.0 + 2.0 * np.sin(years * 1.7), 2.0 + 2.0 * np.cos(years * 1.3)]
    )
    loss = np.where(index > 2.5, (index - 2.5) * 10.0, 0.0)
    return OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(2000, 2000 + n),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=loss,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
            licences=({"role": "table", "licence": "CC0-1.0"},),
            dependence={"kind": "data"},
        ),
    )


def _step(level: float | Free, *, asset: str, label: str | None = "step") -> Cover:
    return Cover(
        INDEX,
        StepPayout([3.0], [level]),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _joint(sample: OutcomeSample) -> JointDesignResult:
    return design_joint(
        [_step(Free(), asset=a) for a in sample.asset_ids], sample, MSE()
    )


FIXED = {"A": _step(10.0, asset="A", label="fixed"), "B": _step(5.0, asset="B")}


# -- the rows ------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["value", "loss"])
def test_allocate_returns_compare_s_allocated_rows_alone(rule: Any) -> None:
    sample = _sample()
    design = _joint(sample)
    alone = allocate(
        [design], sample, split="in_sample", allocation=rule, values=VALUES
    )
    beside = compare(
        [design], sample, split="in_sample", allocation=rule, values=VALUES
    ).to_pylist()
    level = f"asset_by_{rule}"
    assert [row["level"] for row in beside] == ["total", level, level]
    assert alone.to_pylist() == beside[1:]
    assert alone.schema.equals(
        compare([design], sample, split="in_sample").schema, check_metadata=False
    )
    for row in alone.to_pylist():
        assert (row["scope"], row["premium_basis"], row["premium"]) == (
            "joint",
            None,
            None,
        )
        assert row["cover_hash"] == design.covers_hash
        assert row["design_id"] == design.design_id


def test_the_shares_add_up_to_the_total_row() -> None:
    sample = _sample()
    design = _joint(sample)
    total = compare([design], sample, split="in_sample").to_pylist()[0]
    rows = allocate(
        [design], sample, split="in_sample", allocation="loss", values=VALUES
    ).to_pylist()
    for name in ("expected_loss", "expected_payout", "expected_uncovered"):
        assert sum(row[name] for row in rows) == pytest.approx(total[name])


def test_several_designs_and_a_held_out_block() -> None:
    sample = _sample(30)
    train, test = holdout(sample, 0.6)
    designs = [
        _joint(train),
        design_joint(
            [_step(Free(), asset=a) for a in train.asset_ids],
            train,
            MSE(),
            budget=Budget(ratio=0.5),
        ),
    ]
    table = allocate(
        designs, test, split="out_of_sample", allocation="value", values=VALUES
    )
    rows = table.to_pylist()
    assert [(row["design_id"], row["asset_id"]) for row in rows] == [
        (design.design_id, asset) for design in designs for asset in ("A", "B")
    ]
    assert {row["split"] for row in rows} == {"out_of_sample"}
    assert {row["sample_hash"] for row in rows} == {test.content_hash}
    (evaluation,) = EvaluationMetadata.from_table(table).setup.evaluations
    assert evaluation["leakage_check"] == "passed"
    with pytest.raises(SampleError, match="in sample means on the design sample"):
        allocate(designs, test, split="in_sample", allocation="value", values=VALUES)


# -- the metadata, the text, files ---------------------------------------------


def test_the_metadata_records_the_rule_and_no_pricing() -> None:
    sample = _sample()
    design = _joint(sample)
    table = allocate(
        [design], sample, split="in_sample", allocation="value", values=VALUES
    )
    metadata = EvaluationMetadata.from_table(table)
    (evaluation,) = metadata.setup.evaluations
    assert evaluation["level"] == ["asset_by_value"]
    assert evaluation["scope"] == ["joint"]
    assert evaluation["allocation"] == "value"
    assert evaluation["pricing"] is None
    assert metadata.assets["values_sha256"] is not None
    assert design.covers_hash in {entry.hash for entry in metadata.covers}
    assert [record["design_id"] for record in metadata.designs] == [design.design_id]
    metadata.check_table(table)


def test_the_setup_text_states_the_rule_and_no_premium() -> None:
    sample = _sample()
    table = allocate(
        [_joint(sample)], sample, split="in_sample", allocation="loss", values=VALUES
    )
    lines = setup_text(table).splitlines()
    assert lines[0].startswith(
        "Level: asset_by_loss, one row per asset (A and B): its share of the joint "
        "design's total payout in proportion to its loss in the draw"
    )
    assert "an accounting rule and not the payout of the asset's cover" in lines[0]
    assert "total, the sum over" not in lines[0]
    pricing = [line for line in lines if line.startswith("Pricing: ")]
    assert pricing == [
        "Pricing: allocated rows hold no premium and no net-loss values, as a "
        "pricing prices a joint design's total payout and not a part of it; a "
        "design's line states its pricing."
    ]


def test_a_table_of_allocated_rows_round_trips_and_has_a_long_form(
    tmp_path: Path,
) -> None:
    sample = _sample()
    table = allocate(
        [_joint(sample)], sample, split="in_sample", allocation="value", values=VALUES
    )
    write_comparison(table, tmp_path / "allocated.parquet")
    read = pq.read_table(tmp_path / "allocated.parquet")
    assert read.to_pylist() == table.to_pylist()
    assert setup_text(read) == setup_text(table)
    long = to_long(table)
    n_metrics = table.num_columns - len(_identity_columns(table.column_names))
    assert long.num_rows == n_metrics * table.num_rows


# -- what allocate refuses -----------------------------------------------------


def test_allocate_takes_joint_designs_only() -> None:
    sample = _sample()
    with pytest.raises(TypeError, match="JointDesignResult objects, got dict"):
        UNTYPED_ALLOCATE(
            [_joint(sample), FIXED],
            sample,
            split="in_sample",
            allocation="value",
            values=VALUES,
        )
    with pytest.raises(TypeError, match="sequence of JointDesignResult"):
        UNTYPED_ALLOCATE(
            _joint(sample), sample, split="in_sample", allocation="value", values=VALUES
        )
    with pytest.raises(EvaluationError, match="at least one joint design"):
        allocate([], sample, split="in_sample", allocation="value", values=VALUES)


@pytest.mark.parametrize(
    ("allocation", "values", "message"),
    [
        (None, VALUES, "allocate needs a rule"),
        ("area", VALUES, "allocation must be"),
        ("value", None, "needs values="),
        ("value", {"A": 1.0}, "every asset of the sample"),
        ("value", {"A": 0.0, "B": 0.0}, "all zero"),
    ],
)
def test_allocate_refuses_a_missing_rule_or_values(
    allocation: Any, values: Any, message: str
) -> None:
    sample = _sample()
    with pytest.raises(EvaluationError, match=message):
        UNTYPED_ALLOCATE(
            [_joint(sample)],
            sample,
            split="in_sample",
            allocation=allocation,
            values=values,
        )


def test_allocate_refuses_a_design_twice() -> None:
    sample = _sample()
    design = _joint(sample)
    with pytest.raises(EvaluationError, match="repeat a row key"):
        allocate(
            [design, design],
            sample,
            split="in_sample",
            allocation="value",
            values=VALUES,
        )


# -- no table holds allocated rows beside covers that stand on their own -------


def test_compare_refuses_allocated_rows_beside_other_entries() -> None:
    sample = _sample()
    with pytest.raises(EvaluationError, match=r"positions \[1\] are not joint"):
        compare(
            [_joint(sample), FIXED],
            sample,
            split="in_sample",
            allocation="value",
            values=VALUES,
        )
    # Without an allocation the two stand in one table, a total row each.
    rows = compare([_joint(sample), FIXED], sample, split="in_sample").to_pylist()
    assert [(row["level"], row["scope"]) for row in rows] == [
        ("total", "joint"),
        ("asset", "individual"),
        ("asset", "individual"),
        ("total", "individual"),
    ]


def test_evaluate_refuses_an_allocation_of_a_set_of_covers() -> None:
    with pytest.raises(EvaluationError, match="no entry is a joint design"):
        evaluate(FIXED, _sample(), split="in_sample", allocation="value", values=VALUES)


def test_pooled_folds_and_intervals_refuse_them_too() -> None:
    sample = _sample(30)
    folds = year_block_kfold(sample, 3)
    joint = [_joint(train) for train, _ in folds]
    with pytest.raises(EvaluationError, match="not joint designs"):
        compare_pooled([joint, FIXED], folds, allocation="value", values=VALUES)
    with pytest.raises(EvaluationError, match="not joint designs"):
        UNTYPED_INTERVALS(
            [_joint(sample), FIXED],
            sample,
            split="in_sample",
            rng=np.random.default_rng(1),
            n_bootstrap=5,
            allocation="value",
            values=VALUES,
        )
    # Joint designs alone keep their pooled allocated rows.
    rows = compare_pooled([joint], folds, allocation="value", values=VALUES)
    level = "asset_by_value"
    assert rows.column("level").to_pylist() == ["total", level, level]
    assert compare_pooled([FIXED], folds).num_rows == 3
