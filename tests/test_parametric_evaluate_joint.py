"""Comparison tables on samples of several assets: totals, allocations, the setup.

The joint fixture cases are read as JSON with plain numpy, as
`tests/fixtures/parametric/README.md` describes; no reference code is
imported. Every design case of several assets in the folder is judged, so new
ones are picked up when they are copied. The tests build tables only to check
their rows and columns; nothing here is a finding about a design.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from numpy.typing import ArrayLike

from crc_sdk.parametric import (
    MSE,
    Budget,
    CostOfCapitalLoad,
    Cover,
    DesignOptions,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    JointDesignResult,
    LeakageError,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    compare,
    design,
    design_joint,
    evaluate,
    holdout,
    setup_text,
    to_long,
    write_comparison,
)
from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.constraints import budget_from_document, limit_from_document
from crc_sdk.parametric.joint import covers_hash
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    IDENTITY_COLUMNS,
    EvaluationSetup,
    _identity_columns,
)
from crc_sdk.parametric.metrics import MetricOptions, metric_columns, metric_values
from crc_sdk.parametric.objectives import objective_from_document
from crc_sdk.parametric.pricing import pricing_from_document

UNTYPED_EVALUATE: Any = evaluate
UNTYPED_COMPARE: Any = compare

CASES = Path(__file__).parent / "fixtures" / "parametric" / "cases"
UNITLESS = (
    "coverage_ratio",
    "payout_to_loss_ratio",
    "correlation",
    "tail_coverage_",
    "cvar_uncovered_to_loss_",
    "false_negative_rate_",
    "false_positive_rate_",
)
NO_PREMIUM = {
    column.name
    for column in metric_columns(MetricOptions())
    if column.metric in ("premium", "var_net_loss", "cvar_net_loss", "net_loss_rp")
}


# -- the fixture cases -------------------------------------------------------


def _case(path: Path) -> dict[str, Any]:
    case: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return case


def _joint_paths() -> list[Path]:
    return [
        path for path in sorted(CASES.glob("*.json")) if _case(path)["scope"] == "joint"
    ]


UNIQUE_CASES = [p for p in _joint_paths() if _case(p)["expected"]["unique"]]
OTHER_CASES = [p for p in _joint_paths() if not _case(p)["expected"]["unique"]]


def _case_sample(case: Mapping[str, Any]) -> OutcomeSample:
    block = case["sample"]
    return OutcomeSample(
        asset_ids=tuple(block["asset_ids"]),
        period=np.asarray(block["period"], dtype=np.int64),
        probability=np.asarray(block["probability"], dtype=np.float64),
        index=np.asarray(block["index"], dtype=np.float64),
        loss=np.asarray(block["loss"], dtype=np.float64),
        hazard=None,
        index_unit=block["index_unit"],
        loss_unit=block["loss_unit"],
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": case["case_id"]},),
            semantics_source="caller",
            dependence={"kind": "data"},
        ),
    )


def _check_metrics(row: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    scale = float(max(sum(line) for line in case["sample"]["loss"]))
    relative = case["tolerances"]["metrics"]
    expected = case["expected"]["metrics"]
    assert expected
    for name, value in expected.items():
        got = row[name]
        floor = 1e-9 if name.startswith(UNITLESS) else 1e-9 * scale
        if isinstance(value, dict):
            # A quantile on a step: either atom passes the fixture; the
            # module's rule gives the lower one, so check that one.
            lower, upper = value["either"]
            assert lower <= upper
            value = lower
        assert got is not None, name
        assert abs(got - value) <= relative * abs(value) + floor, (name, got, value)


def test_the_joint_cases_divide_into_five_and_eleven() -> None:
    assert (len(UNIQUE_CASES), len(OTHER_CASES)) == (5, 11)


@pytest.mark.parametrize("path", UNIQUE_CASES, ids=lambda path: path.stem)
def test_the_total_row_of_a_unique_joint_case(path: Path) -> None:
    case = _case(path)
    sample = _case_sample(case)
    result = design_joint(
        [Cover.from_document(document) for document in case["cover"]],
        sample,
        objective_from_document(case["objective"]),
        pricing=pricing_from_document(case["pricing"]),
        budget=budget_from_document(case["budget"]),
        limit=limit_from_document(case["limit"]),
        options=DesignOptions(ranges=True),
    )
    assert result.unique is True
    table = compare([result], sample, split="in_sample")
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert (row["level"], row["scope"]) == ("total", "joint")
    assert row["asset_id"] is None
    assert row["premium_basis"] == "design"
    assert row["cover_hash"] == result.covers_hash
    assert row["design_id"] == result.design_id
    assert row["unique"] is True
    _check_metrics(row, case)


@pytest.mark.parametrize("path", OTHER_CASES, ids=lambda path: path.stem)
def test_the_total_row_of_a_non_unique_joint_case(path: Path) -> None:
    case = _case(path)
    sample = _case_sample(case)
    expected = case["expected"]
    resolved = {
        asset: Cover.from_document(document).with_values(expected["parameters"][asset])
        for asset, document in zip(sample.asset_ids, case["cover"], strict=True)
    }
    table = evaluate(
        resolved,
        sample,
        split="in_sample",
        pricing=pricing_from_document(case["pricing"]),
        premium=expected["premium"],
    )
    rows = table.to_pylist()
    assert [row["level"] for row in rows] == ["asset"] * 3 + ["total"]
    total = rows[-1]
    assert (total["scope"], total["asset_id"]) == ("individual", None)
    assert total["premium_basis"] == "stated"
    assert total["premium"] == expected["premium"]
    _check_metrics(total, case)


# -- small samples -----------------------------------------------------------


def _sample(
    index: ArrayLike,
    loss: ArrayLike,
    *,
    asset_ids: tuple[str, ...] = ("A", "B"),
    period: ArrayLike | None = None,
    lineage: str | None = None,
) -> OutcomeSample:
    values = np.asarray(index, dtype=np.float64)
    n = values.shape[0]
    return OutcomeSample(
        asset_ids=asset_ids,
        period=np.arange(n, dtype=np.int64) if period is None else np.asarray(period),
        probability=np.full(n, 1.0 / n),
        index=values,
        loss=np.asarray(loss, dtype=np.float64),
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "test"},),
            semantics_source="caller",
            licences=({"role": "table", "licence": "CC0-1.0"},),
            lineage=lineage,
            dependence={"kind": "data"},
        ),
    )


INDEX = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
# A's trigger is reached in years 1 and 2, B's in year 3; year 2 has no loss.
BOOK = _sample(
    [[0.0, 0.0], [3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
    [[0.0, 0.0], [40.0, 0.0], [0.0, 0.0], [0.0, 20.0], [0.0, 0.0]],
)
VALUES = {"A": 300.0, "B": 100.0}
PROBABILITY = BOOK.probability
TOTAL_LOSS = BOOK.loss.sum(axis=1)
# The MSE design pays the mean loss of each cover's triggered years, 20 each.
TOTAL_PAYOUT = np.array([0.0, 20.0, 20.0, 20.0, 0.0])


def _step(
    level: float | Free, *, asset: str | None = None, label: str | None = None
) -> Cover:
    return Cover(
        INDEX,
        StepPayout([1.0], [level]),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _joint(*, label: str | None = None) -> JointDesignResult:
    templates = [_step(Free(), asset=a, label=label) for a in BOOK.asset_ids]
    return design_joint(templates, BOOK, MSE())


FIXED = {"A": _step(20.0, asset="A"), "B": _step(20.0, asset="B")}


def _by_level(table: pa.Table) -> dict[tuple[str, str | None], dict[str, Any]]:
    return {(row["level"], row["asset_id"]): row for row in table.to_pylist()}


def _setup_line(table: Any, start: str) -> str:
    return next(line for line in setup_text(table).splitlines() if start in line)


# -- the rows of a joint design ----------------------------------------------


def test_a_joint_design_gets_one_total_row() -> None:
    result = _joint()
    table = compare([result], BOOK, split="in_sample")
    assert tuple(table.column_names[: len(IDENTITY_COLUMNS)]) == IDENTITY_COLUMNS
    assert table.column_names.index("scope") == table.column_names.index("level") + 1
    rows = table.to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["level"] == "total" and row["scope"] == "joint"
    assert row["asset_id"] is None
    assert row["cover_hash"] == result.covers_hash == covers_hash(result.covers)
    assert row["design_id"] == result.design_id
    assert row["premium_basis"] == "design"
    assert row["premium"] == result.premium
    assert row["unique"] is True and row["ties"] == 0
    assert row["degenerate_tail"] is None
    assert row["sample_hash"] == BOOK.content_hash
    assert row["fold"] == 0 and row["split"] == "in_sample"
    schema = table.schema
    assert schema.field("asset_id").nullable
    assert schema.field("premium_basis").nullable
    assert not schema.field("scope").nullable


def test_a_total_row_is_the_sum_of_the_losses_against_the_sum_of_the_payouts() -> None:
    row = compare([_joint()], BOOK, split="in_sample").to_pylist()[0]
    # L = (0, 40, 0, 20, 0), payout = (0, 20, 20, 20, 0), both over five years.
    assert row["expected_loss"] == pytest.approx(12.0)
    assert row["expected_payout"] == pytest.approx(12.0)
    assert row["expected_uncovered"] == pytest.approx(4.0)
    assert row["coverage_ratio"] == pytest.approx(8.0 / 12.0)
    expected = metric_values(
        TOTAL_LOSS,
        TOTAL_PAYOUT,
        PROBABILITY,
        premium=row["premium"],
        options=MetricOptions(),
        unidentified=np.zeros(5, dtype=np.bool_),
    )
    for name, value in expected.items():
        assert row[name] == pytest.approx(value, abs=1e-6), name


def test_a_stated_premium_is_the_price_of_the_total() -> None:
    row = evaluate(_joint(), BOOK, split="in_sample", premium=3.5).to_pylist()[0]
    assert row["premium"] == 3.5
    assert row["premium_basis"] == "stated"


@pytest.mark.parametrize("rule", ["value", "loss"])
def test_allocated_rows_need_a_rule_and_add_up_to_the_total(rule: Any) -> None:
    result = _joint()
    table = compare(
        [result], BOOK, split="in_sample", allocation=rule, values=VALUES
    ).to_pylist()
    level = f"asset_by_{rule}"
    assert [(row["level"], row["asset_id"]) for row in table] == [
        ("total", None),
        (level, "A"),
        (level, "B"),
    ]
    total, *allocated = table
    for row in allocated:
        assert row["scope"] == "joint"
        assert row["cover_hash"] == result.covers_hash
        assert row["design_id"] == result.design_id
        assert row["unique"] is True and row["ties"] == 0
    for name in ("expected_loss", "expected_payout"):
        assert sum(row[name] for row in allocated) == pytest.approx(total[name])
    if rule == "loss":
        assert sum(row["expected_uncovered"] for row in allocated) == pytest.approx(
            total["expected_uncovered"]
        )


@pytest.mark.parametrize("rule", ["value", "loss"])
def test_premium_and_net_loss_columns_are_null_on_allocated_rows(rule: Any) -> None:
    table = compare([_joint()], BOOK, split="in_sample", allocation=rule, values=VALUES)
    total, *allocated = table.to_pylist()
    assert total["premium_basis"] == "design"
    assert all(total[name] is not None for name in NO_PREMIUM)
    for row in allocated:
        assert row["premium_basis"] is None
        assert all(row[name] is None for name in NO_PREMIUM)
        others = set(row) - set(_identity_columns(row)) - NO_PREMIUM
        assert all(row[name] is not None for name in others - {"correlation"})


def test_the_value_rule_shares_each_draw_by_the_values() -> None:
    rows = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    by_level = _by_level(rows)
    share = {"A": 0.75, "B": 0.25}
    for column, asset in enumerate(BOOK.asset_ids):
        row = by_level[("asset_by_value", asset)]
        expected = metric_values(
            BOOK.loss[:, column],
            TOTAL_PAYOUT * share[asset],
            PROBABILITY,
            premium=0.0,
            options=MetricOptions(),
            unidentified=np.zeros(5, dtype=np.bool_),
        )
        for name, value in expected.items():
            if name not in NO_PREMIUM:
                assert row[name] == pytest.approx(value, abs=1e-6), (asset, name)
    # Payouts go to assets without a loss in that year, so E[U] does not add up.
    total = by_level[("total", None)]["expected_uncovered"]
    parts = sum(
        by_level[("asset_by_value", a)]["expected_uncovered"] for a in BOOK.asset_ids
    )
    assert parts != pytest.approx(total)


def test_the_loss_rule_shares_each_draw_by_its_losses() -> None:
    by_level = _by_level(
        compare([_joint()], BOOK, split="in_sample", allocation="loss", values=VALUES)
    )
    # Year 1: all of it to A. Year 2 has no loss, so the value shares, 15 and 5.
    # Year 3: all of it to B.
    a = by_level[("asset_by_loss", "A")]
    b = by_level[("asset_by_loss", "B")]
    assert a["expected_payout"] == pytest.approx((20.0 + 15.0) / 5.0)
    assert b["expected_payout"] == pytest.approx((5.0 + 20.0) / 5.0)
    assert a["expected_uncovered"] == pytest.approx(20.0 / 5.0)
    assert b["expected_uncovered"] == pytest.approx(0.0, abs=1e-9)
    assert a["expected_loss"] == pytest.approx(8.0)
    assert b["expected_loss"] == pytest.approx(4.0)


def test_a_zero_value_takes_no_share_of_a_payout() -> None:
    by_level = _by_level(
        compare(
            [_joint()],
            BOOK,
            split="in_sample",
            allocation="value",
            values={"A": 1.0, "B": 0.0},
        )
    )
    assert by_level[("asset_by_value", "B")]["expected_payout"] == 0.0
    assert by_level[("asset_by_value", "A")]["expected_payout"] == pytest.approx(12.0)


# -- the rows of a set of covers ---------------------------------------------


def test_a_set_of_covers_gets_asset_rows_and_a_total() -> None:
    table = evaluate(FIXED, BOOK, split="in_sample")
    rows = table.to_pylist()
    assert [(row["level"], row["asset_id"]) for row in rows] == [
        ("asset", "A"),
        ("asset", "B"),
        ("total", None),
    ]
    assert {row["scope"] for row in rows} == {"individual"}
    a, b, total = rows
    assert a["cover_hash"] == FIXED["A"].definition_hash
    assert total["cover_hash"] == covers_hash(FIXED)
    assert total["premium_basis"] == "evaluation"
    assert total["premium"] == pytest.approx(a["premium"] + b["premium"])
    for name in ("design_id", "unique", "ties", "degenerate_tail"):
        assert total[name] is None
    assert a["expected_loss"] == pytest.approx(8.0)
    assert b["expected_loss"] == pytest.approx(4.0)
    assert total["expected_loss"] == pytest.approx(12.0)
    assert total["expected_payout"] == pytest.approx(12.0)
    assert total["expected_uncovered"] == pytest.approx(4.0)


def test_a_mapping_may_list_its_assets_in_any_order() -> None:
    forward = evaluate(FIXED, BOOK, split="in_sample").to_pylist()
    reverse = evaluate(dict(reversed(FIXED.items())), BOOK, split="in_sample")
    assert reverse.to_pylist() == forward


def test_each_cover_reads_its_own_assets_column() -> None:
    reordered = _sample(BOOK.index[:, ::-1], BOOK.loss[:, ::-1], asset_ids=("B", "A"))
    rows = evaluate(FIXED, reordered, split="in_sample").to_pylist()
    assert [row["asset_id"] for row in rows] == ["B", "A", None]
    assert rows[1]["expected_loss"] == pytest.approx(8.0)
    assert rows[0]["expected_loss"] == pytest.approx(4.0)
    assert rows[-1]["expected_payout"] == pytest.approx(12.0)


def test_the_total_premium_of_a_set_is_one_per_contract() -> None:
    # The covers pay in different years. Each contract's price is
    # 0.5 E[payout] + 0.5 CVaR_0.75(payout): 15 and 5. On the total payout
    # the same pricing would give 16.
    sample = _sample(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
        [[0.0, 0.0], [40.0, 0.0], [0.0, 20.0], [0.0, 0.0]],
    )
    covers = {"A": _step(24.0, asset="A"), "B": _step(8.0, asset="B")}
    rows = evaluate(
        covers, sample, split="in_sample", pricing=CostOfCapitalLoad(0.5, 0.75)
    ).to_pylist()
    assert [row["premium"] for row in rows] == pytest.approx([15.0, 5.0, 20.0])
    assert rows[-1]["premium_basis"] == "evaluation"


def test_a_stated_premium_states_the_total_of_a_set_only() -> None:
    rows = evaluate(FIXED, BOOK, split="in_sample", premium=7.0).to_pylist()
    assert [row["premium_basis"] for row in rows] == [
        "evaluation",
        "evaluation",
        "stated",
    ]
    assert [row["premium"] for row in rows] == pytest.approx([8.0, 4.0, 7.0])


def test_a_set_of_designs_keeps_each_designs_premium() -> None:
    designs = {
        asset: design(_step(Free(), asset=asset), BOOK.asset(asset), MSE())
        for asset in BOOK.asset_ids
    }
    table = compare([designs], BOOK, split="in_sample")
    a, b, total = table.to_pylist()
    assert (a["design_id"], b["design_id"]) == (
        designs["A"].design_id,
        designs["B"].design_id,
    )
    assert [row["premium_basis"] for row in (a, b, total)] == ["design"] * 3
    assert total["premium"] == pytest.approx(
        designs["A"].premium + designs["B"].premium
    )
    assert total["design_id"] is None
    assert total["cover_hash"] == covers_hash(
        {asset: result.cover for asset, result in designs.items()}
    )
    metadata = EvaluationMetadata.from_table(table)
    assert {record["design_id"] for record in metadata.designs} == {
        result.design_id for result in designs.values()
    }
    assert {tuple(sample["roles"]) for sample in metadata.setup.samples} == {
        ("evaluation",),
        ("design",),
    }
    assert metadata.setup.evaluations[0]["design_samples"] == sorted(
        result.design_sample.content_hash for result in designs.values()
    )


def test_a_set_of_covers_may_share_a_template_for_no_asset() -> None:
    cover = _step(20.0)
    rows = evaluate({"A": cover, "B": cover}, BOOK, split="in_sample").to_pylist()
    assert [row["cover_hash"] for row in rows[:2]] == [cover.definition_hash] * 2


# -- labels ------------------------------------------------------------------


def test_a_shared_label_names_the_set() -> None:
    labelled = {
        "A": _step(20.0, asset="A", label="storm"),
        "B": _step(20.0, asset="B", label="storm"),
    }
    total = evaluate(labelled, BOOK, split="in_sample").to_pylist()[-1]
    assert total["cover_label"] == "storm"
    joint = compare([_joint(label="storm")], BOOK, split="in_sample").to_pylist()[0]
    assert joint["cover_label"] == "storm"


def test_without_a_shared_label_the_set_takes_its_hash_prefix() -> None:
    mixed = {
        "A": _step(20.0, asset="A", label="one"),
        "B": _step(20.0, asset="B", label="two"),
    }
    total = evaluate(mixed, BOOK, split="in_sample").to_pylist()[-1]
    assert total["cover_label"] == total["cover_hash"].removeprefix("sha256:")[:12]
    unlabelled = evaluate(FIXED, BOOK, split="in_sample").to_pylist()[-1]
    assert (
        unlabelled["cover_label"]
        == unlabelled["cover_hash"].removeprefix("sha256:")[:12]
    )


# -- one table, the metadata, files -------------------------------------------


def test_the_scope_column_keeps_the_rows_of_each_scope_apart() -> None:
    table = compare([_joint(), FIXED], BOOK, split="in_sample")
    assert [(row["scope"], row["level"]) for row in table.to_pylist()] == [
        ("joint", "total"),
        ("individual", "asset"),
        ("individual", "asset"),
        ("individual", "total"),
    ]


def test_a_repeated_entry_raises() -> None:
    result = _joint()
    with pytest.raises(EvaluationError, match="repeat a row key"):
        compare([result, result], BOOK, split="in_sample")
    with pytest.raises(EvaluationError, match="repeat a row key"):
        compare([FIXED, FIXED], BOOK, split="in_sample")


def test_the_evaluation_records_the_levels_the_scopes_and_the_rule() -> None:
    table = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    metadata = EvaluationMetadata.from_table(table)
    evaluation = metadata.setup.evaluations[0]
    assert evaluation["level"] == ["total", "asset_by_value"]
    assert evaluation["scope"] == ["joint"]
    assert evaluation["allocation"] == "value"
    assert metadata.setup.samples[0]["dependence"] == {"kind": "data"}


def test_a_set_of_covers_records_no_rule() -> None:
    metadata = EvaluationMetadata.from_table(evaluate(FIXED, BOOK, split="in_sample"))
    evaluation = metadata.setup.evaluations[0]
    assert evaluation["level"] == ["asset", "total"]
    assert evaluation["scope"] == ["individual"]
    assert evaluation["allocation"] is None
    assert metadata.assets["values_sha256"] is None


def test_the_values_are_only_hashed() -> None:
    table = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    metadata = EvaluationMetadata.from_table(table)
    assert metadata.assets["count"] == 2
    assert metadata.assets["values_sha256"] == content_hash(
        {"asset_values": [["A", 300.0], ["B", 100.0]], "value_column": None}
    )
    text = table.schema.metadata[EVALUATION_METADATA_KEY.encode()].decode()
    assert "300.0" not in text
    other = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values={"A": 1, "B": 3}
    )
    assert (
        EvaluationMetadata.from_table(other).assets["values_sha256"]
        != metadata.assets["values_sha256"]
    )


def test_the_metadata_describes_the_set_a_total_row_names() -> None:
    result = _joint(label="storm")
    table = compare([result], BOOK, split="in_sample")
    metadata = EvaluationMetadata.from_table(table)
    metadata.check_table(table)
    entries = {entry.hash: entry for entry in metadata.covers}
    assert set(entries) == {
        result.covers_hash,
        *(cover.definition_hash for cover in result.covers.values()),
    }
    entry = entries[result.covers_hash]
    assert entry.label == "storm"
    assert entry.document == {
        "kind": "cover_set",
        "covers": {a: c.definition_hash for a, c in result.covers.items()},
    }
    record = metadata.designs[0]
    assert record["design_id"] == result.design_id
    assert record["covers_hash"] == result.covers_hash
    assert record["cover_label"] == "storm"
    assert metadata.solvers == (dict(result.solver),)


def test_a_table_round_trips_through_a_file(tmp_path: Path) -> None:
    table = compare(
        [_joint(), FIXED],
        BOOK,
        split="in_sample",
        allocation="loss",
        values=VALUES,
    )
    path = tmp_path / "comparison.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.to_pylist() == table.to_pylist()
    assert EvaluationMetadata.from_table(read) == EvaluationMetadata.from_table(table)


def test_to_long_keeps_the_null_asset_of_a_total() -> None:
    table = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    long = to_long(table)
    n_metrics = table.num_columns - len(_identity_columns(table.column_names))
    assert long.num_rows == n_metrics * table.num_rows
    assert long.column_names[: len(IDENTITY_COLUMNS)] == list(IDENTITY_COLUMNS)
    premiums = long.filter(pc.equal(long["column"], "premium")).to_pylist()
    assert {(row["level"], row["asset_id"]) for row in premiums} == {
        ("total", None),
        ("asset_by_value", "A"),
        ("asset_by_value", "B"),
    }
    assert [row["value"] is None for row in premiums] == [False, True, True]


def test_merge_keeps_the_hash_of_the_values() -> None:
    result = _joint()
    allocated = compare(
        [result], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    plain = evaluate(FIXED, BOOK, split="in_sample")
    merged = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(plain), EvaluationMetadata.from_table(allocated)
    )
    assert merged.assets["values_sha256"] is not None
    assert (
        merged.assets["values_sha256"]
        == (EvaluationMetadata.from_table(allocated).assets["values_sha256"])
    )
    merged.check_table(pa.concat_tables([allocated, plain]))
    other = compare(
        [result], BOOK, split="in_sample", allocation="value", values={"A": 1, "B": 3}
    )
    with pytest.raises(EvaluationError, match="asset records"):
        EvaluationMetadata.merge(
            EvaluationMetadata.from_table(allocated),
            EvaluationMetadata.from_table(other),
        )


def test_a_table_written_before_the_scope_column_still_reads() -> None:
    table = evaluate(_step(20.0, asset="A"), BOOK.asset("A"), split="in_sample")
    earlier = table.drop_columns(["scope"])
    metadata = EvaluationMetadata.from_table(earlier)
    metadata.check_table(earlier)
    long = to_long(earlier)
    assert "scope" not in long.column_names
    assert setup_text(earlier).startswith("Level: asset (A). Scenario: baseline.")


# -- out of sample -----------------------------------------------------------


def _long_book(n: int = 20) -> OutcomeSample:
    index = np.column_stack(
        [
            np.where(np.arange(n) % 2 == 0, 3.0, 0.0),
            np.where(np.arange(n) % 3, 3.0, 0.0),
        ]
    )
    loss = np.column_stack([index[:, 0] * 4.0, index[:, 1] * 2.0])
    return _sample(index, loss)


def _pair() -> list[Cover]:
    return [_step(Free(), asset="A"), _step(Free(), asset="B")]


def test_a_joint_design_out_of_sample_keeps_its_premium() -> None:
    train, test = holdout(_long_book())
    result = design_joint(_pair(), train, MSE())
    table = compare([result], test, split="out_of_sample")
    row = table.to_pylist()[0]
    assert row["split"] == "out_of_sample"
    assert row["premium_basis"] == "design"
    assert row["premium"] == result.premium
    assert row["sample_hash"] == test.content_hash
    evaluation = EvaluationMetadata.from_table(table).setup.evaluations[0]
    assert evaluation["method"] == "holdout"
    assert evaluation["design_samples"] == [train.content_hash]
    assert evaluation["leakage_check"] == "passed"


def test_a_joint_design_is_checked_for_leakage() -> None:
    train, _ = holdout(_long_book())
    result = design_joint(_pair(), train, MSE())
    with pytest.raises(LeakageError):
        compare([result], train, split="out_of_sample")


def test_a_joint_design_in_sample_means_on_its_design_sample() -> None:
    train, test = holdout(_long_book())
    result = design_joint(_pair(), train, MSE())
    with pytest.raises(SampleError, match="in sample means"):
        compare([result], test, split="in_sample")


def test_a_set_of_designs_is_checked_asset_by_asset() -> None:
    train, test = holdout(_long_book())
    designs = {
        asset: design(_step(Free(), asset=asset), train.asset(asset), MSE())
        for asset in train.asset_ids
    }
    table = compare([designs], test, split="out_of_sample")
    rows = table.to_pylist()
    assert [row["premium_basis"] for row in rows] == ["design"] * 3
    evaluation = EvaluationMetadata.from_table(table).setup.evaluations[0]
    assert evaluation["method"] == "holdout"
    assert evaluation["leakage_check"] == "passed"
    with pytest.raises(LeakageError):
        compare([designs], train, split="out_of_sample")
    compare([designs], train, split="in_sample")
    with pytest.raises(SampleError, match="in sample means"):
        compare([designs], test, split="in_sample")


def test_the_parts_of_a_joined_sample_keep_their_lineage() -> None:
    def part(asset: str, *, lineage: str) -> OutcomeSample:
        return OutcomeSample(
            asset_ids=(asset,),
            period=np.arange(4, dtype=np.int64),
            probability=np.full(4, 0.25),
            index=np.array([0.0, 3.0, 0.0, 3.0]),
            loss=np.array([0.0, 8.0, 0.0, 4.0]),
            hazard=None,
            index_unit="m/s",
            loss_unit="EUR",
            semantics="annual",
            provenance=SampleProvenance(
                source_kind="table",
                sources=({"role": "table", "name": lineage},),
                semantics_source="caller",
                lineage=lineage,
            ),
        )

    a, b = part("A", lineage="source-a"), part("B", lineage="source-b")
    joined = OutcomeSample.join([a, b])
    designs = {
        asset: design(_step(Free(), asset=asset), each, MSE())
        for asset, each in (("A", a), ("B", b))
    }
    compare([designs], joined, split="in_sample")
    with pytest.raises(LeakageError):
        compare([designs], joined, split="out_of_sample")


def test_covers_given_as_is_take_a_design_sample_of_those_assets() -> None:
    train, test = holdout(_long_book())
    covers = {"A": _step(1.0, asset="A"), "B": _step(1.0, asset="B")}
    with pytest.raises(SampleError, match="needs the design sample"):
        evaluate(covers, test, split="out_of_sample")
    table = evaluate(covers, test, split="out_of_sample", design_sample=train)
    assert table.num_rows == 3
    with pytest.raises(SampleError, match="holds the assets"):
        evaluate(covers, test, split="out_of_sample", design_sample=train.asset("A"))


def test_a_design_sample_must_be_the_designs_own() -> None:
    train, test = holdout(_long_book())
    result = design_joint(_pair(), train, MSE())
    with pytest.raises(EvaluationError, match="differs from the design's own"):
        compare([result], test, split="out_of_sample", design_sample=test)
    designs = {
        asset: design(_step(Free(), asset=asset), train.asset(asset), MSE())
        for asset in train.asset_ids
    }
    with pytest.raises(EvaluationError, match="differs from the design's own"):
        compare([designs], test, split="out_of_sample", design_sample=test)
    compare([designs], test, split="out_of_sample", design_sample=train)


# -- errors ------------------------------------------------------------------


def test_one_cover_on_several_assets_names_the_ways_out() -> None:
    with pytest.raises(SampleError) as raised:
        evaluate(_step(20.0, asset="A"), BOOK, split="in_sample")
    message = str(raised.value)
    assert "asset(" in message and "mapping" in message and "design_joint" in message


def test_a_mapping_on_one_asset_raises() -> None:
    with pytest.raises(SampleError, match="several assets"):
        evaluate({"A": _step(20.0, asset="A")}, BOOK.asset("A"), split="in_sample")


def test_a_set_must_hold_exactly_the_samples_assets() -> None:
    with pytest.raises(SampleError, match="every asset"):
        evaluate({"A": FIXED["A"]}, BOOK, split="in_sample")
    with pytest.raises(SampleError, match="every asset"):
        evaluate({**FIXED, "C": _step(1.0)}, BOOK, split="in_sample")
    with pytest.raises(SampleError, match="every asset"):
        evaluate({}, BOOK, split="in_sample")


def test_a_cover_must_suit_its_asset_in_a_set() -> None:
    with pytest.raises(SampleError, match="written for asset 'B'"):
        evaluate({"A": FIXED["B"], "B": FIXED["B"]}, BOOK, split="in_sample")
    usd = Cover(INDEX, StepPayout([1.0], [1.0]), payout_unit="USD", asset_id="B")
    with pytest.raises(SampleError, match="USD"):
        evaluate({"A": FIXED["A"], "B": usd}, BOOK, split="in_sample")
    template = _step(Free(), asset="B")
    with pytest.raises(Exception, match="free parameters"):
        evaluate({"A": FIXED["A"], "B": template}, BOOK, split="in_sample")


def test_a_set_may_not_mix_covers_and_designs() -> None:
    designed = design(_step(Free(), asset="B"), BOOK.asset("B"), MSE())
    with pytest.raises(EvaluationError, match="not a mix"):
        evaluate({"A": FIXED["A"], "B": designed}, BOOK, split="in_sample")


def test_a_design_in_a_set_must_be_of_its_own_asset() -> None:
    other = design(_step(Free(), asset=None), BOOK.asset("A"), MSE())
    designs = {
        "A": design(_step(Free(), asset="A"), BOOK.asset("A"), MSE()),
        "B": other,
    }
    with pytest.raises(SampleError, match="asset 'B'"):
        compare([designs], BOOK, split="in_sample")


def test_a_joint_design_must_hold_the_samples_assets() -> None:
    result = _joint()
    other = _sample(
        [[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], asset_ids=("A", "C")
    )
    with pytest.raises(SampleError, match="joint design is for"):
        compare([result], other, split="out_of_sample")
    with pytest.raises(SampleError, match="joint design is for"):
        compare([result], BOOK.asset("A"), split="in_sample")


@pytest.mark.parametrize(
    ("allocation", "values", "message"),
    [
        ("area", VALUES, "allocation must be"),
        ("value", None, "needs values="),
        ("loss", None, "needs values="),
        (None, VALUES, "without allocation="),
        ("value", {"A": 1.0}, "every asset"),
        ("value", {**VALUES, "C": 1.0}, "every asset"),
        ("value", {"A": -1.0, "B": 1.0}, "'A'.*>= 0"),
        ("value", {"A": float("nan"), "B": 1.0}, "'A'.*finite"),
        ("value", {"A": float("inf"), "B": 1.0}, "'A'.*finite"),
        ("value", {"A": True, "B": 1.0}, "'A'.*number"),
        ("value", {"A": "1", "B": 1.0}, "'A'.*number"),
        ("value", {"A": 0.0, "B": 0.0}, "all zero"),
    ],
)
def test_invalid_allocations_raise(allocation: Any, values: Any, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        UNTYPED_COMPARE(
            [_joint()],
            BOOK,
            split="in_sample",
            allocation=allocation,
            values=values,
        )


def test_values_of_the_wrong_type_raise() -> None:
    with pytest.raises(TypeError):
        UNTYPED_COMPARE(
            [_joint()], BOOK, split="in_sample", allocation="value", values=[1.0, 2.0]
        )


def test_an_allocation_without_a_joint_design_raises() -> None:
    with pytest.raises(EvaluationError, match="no entry is a joint design"):
        UNTYPED_COMPARE(
            [FIXED], BOOK, split="in_sample", allocation="value", values=VALUES
        )
    with pytest.raises(EvaluationError, match="no entry is a joint design"):
        UNTYPED_EVALUATE(
            _step(20.0, asset="A"),
            BOOK.asset("A"),
            split="in_sample",
            allocation="value",
            values={"A": 1.0},
        )


def test_entries_of_the_wrong_type_raise() -> None:
    with pytest.raises(TypeError):
        UNTYPED_COMPARE(_joint(), BOOK, split="in_sample")
    with pytest.raises(TypeError):
        UNTYPED_COMPARE(FIXED, BOOK, split="in_sample")
    with pytest.raises(TypeError):
        UNTYPED_EVALUATE({"A": "cover", "B": "cover"}, BOOK, split="in_sample")
    with pytest.raises(TypeError):
        UNTYPED_EVALUATE({1: FIXED["A"], 2: FIXED["B"]}, BOOK, split="in_sample")


# -- the setup in words ------------------------------------------------------


def test_the_setup_states_the_total_and_the_rule() -> None:
    table = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    assert setup_text(table).splitlines()[0] == (
        "Level: total, the sum over A and B in each draw: the owner's total loss "
        "against the total payout; asset_by_value, one row per asset (A and B): its "
        "share of the joint design's total payout in proportion to its stated "
        "value, an accounting rule and not the payout of the asset's cover, with "
        "no premium or net-loss values. Scope: joint, one objective and one budget "
        "on the owner's total. Scenario: baseline. Basis: annual, one draw is one "
        "year."
    )


def test_the_setup_states_the_loss_rule() -> None:
    table = compare(
        [_joint()], BOOK, split="in_sample", allocation="loss", values=VALUES
    )
    line = setup_text(table).splitlines()[0]
    assert (
        "asset_by_loss, one row per asset (A and B): its share of the joint "
        "design's total payout in proportion to its loss in the draw, and to its "
        "stated value in a draw without loss" in line
    )


def test_the_setup_of_a_set_of_covers() -> None:
    text = setup_text(evaluate(FIXED, BOOK, split="in_sample"))
    assert text.splitlines()[0] == (
        "Level: asset, one row per asset (A and B); total, the sum over A and B in "
        "each draw: the owner's total loss against the total payout. Scope: "
        "individual, each cover standing on its own. Scenario: baseline. Basis: "
        "annual, one draw is one year."
    )
    assert (
        "The total row of a set of covers holds the sum of their premiums, one "
        "per contract, unless a total premium is stated." in text
    )


def test_the_setup_states_the_dependence() -> None:
    table = compare([_joint()], BOOK, split="in_sample")
    assert (
        "Dependence between the assets: the data's own, the rows being the same "
        "years of the data." in _setup_line(table, "Sample:")
    )


def _dependence_text(document: Mapping[str, Any]) -> str:
    sample = {
        "sample_hash": "sha256:" + "1" * 64,
        "roles": ["evaluation"],
        "source_kind": "monte_carlo",
        "asset_ids": ["A", "B"],
        "basis_risk": None,
        "dependence": document,
        "loss_model": None,
        "loss_unit": "EUR",
    }
    setup = EvaluationSetup(
        metric_options=MetricOptions().to_document(),
        samples=(sample,),
        evaluations=(),
    )
    return setup_text(setup)


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"kind": "independent"}, "Independent(), a stated assumption."),
        ({"kind": "comonotonic"}, "Comonotonic(), a stated assumption."),
        (
            {
                "kind": "empirical_copula",
                "content_hash": "sha256:" + "a" * 64,
                "m": 30,
                "first_year": 1990,
                "last_year": 2019,
                "sources": [],
                "licences": [],
                "k": 40,
            },
            "EmpiricalCopula, a stated assumption: the ranks of 30 reference years, "
            "1990 to 2019 (aaaaaaaaaaaa), 40 draws per year.",
        ),
        ({"kind": "other"}, "other, a stated assumption."),
    ],
    ids=["independent", "comonotonic", "copula", "unknown"],
)
def test_the_setup_names_the_dependence_model(
    document: Mapping[str, Any], expected: str
) -> None:
    assert f"Dependence between the assets: {expected}" in _dependence_text(document)


def test_the_setup_of_a_joined_sample_reads_the_assets_models() -> None:
    def part(asset: str, loss_model: Mapping[str, Any]) -> OutcomeSample:
        return OutcomeSample(
            asset_ids=(asset,),
            period=np.arange(4, dtype=np.int64),
            probability=np.full(4, 0.25),
            index=np.array([0.0, 3.0, 0.0, 3.0]),
            loss=np.array([0.0, 8.0, 0.0, 4.0]),
            hazard=None,
            index_unit="m/s",
            loss_unit="EUR",
            semantics="annual",
            provenance=SampleProvenance(
                source_kind="annual",
                sources=({"role": "hazard", "name": "series"},),
                semantics_source="caller",
                loss_model=loss_model,
                basis_risk={"kind": "no_basis_risk"},
                lineage=f"source-{asset}",
            ),
        )

    def model(name: str) -> dict[str, Any]:
        return {"impact": {"kind": "expression", "name": name}, "value": {}}

    joined = OutcomeSample.join([part("A", model("one")), part("B", model("two"))])
    line = _setup_line(evaluate(FIXED, joined, split="in_sample"), "Sample:")
    assert "Basis risk: no model." in line
    assert (
        "Loss: A: impact expression one, the damage ratio, in EUR; B: impact "
        "expression two, the damage ratio, in EUR." in line
    )
    same = OutcomeSample.join([part("A", model("one")), part("B", model("one"))])
    line = _setup_line(evaluate(FIXED, same, split="in_sample"), "Sample:")
    assert "Loss: impact expression one, the damage ratio, in EUR." in line


def test_the_setup_of_a_table_states_the_levels_of_its_rows() -> None:
    table = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    totals = table.filter(pc.equal(table["level"], "total"))
    line = setup_text(totals).splitlines()[0]
    assert line.startswith("Level: total, the sum over A and B")
    assert "asset_by_value" not in line


def test_a_total_sums_only_the_assets_of_its_own_evaluation() -> None:
    joint = EvaluationMetadata.from_table(compare([_joint()], BOOK, split="in_sample"))
    single = EvaluationMetadata.from_table(
        evaluate(_step(20.0, asset="A"), BOOK.asset("A"), split="in_sample")
    )
    line = setup_text(EvaluationMetadata.merge(joint, single)).splitlines()[0]
    assert "total, the sum over A and B in each draw" in line
    assert " or over " not in line


def test_a_comparison_at_several_levels_keeps_one_split_line() -> None:
    allocated = compare(
        [_joint()], BOOK, split="in_sample", allocation="value", values=VALUES
    )
    plain = evaluate(FIXED, BOOK, split="in_sample")
    metadata = EvaluationMetadata.merge(
        EvaluationMetadata.from_table(allocated),
        EvaluationMetadata.from_table(plain),
    )
    lines = setup_text(metadata).splitlines()
    assert sum(line.startswith("Split: ") for line in lines) == 1
    assert lines[0].count("Scope: individual and joint") == 1


RARER = "draws are rarer than the curve's rarest fitted return period."
FREQUENT_SIDE = "draws are more frequent than the curve's most frequent one."
ASSUMPTION = "The dependence between the assets is an assumption: they are independent."


def _simulated(
    notes: list[str], *, assets: list[str], digit: str = "1", roles: Any = None
) -> dict[str, Any]:
    """A setup sample of several assets, as a sample of curves records it."""
    return {
        "asset_ids": assets,
        "basis_risk": None,
        "dependence": {"kind": "independent"},
        "draws": 400,
        "lineage": f"lineage-{digit}",
        "loss_model": None,
        "loss_unit": "EUR",
        "notes": notes,
        "roles": roles or ["design", "evaluation"],
        "sample_hash": "sha256:" + digit * 64,
        "source_kind": "monte_carlo",
    }


def _notes_line(*samples: dict[str, Any], evaluations: Any = ()) -> str:
    setup = {
        "basis": "annual",
        "metric_options": MetricOptions().to_document(),
        "samples": list(samples),
        "evaluations": list(evaluations),
    }
    lines = setup_text(setup).splitlines()
    return next(line for line in lines if line.startswith("Sample notes: "))


def test_counted_notes_of_each_asset_are_written_once() -> None:
    line = _notes_line(
        _simulated(
            [
                ASSUMPTION,
                f"Asset a: 3 of 400 {RARER}",
                f"Asset a: 7 of 400 {FREQUENT_SIDE}",
                f"Asset b: 5 of 400 {RARER}",
            ],
            assets=["a", "b"],
        )
    )
    assert line == (
        f"Sample notes: {ASSUMPTION} 3 of 400 draws in the design sample for a and "
        "5 of 400 in the design sample for b are rarer than the curve's rarest "
        "fitted return period. 7 of 400 draws in the design sample for a are more "
        "frequent than the curve's most frequent one."
    )


def test_a_counted_note_of_one_asset_takes_a_singular_verb() -> None:
    line = _notes_line(
        _simulated(
            ["Asset a: 1 of 400 draws is rarer than the curve's last."],
            assets=["a", "b"],
        )
    )
    assert line == (
        "Sample notes: 1 of 400 draws in the design sample for a is rarer than "
        "the curve's last."
    )


def test_equal_counts_of_each_asset_keep_their_assets() -> None:
    line = _notes_line(
        _simulated(
            [f"Asset a: 3 of 400 {RARER}", f"Asset b: 3 of 400 {RARER}"],
            assets=["a", "b"],
        )
    )
    assert "3 of 400 draws in the design sample for a and 3 of 400" in line


def test_a_large_book_states_the_range_of_each_notes_counts() -> None:
    assets = [f"site-{n}" for n in range(8)]
    notes = [ASSUMPTION]
    notes += [f"Asset {a}: {n + 2} of 400 {RARER}" for n, a in enumerate(assets)]
    notes += [f"Asset {a}: 6 of 400 {FREQUENT_SIDE}" for a in assets]
    line = _notes_line(_simulated(notes, assets=assets))
    assert line == (
        f"Sample notes: {ASSUMPTION} 2 to 9 of 400 draws in the design sample for "
        "each of 8 assets are rarer than the curve's rarest fitted return period. "
        "6 of 400 draws in the design sample for each of 8 assets are more frequent "
        "than the curve's most frequent one."
    )


def test_a_large_book_states_each_samples_range() -> None:
    assets = [f"site-{n}" for n in range(6)]
    design_notes = [f"Asset {a}: {n + 1} of 400 {RARER}" for n, a in enumerate(assets)]
    fresh_notes = [f"Asset {a}: {n + 3} of 400 {RARER}" for n, a in enumerate(assets)]
    line = _notes_line(
        _simulated(design_notes, assets=assets, digit="1", roles=["design"]),
        _simulated(fresh_notes, assets=assets, digit="2", roles=["evaluation"]),
        evaluations=[
            {
                "sample_hash": "sha256:" + "2" * 64,
                "split": "out_of_sample",
                "fold": 0,
                "method": "fresh_draws",
            }
        ],
    )
    assert line == (
        "Sample notes: 1 to 6 of 400 draws in the design sample for each of 6 "
        "assets and 3 to 8 of 400 in the fresh draws for each of 6 assets are "
        "rarer than the curve's rarest fitted return period."
    )


def test_a_note_that_assets_share_is_written_once() -> None:
    line = _notes_line(
        _simulated(
            ["Asset a: One note.", "Asset b: One note.", "Asset b: Another."],
            assets=["a", "b"],
        )
    )
    assert line == "Sample notes: Assets a and b: One note. Asset b: Another."
    assets = [f"site-{n}" for n in range(6)]
    line = _notes_line(_simulated([f"Asset {a}: One." for a in assets], assets=assets))
    assert line == "Sample notes: 6 assets: One."


def test_an_asset_id_may_hold_a_colon_in_a_note() -> None:
    line = _notes_line(
        _simulated(
            [f"Asset a: b: 3 of 400 {RARER}", f"Asset c: 5 of 400 {RARER}"],
            assets=["a: b", "c"],
        )
    )
    assert "3 of 400 draws in the design sample for a: b and 5 of 400" in line


def test_a_note_naming_no_asset_of_the_sample_stays_as_it_is() -> None:
    line = _notes_line(_simulated([f"Asset z: 3 of 400 {RARER}"], assets=["a", "b"]))
    assert line == f"Sample notes: Asset z: 3 of 400 {RARER}"


def test_the_notes_of_a_curve_sample_of_several_assets() -> None:
    # The shape sample_curves records: the model's sentence first, then one
    # note per asset, and the document of the model in `dependence`.
    copula = {
        "kind": "empirical_copula",
        "content_hash": "sha256:" + "b" * 64,
        "sources": [{"role": "hazard", "name": "era5"}],
        "licences": [{"role": "hazard", "licence": "CC-BY-4.0"}],
        "first_year": 1991,
        "last_year": 2020,
        "m": 30,
        "k": 2,
    }
    sentence = "The dependence between the assets is an assumption: their draws."
    sample = OutcomeSample(
        asset_ids=("A", "B"),
        period=np.arange(5, dtype=np.int64),
        probability=np.full(5, 0.2),
        index=BOOK.index,
        loss=BOOK.loss,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="monte_carlo",
            sources=({"role": "hazard", "name": "curve"},),
            semantics_source="dataset",
            probability_semantics="annual",
            notes=(
                sentence,
                "Asset A: 1 of 5 draws is rarer than the last.",
                "Asset B: 2 of 5 draws are rarer than the last.",
            ),
            lineage="curves",
            dependence=copula,
        ),
    )
    table = evaluate(FIXED, sample, split="in_sample")
    assert _setup_line(table, "Sample notes:") == (
        f"Sample notes: {sentence} 1 of 5 draws in the evaluation sample for A and "
        "2 of 5 in the evaluation sample for B are rarer than the last."
    )
    assert (
        "Dependence between the assets: EmpiricalCopula, a stated assumption: the "
        "ranks of 30 reference years, 1991 to 2020 (bbbbbbbbbbbb), 2 draws per year."
        in _setup_line(table, "Sample:")
    )


def test_the_setup_of_a_joint_design() -> None:
    sample = _sample(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
        [[0.0, 0.0], [40.0, 0.0], [0.0, 20.0], [0.0, 0.0]],
    )
    result = design_joint(
        [_step(Free(), asset="A"), _step(Free(), asset="B")],
        sample,
        ExpectedUncovered(),
        budget=Budget(ratio=0.5),
        options=DesignOptions(ranges=True),
    )
    table = compare([result], sample, split="in_sample")
    text = setup_text(table)
    line = _setup_line(table, "Design ")
    assert (
        "scope: joint, one objective, one budget and one premium on the total of "
        "A and B, with one cover per asset; objective: E[U]" in line
    )
    assert (
        "budget: premium at most 0.5 x E[L] of the total of the design sample" in line
    )
    assert f"design premium: {result.premium:g} EUR on the total" in line
    assert "unique optimum: no" in line
    assert "the amounts of each asset are one member of an optimal set" in line
    assert "A joint design's premium is that of one contract on the total." in text


def test_the_setup_of_a_joint_design_names_unidentified_amounts_by_asset() -> None:
    # B's cover never pays, so its level is unidentified.
    sample = _sample(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 30.0], [0.0, 0.0], [0.0, 0.0]],
    )
    result = design_joint(
        [_step(Free(), asset="A"), _step(Free(), asset="B")],
        sample,
        ExpectedUncovered(),
        budget=Budget(amount=7.5),
    )
    line = _setup_line(compare([result], sample, split="in_sample"), "Design ")
    assert "amounts no design year reaches: B levels[0]" in line
    assert "budget: premium at most 7.5 EUR on the total" in line


def test_a_document_without_the_new_fields_reads_as_rows_of_single_assets() -> None:
    table = evaluate(_step(20.0, asset="A"), BOOK.asset("A"), split="in_sample")
    metadata = EvaluationMetadata.from_table(table)
    document = json.loads(metadata.to_json())
    for evaluation in document["setup"]["evaluations"]:
        for key in ("level", "scope", "allocation"):
            del evaluation[key]
    for sample in document["setup"]["samples"]:
        for key in ("dependence", "parts"):
            del sample[key]
    assert setup_text(document).startswith("Level: asset (A). Scenario: baseline.")
    assert setup_text(document) == setup_text(metadata)


def test_the_setup_of_rows_of_single_assets_states_no_scope() -> None:
    table = evaluate(_step(20.0, asset="A"), BOOK.asset("A"), split="in_sample")
    assert "Scope:" not in setup_text(table)
    assert "Dependence" not in setup_text(table)
    evaluation = EvaluationMetadata.from_table(table).setup.evaluations[0]
    assert evaluation["level"] == ["asset"]
    assert evaluation["scope"] == ["individual"]
    assert evaluation["allocation"] is None
    assert table.to_pylist()[0]["scope"] == "individual"
