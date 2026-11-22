"""Designs under `Clustered`: parity with the golden fixtures' cluster cases.

The cases are read as JSON with plain numpy, as
`tests/fixtures/parametric/README.md` describes; no reference code is
imported. Every design case of the cluster scope in the folder is judged: its
design problems (one under `joint` true, one per cluster under `joint` false),
then the rows of `evaluate`: the total, each cluster's own row under `joint`
false, and the allocated rows under both rules, each cluster's payout shared
over its own assets. Under `joint` true the module shows no row per cluster,
so a case's cluster blocks are checked on each cluster's cover evaluated on its
own loss.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crc_sdk.parametric import (
    ClusterDesign,
    Clustered,
    ClusteredDesignResult,
    ClusterIndex,
    Cover,
    DesignOptions,
    Free,
    OutcomeSample,
    SampleProvenance,
    design_joint,
    evaluate,
)
from crc_sdk.parametric.constraints import budget_from_document, limit_from_document
from crc_sdk.parametric.evaluate import Allocation
from crc_sdk.parametric.joint import covers_hash
from crc_sdk.parametric.metrics import MetricOptions
from crc_sdk.parametric.objectives import objective_from_document
from crc_sdk.parametric.pricing import pricing_from_document

CASES = Path(__file__).parent / "fixtures" / "parametric" / "cases"

UNITLESS = (
    "coverage_ratio",
    "payout_to_loss_ratio",
    "correlation",
    "payout_frequency",
    "unidentified_share",
    "tail_coverage_",
    "cvar_uncovered_to_loss_",
    "false_negative_rate_",
    "false_positive_rate_",
)
NET_LOSS = ("premium", "var_net_loss_", "cvar_net_loss_", "net_loss_rp")


def _case(path: Path) -> dict[str, Any]:
    case: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return case


def _cluster_cases() -> list[Path]:
    return [
        path
        for path in sorted(CASES.glob("*.json"))
        if _case(path)["scope"] == "cluster"
    ]


CLUSTER_CASES = _cluster_cases()


def _sample(case: Mapping[str, Any]) -> OutcomeSample:
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


def _scope(case: Mapping[str, Any]) -> Clustered:
    clusters = case["clusters"]
    return Clustered(
        clusters["assignment"], values=clusters["values"], joint=clusters["joint"]
    )


def _templates(case: Mapping[str, Any]) -> list[Cover]:
    return [Cover.from_document(document) for document in case["cover"]]


def _design(case: Mapping[str, Any]) -> ClusteredDesignResult:
    return design_joint(
        _templates(case),
        _sample(case),
        objective_from_document(case["objective"]),
        pricing=pricing_from_document(case["pricing"]),
        budget=budget_from_document(case["budget"]),
        limit=limit_from_document(case["limit"]),
        options=DesignOptions(ranges=True),
        scope=_scope(case),
    )


def _scale(case: Mapping[str, Any]) -> float:
    return float(max(sum(row) for row in case["sample"]["loss"]))


def _close(actual: float, target: float, relative: float, floor: float) -> bool:
    return abs(actual - target) <= relative * abs(target) + floor


def _check_metrics(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerance: float,
    scale: float,
    where: str,
) -> None:
    """Every column of `expected` against the row, as the README compares them."""
    for name, value in expected.items():
        actual = row[name]
        if value is None:
            assert actual is None, (where, name, actual)
            continue
        assert actual is not None, (where, name)
        floor = 1e-9 if name.startswith(UNITLESS) else 1e-9 * scale
        if isinstance(value, Mapping):
            assert any(
                _close(actual, either, tolerance, floor) for either in value["either"]
            ), (where, name, actual, value)
        else:
            assert _close(actual, value, tolerance, floor), (where, name, actual, value)


def _with_expected(template: Cover, values: Mapping[str, float]) -> Cover:
    """The template resolved with a case's parameters, within their bounds.

    The cases write 12 significant digits, so a level at its cap, the
    cluster's largest loss, can be written a hair above it.
    """
    parameters = template.payout.parameters()
    bounded: dict[str, float] = {}
    for name, value in values.items():
        free = parameters[name]
        assert isinstance(free, Free) and free.upper is not None
        assert free.lower <= value <= free.upper * (1.0 + 1e-11), (name, value)
        bounded[name] = min(value, free.upper)
    return template.with_values(bounded)


def _index_of(cover: Cover) -> ClusterIndex:
    index = cover.cluster_index
    assert index is not None
    return index


def _resolved(
    case: Mapping[str, Any], result: ClusteredDesignResult
) -> ClusteredDesignResult:
    """The result with the case's own optimum in place of the module's.

    A non-unique design's premium and metrics depend on which member of the
    optimal set a solver returns, so the rows are compared on the templates
    resolved with `expected.parameters`: each cluster's template on the
    module's cluster index, and the case's premiums.
    """
    expected = case["expected"]
    covers = {
        cluster: _with_expected(
            Cover(
                _index_of(result.covers[cluster]),
                template.payout,
                payout_unit=template.payout_unit,
                asset_id=cluster,
            ),
            expected["parameters"][cluster],
        )
        for cluster, template in zip(result.cluster_ids, _templates(case), strict=True)
    }
    clusters = (
        None
        if result.clusters is None
        else {
            cluster: dataclasses.replace(
                design, premium=expected["clusters"][cluster]["premium"]
            )
            for cluster, design in result.clusters.items()
        }
    )
    return dataclasses.replace(
        result,
        covers=covers,
        covers_hash=covers_hash(covers),
        premium=expected["premium"],
        clusters=clusters,
    )


def test_the_cluster_cases_are_present() -> None:
    names = {path.stem for path in CLUSTER_CASES}
    assert names == {
        f"step-{objective}-{pricing}-cluster-{setting}-{data}"
        for objective in ("mse", "eu", "wcvar-b0_9")
        for pricing in ("pure", "coc")
        for setting in ("joint", "separate")
        for data in ("synthetic", "thesis")
    }
    cases = [_case(path) for path in CLUSTER_CASES]
    modes = [case["compare_parameters"] for case in cases]
    assert (modes.count("exact"), modes.count("within_ranges")) == (19, 5)
    for case in cases:
        assert case["kind"] == "design" and case["split"] is None
        clusters = case["clusters"]
        assert clusters["joint"] is case["case_id"].endswith(
            ("-cluster-joint-synthetic", "-cluster-joint-thesis")
        )
        assert len(case["cover"]) == len(clusters["cluster_ids"]) == 2
        # The assignment states C1 first, so the module's order of the
        # clusters, the assignment's, is the cases' `cluster_ids`.
        assert _scope(case).cluster_ids == tuple(clusters["cluster_ids"])


@pytest.mark.parametrize("path", CLUSTER_CASES, ids=lambda path: path.stem)
def test_cluster_design_parity(path: Path) -> None:
    case = _case(path)
    expected = case["expected"]
    tolerances = case["tolerances"]
    clusters = case["clusters"]
    scale = _scale(case)
    sample = _sample(case)
    result = _design(case)
    assert result.cluster_ids == tuple(clusters["cluster_ids"])
    assert result.asset_ids == sample.asset_ids
    assert result.joint is clusters["joint"]

    # The index: each cover on its cluster's members and weights.
    for column, cluster in enumerate(result.cluster_ids):
        cover = result.covers[cluster]
        index = cover.cluster_index
        assert isinstance(index, ClusterIndex)
        assert cover.asset_id == cluster
        assert index.members == tuple(clusters["members"][cluster])
        assert dict(index.weights) == clusters["weights"][cluster]
        values = np.asarray(sample.index)[
            :, [sample.asset_ids.index(asset) for asset in index.members]
        ] @ np.array(list(index.weights.values()))
        np.testing.assert_allclose(
            values, np.asarray(clusters["index"])[:, column], rtol=1e-12, atol=0.0
        )

    power = 2 if case["objective"]["kind"] == "mse" else 1
    floor = 1e-9 * scale**power
    allowance = tolerances["parameters"] * scale
    # Each design problem's objective, uniqueness and parameter mode: the whole
    # design's under `joint` true, each cluster's under `joint` false.
    problems: dict[str, tuple[float, bool, str]]
    if clusters["joint"]:
        assert result.clusters is None
        assert result.objective_value is not None
        assert _close(
            result.objective_value,
            expected["objective_value"],
            tolerances["objective_value"],
            floor,
        )
        problems = {
            cluster: (math.nan, expected["unique"], case["compare_parameters"])
            for cluster in result.cluster_ids
        }
        if expected["unique"] is False:
            assert result.unique is not True
    else:
        assert result.objective_value is None and expected["objective_value"] is None
        assert result.clusters is not None
        problems = {}
        for cluster, block in expected["clusters"].items():
            own = result.clusters[cluster]
            assert isinstance(own, ClusterDesign)
            assert _close(
                own.objective_value,
                block["objective_value"],
                tolerances["objective_value"],
                floor,
            ), (cluster, own.objective_value, block["objective_value"])
            if block["unique"] is False:
                assert own.unique is not True, cluster
            problems[cluster] = (
                own.objective_value,
                block["unique"],
                block["compare_parameters"],
            )
        if expected["unique"] is False:
            assert result.unique is not True

    for cluster, values in expected["parameters"].items():
        unidentified = expected["unidentified"][cluster]
        assert set(result.unidentified[cluster]) == set(unidentified), cluster
        mode = problems[cluster][2]
        for name, value in values.items():
            if name in unidentified:
                continue
            if mode == "exact":
                actual = result.parameters[cluster][name]
                assert abs(actual - value) <= allowance, (cluster, name, actual, value)
            else:
                assert mode == "within_ranges"
                assert result.ranges is not None
                low, high = result.ranges[cluster][name]
                assert low - allowance <= value <= high + allowance, (cluster, name)

    # The premium: the design premium on the total under `joint` true, the sum
    # of the clusters' premiums under `joint` false.
    if expected["unique"]:
        assert _close(
            result.premium, expected["premium"], tolerances["premium"], 1e-9 * scale
        )
    if result.clusters is not None:
        assert result.premium == pytest.approx(
            math.fsum(own.premium for own in result.clusters.values()), rel=1e-15
        )
        for cluster, block in expected["clusters"].items():
            if block["unique"]:
                assert _close(
                    result.clusters[cluster].premium,
                    block["premium"],
                    tolerances["premium"],
                    1e-9 * scale,
                ), cluster


@pytest.mark.parametrize("path", CLUSTER_CASES, ids=lambda path: path.stem)
@pytest.mark.parametrize("rule", ["value", "loss"])
def test_cluster_rows_parity(path: Path, rule: Allocation) -> None:
    case = _case(path)
    expected = case["expected"]
    clusters = case["clusters"]
    tolerance = case["tolerances"]["metrics"]
    scale = _scale(case)
    sample = _sample(case)
    result = _design(case)
    unique = expected["unique"]
    entry = result if unique else _resolved(case, result)
    table = evaluate(
        entry,
        sample,
        split="in_sample",
        premium=None if unique or not clusters["joint"] else expected["premium"],
        allocation=rule,
        values=clusters["values"],
    )
    rows = table.to_pylist()
    level = {"value": "asset_by_value", "loss": "asset_by_loss"}[rule]
    cluster_rows = [] if clusters["joint"] else list(result.cluster_ids)
    assert [(row["level"], row["cluster_id"], row["asset_id"]) for row in rows] == [
        *(("cluster", cluster, None) for cluster in cluster_rows),
        ("total", None, None),
        *((level, None, asset) for asset in sample.asset_ids),
    ]
    assert {row["scope"] for row in rows} == {"cluster"}
    by_level = {
        (row["level"], row["cluster_id"] or row["asset_id"]): row for row in rows
    }
    _check_metrics(
        by_level[("total", None)], expected["metrics"], tolerance, scale, "total"
    )
    assert by_level[("total", None)]["premium_basis"] == (
        "design" if unique or not clusters["joint"] else "stated"
    )
    for cluster in cluster_rows:
        row = by_level[("cluster", cluster)]
        assert row["premium_basis"] == "design"
        _check_metrics(
            row, expected["clusters"][cluster]["metrics"], tolerance, scale, cluster
        )
    for asset in sample.asset_ids:
        row = by_level[(level, asset)]
        assert row["premium_basis"] is None
        _check_metrics(
            row, expected["allocation"][rule][asset], tolerance, scale, asset
        )


def _one_cluster(
    case: Mapping[str, Any], cluster: str, column: int
) -> tuple[OutcomeSample, Cover]:
    """A cluster's own sample, its index and loss, and its cover on that index.

    The case's `clusters.index` is the core's index, and a cluster's loss the
    sum of its assets' losses.
    """
    clusters = case["clusters"]
    sample = _sample(case)
    members = [sample.asset_ids.index(asset) for asset in clusters["members"][cluster]]
    own = OutcomeSample(
        asset_ids=(cluster,),
        period=sample.period,
        probability=sample.probability,
        index=np.asarray(clusters["index"], dtype=np.float64)[:, [column]],
        loss=np.asarray(sample.loss)[:, members].sum(axis=1, keepdims=True),
        hazard=None,
        index_unit=sample.index_unit,
        loss_unit=sample.loss_unit,
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": f"{case['case_id']}-{cluster}"},),
            semantics_source="caller",
        ),
    )
    template = _templates(case)[column]
    return own, _with_expected(template, case["expected"]["parameters"][cluster])


@pytest.mark.parametrize(
    "path",
    [path for path in CLUSTER_CASES if _case(path)["clusters"]["joint"]],
    ids=lambda path: path.stem,
)
def test_a_joint_cases_cluster_blocks_are_each_cover_on_its_own_loss(
    path: Path,
) -> None:
    """Under `joint` true the module has no row per cluster (the owner's
    decision 1), so a case's cluster blocks, the core's entity rows, check each
    cover's payout against its cluster's loss. They hold no premium."""
    case = _case(path)
    scale = _scale(case)
    for column, cluster in enumerate(case["clusters"]["cluster_ids"]):
        block = case["expected"]["clusters"][cluster]
        assert block["premium"] is None and block["objective_value"] is None
        own, cover = _one_cluster(case, cluster, column)
        row = evaluate(
            cover, own, split="in_sample", options=MetricOptions()
        ).to_pylist()[0]
        _check_metrics(
            row,
            {
                name: value
                for name, value in block["metrics"].items()
                if not name.startswith(NET_LOSS)
            },
            case["tolerances"]["metrics"],
            scale,
            cluster,
        )
        assert all(
            value is None
            for name, value in block["metrics"].items()
            if name.startswith(NET_LOSS)
        )


@pytest.mark.parametrize("path", CLUSTER_CASES, ids=lambda path: path.stem)
def test_cluster_cases_follow_the_owners_rules(path: Path) -> None:
    """Where the cases depart from the core, the module's rows follow them.

    Under `joint` false the total's premium is the sum of the clusters'
    premiums; under both settings the allocated payouts of a cluster's assets
    add up to its cluster's payout, whatever the rule.
    """
    case = _case(path)
    clusters = case["clusters"]
    sample = _sample(case)
    result = _design(case)
    for rule in ("value", "loss"):
        rows = evaluate(
            result,
            sample,
            split="in_sample",
            allocation=rule,
            values=clusters["values"],
        ).to_pylist()
        total = next(row for row in rows if row["level"] == "total")
        shares = {row["asset_id"]: row for row in rows if row["asset_id"] is not None}
        assert math.fsum(row["expected_payout"] for row in shares.values()) == (
            pytest.approx(total["expected_payout"], rel=1e-12)
        )
        if not clusters["joint"]:
            own = {row["cluster_id"]: row for row in rows if row["level"] == "cluster"}
            assert total["premium"] == pytest.approx(
                math.fsum(row["premium"] for row in own.values()), rel=1e-15
            )
            for cluster, members in clusters["members"].items():
                assert math.fsum(
                    shares[asset]["expected_payout"] for asset in members
                ) == pytest.approx(own[cluster]["expected_payout"], rel=1e-12)
