"""The cluster scope: `Clustered`, `ClusterIndex`, its covers and its rows.

Synthetic numbers, offline. The parity with the golden fixtures is in
`tests/test_parametric_cluster.py`; this file checks the API, its records and
its refusals. Nothing here is a finding about a cover or a design, and nothing
sets a cluster design against another scope.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    ClusterDesign,
    Clustered,
    ClusteredDesignResult,
    ClusterIndex,
    CompositeIndex,
    Cover,
    CoverDefinitionError,
    CoverPackage,
    DesignError,
    DesignOptions,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexDefinition,
    Joint,
    LinearPayout,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    UserObjective,
    WeightedCVaR,
    allocate,
    allocate_scenarios,
    compare,
    compare_pooled,
    design,
    design_joint,
    evaluate,
    holdout,
    payout_breakdown,
    setup_text,
    to_long,
    write_comparison,
)
from crc_sdk.parametric.constraints import Budget, Limit
from crc_sdk.parametric.cover import CLUSTER_COVER_SCHEMA_VERSION
from crc_sdk.parametric.joint import _cluster_index_values
from crc_sdk.parametric.payouts import SumOf

UNTYPED: Any = design_joint
UNTYPED_CLUSTERED: Any = Clustered
UNTYPED_ALLOCATE: Any = allocate
UNTYPED_POOLED: Any = compare_pooled
UNTYPED_SCENARIOS: Any = allocate_scenarios
WIND = IndexDefinition("wind", "m/s", aggregation="annual_maximum")
ASSIGNMENT = {"A": "north", "B": "north", "C": "south"}
VALUES = {"A": 3.0e6, "B": 1.0e6, "C": 2.0e6}


def _sample(n: int = 40, *, dependence: dict[str, Any] | None = None) -> OutcomeSample:
    """Three assets over `n` years of a seeded synthetic wind and its losses."""
    rng = np.random.default_rng(20261122)
    index = rng.gumbel(30.0, 6.0, size=(n, 3))
    loss = np.where(index > 33.0, (index - 33.0) * 1e5, 0.0) * [1.0, 0.5, 0.8]
    return OutcomeSample(
        asset_ids=("A", "B", "C"),
        period=np.arange(1980, 1980 + n),
        probability=np.full(n, 1.0 / n),
        index=index,
        loss=loss,
        hazard=None,
        index_unit="m/s",
        loss_unit="EUR",
        semantics="annual",
        provenance=SampleProvenance(
            source_kind="table",
            sources=({"role": "table", "name": "synthetic book"},),
            semantics_source="caller",
            licences=({"role": "table", "licence": "CC0-1.0"},),
            dependence=dependence or {"kind": "data"},
        ),
    )


def _index_of(cover: Cover) -> ClusterIndex:
    index = cover.cluster_index
    assert index is not None
    return index


def _template(label: str | None = "step", *, asset: str | None = None) -> Cover:
    return Cover(
        WIND,
        StepPayout([33.0, 38.0, 45.0], [Free(), Free(), Free()]),
        payout_unit="EUR",
        asset_id=asset,
        label=label,
    )


def _scope(joint: bool = True, **changes: Any) -> Clustered:
    return Clustered(
        changes.get("assignment", ASSIGNMENT),
        values=changes.get("values", VALUES),
        joint=joint,
    )


def _design(
    joint: bool = True, sample: OutcomeSample | None = None, **terms: Any
) -> ClusteredDesignResult:
    return design_joint(
        [_template(), _template()],
        _sample() if sample is None else sample,
        terms.pop("objective", ExpectedUncovered()),
        budget=terms.pop("budget", Budget(ratio=0.8)),
        scope=_scope(joint),
        **terms,
    )


# -- Clustered -------------------------------------------------------------------


def test_clustered_takes_the_callers_labels_in_the_assignments_order() -> None:
    scope = Clustered(
        {"C": "south", "A": "north", "B": "north"}, values=VALUES, joint=False
    )
    assert scope.cluster_ids == ("south", "north")
    assert dict(scope.members) == {"south": ("C",), "north": ("A", "B")}
    assert {key: dict(value) for key, value in scope.weights.items()} == {
        "south": {"C": 1.0},
        "north": {"A": 0.75, "B": 0.25},
    }
    assert scope.to_document() == {
        "assignment": {"C": "south", "A": "north", "B": "north"},
        "joint": False,
        "kind": "clustered",
    }
    # The values weight the index but are not part of the document.
    assert "values" not in json.dumps(scope.to_document())
    assert scope.index("north", WIND) == ClusterIndex(WIND, {"A": 0.75, "B": 0.25})
    assert repr(scope) == (
        "Clustered({'C': 'south', 'A': 'north', 'B': 'north'}, joint=False)"
    )


def test_clustered_compares_by_assignment_values_and_setting() -> None:
    scope = _scope()
    assert scope == _scope() and hash(scope) == hash(_scope())
    assert scope != _scope(joint=False)
    assert scope != _scope(values={**VALUES, "C": 1.0})
    assert scope != Clustered({"B": "north", "A": "north", "C": "south"}, values=VALUES)
    assert scope != Joint()
    assert pickle.loads(pickle.dumps(scope)) == scope


@pytest.mark.parametrize(
    ("assignment", "values", "joint", "error", "match"),
    [
        ("A", VALUES, True, TypeError, "assignment must map"),
        (ASSIGNMENT, ["A"], True, TypeError, "values must map"),
        (ASSIGNMENT, VALUES, 1, TypeError, "joint must be a bool"),
        ({}, VALUES, True, SampleError, "an assignment of one asset or more"),
        ({"A": " north"}, {"A": 1.0}, True, SampleError, "a cluster id must be"),
        ({1: "north"}, {1: 1.0}, True, SampleError, "an asset id must be"),
        (ASSIGNMENT, {"A": 1.0, "B": 1.0}, True, SampleError, r"lacks \['C'\]"),
        (
            ASSIGNMENT,
            {**VALUES, "D": 1.0},
            True,
            SampleError,
            r"names \['D'\]",
        ),
        (ASSIGNMENT, {**VALUES, "A": -1.0}, True, SampleError, "finite number >= 0"),
        (ASSIGNMENT, {**VALUES, "A": math.nan}, True, SampleError, "finite"),
        (ASSIGNMENT, {**VALUES, "A": True}, True, SampleError, "finite"),
        (
            ASSIGNMENT,
            {"A": 0.0, "B": 0.0, "C": 1.0},
            True,
            SampleError,
            "cluster 'north'.*all zero",
        ),
    ],
)
def test_clustered_refuses_an_invalid_assignment_or_values(
    assignment: Any, values: Any, joint: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        UNTYPED_CLUSTERED(assignment, values=values, joint=joint)


def test_clustered_needs_the_values() -> None:
    with pytest.raises(TypeError, match="values"):
        UNTYPED_CLUSTERED(ASSIGNMENT)


# -- ClusterIndex and the cluster's cover ------------------------------------------


def test_a_cluster_index_names_its_members_and_weights() -> None:
    index = ClusterIndex(WIND, {"A": 0.75, "B": 0.25})
    assert index.members == ("A", "B")
    assert (index.tail, index.unit, index.hazard_name) == ("upper", "m/s", "wind")
    document = index.to_document()
    assert document == {
        "index": WIND.to_document(),
        "kind": "cluster",
        "members": [
            {"asset_id": "A", "weight": 0.75},
            {"asset_id": "B", "weight": 0.25},
        ],
    }
    assert ClusterIndex.from_document(json.loads(json.dumps(document))) == index
    assert index != ClusterIndex(WIND, {"B": 0.25, "A": 0.75})
    assert index != WIND
    assert hash(index) == hash(ClusterIndex(WIND, {"A": 0.75, "B": 0.25}))
    assert pickle.loads(pickle.dumps(index)) == index
    assert ClusterIndex(WIND, {"C": 1}).weights["C"] == 1.0


@pytest.mark.parametrize(
    ("index", "weights", "match"),
    [
        (CompositeIndex({"wind": WIND, "rain": WIND}), {"A": 1.0}, "IndexDefinition"),
        (WIND, [("A", 1.0)], "weights must map"),
        (WIND, {}, "one member or more"),
        (WIND, {"": 1.0}, "asset id"),
        (WIND, {"A": -0.5, "B": 1.5}, "finite number >= 0"),
        (WIND, {"A": True}, "finite number >= 0"),
        (WIND, {"A": 0.5, "B": 0.4}, "sum to 1"),
    ],
)
def test_a_cluster_index_refuses_invalid_terms(
    index: Any, weights: Any, match: str
) -> None:
    with pytest.raises(CoverDefinitionError, match=match):
        ClusterIndex(index, weights)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"kind": "composite"}, "kind 'cluster'"),
        ({"extra": 1}, "has the keys"),
        ({"members": {"A": 1.0}}, "members is a list"),
        ({"members": [{"asset_id": "A"}]}, "each member"),
        (
            {
                "members": [
                    {"asset_id": "A", "weight": 0.5},
                    {"asset_id": "A", "weight": 0.5},
                ]
            },
            "repeats the member 'A'",
        ),
    ],
)
def test_a_cluster_index_document_is_checked(change: Any, match: str) -> None:
    document = {**ClusterIndex(WIND, {"A": 1.0}).to_document(), **change}
    with pytest.raises(CoverDefinitionError, match=match):
        ClusterIndex.from_document(document)


def test_a_clusters_cover_round_trips_and_reads_no_one_asset() -> None:
    index = ClusterIndex(WIND, {"A": 0.75, "B": 0.25})
    template = Cover(
        index,
        StepPayout([33.0, 38.0], [Free(), Free()]),
        payout_unit="EUR",
        asset_id="north",
        label="north step",
    )
    cover = template.with_values({"levels[0]": 1e5, "levels[1]": 2e5})
    document = cover.to_document()
    assert document["schema_version"] == CLUSTER_COVER_SCHEMA_VERSION == "1.2"
    assert document["index"] == index.to_document()
    assert document["asset_id"] == "north"
    again = Cover.from_json(cover.to_json(), label="north step")
    assert again == cover and again.cluster_index == index
    assert (
        cover.cluster_index == index
        and Cover(WIND, cover.payout, payout_unit="EUR").cluster_index is None
    )
    for read in ("index", "index_definition"):
        with pytest.raises(CoverDefinitionError, match="cluster_index"):
            getattr(cover, read)
    np.testing.assert_array_equal(
        cover.payout_values([30.0, 35.0, 40.0]), [0.0, 1e5, 2e5]
    )
    text = cover.terms_text()
    assert "Cluster: north" in text and "Asset:" not in text
    assert "the assets A (weight 0.75), B (weight 0.25)" in text
    assert "1.2" in text.splitlines()[-1]
    unnamed = Cover(index, cover.payout, payout_unit="EUR")
    assert "Cluster: not named" in unnamed.terms_text()
    with pytest.raises(CoverDefinitionError, match="differs from the index's tail"):
        Cover(index, LinearPayout(40.0, 30.0, 1e5, tail="lower"), payout_unit="EUR")
    with pytest.raises(CoverDefinitionError, match="ClusterIndex"):
        Cover("wind", cover.payout, payout_unit="EUR")  # type: ignore[arg-type]


def test_evaluate_and_design_refuse_a_clusters_cover_on_its_own() -> None:
    sample = _sample()
    cover = _design().covers["north"]
    one = sample.asset("A")
    with pytest.raises(EvaluationError, match="through its ClusteredDesignResult"):
        evaluate(cover, one, split="in_sample")
    with pytest.raises(EvaluationError, match="through its ClusteredDesignResult"):
        compare([{"A": cover, "B": cover, "C": cover}], sample, split="in_sample")
    with pytest.raises(EvaluationError, match="through its ClusteredDesignResult"):
        CoverPackage({"wind": cover, "rain": cover})
    template = Cover(
        _index_of(cover),
        StepPayout([33.0], [Free()]),
        payout_unit="EUR",
    )
    with pytest.raises(DesignError, match="design_joint"):
        design(template, one, MSE())


# -- design_joint under Clustered ---------------------------------------------------


@pytest.mark.parametrize("joint", [True, False])
def test_a_cluster_design_is_keyed_by_cluster(joint: bool) -> None:
    sample = _sample()
    result = _design(joint, sample, options=DesignOptions(ranges=True))
    assert isinstance(result, ClusteredDesignResult)
    assert result.cluster_ids == ("north", "south")
    assert result.asset_ids == ("A", "B", "C")
    assert result.joint is joint
    assert dict(result.members) == {"north": ("A", "B"), "south": ("C",)}
    assert set(result.parameters) == set(result.unidentified) == {"north", "south"}
    assert result.ranges is not None and set(result.ranges) == {"north", "south"}
    for cluster, cover in result.covers.items():
        assert cover.asset_id == cluster and cover.label == "step"
        assert cover.cluster_index == _scope().index(cluster, WIND)
    record = result.to_record()
    assert record["scope"] == _scope(joint).to_document()
    assert record["cluster_ids"] == ["north", "south"]
    assert record["members"] == {"north": ["A", "B"], "south": ["C"]}
    assert set(record["cover_hashes"]) == {"north", "south"}
    json.dumps(record)
    expected_loss = math.fsum((sample.probability * sample.loss.sum(axis=1)).tolist())
    assert result.expected_loss == pytest.approx(expected_loss, rel=1e-12)
    if joint:
        assert result.clusters is None
        assert result.objective_value is not None
        assert result.premium <= 0.8 * result.expected_loss * (1 + 1e-9)
    else:
        assert result.clusters is not None and result.objective_value is None
        assert (result.p_zero_loss, result.tail_outcomes) == (None, None)
        own = result.clusters["north"]
        assert isinstance(own, ClusterDesign)
        assert result.premium == pytest.approx(
            math.fsum(each.premium for each in result.clusters.values())
        )
        assert result.budget_bound == pytest.approx(
            math.fsum(each.budget_bound or 0.0 for each in result.clusters.values())
        )
        for cluster, each in result.clusters.items():
            # Each cluster's own budget: 0.8 times its own expected loss.
            assert each.budget_bound == pytest.approx(0.8 * each.expected_loss)
            assert each.premium <= 0.8 * each.expected_loss * (1 + 1e-9), cluster
        assert record["clusters"]["north"] == own.to_record()


def test_a_one_asset_cluster_is_its_assets_index() -> None:
    sample = _sample()
    result = _design()
    south = result.covers["south"].cluster_index
    assert south is not None
    np.testing.assert_array_equal(
        _cluster_index_values(south, sample), sample.index[:, 2]
    )
    north = result.covers["north"].cluster_index
    assert north is not None
    np.testing.assert_allclose(
        _cluster_index_values(north, sample),
        0.75 * sample.index[:, 0] + 0.25 * sample.index[:, 1],
        rtol=1e-15,
    )


def test_the_design_id_holds_the_weights_and_the_order() -> None:
    sample = _sample()
    first = _design()
    assert first.design_id == _design().design_id
    other = design_joint(
        [_template(), _template()],
        sample,
        ExpectedUncovered(),
        budget=Budget(ratio=0.8),
        scope=_scope(values={**VALUES, "B": 2.0e6}),
    )
    assert other.scope == first.scope and other.design_id != first.design_id
    assert _design(joint=False).design_id != first.design_id
    two = [
        _template(),
        Cover(WIND, StepPayout([35.0], [Free()]), payout_unit="EUR"),
    ]
    forward = design_joint(two, sample, MSE(), scope=_scope())
    backward = design_joint(
        two,
        sample,
        MSE(),
        scope=Clustered({"C": "south", "A": "north", "B": "north"}, values=VALUES),
    )
    assert forward.scope == backward.scope
    assert forward.design_id != backward.design_id


def test_a_joint_design_is_unchanged_by_the_scope_argument() -> None:
    sample = _sample()
    templates = [_template(asset=asset) for asset in sample.asset_ids]
    implied = design_joint(templates, sample, MSE())
    stated = design_joint(templates, sample, MSE(), scope=Joint())
    assert implied.design_id == stated.design_id
    assert implied.to_record() | {"solve": None} == stated.to_record() | {"solve": None}


def _free(
    index: Any = WIND,
    payout: Any = None,
    *,
    unit: str = "EUR",
    asset: str | None = None,
) -> Cover:
    return Cover(
        index,
        StepPayout([33.0], [Free()]) if payout is None else payout,
        payout_unit=unit,
        asset_id=asset,
    )


COMPOSITE = CompositeIndex({"wind": WIND, "gust": WIND})
REFUSED_TEMPLATES = {
    "one template": ([_free()], SampleError, "one template per cluster"),
    "another cluster": (
        [_free(asset="A"), _free()],
        SampleError,
        "for cluster 'north' is written for 'A'",
    ),
    "a cluster index": (
        [_free(ClusterIndex(WIND, {"A": 1.0})), _free()],
        SampleError,
        "pays on a ClusterIndex",
    ),
    "several components": (
        [
            _free(
                COMPOSITE,
                SumOf(
                    {
                        "wind": StepPayout([33.0], [Free()]),
                        "gust": StepPayout([33.0], [Free()]),
                    }
                ),
            ),
            _free(),
        ],
        SampleError,
        "several components",
    ),
    "another unit": ([_free(unit="USD"), _free()], SampleError, "pays in 'USD'"),
    "a free threshold": (
        [
            _free(payout=StepPayout([Free(30.0, 40.0, grid=[30.0, 40.0])], [Free()])),
            _free(),
        ],
        DesignError,
        "cluster 'north': a joint design takes fixed thresholds only",
    ),
    "nothing free": (
        [_free(payout=StepPayout([33.0], [1.0]))] * 2,
        ValueError,
        "nothing to design",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED_TEMPLATES))
def test_design_joint_refuses_templates_that_dont_suit_the_clusters(case: str) -> None:
    templates, error, match = REFUSED_TEMPLATES[case]
    with pytest.raises(error, match=match):
        design_joint(templates, _sample(), MSE(), scope=_scope())


def test_design_joint_refuses_an_assignment_that_doesnt_match_the_sample() -> None:
    sample = _sample()
    templates = [_free(), _free()]
    left_out = Clustered({"A": "north", "B": "south"}, values={"A": 1.0, "B": 1.0})
    with pytest.raises(SampleError, match=r"leaves out \['C'\] and names \[\]"):
        design_joint(templates, sample, MSE(), scope=left_out)
    unknown = Clustered({**ASSIGNMENT, "D": "east"}, values={**VALUES, "D": 1.0})
    with pytest.raises(
        SampleError, match=r"names \['D'\].*clusters \['east'\] hold no asset"
    ):
        design_joint([*templates, _free()], sample, MSE(), scope=unknown)


def test_design_joint_refuses_a_one_asset_sample_and_a_grid_objective() -> None:
    one = Clustered({"A": "north"}, values={"A": 1.0})
    with pytest.raises(DesignError, match="holds one, 'A': use design"):
        design_joint([_free()], _sample().asset("A"), MSE(), scope=one)
    objective = UserObjective(lambda loss, payout, probability: 0.0, name="zero")
    with pytest.raises(DesignError, match="UserObjective"):
        design_joint([_free(), _free()], _sample(), objective, scope=_scope())
    with pytest.raises(TypeError, match="one per cluster"):
        UNTYPED(_free(), _sample(), MSE(), scope=_scope())
    with pytest.raises(TypeError, match=r"Joint\(\) or Clustered"):
        UNTYPED([_free(), _free()], _sample(), MSE(), scope="clustered")


def test_a_cluster_design_keeps_limit_and_budget_per_cover_and_problem() -> None:
    sample = _sample()
    result = _design(False, sample, limit=Limit(2e5))
    for cover in result.covers.values():
        levels = cover.payout.parameters().values()
        assert max(float(level) for level in levels if isinstance(level, float)) <= (
            2e5 * (1 + 1e-9)
        )
    assert result.limit == Limit(2e5).to_document()
    with pytest.raises(DesignError, match="refine"):
        _design(options=DesignOptions(refine=True))


# -- the rows ------------------------------------------------------------------


@pytest.mark.parametrize("joint", [True, False])
def test_the_rows_of_a_cluster_design(joint: bool) -> None:
    sample = _sample()
    result = _design(joint, sample)
    rows = evaluate(result, sample, split="in_sample").to_pylist()
    shape = [(row["level"], row["cluster_id"], row["asset_id"]) for row in rows]
    if joint:
        assert shape == [("total", None, None)]
    else:
        assert shape == [
            ("cluster", "north", None),
            ("cluster", "south", None),
            ("total", None, None),
        ]
    total = rows[-1]
    assert {row["scope"] for row in rows} == {"cluster"}
    assert {row["design_id"] for row in rows} == {result.design_id}
    assert total["cover_hash"] == result.covers_hash
    assert total["cover_label"] == "step"
    assert total["premium"] == pytest.approx(result.premium, rel=1e-15)
    assert total["design_budget"] == pytest.approx(result.budget_bound)
    assert total["expected_loss"] == pytest.approx(result.expected_loss, rel=1e-12)
    assert total["unique"] == result.unique
    if not joint:
        assert result.clusters is not None
        for row in rows[:2]:
            own = result.clusters[row["cluster_id"]]
            cover = result.covers[row["cluster_id"]]
            assert row["cover_hash"] == cover.definition_hash
            assert (row["premium"], row["premium_basis"]) == (own.premium, "design")
            assert row["design_budget"] == own.budget_bound
            assert row["unique"] == own.unique
        assert total["premium"] == pytest.approx(
            math.fsum(row["premium"] for row in rows[:2]), rel=1e-15
        )
        assert total["expected_payout"] == pytest.approx(
            math.fsum(row["expected_payout"] for row in rows[:2]), rel=1e-12
        )


def test_a_stated_premium_states_the_total_and_keeps_the_cluster_rows() -> None:
    sample = _sample()
    result = _design(False, sample)
    rows = evaluate(result, sample, split="in_sample", premium=1234.0).to_pylist()
    assert [row["premium_basis"] for row in rows] == ["design", "design", "stated"]
    assert rows[-1]["premium"] == 1234.0
    assert result.clusters is not None
    assert rows[0]["premium"] == result.clusters["north"].premium


def test_out_of_sample_a_cluster_design_keeps_its_design_premium() -> None:
    sample = _sample(60)
    train, test = holdout(sample, 0.6)
    result = _design(False, train)
    rows = evaluate(result, test, split="out_of_sample").to_pylist()
    assert {row["split"] for row in rows} == {"out_of_sample"}
    assert [row["premium"] for row in rows[:2]] == [
        own.premium for own in (result.clusters or {}).values()
    ]
    with pytest.raises(SampleError, match="in sample means on the design sample"):
        evaluate(result, test, split="in_sample")
    with pytest.raises(SampleError, match="the sample holds"):
        evaluate(result, test.asset("A"), split="out_of_sample")
    with pytest.raises(EvaluationError, match="design_sample= differs"):
        evaluate(result, test, split="out_of_sample", design_sample=test)


@pytest.mark.parametrize("joint", [True, False])
@pytest.mark.parametrize("rule", ["value", "loss"])
def test_allocated_rows_share_each_clusters_payout_over_its_assets(
    joint: bool, rule: Any
) -> None:
    sample = _sample()
    result = _design(joint, sample)
    beside = compare(
        [result], sample, split="in_sample", allocation=rule, values=VALUES
    ).to_pylist()
    alone = allocate(
        [result], sample, split="in_sample", allocation=rule, values=VALUES
    )
    shares = [row for row in beside if row["level"] == f"asset_by_{rule}"]
    assert alone.to_pylist() == shares
    assert [row["asset_id"] for row in shares] == ["A", "B", "C"]
    for row in shares:
        assert (row["scope"], row["cluster_id"], row["premium"]) == (
            "cluster",
            None,
            None,
        )
        assert row["cover_hash"] == result.covers_hash
    # A cluster's assets share its payout: north's two, south's one.
    total = next(row for row in beside if row["level"] == "total")
    payouts = {row["asset_id"]: row["expected_payout"] for row in shares}
    assert math.fsum(payouts.values()) == pytest.approx(total["expected_payout"])
    north = result.covers["north"]
    paid = north.payout_values(_cluster_index_values(_index_of(north), sample))
    expected_north = math.fsum((sample.probability * paid).tolist())
    assert payouts["A"] + payouts["B"] == pytest.approx(expected_north, rel=1e-12)
    if rule == "value":
        assert payouts["A"] == pytest.approx(0.75 * expected_north, rel=1e-12)
    metadata = EvaluationMetadata.from_table(alone)
    (evaluation,) = metadata.setup.evaluations
    assert evaluation["scope"] == ["cluster"] and evaluation["pricing"] is None
    assert "its share of its cluster's payout" in setup_text(alone)
    assert "a cluster's payout and not a part of it" in setup_text(alone)


def test_an_allocation_needs_a_value_in_every_cluster() -> None:
    sample = _sample()
    with pytest.raises(EvaluationError, match="cluster 'south'"):
        evaluate(
            _design(),
            sample,
            split="in_sample",
            allocation="value",
            values={"A": 1.0, "B": 1.0, "C": 0.0},
        )


def test_allocation_takes_joint_and_cluster_designs_together_only() -> None:
    sample = _sample()
    clustered = _design()
    joint = design_joint(
        [_template(), _template(), _template()],
        sample,
        ExpectedUncovered(),
        budget=Budget(ratio=0.8),
    )
    table = compare(
        [joint, clustered], sample, split="in_sample", allocation="loss", values=VALUES
    )
    assert table.column("scope").to_pylist() == ["joint"] * 4 + ["cluster"] * 4
    text = setup_text(table)
    assert "the joint design's total payout, or under Clustered its cluster's" in text
    covers = {asset: cover for asset, cover in joint.covers.items()}
    with pytest.raises(EvaluationError, match="positions"):
        compare(
            [clustered, covers],
            sample,
            split="in_sample",
            allocation="value",
            values=VALUES,
        )
    with pytest.raises(TypeError, match="ClusteredDesignResult or JointDesignResult"):
        UNTYPED_ALLOCATE(
            [covers], sample, split="in_sample", allocation="value", values=VALUES
        )


def test_the_metadata_and_the_setup_text_state_the_assignment_and_setting() -> None:
    sample = _sample()
    joint = _design(True, sample)
    apart = _design(False, sample)
    table = compare([joint, apart], sample, split="in_sample")
    metadata = EvaluationMetadata.from_table(table)
    (evaluation,) = metadata.setup.evaluations
    assert evaluation["level"] == ["total", "cluster"]
    assert evaluation["scope"] == ["cluster"]
    kinds = {entry.hash: entry.document.get("kind") for entry in metadata.covers}
    assert kinds[joint.covers_hash] == kinds[apart.covers_hash] == "cluster_set"
    documents = {record["design_id"]: record for record in metadata.designs}
    assert documents[apart.design_id]["clusters"]["south"]["premium"] == (
        (apart.clusters or {})["south"].premium
    )
    text = setup_text(table)
    assert "cluster, one row per cluster (north of A and B, south of C)" in text
    assert "designed together or cluster by cluster" in text
    assert (
        "A design under Clustered(joint=True) has the premium of one contract" in text
    )
    assert "the total row the sum of the clusters' premiums" in text
    assert "budget: premium at most 0.8 x E[L] of each cluster" in text
    assert "design premiums: north" in text
    assert "on the value-weighted mean of its assets' indices" in text
    alone = setup_text(evaluate(joint, sample, split="in_sample"))
    assert "Scope: cluster" in alone and "designed together, with one" in alone


def test_a_comparison_round_trips_with_its_cluster_id_column(tmp_path: Path) -> None:
    sample = _sample()
    table = evaluate(_design(False, sample), sample, split="in_sample")
    assert table.schema.field("cluster_id").type == pa.string()
    path = tmp_path / "clusters.parquet"
    write_comparison(table, path)
    long = to_long(table)
    assert set(long.column("cluster_id").to_pylist()) == {"north", "south", None}
    # A table written before the column existed reads as before.
    earlier = table.drop_columns(["cluster_id"])
    EvaluationMetadata.from_table(earlier).check_table(earlier)
    assert "cluster_id" not in to_long(earlier).column_names


def test_rank_blocks_flag_the_rows_that_read_several_assets() -> None:
    copula = {
        "content_hash": "sha256:" + "a" * 64,
        "first_year": 1990,
        "k": 1,
        "kind": "empirical_copula",
        "last_year": 2029,
        "licences": [],
        "m": 40,
        "sources": [],
    }
    sample = _sample(dependence=copula)
    result = _design(False, sample, objective=WeightedCVaR(alpha=0.9, beta=0.9))
    rows = evaluate(
        result, sample, split="in_sample", allocation="value", values=VALUES
    ).to_pylist()
    years = {
        (row["level"], row["cluster_id"] or row["asset_id"]): row[
            "copula_reference_years"
        ]
        for row in rows
    }
    assert years == {
        ("cluster", "north"): 40,
        ("cluster", "south"): None,
        ("total", None): 40,
        ("asset_by_value", "A"): 40,
        ("asset_by_value", "B"): 40,
        ("asset_by_value", "C"): None,
    }


def test_a_payout_breakdown_refuses_a_cluster_design() -> None:
    # The functions built on compare take one, and test_parametric_cluster_functions
    # checks their rows. A breakdown parts one asset's payout over its hazards.
    sample = _sample(30)
    with pytest.raises(EvaluationError, match="parts one asset's payout"):
        payout_breakdown([_design(sample=sample)], sample, split="in_sample")


def test_a_cluster_design_is_read_in_its_loss_unit_only() -> None:
    sample = _sample()
    result = _design(sample=sample)
    other = OutcomeSample(
        asset_ids=sample.asset_ids,
        period=sample.period,
        probability=sample.probability,
        index=sample.index,
        loss=sample.loss,
        hazard=None,
        index_unit="m/s",
        loss_unit="USD",
        semantics="annual",
        provenance=sample.provenance,
    )
    with pytest.raises(SampleError, match="pays in 'EUR'"):
        evaluate(result, other, split="out_of_sample", design_sample=None)


def test_the_setup_text_states_each_clusters_degenerate_tail() -> None:
    sample = _sample()
    # South's loss is zero in more than half the years, north's in fewer.
    south = sample.loss[:, 2] == 0.0
    north = (sample.loss[:, 0] == 0.0) & (sample.loss[:, 1] == 0.0)
    assert north.mean() < 0.5 < south.mean()
    flagged = _design(False, sample, objective=WeightedCVaR(alpha=0.9, beta=0.5))
    assert flagged.clusters is not None
    assert flagged.clusters["south"].degenerate_tail == "expected_uncovered"
    assert flagged.clusters["north"].degenerate_tail is None
    rows = evaluate(flagged, sample, split="in_sample").to_pylist()
    assert [row["degenerate_tail"] for row in rows] == [
        None,
        "expected_uncovered",
        None,
    ]
    text = setup_text(evaluate(flagged, sample, split="in_sample"))
    assert "degenerate tail of the clusters: south (expected_uncovered)" in text
    plain = _design(False, sample, objective=WeightedCVaR(alpha=0.9, beta=0.9))
    assert "tail: not degenerate in any cluster" in setup_text(
        evaluate(plain, sample, split="in_sample")
    )
