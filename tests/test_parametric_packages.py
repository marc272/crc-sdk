"""`CoverPackage`, `split_budget` and `payout_breakdown`.

The numbers are synthetic and chosen so that sums and shares can be read off.
The tests check what the module computes and records; nothing here is a finding
about a cover or a cover form.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from crc_sdk.parametric import (
    MSE,
    BinaryPayout,
    Budget,
    CompositeIndex,
    CostOfCapitalLoad,
    Cover,
    CoverDefinitionError,
    CoverPackage,
    EvaluationError,
    EvaluationMetadata,
    ExpectedUncovered,
    Free,
    IndexComponent,
    IndexDefinition,
    LeakageError,
    Limit,
    MaxOf,
    MetricOptions,
    OutcomeSample,
    SampleError,
    SampleProvenance,
    StepPayout,
    SumOf,
    UnresolvedParameterError,
    compare,
    design,
    evaluate,
    holdout,
    payout_breakdown,
    register_payout,
    setup_text,
    split_budget,
    to_long,
    write_comparison,
)
from crc_sdk.parametric.index import Tail
from crc_sdk.parametric.joint import covers_hash
from crc_sdk.parametric.metadata import IDENTITY_COLUMNS, _identity_columns
from crc_sdk.parametric.parameters import _resolve_values
from crc_sdk.parametric.payouts import _REGISTRY

RAIN = IndexDefinition("rain", "mm/day", aggregation="annual_maximum")
HEAT = IndexDefinition("heat", "degC", aggregation="annual_maximum")
TWO = CompositeIndex({"rain": RAIN, "heat": HEAT})

N = 12
RAIN_VALUES = np.array(
    [10.0, 60.0, 20.0, 80.0, 30.0, 15.0, 70.0, 25.0, 55.0, 12.0, 65.0, 22.0]
)
HEAT_VALUES = np.array(
    [30.0, 25.0, 40.0, 20.0, 35.0, 42.0, 45.0, 38.0, 20.0, 44.0, 41.0, 18.0]
)
RAIN_LOSS = np.array([0.0, 10.0, 0.0, 6.0, 0.0, 0.0, 8.0, 0.0, 5.0, 0.0, 7.0, 0.0])
HEAT_LOSS = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 6.0, 6.0, 0.0, 3.0, 5.0, 0.0])
COMBINED = np.minimum(RAIN_LOSS + HEAT_LOSS, 12.0)
"""The combined loss caps at 12, so in some years it is below the sum of the own."""


def _provenance() -> SampleProvenance:
    return SampleProvenance(
        source_kind="table",
        sources=(),
        semantics_source="caller",
        hazard_dependence={"kind": "data"},
        combine={"kind": "table"},
    )


def _sample(**changes: Any) -> OutcomeSample:
    fields: dict[str, Any] = {
        "asset_ids": ("site",),
        "period": np.arange(2000, 2000 + N),
        "probability": np.full(N, 1 / N),
        "index": RAIN_VALUES,
        "loss": COMBINED,
        "hazard": None,
        "index_unit": "mm/day",
        "loss_unit": "EUR",
        "semantics": "annual",
        "provenance": _provenance(),
        "index_components": {
            "rain": IndexComponent(
                RAIN_VALUES,
                unit="mm/day",
                tail="upper",
                hazard=None,
                own_loss=RAIN_LOSS,
            ),
            "heat": IndexComponent(
                HEAT_VALUES, unit="degC", tail="upper", hazard=None, own_loss=HEAT_LOSS
            ),
        },
    }
    fields.update(changes)
    untyped: Any = OutcomeSample
    built: OutcomeSample = untyped(**fields)
    return built


def _one_hazard() -> OutcomeSample:
    return _sample(
        index_components=None,
        provenance=SampleProvenance(
            source_kind="table", sources=(), semantics_source="caller"
        ),
    )


def _rain_cover(amount: Any = 8.0, **rest: Any) -> Cover:
    return Cover(RAIN, BinaryPayout(50.0, amount), payout_unit="EUR", **rest)


def _heat_cover(amount: Any = 5.0, **rest: Any) -> Cover:
    return Cover(HEAT, BinaryPayout(38.0, amount), payout_unit="EUR", **rest)


def _as_is_package(**rest: Any) -> CoverPackage:
    return CoverPackage(
        {"rain": _rain_cover(label="rain"), "heat": _heat_cover(label="heat")}, **rest
    )


def _column(table: Any, name: str) -> list[Any]:
    values: list[Any] = table.column(name).to_pylist()
    return values


def _row(table: Any, **where: Any) -> dict[str, Any]:
    rows = [
        row
        for row in table.to_pylist()
        if all(row[key] == value for key, value in where.items())
    ]
    assert len(rows) == 1, (where, len(rows))
    row: dict[str, Any] = rows[0]
    return row


class _Designs:
    """A holdout of the sample, and each hazard's contract on the training block."""

    def __init__(self, *, pricing: Any = None, limits: tuple[Any, Any] = (None, None)):
        self.sample = _sample()
        self.train, self.test = holdout(self.sample, train_fraction=0.75)
        pricing_kw = {} if pricing is None else {"pricing": pricing}
        budgets = split_budget(
            Budget(amount=3.0), self.train, shares={"rain": 0.6, "heat": 0.4}
        )
        self.budgets = budgets
        self.rain = design(
            _rain_cover(Free(), label="rain"),
            self.train.component("rain"),
            ExpectedUncovered(),
            budget=budgets["rain"],
            limit=limits[0],
            **pricing_kw,
        )
        self.heat = design(
            _heat_cover(Free(), label="heat"),
            self.train.component("heat"),
            ExpectedUncovered(),
            budget=budgets["heat"],
            limit=limits[1],
            **pricing_kw,
        )
        self.package = CoverPackage(
            {"rain": self.rain, "heat": self.heat}, label="separate"
        )


# -- the package -------------------------------------------------------------


def test_a_package_is_the_sum_of_its_covers() -> None:
    package = _as_is_package(label="separate")
    assert list(package.covers) == ["rain", "heat"]
    assert package.payout == SumOf(
        {"rain": BinaryPayout(50.0, 8.0), "heat": BinaryPayout(38.0, 5.0)}
    )
    assert package.package_hash == covers_hash(package.covers)
    assert package.label == "separate"
    assert package.index == TWO


def test_as_cover_is_the_same_payout_as_one_contract() -> None:
    package = _as_is_package(label="separate")
    cover = package.as_cover()
    index = np.column_stack([RAIN_VALUES, HEAT_VALUES])
    assert cover.payout_values(index).tolist() == [
        8.0 * (r >= 50.0) + 5.0 * (h >= 38.0)
        for r, h in zip(RAIN_VALUES, HEAT_VALUES, strict=True)
    ]
    assert cover.label == "separate" and cover.index_definition == TWO


def test_the_package_hash_differs_from_the_one_contract_hash() -> None:
    package = _as_is_package()
    assert package.package_hash != package.as_cover().definition_hash


def test_a_package_pickles() -> None:
    package = _as_is_package(label="separate")
    read = pickle.loads(pickle.dumps(package))
    assert read.package_hash == package.package_hash and read.label == "separate"


def test_a_package_keeps_each_key_as_the_component_it_reads() -> None:
    wet = Cover(
        IndexDefinition("rain", "mm/day", aggregation="annual_maximum"),
        BinaryPayout(50.0, 8.0),
        payout_unit="EUR",
    )
    sample = _sample()
    components = sample.index_components
    assert components is not None
    renamed = _sample(
        index_components={"wet": components["rain"], "heat": components["heat"]}
    )
    package = CoverPackage({"wet": wet, "heat": _heat_cover()})
    table = evaluate(package, renamed, split="in_sample")
    row = _row(table, cover_hash=wet.definition_hash)
    assert row["components"] == "wet"
    assert row["expected_payout"] == pytest.approx(8.0 * (RAIN_VALUES >= 50.0).mean())


@pytest.mark.parametrize(
    ("build", "kind", "message"),
    [
        pytest.param(lambda: CoverPackage([]), TypeError, "map each", id="not-mapping"),  # type: ignore[arg-type]
        pytest.param(
            lambda: CoverPackage({"rain": _rain_cover()}),
            EvaluationError,
            "two covers",
            id="one",
        ),
        pytest.param(
            lambda: CoverPackage({"a.b": _rain_cover(), "heat": _heat_cover()}),
            EvaluationError,
            "component name",
            id="bad-name",
        ),
        pytest.param(
            lambda: CoverPackage({"rain": _rain_cover(), "heat": 1.0}),  # type: ignore[dict-item]
            TypeError,
            "a Cover or a designed cover",
            id="not-a-cover",
        ),
        pytest.param(
            lambda: CoverPackage(
                {
                    "rain": Cover(
                        TWO,
                        SumOf(
                            {
                                "rain": BinaryPayout(50.0, 1.0),
                                "heat": BinaryPayout(38.0, 1.0),
                            }
                        ),
                        payout_unit="EUR",
                    ),
                    "heat": _heat_cover(),
                }
            ),
            EvaluationError,
            "several components",
            id="composite",
        ),
        pytest.param(
            lambda: CoverPackage({"rain": _rain_cover(Free()), "heat": _heat_cover()}),
            UnresolvedParameterError,
            "template",
            id="template",
        ),
        pytest.param(
            lambda: CoverPackage(
                {
                    "rain": _rain_cover(),
                    "heat": Cover(HEAT, BinaryPayout(38.0, 5.0), payout_unit="USD"),
                }
            ),
            EvaluationError,
            "payout_unit",
            id="units",
        ),
        pytest.param(
            lambda: CoverPackage(
                {"rain": _rain_cover(asset_id="a"), "heat": _heat_cover(asset_id="b")}
            ),
            EvaluationError,
            "asset_id",
            id="assets",
        ),
        pytest.param(
            lambda: CoverPackage(
                {"rain": _rain_cover(asset_id="a"), "heat": _heat_cover()}
            ),
            EvaluationError,
            "asset_id",
            id="asset-and-none",
        ),
        pytest.param(
            lambda: _as_is_package(label="  "), EvaluationError, "label", id="label"
        ),
    ],
)
def test_an_invalid_package_raises(build: Any, kind: Any, message: str) -> None:
    with pytest.raises(kind, match=message):
        build()


def test_a_mix_of_covers_and_designs_raises() -> None:
    designs = _Designs()
    with pytest.raises(EvaluationError, match="not a mix"):
        CoverPackage({"rain": designs.rain, "heat": _heat_cover()})


# -- the rows ----------------------------------------------------------------


def test_a_package_pays_their_sum_and_has_the_sum_of_their_premiums() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    assert _column(table, "level") == ["asset", "asset", "package"]
    rain = _row(table, cover_label="rain")
    heat = _row(table, cover_label="heat")
    package = _row(table, level="package")
    assert rain["premium"] == designs.rain.premium
    assert heat["premium"] == designs.heat.premium
    assert package["premium"] == pytest.approx(
        designs.rain.premium + designs.heat.premium, abs=1e-12
    )
    assert package["expected_payout"] == pytest.approx(
        rain["expected_payout"] + heat["expected_payout"], abs=1e-12
    )
    assert package["premium_basis"] == "design"
    index = np.column_stack(
        [
            designs.test.component("rain").index[:, 0],
            designs.test.component("heat").index[:, 0],
        ]
    )
    summed = designs.rain.cover.payout_values(
        index[:, 0]
    ) + designs.heat.cover.payout_values(index[:, 1])
    expected = float((designs.test.probability * summed).sum())
    assert package["expected_payout"] == pytest.approx(expected, abs=1e-12)


def test_every_row_is_on_the_combined_loss() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    combined = float((designs.test.probability * designs.test.loss[:, 0]).sum())
    assert _column(table, "expected_loss") == pytest.approx([combined] * 3)
    assert set(_column(table, "sample_hash")) == {designs.test.content_hash}


def test_the_package_row_has_the_columns_of_the_decision() -> None:
    designs = _Designs()
    package = _row(
        compare([designs.package], designs.test, split="out_of_sample"),
        level="package",
    )
    assert package["asset_id"] == "site"
    assert package["scope"] == "individual"
    assert package["cover_hash"] == designs.package.package_hash
    assert package["cover_label"] == "separate"
    for key in ("design_id", "unique", "ties", "degenerate_tail"):
        assert package[key] is None
    assert package["components"] == "rain+heat"
    assert package["design_budget"] == pytest.approx(3.0, abs=1e-12)


def test_the_contract_rows_carry_their_own_design() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    rain = _row(table, cover_label="rain")
    assert rain["design_id"] == designs.rain.design_id
    assert rain["cover_hash"] == designs.rain.cover.definition_hash
    assert rain["components"] == "rain"
    assert rain["design_budget"] == pytest.approx(1.8, abs=1e-12)


def test_a_package_without_a_label_takes_the_hash() -> None:
    package = _as_is_package()
    row = _row(evaluate(package, _sample(), split="in_sample"), level="package")
    assert row["cover_label"] == package.package_hash.removeprefix("sha256:")[:12]


def test_a_budget_missing_from_one_contract_leaves_the_sum_null() -> None:
    sample = _sample()
    rain = design(
        _rain_cover(Free(), label="rain"),
        sample.component("rain"),
        MSE(),
        budget=Budget(amount=1.0),
    )
    heat = design(_heat_cover(Free(), label="heat"), sample.component("heat"), MSE())
    table = evaluate(
        CoverPackage({"rain": rain, "heat": heat}), sample, split="in_sample"
    )
    assert _row(table, level="package")["design_budget"] is None
    assert _row(table, cover_label="rain")["design_budget"] == 1.0
    assert _row(table, cover_label="heat")["design_budget"] is None


def test_covers_given_as_is_are_priced_on_the_block() -> None:
    sample = _sample()
    table = evaluate(
        _as_is_package(), sample, split="in_sample", pricing=CostOfCapitalLoad(0.2, 0.9)
    )
    assert _column(table, "premium_basis") == ["evaluation"] * 3
    assert _column(table, "design_budget") == [None] * 3
    package = _row(table, level="package")
    rain = _row(table, cover_label="rain")
    heat = _row(table, cover_label="heat")
    assert package["premium"] == pytest.approx(rain["premium"] + heat["premium"])


def test_a_stated_premium_states_the_total_only() -> None:
    table = evaluate(_as_is_package(), _sample(), split="in_sample", premium=99.0)
    package = _row(table, level="package")
    assert (package["premium"], package["premium_basis"]) == (99.0, "stated")
    contracts = [row for row in table.to_pylist() if row["level"] == "asset"]
    assert {row["premium_basis"] for row in contracts} == {"evaluation"}


def test_a_loaded_premium_is_per_contract() -> None:
    # At q = 0.5 the worst years of each contract aren't the same years.
    pricing = CostOfCapitalLoad(0.1, 0.5)
    sample = _sample()
    package = _as_is_package(label="separate")
    table = evaluate(package, sample, split="in_sample", pricing=pricing)
    row = _row(table, level="package")
    index = np.column_stack([RAIN_VALUES, HEAT_VALUES])
    rain_paid = package.covers["rain"].payout_values(RAIN_VALUES)
    heat_paid = package.covers["heat"].payout_values(HEAT_VALUES)
    p = sample.probability
    separate = pricing.premium(rain_paid, p) + pricing.premium(heat_paid, p)
    one_contract = pricing.premium(package.as_cover().payout_values(index), p)
    assert row["premium"] == pytest.approx(separate, abs=1e-9)
    assert one_contract < separate - 1e-6


def test_each_cover_keeps_its_own_limit() -> None:
    designs = _Designs(limits=(Limit(0.5), Limit(0.25)))
    values = designs.package.payout.parameters()
    assert values["rain.amount"] == pytest.approx(0.5, abs=1e-9)
    assert values["heat.amount"] == pytest.approx(0.25, abs=1e-9)
    records = {
        record["cover_hash"]: record
        for record in EvaluationMetadata.from_table(
            compare([designs.package], designs.test, split="out_of_sample")
        ).designs
    }
    assert records[designs.rain.cover.definition_hash]["limit"] == {"amount": 0.5}
    assert records[designs.heat.cover.definition_hash]["limit"] == {"amount": 0.25}
    assert "limit" not in designs.package.as_cover().to_document()


def test_a_package_stands_beside_a_cover_on_several_components() -> None:
    designs = _Designs()
    holistic = design(
        Cover(
            TWO,
            SumOf(
                {"rain": BinaryPayout(50.0, Free()), "heat": BinaryPayout(38.0, Free())}
            ),
            payout_unit="EUR",
            label="holistic",
        ),
        designs.train,
        ExpectedUncovered(),
        budget=Budget(amount=3.0),
    )
    table = compare([holistic, designs.package], designs.test, split="out_of_sample")
    assert _column(table, "level") == ["asset", "asset", "asset", "package"]
    assert _column(table, "components") == ["rain+heat", "rain", "heat", "rain+heat"]
    assert _column(table, "design_budget") == pytest.approx([3.0, 1.8, 1.2, 3.0])
    assert "rank" not in table.column_names and "best" not in table.column_names


def test_a_package_can_sit_on_a_sample_that_holds_more_components() -> None:
    sample = _sample()
    components = sample.index_components
    assert components is not None
    third = IndexComponent(
        RAIN_VALUES, unit="m", tail="upper", hazard=None, own_loss=None
    )
    wider = _sample(index_components={**components, "wind": third})
    table = evaluate(_as_is_package(), wider, split="in_sample")
    assert _row(table, level="package")["components"] == "rain+heat"


def test_listing_a_contract_again_repeats_a_key() -> None:
    designs = _Designs()
    with pytest.raises(EvaluationError, match="repeat a row key"):
        compare([designs.package, designs.rain], designs.test, split="out_of_sample")


def test_a_package_needs_a_sample_with_its_components() -> None:
    with pytest.raises(SampleError, match="one index"):
        evaluate(_as_is_package(), _one_hazard(), split="in_sample")


def test_a_mapping_of_covers_on_several_hazards_names_the_package() -> None:
    with pytest.raises(SampleError, match="CoverPackage"):
        evaluate({"site": _rain_cover()}, _sample(), split="in_sample")


def test_a_package_for_another_asset_raises() -> None:
    package = CoverPackage(
        {"rain": _rain_cover(asset_id="a"), "heat": _heat_cover(asset_id="a")}
    )
    with pytest.raises(SampleError, match="written for asset"):
        evaluate(package, _sample(), split="in_sample")


def test_a_package_in_the_wrong_unit_raises() -> None:
    package = CoverPackage(
        {
            "rain": Cover(RAIN, BinaryPayout(50.0, 1.0), payout_unit="USD"),
            "heat": Cover(HEAT, BinaryPayout(38.0, 1.0), payout_unit="USD"),
        }
    )
    with pytest.raises(SampleError, match="pays in"):
        evaluate(package, _sample(), split="in_sample")


def test_each_contract_is_checked_for_leakage() -> None:
    designs = _Designs()
    leaking = design(
        _rain_cover(Free(), label="rain"), designs.sample.component("rain"), MSE()
    )
    package = CoverPackage({"rain": leaking, "heat": designs.heat})
    with pytest.raises(LeakageError):
        evaluate(package, designs.test, split="out_of_sample")


def test_in_sample_is_each_contracts_own_view() -> None:
    designs = _Designs()
    table = evaluate(designs.package, designs.train, split="in_sample")
    assert _column(table, "split") == ["in_sample"] * 3
    with pytest.raises(SampleError, match="design sample"):
        evaluate(designs.package, designs.test, split="in_sample")


def test_a_stated_design_sample_is_compared_with_each_view() -> None:
    designs = _Designs()
    table = evaluate(
        designs.package,
        designs.test,
        split="out_of_sample",
        design_sample=designs.train,
    )
    assert table.num_rows == 3
    with pytest.raises(EvaluationError, match="differs"):
        evaluate(
            designs.package,
            designs.train,
            split="in_sample",
            design_sample=designs.test,
        )


def test_covers_as_is_out_of_sample_need_a_design_sample() -> None:
    train, test = holdout(_sample(), train_fraction=0.75)
    with pytest.raises(SampleError, match="needs the design sample"):
        evaluate(_as_is_package(), test, split="out_of_sample")
    table = evaluate(_as_is_package(), test, split="out_of_sample", design_sample=train)
    assert table.num_rows == 3


def test_a_contract_designed_on_another_asset_raises() -> None:
    designs = _Designs()
    fields = {"asset_ids": ("other",)}
    other = design(
        _rain_cover(Free(), label="rain"),
        _sample(**fields).component("rain"),
        MSE(),
    )
    package = CoverPackage({"rain": other, "heat": designs.heat})
    with pytest.raises(SampleError, match="different assets"):
        evaluate(package, designs.test, split="out_of_sample")


# -- the metadata ------------------------------------------------------------


def test_the_metadata_holds_the_package_and_its_covers() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    metadata = EvaluationMetadata.from_table(table)
    entries = {entry.hash: entry for entry in metadata.covers}
    assert entries[designs.package.package_hash].document == {
        "kind": "package",
        "covers": {
            "rain": designs.rain.cover.definition_hash,
            "heat": designs.heat.cover.definition_hash,
        },
    }
    assert entries[designs.package.package_hash].label == "separate"
    assert designs.rain.cover.definition_hash in entries
    assert {record["design_id"] for record in metadata.designs} == {
        designs.rain.design_id,
        designs.heat.design_id,
    }
    evaluation = metadata.setup.evaluations[0]
    assert evaluation["level"] == ["asset", "package"]
    assert set(evaluation["design_samples"]) == {
        designs.rain.design_sample.content_hash,
        designs.heat.design_sample.content_hash,
    }


def test_a_one_hazard_comparison_keeps_its_setup() -> None:
    cover = _rain_cover(label="rain")
    table = evaluate(cover, _one_hazard(), split="in_sample")
    evaluation = EvaluationMetadata.from_table(table).setup.evaluations[0]
    assert "rows" not in evaluation and "packages" not in evaluation
    assert _column(table, "components") == [None]
    assert _column(table, "design_budget") == [None]


def test_a_design_on_one_hazard_records_its_budget_bound() -> None:
    sample = _one_hazard()
    result = design(_rain_cover(Free()), sample, MSE(), budget=Budget(ratio=0.5))
    row = evaluate(result, sample, split="in_sample").to_pylist()[0]
    assert row["design_budget"] == result.budget_bound
    assert row["components"] is None


def test_a_table_with_a_package_writes_and_reads(tmp_path: Path) -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    path = tmp_path / "package.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert _column(read, "level") == ["asset", "asset", "package"]
    assert EvaluationMetadata.from_table(read) == EvaluationMetadata.from_table(table)


def test_to_long_keeps_the_new_columns() -> None:
    table = evaluate(_as_is_package(), _sample(), split="in_sample")
    long = to_long(table)
    assert long.column_names[: len(IDENTITY_COLUMNS)] == list(IDENTITY_COLUMNS)
    assert set(_column(long, "components")) == {"rain", "heat", "rain+heat"}
    n_identity = len(_identity_columns(table.column_names))
    assert long.num_rows == 3 * (table.num_columns - n_identity)


def test_a_table_without_the_new_columns_still_reads(tmp_path: Path) -> None:
    table = evaluate(_rain_cover(label="rain"), _one_hazard(), split="in_sample")
    older = table.drop_columns(["components", "design_budget"])
    path = tmp_path / "older.parquet"
    write_comparison(older, path)
    assert "components" not in pq.read_table(path).column_names
    assert "components" not in to_long(older).column_names
    assert "Level: asset" in setup_text(older)


def test_the_setup_states_a_package() -> None:
    designs = _Designs(limits=(Limit(0.5), None))
    table = compare([designs.package], designs.test, split="out_of_sample")
    text = setup_text(table)
    assert "package, the separate contracts of site as one row" in text
    assert "The package row holds the sum of its contracts' premiums" in text
    line = next(item for item in text.splitlines() if item.startswith("Package "))
    assert line.startswith("Package separate: rain (budget at most 1.8 EUR")
    assert "limit 0.5 EUR" in line and "no limit stated at design" in line
    assert "the budgets sum to 3 EUR" in line
    assert "Each cover keeps its own limit" in line
    assert "A loading is not additive across contracts" in line


def test_the_setup_says_what_every_row_reads() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    lines = setup_text(table).splitlines()
    assert (
        "Row rain (asset): reads rain; designed on the own loss of component rain; "
        "budget: premium at most 1.8 EUR; pricing: the pure premium, P = E[payout]."
        in lines
    )
    assert (
        "Row separate (package): reads rain and heat; each contract designed or "
        "given on its own; budget: premium at most 3 EUR; pricing: each contract on "
        "its own terms." in lines
    )


def test_a_setup_alone_does_not_call_a_design_given_as_is() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    text = setup_text(EvaluationMetadata.from_table(table).setup)
    assert "Row rain (asset): reads rain; designed, with no design record" in text
    assert "reads rain; given as is" not in text


def test_the_setup_says_a_cover_given_as_is_is_not_designed() -> None:
    text = setup_text(evaluate(_as_is_package(label="p"), _sample(), split="in_sample"))
    assert "given as is; no design budget" in text
    assert "not every contract has a design budget" in text
    assert "; the premiums sum to " in text


def test_the_setup_states_a_stated_premium() -> None:
    table = evaluate(
        _as_is_package(label="p"), _sample(), split="in_sample", premium=9.0
    )
    assert "its row uses the stated premium 9 EUR" in setup_text(table)


def test_the_setup_of_filtered_rows_drops_the_package() -> None:
    designs = _Designs()
    table = compare([designs.package], designs.test, split="out_of_sample")
    rows = table.filter(pc.equal(table.column("level"), pa.scalar("asset")))
    text = setup_text(rows)
    assert not any(line.startswith("Package ") for line in text.splitlines())
    assert "Row rain (asset)" in text and "Row separate" not in text


# -- split_budget ------------------------------------------------------------


def test_split_budget_by_stated_shares_sums_to_the_bound() -> None:
    budgets = split_budget(
        Budget(amount=10.0), _sample(), shares={"rain": 0.3, "heat": 0.7}
    )
    assert budgets["rain"] == Budget(amount=3.0)
    assert budgets["heat"].amount == pytest.approx(7.0, abs=1e-12)


def test_split_budget_of_a_ratio_is_on_the_combined_loss() -> None:
    sample = _sample()
    budgets = split_budget(Budget(ratio=0.5), sample, shares={"rain": 0.5, "heat": 0.5})
    combined = float(COMBINED.mean())
    assert budgets["rain"].amount == pytest.approx(0.25 * combined, abs=1e-12)
    assert sum(b.amount or 0.0 for b in budgets.values()) == pytest.approx(
        0.5 * combined, abs=1e-12
    )


def test_split_budget_by_own_loss() -> None:
    budgets = split_budget(Budget(amount=10.0), _sample(), shares="own_loss")
    rain, heat = float(RAIN_LOSS.mean()), float(HEAT_LOSS.mean())
    assert list(budgets) == ["rain", "heat"]
    assert budgets["rain"].amount == pytest.approx(10.0 * rain / (rain + heat))
    assert budgets["heat"].amount == pytest.approx(10.0 * heat / (rain + heat))


def test_split_budget_of_a_part_of_the_components() -> None:
    sample = _sample()
    components = sample.index_components
    assert components is not None
    third = IndexComponent(
        RAIN_VALUES, unit="m", tail="upper", hazard=None, own_loss=None
    )
    wider = _sample(index_components={**components, "wind": third})
    budgets = split_budget(Budget(amount=4.0), wider, shares={"rain": 1.0, "heat": 0.0})
    assert list(budgets) == ["rain", "heat"]
    with pytest.raises(SampleError, match="own loss"):
        split_budget(Budget(amount=4.0), wider, shares="own_loss")


@pytest.mark.parametrize(
    ("shares", "kind", "message"),
    [
        pytest.param("other", EvaluationError, "'own_loss' or a mapping", id="other"),
        pytest.param(0.5, TypeError, "'own_loss' or a mapping", id="number"),
        pytest.param({"rain": 0.5}, EvaluationError, "sum to 1", id="short"),
        pytest.param(
            {"rain": 0.5, "wind": 0.5}, EvaluationError, "does not hold", id="unknown"
        ),
        pytest.param(
            {"rain": -0.5, "heat": 1.5}, EvaluationError, ">= 0", id="negative"
        ),
        pytest.param({"rain": True, "heat": 0.0}, EvaluationError, ">= 0", id="bool"),
    ],
)
def test_invalid_shares_raise(shares: Any, kind: Any, message: str) -> None:
    with pytest.raises(kind, match=message):
        split_budget(Budget(amount=1.0), _sample(), shares=shares)


def test_split_budget_needs_components() -> None:
    with pytest.raises(SampleError, match="has none"):
        split_budget(Budget(amount=1.0), _one_hazard(), shares="own_loss")


def test_split_budget_needs_a_budget_and_a_sample() -> None:
    with pytest.raises(TypeError, match="budget must be"):
        split_budget(1.0, _sample(), shares="own_loss")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sample must be"):
        split_budget(Budget(amount=1.0), 1.0, shares="own_loss")  # type: ignore[arg-type]


def test_split_budget_of_own_losses_all_zero_raises() -> None:
    zeros = np.zeros(N)
    sample = _sample(
        index_components={
            "rain": IndexComponent(
                RAIN_VALUES, unit="mm/day", tail="upper", hazard=None, own_loss=zeros
            ),
            "heat": IndexComponent(
                HEAT_VALUES, unit="degC", tail="upper", hazard=None, own_loss=zeros
            ),
        }
    )
    with pytest.raises(SampleError, match="are zero"):
        split_budget(Budget(amount=1.0), sample, shares="own_loss")


# -- payout_breakdown --------------------------------------------------------


class Stated:
    """A vector family with a `breakdown` it states; `whole` omits one."""

    kind: ClassVar[str] = "test_stated"
    index_names: ClassVar[tuple[str, ...]] = ("rain", "heat")
    tail: ClassVar[tuple[Tail, ...]] = ("upper", "upper")

    def __init__(self, amount: Any, parts: Any = None) -> None:
        self.amount = amount
        self.parts = parts

    def parameters(self) -> dict[str, float | Free]:
        return {"amount": self.amount}

    def with_values(self, values: Mapping[str, float]) -> Self:
        checked = _resolve_values(self.parameters(), values)
        return type(self)(checked.get("amount", self.amount), self.parts)

    def evaluate(self, index: Any) -> Any:
        return self.amount * (index[:, 0] >= 50.0).astype(np.float64)

    def breakdown(self, index: Any) -> Any:
        if self.parts is not None:
            return self.parts(self.evaluate(index))
        paid = self.evaluate(index)
        return {"first": 0.25 * paid, "second": 0.75 * paid}

    def to_document(self) -> dict[str, Any]:
        return {"amount": self.amount, "kind": self.kind}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(document["amount"])


class Whole:
    """A vector family without a `breakdown`."""

    kind: ClassVar[str] = "test_whole"
    index_names: ClassVar[tuple[str, ...]] = ("rain", "heat")
    tail: ClassVar[tuple[Tail, ...]] = ("upper", "upper")

    def __init__(self, amount: Any) -> None:
        self.amount = amount

    def parameters(self) -> dict[str, float | Free]:
        return {"amount": self.amount}

    def with_values(self, values: Mapping[str, float]) -> Self:
        return type(self)(values.get("amount", self.amount))

    def evaluate(self, index: Any) -> Any:
        return self.amount * (index[:, 0] >= 50.0).astype(np.float64)

    def to_document(self) -> dict[str, Any]:
        return {"amount": self.amount, "kind": self.kind}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls(document["amount"])


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    families: tuple[type[Any], ...] = (Stated, Whole)
    for family in families:
        register_payout(family)
    yield
    for family in families:
        del _REGISTRY[family.kind]


def _sum_cover(rain: float = 8.0, heat: float = 5.0) -> Cover:
    return Cover(
        TWO,
        SumOf({"rain": BinaryPayout(50.0, rain), "heat": BinaryPayout(38.0, heat)}),
        payout_unit="EUR",
        label="sum",
    )


def _max_cover(rain: float = 8.0, heat: float = 5.0, scale: float = 1.0) -> Cover:
    return Cover(
        TWO,
        MaxOf(
            {"rain": BinaryPayout(50.0, rain), "heat": BinaryPayout(38.0, heat)},
            scale=scale,
        ),
        payout_unit="EUR",
        label="max",
    )


def _parts_add_up(entry: Any, table: Any, level: str = "asset") -> None:
    compared = _row(
        compare([entry], _sample(), split="in_sample"),
        level=level,
        **({} if level == "package" else {"cover_label": entry.label}),
    )
    rows = table.to_pylist()
    assert sum(row["expected_payout"] for row in rows) == pytest.approx(
        compared["expected_payout"], abs=1e-12
    )
    assert sum(row["share"] for row in rows) == pytest.approx(1.0, abs=1e-12)


def test_a_sum_of_breaks_down_into_its_components() -> None:
    cover = _sum_cover()
    table = payout_breakdown([cover], _sample(), split="in_sample")
    assert _column(table, "part") == ["rain", "heat"]
    assert _column(table, "part_kind") == ["component", "component"]
    rain = _row(table, part="rain")
    assert rain["expected_payout"] == pytest.approx(8.0 * (RAIN_VALUES >= 50.0).mean())
    assert rain["payout_frequency"] == pytest.approx((RAIN_VALUES >= 50.0).mean())
    assert rain["cover_label"] == "sum" and rain["components"] == "rain+heat"
    _parts_add_up(cover, table)


def test_a_max_of_is_an_attribution() -> None:
    cover = _max_cover()
    table = payout_breakdown([cover], _sample(), split="in_sample")
    assert set(_column(table, "part_kind")) == {"attribution"}
    rain, heat = RAIN_VALUES >= 50.0, HEAT_VALUES >= 38.0
    assert _row(table, part="rain")["expected_payout"] == pytest.approx(
        8.0 * rain.mean()
    )
    assert _row(table, part="heat")["expected_payout"] == pytest.approx(
        5.0 * (heat & ~rain).mean()
    )
    _parts_add_up(cover, table)


def test_a_max_of_shares_a_tie_equally() -> None:
    cover = _max_cover(rain=4.0, heat=4.0, scale=2.0)
    table = payout_breakdown([cover], _sample(), split="in_sample")
    rain, heat = RAIN_VALUES >= 50.0, HEAT_VALUES >= 38.0
    both = rain & heat
    assert _row(table, part="rain")["expected_payout"] == pytest.approx(
        8.0 * (rain & ~heat).mean() + 4.0 * both.mean()
    )
    assert _row(table, part="heat")["expected_payout"] == pytest.approx(
        8.0 * (heat & ~rain).mean() + 4.0 * both.mean()
    )
    _parts_add_up(cover, table)


def test_a_one_index_cover_is_its_component() -> None:
    cover = _rain_cover(label="rain")
    sample = _sample()
    table = payout_breakdown([cover], sample, split="in_sample")
    row = table.to_pylist()[0]
    assert (row["part"], row["part_kind"], row["share"]) == ("rain", "component", 1.0)
    alone = payout_breakdown([cover], _one_hazard(), split="in_sample").to_pylist()[0]
    assert alone["part"] == "rain" and alone["components"] is None


def test_a_package_breaks_down_into_its_contracts() -> None:
    package = _as_is_package(label="separate")
    table = payout_breakdown([package], _sample(), split="in_sample")
    assert _column(table, "part") == ["rain", "heat"]
    assert set(_column(table, "level")) == {"package"}
    assert set(_column(table, "cover_hash")) == {package.package_hash}
    _parts_add_up(package, table, level="package")


def test_a_package_of_designs_keeps_its_identity_in_the_breakdown() -> None:
    designs = _Designs()
    table = payout_breakdown([designs.package], designs.test, split="out_of_sample")
    assert set(_column(table, "premium_basis")) == {"design"}
    assert _column(table, "design_budget") == pytest.approx([3.0, 3.0])


def test_a_user_family_states_its_breakdown() -> None:
    cover = Cover(TWO, Stated(4.0), payout_unit="EUR", label="user")
    table = payout_breakdown([cover], _sample(), split="in_sample")
    assert _column(table, "part") == ["first", "second"]
    assert set(_column(table, "part_kind")) == {"user"}
    _parts_add_up(cover, table)


def test_a_user_family_without_one_is_whole() -> None:
    cover = Cover(TWO, Whole(4.0), payout_unit="EUR", label="whole")
    table = payout_breakdown([cover], _sample(), split="in_sample")
    assert _column(table, "part") == ["whole"]
    assert _column(table, "part_kind") == ["whole"]
    assert _column(table, "share") == [1.0]


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        pytest.param(
            lambda paid: {"a": 0.5 * paid}, "differ from its payout", id="short"
        ),
        pytest.param(lambda paid: paid, "non-empty mapping", id="not-a-mapping"),
        pytest.param(lambda paid: {}, "non-empty mapping", id="empty"),
        pytest.param(lambda paid: {"a": paid[:3]}, "one value per draw", id="shape"),
        pytest.param(lambda paid: {"a": -paid}, "non-negative", id="negative"),
        pytest.param(lambda paid: {1: paid}, "names a part", id="name"),
    ],
)
def test_an_invalid_user_breakdown_raises(parts: Any, message: str) -> None:
    cover = Cover(TWO, Stated(4.0, parts), payout_unit="EUR", label="user")
    with pytest.raises(EvaluationError, match=message):
        payout_breakdown([cover], _sample(), split="in_sample")


def test_a_share_is_null_when_nothing_pays() -> None:
    cover = _sum_cover(0.0, 0.0)
    table = payout_breakdown([cover], _sample(), split="in_sample")
    assert _column(table, "share") == [None, None]
    assert _column(table, "expected_payout") == [0.0, 0.0]


def test_the_payout_frequency_reads_the_threshold() -> None:
    table = payout_breakdown(
        [_sum_cover(8.0, 5.0)],
        _sample(),
        split="in_sample",
        options=MetricOptions(payout_threshold=6.0),
    )
    assert _row(table, part="rain")["payout_frequency"] > 0.0
    assert _row(table, part="heat")["payout_frequency"] == 0.0


def test_a_breakdown_has_the_identity_columns_and_its_own() -> None:
    table = payout_breakdown([_sum_cover()], _sample(), split="in_sample")
    assert table.column_names == [
        *IDENTITY_COLUMNS,
        "part",
        "part_kind",
        "expected_payout",
        "share",
        "payout_frequency",
    ]


def test_a_breakdown_writes_and_reads(tmp_path: Path) -> None:
    table = payout_breakdown([_sum_cover(), _max_cover()], _sample(), split="in_sample")
    path = tmp_path / "breakdown.parquet"
    write_comparison(table, path)
    read = pq.read_table(path)
    assert read.num_rows == 4
    metadata = EvaluationMetadata.from_table(read)
    assert set(metadata.columns) == {"expected_payout", "share", "payout_frequency"}
    assert metadata.columns["share"].unit == "1"
    assert metadata.setup.evaluations[0]["part_kinds"] == ["component", "attribution"]


def test_to_long_of_a_breakdown_keeps_each_part() -> None:
    table = payout_breakdown([_sum_cover()], _sample(), split="in_sample")
    long = to_long(table)
    assert long.num_rows == 2 * 3
    assert set(_column(long, "part")) == {"rain", "heat"}


def test_the_setup_says_a_max_of_breakdown_is_an_attribution() -> None:
    text = setup_text(payout_breakdown([_max_cover()], _sample(), split="in_sample"))
    assert "Payout breakdown: the rows of a MaxOf cover are an attribution" in text
    assert "a payout breakdown holds no premiums" in text
    plain = setup_text(payout_breakdown([_sum_cover()], _sample(), split="in_sample"))
    assert "attribution" not in plain


def test_a_breakdown_checks_the_split() -> None:
    train, test = holdout(_sample(), train_fraction=0.75)
    result = design(
        Cover(
            TWO,
            SumOf(
                {"rain": BinaryPayout(50.0, Free()), "heat": BinaryPayout(38.0, Free())}
            ),
            payout_unit="EUR",
            label="designed",
        ),
        train,
        MSE(),
    )
    table = payout_breakdown([result], test, split="out_of_sample")
    assert set(_column(table, "premium_basis")) == {"design"}
    assert set(_column(table, "design_id")) == {result.design_id}
    with pytest.raises(LeakageError):
        payout_breakdown([result], train, split="out_of_sample")


def test_a_breakdown_refuses_the_same_cover_twice() -> None:
    cover = _sum_cover()
    with pytest.raises(EvaluationError, match="repeat a row key"):
        payout_breakdown([cover, cover], _sample(), split="in_sample")


def test_a_breakdown_needs_one_asset() -> None:
    sample = OutcomeSample.join(
        [
            _one_hazard(),
            dataclasses.replace(_one_hazard(), asset_ids=("other",)),
        ]
    )
    with pytest.raises(SampleError, match="payout_breakdown takes one asset"):
        payout_breakdown([_rain_cover()], sample, split="in_sample")


def test_a_breakdown_takes_a_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        payout_breakdown(_sum_cover(), _sample(), split="in_sample")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="at least one"):
        payout_breakdown([], _sample(), split="in_sample")


def test_a_package_name_cannot_hold_a_separator() -> None:
    with pytest.raises(CoverDefinitionError, match="component name"):
        SumOf({"a+b": BinaryPayout(1.0, 1.0), "c": BinaryPayout(1.0, 1.0)})


def test_a_step_contract_counts_as_a_component() -> None:
    package = CoverPackage(
        {
            "rain": Cover(
                RAIN, StepPayout([50.0, 70.0], [2.0, 6.0]), payout_unit="EUR"
            ),
            "heat": _heat_cover(),
        }
    )
    table = payout_breakdown([package], _sample(), split="in_sample")
    assert _row(table, part="rain")["expected_payout"] == pytest.approx(
        (
            2.0 * ((RAIN_VALUES >= 50.0) & (RAIN_VALUES < 70.0))
            + 6.0 * (RAIN_VALUES >= 70.0)
        ).mean()
    )
