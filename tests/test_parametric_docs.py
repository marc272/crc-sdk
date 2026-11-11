"""`docs/parametric.md` keeps up with the module.

- every public name of `crc_sdk.parametric` appears on the page, in code;
- every ```python block of the page runs, in page order, in one namespace, as a
  reader running the page from the top would run it.
"""

from __future__ import annotations

import math
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest

import crc_sdk.parametric as parametric
from crc_sdk.parametric import ComonotoneIndexWarning
from crc_sdk.workflows import ReturnPeriodExtrapolationWarning

DOCS = Path(__file__).resolve().parents[1] / "docs" / "parametric.md"


def _code(page: str) -> tuple[list[str], list[str]]:
    """The page's inline code spans, and its fenced blocks with their info.

    A span may run over a line break, as Markdown allows.
    """
    prose: list[str] = []
    blocks: list[str] = []
    fence: str | None = None
    lines: list[str] = []
    for line in page.splitlines():
        stripped = line.lstrip()
        if fence is not None:
            if stripped.startswith("```"):
                blocks.append(fence + "\n" + textwrap.dedent("\n".join(lines)) + "\n")
                fence, lines = None, []
            else:
                lines.append(line)
            continue
        if stripped.startswith("```"):
            fence = stripped[3:].strip()
            prose.append("")  # a fence ends any span
            continue
        prose.append(line)
    assert fence is None, "a fenced block of the page is not closed"
    spans = [
        " ".join(span.split())
        for paragraph in "\n".join(prose).split("\n\n")
        for span in re.findall(r"`([^`]+)`", paragraph)
    ]
    return spans, blocks


def _python_blocks(page: str) -> list[str]:
    """The ```python blocks of the page, in order."""
    _, blocks = _code(page)
    return [
        block.partition("\n")[2]
        for block in blocks
        if block.partition("\n")[0] == "python"
    ]


def test_every_public_name_is_on_the_page() -> None:
    spans, blocks = _code(DOCS.read_text(encoding="utf-8"))
    code = "\n".join([*spans, *blocks])
    missing = [
        name
        for name in parametric.__all__
        if not re.search(rf"(?<![\w.]){re.escape(name)}\b", code)
    ]
    assert not missing, (
        f"docs/parametric.md does not name {missing}: document each public name "
        "of crc_sdk.parametric in code on the page"
    )


def test_the_name_check_reads_code_only() -> None:
    page = (
        "`SumOf` and `MaxOf(scale=\nFree())` are named; StepGrid only in prose,\n"
        "and `sample.split_budget` is another name.\n"
        "```text\nCappedSum(cap)\n```\n"
    )
    spans, blocks = _code(page)
    assert spans == ["SumOf", "MaxOf(scale= Free())", "sample.split_budget"]
    code = "\n".join([*spans, *blocks])
    for name, named in (
        ("SumOf", True),
        ("MaxOf", True),
        ("Free", True),
        ("CappedSum", True),
        ("StepGrid", False),
        ("split_budget", False),
    ):
        assert bool(re.search(rf"(?<![\w.]){name}\b", code)) is named, name


def test_every_python_example_on_the_page_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocks = _python_blocks(DOCS.read_text(encoding="utf-8"))
    assert len(blocks) >= 20
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {}
    quick_start: Any = None
    table_start: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReturnPeriodExtrapolationWarning)
        warnings.simplefilter("ignore", ComonotoneIndexWarning)
        for number, block in enumerate(blocks, 1):
            try:
                code = compile(block, f"{DOCS}, python block {number}", "exec")
                exec(code, namespace)
            except Exception as error:  # pragma: no cover - the message is the point
                raise AssertionError(
                    f"python block {number} of docs/parametric.md fails: {error!r}\n"
                    f"{block}"
                ) from error
            # Later blocks reuse the names, so keep what the quick starts made.
            if number == 1:
                table_start = {
                    name: namespace[name]
                    for name in ("sample", "template", "comparison", "designed")
                }
            if quick_start is None and "result" in namespace:
                quick_start = namespace["result"]
    capsys.readouterr()

    # The table quick start: no semantics, no aggregation, and both rows held out.
    sample = table_start["sample"]
    assert (sample.semantics, sample.index_unit, sample.loss_unit) == (
        "annual",
        "mm/day",
        "EUR",
    )
    assert table_start["template"].index.aggregation is None
    comparison = table_start["comparison"]
    assert comparison.column("cover_label").to_pylist() == ["step", "linear"]
    assert set(comparison.column("split").to_pylist()) == {"out_of_sample"}
    assert set(comparison.column("n_rows").to_pylist()) == {12}
    assert comparison.column("premium_basis").to_pylist() == ["design", "evaluation"]
    assert table_start["designed"].budget_use == pytest.approx(1.0)

    # The quick start, as "No objective counts the premium" describes it: the
    # E[U] and weighted-CVaR designs spend their budget, the MSE design doesn't.
    mse, expected_uncovered, weighted_cvar = quick_start.designs
    assert expected_uncovered.budget_use == pytest.approx(1.0)
    assert weighted_cvar.budget_use == pytest.approx(1.0)
    assert mse.budget_use < 1.0

    # The holistic cover and the package, as the page says the rows are.
    table = namespace["table"]
    assert table.column("cover_label").to_pylist() == [
        "grid",
        "rain",
        "heat",
        "separate",
    ]
    assert table.column("level").to_pylist() == ["asset", "asset", "asset", "package"]
    assert table.column("components").to_pylist() == [
        "rx1day+txx",
        "rx1day",
        "txx",
        "rx1day+txx",
    ]
    assert set(table.column("split").to_pylist()) == {"out_of_sample"}
    assert set(table.column("n_rows").to_pylist()) == {12}
    assert len(set(table.column("expected_loss").to_pylist())) == 1  # combined loss
    premiums = table.column("premium").to_pylist()
    assert premiums[3] == pytest.approx(premiums[1] + premiums[2])
    budgets = table.column("design_budget").to_pylist()
    assert budgets[3] == pytest.approx(budgets[0])  # split_budget keeps the total

    # The parts of each entry add up to its expected payout.
    parts = namespace["parts"]
    for label in ("grid", "separate"):
        rows = [row for row in parts.to_pylist() if row["cover_label"] == label]
        (entry,) = [row for row in table.to_pylist() if row["cover_label"] == label]
        assert math.fsum(row["expected_payout"] for row in rows) == pytest.approx(
            entry["expected_payout"]
        )
    assert {row["part_kind"] for row in parts.to_pylist()} == {"region", "component"}

    # The curve route: the stated copula is recorded, and the cover is designed.
    drawn = namespace["drawn_table"]
    assert drawn.column("components").to_pylist() == ["rx1day+txx"]
    assert namespace["designed"].route == "lp"
    assert namespace["design_draws"].provenance.hazard_dependence["kind"] == (
        "gaussian_copula"
    )
