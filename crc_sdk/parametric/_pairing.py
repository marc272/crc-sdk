"""Whether a cover reads what a sample holds: one check for every route.

`design`, `design_joint`, `evaluate`, `compare`, `payout_breakdown` and the
functions built on them call `_check_reads` wherever a cover meets a sample.
It checks what `_check_cover_index` checks (the index unit, and on a sample of
several hazards each component's name, unit and tail), and then the hazard.

**The hazard.** A cover's `IndexDefinition.hazard_name` must be the name the
sample records for the series its index reads, compared exactly: crc-sdk has no
crosswalk of hazard names. That record is

- on a per-year sample with a separate index series (`index=`), the hazard that
  series' source names (its `recipe` or `hazard_name`, or a one-valued `recipe`
  column), since the index may read another hazard than the loss on purpose;
- on a sample of several hazards, each component's own record;
- on a sample of several assets, each asset's own;
- otherwise the sample's `provenance.hazard`. A pool records one when every run
  names it, and `sample_annual_pool` refuses runs that name two. A pool with a
  run that names none records none, and then each run that names one is
  compared.

A sample that records no hazard there is read as before: only its units are
compared, and the hazard is the caller's to know. A mismatch raises
`SampleError`, naming both hazards.

This module does no I/O.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from crc_sdk.parametric.errors import SampleError
from crc_sdk.parametric.index import CompositeIndex, IndexDefinition
from crc_sdk.parametric.sample import (
    OutcomeSample,
    SampleProvenance,
    _check_cover_index,
    _named_in_source,
)

_STATED_BY: Mapping[str, str] = {
    "curve_row": "its canonical row",
    "source_plan": "its source",
    "table_column": "its recipe column",
    "stated": "hazard_name=",
}
"""Who names a recorded hazard, as an error says it."""


def _check_reads(
    index: IndexDefinition | CompositeIndex,
    sample: OutcomeSample,
    asset: str | None = None,
) -> None:
    """Raise `SampleError` unless a cover on `index` reads what `sample` holds.

    `asset` is the asset whose column the cover reads, on a sample of several
    assets; `None` checks every asset.
    """
    _check_cover_index(index, sample)
    provenance = sample.provenance
    if sample.index_components is not None:
        components = provenance.components or {}
        wanted = (
            dict(index.components)
            if isinstance(index, CompositeIndex)
            else {index.hazard_name: index}
        )
        for name, definition in wanted.items():
            own = components.get(name)
            if own is not None:
                _compare(definition.hazard_name, own, f"component {name!r}")
        return
    assert isinstance(index, IndexDefinition)
    for owner, leaf in _leaves(provenance, asset):
        _compare(index.hazard_name, leaf, owner)


def _leaves(
    provenance: SampleProvenance, asset: str | None
) -> Iterator[tuple[str, SampleProvenance | Mapping[str, Any]]]:
    """Each asset's own record, with who it is; the sample's for one asset."""
    parts = provenance.parts
    if not parts:
        yield "the sample", provenance
    elif asset is not None:
        yield f"asset {asset!r}", parts.get(asset, provenance)
    else:
        for name, part in parts.items():
            yield f"asset {name!r}", part


def _field(leaf: SampleProvenance | Mapping[str, Any], name: str) -> Any:
    return leaf.get(name) if isinstance(leaf, Mapping) else getattr(leaf, name)


def _series_hazard(source: Mapping[str, Any]) -> str | None:
    """The hazard a series' source names: its `recipe` or `hazard_name`, or a column."""
    series = source.get("series")
    if not isinstance(series, Mapping):
        return None
    document = series.get("source")
    named = _named_in_source(document) if isinstance(document, Mapping) else None
    if named is not None:
        return named
    recorded = series.get("table_columns")
    column = recorded.get("recipe") if isinstance(recorded, Mapping) else None
    if not isinstance(column, Mapping) or column.get("distinct") != 1:
        return None
    values = column.get("values")
    return _named_in_source({"recipe": values[0]}) if values else None


def _compare(name: str, leaf: SampleProvenance | Mapping[str, Any], owner: str) -> None:
    """Raise `SampleError` unless `leaf` records `name` for what its index reads."""
    index_sources = [
        source
        for source in _field(leaf, "sources") or ()
        if source.get("role") == "index"
    ]
    if index_sources:
        for source in index_sources:
            found = _series_hazard(source)
            if found is not None and found != name:
                raise SampleError(
                    f"the cover pays on the hazard {name!r}, but the index series of "
                    f"{owner} reads {found!r}, as its source names it; hazard names "
                    "are compared exactly"
                )
        return
    record = _field(leaf, "hazard")
    if record is None:
        _compare_runs(name, leaf, owner)
        return
    if record["name"] == name:
        return
    stated_by = _STATED_BY.get(str(record["from"]), str(record["from"]))
    raise SampleError(
        f"the cover pays on the hazard {name!r}, but {owner} reads "
        f"{record['name']!r}, as {stated_by} names it; hazard names are compared "
        "exactly, so state the cover's hazard_name as the sample records it, or "
        "read the cover on a sample of its own hazard"
    )


def _compare_runs(
    name: str, leaf: SampleProvenance | Mapping[str, Any], owner: str
) -> None:
    """Raise `SampleError` unless every run of a pool that names a hazard names `name`.

    A pool records no hazard when one of its runs names none, but the runs
    that name one still say what their rows read.
    """
    for source in _field(leaf, "sources") or ():
        if source.get("role") != "hazard" or "model" not in source:
            continue
        found = _series_hazard(source)
        if found is not None and found != name:
            raise SampleError(
                f"the cover pays on the hazard {name!r}, but run {source['model']} "
                f"{source.get('member')} of {owner} reads {found!r}, as its source "
                "names it; hazard names are compared exactly"
            )
