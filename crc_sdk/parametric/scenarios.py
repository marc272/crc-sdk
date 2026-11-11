"""Several scenarios side by side, and what a set of them refuses.

A scenario is one block of rows a cover can be evaluated on: draws from a curve
at one horizon or from a window of horizons, the years of a per-year series, or
the model-years of a pool. A loop over such blocks runs whatever it is given, so
a difference between two rows may come from the hazard or from anything else
that changed between them: the assets, the units, the loss model, the draws.
`ScenarioSet` holds the blocks and refuses a set in which anything but the hazard
differs. It says nothing about which scenario is more likely or which product is
better; it only keeps the comparison to what was stated.

`ScenarioError` names the two scenarios and the field. The refusals that need
the samples run at construction here, and at execution on a plan, through
`_check_scenarios`. Those that need only the labels run at construction.

`compare_scenarios(covers, scenarios, ...)` puts every cover or design on every
scenario's block in one table, out of sample only: how a fixed cover drifts
across scenarios, and a design per scenario, are slices of it. `scenario` says
where a row is evaluated and `design_scenario` where it was designed.
`to_drift(table, reference=)` adds each row's difference and ratio against the
row of the same cover or design on a reference scenario. Neither ranks anything:
a row is a measurement of a stated setup, and `setup_text` states it.

This module does no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import EvaluationError, SampleError, ScenarioError
from crc_sdk.parametric.evaluate import (
    Allocation,
    CoverPackage,
    EvaluationEntry,
    _allocation,
    _block_fields,
    _check_allocated,
    _check_keys,
    _CoverSet,
    _describe_rows,
    _entries,
    _Entry,
    _fold,
    _Item,
    _JointEntry,
    _metadata,
    _options,
    _prepared,
    _pricing,
    _rows,
    _Run,
    _sample,
    _schema,
    _split_step,
    _values,
    _values_hash,
    to_long,
)
from crc_sdk.parametric.index import (
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    CompositeIndex,
)
from crc_sdk.parametric.intervals import (
    _apply_premiums,
    _Arrays,
    _arrays,
    _block_read,
    _metrics,
    _no_premium_names,
    _short_block,
)
from crc_sdk.parametric.metadata import (
    EVALUATION_METADATA_KEY,
    KEY_COLUMNS,
    EvaluationMetadata,
    _replaced,
)
from crc_sdk.parametric.metrics import (
    MetricColumn,
    MetricOptions,
    _with_repriced,
    metric_columns,
)
from crc_sdk.parametric.pricing import Pricing, PurePremium
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _check_cover_index,
    _check_known_pathway,
    _identifier,
    _pathway_statements,
    _share_a_source,
    _source_columns,
    _text,
    _trend_of,
    _year,
)
from crc_sdk.pathways import pathways_equivalent
from crc_sdk.types import TemporalWindow
from crc_sdk.workflows.distributions import warn_if_outside_window

__all__ = ["ScenarioSet", "compare_scenarios", "to_drift"]

_Pair = tuple[str | None, int | None]
"""A (pathway, horizon) as a record states it; either may be null."""

_Members = tuple[tuple[str, int], ...]
"""A scenario's (pathway, horizon) pairs, in horizon order; one for one horizon."""

_FIXED = "; a set fixes everything but the hazard's values"


# -- what a sample's records say -----------------------------------------------


@dataclass(frozen=True, eq=False)
class _Scenario:
    """One scenario of a set and what its records say it reads."""

    sample: OutcomeSample
    document: Mapping[str, Any]
    leaves: tuple[tuple[str, Mapping[str, Any]], ...]
    """Each asset's or component's own provenance document, with who it is."""
    product: str | None
    members: tuple[_Pair, ...]
    """A window's members, or the one (pathway, horizon) of anything else."""

    @property
    def label(self) -> str:
        return self.sample.scenario


def _leaves(
    document: Mapping[str, Any], owner: str
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Each asset's or component's own provenance, or the sample's if it has one."""
    for key, kind in (("parts", "asset"), ("components", "component")):
        nested = document.get(key)
        if nested:
            return tuple((f"{kind} {name!r}", own) for name, own in nested.items())
    return ((owner, document),)


def _read(sample: OutcomeSample) -> _Scenario:
    """What `sample`'s records say, after a strict read of them."""
    columns = _source_columns(sample, strict=True)
    document = sample.provenance.to_document()
    window = (document.get("details") or {}).get("window")
    members: tuple[_Pair, ...]
    if window is None:
        members = ((columns.pathway, columns.horizon),)
    else:
        members = tuple((_text(pair[0]), _year(pair[1])) for pair in window["members"])
        for pathway, _ in members:
            if pathway is not None:
                _check_known_pathway(pathway, sample.scenario)
    return _Scenario(
        sample,
        document,
        _leaves(document, f"asset {sample.asset_ids[0]!r}"),
        columns.product,
        members,
    )


def _curve_sources(leaf: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The curves a leaf reads: one, or one per member of a window."""
    return [
        source
        for source in leaf.get("sources") or ()
        if source.get("role") == "hazard" and source.get("kind") == "curve"
    ]


def _source_pair(source: Mapping[str, Any]) -> _Pair:
    """The (pathway, horizon) a curve's own row states."""
    row = source.get("row") or {}
    return _text(row.get("pathway")), _year(row.get("horizon"))


def _same_pair(first: _Pair, second: _Pair) -> bool:
    """Whether two pairs name one pathway and horizon, null equal to null."""
    if first[1] != second[1]:
        return False
    if first[0] is None or second[0] is None:
        return first[0] is None and second[0] is None
    return pathways_equivalent(first[0], second[0])


def _same_members(first: Sequence[_Pair], second: Sequence[_Pair]) -> bool:
    return len(first) == len(second) and all(
        _same_pair(ours, theirs) for ours, theirs in zip(first, second, strict=True)
    )


# -- the comparisons -----------------------------------------------------------


def _differ(
    first: _Scenario, second: _Scenario, field: str, detail: str
) -> ScenarioError:
    return ScenarioError(
        f"scenarios {first.label!r} and {second.label!r} differ in {field}: {detail}"
    )


def _require_same(
    scenarios: Sequence[_Scenario],
    field: str,
    read: Callable[[_Scenario], Any],
    *,
    show: bool = True,
) -> None:
    """Raise `ScenarioError` unless `read` gives every scenario the reference's."""
    first = scenarios[0]
    ours = read(first)
    for other in scenarios[1:]:
        theirs = read(other)
        if theirs != ours:
            detail = (
                f"{first.label!r} has {ours!r} and {other.label!r} has {theirs!r}"
                if show
                else "their records are not the same"
            )
            raise _differ(first, other, field, detail + _FIXED)


def _per_asset(scenario: _Scenario, key: str) -> dict[str, Any]:
    """A document of `scenario` by asset: in `parts` on a sample of several assets."""
    parts = scenario.document.get("parts")
    if parts:
        return {f"asset {name!r}": own.get(key) for name, own in parts.items()}
    return {f"asset {scenario.sample.asset_ids[0]!r}": scenario.document.get(key)}


def _require_same_per_asset(scenarios: Sequence[_Scenario], field: str) -> None:
    """Raise `ScenarioError` unless each asset's `field` is the reference's."""
    first = scenarios[0]
    ours = _per_asset(first, field)
    for other in scenarios[1:]:
        theirs = _per_asset(other, field)
        for owner, document in ours.items():
            if theirs[owner] != document:
                raise _differ(
                    first,
                    other,
                    field,
                    f"the documents of {owner} are not the same{_FIXED}",
                )


def _check_one_kind(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for curve draws beside per-year rows."""

    def kind(scenario: _Scenario) -> str:
        return scenario.sample.provenance.source_kind

    first = scenarios[0]
    for other in scenarios[1:]:
        if (kind(other) == "monte_carlo") != (kind(first) == "monte_carlo"):
            raise _differ(
                first,
                other,
                "source_kind",
                f"{first.label!r} is {kind(first)!r} and {other.label!r} is "
                f"{kind(other)!r}; a set holds all curves or all per-year rows",
            )


def _components_of(scenario: _Scenario) -> Mapping[str, Any]:
    return scenario.sample.index_components or {}


def _check_components(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for samples of several hazards that differ in them."""
    _require_same(
        scenarios, "components", lambda scenario: list(_components_of(scenario))
    )
    if not _components_of(scenarios[0]):
        return
    for field in ("unit", "tail"):
        _require_same(
            scenarios,
            f"each component's {field}",
            lambda scenario: {
                name: getattr(component, field)
                for name, component in _components_of(scenario).items()
            },
        )
    _require_same(
        scenarios,
        "combine",
        lambda scenario: scenario.document.get("combine"),
        show=False,
    )


def _reads_no_loss(own: Mapping[str, Any]) -> bool:
    """Whether a component has no impact; a table doesn't say what its loss reads."""
    return own.get("source_kind") != "table" and own.get("loss_model") is None


def _name_in(document: Mapping[str, Any]) -> str | None:
    hazard = document.get("hazard")
    return None if hazard is None else str(hazard["name"])


def _hazard_names(scenario: _Scenario) -> dict[str, str | None]:
    """The hazard each series its loss reads names: by component, or one entry."""
    components = scenario.document.get("components")
    if components is None:
        return {"the sample": _name_in(scenario.document)}
    return {
        f"component {name!r}": _name_in(own)
        for name, own in components.items()
        if not _reads_no_loss(own)
    }


def _check_hazards(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for a hazard not named, or two hazards, by exact name."""
    for scenario in scenarios:
        for owner, name in _hazard_names(scenario).items():
            if name is None:
                raise ScenarioError(
                    f"scenario {scenario.label!r} names no hazard_name for {owner}: "
                    "every scenario of a set names the hazard its loss reads; its "
                    "source states none, so state hazard_name="
                )
    _require_same(scenarios, "hazard_name", _hazard_names)


def _check_stated_pathways(scenario: _Scenario) -> None:
    """Raise `ScenarioError` for a stated pathway its source contradicts."""
    for owner, leaf in scenario.leaves:
        for source in leaf.get("sources") or ():
            if source.get("role") != "hazard":
                continue
            stated = _pathway_statements(source)
            for place, label in stated[1:]:
                if not pathways_equivalent(stated[0][1], label):
                    raise ScenarioError(
                        f"scenario {scenario.label!r} contradicts itself on the "
                        f"pathway of {owner}: {stated[0][0]} states {stated[0][1]!r} "
                        f"and {place} states {label!r}; a stated pathway must agree "
                        "with its source"
                    )


def _check_distinct_reads(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for two scenarios of one pathway, horizon and product."""
    for i, first in enumerate(scenarios):
        for second in scenarios[i + 1 :]:
            if first.product == second.product and _same_members(
                first.members, second.members
            ):
                raise ScenarioError(
                    f"scenarios {first.label!r} and {second.label!r} read one pathway, "
                    f"horizon and product: {list(first.members)} of "
                    f"{first.product!r}; a set holds each once"
                )


def _check_products(scenarios: Sequence[_Scenario], mixed_products: bool) -> None:
    """Raise `ScenarioError` for two products in a set, unless they are stated."""
    if mixed_products:
        return
    first = scenarios[0]
    for other in scenarios[1:]:
        if other.product != first.product:
            raise _differ(
                first,
                other,
                "product",
                f"{first.label!r} reads {first.product!r} and {other.label!r} reads "
                f"{other.product!r}, so a difference between them includes the "
                "products; pass mixed_products=True to state that",
            )


# -- per-year scenarios --------------------------------------------------------


def _trend_option(scenario: _Scenario) -> dict[str, Any]:
    """The trend option as stated; a reference year and fitted slopes may differ."""
    trend = _trend_of(scenario.sample)
    return {
        key: trend[key] for key in ("kind", "n", "method", "min_years") if key in trend
    }


def _pool_models(scenario: _Scenario) -> frozenset[tuple[str, str]]:
    pool = scenario.document["pool"]
    return frozenset((str(run["model"]), str(run["member"])) for run in pool["runs"])


def _check_pools(scenarios: Sequence[_Scenario], mixed_products: bool) -> None:
    """Raise `ScenarioError` unless pools hold one set of models."""
    pools = [scenario for scenario in scenarios if scenario.document.get("pool")]
    if not pools:
        return
    plain = [scenario for scenario in scenarios if not scenario.document.get("pool")]
    if plain and not mixed_products:
        raise _differ(
            pools[0],
            plain[0],
            "pool",
            f"{pools[0].label!r} pools model runs and {plain[0].label!r} is not a "
            "pool; pass mixed_products=True to state that",
        )
    _require_same(pools, "pool's (model, member) pairs", _pool_models)


_SLACK = 1.0 + 1e-9
"""The slack `AnnualSeries.nearest_cell` allows on half a grid step."""

_STATION_COLUMNS = ("station", "station_id")
"""The columns that name a station, where a series names no station column."""


@dataclass(frozen=True)
class _Place:
    """Where one per-year series is read: a matched grid cell, or its stations.

    A cell is its centre and its grid step, `(latitude, longitude)` in degrees;
    a record without a step states a point, a step of 0.
    """

    kind: str
    centre: tuple[float, float] = (0.0, 0.0)
    step: tuple[float, float] = (0.0, 0.0)
    stations: tuple[str, ...] = ()

    def __str__(self) -> str:
        if self.kind == "station":
            noun = "station" if len(self.stations) == 1 else "stations"
            return f"at {noun} {', '.join(repr(each) for each in self.stations)}"
        latitude, longitude = self.centre
        text = (
            f"in the grid cell centred at latitude {latitude:g}, "
            f"longitude {longitude:g}"
        )
        if any(self.step):
            text += f" (step {self.step[0]:g} by {self.step[1]:g} degrees)"
        return text


def _station_ids(series: Mapping[str, Any]) -> tuple[str, ...]:
    """The stations a series reads, from its station filter alone.

    A series that names its station column (`from_table(station=)`, which a
    plan's `StationColumn` states) reads the station of that filter, whatever
    the column is called. Otherwise a `station` or `station_id` filter or
    one-valued column is the station. A filter on any other column, such as
    `model` or `scenario`, names no station.
    """
    where = series.get("where") or {}
    read = series.get("columns")
    named = read.get("station") if isinstance(read, Mapping) else None
    if isinstance(named, str):
        return (str(where[named]),) if named in where else ()
    found = {str(value) for name, value in where.items() if name in _STATION_COLUMNS}
    columns = series.get("table_columns") or {}
    for name in _STATION_COLUMNS:
        column = columns.get(name) or {}
        if column.get("distinct") == 1:
            found.update(str(value) for value in column.get("values") or ())
    return tuple(sorted(found))


def _series_place(series: Mapping[str, Any]) -> _Place | None:
    """Where a series' record says it is read; `None` if it says neither."""
    cell = series.get("cell")
    if isinstance(cell, Mapping):
        spacing = series.get("spacing_degrees") or {}
        return _Place(
            "cell",
            centre=(float(cell["latitude"]), float(cell["longitude"])),
            step=(
                float(spacing.get("latitude") or 0.0),
                float(spacing.get("longitude") or 0.0),
            ),
        )
    stations = _station_ids(series)
    return _Place("station", stations=stations) if stations else None


_Read = tuple[_Place, str]
"""A place, and whose series is read there, as an error names it."""


def _series_places(scenario: _Scenario) -> dict[tuple[str, str], list[_Read]]:
    """Each asset's or component's places, by owner and role: hazard or index."""
    held: dict[tuple[str, str], list[_Read]] = {}
    for owner, leaf in scenario.leaves:
        for source in leaf.get("sources") or ():
            role, series = source.get("role"), source.get("series")
            if role not in ("hazard", "index") or not isinstance(series, Mapping):
                continue
            place = _series_place(series)
            if place is None:
                continue
            whose = f"the {role} series of {owner}"
            if "model" in source:
                whose += f" (run {source['model']} {source.get('member')})"
            own = held.setdefault((owner, str(role)), [])
            if all(place != each for each, _ in own):
                own.append((place, whose))
    return held


def _offsets(first: _Place, second: _Place) -> tuple[float, float]:
    """`second`'s centre less `first`'s, in degrees; longitudes modulo 360."""
    d_latitude = second.centre[0] - first.centre[0]
    d_longitude = (second.centre[1] - first.centre[1] + 180.0) % 360.0 - 180.0
    return d_latitude, d_longitude


def _one_grid(first: _Place, second: _Place) -> bool:
    """Whether two cells are of one grid: one step, centres whole steps apart."""
    for ours, theirs, offset in zip(
        first.step, second.step, _offsets(first, second), strict=True
    ):
        if ours <= 0.0 or not math.isclose(ours, theirs, rel_tol=1e-9):
            return False
        steps = offset / ours
        if abs(steps - round(steps)) > 1e-6:
            return False
    return True


def _one_series_place(first: _Place, second: _Place) -> bool:
    """Whether two places of per-year series may be one; another kind isn't compared.

    Stations must be the same. Cells of one grid must be the same cell, since a
    point matches one cell of a grid. Cells of two grids must overlap: some
    point must lie within half a grid step of both centres, each by its own
    step, as `AnnualSeries.nearest_cell` matches a cell to a point.
    """
    if first.kind != second.kind:
        return True
    if first.kind == "station":
        return first.stations == second.stations
    offsets = _offsets(first, second)
    if _one_grid(first, second):
        return all(
            abs(offset) < 0.5 * step
            for offset, step in zip(offsets, first.step, strict=True)
        )
    return all(
        abs(offset) <= 0.5 * (ours + theirs) * _SLACK + 1e-9
        for offset, ours, theirs in zip(offsets, first.step, second.step, strict=True)
    )


def _check_places(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for a per-year series of one asset read in two places.

    Each asset's or component's hazard series, and its index series, must be
    read at one place in every scenario, compared pair by pair, like with like:
    a matched cell with a matched cell, stations with stations. A series whose
    record names neither isn't compared.
    """
    read = [(scenario, _series_places(scenario)) for scenario in scenarios]
    for i, (first, ours) in enumerate(read):
        for second, theirs in read[i + 1 :]:
            for key, places in ours.items():
                for place, whose in places:
                    # The same series first, so an error names one run where it can.
                    for other, other_whose in sorted(
                        theirs.get(key, ()), key=lambda read: read[1] != whose
                    ):
                        if _one_series_place(place, other):
                            continue
                        reads = "" if other_whose == whose else f" reads {other_whose}"
                        raise _differ(
                            first,
                            second,
                            place.kind,
                            f"{first.label!r} reads {whose} {place}, and "
                            f"{second.label!r}{reads} {other}"
                            f"{_place_reason(place, other)}{_FIXED}",
                        )


def _place_reason(first: _Place, second: _Place) -> str:
    """Why two places of one kind are two."""
    if first.kind == "station":
        return ""
    if _one_grid(first, second):
        return ": two cells of one grid, where a point matches one"
    return ": no point lies within half a grid step of both centres"


def _check_years(scenarios: Sequence[_Scenario], mixed_products: bool) -> None:
    _require_same(scenarios, "trend", _trend_option)
    _check_pools(scenarios, mixed_products)
    _check_places(scenarios)


# -- curve scenarios -----------------------------------------------------------


def _check_window_pathway(label: str, members: Sequence[_Pair]) -> None:
    """Raise `ScenarioError` unless a window's members are under one pathway."""
    if not all(
        _same_pair((pathway, None), (members[0][0], None)) for pathway, _ in members
    ):
        pathways = list(dict.fromkeys(pathway for pathway, _ in members))
        raise ScenarioError(
            f"scenario {label!r} is a window, which holds one pathway, but its "
            f"members are under {pathways}; state one scenario per pathway"
        )


def _check_window_horizons(label: str, members: Sequence[_Pair]) -> None:
    """Raise `ScenarioError` for a horizon a window holds twice."""
    horizons = [horizon for _, horizon in members]
    repeated = sorted(
        {year for year in horizons if year is not None and horizons.count(year) > 1}
    )
    if repeated:
        raise ScenarioError(
            f"scenario {label!r} is a window, whose members differ in horizon, but "
            f"the horizons {repeated} repeat"
        )


def _check_window(scenario: _Scenario) -> None:
    """Raise `ScenarioError` unless a window is one curve at several horizons."""
    details = scenario.document.get("details") or {}
    if "window" not in details:
        return
    label, members = scenario.label, scenario.members
    _check_window_pathway(label, members)
    _check_window_horizons(label, members)
    draws = details.get("draws")
    if isinstance(draws, int) and draws % len(members):
        raise ScenarioError(
            f"scenario {label!r} is a window of {len(members)} horizons, so its "
            f"n_draws must be a whole multiple of {len(members)}, got {draws}"
        )
    reads: tuple[tuple[str, Callable[[Mapping[str, Any]], Any]], ...] = (
        ("hazard_name", lambda source: (source.get("row") or {}).get("hazard_name")),
        ("cell_index", lambda source: (source.get("row") or {}).get("cell_index")),
        ("dataset", lambda source: source.get("metadata")),
    )
    for owner, leaf in scenario.leaves:
        sources = _curve_sources(leaf)
        for field, read in reads:
            if any(read(source) != read(sources[0]) for source in sources[1:]):
                raise ScenarioError(
                    f"scenario {label!r} is a window, whose members differ in horizon "
                    f"alone, but those of {owner} differ in {field}"
                )


def _check_pairs_held(
    label: str, pairs: Sequence[_Pair], held: Mapping[str, Sequence[_Pair]]
) -> None:
    """Raise `ScenarioError` unless every owner holds every (pathway, horizon).

    `held` maps each asset or hazard to the pairs its data holds. A window never
    shrinks to the members it finds.
    """
    for owner, own in held.items():
        for pair in pairs:
            if not any(_same_pair(pair, other) for other in own):
                raise ScenarioError(
                    f"scenario {label!r} reads the (pathway, horizon) pair {pair}, "
                    f"which the data of {owner} doesn't hold; it holds {list(own)}"
                )


def _check_horizons_in_window(scenario: _Scenario) -> None:
    """Raise `ScenarioError` for a horizon outside its dataset's temporal window."""
    for owner, leaf in scenario.leaves:
        for source in _curve_sources(leaf):
            horizon = _source_pair(source)[1]
            stated = (source.get("metadata") or {}).get("temporal_window")
            if horizon is None or stated is None:
                continue
            window = TemporalWindow.model_validate(stated)
            try:
                warn_if_outside_window(horizon, window, strict=True)
            except ValueError as error:
                raise ScenarioError(
                    f"scenario {scenario.label!r} reads a horizon outside the "
                    f"temporal_window of {owner}: {error}"
                ) from error


_RESOLUTION_SHIFT = 52
"""Where an H3 index holds its resolution, in 4 bits."""


def _h3_resolution(cell: int) -> int:
    return (cell >> _RESOLUTION_SHIFT) & 0xF


def _one_place(first: str, second: str) -> bool:
    """Whether two H3 cells, as hexadecimal text, hold one place.

    At one resolution they must be equal. A point's cells at two resolutions
    don't always nest: the finer cell's parent at the coarser resolution is the
    coarser cell or one of its neighbours. So cells at two resolutions are
    compared at the coarser one, with that slack, which needs the `geometry`
    extra.
    """
    ours, theirs = int(first, 16), int(second, 16)
    if ours == theirs:
        return True
    if _h3_resolution(ours) == _h3_resolution(theirs):
        return False
    try:
        import h3  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError(
            "comparing the cells of curves at two H3 resolutions requires "
            "`pip install crc-sdk[geometry]`"
        ) from error
    coarse, fine = sorted((ours, theirs), key=_h3_resolution)
    coarse_cell, fine_cell = h3.int_to_str(coarse), h3.int_to_str(fine)
    if not (h3.is_valid_cell(coarse_cell) and h3.is_valid_cell(fine_cell)):
        return False
    parent = h3.cell_to_parent(fine_cell, _h3_resolution(coarse))
    return bool(coarse_cell in h3.grid_disk(parent, 1))


def _scenario_cells(
    scenarios: Sequence[_Scenario],
) -> dict[str, list[tuple[_Scenario, str]]]:
    """Each asset's or hazard's distinct curve cells, each with its first scenario."""
    held: dict[str, list[tuple[_Scenario, str]]] = {}
    for scenario in scenarios:
        for owner, leaf in scenario.leaves:
            for source in _curve_sources(leaf):
                cell = (source.get("row") or {}).get("cell_index")
                if not isinstance(cell, str):
                    continue
                own = held.setdefault(owner, [])
                if all(int(cell, 16) != int(seen, 16) for _, seen in own):
                    own.append((scenario, cell))
    return held


def _check_cells(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for an asset or hazard read in two places.

    Each asset's or hazard's curves must come from one place in every scenario,
    compared pair by pair: with the first scenario that records a cell alone, a
    coarse-resolution reference could hold two finer cells of two other
    scenarios that are two places. A curve given as a distribution or as
    `CurveParameters` records no cell, and isn't compared.
    """
    for owner, cells in _scenario_cells(scenarios).items():
        for i, (first, ours) in enumerate(cells):
            for second, theirs in cells[i + 1 :]:
                if _one_place(ours, theirs):
                    continue
                levels = {
                    _h3_resolution(int(ours, 16)),
                    _h3_resolution(int(theirs, 16)),
                }
                slack = (
                    ", which at two resolutions must nest up to one neighbouring cell"
                    if len(levels) > 1
                    else ""
                )
                raise _differ(
                    first,
                    second,
                    "cell_index",
                    f"{first.label!r} reads {owner} in the H3 cell {ours!r} and "
                    f"{second.label!r} in {theirs!r}{slack}" + _FIXED,
                )


@dataclass(frozen=True, eq=False)
class _Curve:
    """One distinct curve of a set, for one asset or hazard."""

    label: str
    pair: _Pair
    product: str | None
    document: Mapping[str, Any]


def _check_equal_curves(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` for equal curves at two (pathway, horizon) pairs.

    Each distinct pair of the set counts once for any one asset or hazard, so a
    window over a time-invariant layer raises and two windows don't for the
    horizons they share. A curve of a kind the record can't describe isn't
    compared.
    """
    held: dict[str, list[_Curve]] = {}
    for scenario in scenarios:
        for owner, leaf in scenario.leaves:
            own = held.setdefault(owner, [])
            for source in _curve_sources(leaf):
                pair = _source_pair(source)
                if not any(
                    curve.product == scenario.product and _same_pair(curve.pair, pair)
                    for curve in own
                ):
                    own.append(
                        _Curve(scenario.label, pair, scenario.product, source["curve"])
                    )
    for owner, curves in held.items():
        for i, first in enumerate(curves):
            for second in curves[i + 1 :]:
                if (
                    not _same_pair(first.pair, second.pair)
                    and first.document == second.document
                    and first.document["kind"] != "other"
                ):
                    names = (
                        f"scenario {first.label!r}"
                        if first.label == second.label
                        else f"scenarios {first.label!r} and {second.label!r}"
                    )
                    raise ScenarioError(
                        f"{names} read an equal curve for {owner} at {first.pair} and "
                        f"{second.pair}: a difference between them would measure "
                        "nothing of the scenarios"
                    )


def _hazard_generators(scenario: _Scenario) -> dict[str, list[tuple[Any, Any]]]:
    """Each asset's or hazard's generator record of its `hazard` step."""
    return {
        owner: [
            (record["bit_generator"], record["state"])
            for record in leaf.get("rng") or ()
            if record.get("step") == "hazard"
        ]
        for owner, leaf in scenario.leaves
    }


def _check_shared_draws(scenarios: Sequence[_Scenario]) -> None:
    """Raise `ScenarioError` unless the curve scenarios share one set of draws."""
    first = scenarios[0]
    ours = _hazard_generators(first)
    for other in scenarios[1:]:
        if not np.array_equal(first.sample.period, other.sample.period):
            raise _differ(
                first,
                other,
                "period",
                "curve scenarios hold the same draw numbers, row by row",
            )
        theirs = _hazard_generators(other)
        for owner, records in ours.items():
            if not records or theirs[owner] != records:
                raise _differ(
                    first,
                    other,
                    "rng",
                    f"the hazard step of {owner} must have one generator record in "
                    "every curve scenario; build each block with a new generator "
                    "from one seed sequence, as reusing one Generator advances it",
                )


def _model_document(document: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """A dependence record without what depends on a window's number of members.

    An `EmpiricalCopula`'s `k` counts one member's draws per reference year, so a
    window beside a single horizon on the same draws records two values of it.
    """
    if document is None or document.get("kind") != "empirical_copula":
        return document
    return {key: value for key, value in document.items() if key != "k"}


def _check_curves(scenarios: Sequence[_Scenario]) -> None:
    for scenario in scenarios:
        _check_window(scenario)
        _check_pairs_held(
            scenario.label,
            scenario.members,
            {
                owner: [_source_pair(source) for source in _curve_sources(leaf)]
                for owner, leaf in scenario.leaves
            },
        )
        _check_horizons_in_window(scenario)
    _check_cells(scenarios)
    _check_equal_curves(scenarios)
    _check_shared_draws(scenarios)
    for field in ("dependence", "hazard_dependence"):
        _require_same(
            scenarios,
            field,
            lambda scenario: _model_document(scenario.document.get(field)),
            show=False,
        )


# -- the one function ----------------------------------------------------------


def _reference_first(
    samples: Iterable[OutcomeSample], reference: str
) -> list[OutcomeSample]:
    """`samples` with the reference first and the others as given."""
    return sorted(samples, key=lambda sample: sample.scenario != reference)


def _check_scenarios(
    samples: Sequence[OutcomeSample], *, reference: str, mixed_products: bool
) -> None:
    """Raise `ScenarioError` for samples that can't stand side by side.

    Every refusal that needs the samples, for a set of built samples at
    construction and for a plan at execution. Each compares a scenario with the
    reference, or each two scenarios where that isn't enough.
    """
    labels = [sample.scenario for sample in samples]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise ScenarioError(
            f"two samples are labelled {repeated[0]!r}: each scenario has its own "
            "label, so name each with scenario="
        )
    if reference not in labels:
        raise ScenarioError(
            f"the reference {reference!r} must be one of the set's scenarios, {labels}"
        )
    scenarios = [_read(sample) for sample in _reference_first(samples, reference)]
    _check_one_kind(scenarios)
    _require_same(scenarios, "asset_ids", lambda scenario: scenario.sample.asset_ids)
    _check_components(scenarios)
    _require_same(scenarios, "index_unit", lambda scenario: scenario.sample.index_unit)
    _require_same(scenarios, "loss_unit", lambda scenario: scenario.sample.loss_unit)
    _require_same_per_asset(scenarios, "loss_model")
    _require_same_per_asset(scenarios, "basis_risk")
    _check_hazards(scenarios)
    for scenario in scenarios:
        _check_stated_pathways(scenario)
    _check_distinct_reads(scenarios)
    _check_products(scenarios, mixed_products)
    if scenarios[0].sample.provenance.source_kind == "monte_carlo":
        _check_curves(scenarios)
    else:
        _check_years(scenarios, mixed_products)


# -- a bound cover ---------------------------------------------------------------


def _reads_at(location: Any, source: Mapping[str, Any]) -> bool:
    """Whether one source of an index is read where `location` says."""
    # A plan holds a set, so plan.py can't be imported as this module loads.
    from crc_sdk.parametric.plan import _same_location, _wrapped_longitude

    if isinstance(location, AtCell):
        cell = (source.get("row") or {}).get("cell_index")
        return (
            source.get("kind") == "curve"
            and isinstance(cell, str)
            and int(cell, 16) == int(location.cell_index, 16)
        )
    series = source.get("series")
    if not isinstance(series, Mapping):
        return False
    if isinstance(location, AtStation):
        return location.station_id in _station_ids(series)
    cell = series.get("cell")
    if not isinstance(cell, Mapping):
        return False
    matched = AtPoint(
        _wrapped_longitude(float(cell["longitude"])), float(cell["latitude"])
    )
    return _same_location(location, matched)


def _check_location(
    cover: Cover, label: str, owner: str, location: Any, leaf: Mapping[str, Any]
) -> None:
    """Raise `ScenarioError` unless every source of `leaf`'s index is at `location`."""
    if isinstance(location, AtAsset):
        return
    held = leaf.get("sources") or ()
    sources = [source for source in held if source.get("role") == "index"] or [
        source for source in held if source.get("role") == "hazard"
    ]
    if not sources or not all(_reads_at(location, source) for source in sources):
        stations = (
            "; a series names its station in a station or station_id filter or "
            "one-valued column, or in the filter that "
            "AnnualSeries.from_table(station=) names, and a filter on another column "
            "names none"
            if isinstance(location, AtStation)
            else ""
        )
        raise ScenarioError(
            f"scenario {label!r} and cover {cover.label or cover.definition_hash!r} "
            f"differ in location: the cover is read at {location!r}, which the index "
            f"of {owner} does not read{stations}"
        )


def _check_bound_cover(cover: Cover, sample: OutcomeSample) -> None:
    """Raise `ScenarioError` unless `sample`'s index is read where `cover` is bound.

    `AtAsset()` reads any scenario's own index at the asset. `AtPoint` needs a
    per-year series whose matched cell is centred there, compared as
    `.index(...)` compares a bound location, `AtStation` a per-year series of
    that station, read from its station filter alone (a filter on `model` or
    `scenario` names no station), and `AtCell` a curve at that cell, every
    member of a window.
    A sample that holds several assets is read at the cover's, or at each if it
    names none, and a sample of several hazards at each component the cover
    reads. What the cover can't read at all is `_check_cover_index`'s error.
    """
    index = cover.index_definition
    _check_cover_index(index, sample)
    document = sample.provenance.to_document()
    label = sample.scenario
    if sample.index_components is not None:
        wanted = (
            dict(index.components)
            if isinstance(index, CompositeIndex)
            else {index.hazard_name: index}
        )
        components = document.get("components") or {}
        for name, definition in wanted.items():
            _check_location(
                cover,
                label,
                f"component {name!r}",
                definition.location,
                components.get(name, document),
            )
        return
    assert not isinstance(index, CompositeIndex)
    parts = document.get("parts")
    assets = sample.asset_ids if cover.asset_id is None else (cover.asset_id,)
    for asset in assets:
        leaf = document if not parts else parts.get(asset, document)
        _check_location(cover, label, f"asset {asset!r}", index.location, leaf)


# -- a plan's labels -------------------------------------------------------------


def _checked_pairs(label: str, given: Any) -> _Members:
    """A scenario's (pathway, horizon) pairs, in horizon order, after their refusals."""
    if isinstance(given, (str, bytes, Mapping)) or not isinstance(given, Iterable):
        raise TypeError(
            f"labels[{label!r}] must be an iterable of (pathway, horizon) pairs, got "
            f"{type(given).__name__}"
        )
    pairs: list[tuple[str, int]] = []
    for pair in given:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(
                f"labels[{label!r}] must hold (pathway, horizon) pairs, got {pair!r}"
            )
        pathway, horizon = pair
        if not isinstance(pathway, str):
            raise TypeError(
                f"labels[{label!r}]: a pathway must be a string, got {pathway!r}"
            )
        year = _year(horizon)
        if year is None:
            raise TypeError(
                f"labels[{label!r}]: a horizon must be a year, got {horizon!r}"
            )
        _check_known_pathway(pathway, label)
        pairs.append((pathway, year))
    if not pairs:
        raise ScenarioError(f"scenario {label!r} lists no (pathway, horizon) pair")
    _check_window_pathway(label, pairs)
    _check_window_horizons(label, pairs)
    return tuple(sorted(pairs, key=lambda pair: pair[1]))


def _checked_labels(labels: Any, reference: str) -> Mapping[str, _Members]:
    """A plan's labels, after the refusals that need only the labels."""
    if not isinstance(labels, Mapping):
        raise TypeError(
            "labels must map each label to its (pathway, horizon) pairs, got "
            f"{type(labels).__name__}"
        )
    held = {
        _checked_text("a scenario's label", label): _checked_pairs(label, given)
        for label, given in labels.items()
    }
    if reference not in held:
        raise ScenarioError(
            f"the reference {reference!r} must be one of the set's scenarios, "
            f"{list(held)}"
        )
    names = list(held)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if _same_members(held[first], held[second]):
                raise ScenarioError(
                    f"scenarios {first!r} and {second!r} read one pathway, horizon "
                    f"and product: {list(held[first])}; a set holds each once"
                )
    return MappingProxyType(held)


def _checked_text(name: str, value: Any) -> str:
    """`value` as a label; `TypeError` for anything but a string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    try:
        return _identifier(name, value)
    except SampleError as error:
        raise ScenarioError(str(error)) from error


def _checked_entries(entries: Any) -> tuple[OutcomeSample, ...]:
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Iterable):
        raise TypeError(
            "entries must be an iterable of OutcomeSample, got "
            f"{type(entries).__name__}"
        )
    held = tuple(entries)
    for entry in held:
        if not isinstance(entry, OutcomeSample):
            raise TypeError(
                f"entries must hold OutcomeSample, got {type(entry).__name__}"
            )
    return held


# -- the set ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False, init=False, repr=False)
class ScenarioSet:
    """One evaluation block per scenario, under a named reference.

    `entries` holds one `OutcomeSample` per scenario, and a sample's label is its
    own `scenario`, so a sample keeps one name. `reference` has no default and is
    the label of one scenario: the module chooses none. A set may hold the
    reference alone. `mixed_products=True` states that a set holds two products,
    the reference included, or a series that isn't a pool beside pools, and that a
    difference between such scenarios includes the products.

    `labels` is for a plan and takes no built samples: it maps a label to its
    (pathway, horizon) pairs, in any order, several pairs being a window of
    horizons of one pathway and one pair a single horizon. A set with `labels=`,
    or with no entries, describes what a plan reads and holds no samples, so
    `labels` (the property) and `sample` raise on it.

    Every scenario of a set is read from curves, or every scenario from per-year
    rows. `ScenarioError` names the two scenarios and the field for scenarios that
    differ in:

    - `asset_ids`, in order; `index_unit` and `loss_unit`, exactly;
    - each asset's `loss_model` and `basis_risk` document;
    - `hazard_name`, compared exactly, with no crosswalk of names;
    - on a sample of several hazards: its components in order, each one's unit,
      tail and hazard name, and the `combine` rule;
    - on per-year rows: the trend option (its kind, with `n` or with `method` and
      `min_years`, while a reference year or fitted slopes may differ) and a
      pool's (model, member) pairs. A series that isn't a pool beside a pool needs
      `mixed_products=True`;
    - on curves: `period` and the generator record of every `hazard` step of each
      asset or hazard, so that row d holds one uniform in every scenario, and the
      `dependence` and `hazard_dependence` documents, whatever the number of a
      window's members;
    - on curves, the H3 cell (`cell_index`) each asset's or hazard's curves are
      read in, so that one asset id doesn't hold another place's curves. Every
      pair of scenarios is compared, so a coarse cell can't hold two finer
      cells of two other scenarios. Cells at two resolutions, from two
      products, are compared at the coarser one, where the finer cell's parent
      must be the coarser cell or one of its neighbours, because a point's
      cells at two resolutions don't always nest; that needs the `geometry`
      extra. A curve given as a distribution or as `CurveParameters` records
      no cell, and isn't compared;
    - on per-year rows, the place each asset's or hazard's series is read at,
      the hazard series and a separate index series each on its own, every run
      of a pool included, so that one asset id doesn't hold another place's
      years. Like is compared with like. Two matched grid cells
      (`AnnualSeries.nearest_cell`) of one grid, one step with centres whole
      steps apart, must be one cell, as a point matches one cell of a grid;
      cells of two grids, as two products or two models give, must overlap, so
      that some point lies within half a grid step of both centres, each by its
      own step. Stations must be the same. A series' station is the filter on
      the column its record names (`AnnualSeries.from_table(station=)`, which
      a plan's `StationColumn` states), whatever it is called, or else a
      `station` or `station_id` filter or one-valued column; a filter on any
      other column, such as `model` or `scenario`, names none. A series that
      names neither a cell nor a station, such as a plain table, isn't
      compared, nor is a cell beside a station;
    - the `product`, unless `mixed_products=True`.

    It also raises for two samples with one label, a reference that is no label,
    a scenario that names no hazard, a pathway `crc_sdk.pathways` doesn't know or
    that a record of the scenario contradicts, and two scenarios with one
    pathway, horizon and product, null equal to null. On curves it raises for
    equal curves at two (pathway, horizon) pairs of one asset or hazard, a horizon
    outside the dataset's `temporal_window`, and a window whose members aren't
    one pathway's distinct horizons of one hazard, cell and dataset.

    **A window of horizons** reads as a list of members. Two scenarios with the
    same members raise, while two windows with one centre and different members
    pass, and so does a single horizon at a window's centre. Each distinct
    (pathway, horizon) of the set counts once for equal curves, so two windows
    may share members, and windows may differ in width.

    A set compares what the samples record and says nothing about which scenario
    or product is better. `TypeError` for arguments of the wrong type.

    A set holds samples and no terms. The covers that `compare_scenarios` puts on
    its scenarios keep their terms in every scenario, and each states them in
    words: `cover.terms_text()`, `result.cover.terms_text()` for a design and
    `package.terms_text()` for a `CoverPackage`.
    """

    entries: tuple[OutcomeSample, ...]
    reference: str
    mixed_products: bool
    _pairs: Mapping[str, _Members] | None

    def __init__(
        self,
        entries: Iterable[OutcomeSample] = (),
        *,
        reference: str,
        labels: Mapping[str, Iterable[tuple[str, int]]] | None = None,
        mixed_products: bool = False,
    ) -> None:
        reference = _checked_text("reference", reference)
        if not isinstance(mixed_products, bool):
            raise TypeError(
                f"mixed_products must be a bool, got {type(mixed_products).__name__}"
            )
        held = _checked_entries(entries)
        if held and labels is not None:
            raise TypeError(
                "labels= describes the scenarios a plan reads, and a set holds built "
                "samples or such descriptions, never both; a built sample's label "
                "is its own scenario="
            )
        pairs = None if labels is None else _checked_labels(labels, reference)
        if held:
            _check_scenarios(held, reference=reference, mixed_products=mixed_products)
        object.__setattr__(self, "entries", held)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "mixed_products", mixed_products)
        object.__setattr__(self, "_pairs", pairs)

    def _built(self) -> tuple[OutcomeSample, ...]:
        """The built samples; raises on a set that describes what a plan reads."""
        if self._pairs is not None:
            raise TypeError(
                "this set describes the scenarios a plan reads and holds no built "
                "samples"
            )
        if not self.entries:
            raise ScenarioError(
                "this set holds no built samples: a set of samples holds one scenario "
                "or more"
            )
        return self.entries

    @property
    def labels(self) -> tuple[str, ...]:
        """The scenarios' labels: the reference first, then as given.

        Raises `TypeError` on a set made with `labels=`, and `ScenarioError` on
        an empty set.
        """
        return tuple(
            sample.scenario
            for sample in _reference_first(self._built(), self.reference)
        )

    def sample(self, label: str) -> OutcomeSample:
        """Return the scenario of `label`.

        Raises `ScenarioError` for a label the set doesn't hold, `TypeError` on a
        set made with `labels=`, and `ScenarioError` on an empty set.
        """
        built = self._built()
        for sample in built:
            if sample.scenario == label:
                return sample
        raise ScenarioError(
            f"the set holds no scenario {label!r}; it holds "
            f"{[sample.scenario for sample in built]}"
        )

    def __repr__(self) -> str:
        described = (
            f"{len(self.entries)} samples"
            if self._pairs is None
            else f"labels={list(self._pairs)}"
        )
        return (
            f"ScenarioSet({described}, reference={self.reference!r}, "
            f"mixed_products={self.mixed_products})"
        )


# -- one table -------------------------------------------------------------------


def _design_label_of(entry: _Entry) -> list[str]:
    """The labels of the design samples of `entry`; empty for covers given as is."""
    if isinstance(entry, _JointEntry):
        return [entry.design.design_sample.scenario]
    if isinstance(entry, _Item):
        return [] if entry.design is None else [entry.design.design_sample.scenario]
    items = (
        entry._items.values()
        if isinstance(entry, CoverPackage)
        else entry.items.values()
    )
    return [
        item.design.design_sample.scenario for item in items if item.design is not None
    ]


def _checked_design_label(entry: _Entry, labels: Sequence[str]) -> str | None:
    """The scenario `entry` was designed on, or `None` for covers given as is.

    `ScenarioError` for a design sample whose label isn't one of the set's, and
    for designs of one mapping or package with different labels.
    """
    found = list(dict.fromkeys(_design_label_of(entry)))
    if not found:
        return None
    if len(found) > 1:
        raise ScenarioError(
            f"the designs of one mapping or package must be made on one scenario, but "
            f"their design samples are labelled {found}"
        )
    if found[0] not in labels:
        raise ScenarioError(
            f"a design's design sample is labelled {found[0]!r}, which is not one of "
            f"the set's scenarios, {list(labels)}"
        )
    return found[0]


def _checked_design_block(
    design_sample: Any, labels_of: Sequence[str | None], reference: str
) -> OutcomeSample | None:
    """`design_sample=`, the reference's design block, if any entry needs one."""
    if design_sample is None:
        if None in labels_of:
            raise SampleError(
                "a cover given as is needs design_sample=, the reference's design "
                "block: the sample its terms were chosen on, and where it is priced"
            )
        return None
    given = _sample(design_sample, "design_sample")
    if given.scenario != reference:
        raise ScenarioError(
            f"design_sample= is labelled {given.scenario!r}, but it is the reference's "
            f"design block, so its label must be {reference!r}"
        )
    return given


def _covers_of(entry: _Entry) -> tuple[Cover, ...]:
    """What each cover of `entry` reads, for the bound-location check."""
    if isinstance(entry, _Item):
        return (entry.cover,)
    if isinstance(entry, _CoverSet):
        return tuple(entry.covers.values())
    if isinstance(entry, CoverPackage):
        return (entry.as_cover(),)
    return tuple(entry.design.covers.values())


def _repriced(
    row: Mapping[str, Any], arrays: _Arrays, pricing: Pricing, probability: Any
) -> float | None:
    """`pricing` on this block's payouts, per contract; null on an allocated row."""
    if row["premium_basis"] is None:
        return None
    return (
        math.fsum(pricing.premium(payout, probability) for payout in arrays.contracts)
        + 0.0
    )


@dataclass(frozen=True, eq=False)
class _Terms:
    """What every block of `compare_scenarios` shares."""

    entries: tuple[_Entry, ...]
    design_labels: tuple[str | None, ...]
    reference: str
    given: OutcomeSample | None
    fold: int
    pricing: Pricing
    premiums: Any
    options: MetricOptions
    allocation: Allocation | None
    values: Mapping[str, float] | None
    columns: tuple[MetricColumn, ...]


def _other_scenario(
    evaluation: dict[str, Any],
    sample: OutcomeSample,
    designs: Sequence[OutcomeSample],
    made: Mapping[str, Any],
) -> None:
    """Make `evaluation` that of a block read with another scenario's terms.

    The method says so whatever the block's own split, which `details` keeps under
    `block`. `"different_sources"` is for per-year sources with no lineage or run
    in common: no row could be compared.
    """
    evaluation["method"] = "other_scenario"
    evaluation["details"] = {"block": dict(made)}
    if sample.provenance.source_kind != "monte_carlo" and not any(
        _share_a_source(design, sample) for design in designs
    ):
        evaluation["leakage_check"] = "different_sources"


def _made(sample: OutcomeSample) -> dict[str, Any]:
    """How a scenario's block was made: its split step, or that it has none."""
    step = _split_step(sample)
    if step is None:
        return {"kind": "as_built"}
    return {
        key: step[key]
        for key in ("kind", "block", "train_fraction", "k", "fold")
        if key in step
    }


def _premium_bases(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each row's cover and premium basis, for the setup text."""
    return [
        {
            key: row[key]
            for key in ("cover_hash", "cover_label", "level", "premium_basis")
        }
        for row in rows
    ]


def _block_rows(
    terms: _Terms, sample: OutcomeSample
) -> tuple[list[dict[str, Any]], list[EvaluationMetadata]]:
    """One scenario's rows, and the metadata of each of its evaluations."""
    positions: dict[str | None, list[int]] = {}
    for i, label in enumerate(terms.design_labels):
        positions.setdefault(label, []).append(i)
    runs: dict[int, _Run] = {}
    prepared: dict[str | None, tuple[dict[str, Any], tuple[OutcomeSample, ...]]] = {}
    for label, chosen in positions.items():
        drawn_up = [terms.entries[i] for i in chosen]
        joint = any(isinstance(entry, _JointEntry) for entry in drawn_up)
        run, evaluation, designs = _prepared(
            drawn_up,
            sample,
            split="out_of_sample",
            fold=terms.fold,
            pricing=terms.pricing,
            premium=None,
            options=terms.options,
            # A design on another scenario brings a design block of its own.
            design_sample=(terms.given if label in (None, terms.reference) else None),
            allocation=terms.allocation if joint else None,
            values=terms.values if joint else None,
            price_on=terms.given if label is None else None,
        )
        runs.update(dict.fromkeys(chosen, run))
        prepared[label] = (evaluation, designs)
        for entry in drawn_up:
            for cover in _covers_of(entry):
                _check_bound_cover(cover, sample)
    groups = _apply_premiums(
        terms.premiums,
        [_rows(entry, runs[i]) for i, entry in enumerate(terms.entries)],
    )
    every_row = np.ones(len(sample), dtype=np.bool_)
    without = _no_premium_names(terms.options)
    for i, (entry, group) in enumerate(zip(terms.entries, groups, strict=True)):
        for row, arrays in zip(group, _arrays(entry, runs[i]), strict=True):
            if row["premium_basis"] == "stated":
                read = _block_read(row, arrays, terms.pricing)
                row.update(
                    _metrics(
                        read, every_row, sample.probability, terms.options, without
                    )
                )
            row["premium_repriced"] = _repriced(
                row, arrays, terms.pricing, sample.probability
            )
    made = _made(sample)
    items: list[EvaluationMetadata] = []
    for label, chosen in positions.items():
        evaluation, designs = prepared[label]
        run = runs[chosen[0]]
        entries = [terms.entries[i] for i in chosen]
        by_entry = [groups[i] for i in chosen]
        _describe_rows(evaluation, run, entries, by_entry)
        evaluation["premium_bases"] = _premium_bases(
            row for group in by_entry for row in group
        )
        evaluation["scenario"] = sample.scenario
        evaluation["design_scenario"] = label
        if (terms.reference if label is None else label) != sample.scenario:
            _other_scenario(evaluation, sample, designs, made)
        items.append(
            _metadata(
                entries,
                sample,
                evaluation=evaluation,
                design_samples=designs,
                options=terms.options,
                columns=terms.columns,
                values_sha256=None if run.values is None else _values_hash(run.values),
            )
        )
    return [row for group in groups for row in group], items


def _hazard_record(scenario: _Scenario) -> Any:
    """The hazard a scenario reads: its record, or each component's."""
    components = scenario.document.get("components")
    if components is None:
        return scenario.document.get("hazard")
    return {
        name: own.get("hazard")
        for name, own in components.items()
        if not _reads_no_loss(own)
    }


def _index_place(owner: str, leaf: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Where a leaf's index is read: its index source, or else its hazard source."""
    held = leaf.get("sources") or ()
    sources = [source for source in held if source.get("role") == "index"] or [
        source for source in held if source.get("role") == "hazard"
    ]
    places: list[dict[str, Any]] = []
    for source in sources:
        series = source.get("series")
        place: dict[str, Any] = {"owner": owner, "kind": "stated"}
        if source.get("kind") == "curve":
            place |= {
                "kind": "h3_cell",
                "cell": (source.get("row") or {}).get("cell_index"),
            }
        elif isinstance(series, Mapping):
            cell = series.get("cell")
            stations = _station_ids(series)
            if isinstance(cell, Mapping):
                place |= {
                    "kind": "nearest_cell",
                    "latitude": float(cell["latitude"]),
                    "longitude": float(cell["longitude"]),
                }
            elif stations:
                place |= {"kind": "station", "station": ", ".join(stations)}
        if place not in places:
            places.append(place)
    return places or [{"owner": owner, "kind": "stated"}]


def _member_rows(sample: OutcomeSample, members: Sequence[Any]) -> list[int] | None:
    """A window's rows per member on this block; row d reads member d mod m."""
    if len(members) < 2:
        return None
    counts = np.bincount(np.mod(sample.period, len(members)), minlength=len(members))
    return [int(count) for count in counts]


def _scenario_document(
    scenario: _Scenario, columns: Sequence[MetricColumn], fold: int
) -> dict[str, Any]:
    """One scenario's entry of `setup.scenarios`, for one block."""
    sample = scenario.sample
    read = _source_columns(sample, strict=True)
    curves = sample.provenance.source_kind == "monte_carlo"
    members = [list(pair) for pair in scenario.members] if curves else None
    block: dict[str, Any] = {
        "fold": fold,
        "made": _made(sample),
        "member_rows": _member_rows(sample, members or ()),
        "n_rows": len(sample),
        "sample_hash": sample.content_hash,
        "short_block_columns": [
            column.name for column in columns if _short_block(column, len(sample))
        ],
    }
    return {
        "blocks": [block],
        "hazard": _hazard_record(scenario),
        "horizon": read.horizon,
        "index_read": [
            place
            for owner, leaf in scenario.leaves
            for place in _index_place(owner, leaf)
        ],
        "members": members,
        "n_rows": len(sample),
        "pathway": read.pathway,
        "product": read.product,
        "source_kind": sample.provenance.source_kind,
    }


def _scenarios_document(
    scenarios: ScenarioSet,
    labels: Sequence[str],
    columns: Sequence[MetricColumn],
    fold: int,
) -> dict[str, Any]:
    """What `setup.scenarios` records of a set."""
    read = [_read(scenarios.sample(label)) for label in labels]
    shared: dict[str, Any] | None = None
    if read[0].sample.provenance.source_kind == "monte_carlo":
        shared = {
            "labels": list(labels),
            "rng": {
                owner: [list(record) for record in records]
                for owner, records in _hazard_generators(read[0]).items()
            },
        }
    return {
        "design_on": None,
        "labels": list(labels),
        "mixed_products": scenarios.mixed_products,
        "reference": scenarios.reference,
        "scenarios": {
            scenario.label: _scenario_document(scenario, columns, fold)
            for scenario in read
        },
        "shared_draws": shared,
    }


def compare_scenarios(
    covers: Sequence[EvaluationEntry],
    scenarios: ScenarioSet,
    *,
    fold: int = 0,
    pricing: Pricing = PurePremium(),
    premiums: Mapping[str, float] | None = None,
    options: MetricOptions = MetricOptions(),
    design_sample: OutcomeSample | None = None,
    allocation: Allocation | None = None,
    values: Mapping[str, float] | None = None,
) -> Any:
    """Evaluate every cover or design on every scenario's block; return one table.

    `covers` is what `compare` takes on the set's samples. Every entry is evaluated
    on every scenario's block, the reference first and then as the set gives them,
    and every row has `split="out_of_sample"`. `scenario` says where a row is
    evaluated and `design_scenario` where it was designed: null for a cover given
    as is, which wasn't designed on any scenario. How a fixed cover drifts across
    scenarios and a design per scenario are slices of this table. Nothing in it
    ranks a pathway, a horizon, a product or a design.

    With one scenario, the reference alone, the rows are `compare`'s on that block,
    except that a cover given as is takes the premium below. The columns are those
    of `compare`, with the metric column `premium_repriced` after `premium`.

    `design_sample` is the reference's design block, and is required when an entry
    is a cover given as is: its terms were chosen there, and it is priced there. Its
    label must be the reference's. A design brings its own, whose label must be one
    of the set's, and the designs of one mapping must share that label.

    `premium` and the net-loss columns stay at the contract price: a design's
    premium (`premium_basis` `"design"`), the premium `premiums=` states for a
    cover given as is, keyed by `cover_label`, a set or package stated whole or not
    at all (`"stated"`), or else `pricing` once, on the reference's design block, not
    on the block evaluated (`"reference"`). `premium_repriced` is what a renewal on
    this block would cost: `pricing` on this block's payouts, the sum of the
    contracts' on a set's `"total"` row or a package's row, on the total payout on a
    joint design's `"total"` row. Under `PurePremium` it equals `expected_payout`.
    Allocated rows have neither.

    A budget is the design's: `Budget(ratio=k)` is a share of each design block's
    expected loss, so it moves with the scenario, and `Budget(amount=B)` doesn't.
    `setup_text` says which, and states every scenario next to the numbers.

    A row whose design sample has another label than its block is a cross row, with
    the terms read on that block as they stand: nothing rescales a threshold set on
    another scenario's values. Its evaluation's `method` is `"other_scenario"`.
    Curves share one set of evaluation draws, separate from the design's, which
    `ScenarioSet` and the leakage check of `compare` enforce; per-year blocks keep
    their own held-out years, and a cross row between per-year sources with no
    lineage or run in common records `leakage_check` `"different_sources"`.

    `fold` is the fold of a k-fold split, one at a time: concatenate folds with
    `pyarrow.concat_tables` and pass `EvaluationMetadata.merge` of their metadata to
    `write_comparison`. `allocation` and `values` are `compare`'s, and `allocation`
    takes joint designs only, across the whole call: any other entry raises
    `EvaluationError`, as in `compare`, whichever scenario each entry was designed
    on, so allocated rows never share a table with the rows of covers that stand on
    their own.

    Raises `TypeError` for a `scenarios` that isn't a `ScenarioSet` or that
    describes what a plan reads, and for arguments of the wrong type;
    `ScenarioError` on an empty set, for a design sample labelled outside the set,
    and for a cover bound to a location a scenario's index doesn't read; `SampleError`
    without the `design_sample` a cover given as is needs; `EvaluationError` for a
    `premiums=` key that matches no cover given as is and for an `allocation` beside
    an entry that isn't a joint design; `LeakageError` for a shared
    period or a reused generator state; and otherwise as `compare` does.
    """
    if not isinstance(scenarios, ScenarioSet):
        raise TypeError(
            f"scenarios must be a ScenarioSet, got {type(scenarios).__name__}"
        )
    labels = scenarios.labels
    entries = tuple(_entries(covers))
    design_labels = tuple(_checked_design_label(entry, labels) for entry in entries)
    reference = scenarios.reference
    given = _checked_design_block(design_sample, design_labels, reference)
    checked_fold = _fold(fold)
    checked_options = _options(options)
    checked_pricing = _pricing(pricing)
    rule = _allocation(allocation)
    reference_block = scenarios.sample(reference)
    checked_values = _values(values, rule, reference_block)
    if rule is not None:
        # Across the whole call: each design scenario's group is prepared on its
        # own, so a joint design of one scenario and a cover set of another would
        # otherwise pass the check `_prepared` makes within a group.
        _check_allocated(entries)
    columns = _with_repriced(metric_columns(checked_options))
    terms = _Terms(
        entries=entries,
        design_labels=design_labels,
        reference=reference,
        given=given,
        fold=checked_fold,
        pricing=checked_pricing,
        premiums=premiums,
        options=checked_options,
        allocation=rule,
        values=checked_values,
        columns=columns,
    )
    rows: list[dict[str, Any]] = []
    items: list[EvaluationMetadata] = []
    for label in labels:
        block_rows, block_items = _block_rows(terms, scenarios.sample(label))
        rows += block_rows
        items += block_items
    _check_keys(rows, KEY_COLUMNS)
    merged = EvaluationMetadata.merge(*items)
    metadata = _replaced(
        merged,
        setup=_replaced(
            merged.setup,
            scenarios=_scenarios_document(scenarios, labels, columns, checked_fold),
        ),
    )
    schema = _schema(columns, metadata, extra=_block_fields(checked_options.betas))
    return pa.Table.from_pylist(rows, schema=schema)


# -- differences from a reference --------------------------------------------------


_PAIR_COLUMNS = (
    "cover_hash",
    "design_id",
    "design_scenario",
    "asset_id",
    "level",
    "split",
    "fold",
)
"""The columns a row is paired on: one cover or design with itself."""

_DRIFT_FIELDS = (
    pa.field("reference_scenario", pa.string(), nullable=False),
    pa.field("reference_product", pa.string()),
    pa.field("reference_n_rows", pa.int32()),
    pa.field("reference_value", pa.float64()),
    pa.field("difference", pa.float64()),
    pa.field("ratio", pa.float64()),
)
"""The columns `to_drift` adds to the long form."""


def _drift_reference(reference: Any, metadata: EvaluationMetadata) -> str:
    """The scenario differences are taken from: stated, or the set's."""
    if reference is not None:
        if not isinstance(reference, str):
            raise TypeError(
                f"reference must be a string, got {type(reference).__name__}"
            )
        return reference
    stated = metadata.setup.scenarios
    if stated is None:
        raise EvaluationError(
            "this table states no set's reference: pass reference= with one of its "
            "scenarios"
        )
    return str(stated["reference"])


def to_drift(table: Any, *, reference: str | None = None) -> Any:
    """Add each row's difference and ratio against a reference scenario.

    `table` is a comparison table, such as one `compare_scenarios` returns. The
    result is its long form (`to_long`), one row per metric, with the columns
    `reference_scenario`, `reference_product`, `reference_n_rows`,
    `reference_value`, `difference` (`value - reference_value`, in the row's `unit`)
    and `ratio` (`value / reference_value`, in unit `"1"`). Both are null where
    either value is null, and `ratio` too where `reference_value` is 0.

    A row is paired with the row of the same (`cover_hash`, `design_id`,
    `design_scenario`, `asset_id`, `level`, `split`, `fold`) on the reference, so it
    compares one cover or design with itself across scenarios, never two covers, two
    designs or two scopes. Rows on the reference itself are left out, since their
    differences are 0. Every metric column is carried, `payout_frequency` and
    `unidentified_share` among them.

    `reference=None` takes the set's reference from the metadata. Any label in the
    table may be stated instead. The result has no rank, no best flag and no ordering
    by a metric. It keeps the table's metadata and adds `setup.drift`, with the
    reference used.

    Raises `TypeError` for a table that isn't a `pyarrow.Table` or a `reference` that
    isn't a string, and `EvaluationError` for a table `to_long` refuses, a table that
    states no reference when none is given, a reference that isn't in the table, or a
    row with no row to pair with on the reference.
    """
    long = to_long(table)
    metadata = EvaluationMetadata.from_table(table)
    chosen = _drift_reference(reference, metadata)
    names = long.column("scenario").unique().to_pylist()
    if chosen not in names:
        raise EvaluationError(
            f"the table holds no scenario {chosen!r}, so it has no reference rows; it "
            f"holds {names}"
        )
    rows = long.select(
        [*_PAIR_COLUMNS, "scenario", "product", "n_rows", "column", "value"]
    ).to_pylist()

    def key_of(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (*(row[name] for name in _PAIR_COLUMNS), row["column"])

    on_reference = {key_of(row): row for row in rows if row["scenario"] == chosen}
    keep = [row["scenario"] != chosen for row in rows]
    paired: list[Mapping[str, Any]] = []
    for row, kept in zip(rows, keep, strict=True):
        if not kept:
            continue
        found = on_reference.get(key_of(row))
        if found is None:
            raise EvaluationError(
                f"the row of cover {row['cover_hash']} on scenario {row['scenario']!r} "
                f"(design {row['design_id']}, level {row['level']!r}, fold "
                f"{row['fold']}) has no row on the reference {chosen!r}"
            )
        paired.append(found)
    values = [row["value"] for row, kept in zip(rows, keep, strict=True) if kept]
    differences: list[float | None] = []
    ratios: list[float | None] = []
    for value, found in zip(values, paired, strict=True):
        base = found["value"]
        known = value is not None and base is not None
        differences.append(None if not known else float(value) - float(base) + 0.0)
        ratios.append(
            None if not known or base == 0.0 else float(value) / float(base) + 0.0
        )
    result = long.filter(pa.array(keep, pa.bool_()))
    additions = (
        [chosen] * len(paired),
        [found["product"] for found in paired],
        [found["n_rows"] for found in paired],
        [found["value"] for found in paired],
        differences,
        ratios,
    )
    for field, column in zip(_DRIFT_FIELDS, additions, strict=True):
        result = result.append_column(field, pa.array(column, field.type))
    updated = _replaced(
        metadata,
        setup=_replaced(
            metadata.setup,
            drift={
                "columns": {
                    "difference": "value - reference_value, in the row's unit",
                    "ratio": "value / reference_value, in unit 1",
                },
                "reference": chosen,
            },
        ),
    )
    stored = dict(long.schema.metadata or {})
    stored[EVALUATION_METADATA_KEY.encode("utf-8")] = updated.to_json().encode("utf-8")
    return result.replace_schema_metadata(stored)
