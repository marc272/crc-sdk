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

This module does no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from crc_sdk.parametric.cover import Cover
from crc_sdk.parametric.errors import EvaluationError, SampleError, ScenarioError
from crc_sdk.parametric.index import (
    AtAsset,
    AtCell,
    AtPoint,
    AtStation,
    CompositeIndex,
)
from crc_sdk.parametric.metadata import _series_pathway
from crc_sdk.parametric.sample import (
    OutcomeSample,
    _check_cover_index,
    _check_known_pathway,
    _identifier,
    _source_columns,
    _text,
    _trend_of,
    _year,
)
from crc_sdk.pathways import pathways_equivalent
from crc_sdk.types import TemporalWindow
from crc_sdk.workflows.distributions import warn_if_outside_window

__all__ = ["ScenarioSet"]

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


def _pathway_statements(source: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every place a hazard source's record states a pathway, and what it states."""
    places: list[tuple[str, Any]]
    if source.get("kind") == "curve":
        places = [
            ("the canonical row", (source.get("row") or {}).get("pathway")),
            ("details=", (source.get("details") or {}).get("pathway")),
        ]
    elif "series" in source:
        series = source["series"] or {}
        document = series.get("source")
        stated = document if isinstance(document, Mapping) else {}
        try:
            column = _series_pathway(
                {"table_columns": series.get("table_columns")}, {}, "the series"
            )
        except EvaluationError:
            column = None
        places = [
            ("the model run", source.get("pathway")),
            ("the series' source", stated.get("pathway")),
            ("the series", series.get("pathway")),
            ("the table's scenario column", column),
        ]
    else:
        places = []
    return [(place, label) for place, label in places if _text(label) is not None]


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


def _check_years(scenarios: Sequence[_Scenario], mixed_products: bool) -> None:
    _require_same(scenarios, "trend", _trend_option)
    _check_pools(scenarios, mixed_products)


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


def _series_stations(series: Mapping[str, Any]) -> set[str]:
    """The stations a series names: in its filter, or as a one-valued column."""
    found = {str(value) for value in (series.get("where") or {}).values()}
    columns = series.get("table_columns") or {}
    for name in ("station", "station_id"):
        column = columns.get(name) or {}
        if column.get("distinct") == 1:
            found.update(str(value) for value in column["values"])
    return found


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
        return location.station_id in _series_stations(series)
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
        raise ScenarioError(
            f"scenario {label!r} and cover {cover.label or cover.definition_hash!r} "
            f"differ in location: the cover is read at {location!r}, which the index "
            f"of {owner} does not read"
        )


def _check_bound_cover(cover: Cover, sample: OutcomeSample) -> None:
    """Raise `ScenarioError` unless `sample`'s index is read where `cover` is bound.

    `AtAsset()` reads any scenario's own index at the asset. `AtPoint` needs a
    per-year series whose matched cell is centred there, compared as
    `.index(...)` compares a bound location, `AtStation` a per-year series of
    that station, and `AtCell` a curve at that cell, every member of a window.
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
