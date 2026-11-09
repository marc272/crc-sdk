"""Outcome samples from fitted curves and from per-year values.

**Monte Carlo from a curve** (`sample_curve`). A canonical curve is a
quantile function over the non-exceedance probability. Each simulated year
draws `u` from `rng.random(n_draws)` and takes `distribution.quantiles(u)`,
the inverse-CDF route of crc-framework's own `sample(size, rng=...)`. A draw
is a year only if the curve describes one year's value, so the curve's
`probability_semantics` must be `annual_exceedance` (the curve of an annual
maximum) or `annual_value_distribution` (an annual value):

- any other declared value raises `ProbabilitySemanticsError`;
- unspecified semantics raise too, unless the caller states one of the two
  with `assume_semantics=`; a stated value that contradicts the declared one
  raises.

A hurdle curve keeps its atom: a dry-year draw gets the atom's value.
`no_data` rows raise; they never become zero. Draws outside the curve's
fitted support are counted on each side: rarer than the longest fitted
return period warns with `ReturnPeriodExtrapolationWarning`; more frequent
than the shortest is recorded and named in the provenance's notes. A
tabulated curve cannot be evaluated outside its knots, so by default its
sample raises; `outside_support="extrapolate"` uses crc-framework's own
extrapolation and `"clamp"` the nearest knot, and the choice is recorded.

**Several assets from curves** (`sample_curves`). Curves say nothing about how
assets move together, so a stated dependence model (`Independent`,
`Comonotonic`, `EmpiricalCopula`, `GaussianCopula` or a `DependenceModel` of the
caller's) is required, and has no default. Each asset is drawn exactly as
`sample_curve` draws it, from its own generator; the model then only puts that
asset's rows in another order.

**Several hazards from curves** (`sample_curve_hazards`). Fitted curves carry no
dependence between hazards either, so the same models state it, across the
named components of one asset. Each component is drawn as `sample_curve` draws
it, from its own generator, the model puts its rows in another order, and the
combined loss is made from the rows as they then stand.

**Per-year values** (`AnnualSeries`, `sample_annual`). The years of an
`annual_extremes()` table (ERA5, CMIP6, NEX-GDDP-CMIP6, stations) become the
rows; nothing is assumed about probabilities. A missing or null year raises,
unless `missing="drop_year"` drops it and lists it in the provenance. A second
series as `index=` reads the index at another cell, station or hazard than the
loss: the dependence between index and loss is then the data's own. `trend=`
states how the climate trend is treated (`AsRecorded`, the default,
`RecentYears` or `LinearTrend`; see `crc_sdk.parametric.trend`), and `scenario=`
is the caller's name for the rows.

**Several hazards, per year** (`sample_annual_hazards`). The series of several
hazards at one asset, over the same years, make one sample with named index
components. A row's values are the same year's, so the dependence between the
hazards is the data's own and is recorded as `{"kind": "data"}`. The rows are the
same years and not the same events: two annual maxima of one year may come from
different events. Two impacts or more need a stated rule for the combined loss
(`crc_sdk.parametric.combine`).

**Pooled model-years** (`ModelRun`, `sample_annual_pool`). One climate model
gives few years for a tail, so the years of several models can be pooled for
one asset under one pathway. Every model-year is one row with probability
`1/N`, in (year, model, member) order, with models and members in UTF-8 byte
order; `period` is the draw number, and each row's identity is recorded in the
provenance's `pool`. Every run must hold the same years, and a model appears
once, whatever its member, so every model has the same weight. A draw mixes
model spread with year-to-year variability, and the rows are not independent
years: rows of one run share that model's climate and rows of one year the
pathway's forcing, so the number of rows is not an effective sample size.

**Hazard units.** Before anything is drawn, these functions check the hazard's
unit (the curve's `value_unit`, the hazard series' `unit`) against the loss
model's impact with `LossModel.check_hazard_unit`: an `ImpactExpression`
calibrated for another unit raises `SampleError`. Impacts that declare no
unit are not checked; the loss model's document records `exposure_unit: None`.

Draw order, with one generator per call: the hazard uniforms first, then the
basis-risk model. The provenance records the generator's state before each
step, so a sample can be rebuilt exactly.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, get_args

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from crc_framework.distributions import (
    FittedDistribution,
    HurdleDistribution,
    PointMassDistribution,
    TabulatedDistribution,
)
from numpy.typing import NDArray
from pydantic import BaseModel

from crc_sdk.parametric._canonical import content_hash
from crc_sdk.parametric.basis_risk import BasisRisk, ImpactNoise, NoBasisRisk
from crc_sdk.parametric.combine import (
    CombineRule,
    _combined_impact,
    _one_impact_record,
    _rule_record,
)
from crc_sdk.parametric.dependence import (
    DependenceModel,
    _checked_order,
    _model_record,
    _require_model,
)
from crc_sdk.parametric.errors import (
    ComonotoneIndexWarning,
    ProbabilitySemanticsError,
    SampleError,
)
from crc_sdk.parametric.index import Tail, _component_name
from crc_sdk.parametric.loss import LossModel
from crc_sdk.parametric.metadata import _year_spans
from crc_sdk.parametric.sample import (
    COMONOTONE_MESSAGE,
    IndexComponent,
    OutcomeSample,
    SampleProvenance,
    _arrow_table,
    _asset_note,
    _by_component,
    _check_json,
    _column,
    _component_note,
    _freeze,
    _generator_record,
    _identifier,
    _integer_values,
    _licence_entry,
    _require_generator,
    _source_document,
    _thaw,
)
from crc_sdk.parametric.trend import AsRecorded, Trend, _require_trend, _Series
from crc_sdk.providers.era5 import ERA5_ATTRIBUTION, ERA5_LICENCE
from crc_sdk.types import CurveParameters, HazardDatasetMetadata
from crc_sdk.workflows.distributions import (
    ReturnPeriodExtrapolationWarning,
    curve_parameters_from_row,
    return_periods_to_probabilities,
)

__all__ = [
    "AnnualSemantics",
    "AnnualSeries",
    "MissingYears",
    "ModelRun",
    "OutsideSupport",
    "sample_annual",
    "sample_annual_hazards",
    "sample_annual_pool",
    "sample_curve",
    "sample_curve_hazards",
    "sample_curves",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

AnnualSemantics = Literal["annual_exceedance", "annual_value_distribution"]
"""The probability semantics under which one curve draw is one year."""

OutsideSupport = Literal["raise", "extrapolate", "clamp"]
"""What a tabulated curve does with draws outside its knots."""

MissingYears = Literal["raise", "drop_year"]
"""What `sample_annual` does with a year without a value."""

_ANNUAL_SEMANTICS: tuple[str, ...] = get_args(AnnualSemantics)
_EARTH_RADIUS_KM = 6371.0088
_ROW_FIELDS = ("hazard_name", "horizon", "pathway", "source_id")
_TABLE_PROVENANCE_COLUMNS = (
    "archive_licence",
    "archive_licence_url",
    "attribution",
    "calendar",
    "datum_epoch",
    "datum_label",
    "grid_label",
    "licence",
    "member",
    "model",
    "original_model_licence",
    "recipe",
    "reference_notice",
    "retrieved_at",
    "scenario",
    "source_url",
    "station",
    "station_id",
    "station_name",
    "version",
)
_MAX_RECORDED_VALUES = 20
_SOURCE_PLANS: tuple[
    tuple[str, tuple[str, ...], tuple[str, ...], Mapping[str, str]], ...
] = (
    (
        "era5",
        ("store_url", "grid_degrees"),
        ("store", "recipe"),
        {"licence": ERA5_LICENCE, "attribution": ERA5_ATTRIBUTION},
    ),
    ("cmip6", ("archive", "catalogue", "pathway"), ("model", "member", "recipe"), {}),
)
"""How the `explain()` documents of ERA5 and CMIP6 plans are read.

For each: the provider, as the plan records it on its fitted curves; the keys
that mark its document; the keys whose values name the dataset; and the terms
crc-sdk holds for the data, when the document does not state them. ERA5's are
those of its fitted curves, without the year of retrieval that their
attribution adds, as `explain()` does not know it.
"""
_PLAN_STATE = frozenset(
    {
        # The cache, and what a fetch reads into it.
        "attribution",
        "cache",
        "cache_dir",
        "calendar",
        "catalogue",
        "licence",
        "mode",
        "resolved_dataset",
        # How the plan runs.
        "execution",
        # The years asked for: each row states its own year.
        "years",
    }
)
"""The keys of a source plan's `explain()` document left out of a lineage."""
_PLAN_PLACE = frozenset({"area", "areas", "station_selection", "stations"})
"""The places a plan asks for, left out where the series' cell or filter names one."""
_ROW_SEARCH = ("asset_location", "spacing_degrees")
"""What a series records of how it found its rows, which are the same either way."""


# -- shared checks ------------------------------------------------------------


def _check_models(loss_model: Any, basis_risk: Any, asset_id: str) -> None:
    if not isinstance(loss_model, LossModel):
        raise TypeError(
            f"loss_model must be a LossModel, got {type(loss_model).__name__}"
        )
    if basis_risk is None or not isinstance(basis_risk, BasisRisk):
        raise TypeError(
            "basis_risk must be a BasisRisk model; pass NoBasisRisk() to state "
            "that the index is the hazard at the asset"
        )
    if loss_model.asset_id is not None and loss_model.asset_id != asset_id:
        raise SampleError(
            f"the loss model is bound to asset {loss_model.asset_id!r}, "
            f"not {asset_id!r}"
        )


def _apply_basis_risk(
    model: BasisRisk,
    index: FloatArray,
    hazard: FloatArray,
    impact: FloatArray,
    rng: np.random.Generator | None,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Apply a basis-risk model, check its output and summarise its effect."""
    result = model.apply(index.copy(), hazard.copy(), impact.copy(), rng)
    if not isinstance(result, tuple) or len(result) != 2:
        raise SampleError(
            f"basis-risk model {model.name!r} must return (index, impact)"
        )
    try:
        new_index = np.asarray(result[0], dtype=np.float64)
        new_impact = np.asarray(result[1], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SampleError(
            f"basis-risk model {model.name!r} returned non-numbers"
        ) from error
    if new_index.shape != index.shape or new_impact.shape != impact.shape:
        raise SampleError(
            f"basis-risk model {model.name!r} changed the shape of the index or impact"
        )
    if not np.isfinite(new_index).all():
        raise SampleError(
            f"basis-risk model {model.name!r} returned non-finite indices"
        )
    if not np.isfinite(new_impact).all() or (new_impact < 0).any():
        raise SampleError(
            f"basis-risk model {model.name!r} returned negative or non-finite impacts"
        )
    before = math.fsum(impact.tolist())
    summary: dict[str, Any] = {
        "impact_mean_ratio": (
            math.fsum(new_impact.tolist()) / before if before > 0 else None
        ),
        "index_changed": not np.array_equal(new_index, index),
    }
    if isinstance(model, ImpactNoise) and model.upper is not None:
        summary["capped_draws"] = int((new_impact >= model.upper).sum())
    return new_index, new_impact, summary


def _basis_risk_notes(model: BasisRisk, separate_index: bool) -> list[str]:
    if separate_index:
        return ["The index is read from another series than the loss."]
    if isinstance(model, NoBasisRisk):
        return ["No basis-risk model: the index and the loss are comonotone."]
    return []


def _warn_comonotone(model: BasisRisk, separate_index: bool) -> None:
    if isinstance(model, NoBasisRisk) and not separate_index:
        warnings.warn(COMONOTONE_MESSAGE, ComonotoneIndexWarning, stacklevel=3)


# -- Monte Carlo from a curve -------------------------------------------------


def _semantics(
    metadata: HazardDatasetMetadata, assume: Any
) -> tuple[str, Literal["dataset", "caller"]]:
    if assume is not None and assume not in _ANNUAL_SEMANTICS:
        raise ProbabilitySemanticsError(
            f"assume_semantics must be one of {_ANNUAL_SEMANTICS}, got {assume!r}"
        )
    declared = metadata.probability_semantics
    if declared is None:
        if assume is None:
            raise ProbabilitySemanticsError(
                "the dataset declares no probability_semantics, so a draw from its "
                "curve is not known to be a year; state assume_semantics="
                "'annual_exceedance' (an annual maximum) or "
                "'annual_value_distribution' (an annual value) if it is"
            )
        return str(assume), "caller"
    if declared not in _ANNUAL_SEMANTICS:
        raise ProbabilitySemanticsError(
            f"probability_semantics={declared!r} is not a distribution of annual "
            f"values, so a draw from the curve is not a year; sampling needs one of "
            f"{_ANNUAL_SEMANTICS}"
        )
    if assume is not None and assume != declared:
        raise ProbabilitySemanticsError(
            f"assume_semantics={assume!r} contradicts the dataset's declared "
            f"probability_semantics={declared!r}"
        )
    return declared, "dataset"


def _fitted_document(distribution: FittedDistribution) -> dict[str, Any]:
    return {
        "family": distribution.family,
        "location": distribution.location,
        "scale": distribution.scale,
        "shape": None if distribution.shape is None else float(distribution.shape),
    }


def _curve_document(distribution: Any) -> dict[str, Any]:
    if isinstance(distribution, HurdleDistribution):
        return {
            "kind": "hurdle",
            **_fitted_document(distribution.base),
            "atom_probability": distribution.atom_probability,
            "atom_location": distribution.atom_location,
        }
    if isinstance(distribution, FittedDistribution):
        return {"kind": "fitted", **_fitted_document(distribution)}
    if isinstance(distribution, PointMassDistribution):
        return {"kind": "point_mass", "location": distribution.location}
    if isinstance(distribution, TabulatedDistribution):
        return {
            "kind": "tabulated",
            "interpolation": distribution.interpolation,
            "extrapolate": bool(distribution.extrapolate),
            "probabilities": distribution.probabilities.tolist(),
            "values": distribution.values.tolist(),
        }
    return {"kind": "other", "type": type(distribution).__name__}


def _plain(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def _resolve_curve(curve: Any) -> tuple[Any, dict[str, Any] | None]:
    """Return the curve's distribution and, for a canonical row, its key."""
    row: dict[str, Any] | None = None
    if isinstance(curve, CurveParameters):
        parameters: CurveParameters = curve
    elif isinstance(curve, Mapping):
        try:
            parameters = curve_parameters_from_row(curve)
        except ValueError as error:
            raise SampleError(
                f"the canonical row is not a valid curve: {error}"
            ) from error
        row = {name: _plain(curve.get(name)) for name in _ROW_FIELDS}
        cell = _plain(curve.get("cell_index"))
        row["cell_index"] = None if cell is None else format(int(cell), "x")
    elif callable(getattr(curve, "quantiles", None)):
        return curve, None
    else:
        raise TypeError(
            "curve must be a crc-framework distribution, CurveParameters or a "
            f"canonical row mapping, got {type(curve).__name__}"
        )
    if parameters.curve_kind == "no_data":
        raise SampleError(
            f"the curve has no data ({parameters.curve_type}); a no_data row is "
            "never sampled as zero"
        )
    return parameters.to_distribution(), row


def _support(metadata: HazardDatasetMetadata) -> dict[str, Any] | None:
    """The fitted support as non-exceedance probabilities, frequent and rare end."""
    tail = metadata.return_period_tail
    if metadata.return_period_support is not None:
        shortest, longest = metadata.return_period_support
        frequent, rare = return_periods_to_probabilities((shortest, longest), tail=tail)
        return {
            "basis": "return_period_support",
            "return_periods": [shortest, longest],
            "frequent_probability": frequent,
            "rare_probability": rare,
        }
    if metadata.source_probability_support is not None:
        lower, upper = metadata.source_probability_support
        frequent, rare = (lower, upper) if tail == "upper" else (upper, lower)
        return {
            "basis": "source_probability_support",
            "return_periods": None,
            "frequent_probability": frequent,
            "rare_probability": rare,
        }
    return None


def _draw_count(n_draws: Any) -> int:
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise SampleError(f"n_draws must be a positive integer, got {n_draws!r}")
    if int(n_draws) < 1:
        raise SampleError(f"n_draws must be a positive integer, got {n_draws!r}")
    return int(n_draws)


@dataclass(frozen=True, eq=False)
class _CurveDraws:
    """One curve's draws, in draw order, and what a sample records of them.

    `impact` and `loss` are `None` for a hazard with no impact.
    """

    uniforms: FloatArray
    hazard: FloatArray
    index: FloatArray
    impact: FloatArray | None
    loss: FloatArray | None
    provenance: SampleProvenance
    rarer: int
    """How many draws are rarer than the curve's fitted support."""


def sample_curve(
    curve: Any,
    *,
    metadata: HazardDatasetMetadata,
    n_draws: int,
    rng: np.random.Generator,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    assume_semantics: AnnualSemantics | None = None,
    outside_support: OutsideSupport = "raise",
    details: Mapping[str, Any] | None = None,
) -> OutcomeSample:
    """Draw `n_draws` simulated years from a fitted curve.

    `curve` is a crc-framework distribution (for example from
    `distribution_from_hazard_row`), `CurveParameters`, or a canonical row
    mapping with the curve columns, whose key (hazard, horizon, pathway,
    cell, source id) is then recorded. `metadata` is the curve's dataset
    metadata (`HazardDatasetMetadata`): its probability semantics, tail,
    fitted support, unit and source. `details` adds caller information to
    the provenance, for example the file's SHA-256.

    Each draw is one year with probability `1/n_draws`; `period` is the draw
    number. The index is the drawn hazard in the dataset's `value_unit`; the
    loss is `loss_model` applied to it, after `basis_risk`. `rng` is
    required: the uniforms are drawn first, then the basis-risk model draws.

    Raises `ProbabilitySemanticsError` under the semantics rules of the
    module docstring; `SampleError` for a `value_unit` the loss model's
    impact is not calibrated for (checked before any draw), a `no_data`
    curve, a non-finite quantile, a tabulated curve sampled outside its
    knots with `outside_support="raise"`, or invalid arguments; `TypeError`
    for wrong types. Warns with `ReturnPeriodExtrapolationWarning` for draws rarer than
    the fitted support, and with `ComonotoneIndexWarning` for
    `NoBasisRisk()`.
    """
    asset = _identifier("asset_id", asset_id)
    _check_models(loss_model, basis_risk, asset)
    draws = _draw_curve(
        curve,
        metadata=metadata,
        n_draws=n_draws,
        rng=rng,
        loss_model=loss_model,
        basis_risk=basis_risk,
        asset=asset,
        assume_semantics=assume_semantics,
        outside_support=outside_support,
        details=details,
    )
    assert draws.loss is not None
    n = draws.hazard.shape[0]
    sample = OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=draws.index,
        loss=draws.loss,
        hazard=None if np.array_equal(draws.index, draws.hazard) else draws.hazard,
        index_unit=metadata.value_unit,
        loss_unit=loss_model.value_unit,
        semantics="annual",
        provenance=draws.provenance,
    )
    _warn_rarer(draws.rarer, n)
    _warn_comonotone(basis_risk, separate_index=False)
    return sample


def _warn_rarer(count: int, n: int, whose: str = "") -> None:
    """Warn if `count` of the `n` draws are rarer than the fitted support."""
    if not count:
        return
    rarer = (
        "is rarer than the curve's fitted support; its value is"
        if count == 1
        else "are rarer than the curve's fitted support; their values are"
    )
    warnings.warn(
        f"{whose}{count} of {n} draws {rarer} extrapolated",
        ReturnPeriodExtrapolationWarning,
        stacklevel=3,
    )


def _draw_curve(
    curve: Any,
    *,
    metadata: HazardDatasetMetadata,
    n_draws: int,
    rng: np.random.Generator,
    loss_model: LossModel | None,
    basis_risk: BasisRisk,
    asset: str,
    assume_semantics: AnnualSemantics | None,
    outside_support: OutsideSupport,
    details: Mapping[str, Any] | None,
) -> _CurveDraws:
    """`sample_curve`'s draws and records, without its sample and its warnings.

    The caller has checked `loss_model` and `basis_risk`. With `loss_model=None`
    the hazard has no impact: the basis-risk model doesn't act on it or draw
    for it, and the draws have no impact and no loss.
    """
    if not isinstance(metadata, HazardDatasetMetadata):
        raise TypeError(
            f"metadata must be HazardDatasetMetadata, got {type(metadata).__name__}"
        )
    n = _draw_count(n_draws)
    rng = _require_generator(rng)
    # Before any draw, so a refused call leaves the generator untouched.
    if loss_model is not None:
        loss_model.check_hazard_unit(metadata.value_unit)
    if outside_support not in get_args(OutsideSupport):
        raise SampleError(
            f"outside_support must be one of {get_args(OutsideSupport)}, "
            f"got {outside_support!r}"
        )
    semantics, semantics_source = _semantics(metadata, assume_semantics)
    distribution, row = _resolve_curve(curve)
    caller_details = _source_document(details)
    curve_document = _curve_document(distribution)
    notes: list[str] = []
    if semantics_source == "caller":
        notes.append(
            f"The dataset declares no probability semantics; the caller stated "
            f"{semantics!r}."
        )

    rng_records = [_generator_record(rng, "hazard")]
    uniforms = rng.random(n)
    evaluated = uniforms
    tabulated: dict[str, Any] | None = None
    if isinstance(distribution, TabulatedDistribution) and not distribution.extrapolate:
        lowest = float(distribution.probabilities[0])
        highest = float(distribution.probabilities[-1])
        below_knots = int((uniforms < lowest).sum())
        above_knots = int((uniforms > highest).sum())
        tabulated = {
            "knots": [lowest, highest],
            "outside_support": outside_support,
            "below_knots": below_knots,
            "above_knots": above_knots,
        }
        outside = below_knots + above_knots
        falls, was = ("falls", "was") if outside == 1 else ("fall", "were")
        if outside and outside_support == "raise":
            them = "it" if outside == 1 else "them"
            raise SampleError(
                f"{outside} of {n} draws {falls} outside the tabulated curve's "
                f"knots [{lowest}, {highest}]; pass outside_support='extrapolate' "
                f"or 'clamp' to evaluate {them}"
            )
        if outside and outside_support == "clamp":
            evaluated = np.clip(uniforms, lowest, highest)
            notes.append(
                f"{outside} of {n} draws fell outside the tabulated knots and {was} "
                "clamped to the nearest knot."
            )
        elif outside:
            distribution = TabulatedDistribution(
                distribution.probabilities,
                distribution.values,
                interpolation=distribution.interpolation,
                extrapolate=True,
            )
            notes.append(
                f"{outside} of {n} draws fell outside the tabulated knots and {was} "
                "extrapolated by crc-framework."
            )
    try:
        hazard = np.asarray(distribution.quantiles(evaluated), dtype=np.float64)
    except ValueError as error:
        raise SampleError(
            f"the curve cannot be evaluated at a draw: {error}"
        ) from error
    if hazard.shape != (n,) or not np.isfinite(hazard).all():
        raise SampleError("the curve returned non-finite or misshaped quantiles")

    atom: dict[str, Any] | None = None
    if isinstance(distribution, HurdleDistribution):
        atom = {
            "probability": distribution.atom_probability,
            "location": distribution.atom_location,
            "draws": int((uniforms <= distribution.atom_probability).sum()),
        }

    support = _support(metadata)
    counts: dict[str, int] | None = None
    if support is None:
        notes.append(
            "The curve declares no fitted support, so draws outside it are not counted."
        )
    else:
        frequent = support["frequent_probability"]
        rare = support["rare_probability"]
        if metadata.return_period_tail == "upper":
            below = int((uniforms < frequent).sum())
            above = int((uniforms > rare).sum())
        else:
            below = int((uniforms > frequent).sum())
            above = int((uniforms < rare).sum())
        counts = {"below": below, "above": above}
        periods = support["return_periods"]
        if below:
            shortest = f" ({periods[0]:g} years)" if periods is not None else ""
            notes.append(
                f"{below} of {n} draws {'is' if below == 1 else 'are'} more frequent "
                f"than the shortest fitted return period{shortest}: the curve is "
                "extrapolated there."
            )
        if above:
            longest = f" ({periods[1]:g} years)" if periods is not None else ""
            notes.append(
                f"{above} of {n} draws {'is' if above == 1 else 'are'} rarer than the "
                f"longest fitted return period{longest}: the curve is extrapolated "
                "there."
            )
    pooling = metadata.ensemble.pooling if metadata.ensemble is not None else None
    if pooling == "pooled":
        notes.append(
            "The curve pools an ensemble: a draw mixes model spread with "
            "year-to-year variability."
        )

    index = hazard
    impact: FloatArray | None = None
    loss: FloatArray | None = None
    basis_summary: dict[str, Any] | None = None
    if loss_model is not None:
        impact = loss_model.impact_values(hazard)
        rng_records.append(_generator_record(rng, "basis_risk"))
        index, impact, basis_summary = _apply_basis_risk(
            basis_risk, hazard, hazard, impact, rng
        )
        loss = loss_model.loss_values(impact)
        notes += _basis_risk_notes(basis_risk, separate_index=False)

    source = {
        "role": "hazard",
        "kind": "curve",
        "metadata": metadata.model_dump(mode="json"),
        "curve": curve_document,
        "row": row,
        "details": caller_details,
    }
    _check_json("the curve's source", source)
    provenance = SampleProvenance(
        source_kind="monte_carlo",
        sources=(source,),
        semantics_source=semantics_source,
        probability_semantics=semantics,
        loss_model=None if loss_model is None else loss_model._document(asset),
        basis_risk=basis_risk.to_document(),
        rng=tuple(rng_records),
        licences=(_licence_entry("hazard", metadata.source.model_dump(mode="json")),),
        details={
            "draws": n,
            "tail": metadata.return_period_tail,
            "support": support,
            "outside_support": counts,
            "atom": atom,
            "tabulated": tabulated,
            "ensemble_pooling": pooling,
            "basis_risk": basis_summary,
        },
        notes=tuple(notes),
        lineage=content_hash(
            {
                "kind": "monte_carlo",
                "asset_id": asset,
                "source": source,
                "rng": rng_records[0],
            }
        ),
    )
    return _CurveDraws(
        uniforms=uniforms,
        hazard=hazard,
        index=index,
        impact=impact,
        loss=loss,
        provenance=provenance,
        rarer=0 if counts is None else counts["above"],
    )


# -- several columns from curves -----------------------------------------------


def _check_own_generators(rngs: Mapping[str, np.random.Generator], owners: str) -> None:
    """Raise `SampleError` if two columns' generators are in the same state.

    Equal states would give equal draws, whatever the model says of them.
    """
    states = {
        column: _generator_record(rng, "hazard")["state"]
        for column, rng in rngs.items()
    }
    if len(set(states.values())) < len(states):
        raise SampleError(
            f"rngs must give each {owners} its own generator state: two {owners}s "
            "with the same state would draw the same uniforms"
        )


def _column_fields(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    key: str,
    note: Callable[[str, str], str],
    record: Mapping[str, Any],
    sentence: str,
    n: int,
) -> dict[str, Any]:
    """The provenance fields of a sample of several curves' columns.

    `documents` holds each column's own provenance; `key` is what each of its
    sources and generator records names the column as.
    """
    declared = sorted({own["probability_semantics"] for own in documents.values()})
    stated = {own["semantics_source"] for own in documents.values()}
    return {
        "source_kind": "monte_carlo",
        "sources": tuple(
            {**source, key: column}
            for column, own in documents.items()
            for source in own["sources"]
        ),
        # Columns that differ here are stated side by side because the caller says.
        "semantics_source": stated.pop() if len(stated) == 1 else "caller",
        "probability_semantics": ", ".join(declared),
        "rng": tuple(
            {**generator, key: column}
            for column, own in documents.items()
            for generator in own["rng"]
        ),
        "licences": tuple(
            _distinct(
                [
                    *(entry for own in documents.values() for entry in own["licences"]),
                    *(
                        {**entry, "role": "dependence"}
                        for entry in record.get("licences", ())
                    ),
                ]
            )
        ),
        "details": {"draws": n, "periods": [0, n - 1]},
        "notes": (
            sentence,
            *(
                note(column, text)
                for column, own in documents.items()
                for text in own["notes"]
            ),
        ),
    }


def _moved_documents(
    documents: Mapping[str, Mapping[str, Any]], order: IntArray, kind: str
) -> dict[str, dict[str, Any]]:
    """Each column's own provenance, with a step if the model moved its rows."""
    in_draw_order = np.arange(order.shape[0], dtype=np.int64)
    return {
        column: {
            **own,
            "selection": (
                []
                if np.array_equal(order[:, j], in_draw_order)
                else [{"kind": "reorder", "dependence": kind}]
            ),
        }
        for j, (column, own) in enumerate(documents.items())
    }


# -- several assets from curves ------------------------------------------------


def _by_asset(name: str, value: Any, assets: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a mapping by asset id, got {type(value).__name__}"
        )
    if set(value) != set(assets):
        raise SampleError(
            f"{name} must name the assets of curves, {sorted(assets)}; "
            f"got {sorted(value)}"
        )
    return value


def _one_unit(name: str, units: Mapping[str, str]) -> str:
    if len(set(units.values())) > 1:
        raise SampleError(
            f"sample_curves needs one {name} for every asset, got {dict(units)}"
        )
    return next(iter(units.values()))


def sample_curves(
    curves: Mapping[str, Any],
    *,
    dependence: DependenceModel,
    metadata: Mapping[str, HazardDatasetMetadata],
    n_draws: int,
    rngs: Mapping[str, np.random.Generator],
    loss_models: Mapping[str, LossModel],
    basis_risk: BasisRisk,
    assume_semantics: AnnualSemantics | None = None,
    outside_support: OutsideSupport = "raise",
    details: Mapping[str, Any] | None = None,
) -> OutcomeSample:
    """Draw simulated years for several assets under a stated dependence.

    `curves`, `metadata`, `rngs` and `loss_models` are mappings by asset id;
    the order of `curves` sets `asset_ids`. `dependence` is required: curves
    say nothing about how assets move together, so the dependence is an
    assumption the caller states with `Independent()`, `Comonotonic()`,
    `EmpiricalCopula(reference)`, `GaussianCopula(correlation)` or a
    `DependenceModel` of the caller's, which passes the gate described in
    `crc_sdk.parametric.dependence`. `basis_risk` is one model for every asset;
    `details` is added to every asset's provenance; the other arguments are
    those of `sample_curve`.

    Each asset is drawn by `sample_curve` with its own generator: the uniforms
    first, then the basis-risk model. The dependence model then puts that
    asset's rows in another order, by the ranks of the uniforms; a row's
    hazard, index and loss stay together, no value changes and no further
    random number is used. So each column holds the rows of `sample_curve`
    with the same generator, and under `Independent()` in the same order.
    `period` is the row number, 0 to `n_draws - 1`.

    The provenance's `dependence` holds the model's document: for an
    `EmpiricalCopula`, the reference's content hash, sources, licences, first
    and last year, `m` and `k`. `parts` holds each asset's own provenance,
    whose `selection` records a reordering, and `notes` state the model and
    call it an assumption. The lineage is built from the assets' lineages and
    not from the model, as the same generators give the same draws under every
    model: a design and an evaluation need different generators, whichever
    models they state.

    Raises `SampleError` for fewer than two assets, mappings that name other
    assets, two assets with the same generator state, assets in different
    units, an `n_draws` or a reference the model cannot use (all before any
    draw), a model that doesn't pass the gate, a generator whose bit generator
    is not one of numpy's, and for what `sample_curve` raises for; `TypeError`
    for a missing dependence model, a bare callable as one and wrong types.
    Warns as `sample_curve` does, for each asset.
    """
    if not isinstance(curves, Mapping):
        raise TypeError(
            f"curves must be a mapping by asset id, got {type(curves).__name__}"
        )
    assets = tuple(_identifier("asset id", asset) for asset in curves)
    if len(assets) < 2:
        raise SampleError(
            f"sample_curves needs two assets or more, got {list(assets)}; "
            "use sample_curve for one"
        )
    _require_model(dependence, "assets")
    metadata = _by_asset("metadata", metadata, assets)
    rngs = _by_asset("rngs", rngs, assets)
    loss_models = _by_asset("loss_models", loss_models, assets)
    n = _draw_count(n_draws)
    for asset in assets:
        if not isinstance(metadata[asset], HazardDatasetMetadata):
            raise TypeError(
                "metadata must hold HazardDatasetMetadata, got "
                f"{type(metadata[asset]).__name__} for asset {asset!r}"
            )
        _require_generator(rngs[asset])
        _check_models(loss_models[asset], basis_risk, asset)
    index_unit = _one_unit(
        "index unit", {asset: metadata[asset].value_unit for asset in assets}
    )
    loss_unit = _one_unit(
        "loss unit", {asset: loss_models[asset].value_unit for asset in assets}
    )
    _check_own_generators(rngs, "asset")
    # Before any draw, so a refused call leaves the generators untouched.
    document, sentence = _model_record(dependence, n, assets, "assets")

    parts = {
        asset: sample_curve(
            curves[asset],
            metadata=metadata[asset],
            n_draws=n,
            rng=rngs[asset],
            loss_model=loss_models[asset],
            basis_risk=basis_risk,
            asset_id=asset,
            assume_semantics=assume_semantics,
            outside_support=outside_support,
            details=details,
        )
        for asset in assets
    }
    # Each part recorded the state its uniforms came from, so these are them.
    uniforms = np.column_stack(
        [parts[asset].provenance.generator(0).random(n) for asset in assets]
    )
    order = _checked_order(dependence, document["kind"], uniforms, assets)

    def reordered(values: Mapping[str, FloatArray]) -> FloatArray:
        return np.column_stack(
            [values[asset][order[:, j], 0] for j, asset in enumerate(assets)]
        )

    documents = {asset: parts[asset].provenance.to_document() for asset in assets}
    provenance = SampleProvenance(
        **_column_fields(
            documents,
            key="asset_id",
            note=_asset_note,
            record=document,
            sentence=sentence,
            n=n,
        ),
        basis_risk=basis_risk.to_document(),
        lineage=content_hash(
            {
                "kind": "monte_carlo",
                "lineages": {asset: parts[asset].lineage for asset in assets},
            }
        ),
        dependence=document,
        parts=_moved_documents(documents, order, document["kind"]),
    )
    hazards = {
        asset: part.index if part.hazard is None else part.hazard
        for asset, part in parts.items()
    }
    return OutcomeSample(
        asset_ids=assets,
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=reordered({asset: part.index for asset, part in parts.items()}),
        loss=reordered({asset: part.loss for asset, part in parts.items()}),
        hazard=(
            None
            if all(part.hazard is None for part in parts.values())
            else reordered(hazards)
        ),
        index_unit=index_unit,
        loss_unit=loss_unit,
        semantics="annual",
        provenance=provenance,
    )


# -- several hazards from curves -----------------------------------------------


def sample_curve_hazards(
    curves: Mapping[str, Any],
    *,
    dependence: DependenceModel,
    metadata: Mapping[str, HazardDatasetMetadata],
    n_draws: int,
    rngs: Mapping[str, np.random.Generator],
    loss_models: Mapping[str, LossModel],
    combine: CombineRule | None = None,
    basis_risk: BasisRisk,
    asset_id: str,
    assume_semantics: AnnualSemantics | None = None,
    outside_support: OutsideSupport = "raise",
    details: Mapping[str, Any] | None = None,
    scenario: str = "baseline",
) -> OutcomeSample:
    """Draw simulated years of several hazards at one asset, under a stated dependence.

    `curves`, `metadata`, `rngs` and `loss_models` are mappings by component
    name, and the order of `curves` sets the components: the first is the
    primary, whose values the sample's `index` holds. Two or more, each named
    without surrounding whitespace and without `.`, `+`, `[` or `]`. Each
    component's unit is its dataset's `value_unit` and its tail the dataset's
    `return_period_tail`. `details` is added to every component's provenance;
    `scenario` is the caller's name for the rows; the other arguments are those
    of `sample_curve`.

    **The dependence.** `dependence` is required: fitted curves carry no
    dependence between hazards, so it is an assumption the caller states with
    `Independent()`, `Comonotonic()`, `EmpiricalCopula(reference)`,
    `GaussianCopula(correlation)` or a `DependenceModel` of the caller's, which
    passes the gate described in `crc_sdk.parametric.dependence`.
    `EmpiricalCopula`'s reference is a sample of per-year rows holding exactly
    these components, as `sample_annual_hazards` makes one. The provenance's
    `hazard_dependence` holds the model's record, and the notes and the setup
    text call the dependence an assumption. Every model orders the draws by
    value, not by severity, which the setup text says when the tails are mixed.

    **The draws.** Each component is what `sample_curve` makes with that
    component's generator: the uniforms first, then the basis-risk model. The
    model then puts that component's rows in another order, by the ranks of the
    uniforms: a row's hazard, index and own loss stay together, no value changes
    and no further random number is used. So each component holds the rows of
    `sample_curve`, and under `Independent()` `sample.component(name)` has its
    `content_hash` for the same generator. `period` is the row number, 0 to
    `n_draws - 1`, and each row has probability `1/n_draws`.

    **The loss.** A component's impact comes from its `LossModel`, on its
    hazard, after the basis-risk model, and its `own_loss` is that impact times
    the value. After the reorder, `combine` takes the impacts of each row, and
    the sample's `loss` is its result times the value. It is required when two
    or more components have an impact, and refused with fewer. A component left
    out of `loss_models` has no impact: it needs no rule, has no `own_loss`, and
    the basis-risk model doesn't act on it or draw for it. With one impact the
    sample's loss is that component's own, and `combine` records that. The loss
    models must have one value, one `value_unit` and one `value_column`.

    The lineage is built from the components' lineages and not from the model,
    as the same generators give the same draws under every model: a design and
    an evaluation need different generators, whichever models they state.

    Raises `SampleError` for fewer than two curves or an invalid name, mappings
    that name other components, no impact at all, a hazard unit a component's
    impact is not calibrated for, loss models with different values, units or
    value columns, a missing or invalid rule, two components with the same
    generator state, an `n_draws` or a reference the model cannot use (all
    before any draw), a model that doesn't pass the gate, a rule's invalid
    result, and for what `sample_curve` raises for. Raises `TypeError` for a
    missing dependence model, a bare callable as one or as `combine`, and wrong
    types. Warns as `sample_curve` does: for each component with draws rarer than
    its fitted support, and once for `NoBasisRisk()`.
    """
    if not isinstance(curves, Mapping):
        raise TypeError(
            "curves must map each component name to a curve, got "
            f"{type(curves).__name__}"
        )
    names = tuple(curves)
    if len(names) < 2:
        raise SampleError(
            f"sample_curve_hazards takes two hazards or more, got {len(names)}; "
            "for one hazard use sample_curve"
        )
    for name in names:
        _component_name(name, SampleError)
    _require_model(dependence, "hazards")
    asset = _identifier("asset_id", asset_id)
    scenario = _identifier("scenario", scenario)
    datasets = _by_component("metadata", metadata, names, every=True)
    generators = _by_component("rngs", rngs, names, every=True)
    n = _draw_count(n_draws)
    for name in names:
        if not isinstance(datasets[name], HazardDatasetMetadata):
            raise TypeError(
                "metadata must hold HazardDatasetMetadata, got "
                f"{type(datasets[name]).__name__} for component {name!r}"
            )
        _require_generator(generators[name])
    models = _hazard_models(
        loss_models,
        names,
        {name: datasets[name].value_unit for name in names},
        basis_risk,
        asset,
    )
    rule_record = _combine_record(combine, [name for name in names if name in models])
    _check_own_generators(generators, "component")
    # Before any draw, so a refused call leaves the generators untouched.
    document, sentence = _model_record(dependence, n, names, "hazards")

    draws = {
        name: _draw_curve(
            curves[name],
            metadata=datasets[name],
            n_draws=n,
            rng=generators[name],
            loss_model=models.get(name),
            basis_risk=basis_risk,
            asset=asset,
            assume_semantics=assume_semantics,
            outside_support=outside_support,
            details=details,
        )
        for name in names
    }
    order = _checked_order(
        dependence,
        document["kind"],
        np.column_stack([each.uniforms for each in draws.values()]),
        names,
    )
    components: dict[str, IndexComponent] = {}
    impacts: dict[str, FloatArray] = {}
    for j, (name, each) in enumerate(draws.items()):
        rows = order[:, j]
        index, hazard = each.index[rows], each.hazard[rows]
        if each.impact is not None:
            impacts[name] = each.impact[rows]
        components[name] = IndexComponent(
            index,
            unit=datasets[name].value_unit,
            tail=datasets[name].return_period_tail,
            hazard=None if np.array_equal(index, hazard) else hazard,
            own_loss=None if each.loss is None else each.loss[rows],
        )
    loss = _combined_loss(models, combine, rule_record, impacts)
    first = next(iter(components.values()))
    own = {name: each.provenance.to_document() for name, each in draws.items()}
    loss_documents = [
        {"name": name, "loss_model": own[name]["loss_model"]}
        for name in names
        if name in models
    ]
    sample = OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(n, dtype=np.int64),
        probability=np.full(n, 1.0 / n),
        index=first.values,
        loss=loss,
        hazard=first.hazard,
        index_unit=first.unit,
        loss_unit=models[next(iter(impacts))].value_unit,
        semantics="annual",
        provenance=SampleProvenance(
            **_column_fields(
                own,
                key="component",
                note=_component_note,
                record=document,
                sentence=sentence,
                n=n,
            ),
            # A list, because canonical JSON sorts keys and the order is the
            # components'.
            loss_model={"kind": "components", "components": loss_documents},
            basis_risk=basis_risk.to_document(),
            lineage=content_hash(
                {
                    "kind": "monte_carlo_hazards",
                    "asset_id": asset,
                    "lineages": {
                        name: each.provenance.lineage for name, each in draws.items()
                    },
                }
            ),
            hazard_dependence=document,
            combine=rule_record,
            components=_moved_documents(own, order, document["kind"]),
        ),
        scenario=scenario,
        index_components=components,
    )
    for name, each in draws.items():
        _warn_rarer(each.rarer, n, f"component {name}: ")
    _warn_comonotone(basis_risk, separate_index=False)
    return sample


# -- per-year values -----------------------------------------------------------


def _year_array(name: str, value: Any) -> IntArray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.ndim != 1 or array.size == 0:
        raise SampleError(f"{name} must be a non-empty 1-D sequence of integer years")
    years = array.astype(np.int64)
    if years.size > 1 and not (np.diff(years) > 0).all():
        raise SampleError(f"{name} must be strictly increasing")
    return years


def _requested_years(years: Any) -> IntArray | None:
    if years is None:
        return None
    array = np.asarray(years)
    if array.dtype.kind not in "iu" or array.ndim != 1 or array.size == 0:
        raise SampleError("years must be a non-empty sequence of integer years")
    unique = np.unique(array.astype(np.int64))
    if unique.size != array.size:
        raise SampleError("years must not repeat")
    return unique


def _recorded_columns(rows: Any) -> dict[str, Any]:
    recorded: dict[str, Any] = {}
    for name in _TABLE_PROVENANCE_COLUMNS:
        if name not in rows.column_names:
            continue
        values = sorted(
            {str(value) for value in rows.column(name).to_pylist() if value is not None}
        )
        recorded[name] = {
            "values": values[:_MAX_RECORDED_VALUES],
            "distinct": len(values),
        }
    return recorded


def _source_plan_facts(document: Mapping[str, Any]) -> dict[str, str]:
    """What an ERA5 or CMIP6 `explain()` document implies but does not state.

    These plans' documents state what was asked for, not who holds it: the
    provider and the dataset, and for ERA5 the terms of its data. Any other
    document implies nothing.
    """
    for provider, marks, names, terms in _SOURCE_PLANS:
        parts = [
            part for part in map(document.get, names) if isinstance(part, str) and part
        ]
        if len(parts) == len(names) and all(mark in document for mark in marks):
            return {"provider": provider, "dataset": " ".join(parts), **terms}
    return {}


def _source_plan_document(plan: Any) -> Any:
    """A source plan's `explain()` document, with `format="json"` where it takes one.

    Plans such as NOAA's water levels take no `format`: their `explain()`
    returns the document itself.
    """
    parameters = inspect.signature(plan.explain).parameters.values()
    takes_format = any(
        parameter.name == "format" or parameter.kind is parameter.VAR_KEYWORD
        for parameter in parameters
    )
    return plan.explain(format="json") if takes_format else plan.explain()


def _series_source_document(source: Any) -> dict[str, Any]:
    """Return a per-year series' `source` as a plain JSON document.

    As `_source_document`, and `source` may be a source plan too: then its
    `explain()` is the document, and nothing else of the plan is called, so
    no data is fetched. What an ERA5 or CMIP6 document implies is added,
    unless the document states its own.
    """
    if callable(getattr(source, "explain", None)):
        source = _source_plan_document(source)
        if not isinstance(source, Mapping):
            raise SampleError(
                "a source plan's explain() must return a mapping, "
                f"got {type(source).__name__}"
            )
    elif source is not None and not isinstance(source, (str, Mapping, BaseModel)):
        raise TypeError(
            "source must be a name, a SourceProvenance, a mapping or a plan with "
            f"explain(), got {type(source).__name__}"
        )
    document = _source_document(source)
    return {**_source_plan_facts(document), **document}


@dataclass(frozen=True, eq=False)
class AnnualSeries:
    """One hazard's value per year at one location.

    `years` are strictly increasing integer years; `values` holds one value
    per year in `unit`, with NaN for a year without a value. `provenance`
    describes the series: its source, the table's columns and filter, the
    matched cell or station. Build one with `from_table` or `nearest_cell`
    from an `annual_extremes()` table, or directly.

    Raises `SampleError` for unordered years, mismatched lengths, infinite
    values, an empty unit or a provenance that is not plain JSON.
    """

    years: IntArray
    values: FloatArray
    unit: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        years = _year_array("years", self.years)
        try:
            values = np.array(self.values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise SampleError("values must be numeric") from error
        if values.shape != years.shape:
            raise SampleError(
                f"values must have one entry per year, shape {years.shape}; "
                f"got {values.shape}"
            )
        if np.isinf(values).any():
            raise SampleError("values must be finite, or NaN for a missing year")
        _identifier("unit", self.unit)
        if not isinstance(self.provenance, Mapping):
            raise SampleError("provenance must be a mapping")
        _check_json("provenance", self.provenance)
        years.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "provenance", _freeze(_thaw(self.provenance)))

    def __len__(self) -> int:
        return int(self.years.shape[0])

    @property
    def missing_years(self) -> tuple[int, ...]:
        """The years without a value."""
        return tuple(int(year) for year in self.years[np.isnan(self.values)])

    @classmethod
    def _from_rows(
        cls,
        rows: Any,
        *,
        year: str,
        value: str,
        unit: str,
        years: IntArray | None,
        provenance: dict[str, Any],
    ) -> AnnualSeries:
        what = "the annual table"
        row_years = _integer_values(_column(rows, year, what), year)
        column = _column(rows, value, what)
        if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
            raise SampleError(f"column {value!r} must be numeric, not {column.type}")
        row_values = np.asarray(
            column.cast(pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if np.isinf(row_values).any():
            raise SampleError(f"column {value!r} holds infinite values")
        unique, counts = np.unique(row_years, return_counts=True)
        if (counts > 1).any():
            raise SampleError(
                f"years {unique[counts > 1][:10].tolist()} have several rows; "
                "select one location (where=, nearest_cell) first"
            )
        unit = _identifier("unit", unit)
        if "unit" in rows.column_names:
            stated = {u for u in rows.column("unit").to_pylist() if u is not None}
            if stated and stated != {unit}:
                raise SampleError(
                    f"the table's unit column says {sorted(stated)}, not {unit!r}"
                )
        expected = (
            np.arange(unique[0], unique[-1] + 1, dtype=np.int64)
            if years is None
            else years
        )
        values = np.full(expected.shape[0], np.nan)
        inside = np.isin(row_years, expected)
        values[np.searchsorted(expected, row_years[inside])] = row_values[inside]
        provenance["table_columns"] = _recorded_columns(rows)
        return cls(years=expected, values=values, unit=unit, provenance=provenance)

    @classmethod
    def from_table(
        cls,
        table: Any,
        *,
        year: str = "year",
        value: str = "value",
        where: Mapping[str, Any] | None = None,
        unit: str,
        years: Sequence[int] | None = None,
        source: Any = None,
    ) -> AnnualSeries:
        """Read one location's series from a per-year table.

        `year` and `value` name the columns. `where` keeps the rows whose
        columns equal the given values, for example `{"station":
        "USW00094728"}`; after it, each year may appear once. A null value
        (a station's incomplete year) is a missing year. `years` states the
        years the series must cover; rows outside them are left out, and
        years without a row are missing. By default the series covers every
        year from the first to the last row, so a gap is a missing year.
        `source` describes the table for the provenance: a name, a
        `SourceProvenance`, a mapping such as the plan's
        `explain(format="json")`, or the source plan itself, of which only
        `explain()` is called. An ERA5 or CMIP6 plan's document states no
        provider or dataset, so they are named from what it does state, and
        ERA5's states no terms, so crc-sdk's `ERA5_LICENCE` and
        `ERA5_ATTRIBUTION` are recorded; a mapping's own `provider`,
        `dataset`, `licence` and `attribution` win. Columns such as
        `attribution`, `archive_licence`, `source_url` or `station` are
        recorded too. If the table has a `unit` column, it must equal `unit`.

        What a source plan's document says of its cache, its run and the
        years it asks for is left out of a sample's lineage, as is the time a
        table's `retrieved_at` column records, so one source links across
        fetches and windows. A `source` of the caller's own is kept whole.

        `table` is an in-memory Arrow table, record batches (for example
        NEX-GDDP-CMIP6's `annual_extremes()` iterator) or any table Arrow
        reads. Raises `SampleError` for missing columns, unmatched filters,
        repeated years or a unit mismatch.
        """
        data = _arrow_table(table, "an annual table")
        rows = data
        filter_document: dict[str, Any] | None = None
        if where is not None:
            if not isinstance(where, Mapping) or not where:
                raise SampleError("where must be a non-empty mapping of column values")
            mask: Any = None
            for name, wanted in where.items():
                column = _column(data, name, "the annual table")
                if isinstance(wanted, float) and not math.isfinite(wanted):
                    raise SampleError(f"where[{name!r}] must be finite")
                if not isinstance(wanted, (str, int, float, bool)):
                    raise SampleError(
                        f"where[{name!r}] must be a string, number or bool"
                    )
                try:
                    matches = pc.equal(column, pa.scalar(wanted))
                except pa.ArrowException as error:
                    raise SampleError(
                        f"where[{name!r}] cannot be compared with column {name!r}"
                    ) from error
                mask = matches if mask is None else pc.and_(mask, matches)
            rows = data.filter(pc.fill_null(mask, False))
            filter_document = dict(where)
        if rows.num_rows == 0:
            raise SampleError(f"no rows of the annual table match where={where!r}")
        provenance: dict[str, Any] = {
            "kind": "table",
            "source": _series_source_document(source),
            "columns": {"year": year, "value": value},
            "where": filter_document,
        }
        return cls._from_rows(
            rows,
            year=year,
            value=value,
            unit=unit,
            years=_requested_years(years),
            provenance=provenance,
        )

    @classmethod
    def nearest_cell(
        cls,
        table: Any,
        *,
        longitude: float,
        latitude: float,
        unit: str,
        year: str = "year",
        value: str = "value",
        longitude_column: str = "longitude",
        latitude_column: str = "latitude",
        spacing: tuple[float, float] | None = None,
        years: Sequence[int] | None = None,
        source: Any = None,
    ) -> AnnualSeries:
        """Read the series of the grid cell whose centre is nearest a point.

        For per-cell tables such as ERA5's or CMIP6's `annual_extremes()`
        (`year`, `latitude`, `longitude`, `value`). The cell is accepted only
        if its centre lies within half a grid step of the point in latitude
        and in longitude; otherwise this raises. `spacing` is the grid step
        `(latitude, longitude)` in degrees; by default the smallest step
        between the table's distinct latitudes and longitudes. Longitudes are
        compared modulo 360. The cell centre and the step are recorded, not
        the point itself. By default the series must cover every year the
        table holds: a year the cell lacks is missing.
        `years`, `source` and the other arguments are as in `from_table`.

        Raises `SampleError` if no cell centre is close enough, two are
        equally near, or the spacing cannot be inferred.
        """
        lat0 = _coordinate("latitude", latitude, 90.0)
        lon0 = _coordinate("longitude", longitude, 360.0)
        data = _arrow_table(table, "an annual table")
        what = "the annual table"
        lat = _coordinates(_column(data, latitude_column, what), latitude_column)
        lon = _coordinates(_column(data, longitude_column, what), longitude_column)
        if lat.size == 0:
            raise SampleError("the annual table has no rows")
        if spacing is None:
            step_lat = _smallest_step(lat, latitude_column)
            step_lon = _smallest_step(lon, longitude_column)
        else:
            if not isinstance(spacing, Sequence) or len(spacing) != 2:
                raise SampleError("spacing must be a (latitude, longitude) pair")
            step_lat = _positive_step("spacing[0]", spacing[0])
            step_lon = _positive_step("spacing[1]", spacing[1])
        centres = np.unique(np.stack([lat, lon], axis=1), axis=0)
        d_lat = centres[:, 0] - lat0
        d_lon = (centres[:, 1] - lon0 + 180.0) % 360.0 - 180.0
        slack = 1.0 + 1e-9
        inside = (np.abs(d_lat) <= 0.5 * step_lat * slack) & (
            np.abs(d_lon) <= 0.5 * step_lon * slack
        )
        distance = _great_circle_km(lat0, lon0, centres[:, 0], centres[:, 1])
        if not inside.any():
            nearest = int(np.argmin(distance))
            raise SampleError(
                f"no grid cell centre lies within half a grid step "
                f"({step_lat:g}, {step_lon:g} degrees) of the point; the nearest, "
                f"({centres[nearest, 0]:g}, {centres[nearest, 1]:g}), is "
                f"{distance[nearest]:.1f} km away"
            )
        candidates = np.flatnonzero(inside)
        best = candidates[np.argsort(distance[candidates], kind="stable")]
        if best.size > 1 and math.isclose(
            distance[best[0]], distance[best[1]], rel_tol=1e-12, abs_tol=1e-9
        ):
            raise SampleError(
                "the point is equally near two grid cell centres; choose the cell "
                "with from_table(where=...)"
            )
        cell_lat, cell_lon = (float(v) for v in centres[best[0]])
        rows = data.filter(
            pc.and_(
                pc.equal(data.column(latitude_column), cell_lat),
                pc.equal(data.column(longitude_column), cell_lon),
            )
        )
        requested = _requested_years(years)
        if requested is None:
            all_years = _integer_values(_column(data, year, what), year)
            requested = np.unique(all_years)
        provenance: dict[str, Any] = {
            "kind": "nearest_cell",
            "source": _series_source_document(source),
            "columns": {
                "year": year,
                "value": value,
                "latitude": latitude_column,
                "longitude": longitude_column,
            },
            "cell": {"latitude": cell_lat, "longitude": cell_lon},
            "spacing_degrees": {"latitude": step_lat, "longitude": step_lon},
        }
        return cls._from_rows(
            rows,
            year=year,
            value=value,
            unit=unit,
            years=requested,
            provenance=provenance,
        )


def _coordinate(name: str, value: Any, bound: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a number, got {value!r}")
    number = float(value)
    low = -90.0 if bound == 90.0 else -180.0
    if not math.isfinite(number) or not low <= number <= bound:
        raise SampleError(f"{name} must be within [{low:g}, {bound:g}], got {value!r}")
    return number


def _coordinates(column: Any, name: str) -> FloatArray:
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise SampleError(f"column {name!r} must be numeric, not {column.type}")
    if column.null_count:
        raise SampleError(f"column {name!r} has {column.null_count} null values")
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
    if not np.isfinite(values).all():
        raise SampleError(f"column {name!r} must be finite")
    return values


def _smallest_step(values: FloatArray, name: str) -> float:
    distinct = np.unique(values)
    if distinct.size < 2:
        raise SampleError(
            f"the table has one distinct {name}, so its grid step cannot be "
            "inferred; pass spacing=(latitude_step, longitude_step)"
        )
    return float(np.diff(distinct).min())


def _positive_step(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SampleError(f"{name} must be a positive number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SampleError(f"{name} must be a positive number, got {value!r}")
    return number


def _great_circle_km(
    lat0: float, lon0: float, lat: FloatArray, lon: FloatArray
) -> FloatArray:
    phi0, phi = np.radians(lat0), np.radians(lat)
    d_phi = phi - phi0
    d_lambda = np.radians(lon - lon0)
    half = (
        np.sin(d_phi / 2) ** 2 + np.cos(phi0) * np.cos(phi) * np.sin(d_lambda / 2) ** 2
    )
    distance: FloatArray = (
        2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(half, 0, 1)))
    )
    return distance


def _source_identity(source: Any, *, located: bool) -> Any:
    """`source` as far as it names the data, in a stable form.

    Only a source plan's `explain()` document is cut down, one with a record
    of its cache: `_PLAN_STATE` goes, and `_PLAN_PLACE` too when the series is
    `located`. A caller's own source is kept whole.
    """
    if not isinstance(source, Mapping) or not {"cache", "cache_dir"} & source.keys():
        return source
    left_out = _PLAN_STATE | _PLAN_PLACE if located else _PLAN_STATE
    return {key: value for key, value in source.items() if key not in left_out}


def _series_identity(series: AnnualSeries) -> dict[str, Any]:
    """What identifies a series' source: its provenance, else its data.

    A lineage must not depend on when the data was fetched or how its rows
    were found, or the leakage checks would miss rows two series share. A
    series is `located` when its cell or its filter says which rows it reads.
    """
    identity: dict[str, Any] = {
        "unit": series.unit,
        **{
            key: value
            for key, value in _thaw(series.provenance).items()
            if key not in _ROW_SEARCH
        },
    }
    if "source" in identity:
        located = "cell" in identity or identity.get("where") is not None
        identity["source"] = _source_identity(identity["source"], located=located)
    recorded = identity.get("table_columns")
    if isinstance(recorded, Mapping):
        identity["table_columns"] = {
            name: values for name, values in recorded.items() if name != "retrieved_at"
        }
    if not identity.get("source"):
        digest = hashlib.sha256(
            series.years.astype("<i8").tobytes() + series.values.astype("<f8").tobytes()
        ).hexdigest()
        identity["data"] = f"sha256:{digest}"
    return identity


def _series_licence(role: str, series: AnnualSeries) -> dict[str, Any]:
    provenance = _thaw(series.provenance)
    source = provenance.get("source") or {}
    entry = _licence_entry(role, source if isinstance(source, Mapping) else {})
    recorded = provenance.get("table_columns") or {}

    def joined(*names: str) -> str | None:
        for name in names:
            values = (recorded.get(name) or {}).get("values")
            if values:
                return "; ".join(values)
        return None

    entry["licence"] = entry["licence"] or joined("licence", "archive_licence")
    entry["attribution"] = entry["attribution"] or joined("attribution")
    return entry


def sample_annual(
    hazard: AnnualSeries,
    *,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    index: AnnualSeries | None = None,
    rng: np.random.Generator | None = None,
    missing: MissingYears = "raise",
    trend: Trend = AsRecorded(),
    scenario: str = "baseline",
) -> OutcomeSample:
    """Build a sample whose rows are the years of a per-year series.

    `hazard` is the series the loss model reads. `index`, if given, is the
    series the cover pays on: another cell, station or hazard than the loss
    reads, so the index and the loss differ by the data's own dependence.
    Its years must equal the hazard's. Without it the index is the hazard.

    Each year is one row with probability `1/n`, and `period` is the year.
    A year without a value (in either series) raises, unless
    `missing="drop_year"`; dropped years are listed in the provenance. `rng`
    is needed only by a basis-risk model that draws, such as `ImpactNoise`.

    `trend` states how the climate trend is treated: `AsRecorded()` (the
    default), `RecentYears(n)` or `LinearTrend(...)`
    (`crc_sdk.parametric.trend`). The steps come in this order: the window of
    `RecentYears`; the missing years; the fit and the adjustment of the hazard
    and of the index, one fit each, on the same years with the same reference
    year; the impact on the adjusted hazard; the basis-risk model. The
    sample's `index` and `hazard` hold the adjusted values, and the
    provenance's `trend` records what was done. A sample whose `LinearTrend`
    was fitted on years it is later tested on is refused at evaluation
    (`LeakageError`): to hold out a block, state the design years with
    `LinearTrend(fit_years=...)`.

    `scenario` is the caller's name for the rows, in `to_table`,
    `content_hash` and the comparison's `scenario` column. It is not derived
    from the source (the `scenario` column of a CMIP6 table is its pathway,
    another thing) and the lineage leaves it out, so a relabelled sample still
    counts as sharing its rows.

    Raises `SampleError` for a hazard unit the loss model's impact is not
    calibrated for (checked before any draw; the `index` series is not
    checked, as the loss does not read it), missing years, mismatched years,
    a trend the series cannot take (`RecentYears` above its length, too few
    fit years, fit years it does not hold, a non-finite or newly negative
    adjusted value) or invalid arguments, and `TypeError` for wrong types.
    Warns with `ComonotoneIndexWarning` for `NoBasisRisk()` without an `index`
    series.
    """
    sample = _annual_rows(
        hazard,
        loss_model=loss_model,
        basis_risk=basis_risk,
        asset_id=asset_id,
        index=index,
        rng=rng,
        missing=missing,
        trend=trend,
        scenario=scenario,
    )
    _warn_comonotone(basis_risk, separate_index=index is not None)
    return sample


def _no_value_error(dropped: Sequence[int], which: Sequence[str]) -> SampleError:
    """The error for years without a value; `which` says which series lack them."""
    years_have, them = (
        ("year has", "it") if len(dropped) == 1 else ("years have", "them")
    )
    return SampleError(
        f"{len(dropped)} {years_have} no value ({'; '.join(which)}); pass "
        f"missing='drop_year' to drop {them}, listed in the provenance"
    )


def _dropped_notes(dropped: Sequence[int]) -> list[str]:
    """The note for the years dropped as missing; none if no year was."""
    if not dropped:
        return []
    noun, verb = ("year", "was") if len(dropped) == 1 else ("years", "were")
    return [
        f"{len(dropped)} {noun} without a value {verb} dropped: "
        f"{', '.join(str(year) for year in dropped[:10])}"
        f"{', ...' if len(dropped) > 10 else ''}."
    ]


def _annual_lineage(
    asset_id: str, hazard: AnnualSeries, index: AnnualSeries | None
) -> str:
    """The lineage of a per-year sample, and of the same run in a pool."""
    return content_hash(
        {
            "kind": "annual",
            "asset_id": asset_id,
            "hazard": _series_identity(hazard),
            "index": None if index is None else _series_identity(index),
        }
    )


def _annual_rows(
    hazard: AnnualSeries,
    *,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    index: AnnualSeries | None,
    rng: np.random.Generator | None,
    missing: MissingYears,
    trend: Trend,
    scenario: str,
) -> OutcomeSample:
    """`sample_annual`, without its warning: a plan rebuilds a sample per fold."""
    if not isinstance(hazard, AnnualSeries):
        raise TypeError(f"hazard must be an AnnualSeries, got {type(hazard).__name__}")
    if index is not None and not isinstance(index, AnnualSeries):
        raise TypeError(f"index must be an AnnualSeries, got {type(index).__name__}")
    asset = _identifier("asset_id", asset_id)
    _check_models(loss_model, basis_risk, asset)
    # The loss reads the hazard series; the index series may be in another unit.
    loss_model.check_hazard_unit(hazard.unit)
    if rng is not None:
        rng = _require_generator(rng)
    if missing not in get_args(MissingYears):
        raise SampleError(
            f"missing must be one of {get_args(MissingYears)}, got {missing!r}"
        )
    trend = _require_trend(trend)
    scenario = _identifier("scenario", scenario)
    years = hazard.years
    if index is not None and not np.array_equal(index.years, years):
        only_hazard = np.setdiff1d(years, index.years)[:10].tolist()
        only_index = np.setdiff1d(index.years, years)[:10].tolist()
        raise SampleError(
            "the index series' years differ from the hazard's; only in the hazard: "
            f"{only_hazard}, only in the index: {only_index}"
        )
    first_row = trend._first_row(years.shape[0])
    years = years[first_row:]
    hazard_recorded = hazard.values[first_row:]
    index_recorded = None if index is None else index.values[first_row:]
    hazard_missing = np.isnan(hazard_recorded)
    index_missing = (
        np.isnan(index_recorded) if index_recorded is not None else hazard_missing
    )
    gaps = hazard_missing | index_missing
    dropped = [int(year) for year in years[gaps]]
    if dropped and missing == "raise":
        which = []
        if hazard_missing.any():
            which.append(f"hazard {[int(y) for y in years[hazard_missing]][:10]}")
        if index is not None and index_missing.any():
            which.append(f"index {[int(y) for y in years[index_missing]][:10]}")
        raise _no_value_error(dropped, which)
    keep = ~gaps
    if not keep.any():
        raise SampleError("no year has a value")
    kept_years = years[keep]
    series = [_Series("hazard", hazard.unit, np.array(hazard_recorded[keep]))]
    if index is not None and index_recorded is not None:
        series.append(_Series("index", index.unit, np.array(index_recorded[keep])))
    adjusted, trend_record = trend._apply(years, kept_years, series)
    hazard_values = adjusted[0]
    index_values = adjusted[1] if index is not None else hazard_values.copy()
    n = kept_years.shape[0]

    impact = loss_model.impact_values(hazard_values)
    rng_records = [] if rng is None else [_generator_record(rng, "basis_risk")]
    new_index, impact, basis_summary = _apply_basis_risk(
        basis_risk, index_values, hazard_values, impact, rng
    )
    loss = loss_model.loss_values(impact)

    notes = _dropped_notes(dropped)
    notes += _basis_risk_notes(basis_risk, separate_index=index is not None)
    sources: list[dict[str, Any]] = [
        {"role": "hazard", "unit": hazard.unit, "series": _thaw(hazard.provenance)}
    ]
    licences = [_series_licence("hazard", hazard)]
    if index is not None:
        sources.append(
            {"role": "index", "unit": index.unit, "series": _thaw(index.provenance)}
        )
        licences.append(_series_licence("index", index))
    provenance = SampleProvenance(
        source_kind="annual",
        sources=tuple(sources),
        semantics_source="data",
        loss_model=loss_model._document(asset),
        basis_risk=basis_risk.to_document(),
        rng=tuple(rng_records),
        licences=tuple(licences),
        details={
            "years": [int(kept_years[0]), int(kept_years[-1])],
            "n_years": n,
            "missing": missing,
            "dropped_years": dropped,
            "index_from": "series" if index is not None else "hazard",
            "basis_risk": basis_summary,
        },
        notes=tuple(notes),
        lineage=_annual_lineage(asset, hazard, index),
        trend=trend_record,
    )
    return OutcomeSample(
        asset_ids=(asset,),
        period=kept_years,
        probability=np.full(n, 1.0 / n),
        index=new_index,
        loss=loss,
        hazard=None if np.array_equal(new_index, hazard_values) else hazard_values,
        index_unit=hazard.unit if index is None else index.unit,
        loss_unit=loss_model.value_unit,
        semantics="annual",
        provenance=provenance,
        scenario=scenario,
    )


# -- several hazards, per year -------------------------------------------------


@dataclass(frozen=True)
class _HazardInputs:
    """The checked arguments of `sample_annual_hazards`, by component name."""

    series: dict[str, AnnualSeries]
    tails: dict[str, Tail]
    models: dict[str, LossModel]

    @property
    def impacted(self) -> list[str]:
        """The components with an impact, in component order."""
        return [name for name in self.series if name in self.models]


def _hazard_inputs(
    series: Any, tails: Any, loss_models: Any, basis_risk: Any, asset: str
) -> _HazardInputs:
    """Check the mappings of `sample_annual_hazards`, before anything is drawn."""
    if not isinstance(series, Mapping):
        raise TypeError(
            "series must map each component name to an AnnualSeries, got "
            f"{type(series).__name__}"
        )
    names = list(series)
    if len(names) < 2:
        raise SampleError(
            f"sample_annual_hazards takes two hazards or more, got {len(names)}; "
            "for one hazard use sample_annual"
        )
    for name, each in series.items():
        _component_name(name, SampleError)
        if not isinstance(each, AnnualSeries):
            raise TypeError(
                f"series[{name!r}] must be an AnnualSeries, got {type(each).__name__}"
            )
    severe = _by_component("tails", tails, names, every=True)
    for name, tail in severe.items():
        if tail not in get_args(Tail):
            raise SampleError(
                f"tails[{name!r}] must be 'upper' or 'lower', got {tail!r}"
            )
    models = _hazard_models(
        loss_models,
        names,
        {name: each.unit for name, each in series.items()},
        basis_risk,
        asset,
    )
    return _HazardInputs(series=dict(series), tails=severe, models=models)


def _hazard_models(
    loss_models: Any,
    names: Sequence[str],
    units: Mapping[str, str],
    basis_risk: Any,
    asset: str,
) -> dict[str, LossModel]:
    """Check the loss models of several hazards, before anything is drawn."""
    models = _by_component("loss_models", loss_models, names, every=False)
    if not models:
        raise SampleError(
            "no component has an impact, so the sample would have no loss; give at "
            "least one a LossModel in loss_models"
        )
    for name, model in models.items():
        if not isinstance(model, LossModel):
            raise TypeError(
                f"loss_models[{name!r}] must be a LossModel, got {type(model).__name__}"
            )
        _check_models(model, basis_risk, asset)
        model._require_bound()
        model.check_hazard_unit(units[name])
    _check_one_value(models)
    return models


def _check_one_value(models: Mapping[str, LossModel]) -> None:
    """Raise `SampleError` unless the loss models have one value between them.

    A rule acts on impacts and its result is multiplied by the value, so there
    must be one. Values themselves are not named: a record never holds them.
    """
    for label, stated in (
        ("value_unit", {name: model.value_unit for name, model in models.items()}),
        ("value_column", {name: model.value_column for name, model in models.items()}),
    ):
        if len(set(stated.values())) > 1:
            raise SampleError(
                f"the loss models must have one {label}, but they state {stated}"
            )
    if len({model._value for model in models.values()}) > 1:
        raise SampleError(
            f"the loss models must have one value, but those of {list(models)} differ"
        )


def _combine_record(combine: Any, impacted: Sequence[str]) -> dict[str, Any]:
    """What a sample records of its rule; raises unless one is stated when needed."""
    if len(impacted) >= 2:
        if combine is None:
            raise SampleError(
                f"the components {list(impacted)} each have an impact, so the rule "
                "for their combined loss must be stated, and none is assumed: pass "
                "combine=CappedSum(cap), combine=CompoundDamage() or a CombineRule "
                "of your own"
            )
        return _rule_record(combine)
    if combine is not None:
        raise SampleError(
            f"combine= combines two impacts or more, but only component "
            f"{impacted[0]!r} has one: remove combine=, or add a loss model"
        )
    return _one_impact_record(impacted[0])


def _combined_loss(
    models: Mapping[str, LossModel],
    combine: Any,
    rule_record: Mapping[str, Any],
    impacts: Mapping[str, FloatArray],
) -> FloatArray:
    """A row's loss: its one impact, or the rule's result, times the value."""
    first, only = next(iter(impacts.items()))
    return models[first].loss_values(
        only
        if combine is None
        else _combined_impact(combine, str(rule_record["kind"]), impacts)
    )


def _same_years(series: Mapping[str, AnnualSeries]) -> IntArray:
    """The years every series holds; raises, listing each one's, if they differ."""
    first = next(iter(series.values())).years
    if all(np.array_equal(each.years, first) for each in series.values()):
        return first
    held = "; ".join(
        f"{name}: {_year_spans(each.years.tolist())}" for name, each in series.items()
    )
    raise SampleError(
        "the series must hold the same years, so that a row is one year of every "
        f"hazard; they hold {held}"
    )


def _named_fits(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """A trend record with each fit naming its component, and each one's own.

    The fits were made with the component names as their series' roles, so that
    an error names the series. A record without fits is every component's own.
    """
    fits = [
        {**fit, "role": "hazard", "component": fit["role"]}
        for fit in record.get("fits", ())
    ]
    if not fits:
        return dict(record), {}
    return {**record, "fits": fits}, {
        fit["component"]: {**record, "fits": [fit]} for fit in fits
    }


def _distinct(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The entries, each once, in order."""
    distinct: list[dict[str, Any]] = []
    for entry in entries:
        if entry not in distinct:
            distinct.append(entry)
    return distinct


def sample_annual_hazards(
    series: Mapping[str, AnnualSeries],
    *,
    tails: Mapping[str, Tail],
    loss_models: Mapping[str, LossModel],
    combine: CombineRule | None = None,
    basis_risk: BasisRisk,
    asset_id: str,
    rng: np.random.Generator | None = None,
    missing: MissingYears = "raise",
    trend: Trend = AsRecorded(),
    scenario: str = "baseline",
) -> OutcomeSample:
    """Build a sample of one asset whose rows are the same years of several hazards.

    `series`, `tails` and `loss_models` are mappings by component name, and the
    order of `series` sets the components: the first is the primary, whose
    values the sample's `index` holds. Two or more, each named without
    surrounding whitespace and without `.`, `+`, `[` or `]`. `tails` states each
    component's severe side. Each year is one row with probability `1/n`, and
    `period` is the year.

    The dependence between the hazards is the data's own: the provenance's
    `hazard_dependence` is `{"kind": "data"}`. The rows are the same years, not
    the same events, so two annual maxima of one year may come from different
    events, and the setup text says so.

    The steps come in this order: the window of `RecentYears`; the missing
    years; the fit and the adjustment of every series, one fit each, on the
    same fit years with the same reference year; each component's impact on its
    adjusted hazard; the basis-risk model; the rule. `trend` is one option for
    every series. `basis_risk` is one model for every component, and it draws
    from `rng` in component order, so the first component's draws are those of
    `sample_annual` with the same generator. The provenance's `rng` records the
    generator's state before each component's step.

    **The loss.** A component's impact comes from its `LossModel`, on its
    hazard, after the basis-risk model, and its `own_loss` is that impact times
    the value. `combine` takes the impacts, and the sample's `loss` is its
    result times the value: `CappedSum(cap)`, `CompoundDamage()` or a
    `CombineRule` of the caller's. It is required when two or more components
    have an impact, and refused with fewer. A component left out of
    `loss_models` has no impact: it needs no rule, has no `own_loss`, and the
    basis-risk model doesn't act on it or draw for it. With one impact the
    sample's loss is that component's own, and `combine` records that. The loss
    models must have one value, one `value_unit` and one `value_column`.

    A year without a value in any series raises, unless `missing="drop_year"`
    drops it from all of them and lists it in the provenance. `scenario` is
    the caller's name for the rows, as in `sample_annual`.

    `sample.component(name)` is that hazard's own sample. On the same years,
    with no basis-risk draws or for the first component, it has the content
    hash of `sample_annual` on that series.

    Raises `SampleError` for fewer than two series or an invalid name, tails
    not stated for every component, no impact at all, series that hold
    different years (listing each one's), a hazard unit a component's impact is
    not calibrated for, loss models with different values, units or value
    columns, a missing or invalid rule, missing years, a trend the series cannot
    take or invalid arguments, and `TypeError` for wrong types, a bare callable
    as `combine` included. These are checked before any draw, apart from a
    rule's result. Warns with `ComonotoneIndexWarning` for `NoBasisRisk()`.
    """
    sample = _annual_hazard_rows(
        series,
        tails=tails,
        loss_models=loss_models,
        combine=combine,
        basis_risk=basis_risk,
        asset_id=asset_id,
        rng=rng,
        missing=missing,
        trend=trend,
        scenario=scenario,
    )
    _warn_comonotone(basis_risk, separate_index=False)
    return sample


def _kept_hazards(
    series: Mapping[str, AnnualSeries], missing: MissingYears, trend: Trend
) -> tuple[IntArray, dict[str, FloatArray], list[int], Mapping[str, Any]]:
    """The window, the missing years and the trend, for every series alike.

    Returns the kept years, each series' adjusted values, the years dropped and
    what the trend made.
    """
    all_years = _same_years(series)
    first_row = trend._first_row(all_years.shape[0])
    years = all_years[first_row:]
    recorded = {name: each.values[first_row:] for name, each in series.items()}
    holes = {name: np.isnan(values) for name, values in recorded.items()}
    gaps = np.logical_or.reduce(list(holes.values()))
    dropped = [int(year) for year in years[gaps]]
    if dropped and missing == "raise":
        raise _no_value_error(
            dropped,
            [
                f"{name} {[int(year) for year in years[hole]][:10]}"
                for name, hole in holes.items()
                if hole.any()
            ],
        )
    keep = ~gaps
    if not keep.any():
        raise SampleError("no year has a value in every series")
    kept_years = years[keep]
    adjusted, made = trend._apply(
        years,
        kept_years,
        [
            _Series(name, each.unit, np.array(recorded[name][keep]))
            for name, each in series.items()
        ],
    )
    return kept_years, dict(zip(series, adjusted, strict=True)), dropped, made


def _annual_hazard_rows(
    series: Any,
    *,
    tails: Any,
    loss_models: Any,
    combine: Any,
    basis_risk: BasisRisk,
    asset_id: str,
    rng: np.random.Generator | None,
    missing: MissingYears,
    trend: Trend,
    scenario: str,
) -> OutcomeSample:
    """`sample_annual_hazards`, without its warning: a plan rebuilds per fold."""
    asset = _identifier("asset_id", asset_id)
    inputs = _hazard_inputs(series, tails, loss_models, basis_risk, asset)
    if rng is not None:
        rng = _require_generator(rng)
    if missing not in get_args(MissingYears):
        raise SampleError(
            f"missing must be one of {get_args(MissingYears)}, got {missing!r}"
        )
    trend = _require_trend(trend)
    scenario = _identifier("scenario", scenario)
    rule_record = _combine_record(combine, inputs.impacted)
    kept_years, hazards, dropped, trend_made = _kept_hazards(
        inputs.series, missing, trend
    )

    components: dict[str, IndexComponent] = {}
    impacts: dict[str, FloatArray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    rng_records: dict[str, dict[str, Any]] = {}
    for name, values in hazards.items():
        unit = inputs.series[name].unit
        model = inputs.models.get(name)
        if model is None:
            components[name] = IndexComponent(
                values, unit=unit, tail=inputs.tails[name]
            )
            continue
        impact = model.impact_values(values)
        if rng is not None:
            rng_records[name] = {
                **_generator_record(rng, "basis_risk"),
                "component": name,
            }
        new_index, impacts[name], summaries[name] = _apply_basis_risk(
            basis_risk, values, values, impact, rng
        )
        components[name] = IndexComponent(
            new_index,
            unit=unit,
            tail=inputs.tails[name],
            hazard=None if np.array_equal(new_index, values) else values,
            own_loss=model.loss_values(impacts[name]),
        )
    one_model = inputs.models[inputs.impacted[0]]
    loss = _combined_loss(inputs.models, combine, rule_record, impacts)
    n = kept_years.shape[0]
    first = next(iter(components.values()))
    return OutcomeSample(
        asset_ids=(asset,),
        period=kept_years,
        probability=np.full(n, 1.0 / n),
        index=first.values,
        loss=loss,
        hazard=first.hazard,
        index_unit=first.unit,
        loss_unit=one_model.value_unit,
        semantics="annual",
        provenance=_hazards_provenance(
            inputs,
            asset=asset,
            basis_risk=basis_risk,
            kept_years=kept_years,
            dropped=dropped,
            missing=missing,
            trend_made=trend_made,
            rule_record=rule_record,
            summaries=summaries,
            rng_records=rng_records,
        ),
        scenario=scenario,
        index_components=components,
    )


def _hazards_provenance(
    inputs: _HazardInputs,
    *,
    asset: str,
    basis_risk: BasisRisk,
    kept_years: IntArray,
    dropped: Sequence[int],
    missing: MissingYears,
    trend_made: Mapping[str, Any],
    rule_record: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    rng_records: Mapping[str, Mapping[str, Any]],
) -> SampleProvenance:
    """The provenance of a per-year sample of several hazards, and each one's own."""
    series, models = inputs.series, inputs.models
    details = {
        "years": [int(kept_years[0]), int(kept_years[-1])],
        "n_years": int(kept_years.shape[0]),
        "missing": missing,
        "dropped_years": list(dropped),
        "index_from": "hazard",
    }
    shared_notes = _dropped_notes(dropped)
    own_notes = {
        name: _basis_risk_notes(basis_risk, separate_index=False)
        for name in inputs.impacted
    }
    trend_record, own_trends = _named_fits(trend_made)
    sources = {
        name: {
            "role": "hazard",
            "component": name,
            "unit": each.unit,
            "series": _thaw(each.provenance),
        }
        for name, each in series.items()
    }
    licences = {name: _series_licence("hazard", each) for name, each in series.items()}
    loss_documents = {name: models[name]._document(asset) for name in inputs.impacted}
    return SampleProvenance(
        source_kind="annual",
        sources=tuple(sources.values()),
        semantics_source="data",
        # A list, because canonical JSON sorts keys and the order is the components'.
        loss_model={
            "kind": "components",
            "components": [
                {"name": name, "loss_model": document}
                for name, document in loss_documents.items()
            ],
        },
        basis_risk=basis_risk.to_document(),
        rng=tuple(rng_records.values()),
        licences=tuple(_distinct(licences.values())),
        details={**details, "basis_risk": summaries},
        notes=(
            *shared_notes,
            *(
                _component_note(name, note)
                for name, notes in own_notes.items()
                for note in notes
            ),
        ),
        lineage=content_hash(
            {
                "kind": "annual_hazards",
                "asset_id": asset,
                "hazards": {
                    name: _series_identity(each) for name, each in series.items()
                },
            }
        ),
        trend=trend_record,
        hazard_dependence={"kind": "data"},
        combine=rule_record,
        components={
            name: SampleProvenance(
                source_kind="annual",
                sources=(sources[name],),
                semantics_source="data",
                loss_model=loss_documents.get(name),
                basis_risk=basis_risk.to_document(),
                rng=(rng_records[name],) if name in rng_records else (),
                licences=(licences[name],),
                details={**details, "basis_risk": summaries.get(name)},
                notes=(*shared_notes, *own_notes.get(name, ())),
                trend=own_trends.get(name, trend_record),
            ).to_document()
            for name in series
        },
    )


# -- pooled model-years --------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelRun:
    """One model run's `AnnualSeries` at one location, for `sample_annual_pool`.

    `model`, `member` and `pathway` name the run: for CMIP6 and NEX-GDDP-CMIP6,
    the source id, the variant label and the experiment. They are the caller's
    statement and are recorded with the run.

    Raises `TypeError` for a `series` that is not an `AnnualSeries`, and
    `SampleError` for a name that is not a non-empty string without surrounding
    whitespace.
    """

    series: AnnualSeries
    model: str = field(kw_only=True)
    member: str = field(kw_only=True)
    pathway: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.series, AnnualSeries):
            raise TypeError(
                f"series must be an AnnualSeries, got {type(self.series).__name__}"
            )
        for name in ("model", "member", "pathway"):
            _identifier(name, getattr(self, name))

    @property
    def _name(self) -> str:
        """The run's model and member, as an error message names it."""
        return f"{self.model} {self.member}"


def _stated_recipes(series: AnnualSeries) -> set[str]:
    """The recipes a series' provenance states, if any."""
    provenance = _thaw(series.provenance)
    source = provenance.get("source")
    stated = (source.get("recipe") if isinstance(source, Mapping) else None) or None
    recipes = {str(stated)} if stated is not None else set()
    recorded = provenance.get("table_columns")
    if isinstance(recorded, Mapping):
        recipes |= {
            str(value) for value in (recorded.get("recipe") or {}).get("values", ())
        }
    return recipes


def _checked_runs(runs: Any) -> tuple[ModelRun, ...]:
    """A pool's runs in (model, member) order, after the refusals of a pool."""
    if isinstance(runs, (str, bytes, Mapping)) or not isinstance(runs, Iterable):
        raise TypeError(
            f"runs must be an iterable of ModelRun, got {type(runs).__name__}"
        )
    given = tuple(runs)
    for run in given:
        if not isinstance(run, ModelRun):
            raise TypeError(f"runs must hold ModelRun, got {type(run).__name__}")
    if len(given) < 2:
        raise SampleError(f"a pool needs two runs or more, got {len(given)}")
    pathways = sorted({run.pathway for run in given})
    if len(pathways) > 1:
        raise SampleError(
            f"a pool holds one pathway, but its runs are under {pathways}; "
            "'historical' is a pathway too, so a pool cannot span historical and "
            "SSP experiments"
        )
    units = sorted({run.series.unit for run in given})
    if len(units) > 1:
        raise SampleError(f"a pool needs one unit in every run, got {units}")
    recipes = sorted(
        {recipe for run in given for recipe in _stated_recipes(run.series)}
    )
    if len(recipes) > 1:
        raise SampleError(f"a pool needs one recipe in every run, got {recipes}")
    models = [run.model for run in given]
    repeated = sorted({model for model in models if models.count(model) > 1})
    if repeated:
        members = {
            model: sorted(run.member for run in given if run.model == model)
            for model in repeated
        }
        raise SampleError(
            f"a model appears once in a pool, whatever its member, so that every "
            f"model has the same weight; got {members}"
        )
    return tuple(
        sorted(
            given,
            key=lambda run: (run.model.encode("utf-8"), run.member.encode("utf-8")),
        )
    )


def _pool_years(
    runs: Sequence[ModelRun], trend: Trend, missing: MissingYears
) -> tuple[IntArray, IntArray, list[FloatArray], list[list[int]], list[int]]:
    """The years a pool holds, after the window and the missing years.

    Returns all the years of the runs' windows, the years kept, each run's
    values on them, the years each run had no value for, and the years dropped.
    Raises `SampleError` unless every run holds the same years then.
    """
    windows: list[IntArray] = []
    values: list[FloatArray] = []
    for run in runs:
        try:
            first_row = trend._first_row(len(run.series))
        except SampleError as error:
            raise SampleError(f"run {run._name}: {error}") from error
        windows.append(run.series.years[first_row:])
        values.append(run.series.values[first_row:])
    lacked = [
        [int(year) for year in years[np.isnan(recorded)]]
        for years, recorded in zip(windows, values, strict=True)
    ]
    dropped = sorted({year for years in lacked for year in years})
    if dropped and missing == "raise":
        listed = "; ".join(
            f"{run._name}: {_year_spans(years)}"
            for run, years in zip(runs, lacked, strict=True)
            if years
        )
        years_have, them = (
            ("year has", "it") if len(dropped) == 1 else ("years have", "them")
        )
        raise SampleError(
            f"{len(dropped)} {years_have} no value in a run ({listed}); pass "
            f"missing='drop_year' to drop {them} from every run, listed in the "
            "provenance"
        )
    gone = np.array(dropped, dtype=np.int64)
    held = [years[~np.isin(years, gone)] for years in windows]
    if any(not np.array_equal(years, held[0]) for years in held[1:]):
        listed = "; ".join(
            f"{run._name}: {_year_spans([int(year) for year in years])}"
            for run, years in zip(runs, held, strict=True)
        )
        raise SampleError(
            "every run of a pool must hold the same years, so that the latest "
            f"years are one block for every model; the runs hold {listed}. A stated "
            "years= resolves it"
        )
    kept = held[0]
    if kept.size == 0:
        raise SampleError("no year has a value in every run of the pool")
    on_kept = [
        recorded[np.isin(years, kept)]
        for years, recorded in zip(windows, values, strict=True)
    ]
    return np.unique(np.concatenate(windows)), kept, on_kept, lacked, dropped


def sample_annual_pool(
    runs: Iterable[ModelRun],
    *,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    rng: np.random.Generator | None = None,
    missing: MissingYears = "raise",
    trend: Trend = AsRecorded(),
    scenario: str = "baseline",
) -> OutcomeSample:
    """Pool the model-years of several runs into one asset's sample.

    `runs` holds one `ModelRun` per model, all at one location. Every
    model-year is one row with probability `1/N`, so 3 models of `n` years give
    `3n` rows at `1/(3n)` each. The rows are in (year, model, member) order,
    with models and members in UTF-8 byte order, so the order of `runs` does
    not matter. `period` is the draw number, 0 to `N - 1`. The provenance's
    `pool` records each row's year and run, and each run's model, member,
    pathway, lineage and series.

    The steps come in this order: the window of `RecentYears`, taken from each
    run; the missing years; the check that every run holds the same years; the
    fit and the adjustment, one fit per run on the same years, with the same
    reference year, the last of the pool's years; the impact on the adjusted
    hazard; the basis-risk model. A year without a value in a run raises,
    unless `missing="drop_year"` drops it from every run and lists it with the
    run that lacked it. `rng` is needed only by a basis-risk model that draws:
    `ImpactNoise` draws one factor per row, in row order. `trend` and
    `scenario` are those of `sample_annual`.

    `holdout` and `year_block_kfold` split a pool by its years, so every run
    holds out the same latest years, and the leakage check compares (run
    lineage, year) pairs, so a model-year cannot be on both sides, between
    pools of different runs or between a pool and one of its runs. To hold out
    a block under `LinearTrend`, state the design years with `fit_years=`.

    A draw mixes model spread with year-to-year variability. The rows are not
    independent years: rows of one run share that model's climate and rows of
    one year the pathway's forcing, so the number of rows is not an effective
    sample size.

    Raises `SampleError` for fewer than two runs; runs under different
    pathways, in different units or with different stated recipes; the same
    model twice, whatever the member; runs that do not hold the same years; a
    missing year (unless `missing="drop_year"`); a hazard unit the loss model's
    impact is not calibrated for; a trend a run cannot take; and invalid
    arguments. Raises `TypeError` for wrong types. Warns with
    `ComonotoneIndexWarning` for `NoBasisRisk()`, as for one series.
    """
    sample = _pool_rows(
        runs,
        loss_model=loss_model,
        basis_risk=basis_risk,
        asset_id=asset_id,
        rng=rng,
        missing=missing,
        trend=trend,
        scenario=scenario,
    )
    _warn_comonotone(basis_risk, separate_index=False)
    return sample


def _pool_rows(
    runs: Iterable[ModelRun],
    *,
    loss_model: LossModel,
    basis_risk: BasisRisk,
    asset_id: str,
    rng: np.random.Generator | None,
    missing: MissingYears,
    trend: Trend,
    scenario: str,
) -> OutcomeSample:
    """`sample_annual_pool`, without its warning: a plan rebuilds it per fold."""
    ordered = _checked_runs(runs)
    asset = _identifier("asset_id", asset_id)
    _check_models(loss_model, basis_risk, asset)
    unit = ordered[0].series.unit
    loss_model.check_hazard_unit(unit)
    if rng is not None:
        rng = _require_generator(rng)
    if missing not in get_args(MissingYears):
        raise SampleError(
            f"missing must be one of {get_args(MissingYears)}, got {missing!r}"
        )
    trend = _require_trend(trend)
    scenario = _identifier("scenario", scenario)
    all_years, kept, on_kept, lacked, dropped = _pool_years(ordered, trend, missing)
    series = [
        _Series(f"hazard of run {run._name}", unit, np.array(values))
        for run, values in zip(ordered, on_kept, strict=True)
    ]
    adjusted, trend_record = trend._apply(all_years, kept, series)
    if "fits" in trend_record:
        trend_record["fits"] = [
            {**fit, "role": "hazard", "model": run.model, "member": run.member}
            for fit, run in zip(trend_record["fits"], ordered, strict=True)
        ]
    # One row per year and run: a column per run, read row by row.
    hazard_values = np.column_stack(adjusted).reshape(-1)
    count = hazard_values.shape[0]

    impact = loss_model.impact_values(hazard_values)
    rng_records = [] if rng is None else [_generator_record(rng, "basis_risk")]
    new_index, impact, basis_summary = _apply_basis_risk(
        basis_risk, hazard_values.copy(), hazard_values, impact, rng
    )
    loss = loss_model.loss_values(impact)

    notes: list[str] = []
    if dropped:
        noun, verb = ("year", "was") if len(dropped) == 1 else ("years", "were")
        notes.append(
            f"{len(dropped)} {noun} without a value in a run {verb} dropped from "
            f"every run: {_year_spans(dropped)}."
        )
    notes += _basis_risk_notes(basis_risk, separate_index=False)
    lineages = [_annual_lineage(asset, run.series, None) for run in ordered]
    licences: list[dict[str, Any]] = []
    for run in ordered:
        licence = _series_licence("hazard", run.series)
        if licence not in licences:
            licences.append(licence)
    provenance = SampleProvenance(
        source_kind="annual",
        sources=tuple(
            {
                "role": "hazard",
                "unit": unit,
                "series": _thaw(run.series.provenance),
                "model": run.model,
                "member": run.member,
                "pathway": run.pathway,
            }
            for run in ordered
        ),
        semantics_source="data",
        loss_model=loss_model._document(asset),
        basis_risk=basis_risk.to_document(),
        rng=tuple(rng_records),
        licences=tuple(licences),
        details={
            "years": [int(kept[0]), int(kept[-1])],
            "n_years": int(kept.shape[0]),
            "n_runs": len(ordered),
            "missing": missing,
            "dropped_years": dropped,
            "index_from": "hazard",
            "basis_risk": basis_summary,
        },
        notes=tuple(notes),
        lineage=content_hash(
            {
                "kind": "pool",
                "asset_id": asset,
                "runs": [
                    {"model": run.model, "member": run.member, "lineage": lineage}
                    for run, lineage in zip(ordered, lineages, strict=True)
                ],
            }
        ),
        trend=trend_record,
        pool={
            "pathway": ordered[0].pathway,
            "runs": [
                {
                    "model": run.model,
                    "member": run.member,
                    "pathway": run.pathway,
                    "lineage": lineage,
                    "series": _thaw(run.series.provenance),
                    "dropped_years": years,
                }
                for run, lineage, years in zip(ordered, lineages, lacked, strict=True)
            ],
            "year": np.repeat(kept, len(ordered)).tolist(),
            "run": np.tile(np.arange(len(ordered)), kept.shape[0]).tolist(),
        },
    )
    return OutcomeSample(
        asset_ids=(asset,),
        period=np.arange(count, dtype=np.int64),
        probability=np.full(count, 1.0 / count),
        index=new_index,
        loss=loss,
        hazard=None if np.array_equal(new_index, hazard_values) else hazard_values,
        index_unit=unit,
        loss_unit=loss_model.value_unit,
        semantics="annual",
        provenance=provenance,
        scenario=scenario,
    )
