"""
Radiometric bookkeeping: decibel handling, look counts, and speckle statistics.

The central idea of this package is that a SAR observation of a polygon is an
*estimate with a knowable precision*, not a point value. That precision is
governed by the number of independent looks contributing to the estimate, so
looks are tracked explicitly rather than assumed.

Two distinct quantities are recorded, and they are not interchangeable:

``looks_nominal``
    The multilooking factor implied by the processing configuration, i.e. the
    ratio of the output pixel area to the native single-look resolution cell.
    Deterministic and known before any data is read.

``enl_empirical``
    The equivalent number of looks estimated from the observed intensity
    sample. Always lower than nominal because of range oversampling, spectral
    correlation between adjacent samples, interpolation during terrain
    correction, and genuine scene texture.

The gap between them is informative. For a homogeneous target it measures
processing-induced correlation; for a heterogeneous target it is dominated by
real within-polygon variability, which for agricultural fields is the signal of
interest rather than a nuisance.

Distributional basis
--------------------
For fully developed speckle, an L-look intensity (linear power) observation of a
target with mean backscatter :math:`\\sigma` is Gamma distributed with shape
:math:`L` and scale :math:`\\sigma/L`. Hence

.. math::

    \\mathrm{Var}[I] = \\sigma^2 / L, \\qquad
    \\mathrm{ENL} = \\mathrm{E}[I]^2 / \\mathrm{Var}[I]

and the standard error of a mean over :math:`n` pixels with effective looks
:math:`L` is :math:`\\sigma / \\sqrt{nL}`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_db",
    "to_linear",
    "mean_linear_then_db",
    "nominal_looks",
    "estimate_enl",
    "speckle_standard_error_db",
    "coefficient_of_variation",
    "gamma_change_statistic",
]

# Sentinel-1 IW single-look resolution cell, ground range x azimuth, in metres.
# Used to derive nominal looks from a requested output pixel spacing.
S1_IW_SLC_RESOLUTION_M = (2.7, 22.0)
S1_IW_SLC_PIXEL_SPACING_M = (2.3, 14.1)
S1_IW_GRDH_PIXEL_SPACING_M = (10.0, 10.0)
S1_IW_GRDH_NOMINAL_ENL = 4.4


def to_db(linear: np.ndarray | float) -> np.ndarray:
    """Convert linear power to decibels, mapping non-positive values to NaN."""
    arr = np.asarray(linear, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.where(arr > 0, arr, np.nan))
    return out


def to_linear(db: np.ndarray | float) -> np.ndarray:
    """Convert decibels to linear power."""
    arr = np.asarray(db, dtype=np.float64)
    return np.power(10.0, arr / 10.0)


def mean_linear_then_db(values, already_db: bool = False) -> float:
    """
    Aggregate backscatter correctly.

    Averaging in the decibel domain is biased low, and the size of the bias
    grows with within-polygon heterogeneity. Since heterogeneity is precisely
    what a within-field stability analysis is measuring, the bias is correlated
    with the quantity of interest and cannot be treated as noise. Aggregation
    therefore always happens in linear power.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    linear = to_linear(arr) if already_db else arr
    linear = linear[linear > 0]
    if linear.size == 0:
        return float("nan")
    return float(to_db(linear.mean()))


def nominal_looks(
    output_spacing_m: float,
    source: str = "SLC",
    native_spacing_m: tuple[float, float] | None = None,
) -> dict[str, float]:
    """
    Derive the multilooking factor implied by a requested output spacing.

    This is geometry, not measurement: it describes how many native resolution
    cells are averaged into one output pixel. The result is recorded per
    observation so that downstream models know how the data were formed.

    For GRD input the product is already multilooked and the additional looks
    gained by resampling are reported on top of the product's nominal ENL.
    """
    source = source.upper()
    if source == "SLC":
        native = native_spacing_m or S1_IW_SLC_PIXEL_SPACING_M
        base_looks = 1.0
    elif source.startswith("GRD"):
        native = native_spacing_m or S1_IW_GRDH_PIXEL_SPACING_M
        base_looks = S1_IW_GRDH_NOMINAL_ENL
    else:
        raise ValueError(f"Unknown source product type: {source!r}")

    looks_range = max(1.0, output_spacing_m / native[0])
    looks_azimuth = max(1.0, output_spacing_m / native[1])
    return {
        "looks_range": float(looks_range),
        "looks_azimuth": float(looks_azimuth),
        "looks_nominal": float(base_looks * looks_range * looks_azimuth),
        "native_spacing_range_m": float(native[0]),
        "native_spacing_azimuth_m": float(native[1]),
        "output_spacing_m": float(output_spacing_m),
        "source_product": source,
    }


def estimate_enl(
    intensity: np.ndarray,
    already_db: bool = False,
    min_pixels: int = 16,
    trim_percentile: float | None = None,
) -> float:
    """
    Estimate the equivalent number of looks from an intensity sample.

    ENL is the squared mean over the variance of *linear power*. Computing it
    on decibel values is a common and serious error: the log transform is
    variance-stabilising, so it yields a number with no relationship to the
    look count.

    Parameters
    ----------
    intensity
        Pixel values for one polygon and one acquisition.
    already_db
        Set when the input is in decibels so it can be converted back.
    min_pixels
        Below this count the estimator is too unstable to report. The
        variance of the ENL estimator itself is roughly ``2 L^2 / n``, so at
        n = 16 the relative standard error is already about 35 percent.
    trim_percentile
        If given, symmetrically trim this percentage from each tail before
        estimating. Guards against bright scatterers such as farm buildings or
        standing water edges, which inflate variance and depress ENL.
    """
    arr = np.asarray(intensity, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if already_db:
        arr = to_linear(arr)
    arr = arr[arr > 0]

    if arr.size < min_pixels:
        return float("nan")

    if trim_percentile:
        lo, hi = np.percentile(arr, [trim_percentile, 100.0 - trim_percentile])
        arr = arr[(arr >= lo) & (arr <= hi)]
        if arr.size < min_pixels:
            return float("nan")

    mean = arr.mean()
    var = arr.var(ddof=1)
    if var <= 0:
        return float("nan")
    return float(mean * mean / var)


def coefficient_of_variation(
    intensity: np.ndarray, already_db: bool = False
) -> float:
    """
    Linear-domain coefficient of variation.

    For fully developed speckle with L looks the CV is 1/sqrt(L), so values
    materially above that indicate real texture rather than speckle. This is
    the quantity that distinguishes a uniform canopy from a patchy one.
    """
    arr = np.asarray(intensity, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if already_db:
        arr = to_linear(arr)
    arr = arr[arr > 0]
    if arr.size < 2:
        return float("nan")
    mean = arr.mean()
    if mean <= 0:
        return float("nan")
    return float(arr.std(ddof=1) / mean)


def speckle_standard_error_db(n_pixels: int, enl: float) -> float:
    """
    Approximate standard error of a polygon-mean backscatter, in decibels.

    The mean of ``n`` pixels each with ``enl`` effective looks has relative
    standard error ``1 / sqrt(n * enl)`` in linear power. Propagating through
    the log transform gives the decibel-domain error below. This is the weight
    a downstream panel model should use, and it varies by an order of magnitude
    between a two-hectare plot and a hundred-hectare field.
    """
    if not np.isfinite(enl) or enl <= 0 or n_pixels < 1:
        return float("nan")
    relative = 1.0 / np.sqrt(n_pixels * enl)
    return float((10.0 / np.log(10.0)) * relative)


def gamma_change_statistic(
    intensity_a: np.ndarray,
    intensity_b: np.ndarray,
    looks_a: float,
    looks_b: float,
    already_db: bool = False,
) -> dict[str, float]:
    """
    Likelihood-ratio test for a change in mean backscatter between two dates.

    Under the null of no change, and treating each observation as Gamma
    distributed with known look counts, the ratio of the geometric to the
    arithmetic mean of the two sample means forms a likelihood ratio whose
    logarithm is asymptotically chi-squared with one degree of freedom.

    This is the principled replacement for the fixed decibel thresholds that
    most amplitude change detection uses. Because the null distribution depends
    on the look counts, a threshold calibrated on large fields is wrong on small
    ones, which is exactly the failure mode that look tracking removes.
    """
    a = np.asarray(intensity_a, dtype=np.float64)
    b = np.asarray(intensity_b, dtype=np.float64)
    if already_db:
        a, b = to_linear(a), to_linear(b)
    a = a[np.isfinite(a) & (a > 0)]
    b = b[np.isfinite(b) & (b > 0)]
    if a.size == 0 or b.size == 0:
        return {"log_ratio_db": float("nan"), "statistic": float("nan"), "n_a": 0, "n_b": 0}

    mean_a, mean_b = a.mean(), b.mean()
    la = looks_a * a.size
    lb = looks_b * b.size

    pooled = (la * mean_a + lb * mean_b) / (la + lb)
    stat = 2.0 * (
        la * np.log(pooled / mean_a) + lb * np.log(pooled / mean_b)
    )
    return {
        "log_ratio_db": float(10.0 * np.log10(mean_b / mean_a)),
        "statistic": float(stat),
        "effective_looks_a": float(la),
        "effective_looks_b": float(lb),
        "n_a": int(a.size),
        "n_b": int(b.size),
    }
