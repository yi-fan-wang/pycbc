# Copyright (C) 2026 Yifan Wang
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

"""Periastron-resolved signal-consistency statistic for eccentric templates.

The orbital dynamics define two apastron boundaries around the final three
resolved periastra. The boundaries split the production search template into
three time-domain bins:

    early burst train | penultimate periastron | final periastron + merger

The statistic is the two-degree-of-freedom increment
``chi2(K=3) - chi2(K=2)``. The K=2 partition merges the first two bins, so the
increment tests only the additional penultimate-periastron resolution. Full
complex bin covariance is retained because hard time bins are not orthogonal
in the detector-noise inner product.

SEOBNRv5E dynamics are used only to locate the boundaries. The filter and all
subfilters are exact partitions of the production PyCBC template.
"""

import logging

import numpy

from pycbc.filter import matched_filter_core
from pycbc.types import Array, complex64


class PeriastronChisqNotApplicable(ValueError):
    """Raised when a template has fewer than three resolved periastra."""


def _refine_minimum_times(time, radius, minimum_indices):
    refined = []
    for index in minimum_indices:
        local_time = time[index - 1:index + 2] - time[index]
        local_radius = radius[index - 1:index + 2]
        quadratic, linear, _ = numpy.polyfit(
            local_time, local_radius, 2
        )
        if quadratic <= 0.0:
            refined.append(time[index])
            continue
        offset = -linear / (2.0 * quadratic)
        if local_time[0] <= offset <= local_time[-1]:
            refined.append(time[index] + offset)
        else:
            refined.append(time[index])
    return numpy.asarray(refined)


def extract_periastra(dynamics_time, radius):
    """Locate periastra from local minima of the EOB radial separation."""
    minimum_indices = (
        numpy.flatnonzero(
            (radius[1:-1] < radius[:-2])
            & (radius[1:-1] <= radius[2:])
        )
        + 1
    )
    return _refine_minimum_times(
        dynamics_time, radius, minimum_indices
    )


def find_apastra(dynamics_time, radius, periastra):
    """Locate the maximum separation between each adjacent periastron."""
    apastra = []
    for left, right in zip(periastra[:-1], periastra[1:]):
        between = (dynamics_time > left) & (dynamics_time < right)
        if not numpy.any(between):
            raise ValueError(
                "No dynamics samples lie between adjacent periastra"
            )
        indices = numpy.flatnonzero(between)
        apastra.append(
            dynamics_time[indices[numpy.argmax(radius[indices])]]
        )
    return numpy.asarray(apastra)


def generate_dynamics_metadata(params, sample_rate):
    """Generate reusable periastron-boundary metadata from bank parameters."""
    # These imports are deliberately lazy: pyseobnr is an optional dependency
    # unless the statistic is explicitly enabled.
    import lal
    from pyseobnr.generate_waveform import GenerateWaveform

    f_lower = float(params.f_lower)
    waveform_parameters = {
        "approximant": "SEOBNRv5EHM",
        "mass1": float(params.mass1),
        "mass2": float(params.mass2),
        "spin1z": float(params.spin1z),
        "spin2z": float(params.spin2z),
        "eccentricity": float(params.eccentricity),
        "rel_anomaly": float(params.rel_anomaly),
        "f22_start": f_lower,
        "f_ref": f_lower,
        "deltaT": 1.0 / sample_rate,
        "ModeArray": [(2, 2)],
        "lmax_nyquist": 1,
    }
    generator = GenerateWaveform(waveform_parameters)
    hplus, _ = generator.generate_td_polarizations_conditioned_1()

    total_mass = float(params.mass1 + params.mass2)
    dynamics = numpy.asarray(generator._model.dynamics)
    dynamics_time = dynamics[:, 0] * total_mass * lal.MTSUN_SI
    periastra = extract_periastra(dynamics_time, dynamics[:, 1])
    samples = numpy.asarray(hplus.data.data)
    metadata = {
        "num_periastra": int(len(periastra)),
        "boundary_indices": numpy.asarray([], dtype=numpy.int64),
        "waveform_samples": int(len(samples)),
        "direct_peak_index": int(numpy.argmax(numpy.abs(samples))),
    }
    if len(periastra) >= 3:
        apastra = find_apastra(
            dynamics_time, dynamics[:, 1], periastra
        )
        sample_times = (
            float(hplus.epoch)
            + numpy.arange(len(samples)) * float(hplus.deltaT)
        )
        boundary_times = float(hplus.epoch) + apastra[-2:]
        boundary_indices = numpy.searchsorted(
            sample_times, boundary_times
        ).astype(numpy.int64)
        if (
            numpy.any(boundary_indices <= 0)
            or numpy.any(boundary_indices >= len(samples))
            or numpy.any(numpy.diff(boundary_indices) <= 0)
        ):
            raise ValueError(
                "Periastron chi-square boundaries are not increasing "
                "and internal"
            )
        metadata["boundary_indices"] = boundary_indices
    return metadata


def dynamics_partition(template, sample_rate):
    """Return the two final apastron boundaries for one search template."""
    metadata = generate_dynamics_metadata(
        template.params, sample_rate
    )
    if metadata["num_periastra"] < 3:
        raise PeriastronChisqNotApplicable(
            "Incremental K=3 periastron chi-square requires at least "
            "three resolved periastra"
        )
    return metadata


def partition_from_bank_metadata(params):
    """Return a stored dynamics partition, or ``None`` if none is present."""
    field_names = (
        "ecc_tchisq_boundary_index_0",
        "ecc_tchisq_boundary_index_1",
        "ecc_tchisq_waveform_samples",
        "ecc_tchisq_direct_peak_index",
    )
    values = [getattr(params, name, None) for name in field_names]
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        missing = [
            name for name, exists in zip(field_names, present) if not exists
        ]
        raise ValueError(
            "Incomplete stored periastron partition metadata; missing "
            + ", ".join(missing)
        )

    num_periastra = int(params.ecc_tchisq_num_periastra)
    boundaries = numpy.asarray(values[:2], dtype=numpy.int64)
    waveform_samples = int(values[2])
    direct_peak_index = int(values[3])
    if num_periastra < 3:
        raise PeriastronChisqNotApplicable(
            "Stored template has fewer than three resolved periastra"
        )
    if (
        numpy.any(boundaries <= 0)
        or numpy.any(numpy.diff(boundaries) <= 0)
        or boundaries[-1] >= waveform_samples
        or direct_peak_index < 0
        or direct_peak_index >= waveform_samples
    ):
        raise ValueError("Stored periastron partition metadata is invalid")
    return {
        "num_periastra": num_periastra,
        "boundary_indices": boundaries,
        "waveform_samples": waveform_samples,
        "direct_peak_index": direct_peak_index,
    }


def split_production_template(template, sample_rate, partition):
    """Split the production template into three exactly closing time bins."""
    delta_t = 1.0 / sample_rate
    production_time = template.to_timeseries(delta_t=delta_t)
    production_peak_index = int(
        numpy.argmax(numpy.abs(numpy.asarray(production_time)))
    )
    production_start_index = (
        production_peak_index - partition["direct_peak_index"]
    ) % len(production_time)
    mapped_indices = (
        production_start_index
        + numpy.arange(partition["waveform_samples"])
    ) % len(production_time)

    assigned = production_time.copy()
    assigned.clear()
    bins = []
    start = 0
    for end in partition["boundary_indices"]:
        segment = production_time.copy()
        segment.clear()
        indices = mapped_indices[start:end]
        segment.data[indices] = production_time.data[indices]
        bins.append(segment)
        assigned += segment
        start = int(end)
    # Conditioning samples outside the direct waveform support belong to the
    # final merger bin. This makes the sum exactly equal to the search template.
    bins.append(production_time - assigned)
    frequency_bins = [
        segment.to_frequencyseries(
            delta_f=template.delta_f
        ).astype(complex64)
        for segment in bins
    ]
    if len(frequency_bins) != 3:
        raise RuntimeError(
            "Incremental periastron chi-square requires exactly three bins"
        )
    return frequency_bins


def noise_inner(left, right, psd, low_frequency_cutoff):
    """Return the complex one-sided detector-noise inner product."""
    delta_f = float(left.delta_f)
    kmin = int(low_frequency_cutoff / delta_f)
    psd_values = numpy.asarray(psd, dtype=numpy.float64)
    valid = (
        (numpy.arange(len(psd_values)) >= kmin)
        & numpy.isfinite(psd_values)
        & (psd_values > 0.0)
    )
    left_values = numpy.asarray(left, dtype=numpy.complex128)
    right_values = numpy.asarray(right, dtype=numpy.complex128)
    return (
        4.0
        * delta_f
        * numpy.sum(
            left_values[valid].conj()
            * right_values[valid]
            / psd_values[valid]
        )
    )


def bin_covariance(template, bins, psd, low_frequency_cutoff):
    """Build the normalized full complex covariance of the three bins."""
    full_norm = float(
        numpy.real(
            noise_inner(
                template, template, psd, low_frequency_cutoff
            )
        )
    )
    if not numpy.isfinite(full_norm) or full_norm <= 0.0:
        raise ValueError("Template has a non-positive detector-noise norm")
    covariance = numpy.empty((3, 3), dtype=numpy.complex128)
    for row in range(3):
        for column in range(3):
            covariance[row, column] = (
                noise_inner(
                    bins[row],
                    bins[column],
                    psd,
                    low_frequency_cutoff,
                )
                / full_norm
            )
    covariance = 0.5 * (covariance + covariance.conj().T)
    return covariance, full_norm


def residual_model(covariance):
    """Return the projection and pseudoinverse for covariance-aware residuals."""
    num_bins = len(covariance)
    ones = numpy.ones(num_bins, dtype=numpy.complex128)
    expected = covariance @ ones
    projection = (
        numpy.eye(num_bins)
        - numpy.outer(expected, ones.conj())
    )
    residual_covariance = (
        projection @ covariance @ projection.conj().T
    )
    residual_covariance = 0.5 * (
        residual_covariance + residual_covariance.conj().T
    )
    eigenvalues, eigenvectors = numpy.linalg.eigh(
        residual_covariance
    )
    tolerance = max(eigenvalues[-1] * 1e-10, 1e-14)
    keep = eigenvalues > tolerance
    inverse = (
        eigenvectors[:, keep]
        @ numpy.diag(1.0 / eigenvalues[keep])
        @ eigenvectors[:, keep].conj().T
    )
    return projection, inverse, int(numpy.sum(keep))


def covariance_chisq(values, covariance):
    """Evaluate the covariance-aware chi-square for many complex bin vectors."""
    projection, inverse, rank = residual_model(covariance)
    residual = numpy.asarray(values) @ projection.T
    quadratic = numpy.real(
        numpy.einsum(
            "bi,ij,bj->b",
            residual.conj(),
            inverse,
            residual,
        )
    )
    return 2.0 * quadratic, 2 * rank


def incremental_k3_chisq(values, covariance):
    """Return reduced ``chi2(K=3) - chi2(K=2)`` and its two dof."""
    values = numpy.asarray(values, dtype=numpy.complex128)
    if values.ndim == 1:
        values = values[numpy.newaxis, :]
    if values.shape[1] != 3 or covariance.shape != (3, 3):
        raise ValueError("K=3 values and a 3x3 covariance are required")

    chi3, dof3 = covariance_chisq(values, covariance)
    aggregation = numpy.asarray(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=numpy.complex128,
    )
    values_k2 = values @ aggregation.T
    covariance_k2 = (
        aggregation @ covariance @ aggregation.conj().T
    )
    chi2, dof2 = covariance_chisq(values_k2, covariance_k2)
    incremental_dof = dof3 - dof2
    if incremental_dof != 2:
        raise RuntimeError(
            f"Expected two incremental dof, found {incremental_dof}"
        )

    incremental = chi3 - chi2
    scale = numpy.maximum.reduce(
        (numpy.abs(chi3), numpy.abs(chi2), numpy.ones(len(chi3)))
    )
    if numpy.any(incremental < -1e-8 * scale):
        raise RuntimeError(
            "Nested periastron chi-square decreased beyond numerical tolerance"
        )
    incremental = numpy.maximum(incremental, 0.0)
    return incremental / incremental_dof, incremental_dof


class SingleDetPeriastronChisq:
    """Calculate incremental K=3 periastron chi-square for search triggers."""

    def __init__(
        self,
        enabled,
        sample_rate,
        low_frequency_cutoff,
    ):
        self.do = bool(enabled)
        self.sample_rate = float(sample_rate)
        self.low_frequency_cutoff = float(low_frequency_cutoff)
        self._template_key = None
        self._template_bins = None
        self._not_applicable = False

    @staticmethod
    def _key(template):
        params = template.params
        return (
            int(params.template_hash),
            len(template),
            float(template.delta_f),
        )

    def _prepare_template(self, template):
        key = self._key(template)
        if key == self._template_key:
            return
        self._template_key = key
        self._template_bins = None
        self._not_applicable = False
        recorded_periastra = getattr(
            template.params, "ecc_tchisq_num_periastra", None
        )
        if (
            recorded_periastra is not None
            and int(recorded_periastra) < 3
        ):
            self._not_applicable = True
            return
        try:
            partition = partition_from_bank_metadata(template.params)
            if partition is None:
                partition = dynamics_partition(
                    template, self.sample_rate
                )
            if (
                recorded_periastra is not None
                and partition["num_periastra"] != int(recorded_periastra)
            ):
                raise RuntimeError(
                    "Stored ecc_tchisq_num_periastra disagrees with "
                    "generated SEOBNRv5E dynamics"
                )
            self._template_bins = split_production_template(
                template, self.sample_rate, partition
            )
        except PeriastronChisqNotApplicable:
            self._not_applicable = True

    def values(self, template, psd, stilde, snrv, norm, indices):
        """Return reduced Delta-K3 and dof at selected matched-filter samples."""
        if not self.do:
            return None, None
        indices = numpy.asarray(indices, dtype=numpy.int64)
        self._prepare_template(template)
        if self._not_applicable:
            return (
                numpy.zeros(len(indices), dtype=numpy.float32),
                numpy.zeros(len(indices), dtype=numpy.uint32),
            )

        bins = self._template_bins
        covariance, full_norm = bin_covariance(
            template, bins, psd, self.low_frequency_cutoff
        )
        full_values = (
            numpy.asarray(snrv, dtype=numpy.complex128) * norm
        )

        # Only two additional IFFTs are needed. Exact closure supplies the
        # final-bin response from the already computed full-template response.
        bin_values = []
        for subtemplate in bins[:-1]:
            subseries, _, subnorm = matched_filter_core(
                subtemplate,
                stilde,
                psd=None,
                low_frequency_cutoff=self.low_frequency_cutoff,
                h_norm=full_norm,
            )
            bin_values.append(
                numpy.asarray(subseries.take(indices)) * subnorm
            )
        final_values = full_values - bin_values[0] - bin_values[1]
        values = numpy.column_stack(
            (bin_values[0], bin_values[1], final_values)
        )
        reduced, dof = incremental_k3_chisq(
            values, covariance
        )
        logging.debug(
            "Calculated incremental periastron chi-square for %d triggers",
            len(indices),
        )
        return (
            numpy.asarray(reduced, dtype=numpy.float32),
            numpy.full(len(indices), dof, dtype=numpy.uint32),
        )

    def mismatch_lambda_values(self, template, dof):
        """Return the bank-cell mismatch coefficient associated with triggers.

        The coefficient must be precomputed and stored in the template bank as
        ``ecc_tchisq_lambda``. Missing values remain NaN so a ranking function
        cannot silently interpret an uncalibrated applicable template as
        having zero mismatch.
        """
        if not self.do or dof is None:
            return None
        value = float(
            getattr(template.params, "ecc_tchisq_lambda", numpy.nan)
        )
        values = numpy.full(len(dof), value, dtype=numpy.float32)
        values[numpy.asarray(dof) == 0] = 0.0
        return values
