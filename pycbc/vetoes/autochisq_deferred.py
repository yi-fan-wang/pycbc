# Copyright (C) 2026 Yifan Wang
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
"""Deferred evaluation of the auto chi-squared test.

The standard :class:`~pycbc.vetoes.autochisq.SingleDetAutoChisq` evaluates the
auto chi-squared for every trigger as soon as its template/segment pair has
been filtered.  Nearly all of those triggers are subsequently discarded by
``EventManager.consolidate_events``, so the work is wasted.

This module evaluates the same quantity, from the same public
``autochisq_from_precomputed`` formula, but only for the triggers that survive
to the final output.  During filtering it captures the handful of complex SNR
samples the test actually reads -- ``stride``-spaced offsets around each
trigger -- and stores them keyed by ``(template_hash, time_index)``.  After the
final consolidation it rebuilds each template autocorrelation once and fills
``cont_chisq`` in place.

Two properties make this exact rather than approximate:

* ``keep_loudest_stat`` must be ``newsnr``, which does not depend on
  ``cont_chisq``.  Trigger membership is therefore identical whether the auto
  chi-squared is computed early or late.
* The captured samples are the *only* SNR values the test reads, so the
  deferred evaluation is fed bit-identical inputs.

Captured rows are pruned after every consolidation: a row that is no longer in
``event_manager.events`` can never re-enter the final output, so keeping it
resident only inflates peak memory.  A retained event whose row is missing is a
programming error and raises rather than silently producing a wrong value.

Only the forward-template, single-process configuration used by the O4a
eccentric search is supported; anything else raises.
"""

from collections import defaultdict
import logging
import time

import numpy

from pycbc.filter import make_frequency_series, matched_filter_core
from pycbc.types import Array, complex_same_precision_as, zeros
from pycbc.vetoes.autochisq import autochisq_from_precomputed

__all__ = [
    'autochisq_offsets',
    'capture_snr_neighborhoods',
    'evaluate_snr_neighborhoods',
    'stored_trigger_magnitudes',
    'DeferredAutochisq',
]


def stored_trigger_magnitudes(snr_values):
    """Return trigger magnitudes with the exact semantics of ``abs()`` on the
    stored complex64 scalar.

    The capture selection and the retained-event checks in prune/finalize must
    agree bit for bit on which triggers are above the activation threshold.
    The event manager stores ``snrv * norm`` in a complex64 field, and reading
    one event back gives a numpy complex64 scalar whose Python ``abs()``
    computes the magnitude in double precision and rounds the result to
    float32.  numpy's vectorized ``abs`` on a complex64 *array* instead
    computes in single precision throughout, and for a magnitude within one
    float32 ulp of the threshold the two can disagree (observed in production:
    bits ``40ae4dba/3f4304be`` give 5.5 vectorized but 5.5000005 as a scalar).
    Every threshold comparison in this module therefore goes through this one
    helper: cast the stored complex64 values to complex128, take the double
    precision magnitude, round to float32.
    """
    arr = numpy.asarray(snr_values, dtype=numpy.complex64)
    return numpy.abs(arr.astype(numpy.complex128)).astype(numpy.float32)


def autochisq_offsets(instance, series_length):
    """Return the exact SNR offsets read by one autochisq configuration.

    Parameters
    ----------
    instance : SingleDetAutoChisq
        The configured test whose ``stride``, ``num_points`` and ``one_sided``
        settings determine which samples are read.
    series_length : int
        Length of the SNR series the test would have been handed.

    Returns
    -------
    numpy.ndarray
        Signed sample offsets relative to a trigger, including zero.
    """
    num_points = min(instance.num_points, int(series_length / instance.stride))
    left = numpy.arange(-instance.stride * num_points, 0, instance.stride,
                        dtype=numpy.int64)
    right = numpy.arange(instance.stride, instance.stride * num_points + 1,
                         instance.stride, dtype=numpy.int64)
    if instance.one_sided == 'left':
        offsets = left
    elif instance.one_sided == 'right':
        offsets = right
    else:
        offsets = numpy.concatenate((left, right))
    insert_at = len(left) if instance.one_sided is None else 0
    return numpy.insert(offsets, insert_at, 0)


def autochisq_degrees_of_freedom(instance, series_length):
    """Reproduce the test's output degrees of freedom without an FFT."""
    num_points = min(instance.num_points, int(series_length / instance.stride))
    dof = num_points
    if instance.one_sided is None:
        dof *= 2
    if instance.two_phase:
        dof *= 2
    if instance.take_maximum_value:
        dof = instance.dof
    return dof


def capture_snr_neighborhoods(instance, snr, indices, norm, threshold):
    """Capture normalized SNR samples for triggers strictly above threshold.

    Returns the selected original indices, the relative offsets (including
    zero) and one row of complex SNR samples per selected trigger.  The
    circular indexing of the original implementation is reproduced explicitly.
    """
    index_array = numpy.asarray(indices, dtype=numpy.int64)
    snr_array = numpy.asarray(snr)
    # Round the product to the complex64 precision the event manager stores,
    # then compare with the same magnitude semantics prune/finalize apply to
    # the stored scalar.  See stored_trigger_magnitudes for why neither step
    # is optional at the activation boundary.
    stored_snr = (snr_array[index_array] * norm).astype(numpy.complex64)
    selected = stored_trigger_magnitudes(stored_snr) > float(threshold)
    selected_indices = index_array[selected]
    offsets = autochisq_offsets(instance, len(snr))
    if len(selected_indices) == 0:
        dtype = (snr_array[:1] * norm).dtype
        return selected_indices, offsets, numpy.empty((0, len(offsets)),
                                                      dtype=dtype)

    sample_indices = (selected_indices[:, numpy.newaxis]
                      + offsets[numpy.newaxis, :]) % len(snr)
    return selected_indices, offsets, snr_array[sample_indices] * norm


def evaluate_snr_neighborhoods(instance, offsets, samples, autocorrelation):
    """Evaluate captured samples with the unchanged autochisq formula."""
    samples = numpy.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != len(offsets):
        raise ValueError("samples must have shape (triggers, offsets)")
    radius = instance.stride * min(
        instance.num_points, int(len(autocorrelation) / instance.stride))
    series_length = 2 * radius + 1
    center = radius
    positions = center + numpy.asarray(offsets, dtype=numpy.int64)
    if positions.min(initial=0) < 0 or positions.max(initial=0) >= series_length:
        raise ValueError("captured offsets do not fit the deferred series")

    values = numpy.empty(len(samples), dtype=numpy.float64)
    dof = instance.dof
    # One scratch series is reused; every row overwrites the same positions.
    sparse_snr = numpy.zeros(series_length, dtype=samples.dtype)
    for row_index, row in enumerate(samples):
        sparse_snr.fill(0)
        sparse_snr[positions] = row
        dof, result = autochisq_from_precomputed(
            sparse_snr, sparse_snr, autocorrelation,
            numpy.array([center], dtype=numpy.int64),
            stride=instance.stride,
            num_points=instance.num_points,
            oneside=instance.one_sided,
            twophase=instance.two_phase,
            maxvalued=instance.take_maximum_value)
        values[row_index] = result[0]
    return values, dof


def forward_autocorrelation(instance, template, psd, low_frequency_cutoff,
                            high_frequency_cutoff=None):
    """Build one forward-template autocorrelation, reusing FFT buffers.

    The buffers are attached to ``instance`` and reused across templates whose
    frequency-domain layout matches, which avoids reallocating two full
    time-series per retained template group.  The autocorrelation itself is
    always recomputed: a reloaded bank can recycle Python object ids, so
    caching on ``id(template)`` would be unsafe here.
    """
    if instance.reverse_template:
        raise ValueError("deferred autochisq requires a forward template")
    htilde = make_frequency_series(template)
    time_length = (len(htilde) - 1) * 2
    layout = (time_length, htilde.precision, float(htilde.delta_f),
              low_frequency_cutoff, high_frequency_cutoff)
    if getattr(instance, '_deferred_buffer_layout', None) != layout:
        dtype = complex_same_precision_as(htilde)
        instance._deferred_time = zeros(time_length, dtype=dtype)
        instance._deferred_correlation = zeros(time_length, dtype=dtype)
        instance._deferred_buffer_layout = layout

    autocorrelation, _, _ = matched_filter_core(
        htilde, htilde, psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        h_norm=template.sigmasq(psd),
        out=instance._deferred_time,
        corr_out=instance._deferred_correlation)
    autocorrelation *= 1.0 / autocorrelation[0]
    return Array(autocorrelation, copy=False)


class DeferredAutochisq(object):
    """Capture sparse SNR samples and fill ``cont_chisq`` after consolidation.

    Parameters
    ----------
    threshold : float
        Only triggers whose normalized raw SNR is strictly above this value are
        captured and later evaluated; everything else keeps ``cont_chisq = 0``,
        exactly as the eager raw-SNR activation does.
    autocorrelation_builder : callable, optional
        Overrides how the per-template autocorrelation is rebuilt.  Used by the
        unit tests; production uses :func:`forward_autocorrelation`.
    """

    def __init__(self, threshold, autocorrelation_builder=None):
        self.threshold = float(threshold)
        self.autocorrelation_builder = (autocorrelation_builder
                                        or forward_autocorrelation)
        self.bank = None
        self.bank_index_by_hash = {}
        self.group_metadata = {}
        self.records = {}
        self.offsets = None
        self.instance = None
        self.capture_calls = 0
        self.input_trigger_total = 0
        self.captured_trigger_total = 0
        self.captured_sample_count = 0
        self.captured_sample_bytes_total = 0
        self.final_trigger_total = 0
        self.final_above_threshold = 0
        self.final_evaluated = 0
        self.autocorrelation_recomputations = 0
        self.lookup_misses = 0
        self.duplicate_keys = 0
        self.capture_wall_s = 0.0
        self.finalize_wall_s = 0.0
        self.consolidation_prune_calls = 0
        self.pruned_record_total = 0
        self.pruned_group_total = 0
        self.peak_record_count = 0
        self.finalized = False

    @staticmethod
    def template_hash(template):
        """Return the integer template hash written to trigger HDF files."""
        return int(template.params.template_hash)

    def register_bank_template(self, bank, index, template):
        """Bind a stable template hash to its row in the active FilterBank."""
        if isinstance(index, slice):
            return
        value = self.template_hash(template)
        previous = self.bank_index_by_hash.setdefault(value, int(index))
        if previous != int(index):
            raise RuntimeError("template hash %s maps to rows %s and %s"
                               % (value, previous, index))
        if self.bank is not None and self.bank is not bank:
            raise RuntimeError("more than one FilterBank entered deferred run")
        self.bank = bank

    def capture(self, instance, snr, indices, template, psd, norm,
                stilde=None, low_frequency_cutoff=None,
                high_frequency_cutoff=None):
        """Stand in for ``SingleDetAutoChisq.values`` during filtering.

        Returns neutral zeros so the trigger record keeps its shape; the real
        values are written by :meth:`finalize`.
        """
        started = time.perf_counter()
        if instance.reverse_template:
            raise ValueError("deferred autochisq requires a forward template")
        if not (instance.do and len(indices) > 0):
            return None, None
        self.instance = instance
        self.capture_calls += 1
        self.input_trigger_total += len(indices)
        selected_indices, offsets, samples = capture_snr_neighborhoods(
            instance, snr, indices, norm, self.threshold)
        self.captured_trigger_total += len(selected_indices)
        self.captured_sample_count += samples.size
        self.captured_sample_bytes_total += samples.nbytes

        dof = autochisq_degrees_of_freedom(instance, len(snr))
        instance.dof = dof
        if len(selected_indices) == 0:
            self.capture_wall_s += time.perf_counter() - started
            return numpy.zeros(len(indices), dtype=numpy.float64), dof

        if self.offsets is None:
            self.offsets = offsets.copy()
        elif not numpy.array_equal(self.offsets, offsets):
            raise RuntimeError("deferred autochisq offsets changed within a run")

        value = self.template_hash(template)
        bank_index = self.bank_index_by_hash.get(value)
        if bank_index is None:
            raise RuntimeError("template hash %s has no FilterBank row" % value)

        # ``pycbc_inspiral`` calls autochisq with indices already shifted by
        # ``stilde.analyze.start``, but writes the final ``time_index`` by
        # shifting the original matched-filter indices by
        # ``stilde.cumulative_index``.  Remove the analysis-slice shift so
        # capture and finalization share one global sample coordinate.
        cumulative_index = int(stilde.cumulative_index)
        analysis_start = int(stilde.analyze.start)
        group = (value, id(psd))
        metadata = {'template_hash': value,
                    'bank_index': bank_index,
                    'psd': psd,
                    'low_frequency_cutoff': low_frequency_cutoff,
                    'high_frequency_cutoff': high_frequency_cutoff}
        previous_metadata = self.group_metadata.setdefault(group, metadata)
        if previous_metadata['bank_index'] != bank_index:
            raise RuntimeError("inconsistent deferred group metadata %s"
                               % (group,))

        for local_index, row in zip(selected_indices, samples):
            key = (value, int(local_index) + cumulative_index - analysis_start)
            if key in self.records:
                self.duplicate_keys += 1
                old_group, old_row = self.records[key]
                if old_group != group or not numpy.array_equal(old_row, row):
                    raise RuntimeError("conflicting deferred trigger key %s"
                                       % (key,))
                continue
            self.records[key] = (group, row.copy())

        self.peak_record_count = max(self.peak_record_count, len(self.records))
        self.capture_wall_s += time.perf_counter() - started
        return numpy.zeros(len(indices), dtype=numpy.float64), dof

    @staticmethod
    def _event_key(event_manager, event):
        parameter = event_manager.template_params[event['template_id']]['tmplt']
        return (int(parameter.template_hash), int(event['time_index']))

    def prune_after_consolidation(self, event_manager):
        """Drop captured rows that a completed consolidation already rejected.

        ``pycbc_inspiral`` consolidates after every template chunk when
        ``--finalize-events-template-rate`` is set.  Once the unchanged newsnr
        keep-loudest rule has run, a captured row absent from
        ``event_manager.events`` can never re-enter the final output.
        """
        if self.finalized or not self.records:
            return
        retained_keys = set()
        missing = []
        magnitudes = stored_trigger_magnitudes(event_manager.events['snr'])
        for event, magnitude in zip(event_manager.events, magnitudes):
            if magnitude <= self.threshold:
                continue
            key = self._event_key(event_manager, event)
            retained_keys.add(key)
            if key not in self.records:
                missing.append(key)
        if missing:
            raise RuntimeError(
                "consolidation retained deferred events without captured "
                "records: %s (total %s)" % (missing[:3], len(missing)))

        old_record_count = len(self.records)
        old_group_count = len(self.group_metadata)
        self.records = {key: self.records[key] for key in retained_keys}
        retained_groups = {group for group, _ in self.records.values()}
        self.group_metadata = {
            group: metadata
            for group, metadata in self.group_metadata.items()
            if group in retained_groups}
        self.consolidation_prune_calls += 1
        self.pruned_record_total += old_record_count - len(self.records)
        self.pruned_group_total += old_group_count - len(self.group_metadata)

    def finalize(self, event_manager):
        """Evaluate the retained above-threshold events and fill cont_chisq."""
        if self.finalized:
            return
        self.finalized = True
        started = time.perf_counter()
        events = event_manager.events
        self.final_trigger_total = len(events)
        if len(events) == 0:
            self.finalize_wall_s = time.perf_counter() - started
            return
        if self.instance is None or self.bank is None:
            raise RuntimeError("final events exist but nothing was captured")
        if event_manager.opt.keep_loudest_stat != 'newsnr':
            raise RuntimeError(
                "deferred autochisq requires keep_loudest_stat=newsnr so "
                "trigger membership is independent of cont_chisq")

        grouped = defaultdict(list)
        magnitudes = stored_trigger_magnitudes(events['snr'])
        for event_index, event in enumerate(events):
            if magnitudes[event_index] <= self.threshold:
                continue
            self.final_above_threshold += 1
            key = self._event_key(event_manager, event)
            record = self.records.get(key)
            if record is None:
                self.lookup_misses += 1
                continue
            group, samples = record
            grouped[group].append((event_index, samples))
        if self.lookup_misses:
            raise RuntimeError("missing %s retained deferred SNR records"
                               % self.lookup_misses)

        logging.info("Evaluating deferred autochisq for %s triggers in %s "
                     "template groups", self.final_above_threshold,
                     len(grouped))
        for group, retained in grouped.items():
            metadata = self.group_metadata[group]
            template = self.bank[metadata['bank_index']]
            autocorrelation = self.autocorrelation_builder(
                self.instance, template, metadata['psd'],
                metadata['low_frequency_cutoff'],
                metadata['high_frequency_cutoff'])
            self.autocorrelation_recomputations += 1
            output_indices = [item[0] for item in retained]
            samples = numpy.stack([item[1] for item in retained])
            values, _ = evaluate_snr_neighborhoods(
                self.instance, self.offsets, samples, autocorrelation)
            events['cont_chisq'][output_indices] = values
            self.final_evaluated += len(values)
        self.finalize_wall_s = time.perf_counter() - started

    def audit(self):
        """Return JSON-serializable counters describing the completed run."""
        return {
            'raw_snr_activation_threshold': self.threshold,
            'capture_calls': self.capture_calls,
            'input_trigger_total': self.input_trigger_total,
            'captured_trigger_total': self.captured_trigger_total,
            'captured_sample_count': self.captured_sample_count,
            'captured_sample_bytes_total': self.captured_sample_bytes_total,
            'captured_sample_bytes': sum(row.nbytes
                                         for _, row in self.records.values()),
            'deferred_record_count': len(self.records),
            'deferred_group_count': len(self.group_metadata),
            'consolidation_prune_calls': self.consolidation_prune_calls,
            'pruned_record_total': self.pruned_record_total,
            'pruned_group_total': self.pruned_group_total,
            'peak_record_count': self.peak_record_count,
            'shared_offset_count': (len(self.offsets)
                                    if self.offsets is not None else 0),
            'final_trigger_total': self.final_trigger_total,
            'final_above_threshold': self.final_above_threshold,
            'final_evaluated': self.final_evaluated,
            'autocorrelation_recomputations':
                self.autocorrelation_recomputations,
            'lookup_misses': self.lookup_misses,
            'duplicate_keys': self.duplicate_keys,
            'capture_wall_s': self.capture_wall_s,
            'finalize_wall_s': self.finalize_wall_s,
            'finalized': self.finalized,
        }
