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
"""Equivalence tests for deferred auto chi-squared evaluation.

Each test compares the deferred path against the unchanged
``autochisq_from_precomputed`` result the eager path would have produced.
"""

import unittest
from types import SimpleNamespace

import numpy

from pycbc.vetoes.autochisq import (SingleDetAutoChisq,
                                    autochisq_from_precomputed)
from pycbc.vetoes.autochisq_deferred import (DeferredAutochisq,
                                             capture_snr_neighborhoods,
                                             evaluate_snr_neighborhoods)


class DeferredAutochisqTest(unittest.TestCase):
    def compare_case(self, onesided, twophase, indices):
        """Deferred evaluation must reproduce the eager values bit for bit."""
        rng = numpy.random.default_rng(20260804)
        snr = (rng.normal(size=4096)
               + 1j * rng.normal(size=4096)).astype(numpy.complex64)
        norm = numpy.float32(0.73125)
        autocorrelation = (0.1 * rng.normal(size=4096)
                           + 0.1j * rng.normal(size=4096)).astype(numpy.complex64)
        autocorrelation[0] = 1.0 + 0.0j
        instance = SingleDetAutoChisq(4, 128, onesided=onesided,
                                      twophase=twophase)
        index_array = numpy.asarray(indices, dtype=numpy.int64)
        normalized_snr = snr * norm
        expected_dof, expected = autochisq_from_precomputed(
            normalized_snr, normalized_snr, autocorrelation, index_array,
            stride=instance.stride, num_points=instance.num_points,
            oneside=instance.one_sided, twophase=instance.two_phase,
            maxvalued=instance.take_maximum_value)
        selected, offsets, samples = capture_snr_neighborhoods(
            instance, snr, index_array, norm, threshold=-1.0)
        actual, actual_dof = evaluate_snr_neighborhoods(
            instance, offsets, samples, autocorrelation)
        numpy.testing.assert_array_equal(selected, index_array)
        numpy.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_dof, expected_dof)

    def test_two_sided_two_phase_including_wrap_boundaries(self):
        self.compare_case(None, True, [0, 3, 512, 2048, 4092, 4095])

    def test_one_sided_and_one_phase_variants(self):
        self.compare_case('left', False, [0, 700, 4095])
        self.compare_case('right', True, [0, 700, 4095])

    def test_threshold_is_strict_and_empty_capture_is_typed(self):
        instance = SingleDetAutoChisq(4, 2, twophase=True)
        snr = numpy.zeros(64, dtype=numpy.complex64)
        snr[10] = 5.5 + 0j
        snr[20] = numpy.nextafter(numpy.float32(5.5), numpy.float32(6.0))
        selected, offsets, samples = capture_snr_neighborhoods(
            instance, snr, [10, 20], numpy.float32(1.0), threshold=5.5)
        numpy.testing.assert_array_equal(selected, [20])
        self.assertEqual(samples.shape, (1, len(offsets)))

        selected, offsets, samples = capture_snr_neighborhoods(
            instance, snr, [10], numpy.float32(1.0), threshold=5.5)
        self.assertEqual(len(selected), 0)
        self.assertEqual(samples.dtype, numpy.complex64)

    def test_fills_only_retained_above_threshold_events(self):
        """The two coordinate systems used by pycbc_inspiral must line up."""
        rng = numpy.random.default_rng(20260804)
        snr = (rng.normal(size=4096)
               + 1j * rng.normal(size=4096)).astype(numpy.complex64)
        trigger_indices = numpy.array([100, 800, 1500], dtype=numpy.int64)
        analysis_start = 256
        cumulative_index = 10000
        # Auto-chisq is called with analysis-slice coordinates, but the final
        # time_index uses cumulative coordinates.
        autochisq_indices = trigger_indices + analysis_start
        snr[autochisq_indices[0]] = 7.0 + 1.0j
        snr[autochisq_indices[1]] = 4.0 + 0.0j
        snr[autochisq_indices[2]] = 6.0 - 2.0j
        norm = numpy.float32(1.0)
        autocorrelation = (0.1 * rng.normal(size=4096)
                           + 0.1j * rng.normal(size=4096)).astype(numpy.complex64)
        autocorrelation[0] = 1.0 + 0.0j
        instance = SingleDetAutoChisq(4, 128, twophase=True)
        parameter = SimpleNamespace(template_hash=12345)
        template = SimpleNamespace(params=parameter)

        class FakeBank(object):
            def __getitem__(self, index):
                self.last_index = index
                return template

        bank = FakeBank()
        autocorrelation_calls = []

        def build_autocorrelation(*_args):
            autocorrelation_calls.append(True)
            return autocorrelation

        state = DeferredAutochisq(
            5.5, autocorrelation_builder=build_autocorrelation)
        state.register_bank_template(bank, 0, template)
        neutral, _ = state.capture(
            instance, snr, autochisq_indices, template, object(), norm,
            stilde=SimpleNamespace(cumulative_index=cumulative_index,
                                   analyze=slice(analysis_start, None)),
            low_frequency_cutoff=20.0)
        numpy.testing.assert_array_equal(neutral, [0.0, 0.0, 0.0])

        event_dtype = [('template_id', numpy.int64),
                       ('time_index', numpy.int64),
                       ('snr', numpy.complex64),
                       ('cont_chisq', numpy.float32)]
        events = numpy.zeros(3, dtype=event_dtype)
        events['template_id'] = 0
        events['time_index'] = trigger_indices + cumulative_index
        events['snr'] = snr[autochisq_indices] * norm
        event_manager = SimpleNamespace(
            events=events,
            opt=SimpleNamespace(keep_loudest_stat='newsnr'),
            template_params=[{'tmplt': parameter}])
        state.finalize(event_manager)

        expected_dof, expected = autochisq_from_precomputed(
            snr * norm, snr * norm, autocorrelation,
            autochisq_indices[[0, 2]],
            stride=instance.stride, num_points=instance.num_points,
            oneside=instance.one_sided, twophase=instance.two_phase,
            maxvalued=instance.take_maximum_value)
        self.assertEqual(expected_dof, 512)
        numpy.testing.assert_array_equal(events['cont_chisq'][[0, 2]],
                                         expected.astype(numpy.float32))
        # The sub-threshold trigger keeps the neutral value.
        self.assertEqual(events['cont_chisq'][1], 0.0)
        # One autocorrelation per retained template group, not per trigger.
        self.assertEqual(len(autocorrelation_calls), 1)
        self.assertEqual(state.captured_trigger_total, 2)
        self.assertEqual(state.final_above_threshold, 2)
        self.assertEqual(state.final_evaluated, 2)
        self.assertEqual(state.autocorrelation_recomputations, 1)
        self.assertEqual(state.lookup_misses, 0)

    def test_consolidation_prunes_only_discarded_records(self):
        parameter = SimpleNamespace(template_hash=12345)
        group = (12345, 9876)
        state = DeferredAutochisq(5.5)
        state.records = {
            (12345, 100): (group, numpy.ones(5, dtype=numpy.complex64)),
            (12345, 200): (group, numpy.ones(5, dtype=numpy.complex64))}
        state.group_metadata = {group: {'template_hash': 12345}}
        state.peak_record_count = 2

        event_dtype = [('template_id', numpy.int64),
                       ('time_index', numpy.int64),
                       ('snr', numpy.complex64)]
        events = numpy.zeros(1, dtype=event_dtype)
        events['template_id'] = 0
        events['time_index'] = 100
        events['snr'] = 6.0 + 0.0j
        event_manager = SimpleNamespace(
            events=events, template_params=[{'tmplt': parameter}])

        state.prune_after_consolidation(event_manager)
        self.assertEqual(set(state.records), {(12345, 100)})
        self.assertEqual(set(state.group_metadata), {group})
        self.assertEqual(state.consolidation_prune_calls, 1)
        self.assertEqual(state.pruned_record_total, 1)
        self.assertEqual(state.pruned_group_total, 0)
        self.assertEqual(state.peak_record_count, 2)

        # A retained event without a captured row is a bug, not a silent zero.
        events['time_index'] = 999
        with self.assertRaisesRegex(RuntimeError, 'without captured records'):
            state.prune_after_consolidation(event_manager)


if __name__ == '__main__':
    unittest.main()


class StoragePrecisionSelectionTest(unittest.TestCase):
    """Capture selection must use the stored-scalar magnitude semantics.

    pycbc_inspiral stores snrv * norm in a complex64 events field.  Reading
    one event back gives a numpy complex64 scalar whose Python ``abs()``
    computes the magnitude in double precision and rounds to float32, while
    numpy's vectorized abs on a complex64 array computes in single precision
    throughout.  For a magnitude within one float32 ulp of the activation
    threshold the two disagree; capture must match the scalar semantics the
    prune/finalize checks apply, or the fail-closed check trips -- the
    2026-08-05 production failure.
    """

    # Exact bits of the production trigger that failed deterministically:
    # node inspiral-FULL_DATA-L1_ID3_ID0016686, key (-1490843478099418878,
    # 585953).  Vectorized abs gives 5.5 (40b00000); scalar abs gives
    # 5.5000005 (40b00001).
    PROD_RE = numpy.frombuffer(bytes.fromhex("40ae4dba"), ">f4")[0]
    PROD_IM = numpy.frombuffer(bytes.fromhex("3f4304be"), ">f4")[0]

    def production_value(self):
        return numpy.complex64(complex(float(self.PROD_RE),
                                       float(self.PROD_IM)))

    def test_production_bits_expose_the_abs_discrepancy(self):
        value = self.production_value()
        from pycbc.vetoes.autochisq_deferred import stored_trigger_magnitudes
        vector = numpy.abs(numpy.array([value], dtype=numpy.complex64))[0]
        scalar = abs(value)
        self.assertFalse(vector > 5.5)          # single precision: 5.5
        self.assertTrue(scalar > 5.5)           # double then round: 5.5000005
        helper = stored_trigger_magnitudes([value])[0]
        self.assertEqual(helper, numpy.float32(scalar))

    def test_production_trigger_is_captured(self):
        value = self.production_value()
        instance = SingleDetAutoChisq(4, 2, twophase=True)
        snr = numpy.zeros(4096, dtype=numpy.complex64)
        snr[100] = value
        selected, offsets, samples = capture_snr_neighborhoods(
            instance, snr, [100], 1.0, threshold=5.5)
        # Pre-fix the vectorized abs said 5.5 and skipped it, so the record
        # the prune/finalize checks demand was never stored.
        numpy.testing.assert_array_equal(selected, [100])

    def test_selection_mask_matches_prune_semantics(self):
        """Selection must equal an independent per-event scalar-abs loop over
        the stored complex64 values, i.e. exactly what prune used to do."""
        norm = numpy.float64(0.73125)
        rng = numpy.random.default_rng(20260806)
        mags = 5.5 + rng.uniform(-2e-6, 2e-6, size=5000)
        phases = rng.uniform(0.0, 2.0 * numpy.pi, size=5000)
        snr = ((mags / norm) * numpy.exp(1j * phases)).astype(numpy.complex64)
        instance = SingleDetAutoChisq(4, 2, twophase=True)
        selected, _, _ = capture_snr_neighborhoods(
            instance, snr, numpy.arange(5000), norm, threshold=5.5)
        stored = (snr * norm).astype(numpy.complex64)
        expected = numpy.array(
            [i for i, v in enumerate(stored) if abs(v) > 5.5],
            dtype=numpy.int64)
        numpy.testing.assert_array_equal(selected, expected)
