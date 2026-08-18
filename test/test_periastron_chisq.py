"""Tests for the covariance-aware incremental periastron chi-square."""

import unittest
from types import SimpleNamespace

import numpy

from pycbc.vetoes.periastron_chisq import (
    PeriastronChisqNotApplicable,
    covariance_chisq,
    incremental_k3_chisq,
    partition_from_bank_metadata,
    residual_model,
)


class TestPeriastronChisq(unittest.TestCase):
    def setUp(self):
        # A Hermitian positive-definite covariance whose bins are deliberately
        # non-orthogonal and have unequal expected fractions.
        factor = numpy.asarray(
            [
                [0.48 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j],
                [0.12 + 0.08j, 0.35 + 0.00j, 0.00 + 0.00j],
                [0.06 - 0.03j, 0.10 + 0.05j, 0.28 + 0.00j],
            ]
        )
        covariance = factor @ factor.conj().T
        self.covariance = covariance / numpy.real(numpy.sum(covariance))

    def test_residual_ranks(self):
        _, _, rank3 = residual_model(self.covariance)
        aggregation = numpy.asarray(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        covariance2 = (
            aggregation
            @ self.covariance
            @ aggregation.conj().T
        )
        _, _, rank2 = residual_model(covariance2)
        self.assertEqual(rank3, 2)
        self.assertEqual(rank2, 1)

    def test_exact_signal_has_zero_increment(self):
        expected = self.covariance @ numpy.ones(3)
        values = numpy.outer(
            numpy.asarray([5.0 + 2.0j, 10.0 - 1.0j]),
            expected,
        )
        reduced, dof = incremental_k3_chisq(
            values, self.covariance
        )
        self.assertEqual(dof, 2)
        numpy.testing.assert_allclose(reduced, 0.0, atol=1e-12)

    def test_increment_is_nested_difference(self):
        values = numpy.asarray(
            [
                [3.0 + 1.0j, 0.2 - 0.8j, 4.0 + 0.3j],
                [2.0 - 0.1j, 3.0 + 1.5j, 0.5 - 0.2j],
            ]
        )
        chi3, dof3 = covariance_chisq(
            values, self.covariance
        )
        aggregation = numpy.asarray(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        chi2, dof2 = covariance_chisq(
            values @ aggregation.T,
            aggregation
            @ self.covariance
            @ aggregation.conj().T,
        )
        reduced, dof = incremental_k3_chisq(
            values, self.covariance
        )
        self.assertEqual(dof3 - dof2, dof)
        numpy.testing.assert_allclose(
            reduced * dof, chi3 - chi2, rtol=1e-12, atol=1e-12
        )

    def test_partition_from_complete_bank_metadata(self):
        params = SimpleNamespace(
            ecc_tchisq_num_periastra=6,
            ecc_tchisq_boundary_index_0=731,
            ecc_tchisq_boundary_index_1=863,
            ecc_tchisq_waveform_samples=1251,
            ecc_tchisq_direct_peak_index=997,
        )
        partition = partition_from_bank_metadata(params)
        self.assertEqual(partition["num_periastra"], 6)
        numpy.testing.assert_array_equal(
            partition["boundary_indices"], [731, 863]
        )
        self.assertEqual(partition["waveform_samples"], 1251)
        self.assertEqual(partition["direct_peak_index"], 997)

    def test_partition_without_bank_metadata_falls_back(self):
        self.assertIsNone(
            partition_from_bank_metadata(
                SimpleNamespace(ecc_tchisq_num_periastra=6)
            )
        )

    def test_stored_not_applicable_partition(self):
        params = SimpleNamespace(
            ecc_tchisq_num_periastra=2,
            ecc_tchisq_boundary_index_0=-1,
            ecc_tchisq_boundary_index_1=-1,
            ecc_tchisq_waveform_samples=920,
            ecc_tchisq_direct_peak_index=385,
        )
        with self.assertRaises(PeriastronChisqNotApplicable):
            partition_from_bank_metadata(params)


if __name__ == "__main__":
    unittest.main()
