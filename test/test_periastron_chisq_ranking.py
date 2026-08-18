"""Tests for competition-normalized periastron-chi-square ranking."""

import unittest

import numpy

from pycbc.events import ranking


class TestPeriastronChisqRanking(unittest.TestCase):
    def setUp(self):
        self.snr = numpy.asarray([8.0, 8.0, 8.0, 8.0])
        self.raw = numpy.asarray([50.0, 50.0, 8.0, 8.0])
        self.dof = numpy.asarray([0, 2, 2, 2])
        self.mismatch = numpy.asarray([numpy.nan, 0.1, 0.0, 0.1])

    def test_not_applicable_is_exactly_neutral(self):
        base = numpy.full(4, 8.0)
        result = ranking.apply_periastron_chisq_reweighting(
            base,
            self.snr,
            self.raw,
            self.dof,
            self.mismatch,
            periastron_chisq_onset=1.0,
            periastron_chisq_power=3.0,
        )
        self.assertEqual(result[0], 8.0)

    def test_competition_normalization_weakens_mismatch_penalty(self):
        base = numpy.full(4, 8.0)
        result = ranking.apply_periastron_chisq_reweighting(
            base,
            self.snr,
            self.raw,
            self.dof,
            self.mismatch,
            periastron_chisq_onset=1.0,
            periastron_chisq_power=3.0,
        )
        self.assertGreater(result[1], 0.0)
        self.assertGreater(result[3], result[2])
        self.assertLess(result[3], 8.0)
        ratio = self.raw[2]
        expected = 8.0 * (
            0.5 * (1.0 + ratio ** 3.0)
        ) ** (-1.0 / 6.0)
        self.assertAlmostEqual(result[2], expected)

    def test_missing_applicable_lambda_fails_closed(self):
        with self.assertRaises(ValueError):
            ranking.apply_periastron_chisq_reweighting(
                numpy.asarray([8.0]),
                numpy.asarray([8.0]),
                numpy.asarray([2.0]),
                numpy.asarray([2]),
                numpy.asarray([numpy.nan]),
            )

    def test_registered_combined_wrapper(self):
        trigs = {
            "snr": self.snr,
            "chisq": numpy.full(4, 2.0),
            "chisq_dof": numpy.full(4, 2.0),
            "sg_chisq": numpy.ones(4),
            "psd_var_val": numpy.ones(4),
            "cont_chisq": numpy.full(4, 512.0),
            "cont_chisq_dof": numpy.full(4, 512.0),
            "ecc_tchisq": self.raw,
            "ecc_tchisq_dof": self.dof,
            "ecc_tchisq_lambda": self.mismatch,
        }
        name = (
            "newsnr_sgveto_psdvar_scaled_threshold_"
            "autochisq_periastron_chisq"
        )
        result = ranking.get_sngls_ranking_from_trigs(
            trigs,
            name,
            periastron_chisq_onset=1.0,
            periastron_chisq_power=3.0,
        )
        self.assertEqual(result.dtype, numpy.float32)
        self.assertEqual(result[0], 8.0)
        self.assertIn(
            "ecc_tchisq_lambda", ranking.reqd_datasets[name]
        )


if __name__ == "__main__":
    unittest.main()
