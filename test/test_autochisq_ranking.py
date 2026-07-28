import importlib.util
from pathlib import Path
import unittest

import numpy


RANKING_PATH = (
    Path(__file__).resolve().parents[1] / 'pycbc' / 'events' / 'ranking.py'
)
SPEC = importlib.util.spec_from_file_location(
    'stage2_continuous_ranking',
    RANKING_PATH,
)
ranking = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranking)


class TestAutochisqRanking(unittest.TestCase):
    def setUp(self):
        self.snr = numpy.full(3, 8.0)
        self.bchisq = numpy.array([1.5, 2.0, 2.5])
        self.sgchisq = numpy.ones(3)
        self.psdvar = numpy.ones(3)
        self.cont_dof = numpy.full(3, 512.0)
        self.reduced_auto = numpy.array([1.0, 1.2, 2.4])
        self.cont_chisq = self.reduced_auto * self.cont_dof

    def calculate(self, function, cont_chisq=None):
        if cont_chisq is None:
            cont_chisq = self.cont_chisq
        return function(
            self.snr,
            self.bchisq,
            self.sgchisq,
            self.psdvar,
            cont_chisq,
            self.cont_dof,
            autochisq_onset=1.2,
            autochisq_power=0.75,
        )

    def test_threshold_boundary_and_continuous_branch(self):
        thresholded = self.calculate(
            ranking.newsnr_sgveto_psdvar_scaled_threshold_autochisq
        )
        continuous = self.calculate(
            ranking.newsnr_sgveto_psdvar_scaled_autochisq
        )
        base_continuous = ranking.newsnr_sgveto_psdvar_scaled(
            self.snr,
            self.bchisq,
            self.sgchisq,
            self.psdvar,
        )

        self.assertEqual(thresholded[0], continuous[0])
        self.assertEqual(thresholded[1], continuous[1])
        self.assertEqual(continuous[0], base_continuous[0])
        self.assertEqual(continuous[1], base_continuous[1])

        ratio = self.reduced_auto[2] / 1.2
        penalty = (
            0.5 * (1.0 + ratio ** (6.0 * 0.75))
        ) ** (-1.0 / 6.0)
        self.assertAlmostEqual(thresholded[2], penalty)
        self.assertAlmostEqual(
            continuous[2],
            base_continuous[2] * penalty,
        )
        self.assertNotEqual(thresholded[2], continuous[2])

    def test_nonfinite_autochisq_is_neutral(self):
        cont_chisq = numpy.full(3, numpy.nan)
        thresholded = self.calculate(
            ranking.newsnr_sgveto_psdvar_scaled_threshold_autochisq,
            cont_chisq,
        )
        continuous = self.calculate(
            ranking.newsnr_sgveto_psdvar_scaled_autochisq,
            cont_chisq,
        )

        expected_thresholded = (
            ranking.newsnr_sgveto_psdvar_scaled_threshold(
                self.snr,
                self.bchisq,
                self.sgchisq,
                self.psdvar,
            )
        )
        expected_continuous = ranking.newsnr_sgveto_psdvar_scaled(
            self.snr,
            self.bchisq,
            self.sgchisq,
            self.psdvar,
        )
        numpy.testing.assert_array_equal(
            thresholded,
            expected_thresholded,
        )
        numpy.testing.assert_array_equal(
            continuous,
            expected_continuous,
        )

    def test_registered_wrappers_and_required_datasets(self):
        trigs = {
            'snr': self.snr,
            'chisq': self.bchisq * 2.0,
            'chisq_dof': numpy.full(3, 2.0),
            'sg_chisq': self.sgchisq,
            'psd_var_val': self.psdvar,
            'cont_chisq': self.cont_chisq,
            'cont_chisq_dof': self.cont_dof,
        }
        thresholded_name = (
            'newsnr_sgveto_psdvar_scaled_threshold_autochisq'
        )
        continuous_name = 'newsnr_sgveto_psdvar_scaled_autochisq'

        thresholded = ranking.get_sngls_ranking_from_trigs(
            trigs,
            thresholded_name,
            autochisq_onset=1.2,
            autochisq_power=0.75,
        )
        continuous = ranking.get_sngls_ranking_from_trigs(
            trigs,
            continuous_name,
            autochisq_onset=1.2,
            autochisq_power=0.75,
        )

        self.assertEqual(thresholded.dtype, numpy.float32)
        self.assertEqual(continuous.dtype, numpy.float32)
        self.assertEqual(
            ranking.reqd_datasets[thresholded_name],
            ranking.reqd_datasets[continuous_name],
        )
        self.assertIn(
            'cont_chisq',
            ranking.reqd_datasets[continuous_name],
        )
        self.assertIn(
            'cont_chisq_dof',
            ranking.reqd_datasets[continuous_name],
        )


if __name__ == '__main__':
    unittest.main()
