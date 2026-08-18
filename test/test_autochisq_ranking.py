import unittest

import numpy

from pycbc.events import ranking


class TestAutochisqRanking(unittest.TestCase):
    def setUp(self):
        self.snr = numpy.array([8.0, 8.0, 8.0])
        self.bchisq = numpy.ones(3)
        self.sgchisq = numpy.ones(3)
        self.psdvar = numpy.ones(3)
        self.cont_dof = numpy.full(3, 512.0)
        self.reduced_auto = numpy.array([1.0, 1.2, 2.4])
        self.cont_chisq = self.reduced_auto * self.cont_dof

    def test_penalty_is_continuous_and_neutral_below_onset(self):
        result = (
            ranking.newsnr_sgveto_psdvar_scaled_threshold_autochisq(
                self.snr,
                self.bchisq,
                self.sgchisq,
                self.psdvar,
                self.cont_chisq,
                self.cont_dof,
                autochisq_onset=1.2,
                autochisq_power=0.75,
            )
        )
        self.assertEqual(result[0], 8.0)
        self.assertEqual(result[1], 8.0)
        self.assertLess(result[2], 8.0)

        ratio = 2.0
        expected = 8.0 * (
            0.5 * (1.0 + ratio ** (6.0 * 0.75))
        ) ** (-1.0 / 6.0)
        self.assertAlmostEqual(result[2], expected)

    def test_nonfinite_autochisq_is_neutral(self):
        result = (
            ranking.newsnr_sgveto_psdvar_scaled_threshold_autochisq(
                8.0,
                1.0,
                1.0,
                1.0,
                numpy.nan,
                512.0,
            )
        )
        self.assertEqual(result, 8.0)

    def test_power_penalty_is_neutral_below_onset(self):
        result = (
            ranking.newsnr_sgveto_psdvar_scaled_threshold_autochisq_power(
                self.snr,
                self.bchisq,
                self.sgchisq,
                self.psdvar,
                self.cont_chisq,
                self.cont_dof,
                autochisq_onset=1.2,
                autochisq_power=0.75,
            )
        )
        self.assertEqual(result[0], 8.0)
        self.assertEqual(result[1], 8.0)
        self.assertAlmostEqual(result[2], 8.0 * 2.0 ** (-0.75))

    def test_registered_wrapper_and_required_datasets(self):
        trigs = {
            'snr': self.snr,
            'chisq': numpy.full(3, 2.0),
            'chisq_dof': numpy.full(3, 2.0),
            'sg_chisq': self.sgchisq,
            'psd_var_val': self.psdvar,
            'cont_chisq': self.cont_chisq,
            'cont_chisq_dof': self.cont_dof,
        }
        name = 'newsnr_sgveto_psdvar_scaled_threshold_autochisq'
        result = ranking.get_sngls_ranking_from_trigs(
            trigs,
            name,
            autochisq_onset=1.2,
            autochisq_power=0.75,
        )
        self.assertEqual(result.dtype, numpy.float32)
        self.assertIn('cont_chisq', ranking.reqd_datasets[name])
        self.assertIn('cont_chisq_dof', ranking.reqd_datasets[name])

        power_name = (
            'newsnr_sgveto_psdvar_scaled_threshold_autochisq_power'
        )
        power_result = ranking.get_sngls_ranking_from_trigs(
            trigs,
            power_name,
            autochisq_onset=1.2,
            autochisq_power=0.75,
        )
        self.assertEqual(power_result.dtype, numpy.float32)
        self.assertEqual(
            ranking.reqd_datasets[power_name],
            ranking.reqd_datasets[name],
        )


if __name__ == '__main__':
    unittest.main()
