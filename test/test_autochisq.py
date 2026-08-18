from pycbc.fft.fftw import set_measure_level
set_measure_level(0)
from pycbc.filter import  matched_filter_core
from pycbc.types import Array, TimeSeries, FrequencySeries
from pycbc.waveform import get_td_waveform
from pycbc.vetoes import (
    make_frequency_series,
    autochisq_from_precomputed,
    select_autocorrelation_peak_lags,
)
import numpy as np
from math import cos, sin, pi, exp
import unittest
from utils import parse_args_all_schemes, simple_exit

_scheme, _context = parse_args_all_schemes("Auto Chi-squared Veto")


class TestAutochisquare(unittest.TestCase):
    def setUp(self):

        self.Msun = 4.92549095e-6
        self.sample_rate = 4096
        self.segment_length = 256
        self.low_frequency_cutoff = 30.0

        # chirp params
        self.m1 = 2.0
        self.m2 = 2.5
        self.del_t = 1.0/self.sample_rate
        self.Dl = 40.0
        self.iota = 1.0
        self.phi_c = 2.0
        self.tc_indx = 86*self.sample_rate ## offset from the beginnig of a segment
        self.fmax = 1.0/(6.**1.5 *pi *(self.m1+self.m2)*self.Msun)


        self.zeta = 1.0
        self.thetaS = 0.5
        self.phiS = 2.781

        self.Fp = 0.5*cos(2.0*self.zeta)*(1.0 + cos(self.thetaS)*cos(self.thetaS))*cos(2.0*self.phiS) - \
                sin(2.*self.zeta)*cos(self.thetaS)*sin(2.*self.phiS)

        self.Fc = 0.5*sin(2.0*self.zeta)*(1.0 + cos(self.thetaS)*cos(self.thetaS))*cos(2.0*self.phiS) + \
                 cos(2.*self.zeta)*sin(self.thetaS)*sin(2.*self.phiS)


        # params of sin-gaussian
        self.Q = 1.e-1
        self.om = 200.0*pi*2.0

        # use flat psd
        self.seg_len_idx = self.segment_length * self.sample_rate
        self.psd_len = int(self.seg_len_idx/2+1)

        self.Psd = np.ones(self.psd_len)*2.0e-46

        #  generate waveform and chirp signal

        hp, hc = get_td_waveform(approximant="SpinTaylorT5", mass1=self.m1, mass2=self.m2, \
                        delta_t=self.del_t, f_lower=self.low_frequency_cutoff, distance=self.Dl, \
                        inclination=self.iota, coa_phase=self.phi_c)

        # signal which is a noiseless data
        thp = np.zeros(self.seg_len_idx)
        thp[self.tc_indx:len(hp)+self.tc_indx] = hp
        thc = np.zeros(self.seg_len_idx)
        thc[self.tc_indx:len(hc)+self.tc_indx] = hc

        fct = 10.0/15.21377
        self.sig1 = fct*(self.Fp*thp + self.Fc*thc)

        #### template
        h = np.zeros(self.seg_len_idx)
        h[0:len(hp)] = hp
        hpt = TimeSeries(h, self.del_t)
        self.htilde = make_frequency_series(hpt)

        # generate sin-gaussian signal
        time = np.arange(0, len(hp))*self.del_t
        Nby2 = int(len(hp)/2)
        sngt = np.zeros(len(hp))
        for i in range(len(hp)):
            sngt[i] = 9.0e-21*exp(-(time[i]-time[Nby2])**2/self.Q)*sin(self.om*time[i])

        self.sig2 = np.zeros(self.seg_len_idx)
        self.sig2[self.tc_indx:len(sngt)+self.tc_indx] = sngt


    def test_chirp(self):
        ### use a chirp as a signal


        sigt = TimeSeries(self.sig1, self.del_t)
        sig_tilde = make_frequency_series(sigt)

        del_f = sig_tilde.get_delta_f()
        psd = FrequencySeries(self.Psd, del_f)
        flow = self.low_frequency_cutoff

        with _context:
            hautocor, hacorfr, hnrm = matched_filter_core(self.htilde, self.htilde, psd=psd, \
                        low_frequency_cutoff=flow, high_frequency_cutoff=self.fmax)
            hautocor = hautocor * float(np.real(1./hautocor[0]))

            snr, cor, nrm = matched_filter_core(self.htilde, sig_tilde, psd=psd, \
                        low_frequency_cutoff=flow, high_frequency_cutoff=self.fmax)


        hacor = Array(hautocor, copy=True)
        indx = np.array([352250, 352256, 352260])

        snr = snr*nrm


        with _context:
           dof, achisq = \
               autochisq_from_precomputed(snr, snr, hacor, indx, stride=3,
                                          num_points=20)

        obt_snr = abs(snr[indx[1]])
        obt_ach = achisq[1]
        self.assertTrue(obt_snr > 10.0 and obt_snr < 12.0)
        self.assertTrue(obt_ach < 3.e-3)
        self.assertTrue(achisq[0] > 20.0)
        self.assertTrue(achisq[2] > 20.0)


        #with _context:
    #       dof, achi_list = autochisq(self.htilde, sig_tilde, psd,  stride=3,  num_points=20, \
        #          low_frequency_cutoff=flow, high_frequency_cutoff=self.fmax, max_snr=True)

        #self.assertTrue(obt_snr == achi_list[0, 1])
        #self.assertTrue(obt_ach == achi_list[0, 2])

    #    for i in range(1, len(achi_list)):
        #   self.assertTrue(achi_list[i,2] > 4.0)

    def test_sg(self):
        ### use a sin-gaussian as a signal

        sigt = TimeSeries(self.sig2, self.del_t)
        sig_tilde = make_frequency_series(sigt)

        del_f = sig_tilde.get_delta_f()
        psd = FrequencySeries(self.Psd, del_f)
        flow = self.low_frequency_cutoff

        with _context:
            hautocor, hacorfr, hnrm = matched_filter_core(self.htilde, self.htilde, psd=psd, \
                        low_frequency_cutoff=flow, high_frequency_cutoff=self.fmax)
            hautocor = hautocor * float(np.real(1./hautocor[0]))


            snr, cor, nrm = matched_filter_core(self.htilde, sig_tilde, psd=psd, \
                        low_frequency_cutoff=flow, high_frequency_cutoff=self.fmax)



        hacor = Array(hautocor.real(), copy=True)

        indx = np.array([301440, 301450, 301460])

        snr = snr*nrm

        with _context:
           dof, achisq = \
               autochisq_from_precomputed(snr, snr, hacor, indx, stride=3,
                                          num_points=20)

        obt_snr = abs(snr[indx[1]])
        obt_ach = achisq[1]
        self.assertTrue(obt_snr > 12.0 and obt_snr < 15.0)
        self.assertTrue(obt_ach > 6.8e3)
        self.assertTrue(achisq[0] > 6.8e3)
        self.assertTrue(achisq[2] > 6.8e3)

    def test_peak_lag_selection(self):
        autocorr = np.zeros(2048, dtype=np.complex128)
        autocorr[0] = 1.0
        for lag, value in ((100, 0.8), (200, 0.7), (300, 0.6)):
            autocorr[lag] = value
            autocorr[-lag] = value

        lags = select_autocorrelation_peak_lags(
            autocorr,
            num_peaks=3,
            max_lag=400,
            min_lag=20,
            min_separation=50,
        )
        np.testing.assert_array_equal(lags, [-300, -200, -100, 100, 200, 300])

    def test_peak_selected_signal_and_single_burst(self):
        autocorr = np.zeros(2048, dtype=np.complex128)
        autocorr[0] = 1.0
        for lag, value in ((100, 0.8), (200, 0.7), (300, 0.6)):
            autocorr[lag] = value
            autocorr[-lag] = value

        lags = select_autocorrelation_peak_lags(
            autocorr,
            num_peaks=3,
            max_lag=400,
            min_lag=20,
            min_separation=50,
        )
        trigger_index = 1000
        signal_snr = np.zeros(2048, dtype=np.complex128)
        signal_snr[trigger_index] = 10.0
        signal_snr[trigger_index + lags] = 10.0 * autocorr[lags]

        single_burst_snr = np.zeros(2048, dtype=np.complex128)
        single_burst_snr[trigger_index] = 10.0

        signal_dof, signal_chisq = autochisq_from_precomputed(
            signal_snr,
            signal_snr,
            autocorr,
            np.array([trigger_index]),
            twophase=True,
            lag_indices=lags,
        )
        burst_dof, burst_chisq = autochisq_from_precomputed(
            single_burst_snr,
            single_burst_snr,
            autocorr,
            np.array([trigger_index]),
            twophase=True,
            lag_indices=lags,
        )

        self.assertEqual(signal_dof, 12)
        self.assertEqual(burst_dof, 12)
        self.assertAlmostEqual(signal_chisq[0], 0.0, places=12)
        self.assertGreater(burst_chisq[0] / burst_dof, 10.0)


    #    with _context:
        #    dof, achi_list = autochisq(self.htilde, sig_tilde, psd,  stride=3,  num_points=20, \
        #          low_frequency_cutoff=flow, high_frequency_cutoff=self.fmax, max_snr=True)

        #self.assertTrue(obt_snr == achi_list[0, 1])
        #self.assertTrue(obt_ach == achi_list[0, 2])

    #    for i in range(1, len(achi_list)):
        #   self.assertTrue(achi_list[i,2] > 2.e3)



suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAutochisquare))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
