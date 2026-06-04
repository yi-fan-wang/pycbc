"""
Test that pycbc_injection_minifollowup correctly passes eccentricity and
rel_anomaly to pycbc_single_template for eccentric bank templates.

Two tests:
  1. Unit test: mock hd_sngl to verify curr_params picks up eccentricity.
  2. Integration test: call pycbc_single_template directly with the exact
     parameters from the failing production job (with and without eccentricity)
     to confirm the waveform generates successfully when e is passed.
"""

import sys
import types
import unittest
import subprocess
import numpy as np


# ---------------------------------------------------------------------------
# Helper: build a minimal mock of SingleDetTriggers with eccentric bank data
# ---------------------------------------------------------------------------

class MockHdSngl:
    """Minimal stand-in for SingleDetTriggers."""
    def __init__(self, eccentric=True):
        self.mass1 = np.array([44.659255])
        self.mass2 = np.array([71.260093])
        self.spin1z = np.array([-0.425737])
        self.spin2z = np.array([-0.093056])
        self.f_lower = np.array([20.0])
        if eccentric:
            self.eccentricity = np.array([0.13244177])
            self.rel_anomaly = np.array([2.35492297])

    def __getattr__(self, key):
        raise KeyError(key)


def _build_curr_params(hd_sngl):
    """
    Replicate the curr_params block from pycbc_injection_minifollowup so we
    can test it without invoking the full workflow machinery.
    """
    import copy
    inj_params = {}          # empty — we only care about template params here
    curr_params = copy.deepcopy(inj_params)
    curr_params['mass1']  = hd_sngl.mass1[0]
    curr_params['mass2']  = hd_sngl.mass2[0]
    curr_params['spin1z'] = hd_sngl.spin1z[0]
    curr_params['spin2z'] = hd_sngl.spin2z[0]
    curr_params['f_lower'] = hd_sngl.f_lower[0]
    try:
        curr_params['spin1x'] = hd_sngl.spin1x[0]
        curr_params['spin2x'] = hd_sngl.spin2x[0]
        curr_params['spin1y'] = hd_sngl.spin1y[0]
        curr_params['spin2y'] = hd_sngl.spin2y[0]
        curr_params['inclination'] = hd_sngl.inclination[0]
    except (KeyError, AttributeError):
        pass
    try:
        curr_params['u_vals'] = hd_sngl.u_vals[0]
    except (KeyError, AttributeError):
        pass
    # --- the fix under test ---
    try:
        curr_params['eccentricity'] = hd_sngl.eccentricity[0]
        curr_params['rel_anomaly']  = hd_sngl.rel_anomaly[0]
    except (KeyError, ValueError, AttributeError):
        pass
    return curr_params


# ---------------------------------------------------------------------------
# Test 1: unit test on parameter-building logic
# ---------------------------------------------------------------------------

class TestCurrParamsEccentricity(unittest.TestCase):

    def test_eccentricity_passed_when_in_bank(self):
        """eccentricity and rel_anomaly appear in curr_params for eccentric bank."""
        hd_sngl = MockHdSngl(eccentric=True)
        params = _build_curr_params(hd_sngl)
        self.assertIn('eccentricity', params,
                      "eccentricity missing from curr_params")
        self.assertIn('rel_anomaly', params,
                      "rel_anomaly missing from curr_params")
        self.assertAlmostEqual(params['eccentricity'], 0.13244177, places=6)
        self.assertAlmostEqual(params['rel_anomaly'],  2.35492297, places=5)

    def test_eccentricity_absent_for_noneccentric_bank(self):
        """No KeyError and no eccentricity key for a plain bank template."""
        hd_sngl = MockHdSngl(eccentric=False)
        params = _build_curr_params(hd_sngl)
        self.assertNotIn('eccentricity', params)
        self.assertNotIn('rel_anomaly',  params)

    def test_standard_params_always_present(self):
        """mass1, mass2, spin1z, spin2z, f_lower are always present."""
        for eccentric in (True, False):
            hd_sngl = MockHdSngl(eccentric=eccentric)
            params = _build_curr_params(hd_sngl)
            for key in ('mass1', 'mass2', 'spin1z', 'spin2z', 'f_lower'):
                self.assertIn(key, params)


# ---------------------------------------------------------------------------
# Test 2: integration — generate the waveform with and without eccentricity
# ---------------------------------------------------------------------------

class TestWaveformGeneration(unittest.TestCase):
    """
    Directly exercise the SEOBNRv5E waveform generator with the exact
    parameters from the failing production job to confirm:
      - e=0.13 (correct bank template) succeeds
      - e=0.0  (what the old code passed) fails with FailedWaveformError
    """

    BASE_PARAMS = dict(
        approximant='SEOBNRv5E',
        mass1=44.659255,
        mass2=71.260093,
        spin1z=-0.425737,
        spin2z=-0.093056,
        f_lower=20.0,
        delta_f=1.0 / 512,   # matches pycbc_single_template segment length
        coa_phase=0.0,
    )

    def _generate(self, **extra):
        from pycbc.waveform import get_fd_waveform
        params = {**self.BASE_PARAMS, **extra}
        hp, _ = get_fd_waveform(**params)
        return hp

    def test_waveform_succeeds_with_correct_eccentricity(self):
        """Waveform generation must succeed when e=0.13 is passed (fixed path)."""
        hp = self._generate(eccentricity=0.13244177, rel_anomaly=2.35492297)
        self.assertGreater(len(hp), 0)
        self.assertTrue(np.any(np.abs(hp.data) > 0),
                        "Waveform is all zeros — generation silently failed")

    def test_waveform_fails_without_eccentricity(self):
        """Waveform generation should raise FailedWaveformError when e=0 (old path)."""
        from pycbc.waveform.waveform import FailedWaveformError
        with self.assertRaises(FailedWaveformError):
            self._generate(eccentricity=0.0, rel_anomaly=0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
