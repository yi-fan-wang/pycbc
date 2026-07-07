# Copyright (C) 2026 Yifan Wang
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""GPU-batched Gaussian noise likelihood.

This model evaluates the standard Gaussian-noise log likelihood ratio
    loglr = sum_det Re<h|d> - <h|h>/2
for B parameter sets at once on the GPU (CuPy), using the batched
aligned-spin TaylorF2 generator in pycbc.waveform.batched_taylorf2. One
kernel launch produces the whole [B, Nf] waveform matrix per detector and
the inner products are two matrix operations, so the per-likelihood cost is
amortized (~500x over the scalar path for BNS-scale data).

Vectorized samplers should call ``loglr_batch``; the scalar interface used
by standard pycbc samplers routes through a batch of size 1 and therefore
stays consistent with the batched path.

Conventions (self-consistent between the batch and scalar paths of this
model, but NOT phase/time-referenced identically to lalsimulation TaylorF2):
``tc`` is the geocentric merger time; the waveform row phase is the SPA
polynomial with the merger at ``tc``; ``coa_phase`` enters as -2*phi_c.
"""
import numpy

from pycbc.detector import Detector
from .gaussian_noise import BaseGaussianNoise


class BatchedGaussianNoise(BaseGaussianNoise):
    r"""Gaussian-noise model with a GPU-batched TaylorF2 likelihood.

    Requires static or variable params: mass1, mass2, spin1z, spin2z,
    distance, tc, inclination, polarization, coa_phase, and static ra, dec.
    """
    name = 'batched_gaussian_noise'

    def __init__(self, variable_params, data, low_frequency_cutoff,
                 psds=None, high_frequency_cutoff=None, normalize=False,
                 static_params=None, **kwargs):
        super(BatchedGaussianNoise, self).__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        self._det_objects = {ifo: Detector(ifo) for ifo in self.data}
        self._generator = None
        self._dev = None
        d0 = list(self.data.values())[0]
        self.flen = len(d0)
        self.delta_f = d0.delta_f
        self.epoch = float(d0.start_time)
        self.f_lower = min(self._f_lower.values())

    def _ensure_device(self):
        """Lazily build the generator and move data products to the GPU."""
        if self._generator is not None:
            return
        import cupy as cp
        from pycbc.waveform.batched_taylorf2 import BatchSPATaylorF2
        self._generator = BatchSPATaylorF2(self.flen, self.delta_f,
                                           self.f_lower)
        self._dev = {}
        for det in self.data:
            ipsd = numpy.zeros(self.flen)
            kmin, kmax = self._kmin[det], self._kmax[det]
            invp = numpy.asarray(self._invpsds[det].numpy(), dtype=float)
            ipsd[kmin:kmax] = invp[kmin:kmax]
            wdata = numpy.asarray(self.data[det].numpy()) * ipsd
            self._dev[det] = (cp.asarray(wdata), cp.asarray(ipsd))

    def _all_param_arrays(self, params):
        """Merge variable-param arrays with (broadcast) static params."""
        B = len(next(iter(params.values())))
        out = {}
        for k, v in self.static_params.items():
            out[k] = numpy.full(B, v)
        for k, v in params.items():
            out[k] = numpy.asarray(v, dtype=float)
        return out, B

    def loglr_batch(self, params):
        """loglr for a batch of parameter sets.

        Parameters
        ----------
        params : dict of (B,) arrays keyed by parameter name (variable
            params at least; statics are filled in).

        Returns
        -------
        (B,) numpy array of loglr values.
        """
        import cupy as cp
        from pycbc.waveform.batched_taylorf2 import (
            detector_response_rows, batch_loglr_terms)
        self._ensure_device()
        p, B = self._all_param_arrays(params)
        p['dt'] = p['tc'] - self.epoch
        cols, kend = self._generator.intrinsic_arrays(p)
        rows = detector_response_rows(self._det_objects, p, p['ra'][0],
                                      p['dec'][0], float(p['tc'][0]))
        total = cp.zeros(B)
        for det, (amp_scale, dt_det, phi0) in rows.items():
            H = self._generator.generate(cols, kend, amp_scale, dt_det, phi0)
            wdata, ipsd = self._dev[det]
            hd, hh = batch_loglr_terms(H, wdata, ipsd, self.delta_f)
            total += hd - 0.5 * hh
        return cp.asnumpy(total)

    def _loglr(self):
        """Scalar interface: a batch of one, so any sampler works."""
        cp = self.current_params
        params = {k: numpy.array([cp[k]]) for k in self.variable_params}
        return float(self.loglr_batch(params)[0])
