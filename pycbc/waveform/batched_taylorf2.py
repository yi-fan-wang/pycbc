"""Batched aligned-spin TaylorF2 waveforms on the GPU (CuPy).

One kernel launch computes a [B, Nf] complex128 matrix of stationary-phase
TaylorF2 waveforms for B parameter sets. PN phasing coefficients are taken
from lalsimulation.SimInspiralTaylorF2AlignedPhasing on the host (same source
as pycbc.waveform.spa_tmplt, ~microseconds per point), so only the
per-frequency evaluation runs on the device.

The dominant-mode detector response is a per-row complex scalar,
    h_det = (F+ (1+cos^2 i)/2 - i Fx cos i) * h0 * exp(-2 pi i f dt_det),
so multi-detector waveforms reuse the same kernel via per-row amplitude
scale, constant phase offset, and time shift.

This module imports cupy lazily; importing it on a CPU-only node is safe.
"""
import numpy
from math import log, sqrt

from pycbc.constants import PI, MTSUN_SI, PC_SI, MRSUN_SI
from pycbc.libutils import import_optional

lal = import_optional('lal')
lalsimulation = import_optional('lalsimulation')

PI_4 = PI / 4.0
TWOPI = 2.0 * PI
LOG4 = log(4.0)

KERNEL_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void batch_spa(
    complex<double>* h,
    const double* pfaN, const double* pfa2, const double* pfa3,
    const double* pfa4, const double* pfa5, const double* pfl5,
    const double* pfa6, const double* pfl6, const double* pfa7,
    const double* piM, const double* amp, const double* dt,
    const double* phi0,
    const int* kmin, const int* kend,
    const double delta_f, const int nf, const int nb)
{
    int b = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb || k >= nf) return;

    long long idx = (long long) b * nf + k;
    if (k < kmin[b] || k >= kend[b]) {
        h[idx] = complex<double>(0.0, 0.0);
        return;
    }

    double f = k * delta_f;
    double v = cbrt(piM[b] * f);
    double logv = log(v);
    double v2 = v * v, v3 = v2 * v, v4 = v2 * v2;
    double v5 = v2 * v3, v6 = v3 * v3, v7 = v3 * v4;

    double phasing = 1.0
        + pfa2[b] * v2
        + pfa3[b] * v3
        + pfa4[b] * v4
        + (pfa5[b] + pfl5[b] * logv) * v5
        + (pfa6[b] + pfl6[b] * (logv + %LOG4%)) * v6
        + pfa7[b] * v7;
    phasing *= pfaN[b] / v5;
    phasing -= %PI_4%;
    phasing += %TWOPI% * f * dt[b] + phi0[b];

    double s, c;
    sincos(phasing, &s, &c);
    double a = amp[b] * pow(f, -7.0 / 6.0);
    h[idx] = complex<double>(c * a, -s * a);
}
""".replace('%LOG4%', repr(LOG4)).replace('%PI_4%', repr(PI_4)) \
   .replace('%TWOPI%', repr(TWOPI))


def spa_amplitude_factor(m1, m2):
    """Identical to pycbc.waveform.spa_tmplt.spa_amplitude_factor."""
    eta = m1 * m2 / (m1 + m2) ** 2
    FTaN = 32.0 * eta * eta / 5.0
    dETaN = -eta
    piM = PI * (m1 + m2) * MTSUN_SI
    amp0 = (4.0 * m1 * m2 / (1e6 * PC_SI) * MRSUN_SI * MTSUN_SI
            * sqrt(PI / 12.0))
    return -sqrt(-dETaN / FTaN) * amp0 * piM ** (-7.0 / 6.0)


def phasing_coeffs(m1, m2, s1z, s2z):
    """Aligned-spin PN phasing, normalized as in pycbc spa_tmplt."""
    phasing = lalsimulation.SimInspiralTaylorF2AlignedPhasing(
        float(m1), float(m2), float(s1z), float(s2z), lal.CreateDict())
    pfaN = phasing.v[0]
    return (pfaN, phasing.v[2] / pfaN, phasing.v[3] / pfaN,
            phasing.v[4] / pfaN, phasing.v[5] / pfaN,
            phasing.vlogv[5] / pfaN,
            (phasing.v[6] - phasing.vlogv[6] * LOG4) / pfaN,
            phasing.vlogv[6] / pfaN, phasing.v[7] / pfaN)


def f_isco(m1, m2):
    return 1.0 / (6.0 ** 1.5 * PI * (m1 + m2) * MTSUN_SI)


class BatchSPATaylorF2(object):
    """Batched aligned-spin TaylorF2 generator producing device arrays."""

    def __init__(self, flen, delta_f, f_lower):
        import cupy as cp
        self.cp = cp
        self.flen = int(flen)
        self.delta_f = float(delta_f)
        self.kmin_default = int(f_lower / delta_f)
        self.kernel = cp.RawKernel(KERNEL_SRC, 'batch_spa')

    def intrinsic_arrays(self, params):
        """Per-waveform host scalars from intrinsic parameters.

        params: dict of (B,) arrays with mass1, mass2, spin1z, spin2z.
        Returns (coeffs dict of (B,) float64 arrays, kend int32 array).
        """
        B = len(params['mass1'])
        names = ('pfaN', 'pfa2', 'pfa3', 'pfa4', 'pfa5', 'pfl5',
                 'pfa6', 'pfl6', 'pfa7')
        cols = {k: numpy.empty(B) for k in names + ('piM', 'amp0')}
        kend = numpy.empty(B, dtype=numpy.int32)
        for i in range(B):
            m1, m2 = params['mass1'][i], params['mass2'][i]
            vals = phasing_coeffs(m1, m2, params['spin1z'][i],
                                  params['spin2z'][i])
            for k, v in zip(names, vals):
                cols[k][i] = v
            cols['piM'][i] = PI * (m1 + m2) * MTSUN_SI
            cols['amp0'][i] = spa_amplitude_factor(m1, m2)
            kend[i] = min(int(f_isco(m1, m2) / self.delta_f), self.flen)
        return cols, kend

    def generate(self, cols, kend, amp_scale, dt, phi0):
        """Launch the kernel. cols/kend from intrinsic_arrays; amp_scale, dt,
        phi0 are per-row (B,) host arrays (detector-dependent)."""
        cp = self.cp
        B = len(kend)
        dev = {k: cp.asarray(v) for k, v in cols.items()}
        H = cp.empty((B, self.flen), dtype=cp.complex128)
        nt = 256
        grid = ((self.flen + nt - 1) // nt, B)
        self.kernel(grid, (nt,), (
            H, dev['pfaN'], dev['pfa2'], dev['pfa3'], dev['pfa4'],
            dev['pfa5'], dev['pfl5'], dev['pfa6'], dev['pfl6'], dev['pfa7'],
            dev['piM'],
            cp.asarray(dev['amp0'] * cp.asarray(amp_scale)),
            cp.asarray(numpy.asarray(dt, dtype=numpy.float64)),
            cp.asarray(numpy.asarray(phi0, dtype=numpy.float64)),
            cp.full(B, self.kmin_default, dtype=cp.int32),
            cp.asarray(kend),
            numpy.float64(self.delta_f), numpy.int32(self.flen),
            numpy.int32(B)))
        return H


def detector_response_rows(detectors, params, ra, dec, t_gps):
    """Per-detector, per-row response scalars for the dominant mode.

    params must carry (B,) arrays: inclination, polarization, coa_phase,
    distance, dt (geocentric time offset), tc (geocentric merger gps time,
    used for the antenna pattern and light-travel delay of each row).
    Returns {det: (amp_scale, dt_det, phi0)} with (B,) host arrays.
    """
    cosi = numpy.cos(params['inclination'])
    ap = 0.5 * (1.0 + cosi ** 2)
    ac = cosi
    B = len(cosi)
    tcs = params.get('tc', numpy.full(B, t_gps))
    out = {}
    for name, det in detectors.items():
        fp = numpy.empty(B)
        fc = numpy.empty(B)
        delay = numpy.empty(B)
        for i in range(B):
            fp[i], fc[i] = det.antenna_pattern(
                ra, dec, params['polarization'][i], tcs[i])
            delay[i] = det.time_delay_from_earth_center(ra, dec, tcs[i])
        c = fp * ap - 1j * fc * ac
        amp_scale = numpy.abs(c) / params['distance']
        # h = a exp(-i phi): a row phase offset X enters as phi -> phi - X
        phi0 = -numpy.angle(c) - 2.0 * params['coa_phase']
        dt_det = params['dt'] + delay
        out[name] = (amp_scale, dt_det, phi0)
    return out


def batch_loglr_terms(H, whitened_data, inv_psd, delta_f):
    """Return (<h|d>_Re, <h|h>) per row, on device.

    whitened_data = data * inv_psd precomputed (Nf,), inv_psd zero outside
    the analysis band.
    """
    import cupy as cp
    hd = 4.0 * delta_f * (H.conj() @ whitened_data).real
    hh = 4.0 * delta_f * cp.einsum('ij,ij,j->i', H.conj(), H, inv_psd).real
    return hd, hh
