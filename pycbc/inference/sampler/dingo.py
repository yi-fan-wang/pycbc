""" Sampler that draws posterior samples from a trained Dingo model
(neural posterior estimation) instead of running a stochastic sampler.

Dingo (https://github.com/dingo-gw/dingo) performs amortized inference:
a normalizing flow is trained to represent p(theta|d) for data drawn from
the prior and detector noise distribution. At inference time the network is
conditioned on the observed data and posterior samples are produced with a
single forward pass.

This sampler builds the Dingo "context" (frequency-domain strain and ASDs)
from the data held by the PyCBC model, so the data selection and conditioning
are controlled by the standard ``[data]`` section of the PyCBC inference
configuration file. The network's segment (duration T = 1/delta_f, ending at
time_buffer after the trigger) is cropped out of the model's data segment,
which must contain it; the sample rate must be at least twice the network's
maximum frequency.

The analysis segment is Tukey-windowed in the time domain to match Dingo's
event-data conditioning. This is essential, not cosmetic: without a window,
power from loud spectral lines (violin modes, power-line harmonics) leaks
across the whole band through the sinc sidelobes of the segment FFT and the
whitened data no longer look like the training distribution. (PyCBC's own
likelihoods instead absorb leakage via inverse-PSD truncation, which is not
available here.)
"""

import logging

import numpy
from scipy import special

from pycbc import conversions
from pycbc.inference import models
from pycbc.inference.io import PosteriorFile
from pycbc.pool import choose_pool

from .base import BaseSampler, setup_output


def _call_loglikelihood(params):
    """Update the global model instance and return its loglikelihood.

    Module-level so that it can be pickled for multiprocessing pools.
    """
    models._global_instance.update(**params)
    return models._global_instance.loglikelihood


# Map from Dingo (bilby-style) parameter names to PyCBC names for the
# parameters that translate one-to-one. Masses and coalescence time are
# handled separately.
DINGO_PARAM_MAP = {
    'luminosity_distance': 'distance',
    'theta_jn': 'inclination',
    'psi': 'polarization',
    'phase': 'coa_phase',
    'ra': 'ra',
    'dec': 'dec',
    'chi_1': 'spin1z',
    'chi_2': 'spin2z',
    'a_1': 'spin1_a',
    'a_2': 'spin2_a',
    'tilt_1': 'spin1_polar',
    'tilt_2': 'spin2_polar',
}


class DingoSampler(BaseSampler):
    """Draws samples from a trained Dingo network conditioned on the data
    stored in the PyCBC model.

    Parameters
    ----------
    model : Model
        An instance of a model from ``pycbc.inference.models``. Must be a
        data-based model (it provides ``data`` and ``psds``).
    model_file : str
        Path to the trained Dingo model (``.pt`` file).
    trigger_time : float
        GPS time of the event. Dingo samples ``geocent_time`` relative to
        this time; it is also used to correct the sky location for the
        detector positions at the network reference time.
    num_samples : int, optional
        Number of posterior samples to draw (default 5000).
    batch_size : int, optional
        Batch size for sampling on the GPU (default: ``num_samples``).
    device : str, optional
        Torch device, e.g. 'cuda' or 'cpu'. Defaults to 'cuda' when
        available.
    time_buffer : float, optional
        Time between the trigger and the end of the analysis segment that
        the network was trained with (Dingo's ``post_trigger_duration``;
        default 2.0 s). The data are cyclically time-shifted so that the
        trigger sits at this position, since PyCBC truncates segment
        boundaries to integer GPS times.
    window_roll_off : float, optional
        Roll-off time (s) of the Tukey window applied to the analysis
        segment, matching Dingo's event-data conditioning (default 1.0,
        the dingo_pipe default).
    importance_sample : bool, optional
        Reweight the network samples to the exact posterior defined by the
        PyCBC model's likelihood and the Dingo network's prior:
        ``log w = loglikelihood + log pi_dingo - log q_flow``. Any of the
        model's variable_params that the network does not produce (e.g.
        beyond-GR deviation parameters) are drawn from the model's prior,
        so their prior and proposal densities cancel in the weights. The
        estimated evidence and effective sample size are stored in the
        output file. Default False.
    """
    name = 'dingo'

    def __init__(self, model, model_file, trigger_time,
                 num_samples=5000, batch_size=None, device=None,
                 time_buffer=2.0, window_roll_off=1.0,
                 importance_sample=False, nprocesses=1, use_mpi=False):
        super().__init__(model)

        import torch
        from dingo.core.posterior_models.build_model import (
            build_model_from_kwargs)
        from dingo.gw.inference.gw_samplers import GWSampler

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_samples = int(num_samples)
        self.batch_size = (int(batch_size) if batch_size is not None
                           else self.num_samples)
        self.trigger_time = float(trigger_time)
        self.time_buffer = float(time_buffer)
        self.window_roll_off = float(window_roll_off)
        if isinstance(importance_sample, str):
            importance_sample = importance_sample.lower() in ('true', '1',
                                                              'yes')
        self.importance_sample = importance_sample
        # The global instance must be set before the pool forks its worker
        # processes, so that they inherit it.
        models._global_instance = model
        self.pool = choose_pool(mpi=use_mpi, processes=nprocesses)
        self._samples = {}
        self.meta = {}

        logging.info("Loading Dingo model from %s onto %s",
                     model_file, device)
        posterior_model = build_model_from_kwargs(
            filename=model_file, device=device, load_training_info=False)
        self.dingo_sampler = GWSampler(model=posterior_model)

        self.dingo_sampler.context = self._context_from_model()
        self.dingo_sampler.event_metadata = {
            'time_event': self.trigger_time}

    def _context_from_model(self):
        """Assemble the Dingo context (strain and ASDs on the network's
        frequency grid) from the PyCBC model's data and PSDs.

        The model's data segment may be longer than the network's segment
        (e.g. because of psd-inverse-length padding) and its boundaries are
        truncated to integer GPS times by PyCBC, whereas Dingo networks
        expect a segment of duration T = 1/delta_f ending at
        trigger_time + time_buffer. The segment is therefore realigned via
        a cyclic frequency-domain time shift, cropped to T in the time
        domain, and Tukey-windowed like Dingo's event data. If the model
        segment is longer than T, the crop contains only contiguous data;
        otherwise a small amount of noise wraps around into the window
        roll-off region.
        """
        from scipy.signal.windows import tukey
        from dingo.gw.domains import MultibandedFrequencyDomain

        domain = self.dingo_sampler.domain
        if isinstance(domain, MultibandedFrequencyDomain):
            # Event data is provided on the underlying uniform grid; the
            # sampler decimates it internally.
            domain = domain.base_domain
        delta_f = domain.delta_f
        nbins = len(domain)
        duration = 1 / delta_f
        seg_start = self.trigger_time + self.time_buffer - duration
        domain_freqs = numpy.arange(nbins) * delta_f

        waveform = {}
        asds = {}
        for det in self.dingo_sampler.detectors:
            try:
                data = self.model.data[det]
                psd = self.model.psds[det]
            except (AttributeError, KeyError):
                raise ValueError(
                    "The Dingo network requires data for detector {}, but "
                    "the model does not provide it. Check the instruments "
                    "in the [data] section.".format(det))
            data_duration = 1 / float(data.delta_f)
            sample_rate = 2 * (len(data) - 1) * float(data.delta_f)
            if sample_rate < 2 * domain.f_max * (1 - 1e-10):
                raise ValueError(
                    "Data sample rate {} Hz is below twice the Dingo "
                    "network f_max = {} Hz.".format(sample_rate,
                                                    domain.f_max))
            if data_duration < duration * (1 - 1e-10):
                raise ValueError(
                    "Data segment for {} is shorter than the Dingo "
                    "network segment T = {} s.".format(det, duration))
            # Shift so that the data segment starts at seg_start.
            shift = seg_start - float(data.epoch)
            wrapped = max(0, -shift) + max(0, shift + duration
                                           - data_duration)
            if wrapped > duration / 2:
                raise ValueError(
                    "Data segment for {} covers less than half of the "
                    "Dingo segment [{:.2f}, {:.2f}]; check the analysis "
                    "start/end times.".format(det, seg_start,
                                              seg_start + duration))
            if wrapped > 0:
                logging.info("%.3f s of %s data wrap cyclically into the "
                             "window roll-off region", wrapped, det)
            data_freqs = numpy.arange(len(data)) * float(data.delta_f)
            fd = numpy.asarray(data.data, dtype=numpy.complex128)
            fd *= numpy.exp(2j * numpy.pi * data_freqs * shift)
            # Crop to the network segment duration and window. The rfft /
            # irfft round trip preserves the continuous-FT normalization
            # since the sample rate is unchanged.
            td = numpy.fft.irfft(fd)
            td = td[:int(round(duration * sample_rate))]
            td *= tukey(len(td), 2 * self.window_roll_off / duration)
            strain = numpy.fft.rfft(td)[:nbins]
            # Dingo expects the trigger at cyclic time zero of the segment
            # (dingo.gw.data.data_preparation applies
            # cyclic_time_shift(time_buffer) after windowing); rotate by the
            # buffer time. This is a pure phase factor, applied after the
            # window so that the taper stays on the physical segment edges.
            strain *= numpy.exp(-2j * numpy.pi * domain_freqs
                                * self.time_buffer)
            waveform[det] = strain
            psd_freqs = numpy.arange(len(psd)) * float(psd.delta_f)
            asds[det] = numpy.interp(
                domain_freqs, psd_freqs,
                numpy.sqrt(numpy.asarray(psd.data, dtype=numpy.float64)))
        return {'waveform': waveform, 'asds': asds}

    @classmethod
    def from_config(cls, cp, model, output_file=None, nprocesses=1,
                    use_mpi=False):
        """Initializes the sampler from the given config file.

        Options in the ``[sampler]`` section:

        * ``model-file`` (required): path to the trained Dingo network.
        * ``num-samples``, ``batch-size``, ``device``, ``trigger-time``:
          optional; ``trigger-time`` defaults to the value in the
          ``[data]`` section.
        """
        section = 'sampler'
        model_file = cp.get(section, 'model-file')
        kwargs = {}
        for opt in ('num-samples', 'batch-size', 'device', 'trigger-time',
                    'time-buffer', 'window-roll-off', 'importance-sample'):
            if cp.has_option(section, opt):
                kwargs[opt.replace('-', '_')] = cp.get(section, opt)
        if 'trigger_time' not in kwargs:
            kwargs['trigger_time'] = cp.get('data', 'trigger-time')
        obj = cls(model, model_file, nprocesses=nprocesses, use_mpi=use_mpi,
                  **kwargs)
        setup_output(obj, output_file, check_nsamples=False, validate=False)
        return obj

    @property
    def io(self):
        return PosteriorFile

    @property
    def samples(self):
        """Dict of posterior samples, keyed by PyCBC parameter names."""
        return self._samples

    @property
    def model_stats(self):
        return None

    def run(self):
        self.dingo_sampler.run_sampler(self.num_samples,
                                       batch_size=self.batch_size)
        df = self.dingo_sampler.samples
        logging.info("Dingo sampling complete; converting %d samples to "
                     "PyCBC conventions", len(df))
        out = self._convert_samples(df)
        # The flow has no hard prior boundaries, so a small fraction of
        # samples may fall outside the physical range (e.g. mass_ratio <= 0),
        # yielding NaNs in the conversions or -inf prior density. These have
        # zero posterior support; drop them.
        keep = numpy.ones(len(df), dtype=bool)
        for values in out.values():
            keep &= numpy.isfinite(values)
        log_prior = None
        if self.importance_sample:
            log_prior = self._dingo_log_prior(df)
            keep &= numpy.isfinite(log_prior)
        if not keep.all():
            logging.info("Dropping %d samples outside the physical "
                         "parameter range", (~keep).sum())
            out = {name: values[keep] for name, values in out.items()}
        if self.importance_sample:
            self._reweigh_samples(out, log_prior[keep])
        self._samples = out

    def _reweigh_samples(self, out, log_prior):
        """Importance-sample the network output against the PyCBC model.

        Computes ``log w = loglikelihood + log_prior - log_proposal``, where
        the likelihood is evaluated by the PyCBC model, the prior is the
        Dingo network's training prior, and the proposal is the flow density
        (both in Dingo's parameter space; the likelihood value is
        coordinate-independent). ``out`` is modified in place.
        """
        nsamples = len(log_prior)
        log_proposal = out['log_prob']
        # Variable params the network does not produce are drawn from the
        # model's prior, so prior and proposal densities cancel for them.
        variable_params = list(self.model.variable_params)
        extra = [p for p in variable_params if p not in out]
        if extra:
            logging.info("Drawing %s from the model prior", extra)
            rvs = self.model.prior_rvs(nsamples)
            for param in extra:
                out[param] = numpy.asarray(rvs[param], dtype=numpy.float64)
        logging.info("Evaluating the model loglikelihood for %d samples",
                     nsamples)
        param_dicts = [{p: out[p][i] for p in variable_params}
                       for i in range(nsamples)]
        loglikelihood = numpy.array(
            self.pool.map(_call_loglikelihood, param_dicts))
        log_weight = loglikelihood + log_prior - log_proposal
        log_norm = special.logsumexp(log_weight)
        ess = numpy.exp(-special.logsumexp(2 * (log_weight - log_norm)))
        self.meta['log_evidence'] = log_norm - numpy.log(nsamples)
        self.meta['ess'] = ess
        self.meta['sample_efficiency'] = ess / nsamples
        logging.info("Importance sampling: log_evidence = %.3f, effective "
                     "sample size = %.1f of %d (efficiency %.2f%%)",
                     self.meta['log_evidence'], ess, nsamples,
                     100 * self.meta['sample_efficiency'])
        out['loglikelihood'] = loglikelihood
        out['log_prior'] = log_prior
        out['log_weight'] = log_weight

    def _dingo_log_prior(self, df):
        """Evaluate the Dingo training prior density for the raw network
        samples, in Dingo's parameter space.
        """
        from dingo.gw.gwutils import get_extrinsic_prior_dict
        from dingo.gw.prior import build_prior_with_defaults

        metadata = self.dingo_sampler.metadata
        intrinsic = metadata['dataset_settings']['intrinsic_prior']
        extrinsic = get_extrinsic_prior_dict(
            metadata['train_settings']['data']['extrinsic_prior'])
        prior = build_prior_with_defaults({**intrinsic, **extrinsic})
        theta = {p: df[p].to_numpy(dtype=numpy.float64)
                 for p in self.dingo_sampler.inference_parameters}
        with numpy.errstate(divide='ignore'):
            return numpy.asarray(prior.ln_prob(theta, axis=0),
                                 dtype=numpy.float64)

    def _convert_samples(self, df):
        """Convert a DataFrame of Dingo samples to a dict of arrays with
        PyCBC parameter names.
        """
        # The network outputs float32; cast up front so that e.g. adding the
        # GPS trigger time to geocent_time does not lose the sub-second
        # information.
        cols = {c: df[c].to_numpy(dtype=numpy.float64) for c in df.columns}
        out = {}
        for dingo_name, pycbc_name in DINGO_PARAM_MAP.items():
            if dingo_name in cols:
                out[pycbc_name] = cols[dingo_name]
        # Dingo uses mass_ratio = m2/m1 <= 1; PyCBC conventions take
        # q = m1/m2 >= 1.
        if 'chirp_mass' in cols and 'mass_ratio' in cols:
            invq = 1.0 / cols['mass_ratio']
            out['mchirp'] = cols['chirp_mass']
            out['q'] = invq
            out['mass1'] = conversions.mass1_from_mchirp_q(
                cols['chirp_mass'], invq)
            out['mass2'] = conversions.mass2_from_mchirp_q(
                cols['chirp_mass'], invq)
        # Dingo geocent_time is relative to the trigger time.
        if 'geocent_time' in cols:
            out['tc'] = self.trigger_time + cols['geocent_time']
        # Pass through anything not covered above (e.g. log_prob) under
        # its original name.
        handled = set(DINGO_PARAM_MAP) | {'chirp_mass', 'mass_ratio',
                                          'geocent_time'}
        for name, values in cols.items():
            if name not in handled:
                out.setdefault(name, values)
        return out

    def finalize(self):
        with self.io(self.checkpoint_file, "a") as fp:
            fp.write_samples(samples=self._samples)
            for key, value in self.meta.items():
                fp[fp.sampler_group].attrs[key] = value

    checkpoint = resume_from_checkpoint = run
