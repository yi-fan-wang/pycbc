
from .gaussian_noise import BaseGaussianNoise

class TimeDomainGaussian(BaseGaussianNoise):
	r"""A time domain model for Gaussian noise.
    """
    def __init__(self, variable_params, data, **kwargs):
        # we'll want the time-domain data, so store that
        self._td_data = {}
        # set up the boiler-plate attributes
        super().__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)

    @classmethod
    def from_config(cls, cp, data_section='data', data=None, psds=None,
                    **kwargs):
        """Adds highpass filtering to keyword arguments based on config file.
        """
        if cp.has_option(data_section, 'strain-high-pass') and \
            'highpass_waveforms' not in kwargs:
            kwargs['highpass_waveforms'] = float(cp.get(data_section,
                                                        'strain-high-pass'))
        return super().from_config(cp, data_section=data_section,
                                   data=data, psds=psds,
                                   **kwargs)

	@BaseDataModel.data.setter
    def data(self, data):
        """Store a copy of the FD and TD data."""
        BaseDataModel.data.fset(self, data)
        # store the td version
        self._td_data = {det: d.to_timeseries() for det, d in data.items()}

    @property
    def td_data(self):
        """The data in the time domain."""
        return self._td_data

    def _loglr(self):
        r"""Computes the log likelihood ratio.
        Returns
        -------
        float
            The value of the log likelihood ratio evaluated at the given point.
        """
        return self.loglikelihood - self.lognl

    def _loglikelihood(self):
        r"""Computes the log likelihood after removing the power within the
        given time window,

        .. math::
            \log p(d|\Theta) = -\frac{1}{2} \sum_i
             \left< d_i - h_i(\Theta) | d_i - h_i(\Theta) \right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood.
        """
        # generate the template waveform
        try:
            wfs = self.get_waveforms()
        except NoWaveformError:
            return self._nowaveform_logl()
        except FailedWaveformError as e:
            if self.ignore_failed_waveforms:
                return self._nowaveform_logl()
            raise e
        logl = 0.
        self.current_proj.clear()
        for det, h in wfs.items():
            invpsd = self._invpsds[det]
            norm = self.det_lognorm(det)
            # we always filter the entire segment starting from kmin, since the
            # gated series may have high frequency components
            slc = slice(self._kmin[det], self._kmax[det])
            # calculate the residual
            data = self.td_data[det]
            ht = h.to_timeseries()
            res = data - ht
            rr = 4 * invpsd.delta_f * rtilde[slc].inner(gated_rtilde[slc]).real
            logl += norm - 0.5*rr
        return float(logl)