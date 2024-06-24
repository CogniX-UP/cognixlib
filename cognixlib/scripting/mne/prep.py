import mne 
import pyprep
from pyprep.removeTrend import removeTrend
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.reference import Reference

import numpy as np

def runline(y, n, dn):
    """Perform local linear regression on a channel of EEG data.

    A re-implementation of the ``runline`` function from the Chronux package
    for MATLAB [1]_.

    Parameters
    ----------
    y : np.ndarray
        A 1-D array of data from a single EEG channel.
    n : int
        Length of the detrending window.
    dn : int
        Length of the window step size.

    Returns
    -------
    y: np.ndarray
       The detrended signal for the given EEG channel.

    References
    ----------
    .. [1] http://chronux.org/

    """
    nt = y.shape[0]
    y_line = np.zeros((nt, 1))
    norm = np.zeros((nt, 1))
    nwin = int(np.ceil((nt - n) / dn))
    yfit = np.zeros((nwin, n))
    xwt = (np.arange(1, n + 1) - n / 2) / (n / 2)
    wt = np.power(1 - np.power(np.absolute(xwt), 3), 3)
    for j in range(0, nwin):
        tseg = y[dn * j : dn * j + n]
        y1 = np.mean(tseg)
        y2 = np.mean(np.multiply(np.arange(1, n + 1), tseg)) * (2 / (n + 1))
        a = np.multiply(np.subtract(y2, y1), 6 / (n - 1))
        b = np.subtract(y1, a * (n + 1) / 2)
        yfit[j, :] = np.multiply(np.arange(1, n + 1), a) + b
        y_line[j * dn : j * dn + n] = y_line[j * dn : j * dn + n] + np.reshape(
            np.multiply(yfit[j, :], wt), (n, 1)
        )
        norm[j * dn : j * dn + n] = norm[j * dn : j * dn + n] + np.reshape(wt, (n, 1))

    for i in range(0, len(norm)):
        if norm[i] > 0:
            y_line[i] = y_line[i] / norm[i]
    indx = (nwin - 1) * dn + n - 1
    npts = len(y) - indx + 1
    y_line[indx - 1 :] = np.reshape(
        (np.multiply(np.arange(n + 1, n + npts + 1), a) + b), (npts, 1)
    )
    for i in range(0, len(y_line)):
        y[i] = y[i] - y_line[i]
    return y

def remove_trend_data(
    data: np.ndarray,
    sample_rate: float,
    detrendType: str='high pass',
    detrendCutoff=1.0,
    detrendChannels: list = None
) -> np.ndarray:
    """Remove trends (i.e., slow drifts in baseline) from an array of EEG data."""
    if len(data.shape) == 1:
        data = np.reshape(data, (1, data.shape[0]))
    
    if detrendType.lower() == 'high pass':
        data = mne.filter.filter_data(
            data = data,
            sfreq=sample_rate,
            l_freq=detrendCutoff,
            h_freq=None,
            picks = detrendChannels
        )

    elif detrendType.lower() == 'high pass sinc':
        fOrder = np.round(14080 * sample_rate / 512)
        fOrder = int(fOrder + fOrder % 2)
        data = mne.filter.filter_data(
            data = data,
            sfreq = sample_rate,
            l_freq=1,
            h_freq=None,
            picks=detrendChannels,
            filter_length=fOrder,
            fir_window='blackman'
        )
    
    elif detrendType.lower() == 'local detrend':
        if detrendChannels is None:
            detrendChannels = np.arange(0,data.shape[0])
        windowSize = 1.5/detrendCutoff
        windowSize = np.minimum(windowSize,data.shape[1])
        stepSize = 0.02
        data = np.transpose(data)
        n = np.round(sample_rate * windowSize)
        dn = np.round(sample_rate * stepSize)

        if dn > n or dn < 1:
            print("Step size should be less than the window size and contain at least 1 sample")
        if n == data.shape[0]:
            pass
        else:
            for ch in detrendChannels:
                data[:,ch] = runline(data[:,ch],int(n),int(dn))
        data = np.transpose(data)

    else:
        print("No filtering/detrending performed since the detrend type did not match")

    return data


def line_noise_removal_prep(raw_eeg: mne.io.Raw, sfreq: float, linenoise: float):
    EEG_raw = raw_eeg.get_data()
    EEG_new = removeTrend(EEG_raw, sfreq, matlab_strict=False)

    # Step 2: Removing line noise
    EEG_clean = mne.filter.notch_filter(
        EEG_new,
        Fs=sfreq,
        freqs=linenoise,
        method="spectrum_fit",
        mt_bandwidth=2,
        p_value=0.01,
        filter_length="10s",
    )
    # Add Trend back
    EEG = EEG_raw - EEG_new + EEG_clean
    raw_eeg._data = EEG
    return raw_eeg

def referencing_prep(
    raw_eeg: mne.io.Raw,
    ref_chs: str | list,
    reref_chs: str | list,
    line_freqs: float | list,
    max_iterations=4,
    ransac=True,
    channel_wise=False,
    max_chunk_size:int = None
):

    ch_names_all = raw_eeg.ch_names.copy()
    ch_types_all = raw_eeg.get_channel_types()
    ch_names_eeg = [
        ch_names_all[i]
        for i in range(len(ch_names_all))
        if ch_types_all[i] == 'eeg'
        ]
    
    raw_eeg.pick_channels(ch_names_eeg)
    
    prep_params = {
        'ref_chs':ref_chs,
        'reref_chs':reref_chs,
        'line_freqs':line_freqs
    }
    
    if prep_params["ref_chs"] == "eeg":
        prep_params["ref_chs"] = ch_names_eeg
    if prep_params["reref_chs"] == "eeg":
        prep_params["reref_chs"] = ch_names_eeg
    if "max_iterations" not in prep_params.keys():
        prep_params["max_iterations"] = max_iterations

    ransac_settings = {
        "ransac": ransac,
        "channel_wise": channel_wise,
        "max_chunk_size": max_chunk_size,
    }
    
    reference = Reference(
        raw = raw_eeg,
        params = prep_params,
        random_state = 1,
        matlab_strict = False
        **ransac_settings
    )
    
    reference.perform_reference(prep_params["max_iterations"])
    raw_eeg = reference.raw
    noisy_channels_original = reference.noisy_channels_original
    noisy_channels_before_interpolation = (
        reference.noisy_channels_before_interpolation
    )
    noisy_channels_after_interpolation = (
        reference.noisy_channels_after_interpolation
    )
    bad_before_interpolation = reference.bad_before_interpolation
    EEG_before_interpolation = reference.EEG_before_interpolation
    reference_before_interpolation = reference.reference_signal
    reference_after_interpolation = reference.reference_signal_new
    interpolated_channels = reference.interpolated_channels
    still_noisy_channels = reference.still_noisy_channels
    
    return raw_eeg

def noisychannelsprep(raw_eeg:mne.io.Raw):
    noisy_detector = NoisyChannels(raw_eeg,random_state=1)
    noisy_detector.find_bad_by_nan_flat()
    return raw_eeg
    
    