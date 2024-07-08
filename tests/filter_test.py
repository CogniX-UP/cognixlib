"""
Some good-to-know facts.

1)  MNE's filter function is really slow, for some reason. At least for FIR filters. It's only fast for larger
    signals.
2)  scipy's oaconvole is quite faster than MNE's corresponding overlap-add method.
    Furthermore, there is a mode for removing phase lag/delay.
3)  In a real-time BCI setting, there will always be a phase delay. In essence, we
    should always attempt to have minimum phase delay, otherwise things won's work
    correctly.
    
    For BCIs, we should always aim for minimum phase-delay, since it is not possible
    to remove the delay in real-time (to remove it we'd need future values)
4)  We really need to parallelize stuff and not between processes.

lfilter == np.convolve == real time

In realtime, applying lfilter and np.convolve and real time FIRfilter all yield the
same result.

For attempting to correct phase delay, filtfilt, oaconvolve and mne all yield different
results.

For FIR filters, the forward-backwards and time shifting are identical bar the change
in amplitude. If we don't have FIR filters, then it's better to do forward-backward
for zero-phase filtering.

Finally, we have edge and transient effects. I think these are automatically handled when
using overlap-add in the case of FIR filters. Either way, in offline situations, this shouldn't
matter since after the transient effects the filtering is identical.
"""

if __name__ != '__main__':
    exit()
    
from scipy.signal import oaconvolve, firwin, minimum_phase, convolve, filtfilt, lfilter
from scipy.fft import fft, fftfreq
from mne.filter import create_filter, _overlap_add_filter
import numpy as np
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from joblib import parallel_backend
from random import randint

from cognixlib.scripting.processing.filters import FIROfflineApplier, FIROnlineApplier

fs = 2048
# t = int(fs*10800) # 3 hours
t = int(fs*1) # 1 second
channels = 8
channel_in = np.longdouble(35 * np.random.rand(t, channels))

#%%

h_t = firwin(845, [8, 30], pass_zero=False, fs=fs)
# k = minimum_phase(h_t)
#%%
h = create_filter(
    data=None, 
    sfreq=fs, 
    l_freq=8, 
    h_freq=30,
    l_trans_bandwidth=8,
    h_trans_bandwidth=8,
    phase='zero-double'
)
# h = h_t

t0 = time.perf_counter()
# this is much faster due to subprocess copying data
with parallel_backend('threading'):
    mne_res = _overlap_add_filter(channel_in.T, h, phase="zero",n_jobs=-1)
t1 = time.perf_counter()
print(f"MNE RES: {t1-t0}")

h_m = np.tile(h, (channels, 1)).T
t0 = time.perf_counter()
filt_filt = filtfilt(h, 1, channel_in, axis=0, padtype='even', padlen=2000)
t1 = time.perf_counter()
print(f"FILT FILT RES: {t1-t0}")

t0 = time.perf_counter()
# pad = 50
# ch = np.pad(channel_in, [(pad, pad), (0, 0)], mode='reflect') # how to pad like filtfilt
sci_res = oaconvolve(channel_in, h_m, 'same', 0)
h_r = h_m[::-1]
sci_res = oaconvolve(sci_res, h_r, 'same', 0)[0:t]
t1 = time.perf_counter()
print(f"SCIPY RES: {t1-t0}")

# FIR Offline
fir_offline = FIROfflineApplier(h_m, channels, phase=FIROfflineApplier.Phase.ZERO)
t0 = time.perf_counter()
fir_off_res = fir_offline.filter(channel_in)
t1 = time.perf_counter()
print(f"FIR_OFF: {t1-t0}")

step = 512
real_res = np.zeros((t, channels))
fir_online = FIROnlineApplier(h_m, channels, block_size=step, method='ols')
frame_start = 0

t0 = time.perf_counter()
index = 0
counter = 0

while counter <= channel_in.shape[0]:
    index += 1
    real_step = randint(40, 80)
    res = fir_online.filter(channel_in[counter: counter + real_step])
    if res is not None:
        real_res[frame_start:frame_start + len(res)] = res
        frame_start += len(res)
    
    counter += real_step
    
t1 = time.perf_counter()

print(f"REAL RES: {(t1-t0)/index}")
t0 = time.perf_counter()
np_res = np.zeros((t, channels))
for i in range(channels):
    np_res[:, i] = convolve(channel_in[:,i], h)[0:t]
t1 = time.perf_counter()
print(f"NP RES: {t1-t0}")

t0 = time.perf_counter()
l_filt = lfilter(h, 1, channel_in, axis=0)
t1 = time.perf_counter()
print(f"Lfilter RES: {t1-t0}")

fig, axs = plt.subplots(2, 1)
top = axs[0]

top.plot(filt_filt[:, 2], label='Filt Filt Signal')
top.plot(sci_res[:, 2], label='SciPy Filtered Signal')
top.plot(fir_off_res[:, 2], label='FIR Offline Signal')
top.plot(mne_res.T[:,2], label='MNE Filtered Signal', linestyle='dashed')
#plt.plot(np.abs(filt_filt[:, 3]-sci_res[:,3]), label='Diff')
top.legend()
top.set_title('Comparison Zero Phase')

bot = axs[1]
bot.plot(real_res[:,0], label='Real Time Filtered')
bot.plot(np_res[:,0], label="NP Convole")
bot.plot(l_filt[:, 0], label='Lfilter')
bot.legend()
bot.set_title('Comparison of Real Time')

plt.figure()
ind = slice(0, 100) 

fft_res = fft(h)[ind]
fft_freq = fftfreq(len(h), d=1/fs)[ind]
plt.plot(fft_freq, np.abs(fft_res), label='Frequence Response MNE')

fft_zero = fft(np.convolve(h, h[::-1], mode='same'))[ind]
fft_zero_freq = fftfreq(len(h), d=1/fs)[ind]
plt.plot(fft_zero_freq, np.abs(fft_zero), label='Zero Frequence Response MNE')

fft_res_ht = fft(h_t)[ind]
fft_freq_ht = fftfreq(len(h_t), d=1/fs)[ind]
plt.plot(fft_freq_ht, np.abs(fft_res_ht), label='Freq Response SCIPY')

plt.legend()
plt.title('Filter FFT')

plt.show()