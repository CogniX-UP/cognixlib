"""
Some good-to-know facts.

1)  MNE's filter function is really slow, for some reason. At least for FIR filters.
2)  scipy's oaconvole is quite faster than MNE's corresponding overlap-add method.
    Furthermore, there is a mode for removing phase lag/delay.
3)  In a real-time BCI setting, there will always be a phase delay. In essence, we
    should always attempt to have minimum phase delay, otherwise things won's work
    correctly.
    
    For BCIs, we should always aim for minimum phase-delay, since it is not possible
    to remove the delay in real-time (to remove it we'd need future values)
4)  We really need to parallelize stuff and not between processes.
"""

if __name__ != '__main__':
    exit()
    
import mne
from FIRconv import FIRfilter
from scipy.signal import oaconvolve
from scipy.fft import fft, fftfreq
from mne.filter import create_filter, _overlap_add_filter
import numpy as np
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

fs = 2048
t = fs * 2
times = 16
channel_in = np.longdouble(35 * np.random.rand(t, times))
print(channel_in.shape)

h = create_filter(
    data=None, 
    sfreq=fs, 
    l_freq=8, 
    h_freq=30,
    filter_length=845,
    l_trans_bandwidth=8,
    h_trans_bandwidth=30
)

t0 = time.perf_counter()
mne_res = _overlap_add_filter(channel_in.T, h, phase="zero",n_jobs=-1)
t1 = time.perf_counter()
print(f"MNE RES: {t1-t0}")

t0 = time.perf_counter()
sci_res = np.zeros((t, times))
for i in range(times):
    sci_res[:,i] = oaconvolve(channel_in[:,i], h, mode="same")
t1 = time.perf_counter()
print(f"SCI RES: {t1-t0}")

step = 512
real_res = np.zeros((t, times))
firs = [FIRfilter(blockSize=step, h=h, normalize=False) for i in range(times)]
frame_start = 0
frame_end = step
t0 = time.perf_counter()
index = 0
while frame_end <= channel_in.shape[0]:
    index += 1
    for i in range(times):
        res = firs[i].process(channel_in[:,i][frame_start:frame_end])
        real_res[:,i][frame_start:frame_end] = res
    frame_start = frame_end
    frame_end += step
    
t1 = time.perf_counter()
print(f"REAL RES: {(t1-t0)/index}")
t0 = time.perf_counter()
np_res = np.convolve(channel_in[:,0], h)[0:t]
t1 = time.perf_counter()
print(f"NP RES: {t1-t0}")
print(mean_squared_error(real_res[:,0], np_res))

plt.figure()
plt.plot(sci_res[:,0], label='SciPy Filtered Signal')
plt.plot(mne_res.T[:,0], label='MNE Filtered Signal', linestyle='dashed')
plt.plot(real_res[:,0], label='Real Time Filtered')
plt.plot(np_res, label="NP Convole")
plt.legend()
plt.title('Comparison of Filtered Signals')

plt.figure()
fft_res = fft(h)
fft_freq = fftfreq(len(h), d=1/fs)

plt.plot(fft_freq, np.abs(fft_res), label='Frequence Response')
plt.title('Filter FFT')

plt.show()