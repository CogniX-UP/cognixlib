if __name__ != '__main__':
    exit()
    
import numpy as np
from random import randint
from cognixlib.scripting.processing.filters import FIROfflineApplier, FIROnlineApplier
from cognixlib.scripting.data.signals import StreamSignal
from mne.filter import create_filter

fs = 2048
# t = int(fs*10800) # 3 hours
t = int(fs*10) # 1 second
channels = 8
channel_in = np.longdouble(35 * np.random.rand(t, channels))
timestamps = np.random.rand(t)

labels = [f'{i}' for i in range(channels)]

signal = StreamSignal(
    timestamps=timestamps,
    labels=labels,
    data=channel_in,
    signal_info=None,
)


h = create_filter(
    data=None, 
    sfreq=fs, 
    l_freq=8, 
    h_freq=30,
    l_trans_bandwidth=8,
    h_trans_bandwidth=8,
    phase='zero-double'
)
h_m = np.tile(h, (channels, 1)).T

step = 1024
real_res = np.zeros((t, channels))
times_res = np.zeros(t)
fir_online = FIROnlineApplier(h_m, channels, block_size=step, method='ols')
frame_start = 0

index = 0
counter = 0

while counter <= channel_in.shape[0]:
    index += 1
    real_step = randint(50, 850)
    res = fir_online.filter(signal[counter: counter + real_step, :])
    
    if res is not None:
        l = len(res.data)
        real_res[frame_start:frame_start + l] = res.data
        times_res[frame_start:frame_start + l] = res.timestamps
        frame_start += l
    
    counter += real_step

print(len(timestamps), len(times_res))
print(timestamps[511], times_res[511])
print(np.where(timestamps != times_res))

print(np.array_equal(timestamps, times_res))