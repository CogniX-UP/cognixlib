if __name__ != '__main__':
    exit()
    
from cognixlib.scripting.features.fbcsp import FBCSP_Binary
from cognixlib.scripting.data.signals import Signal
import numpy as np
import time

trials = 10
bands = 2
samples = 6000
channels = 16

filt_class_one: list[np.ndarray] = [
    [
        Signal(
            np.random.rand(samples, channels),
            None
        )
        for _ in range(trials)
    ]
    for _ in range(bands)
]

filt_class_two: list[np.ndarray] = [
    [
    Signal(
        np.random.rand(samples, channels),
        None
    )
    for _ in range(trials)
    ]
    for _ in range(bands)
]

trials = {
    'class one': filt_class_one,
    'class two': filt_class_two
}

fbcsp = FBCSP_Binary(
    m=2,
    n_features=2,
    filt_trials=trials
)

t0 = time.perf_counter()
features = fbcsp.fit()
t1 = time.perf_counter()
print(t1-t0)
print('')
print(features.shape, features.labels, features.classes)