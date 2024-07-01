if __name__ != '__main__':
    exit()
    
from cognixlib.scripting.features.fbcsp import FBCSP_Binary
from cognixlib.scripting.data.signals import Signal
from cognixlib.scripting.prediction.scikit.classification import SVMClassifier

import numpy as np
import time
import random

num_trials = 10
bands = 1
# variable sample length to test the algorithm
samples_range = (5600, 6000)
channels = 16

def create_filtered_trials(num_trials, bands, channels, samples_range) -> list[np.ndarray]:
    return [
        [
            Signal(
                np.random.rand(random.randint(samples_range[0], samples_range[1]), channels),
                None
            )
            for _ in range(num_trials)
        ]
        for _ in range(bands)
    ]
    
filt_class_one: list[np.ndarray] = create_filtered_trials(num_trials, bands, channels, samples_range)

filt_class_two: list[np.ndarray] = create_filtered_trials(num_trials, bands, channels, samples_range)

trials = {
    'class one': filt_class_one,
    'class two': filt_class_two
}

fbcsp = FBCSP_Binary(
    m=4,
    n_features=4,
    filt_trials=trials
)

t0 = time.perf_counter()
features = fbcsp.fit(class_labels=['left', 'right'])
t1 = time.perf_counter()

print(f'FBCSP Fit: {t1-t0} secs')
print(features.shape, features.labels, features.classes)

t0 = time.perf_counter()
svm = SVMClassifier()
svm.fit(features)
t1 = time.perf_counter()
print(f'FBCSP Train: {t1-t0} secs')

test_sig = create_filtered_trials(
    num_trials=1,
    bands=bands,
    channels=channels,
    samples_range=samples_range,
)

t0 = time.perf_counter()
feats = fbcsp.extract_features(test_sig)
t1 = time.perf_counter()
pred = svm.predict(feats)
t2 = time.perf_counter()
print(f'FBCS:\nExtract: {t1-t0} sec\nEval: {t2-t1} sec\nTotal: {t2-t0} sec')
print(pred)