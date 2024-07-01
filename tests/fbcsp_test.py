if __name__ != '__main__':
    exit()
    
from cognixlib.scripting.features.fbcsp import FBCSP_Binary
from cognixlib.scripting.data.signals import Signal
from cognixlib.scripting.prediction.scikit.classification import SVMClassifier, LDAClassifier

import numpy as np
import time
import random

num_trials = 10
bands = 10
# variable sample length to test the algorithm
csp_samples_range = (5*1900, 5*2048)
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
    
filt_class_one: list[np.ndarray] = create_filtered_trials(num_trials, bands, channels, csp_samples_range)
filt_class_two: list[np.ndarray] = create_filtered_trials(num_trials, bands, channels, csp_samples_range)

csp_trials = {
    'left': filt_class_one,
    'right': filt_class_two
}

fbcsp = FBCSP_Binary(
    m=2,
    n_features=4,
)

t0 = time.perf_counter()
fbcsp.calc_spat_filts(csp_trials)
t1 = time.perf_counter()
print(f'FBCSP Spatial Filters: {t1-t0} secs\n')

filt_class_one: list[np.ndarray] = create_filtered_trials(40, bands, channels, (1900, 2048))
filt_class_two: list[np.ndarray] = create_filtered_trials(40, bands, channels, (1900, 2048))
feat_trials = {
    'left': filt_class_one,
    'right': filt_class_two 
}

t0 = time.perf_counter()
features = fbcsp.select_features(feat_trials, ['left', 'right'])
t1 = time.perf_counter()
print(f'FBCSP Feature Selection: {t1-t0} secs\n')

t0 = time.perf_counter()
#model = SVMClassifier(probability=True)
model = LDAClassifier()
model.fit(features)
t1 = time.perf_counter()
print(f'FBCSP Train: {t1-t0} secs\n')

test_sig = create_filtered_trials(
    num_trials=1,
    bands=bands,
    channels=channels,
    samples_range=(1900, 2048),
)

t0 = time.perf_counter()
feats = fbcsp.extract_features(test_sig)
t1 = time.perf_counter()
pred = model.predict(feats)
t2 = time.perf_counter()
print(f'FBCSP Predict:\n\tExtract: {t1-t0} sec\n\tEval: {t2-t1} sec\n\tTotal: {t2-t0} sec')
print(pred)
print(model.predict_proba(feats))

# Saving

t0 = time.perf_counter()
j = fbcsp.to_json()
t1 = time.perf_counter()
print(f'To json: {t1-t0} secs')

t0 = time.perf_counter()
fbcsp.from_json(j)
t1 = time.perf_counter()
print(f'From json: {t1-t0} secs')

feats = fbcsp.extract_features(test_sig)
print(model.predict(feats))
