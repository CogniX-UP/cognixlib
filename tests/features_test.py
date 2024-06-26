from cognixlib.scripting.data import *
import numpy as np

labels = ['x1', 'x2', 'x3']
data = np.array(
    [
        [1, 3, 4],
        [-1, 4, 5],
        [2, -1, 4],
        [5, 1, 2],
        [-1, 8, -4],
        [2, 3, 1]
    ]
)

f = LabeledSignal(labels=labels, data=data, signal_info=None)
f_new = f[f > -1]
assert type(f_new) == Signal
t = f[2:5, [1]]
assert len(t.data.shape) == 2
t = f['x2':'x3']

# Filter by a column
rows = np.any(f['x1'] > -1, axis=1)
f_new = f[rows, :]
print(f_new, f_new.labels)

# Filter by a row
cols = np.any(f[[1], :] > -1, axis=0)
f_new = f[:, cols]
print(f_new, f_new.labels)