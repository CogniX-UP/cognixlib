if __name__ != '__main__':
    exit()
    
from cognixlib.scripting.data import *
import numpy as np

labels = ['x1', 'x2', 'x3']
data = np.zeros((6, 3), dtype=object)
rnd = np.random.rand(10, 10)
sig = Signal(data=rnd, signal_info=None)

data.fill(sig)

f = LabeledSignal(labels=labels, data=data, signal_info=None)
new_sig = f[1, :]
print(new_sig.data.shape)
new_sig = f[1:2, :]
print(np.array_equal(sig.data, f[1, 2].data.data))