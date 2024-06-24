
if __name__ != '__main__':
    exit()

import numpy as np
import pandas as pd
import time
import cognixlib.api.data as cl
t0 = time.perf_counter()
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])
columns = ['A', 'B', 'C']

t1 = time.perf_counter()
# Create a DataFrame with the NumPy array and additional metadata
df = pd.DataFrame(data, columns=columns)
df['Metadata'] = ['meta1', 'meta2', 'meta2', 'meta3'] # this breaks contiguous stuff
t2 = time.perf_counter()

c_dict = {
    'meta1': (0, 1),
    'meta2': (1, 3),
    'meta3': (3, 4)
}
f = cl.FeatureSignal(
    columns,
    c_dict,
    data,
    None,
)

t3 = time.perf_counter()

print(f'Data Creation: {t1-t0}')
print(f'Pandas Creation: {t2-t1}')
print(f'Feature Creation: {t3-t2}')

check = np.array_equal(
    df['A'],
    f.ldm['A'].data
)
print(check)
print(df.values.flags['C_CONTIGUOUS'])