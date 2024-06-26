from cognixlib.scripting.data import *
import numpy as np
import time

def run():
    
    # LABELS TEST
    size = 5
    # x0 until x4
    labels = [f"x{i}" for i in range(size)]
    data = np.random.rand(3, size)
    a = LabeledSignal(labels, data, None)
    
    assert np.array_equal(
        a['x1':'x3'].data, 
        a[:, 1:4].data
    ) # x3 is inclusive
    assert np.array_equal(
        a['x2'].data, 
        a[:, [2]].data
    )
    assert np.array_equal(
        a[['x1', 'x2', 'x4']].data, 
        a[:, [1, 2, 4]].data
    )
    

    # STREAM TEST - MUST BE USED IN LSL
    timestamps = np.random.rand(size)
    
    b = StreamSignal(timestamps, labels, data, None)
    
    # CLASS TEST - MUST BE USED FOR TRAINING
    # same labels as above, x0 until x4
    data = np.random.rand(25, size)
    classes = {
        "george": (0, 14), # EXCLUSIVE
        "john": (14, 25)
    }
    f1 = FeatureSignal(
        labels,
        classes,
        data,
        None
    )
    
    assert np.array_equal(
        f1["george"].data, 
        f1[0:14].data
    )
    assert np.array_equal(
        f1["john"].data, 
        f1[14:25].data
    )
    
    # CLASS MERGE TEST
    data = np.random.rand(30, size)
    classes = {
        "john": (10, 20),
        "george": (0, 10),
        "dam": (20, 30)
    }
    f2 = FeatureSignal(
        labels,
        classes,
        data,
        None,
    )
    
    f_signal = FeatureSignal.concat_classes(f1, f2)

    assert np.array_equal(
        f_signal['george'].data[0:14],
        f1['george'].data
    )
    assert np.array_equal(
        f_signal['john'].data[11:21],
        f2['john'].data
    )
    
    # REMOVE TESTs
    
    # Class remove
    t0 = time.perf_counter()
    new_sig = f2[5:15, :]
    t1 = time.perf_counter()
    print(t1-t0)
    assert np.array_equal(
        f2['john'].data[0:5, :],
        new_sig['john'].data
    )
    
    mask = np.ones(f2.data.shape, dtype=np.bool_)
    mask[15:23] = False
    mask[:, 2:4] = False
    new_sig = f2[mask]
    test_np = f2['dam'].data[3:, [0, 1, 4]]

    print(new_sig.classes)
    assert np.array_equal(
        test_np,
        new_sig['dam'].data
    )
    
    # test for removing class
    new_data = np.random.rand(100000, 5)
    t0 = time.perf_counter()
    labels = ['A', 'B', 'C', 'D', 'E']
    classes = {
        "k0": (0, 10000),
        "k1": (10000, 20000),
        "k2": (20000, 30000),
        "k3": (30000, 40000),
        "k4": (40000, 50000),
        "k5": (50000, 60000),
        "k6": (60000, 70000),
        "k7": (70000, 80000),
        "k8": (80000, 90000),
        "k9": (90000, 100000)
    }
    new_sig = FeatureSignal(
        labels,
        classes,
        new_data,
        None
    )
    
    test_array = [i for i in range(0, 100000, 20)]
    t1 = time.perf_counter()
    test_sig = new_sig[test_array, :]
    t2 = time.perf_counter()
    
    assert np.array_equal(
        new_sig['k4'][np.arange(0, 10000, 20), :].data,
        test_sig['k4'].data
    )
    print(f"WITHOUT ROWS: {(t2-t1)}")
    test_slice = slice(49500, 76542)
    test_sig = new_sig[test_slice, :]
    assert np.array_equal(
        new_sig['k7'][0:6542, :].data,
        test_sig['k7'].data,
    )
    
    t = new_sig['B':'D']['k3':'k6']
    print(t, type(t), t.classes, t.labels)
    
if __name__ == '__main__':
    run()