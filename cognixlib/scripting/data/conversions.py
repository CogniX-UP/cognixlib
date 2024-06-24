"""
Defines conversions between primitive types

Used for changing from numpy type to lsl type
"""

import numpy as np
from numpy import dtypes
import pylsl
from types import MappingProxyType

__str_to_np = {
    'float32': np.dtype(np.float32),
    'float64': np.dtype(np.float64),
    'int8': np.dtype(np.int8),
    'int16': np.dtype(np.int16),
    'int32': np.dtype(np.int32),
    'int64': np.dtype(np.int64),
    'string': np.dtype(np.string_),    
}

str_to_np = MappingProxyType(__str_to_np)

__np_to_lsl = {
    np.dtype(np.float32): pylsl.cf_float32,
    np.dtype(np.float64): pylsl.cf_double64,
    np.dtype(np.int8): pylsl.cf_int8,
    np.dtype(np.int16): pylsl.cf_int16,
    np.dtype(np.int32): pylsl.cf_int32,
    np.dtype(np.int64): pylsl.cf_int64,
    np.dtype(np.string_): pylsl.cf_string,
}

np_to_lsl = MappingProxyType(__np_to_lsl)

__lsl_to_np = {
    pylsl.cf_float32: np.dtype(np.float32),
    pylsl.cf_double64: np.dtype(np.float64),
    pylsl.cf_int8: np.dtype(np.int8),
    pylsl.cf_int16: np.dtype(np.int16),
    pylsl.cf_int32: np.dtype(np.int32),
    pylsl.cf_int64: np.dtype(np.int64),
    pylsl.cf_string: np.dtype(np.string_),
}

lsl_to_np = MappingProxyType(__lsl_to_np)

___str_to_lsl = {
    'float32': pylsl.cf_float32,
    'float64': pylsl.cf_double64,
    'int8': pylsl.cf_int8,
    'int16': pylsl.cf_int16,
    'int32': pylsl.cf_int32,
    'int64': pylsl.cf_int64,
    'string': pylsl.cf_string,
}

str_to_lsl = MappingProxyType(___str_to_lsl)

__lsl_to_str = {
    pylsl.cf_float32: 'float32',
    pylsl.cf_double64: 'float64',
    pylsl.cf_int8: 'int8',
    pylsl.cf_int16: 'int16',
    pylsl.cf_int32: 'int32',
    pylsl.cf_int64: 'int64',
    pylsl.cf_string: 'string',
}

lsl_to_str = MappingProxyType(__lsl_to_str)

def get_lsl_format(key: np.dtype | str | int):
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return ___str_to_lsl[key]
    elif isinstance(key, np.dtype):
        return __np_to_lsl[key]