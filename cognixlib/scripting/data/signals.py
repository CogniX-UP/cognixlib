"""
Defines the core functionalities and data types for :code:`cognixlib`

The Signal classes are special classes that wrap numpy arrays with additional
functionality and potentially metadata. Each Signal object might have its own
restrictions over dimensionality and metadata. The basic :class:`Signal` object
has no restrictions.

The other built-in signal objects, namely :class:`TimeSignal`, :class:`LabeledSignal`,
:class:`FeatureSignal`, :class:`StreamSignal` typically operate as 2D arrays (with 
arbitrary data inside). Any operation that may reduce the matrix to 1D will also reduce
the signal type to the generic :class:`Signal`. The same applies to boolean masking.
This happens because the above classes' metadata relies heavily on the assumption 
that the array/matrix is 2D.

Typically, in numpy, the only way to keep the structure of an array is by using a tuple
of slices for any extraction operation. 
"""

from __future__ import annotations
from collections.abc import Sequence, Mapping
from .mixin import *
from sys import maxsize
from itertools import chain
from copy import copy, deepcopy
from beartype.door import is_bearable
import numpy as np

from .conversions import *

class SignalKey:
    """
    The numpy nature of the signal classes make them inherently
    unhasable. This is a python object that every signal
    instance creates that acts as its unique identifier.
    """
    
    def __init__(self, sig: Signal):
        self._signal = sig
    
    @property
    def signal(self):
        return self._signal
    
class SignalInfo:
    """
    A signal info carries metadata about the signal
    
    At its base form, it's a simple class with **kwargs for storing
    information. The **kwargs are stored as attribute to the instance
    of the class.
    
    """
    
    def __init__(
        self,
        **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
class Signal:
    """
    Represents the data for signal processing
    
    The data is a numpy array of any shape. What the shape describes 
    is left to the metadata. Some nodes may require a signal of 
    specific type, shape, etc.
    
    i.e.
    In EEG, a 2x2 shape might represent samples x channels.
    In image processing,a 2x2 shape typically represents an image.
    """
    
    @classmethod
    def concat(*signals: Signal, axis=0):
        if len(signals) == 1:
            return signals[0]
        
        datas = [
            sig.data for sig in signals
        ]
        
        data_comb = np.concatenate(datas, axis)
        
        return Signal(
            data_comb,
            signals[0].info
        )
        
    def __init__(
        self,  
        data: np.ndarray,
        signal_info: SignalInfo 
    ):
        self._data = data
        self._info = signal_info
        self._unique_key = SignalKey(self)
    
    @property
    def unique_key(self) -> SignalKey:
        """A unique identifier for this signal instance"""
        return self._unique_key
    
    @property
    def info(self) -> SignalInfo:
        """Metadata information regarding the signal"""
        return self._info
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
    
    def __str__(self):
        return str(self.data)
    
    def __getitem__(self, key):
        return Signal(self._new_data(key), self.info)
    
    def __setitem__(self, key, newvalue):
        self.data[key] = newvalue
    
    def __add__(self, other):
        add = self._extract_data(other)
        return Signal(self.data + add, self.info)
    
    def __sub__(self, other):
        sub = self._extract_data(other)
        return Signal(self.data - sub, self.info)
    
    def __eq__(self, other):
        return self.data == self._extract_data(other)
    
    def __ne__(self, other):
        return self.data != self._extract_data(other)
    
    def __lt__(self, other):
        return self.data < self._extract_data(other)
    
    def __le__(self, other):
        return self.data <= self._extract_data(other)
    
    def __gt__(self, other):
        return self.data > self._extract_data(other)
    
    def __ge__(self, other):
        return self.data >= self._extract_data(other)
    
    def _extract_data(self, other):
        return other.data if isinstance(other, Signal) else other
    
    def copy(self, copydata=True):
        """
        Shallow copy of the signal
        
        Returns a copy of the signal. If :code:`copydata` is True,
        then the data is copied.
        """
        new_sig = copy(self)
        new_sig._unique_key = SignalKey(self)
        if copydata:
            new_sig._data = new_sig.data.copy()
        return new_sig
    
    def deepcopy(self):
        """Returns a deep copy of the signal"""
        return deepcopy(self)  
    
    def _new_data(self, key):
        """
        Generates the new data from a key, based on whether the key is
        a mask or anything else
        """
        if isinstance(key, np.ndarray) and key.dtype==np.bool_:
            rows = np.any(key, axis=1)
            cols = np.any(key, axis=0)
            new_data = self.data[rows, :]
            new_data = self.data[:, cols]
        else:
            new_data = self.data[key]
        return new_data
            
    def _check_reduction(self, key) -> Signal | None:
        """
        Checks to see if the signal must be reduced to a generic :class:`signal` 
        rather than keep its current type.
        
        Casting down to a generic signal will occur if the key is a boolean key
        of different shape than the original data or if the key is not of type
        :code:`tuple[slice | Sequence[float], slice | Sequence[float]]
        
        Only then will the signal keep the 2D aspect.
        """
        if (
            isinstance(key, np.ndarray) and
            key.dtype == np.bool_ and
            key.shape == self.data.shape
        ):
            return None
        
        if (
            isinstance(key, tuple) and
            isinstance(key[0], (slice, Sequence, np.ndarray)) and 
            isinstance(key[1], (slice, Sequence, np.ndarray))
        ):
            return None
        
        return Signal(self.data[key], None)
 
class TimeSignal(Signal, Timestamped):
    """
    Represents signal data with additional timestamps per sample
    
    The timestamps should map to the first dimension of the signal.
    
    i.e. A video stream, a single channel of EEG
    """
    
    @classmethod
    def concat(*signals: TimeSignal, axis=0):
        """Concatenates multiple :code:`TimeSignal` (s) together."""
        if len(signals) == 1:
            return signals[0]
        
        datas = [
            sig.data for sig in signals
        ]
        timetables = [
            sig.tms for sig in signals
        ]
        
        if axis==0:
            data_conc = np.concatenate(datas, 0)
            tms_concat = np.concatenate(timetables)
        else:
            for i in range(len(timetables) - 1):
                assert np.array_equal(timetables[i], timetables[i+1]), "Timestamp array must be the same accross all signals"
            
            for i in range(len(datas) - 1):
                assert datas[i].shape == datas[i+1].shape, "Datas must share the same shape"
            
            data_conc = np.concatenate(datas, 1)
            tms_concat = timetables[0]
        
        return TimeSignal(
            tms_concat,
            data_conc,
            signals[0].info
        )
        
    def __init__(self, timestamps: Sequence[float], data: np.ndarray, signal_info: SignalInfo):
        Signal.__init__(self, data, signal_info)
        Timestamped.__init__(self, timestamps)
        
    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps
    
    @timestamps.setter
    def timestamps(self, value: np.ndarray):
        self._timestamps = value
    
    def copy(self, copydata=True) -> TimeSignal:
        new_sig = super().copy(copydata)
        return new_sig

    def __getitem__(self, key):
        
        reduce_check = self._check_reduction(key)
        if reduce_check is not None:
            return reduce_check
        
        # At this point, both are slices
        return TimeSignal(
            self._extract_timestamps(key),
            self._new_data(key),
            self.info
        )
    
    def _extract_timestamps(self, key: tuple[slice, slice] | np.ndarray):
        """Extracts the timestamps when the key is a tuple"""
        if isinstance(key, np.ndarray):
            rows = np.any(key, axis=1)
        else:
            rows, _ = key
        return self.timestamps[rows]

class LabeledSignal(Signal, Labeled):
    """
    Represents signal data that is mapped to specific labels.
    """
    
    @classmethod
    def concat(self, *signals: LabeledSignal, axis=0):
        """
        Concatenates multiple :code:`LabeledSignal` (s). For a vertical 
        concatentation, the labels must be the same between the signals.
        """
        if len(signals) == 1:
            return signals[0]
        
        datas = [
            sig.data for sig in signals
        ]
        label_lists = [
            sig.labels for sig in signals
        ]
        
        if axis==0:
            data_con = np.concatenate(datas, axis=0)
            label_conc = list(chain(*label_lists))
        else:
            for i in range(len(label_lists) - 1):
                assert label_lists[i] == label_lists[i+1], "Labels must be the same for vertical concat!"
            data_con = np.concatenate(datas, axis=1)
            label_conc = label_lists[0]
        
        return LabeledSignal(
            label_conc,
            data_con,
            signals[0].info
        )
    
    def __init__(
        self,
        labels: Sequence[str], 
        data: np.ndarray, 
        signal_info: SignalInfo,
        make_lowercase=False,
    ):
        Signal.__init__(self, data, signal_info)
        Labeled.__init__(self, labels, make_lowercase)
        if isinstance(labels, np.ndarray):
            labels = labels.flat
        self._label_to_index: dict[str, int] = {
            label:index for index, label in enumerate(labels)
        }
        
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, value: np.ndarray):
        self._labels = value
    
    @property
    def label_indices(self) -> Mapping[str, int]:
        return self._label_to_index
    
    def label_index(self, label: str) -> int:
        if not label in self._label_to_index:
            return -1
        return self._label_to_index[label]
    
    def __getitem__(self, key):
        # Handle the cases of strings first
        label_check = self._check_labels(key)
        if label_check is not None:
            return label_check
        
        # dim reduce check
        reduce_check = self._check_reduction(key)
        if reduce_check is not None:
            return reduce_check
        
        return LabeledSignal(
            self._extract_labels(key),
            self._new_data(key),
            self.info
        )
    
    def _check_labels(self, key) -> LabeledSignal | None:
        """Checks if the key includes any kind of string handling."""
        if isinstance(key, str):
            index = self.label_index(key)
            labels = np.array(self.labels[index])
            return LabeledSignal(
                labels,
                self.data[:, index:index+1],
                self.info
            )
        elif is_bearable(key, Sequence[str]):
            indices = [
                self.label_index(label) 
                for label in key
                if self.label_index(label) > 0
            ]
            stop, start = indices[-1], indices[0]
            consecutive = len(indices) == (stop - start + 1)
            if consecutive:
                indices = slice(start, stop + 1)
            return LabeledSignal(
                self.labels[indices],
                self.data[:, indices],
                self.info
            )
        elif (
            isinstance(key, slice) and
            isinstance(key.start, str) and
            isinstance(key.stop, str)
        ):
            start = self.label_index(key.start)
            stop = self.label_index(key.stop)
            if start < stop:
                return LabeledSignal(
                    self.labels[start:stop + 1],
                    self.data[:, start:stop + 1],
                    self.info
                )
                
        return None
    
    def _extract_labels(self, key: tuple[slice, slice] | np.ndarray):
        """Extracts the subset of labels"""
        # this is meant to work with a mask of the same shape with the 
        # data
        if isinstance(key, np.ndarray):
            cols = np.any(key, axis=0)
        else:
            _, cols = key
        return self.labels[cols]
    
class StreamSignalInfo(SignalInfo, StreamConfig):
    """Information regard a Stream Signal"""
    def __init__(
        self,
        nominal_srate: int, 
        signal_type: str, 
        data_format: str, 
        name: str, 
        **kwargs
    ):
        SignalInfo.__init__(self, **kwargs)
        StreamConfig.__init__(
            self,
            nominal_srate,
            signal_type,
            data_format,
            name 
        )
    
    @property
    def data_format_np(self):
        """Returns the data format of this info, based on numpy convention."""
        return str_to_np[self.data_format]
    
    @property
    def data_format_lsl(self):
        """Returns the data format of this info, based on LSL convention."""
        return lsl_to_np[self.data_format]
        
class StreamSignal(TimeSignal, LabeledSignal):
    """
    Represents time signal data with additional channels / labels.
    
    In this context, a timestamp doesn't correspond to a singular unit
    of the signal, but rather multitle samples from different sources
    (devices) recorded at the same time.
    
    i.e. multiple video streams, a full EEG configuration (all channels)
    
    This object is essentially a superset of the TimeSignal class and
    will most likely be used in the majority of cases, even if there was
    only one channel / label.
    """
    
    def __init__(
        self, 
        timestamps: Sequence[float], 
        labels: Sequence[str],
        data: np.ndarray, 
        signal_info: StreamSignalInfo,
        make_lowercase=False
    ):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        TimeSignal.__init__(self, timestamps, data, None)
        LabeledSignal.__init__(self, labels, data, None, make_lowercase)
        self._data = data
        self._info = signal_info
    
    @property
    def info(self) -> StreamSignalInfo:
        return self._info

    def __getitem__(self, key):
        # label check, rows are not affected / timestamps
        label_check = self._check_labels(key)
        if label_check is not None:
            return StreamSignal(
                self.timestamps,
                label_check.labels,
                label_check.data,
                self.info
            )
        
        # reduce check
        reduce_check = self._check_reduction(key)
        if reduce_check is not None:
            return reduce_check
        
        new_timestamps = self._extract_timestamps(key)
        new_labels = self._extract_labels(key)
        new_data = self._new_data(key)
        
        return StreamSignal(
            new_timestamps,
            new_labels,
            new_data,
            self.info
        )

class FeatureSignal(LabeledSignal):
    """
    Represents a signal whose rows correspond to a feature class
    and whose columns correspond to a feature label. Can be used
    for machine learning purposes.
    """
    
    @classmethod
    def concat_classes(self, *signals: FeatureSignal):
        """
        Concatenates two Feature Signals based on their classes.
        The resulting signal will have the data reordered in such
        a way that a class has all its data sequentially in memory.
        
        This means that the features / labels of the signals must
        be the same, both in length and in order.
        """
        
        if len(signals) == 1:
            return signals[0]
        
        label_arrs = [signal.labels for signal in signals]
        for i in range(len(label_arrs)-1):
            assert np.array_equal(label_arrs[i], label_arrs[i+1]), "Signal labels are not the same!"
        
        # ideally, this should be implemented in cython
        # or as an external library
        
        # extract all the available classes
        class_labels = {
            class_label
            for signal in signals
            for class_label in signal.classes
        }
        
        # We scan the data through its labels and add
        # whatever data we find to this list. In the
        # end, we'll have the data we want sorted by
        # the labels in sequence
        class_sorted_datas: list[np.ndarray] = []
        
        # the index dictionary for the classes
        class_index_dict: dict[str, tuple[int, int]] = {}
        offset = 0
        for class_label in class_labels:
            
            # final length of the current label
            label_data_len = 0
            for signal in signals:
                if class_label in signal.classes:
                    label_data = signal[class_label].data
                    class_sorted_datas.append(label_data)
                    label_data_len += label_data.shape[0]
            
            class_index_dict[class_label] = (offset, offset + label_data_len)
            offset += label_data_len
        
        classes_data = np.concatenate(class_sorted_datas, axis=0)
        return FeatureSignal(
            signals[0].labels,
            class_index_dict,
            classes_data,
            None
        )
            
    def __init__(
        self, 
        labels: Sequence[str],
        classes: dict[str, tuple[int, int]],
        data: np.ndarray, 
        signal_info: SignalInfo,
        sort=True,
    ):
        super().__init__(labels, data, signal_info)
        self.classes = classes
        self._build_succession(sort)
    
    def _build_succession(self, sort=True):
        # Optimized
        if sort:
            self.classes = dict(
                sorted(
                    self.classes.items(), 
                    key=lambda item:item[1][1] # sort by value and by the end of the indices
                )
            )
        
        self._succ_classes_list = list(self.classes.keys())
        
    def __getitem__(self, key):
        
        # check for label specific feature extraction
        label_check = self._check_labels(key)
        if label_check is not None:
            return label_check
        
        # check for class specific feature extraction
        class_check = self._check_classes(key)
        if class_check is not None:
            return class_check
        
        # reduce check
        reduce_check = self._check_reduction(key)
        if reduce_check is not None:
            return reduce_check
        
        new_labels = self._extract_labels(key)
        if isinstance(key, np.ndarray) and key.dtype==np.bool_:
            rows = np.any(key, axis=1)
            cols = np.any(key, axis=0)
            new_data = self.data[rows, :]
            new_data = self.data[:, cols]
            new_classes = self._extract_classes(rows)
        else:
            new_classes = self._extract_classes_by_key(key)
            new_data = self.data[key]
            
        return FeatureSignal(
            new_labels,
            new_classes,
            new_data,
            self.info
        )
    
    def _check_labels(self, key) -> FeatureSignal | None:
        if self._check_key_names(key, self.labels):
            label_check = super()._check_labels(key)
            
            # no rows are affected by this
            if label_check is not None:
                return FeatureSignal(
                    label_check.labels,
                    self.classes,
                    label_check.data,
                    self.info,
                    False
                )
                
        return None
    
    def _check_classes(self, key) -> FeatureSignal | None:
        if self._check_key_names(key, self._succ_classes_list):
            
            if isinstance(key, str):
                start, stop = self.classes[key]
                klass = self._extract_classes(slice(start, stop))
                return FeatureSignal(
                    self.labels,
                    klass,
                    self.data[start:stop, :],
                    self.info,
                    False,
                )
            elif is_bearable(key, Sequence[str]):
                indices = [
                    self.classes[label] 
                    for label in key
                ]
                _, stop = indices[-1]
                start, _ = indices[0]
                consecutive = len(indices) == (stop - start)
                if consecutive:
                    indices = slice(start, stop)
                classes = self._extract_classes(indices)
                return FeatureSignal(
                    self.labels,
                    classes,
                    self.data[indices, :],
                    self.info,
                    False,
                )
            elif (
                isinstance(key, slice) and
                isinstance(key.start, str) and
                isinstance(key.stop, str)
            ):
                start, _ = self.classes[key.start]
                _, stop = self.classes[key.stop]
                if start < stop:
                    s = slice(start, stop)
                    return FeatureSignal(
                        self.labels,
                        self._extract_classes(s),
                        self.data[s, :],
                        self.info,
                        False
                    )
                
        return None
    
    def _check_key_names(self, key, check_list: Sequence[str]):
        """
        Checks if the key is related to the corresponding list
        Only checks the first element if the key is a list.
        """
        return (
            (
                isinstance(key, str) and
                key in check_list
            ) or 
            (
                is_bearable(key, Sequence[str]) and
                key[0] in check_list        
            ) or
            (
                isinstance(key, slice) and
                isinstance(key.start, str) and
                isinstance(key.stop, str) and
                key.start in check_list and
                key.stop in check_list
            )
        )
    
    def _extract_classes_by_key(self, key: tuple | np.ndarray):
        if isinstance(key, np.ndarray):
            rows = np.any(key, axis=1)
        else:
            rows, _ = key
        return self._extract_classes(rows)
        
    def _extract_classes(self, rows: int | Sequence[int] | slice | np.ndarray):
        # The costs of this function are complementary. When deleting a small subsection,
        # the deletion construction of the new array takes time. When deleting a large
        # subsection, the reconstruction of the classes hierarchy takes time.
        
        if isinstance(rows, int):
            rows = [rows]
        
        if isinstance(rows, slice):
            start = rows.start if rows.start else 0
            end = rows.stop if rows.stop else len(self.data)
            step = rows.step if rows.step else 1
            
            rows_to_include = np.arange(start, end, step)
            
        elif isinstance(rows, Sequence):
            rows_to_include = np.array(rows, dtype=np.int64)
            rows_to_include.sort()
        elif isinstance(rows, np.ndarray):
            if rows.dtype == np.bool_:
                rows_to_include = np.where(rows)[0]
            else:
                rows_to_include = rows
            
        min_include_index = rows_to_include[0]
        max_include_index = rows_to_include[-1]
            
        classes = self.classes
        class_include_count: dict[str, int] = {}
        min_class_index = maxsize
        max_class_index = -1
        succ_classes_list = self._succ_classes_list
        
        
        # The rows to remove are sorted here
        # The classes are also sorted
        len_inc = len(rows_to_include)
        last_found_index = 0
        # will always be true currently
        is_numpy = isinstance(rows_to_include, np.ndarray)
        
        inc_start = rows_to_include[0]
        inc_end = rows_to_include[-1] + 1
        search_space = inc_end - inc_start
        
        for klass_index, klass in enumerate(succ_classes_list):
            start, end = classes[klass]
            min_class_index = min(min_class_index, klass_index)
            max_class_index = max(max_class_index, klass_index)
            
            # avoid redundant checks
            if end < min_include_index:
                continue
            
            if len_inc == search_space:
                include_count = min(end, inc_end)-max(start, inc_start)
                if include_count > 0:
                    class_include_count[klass] = include_count
            elif is_numpy:
                result = np.where(
                    np.logical_and(
                        start <= rows_to_include,
                        rows_to_include < end
                    )
                )
                
                if result:
                    arr, = result
                    class_include_count[klass] = len(arr)
            
            # this will never be used currently
            else:    
                for i in range(last_found_index, len_inc):
                    r = rows_to_include[i]
                    if start <= r and r < end:
                        last_found_index = i
                        if klass not in class_include_count:
                            class_include_count[klass] = 0
                        class_include_count[klass] += 1
            
            # early exit after we processed the segment
            if end >= max_include_index:
                break
        
        new_classes = {}
        offset = 0
        
        for klass, inc_count in class_include_count.items():
            if inc_count <= 0:
                continue
            new_classes[klass] = (offset, offset + inc_count)
            offset += inc_count
        
        return new_classes
    