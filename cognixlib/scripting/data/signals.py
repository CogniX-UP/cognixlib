"""Defines the core functionalities and data types for Cognix"""
from __future__ import annotations
from collections.abc import Sequence
from .mixin import *
from sys import maxsize
from itertools import chain
from copy import copy, deepcopy
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
    def unique_key(self):
        """A unique identifier for this signal instance"""
        return self._unique_key
    
    @property
    def info(self):
        """Metadata information regarding the signal"""
        return self._info
    
    @property
    def data(self):
        return self._data
    
    def __str__(self):
        return str(self.data)
    
    def __getitem__(self, key):
        return Signal(self.data[key], self.info)
    
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
    
    def _extract_data(other):
        return other.data if isinstance(other, Signal) else other
    
    def copy(self, copydata=True):
        new_sig = copy(self)
        new_sig._unique_key = SignalKey(self)
        if copydata:
            new_sig._data = new_sig.data.copy()
        return new_sig
    
    def deepcopy(self):
        return deepcopy(self)
    
    def filter_rows(self, condition: np.ndarray):
        """
        Filters the rows based on a numpy condition
        
        If the condition is not met, the whole row will be removed
        """
        rows_to_keep = np.where(np.all(condition, axis=1))[0]
        new_data = self.data[rows_to_keep, :]
        return Signal(new_data, self.info)
    
    def filter_columns(self, condition: np.ndarray):
        """
        Filters the columns based on a numpy condition
        
        If the condition is not met, the whole column will be removed
        """
        cols_to_keep = np.where(np.all(condition, axis=0))[0]
        new_data = self.data[:, cols_to_keep]
        return Signal(new_data, self.info)           

class TimeSignal(Signal, Timestamped):
    """
    Represents signal data with additional timestamps per sample
    
    The timestamps should map to the first dimension of the signal.
    
    i.e. A video stream, a single channel of EEG
    """
    
    @classmethod
    def concat(*signals: TimeSignal, axis=0):
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
                
    class DataMap:
        """
        Maps from time indices to data indices
        
        Since timestamps are row-related, there is no
        conversion between time and data indices.
        """
        def __init__(self, data: np.ndarray, timestamps: Sequence[float], info: SignalInfo):
            self.data = data
            self.timestamps = timestamps
            self.info = info
        
        def __str__(self):
            return str(self.data)
        
        def __getitem__(self, key):
            sub_data = self.data[key]
            sub_times = self.timestamps[key]
            return TimeSignal.DataMap(sub_data, sub_times, self.info)
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def signal(self):
            """Converts the DataMap back to a TimeSignal"""
            return TimeSignal(self.data, self.timestamps, self.info)
        
    def __init__(self, timestamps: Sequence[float], data: np.ndarray, signal_info: SignalInfo):
        Signal.__init__(self, data, signal_info)
        Timestamped.__init__(self, timestamps)
        self._time_datamap = TimeSignal.DataMap(data, timestamps, signal_info)
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        self._time_datamap.data = value
        
    @property
    def timestamps(self):
        return self._timestamps
    
    @timestamps.setter
    def timestamps(self, value: np.ndarray):
        self._timestamps = value
        self._time_datamap.timestamps = value
        
    @property
    def time_datamap(self):
        """Retrieves an object with mapped indices from timestamps to data"""
        return self._time_datamap
    
    @property
    def tdm(self):
        """Shorthand for time_datamap"""
        return self.time_datamap
    
    def copy(self, copydata=True):
        new_sig = super().copy(copydata)
        new_sig._time_datamap = TimeSignal.DataMap(
            new_sig.data, 
            new_sig.timestamps, 
            new_sig.info
        )
        return new_sig

    def __getitem__(self, key):
        return TimeSignal(
            self._extract_timestamps(key),
            self.data[key],
            self.info
        )
    
    def _extract_timestamps(self, key):
        if isinstance(key, (slice, Sequence, np.ndarray, int)):
            new_times = self.timestamps[key]
        elif isinstance(key, tuple):
            left, _ = key
            new_times = self.timestamps[left]
        else:
            new_times = self.timestamps
        return new_times

class LabeledSignal(Signal, Labeled):
    """
    Represents signal data that is mapped to specific labels.
    """
    
    @classmethod
    def concat(self, *signals: LabeledSignal, axis=0):
        
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
        
    class DataMap:
        """
        Maps from label indices to data indices
        
        Since labels are column-related, a conversion must be made
        between label indices and data indices.
        """
        def __init__(self, data: np.ndarray, labels: np.ndarray, info: SignalInfo):
            self.data = data
            self.info = info
            self.labels = labels
        
        @property
        def labels(self):
            return self._labels
        
        @labels.setter
        def labels(self, value: np.ndarray):
            self._labels = value
            self._label_to_index: dict[str, int] = {
                label:index for index, label in enumerate(value.flat)
            }
            
        def __str__(self):
            return f"{self.labels}\n{self.data}"
        
        def __getitem__(self, key):
            old_key = key
            key = self.convert_to_indices(key)
            sub_data = self.data[:, key]
            if isinstance(key, list):
                sub_labels = np.array(old_key)
            else:
                sub_labels = self.labels[key]
            return LabeledSignal.DataMap(sub_data, sub_labels, self.info)
        
        def __setitem__(self, key, value):
            key = self.convert_to_indices(key)
            self.data[:, key] = value
        
        def signal(self):
            """Converts the DataMap back to a LabeledSignal"""
            return LabeledSignal(self.labels, self.data, self.info)
        
        def convert_to_indices(self, key):
            result = key
            if isinstance(key, str):
                result = self._label_to_index[key]
            elif isinstance(key, list):
                result = []
                for k in key:
                    result.append(self._label_to_index[k])
            elif isinstance(key, slice):
                start = (
                    self._label_to_index[key.start] 
                    if isinstance(key.start, str)
                    else key.start
                )
                stop = (
                    self._label_to_index[key.stop]
                    if isinstance(key.stop, str)
                    else key.stop
                )
                stop += 1
                stop = min(len(self.labels), stop)
                result = slice(start, stop)
            return result
    
    def __init__(
        self,
        labels: Sequence[str], 
        data: np.ndarray, 
        signal_info: SignalInfo,
        make_lowercase=False,
    ):
        Signal.__init__(self, data, signal_info)
        Labeled.__init__(self, labels, make_lowercase)
        self._label_datamap = LabeledSignal.DataMap(data, self.labels, signal_info)

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        self._label_datamap.data = value
        
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, value: np.ndarray):
        self._labels = value
        self._label_datamap.labels = value
    
    @property
    def label_datamap(self):
        """Retrieves an object with mapped indices from labels to data"""
        return self._label_datamap

    @property
    def ldm(self):
        """Shorthand for label_datamap"""
        return self.label_datamap
    
    def copy(self, copydata=True):
        new_sig = super().copy(copydata)
        new_sig._label_datamap = LabeledSignal.DataMap(
            new_sig.data,
            new_sig.labels,
            new_sig.info
        )
        return new_sig
    
    def __getitem__(self, key):
        return LabeledSignal(
            self._extract_labels(key),
            self.data[key],
            self.info
        )
    
    def _extract_labels(self, key):
        if isinstance(key, tuple):
            _, right = key
            if isinstance(right, (slice, Sequence, np.ndarray, int)):
                new_labels = self.labels[right]
        else:
            new_labels = self.labels
        
        return new_labels
    
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
        return str_to_np[self.data_format]
    
    @property
    def data_format_lsl(self):
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
    def info(self):
        return self._info
    
    def copy(self, copydata=True):
        new_sig: StreamSignal = Signal.copy(self, copydata)
        new_sig._label_datamap = LabeledSignal.DataMap(
            new_sig.data,
            new_sig.labels,
            new_sig.info
        )
        new_sig._time_datamap = TimeSignal.DataMap(
            new_sig.data,
            new_sig.timestamps,
            new_sig.info
        )
        return new_sig

    def __getitem__(self, key):
        new_timestamps = self._extract_timestamps(key)
        new_labels = self._extract_labels(key)
        new_data = self.data[key]
        return StreamSignal(
            new_timestamps,
            new_labels,
            new_data,
            self.info
        )

class FeatureSignal(LabeledSignal):
    """
    Represents a signal whose rows correspond to a feature class
    and whose columns correspond to a feature label 
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
                if class_label in signal.cdm:
                    label_data = signal.cdm[class_label].data
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
            
    class DataMap:
        
        def __init__(
            self,
            labels: Sequence[str],
            class_dict: dict[str, tuple[int, int]],
            data: np.ndarray,
            signal_info: SignalInfo,
            sort=True,
        ):
            self.labels = labels
            self.classes = class_dict
            self.data = data
            self.info = signal_info
            
            self._succ_classes_list: list[str] = None
            self._build_succession(sort)
        
        @property
        def successive_classes(self):
            """The class list in order of succession"""
            return self._succ_classes_list
        
        def get(self, key):
            try:
                return self[key]
            except:
                return None
        
        def __contains__(self, class_name):
            return class_name in self.classes    
            
        def __getitem__(self, key):
            if isinstance(key, str):
                class_range = self.classes[key]
                subclasses = {key: class_range}
                start, stop = class_range
                subdata = self.data[start:stop]
                sort=False
            elif isinstance(key, list):
                minstart = maxsize
                maxstop = -1
                subclasses: dict[str, tuple[int, int]] = {}
                for k in key:
                    class_range = self.classes[k]
                    subclasses[k] = class_range
                    start, stop = class_range
                    minstart = min(minstart, start)
                    maxstop = max(maxstop, stop)
                
                subdata = self.data[start:stop]
                sort=True
                    
            else:
                raise KeyError(f"Incompatible Key Type. Must be {str} or {list}")
            
            return FeatureSignal.DataMap(
                self.labels,
                subclasses,
                subdata,
                self.info,
                sort,
            )
        
        def signal(self):
            return FeatureSignal(
                None,
                None,
                None,
                None,
                classes_datamap=self,
            )
        
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
            
    def __init__(
        self, 
        labels: Sequence[str],
        class_dict: dict[str, tuple[int, int]],
        data: np.ndarray, 
        signal_info: SignalInfo,
        sort=True,
        classes_datamap: FeatureSignal.DataMap=None
    ):
        if not classes_datamap:
            super().__init__(labels, data, signal_info)
            self._class_datamap = FeatureSignal.DataMap(
                labels, 
                class_dict, 
                data, 
                signal_info,
                sort
            )
            self.classes = self._class_datamap.classes
        else:
            super().__init__(
                classes_datamap.labels,
                classes_datamap.data,
                classes_datamap.info
            )
            self.classes = classes_datamap.classes
            self._class_datamap = classes_datamap
            
    @property
    def class_datamap(self):
        """Map from the classes to the portion of the signal"""
        return self._class_datamap
    
    @property
    def cdm(self):
        """Shorthand for class_datamap"""
        return self._class_datamap
    
    def __getitem__(self, key):
        new_labels = self._extract_labels(key)
        new_classes = self._extract_classes_by_key(key)
        return FeatureSignal(
            new_labels,
            new_classes,
            self.data[key],
            self.info
        )
    
    def _extract_classes_by_key(self, key):
        if isinstance(key, (slice, Sequence, np.ndarray, int)):
            new_classes = self._extract_classes(key)
        elif isinstance(key, tuple):
            left, _ = key
            new_classes = self._extract_classes(left)
        else:
            new_classes = self.classes
        return new_classes
    
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
            rows_to_include = np.array(rows, dtype='int64')
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
        succ_classes_list = self.cdm._succ_classes_list
        
        
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
    