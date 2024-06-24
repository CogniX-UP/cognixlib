"""
This module defines ways to extract segments out of a signal based on timestamps
or simple windowing. 
"""
from __future__ import annotations
from ..data.signals import TimeSignal, SignalKey
from ..data.circ_buffer import CircularBuffer
from numpy import full, nan, float64, ndarray
from collections.abc import Sequence, Iterable, Mapping
from abc import ABC, abstractmethod
from enum import IntEnum

# SEGMENTS

class SegmentFinder(ABC):
    """
    Extracts segments from a signal based on marker timestamps relative to
    the signal time axis. Multiple markers can be used.
    """
    
    class TimestampCache:
        
        def __init__(self, finder: SegmentFinder, cache_length=1000):
            self.current_length = 0
            self._cache = full(shape=cache_length, fill_value=nan,dtype=float64)
            self._finder = finder
            self._buffer: CircularBuffer = None
        
        def append(self, tmps: float | Sequence[float] | ndarray):
            if isinstance(tmps, (Sequence, ndarray)):
                l_t = len(tmps)
                self._cache[self.current_length:self.current_length + l_t] = tmps
                self.current_length += l_t
            else:
                self._cache[self.current_length] = tmps
                self.current_length += 1
        
        def sort(self):
            self._cache[0:self.current_length].sort()
        
        def segment(self, offset: tuple[float, float], data_ref: TimeSignal):
            output_data: Sequence[TimeSignal] = []
            cl = self.current_length
            
            for i in range(cl):
                segment, timestamps = self._finder._buffer.segment(
                    self._cache[i],
                    offset,
                )
                
                if segment is not None:
                    self.current_length -= 1
                    new_sig = data_ref.copy(False)
                    new_sig.data = segment
                    new_sig.timestamps = timestamps
                    output_data.append(new_sig)
            
            return output_data
            
    def __init__(
        self, 
        marker_names: Iterable[str],
        marker_cache_length=1000,
    ):
        self._data_ref: TimeSignal = None
        self._marker_tm_cache = {
            m_name: SegmentFinder.TimestampCache(self, marker_cache_length)
            for m_name in marker_names
        }
        self._buffer: CircularBuffer = None
    
    @abstractmethod
    def update_data(self, data_signal: TimeSignal):
        pass

    def update_markers(self, marker_signal: TimeSignal | Sequence[tuple[str, float]]):        
        
        changed_caches: set[SegmentFinder.TimestampCache] = set()
        m_len = (
            marker_signal.data.shape[0] 
            if isinstance(marker_signal, TimeSignal)
            else len(marker_signal)
        )
        
        for i in range(m_len):
            if isinstance(marker_signal, TimeSignal):
                m_name = marker_signal.data[i][0]
                tms = marker_signal.timestamps[i]
            else:
                m_name, tms = marker_signal
            
            if m_name in self._marker_tm_cache:
                m_cache = self._marker_tm_cache[m_name]
                changed_caches.add(m_cache)
                tms = marker_signal.timestamps[i]
                m_cache.append(tms)

        if not changed_caches:
            return False
        for cache in changed_caches:
            cache.sort()
        return True

    def segments(self, offset: tuple[float, float]):
        result: dict[str, list[TimeSignal]] = {}
        if not self._data_ref:
            return result
        
        for m_name, cache in self._marker_tm_cache.items():
            t_signals = cache.segment(offset, self._data_ref)
            if t_signals:
                result[m_name] = t_signals
        
        return result            

class SegmentFinderOffline(SegmentFinder):
    
    def __init__(
        self, 
        marker_names: Iterable[str], 
        marker_cache_length=1000,
        data: TimeSignal=None
    ):
        super().__init__(marker_names, marker_cache_length)
        if data:
            self.update_data(data)
    
    def update_data(self, data_signal: TimeSignal):
        self._data_ref = data_signal
        self._buffer = CircularBuffer.create(
            data_signal.data,
            data_signal.timestamps,
        )
        
class SegmentFinderOnline(SegmentFinder):
    
    def __init__(
        self, 
        marker_names: Iterable[str], 
        marker_cache_length=1000,
        buffer_dur=0.0,
        nominal_srate=-1,
        data_signal: TimeSignal = None
    ):
        super().__init__(marker_names, marker_cache_length)
        self.set_buffer_info(buffer_dur, nominal_srate)
        
        if data_signal:
            self.update_data(data_signal)
    
    def set_buffer_info(self, buffer_dur: float, srate: int):
        self._buffer_dur = buffer_dur
        self._nominal_srate = srate
        
    def update_data(
        self, 
        data_signal: TimeSignal
    ):        
        
        if not self._buffer:
            self._data_ref = data_signal
            self._buffer = CircularBuffer(
                sampling_frequency=self._nominal_srate,
                buffer_duration=self._buffer_dur,
                start_time=data_signal.timestamps[0],
                channels_count=data_signal.data.shape[1]
            )
        
        self._buffer.append(data_signal.data, data_signal.timestamps)
    
    def is_buffer_init(self):
        return self._buffer is not None


# WINDOWING
class BaseOverlap(ABC):
    """Calculates the step of overlaps for the window finder"""
    
    @abstractmethod
    def step(self, wnd_length: float):
        pass

class ZeroOverlap(BaseOverlap):
    
    def step(self, wnd_length: float):
        return wnd_length

class StepOverlap(BaseOverlap):
    
    def __init__(self, overlap_step: float):
        self.overlap_step = overlap_step
        
    def step(self, wnd_length: float):
        return min(wnd_length, self.overlap_step)

class PercentOverlap(BaseOverlap):
    
    def __init__(self, percent_overlap: float):
        self.percent_overlap = percent_overlap
    
    def step(self, wnd_length: float):
        return max(
            0,
            (1-self.percent_overlap) * wnd_length
        )
        
class WindowFinder(ABC):
    """A searcher for smaller windows, overlapping or not, in a bigger signal."""
        
    def __init__(
        self,
        window_length: float,
        error_margin: float,
        dts_error_scale: float=1.5,
        overlap: BaseOverlap=ZeroOverlap()
    ):
        self.window_length = window_length
        self.error_margin = error_margin
        self.dts_error_scale = dts_error_scale
        self.overlap = overlap
    
    @abstractmethod
    def extract_windows(
        self, signal: TimeSignal | Sequence[TimeSignal]
    ) -> tuple[Sequence[TimeSignal], Mapping[SignalKey, Sequence[TimeSignal]]]:
        pass

class WindowFinderOffline(WindowFinder):
    
    def extract_windows(
        self, signal: TimeSignal | Sequence[TimeSignal]
    ) -> tuple[Sequence[TimeSignal], Mapping[SignalKey, Sequence[TimeSignal]]]:
        if isinstance(signal, TimeSignal):
            signal = [signal]
        
        l_result: list[TimeSignal] = []
        map_result: dict[SignalKey, list[TimeSignal]] = {}
        buffer = CircularBuffer.empty()
        
        step = self.overlap.step(self.window_length)
        for signal in signal:
            
            buffer.reset(signal.data, signal.timestamps)
            time = 0
            sig_start = signal.tms[0]
            
            while time <= signal.duration - self.window_length + self.error_margin:
                
                seg, tms = buffer.segment(
                    time + sig_start,
                    (0, self.window_length),
                    self.error_margin,
                    self.dts_error_scale
                )
                
                if seg is not None:
                    if signal.unique_key not in map_result:
                        map_result[signal.unique_key] = []
                    
                    w_sig = signal.copy(False)
                    w_sig.timestamps = tms
                    w_sig.data = seg
                    
                    map_result[signal.unique_key].append(w_sig)
                    l_result.append(w_sig)
                    
                    # due to finite sampling, we have to add
                    # the actual duration instead of the requested one
                    time += step
        
        return l_result, map_result

class WindowFinderOnline(WindowFinder):
    
    def __init__(
        self, 
        window_length: float, 
        error_margin: float,
        dts_error_scale: float = 1.5, 
        overlap: BaseOverlap = ZeroOverlap(),
        extra_buffer: float = 0.5,
        srate: int = 0,
        start_data: TimeSignal = None
    ):
        super().__init__(window_length, error_margin, dts_error_scale, overlap)
        self.srate=srate
        self.extra_buffer = extra_buffer
        self.current_time = 0
        self.first_window = True
        if self.srate > 0 and start_data:
            self.init_buffer(srate, start_data)
        else:
            self.buffer: CircularBuffer = None
    
    @property
    def step(self):
        return self.overlap.step(self.window_length)
    
    def init_buffer(self, srate: float, start_data: TimeSignal):
        self.buffer = CircularBuffer(
                srate,
                self.window_length + self.extra_buffer,
                start_data.tms[0],
                start_data.data.shape[1]
            )
    
    def is_buffer_init(self):
        return self.buffer is not None
        
    def extract_windows(
        self, signal: TimeSignal
    ) -> tuple[Sequence[TimeSignal], Mapping[SignalKey, Sequence[TimeSignal]]]:
        
        data_dur = signal.duration
        self.current_time += data_dur
        
        # if the buffer is not large enough to hold
        # all the incoming data, we attempt to expand it
        self.buffer.append_expand(
            signal.data,
            signal.timestamps
        )
        
        l_result: list[TimeSignal] = []
        map_result: dict[SignalKey, list[TimeSignal]] = {}
        
        # check for first window
        if self.first_window and self.current_time >= self.window_length:
            extra = self.current_time - self.window_length
            seg, tms = self.buffer.segment_current(
                (-self.window_length-extra, -extra),
                self.error_margin,
                self.dts_error_scale
            )
            
            if seg is not None:
                w_sig = signal.copy(False)
                w_sig.timestamps = tms
                w_sig.data = seg
                
                if signal.unique_key not in map_result:
                    map_result[signal.unique_key] = []
                
                map_result[signal.unique_key].append(w_sig)
                l_result.append(w_sig)

                # due to finite sampling, we have to subtract
                # the actual duration, which may be slightly
                # different than the window duration
                self.current_time -= w_sig.duration
                
            self.first_window = False
        
        # check for any residual windows by applying overlap
        if not self.first_window:
            step = self.step
            while self.current_time >= step:
                extra = self.current_time - step
                seg, tms = self.buffer.segment_current(
                    (-extra-self.window_length, -extra),
                    self.error_margin,
                    self.dts_error_scale
                )
                
                if seg is not None:
                    w_sig = signal.copy(False)
                    w_sig.timestamps = tms
                    w_sig.data = seg
                    
                    if signal.unique_key not in map_result:
                        map_result[signal.unique_key] = []
                    
                    map_result[signal.unique_key].append(w_sig)
                    l_result.append(w_sig)

                    self.current_time -= self.step + (w_sig.duration - self.window_length)
        
        return l_result, map_result