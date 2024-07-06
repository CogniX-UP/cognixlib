"""
A filtering module consisting of wrappers or unique implementations of filtering.

This module's approach is object-oriented and not function oriented
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ..data import Signal, TimeSignal
from mne.filter import (
    create_filter,
    _overlap_add_filter,
    _iir_filter,
)
from scipy.signal import (
    oaconvolve, 
    convolve, 
    minimum_phase,
    firwin
)
from FIRconv import FIRfilter
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

import numpy as np

class FilterType(StrEnum):
    PASS='pass'
    """Either a lowpass or a bandpass, depending on the cutoff frequencies."""
    STOP='stop'
    """Either a highpass or a bandstop, depending on the cutoff frequencies."""
    LOWPASS='lowpass'
    HIGHPASS='highpass'
    BANDPASS='bandpass'
    BANDSTOP='bandstop'

ft = FilterType

# DESIGNERS

class FIRDesigner(ABC):
    """Designs an FIR filter."""
    
    def __init__(
        self,
        fs: int,
        f_type: FilterType,
        f_freq: int | Sequence[int],
        min_phase=False,
    ):
        self.fs = fs
        self.f_type = f_type
        self.f_freq = f_freq
        self.min_phase = min_phase

    @abstractmethod
    def create_filter(self) -> np.ndarray:
        """Creates an FIR filter."""
        pass

class FIRDesignerScipy(FIRDesigner):
    """Designs an FIR filter using Scipy's `fir_win <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html>_`"""
    
    # TODO: There are many more windows in firwin. Check them out
    class Window(StrEnum):
        HAMMING='hamming'
        HANN='hann'
        BLACKMAN='blackman'
    
    def __init__(
        self, 
        fs: int, 
        f_type: FilterType, 
        f_freq: int | Sequence[int],
        length: int,
        window: FIRDesignerScipy.Window = Window.HAMMING,
        min_phase=False
    ):
        super().__init__(fs, f_type, f_freq, min_phase)
        self.length = length
        self.window = window
    
    def create_filter(self) -> np.ndarray:
        pass_zero = self.f_type
        if pass_zero == ft.PASS:
            pass_zero = True
        elif pass_zero == ft.STOP:
            pass_zero = False
        
        h = firwin(self.length, self.f_freq, pass_zero=pass_zero, fs=self.fs)
        if self.min_phase:
            h = minimum_phase(h)
        return h

class FIRDesignerMNE(FIRDesigner):
    """Designs an FIR filter using MNE's 'create_filter <https://mne.tools/stable/generated/mne.filter.create_filter.html>_'"""
    
    class Window(StrEnum):
        HAMMING='hamming'
        HANN='hann'
        BLACKMAN='blackman'
        
    def __init__(
        self, 
        fs: int, 
        f_type: FilterType,
        f_freq: int | tuple[int, int],
        trans_bandwidth: Sequence[int] = ['auto', 'auto'],
        window: Window = Window.HAMMING,
        min_phase=False
    ):
        super().__init__(fs, f_type, min_phase)
        self.f_freq = f_freq
        self.trans_bandwidth = trans_bandwidth
        self.window = window
    
    def create_filter(self) -> np.ndarray:
        
        f_freq = self.f_freq
        if isinstance(f_freq, Integral):
            f_freq = [f_freq]
            
        n_f_freq = len(f_freq)
        if (
            self.f_type == ft.LOWPASS or 
            self.f_type == ft.HIGHPASS and
            n_f_freq != 1
        ):
            raise RuntimeWarning(f'A {self.f_type} filter was requested but got more cutoffs: {f_freq}')

        if (
            self.f_type == ft.LOWPASS or
            (
                self.f_type == ft.PASS and 
                n_f_freq == 1
            ) 
        ):
            l, h = None, f_freq[0]
        
        elif (
            self.f_type == ft.HIGHPASS or
            (
                self.f_type == ft.PASS and
                n_f_freq == 1
            )
        ):
            l, h = f_freq[0], None
        
        elif self.f_type == ft.BANDPASS or self.f_type == ft.PASS:
            l, h = f_freq
        elif self.f_type == ft.BANDSTOP or self.f_type == ft.STOP:
            h, l = f_freq
        
        l_trans, h_trans = self.trans_bandwidth
        h = create_filter(
            data=None,
            sfreq=self.fs,
            l_freq=l,
            h_freq=h,
            l_trans_bandwidth=l_trans,
            h_trans_bandwidth=h_trans,
            fir_window=self.window
        )
        if self.min_phase:
            h = minimum_phase(h)
        return h
        
# APPLIERS
    
class FilterApplier(ABC):
    """The basic definition of an object that applies a filter"""
    
    class Phase(StrEnum):
        PASS='pass'
        """Applies the filter once. Known also as the forward method."""
        ZERO='zero'
        """
        Applies the filter so that it removes phase distortion. Known also
        as forward-backwards method.
        """
    
    @abstractmethod
    def filter(
        self, 
        signal:  Signal | np.ndarray
    ) -> Signal | np.ndarray | None:
        """
        Filters an incoming signal or numpy array.
        
        An implementation of this method should consider that many filters
        can be applied by the same object, hence the possibility of a 
        :code:`Sequence` return type.
        
        Furthermore, the incoming signal type should be identical to the input
        type.
        """
        pass

class FIRApplier(FilterApplier):
    """
    FIR Filter Applier 
    """
    
    def __init__(
        self,
        h: np.ndarray,
        channels=1
    ):
        self._h = h
        self._h_apply = h
        if self._h.ndim == 1 and channels > 1:
            self._h_apply = np.tile(h, (channels, 1)).T
        
        self._changed = False
        super().__init__()
    
    @property
    def h(self) -> np.ndarray:
        return self._h
    
    @h.setter
    def h(self, value: np.ndarray):
        self._h = value
        self._h_apply = value
        self._changed = True
    
    def _reconstruct_h(self, data: np.ndarray):
        if (
            not self._changed and
            self._h_apply.shape  == data.shape
        ):
            return
        
        self._changed = False
        self._h_apply = self._h
        channels = data.shape[1]
        if self._h.ndim == 1:
            self._h_apply = np.tile(self._h, (channels, 1)).T

class FIROfflineApplier(FIRApplier):
    """
    Filters the whole signal using an FIR filter. Should always return the same size as the signal.
    
    If the FIR designed has a linear-phase, we can shift the signal instead of forward-backward
    filtering it to remove delays. This is close enough to zero-phase filtering. Otherwise, we 
    could simply forward-backward filter it, which is more accurate but requires more time and
    will have a slight change in amplitude.
    """
    
    class Phase(StrEnum):
        PASS='pass'
        """Applies the filter once. Known also as the forward method."""
        ZERO='zero'
        """
        Applies the filter so that it removes phase distortion. Known also
        as forward-backwards method.
        """
        ZERO_SHIFT='zero-shift'
        """
        Applies the filter once and accounts for phase distortion by shifting
        it to the left. This is faster than the :code:`ZERO` method, but only
        works when we have a linear phase FIR filter. 
        """
        
    class Method(StrEnum):
        """The method with which to apply the convolution in the :code:`FIROfflineApplier` filter."""
        # AUTO='auto'
        DIRECT='direct'
        FFT='fft'
        OVERLAP_ADD='overlap-add'
        
    def __init__(
        self, 
        h: np.ndarray, 
        channels=1,
        phase: Phase = Phase.ZERO_SHIFT,
        method: Method = Method.OVERLAP_ADD,
    ):
        super().__init__(h, channels)
        self._phase = phase
        self._method = method
    
    def filter(
        self, 
        signal: Signal | np.ndarray,
    ) -> Signal | np.ndarray | None:

        is_signal = isinstance(signal, Signal)
        data = signal.data if is_signal else signal
        samples, channels = data.shape
        self._reconstruct_h(data)
        
        foa = FIROfflineApplier
        phase_mode = 'full' if self._phase == foa.Phase.PASS else 'same'
        
        # Using the notation [0:samples] means that we're getting the
        # correct result for either Phase.ZERO or Phase.PASS
        
        if self._method == foa.Method.OVERLAP_ADD:
            conv_res = oaconvolve(data, self._h_apply, mode=phase_mode, axes=0)[0:samples]
            if self._phase == foa.Phase.ZERO:
                h_r = self._h_apply[::-1]
                conv_res = oaconvolve(conv_res, h_r, mode=phase_mode, axes=0)
        else:
            conv_res = np.zeros((samples, channels))
            # method == methdo for convolve in DIRECT or FFT
            for i in range(channels):
                conv_res[:, i] = convolve(
                    data[:,i], 
                    self._h_apply[:, i],
                    mode=phase_mode,
                    method=self._method
                )[0:samples]
            
            if self._phase == foa.Phase.ZERO:
                h_r = self._h_apply[::-1]
                for i in range(channels):
                    conv_res[:, i] = convolve(
                        conv_res[:,i],
                        h_r,
                        mode=phase_mode,
                        method=self._method
                    )
        
        if not is_signal:
            return conv_res
        
        sig_res = signal.copy(False)
        sig_res.data = conv_res
        return sig_res
        
class FIROnlineApplier(FIRApplier):
    """
    Designed to filter an incoming signal using an FIR (Finite Impulse Response). As this
    is online filtering, it needs to be causal, hence zero-phase is not achievable.
    """
    
    class Method(StrEnum):
        OVERLAP_ADD = 'overlap-add'
        OVERLAP_SAVE = 'overlap-save'
        UPOLS = 'upols'
        
    def __init__(
        self, 
        h: np.ndarray, 
        channels=1,
        method=Method.OVERLAP_SAVE,
        block_size=512,
    ):
        super().__init__(h, channels)
        
        self._method = method
        self._block_size = block_size
        self._applier = FIRfilter(method, block_size, self._h_apply, normalize=False)
        
        # initial buffer
        self._data_buffer = np.zeros((3 * block_size, channels))
        # in case this is a Stream Signal
        self._times_buffer = np.zeros(3 * block_size)
        self._curr_size = 0
    
    @property
    def apply_method(self) -> Method:
        return self._method
        
    def filter(
        self, 
        signal: TimeSignal | Signal | np.ndarray
    ) -> TimeSignal | Signal | np.ndarray | None:
        
        is_time_signal = isinstance(signal, TimeSignal)
        
        data = signal.data if isinstance(signal, Signal) else signal
        rows, cols = data.shape
        
        if self._curr_size <= len(self._data_buffer):
            self._data_buffer[self._curr_size: self._curr_size + rows] = data
            if is_time_signal:
                self._times_buffer[self._curr_size: self._curr_size + rows] = signal.timestamps
        else:
            self._data_buffer = np.append(self._data_buffer, data, 0)
            if is_time_signal:
                self._times_buffer = np.append(self._times_buffer, signal.timestamps, 0)
        self._curr_size += rows
        
        if self._curr_size < self._block_size:
            return None
        
        blocks = int(self._curr_size / self._block_size)
        res_data = np.zeros((blocks * self._block_size, cols))
        if is_time_signal:
            res_time = np.zeros_like(blocks * self._block_size)
            
        frame_start = 0
        frame_end = self._block_size
        for _ in range(blocks):
            res_data[frame_start:frame_end] = self._applier.process(self._data_buffer[frame_start:frame_end])
            frame_start = frame_end
            frame_end += self._block_size
        
        # if there are any values left i.e. the buffer isn't completely empty
        ret_block = self._block_size * blocks
        left_size = self._curr_size - ret_block
        # perhaps we could use np.roll
        self._data_buffer[0:left_size] = self._data_buffer[ret_block:self._curr_size]
        
        if is_time_signal:
            res_time = self._times_buffer[0:ret_block].copy()
        
        self._curr_size -= ret_block
        
        # return results
        if isinstance(signal, np.ndarray):
            return res_data
        elif isinstance(signal, Signal):
            sig_res = signal.copy(False)
            sig_res.data = res_data
            if is_time_signal:
                sig_res.timestamps = res_time
            return sig_res
        

# TODO implement this
# class FIRApplierMNE(FIRApplier):
#     pass
    

        