"""
A filtering module consisting of wrappers or unique implementations of filtering.

This module's approach is object-oriented and not function oriented
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..data import Signal, StreamSignal, StreamSignalInfo
from mne.filter import (
    create_filter,
    _overlap_add_filter,
    _iir_filter,
)
from scipy.signal import oaconvolve, convolve
from FIRconv import FIRfilter
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

class FilterType(StrEnum):
    LOWPASS='lowpass'
    HIGHPASS='highpass'
    BANDPASS='bandpass'
    BANDSTOP='bandstop'
    
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
        
    @classmethod
    @abstractmethod
    def ensure_correct_type(cls, signal: Signal) -> Signal:
        pass
    
    @abstractmethod
    def filter(
        self, 
        signal: Signal | np.ndarray
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
        remove_delay=True,
    ):
        super().__init__(h, channels)
        self._phase = phase
        self._method = method
        self._remove_delay = remove_delay
    
    def filter(
        self, 
        signal: Signal | np.ndarray,
    ) -> Signal | np.ndarray | None:

        is_signal = isinstance(signal, Signal)
        data = signal.data if is_signal else signal
        samples, channels = data.shape[0]
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
        block_size=512
    ):
        super().__init__(h, channels)
        
        self._method = method
        self._block_size = block_size
        self._applier = FIRfilter(method, block_size, self._h_apply)
        
        # initial buffer
        self._buffer = np.zeros(2*block_size, channels)
        self._curr_idx = 0
    
    def filter(
        self, 
        signal: Signal | np.ndarray
    ) -> Signal | np.ndarray | None:
        
        return super().filter(signal)
        
        
        


# TODO implement this
class FIRApplierMNE(FIRApplier):
    pass
    

        