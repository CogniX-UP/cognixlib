"""
A filtering module consisting of wrappers or unique implementations of filtering.

This module's approach is object-oriented and not function oriented
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ..data import Signal, StreamSignal, StreamSignalInfo
from mne.filter import (
    create_filter,
    _overlap_add_filter,
    _iir_filter,
)
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

class Phase(StrEnum):
    ZERO="zero"
    ZERO_DOUBLE="zero-double"
    MINIMUM="minimum"
    MINIMUM_HALF="minimum-half"

class FilterWindow(StrEnum):
    """Defines standard window types"""
    HAMMING="hamming"
    HANN="hann"
    BLACKMAN="blackman"
    
@dataclass
class FilterParams:
    """
    Specifies standard filter parameters common to many filter
    types from the sci-py library.
    """
    low_freq: float = None
    high_freq: float = None
    l_trans_bandwidth: float = "auto"
    h_trans_bandwidth: float = "auto"
    filter_length: str | int = "auto"
    phase: Phase = Phase.ZERO
    
class Filter(ABC):
    """The basic definition of a filter"""
        
    @classmethod
    @abstractmethod
    def ensure_correct_type(cls, signal: Signal) -> Signal:
        pass
    
    @abstractmethod
    def filter(self, signal: Signal, copy=True) -> Signal:
        pass

class MNEFilter(Filter):
    
    @classmethod
    def ensure_correct_type(cls, signal: Signal) -> Signal:
        if signal.data.dtype != np.float64:
            signal = signal.copy(False)
            signal.data = signal.data.astype(np.float64)
        return signal 
    
class FIRFilter(MNEFilter):
    """
    An implementation of an FIR filter that handles
    the typical cases of lowpass, highpass, bandpass
    and bandstop.
    """
    
    def __init__(
        self, 
        sfreq: float | StreamSignalInfo,
        params: FilterParams,
        wnd: FilterWindow = FilterWindow.HAMMING,
        data_valid: Signal = None
    ):
        self._params = params
        self.sfreq = sfreq
        if isinstance(sfreq, StreamSignalInfo):
            self.sfreq = sfreq.nominal_srate
        
        if data_valid:
            data_valid = self.ensure_correct_type(data_valid) 
            data_check = data_valid.data.T if data_valid else None
        else:
            data_check = None
        
        self._fir = create_filter(
            data=data_check,
            sfreq=self.sfreq,
            l_freq=params.low_freq,
            h_freq=params.high_freq,
            filter_length=params.filter_length,
            method="fir",
            phase=params.phase,
            fir_window=wnd,
            fir_design="firwin"
        )
        
    def filter(self, signal: Signal, copy=True):
        result_signal = signal.copy(False)
        result_signal = self.ensure_correct_type(result_signal)
                
        filt_data = _overlap_add_filter(
            result_signal.data.T,
            self._fir,
            None,
            self._params.phase,
            copy=copy
        )
        result_signal.data = filt_data.T
        return result_signal

class IIRFilter(MNEFilter):
    """An implementation of an IIR Filter"""
    
    def __init__(
        self,
        sfreq: float | StreamSignalInfo,
        params: FilterParams,
        phase: Phase = Phase.ZERO,
        data_valid: Signal=None
    ):
        self._params = params
        self._phase = phase
        self.sfreq = sfreq
        if isinstance(sfreq, StreamSignalInfo):
            self.sfreq = sfreq.nominal_srate
        
        if data_valid:
            data_valid = self.ensure_correct_type(data_valid) 
            data_check = data_valid.data.T
        else:
            data_check = None
        
        self._irr = create_filter(
            data=data_check,
            l_freq=params.low_freq,
            h_freq=params.high_freq,
            phase=phase,
            method="iir",
            filter_length=params.filter_length,
        )
        
    def filter(self, signal: Signal, copy=True):
        result = signal.copy(False)
        result = self.ensure_correct_type(result)
        
        filt_data = _iir_filter(
            result.data.T,
            self._irr,
            None,
            -1,
            copy,
            self._phase,
        )
        result.data = filt_data.T
        return result.data
        
        