from __future__ import annotations

import mne
import numpy as np

from mne_icalabel import label_components  

from cognixcore import PortConfig
from cognixcore.config.traits import *
from traitsui.api import *

from collections.abc import Sequence

from ...scripting.data import (
    Signal, 
    StreamSignal, 
    LabeledSignal,
    CircularBuffer
)
from ...scripting.processing.manipulation import *

from ...scripting.processing.filters import (
    FilterParams,
    Phase,
    FIRFilter,
    IIRFilter,
    Phase,
    FilterWindow,
)

from ...scripting.mne.prep import (
    remove_trend_data,
    line_noise_removal_prep,
    referencing_prep,
    noisychannelsprep
)

class SegmentationNode(Node):
    """
    Outputs segments of a StreamSignal based on timestamps
    of a list of markers.
    """
    
    title = 'Segment'
    version = '0.1'

    class Config(NodeTraitsConfig):
        offset: tuple[float, float] = CX_Tuple(-0.5, 0.5)
        mode: str = Enum(
            'input', 
            'buffer', 
            desc='Whether to use the input to decide buffer size (offline) or manually (online)'
        )
        buffer_duration: float = CX_Float(10.0, desc = 'Duration of the buffer in seconds', visible_when='mode=="buffer"')
        timestamp_buffer: int = CX_Int(1000, desc = 'Maximum amount of timestamps that can be processed')
        debug: bool = Bool(False, desc='debug message when data is segmented')
        markers: PortList = Instance(
            PortList,
            lambda: PortList(
                list_type=PortList.ListType.OUTPUTS | PortList.ListType.INPUTS,
                min_port_count=1,
                inp_params=PortList.Params(
                  allowed_data=StreamSignal | Sequence[tuple[str, float]],
                  prefix="in_",  
                ),
                out_params=PortList.Params(
                    allowed_data=StreamSignal | Sequence[StreamSignal],
                    suffix="_data",
                )
            ),
            style='custom',
            desc='the markers to segment the signal by'
        )
        
    init_inputs = [
        PortConfig(label='data', allowed_data=StreamSignal),
    ]
    init_outputs = [
        PortConfig(label='og_data', allowed_data=StreamSignal)
    ]

    @property
    def config(self) -> Config:
        return self._config
    
    def init(self):
        
        self._valid_markers: list[str] = self.config.markers.valid_names
        # find valid markers
        self._m_name_to_port: dict[str, int] = {
            m_name: index for index, m_name in enumerate(self._valid_markers)
        }
        
        self._seg_finder: SegmentFinder = None
        
        if self.config.mode == "input":
            self._seg_finder = SegmentFinderOffline(
                marker_names=self._valid_markers,
            )
        else:
            self._seg_finder = SegmentFinderOnline(
                marker_names=self._valid_markers,
            )
    
    def update_event(self, inp=-1):
        
        sig = self.input(inp)
        if sig is None:
            return
        
        if inp==0:
            sig: StreamSignal = sig
            if isinstance(self._seg_finder, SegmentFinderOnline) and not self._seg_finder.is_buffer_init():
                self._seg_finder.set_buffer_info(
                    self.config.buffer_duration, 
                    sig.info.nominal_srate,
                )

            self._seg_finder.update_data(sig)
        
        else:
            
            self._seg_finder.update_markers(sig)
        
        signal_segments = self._seg_finder.segments(self.config.offset)
        if signal_segments:
            
            if self.config.debug:
                msg = f"ERATE: {self._seg_finder._buffer.effective_srate}\n"
                msg += f"Found Marker Types: {signal_segments.keys()}\n"
            
            for m_name, signals in signal_segments.items():
                out_index = self._m_name_to_port[m_name]
                self.set_output(out_index, signals)
                
                # debug only
                if self.config.debug:
                    msg += f"\n For Marker: {m_name}"
                    for sig in signals:
                        s_dur = sig.timestamps[-1] - sig.timestamps[0]
                        msg += f"\t[{sig.timestamps[0]} : {sig.timestamps[-1]}] dur:[{s_dur}]</br>"
            
            if self.config.debug:
                self.logger.info(msg)          

class WindowNode(Node):
    """
    The Window Node has two functions.
    
    input:
        In the input mode, it accepts a StreamSignal as a whole
        input and attempts to segment it into windows, depending
        on the window length and overlap given.
    buffer:
        In the buffer mode, it acts as a buffer. Incoming data
        must be of type StreamSignal. Each time the node updates,
        it adds the input to the buffer and outputs the Window
        if it is detected, depending on the overlap given.
        
        In the buffer mode, it is assumed that small chunks of a 
        bigger signal are incoming to the buffer. In the case of 
        a bigger chunk, to avoid data loss, the buffer is extended
        to hold the additional data.
        
        Essentially, buffer mode is designed for real time processing.
    """
    
    title = 'Window'
    version = '0.1'

    buffer_mode = 'buffer'
    input_mode = 'input'
    
    step_mode = 'step'
    percent_mode = 'percent'
    
    class Config(NodeTraitsConfig):
        mode: str = Enum('input', 'buffer', desc='the operation mode of the Window Node')
        window_length: float = CX_Float(0.5, desc='length of the window in seconds')
        error_margin: float = CX_Float(0.0, desc='error margin for when the segment is slightly smaller than requested')
        dts_error_scale: float = Range(
            1.0, 
            3.0, 
            1.5, 
            desc="how close to the effective sample rate we want to reduce the error to"
        )
        extra_buffer: float = CX_Float(0.5, desc="initialize the buffer with extra amount of seconds")
        overlap_mode: str = Enum('none', 'step', 'percent', desc='how the segmented windows should be overlapped')
        overlap_step: float = Range(0.01, None, 0.5, exclude_high=True, desc='overlap step between the windows, in seconds')
        overlap_percent: float = Range(0.0, 0.99, desc='overlap percent between the windows')
        # the 0.99 means that at max, 100 windows can be created with a step of x * window_time
        debug: bool = Bool(False, desc='debug message when data is segmented')
        ports: PortList = Instance(
            PortList,
            lambda: PortList(
                list_type= PortList.ListType.INPUTS | PortList.ListType.OUTPUTS,
                inp_params=PortList.Params(
                    allowed_data=StreamSignal | Sequence[StreamSignal]
                ),
                out_params=PortList.Params(
                    allowed_data=StreamSignal | Sequence[StreamSignal],
                    suffix="_wnds"
                )
            ),
            style='custom'
        )
        # to show the borders correctly, the scrollable must be in an outside group
        traits_view = View(
            Group(
                Group(
                    Item('mode'),
                    Item('window_length'),
                    Item('error_margin'),
                    Item('dts_error_scale'),
                    Item('extra_buffer'),
                    Item('overlap_mode'),
                    Group(
                        Item(
                            'overlap_step', 
                            label='step', 
                            visible_when='overlap_mode=="step"',
                        ),
                        Item(
                            'overlap_percent',
                            label='percent',
                            visible_when='overlap_mode=="percent"',
                        ),
                    ),
                    Item('debug'),
                    Item('ports', style='custom'),
                ),
                label='configuration',
                scrollable=True,
            )
        )
 
    def init(self):
        
        if self.config.overlap_mode == "none":
            self.overlap = ZeroOverlap()
        elif self.config.overlap_mode == "step":
            self.overlap = StepOverlap(self.config.overlap_step)
        else:
            self.overlap = PercentOverlap(self.config.overlap_percent)
        
        def create_finder():
            if self.config.mode == "input":
                return WindowFinderOffline(
                    self.config.window_length,
                    self.config.error_margin,
                    self.config.dts_error_scale,
                    self.overlap
                )
            else:
                return WindowFinderOnline(
                    self.config.window_length,
                    self.config.error_margin,
                    self.config.dts_error_scale,
                    self.overlap,
                    self.config.extra_buffer,
                )
            
        self.port_ind_to_finder: dict[int, WindowFinder] ={
            index: create_finder()
            for index in range(len(self._outputs))
        }
    
    @property
    def config(self) -> WindowNode.Config:
        return self._config
    
    def update_event(self, inp=-1):
        
        signals: StreamSignal | Sequence[StreamSignal] = self.input(inp)
        if not signals:
            return
        
        wnd_finder = self.port_ind_to_finder[inp]
        if not wnd_finder:
            return
        
        # we're assuming that in real time it's always a time signal and not a list
        if isinstance(wnd_finder, WindowFinderOnline) and not wnd_finder.is_buffer_init():
            wnd_finder.init_buffer(signals.info.nominal_srate, signals)
        
        wnds_list, wnds_map = wnd_finder.extract_windows(signals)
        if not wnds_map:
            return
        
        # a sequence of signals will output a sequence of windows
        self.set_output(inp, wnds_list)
        
        if not self.config.debug:
            return
        
        msg = ''
        if isinstance(signals, StreamSignal):
            signals = [signals]
        
        for sig in signals:
            msg += f"SIGNAL: [{sig.tms[0]} - {sig.tms[-1]}], DUR: {sig.duration}\n"
            wnds = wnds_map.get(sig.unique_key)
            msg += f"WINDOWS EXTRACTED: {len(wnds)}\n"
            
            for wnd in wnds:
                msg += f"WINDOW: [{wnd.tms[0]} - {wnd.tms[-1]}], DUR: {wnd.duration}\n"
            msg += "\n"
        
        self.logger.info(msg)

                
 
class StreamSignalSelectionNode(Node):
    """
    Selects part of a stream signal based on its channels. An option
    allows for the labels given to be forced to lowercase.
    
    Typical EEG labels for a 32 cap are:
        fp1, af3, f7, f3, fc1, fc5, t7, c3, cp1, cp5, p7, p3,
        pz, po3, o1, oz, o2, po4, p4, p8, cp6, cp2, c4, t8, fc6,
        fc2, f4, f8, af4, fp2, fz, cz
    """
    
    title = 'Select Channels'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        
        labels_egg = ['Fp1', 'Af3', 'F7', 'F3', 'Fc1', 'Fc5', 'T7',
            'C3', 'Cp1', 'Cp5', 'P7', 'P3', 'Pz', 'Po3', 'O1', 'Oz', 'O2', 'Po4',
            'P4', 'P8', 'Cp6', 'Cp2', 'C4', 'T8', 'Fc6', 'Fc2', 'F4', 'F8', 'Af4',
            'Fp2', 'Fz', 'Cz'
        ]
        
        force_lowercase: bool = Bool(True)
        labels: list[str] = List(CX_Str())
    
    init_inputs = [
        PortConfig(label='in', allowed_data=StreamSignal)
    ]
    init_outputs = [
        PortConfig(label='out', allowed_data=StreamSignal)
    ]

    @property
    def config(self) -> StreamSignalSelectionNode.Config:
        return self._config
    
    def init(self):
        self.chan_inds = None
        if self.config.force_lowercase:
            self.selected_labels = set(self.config.labels)
        else:
            self.selected_labels = { label.lower() for label in self.config.labels }
        
        # these help for potential optimization
        # check _optimize_searchs
        # their values will be set to an empty "" if not
        self.start_ind: int = None
        self.stop_ind: int = None
        
    def update_event(self, inp=-1):
        
        signal: StreamSignal = self.input(inp)
        if not signal:
            return 
        
        if not self.start_ind:
            self._optimize_search(signal)
        
        if isinstance(self.start_ind, str):
            # no optimization found
            subdata = signal.data[self.chan_inds]
        else:
            subdata = signal.data[self.start_ind: self.stop_ind]
        
        subsignal = StreamSignal(
            signal.timestamps,
            self.selected_labels,
            subdata,
            signal.info,
            False
        )
        
        self.set_output(0, subsignal)
    
    def _optimize_search(self, signal: LabeledSignal):
        """
        Checks whether the selected labels are in sequence in
        the data signal (their corresponding indices are sequenced 
        integers). If they are, the selection is optimized since
        the internal numpy array is not copied.
        """
        
        # if no labels were given, choose all labels
        if not self.selected_labels:
            self.selected_labels = signal.labels
        # find the indices
        
        print(signal.ldm._label_to_index)
        self.chan_inds = [
            signal.ldm._label_to_index[label] 
            for label in self.selected_labels 
        ]
        self.chan_inds.sort()
        # this means that we have a sequence (0, 1, 2, 3..)
        if (self.chan_inds[-1] - self.chan_inds[0] == len(self.chan_inds)):
            self.start_ind = self.chan_inds[0]
            self.stop_ind = self.chan_inds[-1]
            # make the selected labels in order
            self.selected_labels = [
                self.selected_labels[i] for i in self.chan_inds
            ]
        else:
            self.start_ind = ''
            self.stop_ind = ''

class BadConfidenceRemoval(Node):
    title = 'Confidence Guard'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        conf_label: str = CX_Str("confidence")
        conf_level: float = Range(0.0, 1.0, 0.9, desc="all samples below this confidence will be removed")
    
    @property
    def config(self) -> BadConfidenceRemoval.Config:
        return self._config
    
    init_inputs = [
        PortConfig("in", allowed_data=StreamSignal)
    ]
    init_outputs = [
        PortConfig("out", allowed_data=StreamSignal)
    ]
    
    def update_event(self, inp=-1):
        sig: StreamSignal = self.input(inp)
        if not sig:
            return
        conf_label = self.config.conf_label
        if conf_label not in sig.ldm:
            self.logger.warn(f"No {conf_label} found in incoming signal")
            return
        
        conf_data = sig.ldm[conf_label]
        print(conf_data)
        
class FIRFilterNode(Node):
    title = 'FIR Filter'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        low_freq: float = CX_Float(desc='the low frequency of the filter')
        high_freq: float = CX_Float(desc='the high frequency of the fitler')
        l_trans_bandwidth:float = CX_Float(0.0, desc='the width of the transition band at the low cut-off frequency in Hz')
        h_trans_bandwidth:float = CX_Float(0.0, desc='the width of the transition band at the high cut-off frequency in Hz')
        filter_length: str = CX_String('auto',desc='a string representing the length of the filter (i.e 5s, 5ms etc)')
        phase: str = Enum(Phase, desc='the phase of the filter')
        window: str = Enum(FilterWindow, desc='a choice of windows for an fir filter')
        ports: PortList = Instance(
            PortList,
            lambda: PortList(
                list_type=PortList.ListType.OUTPUTS | PortList.ListType.INPUTS,
                inp_params=PortList.Params(
                    allowed_data=StreamSignal | Sequence[StreamSignal]
                ),
                out_params=PortList.Params(
                    allowed_data=StreamSignal | Sequence[StreamSignal],
                    suffix="_filt"
                )
            ),
            style='custom'
        )
    
    @property
    def config(self) -> FIRFilterNode.Config:
        return self._config
    
    def init(self):
        # TODO might include length later
        c = self.config
        
        try:
            filter_length = int(c.filter_length)
        except:
            filter_length = c.filter_length
        
        self.fir_params = FilterParams(
            c.low_freq,
            c.high_freq,
            c.l_trans_bandwidth if c.l_trans_bandwidth != 0 else "auto",
            c.h_trans_bandwidth if c.h_trans_bandwidth != 0 else "auto",
            filter_length,
            c.phase 
        )
        
        self.fir_filter: FIRFilter = None
                
    def update_event(self, inp=-1):
        
        signals: StreamSignal = self.input(inp)
        if not signals:
            return
        
        if not isinstance(signals, Sequence):
            signals = [signals]
        
        if not self.fir_filter:
            first_signal = signals[0]
            info = first_signal.info
            guard_signal = FIRFilter.ensure_correct_type(first_signal)
            if first_signal.unique_key != guard_signal.unique_key:
                self.logger.warn(
                    f"Incompatible Numpy Type: Converting from {first_signal.data.dtype} to {guard_signal.data.dtype}"
                )
            
            self.fir_filter = FIRFilter(
                info.nominal_srate,
                self.fir_params,
                self.config.window,
                guard_signal,
            )
            
        filtered_sigs: Sequence = []
        for sig in signals:
            sig = FIRFilter.ensure_correct_type(sig)
            f_sig = self.fir_filter.filter(sig)
            filtered_sigs.append(f_sig)
        
        if len(filtered_sigs) == 1:
            self.set_output(inp, filtered_sigs[0])
        else:
            self.set_output(inp, filtered_sigs)
    
class IIRFilterNode(Node):
    title = 'IIR Filter'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        l_freq: float = CX_Float(desc='the low frequency of the filter')
        h_freq: float = CX_Float(desc='the high frequency of the fitler')
        phase:str = Enum('zero','zero-double','forward',desc='the phase of the filter')
        
    init_inputs = [PortConfig(label='data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='filtered data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    
    @property
    def config(self) -> IIRFilterNode.Config:
        return self._config
    
    def init(self):
        pass
                
    def update_event(self, inp=-1):
        
        signal:StreamSignal = self.input(inp)
        if not signal:
            return
        
        if not isinstance(signal,list):
            signal = [signal]
        
        
        list_of_filtered_sigs:Sequence = []
        for sig in signal:
            
            filtered_signal:StreamSignal = sig.copy()
            
            filtered_data = mne.filter.filter_data(
                data = sig.data.Τ,
                sfreq = sig.info.nominal_srate,
                l_freq = self.config.l_freq,
                h_freq = self.config.h_freq,
                phase = self.config.phase,
                method = 'iir'
                )
            
            filtered_signal._data = filtered_data.Τ
            
            list_of_filtered_sigs.append(filtered_signal)
            
        if len(list_of_filtered_sigs) == 1:
            self.set_output(0, list_of_filtered_sigs[0])
        else:
            self.set_output(0, list_of_filtered_sigs)
                    
class NotchFilterNode(Node):
    title = 'Notch Filter'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        freqs:float = CX_Float(50.0,desc='set the line noise freqeuncy')
        filter_length_sec:str = CX_Str(desc='specify filter length in seconds')
        filter_length_samples:int = CX_Int(desc='specify filter length in samples')
        method:str = Enum('fir','iir','spectrum_fit',desc='the method in which the filter is created')
        mt_bandwidth: float = CX_Float(desc='the bandwidth of the multitaper windowing function in Hz')
        p_value:float = CX_Float(0.05,desc='p-value to use in F-test thresholding to determine significant sinusoidal components')
        phase: str = Enum('zero','minimum','zero-double','minimum-half',desc='phase of the filter')
        fir_window: str = Enum('hamming','hann','blackman')
        fir_design: str = Enum('firwin','firwin2')

        
    init_inputs = [PortConfig(label='data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='filtered data',allowed_data=LabeledSignal | Sequence[StreamSignal])]
    
    @property
    def config(self) -> NotchFilterNode.Config:
        return self._config
    
    def init(self):

        print("PRINT",self.config.filter_length_sec,self.config.filter_length_samples)
        self.line_freq = self.config.freqs
        self.filter_length_input = self.config.filter_length_sec if self.config.filter_length_sec else self.config.filter_length_samples
        print("WHAT",self.filter_length_input)
        if not self.filter_length_input:
            self.filter_length_input = 'auto'
        print("WHAT",self.filter_length_input)
        self.method = self.config.method
        self.bandwidth = self.config.mt_bandwidth if self.config.mt_bandwidth else None
        self.p_value = self.config.p_value
        self.phase = self.config.phase
        self.fir_window = self.config.fir_window
        self.fir_design = self.config.fir_design
        
    def update_event(self,inp=-1):
        signal: StreamSignal = self.input(inp)
        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_filtered_sigs:Sequence = []
        for sig in signal:

            filtered_signal:StreamSignal = sig.copy()

            sampling_freq = sig.info.nominal_srate   
            freqs = np.arange(self.line_freq,sampling_freq/2,self.line_freq) 
        
            filtered_data = mne.filter.notch_filter(
                x = sig.data.Τ,
                Fs = sampling_freq,
                freqs = freqs,
                filter_length = self.filter_length_input,
                method = self.method,
                mt_bandwidth= self.bandwidth,
                p_value=self.p_value,
                n_jobs=-1,
                phase = self.phase,
                fir_window = self.fir_window,
                fir_design= self.fir_design
            )

            filtered_signal._data = filtered_data.Τ
            list_of_filtered_sigs.append(filtered_signal)

        if len(list_of_filtered_sigs) == 1:
            self.set_output(0, list_of_filtered_sigs[0])
        else:
            self.set_output(0, list_of_filtered_sigs)

# TODO CHECK CODE FROM HERE

class ResamplingNode(Node):
    title = 'Resampling Data'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        upsample_factor:float = CX_Float(1.0,desc='factor to upsample by')
        downsample_factor:float = CX_Float(1.0,desc='factor to downsample by')
        method:str = Enum('fft','polyphase',desc='resampling method to use')

    init_inputs = [PortConfig(label='data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='filtered data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    
    @property
    def config(self) -> ResamplingNode.Config:
        return self._config
    
    def init(self):
        self.upsample_factor = self.config.upsample_factor
        self.downsample_factor = self.config.downsample_factor
        self.method = self.config.method
        
    def update_event(self,inp=-1):
        signal: StreamSignal = self.input(inp)
        if not signal: 
            return
            
        if not isinstance(signal,list):
            signal = [signal]
        
        
        list_of_resampled_sigs:Sequence = []
        for sig in signal:
            
            resampled_signal:StreamSignal = sig.copy()

            resampled_data = mne.filter.resample(
                x = sig.data,
                up = self.upsample_factor,
                down = self.downsample_factor,
                n_jobs=-1,
                method = self.method
            )

            print(resampled_signal.timestamps)

            resampled_signal._data = resampled_data
            resampled_signal._timestamps = np.linspace(resampled_signal.timestamps[0],resampled_signal.timestamps[-1],resampled_data.shape[1])
            
            print(resampled_signal.timestamps)

            list_of_resampled_sigs.append(resampled_signal)
            
        if len(list_of_resampled_sigs) == 1:
            self.set_output(0, list_of_resampled_sigs[0])
        else:
            self.set_output(0, list_of_resampled_sigs)
          
class RemoveTrendNode(Node):
    title = 'Remove Trend from EEG'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        detrendType: str = Enum('high pass','high pass sinc','local detrend',desc='type of detrending to be performed')
        detrendCutoff:float = CX_Float(1.0,desc='the high-pass cutoff frequency to use for detrending (in Hz)')
        
    init_inputs = [PortConfig(label='data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='filtered data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    
    @property
    def config(self) -> RemoveTrendNode.Config:
        return self._config
    
    def init(self):
        self.detrend_type = self.config.detrendType
        self.detrend_cutoff = self.config.detrendCutoff
        
    def update_event(self,inp=-1):
        signal: StreamSignal = self.input(inp)
        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]
        
        list_of_filtered_sigs:Sequence = []
        for sig in signal:

            filtered_signal = sig.copy()

            sampling_freq = sig.info.nominal_srate
                    
            filtered_data = remove_trend_data(
                data = sig.data,
                sample_rate=sampling_freq,
                detrendType=self.detrend_type,
                detrendCutoff=self.detrend_cutoff
            )

            filtered_signal._data = filtered_data
            list_of_filtered_sigs.append(filtered_signal)
            
        if len(list_of_filtered_sigs) == 1:
            self.set_output(0, list_of_filtered_sigs[0])
        else:
            self.set_output(0, list_of_filtered_sigs)
    
class SignalToMNENode(Node):
    title = 'Data to MNE'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        montage:str = Enum('biosemi32','biosemi16','biosemi64','standard_1005','standard_1020') 
        
    init_inputs = [PortConfig(label='data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='MNE',allowed_data=mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    
    @property
    def config(self) -> SignalToMNENode.Config:
        return self._config
    
    def init(self):
        self.montage = self.config.montage
        
    def update_event(self,inp=-1):
        signal: StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_signals: Sequence = []
        for sig in signal:

            info = mne.create_info(
                ch_names = sig.labels,
                sfreq = sig.info.nominal_srate,
                ch_types = 'eeg'
            )
            
            raw_mne = mne.io.RawArray(
                data = sig.data,
                info = info
            )

            print(raw_mne,type(raw_mne))
            
            montage = mne.channels.make_standard_montage(self.config.montage)
            
            if self.config.montage == "biosemi32":
                missing_channels =  ['Af3', 'Fc1', 'Fc5', 'Cp1', 'Cp5', 'Po3', 'Po4', 'Cp6', 'Cp2', 'Fc6', 'Fc2', 'Af4']
                for x in range(len(missing_channels)):
                    for j in range(len(montage.ch_names)):
                        if len(montage.ch_names[j]) == 3 and montage.ch_names[j].lower().capitalize() == missing_channels[x]:
                            montage.ch_names[j] = missing_channels[x]

            raw_mne.set_montage(montage)
            list_of_signals.append(raw_mne)
        
        if len(list_of_signals) == 1:
            self.set_output(0,list_of_signals[0])            
        else:
            self.set_output(0,list_of_signals)

class MNEToSignalNode(Node):
    title = 'MNE to Signal'
    version = '0.1'

    class Config(NodeTraitsConfig):
        stream_name:str = CX_Str('Data Marker',desc='name of the stream')
        stream_type:str = CX_Str('Data',desc='type of stream')

    init_inputs = [PortConfig(label='MNE data',allowed_data=mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label='data',allowed_data=StreamSignal | Sequence[StreamSignal])]
    
    def init(self):
        pass

    @property
    def config(self) -> MNEToSignalNode.Config:
        return self._config

    def update_event(self,inp=-1):
        raws: mne.io.array.array.RawArray = self.input(inp).load_data()
        if not raws:
            return 

        if not isinstance(raws,list):
            raws = [raws]

        streamsignal_list = []
        for raw in raws:
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            channels = raw.info['ch_names']
            channel_dict = {i:channels[i] for i in range(len(channels))}

            info = StreamInfo(
                name = self.config.stream_name,
                type = self.config.stream_type,
                channel_count = len(channels),
                nominal_srate = sfreq,
                channel_format = 'float32',
                source_id = str(1)
            )

            stream_info = LSLSignalInfo(info)

            signal = StreamSignal(
                timestamps = np.linspace(0.0,data.shape[1] * (1/sfreq),data.shape[1]),
                data = data,
                labels = channels,
                signal_info = stream_info,
                make_lowercase = False
            )

            streamsignal_list.append(signal)

            print(data,signal.data,signal.timestamps)

        if len(streamsignal_list) == 1:
            self.set_output(0, streamsignal_list[0])
        else:
            self.set_output(0, streamsignal_list)

class RemovalLineNoisePrepNode(Node):
    title = 'Line Noise Removal PREP'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        line_freq:float = CX_Float(50.0,desc='line noise frequency')
                
    init_inputs = [PortConfig(label='data',allowed_data=mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label='filtered_data',allowed_data=mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    
    @property
    def config(self) -> RemovalLineNoisePrepNode.Config:
        return self._config
    
    def init(self):
        self.line_freq = self.config.line_freq
        
    def update_event(self,inp=-1):
        signal: mne.io.array.array.RawArray = self.input(inp)
        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_filtered_sigs:Sequence = []
        for sig in signal:

            raw_sig = sig.load_data()
         
            filtered_sig = line_noise_removal_prep(
                raw_eeg = raw_sig,
                sfreq = raw_sig.info['sfreq'],
                linenoise = np.arange(self.line_freq,raw_sig.info['sfreq']/2,self.line_freq)
            )
            
            list_of_filtered_sigs.append(filtered_sig)
        
        if len(list_of_filtered_sigs) == 1:
            self.set_output(0, list_of_filtered_sigs[0])
        else:
            self.set_output(0, list_of_filtered_sigs)

class ReferencingPrepNode(Node):
    title = 'Referencing EEG Prep'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        ref_chs:str|list = CX_Str('eeg')
        reref_chs:str|list = CX_Str('eeg')
        line_freqs:float = CX_Float(50.0,desc='line noise frequency')
        max_iterations:int = CX_Int(4)
        ransac:bool = Bool()
        channel_wise:bool = Bool()
    
    init_inputs = [PortConfig(label = 'data', allowed_data= mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label = 'referenced data',allowed_data = mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    
    @property
    def config(self) -> ReferencingPrepNode.Config:
        return self._config
    
    def init(self):
        self.ref_chs = self.config.ref_chs
        self.reref_chs = self.config.reref_chs
        self.line_freqs = self.config.line_freqs
        self.max_iterations = self.config.max_iterations
        self.ransac = self.config.ransac
        self.channel_wise = self.config.channel_wise

    def update_event(self,inp=-1):
        signal: mne.io.array.array.RawArray = self.input(inp)

        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_referenced_sigs:Sequence = []
        for sig in signal:

            raw_sig = sig.load_data() 
            
            sfreq = raw_sig.info['sfreq']
            
            self.line_freqs_list = np.arange(self.line_freqs, sfreq/2, self.line_freqs)
            print(self.line_freqs_list)
            
            referenced_eeg = referencing_prep(
                raw_eeg = raw_sig,
                ref_chs = self.ref_chs,
                reref_chs = self.reref_chs,
                line_freqs = self.line_freqs_list,
                max_iterations = self.max_iterations,
                ransac = self.ransac,
                channel_wise = self.channel_wise
            )

            list_of_referenced_sigs.append(referenced_eeg)
            
        if len(list_of_referenced_sigs) == 1:
            self.set_output(0, list_of_referenced_sigs[0])
        else:
            self.set_output(0, list_of_referenced_sigs)

class NoisyChannelPrepNode(Node):
    title = 'Noisy Channels Prep'
    version = '0.1'
    
    init_inputs = [PortConfig(label = 'data', allowed_data= mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label = 'noisy channel detected data',allowed_data = mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]

    def update_event(self,inp=-1):
        signal: mne.io.array.array.RawArray = self.input(inp).load_data()
        
        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_sigs = []
        for sig in signal:

            raw = sig.load_data()
                       
            new_eeg = noisychannelsprep(raw)

            print("NEW DATA",new_eeg)
            
            list_of_sigs.append(new_eeg)
        
        if len(list_of_sigs) == 1:
            self.set_output(0, list_of_sigs[0])
        else:
            self.set_output(0, list_of_sigs)

class InterpolateEEGNode(Node):
    title = 'Interpolate bad channels MNE'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        reset_bads:bool = Bool()
        mode:str = Enum('accurate','fast')
    
    init_inputs = [PortConfig(label = 'data', allowed_data= mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label = 'interpolated data',allowed_data = mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]

    @property
    def config(self) -> InterpolateEEGNode.Config:
        return self._config

    def update_event(self,inp=-1):
        signal: mne.io.array.array.RawArray = self.input(inp)

        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_sigs = []
        for sig in signal:
            
            raw = sig.load_data()

            new_eeg = raw.copy().interpolate_bads(
                reset_bads = self.config.reset_bads,
                mode = self.config.mode
            )

            print("NEW DATA",new_eeg)
            
            list_of_sigs.append(new_eeg)
        
        if len(list_of_sigs) == 1:
            self.set_output(0, list_of_sigs[0])
        else:
            self.set_output(0, list_of_sigs)
 
class RepairArtifactsICANode(Node):
    title = 'Repair Artifacts with ICA'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        n_components:int = CX_Int()
        method:str = Enum('fastica','infomax','picard')
    
    init_inputs = [PortConfig(label = 'data', allowed_data= mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label = 'cleaned data',allowed_data = mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]

    @property
    def config(self) -> RepairArtifactsICANode.Config:
        return self._config
    
    def init(self):
        self.n_components = self.config.n_components if self.config.n_components else None
        self.method = self.config.method

    def update_event(self,inp=-1):
        signal: mne.io.array.array.RawArray = self.input(inp).load_data()

        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_sigs = []
        for sig in signal:

            raw = sig.load_data()
            
            ica = mne.preprocessing.ICA(
                n_components=self.n_components,
                method=self.method,
                max_iter = 'auto',
                random_state=42
            )
            
            ica.fit(raw)
            ic_labels = label_components(raw,ica,method='iclabel')
            
            labels = ic_labels['labels']
            
            exclude_idx = [idx for idx,label in enumerate(labels) if label not in ['brain','other']]
            
            cleaned_signal = ica.apply(signal,exclude=exclude_idx)

            print("NEW DATA",cleaned_signal)
            
            list_of_sigs.append(cleaned_signal)

        if len(list_of_sigs) == 1:
            self.set_output(0, list_of_sigs[0])
        else:
            self.set_output(0, list_of_sigs)
            
class AverageReferenceNode(Node):
    title = 'Average Rereference on EEG'
    version = '0.1'
    
    init_inputs = [PortConfig(label = 'data', allowed_data=mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]
    init_outputs = [PortConfig(label = 'rereferenced data',allowed_data =mne.io.array.array.RawArray | Sequence[mne.io.array.array.RawArray])]

    def update_event(self,inp=-1):
        signal: mne.io.array.array.RawArray = self.input(inp)

        if not signal:
            return 
        
        if not isinstance(signal,list):
            signal = [signal]

        list_of_sigs = []
        for sig in signal:

            raw = sig.load_data()
            
            rereferenced_data = mne.set_eeg_reference(
                raw,
                ref_channels = 'average',
                copy=True
            )
            
            print("NEW DATA",rereferenced_data)

            list_of_sigs.append(rereferenced_data[0])

        if len(list_of_sigs) == 1:
            self.set_output(0, list_of_sigs[0])
        else:
            self.set_output(0, list_of_sigs)