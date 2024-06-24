from __future__ import annotations

from pylsl import (
    resolve_stream,
    resolve_bypred,
    StreamInlet,
    StreamOutlet,
    StreamInfo,
    proc_clocksync , 
    proc_dejitter, 
    proc_monotonize,
    proc_threadsafe ,
)

from cognixcore import (
    Node,
    FrameNode,
    ProgressState, 
    PortConfig,
)
from cognixcore.config.traits import *
from threading import Thread
from traitsui.api import CheckListEditor

import numpy as np
from ...scripting.data import (
    StreamSignal, 
    StreamSignalInfo, 
    LabeledSignal,
    TimeSignal,
    Signal
)
from ...scripting.data.conversions import np_to_lsl, lsl_to_np, lsl_to_str

class LSLSignalInfo(StreamSignalInfo):
    
    def __init__(self, lsl_info: StreamInfo):
        self._lsl_info = lsl_info
        
        stream_xml = lsl_info.desc()
        chans_xml = stream_xml.child("channels")
        chan_xml_list = []
        ch = chans_xml.child("channel")
        while ch.name() == "channel":
            chan_xml_list.append(ch)
            ch = ch.next_sibling("channel")
            
        self.channels: dict[str, int] =  {
            chan_xml_list[c_index].child_value('label'): c_index 
            for c_index in range(len(chan_xml_list))
        }
        
        super().__init__(
            nominal_srate=lsl_info.nominal_srate(),
            signal_type=lsl_info.type(),
            data_format=lsl_to_str[lsl_info.channel_format()],
            name=lsl_info.name(),
        )

class LSLInputNode(FrameNode):
    """
    A node that awaits for an LSL stream to be found 
    and then passes the data to its output ports.
    """
    
    class Config(NodeTraitsConfig):
        
        stream_name: str = CX_Str('stream_name', desc='Stream name')
        stream_type: str = CX_Str('stream_type', desc = 'Stream type')
        search_action: str = Enum('name','type', desc='Filtering of streams based of specific value')    
        define_buffer:bool = Bool(True, desc='the creation of a buffer for the saving of the data samples')
        buffer_size: int = CX_Int(3276, desc='the size of the buffer in which the data samples are stored')
        lowercase_labels: bool = Bool(False, desc="if checked, makes all the incoming labels into lowercase")
        invert: bool = Bool(
            False, 
            desc='if checked, inverts the data i.e (channels x samples) => (samples x channels)'
        )
        debug: bool = Bool(False, desc="when true, logs debug messages")

        #### configuration for buffer size of data received
        #### (define buffer) Boolean for transformation from list to np.array (if False transform else keep)
        processing_flag_mode: int = List(
            editor=CheckListEditor(
                values=
                [
                    (proc_clocksync, 'synchronization'),  
                    (proc_dejitter, 'dejitter'),
                    (proc_monotonize, 'monotonize'),
                    (proc_threadsafe, 'threadsafe')
                ],
                cols=2
            ),
            style='custom'
        )

        
    
    title = 'LSL Input'
    version = '0.1'
    init_outputs = [PortConfig(label='data', allowed_data=StreamSignal)]
    
    def __init__(self, params):
        super().__init__(params)
        
        self.inlet: StreamInlet = None
        self.formats = ['double64','float32','int32',str,'int16','int8','int64']
        # The stream won't change after initialization
        self.cached_labels: list[str] = None 
    
    @property
    def config(self) -> LSLInputNode.Config:
        return self._config
    
    def init(self):
        self.t = None
        self.force_stop = False
        self.inlet = None
        self.signal_info: LSLSignalInfo = None
        self.buffer:np.ndarray | list = None
        
        ### Check boolean for buffer
        self.buffer_size = self.config.buffer_size
        self.define_buffer = self.config.define_buffer
        self.stream_name = self.config.stream_name
        self.stream_type = self.config.stream_type
        self.search_action = self.config.search_action
        self.processing_flag_mode = self.config.processing_flag_mode
        
        flags = 0
        for flag in self.processing_flag_mode:
            flags |= flag
        
        def _search_stream():
            
            self.progress = None
            self.progress = ProgressState(1,-1,'Searching stream')
            
            while True:
                
                if self.search_action == 'name':
                    results = resolve_bypred(f"name='{self.stream_name}'", 1, 3)
                elif self.search_action == 'type':
                    results = resolve_bypred(f"type='{self.stream_type}'", 1, 3)
                
                if results or self.force_stop:
                    break
            
            if results:

                flags = 0
                for flag in self.processing_flag_mode:
                    flags |= flag
                                                                 
                self.inlet = StreamInlet(results[0], processing_flags=flags)
                self.signal_info = LSLSignalInfo(self.inlet.info())
                
                labels = self.signal_info.channels.keys()
                self.cached_labels = (
                    list(labels) if not self.config.lowercase_labels
                    else [label.lower() for label in labels]
                )
                
                if self.define_buffer: 
                    
                    dtype = lsl_to_np[self.inlet.channel_format]
                    self.buffer = np.zeros(
                            (self.buffer_size * 2, int(self.inlet.channel_count)),
                            dtype=dtype
                        )
    
                self.set_progress_msg('Streaming!')
            
        self.t = Thread(target=_search_stream)
        self.t.start()

    def stop(self):
        
        self.force_stop = True
        if self.t:
            self.t.join()
            
        if self.inlet:
            self.inlet.close_stream()
            
        self.set_progress_value(-1,'Stopping stream...')
        self.progress = None
    
    def frame_update_event(self):
        if not self.inlet or (self.define_buffer and self.buffer is None):
            return
        
        if self.define_buffer:
            # numpy buffer used
            if self.formats[self.inlet.channel_format]!=str:
            
                _, timestamps = self.inlet.pull_chunk(max_samples=self.buffer_size * 2, dest_obj=self.buffer)
                if timestamps:
                    samples = self.buffer[:len(timestamps),:]
            # there's an issue with the str (marker) streams
            # since the marker streams are probably few, we
            # don't use the buffer
            else:
                samples, timestamps = self.inlet.pull_chunk()
                if timestamps:
                    samples = np.array(samples)
        
        else:
            samples, timestamps = self.inlet.pull_chunk()
            if timestamps:
                samples = np.array(samples)
        
        if self.config.debug and timestamps:
            self.logger.info(f'{timestamps}, {samples}')
            
        if not timestamps:
            return
        
        timestamps = np.array(timestamps, dtype='float64')
        
        if self.config.invert:
            samples = samples.T
            
        signal = StreamSignal(
            timestamps, 
            self.cached_labels,
            samples, 
            self.signal_info
        )
        self.set_output(0, signal)

class LSLOutputNode(Node):
    """An LSL Output Stream. This is an irregular stream."""
    
    title='LSL Output'
    version='0.1'
    
    class Config(NodeTraitsConfig):
        name: str = CX_Str('stream_name')
        type: str = CX_Str('')
        unit_type: str = CX_Str('')
    
    init_inputs = [
        PortConfig('in', allowed_data=Signal)
    ]
    
    @property
    def config(self) -> LSLOutputNode.Config:
        return self._config
    
    def init(self):
        self.stream_info: StreamInfo = None
        self.stream_out: StreamOutlet = None
    
    def stop(self):
        if self.stream_out:
            del self.stream_out
            self.stream_out = None
    
    def update_event(self, inp=-1):
        
        signal: Signal = self.input(0)
        if not signal:
            return        
        
        is_time_sig = isinstance(signal, TimeSignal)
        is_label_sig = isinstance(signal, LabeledSignal)
        
        if not self.stream_out:
            dtype = signal.data.dtype
            chann_count = (
                len(signal.labels)
                if is_label_sig
                else signal.data.shape[1]
            )
            
            self.stream_info = StreamInfo(
                self.config.name,
                self.config.type,
                channel_count= chann_count,
                channel_format=np_to_lsl[dtype],
            )
            
            if is_label_sig:
                desc = self.stream_info.desc()
                chann_info = desc.append_child('channels')
                for label in signal.labels:
                    channel = chann_info.append_child('channel')
                    channel.append_child_value('label', label)
                    if self.config.type:
                        channel.append_child_value('type', self.config.type)
                    if self.config.unit_type:
                        channel.append_child_value('unit', self.config.unit_type)
            
            self.stream_out = StreamOutlet(self.stream_info)
            print(f"Created an outlet with {chann_count} channels")
        
        
        if is_time_sig:
            self.stream_out.push_chunk(
                signal.data,
                signal.timestamps,
            )
        else:
            
            cols = signal.data.shape[1]
            for i in range(cols):
                data = signal.data[:,i]
                self.stream_out.push_sample(data)
        
        
        
