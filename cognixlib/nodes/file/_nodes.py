from __future__ import annotations

from cognixcore import (
    Node,
    Flow,
    PortConfig,
    ProgressState,
)
from cognixcore.config.traits import *

import numpy as np
from pylsl import (
    StreamInlet, 
    StreamInfo,
    local_clock,
    IRREGULAR_RATE,
    cf_string
)

import os
import pyxdf
import json
import polars

from collections.abc import Mapping

from ...scripting.file.xdf import XDFWriter, XDFHeader
from ...scripting.data import StreamSignal
from ...scripting.data.conversions import get_lsl_format, lsl_to_str
from ..stream import LSLSignalInfo

class XDFWriterNode(Node):
    title = 'XDF Writer'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        default_filename: str = CX_Str("model", desc="the default file name")
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )
        
    init_inputs = [
        PortConfig(label='s0', allowed_data=StreamSignal), 
        PortConfig(label='s1', allowed_data=StreamSignal)
    ]
    
    def __init__(self, flow: Flow):
        super().__init__(flow)

        def add_stream():
            self.create_input(
                PortConfig(f's{len(self._inputs)}', allowed_data=StreamSignal)
            )
            
        self.add_generic_action(
            'add stream',
            add_stream
        )
        
        def remove_stream():
            len_inp = len(self._inputs)
            if len_inp > 1:
                self.delete_input(len_inp - 1)
            
        self.add_generic_action(
            'remove stream',
            remove_stream
        )
    
    @property
    def config(self) -> XDFWriterNode.Config:
        return self._config
    
    def init(self):
        self.inlets: dict[int, StreamInlet] = {}
        self.start_time = local_clock()
        self.stream_ids: set[int] = set()     
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname) 
        if not filename or isinstance(filename, str) == False:
            filename = self.config.default_filename
            
        if dir: 
            self.path = os.path.join(dir, filename)
        
        self.xdfile = XDFWriter(self.path, True)
    
    def stop(self):
        self.xdfile.close_file()
    
    def update_event(self,inp=-1):
          
        if inp not in self.stream_ids:
            
            signal: StreamSignal = self.input(inp)
            if signal is None:
                return False
            if (
                'Marker' in signal.info.signal_type and 
                (
                    signal.info.nominal_srate != IRREGULAR_RATE or 
                    signal.info.data_format != lsl_to_str[cf_string]
                )
            ):
                return 
            
            header = XDFHeader(
                name=signal.info.name,
                type_=signal.info.signal_type,
                channels=signal.labels.tolist(),
                nominal_srate=signal.info.nominal_srate,
                channel_format=signal.info.data_format,
                time_created=self.start_time
            )

            self.xdfile.add_stream(inp, header)
            self.stream_ids.add(inp)
        
        signal: StreamSignal = self.input(inp)
        if signal is None:
            return
        
        samples = signal.data
        timestamps = signal.timestamps
        
        self.xdfile.write_data(
            inp,
            samples,
            timestamps
        )
          
class XDFImporterNode(Node):
    title = 'XDF Import'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        default_file_name: str = CX_Str("filename")
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )
        lowercase_labels: bool = Bool(False, desc="if checked, makes all the incoming labels into lowercase")
        ports: PortList = Instance(
            PortList,
            lambda:PortList(
                list_type=PortList.ListType.OUTPUTS,
                min_port_count=1,
                out_params=PortList.Params(
                        allowed_data=StreamSignal
                    ),
                ), 
            style='custom'
        )
            
    init_outputs = [PortConfig(label='streams',allowed_data=Mapping[str, StreamSignal])]
        
    @property
    def config(self) -> XDFImporterNode.Config:
        return self._config
    
    def init(self):
        
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname) 
        if not filename or not isinstance(filename, str):
            self.logger.debug(f"Variable wasn't a string!")
            filename = self.config.default_file_name
        path_file = None
        
        if filename and dir:
            path_file = os.path.join(dir, f'{filename}.xdf')
    
        self.stream_collection: dict[str, StreamSignal] = {}
        
        if path_file and os.path.exists(path_file):
            streams, header = pyxdf.load_xdf(path_file)

            for stream in streams:
                stream_name = stream['info']['name'][0]
                stream_type = stream['info']['type'][0]
                stream_channel_count = stream['info']['channel_count'][0]
                stream_srate = stream['info']['nominal_srate'][0]
                stream_format = stream['info']['channel_format'][0]
                stream_id = stream['info']['stream_id']
                stream_data = stream['time_series']
                stream_timestamps = stream['time_stamps']
                
                if 'channels' in stream['info']:
                    channels_list = stream['info']['channels'][0]['channel']
                    stream_channels = [
                        val['label'][0] for val in channels_list
                    ]
                else:
                    stream_channels = [
                        f"channel_{i}" for i in range(int(stream_channel_count))
                    ]

                self.logger.debug(stream_channels)

                info = StreamInfo(
                    name = stream_name,
                    type = stream_type,
                    channel_count = int(stream_channel_count),
                    nominal_srate = float(stream_srate),
                    channel_format = get_lsl_format(stream_format),
                    source_id = str(stream_id)
                ) 
                
                stream_info = LSLSignalInfo(info)
                
                if isinstance(stream_data, Sequence):
                    stream_data = np.array(stream_data)
                    
                self.stream_collection[stream_name] = StreamSignal(
                    timestamps=stream_timestamps,
                    data=stream_data,
                    labels=stream_channels,
                    signal_info=stream_info,
                    make_lowercase=self.config.lowercase_labels
                )
        
        else:
            self.logger.error(msg=f'The path {path_file} doesnt exist')
              
    def update_event(self, inp=-1):
        self.set_output(0, self.stream_collection)
        self.logger.info(self.stream_collection.keys())
        
        valid_names = self.config.ports.valid_names
        for index, valid_name in enumerate(valid_names):
            stream = self.stream_collection.get(valid_name)
            if stream:
                self.set_output(index + 1, stream)
                                 
class SelectStreamNode(Node):
    title = 'Select Stream'
    version = '0.1' 

    class Config(NodeTraitsConfig):
        stream_name: str = CX_Str('stream name',desc='the stream name to get data')

    init_inputs = [PortConfig(label = 'streams',allowed_data=Mapping[str,StreamSignal])]
    init_outputs = [PortConfig(label = 'selected stream',allowed_data=StreamSignal)]

    @property
    def config(self) -> SelectStreamNode.Config:
        return self._config
    
    def init(self):
        self.dict_streams = None
        self.stream_name = self.config.stream_name
        
    def update_event(self, inp=-1):

        self.dict_streams = self.input(inp)

        if self.dict_streams and self.stream_name in self.dict_streams.keys():
            self.set_output(0,self.dict_streams[self.stream_name])
        
        
class DataFramePolarsNode(Node):
    title = 'DataFrame Import (Polars)'
    version = '0.1'

    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        default_file_name: str = CX_Str('filename')
        varname:str = CX_Str(
            'file name',
            desc='the file name will be extracted from this if there is a string variable'
        )
    
    init_outputs = [PortConfig(label='DataFrame',allowed_data=polars.dataframe.frame.DataFrame)]

    @property
    def config(self) -> DataFramePolarsNode.Config:
        return self._config

    def init(self):
        self.q = None
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname)
        if not filename or not isinstance(filename,str):
            self.logger.debug(f"Variable wasn't a string!")
            filename = self.config.default_file_name
        path_file = None

        if filename and dir:
            path_file = os.path.join(dir,f'{filename}.csv')

        if path_file and os.path.exists(path_file):
            self.q = polars.scan_csv(path_file)
        else:
            self.logger.error(msg=f'The path {path_file} doesnt exist')
            
    def update_event(self, inp=-1):
        if self.q is not None:
            df = self.q.collect()
            self.set_output(0,df)
            