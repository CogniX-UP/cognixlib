from __future__ import annotations

from cognixcore.config import NodeConfig
import numpy as np
import mne
import os,joblib

from ...scripting.data import (
    Signal,
    TimeSignal,
    LabeledSignal,
    FeatureSignal,
    StreamSignal
)

from cognixcore import (
    Flow, 
    Node, 
    FrameNode, 
    PortConfig,
)
from cognixcore.config.traits import *
from traitsui.api import CheckListEditor

from collections.abc import Sequence, Mapping
from ...scripting.statistics import *
from ...scripting.features.csp import FBCSP
from ...scripting.features.filter_bank import FilterBank

class MergeFeaturesNode(Node):
    """A node that merges features signals together."""
    
    title = 'Merge Features'
    version = '0.1'
    
    init_inputs = [
        PortConfig('f0', allowed_data=FeatureSignal | Sequence[FeatureSignal]),
        PortConfig('f1', allowed_data=FeatureSignal | Sequence[FeatureSignal])
    ]
    init_outputs = [
        PortConfig('merged', allowed_data=FeatureSignal)
    ]
    
    def __init__(self, flow: Flow):
        super().__init__(flow)
        
        def add_port():
            self.create_input(
                PortConfig(f'f{len(self._inputs)}', allowed_data=FeatureSignal | Sequence[FeatureSignal])
            )
        def remove_port():
            self.delete_input(len(self._inputs) - 1)
        
        self.add_generic_action('add input', add_port)
        self.add_generic_action('remove input', remove_port)
    
    def update_event(self, inp=-1):
        inp_count = len(self._inputs)
        f_signals: list[FeatureSignal] = []
        for i in range(inp_count):
            f_signal = self.input(i)
            if isinstance(f_signal, FeatureSignal):
                f_signals.append(f_signal)
            elif isinstance(f_signal, Sequence):
                f_signals.extend(f_signal)
        
        if not f_signals:
            return
        
        self.set_output(0, FeatureSignal.concat_classes(f_signals)) 
    
class FBCSPTrainRealTimeNode(Node):
    title = 'FBCSP Training'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        n_filters: int = CX_Int(2,desc='the number of windows used for FBSCP')
        srate: float = CX_Float(2048.0,desc='sampling frequency of the signal')
        min_freq: float = CX_Float(0.0,desc='the minimum frequency in Hz in which the FBSCP functions -')
        max_freq: float = CX_Float(0.0,desc='the maximum frequency in Hz in which the FBSCP functions -')
        freq_bands_split: int = CX_Int(10,desc='how to split the frequency band')
        
        directory: str = Directory(desc='the saving directory')
        default_filename: str = CX_Str("model", desc="the default file name")
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )
        
        save_button: bool = Bool()
    
    init_inputs = [
        PortConfig(label='data_class1', allowed_data=Sequence[LabeledSignal] | LabeledSignal),
        PortConfig(label='data_class2',allowed_data=Sequence[LabeledSignal] | LabeledSignal)
    ]
    init_outputs = [PortConfig(label='spatial filters',allowed_data=Mapping)]
    
    @property
    def config(self) -> FBCSPTrainRealTimeNode.Config:
        return self._config 
        
    def init(self):
        self.dict = dict()

        self.srate = self.config.srate
        self.min_freq = self.config.min_freq if self.config.min_freq != 0.0 else None
        self.max_freq = self.config.max_freq if self.config.min_freq != 0.0 else None
        self.freq_splits = self.config.freq_bands_split if self.config.freq_bands_split!=0 else None
        self.n_filters = self.config.n_filters
        self.fbank = None

        self.save_button = self.config.save_button

        self.signal1 = None
        self.signal2 = None
        
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname) 
        if not filename or isinstance(filename, str) == False:
            filename = self.config.default_filename
        self.path = os.path.join(dir, filename)
        
    def stop(self):
        print(self.fbank)
        if self.save_button and self.fbank:
            joblib.dump(value=self.dict,filename=f'{self.path}.joblib')
        self.fbcsp_feature_extractor = None
        
    def update_event(self, inp=-1):
        if inp==0: 
            self.signal1: Sequence[LabeledSignal] = self.input(inp)
            if isinstance(self.signal1, LabeledSignal):
                self.signal1 = [self.signal1]
        if inp==1: 
            self.signal2: Sequence[LabeledSignal] = self.input(inp)
            if isinstance(self.signal2, LabeledSignal):
                self.signal2 = [self.signal2]
            
        if self.signal1 and self.signal2:

            if not self.fbank: 
                print(self.path,os.path.exists(self.path + '.joblib'))
                      
                self.fbank = FilterBank(
                            fs = self.srate,
                            fmin = self.min_freq,
                            fmax = self.max_freq,
                            splits = self.freq_splits
                        )
                fbank_coeff = self.fbank.get_filter_coeff()
                self.dict['fbank'] = self.fbank
                
                self.fbcsp_feature_extractor = FBCSP(self.n_filters)
                
            min_size = min(min([sig1.data.shape[0] for sig1 in self.signal1]) , min([sig2.data.shape[0] for sig2 in self.signal2]))
            
            data1 = [_ for sig1 in self.signal1 for _ in (sig1.data.transpose()[:,:min_size] if sig1.data.ndim == 3 else [sig1.data.transpose()[:,:min_size]])] 
            data2 = [_ for sig2 in self.signal2 for _ in (sig2.data.transpose()[:,:min_size] if sig2.data.ndim == 3 else [sig2.data.transpose()[:,:min_size]])] 
            
            data = np.array(data1 + data2)          
            
            sum_of_data_class1 = (
                self.signal1[0].data.shape[0] if self.signal1[0].data.ndim == 3 else 1
                    if len(self.signal1) == 1 
                    else sum([sig1.data.shape[0] if sig1.data.ndim == 3 else 1 for sig1 in self.signal1])
                )
            
            sum_of_data_class2 = (
                self.signal2[0].data.shape[0] if self.signal2[0].data.ndim == 3 else 1
                    if len(self.signal2) == 1 
                    else sum([sig2.data.shape[0] if sig2.data.ndim == 3 else 1 for sig2 in self.signal2])
                )
            
            classes = {
                '0' : (0,sum_of_data_class1),
                '1' : (sum_of_data_class1,sum_of_data_class1 + sum_of_data_class2)
            }
                
            class_labels = []
            
            for class_label, (start_idx, end_idx) in classes.items():
                for i in range(start_idx,end_idx):
                    class_labels.append(class_label)
            
            filtered_data = self.fbank.filter_data(data)
            
            self.fbcsp_feature_extractor.fit(filtered_data,class_labels)

            self.dict['fbcsp_filters'] = self.fbcsp_feature_extractor

            self.signal1,self.signal2 = None,None

            self.set_output(0,self.dict)
            
            
class FBCSPTransformRealTimeNode(Node):
    title = 'FBCSP Transform'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )
    
    init_inputs = [
        PortConfig(label='data_class1',allowed_data=Sequence[LabeledSignal] | LabeledSignal),
        PortConfig(label='data_class2',allowed_data=Sequence[LabeledSignal] | LabeledSignal),
        PortConfig(label='spatial filters',allowed_data=Mapping)
    ]
    init_outputs = [PortConfig(label='features',allowed_data=FeatureSignal)]
    
    @property
    def config(self) -> FBCSPTransformRealTimeNode.Config:
        return self._config 
        
    def init(self):
        self.fbank = None
        self.fbcsp_feature_extractor = None
        self.signal1 = None
        self.signal2 = None
        self.dict = None
        self.file_exists = False
        
        self.path_file = None
        self.dict_file = None
        self.dict_node = None
        
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname) 
        
        if filename and dir and isinstance(filename, str)!=False:
            self.path_file = f'{dir}/{filename}'
        
        if self.path_file and os.path.exists(self.path_file + '.joblib'):
            self.file_exists = True
            self.dict_file = joblib.load(self.path_file + '.joblib')
        else:
            self.logger.error(msg='The path doenst exist')
        
    def update_event(self, inp=-1):
        if inp==0: 
            self.signal1: Sequence[LabeledSignal] = self.input(inp)
            if isinstance(self.signal1,LabeledSignal):
                self.signal1 = [self.signal1]
        
        if inp==1: 
            self.signal2: Sequence[LabeledSignal] = self.input(inp)
            if isinstance(self.signal2,LabeledSignal):
                self.signal2 = [self.signal2]
                
        if inp==2: 
            self.dict_node: Mapping = self.input(inp)

        print(type(self.signal1),type(self.signal2),type(self.dict_node),type(self.dict_file))
        if self.signal1 and self.signal2 and (self.dict_file or self.dict_node):
                        
            self.dict = self.dict_node if self.dict_node else self.dict_file
                             
            self.fbank = self.dict['fbank']
            
            self.fbcsp_feature_extractor = self.dict['fbcsp_filters']

            min_size = min(min([sig1.data.shape[0] for sig1 in self.signal1]) , min([sig2.data.shape[0] for sig2 in self.signal2]))
            
            data1 = [_ for sig1 in self.signal1 for _ in (sig1.data.transpose()[:,:min_size] if sig1.data.ndim == 3 else [sig1.data.transpose()[:,:min_size]])] 
            data2 = [_ for sig2 in self.signal2 for _ in (sig2.data.transpose()[:,:min_size] if sig2.data.ndim == 3 else [sig2.data.transpose()[:,:min_size]])] 
                      
            data = np.array(data1 + data2)          
            
            sum_of_data_class1 = (
                self.signal1[0].data.shape[0] if self.signal1[0].data.ndim == 3 else 1
                    if len(self.signal1) == 1 
                    else sum([sig1.data.shape[0] if sig1.data.ndim == 3 else 1 for sig1 in self.signal1])
                )
            
            sum_of_data_class2 = (
                self.signal2[0].data.shape[0] if self.signal2[0].data.ndim == 3 else 1
                    if len(self.signal2) == 1 
                    else sum([sig2.data.shape[0] if sig2.data.ndim == 3 else 1 for sig2 in self.signal2])
                )
            
            classes = {
                '0' : (0,sum_of_data_class1),
                '1' : (sum_of_data_class1,sum_of_data_class1 + sum_of_data_class2)
            }
            
            print(classes)            
            
            class_labels = []
            
            for class_label, (start_idx, end_idx) in classes.items():
                for i in range(start_idx,end_idx):
                    class_labels.append(class_label)
            
            filtered_data = self.fbank.filter_data(data)
            
            features = self.fbcsp_feature_extractor.transform(filtered_data)
            label_features = [f'feature_{i}' for i in range(features.shape[1])]
            
            signal_features = FeatureSignal(
                labels=label_features,
                class_dict = classes,
                data = features,
                signal_info = None
            )
            
            print(features.shape)

            self.signal1,self.signal2,self.dict_node = None,None,None
            
            self.set_output(0,signal_features)
            
    
class FBCSPTransformNodeForOneSignal(Node):
    title = 'FBCSP Transform Per Window'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )
    
    init_inputs = [
        PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal),
        PortConfig(label='spatial filters',allowed_data=Mapping)
    ]
    init_outputs = [PortConfig(label='features',allowed_data=FeatureSignal)]
    
    @property
    def config(self) -> FBCSPTransformNodeForOneSignal.Config:
        return self._config 
        
    def init(self):
        self.fbank = None
        self.fbcsp_feature_extractor = None
        self.signal = None
        self.dict = None
        self.file_exists = False
        
        self.path_file = None
        self.dict_file = None
        self.dict_node = None
        
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname) 
        
        if filename and dir and isinstance(filename, str)!=False:
            self.path_file = f'{dir}/{filename}'
        
        if self.path_file and os.path.exists(self.path_file + '.joblib'):
            self.file_exists = True
            self.dict_file = joblib.load(self.path_file + '.joblib')
        else:
            self.logger.error(msg='The path doenst exist')
        
    def update_event(self, inp=-1):
        if inp==0: 
            self.signal: Sequence[LabeledSignal] = self.input(inp)
            if isinstance(self.signal,LabeledSignal):
                self.signal = [self.signal]
        
        if inp==1: 
            self.dict_node: Mapping = self.input(inp)

        if self.signal and (self.dict_file or self.dict_node):
                        
            self.dict = self.dict_node if self.dict_node else self.dict_file
                             
            self.fbank = self.dict['fbank']
            
            self.fbcsp_feature_extractor = self.dict['fbcsp_filters']

            min_size = min([sig.data.shape[0] for sig in self.signal])
            
            data = [_ for sig in self.signal for _ in (sig.data.transpose()[:,:min_size] if sig.data.ndim == 3 else [sig.data.transpose()[:,:min_size]])]                  
            data = np.array(data)          
            
            sum_of_data_class = (
                self.signal[0].data.shape[0] if self.signal[0].data.ndim == 3 else 1
                    if len(self.signal) == 1 
                    else sum([sig.data.shape[0] if sig.data.ndim == 3 else 1 for sig in self.signal])
                )
            
            classes = {
                '0' : (0,sum_of_data_class),
            }

            class_labels = []
            
            for class_label, (start_idx, end_idx) in classes.items():
                for i in range(start_idx,end_idx):
                    class_labels.append(class_label)
            
            filtered_data = self.fbank.filter_data(data)
            
            features = self.fbcsp_feature_extractor.transform(filtered_data)
            label_features = [f'feature_{i}' for i in range(features.shape[1])]
            
            signal_features = FeatureSignal(
                labels=label_features,
                class_dict = classes,
                data = features,
                signal_info = None
            )
            
            print(features.shape)

            self.signal,self.dict_node = None,None
            
            self.set_output(0,signal_features)
     
     
     
     
     
     
     
     
     
     
     
     
     
            
class PSDMultitaperNode(Node):
    title = 'Power Spectral Density with Multitaper'
    version = '0.1'


    class Config(NodeTraitsConfig):
        fmin: float = CX_Float(0.0,desc='the lower-bound on frequencies of interest')
        fmax: float = CX_Float(desc='the upper-bound on frequencies of interest')
        bandwidth: float = CX_Float(desc='the frequency bandwidth of the multi-taper window funciton in Hz')
        normalization: str = Enum('full','length',desc='normalization strategy')
        output : str = Enum('power','complex')
        class_: str = CX_Str('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]

    @property
    def config(self) -> PSDMultitaperNode.Config:
        return self._config

    def init(self):
        self.fmin = self.config.fmin

        self.fmax = np.inf
        if self.config.fmax: self.fmax = self.config.fmax

        self.bandwidth = None
        if self.config.bandwidth: self.bandwidth = self.config.bandwidth

        self.normalization = self.config.normalization

        self.output = self.config.output

    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            results = mne.time_frequency.psd_array_multitaper(
                x = sig.data,
                sfreq = sig.info.nominal_srate,
                fmin = self.fmin,
                fmax = self.fmax,
                bandwidth= self.bandwidth,
                normalization= self.normalization,
                output = self.output,
                n_jobs= -1
            )

            labels = [f'feature_{freq}_Hz' for freq in results[1]]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,results[0].shape[0])},
                data = results[0],
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        print(list_of_features)
        
        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
        
class PSDWelchNode(Node):
    title = 'Power Spectral Density with Welch'
    version = '0.1'


    class Config(NodeTraitsConfig):
        fmin: float = CX_Float(0.0,desc='the lower-bound on frequencies of interest')
        fmax: float = CX_Float(desc='the upper-bound on frequencies of interest')
        n_fft: int = CX_Int(256, desc='The length of FFT used')
        n_overlap: int = CX_Int(0, desc='the number of points of overlap between segments')
        n_per_seg: int = CX_Int(desc='length of each Welch segment')
        average: str = Enum('mean','median','none',desc='how to average the segments')
        output: str = Enum('power','complex')
        class_: str = CX_Str('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]

    @property
    def config(self) -> PSDWelchNode.Config:
        return self._config

    def init(self):
        self.fmin = self.config.fmin

        self.fmax = np.inf
        if self.config.fmax: self.fmax = self.config.fmax

        self.n_fft = self.config.n_fft

        self.n_overlap = self.config.n_overlap

        self.n_per_seg = None
        if self.config.n_per_seg: self.n_per_seg = self.config.n_per_seg

        self.average = self.config.average if self.config.average!= 'none' else None

        self.output = self.config.output

    def update_event(self, inp=-1):
        signal:Signal =self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            results = mne.time_frequency.psd_array_welch(
                x = sig.data,
                sfreq = sig.info.nominal_srate,
                fmin = self.fmin,
                fmax = self.fmax,
                n_fft= self.n_fft,
                n_overlap= self.n_overlap,
                n_per_seg= self.n_per_seg,
                n_jobs= -1,
                output= self.output,
                window = 'hamming'
            )

            labels = [f'feature_{freq}_Hz' for freq in results[1]]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,results[0].shape[0])},
                data = results[0],
                signal_info = sig.info
            )

            list_of_features.append(signal_features)
        
        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class MorletTFNode(Node):
    title = 'Time-Frequency Representetion Morlet'
    version = '0.1'

    class Config(NodeTraitsConfig):
        n_cycles: int = CX_Int(7,desc='number of cycles in the wavelet')
        zero_mean: bool = Bool()
        use_fft: bool = Bool()
        output: str = Enum('complex','power','phase','avg_power','itc','avg_power_itc')
        f_min: float = CX_Float(0.0,desc='minimum frequency')
        f_max: float = CX_Float(0.0,desc='maximum frequency')
        f_splits: int = CX_Int(desc='number of splits')
        class_: str = CX_String('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='out',allowed_data = FeatureSignal | Sequence[FeatureSignal])]

    @property
    def config(self) -> MorletTFNode.Config:
        return self._config

    def init(self):
        self.n_cycles = self.config.n_cycles
        self.zero_mean = self.config.zero_mean
        self.use_fft = self.config.use_fft
        self.output = self.config.output
        self.fmin = self.config.f_min
        self.fmax = self.config.f_max if self.config.f_max else 0.0
        self.fplits = self.config.f_splits if self.config.f_splits else 0

    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            if self.fmax == 0.0: self.fmax = sig.info.nominal_srate
            if self.fplits == 0: self.fplits = int((self.fmax - self.fmin)/5)

            freqs = np.linspace(self.fmix,self.fmax,self.fplits)

            out = mne.time_frequency.tfr_array_morlet(
                data = sig.data,
                sfreq = sig.info.nominal_srate,
                freqs = freqs,
                n_cycles = self.n_cycles,
                zero_mean = self.zero_mean,
                use_fft = self.use_fft,
                output = self.output
            )

            if self.output in ['complex', 'phase', 'power']:
                out_reshaped = out.reshape(out.shape[0],out.shape[1] * out.shape[2] * out.shape[3])
                labels = [f'feature_channel_{chan}_{freq}_{n}' for chan in range(out.shape[1]) for freq in freqs for n in range(out.shape[3])]
            else:
                out_reshaped = out.reshape(out.shape[0],out.shape[1] * out.shape[2])
                labels = [f'feature_{freq}_{n}' for freq in freqs for n in range(out.shape[2])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,out_reshaped.shape[0])},
                data = out_reshaped,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class MultitaperTFNode(Node):
    title = 'Time-Frequency Representetion Multitaper'
    version = '0.1'

    class Config(NodeTraitsConfig):
        n_cycles: int = CX_Int(7,desc='number of cycles in the wavelet')
        zero_mean: bool = Bool()
        use_fft: bool = Bool()
        time_bandwidth: float = CX_Float(4.0,desc='product between the temporal window length (in seconds) and the full frequency bandwidth (in Hz).')
        output = str = Enum('complex','power','phase','avg_power','itc','avg_power_itc')
        f_min: float = CX_Float(0.0,desc='minimum frequency')
        f_max: float = CX_Float(0.0,desc='maximum frequency')
        f_splits: int = CX_Int(desc='number of splits')
        class_: str = CX_String('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='out',allowed_data = FeatureSignal | Sequence[FeatureSignal])]

    @property
    def config(self) -> MultitaperTFNode.Config:
        return self._config

    def init(self):
        self.n_cycles = self.config.n_cycles
        self.zero_mean = self.config.zero_mean
        self.time_bandwidth = self.config.time_bandwidth if self.config.time_bandwidth >= 2.0 else 4.0
        self.use_fft = self.config.use_fft
        self.output = self.config.output
        self.fmin = self.config.f_min
        self.fmax = self.config.f_max if self.config.f_max else 0.0
        self.fplits = self.config.f_splits if self.config.f_splits else 0

    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            if self.fmax == 0.0: self.fmax = sig.info.nominal_srate
            if self.fplits == 0: self.fplits = int((self.fmax - self.fmin)/5)

            freqs = np.linspace(self.fmix,self.fmax,self.fplits)

            out = mne.time_frequency.tfr_array_multitaper(
                data = sig.data,
                sfreq = sig.info.nominal_srate,
                freqs = freqs,
                n_cycles= self.n_cycles,
                zero_mean= self.zero_mean,
                time_bandwidth= self.time_bandwidth,
                use_fft= self.use_fft,
                output= self.output,
                n_jobs= -1
            )

            if self.output in ['complex', 'phase']:
                out_reshaped = out.reshape(out.shape[0],out.shape[1] * out.shape[2] * out.shape[3] * out.shape[4])
                labels = [f'feature_taper_{taper}_channel_{chan}_{freq}_{n}' for taper in range(out.shape[1]) for chan in range(out.shape[1]) for freq in freqs for n in range(out.shape[3])]
            elif self.output == 'power':
                out_reshaped = out.reshape(out.shape[0],out.shape[1] * out.shape[2] * out.shape[3])
                labels = [f'feature_channel_{chan}_{freq}_{n}' for chan in range(out.shape[1]) for freq in freqs for n in range(out.shape[3])]
            else:
                out_reshaped = out.reshape(out.shape[0],out.shape[1] * out.shape[2])
                labels = [f'feature_{freq}_{n}' for freq in freqs for n in range(out.shape[2])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,out_reshaped.shape[0])},
                data = out_reshaped,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class StockwellTFNode(Node):
    title = 'Time-Frequency Representetion Stockwell Transform'
    version = '0.1'

    class Config(NodeTraitsConfig):
        fmin: float = CX_Float(desc='the lower-bound on frequencies of interest')
        fmax: float = CX_Float(desc='the upper-bound on frequencies of interest')
        n_fft: int = CX_Int(desc='the length of the windows used for FFT')
        width: float = CX_Float(1.0,desc='the width of the Gaussian Window')
        class_: str = CX_Str('class name',desc='class name')
        return_itc: bool = Bool()

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='st_power',allowed_data = FeatureSignal | Sequence[FeatureSignal])]

    @property
    def config(self) -> StockwellTFNode.Config:
        return self._config

    def init(self):
        self.fmin = self.config.fmin if self.config.fmin else None
        self.fmax = self.config.fmax if self.config.fmax else None
        self.n_fft = self.config.n_fft if self.config.n_fft else None
        self.width = self.config.width
        self.return_itc = self.config.return_itc

    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            out = mne.time_frequency.tfr_array_stockwell(
                data = sig.data,
                sfreq = sig.info.nominal_srate,
                fmin = self.fmin,
                fmax= self.fmax,
                n_fft= self.n_fft,
                width = self.width,
                return_itc= False,
                n_jobs= -1
            )

            out_reshaped = out[0].reshape(out.shape[0],out.shape[1] * out.shape[2])
            labels = [f'frequency_{freq}_feature_{i}' for freq in out[2] for i in range(out[0].shape[2])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,sig.data.shape[0])},
                data = out_reshaped,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class STFTNode(Node):
    ## 2d array 

    title = 'Short Time Fourier Transform'
    version = '0.1'

    class Config(NodeTraitsConfig):
        wsize: int = CX_Int(4,desc='length of the STFT window in samples(must be a multiple of 4)')
        tstep: int = CX_Int(2,desc='step between successive windows in samples')
        class_:stt = CX_Str('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='out',allowed_data = FeatureSignal | Sequence[FeatureSignal])]

    @property
    def config(self) -> STFTNode.Config:
        return self._config

    def init(self):
        self.wsize = self.config.wsize
        self.tstep = self.config.tstep

    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            output = mne.time_frequency.stft(
                x = sig.data,
                wsize = self.wsize,
                tstep = self.tstep
            )

            out_reshaped = output.reshape(output.shape[0],output.shape[1] * output.shape[2])
            labels = [f'feature_{i}' for i in range(out_reshaped.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = out_reshaped,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class FourierCSDNode(Node):
    title = 'Cross-spectral density with Fourier Transform'
    version = '0.1'

    class Config(NodeTraitsConfig):
        fmin: float = CX_Float(0.0,desc='the lower-bound on frequencies of interest')
        fmax: float = CX_Float(desc='the upper-bound on frequencies of interest')
        t0: float = CX_Float(0,desc='time of the first sample relative to the onset of the epoch (in sec)')
        tmin: float = CX_Float(desc='minimum time instant to consider, in seconds')
        tmax: float = CX_Float(desc='maximum time instant to consider, in seconds')
        n_fft: int = CX_Int(desc='length of the fft')


    init_inputs = [PortConfig(label='data',allowed_data=Signal)]
    init_outputs = [PortConfig(label='csd')]

    @property
    def config(self) -> FourierCSDNode.Config:
        return self._config

    def init(self):
        self.fmin = self.config.fmin
        self.fmax = self.config.fmax if self.config.fmax else np.inf
        self.t0 = self.config.t0
        self.tmin = self.config.tmin if self.config.tmin else None
        self.tmax = self.config.tmax if self.config.tmax else None
        self.nfft = self.config.n_fft if self.config.n_fft else None

    def update_event(self, inp=-1):
        if inp == 0: signal:Signal =self.input_payload(inp)
        if signal:
            features = signal.copy()

            out = mne.time_frequency.csd_array_fourier(
                X = signal.data,
                sfreq = signal.info.nominal_srate,
                t0 = self.t0,
                fmin= self.fmin,
                fmax = self.fmax,
                tmin = self.tmin,
                tmax = self.tmax,
                ch_names = signal.info.channels.values(),
                n_fft= self.nfft,
                n_jobs= -1
            )

            self.set_output(0,out)



class MultitaperCSDNode(Node):
    title = 'Cross-spectral density with Multitaper Transform'
    version = '0.1'


    class Config(NodeTraitsConfig):
        fmin: float = CX_Float(0.0,desc='the lower-bound on frequencies of interest')
        fmax: float = CX_Float(desc='the upper-bound on frequencies of interest')
        t0: float = CX_Float(0,desc='time of the first sample relative to the onset of the epoch (in sec)')
        tmin: float = CX_Float(desc='minimum time instant to consider, in seconds')
        tmax: float = CX_Float(desc='maximum time instant to consider, in seconds')
        n_fft: int = CX_Int(desc='length of the fft')
        bandwidth:float = CX_Float(desc='the bandwidth of the multitaper windowing function in Hz')

    init_inputs = [PortConfig(label='data',allowed_data=Signal)]
    init_outputs = [PortConfig(label='csd')]

    @property
    def config(self) -> MultitaperCSDNode.Config:
        return self._config

    def init(self):
        self.fmin = self.config.fmin
        self.fmax = self.config.fmax if self.config.fmax else np.inf
        self.t0 = self.config.t0
        self.tmin = self.config.tmin if self.config.tmin else None
        self.tmax = self.config.tmax if self.config.tmax else None
        self.nfft = self.config.n_fft if self.config.n_fft else None
        self.bandwidth = self.config.bandwidth if self.config.bandwidth else None

    def update_event(self, inp=-1):
        if inp == 0: signal:Signal =self.input_payload(inp)
        if signal:
            features = signal.copy()

            out = mne.time_frequency.csd_array_multitaper(
                X = signal.data,
                sfreq = signal.info.nominal_srate,
                t0 = self.t0,
                fmin= self.fmin,
                fmax = self.fmax,
                tmin = self.tmin,
                tmax = self.tmax,
                ch_names = signal.info.channels.values(),
                n_fft= self.nfft,
                bandwidth = self.bandwidth,
                n_jobs= -1
            )

            self.set_output(0,out)



class MorletCSDNode(Node):
    title = 'Cross-spectral density with Multitaper Transform'
    version = '0.1'


    class Config(NodeTraitsConfig):
        t0: float = CX_Float(0,desc='time of the first sample relative to the onset of the epoch (in sec)')
        tmin: float = CX_Float(desc='minimum time instant to consider, in seconds')
        tmax: float = CX_Float(desc='maximum time instant to consider, in seconds')
        use_fft:bool = Bool()
        n_cycles: float = CX_Float(7,desc='number of cycles in the wavelet')

    init_inputs = [PortConfig(label='data',allowed_data=Signal),PortConfig(label='freqs')]
    init_outputs = [PortConfig(label='csd')]

    @property
    def config(self) -> MorletCSDNode.Config:
        return self._config

    def init(self):
        self.t0 = self.config.t0
        self.tmin = self.config.tmin if self.config.tmin else None
        self.tmax = self.config.tmax if self.config.tmax else None
        self.use_fft = self.config.use_fft
        self.n_cycles = self.config.n_cycles if self.config.n_cycles else None

    def update_event(self, inp=-1):
        if inp == 0: signal:Signal =self.input_payload(inp)
        if inp == 1: freqs = self.input_payload(inp)
        if signal and freqs:
            features = signal.copy()

            out = mne.time_frequency.csd_array_morlet(
                X = signal.data,
                sfreq= signal.info.nominal_srate,
                frequencies= freqs,
                ch_names= signal.info.channels.values(),
                t0 = self.t0,
                tmin = self.tmin,
                tmax = self.tmax,
                use_fft= self.use_fft,
                n_cycles= self.n_cycles,
                n_jobs= -1
            )

            self.set_output(0,out)


class MeanNode(Node):
    title = 'Mean'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> MeanNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'mean_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class VarNode(Node):
    title = 'Variance'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> VarNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'var_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class StdNode(Node):
    title = 'Standard deviation'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> StdNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'std_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class PTPNode(Node):
    title= 'Peak-to-Peak Value'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> PTPNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'ptp_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class SkewNode(Node):
    title = 'Skewness'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> SkewNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'skewness_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class KurtNode(Node):
    title = 'Kurtosis'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> KurtNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'kurtosis_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class RMSNode(Node):
    title = 'RMS value'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> RMSNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'rms_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
                       
class MobilityNode(Node):
    title = 'Hjorth Mobility'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> MobilityNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'mobility_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
               
class ComplexityNode(Node):
    title = 'Hjorth Complexity'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> ComplexityNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'complexity_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
                  
class Quantile75thNode(Node):
    title = '75th quantile'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> Quantile75thNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'q75th_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
             
class Quantile25thNode(Node):
    title = '25th quantile'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> Quantile25thNode.Config:
        return self._config
    
    def init(self):
        classes = list(Statistics.subclasses.values())
        self.func = self.stats_selected[0]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            func = self.func(data)
            feature = func.calculate_stat()
            
            labels = [f'q25th_value_channel_{chan}' for chan in range(feature.shape[1])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
                        
class StatisticsNode(Node):
    title = 'Basic Statistics'
    version = '0.1'

    class Config(NodeTraitsConfig):
        stats_selected: int = List(
            editor = CheckListEditor(
                values = [
                    (0,'mean'),
                    (1,'var'),
                    (2,'std'),
                    (3,'ptp'),
                    (4,'skewness'),
                    (5,'kurtosis'),
                    (6,'rms'),
                    (7,'mobility'),
                    (8,'complexity'),
                    (9,'75th quantile'),
                    (10,'25th quantile')
                ],
                cols=2
            ),
            style='custom'
        )
        class_:str = CX_Str('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    @property
    def config(self) -> StatisticsNode.Config:
        return self._config
    
    def init(self):
        self.stats = ['mean','var','std','ptp','skewness','kurtosis','rms','mobility','complexity','75th quantile','25th quantile']
        self.stats_selected = self.config.stats_selected
        classes = list(Statistics.subclasses.values())
        self.funcs = [classes[_] for _ in self.stats_selected]
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data

            if data.ndim == 2:
                data = data.reshape(1,data.shape[0],data.shape[1])

            features = np.zeros((data.shape[0],data.shape[1],len(self.funcs)))
            labels = [f'feature_{self.stats[stat]}' for stat in self.stats_selected]

            for index,func in enumerate(self.funcs):
                func = func(data)
                feature = func.calculate_stat()
                features[:,:,index] = feature

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,data.shape[0])},
                data = features,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        print

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class ApproximateEntopyNode(Node):
    #2d arrays only (n_channels,times)

    title = 'Approximate Entropy'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> ApproximateEntopyNode.Config:
        return self._config
    
    def init(self):
        pass
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            feature = univariate.compute_app_entropy(data)
            
            labels = [f'approximate_entropy_channel_{i}' for i in range(data.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class SampleEntropyNode(Node):
    title = 'Sample Entropy'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')

    def config(self) -> SampleEntropyNode.Config:
        return self._config
    
    def init(self):
        pass
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            feature = univariate.compute_samp_entropy(data)
            
            labels = [f'sample_entropy_channel_{i}' for i in range(data.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class SVDEntropyNode(Node):
    title = 'SVD Entropy'
    version = '0.1'
    
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    class Config(NodeTraitsConfig):
        class_: str = CX_String('class name',desc='class name')
        tau: int = CX_Int(2,desc='the delay (number of samples)')

    def config(self) -> SVDEntropyNode.Config:
        return self._config
    
    def init(self):
        pass
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            feature = univariate.compute_svd_entropy(data=data,tau=self.config.tau)
            
            labels = [f'svd_entropy_channel_{i}' for i in range(data.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class WaveletEnergyNode(Node):
    title = 'Wavelet Transform Energy'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        wavelet_name:str = Enum('haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10','morl','cmorl','gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8',desc='the wavelet name from which we calculate the wavelet coefficients and their energy')
        class_: str = CX_String('class name',desc='class name')

    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    @property
    def config(self) -> WaveletEnergyNode.Config:
        return self._config 
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            feature = univariate.compute_wavelet_coef_energy(data)
            
            labels = [f'wavelet_energy_channel_{i}' for i in range(feature.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)
            
class PLVNode(Node):
    title = 'Phase Locking Value (PLV)'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        include_diagonal: bool = Bool()
        class_: str = CX_String('class name',desc='class name')
        
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
        
    @property
    def config(self) -> PLVNode.Config:
        return self._config 
    
    def init(self):
        pass

    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            feature = bivariate.compute_phase_lock_val(data,include_diag=self.config.include_diagonal)
            
            labels = [f'plv_feature_{i}' for i in range(feature.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class CorrelationNode(Node):
    title = 'Pearson Correlation (time-domain)'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        include_diagonal: bool = Bool()
        return_eigenvalues: bool = Bool()
        class_: str = CX_String('class name',desc='class name')
        
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    @property
    def config(self) -> CorrelationNode.Config:
        return self._config 
    
    def init(self):
        pass
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            feature = bivariate.compute_time_corr(data,with_eigenvalues=self.config.return_eigenvalues,include_diag=self.config.include_diagonal)
            
            labels = [f'correlation_{i}' for i in range(feature.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class CorrelationSpectralNode(Node):
    title = 'Pearson Correlation (time-domain)'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        include_diagonal: bool = Bool()
        return_eigenvalues: bool = Bool()
        psd_method:str = Enum('welch','multitaper','fft')
        class_: str = CX_String('class name',desc='class name')
        
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
     
    @property
    def config(self) -> CorrelationSpectralNode.Config:
        return self._config 
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            
            feature = bivariate.compute_spect_corr(
                sfreq= signal.info.nominal_srate,
                data = data,
                with_eigenvalues= self.config.return_eigenvalues,
                include_diag= self.config.include_diagonal,
                psd_method= self.config.psd_method
            )

            labels = [f'spectral_correlation_feature_{i}' for i in range(feature.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)

class PowerSpectrumNode(Node):
    title = 'Power Spectrum'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        fmin: float = CX_Float(0.0,desc='minimum frequency for the calculation of the power spectrum')
        fmax: float = CX_Float(100.0,desc='maximum frequency for the calculation of the power spectrum')
        n_splits: int = CX_Int(5,desc='number of splits in the specified range of frequencies')
        normalization: bool = Bool()
        psd_method: str = Enum('welch','multitaper','fft',desc='method to calculate the power spectrum')
        log_calculation: bool = Bool()
        class_: str = CX_String('class name',desc='class name')
        
        
    init_inputs = [PortConfig(label='data',allowed_data=Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal])]
    init_outputs = [PortConfig(label='features',allowed_data = FeatureSignal | Sequence[FeatureSignal])]
    
    @property
    def config(self) -> PowerSpectrumNode.Config:
        return self._config
    
    def init(self):
        self.fmin = self.config.fmin if self.config.fmin else None
        self.fmax = self.config.fmax if self.config.fmax else None
        self.n_splits = self.config.n_splits if self.config.n_splits else None

        if not self.fmin or not self.fax or not self.n_splits:
            self.freq_bands = np.array([0.5, 4., 8., 13., 30., 100.])
        else:
            self.freq_bands = np.linspace(self.fmin,self.fmax,self.n_splits)
        
        self.normalization = self.config.normalization
        self.psd_method = self.config.psd_method
        self.log_calculation = self.config.log_calculation
    
    def update_event(self, inp=-1):
        signal:Sequence[LabeledSignal] | LabeledSignal | StreamSignal | Sequence[StreamSignal] = self.input(inp)

        if not signal:
            return
        
        if not isinstance(signal,Sequence):
            signal = [signal]
        
        list_of_features = []
        for sig in signal:

            data = sig.data
            
            feature = univariate.compute_pow_freq_bands(
                data = signal.data,
                sfreq = signal.info.nominal_srate,
                freq_bands = self.freq_bands,
                normalize = self.normalization,
                psd_method= self.psd_method,
                log = self.log_calculation
            )
            
            labels = [f'power_spectrum_feature_{i}' for i in range(feature.shape[0])]

            signal_features = FeatureSignal(
                labels=labels,
                class_dict = {f'{self.config.class_}':(0,1)},
                data = feature,
                signal_info = sig.info
            )

            list_of_features.append(signal_features)

        if len(list_of_features) == 1:
            self.set_output(0, list_of_features[0])
        else:
            self.set_output(0, list_of_features)


