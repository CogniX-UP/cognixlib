from __future__ import annotations
from cognixcore import (
    Flow, 
    Node, 
    FrameNode, 
    PortConfig,
)
from cognixcore.config.traits import *
from traitsui.api import CheckListEditor
from ...scripting.data import FeatureSignal, LabeledSignal
from ...scripting.features.filter_bank import FilterBank
from ...scripting.features.csp import FBCSP

import joblib
import os
import numpy as np

class FBCSPTrainNode(Node):
    title = 'FBCSP Training'
    
    class Config(NodeTraitsConfig):
        n_filters: int = CX_Int(2,desc='the number of windows used for FBSCP')
        srate: float = CX_Float(2048.0,desc='sampling frequency of the signal')
        min_freq: float = CX_Float(0.0,desc='the minimum frequency in Hz in which the FBSCP functions -')
        max_freq: float = CX_Float(0.0,desc='the maximum frequency in Hz in which the FBSCP functions -')
        freq_bands_split: int = CX_Int(10,desc='how to split the frequency band')
        filename:str = CX_Str('filename',desc='the name of the model in which to save')
        f: str = File('Some path',desc='path of the model to import')
        save_button: bool = Bool()
    
    init_inputs = [PortConfig(label='data',allowed_data=FeatureSignal)]
    init_outputs = [PortConfig(label='features',allowed_data=FeatureSignal)]
    
    @property
    def config(self) -> FBCSPTrainNode.Config:
        return self._config 
        
    def init(self):
        self.srate = self.config.srate
        self.min_freq = self.config.min_freq if self.config.min_freq != 0.0 else None
        self.max_freq = self.config.max_freq if self.config.min_freq != 0.0 else None
        self.freq_splits = self.config.freq_bands_split if self.config.freq_bands_split!=0 else None
        self.n_filters = self.config.n_filters
        self.fbank = None
        self.filename = self.config.filename
        self.path_file = self.config.f
        self.save_button = self.config.save_button

        self.fbcsp_feature_extractor = FBCSP(self.n_filters)
        if self.path_file != 'Some path' and os.path.exists(self.path_file):
            self.fbcsp_feature_extractor = joblib.load(self.path_file)

    def stop(self):
        if self.save_button:
            joblib.dump(self.fbcsp_feature_extractor,f'{self.filename}.joblib')
        self.fbcsp_feature_extractor = None

    def update_event(self, inp=-1):
        signal: FeatureSignal = self.input(inp)
        signal = self.input(inp)
        if signal:

            if not self.fbank:       
                self.fbank = FilterBank(
                            fs = self.srate,
                            fmin = self.min_freq,
                            fmax = self.max_freq,
                            splits = self.freq_splits
                        )
                fbank_coeff = self.fbank.get_filter_coeff()
                
            data = signal.data
            classes = signal.classes
            
            labels = []
            
            for class_label, (start_idx, end_idx) in classes.items():
                for i in range(start_idx,end_idx):
                    labels.append(class_label)
            
            filtered_data = self.fbank.filter_data(data)
            self.fbcsp_feature_extractor.fit(filtered_data,labels)

            features = self.fbcsp_feature_extractor.transform(filtered_data)
            label_features = [f'feature_{i}' for i in range(features.shape[1])]
            
            signal_features = FeatureSignal(
                labels=label_features,
                class_dict = signal.classes,
                data = features,
                signal_info = None
            )
            self.set_output(0,signal_features)
            
class FBCSPTransformNode(Node):
    title = 'FBCSP Transform'
    
    class Config(NodeTraitsConfig):
        n_filters: int = CX_Int(2,desc='the number of windows used for FBSCP')
        srate: float = CX_Float(2048.0,desc='sampling frequency of the signal')
        min_freq: float = CX_Float(0.0,desc='the minimum frequency in Hz in which the FBSCP functions -')
        max_freq: float = CX_Float(0.0,desc='the maximum frequency in Hz in which the FBSCP functions -')
        freq_bands_split: int = CX_Int(10,desc='how to split the frequency band')
        f: str = File('Some path',desc='path of the model to import')
    
    init_inputs = [PortConfig(label='data',allowed_data=FeatureSignal)]
    init_outputs = [PortConfig(label='features',allowed_data=FeatureSignal)]
    
    @property
    def config(self) -> FBCSPTransformNode.Config:
        return self._config 
        
    def init(self):
        self.srate = self.config.srate
        self.min_freq = self.config.min_freq if self.config.min_freq != 0.0 else None
        self.max_freq = self.config.max_freq if self.config.min_freq != 0.0 else None
        self.freq_splits = self.config.freq_bands_split if self.config.freq_bands_split!=0 else None
        self.n_filters = self.config.n_filters
        self.fbank = None
        self.fbcsp_feature_extractor = None
        self.path_file = self.config.f

        self.file_exists = False
        if self.path_file != 'Some path' and os.path.exists(self.path_file):
            self.file_exists = True

    def update_event(self, inp=-1):
        signal: FeatureSignal = self.input(inp)

        if signal and self.file_exists:

            if not self.fbank:          
                self.fbank = FilterBank(
                            fs = self.srate,
                            fmin = self.min_freq,
                            fmax = self.max_freq,
                            splits = self.freq_splits
                        )
                fbank_coeff = self.fbank.get_filter_coeff()
                self.fbcsp_feature_extractor = joblib.load(self.path_file)

            data = signal.data
            classes = signal.classes
            
            labels = []
            
            for class_label, (start_idx, end_idx) in classes.items():
                for i in range(start_idx,end_idx):
                    labels.append(class_label)
            
            filtered_data = self.fbank.filter_data(data)

            features = self.fbcsp_feature_extractor.transform(filtered_data)
            label_features = [f'feature_{i}' for i in range(features.shape[1])]
            
            signal_features = FeatureSignal(
                labels=label_features,
                class_dict = signal.classes,
                data = features,
                signal_info = None
            )
            
            self.set_output(0,signal_features)            

class FBCSPTrainOnlineVersionNode(Node):
    title = 'FBCSP Training'
    
    class Config(NodeTraitsConfig):
        n_filters: int = CX_Int(2,desc='the number of windows used for FBSCP')
        srate: float = CX_Float(2048.0,desc='sampling frequency of the signal')
        min_freq: float = CX_Float(0.0,desc='the minimum frequency in Hz in which the FBSCP functions -')
        max_freq: float = CX_Float(0.0,desc='the maximum frequency in Hz in which the FBSCP functions -')
        freq_bands_split: int = CX_Int(10,desc='how to split the frequency band')
        filename:str = CX_Str('filename',desc='the name of the model in which to save')
        f: str = File('Some path',desc='path of the model to import')
        save_button: bool = Bool()
    
    init_inputs = [PortConfig(label='data_class1',allowed_data=Sequence[LabeledSignal]),PortConfig(label='data_class2',allowed_data=Sequence[LabeledSignal])]
    init_outputs = [PortConfig(label='features',allowed_data=FeatureSignal)]
    
    @property
    def config(self) -> FBCSPTrainOnlineVersionNode.Config:
        return self._config 
        
    def init(self):
        self.srate = self.config.srate
        self.min_freq = self.config.min_freq if self.config.min_freq != 0.0 else None
        self.max_freq = self.config.max_freq if self.config.min_freq != 0.0 else None
        self.freq_splits = self.config.freq_bands_split if self.config.freq_bands_split!=0 else None
        self.n_filters = self.config.n_filters
        self.fbank = None
        self.filename = self.config.filename
        self.path_file = self.config.f
        self.save_button = self.config.save_button

        self.fbcsp_feature_extractor = FBCSP(self.n_filters)
        if self.path_file != 'Some path' and os.path.exists(self.path_file):
            self.fbcsp_feature_extractor = joblib.load(self.path_file)

        self.signal1 = None
        self.signal2 = None
        
    def stop(self):
        if self.save_button:
            joblib.dump(self.fbcsp_feature_extractor,f'{self.filename}.joblib')
        self.fbcsp_feature_extractor = None

    def update_event(self, inp=-1):
        if inp==0: self.signal1: Sequence[LabeledSignal] = self.input(inp)
        if inp==1: self.signal2: Sequence[LabeledSignal] = self.input(inp)

        if self.signal1 and self.signal2:

            if not self.fbank:       
                self.fbank = FilterBank(
                            fs = self.srate,
                            fmin = self.min_freq,
                            fmax = self.max_freq,
                            splits = self.freq_splits
                        )
                fbank_coeff = self.fbank.get_filter_coeff()
            
            data = [sig1.data for sig1 in self.signal1] + [sig2.data for sig2 in self.signal2]
            data = np.array(data)
            
            classes = {
                '0':(0,len(self.signal1)),
                '1':(len(self.signal1),len(self.signal1)+len(self.signal2)),
            }
            
            class_labels = []
            
            for class_label, (start_idx, end_idx) in classes.items():
                for i in range(start_idx,end_idx):
                    class_labels.append(class_label)
            
            filtered_data = self.fbank.filter_data(data)
            self.fbcsp_feature_extractor.fit(filtered_data,class_labels)

            features = self.fbcsp_feature_extractor.transform(filtered_data)
            label_features = [f'fbcsp_{i}' for i in range(features.shape[1])]
            
            signal_features = FeatureSignal(
                labels=label_features,
                class_dict = classes,
                data = features,
                signal_info = None
            )
            
            self.set_output(0,signal_features)
            
            
class FBCSPTransformOnlineNode(Node):
    title = 'FBCSP Transform'
    
    class Config(NodeTraitsConfig):
        n_filters: int = CX_Int(2,desc='the number of windows used for FBSCP')
        srate: float = CX_Float(2048.0,desc='sampling frequency of the signal')
        min_freq: float = CX_Float(0.0,desc='the minimum frequency in Hz in which the FBSCP functions -')
        max_freq: float = CX_Float(0.0,desc='the maximum frequency in Hz in which the FBSCP functions -')
        freq_bands_split: int = CX_Int(10,desc='how to split the frequency band')
        f: str = File('Some path',desc='path of the model to import')
    
    init_inputs = [PortConfig(label='data_class1',allowed_data=Sequence[LabeledSignal]),PortConfig(label='data_class2',allowed_data=Sequence[LabeledSignal])]
    init_outputs = [PortConfig(label='features',allowed_data=FeatureSignal)]
    
    @property
    def config(self) -> FBCSPTransformOnlineNode.Config:
        return self._config 
        
    def init(self):
        self.min_freq = self.config.min_freq if self.config.min_freq != 0.0 else None
        self.max_freq = self.config.max_freq if self.config.min_freq != 0.0 else None
        self.freq_splits = self.config.freq_bands_split if self.config.freq_bands_split!=0 else None
        self.n_filters = self.config.n_filters
        self.fbank = None
        self.fbcsp_feature_extractor = None
        self.path_file = self.config.f
        self.srate = self.config.srate

        self.file_exists = False
        if self.path_file != 'Some path' and os.path.exists(self.path_file):
            self.file_exists = True
            
        self.signal1 = None
        self.signal2 = None
    
    def update_event(self, inp=-1):
        if inp==0: self.signal1: Sequence[LabeledSignal] = self.input(inp)
        if inp==1: self.signal2: Sequence[LabeledSignal] = self.input(inp)

        if self.signal1 and self.signal2 and self.file_exists:
    
            if not self.fbank:          
                self.fbank = FilterBank(
                            fs = self.srate,
                            fmin = self.min_freq,
                            fmax = self.max_freq,
                            splits = self.freq_splits
                        )
                fbank_coeff = self.fbank.get_filter_coeff()
                self.fbcsp_feature_extractor = joblib.load(self.path_file)
        
            data = [sig1.data for sig1 in self.signal1] + [sig2.data for sig2 in self.signal2]
            data = np.array(data)            
            
            classes = {
                '0':(0,len(self.signal1)),
                '1':(len(self.signal1),len(self.signal1)+len(self.signal2)),
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
            
            self.set_output(0,signal_features)