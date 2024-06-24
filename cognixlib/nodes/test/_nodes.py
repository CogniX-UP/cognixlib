from __future__ import annotations
from random import randint
from sklearn import datasets
from cognixcore import Node, FrameNode, PortConfig
from cognixcore.config.traits import *
from traitsui.api import EnumEditor

import logging
from random import randint
import numpy as np
from collections.abc import Sequence

from pylsl import (
    StreamInlet, 
    StreamInfo,
    local_clock,
    IRREGULAR_RATE,
    cf_string
)
from ..stream import LSLSignalInfo
from ...scripting.data import StreamSignal

from ...scripting.data import Signal, TimeSignal, LabeledSignal, FeatureSignal

class TestStreamNode(FrameNode):
    
    title = "Classification Data Generator"
    version='0.1'
    
    init_outputs = [PortConfig(label='data'), PortConfig(label='class')]

    def __init__(self, params):
        super().__init__(params)
        iris = datasets.load_iris()
        self.X, self.y = iris.data, iris.target

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print(self.X_train.shape)
    
    def frame_update_event(self) -> bool:
        self.set_output(0, self.X)
        self.set_output(1, self.y)

class TestStreamSignalNode(FrameNode):
    title = 'Test Signal Node'
    version = '0.1'
    
    init_outputs = [PortConfig('stream signal',allowed_data=StreamSignal)]
    
    def init(self):
        self.current_time = 0
        
        self.info = StreamInfo(
                name = "Data Stream",
                type = 'EEG',
                channel_count = 32,
                nominal_srate = float(2048),
                channel_format = 'float32',
                source_id = str(1)
            )
        
        self.stream_info = LSLSignalInfo(self.info)
        
    def frame_update_event(self):
        self.current_time += self.player.delta_time
        if self.current_time >= 2:
            self.current_time = 0
            signal = StreamSignal(
                timestamps = np.linspace(start=1.0,stop=5.0,num=100),
                data = np.random.rand(32,100),
                labels = [f'channel_{i}' for i in range(32)],
                signal_info = self.stream_info,
                make_lowercase = False
            )
            self.set_output(0,signal)

class TestRandomGeneratorNode(FrameNode):
    
    title = "Random Number Frame Node"
    version = "0.1"
    
    init_outputs = [PortConfig('result')]
    
    def init(self):
        self.current_time = 0
        
    def frame_update_event(self):
        
        self.current_time += self.player.delta_time
        if self.current_time >= 5:
            self.current_time = 0
            self.set_output(0, randint(10, 50))
            
class TestRandomFeaturesNode(FrameNode):
    
    title = "Random Feature Node"
    version = "0.1"
    
    init_outputs = [PortConfig('result',allowed_data=FeatureSignal)]
    
    def init(self):
        self.current_time = 0
        
    def frame_update_event(self):
        
        self.current_time += self.player.delta_time
        if self.current_time >= 5:
            self.current_time = 0
            data = np.random.rand(100,40)
            classes = {
                '0':(0,50),
                '1':(0,50)
            }
            labels = [f'feature{i}' for i in range(40)]
            signal = FeatureSignal(data = data,class_dict=classes,signal_info=None,labels=labels)
            self.set_output(0, signal)

class TestRandomFeaturesNode1(FrameNode):
    
    title = "Random Feature Node 1"
    version = "0.1"
    
    init_outputs = [PortConfig('result',allowed_data=Sequence[LabeledSignal])]
    
    def init(self):
        self.current_time = 0
        
    def frame_update_event(self):
        
        self.current_time += self.player.delta_time
        if self.current_time >= 5:
            self.current_time = 0
            data = np.random.rand(50,32,100)
            labels = [f'feature{i}' for i in range(100)]
            signal = [LabeledSignal(data = data_,signal_info=None,labels=labels) for data_ in data]
            self.set_output(0, signal)

class TestRandomFeaturesNode2(FrameNode):
    
    title = "Random Feature Node 2"
    version = "0.1"
    
    init_outputs = [PortConfig('result',allowed_data=Sequence[LabeledSignal])]
    
    def init(self):
        self.current_time = 0
        
    def frame_update_event(self):
        
        self.current_time += self.player.delta_time
        if self.current_time >= 5:
            self.current_time = 0
            data = np.random.rand(50,32,100)
            labels = [f'feature{i}' for i in range(100)]
            signal = [LabeledSignal(data = data_,signal_info=None,labels=labels) for data_ in data]
            self.set_output(0, signal)

class TestLogNode(Node):
    """A node for testing log messages!"""
    title = "Logging Test"
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        
        msg_lvl: int = CX_Int(
            logging.NOTSET,
            editor=EnumEditor(
                values={
                    logging.NOTSET: "1:NOTSET",
                    logging.DEBUG: "2:DEBUG",
                    logging.INFO: "3:INFO",
                    logging.WARNING: "4:WARNING",
                    logging.ERROR: "5:ERROR",
                    logging.CRITICAL: "6:CRITICAL",    
                }
            ),
            desc="The logging level."
        )
        msg: str = CX_Str('some message', desc="The message to be sent")
        log_button = Button('LOG')
        
        @observe('log_button')
        def on_click(self, e):
            self.node.logger.log(self.msg_lvl, self.msg)
        
    @property
    def config(self) -> TestLogNode.Config:
        return self._config
    
    def start(self):
        self.logger.log(self.config.msg_lvl, self.config.msg) 
    
    def update_event(self, inp=-1):
        self.start()

class TestAddNode(Node):
    """A node for adding stuff together"""
    title = "Add Test Node"
    version = "0.1"
    
    init_inputs = [PortConfig('x'), PortConfig('y')]
    init_outputs = [PortConfig('result')]
        
    def init(self):
        self.values: dict[int, Any] = {}
    
    def update_event(self, inp=-1):
        self.values[inp] = self.input(inp)
        result = 0
        for value in self.values.values():
            result += value
        self.set_output(0, result) 
        
    
    

            

