"""Defines the base class fot classifiers of sklearn"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from cognixcore import PortConfig
from abc import ABC
import numpy as np

from ..data.signals import FeatureSignal

class BasePredictor(ABC):

    @abstractmethod
    def train(self, f_signal_train: FeatureSignal):
        pass

    @abstractmethod
    def test(self, f_signal_test: FeatureSignal):
        pass

    def predict(self,f_signal_test: FeatureSignal):
        pass

    @abstractmethod
    def save_model(self, path:str):
        pass

    @abstractmethod
    def load_model(self, path:str):
        pass
    

