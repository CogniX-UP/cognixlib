"""Defines the base class fot classifiers of sklearn"""

from abc import ABC, abstractmethod
from abc import ABC

from ..data.signals import FeatureSignal, LabeledSignal

class BasePredictor(ABC):
    """A base predictor is an object used for prediction."""
    
    @abstractmethod
    def fit(self, signal: FeatureSignal):
        """Trains/fits the model for prediction."""
        pass

    @abstractmethod
    def test(self, signal: FeatureSignal):
        """Tests the model using various methods."""
        pass
    
    @abstractmethod
    def predict(self, signal: LabeledSignal):
        """Predicts an output using a data sample"""
        pass

    @abstractmethod
    def save(self, path:str):
        """Saves the trained model to the path"""
        pass

    @abstractmethod
    def load(self, path:str):
        """Loads the model from the path"""
        pass
    

