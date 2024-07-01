"""Defines the base class fot classifiers of sklearn"""

from abc import ABC, abstractmethod
from abc import ABC

from ..data.signals import Signal, FeatureSignal
from dataclasses import dataclass
from collections.abc import Sequence, Set, Mapping
from enum import IntFlag

class BasePredictor(ABC):
    """A contruct for any kind of prediction model."""
        
    @abstractmethod
    def fit(self, signal: FeatureSignal):
        """Trains/fits the model for prediction."""
        pass
    
    @abstractmethod
    def predict(self, signal: Signal):
        """Predicts an output using a data sample. Each row is a data point."""
        pass

    @abstractmethod
    def save(self, path:str):
        """Saves the trained model to the path"""
        pass

    @abstractmethod
    def load(self, path:str):
        """Loads the model from the path"""
        pass

class BaseClassifier(BasePredictor):
    """A construct for any kind of classification"""
    
    def __init__(self):
        
        super().__init__()
        
        self._classes = Sequence[str] = None
    
    @property
    def classes(self) -> Sequence[str]:
        return self._classes
    
    @abstractmethod
    def fit(self, signal: FeatureSignal):
        """Fits/trains the classifier based on a training set"""
        pass
    
    @abstractmethod
    def predict(self, signal: Signal) -> Sequence[str]:
        """
        Each row of the :code:`Signal` is a data point to
        predict a class for.
        
        Returns:
            A sequence of the class predictions for each data point.
        """
        pass

    @abstractmethod
    def predict_proba(self, signal: Signal):
        pass

class Validator(ABC):
    
    def __init__(
        self, 
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics
    
    @abstractmethod
    def score(
        self, 
        signal: FeatureSignal, 
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ) -> Mapping:
        """Returns the scores for various scoring functions."""
        pass
    
    
    

