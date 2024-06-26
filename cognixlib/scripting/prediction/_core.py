"""Defines the base class fot classifiers of sklearn"""

from abc import ABC, abstractmethod
from abc import ABC

from ..data.signals import Signal, FeatureSignal
from dataclasses import dataclass
from collections.abc import Sequence
from enum import IntFlag

class BasePredictor(ABC):
    """A contruct for any kind of prediction model."""
        
    @abstractmethod
    def fit(self, signal: FeatureSignal):
        """Trains/fits the model for prediction."""
        pass

    @abstractmethod
    def test(self, signal: FeatureSignal):
        """Tests the model using various methods."""
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
    
    class MetricFlag(IntFlag):
        """Determines what metrics a classifier will produce"""
        NONE=0
        ACCURACY=1
        PRECISION=2
        RECALL=4
        F1_SCORE=8
        
        @classmethod
        def all(cls):
            return cls.ACCURACY | cls.PRECISION | cls.RECALL | cls.F1_SCORE
        
    @dataclass
    class Metrics:
        """
        Metric results of models tesing / fit. Attributes can 
        be None, depending on the metric requested.
        """
        
        accuracy: float = None
        precision: float = None
        recall: float = None
        f1: float = None
        
        def __str__(self) -> str:
            res = 'Metrics:\n'
            if self.accuracy:
                res += f"Accuracy: {self.accuracy}\n"
            if self.precision:
                res += f"Precision: {self.precision}"
            if self.recall:
                res += f"Recall: {self.recall}"
            if self.f1:
                res += f"F1 Score: {self.f1}"
            
            return res
    
    def __init__(self, class_labels: Sequence[str] = None, m_flag: MetricFlag = MetricFlag.all()):
        """

        Args:
            class_labels (Sequence[str], optional): Ordered labels of the classes. Defaults to None.
        """
        super().__init__()
        self.class_labels = class_labels
        self.metric_flag = m_flag
        
    def fit(self, signal: FeatureSignal) -> Metrics | None:
        return self.fit_class(signal, self.metric_flag)
    
    @abstractmethod
    def fit_class(self, signal: FeatureSignal, m_flag: MetricFlag = None) -> Metrics | None:
        pass
    
    def predict(self, signal: Signal) -> Sequence[str]:
        return self.predict_class(signal, self.class_labels)
    
    @abstractmethod
    def predict_class(
        self, 
        signal: Signal, 
        class_labels: Sequence[str] | None = None
    ) -> Sequence[str]:
        """
        Each row of the :code:`LabeledSignal` is a data point to
        predict a class for.
        
        Returns:
            A sequence of the class predictions for each data point.
        """
        pass
    

