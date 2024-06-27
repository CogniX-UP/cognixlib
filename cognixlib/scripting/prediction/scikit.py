"""Defines the base for scikit classifiers and the classifiers themselves"""
from __future__ import annotations
from ._core import BasePredictor, BaseClassifier
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    ShuffleSplit,
    train_test_split
)
import os
from sklearn.preprocessing import LabelEncoder
import joblib
from ..data.signals import FeatureSignal, Signal

class SciKitClassifier(BaseClassifier):
    """The base class for a `SciKit <https://scikit-learn.org/stable/index.html>`_ classifier"""
    
    def __init__(
        self, 
        model: BaseEstimator | ClassifierMixin,
        m_flag = BaseClassifier.MetricFlag.all()
    ):
        super().__init__(m_flag)
        self._model = model
    
    @property
    def model(self) -> BaseEstimator | ClassifierMixin:
        return self._model
    
    def fit(
        self, 
        signal: FeatureSignal, 
    ):
        
        X = signal.data
        classes = signal.classes
        
        # we assume the feature signal is sorted
        y = []
        for class_label, (start_idx, end_idx) in classes.items():
            for _ in range(start_idx, end_idx):
                y.append(class_label)

        # In scikit-learn, X and Y can be python lists, dataframes or numpy arrays
        self.model.fit(X, y)

    def predict(self, signal: Signal):
        X = signal.data
        return self.model.predict(X)
      
    def save(self, path:str):
        path = f"{path}.joblib"
        joblib.dump(self, filename=path)

    def load(self, path:str):
        path = f"{path}.joblib"
        self.__dict__.update(joblib.load(path))

class SVMClassifier(SciKitClassifier):

    def __init__(self,params:dict):
        super().__init__(model = SVC(**params))
        
class RFClassifier(SciKitClassifier):

    def __init__(self,params:dict):
        super().__init__(model = RandomForestClassifier(**params))
        
class LogisticRegressionClassifier(SciKitClassifier):

    def __init__(self,params:dict):
        super().__init__(model = LogisticRegression(**params))

class LDAClassifier(SciKitClassifier):

    def __init__(self,params:dict):
        super().__init__(model = LinearDiscriminantAnalysis(**params))
   
class CrossValidation:
    
    subclasses = {}
    
    def __init_subclass__(cls,**kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, kfold: int, train_test_split: float):
        self.kfold = kfold
        self.train_test_split = train_test_split
        self.average_setting = ''
        self.cv_model = None
        
    def calculate_cv_score(self, model, f_signal: FeatureSignal):
        X = f_signal.data
        classes = f_signal.classes
        
        len_classes = len(list(classes.keys()))
        
        if len_classes > 2:
            self.average_setting = '_macro'
        
        y = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for _ in range(start_idx,end_idx):
                y.append(class_label)

        cv_accuracy = cross_val_score(model, X, y, cv=self.cv_model, scoring='accuracy').mean()
        cv_precision = cross_val_score(model, X, y, cv=self.cv_model, scoring=f'precision{self.average_setting}').mean()
        cv_recall = cross_val_score(model, X, y, cv=self.cv_model, scoring=f'recall{self.average_setting}').mean()
        cv_f1 = cross_val_score(model, X ,y , cv=self.cv_model, scoring=f'f1{self.average_setting}').mean()

        return cv_accuracy,cv_precision,cv_recall,cv_f1
           

class KFoldClass(CrossValidation):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = KFold(n_splits=kfold)
    
class StratifiedKFoldClass(CrossValidation):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = StratifiedKFold(n_splits=kfold)
    
class LeaveOneOutClass(CrossValidation):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = LeaveOneOut()
    
class ShuffleSplitClass(CrossValidation):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = ShuffleSplit(train_size=1-train_test_split,test_size=train_test_split,n_splits=kfold)
    