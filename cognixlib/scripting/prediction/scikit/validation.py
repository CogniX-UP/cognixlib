"""Defines various cross validation classes that wrap scikit."""

from collections.abc import Mapping
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    ShuffleSplit,
    train_test_split
)
from ...data.signals import FeatureSignal
from .._core import Validator, BasePredictor


class ScikitValidator(Validator):

    def __init__(
        self,
        predictor: BasePredictor = None
    ):
        self.predictor = predictor
    
    def score(self, signal: FeatureSignal, predictor: BasePredictor = None) -> Mapping:
        
        model = predictor if predictor else self.predictor
        X = signal.data
        classes = signal.classes
        
        len_classes = len(list(classes.keys()))
        
        if len_classes > 2:
            self.average_setting = '_macro'
        
        y = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for _ in range(start_idx,end_idx):
                y.append(class_label)

        cv_accuracy = cross_val_score(model, X, y, cv=self._cv_model, scoring='accuracy').mean()
        cv_precision = cross_val_score(model, X, y, cv=self._cv_model, scoring=f'precision{self.average_setting}').mean()
        cv_recall = cross_val_score(model, X, y, cv=self._cv_model, scoring=f'recall{self.average_setting}').mean()
        cv_f1 = cross_val_score(model, X ,y , cv=self._cv_model, scoring=f'f1{self.average_setting}').mean()

        return cv_accuracy,cv_precision,cv_recall,cv_f1
           

class KFoldValidator(ScikitValidator):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = KFold(n_splits=kfold)
    
class StratifiedKFoldValidator(ScikitValidator):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = StratifiedKFold(n_splits=kfold)
    
class LeaveOneOutValidator(ScikitValidator):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = LeaveOneOut()
    
class ShuffleSplitValidator(ScikitValidator):
    
    def __init__(self,kfold:int,train_test_split:float,binary_classification:bool):
        super().__init__(kfold,train_test_split, binary_classification)
        self.cv_model = ShuffleSplit(train_size=1-train_test_split,test_size=train_test_split,n_splits=kfold)