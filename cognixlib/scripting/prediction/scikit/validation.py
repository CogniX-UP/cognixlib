"""Defines various cross validation classes that wrap scikit."""

from collections.abc import Mapping
from sklearn.model_selection import (
    cross_validate,
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    # TODO add more
)
from ...data.signals import FeatureSignal
from .._core import Validator, BasePredictor
from collections.abc import Sequence

class ScikitValidator(Validator):

    def __init__(
        self,
        validator,
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ):
        super().__init__(predictor, metrics)
        self.validator = validator
    
    def score(
        self, 
        signal: FeatureSignal, 
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None
    ) -> tuple[Mapping, Mapping]:
        
        model = predictor if predictor else self.predictor
        scoring = metrics if metrics else self.metrics
        
        X = signal.data
        classes = signal.classes
        
        y = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for _ in range(start_idx,end_idx):
                y.append(class_label)

        cross_res: Mapping = cross_validate(model, X, y, cv=self.validator, scoring=scoring)
        res = {
            name: val.mean()
            for name, val in cross_res.items()
        }
        return res, cross_res

class KFoldValidator(ScikitValidator):
    
    def __init__(
        self,
        kfold=5,
        shuffle=False,
        random_state: int = None,
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ):
        self.validator = KFold(
            n_splits=kfold,
            shuffle=shuffle,
            random_state=random_state
        )
        super().__init__(self.validator, predictor, metrics)

class RepeatedKFoldValidator(ScikitValidator):
    
    def __init__(
        self,
        kfold: int = 5,
        repeats: int = 10,
        random_state: int = None,
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ):
        self.validator = RepeatedKFold(
            n_splits=kfold,
            n_repeats=repeats,
            random_state=random_state
        )
        super().__init__(self.validator, predictor, metrics)
        
class StratifiedKFoldValidator(ScikitValidator):
    
    def __init__(
        self,
        kfold=5,
        shuffle=False,
        random_state: int = None,
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ):
        self.validator = StratifiedKFold(
            n_splits=kfold,
            shuffle=shuffle,
            random_state=random_state
        )
        super().__init__(self.validator, predictor, metrics)
    
class LeaveOneOutValidator(ScikitValidator):
    
    def __init__(
        self,
        predictor: BasePredictor = None,
        metrics: Sequence[str] = None,
    ):
        super().__init__(LeaveOneOut(), predictor, metrics)

class LeavePOutValidator(ScikitValidator):
    
    def __init__(
        self, 
        p: int,
        predictor: BasePredictor = None, 
        metrics: Sequence[str] = None,
    ):
        super().__init__(LeavePOut(p), predictor, metrics)
    
class ShuffleSplitValidator(ScikitValidator):
    
    def __init__(
        self,
        n_splits=10,
        test_percent: float = None,
        random_state: int = None,
        predictor: BasePredictor = None, 
        metrics: Sequence[str] = None,
    ):
        train_percent = None
        if test_percent is not None:
            train_percent = 1 - test_percent
        self.validator = ShuffleSplit(
            n_splits=n_splits,
            test_size=test_percent,
            train_size=train_percent,
            random_state=random_state
        )
        
        super().__init__(self.validator, predictor, metrics)