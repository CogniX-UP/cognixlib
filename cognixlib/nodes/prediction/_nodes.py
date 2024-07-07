from __future__ import annotations
from cognixcore import Flow, Node, PortConfig
from cognixcore.config import NodeConfig
from cognixcore.config.traits import *

import os
import numpy as np

from ...scripting.data import LabeledSignal,FeatureSignal

from ...scripting.prediction._core import BasePredictor, BaseClassifier
from ...scripting.prediction.scikit.classification import (
    SVMClassifier,
    RFClassifier,
    SciKitClassifier,
    LogisticRegressionClassifier,
    LDAClassifier,
)
from ...scripting.prediction.scikit.validation import (
    Validator,
    KFoldValidator,
    LeaveOneOutValidator,
    StratifiedKFoldValidator,
)

class ModelNode(Node):
    """A node that outputs a model"""
    
    def __init__(self, flow: Flow, config: NodeConfig = None):
        super().__init__(flow, config)
        self.model = None
    
    def update_event(self, inp=-1):
        self.set_output(0, self.model)

class SVMNode(ModelNode):
    title = 'SVM Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        C : float = CX_Float(1.0)
        degree: int = CX_Int(3)
        kernel: str = Enum(SVMClassifier.Kernel.RBF, values=SVMClassifier.Kernel)
        gamma: str = Enum('scale','auto', 'manual')
        gamma_value: float = CX_Float(0.0, visible_when="gamma=='manual'")

    init_outputs = [PortConfig(label='model', allowed_data=SVMClassifier)]
    
    @property
    def config(self) -> SVMNode.Config:
        return self._config

    def init(self):
        
        gamma = self.config.gamma
        if gamma == 'manual':
            gamma = self.config.gamma_value
            
        self.model = SVMClassifier(
            C=self.config.C,
            kernel=self.config.kernel,
            degree=self.config.degree,
            gamma=gamma,
        )
        
class LDANode(ModelNode):
    title = 'LDA Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        solver: str = Enum('svd','lsqr','eigen',desc='solver to use')
        shrinkage: str = Enum('auto', 'manual')
        shrinkage_value: float = Range(0.5, min=0.0, max=0.0, visible_when="shrinkage=='manual'", desc='shrinkage value')

    init_inputs = []
    init_outputs = [PortConfig(label='model', allowed_data=LDAClassifier)]
    
    @property
    def config(self) -> LDANode.Config:
        return self._config

    def init(self):
        shrinkage = self.config.shrinkage
        if shrinkage == 'manual':
            shrinkage = self.config.shrinkage_value
            
        self.model = LDAClassifier(
            solver=self.config.solver,
            shrinkage=shrinkage
        )
    
class RandomForestNode(ModelNode):
    title = 'Random Forest Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        n_estimators : int = CX_Int(100,desc='the number of trees in the forest')
        criterion: str = Enum(RFClassifier.Criterion, desc='the number of trees in the forest')
        max_depth: str = Enum('auto', 'manual')
        max_depth_value: int = CX_Int(0, visible_when="max_depth=='manual'", desc='the maximum depth of the tree')
        mini_samples_split: int|float = CX_Int(2, desc='the minimum number of samples required to split an internal node')
        mini_samples_leaf: int|float = CX_Int(1, desc='the minimum number of samples required to be at a leaf node')
        max_features: str = Enum('sqrt','log2', 'manual', desc='the number of features to consider when looking for the best split')
        max_features_val: float = CX_Float(0)  
        max_leaf_nodes: str = Enum('auto', 'manual')
        max_leaf_nodes_val: int = CX_Int(0, visible_when="max_leaf_nodes=='manual'", desc='grow trees with max_leaf_nodes in best-first fashion')

    init_outputs = [PortConfig(label='model',allowed_data=RFClassifier)]

    @property
    def config(self) -> RandomForestNode.Config:
        return self._config

    def init(self):
        
        max_depth = None if self.config.max_depth == 'auto' else self.config.max_depth_value
        max_leaf_nodes = None if self.config.max_leaf_nodes == 'auto' else self.config.max_leaf_nodes_val
        max_features = self.config.max_features if self.config.max_features != 'manual' else self.config.max_features_val
        
        self.model = RFClassifier(
            n_estimators=self.config.n_estimators,
            criterion=self.config.criterion,
            max_depth=max_depth,
            mini_samples_split=self.config.mini_samples_split,
            mini_samples_leaf=self.config.mini_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes
        )

_LR = LogisticRegressionClassifier

class LogisticRegressionNode(ModelNode):
    title = 'Logistic Regression Classifier'
    version = '0.1'

    _LR = LogisticRegressionClassifier
    
    class Config(NodeTraitsConfig):
        penalty: str = Enum(_LR.Penalty.L2, values=_LR.Penalty, desc='specify the norm of the penalty')
        tol:float = CX_Float(1e-4, desc='tolerance for stopping criteria')
        C:float = CX_Float(1.0, desc='inverse of regularization strength; must be a positive float')
        solver:str = Enum(_LR.Solver, desc='algorithm to use in the optimization problem')
        max_iter:int = CX_Int(100, desc='maximum number of iterations taken for the solvers to converge')

    init_outputs = [PortConfig(label='model', allowed_data=LogisticRegressionClassifier)]

    @property
    def config(self) -> LogisticRegressionNode.Config:
        return self._config

    def init(self):
        
        penalty = None if self.config.penalty == _LR.Penalty.NONE else self.config.penalty
        self.model = LogisticRegressionClassifier(
            penalty=penalty,
            tol=self.config.tol,
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter
        )

class TrainNode(Node):

    title = 'Train Classifier'
    version = '0.1'
    
    init_inputs = [
        PortConfig(label='data',allowed_data=FeatureSignal),
        PortConfig(label='model',  allowed_data=SciKitClassifier)
    ]
    init_outputs = [
        PortConfig(label='model',allowed_data=SciKitClassifier),
        PortConfig(label='train metrics',allowed_data=LabeledSignal)
    ]

    def init(self):

        self.signal = None
        self.classifier: SciKitClassifier = None

    def update_event(self, inp=-1):

        if inp == 0:
            self.signal = self.input(inp)
        elif inp == 1:
            self.classifier: SciKitClassifier = self.input(inp)

        if self.signal is not None and self.classifier:
            
            self.classifier.fit(self.signal)
            self.set_output(0, self.classifier)


class TrainTestSplitNode(Node):
    title = 'Train Test Split'
    version = '0.1'

    class Config(NodeTraitsConfig):
        train_test_split:float= CX_Float(0.2,desc='split of data between train and test data')

    init_inputs = [PortConfig(label='data',allowed_data=FeatureSignal)]
    init_outputs = [PortConfig(label='train data',allowed_data=FeatureSignal),
                    PortConfig(label='test data',allowed_data=FeatureSignal)]

    @property
    def config(self) -> TrainTestSplitNode.Config:
        return self._config
    
    def init(self):

        self.tt_split = self.config.train_test_split
        self.model = SciKitClassifier(model = None)

    def update_event(self, inp=-1):

        signal = self.input(inp)
        
        if signal:

            train_signal,test_signal = self.model.split_data(signal=signal,test_size=self.tt_split)
            self.set_output(0, train_signal)
            self.set_output(1, test_signal)

class CrossValidationNode(Node):

    title = 'Cross Validation'
    version = '0.1'
    
    class Config(NodeTraitsConfig):
        folds: int = CX_Int(5,desc='the number of folds to split data for cross validation')
        splitter_type:str = Enum('KFold','Stratified','LeaveOneOut','ShuffleSplit')
        train_test_split:float= CX_Float(0.2,desc='split of data between train and test data')

    init_inputs = [PortConfig(label='data',allowed_data=FeatureSignal),PortConfig(label='model',allowed_data=SciKitClassifier)]
    init_outputs = [PortConfig(label = 'cv_metrics',allowed_data=LabeledSignal)]

    @property
    def config(self) -> CrossValidationNode.Config:
        return self._config

    def init(self):

        self.signal = None
        self.classifier = None
        self.load_model = False

        cv_class = next((cls for name, cls in CrossValidation.subclasses.items() if self.config.splitter_type in name), None)
        self.cv_model: CrossValidation = cv_class(
            kfold=self.config.folds, 
            train_test_split = self.config.train_test_split
        )
        print(self.cv_model)

    def update_event(self, inp=-1):

        if inp == 0:self.signal = self.input(inp)
        if inp == 1:self.classifier:SciKitClassifier = self.input(inp)

        if self.signal and self.classifier:

            cv_accuracy,cv_precision,cv_recall,cv_f1 = self.cv_model.calculate_cv_score(model=self.classifier.model,f_signal=self.signal)
            
            metrics_signal = LabeledSignal(
                labels=['cv_accuracy','cv_precision','cv_recall','cv_f1'],
                data = np.array([cv_accuracy,cv_precision,cv_recall,cv_f1]),
                signal_info = None
            )
            
            self.set_output(0,metrics_signal)

class SaveModel(Node):
    title = 'Save Model'
    version = '0.1'

    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        default_filename: str = CX_Str("model", desc="the default file name")
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )

    init_inputs = [PortConfig(label='model',allowed_data=SciKitClassifier)]

    def init(self):
        self.path = None
        self.classifier: SciKitClassifier = None
        
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname) 
        if not filename or isinstance(filename, str) == False:
            filename = self.config.default_filename
        
        if dir:
            self.path = os.path.join(dir, filename)
        
    @property
    def config(self) -> SaveModel.Config:
        return self._config

    def update_event(self, inp=-1):
        if inp == 0:
            self.classifier = self.input(0)      
    
    def stop(self):
        if self.classifier and self.path:
            self.classifier.save(self.path)
            
class LoadModel(Node):
    title = 'Load Model'
    version = '0.1'

    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
        default: str = CX_Str('model')
        varname: str = CX_Str(
            "model", 
            desc="the file name will be extracted from this if there is a string variable"
        )

    init_outputs = [PortConfig(label='model',allowed_data=SciKitClassifier)]

    @property
    def config(self) -> LoadModel.Config:
        return self._config
    
    def init(self):
        
        dir = self.config.directory
        filename = self.var_val_get(self.config.varname)
        if filename is None:
            filename = self.config.default 
        path_file = None
        
        path_file = os.path.join(dir, filename)
        print(path_file)
        
        if path_file:
            self.classifier = SciKitClassifier(None)
            self.classifier.load(path=path_file)
        
    def update_event(self, inp=-1):
        if self.classifier.model:
            self.set_output(0, self.classifier)
        else:
            self.logger.error(msg='The path doesnt exist')


class TestNode(Node):
    title = 'Test Classifier'
    version = '0.1'

    init_inputs = [
        PortConfig(label='data',allowed_data=FeatureSignal),
        PortConfig(label='model',allowed_data=SciKitClassifier)
    ]
    init_outputs = [PortConfig(label = 'test metrics',allowed_data=LabeledSignal)]

    def init(self):

        self.signal = None
        self.model = None
        self.load_model = None

    def update_event(self, inp=-1):

        if inp == 0:self.signal = self.input(inp)
        if inp == 1:self.model:SciKitClassifier = self.input(inp)

        if self.signal and self.model:
            test_accuracy,test_precision,test_recall,test_f1 = self.model.test(signal=self.signal)
            
            metrics_signal = LabeledSignal(
                labels=['test_accuracy','test_precision','test_recall','test_f1'],
                data = np.array([test_accuracy,test_precision,test_recall,test_f1]),
                signal_info = None
            )
            
            self.set_output(0,metrics_signal)



class ClassifyNode(Node):
    title = 'Classify'
    version = '0.1'

    init_inputs = [
        PortConfig(label='data', allowed_data=LabeledSignal), 
        PortConfig(label='model',allowed_data=BaseClassifier)
    ]
    init_outputs = [
        PortConfig(label='predictions',allowed_data=LabeledSignal)
    ]

    def init(self):

        self.signal = None
        self.model = None
        self.load_model = None

    def update_event(self, inp=-1):

        if inp == 0:self.signal = self.input(inp)
        if inp == 1:self.model:BaseClassifier = self.input(inp)

        if self.signal and self.model:
            predictions = self.model.predict(signal=self.signal)
            
            metrics_signal = LabeledSignal(
                labels=['prediction'],
                data = np.array([predictions]),
                signal_info = None
            )
            
            self.set_output(0,metrics_signal)
