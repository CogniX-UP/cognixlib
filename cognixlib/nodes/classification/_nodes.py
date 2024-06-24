from __future__ import annotations
from cognixcore import Flow, Node, PortConfig
from cognixcore.config import NodeConfig
from cognixcore.config.traits import *

import os
import numpy as np

from ...scripting.data import LabeledSignal,FeatureSignal

from ...scripting.prediction.core import BasePredictor
from ...scripting.prediction.scikit import (
    SVMClassifier,
    RFClassifier,
    SciKitClassifier,
    LogisticRegressionClassifier,
    CrossValidation,
    LDAClassifier,
    KFoldClass,
    LeaveOneOutClass,
    StratifiedKFoldClass
)

class ModelNode(Node):
    """A node that outputs a model"""
    
    def __init__(self, flow: Flow, config: NodeConfig = None):
        super().__init__(flow, config)
        self.model = None
    
    def update_event(self, inp=-1):
        if self.model:
            self.set_output(0, self.model)

class SVMNode(ModelNode):
    title = 'SVM Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        C : float = CX_Float(1.0)
        degree: int = CX_Int(3)
        kernel: str = Enum('linear','poly','rbf','sigmoid','precomputed')
        gamma: str = Enum('scale','auto')
        gamma_float: float = CX_Float(0.0)
        selection_of_gamma: bool = Bool('gamma value from choices?')

    init_inputs = []
    init_outputs = [PortConfig(label='model', allowed_data=SciKitClassifier)]
    
    @property
    def config(self) -> SVMNode.Config:
        return self._config

    def init(self):
        self.params = {'C':self.config.C,
                        'degree':self.config.degree,
                        'kernel':self.config.kernel,
                        'gamma':self.config.gamma if self.config.selection_of_gamma else self.config.gamma_float
                        }
        self.model = SVMClassifier(self.params)
        self.set_output(0, self.model)
        
class LDANode(ModelNode):
    title = 'LDA Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        solver: str = Enum('svd','lsqr','eigen',desc='solver to use')
        shrinkage: float = CX_Float(0.5,desc='shrinkage parameter between 0 and 1')
        shrinkage_default: bool = Bool()

    init_inputs = []
    init_outputs = [PortConfig(label='model', allowed_data=SciKitClassifier)]
    
    @property
    def config(self) -> LDANode.Config:
        return self._config

    def init(self):
        self.params = {'solver':self.config.solver,
                        'shrinkage':self.config.shrinkage_default if self.config.shrinkage_default else self.config.shrinkage
                        }
        
        self.model = LDAClassifier(self.params)
        self.set_output(0, self.model)
    
class RandomForestNode(ModelNode):
    title = 'Random Forest Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        n_estimators : int = CX_Int(100,desc='the number of trees in the forest')
        criterion: str = Enum('gini','entropy','log loss',desc='the number of trees in the forest')
        max_depth: int = CX_Int(0,desc='the maximum depth of the tree')
        min_samples_split: int|float = CX_Int(2,desc='the minimum number of samples required to split an internal node')
        min_samples_leaf: int|float = CX_Int(1,desc='the minimum number of samples required to be at a leaf node')
        max_features: str = Enum('sqrt','log2',0,desc='the number of features to consider when looking for the best split')       
        max_leaf_nodes: int = CX_Int(0,desc='grow trees with max_leaf_nodes in best-first fashion')

    init_outputs = [PortConfig(label='model',allowed_data=SciKitClassifier)]
    init_inputs = []

    @property
    def config(self) -> RandomForestNode.Config:
        return self._config

    def init(self):
        self.params = {
            'n_estimators': self.config.n_estimators,
            'criterion':self.config.criterion,
            'max_depth':self.config.max_depth if self.config.max_depth !=0 else None,
            'min_samples_split':self.config.min_samples_split,
            'min_samples_leaf':self.config.min_samples_leaf,
            'max_features':self.config.max_features if self.config.max_features !=0 else None,
            'max_leaf_nodes':self.config.max_leaf_nodes if self.config.max_leaf_nodes !=0 else None,
        }

        print(self.params)
        
        self.model = RFClassifier(self.params)
        self.set_output(0, self.model)

class LogisticRegressionNode(ModelNode):
    title = 'Logistic Regression Classifier'
    version = '0.1'

    class Config(NodeTraitsConfig):
        penalty: str = Enum('l1','l2','elasticnet','None',desc='specify the norm of the penalty')
        tol:float = CX_Float(1e-4,desc='tolerance for stopping criteria')
        C:float = CX_Float(1.0,desc='inverse of regularization strength; must be a positive float')
        solver:str = Enum('lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga',desc='algorithm to use in the optimization problem')
        max_iter:int = CX_Int(100,desc='maximum number of iterations taken for the solvers to converge')

    init_outputs = [PortConfig(label='model',allowed_data=SciKitClassifier)]
    init_inputs = []

    @property
    def config(self) -> LogisticRegressionNode.Config:
        return self._config

    def init(self):
        self.params = {
            'penalty' : self.config.penalty,
            'tol' : self.config.tol,
            'C' : self.config.C,
            'solver' : self.config.solver,
            'max_iter' : self.config.max_iter,
        }

        self.model = LogisticRegressionClassifier(self.params)
        self.set_output(0, self.model)

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

    @property
    def config(self) -> TrainNode.Config:
        return self._config

    def init(self):

        self.signal = None
        self.classifier = None
        self.load_model = False
        self.model_exists = False

    def update_event(self, inp=-1):

        if inp == 0:self.signal = self.input(inp)
        if inp == 1 and not self.model_exists:self.classifier:SciKitClassifier = self.input(inp)

        if self.signal and self.classifier:

            self.model_exists = True

            (
                trained_model,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1
            ) = self.classifier.train(self.signal)
            
            metrics_signal = LabeledSignal(
                labels=['train_accuracy','train_precision','train_recall','train_f1'],
                data = np.array([train_accuracy,train_precision,train_recall,train_f1]),
                signal_info = None
            )
            
            self.set_output(0, self.classifier)
            self.set_output(1, metrics_signal)
            


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

            train_signal,test_signal = self.model.split_data(f_signal=signal,test_size=self.tt_split)
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
            self.classifier.save_model(self.path)
            
class LoadModel(Node):
    title = 'Load Model'
    version = '0.1'

    class Config(NodeTraitsConfig):
        directory: str = Directory(desc='the saving directory')
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
        path_file = None
        
        print(path_file)
        if filename and dir and isinstance(filename, str)!=False:
            path_file = f'{dir}/{filename}'
            
        print(path_file,os.path.exists(path_file))
        
        if path_file:
            self.classifier = SciKitClassifier(None)
            self.model = self.classifier.load_model(path=path_file)
            if self.model:
                self.set_output(0, SciKitClassifier(self.model))
            else:
                self.logger.error(msg='The path doesnt exist')
        
    def update_event(self, inp=-1):
        return super().update_event(inp)


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
            test_accuracy,test_precision,test_recall,test_f1 = self.model.test(f_signal_test=self.signal)
            
            metrics_signal = LabeledSignal(
                labels=['test_accuracy','test_precision','test_recall','test_f1'],
                data = np.array([test_accuracy,test_precision,test_recall,test_f1]),
                signal_info = None
            )
            
            self.set_output(0,metrics_signal)



class PredictNode(Node):
    title = 'Predictor Classifier'
    version = '0.1'

    init_inputs = [PortConfig(label='data',allowed_data=FeatureSignal), PortConfig(label='model',allowed_data=SciKitClassifier)]
    init_outputs = [PortConfig(label='predictions',allowed_data=LabeledSignal)]

    def init(self):

        self.signal = None
        self.model = None
        self.load_model = None

    def update_event(self, inp=-1):

        if inp == 0:self.signal = self.input(inp)
        if inp == 1:self.model:SciKitClassifier = self.input(inp)

        if self.signal and self.model:
            predictions = self.model.predict(f_signal_test=self.signal)
            
            metrics_signal = LabeledSignal(
                labels=['prediction'],
                data = np.array([predictions]),
                signal_info = None
            )
            
            self.set_output(0,metrics_signal)
