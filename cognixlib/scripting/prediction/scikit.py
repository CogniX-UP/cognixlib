"""Defines the base oof scikit classifiers and the classifiers themselves"""
from __future__ import annotations
from abc import ABC, abstractmethod
from .core import BasePredictor
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
from ..data.signals import FeatureSignal

##### from .core import SignalInfo,Signal
### X_train and Y_train are to change to just a Signal object

class SciKitClassifier(BasePredictor):
    
    def __init__(self,model):
        self.model = model

    def train(self,f_signal_train: FeatureSignal):
    
        X_train = f_signal_train.data
        classes = f_signal_train.classes
        
        len_classes = len(list(classes.keys()))
        
        average_setting = 'binary'
        if len_classes > 2:
            average_setting = 'macro'

        class_labels = {list(classes.keys())[_]:_ for _ in range(len(list(classes.keys())))}
        
        Y_train = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for i in range(start_idx,end_idx):
                Y_train.append(class_label)

        Y_train = np.array(Y_train)

        Y_train = np.array([class_labels[class_] for class_ in Y_train])

        self.model.fit(X_train,Y_train)

        y_pred = self.model.predict(X_train)

        train_accuracy = accuracy_score(Y_train,y_pred)
        train_precision = precision_score(Y_train,y_pred,average=average_setting)
        train_recall = recall_score(Y_train,y_pred,average=average_setting)
        train_f1 = f1_score(Y_train,y_pred,average=average_setting)

        return self.model,train_accuracy,train_precision,train_recall,train_f1
    
    def predict(self,f_signal_test: FeatureSignal):
        X_test = f_signal_test.data
        classes = f_signal_test.classes

        class_labels = {_:list(classes.keys())[_] for _ in range(len(list(classes.keys())))}

        y_pred = self.model.predict(X_test)

        print(y_pred)

        y_predictions = [class_labels[pred] for pred in y_pred]

        print(y_predictions)

        return y_predictions


    def test(self,f_signal_test: FeatureSignal):        
        X_test = f_signal_test.data
        classes = f_signal_test.classes

        class_labels = {list(classes.keys())[_]:_ for _ in range(len(list(classes.keys())))}
        
        Y_test = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for i in range(start_idx,end_idx):
                Y_test.append(class_label)

        Y_test = np.array(Y_test)

        print(Y_test)

        Y_test = np.array([class_labels[class_] for class_ in Y_test])

        print(Y_test)

        y_pred = self.model.predict(X_test)

        test_accuracy = accuracy_score(Y_test,y_pred)
        test_precision = precision_score(Y_test,y_pred)
        test_recall = recall_score(Y_test,y_pred)
        test_f1 = f1_score(Y_test,y_pred)
        return test_accuracy,test_precision,test_recall,test_f1

    def split_data(self,f_signal:FeatureSignal,test_size:float):
        
        X = f_signal.data
        classes = f_signal.classes
              
        Y = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for i in range(start_idx,end_idx):
                Y.append(class_label)

        Y = np.array(Y)

        encoder = LabelEncoder()
        Y = encoder.fit_transform(Y)

        unique_classes = np.unique(Y)  

        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=1)

        count = 0
        features_train = []
        classes_train = {}
        for class_ in unique_classes:
            x = X_train[Y_train == class_]
            classes_train[class_] = (count,count + x.shape[0])
            count += x.shape[0]
            features_train.append(x)
            
        features_train = np.concatenate(features_train)
        
        train_signal = FeatureSignal(labels=f_signal.labels,class_dict=classes_train,data=features_train,signal_info=None)       
        
        count = 0
        features_test = []
        classes_test = {}
        for class_ in unique_classes:
            x = X_test[Y_test == class_]
            classes_test[class_] = (count,count + x.shape[0])
            count += x.shape[0]
            features_test.append(x)
            
        features_test = np.concatenate(features_test)

        test_signal = FeatureSignal(labels=f_signal.labels,class_dict=classes_test,data=features_test,signal_info=None)       

        
        return train_signal,test_signal
      
    def save_model(self,path:str):
        path = f"{path}.joblib"
        joblib.dump(self.model, filename=path)

    def load_model(self,path:str):
        path = f"{path}.joblib"
        print(path,os.path.exists(path))
        if os.path.exists(path):
            return joblib.load(path)   
        else:
            return False

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

    def __init__(self,kfold:int,train_test_split:float):
        self.kfold = kfold
        self.train_test_split = train_test_split
        self.average_setting = ''
        self.cv_model = None
        
    def calculate_cv_score(self,model,f_signal:FeatureSignal):
        X = f_signal.data
        classes = f_signal.classes
        
        len_classes = len(list(classes.keys()))
        
        if len_classes > 2:
            self.average_setting = '_macro'
        
        Y = []
            
        for class_label, (start_idx, end_idx) in classes.items():
            for i in range(start_idx,end_idx):
                Y.append(class_label)

        Y = np.array(Y)
        encoder = LabelEncoder()
        Y = encoder.fit_transform(Y)

        print(model,self.cv_model)
        
        cv_accuracy = cross_val_score(model,X,Y,cv=self.cv_model,scoring='accuracy').mean()
        cv_precision = cross_val_score(model,X,Y,cv=self.cv_model,scoring=f'precision{self.average_setting}').mean()
        cv_recall = cross_val_score(model,X,Y,cv=self.cv_model,scoring=f'recall{self.average_setting}').mean()
        cv_f1 = cross_val_score(model,X,Y,cv=self.cv_model,scoring=f'f1{self.average_setting}').mean()

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
    