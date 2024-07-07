"""
This module wraps various classes and utilities from `scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_ into a more friendly set-up. The classification wrappers are based on `:class:cognixlib.prediction.BaseClassifier`.
"""
from __future__ import annotations
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from enum import StrEnum

import joblib
import numpy as np

from ...data.signals import FeatureSignal, Signal
from .._core import BasePredictor, BaseClassifier

from typing import Literal

# Metrics
class ClassificationMetrics(StrEnum):
    """
    Defines all the classification metrics as strings as given by `scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_
    """
    ACCURACY = 'accuracy'
    BALANCED_ACCURACY = 'balanced_accuracy'
    TOP_K_ACCURACY = 'top_k_accuracy'
    AVG_PRECISION = 'average_precision'
    NEG_BRIER_SCORE = 'neg_brier_score'
    F1 = 'f1'
    F1_MICRO = 'f1_micro'
    F1_MACRO = 'f1_macro'
    F1_WEIGHTED = 'f1_weighted'
    F1_SAMPLES = 'f1_samples'
    NEG_LOG_LOSS = 'neg_log_loss' # requires predict_proba
    PRECISION = 'precision'
    PRECISION_MICRO ='precision_micro'
    PRECISION_MACRO = 'precision_macro'
    PRECISION_WEIGHTED = 'precision_weighted'
    PRECISION_SAMPLES = 'precision_samples'
    RECALL = 'recall'
    RECALL_MICRO = 'recall_micro'
    RECALL_MACRO = 'recall_macro'
    RECALL_WEIGHTED = 'recall_weighted'
    RECALL_SAMPLES = 'recall_samples'
    JACCARD = 'jaccard'
    JACCARD_MICRO = 'jaccard_micro'
    JACCARD_MACRO = 'jaccard_macro'
    JACCARD_WEIGHTED = 'jaccard_weighted'
    JACCARD_SAMPLES = 'jaccard_samples'
    ROC_AUC = 'roc_auc'
    ROC_AUC_OVR = 'roc_auc_ovr'
    ROC_AUC_OVO = 'roc_auc_ovo'
    ROC_AUC_OVR_WEIGHTED = 'roc_auc_ovr_weighted'
    ROC_AUC_OVO_WEIGHTED = 'roc_auc_ovo_weighted'
    D2_LOG_LOSS_SCORE = 'd2_log_loss_score'
    
# Classifiers
class SciKitClassifier(BaseClassifier):
    """The base class for a `SciKit <https://scikit-learn.org/stable/index.html>`_ classifier"""
    
    def __init__(
        self, 
        model: BaseEstimator | ClassifierMixin,
    ):
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

    def predict_proba(self, signal: Signal):
        X = signal.data
        return self.model.predict_proba(X)
      
    def save(self, path:str):
        path = f"{path}.joblib"
        joblib.dump(self.model, filename=path)

    def load(self, path:str):
        path = f"{path}.joblib"
        self._model = joblib.load(path)

class SVMClassifier(SciKitClassifier):
    """Wrapper for the `SVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_ in scikit"""
    
    class Kernel(StrEnum):
        """Allowed SVM Kernels"""
        LINEAR='linear'
        POLY='poly'
        RBF='rbf'
        SIGMOID='sigmoid'
        PRECOMPUTED='precomputed'
    
    class Gamma(StrEnum):
        """Allowed Gamma Values"""
        SCALE='scale'
        AUTO='auto'
    
    class DFS(StrEnum):
        """Allowed decision function shapes"""
        OVO='ovo'
        OVR='ovr'
        
    def __init__(
        self,
        C=1.0,
        # TODO include callable and precomputed
        kernel: Kernel | str = Kernel.RBF,
        degree=3,
        gamma: Gamma | str | float = Gamma.SCALE,
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200.0,
        class_weight: dict | Literal['balanced'] | None = None,
        verbose=False,
        max_iter=-1,
        decision_function_shape: DFS | str = DFS.OVR,
        break_ties=False,
        random_state: int | None = None,
        external_model: SVC = None,
    ):
        model = external_model
        if model is None:
            model = SVC(
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                shrinking=shrinking,
                probability=probability,
                tol=tol,
                cache_size=cache_size,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
                decision_function_shape=decision_function_shape,
                break_ties=break_ties,
                random_state=random_state,
            )
        super().__init__(model=model)
    
    @property
    def model(self) -> SVC:
        return self._model
        
class RFClassifier(SciKitClassifier):
    """Wrapper for the `Random Forest <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ in scikit."""
    
    class Criterion(StrEnum):
        "Allowed data for criterion"
        GINI='gini'
        ENTROPY='entropy'
        LOG_LOSS='log_loss'
    
    class MaxFeatures(StrEnum):
        "Allowed data for max features"
        SQRT='sqrt'
        LOG2='log2'
        
    class ClassWeight(StrEnum):
        """Allowed values for class weight"""
        BALANCED='balanced'
        BALANCED_SUBSAMPLE='balanced_subsample'
        
    def __init__(
        self,
        n_estimators=100,
        criterion: Criterion | str = Criterion.GINI,
        max_depth: int = None,
        mini_samples_split: int | float = 2,
        mini_samples_leaf: int | float = 1,
        min_weight_fraction_leaf=0.0,
        max_features: MaxFeatures | None | int | float = MaxFeatures.SQRT,
        max_leaf_nodes: int = None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        # TODO add callable
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int | None = None,
        verbose=0,
        warm_start=False,
        class_weight: ClassWeight | dict | list[dict] = None,
        ccp_alpha=0.0,
        max_samples: int | float = None,
        monotonic_cst: np.ndarray = None,
        external_model: RandomForestClassifier = None,
    ):
        model = external_model
        if model is None:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                mini_samples_split=mini_samples_split,
                mini_samples_leaf=mini_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
            )
        super().__init__(model=model)
    
    @property
    def model(self) -> RandomForestClassifier:
        return self._model
        
class LogisticRegressionClassifier(SciKitClassifier):
    """Wrapper for `Logistic Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ from scikit."""
    
    class Penalty(StrEnum):
        """Allowed values for penalty"""    
        L1='l1'
        L2='l2'
        ELASTICNET='elasticnet'
        NONE='None'
    
    class Solver(StrEnum):
        """Allowed values for solvers"""
        LBFGS='lbfgs'
        LIBLINEAR='liblinear'
        NEWTON_CG='newton-cg'
        NEWTON_CHOLESKY='newton_cholesky'
        SAG='sag'
        SAGA='saga'
    
    class MultiClass(StrEnum):
        """Allowed values for multiclass"""
        AUTO='auto'
        OVR='ovr'
        MULTINOMIAL='multinomial'
        
    def __init__(
        self,
        penalty: Penalty | str = Penalty.L2,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight: dict | str = None,
        random_state: int | None = None,
        solver: Solver | str = Solver.LBFGS,
        max_iter=100,
        multi_class: MultiClass | str = MultiClass.AUTO,
        verbose=0,
        warm_start=False,
        n_jobs: int = None,
        l1_ratio: float = None,
        external_model: LogisticRegression = None,
    ):
        model = external_model
        if model is None:
            model = LogisticRegression(
                penalty=penalty,
                dual=dual,
                tol=tol,
                C=C,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                random_state=random_state,
                solver=solver,
                max_iter=max_iter,
                multi_class=multi_class,
                verbose=verbose,
                warm_start=warm_start,
                n_jobs=n_jobs,
                l1_ratio=l1_ratio,
            )
        super().__init__(model=model)
    
    @property
    def model(self) -> LogisticRegression:
        return self._model

class LDAClassifier(SciKitClassifier):
    """Wrapper for `Linear Discriminant Analysis <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html>`_ from scikit."""
    
    class Solver(StrEnum):
        """Allowed values for solver"""
        SVD='svd'
        LSQR='lsqr'
        EIGEN='eigen'
        
    def __init__(
        self,
        solver: Solver | str = Solver.SVD,
        shrinkage: Literal['auto'] | float | None = None,
        priors: np.ndarray = None,
        n_components: int = None,
        store_covariance=False,
        tol=1.0e-4,
        covariance_estimartor=None,
        external_model: LinearDiscriminantAnalysis = None,
    ):
        model = external_model
        if model is None:
            model = LinearDiscriminantAnalysis(
                solver=solver,
                shrinkage=shrinkage,
                priors=priors,
                n_components=n_components,
                store_covariance=store_covariance,
                tol=tol,
                covariance_estimator=covariance_estimartor,
            )
        super().__init__(model=model)
    
    @property
    def model(self) -> LinearDiscriminantAnalysis:
        return self._model