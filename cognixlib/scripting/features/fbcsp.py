"""
This module implements the FBCSP algorithm, as described in the original
`paper <https://doi.org/10.1109/IJCNN.2008.4634130>`_. The implementation
is based on this `repo <https://github.com/jesus-333/FBCSP-Python>`.

A more detailed approach on the FBCSP can be found in this
`paper <https://doi.org/10.3389/fnins.2012.00039>`_.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_classif
from collections.abc import Sequence, Mapping, MutableSequence
from orjson import dumps, loads, OPT_INDENT_2, OPT_SERIALIZE_NUMPY
from os.path import join
from ..data.signals import Signal, LabeledSignal, StreamSignal, FeatureSignal

import numpy as np
import scipy.signal
import scipy.linalg as la


Trials = Sequence[LabeledSignal]
PerBandTrials = Sequence[Trials]

PerBandCSP = Sequence[np.ndarray] # trial x channels x samples

class FBCSP_Binary:
    """
    The original binary version of the FBCSP algorithm.
    
    Note that this implementation doesn't concern itself with the filtering
    part. This should be external to the algorithm.
    """     
    
    def __init__(
        self,
        m=2,
        n_features=4,
        # {class: band[trials]}
        filt_trials: Mapping[str, PerBandTrials] = None,
    ):
        self._m = m
        self._n_features = n_features
        self._filt_trials = filt_trials
        self._fitted = False
        # per band CSP projection matrices
        # rows of single W: stationary spatial filters
        # columns of single W^T: common spatial patterns 
        self._w_per_band: list[np.ndarray] = None
        # {class: band[trial]}
        self._classes_csp: dict[str, list[np.ndarray]] = None
        # {class: band[features]}
        self._classes_features: dict[str, list[np.ndarray]] = None
        # (og band, feat index in og band), these are the selected features
        self._feature_items: list[tuple[int, int]] = None 
    
    @property
    def fitted(self) -> bool:
        return self._fitted
    
    @property
    def m(self) -> int:
        return self._m
    
    @m.setter
    def m(self, m: int):
        self._m = m
        self._fitted = False
    
    @property
    def n_features(self) -> int:
        return self._n_features
            
    @n_features.setter
    def n_features(self, n_features: int):
        self._n_features = n_features
        self._fitted= False
    
    @property
    def filt_trials(self):
        return self._filt_trials

    @filt_trials.setter
    def filt_trials(self, value: Mapping[str, PerBandTrials]):
        self._filt_trials = value
        self._fitted = False
    
    def save(self, path: str, name: str, ext='json'):
        full_path = join(path, f'{name}.{ext}')
        with open(full_path, 'wb') as file:
            file.write(self.to_json())
    
    def load(self, path: str):
        
        with open(path, 'rb') as file:
            json_bytes = file.read()
        self.filt_trials(json_bytes)
    
    def to_json(self):
        data = {
            '_m': self._m,
            '_n_features': self._n_features,
            '_w_per_band': self._w_per_band,
            '_feature_items': self._feature_items
        }
        return dumps(data, option = OPT_SERIALIZE_NUMPY)

    def from_json(self, json: bytes):
        data: dict = loads(json)
        for key, value in data.items():
            if key == '_w_per_band':
                value = [np.array(v) for v in value]
            setattr(self, key, value)
        
    def extract_features(self, band_trials: PerBandTrials) -> LabeledSignal:
        """
        Extracts the features from filtered trials from a trial.
        The per band trials should come in the same order as the
        trials given when the FBCSP was "trained".
        """
        
        features_list = []
        n_bands = len(band_trials) # how many bands 
        n_trials = len(band_trials[0]) # how many trials per band
        n_tot_feats = len(self._feature_items)
        
        feat_input = np.zeros((n_trials, n_tot_feats))
        
        # Feature extraction
        
        # This should (can it?) be optimized at some point
        for i in range(n_bands):
            W = self._w_per_band[i]
            band_trial = band_trials[i]
            spatial_trial = self._apply_spatial_filter(band_trial, W)
            # log var -> features
            features = self._log_var(spatial_trial)
            features_list.append(features)
        
        # Feature selection
        feat_labels = []
        for i in range(n_tot_feats):
            band_pos, feat_index = self._feature_items[i]
            feat_labels.append(self._create_feat_label(band_pos, feat_index))
            feat_input[:, i] = features_list[band_pos][:, feat_index]
        return LabeledSignal(feat_labels, feat_input, None)
        
    def fit(
        self, 
        filt_trials: Mapping[str, PerBandTrials] = None
    ):
        """
        Calculates the spatial filters as described in :meth:`spatial_filters`. Then applies
        the logarithm and the covariance to extract the features. Finally, the most important
        features are selected through `code:`MIBIF (Mutal Information) and returned as a
        :class:`cognixlib.data.signals.FeatureSignal`.
        
        Note that this is shortcut for calling :meth:`calc_spatial_filters` and :meth:`extract_features`
        together and are applied on the same trials.
        
        For calculating the spatial filters and features on potentially different trials,
        use the above separately.
        """ 
        # Each key corresponds to a trial
        trials = filt_trials if filt_trials else self._filt_trials
    
        # each Stream Signal represents a trial
        # we're extracting the classes, which should be two
        # then extracting how many bands we have
        
        self.calc_spat_filts(trials)
        return self.select_features(trials)
    
    # Features
    
    def select_features(
        self, 
        filt_trials: Mapping[str, PerBandTrials]
    ):
        """
        Calculates the feature matrices of all trials per class. Then applies a feature
        selection algorithm through Mutual Information (MIBIF) and extracts a :class:`cognixlib.data.signals.FeatureSignal`
        containing the features from class one and two.
        
        The selected features will range from [n, 2*n], where n=:attr:`n_features`

        Args:
            filt_trials (Mapping[str, PerBandTrials]): A dict of {class, band[trial]} where trial => :code:`Sequence[LabeledSignal]`
            class_labels (Sequence[str], optional): _description_. Optional classes names.
        """
        
        # Calculate the feature matrices using log var
        self._spatial_filter(filt_trials)
        self._classes_features = dict(**self._classes_csp)
        for cls, per_band_csp in self._classes_csp.items():
            
            self._classes_features[cls] = [
                self._log_var(band_csp) for band_csp in per_band_csp
            ]
        
        # Mutual Information
        
        mutual_infos: list[np.ndarray] = []
        
        classes = list(filt_trials.keys())
        class_one, class_two = classes
        num_bands = len(self._classes_features[class_one])
        
        # each band will have its own mutual information
        for i in range(num_bands):
            feat_one = self._classes_features[class_one][i]
            feat_two = self._classes_features[class_two][i]
            
            all_features = np.zeros(
                (
                    feat_one.shape[0] + feat_two.shape[0],
                    feat_one.shape[1]
                )
            )
            
            n_feat = feat_one.shape[0]
            all_features[0:n_feat] = feat_one
            all_features[n_feat: ] = feat_two
            
            # labeling scheme
            labels = np.ones(all_features.shape[0])
            labels[0: n_feat] = 2

            mutual_info = mutual_info_classif(all_features, labels)
            mutual_infos.append(mutual_info)
        
        m2 = self.m * 2
        # 1D-Array with all the mutual information value
        mutual_info_vec = np.zeros(len(mutual_infos) * m2)
        # The CSP features are coupled (first with last). Save the CSP features
        # 
        other_info_matrix = np.zeros((len(mutual_info_vec), 4))
        
        # convert the infos to a vector
        for i in range(len(mutual_infos)):
            mutual_info = mutual_infos[i]
            for j in range(m2):
                actual_idx = i * m2 + j    
                mutual_info_vec[actual_idx] = mutual_info[j]
                
                other_info_matrix[actual_idx, 0] = i * m2 + m2 - (j + 1) # position of the twin in the vector
                other_info_matrix[actual_idx, 1] = actual_idx # position of the actual feature (in the vector)
                other_info_matrix[actual_idx, 2] = i # current band
                other_info_matrix[actual_idx, 3] = j # position of the original band
        
        sorted_minfo_idices = np.flip(np.argsort(mutual_info_vec))
        sorted_other_info = other_info_matrix[sorted_minfo_idices, :]
        
        feature_items: list[tuple[int, int]] = [] # (og band, og position in og band)
        selected_features = sorted_other_info[:, 1][0:self.n_features]

        # Select the 2 * n most relevant features
        for i in range(self._n_features):
            
            features_item = ((int)(sorted_other_info[i, 2]), int(sorted_other_info[i, 3]))
            feature_items.append(features_item)
            
            # Twin/Couple features of the current features
            current_features_twin = sorted_other_info[i, 0]
            if not current_features_twin in selected_features:
                # add the twin
                twin_idx = sorted_other_info[:, 1] == current_features_twin
                twin_feature_item = (int(sorted_other_info[twin_idx, 2][0]), int(sorted_other_info[twin_idx, 3][0]))
                feature_items.append(twin_feature_item)
        
        feature_items.sort()
        self._feature_items = feature_items
        
        # We want the number of original per class trials, not filtered
        num_trials_one = len(filt_trials[class_one][0])
        num_trials_two = len(filt_trials[class_two][0])
        total_trials = num_trials_one + num_trials_two
        num_features = len(feature_items) # should be 2 * self._n_features
        
        band_check = -1
        features = np.zeros((total_trials, num_features))
        
        feature_labels = []
        for i in range(num_features):
            band, feat_index = feature_items[i]
            feature_labels.append(self._create_feat_label(band, feat_index))
            
            if band != band_check:
                band_check = band    
                band_feat_one = self._classes_features[class_one][band]
                band_feat_two = self._classes_features[class_two][band]
                
            features[0:num_trials_one, i] = band_feat_one[:, feat_index]
            features[num_trials_one: total_trials, i] = band_feat_two[:, feat_index]
        
        feature_classes = {
            class_one: (0, num_trials_one),
            class_two: (num_trials_one, total_trials)
        }
        
        return FeatureSignal(
            feature_labels,
            feature_classes,
            features,
            None,
            False
        )
            
    def _create_feat_label(self, band: int, feat_index: int):
        return f'fcsp_band{band}_f{feat_index}'     
    
    def _log_var(self, trials: np.ndarray):
        """
        Calculates the logarithm and variance of a trial matrix. The trials should
        be csp filtered. Only the first and last :meth:`m` of the CSP filtered signal
        will be considered.
        """
        
        trials = np.concatenate(
            [
                trials[:, :self._m, :],
                trials[:, -self._m:, :]
            ],
            axis=1
        )
        
        features = np.var(trials, 2)
        features = np.log(features)
        return features   
    
    # Spatial Filters
        
    def _spatial_filter(self, filt_trials: Mapping[str, PerBandTrials]):
        
        classes = list(filt_trials.keys())
        class_one, _ = classes
        num_bands = len(filt_trials[class_one])
        
        self._classes_csp = {
            klass: [] for klass in classes
        }
        
        for i in range(num_bands):
            W = self._w_per_band[i]
            
            for klass in classes:    
                
                band_trials = filt_trials[klass][i]
                spatial_filter = self._apply_spatial_filter(band_trials, W)
                self._classes_csp[klass].append(spatial_filter)
                
    
    def _apply_spatial_filter(
        self, 
        trials: Sequence[Signal | np.ndarray] | np.ndarray , 
        W: np.ndarray
    ) -> np.ndarray:
        
        # a trial is always samples x channels
        # Allocation of memory for spatial filtered trials
        n_trials = len(trials)
        first_data = trials[0]
        _, n_channels = first_data.shape
        n_samples = 0
        if isinstance(trials, np.ndarray):
            n_samples = trials.shape[1]
        else:
            for trial in trials:
                n_samples += trial.shape[0]
        
        trials_csp = np.zeros((n_trials, n_channels, n_samples))
        
        count = 0
        for i in range(n_trials):
            trial = trials[i]
            if isinstance(trial, Signal):
                trial = trial.data
                
            trial_samples, _ = trial.shape # remember, it's samples x channels
            # change it to channels x samples
            trial = trial.T
            # we're storing all the trials in one single matrix
            trials_csp[i, :, count:trial_samples + count] = W.dot(trial)
            count += trial_samples
        
        return trials_csp
        
              
    def calc_spat_filts(
        self,
        filt_trials: Mapping[str, PerBandTrials]       
    ):
        """
        Calculates the per band CSP (W) projection matrix that. The rows of the
        matric correspond to stationary spatial filters. The columns of the transposed W correspond to common spatial patterns.

        Args:
            filt_trials (Mapping[str, PerBandTrials]): 
                A dictionary of {class: band[trial]} where trial => :code:`Sequence[LabeledSignal]`. If None is given,
                the trials passed through the constructor will be used.
        """

        self._w_per_band = []
        
        classes = list(filt_trials.keys())
        class_one, class_two = classes
        num_bands = len(filt_trials[class_one])
        
        # W - CSP projection matrix evaluation
        for i in range(num_bands):
            
            trial_one = filt_trials[class_one][i]
            trial_two = filt_trials[class_two][i]
            
            cov_one = self._covariance(trial_one)
            cov_two = self._covariance(trial_two)
            
            R = cov_one + cov_two 
            
            P = self._whitening(R)
            
            cov_one_white = np.dot(
                P, 
                np.dot(cov_one, np.transpose(P))
            )
            cov_two_white = np.dot(
                P,
                np.dot(cov_two, np.transpose(P))
            )
            
            # CSP requires eigenvalues and eigenvector to be sorted in descending
            # order
            
            # eigenvalues, eigenvector
            E, U = la.eig(cov_one_white, cov_two_white)
            order = np.argsort(E)[::-1]
            E = E[order]
            U = U[:, order]
            
            # CSP projection matrix
            W = np.dot(np.transpose(U), P)
            
            self._w_per_band.append(W)
        
        
    def _covariance(self, trials: Trials) -> np.ndarray:
        """
        Calculates the covariance for each trial and returns their average.

        Args:
            trials (Trials):
                A sequence of :class:`StreamSignal` representing the trials of a class.

        Returns:
            np.ndarray: Mean of the covariance alongside the channels.
        """
        n_trials = len(trials)
        n_channels =  trials[0].shape[1]
        
        # A StreamSignal is samples x channels, so we transpose it because
        # we're working with channels x samples in here
        covariance_matrix = np.zeros((n_trials, n_channels, n_channels))  
        for i in range(n_trials):
            trial = trials[i]
            covariance_matrix[i, :, :] = np.cov(trial.data.T)
        
        return np.mean(covariance_matrix, 0)

    def _whitening(self, sigma: np.ndarray, mode=2):
        """
        Calculates the whitening matrix of the input matrix :code:`sigma`

        Args:
            sigma (np.ndarray): A 2D numpy array
            mode (int, optional): Evaluation mode of the whitening matrix. Defaults to 2.

        Returns:
            np.ndarray: 2D numpy Whitening matrix
        """
        
        [u, s, _] = np.linalg.svd(sigma)
        
        if mode != 1 and mode != 2:
            mode = 1
        
        if mode == 1:
            # Whitening constant to prevent zero-division
            epsilon = 1e-5
            
            # ZCA Whitening matrix: U * Lambda * U'
            w = np.dot(
                u,
                np.dot(
                    np.diag(1.0/np.sqrt(s + epsilon)),
                    u.T
                )
            )
        else:
            # eigenvalue decomposition of the covariance matrix
            d, V = np.linalg.eigh(sigma)
            # fudeg factor so that eigenvectors associated with small eigenvalues
            # do not get overamplified
            fudge = 10e-18
            D = np.diag(1. / np.sqrt(d + fudge))
            
            w = np.dot(np.dot(V, D), V.T)
        
        return w    