"""
This module implements the FBCSP algorithm, as described in the original
`paper <https://doi.org/10.1109/IJCNN.2008.4634130>`_. The implementation
is based on this `repo <https://github.com/jesus-333/FBCSP-Python>`.
"""

import numpy as np
import scipy.signal
import scipy.linalg as la

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_classif
from collections.abc import Sequence, Mapping, MutableSequence
from ..data.signals import Signal, LabeledSignal, StreamSignal

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
        # columns of single W^-1: common spatial patterns 
        self._w_per_band: list[np.ndarray] = None
        # {class: band[trial]}
        self._classes_csp: dict[str, list[np.ndarray]] = None
        # {class: band[features]}
        self._classes_features: dict[str, list[np.ndarray]] = None
    
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
    
    def fit(self, filt_trials: Mapping[str, PerBandTrials] = None):
        """
        Calculates the spatial filters as described in :meth:`spatial_filters`. Then applies
        the logarithm and the covariance to extract the features. Finally, the most important
        features are selected through `code:`MIBIF (Mutal Information). 
        """ 
        # Each key corresponds to a trial
        trials = filt_trials if filt_trials else self._filt_trials
    
        # each Stream Signal represents a trial
        # we're extracting the classes, which should be two
        # then extracting how many bands we have
        
        self.calc_spatial_filters(trials)
        self.calc_features()
    
    # Features
    
    def calc_features(self):
        """
        Calculates the feature matrices of all trials per class. Also applies a feature
        selection algorithm through Mutual Information (MIBIF)
        """
        
        # Calculate the feature matrices using log var
        
        self._classes_features = dict(**self._classes_csp)
        for cls, per_band_csp in self._classes_csp.items():
            
            res = []
            for band_csp in per_band_csp:
                res.append(self._log_var(band_csp))
            
            self._classes_features[cls] = res
        
        # Mutual Information
        
        mutual_infos: list[np.ndarray] = []
        
        classes = list(self._filt_trials.keys())
        class_one, class_two = classes
        num_features = len(self._classes_features[class_one])
        
        for i in range(num_features):
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
        mutual_info_vec = np.zeros(9 * m2)
        # The CSP features are coupled (first with last). Save the CSP features
        # 
        other_info_matrix = np.zeros((len(mutual_info_vec), 4))
        
        # convert the infos to a vector
        for i in range(len(mutual_infos)):
            mutual_info = mutual_infos[i]
            for j in range(m2):
                actual_idx = i * m2 + j    
                mutual_info_vec[actual_idx] = mutual_info[j]
                
                other_info_matrix[actual_idx, 0] = i * m2 + m2 - j + 1
        
    def _log_var(self, trials: np.ndarray):
        """
        Calculates the logarithm and variance of a trial matrix. The trials should
        be csp filtered. Only the first and last :meth:`m` of the CSP filtered signal
        will be considered.
        """
        
        idxs = [i for i in range(self._m)]
        for i in reversed(idxs):
            idxs.append(-(i+1))
        trials = trials[:, idxs, :]
        
        features = np.var(trials, 2)
        features = np.log(features)
        
        return features
    
    # Spatial Filters
    
    def calc_spatial_filters(self, filt_trials: Mapping[str, PerBandTrials] = None):
        """
        At first calculates the per band CSP (W) projection matrix that. The rows of the
        matric correspond to stationary spatial filters. The columns of the transposed W correspond to common spatial patterns.
        
        Then, it calculates the spatial filters for each band.
        """
        trials = filt_trials if filt_trials else self._filt_trials
        self._filt_trials = trials

        self._calculate_w(trials)
        self._spatial_filters(trials)
        
    def _spatial_filters(self, filt_trials: Mapping[str, PerBandTrials] = None):
        
        classes = list(filt_trials.keys())
        class_one, _ = classes
        num_bands = len(filt_trials[class_one])
        
        self._classes_csp = {
            klass: [] for klass in classes
        }
        
        for i in range(num_bands):
            W = self._w_per_band[i]
            
            for klass in classes:    
                
                trials = self._filt_trials[klass]
                spatial_filter = self._apply_spatial_filter(trials, W)
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
        
        trials_csp = np.zeros(n_trials, n_channels, n_samples)
        
        for i in range(n_trials):
            trial = trials[i]
            if isinstance(trial, Signal):
                trial = trial.data
            
            # change it to channels x samples
            trial = trial.T
            trials_csp[i] = W.dot(trial)
        
        return trials_csp
        
              
    def _calculate_w(
        self,
        filt_trials: Mapping[str, PerBandTrials]       
    ):
    
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
        
        
    def _covariance(trials: Trials) -> np.ndarray:
        """
        Calculates the covariance for each trial and returns their average.

        Args:
            trials (Trials):
                A sequence of :class:`StreamSignal` representing the trials of a class.

        Returns:
            np.ndarray: Mean of the covariance alongside the channels.
        """
        n_trials = len(trials)
        n_channels = len(trials[0].labels)
        
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
            
        
        
    
class FBCSP_V4():
    
    def __init__(self, data_dict, fs, n_w = 2, n_features = 4, freqs_band = None, filter_order = 3, classifier = None, print_var = True):
        self.fs = fs
        self.trials_dict = data_dict
        self.n_w = n_w
        self.n_features = n_features
        self.n_trials_class_1 = data_dict[list(data_dict.keys())[0]].shape[0]
        self.n_trials_class_2 = data_dict[list(data_dict.keys())[1]].shape[0]
        self.print_var = print_var
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #Filter data section
        
        # Filtered signal list
        self.filtered_band_signal_list = []
        
        # Frequencies band
        if isinstance(freqs_band, np.ndarray):
            self.freqs = freqs_band
        elif(freqs_band == None):
            self.freqs = np.linspace(4, 40, 10)
        else:
            raise ValueError('freqs_band must be a Numpy Array')
        
        self.filterBankFunction(filter_order)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    
        # CSP filters evaluation and application
        
        # CSP filter evaluation
        self.W_list_band = []
        self.evaluateW()
        
        # CSP filter application
        self.features_band_list = []
        self.spatialFilteringAndFeatureExtraction()
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Training of the classifier
        if(classifier != None): self.trainClassifier(classifier = classifier)
        else: self.trainClassifier() 
        
        
    def filterBankFunction(self, filter_order = 3):
        """
        Function that apply fhe fitlering for each pair of frequencies in the list self.freqs.
        The results are saved in a list called self.filtered_band_signal_list. Each element of the list is a diciotinary with key the label of the various class.

        Parameters
        ----------
        filter_order : int, optional
            The order of the filter. The default is 3.

        """
        
        # Cycle for the frequency bands
        for i in range(len(self.freqs) - 1):  
            # Dict for selected band that will contain the various filtered signals
            filt_trial_dict = {}
            
            # "Create" the band
            band = [self.freqs[i], self.freqs[i+1]]
            
            # Cycle for the classes
            for key in self.trials_dict.keys(): 
                # Filter the signal in each class for the selected frequency band
                filt_trial_dict[key] = self.bandFilterTrials(self.trials_dict[key], band[0], band[1], filter_order = filter_order)
            
            # Save the filtered signal in the list
            self.filtered_band_signal_list.append(filt_trial_dict)
        
    
    def bandFilterTrials(self, trials_matrix, low_f, high_f, filter_order = 3):
        """
        Applying a pass-band fitlering to the data. The filter implementation was done with scipy.signal
    
        Parameters
        ----------
        trials_matrix : numpy matrix
            Numpy matrix with the various EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples
        fs : int/double
            Frequency sampling.
        low_f : int/double
            Low band of the pass band filter.
        high_f : int/double
            High band of the pass band filter..
        filter_order : int, optional
            Order of the filter. The default is 3.
    
        Returns
        -------
        filter_trails_matrix : numpy matrix
             Numpy matrix with the various filtered EEG trials. The dimensions of the matrix must be n_trial x n_channel x n_samples.
    
        """
        
        # Evaluate low buond and high bound in the [0, 1] range
        low_bound = low_f / (self.fs/2)
        high_bound = high_f / (self.fs/2)
        
        # Check input data
        if(low_bound < 0): low_bound = 0
        if(high_bound > 1): high_bound = 1
        if(low_bound > high_bound): low_bound, high_bound = high_bound, low_bound
        if(low_bound == high_bound): low_bound, high_bound = 0, 1
        
        b, a = scipy.signal.butter(filter_order, [low_bound, high_bound], 'bandpass')
          
        return scipy.signal.filtfilt(b, a, trials_matrix)
    
    
    def evaluateW(self):
        """
        Evaluate the spatial filter of the CSP algorithm for each filtered signal inside self.filtered_band_signal_list
        Results are saved inside self.W_list_band.    
        """
        
        for filt_trial_dict in self.filtered_band_signal_list:
            # Retrieve the key (class)
            keys = list(filt_trial_dict.keys())
            
            
            keys = list(filt_trial_dict.keys())
            trials_1 = filt_trial_dict[keys[0]]
            trials_2 = filt_trial_dict[keys[1]]
        
            # Evaluate covariance matrix for the two classes
            cov_1 = self.trialCovariance(trials_1)
            cov_2 = self.trialCovariance(trials_2)
            R = cov_1 + cov_2
            
            # Evaluate whitening matrix
            P = self.whitening(R)
            
            # The mean covariance matrices may now be transformed
            cov_1_white = np.dot(P, np.dot(cov_1, np.transpose(P)))
            cov_2_white = np.dot(P, np.dot(cov_2, np.transpose(P)))
            
            # Since CSP requires the eigenvalues and eigenvector be sorted in descending order we find and sort the generalized eigenvalues and eigenvector
            E, U = la.eig(cov_1_white, cov_2_white)
            order = np.argsort(E)
            order = order[::-1]
            E = E[order]
            U = U[:, order]
            
            # The projection matrix (the spatial filter) may now be obtained
            W = np.dot(np.transpose(U), P)
            
            self.W_list_band.append(W)
      
    
    def trialCovariance(self, trials):
        """
        Calculate the covariance for each trial and return their average
    
        Parameters
        ----------
        trials : numpy 3D-matrix
            Trial matrix. The dimensions must be trials x channel x samples
    
        Returns
        -------
        mean_cov : Numpy matrix
            Mean of the covariance alongside channels.
    
        """
        
        n_trials, n_channels, n_samples = trials.shape
        
        covariance_matrix = np.zeros((n_trials, n_channels, n_channels))
        
        for i in range(trials.shape[0]):
            trial = trials[i, :, :]
            covariance_matrix[i, :, :] = np.cov(trial)
            
        mean_cov = np.mean(covariance_matrix, 0)
            
        return mean_cov
    
    
    def whitening(self, sigma, mode = 2):
        """
        Calculate the whitening matrix for the input matrix sigma
    
        Parameters
        ----------
        sigma : Numpy square matrix
            Input matrix.
        mode : int, optional
            Select how to evaluate the whitening matrix. The default is 1.
    
        Returns
        -------
        x : Numpy square matrix
            Whitening matrix.
        """
        [u, s, vh] = np.linalg.svd(sigma)
        
          
        if(mode != 1 and mode != 2): mode == 1
        
        if(mode == 1):
            # Whitening constant: prevents division by zero
            epsilon = 1e-5
            
            # ZCA Whitening matrix: U * Lambda * U'
            x = np.dot(u, np.dot(np.diag(1.0/np.sqrt(s + epsilon)), u.T))
        else:
            # eigenvalue decomposition of the covariance matrix
            d, V = np.linalg.eigh(sigma)
            fudge = 10E-18
         
            # A fudge factor can be used so that eigenvectors associated with small eigenvalues do not get overamplified.
            D = np.diag(1. / np.sqrt(d+fudge))
         
            # whitening matrix
            x = np.dot(np.dot(V, D), V.T)
            
        return x
        
    
    def spatialFilteringAndFeatureExtraction(self):
        # Cycle through frequency band and relative CSP filter
        for filt_trial_dict, W in zip(self.filtered_band_signal_list, self.W_list_band):
            # Features dict for the current frequency band
            features_dict = {}
            
            # Cycle through the classes
            for key in filt_trial_dict.keys():
                # Applying spatial filter
                tmp_trial = self.spatialFilteringW(filt_trial_dict[key], W)
                
                # Features evaluation
                features_dict[key] = self.logVarEvaluation(tmp_trial)
            
            self.features_band_list.append(features_dict)
        
        # Evaluate mutual information between features
        self.mutual_information_list = self.computeFeaturesMutualInformation()
        self.mutual_information_vector, self.other_info_matrix = self.changeShapeMutualInformationList()
        
        # Select features to use for classification
        # List of tuple (each tuple contains the number of the band and the number of the features)
        self.classifier_features = self.selectFeatures()
        
        
    def spatialFilteringW(self, trials, W):
        # Allocate memory for the spatial fitlered trials
        trials_csp = np.zeros(trials.shape)
        
        # Apply spatial fitler
        for i in range(trials.shape[0]): trials_csp[i, :, :] = W.dot(trials[i, :, :])
            
        return trials_csp
    
    
    def logVarEvaluation(self, trials):
        """
        Evaluate the log (logarithm) var (variance) of the trial matrix along the samples axis.
        The sample axis is the axis number 2, counting axis as 0,1,2. 
    
        Parameters
        ----------
        trials : numpy 3D-matrix
            Trial matrix. The dimensions must be trials x channel x samples
    
        Returns
        -------
        features : Numpy 2D-matrix
            Return the features matrix. DImension will be trials x (n_w * 2)
    
        """
        # Select the first and last n rows of the CSP filtered signal
        idx = []
        for i in range(self.n_w): idx.append(i)
        for i in reversed(idx): idx.append(-(i + 1))
        trials = trials[:, idx, :]    
        
        features = np.var(trials, 2)
        features = np.log(features)
        
        return features
            
    def computeFeaturesMutualInformation(self):
        """
        Select the first and last n columns of the various features matrix and compute their mutual inforamation.
        The value of n is self.n_features

        Returns
        -------
        mutual_information_list : List of numpy matrix
            List with the mutual information of the various features.

        """
        
        mutual_information_list = []
                
        # Cycle through the different band
        for features_dict in self.features_band_list:
            # Retrieve features for that band
            keys = list(features_dict.keys())
            feat_1 = features_dict[keys[0]]
            feat_2 = features_dict[keys[1]]
            
            # Save features in a single matrix
            all_features = np.zeros((feat_1.shape[0] + feat_2.shape[0], feat_1.shape[1]))            
            all_features[0:feat_1.shape[0], :] = feat_1
            all_features[feat_1.shape[0]:, :] = feat_2
            
            # Create label vector
            label = np.ones(all_features.shape[0])
            label[0:feat_1.shape[0]] = 2
            
            tmp_mutual_information = mutual_info_classif(all_features, label)
            mutual_information_list.append(tmp_mutual_information)
            
        return mutual_information_list
    
    
    def changeShapeMutualInformationList(self):
        # 1D-Array with all the mutual information value
        mutual_information_vector = np.zeros(9 * 2 * self.n_w)
            
        # Since the CSP features are coupled (First with last etc) in this matrix I save the couple.
        # I will also save the original band and the position in the original band
        other_info_matrix = np.zeros((len(mutual_information_vector), 4))
        
        for i in range(len(self.mutual_information_list)):
            mutual_information = self.mutual_information_list[i]
            
            for j in range(self.n_w * 2):
                # Acual index for the various vector
                actual_idx = i * self.n_w * 2 + j
                
                # Save the current value of mutual information for that features
                mutual_information_vector[actual_idx] = mutual_information[j]
                
                # Save other information related to that feature
                other_info_matrix[actual_idx, 0] = i * self.n_w * 2 + ((self.n_w * 2) - (j + 1)) # Position of the twin (in the vector)
                other_info_matrix[actual_idx, 1] = actual_idx # Position of the actual feature (in the vector)
                other_info_matrix[actual_idx, 2] = i # Current band
                other_info_matrix[actual_idx, 3] = j # Position in the original band
                
        return mutual_information_vector, other_info_matrix
    
    
    def computeMutualInformation2(self):
        """
        Method add to test a different type of mutual information evaluation find in another paper. 
        The results are the same that with the original method. 
        So this method is impemented but not used.

        """

        tot_trials = self.n_trials_class_1 + self.n_trials_class_2
        features_matrix = np.zeros((tot_trials, self.n_features * 2 * 9))
        label_vector = np.zeros(tot_trials)
        
        # Cycle through the different band
        for features_dict, i in zip(self.features_band_list, range(len(self.features_band_list))):
            # Retrieve features for that band
            keys = list(features_dict.keys())
            feat_1 = features_dict[keys[0]]
            feat_2 = features_dict[keys[1]]
            
            # Save features in a single matrix
            all_features = np.zeros((feat_1.shape[0] + feat_2.shape[0], self.n_features * 2))            
            all_features[0:feat_1.shape[0], :] = feat_1
            all_features[feat_1.shape[0]:, :] = feat_2
            
            # Create label vector
            label = np.ones(all_features.shape[0])
            label[0:feat_1.shape[0]] = 2
            
            # Add element to the single variable
            features_matrix[0:tot_trials, (self.n_features * 2) * i:(self.n_features * 2) * (i + 1)] = all_features
            label_vector[0:tot_trials] = label
            
        
        self.mutual_information_vector_V2 = mutual_info_classif(features_matrix, label_vector)
            
    
    def selectFeatures(self):
        """
        Select n features for classification. In this case n is equal to 2 * self.n_features.
        The features selected are the self.n_features with the highest mutual information. 
        Since the CSP features are coupled if the original couple was not selected we add to the list of features the various couple.
        The original algorithm select a variable number of features (and also the V3 implementation has the same behavior). This version select always 2 * self.n_features.
        
        Returns
        -------
        complete_list_of_features : List of tuple
            List that contatin the band for the filter and the position inside the original band.

        """
        
        # Sort features in order of mutual information
        sorted_MI_features_index = np.flip(np.argsort(self.mutual_information_vector))
        sorted_other_info = self.other_info_matrix[sorted_MI_features_index, :]
        
        complete_list_of_features = []
        selected_features = sorted_other_info[:, 1][0:self.n_features]
        
        for i in range(self.n_features):
            # Current features (NOT USED)(added just for clarity during coding)
            # current_features = sorted_other_info[i, 1]
            
            # Twin/Couple feature of the current features
            current_features_twin = sorted_other_info[i, 0]
            
            if(current_features_twin in selected_features): 
                # If I also select its counterpart I only add the current feaures because the counterpart will be added in future iteration of the cycle
                
                # Save the features as tuple with (original band, original position in the original band)
                features_item = (int(sorted_other_info[i, 2]), int(sorted_other_info[i, 3]))
                
                # Add the element to the features vector
                complete_list_of_features.append(features_item)
            else: 
                # If I not select its counterpart I addo both the current features and it's counterpart
                
                # Select and add the current feature
                features_item = (int(sorted_other_info[i, 2]), int(sorted_other_info[i, 3]))
                complete_list_of_features.append(features_item)
                
                # Select and add the twin/couple feature
                idx = sorted_other_info[:, 1] == current_features_twin
                features_item = (int(sorted_other_info[idx, 2][0]), int(sorted_other_info[idx, 3][0]))
                complete_list_of_features.append(features_item)
                
        return sorted(complete_list_of_features)
    
    
    def extractFeaturesForTraining(self):
        # Tracking variable of the band
        old_band = -1
        
        # Return matrix
        features_1 = np.zeros((self.n_trials_class_1, len(self.classifier_features)))
        features_2 = np.zeros((self.n_trials_class_2, len(self.classifier_features)))
        
        # Cycle through the different features
        for i in range(len(self.classifier_features)):
            # Retrieve the position of the features
            features_position = self.classifier_features[i]
            
            # Band of the selected feaures
            current_features_band = features_position[0]
            
            # Check if the band is the same of the previous iteration
            if(current_features_band != old_band):
                # In this case the band is not the same of the previous iteration
                
                old_band = current_features_band
                
                # Retrieve the dictionary with the features of the two classes for the current band
                current_band_features_dict = self.features_band_list[current_features_band]
                
                # Retrieve the matrix of features for the two classes
                keys = list(current_band_features_dict.keys())
                tmp_feat_1 = current_band_features_dict[keys[0]]
                tmp_feat_2 = current_band_features_dict[keys[1]]
                
                # Extract the features
                features_1[:, i] = tmp_feat_1[:, features_position[1]]
                features_2[:, i] = tmp_feat_2[:, features_position[1]]
                
            else: 
                # In this case I'm in the same band of the previous iteration
                
                # Extract the features
                features_1[:, i] = tmp_feat_1[:, features_position[1]]
                features_2[:, i] = tmp_feat_2[:, features_position[1]]
        
                
        return features_1, features_2
    
        
    def trainClassifier(self, train_ratio = 0.75, classifier = None):
        """
        Divide the data in train set and test set and used the data to train the classifier.

        Parameters
        ----------
        n_features : int, optional
            The number of mixture channel to use in the classifier. It must be even and at least as big as 2. The default is 2.
        train_ratio : doble, optional
            The proportion of the data to used as train dataset. The default is 0.75.
        classifier : sklearnn classifier, optional
            Classifier used for the problem. It must be a sklearn classifier. If no classfier was provided the fucntion use the LDA classifier.

        """
        
        features_1, features_2 = self.extractFeaturesForTraining()
        self.n_features_for_classification = features_1.shape[1]
        if(self.print_var): print("Features used for classification: ", self.n_features_for_classification)
    
        # Save both features in a single data matrix
        data_matrix = np.zeros((features_1.shape[0] + features_2.shape[0], features_1.shape[1]))
        data_matrix[0:features_1.shape[0], :] = features_1
        data_matrix[features_1.shape[0]:, :] = features_2
        self.tmp_data_matrix = data_matrix
        
        # Create the label vector
        label = np.zeros(data_matrix.shape[0])
        label[0:features_1.shape[0]] = 1
        label[features_1.shape[0]:] = 2
        self.tmp_label = label
        
        # Create the label dict
        self.tmp_label_dict = {}
        keys = list(self.features_band_list[0].keys())
        self.tmp_label_dict[1] = keys[0]
        self.tmp_label_dict[2] = keys[1]
        
        # Shuffle the data
        perm = np.random.permutation(len(label))
        label = label[perm]
        data_matrix = data_matrix[perm, :]
        
        # Select the portion of data used during training
        if(train_ratio <= 0 or train_ratio >= 1): train_ratio = 0.75
        index_training = int(data_matrix.shape[0] * train_ratio)
        train_data = data_matrix[0:index_training, :]
        train_label = label[0:index_training]
        test_data = data_matrix[index_training:, :]
        test_label = label[index_training:]
        self.tmp_train = [train_data, train_label]
        self.tmp_test = [test_data, test_label]
        
        # Select classifier
        if(classifier == None): self.classifier = LDA()
        else: self.classifier = classifier
        
        # Train Classifier
        self.classifier.fit(train_data, train_label)
        if(self.print_var): print("Accuracy on TRAIN set: ", self.classifier.score(train_data, train_label))
        
        # Test parameters
        if(self.print_var): print("Accuracy on TEST set: ", self.classifier.score(test_data, test_label), "\n")
        
        # print("total: ", self.classifier.score(train_data, train_label) * self.classifier.score(test_data, test_label))
        
        
    def evaluateTrial(self, trials_matrix, plot = True):
        """
        Evalaute trial/trials given in input

        Parameters
        ----------
        trials_matrix : Numpy 3D matrix
            Input matrix of trials. The dimension MUST BE "n. trials x n. channels x n.samples".
            Also in case of single trials the input input dimension must be "1 x n. channels x n.samples".
        plot : Boolean, optional
            If set to true will plot the features of the trial. The default is True.

        Returns
        -------
        y : Numpy vector
            Vector with the label of the respective trial. The length of the vector is the number of trials.
            The label are 1 for class 1 and 2 for class 2.
        
        y_prob : Numpy matrix
            Vector with the label of the respective trial. The length of the vector is the number of trials.
            The label are 1 for class 1 and 2 for class 2.

        """
        
        # Compute and extract the features for the training
        features_input = self.extractFeatures(trials_matrix)
        self.a = features_input

        # Classify the trials
        # print(features_input.shape)
        y = self.classifier.predict(features_input)
        
        # Evaluate the probabilty
        # if(self.classifier.__class__.__name__ == 'LinearDiscriminantAnalysis'):
        #     y_prob = self.classifier.predict_proba(features_input)
        # else:
        #     y_prob = np.zeros(2)
            
        y_prob = self.classifier.predict_proba(features_input)
        
        return y, y_prob
    
    
    def extractFeatures(self, trials_matrix): 
        # List for the features
        features_list = []
        
        # Input for the classifier
        features_input = np.zeros((trials_matrix.shape[0], len(self.classifier_features)))
        
        # Frequency filtering, spatial filtering, features evaluation
        for i in range(len(self.freqs) - 1):              
            # "Create" the band
            band = [self.freqs[i], self.freqs[i+1]]
            
            # Retrieve spatial filter
            W = self.W_list_band[i]
            
            # Frequency and spatial filter
            band_filter_trials_matrix = self.bandFilterTrials(trials_matrix, band[0], band[1])
            spatial_filter_trial = self.spatialFilteringW(band_filter_trials_matrix, W)
            
            # Features evaluation
            features = self.logVarEvaluation(spatial_filter_trial)
            
            features_list.append(features)
            # features_list.append(features[:, idx])
            
        # Features selection
        for i in range(len(self.classifier_features)):
            # Retrieve feature position
            feature_position = self.classifier_features[i]
            
            # Retrieve feature from the evaluated features
            features_input[:, i] = features_list[feature_position[0]][:, feature_position[1]]
            
        return features_input
        