
import scipy.linalg
import numpy as np

class CSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters

    def fit(self,x_train,y_train):
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float32)
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
            cov_x_trial /= np.trace(cov_x_trial)
            cov_x[y_trial, :, :] += cov_x_trial

        cov_x = np.asarray([cov_x[cls]/np.sum(y_labels==cls) for cls in range(2)])
        cov_combined = cov_x[0]+cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined,cov_x[0])
        sort_indices = np.argsort(abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:,sort_indices]
        u_mat = np.transpose(u_mat)

        return eig_values, u_mat

    def transform(self,x_trial,eig_vectors):
        z_trial = np.matmul(eig_vectors, x_trial)
        z_trial_selected = z_trial[:self.m_filters,:]
        z_trial_selected = np.append(z_trial_selected,z_trial[-self.m_filters:,:],axis=0)
        sum_z2 = np.sum(z_trial_selected**2, axis=1)
        sum_z = np.sum(z_trial_selected, axis=1)
        var_z = (sum_z2 - (sum_z ** 2)/z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)
        sum_var_z = sum(var_z)
        return np.log(var_z/sum_var_z)


class FBCSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters
        self.fbcsp_filters_multi=[]

    def fit(self,x_train_fb,y_train):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]            
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters=get_csp(x_train_fb,y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)

            print(len(self.fbcsp_filters_multi))

    def transform(self,x_data,class_idx=0):
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros((n_trials,self.m_filters*2*len(x_data)),dtype=np.float32)
        for i in range(n_fbanks):
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            for k in range(n_trials):
                x_trial = np.copy(x_data[i,k,:,:])
                csp_feat = self.csp.transform(x_trial,eig_vectors)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 2]  = csp_feat[j]
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 1]= csp_feat[-j-1]

        return x_features