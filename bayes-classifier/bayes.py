import os
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Util: Load features and labels from a given folder
# ------------------------------------------------------------
def load_features_and_labels(root_dir):
    X = []
    y = []
    label_map = {}
    class_names = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
    
    for idx, class_name in enumerate(class_names):
        label_map[class_name] = idx
        feature_dir = Path(root_dir) / class_name / "features"
        feature_paths = list(feature_dir.glob("*.npy"))

        for fpath in feature_paths:
            vec = np.load(fpath)
            X.append(vec)
            y.append(idx)
    
    X = np.stack(X)  # (N, d)
    y = np.array(y)  # (N,)
    
    return X, y, label_map

# ------------------------------------------------------------
# Util: Convert int labels to one-hot encoded matrix
# ------------------------------------------------------------
def one_hot_encode(y, num_classes):
    N = len(y)
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), y] = 1
    return one_hot

# ------------------------------------------------------------
class BayesClassifier:
    def __init__(self, k, X, y):
        """
        k: int, number of classes
        X: np array, Nxd array with n samples of dimension d 
        y: ground truth,  Nxk ground truth labels  one hot encoding for k classes 
        """
        self.k = k
        self.X = X
        self.y = y
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.samplesPerClass = None 
        self.mean = np.zeros((self.k, self.d))
        self.cov = np.zeros((self.k, self.d, self.d))
        self.prior = None
        assert np.all(self.y.sum(axis=1) == 1), "Each row in y must be one-hot encoded"

    def computeNumberClasses(self):
        self.samplesPerClass = self.y.sum(axis=0)

    def computePrior(self):
        self.prior = self.samplesPerClass/self.N
        
    def estimateMean(self):
        for i in range(self.k):
            self.mean[i,:] = self.X[self.y[:,i]==1, :].mean(0)
    def estimateCovariance(self, epsilon=1e-6):
        for i in range(self.k):
            X_i = self.X[self.y[:, i] == 1, :]
            X_center = X_i - self.mean[i, :]
            n_i = X_i.shape[0]  # Number of samples in class i
            # Estimate covariance and add regularization
            cov = (X_center.T @ X_center) / (n_i - 1)
            self.cov[i, :, :] = cov + epsilon * np.eye(self.d)

    def fit(self):
        self.computeNumberClasses() 
        self.computePrior()
        self.estimateMean()
        self.estimateCovariance()

    def predict(self, X):
        """
        X: Nxd test samples 
        Output: Nxk one hot encoding for prediction 
        """
        pred = np.zeros([X.shape[0], self.k])
        for i in range(self.k):
            pred[:,i] = -1/2*np.log(np.linalg.det(self.cov[i,:,:])) -1/2* np.diag((X - self.mean[i,:])@np.linalg.inv(self.cov[i,:,:])@ (X - self.mean[i,:]).T) + np.log(self.prior[i])
        pred = np.argmax(pred,1) 
        return pred 
        
