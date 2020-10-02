"""Pointcloud regressors."""
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.base import BaseEstimator
from tofnet.pointcloud.utils import project_mat

class SVDRegressor(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.norm_ = None

    def fit(self, X, _):
        self.norm_ = np.linalg.svd(X)[2][-1]
        self.norm_ = self.norm_.reshape(-1, 1)

    def score(self, X, _):
        return np.mean(self.predict(X))

    def predict(self, X):
        Xproj = project_mat(X, self.norm_)
        return np.linalg.norm(Xproj-X, axis=1)
