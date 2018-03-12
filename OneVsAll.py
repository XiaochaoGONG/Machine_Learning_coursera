import numpy as np
from scipy.optimize import minimize
import sys
import Logistic_Regression as LR
import pdb

def oneVsAll(X, y, lamb):
    m = X.shape[0]
    num_features = X.shape[1] + 1
    num_labels = len(set(y.flatten()))

    initial_theta = np.zeros(num_features)
    X = np.c_[np.ones(m), X]

    all_theta = np.zeros([num_labels, num_features])

    for i in xrange(num_labels):
        res = minimize(LR.costFunctionReg, initial_theta, args = (X, (y == i), lamb), method = None, jac = LR.gradientFunctionReg, options = {'maxiter' : 150})
        all_theta[i, :] = res.x

    return all_theta

def predict(all_theta, X):
    X = np.c_[np.ones(X.shape[0]), X]
    Z = X.dot(all_theta.T)

    res = [np.argmax(z) for z in Z]

    return res
