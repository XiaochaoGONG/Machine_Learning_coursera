import numpy as np
from sklearn.svm import SVC

def gaussianKernel(x1, x2, sigma=2):
    norm = (x1 - x2).T.dot(x1 - x2)
    return np.exp(-norm / (2 * sigma ** 2))

