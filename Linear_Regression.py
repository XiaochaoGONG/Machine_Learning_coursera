import numpy as np

def featureNormalize(X):
    mu = np.mean(X, 0)
    X = X - mu
    sigma = np.std(X, 0)
    X_norm = X / sigma
    return (X_norm, mu, sigma)

def computeCost(X, y, theta):
    m = y.shape[0]
    h = X.dot(theta)
    delta = h - y
    J = (1 / float(2 * m)) * (delta.T.dot(delta))
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    J = np.zeros(iterations)
    for i in xrange(iterations):
        h = X.dot(theta)
        theta = theta - alpha * (1 / float(m)) * X.T.dot(h - y)
        J[i] = computeCost(X, y, theta)
    return (theta, J)

def normalEquation(X, y):
    # theta = inv(X.T * X) * X.T * y
    x = np.mat(X)
    return x.T.dot(x).I.dot(x.T).dot(y)

