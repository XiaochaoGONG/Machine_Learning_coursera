import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def featureNormalize(X):
    mu = np.mean(X, 0)
    X = X - mu
    sigma = np.std(X, 0)
    X_norm = X / sigma
    return (X_norm, mu, sigma)

def costFunction(theta, X, y):
    # divide by m is not necessary because m is constant
    z = X.dot(theta.reshape(-1, 1))
    h = sigmoid(z)

    cost = y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))
    cost = - cost 
    return cost.flatten()

def costFunctionReg(theta, X, y, lamb):
    cost = costFunction(theta, X, y)
    theta[0] = 0
    cost += theta.T.dot(theta) * lamb / 2
    return cost

def gradientFunction(theta, X, y):
    # divide by m is not necessary because m is constant
    z = X.dot(theta.reshape(-1, 1))
    h = sigmoid(z)

    grad = X.T.dot(h - y)
    return grad.flatten()

def gradientFunctionReg(theta, X, y, lamb):
    grad = gradientFunction(theta, X, y)
    theta[0] = 0
    grad += theta * lamb
    return grad

def predict(theta, X):
    Z = X.dot(theta)
    r = [1 if z > 0 else 0 for z in Z]
    r = np.c_[r]
    return r
