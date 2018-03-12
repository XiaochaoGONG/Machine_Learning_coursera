import numpy as np
import pandas as pd
from Logistic_Regression import *
import math
import pdb

def bp():
    pdb.set_trace()

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def randInitializeWeights(L_in, L_out):
    epsilon_init = np.sqrt(6 / float(L_in + L_out))
    W = np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init
    return W

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   yhat, y, m, lamb):
    nn_params = nn_params.flatten()

    Theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : ].reshape(num_labels, (hidden_layer_size + 1))

    theta1 = Theta1[:, 1 : ]
    theta2 = Theta2[:, 1 : ]
    cost = y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
    cost = -np.sum(cost)
    reg = (np.sum(theta1 ** 2) + np.sum(theta2 ** 2)) * lamb / 2

    return (cost + reg) / m

def forwardPropagation(nn_params,
                        input_layer_size,
                        hidden_layer_size,
                        num_labels,
                        x, y, m):

    nn_params = nn_params.flatten()
    Theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : ].reshape(num_labels, (hidden_layer_size + 1))

    a0_b = np.r_[np.ones([1, m]), x]       # (input_layer_size + 1) * m
    z1 = Theta1.dot(a0_b)                  # hidden_layer_size * m
    a1 = sigmoid(z1)
    a1_b = np.r_[np.ones([1, m]), a1]      # (hidden_layer_size + 1) * m
    z2 = Theta2.dot(a1_b)                  # num_labels * m
    a2 = sigmoid(z2)

    fp = {}
    fp['a0'] = a0_b
    fp['a1'] = a1_b
    fp['a2'] = a2
    fp['z1'] = z1
    fp['z2'] = z2

    return fp

def backPropagation(nn_params,
                    input_layer_size,
                    hidden_layer_size,
                    num_labels,
                    fp, y, m, lamb):
    a0 = fp['a0']
    a1 = fp['a1']
    a2 = fp['a2']
    z1 = fp['z1']
    nn_params = nn_params.flatten()

    Theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : ].reshape(num_labels, (hidden_layer_size + 1))

    theta1 = Theta1[:, 1 : ]
    theta2 = Theta2[:, 1 : ]

    delta2 = a2 - y                         # num_labels * m
    delta1 = theta2.T.dot(delta2) * sigmoidGradient(z1)        # hidden_layer_size * m
    
    Delta1 = delta1.dot(a0.T)            # hidden_layer_size * (input_layer_size + 1)
    Delta2 = delta2.dot(a1.T)            # num_labels * (hidden_layer_size + 1)

    Theta1_grad = Delta1 + Theta1 * lamb
    Theta1_grad[:, 0] = Delta1[:, 0]
    Theta2_grad = Delta2 + Theta2 * lamb
    Theta2_grad[:, 0] = Delta2[:, 0]

    grad = np.r_[Theta1_grad.ravel(), Theta2_grad.ravel()] / m
    return grad

def checkNNGradients(lamb = 0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    X = randInitializeWeights(input_layer_size - 1, m).T
    y = 1 + np.mod(range(m), num_labels)

    nn_params = np.r_[Theta1.ravel(), Theta2.ravel()]

    fp = forwardPropagation(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, m)
    cost = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, fp['a2'], y, m, lamb)
    
    grad = backPropagation(nn_params, input_layer_size, hidden_layer_size, num_labels, fp, y, m, lamb)

    # Compute Numerical Gradient
    e = 1e-4
    num_params = len(nn_params)
    perturb = np.zeros(num_params)
    grad_t = np.zeros(num_params)
    print "=======Check Gradients========="
    print "Numerical\tAnalytical"
    for i in xrange(num_params):
        perturb[i] = e
        fp = forwardPropagation(nn_params - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, m)
        cost_t1 = nnCostFunction(nn_params - perturb, input_layer_size, hidden_layer_size, num_labels, fp['a2'], y, m, lamb)
        fp = forwardPropagation(nn_params + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, m)
        cost_t2 = nnCostFunction(nn_params + perturb, input_layer_size, hidden_layer_size, num_labels, fp['a2'], y, m, lamb)

        grad_t[i] = (cost_t2 - cost_t1) / (2 * e)
        perturb[i] = 0
        print "%f\t%f" % (grad_t[i], grad[i])

    diff = np.linalg.norm(grad_t - grad) / np.linalg.norm(grad_t + grad)
    return diff

def predict(Theta1, Theta2, X):
    res = [np.argmax(h) for h in hk]

    return res
