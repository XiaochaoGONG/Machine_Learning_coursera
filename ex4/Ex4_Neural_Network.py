import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
sys.path.append('../')
import os
import NeuralNetwork as nn

import pdb

def bp():
    pdb.set_trace()

def displayData(selected):
    """
    Reform m * n vector

    m: num of item
    n: pixels of one item

    To display the image, reforme pixels as w * h = n

    [Example] 
    Input: 100 * 400
    Output: Large matrix: 10 * 10, 
            For every element Small matrix: 20 * 20

            0 ... 399 (400)     0...19 0...19 0...19 ... (10 * 20)
            .           ===>    20..39 20..39 20..39 ... (10 * 20)
            .                   ...
            99
            (100)               (10 * 20)

    """
    import math
    m = selected.shape[0]
    n = selected.shape[1] 
    
    # For one item, width and height
    example_width = int(math.sqrt(n))
    example_height = n / example_width
    
    # Num of items
    display_rows = int(math.sqrt(m))
    display_cols = m / display_rows

    # Between images padding
    pad = 1
    # Setup blank display
    display_array = np.zeros([(pad + display_rows * (example_height + pad)),\
            (pad + display_cols * (example_width + pad))])

    # Copy each example into a patch on the display array
    for j in range(display_rows):
        for i in range(display_cols):
            max_val = max(abs(selected[i + j * display_cols, :]))
            example = selected[i + j * display_cols, :].reshape(example_height, example_width).T / max_val
            row_base = pad + j * (example_height + pad)
            col_base = pad + i * (example_width + pad)
            display_array[row_base : (row_base + example_height), col_base : (col_base + example_width)] = example
    
    fig, ax = plt.subplots()
    ax.imshow(display_array)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    return ax


if __name__ == '__main__':

    ################################################
    ##  1. Loading and Visualizing Data
    ##  2017.11.14
    ################################################
    # data.keys: X, y
    data = loadmat('ex4/ex4data1.mat')

    X = data['X']
    Y = data['y']
    print 'Shape of X: ', X.shape
    print 'Shape of y: ', Y.shape

    m = X.shape[0]
    num_Features = X.shape[1]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100], :]

    #displayData(sel)
    #plt.show()

    input_layer_size = num_Features
    hidden_layer_size = 25

    ################################################
    ##  2. Loading Weights
    ##  2017.11.14
    ################################################
    data = loadmat('ex4/ex4weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']

    nn_params = np.c_[[Theta1.flatten()], [Theta2.flatten()]]

    ################################################
    ##  3. Computing Cost (Feedforward)
    ##  2017.11.14
    ################################################
    # Check 1

    lamb = 3

    x = X.T
    y = pd.get_dummies(Y.ravel()).as_matrix().T
    num_Labels = y.shape[0]

    fp = nn.forwardPropagation(nn_params, input_layer_size, hidden_layer_size, num_Labels, x, y, m)
    cost = nn.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_Labels, fp['a2'], y, m, lamb)

    print 'Cost with lamb == 3: %6f, should be 0.576051' % cost

    ################################################
    ##  4. Back Propagation
    ##  2017.11.16
    ################################################

    # Test sigmoid gradient
    g_t = nn.sigmoidGradient(np.c_[-1, -0.5, 0, 0.5, 1])
    print 'Test sigmoid gradient', g_t

    grad = nn.backPropagation(nn_params, input_layer_size, hidden_layer_size, num_Labels, fp, y, m, lamb)
    print grad
    # Check gradient function
    diff = nn.checkNNGradients(3)
    print "Diff relative is", diff, "should < 1e-9"

    ################################################
    ##  5. Training Theta
    ##  2017.11.20
    ################################################
    # Initialize Parameters break symmetry breaking
    initial_Theta1 = nn.randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = nn.randInitializeWeights(hidden_layer_size, num_Labels)
    initial_nn_params = np.r_[initial_Theta1.ravel(), initial_Theta2.ravel()]

    maxIter = 5000
    alpha = 3
    for i in range(maxIter):
        fp = nn.forwardPropagation(initial_nn_params, input_layer_size, hidden_layer_size, num_Labels, x, y, m)
        cost = nn.nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_Labels, fp['a2'], y, m, lamb)

        if i % 100 == 0:
            print cost

        if cost < 0.001:
            break

        grad = nn.backPropagation(initial_nn_params, input_layer_size, hidden_layer_size, num_Labels, fp, y, m, lamb)

        initial_nn_params = initial_nn_params - alpha * grad

    ################################################
    ##  6. Predicting
    ##  2017.11.20
    ################################################
    nn_params = initial_nn_params
    fp = nn.forwardPropagation(initial_nn_params, input_layer_size, hidden_layer_size, num_Labels, x, y, m)

    res = np.array([np.argmax(h) for h in fp['a2'].T])

    res = res + 1

    accuracy = sum(res == Y.flatten()) / float(m) * 100
    print 'Accuracy is : %4f' % accuracy
