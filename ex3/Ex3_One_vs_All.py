import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
sys.path.append('../')
import OneVsAll

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
    ##  2017.11.8
    ################################################
    # data.keys: X, y
    data = loadmat('ex3/ex3data1.mat')

    X = data['X']
    y = data['y']
    print 'Shape of X: ', X.shape
    print 'Shape of y: ', y.shape

    # Change value 10 to 0
    y0 = np.c_[[0 if yi == 10 else yi for yi in y.flatten()]]

    m = X.shape[0]
    num_Features = X.shape[1]
    num_Labels = y.shape[1]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100], :]

    displayData(sel)
    plt.show()
    
    ################################################
    ##  2. Computing theta
    ##  2017.11.10
    ################################################
    all_theta = OneVsAll.oneVsAll(X, y0, 0.1)

    ################################################
    ##  3. Predicting accuracy
    ##  2017.11.10
    ################################################
    pred = OneVsAll.predict(all_theta, X)

    accuracy = sum(pred == y0.flatten()) / float(m)
    print 'Predicting accuracy: %f.' % accuracy

    ################################################
    ##  4. Randomly Predicting
    ##  2017.11.12
    ################################################
    rand_indices = np.random.permutation(m)
    for i in xrange(m):
        sel = X[rand_indices[i], :].reshape(1, -1)
        ax = displayData(sel)

        pred = OneVsAll.predict(all_theta, sel)
        ax.set_xlabel('Predict Number: {}'.format(pred))
        plt.show()

        

