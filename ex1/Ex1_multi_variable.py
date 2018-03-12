from Linear_Regression import *

if __name__ == '__main__':
    ############################################
    ##  1. Feature Normalization
    ##  2017.10.28
    ############################################
    print 'Loading data ...'
    
    data = np.loadtxt('ex1/ex1data2.txt', delimiter = ',')
    
    X = data[:, 0:2]
    y = data[:, 2]
    
    if X.shape[0] != y.shape[0]:
        print 'Example is not equal from X to y!'
        exit(-1)

    m = y.shape[0]
    num_Features = X.shape[1] if (len(X.shape) != 1) else 1 
    num_Labels = y.shape[1] if (len(y.shape) != 1) else 1 

    print 'Normalizing Features ...'
    (X_norm, mu, sigma) = featureNormalize(X)


    ############################################
    ##  2. Gradient Descent
    ##  2017.10.31
    ############################################
    print 'Running gradient descent ...'

    alpha = 0.1
    num_iters = 50

    # Add ones
    X_norm = np.c_[np.ones([m, 1]), X_norm]
    y = y.reshape(-1, num_Labels)
    theta = np.zeros([num_Features + 1, 1])

    (theta, J_history) = gradientDescent(X_norm, y, theta, alpha, num_iters)

    ############################################
    ##  3. Predict
    ##  2017.10.31
    ############################################
    X_temp = (np.c_[1650, 3] - mu) / sigma
    X_temp = np.c_[np.ones([1, 1]), X_temp]

    price = X_temp.dot(theta) 
    print 'Predict Price with Gradient Descent is ', price

    
    ############################################
    ##  4. Normal Equation
    ##  2017.11.01
    ############################################
    X = np.c_[np.ones([m, 1]), X]
    theta = normalEquation(X, y)

    X_t = np.c_[1, 1650, 3]
    price = X_t.dot(theta)
    print 'Predict Price with Normal Equation is ', price



