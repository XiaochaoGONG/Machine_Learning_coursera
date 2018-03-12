import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
from Linear_Regression import *
import pdb

def bp():
    pdb.set_trace()

def plotData(X, y):
    plt.figure('Ex1')   # open a new figure window
    plt.scatter(X, y, c = 'r', marker = 'x')

    xmin = min(X); xmax = max(X)
    ymin = min(y); ymax = max(y)
    plt.xlim(xmin - 2, xmax + 2)
    plt.ylim(ymin - 2, ymax + 2)
    
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

if __name__ == '__main__':
    ############################################
    ##  1. Plotting
    ##  2017.10.24
    ############################################
    print 'Ploting Data First ...'
    
    data = np.loadtxt('ex1/ex1data1.txt', delimiter = ',')
    
    X = data[:, 0]
    y = data[:, 1]
    
    if X.shape[0] != y.shape[0]:
        print 'Example is not equal from X to y!'
        exit(-1)
    
    m = y.shape[0]
    
    num_Features = X.shape[1] if (len(X.shape) != 1) else 1 
    num_Labels = y.shape[1] if (len(y.shape) != 1) else 1 
    
    # Plot Data
    plotData(X, y)
    plt.show()
    
    ############################################
    ##  2. Cost and Gradient Descent
    ##  2017.10.25
    ############################################
    
    X = np.c_[np.ones([m, 1]), X]   # Add all ones in the first column
    y = y.reshape(-1, num_Labels)
    theta = np.zeros([num_Features + 1, 1])  # Initialize parameters
    
    # Gradient Descent Settings
    iters = 1500
    alpha = 0.01
    
    print 'Testing the cost function...'
    
    J = computeCost(X, y, theta)
    print 'With theta = [0 , 0], Cost computed = %f' % J
    print 'Expected cost value is 32.07\n'
    
    # With more test
    J = computeCost(X, y, np.c_[[-1, 2]])
    print 'With theta = [-1, 2], Cost computed = %f' % J
    print 'Expected cost value is 54.24\n'
    
    print 'Running Gradient Descent...'
    theta, J = gradientDescent(X, y, theta, alpha, iters)
    print 'Theta found by gradient descent: ', theta
    print 'Expected theta values', np.c_[[-3.6303, 1.1664]]
    
    # Plot the linear fit
    X_raw = X[:, num_Features].reshape(-1, num_Features)
    plotData(X_raw, y)
    plt.plot(X_raw, X.dot(theta), label = 'Gradient Descent')
    # Compare with scikit lib
    regression = LinearRegression()
    regression.fit(X_raw.reshape(-1, num_Features), y)
    plt.plot(X_raw, regression.predict(X_raw), label = 'Scikit Linear Regression')
    plt.legend(loc = 'best')
    plt.show()
    
    # Plot Iterations of computing cost function
    plt.plot(J)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.xlim(0, 1600)
    plt.show()
    
    # Predict values for population size of 35,000 and 70,000
    predict1 = np.c_[1, 3.5].dot(theta) * 10000
    print 'For population = 35,000, we predict a profit of ', predict1
    predict2 = np.c_[1, 7].dot(theta) * 10000
    print 'For population = 35,000, we predict a profit of ', predict2, '\n'
    
    ############################################
    ##  3. Visualizing J
    ##  2017.10.25
    ############################################
    print 'Visualizing J...'
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
    
    for (row, column), v in np.ndenumerate(J_vals):
        t = np.c_[[theta0_vals[column], theta1_vals[row]]]
        J_vals[row, column] = computeCost(X, y, t)
        
    fig = plt.figure(figsize = (15, 6))
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection = '3d')
    
    # Contour plot
    ax1.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
    ax1.scatter(theta[0], theta[1], c = 'r')
    
    # Surface plot
    xx, yy = np.meshgrid(theta0_vals, theta1_vals, indexing='xy')
    ax2.plot_surface(xx, yy, J_vals, rstride=1, cstride=1, alpha=0.7, cmap=plt.cm.jet)
    ax2.set_zlabel('Cost')
    ax2.set_zlim(J_vals.min(), J_vals.max())
    ax2.view_init(elev=15, azim=230)
    
    for ax in fig.axes:
        ax.set_xlabel(r'$\theta_0$', fontsize = 17)
        ax.set_ylabel(r'$\theta_1$', fontsize = 17)
    
    plt.show()
