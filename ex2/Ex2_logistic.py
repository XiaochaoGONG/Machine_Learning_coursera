import matplotlib.pyplot as plt
from Logistic_Regression import *
from scipy.optimize import minimize
import pdb

def bp():
    pdb.set_trace()

def plotData(X, y, label_x, label_y, leg, axes = None):
    pos_idx = [idx for (idx, val) in enumerate(y[:]) if (val == 1)]
    neg_idx = [idx for (idx, val) in enumerate(y[:]) if (val == 0)]
    if axes == None:
        axes = plt.gca()
    axes.scatter(X[pos_idx, 0], X[pos_idx, 1], c = 'r', marker = '+') 
    axes.scatter(X[neg_idx, 0], X[neg_idx, 1], c = 'y', marker = 'o')
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(leg, frameon= True, fancybox = True);

def plotDecisionBoundary(theta, X, y):
    x1_min, x1_max = min(X[:, 1]), max(X[:, 1])
    x2_min, x2_max = min(X[:, 2]), max(X[:, 2])

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    xx = np.c_[np.ones([xx1.ravel().shape[0], 1]), xx1.ravel(), xx2.ravel()]

    z = xx.dot(theta.reshape(-1, 1))
    h = sigmoid(z)
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths = 1, colors = 'b')



if __name__ == '__main__':
    data = np.loadtxt('ex2/ex2data1.txt', delimiter = ',')
    print 'data shape: ', data.shape

    X = data[:, 0:2]
    y = data[:, 2]

    m = y.shape[0]
    num_Features = X.shape[1] if (len(X.shape) != 1) else 1
    num_Labels = y.shape[1] if (len(y.shape) != 1) else 1

    #########################################
    ##  1. Plotting
    ##  2017.11.02
    #########################################
    plt.figure('Ex2')
    leg = ['Admitted', 'Not admitted']
    plotData(X, y, 'Exam 1 score', 'Exam 2 score', leg)


    #########################################
    ##  2. Compute Cost and Gradient
    ##  2017.11.02
    #########################################
    X = np.c_[np.ones([m, 1]), X]
    y = y.reshape(-1, num_Labels)

    initial_theta = np.zeros([num_Features + 1, 1])

    cost = costFunction(initial_theta, X, y)
    grad = gradientFunction(initial_theta, X, y)
    print 'Cost at initial theta (zeros): %f' % (cost / m)
    print 'Expected cost (approx): 0.693\n'
    print 'Gradient at initial theta (zeros):'
    print (grad / m)
    print 'Expected gradients (approx):'
    print ' -0.1000\t-12.0092\t-11.2628\n'

    test_theta = np.c_[[-24, 0.2, 0.2]]
    cost = costFunction(test_theta, X, y)
    grad = gradientFunction(test_theta, X, y)
    print 'Cost at initial theta (zeros): %f' % (cost / m)
    print 'Expected cost (approx): 0.218\n'
    print 'Gradient at initial theta (zeros):'
    print (grad / m)
    print 'Expected gradients (approx):'
    print ' 0.043\t2.566\t2.647\n'

    #########################################
    ##  3. Optimizing
    ##  2017.11.03
    #########################################
    res = minimize(costFunction, initial_theta, args = (X, y), method = None, jac = gradientFunction, options = {'maxiter' : 400})
    print res.message
    if res.status == 0:
        # Success
        theta = res.x.reshape(-1, 1)
        print 'Theta : '
        print theta


    #########################################
    ##  4. Predict and Accuracies
    ##  2017.11.03
    #########################################
    X_test = np.c_[1, 45, 85]
    prob = sigmoid(X_test.dot(theta))
    print 'For a student with score 45 and 85, we predict an admission probability of %f' % prob
    plt.scatter(X_test[:, 1], X_test[:, 2], marker='v')
    leg.append('(45, 85)')
    plt.legend(leg, loc = 'best')
    plotDecisionBoundary(theta, X, y)
    plt.show()
    
    p = predict(theta, X)
    print 'Train Accuracy: %f' % np.mean(p == y)
