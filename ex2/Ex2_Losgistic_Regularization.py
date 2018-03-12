import matplotlib.pyplot as plt
from Logistic_Regression import *
import Ex2_logistic
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import pdb

def bp():
    pdb.set_trace()

if __name__ == '__main__':

    data = np.loadtxt('ex2/ex2data2.txt', delimiter = ',')

    X = np.c_[data[:, 0 : 2]]
    y = np.c_[data[:, 2]]

    m = y.shape[0]
    num_Features = X.shape[1]
    num_Labels = y.shape[1]

    ##############################################
    ##  1. Visualizing Data
    ##  2017.11.05
    ##############################################
    plt.figure('Ex2_Reg')
    leg = ['y = 1', 'y = 0']
    Ex2_logistic.plotData(X, y, 'Microchip Test 1', 'Microchip Test 2', leg)
    plt.show()

    ##############################################
    ##  2. Map Multi Features
    ##  2017.11.05
    ##############################################
    print 'Original Feature Num: %d' % num_Features
    X_origin = X
    # Note that this function inserts a column with 'ones'
    poly = PolynomialFeatures(6)
    X = poly.fit_transform(X)
    num_Features = X.shape[1]
    print 'Actual Feature Num after Polynominalization: %d' % num_Features

    ##############################################
    ##  3. Cost Function and Gradient with Regularization
    ##  2017.11.05
    ##############################################
    cost = costFunctionReg(np.ones(num_Features), X, y, 10)
    grad = gradientFunctionReg(np.ones(num_Features), X, y, 10)
    print 'Cost : %f' % (cost / m)
    print 'Gradient : ' 
    print (grad[0:5] / m)
    
    ##############################################
    ##  4. Computing theta using minimize
    ##  2017.11.05
    ##############################################
    initial_theta = np.zeros(num_Features)
    res = minimize(costFunctionReg, initial_theta, args = (X, y, 0), method = None, jac = gradientFunctionReg, options = {'maxiter' : 3000})

    pred = predict(res.x, X)
    accuracy = sum(pred == y) / float(m)
    print 'Accuracy with lambda = 1: %f' % accuracy

    ##############################################
    ##  5. Plot Decision Boundary
    ##  2017.11.05
    ##############################################
    fig, axes = plt.subplots(1, 3, sharey = True, figsize=(17,5))
    text = ['Overfitting', 'Just work', 'Underfitting']

    for i, lamb in enumerate([0, 1, 100]):
        res = minimize(costFunctionReg, initial_theta, args = (X, y, lamb), method = None, jac = gradientFunctionReg, options = {'maxiter' : 3000})
        
        theta = res.x
        pred = predict(theta, X)
        accuracy = 100 * sum(pred == y) / float(m)

        Ex2_logistic.plotData(X_origin, y, 'Microchip Test 1', 'Microchip Test 2', leg, axes.flatten()[i])

        # Decision Boundary
        x1_min, x1_max = min(X_origin[:,0]), max(X_origin[:,0])
        x2_min, x2_max = min(X_origin[:,1]), max(X_origin[:,1])
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

        z = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta)
        h = sigmoid(z)
        h = h.reshape(xx1.shape)
        axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
        axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), lamb))
        x1_mid = (x1_min + x1_max) / 2.0
        x1_len = x1_max - x1_min
        axes.flatten()[i].text(x1_mid - x1_len / 4.0, x2_min, text[i], fontdict = {'size' : 16})

    plt.show()

