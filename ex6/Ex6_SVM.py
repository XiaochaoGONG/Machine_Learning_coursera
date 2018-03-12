import numpy as np
import sys
sys.path.append('../')
import SVM
import matplotlib.pyplot as plt
from scipy.io import loadmat

import pdb

def bp():
    pdb.set_trace()

def plotData(X, y, label_x, label_y, leg, axes = None):
    pos_idx = (y == 1).ravel()
    neg_idx = (y == 0).ravel()
    if axes == None:
        axes = plt.gca()
    axes.scatter(X[pos_idx, 0], X[pos_idx, 1], c = 'r', marker = '+') 
    axes.scatter(X[neg_idx, 0], X[neg_idx, 1], c = 'y', marker = 'o')
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(leg, frameon= True, fancybox = True);

def plotSVC(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plotData(X, y, 'X1', 'X2', ['y == 1', 'y == 0'])
    #plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    print 'Number of support vectors: ', svc.n_support_

if __name__ == '__main__':
    data1 = loadmat('ex6/ex6data1.mat')

    X1 = data1['X']
    y1 = data1['y']

    print 'X1:', X1.shape
    print 'y1:', y1.shape

    axes = plotData(X1, y1, 'X1', 'X2', ['y == 1', 'y == 0'])
    plt.show()

    clf = SVM.SVC(C=1.0, kernel='linear')
    clf.fit(X1, y1.ravel())
    plotSVC(clf, X1, y1)
    plt.show()

    clf.set_params(C = 100)
    clf.fit(X1, y1.ravel())
    plotSVC(clf, X1, y1)
    plt.show()

    data2 = loadmat('ex6/ex6data2.mat')
    X2 = data2['X']
    y2 = data2['y']
    axes = plotData(X2, y2, 'X1', 'X2', ['y == 1', 'y == 0'])
    plt.show()

    clf2 = SVM.SVC(C = 50, kernel = 'rbf', gamma = 6)
    clf2.fit(X2, y2.ravel())
    plotSVC(clf2, X2, y2)
    plt.show()


