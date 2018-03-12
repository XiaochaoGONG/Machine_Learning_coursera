from math import log
import pdb
import numpy as np

def Ent(D):
    l_total = float(sum(D))

    ent = 0
    for index, value in enumerate(D):
        if value != 0:
            pk = value / l_total
            ent += pk * log(pk) / log(2)

    return -ent

def getElementsClassified(D, pos_index):
    d_pos = D[:pos_index]
    d_neg = D[pos_index:]

    d = set(D)
    cnt_pos = []
    cnt_neg = []
    for index, value in enumerate(d):
        cnt_pos.append(sum(d_pos == value)) 
        cnt_neg.append(sum(d_neg == value))

    return zip(cnt_pos, cnt_neg)

def gainEnt(D, Dv):
    ent_D = Ent(D)
    cnt_total = float(sum(D))
    gain = ent_D
    for i, dv in enumerate(Dv):
        ent_dv = Ent(dv)
        cnt_dv = float(sum(dv))
        gain -= cnt_dv / cnt_total * ent_dv

    return gain



if __name__ == '__main__':

    y = np.ones(17)
    y = getElementsClassified(y, 8)
    print y
    print 'Entro:\t', Ent(y[0])

    X0 = [1, 2, 2, 1, 3, 1, 2, 2, 2, 1, 3, 3, 1, 3, 2, 3, 1]
    X1 = [1] * 5 + [2] * 4 + [3, 3, 1, 2, 2, 2, 1, 1]
    X2 = [1, 2, 1, 2, 1, 1, 1, 1, 2, 3, 3, 1, 1, 2, 1, 1, 2]
    X3 = [1] * 6 + [2, 1] + [2, 1, 3, 3, 2, 2, 1, 3, 2]
    X4 = [1] * 5 + [2] * 4 + [3] * 3 + [1, 1, 2, 3, 2]
    X5 = [1] * 5 + [2] * 2 + [1] * 2 + [2, 1, 2, 1, 1, 2, 1, 1]
    X = np.c_[X0, X1, X2, X3, X4, X5]
    G = []
    for xi in X.T:
        xi_cnt = getElementsClassified(xi, y[0][0])
        xi_gain = gainEnt(y[0], xi_cnt)
        G.append(xi_gain)
    
    print 'Gain:'
    for i, v in enumerate(G):
        print 'X%d:\t' % i, v
    m = max(G)
    a1 = G.index(m)
    print 'Max is X%d\t:' % a1, m
    print '\n'

    Xi = X[:, a1]
    feature_classified = getElementsClassified(Xi, 8)
    X_temp = np.delete(X, a1, axis = 1)
    for i in range(len(feature_classified)):
        print '[Feature %d]' % i 
        X_new = []
        for j, v in enumerate(Xi):
            if v == i + 1:
                X_new.append(X_temp[j, :])

        shape = np.shape(X_new)
        X_new = np.reshape(X_new, shape)
        G = []
        for xi in X_new.T:
            xi_cnt = getElementsClassified(xi, feature_classified[i][0])
            xi_gain = gainEnt(feature_classified[i], xi_cnt)
            G.append(xi_gain)

        print 'Gain:'
        for i, v in enumerate(G):
            index = [i if i < a1 else i + 1][0]
            print 'X%d\t:' % index, v
        m = max(G)
        i = G.index(m)
        index = [i if i < a1 else i + 1][0]
        print 'Max is X%d\t:' % index, m
