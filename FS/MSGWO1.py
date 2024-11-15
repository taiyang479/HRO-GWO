import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
import math
import copy

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def single_binary_conversion(X, thres, dim):
    Xbin = np.zeros([1, dim], dtype='int')
    for d in range(dim):
        s = X[d]
        if s > thres:
            Xbin[0, d] = 1
        else:
            Xbin[0, d] = 0
    return Xbin


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            s = X[i, d]
            if s > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def msgwo(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']

    N = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)
    X_cross = np.zeros([1, dim], dtype='float')

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')

    X_i_GWO = np.zeros([1, dim], dtype='float')
    X_i_DLH = np.zeros([1, dim], dtype='float')

    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]

        if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]

        if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]


    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    print("Iteration:", t + 1)
    print("Best (MSGWO):", curve[0, t])
    t += 1

    X = np.hstack((X, fit))
    groupSize = int(N / 3)
    X = X[np.argsort(X[:,-1])]
    X = X[:, 0:-1]
    while t < max_iter:

            w = 0.01
            a = 2 - t * (2 / max_iter)

            for k in range(2 * groupSize, N):
                trial_rice = np.zeros([N, dim], dtype='float')
                for d in range(dim):
                    n = np.random.randint(low=groupSize, high=2 * groupSize)
                    neighbor = X[n]
                    C = 2 * rand()
                    Dg = abs(C * neighbor[d] - X[k, d])
                    A = 2 * a * rand() - a
                    trial_rice[0, d]= neighbor[d] - A * Dg
                    trial_rice[0, d] = boundary(trial_rice[0, d], lb[0, d], ub[0, d])
                X[k, :] = trial_rice[0, :]

            for j in range(groupSize, 2 * groupSize):
                # Parameter C (3.4)
                for d in range(dim):
                    C1 = 2 * rand()
                    C2 = 2 * rand()
                    C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                    Dalpha = abs(C1 * Xalpha[0, d] - X[j, d])
                    Dbeta = abs(C2 * Xbeta[0, d] - X[j, d])
                    Ddelta = abs(C3 * Xdelta[0, d] - X[j, d])
                # Parameter A (3.3)
                    A1 = 2 * a * rand() - a
                    A2 = 2 * a * rand() - a
                    A3 = 2 * a * rand() - a

                    X1 = Xalpha[0, d] - A1 * Dalpha
                    X2 = Xbeta[0, d] - A2 * Dbeta
                    X3 = Xdelta[0, d] - A3 * Ddelta
                # Update wolf (3.7)
                    X_i_GWO[0, d] = (X1 + X2 + X3) / 3
                    X_i_GWO[0, d] = boundary(X_i_GWO[0, d], lb[0, d], ub[0, d])

                X[j, :] = X_i_GWO[0, :]

            for i in range(0, groupSize):
                p = rand()
                l = -1 + 2 * rand()
                b = 1
                if p < 0.5:
                    for d in range(dim):
                        C1 = 2 * rand()
                        C2 = 2 * rand()
                        C3 = 2 * rand()

                        Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                        Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                        Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])

                        A1 = 2 * a * rand() - a
                        A2 = 2 * a * rand() - a
                        A3 = 2 * a * rand() - a

                        X1 = Xalpha[0, d] - A1 * Dalpha
                        X2 = Xbeta[0, d] - A2 * Dbeta
                        X3 = Xdelta[0, d] - A3 * Ddelta

                        X[i, d] = (X1 + X2 + X3) / 3
                        X[i, d] = boundary(X_i_GWO[0, d], lb[0, d], ub[0, d])
                else:
                    for d in range(dim):
                        C1 = 2 * rand()
                        C2 = 2 * rand()
                        C3 = 2 * rand()
                        Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                        Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                        Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                        D = (Dalpha + Dbeta + Ddelta) / 3

                        X[i, d] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + X[i, d]
                        X[i, d] = boundary(X[0, d], lb[0, d], ub[0, d])

            Xbin = binary_conversion(X, thres, N, dim)
            # Fitness
            for i in range(N):
                fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
                if fit[i, 0] < Falpha:
                    Xalpha[0, :] = X[i, :]
                    Falpha = fit[i, 0]

                if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                    Xbeta[0, :] = X[i, :]
                    Fbeta = fit[i, 0]

                if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                    Xdelta[0, :] = X[i, :]
                    Fdelta = fit[i, 0]


            X = np.hstack((X, fit))
            X = X[np.argsort(X[:, -1])]
            X = X[:, 0:-1]

            curve[0, t] = Falpha.copy()
            print("Iteration:", t + 1)
            print("Best (MSGWO):", curve[0, t])
            t += 1

    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'fitness': curve[0, max_iter - 1]}

    return gwo_data