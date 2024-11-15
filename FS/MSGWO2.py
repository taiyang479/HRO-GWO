import numpy as np
from numpy.random import rand
from FS.functionHO import Fun

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


def msgwo2(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

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
    print("Best (MSGWO2):", curve[0, t])
    t += 1
    neighbor = np.zeros([1, dim], dtype='float')
    step = np.zeros([1, dim], dtype='float')

    while t < max_iter:
        # Coefficient decreases linearly from 2 to 0
        a = 2 - t * (2 / max_iter)
        for i in range(0,N):
            r_v = np.random.uniform(0, 1)
            cp = 2 * a * r_v - a
            if abs(cp) >= 1:
                q = np.random.random()
                if q < 0.5:
                    random_number = np.random.randint(low=0, high=N)
                    neighbor = X[random_number]
                    bin_neibor = single_binary_conversion(neighbor, thres, dim)
                    F_neibor = Fun(xtrain, ytrain, bin_neibor[0, :], opts)
                    bin_x = single_binary_conversion(X[i, :], thres, dim)
                    F_x = Fun(xtrain, ytrain, bin_x[0, :], opts)
                    for d in range(dim):
                        step[0,d] = X[i,d] - neighbor[d]
                        if(F_x < F_neibor):
                            X[i,d] = X[i, d] + r_v * step[0,d]
                            X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                        else:
                            X[i, d] = X[i, d] - r_v * step[0,d]
                            X[i, d] = boundary(X[i,d], lb[0,d], ub[0,d])
                else:
                    for d in range(dim):    
                        x_mean = sum(X[:,d]) / N
                        X[i, d] = r_v * Xalpha[0, d] - x_mean
                        X[i, d] = boundary(X[i,d], lb[0,d], ub[0,d])
            else:
                Cv = abs(Falpha / (Fbeta + Fdelta))
                if np.random.random() < Cv:
                    for d in range(dim):
                        C1 = 2 * rand()
                        C2 = 2 * rand()
                        C3 = 2 * rand()
                        # Compute Dalpha, Dbeta & Ddelta (3.5)
                        Dalpha = abs(C1 * Xalpha[0,d] - X[i, d])
                        Dbeta = abs(C2 * Xbeta[0,d] - X[i, d])
                        Ddelta = abs(C3 * Xdelta[0,d] - X[i, d])
                        # Parameter A (3.3)
                        A1 = 2 * a * rand() - a
                        A2 = 2 * a * rand() - a
                        A3 = 2 * a * rand() - a
                        # Compute X1, X2 & X3 (3.6)
                        X1 = Xalpha[0, d] - A1 * Dalpha
                        X2 = Xbeta[0, d] - A2 * Dbeta
                        X3 = Xdelta[0, d] - A3 * Ddelta
                        # Update wolf (3.7)
                        X[i, d] = (X1 + X2 + X3) / 3
                        # Boundary
                        X[i, d] = boundary(X[i,d], lb[0,d], ub[0,d])
                else:
                    for d in range(dim):
                        A1 = 2 * a * rand() - a
                        C1 = 2 * rand()
                        Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                        X[i, d] = Xalpha[0,d] - A1 * Dalpha
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

        # Binary conversion
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

        curve[0, t] = Falpha.copy()
        print("Iteration:", t + 1)
        print("Best (MSGWO2):", curve[0, t])
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
