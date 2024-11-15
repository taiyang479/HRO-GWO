import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
import math

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


def hrogwo(xtrain, ytrain, opts):
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

    X_i_GWO = np.zeros([1, dim], dtype='float')
    X_i_DLH = np.zeros([1, dim], dtype='float')
    X_cross = np.zeros([1, dim], dtype='float')

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

    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    print("Iteration:", t + 1)
    print("Best (HRO-GWO):", curve[0, t])
    t += 1

    while t < max_iter:
        w = 0.01
        a = 2 * ((1 / 4) ** (t / max_iter)) * np.exp(-(w * (max_iter - t) / max_iter))
        for i in range(N):
            Neibor = []
            p = rand()
            for d in range(dim):
                # Parameter C
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                # Parameter A
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a

                X1 = Xalpha[0, d] - A1 * Dalpha
                X2 = Xbeta[0, d] - A2 * Dbeta
                X3 = Xdelta[0, d] - A3 * Ddelta
                # Update wolf
                X_i_GWO[0, d] = (X1 + X2 + X3) / 3
                X_i_GWO[0, d] = boundary(X_i_GWO[0, d], lb[0, d], ub[0, d])

                if p <= 0.5:
                    A = 2 * a * rand() - a
                    if abs(A) > 0.8:
                        r1 = rand()
                        r2 = rand()
                        r3 = rand()
                        X_cross[0, d] = (r1 * Xalpha[0, d] + r2 * Xbeta[0, d] + r3 * Xdelta[0, d]) / (r1 + r2 + r3)
                    else:
                        X_cross[0, d] = X1
                    if rand() < 0.3:

                        X_cross[0, d] = X[i, d]
                    X_cross[0, d] = boundary(X_cross[0, d], lb[0, d], ub[0, d])

                else:
                    random_number = np.random.randint(low=0, high=N)
                    neighbor = X[random_number]
                    r3 = np.random.random()
                    X_cross[0, d] = r3 * (Xalpha[0, d] - neighbor[d]) + X[i, d]
                    X_cross[0, d] = boundary(X_cross[0, d], lb[0, d], ub[0, d])

            X_cross_bin = single_binary_conversion(X_cross[0, :], thres, dim)
            X_i_GWO_bin = single_binary_conversion(X_i_GWO[0, :], thres, dim)
            F_i_cro = Fun(xtrain, ytrain, X_cross_bin[0, :], opts)
            F_i_GWO = Fun(xtrain, ytrain, X_i_GWO_bin[0, :], opts)

            f = []
            R = np.sqrt(np.sum((X[i, :] - X_i_GWO[0, :]) ** 2))
            for j in range(N):
                if i != j:
                    R_n = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))
                    if R_n <= R:
                        Neibor.append(X[j, :].tolist())
                        f.append(fit[j, 0].tolist())
            Neibor = np.array(Neibor, dtype=float)
            f = np.array(f, dtype=float)


            if len(Neibor) != 0:
                for d in range(dim):
                    k1 = np.random.randint(low=0, high=len(Neibor))
                    k2 = np.random.randint(low=0, high=N)

                    if (f[k1] < fit[k2, 0]):

                        X_i_DLH[0, d] = X[i, d] + rand() * (Neibor[k1, d] - X[k2, d])
                    else:
                        X_i_DLH[0, d] = X[i, d] - rand() * (Neibor[k1, d] - X[k2, d])

                    X_i_DLH[0, d] = boundary(X_i_DLH[0, d], lb[0, d], ub[0, d])
                X_i_DLH_bin = single_binary_conversion(X_i_DLH[0, :], thres, dim)
                F_i_DLH = Fun(xtrain, ytrain, X_i_DLH_bin[0, :], opts)
                if (F_i_GWO < F_i_DLH) and (F_i_GWO < F_i_cro):
                    X[i, :] = X_i_GWO[0, :]
                    fit[i, 0] = F_i_GWO

                elif (F_i_cro < F_i_DLH) and (F_i_cro < F_i_GWO):

                    X[i, :] = X_cross[0, :]
                    fit[i, 0] = F_i_cro

                else:
                    X[i, :] = X_i_DLH[0, :]
                    fit[i, 0] = F_i_DLH

            else:
                if F_i_GWO < F_i_cro:
                    X[i, :] = X_i_GWO[0, :]
                    fit[i, 0] = F_i_GWO

                else:
                    X[i, :] = X_cross[0, :]
                    fit[i, 0] = F_i_cro


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
        print("Best (HRO-GWO):", curve[0, t])
        t += 1

    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    # Create dictionary
    hrogwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'fitness': curve[0, max_iter - 1]}

    return hrogwo_data
