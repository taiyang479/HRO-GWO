import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import neighbors

# error rate
def error_rate(xtrain, ytrain, x, opts):
    k = opts['k']
    classifier_name = opts['classifier_name']
    kf = StratifiedKFold(n_splits=10)
    kf.get_n_splits()
    acc = []
    for train_index, valid_index in kf.split(xtrain, ytrain):
        x_train = xtrain[train_index]
        y_train = ytrain[train_index]
        x_test = xtrain[valid_index]
        y_test = ytrain[valid_index]

        num_train = np.size(x_train, 0)
        num_valid = np.size(x_test, 0)
        xtrain1 = x_train[:, x == 1]
        ytrain1 = y_train.reshape(num_train)
        xvalid = x_test[:, x == 1]
        yvalid = y_test.reshape(num_valid)

        if classifier_name == 'NB':
            NBModel = MultinomialNB(alpha=0.1)
            NBModel.fit(xtrain1, ytrain1)
            NB_pred = NBModel.predict(xvalid)
            acc.append(accuracy_score(yvalid, NB_pred))
        elif classifier_name == 'KNN':
            KNNModel = neighbors.KNeighborsClassifier(n_neighbors=k)
            KNNModel.fit(xtrain1, ytrain1)
            KNN_pred = KNNModel.predict(xvalid)
            Acc = np.sum(yvalid == KNN_pred) / np.size(yvalid, 0)
            acc.append(Acc)


    error = 1 - np.mean(acc)
    return np.mean(acc), error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate
        acc, error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)
        # cost = error

    return cost

