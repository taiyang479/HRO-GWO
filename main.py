import FS.GWO as h_gwo
import FS.GWO as h_sca
import FS.IGWO as h_igwo
import FS.PSO as h_pso
import FS.WOA as h_woa
import FS.HROGWO as h_hrogwo
import FS.HRO as h_hro
import FS.MSGWO1 as h_msgwo
import FS.MSGWO2 as h_msgwo2
import FS.MHRO as h_mhro
import FS.EGA as h_ega

import datetime
import csv
from sklearn import preprocessing
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from skrebate import ReliefF

# load data
def readfile(filepath):
    data = pd.read_csv(filepath, encoding='utf-8')
    data = data.values
    X = np.array(data[:, 0:-1])
    y = np.array(data[:, -1])
    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(X)
    encoder = preprocessing.LabelEncoder()
    y_data = encoder.fit_transform(y)
    labels = x_data
    contents = y_data
    return labels, contents

def to_filters(data,s,filter):
    x_data, y_data = readfile(data)
    dim = np.size(x_data, 1)
    a = int(100/s)
    n = int(dim/a)
    if filter == 'CHI':
        X_new = SelectPercentile(score_func=chi2, percentile=s).fit_transform(x_data, y_data)

    elif filter == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=42)
        model.fit(x_data, y_data)
        importances = model.feature_importances_
        sorted_feature_indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)

        selected_feature_indices = sorted_feature_indices[:n]
        X_new = x_data[:, selected_feature_indices]
    elif filter == 'ReliefF':
        fs = ReliefF()
        fs.fit(x_data, y_data)
        importances = fs.feature_importances_
        sorted_feature_indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)

        selected_feature_indices = sorted_feature_indices[:n]
        X_new = x_data[:, selected_feature_indices]
    else: #filter == 'MI':
        X_new = SelectKBest(mutual_info_classif, k = n).fit_transform(x_data, y_data)

    return X_new, y_data


def selector(name, feat, label, opts):
    if name == 'gwo':
        hh = h_gwo.gwo(feat, label, opts)
    elif name == 'igwo':
        hh = h_igwo.igwo(feat, label, opts)
    elif name == 'pso':
        hh = h_pso.pso(feat, label, opts)
    elif name == 'sca':
        hh = h_sca.sca(feat, label, opts)
    elif name == 'woa':
        hh = h_woa.woa(feat, label, opts)
    elif name == 'hro':
        hh =h_hro.hro(feat, label, opts)
    elif name == 'hrogwo':
        hh = h_hrogwo.hrogwo(feat, label, opts)
    elif name == 'msgwo':
        hh = h_msgwo.msgwo(feat, label, opts)
    elif name == 'msgwo2':
        hh = h_msgwo2.msgwo2(feat, label, opts)
    elif name == 'mhro':
        hh = h_mhro.mhro(feat, label, opts)
    elif name == 'ega':
        hh = h_ega.ega(feat, label, opts)
    else:
        pass    
    return hh


if __name__ == "__main__":
    k = 5     # k-value in KNN
    N = 50    # number of particles
    T = 1000  # maximum number of iterationsna
    s = 10
    num2 = str(s)
    datasets = ['GLIOMA']
    algorithm = ['hrogwo']
    filtering = ['CHI']#'DecisionTree','ReliefF','MI'
    classifierName = ['NB','KNN']
    name = str(classifierName)

    Export = True
    Flag = False
    for dataset in datasets:
        path = './data/' + dataset + '.csv'
        print("path:", path)
        for filter in filtering:
            feat, label = to_filters(path,s,filter)
            for cls in classifierName:
                data = np.linspace(1, T, T)
                data = np.transpose(data)
                data = data.reshape((T, 1))
                for alg in algorithm:
                    print("algorithm：", alg)
                    print("classifier：", cls)
                    ExportToFile = "./FS-result/new-result/" + dataset + "-" + cls + "-" + num2 + "-"+".csv"
                    ExportToFile2 = "./FS-result/new-curve/" + dataset + "-" + cls + "-" + num2 + "-"+ ".csv"
                    opts = {'k': k, 'N': N, 'T': T, 'classifier_name': cls}
                    for i in range(5):
                        print("----------------------{}-----------------------".format(i))
                        starttime = datetime.datetime.now()
                        fmdl = selector(alg, feat, label, opts)
                        sf = fmdl['sf']
                        num_feat = fmdl['nf']
                        fit = fmdl['fitness']
                        curve = fmdl['c']
                        curves = np.transpose(curve)
                        endtime = datetime.datetime.now()
                        time = (endtime - starttime).seconds
                        data = np.hstack((data, curves))
                        if (Export == True):
                            with open(ExportToFile, 'a', newline='\n') as out:
                                writer = csv.writer(out, delimiter=',')
                                if (Flag == False):  # just one time to write the header of the CSV file
                                    header = ["Dataset", "Optimizer", "classifierName", "fitness", "Num","k-value","particles","iterations","Proportion","time","filter"]
                                    writer.writerow(header)
                                a = [dataset, alg, cls, fit, num_feat, k, N, T, s, time, filter]
                                writer.writerow(a)
                            out.close()

                        with open(ExportToFile2, 'a', newline='\n') as out:
                            writer = csv.writer(out, delimiter=',')
                            header = ["Serial Number",'hrogwo']
                            writer.writerow(header)
                            writer.writerows(data)

                        Flag = True
                Flag = False

