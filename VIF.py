## Calculate the VIF of each variable in each loop and remove the variables with VIF>threshold
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def VIF(X, thres):
    col = list(range(X.shape[1]))
    dropped = len(col)

    for var in range(0, dropped):
        x_new = X.iloc[:, col]
        vif = [variance_inflation_factor(X.iloc[:, col].values, ix) for ix in range(X.iloc[:, col].shape[1])]
        col_all = col
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('deleted:', x_new.columns[maxix], '；', 'VIF:', maxvif)
        Xnew = X.iloc[:, col]

    print('Remain Variables:', len(X.columns[col]))
    return list(X.columns[col])

data=pd.read_csv('./c/lasso-x.csv')
x=data.iloc[:,1:]
VIF(x,thres=10.0)