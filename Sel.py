import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RepeatedKFold,GridSearchCV
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from scipy.stats import pearsonr,ttest_ind,levene,spearmanr,kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from statsmodels.stats.outliers_influence import variance_inflation_factor


import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

################ Importing data and pre-processing ##########
#Importing data
data_x = pd.read_csv('./GTVnx.csv')#GTVnx
data_d = pd.read_csv('./GTVnd.csv')#GTVnd
X_x = data_x[data_x.columns[1:]]
y_x = data_x['label']
X_d = data_d[data_d.columns[1:]]
y_d = data_d['label']

################ T test ##########
data_x_a = data_x[:][data_x['label'] == 0]
data_x_b = data_x[:][data_x['label'] == 1]
index = []
for colName in data_x.columns[:]:
    if levene(data_x_a[colName], data_x_b[colName])[1] > 0.05:
        if ttest_ind(data_x_a[colName], data_x_b[colName])[1] < 0.05:
            index.append(colName)
    else:
        if ttest_ind(data_x_a[colName], data_x_b[colName],equal_var=False)[1] < 0.05:
            index.append(colName)
print(len(index))

data_x_a = data_x_a[index]
data_x_b = data_x_b[index]
data_x = pd.concat([data_x_a, data_x_b])

#########Data set processing#######
data_x = shuffle(data_x)
data_x.index = range(len(data_x))
X_x = data_x[data_x.columns[1:]]
scaler = StandardScaler()
scaler.fit(X_x)
X_x = scaler.transform(X_x)
X_x = pd.DataFrame(X_x)
X_x.columns = index[1:]
y_x = data_x['label']

#########Lasso festure screening#######
alphas = np.logspace(-4,1,100)
model_lassoCV = LassoCV(alphas = alphas, max_iter = 100000).fit(X_x,y_x)
coef = pd.Series(model_lassoCV.coef_, index = X_x.columns)
print(model_lassoCV.alpha_)
print('%s %d'%('Lasso picked',sum(coef != 0)))
index = coef[coef != 0].index
X_x = X_x[index]


X_x.to_csv('./Lasso-x.csv')
y_x.to_csv('./Lasso-y.csv')





