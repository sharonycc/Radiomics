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


data_m = pd.read_csv('./AllFeature.csv')
data_m = shuffle(data_m)
data_m.index = range(len(data_m))
X_m = data_m[data_m.columns[1:]]
y_m = data_m['label']

###Grid search
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm1 = svm.SVC(gamma=gamma, C=C)
            svm1.fit(X_m, y_m)
            score = svm1.score(X_m, y_m)
            if score > best_score:
                best_score = score
                best_parameters = {'gamma': gamma, 'C': C}

# ####   grid search end

print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))

C = best_parameters['C']
gamma = best_parameters['gamma']
#model
model_svm = svm.SVC(kernel = 'rbf',C = C, gamma = gamma,probability = True).fit(X_m,y_m)
score_svm = model_svm.score(X_m,y_m)
print(score_svm)
classifier = svm.SVC(kernel = 'rbf',C = C, gamma = gamma,probability = True)


#########ROC########
loo = LeaveOneOut()
loo.get_n_splits(X_m)
print("num：", loo.get_n_splits(X_m))

y_pred = []
for train_index, test_index in loo.split(X_m):
    X_mtrain, X_mtest = X_m.iloc[train_index], X_m.iloc[test_index]
    y_mtrain, y_mtest = y_m.iloc[train_index], y_m.iloc[test_index]
    model_svm = classifier.fit(X_mtrain, y_mtrain)
    y_probs = model_svm.predict_proba(X_mtest)
    x_test_pred = model_svm.predict(X_mtest)

    #     # AdaBoostClassifier
    #     model_bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
    #     model_bdt.fit(X_train, y_train)

    #     # pre
    #     x_test_pred = model_bdt.predict(X_test)

    y_pred.append(x_test_pred)

y_pred = np.array(y_pred)  # list to array

fpr, tpr, threshold = roc_curve(y_m, y_pred)
roc_auc = auc(fpr, tpr)  # auc value

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue',
         lw=2, label='Radiomics ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()
