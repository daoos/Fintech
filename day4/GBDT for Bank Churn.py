# coding=utf8
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from numpy import *
from sklearn import ensemble, cross_validation, metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

print '======================================='
path = '/home/daoos/Documents/xiaoxiang_fintech/day4/';
### Step 1: Prepare the prior work ###
print '步骤1：准备先前的工作'
###read the data with pre-processing, like making up the missing value,
print '通过预处理读取数据，如弥补缺失值，'
modelData = pd.read_csv(path + '/data/modelData.csv', header=0)
allFeatures = list(modelData.columns)
# remove the class label and cust id from features
print '从特征中删除类标签和帐号'
allFeatures.remove('CUST_ID')
allFeatures.remove('CHURN_CUST_IND')

# split the modeling dataset into trainning set and testing set
print '将建模数据集分为训练集和测试集'
X_train, X_test, y_train, y_test = train_test_split(modelData[allFeatures], modelData['CHURN_CUST_IND'], test_size=0.5,
                                                    random_state=9)

y_train.value_counts()

# try 1: using default parameter
print '尝试1：使用默认参数'
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X_train, y_train)
y_pred = gbm0.predict(X_test)
y_predprob = gbm0.predict_proba(X_test)[:, 1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred)
print "AUC Score (Testing): %f" % metrics.roc_auc_score(y_test, y_predprob)

y_pred2 = gbm0.predict(X_train)
y_predprob2 = gbm0.predict_proba(X_train)[:, 1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred2)
print "AUC Score (Testing): %f" % metrics.roc_auc_score(y_train, y_predprob2)

# tunning the number of estimators
print '调整估计数'
param_test1 = {'n_estimators': range(20, 81, 10)}
gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                             min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                             subsample=0.8, random_state=10),
                        param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# tunning the parameters of simple trees
print '调整简单树的参数'
param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, min_samples_leaf=20,
                                                             max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
gsearch2.fit(X_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test3 = {'min_samples_split': range(400, 1001, 100), 'min_samples_leaf': range(60, 101, 10)}
gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=9,
                                                             max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
gsearch3.fit(X_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

## use the tunned parameters to train the model again
print '使用调谐参数再次训练模型'
gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=9, min_samples_leaf=70,
                                  min_samples_split=500, max_features='sqrt', subsample=0.8, random_state=10)
gbm1.fit(X_train, y_train)
y_pred1 = gbm1.predict(X_train)
y_predprob1 = gbm1.predict_proba(X_train)[:, 1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred1)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob1)

y_pred2 = gbm1.predict(X_test)
y_predprob2 = gbm1.predict_proba(X_test)[:, 1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred2)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob2)

### tunning max_features
print '调整最大特征集'
param_test4 = {'max_features': range(5, 31, 2)}
gsearch4 = GridSearchCV(
    estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=9, min_samples_leaf=70,
                                         min_samples_split=500, subsample=0.8, random_state=10),
    param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
gsearch4.fit(X_train, y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

## tunning subsample
print '调整子样本'
param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
gsearch5 = GridSearchCV(
    estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=9, min_samples_leaf=70,
                                         min_samples_split=500, max_features=28, random_state=10),
    param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)
gsearch5.fit(X_train, y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# tunning the learning rate and
print '调整学习率'
gbm2 = GridSearchCV(
    estimator=GradientBoostingClassifier(learning_rate=0.05, n_estimators=70, max_depth=9, min_samples_leaf=70,
                                         min_samples_split=1000, max_features=28, random_state=10, subsample=0.8),
    param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)
gbm2.fit(X_train, y_train)

y_pred1 = gbm2.predict(X_train)
y_predprob1 = gbm2.predict_proba(X_train)[:, 1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred1)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob1)

y_pred2 = gbm2.predict(X_test)
y_predprob2 = gbm2.predict_proba(X_test)[:, 1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred2)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob2)

clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=70, max_depth=9, min_samples_leaf=70,
                                 min_samples_split=1000, max_features=28, random_state=10, subsample=0.8)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
# sort the features by importance in descending order. by default argsort returing asceding order
print '按照重要性降序对特征进行排序。默认情况下argsort保留退出顺序'
features_sorted = argsort(-importances)
import_feautres = [allFeatures[i] for i in features_sorted]
print '=================================================='
