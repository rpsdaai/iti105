import imblearn

# Ref: https://stackoverflow.com/questions/53114668/python-imblearn-make-pipeline-typeerror-last-step-of-pipeline-should-implement
# If you are using the imleaarn framework have to use the imblearn pipeline else will give error
# TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 
# 'passthrough' 'SMOTE(random_state=2)' (type <class 'imblearn.over_sampling._smote.base.SMOTE'>) doesn't
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline

# from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

# XGBoost
import xgboost as xgb

# Gaussian Naive-Bayes
from sklearn.naive_bayes import GaussianNB

# Bagging
from sklearn.ensemble import BaggingClassifier

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# CatBoost
from catboost import CatBoostClassifier

# LightGBM
import lightgbm as lgbm

# Metrics
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report

# from sklearn.model_selection import (
#     train_test_split,
#     RepeatedStratifiedKFold,
#     cross_validate
# )

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import fd_eda

import sys
import logging

# Turn off - DEBUG findfont: score(<Font 'DejaVu Sans' (DejaVuSans-Oblique.ttf) oblique normal 400 normal>) = 1.05
# Ref: https://github.com/matplotlib/matplotlib/issues/14523
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Log to both console + file
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     handlers=[
#                         logging.FileHandler('fraud_detection.log', 'w', 'utf-8'),
#                         logging.StreamHandler(sys.stdout)
#                     ])

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

def do_SplitData(df, cols2drop, nSplits, testSize, randomState, targetLabel):
    log.info('do_SplitData(): cols2drop: ' + ' '.join(cols2drop) + ' nSplits: ' + str(nSplits) + ' testSize: ' + str(testSize) + ' randomState: ' + str(randomState) + ' targetLabel: ' + targetLabel)

    # initialization a generator
    ss_split = StratifiedShuffleSplit(n_splits=nSplits, test_size=testSize, random_state=randomState)

    df_tmp  = fd_eda.doPrepareData(df, cols2drop)

    # Split data folders based on isFraud value, to ensure the good distribution of train and test data responing to isFraud values
    # Get the index values from the generator
    # Separate feature columns and the target column
    feature_cols = [x for x in df_tmp.columns if x != targetLabel]
    log.info('Feature Columns: ' + ' '.join(feature_cols))
    
    # Split data folders based on isFraud value, to ensure the good distribution of train and test data responing to isFraud values
    # Get the index values from the generator
    train_idx, test_idx = next(ss_split.split(df_tmp[feature_cols], df_tmp[targetLabel]))

    X_train = df_tmp.loc[train_idx, feature_cols]
    y_train = df_tmp.loc[train_idx, targetLabel]

    X_test = df_tmp.loc[test_idx, feature_cols]
    y_test = df_tmp.loc[test_idx, targetLabel]

    return X_train, X_test, y_train, y_test


def do_GridSearch(classifierName, classifier, params, xval, X_train, y_train):
    pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state = 2)), ("rus", RandomUnderSampler()),  (classifierName, classifier)])

    # use gridsearch to test all values for n_neighbors
    grid = GridSearchCV(pipeline, params, cv=xval)

    # fit model to training data
    model = grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    # save best model
    lr_best = grid.best_estimator_

    # check best n_neigbors value
    print(grid.best_params_)

    return grid, model

def do_PipelineOnly(classifierName, classifier, X_train, y_train):
    log.info('--> doPipelineOnly(): ' + classifierName)
    pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state = 2)), (classifierName, classifier)])
    model = pipeline.fit(X_train, y_train)
    # y_prediction_base = pipeline.predict(X_test)
    return model, pipeline

# scoring = ['accuracy', 'precision', 'roc_auc']
def do_PipelineWithCV(classifierName, classifier, X_train, y_train, nSplits, score_list):
    log.info('--> do_PipelineWithCV(): ' + classifierName)
    pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state = 2)), (classifierName, classifier)])

    # the oversampling is only applied to the train folds
    cv = RepeatedStratifiedKFold(n_splits=nSplits, n_repeats=1, random_state=0)
    scores = cross_validate(pipeline, X_train, y_train, scoring=score_list, cv=cv, n_jobs=-1)

    return pipeline, scores

# Ref: https://forums.fast.ai/t/how-to-save-confusion-matrix-to-file/38520/4
def do_plotConfusionMatrix(filename, model, xtest, ytest, colorMap):
    log.info('--> do_plotConfusionMatrix(): ' + filename + ' colorMap: ' + colorMap)
    #plt.figure(figsize=(10,10))
    cfm = plot_confusion_matrix(model, xtest, ytest, cmap=colorMap, normalize='true')
    #cfm.plot()
    plt.savefig(filename)
    # plt.grid(False)
    #plt.show()

# Ref: https://stackoverflow.com/questions/65317685/how-to-create-image-of-confusion-matrix-in-python
def do_ClassificationReport(ytest, ypred):
    log.info('--> do_ClassificationReport()')
    report = classification_report(ytest, ypred)
    return report

def do_plotPRCurve(filename, model, xtest, ytest):
    log.info('--> do_plotPRCurve()')
    pr_curve = plot_precision_recall_curve(model, xtest, ytest)
    plt.savefig(filename)
    return pr_curve

def do_plotAucRocCurve(filename, model, xtest, ytest):
    log.info('--> do_plotAucRocCurve()')
    roc_curve = plot_roc_curve(model, xtest, ytest)
    plt.savefig(filename)
    return roc_curve

def do_getAucRocScores(ytest, ypred):
    log.info('--> do_getAucRocScores()')
    return roc_auc_score(ytest, ypred)

def do_getRecallScore(ytest, ypred):
    log.info('--> do_getRecallScore()')
    return recall_score(ytest, ypred)

def do_evaluateModel(pipeline, xtest, ytest):
    log.info('--> do_evaluateModel()')
    ypred = pipeline.predict(xtest)
    log.info('Classification Report: ' + do_ClassificationReport(ytest, ypred))
    log.info('AUC ROC Score: ' + str(do_getAucRocScores(ytest, ypred)))
    log.info('Recall Score: ' + str(do_getRecallScore(ytest, ypred))) 

def do_plotEvaluationCurves(classifierName, model, xtest, ytest, colorMap):
    log.info('--> do_plotEvaluationCurves()')
    do_plotConfusionMatrix(classifierName+'_cm', model, xtest, ytest, colorMap)
    do_plotPRCurve(classifierName+'_pr', model, xtest, ytest)
    do_plotAucRocCurve(classifierName+'_auc_roc', model, xtest, ytest)

# def model_selection(classifier, name, grid, X_train, y_train, scoring):
    
#     gridsearch_cv=GridSearchCV(classifier, 
#                                grid,
#                                cv=5, 
#                                scoring = scoring)
    
#     gridsearch_cv.fit(X_adasyn, y_adasyn)
    
#     results_dict = {}
    
#     results_dict['classifier_name'] = name    
#     results_dict['classifier'] = gridsearch_cv.best_estimator_
#     results_dict['best_params'] = gridsearch_cv.best_params_
#     results_dict['ROC_AUC'] = gridsearch_cv.best_score_
    
#     return(results_dict)

# Explore the dataset to get general feel
if __name__ == '__main__':
    # Read the dataset
    df = fd_eda.read_datatset(fd_eda.dataset_dir, fd_eda.datafile)
    # df_new  = fd_eda.doPrepareData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # # TRAIN TEST SPLITTING
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)
    X_train, X_test, y_train, y_test = do_SplitData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], 1, 0.2, 88, 'isFraud')
    # log.info('X_train: \n')
    # log.info(X_train.head())

    # log.info('X_test: \n')
    # log.info(X_test.head())

    # log.info('y_train: ')
    # log.info(y_train.value_counts())

    # log.info('y_test: ')
    # log.info(y_test.value_counts())    

    # pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state = 2)), ("lr", LogisticRegressionCV(cv=5, max_iter=10000))])
    # lrcv = pipeline.fit(X_train, y_train)
    # y_prediction_base = pipeline.predict(X_test)

    # Logistic Regression
    lr, lr_pipe = do_PipelineOnly("lr", LogisticRegression(max_iter=10000), X_train, y_train)
    log.info (lr_pipe._estimator_type)
    log.info(lr_pipe.get_params())
    log.info(lr_pipe.steps[2][0])
    log.info(lr_pipe.steps[2][1])
    log.info(type(lr_pipe.steps[2][1]))
    exit(0)
    # do_plotConfusionMatrix("lr_cm.png", lr, X_test, y_test, 'Reds')
    # ypred = lr_pipe.predict(X_test)
    # log.info('Classification Report: ' + do_ClassificationReport(y_test, ypred))
    # do_plotPRCurve('lr_pr', lr, X_test, y_test)
    # do_plotAucRocCurve('lr_auc_roc', lr, X_test, y_test)
    # log.info('AUC ROC Score: ' + str(do_getAucRocScores(y_test, ypred)))
    # log.info('Recall Score: ' + str(do_getRecallScore(y_test, ypred)))    
    do_evaluateModel(lr_pipe, X_test, y_test)
    do_plotEvaluationCurves('lr', lr, X_test, y_test, 'Reds')
    fd_eda.do_saveModel('lr.pkl', lr, 'p')

    # lr_cv, lr_pipescores = do_PipelineWithCV("lr", LogisticRegression(max_iter=10000), X_train, y_train, 5, ['accuracy', 'precision', 'roc_auc'])
    # log.info(lr_pipescores)

    # XGBoost
    xgb, xgb_pipe = do_PipelineOnly("xgb", xgb.XGBClassifier(tree_method='exact', max_depth=3, n_estimators=50), X_train, y_train)
    do_evaluateModel(xgb_pipe, X_test, y_test)
    do_plotEvaluationCurves('xgb', xgb, X_test, y_test, 'Reds')
    fd_eda.do_saveModel('xgb.pkl', xgb, 'p')

    # # Gaussian Naive-Bayes
    gnb, gnb_pipe = do_PipelineOnly("gnb", GaussianNB(), X_train, y_train)
    do_evaluateModel(gnb_pipe, X_test, y_test)
    do_plotEvaluationCurves('gnb', gnb, X_test, y_test, 'Greens')
    fd_eda.do_saveModel('gnb.pkl', gnb, 'p')

    # Decision Tree Classifier
    dt, dt_pipe = do_PipelineOnly("dt", DecisionTreeClassifier(random_state=88, max_depth=3), X_train, y_train)
    do_evaluateModel(dt_pipe, X_test, y_test)
    do_plotEvaluationCurves('dt', dt, X_test, y_test, 'Blues')
    fd_eda.do_saveModel('dt.pkl', dt, 'p')    

    # Bagging Classifier
    bg, bg_pipe = do_PipelineOnly("bg", BaggingClassifier(DecisionTreeClassifier()), X_train, y_train)
    do_evaluateModel(bg_pipe, X_test, y_test)
    do_plotEvaluationCurves('bg', bg, X_test, y_test, 'PuBu')
    fd_eda.do_saveModel('bg.pkl', bg, 'p')

    # Catboost
    cb, cb_pipe = do_PipelineOnly("cb", CatBoostClassifier(iterations=5, learning_rate=0.1), X_train, y_train)
    do_evaluateModel(cb_pipe, X_test, y_test)
    do_plotEvaluationCurves('cb', cb, X_test, y_test, 'Purples')
    fd_eda.do_saveModel('cb.pkl', cb, 'p')  

    # LightGBM
    lgbm, lgbm_pipe = do_PipelineOnly("lgbm", lgbm.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1), X_train, y_train)
    do_evaluateModel(lgbm_pipe, X_test, y_test)
    do_plotEvaluationCurves('lgbm', lgbm, X_test, y_test, 'Oranges')
    fd_eda.do_saveModel('lgbm.pkl', lgbm, 'p')