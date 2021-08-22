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

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# CatBoost
from catboost import CatBoostClassifier

# LightGBM
import lightgbm as lgbm

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier

# Metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

results_dict = {}

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
    log.info('--> do_GridSearch(): ' + classifierName)

    # pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state = 2)), ("rus", RandomUnderSampler()),  (classifierName, classifier)])
    pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state = 2)), (classifierName, classifier)])

    # use gridsearch to test all values for n_neighbors
    grid = GridSearchCV(pipeline, params, cv=xval)

    # fit model to training data
    model = grid.fit(X_train, y_train)
    log.info("Grid FIT: \n")
    log.info(model)
    # y_pred = grid.predict(X_test)

    # save best model
    grid_best_model = grid.best_estimator_
    log.info('Best Model: \n')
    log.info(grid_best_model)

    # check best parameters
    log.info('Best parameters:\n')
    log.info(grid.best_params_)

    log.info('Best Score\n')
    log.info(str(grid.best_score_))

    return grid, grid_best_model

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
    plt.close()
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
    plt.close()
    return pr_curve

def do_plotAucRocCurve(filename, model, xtest, ytest):
    log.info('--> do_plotAucRocCurve()')
    roc_curve = plot_roc_curve(model, xtest, ytest)
    plt.savefig(filename)
    plt.close()
    return roc_curve

def do_getAucRocScore(ytest, ypred):
    log.info('--> do_getAucRocScore()')
    return roc_auc_score(ytest, ypred)

def do_getRecallScore(ytest, ypred):
    log.info('--> do_getRecallScore()')
    return recall_score(ytest, ypred)

def do_getPrecisionScore(ytest, ypred):
    log.info('--> do_getPrecisionScore()')
    return precision_score(ytest, ypred)    

def do_getF1Score(ytest, ypred):
    log.info('--> do_getF1Score()')
    return f1_score(ytest, ypred)     

def do_getBalancedAccuracyScore(ytest, ypred):
    log.info('--> do_getBalancedAccuracyScore()')
    return balanced_accuracy_score(ytest, ypred)      

def do_evaluateModel(pipeline, xtest, ytest):
    log.info('--> do_evaluateModel()')

    ypred = pipeline.predict(xtest)

    # Ref: https://www.geeksforgeeks.org/python-nested-dictionary/
    results = {}
    results[pipeline.steps[2][0]] = {}
    results[pipeline.steps[2][0]]['classificationReport'] = do_ClassificationReport(ytest, ypred)
    results[pipeline.steps[2][0]]['auc_roc_score'] = do_getAucRocScore(ytest, ypred)
    results[pipeline.steps[2][0]]['recall_scores'] = do_getRecallScore(ytest, ypred)
    results[pipeline.steps[2][0]]['precision_scores'] = do_getPrecisionScore(ytest, ypred)
    results[pipeline.steps[2][0]]['f1_scores'] = do_getF1Score(ytest, ypred)
    results[pipeline.steps[2][0]]['balanced_accuracy_scores'] = do_getBalancedAccuracyScore(ytest, ypred)

    log.info('Classification Report: ' + do_ClassificationReport(ytest, ypred))
    log.info('AUC ROC Score: ' + str(do_getAucRocScore(ytest, ypred)))
    log.info('Recall Score: ' + str(do_getRecallScore(ytest, ypred))) 
    log.info('Precision Score: ' + str(do_getPrecisionScore(ytest, ypred))) 
    log.info('F1 Score: ' + str(do_getF1Score(ytest, ypred)))
    log.info('Balanced Accuracy Score: ' + str(do_getBalancedAccuracyScore(ytest, ypred)))

    return results

def do_evaluateModel_GS(key, pipeline, xtest, ytest):
    log.info('--> do_evaluateModel()')

    ypred = pipeline.predict(xtest)

    # Ref: https://www.geeksforgeeks.org/python-nested-dictionary/
    results = {}
    results[key] = {}
    results[key]['classificationReport'] = do_ClassificationReport(ytest, ypred)
    results[key]['auc_roc_score'] = do_getAucRocScore(ytest, ypred)
    results[key]['recall_scores'] = do_getRecallScore(ytest, ypred)
    results[key]['precision_scores'] = do_getPrecisionScore(ytest, ypred)
    results[key]['f1_scores'] = do_getF1Score(ytest, ypred)
    results[key]['balanced_accuracy_scores'] = do_getBalancedAccuracyScore(ytest, ypred)

    log.info('Classification Report: ' + do_ClassificationReport(ytest, ypred))
    log.info('AUC ROC Score: ' + str(do_getAucRocScore(ytest, ypred)))
    log.info('Recall Score: ' + str(do_getRecallScore(ytest, ypred))) 
    log.info('Precision Score: ' + str(do_getPrecisionScore(ytest, ypred))) 
    log.info('F1 Score: ' + str(do_getF1Score(ytest, ypred)))
    log.info('Balanced Accuracy Score: ' + str(do_getBalancedAccuracyScore(ytest, ypred)))

    return results

def do_plotEvaluationCurves(classifierName, model, xtest, ytest, colorMap):
    log.info('--> do_plotEvaluationCurves()')
    do_plotConfusionMatrix(classifierName+'_cm', model, xtest, ytest, colorMap)
    do_plotPRCurve(classifierName+'_pr', model, xtest, ytest)
    do_plotAucRocCurve(classifierName+'_auc_roc', model, xtest, ytest)

# Ref: https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html
def do_plotFeatureImportance(filename, importance, names):
    # Define size of bar plot
    fig = plt.figure(figsize = (12, 6))

    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    log.info(feature_importance)
    feature_names = np.array(names)
    log.info(feature_names)

    # Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    log.info(data)
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Plot Searborn bar chart
    ax = sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette="bright")

    # Annotate text on horizontal Seaborn barplot
    # https://www.py4u.net/discuss/243741
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2), xytext=(5, 0), textcoords='offset points', ha="left", va="center")

    # Add chart labels
    plt.title('Visualizing Important Features')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    fig.savefig(filename)
    plt.close()

# Explore the dataset to get general feel
if __name__ == '__main__':
    # Read the dataset
    df = fd_eda.read_datatset(fd_eda.dataset_dir, fd_eda.datafile)
    # df_new  = fd_eda.doPrepareData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # TRAIN TEST SPLITTING
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)
    X_train, X_test, y_train, y_test = do_SplitData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], 1, 0.2, 88, 'isFraud')

    # Logistic Regression **lr and lr_pipe** same type 'imblearn.pipeline.Pipeline'
    lr, lr_pipe = do_PipelineOnly("lr", LogisticRegression(max_iter=10000, random_state=88), X_train, y_train)
    results_dict[lr_pipe.steps[2][0]] = do_evaluateModel(lr_pipe, X_test, y_test)
    do_plotEvaluationCurves('lr', lr, X_test, y_test, 'Reds')
    # Ref: https://cloud.google.com/ai-platform/prediction/docs/using-pipelines
    fd_eda.do_saveModel('lr.pkl', lr, 'p')
    fd_eda.do_saveModel('lr_scaler.pkl', lr_pipe[0], 'p')

    # XGBoost
    xgb, xgb_pipe = do_PipelineOnly("xgb", xgb.XGBClassifier(tree_method='exact', max_depth=3, n_estimators=50, random_state=88), X_train, y_train)
    results_dict[xgb_pipe.steps[2][0]] = do_evaluateModel(xgb_pipe, X_test, y_test)
    do_plotEvaluationCurves('xgb', xgb, X_test, y_test, 'PiYG')
    fd_eda.do_saveModel('xgb.pkl', xgb, 'p')
    fd_eda.do_saveModel('xgb_scaler.pkl', xgb_pipe[0], 'p')

    # Gaussian Naive-Bayes
    gnb, gnb_pipe = do_PipelineOnly("gnb", GaussianNB(), X_train, y_train)
    results_dict[gnb_pipe.steps[2][0]] = do_evaluateModel(gnb_pipe, X_test, y_test)
    do_plotEvaluationCurves('gnb', gnb, X_test, y_test, 'Greens')
    fd_eda.do_saveModel('gnb.pkl', gnb, 'p')
    fd_eda.do_saveModel('gnb_scaler.pkl', gnb_pipe[0], 'p')

    # Decision Tree Classifier
    dt, dt_pipe = do_PipelineOnly("dt", DecisionTreeClassifier(random_state=88, max_depth=3), X_train, y_train)
    results_dict[dt_pipe.steps[2][0]] = do_evaluateModel(dt_pipe, X_test, y_test)
    do_plotEvaluationCurves('dt', dt, X_test, y_test, 'Blues')
    fd_eda.do_saveModel('dt.pkl', dt, 'p') 
    fd_eda.do_saveModel('dt_scaler.pkl', dt_pipe[0], 'p')  

    # Random Forest Classifier
    rfc, rfc_pipe = do_PipelineOnly("rfc", RandomForestClassifier(n_estimators = 50, random_state = 88, n_jobs = -1, oob_score = True), X_train, y_train)
    results_dict[rfc_pipe.steps[2][0]] = do_evaluateModel(rfc_pipe, X_test, y_test)
    do_plotEvaluationCurves('rfc', rfc, X_test, y_test, 'magma')
    do_plotFeatureImportance('feature_importance.png', rfc_pipe.steps[2][1].feature_importances_, X_test.columns.values.tolist())
    fd_eda.do_saveModel('rfc.pkl', rfc, 'p') 
    fd_eda.do_saveModel('rfc_scaler.pkl', rfc_pipe[0], 'p') 

    # Catboost
    cb, cb_pipe = do_PipelineOnly("cb", CatBoostClassifier(iterations=5, learning_rate=0.1, random_state=88), X_train, y_train)
    results_dict[cb_pipe.steps[2][0]] = do_evaluateModel(cb_pipe, X_test, y_test)
    do_plotEvaluationCurves('cb', cb, X_test, y_test, 'Purples')
    fd_eda.do_saveModel('cb.pkl', cb, 'p') 
    fd_eda.do_saveModel('cb_scaler.pkl', cb_pipe[0], 'p')

    # LightGBM
    lgbm, lgbm_pipe = do_PipelineOnly("lgbm", lgbm.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, random_state=88), X_train, y_train)
    results_dict[lgbm_pipe.steps[2][0]] = do_evaluateModel(lgbm_pipe, X_test, y_test)
    do_plotEvaluationCurves('lgbm', lgbm, X_test, y_test, 'Oranges')
    fd_eda.do_saveModel('lgbm.pkl', lgbm, 'p')
    fd_eda.do_saveModel('lgbm_scaler.pkl', lgbm_pipe[0], 'p')

    # Bagging Classifier
    bg, bg_pipe = do_PipelineOnly("bg", BaggingClassifier(DecisionTreeClassifier(random_state=88), random_state=88), X_train, y_train)
    results_dict[bg_pipe.steps[2][0]] = do_evaluateModel(bg_pipe, X_test, y_test)
    do_plotEvaluationCurves('bg', bg, X_test, y_test, 'PuBu')
    fd_eda.do_saveModel('bg.pkl', bg, 'p')
    fd_eda.do_saveModel('bg_scaler.pkl', bg_pipe[0], 'p') 

    # Consolidated Results
    log.info('--> RESULTS\n')
    log.info(results_dict)
    log.info('<-- RESULTS\n')
    fd_eda.do_saveModel('results.pkl', results_dict, 'p')