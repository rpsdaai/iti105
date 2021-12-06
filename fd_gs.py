import pickle
import joblib
# CatBoost
from catboost import CatBoostClassifier

# LightGBM
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import fd_eda
import fd_training

import sys
import logging

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

results_all = {}
grid_results_all = {}

if __name__ == '__main__':
    # Read the dataset
    df = fd_eda.read_datatset(fd_eda.dataset_dir, fd_eda.datafile)
    # df_new  = fd_eda.doPrepareData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # # TRAIN TEST SPLITTING
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)
    X_train, X_test, y_train, y_test = fd_training.do_SplitData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], 1, 0.2, 88, 'isFraud')

    cb_model = CatBoostClassifier(iterations=25, random_state=88)
    # cb_model = CatBoostClassifier(iterations=5, random_state=88)
    xval_value = 5
    '''
    Will get this error if dont prefix the parameters with classifier name used in pipeline!!

    ValueError: Invalid parameter depth for estimator Pipeline(steps=[('scaler', StandardScaler()), ('smote', SMOTE(random_state=2)),
                ('cb',
                 <catboost.core.CatBoostClassifier object at 0x000001B32B030DD8>)]). Check the list of available parameters with `estimator.get_params().keys()`.
    
    '''
    param_grid = {
                    'cb__learning_rate': [0.001, 0.01, 0.5], # default: 0.03
                    'cb__depth': [4, 6, 10], # tree depth
                    'cb__l2_leaf_reg': [10, 15, 25] # L2 regularization default is 3.0
            } 
    # param_grid = {
    #                 'cb__learning_rate': [0.001], # default: 0.03
    #                 'cb__depth': [4], # tree depth
    #                 'cb__l2_leaf_reg': [10] # L2 regularization default is 3.0
    #         }     
    cb_grid, cb_grid_model = fd_training.do_GridSearch('cb', cb_model, param_grid, xval_value, X_train, y_train)
    # grid_results_all[cb_grid_model.estimator.steps[2][0]] = fd_training.do_evaluateModel_GS(cb_grid_model.estimator.steps[2][0], cb_grid, X_test, y_test)
    grid_results_all[cb_grid_model.steps[2][0]] = fd_training.do_evaluateModel_GS(cb_grid_model.steps[2][0], cb_grid_model, X_test, y_test)

    # yPred = cb_grid.predict(X_test)
    # print (yPred)
    fd_training.do_plotEvaluationCurves('cb', cb_grid_model, X_test, y_test, 'Wistia')
    # Old return: model = grid.fit(...) that was why had to use best_estimator_
    # fd_eda.do_saveModel('cb_best_gs.pkl', cb_grid_model.best_estimator_, 'p') 
    # Updated to return best_estimator_ directly
    fd_eda.do_saveModel('cb_best_gs.pkl', cb_grid_model, 'p')

    '''
    Old: cb_grid_model = grid.fit(....)
    log.info(cb_grid_model)
    log.info(type(cb_grid_model))

    log.info("SCALER [0][0]")
    log.info (cb_grid_model.estimator.steps[0][0])
    log.info (type(cb_grid_model.estimator.steps[0][0]))
    log.info("SCALER [0][1]")
    log.info(cb_grid_model.estimator.steps[0][1])
    log.info (type(cb_grid_model.estimator.steps[0][1]))
    '''

    # log.info (type(cb_grid_model.estimator.steps[0][0]))
    # log.info (cb_grid_model.estimator.steps[0])
    # log.info (type(cb_grid_model.estimator.steps[0]))
    # Ref: https://stackoverflow.com/questions/57730192/how-to-save-gridsearchcv-xgboost-model
    # Old returned model = grid.fit()
    # fd_eda.do_saveModel('cb_best_scaler_gs.pkl', cb_grid_model.estimator.steps[0][1], 'p')
    # Updated: returned best_estimator_ directly
    fd_eda.do_saveModel('cb_best_scaler_gs.pkl', cb_grid_model['scaler'], 'p')

    scaler = fd_eda.do_loadModel('cb_best_scaler_gs.pkl')
    model = fd_eda.do_loadModel('cb_best_gs.pkl')
    log.info("SCALER TYPE")
    log.info(type(scaler))
    log.info("MODEL TYPE")
    log.info(type(model))
    # exit(0)

    # boosting_type='gbdt' (default), learning_rate=0.1 (default) 
    lg_model = lgb.LGBMClassifier(random_state=88)
    param_grid = {
                    'lg__num_iterations': [10, 100],
                    'lg__learning_rate': [0.001, 0.01, 0.05],
                    'lg__max_depth': [10, 50]
                }
    cv = 5
    lg_grid, lg_grid_model = fd_training.do_GridSearch('lg', lg_model, param_grid, cv, X_train, y_train)
    # grid_results_all[lg_grid_model.estimator.steps[2][0]] = fd_training.do_evaluateModel_GS(lg_grid_model.estimator.steps[2][0], lg_grid, X_test, y_test)
    grid_results_all[lg_grid_model.steps[2][0]] = fd_training.do_evaluateModel_GS(lg_grid_model.steps[2][0], lg_grid_model, X_test, y_test)
    fd_training.do_plotEvaluationCurves('lg', lg_grid_model, X_test, y_test, 'bwr')
    fd_eda.do_saveModel('lg_best_gs.pkl', lg_grid_model, 'p')
    # fd_eda.do_saveModel('lg_best_scaler_gs.pkl', lg_grid_model.estimator.steps[0][0], 'p')
    fd_eda.do_saveModel('lg_best_scaler_gs.pkl', lg_grid_model['scaler'], 'p')

    # Save results of both models in a file
    fd_eda.do_saveModel('results_gs_all.pkl', grid_results_all, 'p')

    # Test saved models to ensure they are the correct type
    scaler = fd_eda.do_loadModel('lg_best_scaler_gs.pkl')
    model = fd_eda.do_loadModel('lg_best_gs.pkl')
    log.info("SCALER TYPE")
    log.info(type(scaler))
    log.info("MODEL TYPE")
    log.info(type(model))