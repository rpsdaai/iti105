import fd_eda

import ast
import sys
import logging
import pandas as pd

import fd_eda

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

datadir = 'D:/Users/ng_a/My NYP SDAAI/tmp/iti105/'
cb_gs_sclr = fd_eda.do_loadModel(datadir + 'cb_best_scaler_gs.pkl')
cb_gs_model = fd_eda.do_loadModel(datadir + 'cb_best_gs.pkl')

lg_gs_sclr = fd_eda.do_loadModel(datadir + 'lg_best_scaler_gs.pkl')
lg_gs_model = fd_eda.do_loadModel(datadir + 'lg_best_gs.pkl')

datadir_Models = 'D:/Users/ng_a/My NYP SDAAI/tmp/iti105/models/'
lr_sclr = fd_eda.do_loadModel(datadir_Models + 'lr_scaler.pkl')
lr_model = fd_eda.do_loadModel(datadir_Models + 'lr.pkl')

gnb_sclr = fd_eda.do_loadModel(datadir_Models + 'gnb_scaler.pkl')
gnb_model = fd_eda.do_loadModel(datadir_Models + 'gnb.pkl')

xgb_sclr = fd_eda.do_loadModel(datadir_Models + 'xgb_scaler.pkl')
xgb_model = fd_eda.do_loadModel(datadir_Models + 'xgb.pkl')

dt_sclr = fd_eda.do_loadModel(datadir_Models + 'dt_scaler.pkl')
dt_model = fd_eda.do_loadModel(datadir_Models + 'dt.pkl')

rfc_sclr = fd_eda.do_loadModel(datadir_Models + 'rfc_scaler.pkl')
rfc_model = fd_eda.do_loadModel(datadir_Models + 'rfc.pkl')

cb_sclr = fd_eda.do_loadModel(datadir_Models + 'cb_scaler.pkl')
cb_model = fd_eda.do_loadModel(datadir_Models + 'cb.pkl')

lgbm_sclr = fd_eda.do_loadModel(datadir_Models + 'lgbm_scaler.pkl')
lgbm_model = fd_eda.do_loadModel(datadir_Models + 'lgbm.pkl')

bg_sclr = fd_eda.do_loadModel(datadir_Models + 'bg_scaler.pkl')
bg_model = fd_eda.do_loadModel(datadir_Models + 'bg.pkl')

log.info(type(cb_gs_sclr))
log.info(type(cb_gs_model))
log.info(cb_gs_sclr)
log.info(cb_gs_model)
# exit(0)

# log.info(cb_model.named_steps['scaler'])
# Ref: https://amueller.github.io/aml/01-ml-workflow/12-pipelines-gridsearch.html
# GridSearch
# log.info(cb_model['scaler'])
# scaler = cb_model['scaler']

# step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
# 1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
# 1,PAYMENT,1864.28,C1666544295,21249.0,19384.72,M2044282225,0.0,0.0,0,0
data = [[181.0, 181.0, 0.0, 0.0, 0.0, 0.0, 181.0, 0, 0, 0, 0, 1], [
    1864.28, 21249.0, 19384.72, 0.0, 0.0, 0.0, 1864.28, 0, 0, 0, 1, 0]]
df = pd.DataFrame(data, columns=['amount',  'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                  'ErrorBalanceOrigin', 'ErrorBalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])
# 1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
# scaled = lr_sclr.fit_transform(df) GridSearched Scaler
lr_scaled = lr_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('LR Scale Type: \n')
log.info(type(lr_scaled))
log.info('LR Scale Values: \n')
log.info(lr_scaled)
log.info('LR Prediction: \n')
log.info(lr_model.predict(lr_scaled))

gnb_scaled = gnb_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('GNB Scale Type: \n')
log.info(type(gnb_scaled))
log.info('GNB Scale Values: \n')
log.info(gnb_scaled)
log.info('GNB Prediction: \n')
log.info(gnb_model.predict(gnb_scaled))

xgb_scaled = xgb_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('XGB Scale Type: \n')
log.info(type(xgb_scaled))
log.info('XGB Scale Values: \n')
log.info(xgb_scaled)
log.info('XGB Prediction')
log.info(xgb_model.predict(xgb_scaled))

dt_scaled = dt_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('DT Scale Type: \n')
log.info(type(dt_scaled))
log.info('DT Scale Values: \n')
log.info(dt_scaled)
log.info('DT Prediction')
log.info(dt_model.predict(dt_scaled))

rfc_scaled = rfc_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('RFC Scale Type: \n')
log.info(type(rfc_scaled))
log.info('RFC Scale Values: \n')
log.info(rfc_scaled)
log.info('RFC Prediction')
log.info(rfc_model.predict(rfc_scaled))

cb_scaled = cb_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('CB Scale Type: \n')
log.info(type(cb_scaled))
log.info('CB Scale Values: \n')
log.info(cb_scaled)
log.info('CB Prediction')
log.info(cb_model.predict(cb_scaled))

lgbm_scaled = lgbm_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('LGBM Scale Type: \n')
log.info(type(lgbm_scaled))
log.info('LGBM Scale Values: \n')
log.info(lgbm_scaled)
log.info('LGBM Prediction')
log.info(lgbm_model.predict(lgbm_scaled))

bg_scaled = bg_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('BG Scale Type: \n')
log.info(type(bg_scaled))
log.info('BG Scale Values: \n')
log.info(bg_scaled)
log.info('BG Prediction')
log.info(bg_model.predict(bg_scaled))

cb_gs_scaled = cb_gs_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('CB GS Scale Type: \n')
log.info(type(cb_gs_scaled))
log.info('CB GS Scale Values: \n')
log.info(cb_gs_scaled)
log.info('CB GS Prediction')
log.info(cb_gs_model.predict(cb_gs_scaled))

lg_gs_scaled = lg_gs_sclr.transform(df)
# scaled = scaler.transform(df)
log.info('LG GS Scale Type: \n')
log.info(type(lg_gs_scaled))
log.info('LG GS Scale Values: \n')
log.info(lg_gs_scaled)
log.info('LG GS Prediction')
log.info(lg_gs_model.predict(cb_gs_scaled))

log.info(len(cb_model))
log.info(cb_model.steps[0][0])
log.info(cb_model.steps[0][1])
log.info(cb_model.steps[1][0])
log.info(cb_model.steps[1][1])
log.info(cb_model.steps[2][0])
log.info(cb_model.steps[2][1])

# log.info (cb_model.predict(cb_scaled))

# lg_scaled = lg_gs_sclr.transform(df)
# log.info (lg_scaled)
# log.info (lg_gs_model.predict(lg_scaled))
