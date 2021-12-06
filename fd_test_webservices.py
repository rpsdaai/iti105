from flask import Flask, render_template, request, redirect, url_for

import ast
import sys
import logging
import pandas as pd

import fd_eda

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

# datadir = 'D:/Users/ng_a/My NYP SDAAI/IT105-ML-Project/models/'
datadir = 'D:/Users/ng_a/My NYP SDAAI/tmp/iti105/'
# lr_sclr = fd_eda.do_loadModel(datadir + 'cb_scaler.pkl')
# lr_model = fd_eda.do_loadModel(datadir + 'cb.pkl')
sclr = fd_eda.do_loadModel(datadir + 'cb_best_scaler_gs.pkl')
model = fd_eda.do_loadModel(datadir + 'cb_best_gs.pkl')
log.info(type(sclr))
log.info(type(model))

# Ref: https://www.askpython.com/python-modules/flask/create-hello-world-in-flask
# Ref: https://stackoverflow.com/questions/29277581/flask-nameerror-name-app-is-not-defined
# flask - NameError: name 'app' is not defined
app = Flask(__name__)

def transactionType(txnType):
    log.info('--> transactionType: ' + txnType)
    keys = ['Cash In', 'Cash Out', 'Debit', 'Payment', 'Transfer']
    value = 0

    txnDict = dict.fromkeys(keys, value)
    txnDict[txnType] = 1

    log.info(txnDict)
    return txnDict

@app.route('/')
def index():
    log.debug("--> index(): Entry point")
    # return render_template('index.html')
    return render_template('index.html')

@app.route('/fd_service', methods=['GET', 'POST'])
def fd_service():
    if request.method == 'POST':
        log.info("POST")
        amount = float(request.form['amount'])
        print (amount)
        old_bal_org = float(request.form['old_bal_org'])
        old_bal_dest = float(request.form['old_bal_dest'])
        new_bal_org = float(request.form['new_bal_org'])
        new_bal_dest = float(request.form["new_bal_dest"])
        # Ref: https://stackoverflow.com/questions/31662681/flask-handle-form-with-radio-buttons
        txn_type = request.form["txnType"] 
        log.info (transactionType(txn_type)) 
        txnDict = transactionType(txn_type)
        # return redirect(request.url)
    else:
        log.info("GET")
        amount = float(request.args.get['amount'])
        print (amount)
        old_bal_org = float(request.args.get['old_bal_org'])
        old_bal_dest = float(request.args.get['old_bal_dest'])
        new_bal_org = float(request.args.get['new_bal_org'])
        new_bal_dest = float(request.args.get["new_bal_dest"])
        # Ref: https://stackoverflow.com/questions/31662681/flask-handle-form-with-radio-buttons
        txn_type = request.args.get["txnType"] 
        txnDict = transactionType(txn_type)
        log.info (transactionType(txn_type))

    err_bal_org = new_bal_org - old_bal_org + amount
    err_bal_dest = new_bal_dest - old_bal_dest + amount

    log.info(amount)
    log.info(old_bal_org)
    log.info(old_bal_dest)
    log.info(new_bal_org)
    log.info(new_bal_dest)
    log.info(err_bal_org)
    log.info(err_bal_dest)
    log.info(txn_type)

	# Make a list of dictionary items from user input
    mydata = [
		{
			'amount': amount,
			'oldbalanceOrg': old_bal_org,
			'newbalanceOrig': new_bal_org,
			'oldbalanceDest': old_bal_dest,
			'newbalanceDest': new_bal_dest,
			'ErrorBalanceOrigin': err_bal_org,
			'ErrorBalanceDest': err_bal_dest,
			'type_CASH_IN': txnDict['Cash In'],
			'type_CASH_OUT': txnDict['Cash Out'],
			'type_CASH_DEBIT': txnDict['Debit'],
			'type_CASH_PAYMENT': txnDict['Payment'],
            'type_CASH_TRANSFER': txnDict['Transfer']
        }
    ]
    # log.info (mydata)

    # df = pd.DataFrame(mydata)
    # log.info(df.head())

    # scaled = lr_sclr.transform(df)
    # log.info (type(scaled))
    # log.info (scaled)
    # results = lr_model.predict(scaled)
    # log.info (results)

    # log.info ('Fraud Detection RESULTS: ' + str(results))
    
    # if results[0] == 0:
    #     app_status = 'Transaction NORMAL'
    # else:
    #     app_status = "Transaction is FRAUDULENT"
    
    # log.debug('<-- fraud_detection_service()')
    # return app_status

    # return render_template("index.html")
    return redirect(url_for('fraud_detection_service', data=mydata))

@app.route('/fraud_detection_service/<data>', methods=['GET', 'POST'])
def fraud_detection_service(data):
    log.info('--> fraud_detection_service(): ' + data)
    log.info(type(data))

    if request.method == 'POST':
        log.info("POST")
    else:
        log.info('GET')
     
    # log.info("GET args:\n")
    # log.info(request.args.get('amount'))
    # log.info('DATA\n')
    # log.info(dict(data))
	# convert the string representation to a dict
    myData = ast.literal_eval(data)
    # log.info("Convert String to Dictionary: \n")
    # for d in myData:
    #     for key in d:
    #         print (key)

    log.info('Length of list: ' + str(len(myData)) )
    # Ref: https://stackoverflow.com/questions/5236296/how-to-convert-list-of-dict-to-dict
    # myDict = dict(myData[0])
    # log.info(myDict)

    # a_key = "amount"
    # values_of_key = [a_dict[a_key] for a_dict in myData]
    # log.info(values_of_key)

    # log.info(myData)
    # log.info(type(myData))

    # df = pd.DataFrame(myDict)
	# and use it as the input
    # log.info(myDict.values())
    # log.info(myDict.keys())

    # log.info(type(list(myDict.values())))
    # log.info(type(list(myDict.keys())))
    # df = pd.DataFrame(myDict.values(), columns=['amount',  'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'ErrorBalanceOrigin', 'ErrorBalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])
    df = pd.DataFrame([myData])
    log.info(df.head())

    # 1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
    # use fit_transform() for doGridSearch returned Scaler
    # scaled = lr_sclr.fit_transform(df) # only if return scaler direct from GridSearchCV no need after fixing return type of do_GridSearch
    # use transform() for doPipeline returned Scaler
    scaled = sclr.transform(df)

    log.info (type(scaled))
    log.info (scaled)
    results = model.predict(scaled)
    log.info (results)
    
    if results[0] == 0:
         app_status = 'Transaction NORMAL'
    else:
        app_status = "Transaction is FRAUDULENT"
    
    log.debug('<-- fraud_detection_service()')
    # return app_status
    # return "OK"
    return redirect(url_for('fraud_results', data=app_status))

@app.route('/fraud_results/<data>', methods=['GET', 'POST'])
def fraud_results(data):
    # Ref: https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3
    # Ref: https://www.digitalocean.com/community/tutorials/processing-incoming-request-data-in-flask        
    return '''<h1>Fraud Detection Results</h1><p>Verdict: {}'''.format(data)


if __name__ == '__main__':
    app.run(debug=True)