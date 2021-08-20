from flask import Flask, render_template, request, redirect, url_for

import ast
import sys
import logging
import pandas as pd

import fd_eda

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

lr_sclr = fd_eda.do_loadModel('D:/Users/ng_a/My NYP SDAAI/IT105-ML-Project/lr_scaler.pkl')
lr_model = fd_eda.do_loadModel('D:/Users/ng_a/My NYP SDAAI/IT105-ML-Project/lr.pkl')

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
    log.info (mydata)

    df = pd.DataFrame(mydata)
    log.info(df.head())

    scaled = lr_sclr.transform(df)
    log.info (type(scaled))
    log.info (scaled)
    results = lr_model.predict(scaled)
    log.info (results)

    log.info ('Fraud Detection RESULTS: ' + str(results))
    
    if results[0] == 0:
        app_status = 'Transaction NORMAL'
    else:
        app_status = "Transaction is FRAUDULENT"
    
    log.debug('<-- fraud_detection_service()')
    return app_status

    # return render_template("index.html")
    # return redirect(url_for('fraud_detection_service', data=mydata))

@app.route('/fraud_detection_service/<data>', methods=['GET', 'POST'])
def fraud_detection_service(data):
    log.info('--> fraud_detection_service(): ' + data)

    if request.method == 'POST':
        log.info("POST")
    else:
        log.info('GET')
	# convert the string representation to a dict
    # myData = ast.literal_eval(data)
    # log.info(type(myData))

    # df = pd.DataFrame(myData)
	# and use it as the input
    # df = pd.DataFrame(myData, columns=['amount',  'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'ErrorBalanceOrigin', 'ErrorBalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])
    # log.info(df.head())

    # 1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
    # scaled = scaler.transform(df)
    # log.info (type(scaled))
    # log.info (scaled)
    # result = model.predict(scaled)
    # log.info ('Fraud Detection RESULTS: ' + str(result))
    
    # if result[0] == 0:
    #     app_status = 'Transaction NORMAL'
    # else:
    #     app_status = "Transaction is FRAUDULENT"
    
    # log.debug('<-- fraud_detection_service()')
    # return app_status    

if __name__ == '__main__':
    app.run(debug=True)