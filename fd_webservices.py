from flask import Flask, render_template, request, redirect, url_for

import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#ch.setFormatter(formatter)
# add the handlers to the logger
log.addHandler(ch)

# Ref: https://www.askpython.com/python-modules/flask/create-hello-world-in-flask
# Ref: https://stackoverflow.com/questions/29277581/flask-nameerror-name-app-is-not-defined
# flask - NameError: name 'app' is not defined
app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello World'

# return "Web Service is running!"
@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
	log.debug("webhook(): test that web service is running")
	return "Web Service is running!"    

@app.route('/')
def getdata():
   log.debug("--> getdata()")
   amt = request.args.get('Amount')
   frm = request.args.get('From')
   to = request.args.get('To')
   log.debug('Amount: ', amt, ' FROM: ', frm, ' TO: ', to)
   return render_template('getdata.html')
   # if request.method == 'POST':
   #    amt = request.args.get('Amount')
   #    frm = request.args.get('From')
   #    to = request.args.get('To')
   #    app.logger.debug('Amount: ', amt, ' FROM: ', frm, ' TO: ', to)
   #    # return 'welcome %s %s %s' % amt % frm % to 
   #    return redirect(url_for('data', amount = amt, source = frm, destination = to))
   #    # return render_template('getdata.html', error=None)
   # else:
   #    amt = request.args.get('Amount')
   #    frm = request.args.get('From')
   #    to = request.args.get('To')
   #    app.logger.debug(amt)
   #    # return render_template('getdata.html', error=None)
   #    # return 'welcome %s %s %s' % amt % frm % to
   #    return redirect(url_for('data', amount = amt, source = frm, destination = to))   

@app.route('/success',methods = ['POST', 'GET'])  
def print_data():  
   log.debug("--> print_data()")
   if request.method == 'POST':  
      result = request.form  
      log.debug(result.values)
      return render_template("result_data.html",result = result)  

@app.route('/data/<name>/<source>/<destination>', methods = ['POST', 'GET'])
def data(name, source, destination):
   log.debug("--> data(): name = ", name, " source: ", source, " destination: ", destination)
   return "Test SUccess"
   # return 'welcome %s %s %s' % request.form['Amount'] % request.form['From'] % request.form['To']
   #  if request.method == 'GET':
   #      return f"The URL /data is accessed directly. Try going to '/form' to submit form"
   #  if request.method == 'POST':
   #      form_data = request.form
   #      return render_template('data.html',form_data = form_data)

# @app.route('/result', methods = ['POST', 'GET'])
# def result(name, source, destination):
# def result():
#   return 'welcome %s %s %s' % request.form['Amount'] % request.form['From'] % request.form['To']
   # result = request.form
   # return render_template("result.html", result = result)
   # return 'welcome %s %s %s' % name % source % destination
   # if request.method == 'POST':
   #    amt = request.form['Amount']
   #    frm = request.form['From']
   #    to = request.form['To']
   #    return redirect(url_for('result', amount = amt, source = frm, destination = to))
   # else:
   #    amt = request.form['Amount']
   #    frm = request.form['From']
   #    to = request.form['To']
   #    return redirect(url_for('result', amount = amt, source = frm, destination = to))   

# Ref: https://www.quora.com/How-do-I-run-python-flask-file-when-click-a-HTML-button
@app.route('/test_form_service', methods=['GET', 'POST'])
def test_form_service():
   log.debug("--> test_form_service()")
   if request.method == 'POST':
      log.debug("POST")
      amt = request.form['Amount']
      frm = request.form['From']
      to1 = request.form['To']
   else:
      log.debug("GET")
      # amt = request.form.get('Amount')
      # frm = request.form.get('From')
      # to = request.form.get('To')
      amt = request.args.get('Amount')
      frm = request.args.get('From')
      to = request.args.get('To')
	# return redirect(url_for('success',name = fname+' '+lname))
   # log.debug("amt: ", amt, " frm: ", frm, " to: ", to)
	# # Make a list of dictionary items from user input
   # mydata = [
	# 	{
	# 		'Amount': amt,
	# 		'From': frm,
	# 		#'To': to
	# 	}
	# ]
   # log.debug('DATA: ' + str(mydata))

	# return redirect(url_for('success', name=fname, data=mydata))
	# redirect to next url for loan prediction using trained model
   # return redirect(url_for('fraud_detection_service', name=amt, data=mydata))
   return render_template('getdata.html')

@app.route('/fraud_detection_service/<name>/<data>', methods=['GET', 'POST'])
def fraud_detection_service(name, data):
	log.debug('fraud_detection_service()' + str(type(data)))
	log.debug('Name: ' + name)
	log.debug('data contents: ' + data)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
