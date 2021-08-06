# from sklearn.naive_bayes import GaussianNB
import pickle
import joblib

results_all = {}

def do(modelName, name, score):
    results = {}
    results[modelName] = {}
    results[modelName]['cm'] = name
    results[modelName]['auc_roc_score'] = score

    return results

def do_loadModel(filename):
	# log.debug('--> do_loadModel(): ' + filename)
	with open(filename, 'rb') as f:
		model = pickle.load(f)
	return (model)

if __name__ == '__main__':
    results_all['lr'] = do('lr', 'LR', 0.8823)
    results_all['svm'] = do('svm', 'SVM', 0.123)
    # print (results_all)

    resultsDict = do_loadModel('results.pkl')
    # Ref: https://www.programiz.com/python-programming/nested-dictionary
    # for p_id, p_info in resultsDict.items():
    #     print("\nPerson ID:", p_id)
    #     print('\nPINFO: ', p_info)
    #     # for key in p_info:
    #     #     print (key)
    #         # print(key + ':', p_info[key])    
    # print (resultsDict['lr'].keys())
    # print (resultsDict.values())
    test = resultsDict['lr']['lr']['precision_scores']
    print (test)
    print (resultsDict['lr']['lr']['classificationReport'])
    # print (resultsDict['xgb'])
    # print (resultsDict['gnb'])
    # print (resultsDict['dt'])
    # print (resultsDict['rfc'])
    # print (resultsDict['cb'])
    # print (resultsDict['lgbm'])
    # print (resultsDict['bg'])
