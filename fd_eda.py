import warnings

import pandas as pd
# Ref: https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings/32389270
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import sys
import logging

import pickle
import joblib

# Ref: https://miamioh.instructure.com/courses/38817/pages/data-cleaning

# Log to both console + file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler('fraud_detection.log', 'w', 'utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

dataset_dir = 'D:/Users/ng_a/My NYP SDAAI/IT105-ML-Project/Datasets/Synthetic_Financial_Datasets_4_Fraud_Detection/'
datafile = 'PS_20174392719_1491204439457_log.csv'

# Ref: https://stackoverflow.com/questions/21137150/format-suppress-scientific-notation-from-python-pandas-aggregation-results
pd.options.display.float_format = '{:.2f}'.format


# Read the dataset and return the dataframe
def read_datatset(data_directory, filename):
    log.info('\n--> read_dataset(): ' + data_directory + ' ' + filename + '\n')
    df = pd.read_csv(data_directory + filename)
    return df


# Get frequencies of each unique value in a column
# Ref: https://towardsdatascience.com/getting-more-value-from-the-pandas-value-counts-aa17230907a6
def getFrequencies(column, df):
    log.info('--> getFrequencies(): column = ' + column + '\n')
    return df[column].value_counts()


# Returns a count of unique values per column; Omits missing values
# Note the unique method returns missing values as well c.f. to nunique
def getUniqueValues(column, df):
    log.info('--> getUniqueValues(): column = ' + column + '\n')
    return df[column].nunique()


# Returns rows, columns of dataframe
def getDimensions(df):
    log.info('\n--> getDimensions()\n')
    return df.shape

# get min, max, count, mean, standard deviation, median, 25 percentile, 75 percentile
def getStatisticalInfo(df):
    log.info('\n--> getStatisticalInfo()\n')
    log.info(df.describe())


# Check for missing, null values
def checkMissingValues(df):
    log.info('\n--> checkMissingValues()\n')
    # Returns dataFrame mask of bool values for each element in DataFrame that indicates whether an element is an NA value
    log.info(df.isnull())
    return df.isnull()


# Returns the number of NaN values in every column.
def getNumberMissingValues(df):
    log.info('\n--> getNumberMissingValues()\n')
    log.info('\n Missing values sum x 2: ' + str(df.isnull().sum().sum()) + '\n')
    log.info('\n Missing values sum: ' + str(df.isnull().sum()) + '\n')
    return df.isnull().sum().sum()


# Returns column names as list
def getListofColumnNames(df):
    log.info('\n--> getListofColumnNames()\n')
    return df.columns.values.tolist()


def do_FeatureEngineering_TransactionType(df):
    log.info('\n--> do_FeatureEngineering_TransactionType()\n')
    # Make a copy of original dataframe
    df_new = df.copy()

    # Create a new feature column to capture Customer-2-Customer, Customer-2_Merchant
    # Merchant-2-Customer, Merchant-2-Merchant transactions
    df_new.loc[df.nameOrig.str.contains('C') & df.nameDest.str.contains('C'), "txn_type"] = "C2C"
    df_new.loc[df.nameOrig.str.contains('C') & df.nameDest.str.contains('M'), "txn_type"] = "C2M"
    df_new.loc[df.nameOrig.str.contains('M') & df.nameDest.str.contains('C'), "txn_type"] = "M2C"
    df_new.loc[df.nameOrig.str.contains('M') & df.nameDest.str.contains('M'), "txn_type"] = "M2M"

    return df_new


def do_FeatureEngineering_ErrorBalanceAmt(df):
    log.info('\n--> do_FeatureEngineering_ErrorBalanceAmt\n')

    # Ref: https://www.geeksforgeeks.org/how-to-create-an-empty-dataframe-and-append-rows-columns-to-it-in-pandas/
    # Create an empty dataframe
    df_new = pd.DataFrame()

    # Create a new feature column to capture Customer-2-Customer, Customer-2_Merchant
    # Merchant-2-Customer, Merchant-2-Merchant transactions
    df_new['ErrorBalanceOrigin'] = df['newbalanceOrig'] - df['oldbalanceOrg'] + df['amount']
    df_new['ErrorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest'] + df['amount']

    return df_new


def do_FeatureEngineering_Finalize(df, columns2drop):
    log.info('\n--> do_FeatureEngineering_Finalize\n')
    df.drop(columns=columns2drop, inplace=True, axis=1)
    df_tmp = pd.concat([df, do_FeatureEngineering_ErrorBalanceAmt(df)], axis=1)
    return df_tmp


def getCountFraudulentTransactionTypes(df, column):
    log.info('\n--> getCountFraudulentTransactionTypes()\n')
    return df[df['isFraud'] == 1][column].value_counts()

def getCountFraudulentTransactionByAmounts(df, column, columnList):
    log.info('\n--> getCountFraudulentTransactionByAmounts()\n')
    return df.loc[df[column] == 1, columnList].value_counts(ascending=True)

def getCountNormalTransactionTypes(df, column):
    log.info('\n--> getCountNormalTransactionTypes()\n')
    return df[df['isFraud'] == 0][column].value_counts()


def doPrepareAndSplitData(df, columns2Drop):
    # Ref: https://www.decalage.info/en/python/print_list
    log.info('\n--> doPrepareAndSplitData(): columns2Drop = ' + ', '.join(columns2Drop) + '\n')
    
    # Make a copy of dataframe
    df_tmp1 = df.copy()

    # Create empty dataframe
    X = pd.DataFrame()
    y = pd.DataFrame()

    # No missing data
    if getNumberMissingValues(df) == 0:
        log.info("No missing data, proceeding with OHE and feature engineering")
        df_tmp2 = do_FeatureEngineering_Finalize(df_tmp1, columns2Drop)

        # Convert categorical text labels to numericals, where type is one of: CASH_OUT, PAYMENT, CASH_IN, 
        # TRANSFER, DEBIT, CASH_OUT, TRANSFER
        df_tmp = pd.get_dummies(df_tmp2, columns=['type'])

        # Extract out target variable into y 
        y = df_tmp['isFraud']

        # Extract the remaining features into X
        X = df_tmp.drop(['isFraud'], axis = 1)
    else:
        log.info("Missing data, CANNOT proceed with OHE and feature engineering")

    return X, y


def doPrepareData(df, columns2Drop):
    # Ref: https://www.decalage.info/en/python/print_list
    log.info('\n--> doPrepareData(): columns2Drop = ' + ', '.join(columns2Drop) + '\n')
    
    # Make a copy of dataframe
    df_tmp1 = df.copy()

    # No missing data
    if getNumberMissingValues(df) == 0:
        log.info("No missing data, proceeding with OHE and feature engineering")
        df_tmp2 = do_FeatureEngineering_Finalize(df_tmp1, columns2Drop)

        # Convert categorical text labels to numericals, where type is one of: CASH_OUT, PAYMENT, CASH_IN, 
        # TRANSFER, DEBIT, CASH_OUT, TRANSFER
        df_tmp = pd.get_dummies(df_tmp2, columns=['type'])
    else:
        log.info("Missing data, CANNOT proceed with OHE and feature engineering")
        df_tmp = None
        
    # return final dataframe that is OHE
    return df_tmp

'''
Check balances before and after transaction
'''

# Compute sender's balance after $X deducted
def computeSrcBalance(df):
    log.info('\n--> computeSrcBalance()\n')
    # Check sender balance for errors, Series returned
    return (df['oldbalanceOrg'] - df['amount'] != df['newbalanceOrig']).value_counts(normalize=True)

# Compute receiver's balance after $X credited
def computeDestinationBalance(df):
    log.info('\n--> computeDestinationBalance()\n')
    # Check receiver balance for errors, Series returned
    # return (df['oldbalanceDest'] + df['amount'] != df['newbalanceDest']).value_counts(normalize=True)
    return (df['oldbalanceDest'] + df['amount'] > df['newbalanceDest']).value_counts(normalize=True)

# Amount transferred more than what sender has in his/her account
def countSrcAmountTransferredExceedsBalance(df):
    log.info('\n--> countAmountTransferredExceedsBalance()\n')
    return (df['amount'] > df['oldbalanceOrg']).value_counts()


# Amount transferred more than what receiver has in his/her account
def countAmountReceivedExceedsBalance(df):
    log.info('\n--> countAmountTransferredExceedsBalance()\n')
    return (df['amount'] > df['newbalanceDest']).value_counts()

'''
Saving model to file and loading model files
'''
# Save model to disk
def do_saveModel(filename, model, which_library):
	log.debug('--> do_saveModel(): ' + filename + ' library to use: ' + which_library)
	with open(filename, 'wb') as f:
		if which_library == 'p':
			pickle.dump(model, f)
		else:
			joblib.dump(model, f)


# Load model file from disk
def do_loadModel(filename):
	log.debug('--> do_loadModel(): ' + filename)
	with open(filename, 'rb') as f:
		model = pickle.load(f)
	return (model)

# # Save results to disk
# def do_saveResults(filename, model, which_library):
# 	log.debug('--> do_saveResults(): ' + filename + ' library to use: ' + which_library)
# 	with open(filename, 'wb') as f:
# 		if which_library == 'p':
# 			pickle.dump(model, f)
# 		else:
# 			joblib.dump(model, f)


# # Load results from disk
# def do_loadResults(filename):
# 	log.debug('--> do_loadResults(): ' + filename)
# 	with open(filename, 'rb') as f:
# 		model = pickle.load(f)
# 	return (model)    

# Explore the dataset to get general feel
if __name__ == '__main__':
    df = read_datatset(dataset_dir, datafile)
    # log.info(getFrequencies('isFraud', df))
    # log.info(getFrequencies('isFlaggedFraud', df))
    # log.info(getUniqueValues('isFraud', df))
    # log.info(getUniqueValues('isFlaggedFraud', df))
    # log.info(getDimensions(df))
    # log.info(getStatisticalInfo(df))
    # log.info(checkMissingValues(df))
    # log.info(getNumberMissingValues(df))
    # log.info(getListofColumnNames(df))

    # Count the number of fraudulent transactions by amount (isFruad)
    log.info('\n' + str(getCountFraudulentTransactionByAmounts(df, 'isFraud', ['amount', 'isFraud'])) + '\n')

    # Count the number of fraudulent transactions by amount (isFlaggedFraudulent)
    log.info('\n' + str(getCountFraudulentTransactionByAmounts(df, 'isFlaggedFraud', ['amount', 'isFlaggedFraud'])) + '\n')

    df_tmp = do_FeatureEngineering_TransactionType(df)

    # Get fraudulent count for new column, 'txn_type'
    log.info('\n' + str(getCountFraudulentTransactionTypes(df_tmp, 'txn_type')) + '\n')

    # Get normal count for new column, 'txn_type'
    log.info('\n' + str(getCountNormalTransactionTypes(df_tmp, 'txn_type')) + '\n')

    # Get fraudulent count for original column, type
    log.info('\n' + str(getCountFraudulentTransactionTypes(df_tmp, 'type')) + '\n')
    # Get normal count for original column, 'type'
    log.info('\n' + str(getCountNormalTransactionTypes(df_tmp, 'type')) + '\n')

    # log.info('\n' + 'Statistics for Fraudulent Transactions' + '\n')
    # log.info(getStatisticalInfo(df[df['isFraud'] == 1]['amount']))

    # log.info('\n' + 'Statistics for NORMAL Transactions' + '\n')
    # log.info(getStatisticalInfo(df[df['isFraud'] == 0]['amount']))

    # df_new = do_FeatureEngineering_ErrorBalanceAmt(df)
    # df_orig = df.copy()
    # df_orig = do_FeatureEngineering_Finalize(df_orig, df_new, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    # df_orig.head()

    # df = do_FeatureEngineering_Finalize(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    # print (getListofColumnNames(df))

    # X, y  = doPrepareAndSplitData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    # log.info('\n Features DF: \n')
    # log.info(X.head())
    # log.info('\n Target DF: \n')
    # log.info(y.head())