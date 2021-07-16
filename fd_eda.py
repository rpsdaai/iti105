import warnings

import pandas as pd
# Ref: https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings/32389270
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import sys
import logging

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


def getStatisticalInfo(df):
    log.info('\n--> getStatisticalInfo()\n')
    log.info(df.describe())


def checkMissingValues(df):
    log.info('\n--> checkMissingValues()\n')
    log.debug(df.isnull())
    return df.isnull()


def getNumberMissingValues(df):
    log.info('\n--> getNumberMissingValues()\n')
    log.debug(df.isnull().sum().sum())
    return df.isnull().sum().sum()


def getListofColumnNames(df):
    log.info('\n--> getListofColumnNames()\n')
    return df.columns.values.tolist()


def do_FeatureEngineering(df):
    log.info('\n--> do_FeatureEngineering()\n')
    # Make a copy of original dataframe
    df_new = df.copy()

    # Create a new feature column to capture Customer-2-Customer, Customer-2_Merchant
    # Merchant-2-Customer, Merchant-2-Merchant transactions
    df_new.loc[df.nameOrig.str.contains('C') & df.nameDest.str.contains('C'), "txn_type"] = "C2C"
    df_new.loc[df.nameOrig.str.contains('C') & df.nameDest.str.contains('M'), "txn_type"] = "C2M"
    df_new.loc[df.nameOrig.str.contains('M') & df.nameDest.str.contains('C'), "txn_type"] = "M2C"
    df_new.loc[df.nameOrig.str.contains('M') & df.nameDest.str.contains('M'), "txn_type"] = "M2M"

    return df_new


def getCountFraudulentTransactionTypes(df, column):
    log.info('\n--> getCountFraudulentTransactionTypes()\n')
    return df[df['isFraud'] == 1][column].value_counts()


def getCountNormalTransactionTypes(df, column):
    log.info('\n--> getCountFraudulentTransactionTypes()\n')
    return df[df['isFraud'] == 0][column].value_counts()


# Explore the dataset to get general feel
if __name__ == '__main__':
    df = read_datatset(dataset_dir, datafile)
    log.info(getFrequencies('isFraud', df))
    log.info(getFrequencies('isFlaggedFraud', df))
    log.info(getUniqueValues('isFraud', df))
    log.info(getUniqueValues('isFlaggedFraud', df))
    log.info(getDimensions(df))
    log.info(getStatisticalInfo(df))
    log.info(checkMissingValues(df))
    log.info(getNumberMissingValues(df))
    log.info(getListofColumnNames(df))
    df_tmp = do_FeatureEngineering(df)
    log.info('\n' + str(getCountFraudulentTransactionTypes(df_tmp, 'txn_type')) + '\n')
    log.info('\n' + str(getCountNormalTransactionTypes(df_tmp, 'txn_type')) + '\n')
    log.info('\n' + str(getCountFraudulentTransactionTypes(df_tmp, 'type')) + '\n')
    log.info('\n' + str(getCountNormalTransactionTypes(df_tmp, 'type')) + '\n')