import logging
import sys
import warnings  # To ignore any warnings warnings.filterwarnings("ignore")
from enum import IntEnum

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns  # For data visualization
from matplotlib import rcParams

import fd_eda

# Ref: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
rcParams.update({'figure.autolayout': True})

warnings.filterwarnings("ignore")

# Turn off - DEBUG findfont: score(<Font 'DejaVu Sans' (DejaVuSans-Oblique.ttf) oblique normal 400 normal>) = 1.05
# Ref: https://github.com/matplotlib/matplotlib/issues/14523
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     handlers=[
#                         logging.FileHandler(
#                             'fraud_detection.log', 'w', 'utf-8'),
#                         logging.StreamHandler(sys.stdout)
#                     ])
log = logging.getLogger(__name__)

vizDestDir = 'vzDump/'

# Ref: https://www.tutorialspoint.com/enum-in-python
class timeStep(IntEnum):
    dayofWeek = 7
    hourOfDay = 24

# Draw labels above the bar plots
# Ref: https://matplotlib.org/examples/api/barchart_demo.html
# Ref: https://stackoverflow.com/questions/39444665/add-data-labels-to-seaborn-factor-plot
def autolabel(ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    # for rect in rects:
    for rect in ax.patches:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        # if p_height > 0.95: arbitrary; 95% looked good to me.
        if p_height > 0.90:
            # label_position = height - (y_height * 0.05)
            label_position = height - (y_height * 0.1)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                # '%d' % int(height),
                '%.2f' % height,
                ha='center', va='bottom', color='red', fontweight='bold')


def do_plotCategorical(df, column, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('Un-Normalised')
    plt.subplots_adjust(wspace=0.5)  # set spacing between plots

    cp_cx = sns.countplot(df[column], ax=ax1)

    autolabel(cp_cx)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Normalised')
    # train = pd.read_csv("data/train")
    train = df.copy()
    ser = train['Loan_Status'].value_counts(normalize=True)

    df1 = pd.DataFrame(ser).reset_index()
    df1.columns = ['Loan_Status', 'count']

    bp_cx = sns.barplot(df1['Loan_Status'], df1['count'])
    autolabel(bp_cx)

    fig.savefig(filename)
    # plt.show()
    plt.close(fig)


def plotFraudTransactionCounts(df, column, filename):
    log.info('--> plotFraudTransactionCounts(): column = ' + column + ' filename = ' + filename)
    fig = plt.figure(figsize=(6, 6))

    fraud_count = df[column].value_counts()
    fraud_count = fraud_count[:2, ]
    # plt.figure(figsize=(10,5))
    ax = sns.barplot(x=fraud_count.index, y=fraud_count.values, palette=['blue', 'red'])
    # ax.legend(ncol=2, loc="upper right", frameon=True)
    # ax.legend()

    # Ref: https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(6, 6), textcoords='offset points')

    ax.legend(ncol=2, loc="upper right", frameon=True)
    # ax.legend()

    # plt.legend(frameon = False)
    # plt.legend(fraud_count, ['Genuine', 'Fraud']);

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels)
    # print('Handles: ', handles)
    # print('Labels: ', labels)
    plt.title('Normal vs Fraud Transaction Count')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Transaction Type (Normal vs Fraud)', fontsize=12)

    plt.tight_layout()

    fig.savefig(filename)
    # plt.show()
    plt.close(fig)


def plotCountTransactionTypes(df, column, filename):
    log.info('--> plotCountTransactionTypes(): column = ' +
             column + ' filename = ' + filename)
    fig = plt.figure()

    ax = sns.barplot(x=df[column].value_counts().index,
                     y=df[column].value_counts().values)
    plt.legend(df[column].value_counts(), df[column].unique())

    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')),
                    (p.get_x(), p.get_height() * 1.01))

    # Ref: https://stackoverflow.com/questions/65272126/seaborn-how-to-add-legend-to-seaborn-barplot
    patches = [mpl.patches.Patch(color=sns.color_palette()[i], label=t)
               for i, t in enumerate(t.get_text() for t in ax.get_xticklabels())]
    plt.legend(handles=patches, loc="upper right")

    plt.title('Transaction Types vs. Count of Transaction Types')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Transaction Types', fontsize=12)

    print('Patches: ', patches)
    plt.tight_layout()

    fig.savefig(filename)
    # plt.show()
    plt.close(fig)


def plotCountTransactionTypesGrpByFraudTypes(df, columnList, filename):
    # Ref: https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string
    log.info('--> plotountTransactionTypesGrpByFraudTypes(): column = ' +
             ''.join(columnList) + ' filename = ' + filename)
    fig = plt.figure()

    # Ref: https://stackoverflow.com/questions/52089678/how-to-add-custom-strings-to-legend-of-count-value-plots-in-pandas
    # Ref: https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
    ax = df.groupby(columnList).size().plot.bar(
        color=sns.color_palette(), figsize=(7, 6), legend=True, rot=90)

    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')),
                    (p.get_x(), p.get_height() * 1.01))

    # Ref: https://stackoverflow.com/questions/65272126/seaborn-how-to-add-legend-to-seaborn-barplot
    patches = [mpl.patches.Patch(color=sns.color_palette()[i], label=t)
               for i, t in enumerate(t.get_text() for t in ax.get_xticklabels())]
    plt.legend(handles=patches, loc="upper right")

    plt.title('Transaction Types vs. Count of Transaction Types')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Transaction Types, isFraud', fontsize=12)

    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close(fig)


# Plots the correlation between ALL numerical variables. The darker the oolor the higher the correlation
def plotCorrelationMap(df, oheColumn, columns2Drop, filename):
    log.info(
        '--> plotCorrelationMap(): filename = ' + filename + ' oheColumn = ' + oheColumn + ' columns2Drop = ' + ''.join(
            columns2Drop))

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('Correlation Plot', y=1.0)
    # fig = plt.figure() 
    # df_tmp = pd.get_dummies(df, columns=['type'])
    # df_tmp.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
    # df_tmp.head()
    df_tmp = pd.get_dummies(df, columns=[oheColumn])
    df_tmp.drop(columns2Drop, axis=1, inplace=True)
    # df_tmp.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
    sns.heatmap(df_tmp.corr(), annot=True)

    plt.xticks(rotation=60)
    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close(fig)

# Plots the correlation between ALL numerical variables. The darker the oolor the higher the correlation
# This is applied on dataframe that already has been feature engineered
def plotFECorrelationMap(df, filename):
    log.info('--> plotFECorrelationMap(): filename = ' + filename)

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('Correlation Plot', y=1.0)

    sns.heatmap(df.corr(), annot=True)

    plt.xticks(rotation=60)
    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close(fig)

def plotTransactionsOverTime(df, filename):
    log.info('--> plotTransactionsOverTime(): filename = ' + filename)
    # fig = plt.figure() If uncomment this will show 2 plots - an empty one and the rendered one with 3 graphs!!

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    # Ref: https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
    # Ref: https://www.geeksforgeeks.org/matplotlib-axes-axes-hist-in-python/
    # ax1 = fig.add_subplot(131)

    df_tmp = df.filter(['step', 'isFraud'], axis=1)
    # Ref: https://www.statology.org/seaborn-title/

    sns.histplot(df_tmp, x="step", stat='count', ax=ax[0]).set(title='# Transactions (Fraud + Normal) vs Steps',
                                                               xlabel='Hourly Time Steps')

    sns.histplot(df[df["isFraud"] == 1].step, bins=50, color='red', stat='count', ax=ax[1]).set(
        title='# Transactions (Fraud) vs Steps', xlabel='Hourly Time Steps')

    sns.histplot(df[df["isFraud"] == 0].step, bins=50, color='green', stat='count', ax=ax[2]).set(
        title='# Transactions (Normal) vs Steps', xlabel='Hourly Time Steps')

    # plt.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close(fig)


def plotTransactionsOverTimeBy(df, filename, by, xLabel, yLabel, title1, title2):
    log.info('--> plotTransactionsOverTimeByDay(): filename = ' + filename)
    # fig = plt.figure() If uncomment this will show 2 plots - an empty one and the rendered one with 3 graphs!!

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Ref: https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
    # Ref: https://www.geeksforgeeks.org/matplotlib-axes-axes-hist-in-python/
    # ax1 = fig.add_subplot(131)

    fraud = df[df["isFraud"] == 1].step % by.value
    sns.histplot(fraud, bins=by.value, color='red', stat='count', ax=ax[0]).set(
        title=title1, xlabel=xLabel, ylabel=yLabel)

    normal = df[df["isFraud"] == 0].step % by.value
    sns.histplot(normal, bins=by.value, color='green', stat='count', ax=ax[1]).set(
        title=title2, xlabel=xLabel, ylabel=yLabel)

    # plt.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(filename)
    # plt.show()
    plt.close(fig)


def plotFraudByTransactionType(df, filename):
    log.info('--> doPlotFraudByTransactionType(): filename = ' + filename)
    pie, ax = plt.subplots(figsize=[10, 6])
    labels = df[df['isFraud'] == 1].type.unique()

    # Ref: https://pythontic.com/visualization/charts/piechart
    plt.pie(x=df[df['isFraud'] == 1].type.value_counts(), autopct="%.1f%%", labels=labels, pctdistance=0.5,
            explode=(0.1, 0.0))

    # Position the title
    # Ref: https://matplotlib.org/stable/gallery/text_labels_and_annotations/titles_demo.html
    # plt.title("Fraud by Transaction Type", fontsize=14, loc='left')
    plt.title("Fraud by Transaction Type", fontsize=14)

    # plt.axis('equal')
    plt.legend()
    pie.savefig(filename)
    # plt.show()
    plt.close(pie)


def plotBalances(pd_series, plotTitle, filename):
    pie, ax = plt.subplots(figsize=[10, 6])
    plt.pie(x=pd_series, autopct="%.1f%%", labels=pd_series.index, pctdistance=0.5,
            explode=(0.1, 0.0))
    plt.title(plotTitle, fontsize=14)
    plt.legend()
    pie.savefig(filename)
    # plt.show()
    plt.close(pie)


# Skew
# Ref: https://medium.com/@atanudan/kurtosis-skew-function-in-pandas-aa63d72e20de
def do_Skew(df):
    log.info('--> do_Skew()')
    # log.info(df.skew(axis=0).apply(np.log))  # Column wise
    # log.info(df.skew(axis=1).apply(np.log))  # Row wise

    df_skew_log = df.skew(axis=0).apply(np.log)
    df_skew = df.skew(axis=0)
    log.info(df_skew)
    log.info('TYPE: ' + str(type(df_skew)) + ' ' + str(type(df_skew_log)))
    log.info(df_skew.index)
    log.info(df_skew.values)

    # Ref: https://stackoverflow.com/questions/18062135/combining-two-series-into-a-dataframe-in-pandas
    df_skew_all = pd.concat([df_skew, df_skew_log], axis = 1)
    df_skew_all.columns = ['Skew', 'Skew (Log)']
    log.info(df_skew_all)
    log.info(df_skew_all.columns)
    
    log.info('<-- do_Skew()\n')

    return df_skew_all


# Kurtosis
def do_Kurtosis(df):
    log.info('--> do_Kurtosis()\n')

    df_kurt_log = df.kurt(axis=0).apply(np.log) # axis = 0: Column-wise, axis = 1: row-wise
    df_kurt = df.kurt(axis=0)
    log.info(df_kurt)
    log.info('TYPE: ' + str(type(df_kurt)) + ' ' + str(type(df_kurt_log)))
    log.info(df_kurt.index)
    log.info(df_kurt.values)

    # Ref: https://stackoverflow.com/questions/18062135/combining-two-series-into-a-dataframe-in-pandas
    df_kurt_all = pd.concat([df_kurt, df_kurt_log], axis = 1)
    df_kurt_all.columns = ['Kurt', 'Kurt (log)']
    log.info(df_kurt_all)

    log.info('<-- do_Kurtosis()\n')
    return df_kurt_all


# Explore the dataset to get general feel
if __name__ == '__main__':
    df = fd_eda.read_datatset(fd_eda.dataset_dir, fd_eda.datafile)
    # log.info(df['step'].value_counts())

    df_skew = do_Skew(df)
    df_kurt = do_Kurtosis(df)

    # plotTransactionsOverTime(df, vizDestDir + 'txnOverTime')

    # plotTransactionsOverTimeBy(df, vizDestDir + 'txnOverTime_dayOfWeek', timeStep.dayofWeek, 'Day of the Week', '# Transactions',
    #                            'Fraud Transactions by Day', 'Normal Transactions by Day')
    # plotTransactionsOverTimeBy(df, vizDestDir + 'txnOverTime_hourOfDay', timeStep.hourOfDay, 'Hour of Day', '# Transactions',
    #                            'Fraud Transactions by Hour', 'Normal Transactions by Hour')
    # plotFraudTransactionCounts(df, 'isFraud', vizDestDir + 'fraudTxnCounts')
    # plotCountTransactionTypes(df, 'type', vizDestDir + 'typeTxnCounts')
    # plotCountTransactionTypesGrpByFraudTypes(df, ['type','isFraud'], vizDestDir + 'typeFraudTxn')
    # plotCorrelationMap(df, 'type', ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], vizDestDir + 'corrMap') # not feature engineered yet
    # plotFraudByTransactionType(df, vizDestDir + 'fraudByTxnType')
    plotBalances(fd_eda.computeSrcBalance(df), 'Sender Balance Amount Mismatches', vizDestDir + 'senderBalAmtMismatch')
    plotBalances(fd_eda.computeDestinationBalance(df), 'Receipient Balance Amount Mismatches', vizDestDir + 'receipientBalAmtMismatch')
    plotBalances(fd_eda.countSrcAmountTransferredExceedsBalance(df),
                 'Sender Amount Transferred Exceeds Available Balance', vizDestDir + 'senderAmtTrsfrXBal')
    plotBalances(fd_eda.countAmountReceivedExceedsBalance(df), 'Receipient Amount Received Exceeds Available Balance',
                 vizDestDir + 'receipientAmtRecXBal')

    # df_fe = fd_eda.doPrepareData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    # plotFECorrelationMap(df_fe, vizDestDir + 'corrMapFeatureEngineered')
