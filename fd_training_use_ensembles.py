from platform import machine
from matplotlib import colors
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

# Support Vector machine
from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns

import fd_eda
import fd_training

import sys
import logging

# Turn off - DEBUG findfont: score(<Font 'DejaVu Sans' (DejaVuSans-Oblique.ttf) oblique normal 400 normal>) = 1.05
# Ref: https://github.com/matplotlib/matplotlib/issues/14523
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Log to both console + file
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     handlers=[
#                         logging.FileHandler('fraud_detection_1.log', 'w', 'utf-8'),
#                         logging.StreamHandler(sys.stdout)
#                     ])

# __name__ contains the full name of the current module
log = logging.getLogger(__name__)

# Ref: https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html
def plot_feature_importance(filename, importance, names):
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
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette="bright")
    # Add chart labels
    plt.title('Visualizing Important Features')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    fig.savefig(filename)
    plt.close()

def do_plotImportantFeatures(filename, feature_imp, feature_cols):
    log.info('--> do_plotImportantFeatures()')

    fig = plt.figure(figsize = (12, 6))

    # Creating a bar plot
    ax = sns.barplot(x=feature_imp, y=feature_cols, palette="bright")

    # handles, labels = ax.get_legend_handles_labels()
    # labels = X_test.columns.values.tolist()
    # log.info('Labels: ' + ' '.join(labels))

    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")

    # plt.legend()
    # Ref: https://www.delftstack.com/howto/seaborn/legend-seaborn-plot/
    # plt.legend(labels=labels)

    # Ref: https://stackoverflow.com/questions/48541040/saving-figures-using-plt-savefig-on-colaboratory
    fig.savefig(filename)

    # plt.show()    
    plt.close()

def do_plotImportantF(filename, feature_imp, feature_cols):
    # feature_imp_ = pd.Series(feature_imp, index=feature_cols).sort_values(ascending=False)
    fig = plt.figure(figsize=(12,12))
    ax = feature_imp.plot(kind='bar', colormap='Paired')

	# Ref: https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    # do_horizontalLabelPlacement(ax)

    plt.ylabel('Feature Importance Score')
    plt.xlabel('Features')
    plt.title("Visualizing Important Features")

    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

if __name__ == '__main__':
    # Read the dataset
    df = fd_eda.read_datatset(fd_eda.dataset_dir, fd_eda.datafile)
    # X, y  = fd_eda.doPrepareData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # # TRAIN TEST SPLITTING
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)
    X_train, X_test, y_train, y_test = fd_training.do_SplitData(df, ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], 1, 0.2, 88, 'isFraud')

    # exit(0)
    # Bagging Classifier
    # bg, bg_pipe = fd_training.do_PipelineOnly("bg", BaggingClassifier(DecisionTreeClassifier()), X_train, y_train)
    # fd_training.do_evaluateModel(bg_pipe, X_test, y_test)
    # fd_training.do_plotEvaluationCurves('bg', bg, X_test, y_test, 'PuBu')
    # fd_eda.do_saveModel('bg', bg, 'p')

    # lr, lr_pipe = fd_training.do_PipelineOnly("lr", LogisticRegression(max_iter=100), X_train, y_train)
    # log.info(lr_pipe[0])
    # log.info(type(lr_pipe[0]))
    # lr_pipe.steps[0] = lr_pipe[0] = scaler
    # named_steps allows attribute access
    # log.info(lr_pipe[0].named_steps)
    # Ref: https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models
    # fd_eda.do_saveModel('lr_scaler.pkl', lr_pipe[0], 'p')



    rfc, rfc_pipe = fd_training.do_PipelineOnly("rfc", RandomForestClassifier(n_estimators = 50, random_state = 88, n_jobs = -1, oob_score = True), X_train, y_train)

    # fd_training.do_evaluateModel(rfc_pipe, X_test, y_test)
    # fd_training.do_plotEvaluationCurves('rfc', rfc, X_test, y_test, 'BuGn')
    # fd_eda.do_saveModel('rfc.pkl', rfc, 'p') 

    # Ref: https://stackoverflow.com/questions/28822756/getting-model-attributes-from-pipeline
    # Ref: https://scikit-learn.org/stable/modules/compose.html
    # pipeline.named_steps['pca']
    # pipeline.steps[1][1]
    
    feature_imp = pd.Series(rfc_pipe.steps[2][1].feature_importances_, index=X_test.columns.values.tolist()).sort_values(ascending=False)
    # # do_plotImportantFeatures('feature_importance_2.png', feature_imp, X_test.columns.values.tolist()) 
    # plot_feature_importance('feature_importance_1.png', rfc_pipe.steps[2][1].feature_importances_, X_test.columns.values.tolist(), 'Random Forest')
    plot_feature_importance('feature_importance_1.png', rfc_pipe.steps[2][1].feature_importances_, X_test.columns.values.tolist())
    # do_plotImportantF('feature_importance.png', feature_imp, X_test.columns.values.tolist()) 

    # svm, svm_pipe = fd_training.do_PipelineOnly("svm", svm.SVC(), X_train, y_train)
    # fd_training.do_evaluateModel(svm_pipe, X_test, y_test)
    # fd_training.do_plotEvaluationCurves('svm', svm, X_test, y_test, 'BuGn')
    # fd_eda.do_saveModel('svm.pkl', svm, 'p') 
