
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import recordlinkage
import recordlinkage as rl
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from recordlinkage.index import Full
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report


data_prep = __import__('Task1_1')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)
    
feature_scoring= __import__('Task1_3')
try:
    attrlist = feature_scoring.__all__
except AttributeError:
    attrlist = dir(feature_scoring)
for attr in attrlist:
    globals()[attr] = getattr(feature_scoring, attr)
    


# print(features)
features['label'] = 0
for u in links_pred.index.tolist():
    features.loc[u, 'label']=1
# print(features)
# print(features['label'].value_counts())

label_0= features[features['label'] == 0]
# print(label_0)
label_1= features[features['label'] == 1]
# print(label_1)
label_0=label_0.to_numpy()
label_1=label_1.to_numpy()
# print(label_0[:, :-1])
# print(label_0)

# def all_pairs():
#     indexer = recordlinkage.Index()
#     indexer.add(Full())
#     candidate_links_all = indexer.index(df_ACM, df_DBLP)

#     return candidate_links_all

# candidate_links_all = all_pairs()

# features_all = feature_scoring.feature_scores(candidate_links_all)

def split_dataset_classification():

  
   X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(label_1[:, :-1], label_1[:, -1], test_size=0.30, random_state=42)
   k=len(label_1[:, -1].tolist())
   indices = np.random.choice(label_0.shape[0], k, replace=False)
   label_0_new=label_0[indices]
  
   X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(label_0_new[:, :-1], label_0_new[:, -1], test_size=0.30, random_state=42)
   return X_train_0,X_test_0,y_train_0,y_test_0, X_train_1,X_test_1,y_train_1,y_test_1
X_train_0,X_test_0,y_train_0,y_test_0, X_train_1,X_test_1,y_train_1,y_test_1= split_dataset_classification()

X_train = np.concatenate([X_train_0, X_train_1], axis=0)
X_test = np.concatenate([X_test_0, X_test_1], axis=0)
y_train = np.concatenate([y_train_0, y_train_1], axis=0)
y_test = np.concatenate([y_test_0, y_test_1], axis=0)




def LogisticRegressionClassifier():
    logreg = rl.LogisticRegressionClassifier()

    logreg.fit(X_train, y_train)
    print("Intercept: ", logreg.intercept)
    print("Coefficients: ", logreg.coefficients)

    result_logreg = logreg.predict(X_test)

    print(result_logreg)


    rl.fscore(y_test, result_logreg)

# Base SVM Model
#   returns : f1_score
#   displays correlation Plot

def svm():
    svm = SVC()
    clf = SVC(random_state=0)
    svm.fit(X_train, y_train)
    result_svm = svm.predict(X_test)
    score= f1_score(y_test, result_svm)
    SVC(random_state=0)
    cm = confusion_matrix(y_test, result_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return  score




# Since SVM have better results

def hyperparametertuning():
# defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    start = time.time()

    grid.fit(X_train, y_train)
    end = time.time()
    # show the grid search information
    print("[INFO] grid search took {:.2f} seconds".format(
        end - start))
    print("[INFO] grid search best score: {:.2f}%".format(
        grid.best_score_ * 100))
    print("[INFO] grid search best parameters: {}".format(
        grid.best_params_))
    model = grid.best_estimator_
    print("[INFO] evaluating...")

    predictions = model.predict(X_test)
    print(predictions)


    score = f1_score(y_test, predictions)

    print("F1_SCORE_FINAL", score)
    return model


df_x = np.concatenate([X_train, X_test], axis=0)
df_y = np.concatenate([y_train, y_test], axis=0)
def crossvalidaion():
    kf = KFold(n_splits=10, shuffle=True)

    acc_arr = np.empty((10, 1))
    f1_arr = np.empty((10, 1))
    cnf_arr = []
    x = 0

    model = hyperparametertuning()
    for train_index, test_index in kf.split(df_x, df_y):


        X_train, X_test = df_x[train_index], df_x[test_index]
        y_train, y_test = df_y[train_index], df_y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
        print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
        print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
        print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
        print("\n", classification_report(y_pred, y_test))

        cnf_matrix = confusion_matrix(y_test, y_pred)
        acc_arr[x] = accuracy_score(y_test, y_pred)
        f1_arr[x] = f1_score(y_test, y_pred)

        x = x + 1

    print("%0.2f f1 score with a standard deviation of %0.2f" %
          (f1_arr.mean(), f1_arr.std()))
    print("%0.2f accuracy with a standard deviation of %0.2f" %
          (acc_arr.mean(), acc_arr.std()))

crossvalidaion()