
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_prep = __import__('Task1_1')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)
    
print(df_DBLP)
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
   print(len(y_test_1.tolist()),len(y_test_0.tolist()))
   print(X_train_0.shape, X_train_1.shape)
   print(y_train_0.shape, y_train_1.shape)
   return X_train_0,X_test_0,y_train_0,y_test_0, X_train_1,X_test_1,y_train_1,y_test_1
X_train_0,X_test_0,y_train_0,y_test_0, X_train_1,X_test_1,y_train_1,y_test_1= split_dataset_classification()

X_train = np.concatenate([X_train_0, X_train_1], axis=0)
X_test = np.concatenate([X_test_0, X_test_1], axis=0)
y_train = np.concatenate([y_train_0, y_train_1], axis=0)
y_test = np.concatenate([y_test_0, y_test_1], axis=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



def logistika():
    model = LogisticRegression() 
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
    lst_accu_stratified = []
    for train_index, test_index in skf.split(X, y): 
        X_train_fold, X_test_fold = X[train_index], X[test_index] 
        y_train_fold, y_test_fold = y[train_index], y[test_index] 
        model.fit(X_train_fold, y_train_fold) 
        lst_accu_stratified.append(model.score(X_test_fold, y_test_fold))


def LogisticRegressionClassifier():
    logreg = rl.LogisticRegressionClassifier()
    X_train, X_test, Y_train, Y_test = split_dataset_classification()
    # golden_pairs = features
    # golden_matches_index = golden_pairs.index.intersection(links_true)
    # print(golden_matches_index)

    logreg.fit(X_train, Y_train)
    print("Intercept: ", logreg.intercept)
    print("Coefficients: ", logreg.coefficients)

    result_logreg = logreg.predict(features_all)

    print(result_logreg)

    rl.confusion_matrix(links_true, result_logreg, len(features_all))

    rl.fscore(links_true, result_logreg)

def svm():
    svm = SVC()
    clf = SVC(random_state=0)
    svm.fit(X_train, y_train)
    result_svm = svm.predict(X_test)
    score= f1_score(y_test, result_svm)
    SVC(random_state=0)
    cm = confusion_matrix(y_test, result_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
    return  score
# result_svm=svm()
# print(result_svm)



# Since SVM have better results

def hyperparametertuning():
# defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    result_svm = svm.predict(X_test)
    print(result_svm)



hyperparametertuning()
