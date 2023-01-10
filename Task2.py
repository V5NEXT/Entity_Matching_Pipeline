
from sklearn.model_selection import train_test_split
import recordlinkage
import recordlinkage as rl
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from recordlinkage.index import Full
from sklearn.preprocessing import MultiLabelBinarizer



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
    
    
    feature_scoring
df_DBLP, df_ACM = data_prep.preprocessing()
f_score,links_true = feature_scoring.evaluation()

def all_pairs():
    indexer = recordlinkage.Index()
    indexer.add(Full())
    candidate_links_all = indexer.index(df_ACM, df_DBLP)

    return candidate_links_all

candidate_links_all = all_pairs()
# features = feature_scoring.feature_scores()
features_all = feature_scoring.feature_scores(candidate_links_all)
def split_dataset_classification():
    compare_cl = recordlinkage.Compare()

    X_train, X_test = train_test_split(features_all, test_size=0.2)
    Y_train, Y_test = train_test_split(links_true, test_size=0.2)


    # features_test_all = compare_cl.compute(Y_test, df_ACM, df_DBLP)
    # features_test_all

    return X_train,X_test,Y_train,Y_test

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
    svm = rl.SVMClassifier()
    X_train, X_test, Y_train, Y_test = split_dataset_classification()

    # golden_pairs = features
    # golden_matches_index = golden_pairs.index.intersection(links_true)

    svm.fit(X_train, Y_train)
    result_svm = svm.predict(features_all)

    rl.confusion_matrix(links_true, result_svm, len(features_all))

    rl.fscore(links_true, result_svm)


# Since SVM have better results

def hyperparametertuning():
# defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    X_train, X_test, Y_train, Y_test = split_dataset_classification()
    X_train = MultiLabelBinarizer().fit_transform(X_train)
    Y_train = MultiLabelBinarizer().fit_transform(Y_train)

# fitting the model for grid search
    grid.fit(X_train, Y_train)


    result_svm = svm.predict(features_all)
    print(result_svm)


svm()