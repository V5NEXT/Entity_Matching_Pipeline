



Task 2.1 (File : Task2.py)
After succesfully creating the dataset the next task was to find a suitable machine learning model to train and predict based on perfect Mapping



Task 2.2 (File : Task2.py)
We tried between Logistics Regression and SVM. Since SVM yielded better results for us we decided to move ahead with SVM

Base Model Creation
SVM() : Function which uses sklearn's sklearn.svm funstion to train the model

We achieved :

Without Preprocessing : 0.9224880382775119 (Correlation Plot :  Plots/Without_Preprocess.png)
With Preprocessing : 0.9687713699566901  (Correlation Plot :  Plots/After_Preprocess.png)


You can replicate the results by just running the function svm()

Task 2.3 (File : Task2.py)

Hyper-Parameter Tuning
hyperparametertuning_SVM() : Function Uses GridSearchCV from sklearn to try out different combination of C-value, Gamma and kernal to find the best
possible combination.The values used are shown below:

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}

We found out that :

# grid search took 7.31 seconds
# grid search best score: 99.93%
# grid search best parameters: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

And therfore final f-score before cross validation is 1
[0. 0. 0. ... 1. 1. 1.]
F1_SCORE_FINAL
 With Pre-processing : 1.0
 Without Pre-processing : 0.9991489361702128


You can replicate the following results by running the function hyperparametertuning_SVM()

Cross-Validation : Inorder to check for overfitting we did cross validation using sklearn's K-Fold mechanism

Function used : crossvalidation()

We achieved the following results :

Accuracy Score: 1.0000
SVC f1-score  : 1.0000
SVC precision : 1.0000
SVC recall    : 1.0000

               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00       201
         1.0       1.00      1.00      1.00       231

    accuracy                           1.00       432
   macro avg       1.00      1.00      1.00       432
weighted avg       1.00      1.00      1.00       432

1.00 f1 score with a standard deviation of 0.00
1.00 accuracy with a standard deviation of 0.00

This prove that our tuned model was not overfitted and was predicting with a 100% accuracy.

You can recreate the results by running the file crossvalidation()