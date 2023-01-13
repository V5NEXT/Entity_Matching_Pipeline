# Entity_Matching_Pipeline - Task 1
Members :
Angela Agoston,
Vishnu Viswambharan <br/><br/><br/>
[Task 01] File Task1.1.py <br/><br/>
1)Reading the CSV files, performing basic descriptions of each attribute, ploting them, counting unique occurences etc.
Here we observe, that contain schema ACM contains less records than schema DBLP. So, intuitvely, we make an assumption that we should make the ACM fixed as a benchmark, 
and look for its matches.
Then, utilizing special python librares for each step of the following, preproccesing/normalization pipeline: lowering the strings, removing numbers from strings, 
removing special characthers from strings, performing stemming, removing stopwords, removing possible HTML tags.
After the preprocessing is done, a neccesary transformation is done, which will later help in performing blocking.
By simple observing the similiarties between 'venue' attribute values from both schemas, we see that a transforming can be done. 
The pairing between the two value sets can be done based on solemnly Levenstein similarity, by ordering the similarities, fixing and connecting the ACM venue values with the DBLP 
venue values. In such a way, that firstly a match with the highest Levenstein similarity is made. This yields in having a first pair of the two venues which are declared as the same,
the DBLP value for this pair is then removed as a potential candidate for the next matches(pairs) with the ACM venue values. Then, a match with second highest similarity is taken.
The DBLP value for this pair is then also removed as a potential candidate for the next matches. In the end, this results, in having 5 pairs, which can be placed in a dicitonary,
used to rename the venue values from one column to the venue values of the other column, and viceversa. 
<br/>
<br/>

2) File Task1.2.py
By having  columnn 'venue' of one schema renamed to the values of the other one, an efficent blocking scheme can be implemented  based on  having exactly the same 
values on attributes: 'venue', 'year'. This is a sort of a hybrid blocking, because, essentialy it is based on a string distance between the values(venues)-
before the renaming, combined with exact values(year). A pair (A,B) from a block, represents, a record A from ACM, and a record B from DBLP, which have the same venue, 
and year as mentioned.

   <br/>
   <br/>
3) File Task1.3.py
For each pair in the block, a scoring is performed. By being in the same block, for venue, year they get score 1. For attributes author, title, similarity based on 
jaro-winkler distance is calculated. Filtering is done, by a condition, that a pair in the block is a possible match if its overall sum of scores on author, title
is greater than 1.60. Some records from ACM can have multiple potential candidates from DBLP, even after the filtering. However, a record which has the heighest score is choosen afterall.
It is important to note here, that this pairs represent a MultiIndex made by combining indices from both tables. 

In order to perform any evaluation of our perfect matches, PerfectMapping.csv needs to be transformed into pairs. This is done by exploiting the uniqueness of both ids.
Once again, a dictionary can be created used for transforming the (idACM,idDBLP) pairs into pairs ( index of idACM from ACM, index of idDBLP from DBLP).





# Entity_Matching_Pipeline - Task 2 SVM Prediction
 


Task 2.1 (File : Task2.py)
After succesfully creating the dataset the next task was to find a suitable machine learning model to train and predict based on perfect Mapping



Task 2.2 (File : Task2.py)
We tried between Logistics Regression and SVM. Since SVM yielded better results for us we decided to move ahead with SVM

Base Model Creation
SVM() : Function which uses sklearn's sklearn.svm funstion to train the model

We achieved 0.9687713699566901

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
F1_SCORE_FINAL 1.0

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