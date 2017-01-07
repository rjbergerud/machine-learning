# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

## Preparing the Data
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

## Test Train Split
# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split
# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
splitted = train_test_split(X_all, y_all, train_size=num_train, random_state=1)
X_train = splitted[0]
X_test = splitted[1]
y_train = splitted[2]
y_test = splitted[3]

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import make_scorer, f1_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# TODO: Create the parameters list you wish to tune
parameters = {"base_estimator__criterion" : ["gini"],
              "base_estimator__splitter" :   ["best"],
              "base_estimator__max_features" : [ None ],
              "base_estimator__max_depth" : [ None ],
              "n_estimators": [1, 2, 5, 10, 50]
             }
#Just to prove that I'm not messing with any default params here
parameters = {}
adaboost_parameters = {}

clf_base_decisiontree_test_scores = []
clf_base_default_test_scores = []
for x in range(0,100):

    # TODO: Initialize the classifier

    # TODO: Make an f1 scoring function using 'make_scorer' 

    # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
    grid_obj = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), cv=StratifiedShuffleSplit(y_train, test_size=0.2, random_state=2), param_grid=parameters)
    grid_obj_default_base_clf = GridSearchCV(AdaBoostClassifier(), cv=StratifiedShuffleSplit(y_train, test_size=0.2, random_state=2), param_grid = adaboost_parameters)
    
    # TODO: Fit the grid search object to the training data and find the optimal parameters
    grid_obj.fit(X_train, y_train)
    grid_obj_default_base_clf.fit(X_train, y_train)
    
    # Get the estimator
    clf = grid_obj.best_estimator_
    clf_default_base_clf = grid_obj_default_base_clf.best_estimator_
    
    #print grid_obj.grid_scores_
    #print grid_obj_default_base_clf.grid_scores_
    
    clf_base_decisiontree_test_scores.append(float(clf.score(X_test, y_test)))
    clf_base_default_test_scores.append(float(clf_default_base_clf.score(X_test, y_test)))
    

print "Mean of adaboost with base_estimator set to DecisionTreeClassifier", np.array(clf_base_decisiontree_test_scores).mean()
print "Std of adaboost with base_estimator set to DecisionTreeClassifier", np.array(clf_base_decisiontree_test_scores).std()
print " ===="
print "Mean of adaboost with base_estimator left as default", np.array(clf_base_default_test_scores).mean()
print "Std of adaboost with base_estimator left as default", np.array(clf_base_default_test_scores).std()
