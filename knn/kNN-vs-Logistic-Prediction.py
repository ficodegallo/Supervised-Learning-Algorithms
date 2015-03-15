
# coding: utf-8

# In[1]:

import csv
import pandas as pd
import pylab as pl
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
 
df = pd.read_csv('https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/bank-additional-full.csv', delimiter=';')
 
# This creates an array of Trues and False, uniformly distributed such that
# around 30% of the items will be True and the rest will be False
test_idx = np.random.uniform(0, 1, len(df)) <= 0.3

## Run all transformations on the dataset FIRST

JOB_MAP = {
"admin." : 2,
"blue-collar" : 1, 
"entrepreneur" : 1, 
"housemaid" : 1, 
"management" : 3, 
"retired" : 0, 
"self-employed": 0, 
"services" : 2, 
"student" : 0, 
"technician" : 3, 
"unemployed" : 0, 
"unknown" : 0, }

df["job"] = df["job"].apply(lambda value: JOB_MAP.get(value))

MARITAL_MAP = {
"divorced" : 1, 
"married" : 2, 
"single" : 1, 
"unknown" : 0 }
 
df["marital"] = df["marital"].apply(lambda value: MARITAL_MAP.get(value))
 
EDUCATIONAL_MAP = {
"basic.4y" : 1, 
"basic.6y" : 2, 
"basic.9y" : 3, 
"high.school" : 0, 
"illiterate" : 0, 
"professional.course" : 1, 
"university.degree" : 1, 
"unknown" : 0 }
 
df["education"] = df["education"].apply(lambda value: EDUCATIONAL_MAP.get(value))
 
DEFAULT_MAP = {
"no" : 1, 
"yes" : 2, 
"unknown" : 0 }
 
df["default"] = df["default"].apply(lambda value: DEFAULT_MAP.get(value))

## CREATE MAPPING OUT OUTCOMES
OUTCOME_MAP = {
    "no" : 0,
    "yes" : 1
}
df["y"] = df["y"].apply(lambda value: OUTCOME_MAP.get(value))

HOUSING_MAP = {
"no" : 1, 
"yes" : 2, 
"unknown" : 0 
}
 
df["housing"] = df["housing"].apply(lambda value: HOUSING_MAP.get(value))

LOAN_MAP = {
"no" : 1, 
"yes" : 2, 
"unknown" : 0 
}
 
df["loan"] = df["loan"].apply(lambda value: LOAN_MAP.get(value))

CONTACT_MAP = {
"cellular" : 1, 
"telephone" : 2 
}
 
df["contact"] = df["contact"].apply(lambda value: CONTACT_MAP.get(value))

MONTH_MAP = {
"jan" : 1, 
"feb" : 2,
"mar" : 3, 
"apr" : 4,
"may" : 5,
"jun" : 6,
"jul" : 7,
"aug" : 8,
"sep" : 9,
"oct" : 10,
"nov" : 11,
"dec" : 12
}
 
df["month"] = df["month"].apply(lambda value: MONTH_MAP.get(value))

DAY_OF_WEEK_MAP = {
"mon" : 1, 
"tue" : 2,
"wed" : 3, 
"thu" : 4,
"fri" : 5
}
 
df["day_of_week"] = df["day_of_week"].apply(lambda value: DAY_OF_WEEK_MAP.get(value))

POUTCOME_MAP = {
"nonexistent" : 0,
"failure" : 1,
"success" : 2
}
 
df["poutcome"] = df["poutcome"].apply(lambda value: POUTCOME_MAP.get(value))

df.describe



# In[3]:

## After transformations, split the set into a training and test
# The training set will be ~30% of the data
train = df[test_idx==True]
# The test set will be the remaining, ~70% of the data
test = df[test_idx==False]

features = ["age", "job", "marital", "education", "default"]

results = []
# range(1, 51, 2) = [1, 3, 5, 7, ...., 49]
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    # train the classifier
    clf.fit(train[features], train["y"])
    # then make the predictions
    preds = clf.predict(test[features])
    # very simple and terse line of code that will check the accuracy
    # documentation on what np.where does: http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    # Here is a simple example: suppose our predictions where [True, False, True] and the correct values were [True, True, True]
    # The next line says, create an array where when the prediction = correct value, the value is 1, and if not the value is 0.
    # So the np.where would, in this example, produce [1, 0, 1] which would be summed to be 2 and then divided by 3.0 to get 66% accuracy
    accuracy = np.where(preds==test["y"], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
 
    results.append([n, accuracy])
 
results = pd.DataFrame(results, columns=["n", "accuracy"])
 
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()
 


# In[4]:

# ****** Now, let's see how accurate the predictor is ******
results = []
# let's try two different weighting schemes, one where we don't worry about the distance
# another where we weight each point by 1/distance
for w in ['uniform', 'distance', lambda x: np.log(x)]:
    clf = KNeighborsClassifier(7, weights=w)
    w = str(w)
    clf.fit(train[features], train['y'])
    preds = clf.predict(test[features])

    # For an explanation of this line, refer to my explanation of this same line above
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    print "Weights: %s, Accuracy: %3f" % (w, accuracy)

    results.append([w, accuracy])

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results


# In[9]:

## After transformations, split the set into a training and test
# The training set will be ~30% of the data
train = df[test_idx==True]
# The test set will be the remaining, ~70% of the data
test = df[test_idx==False]

features2 = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", 
            "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
            "euribor3m", "nr.employed"]

results = []
# range(1, 51, 2) = [1, 3, 5, 7, ...., 49]
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    # train the classifier
    clf.fit(train[features2], train["y"])
    # then make the predictions
    preds = clf.predict(test[features2])
    # very simple and terse line of code that will check the accuracy
    # documentation on what np.where does: http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    # Here is a simple example: suppose our predictions where [True, False, True] and the correct values were [True, True, True]
    # The next line says, create an array where when the prediction = correct value, the value is 1, and if not the value is 0.
    # So the np.where would, in this example, produce [1, 0, 1] which would be summed to be 2 and then divided by 3.0 to get 66% accuracy
    accuracy = np.where(preds==test["y"], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
 
    results.append([n, accuracy])
 
results = pd.DataFrame(results, columns=["n", "accuracy"])
 
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()


# In[10]:

# ****** Now, let's see how accurate the predictor is ******
results = []
# let's try two different weighting schemes, one where we don't worry about the distance
# another where we weight each point by 1/distance
for w in ['uniform', 'distance', lambda x: np.log(x)]:
    clf = KNeighborsClassifier(7, weights=w)
    w = str(w)
    clf.fit(train[features2], train['y'])
    preds = clf.predict(test[features2])

    # For an explanation of this line, refer to my explanation of this same line above
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    print "Weights: %s, Accuracy: %3f" % (w, accuracy)

    results.append([w, accuracy])

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results


# In[11]:

## After transformations, split the set into a training and test
# The training set will be ~30% of the data
train = df[test_idx==True]
# The test set will be the remaining, ~70% of the data
test = df[test_idx==False]

features3 = ["housing", "loan", "contact", "campaign", "pdays", "previous", "poutcome"]

results = []
# range(1, 51, 2) = [1, 3, 5, 7, ...., 49]
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    # train the classifier
    clf.fit(train[features3], train["y"])
    # then make the predictions
    preds = clf.predict(test[features3])
    # very simple and terse line of code that will check the accuracy
    # documentation on what np.where does: http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    # Here is a simple example: suppose our predictions where [True, False, True] and the correct values were [True, True, True]
    # The next line says, create an array where when the prediction = correct value, the value is 1, and if not the value is 0.
    # So the np.where would, in this example, produce [1, 0, 1] which would be summed to be 2 and then divided by 3.0 to get 66% accuracy
    accuracy = np.where(preds==test["y"], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
 
    results.append([n, accuracy])
 
results = pd.DataFrame(results, columns=["n", "accuracy"])
 
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()


# In[13]:

# ****** Now, let's see how accurate the predictor is ******
results = []
# let's try two different weighting schemes, one where we don't worry about the distance
# another where we weight each point by 1/distance
for w in ['uniform', 'distance', lambda x: np.log(x)]:
    clf = KNeighborsClassifier(5, weights=w)
    w = str(w)
    clf.fit(train[features3], train['y'])
    preds = clf.predict(test[features3])

    # For an explanation of this line, refer to my explanation of this same line above
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    print "Weights: %s, Accuracy: %3f" % (w, accuracy)

    results.append([w, accuracy])

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results


# In[18]:

# show histogram of the data
#df.hist()
#pl.show()

# frequency table that cuts campaign and the outcome of the previous campaign with
# whether or not someone was subscribed to a term deposit
#print pd.crosstab(df['campaign'], df['poutcome'], rownames=['y'])


df['intercept'] = 1.0
df.head()
training_columns = df.columns[0:15]
logit = sm.Logit(df["y"], df[training_columns])
result = logit.fit()
print result.summary()


# In[24]:

def predict(age, job, marital, education, default):
    """
    Outputs predicted probability of the outcome of the campaign
    given age, job, marital, education and default variables of the 
    potential consumer
    """
    return result.predict([age, job, marital, education, default])[0]

print "\nPrediction for Age: 35, Job: Management, Marital: Single, Education: College Degree, Credit Default: No is..."
print predict(35, 3, 1, 1, 1)


# In[17]:

def predict(housing, loan, contact, month, day_of_week, 
            duration, campaign, pdays, previous, poutcome):
    """
    Outputs predicted probability of the outcome of the campaign
    given age, job, marital, education and default variables of the 
    potential consumer
    """
    return result.predict([housing, loan, contact, month, day_of_week, 
            duration, campaign, pdays, previous, poutcome])[0]

print "\nPrediction for this subset of features..."
print predict(3, 2, 1, 4, 3, 195, 1, 999, 0, 0)


# In[8]:

def predict(age, job, marital, education, default, housing, loan, contact, month, day_of_week, 
            duration, campaign, pdays, previous, poutcome):
    """
    Outputs predicted probability of the outcome of the campaign
    given age, job, marital, education and default variables of the 
    potential consumer
    """
    return result.predict([age, job, marital, education, default, housing, loan, contact, month, day_of_week, 
            duration, campaign, pdays, previous, poutcome])[0]

print "\nPrediction for all features..."
print predict(35, 3, 1, 1, 1, 3, 2, 1, 4, 3, 195, 1, 999, 0, 0)


# In[ ]:



