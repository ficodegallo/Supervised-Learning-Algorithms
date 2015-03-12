
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt

from sklearn import datasets
# our support vector machine classifier!
from sklearn import svm

# load digits dataset
digits = datasets.load_digits()


# In[3]:

# let's see what this dataset is all about
print digits.DESCR


# In[15]:

# let's see what keys this digits object has
print digits.keys()

# Let's see what classes we want to put digits into
print digits.target_names


# In[7]:

# Let's pic one and see what it looks like
first_digit = digits.data[0]
print first_digit

# But what do all of those columns mean!?!?! 
# Let's go to the whiteboard and find out...
# in each quadrant how many pixels/boxes are filled in (turned on)


# In[5]:

# let's see what the first digit was tagged as
print digits.target[0]


# In[6]:

# now let's see what digits.images[0] is
# it's just the first_digit array reshaped as an 8 x 8 matrix
print digits.images[0]


# In[102]:

# let's instantiate our Support Vector Classifier
# C is the error threshold
# Gamma is a tuning parameter related to the gradient descent by... 
# controling the speed / distance at which we step through gradient descent
classifier = svm.SVC(gamma=.0001, C=100)

# how many data points do we have?
print len(digits.data)


# In[113]:

# let's create a training set of the first 1597 of the 1797 data points
x_training, y_training = digits.data[:-97], digits.target[:-97]

# now let's train the classifier on the training data
classifier.fit(x_training, y_training)


# In[114]:

print "Prediction {}".format(classifier.predict(digits.data[1590]))
print digits.target[1590]


# In[115]:

# show the image
plt.imshow(
    digits.images[1600],
    cmap=plt.cm.gray_r,
    interpolation="nearest"
)
plt.show()


# In[116]:

# Wow!!!! Let's see what the accuracy is 
# Let's go from digits.data[-200] all the way to digits.data[-1]
correct = 0
indices = range(-200, 0)
for i in indices:
    # if we were correct
    if classifier.predict(digits.data[i])[0] == digits.target[i]:
        correct += 1
accuracy = float(correct) / len(indices)
        
print "Accuracy: {}".format(accuracy)


# In[117]:

# Let's try using kNN
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=25)
# train the classifier
knn_clf.fit(x_training, y_training)

# Let's see what the accuracy is 
# Let's go from digits.data[-200] all the way to digits.data[-1]
correct = 0
indices = range(-200, 0)
for i in indices:
    # if we were correct
    if knn_clf.predict(digits.data[i])[0] == digits.target[i]:
        correct += 1
accuracy = float(correct) / len(indices)
        
print "Accuracy: {}".format(accuracy)


# In[118]:

# Let's try with Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_training, y_training)

# Let's see what the accuracy is 
# Let's go from digits.data[-200] all the way to digits.data[-1]
correct = 0
indices = range(-200, 0)
for i in indices:
    # if we were correct
    if log_reg.predict(digits.data[i])[0] == digits.target[i]:
        correct += 1
accuracy = float(correct) / len(indices)
        
print "Accuracy: {}".format(accuracy)


# In[ ]:



