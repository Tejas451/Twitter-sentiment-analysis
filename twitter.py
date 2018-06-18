# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:31:21 2018

@author: tejas
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re, urllib.request

# Importing the dataset
train = pd.read_csv('train.csv',encoding='latin-1')
test = pd.read_csv('test.csv',encoding='latin-1')
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
corpus1 = []
for i in range(0, len(train.index)):
   # train['Sentiment'][i] = train['Sentiment'][i].strip().decode('utf-8')

    review = re.sub('[^a-zA-Z#@]', ' ', train['SentimentText'][i])
    
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

for i in range(0, len(test.index)):
   # train['Sentiment'][i] = train['Sentiment'][i].strip().decode('utf-8')

    review1 = re.sub('[^a-zA-Z#@]', ' ', test['SentimentText'][i])
    
    review1 = review1.lower()
    review1 = review1.split()
    ps = PorterStemmer()
    review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
    review1 = ' '.join(review1)
    corpus1.append(review1)
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 200)
X = cv.fit_transform(corpus).toarray()
X_test = cv.fit_transform(corpus1).toarray()
y = train.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

# Predicting the Test set results
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#m = confusion_matrix(y_test, y_pred)
sub1=pd.DataFrame(y_pred3,columns=['Sentiment'])

submission1=pd.concat([test['ItemID'], sub1], axis=1)
submission1.to_csv("submission1.csv",index=False)
