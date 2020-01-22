#importing basic packages
import pandas as pd
import numpy as pn
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting= 3)

#implementing data cleaning and data pre-processing using nltk
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i])
    review = review.lower() 
    review = review.split()
    ps=PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#implemnting bag of words algorithm for word embedding
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#splitting data for testing and training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)

#using NaiveBayes a methodology for classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#displaying confusion matrix to see the no of correct predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
(91+55)/200