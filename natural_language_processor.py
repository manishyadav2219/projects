import nltk
nltk.download('stopwords')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t' , quoting = 3)

import re
from nltk .corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =0)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)

y_ = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,f1_score
cm = confusion_matrix(y_test,y_)
print('f1_score : %.2f',f1_score(y_test,y_))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state =0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cmt = confusion_matrix(y_test,y_pred)
print('f1_score : %.2f',f1_score(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 20 , criterion ='entropy' , random_state =0)
rfc.fit(X_train,y_train)

y1_ = rfc.predict(X_test)

cmrfc = confusion_matrix(y_test,y1_)
print('f1_score : %.2f',f1_score(y_test,y1_))