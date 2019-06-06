import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('merged_B1_ISLAB.csv')

#DATA PREPROCESSING
dataset = dataset.drop(dataset.columns[[0, 6, 7,13,14]], axis=1) 
'''del dataset['appName']
del dataset['startDateTime']
del dataset['stopDateTime']
del dataset['destinationTCPFlagsDescription']
del dataset['sourceTCPFlagsDescription']'''

dataset = dataset.drop(dataset.index[[192041,375665]])

X = dataset.iloc[:,0:10].values
Y = dataset.iloc[:,10].values

for i in range(X.shape[0]):
    X[i,8] = int(X[i,8].replace('.',''))
    X[i,5] = int(X[i,5].replace('.',''))

for i in range(X.shape[0]):
    X[i,9] = int(X[i,9])
    
X[:,4]



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label = LabelEncoder()
label2 = LabelEncoder()
X[:,4] = label.fit_transform(X[:,4])
X[:,6] = label2.fit_transform(X[:,6])
ohe = OneHotEncoder(categorical_features = [4])
X = ohe.fit_transform(X).toarray()
ohe1 = OneHotEncoder(categorical_features = [9])
X = ohe1.fit_transform(X).toarray()

label3 = LabelEncoder()
Y = label3.fit_transform(Y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#RFC
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 9)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)'''
'''
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 6)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
'''
'''
#Backward Elimination
X = X[:,1:]
X = np.delete(X, 3, 1)
import statsmodels.formula.api as sm
x_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regr = sm.OLS(endog = Y,exog =x_opt ).fit()
regr.summary()
x_opt = X[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regr = sm.OLS(endog = Y,exog =x_opt ).fit()
regr.summary()

x_opt = X[:, [2,3,4,5,6,7,8,9,10,11,12,13,14]]
regr = sm.OLS(endog = Y,exog =x_opt ).fit()
regr.summary()

from sklearn.cross_validation import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_opt, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train1, y_train1)

y_pred2 = classifier2.predict(X_test1)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test1, y_pred2)

'''
'''
x_opt = X[:, [3,4,5,6,7,8,9,10,11,12,13,14]]
regr = sm.OLS(endog = Y,exog =x_opt ).fit()
regr.summary()
x_opt = X[:, [3,4,6,7,8,9,10,11,12,13,14]]
regr = sm.OLS(endog = Y,exog =x_opt ).fit()
regr.summary()

x_opt = X[:, [3,4,6,8,9,10,11,12,13,14]]
regr = sm.OLS(endog = Y,exog =x_opt ).fit()
regr.summary()
'''