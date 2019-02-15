from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
by_sex_class = df.groupby(['Sex','Pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.mean())

# Impute Age and assign to df.Age
df.Age = by_sex_class.Age.transform(impute_median)
df = pd.get_dummies(df)
#print(df.head())
#print(df.shape)
#print(df.info())

#df.hist()
#scatter_matrix(df)
#plt.show()

X = df.drop('Survived',axis=1)
Y = df['Survived']

#knn = KNeighborsClassifier(n_neighbors=8)
#knn.fit(X,Y)

reg_all = GaussianNB()
reg_all.fit(X,Y)

df2 = pd.read_csv('test.csv')
#print(df2.shape)
df2 = df2.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

by_sex_class_2 = df2.groupby(['Sex','Pclass'])
# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.mean())

# Impute Age and assign to df.Age
df2.Age = by_sex_class_2.Age.transform(impute_median)

meadian_value=df2['Fare'].median()
df2['Fare']=df2['Fare'].fillna(meadian_value)

print(df2.shape)
df2 = pd.get_dummies(df2)
#print(df2.head())
#print(df2.info())

y_pred = reg_all.predict(df2)
print("Test set predictions:\n {}".format(y_pred))
#print(len(y_pred))

df3 = pd.read_csv('test.csv')
df3 = df3['PassengerId']
#print(len(list(df3)))

new = {'PassengerId':list(df3), 'Survived':y_pred}
df4 = pd.DataFrame(new)
print(df4.tail(10))

df4.to_csv('Submission.csv', index= False)