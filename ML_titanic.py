#Use ML to solve the titanic problem
#Problem statement: Probability of a type of traveller to have survived on Titanic

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Cabin','Fare','Embarked'], axis=1)
test_df    = test_df.drop(['Name','Ticket','Cabin','Fare','Embarked'], axis=1)
#titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
#print titanic_df.head(50)

#Create another category called 'child' for Sex. 
age_mean=titanic_df['Age'].mean()
titanic_df['Age'][np.isnan(titanic_df["Age"])]=age_mean
titanic_df['Age']=titanic_df['Age'].astype('int')
titanic_df.loc[titanic_df['Age']<=6, 'Sex']='child'

age_mean=test_df['Age'].mean()
test_df['Age'][np.isnan(test_df["Age"])]=age_mean
test_df['Age']=test_df['Age'].astype('int')
test_df.loc[test_df['Age']<=6, 'Sex']='child'

#All siblings, spouse, parents and child can be moved to a single boolean category
titanic_df['Family']=titanic_df['SibSp']+titanic_df['Parch']
titanic_df.loc[titanic_df['Family']>0,'Family']=1
titanic_df=titanic_df.drop(['SibSp','Parch','Age'],axis=1)

test_df['Family']=test_df['SibSp']+test_df['Parch']
test_df.loc[test_df['Family']>0,'Family']=1
test_df=test_df.drop(['SibSp','Parch','Age'],axis=1)

#Plot the survival rate of travellers according to sex
sex_survived=titanic_df.groupby('Sex').agg(['mean'])
sex_survived.reset_index().plot(x='Sex',y='Survived',kind='Bar')

#Represent Sex by Integers
titanic_df.loc[titanic_df['Sex']=="male",'Sex']=1
titanic_df.loc[titanic_df['Sex']=="female",'Sex']=2
titanic_df.loc[titanic_df['Sex']=="child",'Sex']=3

test_df.loc[test_df['Sex']=="male",'Sex']=1
test_df.loc[test_df['Sex']=="female",'Sex']=2
test_df.loc[test_df['Sex']=="child",'Sex']=3

Pclass_survived=titanic_df.groupby('Pclass').agg(['mean'])
Pclass_survived.reset_index().plot(x='Pclass',y='Survived',kind='Bar')

family_survived=titanic_df.groupby('Family').agg(['mean'])
family_survived.reset_index().plot(x='Family',y='Survived',kind='Bar')

#embark_survived=titanic_df.groupby('Embarked').agg(['mean'])
#embark_survived.reset_index().plot(x='Embarked',y='Survived',kind='Bar')

#plt.show()
X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"].astype(int)
X_test = test_df.drop(['PassengerId'],axis=1).copy()

#Run those awesome models
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print "logreg score is "+str(logreg.score(X_train, Y_train))

#SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print "svc Score is "+str(svc.score(X_train, Y_train))

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print "Random Forest Score is "+str(random_forest.score(X_train, Y_train))

submission = pd.DataFrame({"PassengerId": test_df['PassengerId'],"Survived": Y_pred})
submission.to_csv("~/Projects/Titanic/submission.csv",index=False)
