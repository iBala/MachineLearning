#!/usr/local/bin/python

###################################
#Introduction to Machine Learning
#This script estimates the price of each house in Iowa depending on various paramreters
#Place train.csv and test.csv in the same directory as this script
#Influenced by https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/notebook
#(c) p.balakumaran@gmail.com
###################################

#Basic Modules
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('whitegrid')

#Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Scipy
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
Y_train = train.SalePrice

#Linear Regression
from sklearn import linear_model
linreg = linear_model.LinearRegression()
linreg.fit(X_train,Y_train)
print "LinReg Score is "+str(linreg.score(X_train,Y_train))
Y_test=linreg.predict(X_test)

#KNearestRegressors
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train,Y_train)
print "KNN Score is "+str(knn.score(X_train,Y_train))

#Lasso CV - This model focuses on reducing RMSE
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y_train)
print rmse_cv(model_lasso).mean()

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
#plt.show()

Y_pred = model_lasso.predict(X_test)
Y_pred = np.expm1(model_lasso.predict(X_test))
print Y_pred

submission = pd.DataFrame({"Id": test['Id'],"Price": Y_pred})
submission.to_csv("submission.csv",index=False)
