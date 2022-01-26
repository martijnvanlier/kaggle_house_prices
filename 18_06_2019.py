# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:20:12 2019

@author: Martijn
"""

#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingRegressor

import os
import math

import xgboost as xgb
from sklearn import svm



#Set working directory
os.chdir('C:\\Users\\Martijn\\Desktop\\Boston House Pricing')
os.getcwd()

#Import files
train = pd.read_csv('train.csv')

train.head()
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

#Function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

#Custom scoring function (RMSLE) for Cross-Validation 
from sklearn.metrics import make_scorer 
boston_housing = make_scorer(rmsle, greater_is_better=False)









#------------------------------------------------------
#EXPLORATORY DATA ANALYSIS
#------------------------------------------------------

#Take a first look at the data
train.head()
train.shape
test.shape
train.columns

#Check the number of missings
train.isnull().sum()
train.dtypes

#Count number of columns with missing values in train and test set
sum(train.isnull().sum() > 0)
sum(test.isnull().sum() > 0)

#Count total missing values in train and test set
sum(train.isnull().sum())
sum(test.isnull().sum())

#Test set has more missing values, so combine train/test before eventual imputing 
test['SalePrice'] = 0

train['Key'] = 'train'
test['Key'] = 'test'

complete = pd.concat([train, test])
complete = complete.reset_index(drop=True)

#Check again the number of missings in each column
complete.shape
complete.isnull().sum()
sum(complete.isnull().sum() > 0)

#Inspect distribution of target variable
%matplotlib auto 
sns.distplot(train['SalePrice']) #For inline: %matplotlib inline


#------------------------------------------------------------------------------
#IMPUTATIONS ON COMPLETE DATASET
#------------------------------------------------------------------------------

#1) Delete all columns with 60%+ missings
#2) Quick impute other missings

#Whole dataset contains 2919 observations, delete columns with 2000+ missings (only 4)
max_number_of_nas = 2000
complete = complete.loc[:, (complete.isnull().sum(axis=0) <= max_number_of_nas)]

#Loop through complete data and fill NAs based on datatype
for i in list(complete):
    
    #Check for NAs
    if complete[i].isnull().sum() > 0:
        
        #Check for datatype
        if complete[i].dtype == 'int64':
            complete[i] = complete[i].fillna(complete[i].mean())
        if complete[i].dtype == 'float64':
            complete[i] = complete[i].fillna(complete[i].mean())                
        if complete[i].dtype == 'object':
            complete[i] = complete[i].astype('category').cat.add_categories("imputed").fillna("imputed")
            
#Check if there are still any NAs.
sum(complete.isnull().sum() > 0)

#------------------------------------------------------------------------------
#MODELLING ON COMPLETE DATASET
#------------------------------------------------------------------------------        

#Dumify complete dataset first
complete = pd.get_dummies(complete, drop_first = True)  #Sparse = true generates errors after train/test splitting

#Then split into train and test again
complete_train = complete[complete['Key_train'] == 1]
complete_test = complete[complete['Key_train'] == 0]

#Delete 'Id' and 'Key' from complete_train
complete_train.drop(['Id', 'Key_train'], axis = 1, inplace = True)

#Split into X and y
labels = complete_train["SalePrice"]
predictors = complete_train.drop('SalePrice', axis = 1)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(predictors, labels, test_size=0.3, random_state=42)

#Define models
linreg = linear_model.LinearRegression()
rf = RandomForestRegressor()

ridgereg = linear_model.Ridge()
lassoreg = linear_model.Lasso()

svm = svm.SVR()
gboost = GradientBoostingRegressor()
xgboost = xgb.XGBRegressor()







model0 = linreg.fit(X_train, y_train)
model1 = rf.fit(X_train, y_train)

#View scores on test data
model0.score(X_test, y_test, scoring= boston_housing) #71% 
model1.score(X_test, y_test) #87% R^2 not to bad

#Calculate RMSLE on test data
y_pred_model0 = model0.predict(X_test)
y_pred_model1 = model1.predict(X_test)

rmsle(y_test, y_pred_model0) #0.1477 on test
rmsle(y_test, y_pred_model1) #0.1508 on test



#CROSS VALIDATION SCORES 

#With regular scoring function (R squared)
cross_val_score(linreg, predictors, labels, cv=10).mean() #0.66
cross_val_score(rf, predictors, labels, cv=10).mean() #0.83

cross_val_score(ridgereg, predictors, labels, cv=10).mean() #0.83
cross_val_score(lassoreg, predictors, labels, cv=10).mean() #0.68



#With custom scoring function (RMSLE)
cross_val_score(linreg, predictors, labels, cv=10, scoring = boston_housing).mean() #0.16
cross_val_score(rf, predictors, labels, cv=10, scoring = boston_housing).mean() #0.15

cross_val_score(ridgereg, predictors, labels, cv=10, scoring = boston_housing).mean() #0.167
cross_val_score(lassoreg, predictors, labels, cv=10, scoring = boston_housing).mean() #0.157

cross_val_score(svm, predictors, labels, cv=10, scoring = boston_housing).mean() #0.39
cross_val_score(gboost, predictors, labels, cv=10, scoring = boston_housing).mean() #0.129
cross_val_score(xgboost, predictors, labels, cv=10, scoring = boston_housing).mean() #0.13



#1) How to get cross validation in model1?
#2) Compare different models (measure time as well)
#3) Hyperparameter tuning: Gboost and Xgboost
#4) Stacking models
#5) Predict new results







#------------------------------------------------------------------------------
#HYPER PARAMETER TUNING 
#------------------------------------------------------------------------------

#Xgboost
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


xgboost.get_params

#1) Define grid of range
xgboost_grid = {
        'learning_rate':[0.01, 0.05, 0.1, 0.2], #Typical values between 0.01 and 0.2
        'n_estimators':[500, 750, 1000],
        
        'max_depth':[3, 4, 5, 6, 7, 8, 9, 10, 11, 12], #Typical values between 3-10
        'min_child_weight':[1, 2, 3, 4, 5, 6],
        'gamma':[0, 0.0001, 0.001, 0.01], #Default is 0
        'subsample':[0.5, 0.6, 0.7, 0.8, 0.9, 1], #Typical values between 0.5 -1
        'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1], #Typical values between 0.5 - 1
        
        'reg_alpha':[0, 0.5], #Default is 0
        'reg_lambda':[0.5, 1], #Default is 1
        
        'nthread':[10]        
        }

#2) Random Search 25 times
hypertuning = RandomizedSearchCV(xgboost, xgboost_grid, cv=3, scoring=boston_housing, n_iter = 25)
hypertuning.fit(predictors, labels)

#Check best score and save parameters 
hypertuning.best_score_ #0.125
xgboost_grid_best = hypertuning.best_params_

#Use the saved parameters to train final model
final_settings = xgb.XGBRegressor(objective ='reg:linear', 
                               learning_rate = 0.05, 
                               n_estimators = 500,
                               max_depth = 4,
                               min_child_weight = 1,
                               gamma = 0.001,
                               subsample = 0.8,
                               colsample_bytree = 0.9,
                               reg_alpha = 0,
                               reg_lambda = 1,
                               nthread = 4)

final_model = final_settings.fit(predictors, labels)













#EXHAUSTIVE GRID SEARCH
tuning = GridSearchCV(ridgereg , param_grid, cv=5, scoring=boston_housing)
tuning.fit(predictors, labels)

tuning = GridSearchCV(xgboost , param_grid, cv=3, scoring=boston_housing)
tuning.fit(predictors, labels)

#Check best score and parameters
tuning.best_score_
tuning.best_params_ #alpha = 34


#RANDOM GRID SEARCH 
tuning1 = RandomizedSearchCV(ridgereg , param_grid, cv=10, scoring=boston_housing, n_iter = 25)
tuning1.fit(predictors, labels)

#Check best score and parameters compare with Exhaustive search
tuning1.best_score_
tuning1.best_params_





#------------------------------------------------------------------------------
#STACKING MODELS
#------------------------------------------------------------------------------

#1: Linear regression
#2: Ridge regression
#3: Lasso regression
#4: RandomForest
#5: Gradient Boosting
#6: Extreme gradient boosting
#7: Support Vector Machine

#Check number of CPUs
import multiprocessing
multiprocessing.cpu_count()

#Using vecstack
from vecstack import stacking
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error


# demonstrational and shouldn't be considered as recommended.
models = [linear_model.LinearRegression(),
                     linear_model.Ridge(),
                     linear_model.Lasso(),
                     ExtraTreesRegressor(),
                  RandomForestRegressor(),

         XGBRegressor(objective ='reg:linear', 
                       learning_rate = 0.05, 
                       n_estimators = 500,
                       max_depth = 4,
                       min_child_weight = 1,
                       gamma = 0.001,
                       subsample = 0.8,
                       colsample_bytree = 0.9,
                       reg_alpha = 0,
                       reg_lambda = 1,
                       nthread = 10)]

#Perform stacking
S_train, S_test = stacking(models,
                           X_train, y_train, X_test,   
                           regression=True,            
                           mode='oof_pred_bag',        
                           save_dir=None,              
                           metric=rmsle, 
                           n_folds=6,                  
                           shuffle=True,               
                           random_state=0,
                           verbose=2)
#Look at the results
S_train[:5]
S_test[:5]

#Apply 2nd level model

# Initialize 2nd level model
model = XGBRegressor(objective ='reg:linear', 
                       learning_rate = 0.05, 
                       n_estimators = 500,
                       max_depth = 4,
                       min_child_weight = 1,
                       gamma = 0.001,
                       subsample = 0.8,
                       colsample_bytree = 0.9,
                       reg_alpha = 0,
                       reg_lambda = 1,
                       nthread = 10)
    
# Fit 2nd level model
model = model.fit(S_train, y_train)

# Predict
y_pred = model.predict(S_test)

# Final prediction score
print('Final prediction score: [%.8f]' % rmsle (y_test, y_pred))













#------------------------------------------------------------------------------
#CHECK FEATURE IMPORTANCE
#------------------------------------------------------------------------------

#Now you have 289 predictors; which one is most important?
feature_importances = pd.DataFrame(model1.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance',                                                                 
                                    ascending=False)

#Check out most important features

#1) Check Overall Quality
train['OverallQual'].unique()
sns.distplot(train['OverallQual'])

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data=train)

#2) Check GrLivArea
train['GrLivArea'].describe()
sns.distplot(train['GrLivArea'])

sns.lmplot(x='GrLivArea', y='SalePrice', data=train)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train) 

#3) Check TotalBsmtSF
train['TotalBsmtSF'].describe()
sns.distplot(train['TotalBsmtSF'])

sns.lmplot(x = 'TotalBsmtSF', y = 'SalePrice', data=train)

#Check if top 20 features has any missings in complete dataset
missings = complete.isnull().sum()






#------------------------------------------------------------------------------
#GENERATE SUBMISSION FOR KAGGLE
#------------------------------------------------------------------------------ 

#Delete 'Id', 'Key' and 'SalePrice' from complete_test
complete_test.drop(['Id', 'Key_train', 'SalePrice'], axis = 1, inplace = True)

#Predict on complete_test set and store results in dataframe
predictions = final_model.predict(complete_test)
predictions = pd.DataFrame(predictions)

#Put predictions in sample_submission
sample_submission['SalePrice'] = predictions

#Store results in csv
sample_submission.to_csv('submission3.csv', index = False)



