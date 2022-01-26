#Import required packages
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import metrics
import xgboost as xgb

#1) DATA PRE-PROCESSING

#1.1) Import train and test data from csv file
train = pd.read_csv('/home/ec2-user/environment/Kaggle/train.csv')
test = pd.read_csv('/home/ec2-user/environment/Kaggle/test.csv')
sample_submission = pd.read_csv('/home/ec2-user/environment/Kaggle/sample_submission.csv')

#1.2) Store ID in index (because it doesn't have any predictive power)
train.set_index('Id', inplace=True)
test.set_index('Id', inplace=True)

#1.3) Store X and y variables from training data separate
y = train['SalePrice']
X=train.drop(['SalePrice'], axis=1) #Drop SalePrice (because it's the variable to predict).

#1.4) One-hot-encoding, convert categories to dummies
train = pd.get_dummies(X)   
test = pd.get_dummies(test)     
final_train, final_test = train.align(test, join='inner', axis=1)  #Include only columns in test set which are available in training set.

#Check if number of columns aligns
print(final_train.shape)
print(final_test.shape)

#1.5) Fill NAs with mean and verify results
final_train.fillna(final_train.mean(), inplace=True)
final_test.fillna(final_test.mean(), inplace=True)

print(np.any(np.isnan(final_train)))
print(np.any(np.isnan(final_test)))

#2) TRAIN AND COMPARE MODELS

#2.1) Split into train test sets
X_train, X_test, y_train, y_test = train_test_split(final_train, y, train_size=0.70)

#2.2) Train and compare models

#2.2.1) linear regression model
regression = LinearRegression().fit(X_train, y_train)

#Check score
print(regression.score(X_train, y_train))
print(regression.score(X_test, y_test))

#Use Root Mean Squared Log Error (RMSLE) as evaluation metric
y_train_pred = regression.predict(X_train)
y_test_pred = regression.predict(X_test)

print(metrics.mean_squared_log_error(y_train, y_train_pred, squared=False))
#print(metrics.mean_squared_log_error(y_test, y_test_pred, squared=False)) #MMean Squared Logarithmic Error cannot be used when targets contain negative values.

#2.2.2) Random forest
randomforest = RandomForestRegressor()
randomforest.fit(X_train, y_train)
y_train_pred = randomforest.predict(X_train)
y_test_pred = randomforest.predict(X_test)

#Evaluate model on train/test
print(metrics.mean_squared_log_error(y_train, y_train_pred, squared=False))
print(metrics.mean_squared_log_error(y_test, y_test_pred, squared=False))

#2.2.3) Xgboost
xgboost = xgb.XGBRegressor()
xgboost = xgboost.fit(X_train, y_train)

y_train_pred = xgboost.predict(X_train)
y_test_pred = xgboost.predict(X_test)

#Evaluate model on train/test
print(metrics.mean_squared_log_error(y_train, y_train_pred, squared=False))
print(metrics.mean_squared_log_error(y_test, y_test_pred, squared=False))

#2.3) List most important features and export
feature_importance = pd.DataFrame(xgboost.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance',                                                                 
                                    ascending=False)

feature_importance.to_csv('/home/ec2-user/environment/Kaggle/26_01_2022_feature_importance.csv', index = True)

#2.4) Retrain with 150 most important features
feature_importance.reset_index(level=0, inplace=True)
feature_importance.columns = ['feature','importance']
top_features = feature_importance[feature_importance['importance'] > 0.0001]
features_list = top_features['feature'].tolist()
final_train = train[train.columns.intersection(features_list)]
final_test = test[test.columns.intersection(features_list)]

X_train, X_test, y_train, y_test = train_test_split(final_train, y, train_size=0.70)
xgboost = xgb.XGBRegressor()
xgboost = xgboost.fit(X_train, y_train)
y_train_pred = xgboost.predict(X_train)
y_test_pred = xgboost.predict(X_test)
print(metrics.mean_squared_log_error(y_train, y_train_pred, squared=False))
print(metrics.mean_squared_log_error(y_test, y_test_pred, squared=False))

#3) CREATE PREDICTIONS

#3.1) Make predictions based on xgboost
predictions = xgboost.predict(final_test)

#3.2) Convert results from array to dataframe
predictions = pd.DataFrame(predictions)

#3.3) Give column a name
predictions.columns = ['SalePrice']

#3.4) Put predictions in sample_submission
sample_submission['SalePrice'] = predictions

#3.5) Store results in csv
sample_submission.to_csv('/home/ec2-user/environment/Kaggle/26_01_2022_submission.csv', index = False)
