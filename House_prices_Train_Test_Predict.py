import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

#Import training data from csv file
train = pd.read_csv('D:/Kaggle/House Prices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('D:/Kaggle/House Prices/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('D:/Kaggle/House Prices/house-prices-advanced-regression-techniques/sample_submission.csv')

#Store ID in index (because it doesn't have any predictive power)
train.set_index('Id', inplace=True)
test.set_index('Id', inplace=True)

#1) Store X and y variables from training data separate
y = train['SalePrice']
X=train.drop(['SalePrice'], axis=1) #Drop SalePrice (because it's Y)

#2) One-hot-encoding
ohe= OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(X)
X_ohe=ohe.transform(X)

#3) Check for NAs in dataset
np.any(np.isnan(X))
np.all(np.isfinite(X))

#4) Fill NAs with mean for now (otherwise model generation will give errors)
#X.fillna(X.mean(), inplace=True)

#5) split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, train_size=0.70)

#6) Train linear regression model
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)

#7) Test linear regression model
reg.score(X_test, y_test)

#8) Create predictions for test set

#8.1)Data preparation
test_cleaned=ohe.transform(test)
test_cleaned.fillna(test_cleaned.mean(), inplace=True)
#8.2)Predictions
predictions = reg.predict(test_cleaned)
#8.3) Convert results from array to dataframe
predictions = pd.DataFrame(predictions)
#8.4) Give column a name
predictions.columns = ['SalePrice']
#8.5) Put predictions in sample_submission
sample_submission['SalePrice'] = predictions
#8.6) Store results in csv
sample_submission.to_csv('D:/Kaggle/House Prices/submission_10_01_2022.csv', index = False)
