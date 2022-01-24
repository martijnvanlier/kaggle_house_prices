import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Create Postgres connection
#import sqlalchemy
#postgres_conn = sqlalchemy.create_engine('postgresql://postgres:ckvunitas03@localhost:5432/kaggle_competitions')

#Import training data from Postgress
#train = pd.read_sql('SELECT * FROM december_playground.train_adj', postgres_conn)

#Import training data from csv file
train = pd.read_csv('D:/Kaggle/House Prices/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('D:/Kaggle/House Prices/house-prices-advanced-regression-techniques/test.csv')

#Inspect target variable 
train['SalePrice'].head()

#Check imported datatypes
print(train.dtypes)

#Check datatypes
print(train.dtypes)

#Check memory size
print(train.memory_usage().sum() / (1024**2)) #converting to megabytes

#Check rows and columns
print(train.shape)

#Inspect attributes
print(train.columns)

#Plot distribution of house prices
plt.ion()
output = sns.histplot(data=train, x='SalePrice')

#Inspect all the features and there impact on the target variable

#Step 0: Turn interactive mode off (otherwise a separate window will be created for each graph)
plt.ioff()

#Step 1: Put all features in list
features = train.columns.tolist()

# Step 2: Loop through all features and their relationship with target variable
for x in features:

  #If X is numerical create correlation plot
  if train[x].dtype=='float64':
      output = sns.lmplot(data=train, x=x, y="SalePrice")
      output.figure.savefig("D:/Kaggle/House Prices/EDA/" + str(x) + ".jpg")

  #If X is categorical (object/int64) create barplot
  else:
      output = sns.catplot(data=train, kind="bar", x=x, y="SalePrice")
      output.figure.savefig("D:/Kaggle/House Prices/EDA/" + str(x) + ".jpg")




