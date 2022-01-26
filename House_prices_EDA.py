#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import training data from working directory
train = pd.read_csv('/home/ec2-user/environment/Kaggle/train.csv')
test = pd.read_csv('/home/ec2-user/environment/Kaggle/test.csv')

#Check imported datatypes
print(train.dtypes)

#Check datatypes
print(train.dtypes)

#Check rows and columns
print(train.shape)

#Inspect attributes
print(train.columns)

#Create and store first graph
output = sns.histplot(data=train, x='SalePrice')
output.figure.savefig("/home/ec2-user/environment/Kaggle/graphs/saleprice.jpg")

#Inspect all the features and there impact on the target variable

#Step 1: Put all features in list
features = train.columns.tolist()

# Step 2: Loop through all features and their relationship with target variable
for x in features:

  #If X is numerical create correlation plot
  if train[x].dtype=='float64':
      output = sns.lmplot(data=train, x=x, y="SalePrice")
      output.figure.savefig("/home/ec2-user/environment/Kaggle/graphs/" + str(x) + ".jpg")

  #If X is categorical (object/int64) create barplot
  else:
      output = sns.catplot(data=train, kind="bar", x=x, y="SalePrice")
      output.figure.savefig("/home/ec2-user/environment/Kaggle/graphs/" + str(x) + ".jpg")


