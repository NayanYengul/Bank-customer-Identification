
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

# Read csv file using Pandas
training_data = pd.read_csv("E:\\Nayan\\Sesh\\bank-full_train.csv")

#Remove 'Id' column
training_data_wo_Id = training_data.drop(['ID'], axis=1)

#Remove 'pdays' column as it has random data
training_data_wo_pdays = training_data_wo_Id.drop(['pdays'], axis=1)

#Remove 'duration' column as it has random data
training_data_wo_duration = training_data_wo_pdays.drop(['duration'], axis=1)

#Print total number of missing values of each features
print(training_data_wo_duration.isnull().sum())

#write data into csv
training_data_wo_duration.to_csv("E:\\Nayan\\Sesh\\training_data_missing_val.csv", index = False)

#As per this there are no missing values in the data set
#but, the uknown term in e.q. 'education' column is missing value
#will deal with 'unknown' value as a missing value, using following steps

df = pd.read_csv("E:\\Nayan\\Sesh\\training_data_missing_val.csv")

print("Initial count of null values: \n", df.isnull().sum())

#mark unknown values as missing or NaN
df = df.replace("unknown", np.NaN)

#drop rows where missing values present in Job column
df = df.dropna(subset=['job'])

print("Final null value count: \n", df.isnull().sum())

#Replace NaN values with "unknown" to do further procedure
df = df.replace(np.NaN, "unknown")
df.to_csv("E:\\Nayan\\Sesh\\df_wo_missing_values.csv", index = False)
print(df.isnull().sum())

#Make X  matrix and y vector

X = df.iloc[:, :-1]


#extract only categorical columns
X_categorical = X.select_dtypes(include=[object])
print(X_categorical.head())

from feature_engine.categorical_encoders import OneHotCategoricalEncoder
one_enc = OneHotCategoricalEncoder(top_categories=None, variables=list(X_categorical), drop_last=True)
one_enc.fit(X)
X_transformed = one_enc.transform(X)
print(X_transformed.columns)
print(len(X_transformed.columns))
print(X_transformed.head())

X_transformed.to_csv("E:\\Nayan\\Sesh\\Processed_data.csv", index = False)
