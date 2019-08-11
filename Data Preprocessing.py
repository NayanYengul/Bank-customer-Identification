
import pandas as pd
from sklearn.model_selection import train_test_split

# Read csv file using Pandas
training_data = pd.read_csv("E:\\Nayan\\Sesh\\bank-full_train.csv")

#Remove 'result' column for further requirement
training_data_wo_y = training_data.drop(['y'], axis=1)

#Remove 'Id' column
training_data_wo_Id = training_data_wo_y.drop(['ID'], axis=1)

#Remove 'pdays' column as it has random data
training_data_wo_pdays = training_data_wo_Id.drop(['pdays'], axis=1)

#Remove 'duration' column as it has random data
training_data_wo_duration = training_data_wo_pdays.drop(['duration'], axis=1)

#Remove 'balance' column as it has random data
training_data_wo_balance = training_data_wo_duration.drop(['balance'], axis=1)

print(training_data_wo_balance.info())

#Print total number of missing values of each features
print(training_data_wo_balance.isnull().sum())

#write data into csv
training_data_wo_balance.to_csv("E:\\Nayan\\Sesh\\training_data_missing_val.csv", index = False)

#As per this there are no missing values in the data set
#but, the uknown term in e.q. 'education' column is missing value
#will deal with 'unknown' value as a missing value, using following steps

# Making a list of missing value types
missing_values = ["unknown"]
df = pd.read_csv("E:\\Nayan\\Sesh\\training_data_missing_val.csv", na_values=missing_values)

print(df.isnull().sum())

