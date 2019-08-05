
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