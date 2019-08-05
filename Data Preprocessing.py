
import pandas as pd
from sklearn.model_selection import train_test_split

# Read csv file using Pandas
training_data = pd.read_csv("E:\\Nayan\\Sesh\\bank-full_train.csv")

#fetch unique values of each feature (e.q. education column)
#Other values are mentioned in seperate file
training_data['education'].unique()

#Remove 'result' column for further requirement
training_data_wo_y = training_data.drop(['y'], axis=1)

#replace 'month' by numbers
month_replaced = training_data_wo_y.replace(['aug', 'jul', 'mar', 'may', 'jun', 'jan', 'feb', 'nov', 'apr', 'sep', 'oct', 'dec'],
                                         [8, 7, 3, 5, 6, 1, 2, 11, 4, 9, 10, 12])

#replace 'job' column by numeric values
job_replaced = month_replaced.replace(['blue-collar', 'admin.', 'technician', 'self-employed', 'services', 'entrepreneur',
                                       'management', 'unemployed', 'retired', 'housemaid', 'student', 'unknown'],
                                      [4, 9, 7, 6, 5, 8, 10, 0, 3, 1, 2, -1])
#replace 'marital' column by numeric values
marital_replaced = job_replaced.replace(['married', 'divorced', 'single'], [1, -1, 0])

#replace 'education' with numeric values
education_replaced = marital_replaced.replace(['secondary', 'tertiary', 'primary', 'unknown'], [2, 3, 1, -1])

#replace 'yes/no' results with entire dataset with '1/0'
boolean_replaced = education_replaced.replace(['yes', 'no'], [1, 0])

#replace contact
contact_replaced = boolean_replaced.replace(['cellular', 'unknown', 'telephone'], [2, -1, 1])

#replace poutcomes
poutcome_replaced = contact_replaced.replace(['unknown', 'failure', 'other', 'success'], [-1, 0, -1, 1])

#write numeric dataset to csv file
poutcome_replaced.to_csv("E:\\Nayan\\Sesh\\numeric_training_data1.csv", index = False)

#split the dataset into training and testing
#train_x, test_x = train_test_split(poutcome_replaced, test_size=0.2)


