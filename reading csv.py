#Read csv file
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#Reading csv
with open("E:\\Nayan\\Sesh\\bank-full_test.csv",'r') as csvfile:
    #creating csv object
    csvreader = csv.reader(csvfile)

    #extracting data row by row
    #for row in csvreader:
        #print(row)

# get total number of rows
#print("Total no. of rows: %d"%(csvreader.line_num))

# get data info using Pandas
training_data = pd.read_csv("E:\\Nayan\\Sesh\\bank-full_train.csv")
print(training_data.info())

#split the dataset into training and testing
train_x, test_x = train_test_split(training_data, test_size=0.2)
print(test_x)


#find unique data of a column
#print(training_data['education'].unique())

#Size of data
size_data = training_data.shape




#Data cleasing

#1. Remove result column for further requirement
heads = training_data.head(5)


training_data_wo_y = training_data.drop(['y'], axis=1)


#Unique values of each column
training_data_wo_y['previous'].unique

#replace month by numbers
month_replaced = training_data_wo_y.replace(['aug', 'jul', 'mar', 'may', 'jun', 'jan', 'feb', 'nov', 'apr', 'sep', 'oct', 'dec'],
                                         [8, 7, 3, 5, 6, 1, 2, 11, 4, 9, 10, 12])

#replace job column by numeric values
job_replaced = month_replaced.replace(['blue-collar', 'admin.', 'technician', 'self-employed', 'services', 'entrepreneur',
                                       'management', 'unemployed', 'retired', 'housemaid', 'student', 'unknown'],
                                      [4, 9, 7, 6, 5, 8, 10, 0, 3, 1, 2, -1])


#replace marital column by numeric values
marital_replaced = job_replaced.replace(['married', 'divorced', 'single'], [1, -1, 0])

#replace education with numeric values
education_replaced = marital_replaced.replace(['secondary', 'tertiary', 'primary', 'unknown'], [2, 3, 1, -1])

#replace yes/no with 1/0
boolean_replaced = education_replaced.replace(['yes', 'no'], [1, 0])

#replace contact
contact_replaced = boolean_replaced.replace(['cellular', 'unknown', 'telephone'], [2, -1, 1])


#replace poutcomes

poutcome_replaced = contact_replaced.replace(['unknown', 'failure', 'other', 'success'], [-1, 0, -1, 1])

poutcome_replaced.to_csv("E:\\Nayan\\Sesh\\numeric_training_data.csv", index = False)

#Principal component analysis
#seperating out the features
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
            'pdays', 'previous', 'poutcome', 'ID']
x = poutcome_replaced.loc[:, features].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3',
                          'principal component 4', 'principal component 5'])

print(principalDf)