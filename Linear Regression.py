import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Reading csv
with open("E:\\Nayan\\Sesh\\numeric_training_data1.csv",'r') as csvfile:
    #creating csv object
    diabetes = csv.reader(csvfile)


#Print features and targets
print("Features: ", diabetes.feature_names)

#We are using here, only one feature
diabetes_x = diabetes.data[:, np.newaxis, 2]

#training and testing actual data
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]


#training and testing target
diabetes_y_train = diabetes.target[:-20] #remove last 20 elements
diabetes_y_test = diabetes.target[-20:] #take last 20 elements

#create linear regression object
linear_regression = linear_model.LinearRegression()

#Train the model using training dataset
linear_regression.fit(diabetes_x_train, diabetes_y_train)

#Test the model using testing data
predict_y_pred = linear_regression.predict(diabetes_x_test)

#Regression coefficient
print("coefficients: ", linear_regression.coef_)
print("Intercept: ",linear_regression.intercept_)

#Mean Sqaured Error
print("Mean Sqaured Error: %.2f" % mean_squared_error(diabetes_y_test,predict_y_pred))

#Variance score
print("Variance score: %.2f" % r2_score(diabetes_y_test, predict_y_pred))

#plot outputs
plt.scatter(diabetes_x_test, diabetes_y_test, color = 'black')
plt.plot(diabetes_x_test, predict_y_pred, color='blue', linewidth=2)

#plt.xticks([]) #remove x-ticks

plt.show()



