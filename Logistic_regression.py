#Logistic Regression

import numpy as np
import pandas as pd

#read dataset
X = pd.read_csv("E:\\Nayan Dell Backup\\Nayan\\Sesh\\Processed_data.csv")
total_cols = len(X.axes[1])

data_y = pd.read_csv("E:\\Nayan Dell Backup\\Nayan\\Sesh\\df_wo_missing_values.csv")
data_y = data_y.iloc[:, 14]

#convert y to numeric values
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y_array = labelEncoder_y.fit_transform(data_y)
y = pd.DataFrame(y_array, columns=['output'])

#Variance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

#Drop job_"entrepreneur" column as it is equivalent to self employed
X = X.drop(['job_entrepreneur'], axis=1)

#Drop "Default-no" column as VIF is 49.879119
X = X.drop(['default_no'], axis=1)
print(X.shape)

#Concat X and y
dataX_y = X.join(y)

#check balance or imbalance data
#check if the data is balanced or not
count_class_0, count_class_1 = dataX_y['output'].value_counts()
print("Class_0:", count_class_0)
print("Class_1:", count_class_1)

#divide by class
df_class_0 = dataX_y[dataX_y['output'] == 0]
df_class_1 = dataX_y[dataX_y['output'] == 1]

#Under-sampling
#df_class_under_0 = df_class_0.sample(count_class_1)
#X_under_sampled = pd.concat([df_class_under_0,df_class_1], axis=0)
#print(X_under_sampled['output'].value_counts())

#Over_sampling
df_class_over_1 = df_class_1.sample(count_class_0, replace=True)
X_over_sampled = pd.concat([df_class_0, df_class_over_1], axis=0)
print(X_over_sampled['output'].value_counts())

#X matrix and y vector
X = X_over_sampled.iloc[:, :37].values
y = X_over_sampled.iloc[:, 37].values
print("X_shape:", X.shape)
print(X)
print("y_shape:", y.shape)
print(y)

#Split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)
print("X_test_shape:", X_test.shape)
print("y_test_shape:", y_test.shape)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predict test set result
y_pred = classifier.predict(X_test)
print("y_pred_shape: ", y_pred.shape)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ks test
import scipy.stats as st

#convert y_pred to dataframe
y_test_df = pd.DataFrame(y_test)
y_pred_df = pd.DataFrame(y_pred, index=y_test_df.index)

print(st.kstest(y_pred_df, 'norm'))

