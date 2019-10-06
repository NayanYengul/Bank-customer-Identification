#Logistic Regression

import numpy as np
import pandas as pd

#read dataset
X = pd.read_csv("E:\\Nayan Dell Backup\\Nayan\\Sesh\\Processed_data.csv")

data_y = pd.read_csv("E:\\Nayan Dell Backup\\Nayan\\Sesh\\df_wo_missing_values.csv")
data_y = data_y.iloc[:, 14]
print(data_y.value_counts())

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

#Drop job_"entrepreneur" column
X = X.drop(['job_entrepreneur'], axis=1)

#Drop "Default-no" column as VIF is 49.879119
X = X.drop(['default_no'], axis=1)
print(X.columns)

#Split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)
print(y_test['output'].value_counts())


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predict test set result
y_pred = classifier.predict(X_test)

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
y_pred_df = pd.DataFrame(y_pred, index=y_test.index)

print(st.kstest(y_pred_df, 'norm'))
