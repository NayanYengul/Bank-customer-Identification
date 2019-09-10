#Logistic Regression

import numpy as np
import pandas as pd

#read dataset
X = pd.read_csv("E:\\Nayan\\Sesh\\Processed_data.csv")
total_cols = len(X.axes[1])

data_y = pd.read_csv("E:\\Nayan\\Sesh\\df_wo_missing_values.csv")
data_y = data_y.iloc[:, 14]

#convert y to numeric values
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y_array = labelEncoder_y.fit_transform(data_y)
y = pd.DataFrame(y_array, columns=['output'])


#Split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)


#whether to use or not??????
#scaling of data
#from sklearn.preprocessing import StandardScaler
#Scale = StandardScaler()
#X_train = Scale.fit_transform(X_train)
#X_test = Scale.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predict test set result
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)