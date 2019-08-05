import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Principal component analysis

# Read csv file using Pandas
data = pd.read_csv("E:\\Nayan\\Sesh\\numeric_training_data1.csv")

#seperating out the features
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
            'pdays', 'previous', 'poutcome', 'ID']
x = data.loc[:, features].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3',
                          'principal component 4', 'principal component 5'])

print(principalDf)


