import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
data = pd.DataFrame(data = np.c_[wine['data'],wine['target']],columns = wine['feature_names']+['target'])

X = data.drop(['target'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)


X.shape


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train,y_train)


test_prediction = LR.predict(X_test)


confusion_matrix(y_test,test_prediction)


data.columns


data['target'].value_counts()

data.groupby('target').mean()

from scipy import stats
a = data[data.target==0]
b = data[data.target==1]
c = data[data.target==2]
A = a.drop(['target'], axis=1)
B = b.drop(['target'], axis=1)
stats.ttest_ind(A,B,equal_var=True)

TT = stats.ttest_ind(A,B,equal_var=True)
print(TT.pvalue)
