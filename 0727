import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
data = pd.DataFrame(data = np.c_[wine['data'],wine['target']],columns = wine['feature_names']+['target'])

X = data.drop(['target'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)
