import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi','age', 'class' ]
data = pd.read_csv("./data/pima-indians-diabetes.data.csv", names=header)

# 데이터 전처리 : Min-Max 스케일링
array = data.values
X = array[:,0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.3)

# 모델 선택 및 분할
model = LogisticRegression()
model.fit(X_train, Y_train)

fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring='accuracy')
s = sum(acc)
l = len(acc)
