import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt


header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv',delim_whitespace=True, names=header)

array = data.values
# 독립변수 / 종속변수
X = array[:, 0:13]
Y = array[:, 13]

# 학습데이터 / 테스트데이터
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue')
plt.scatter(range(len(X_test[:15])), y_pred[:15], color='red', marker='*')
plt.xlabel("Index")
plt.ylabel("MEDV ($1,000)")
plt.show()

mse = mean_squared_error(Y_test,y_pred)
print(mse)

kfold = KFold(n_splits=5)
mse= cross_val_score(model, X, Y, scoring="neg_mean_squared_error")
print(mse.mean)