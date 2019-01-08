# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import svm

df = pd.read_csv("/Users/zhangjin13/Documents/data/loss_whole.csv")

dfX = df.drop(['Exited'], axis=1, inplace=False)
df.head()
dfy = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.3, random_state=42)

# # 设置线性回归模块
# model = LogisticRegression()
# # 训练数据，得出参数
# model.fit(dfX, dfy)

model = svm.SVC()
model.fit(dfX, dfy)

y_prob = model.predict(X_train)
y_preb = model.predict(X_test)

print(accuracy_score(y_pred=y_prob, y_true=y_train))
print(accuracy_score(y_pred=y_preb, y_true=y_test))

joblib.dump(model, 'svm_model')

model_1 = joblib.load('svm_model')
model_1.predict(X_test)
