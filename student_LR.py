import sklearn
import  numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn.utils import shuffle
data=pd.read_csv("linear_regression/student-mat.csv", sep=";")
# print(data.columns.tolist(), data.dtypes)
data=data[['G1','G2','G3','studytime','failures','absences', "age", "traveltime", "freetime","health" ]]

predict="G3"
X=np.array(data.drop([predict],axis=1))
Y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y, test_size=0.1)
try:
    pickle_in =open("linear_regression/student.pickle", "rb")
    linear=pickle.load(pickle_in)
except:
    best=0
    for _ in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
        linear=linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc=linear.score(x_test,y_test)
        print(acc)
        if acc>best:
            acc=best
            with open("linear_regression/student.pickle", "wb") as f:
                pickle.dump(linear, f)


print("co-ef", linear.coef_)
print("co-ef", linear.intercept_)
predictions=linear.predict(x_test)
for x in range(len(predictions)):
    print(x)
    print(predictions[x], x_test[x], y_test[x])

x=data["G1"]
y=data['G3']
style.use('ggplot')
plt.scatter(x,y)
plt.xlabel("G1")
plt.ylabel("Final grades")
plt.grid=True
plt.show()