import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
import sklearn
from sklearn import linear_model
import pickle
data=pd.read_csv("housing[1].csv")
data=data.dropna()
data=data[["housing_median_age","total_rooms","total_bedrooms", "population", "households","median_income","median_house_value"]]
predict="median_house_value"
X=np.array(data.drop([predict], axis=1))
Y=np.array(data[predict])
x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

try:
    pickle_in = open("house_predict", "rb")
    linear = pickle.load(pickle_in)
except:
         best=0
         for _ in range(30):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
            linear=linear_model.LinearRegression()
            linear.fit(x_train,y_train)
            acc=linear.score(x_test, y_test)
            if acc>best:
                with open("house_predict", "wb") as file:
                    pickle.dump(linear, file)
            print(acc)

prediction=linear.predict(x_test)
for x in range(len(prediction)):
    print(f"{int(prediction[x]), x_test[x], int(y_test[x])}")

p="median_house_value"
q="median_income"
x=data[p]
y=data[q]
plt.scatter(x,y)
plt.xlabel(p)
plt.ylabel(q)
plt.grid=True
plt.show()



