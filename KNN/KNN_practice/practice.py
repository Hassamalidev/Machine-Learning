import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns


data=pd.read_csv("bank.csv", sep=";")

le=preprocessing.LabelEncoder()
age=le.fit_transform(data["age"])
job=le.fit_transform(data["job"])
marital=le.fit_transform(data["marital"])
education=le.fit_transform(data["education"])
balance=le.fit_transform(data["balance"])
housing=le.fit_transform(data["housing"])
loan=le.fit_transform(data["loan"])
duration=le.fit_transform(data["duration"])
contact=le.fit_transform(data["contact"])
campaign=le.fit_transform(data["campaign"])
pdays=le.fit_transform(data["pdays"])
poutcome=le.fit_transform(data["poutcome"])

predict="education"
X=list(zip(age, job, marital, education, balance, housing, loan, duration, contact, campaign,pdays, poutcome ))
Y=list(education)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y, test_size=0.1)
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
acc=model.score(x_test, y_test)
print(acc)
prediction=model.predict(x_test)
eduction=['tertiary', 'secondary', 'unknown', 'primary']
for x in range(len(prediction)):
    print(f"Predicted: {eduction[prediction[x]]}, Data: {x_test[x]}, Actual: {eduction[y_test[x]]}")

pca =PCA(n_components=2)
X_reduced=pca.fit_transform(X)
df=pd.DataFrame({
    "Age":X_reduced[:, 0],
     "job":X_reduced[:,1],
    "class":[eduction[i] for i in Y]
})
sns.scatterplot(data=df, x="Age", y="job", hue="class", palette="Dark2")
plt.show()
