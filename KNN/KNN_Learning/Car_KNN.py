import matplotlib.pyplot as plt
import sklearn
from  sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


data=pd.read_csv("car.data")
le=preprocessing.LabelEncoder()
buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform((list(data["maint"])))
door=le.fit_transform((list(data["door"])))
persons=le.fit_transform((list(data["persons"])))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))

predict="class"
X=list(zip(buying,maint,door,persons,lug_boot,safety ))
Y=list(cls)

x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(X,Y, test_size=0.1)
model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc=model.score(x_test, y_test)
print(acc)
prediction=model.predict(x_test)
names=["unacc","acc","good","vgood"]
for x in range(len(prediction)):
    print(f"Predicted: {names[prediction[x]]} , Data: {x_test[x]}, Actual:{names[y_test[x]]}")



pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=Y, cmap="viridis", s=20)
plt.legend(handles=scatter.legend_elements()[0], labels=names)
plt.title("Car Evaluation Dataset (PCA Projection to 2D)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.show()

