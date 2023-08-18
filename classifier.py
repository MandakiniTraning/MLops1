import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/MLops1/data/iris.csv")
features = ['sepal_length','sepal_width','petal_length', 'petal_width']
target = 'species'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
#initiate the modelclass
clf = DecisionTreeClassifier(criterion ="entropy")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy of the model is {accuracy_score(y_test,y_pred)*100}")
