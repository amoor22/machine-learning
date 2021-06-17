from sklearn import datasets
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

canc = datasets.load_breast_cancer()

X = canc.data
Y = canc.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
classes = ['malignant', 'benign']
neighbor = KNeighborsClassifier(n_neighbors=13)
neighbor.fit(x_train, y_train)
print(neighbor.score(x_test, y_test))
cls = svm.SVC(kernel="linear")
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
for i in range(len(y_pred)):
    print(classes[y_pred[i]], classes[y_test[i]])
print(acc)