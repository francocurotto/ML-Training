# imports
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# load data
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

# create regressors
linear_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=1, loss="hinge"))])
poly_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1))])
rbf_svc = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5))])

# fit regressors
linear_svc.fit(X, y)
poly_svc.fit(X, y)
rbf_svc.fit(X, y)

# get results
y_pred_linear = linear_svc.predict(X)
y_pred_poly = poly_svc.predict(X)
y_pred_rbf = rbf_svc.predict(X)

# create displays
#plt.figure()
cm_display = ConfusionMatrixDisplay(confusion_matrix(y, y_pred_linear), display_labels=iris["target_names"]).plot()
plt.title("Linear SVC")
#plt.figure()
cm_display = ConfusionMatrixDisplay(confusion_matrix(y, y_pred_poly), display_labels=iris["target_names"]).plot()
plt.title("Poly SVC")
#plt.figure()
cm_display = ConfusionMatrixDisplay(confusion_matrix(y, y_pred_rbf), display_labels=iris["target_names"]).plot()
plt.title("RBF SVC")

plt.show()
