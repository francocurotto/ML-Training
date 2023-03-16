# imports
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# load data
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

# fit regressor
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, C=10))])
log_reg.fit(X, y)

# get results and display
y_pred = log_reg.predict(X)
cm = confusion_matrix(y, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=iris["target_names"]).plot()
print(type(cm_display))
plt.show()
