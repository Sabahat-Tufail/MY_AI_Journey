from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


svm = SVC(kernel='rbf', gamma=0.5, C=1.0)
svm.fit(X_pca, y)


DecisionBoundaryDisplay.from_estimator(
    svm,
    X_pca,
    response_method="predict",
    cmap=plt.cm.Spectral,
    alpha=0.6,
    xlabel='PCA Component 1',
    ylabel='PCA Component 2'
)


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=15, edgecolors="k", cmap=plt.cm.tab10)
plt.title("SVM Decision Boundary on Digits Dataset (2D PCA projection)")
plt.show()