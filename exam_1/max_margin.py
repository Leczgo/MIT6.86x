import matplotlib.pyplot as plt
from sklearn import svm
#from sklearn import inspection
import numpy as np


# we create 40 separable points
X, y = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]]),np.array([-1,-1,-1,-1,-1,1,1,1,1,1])

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
#inspection.DecisionBoundaryDisplay.from_estimator(
#    clf,
#    X,
#    plot_method="contour",
#    colors="k",
#    levels=[-1, 0, 1],
#    alpha=0.5,
#    linestyles=["--", "-", "--"],
#    ax=ax,
#)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()
#print(clf.support_vectors_)
print(clf.coef_[0])
print(clf.intercept_)