#http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
#get iris data
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
%matplotlib inline  
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('x shape: ' + str(iris_X.shape))
print('y unique: ' + str(np.unique(iris_y)))

"""All supervised estimators in scikit-learn implement a fit(X, y) method to fit the model 
and a predict(X) method that, given unlabeled observations X, returns the predicted labels y."""

# Split iris data in train and test data
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

#KNN, create object, fit, predict
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(iris_X_train, iris_y_train) 
print(knn.predict(iris_X_test))
print(iris_y_test)

###Diabetes data & LR
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X =diabetes_X_train, y=diabetes_y_train)
print(regr.coef_)
print('mean : ' + str(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)))
print('R^2 : ' + str(regr.score(diabetes_X_test, diabetes_y_test) ))

# use regularization
alphas = np.logspace(-4, -1, 6)
regr = linear_model.Ridge(alpha=.1)
hyper = {}
for alpha in alphas:
	r = regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train)
	score = r.score(diabetes_X_test, diabetes_y_test)
	hyper[alpha] = score

print(hyper)

##################Model selection: choosing estimators and their parameters
#http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
#SVM
from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

#k fold cross validation DUMMY example
from sklearn.model_selection import KFold, cross_val_score
X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
k_fold = KFold(n_splits=5)
for train_indices, test_indices in k_fold.split(X):
	print('Train: %s | test: %s' % (train_indices, test_indices))

#k_fold cross valdation
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in k_fold.split(X_digits)]

# same thing using built in method
## n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
cross_val_score(estimator = svc, X = X_digits, y = y_digits, cv=k_fold, n_jobs=-1)

# Grid-Search
from sklearn.model_selection import GridSearchCV, cross_val_score
Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1)
clf.fit(X=X_digits[:1000], y=y_digits[:1000])
print(clf.best_score_)
print(clf.best_estimator_.C)
# Prediction performance on test set is not as good as on train set
print(clf.score(X_digits[1000:], y_digits[1000:]))


#cross validated estimators 
"""Cross-validation to set a parameter can be done more efficiently on an algorithm-by-algorithm basis. 
This is why, for certain estimators, scikit-learn exposes Cross-validation: evaluating estimator 
performance estimators that set their parameter automatically by cross-validation:"""
from sklearn import linear_model, datasets
lasso = linear_model.LassoCV(cv=3)
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
lasso.fit(X_diabetes, y_diabetes)
print(lasso.alpha_)


#Unsupervised Learning
#http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html
from sklearn import cluster, datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris) 
print(k_means.labels_[::10])
print(y_iris[::10])

#Connectivity-constrained clustering
import matplotlib.pyplot as plt
from skimage.data import coins
from skimage.transform import rescale
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage.filters import gaussian_filter

# #############################################################################
# Generate data
orig_coins = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect")

X = np.reshape(rescaled_coins, (-1, 1))

# #############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*rescaled_coins.shape)


###############################################Pipelining
#http://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
                         max_iter=10000, tol=1e-5, random_state=0)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [5, 20, 30, 40, 50, 64],
    'logistic__alpha': np.logspace(-4, 4, 5),
}
search = GridSearchCV(estimator = pipe, param_grid = param_grid, iid=False, cv=5,return_train_score=False)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)








