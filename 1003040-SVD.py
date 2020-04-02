import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn import preprocessing


X = datasets.load_diabetes().data
X = preprocessing.normalize(X, axis=0)
pca = decomposition.PCA()
pca.fit(X)
print("The V matrix is ", np.matrix.transpose(pca.components_))
print("The singular values are  ", pca.singular_values_)
print("The 3 most important components for the first 10 data-points is", pca.transform(X)[:10, :3])