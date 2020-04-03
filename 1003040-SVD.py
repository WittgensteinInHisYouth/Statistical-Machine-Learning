import numpy as np
from sklearn import decomposition
from sklearn import datasets



X = datasets.load_diabetes().data
row, col = X.shape
for column_ix in range(col):
    X[:, column_ix] -= np.mean(X[:, column_ix])
    X[:, column_ix] /= np.std(X[:, column_ix])
    
pca = decomposition.PCA()
pca.fit(X)
print("The V matrix is \n", np.matrix.transpose(pca.components_))
print("The singular values are \n ", pca.singular_values_)
print("The 3 most important components for the first 10 data-points is \n", pca.transform(X)[:10, :3])
