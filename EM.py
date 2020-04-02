import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as LA
from matplotlib.patches import Ellipse
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
K = 3
NUMDATAPTS = 150
X, y = make_blobs(n_samples=NUMDATAPTS,  centers=K, shuffle=False, random_state=0,   cluster_std=0.6)
g1 = np.asarray([[2.0, 0], [-0.9, 1]])
g2 = np.asarray([[1.4, 0], [0.5, 0.7]])
mean1 = np.mean(X[: int(NUMDATAPTS/K)])
mean2 = np.mean(X[int(NUMDATAPTS/K): 2*int(NUMDATAPTS/K)])
X[:int(NUMDATAPTS/K)] = np.einsum("nj, ij ->ni", X[: int(NUMDATAPTS/K)]-mean1, g1) + mean1
X[int(NUMDATAPTS/K): 2*int(NUMDATAPTS/K)] = np.einsum(" nj, ij -> ni ", X[int(NUMDATAPTS/K): 2*int(NUMDATAPTS/K)] - mean2, g2) + mean2

X[:, 1] -= 4
#Part a
np.random.seed(123)
mu = X[np.random.choice(NUMDATAPTS, K, replace=False)]
cov = np.array([np.eye(2) for _ in range(K)])
pi = np.array([1/K for _ in range(K)])

#Part b

def multivariate_pdf(x, mean,Cov):
    return multivariate_normal(mean=mean, cov=Cov).pdf([x])

def E_step(X, Mu,Cov,p):

    n, d = X.shape
    K, _ = Mu.shape
    post = np.zeros((n, K))
    likelihood = 0
    for i in range(n):
        for j in range(K):
            post[i, j] = p[j]*multivariate_pdf(X[i, :], Mu[j, :], Cov[j])
        likelihood += np.log(np.sum(post[i, :]))
    for row in post:
        row /= np.sum(row)

    return post, likelihood


def M_step(X, post):

    X, post = np.array(X),np.array(post)
    n, d = X.shape
    _, K = post.shape
    Mu, p, Cov = np.zeros((K, d)), [], []
    for k in range(K):
        Mu[k,:] = (X.T @ post[:, k])/np.sum(post[:, k])
        p.append(np.mean(post[:, k]))
        covariance_mat = 0
        for i in range(n):
            covariance_mat += (post[i, k] * np.outer(X[i, :] - Mu[k, :], X[i, :] - Mu[k, :]))/np.sum(post[:, k])
        Cov.append(covariance_mat)

    return Mu, np.array(Cov), np.array(p)

def run(X, Mu ,Cov, p):
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None
    LL = None
    itertaion = 0
    while prev_LL is None or LL - prev_LL > 1e-3 * abs(LL):
        prev_LL = LL
        post, LL = E_step(X, Mu, Cov, p)
        Mu, Cov, p = M_step(X, post)
        print(LL)
        plot_result(Mu,Cov,itertaion,LL,post)
        itertaion += 1
    return "Converged!"

def plot_result(Mu, Cov,iter,ll, gamma=None) :
    ax = plt.subplot(111, aspect='equal')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5] )
    ax.scatter(X[:, 0], X[:, 1], c=gamma, s=50, cmap=None)
    ax.set_title(" Plot for {} iteration".format(iter)+" with loglikelihood "+str(ll))
    for k in range(K):
        l, v = LA. eig(Cov[k])
        theta = np.arctan(v[1, 0] / v[0, 0])
        e = Ellipse( (Mu[k, 0], Mu[k, 1]), 6 * l[0], 6*l[1], theta * 180 / np.pi)
        e.set_alpha(0.5)
        ax.add_artist(e)
    #ax.title("At iteration {}".format(iter), "the log-likelihood is", ll)
    plt.savefig(" Plot for {} iteration".format(iter)+" with loglikelihood "+str(ll) + ".png")
    plt.show()

print(run(X, mu, cov, pi))

# Part e
plt.figure(figsize=(12, 12))
random_state = 170
y_pred = KMeans(n_clusters=K, random_state=random_state).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Result using K-mean")
plt.savefig("Result using K-mean.png")
plt.show()