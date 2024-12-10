import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import rdata 
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV

# we create two clusters of random points
n_samples_1 = 100
n_samples_2 = 70
centers = [[-0.3, 0.0], [1.0, 1.5]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)
n = n_samples_1 + n_samples_2

#split the data into training and testing

np.random.seed(4)
test_idx = np.random.randint(0, n, int(0.2*n))
X_train = np.delete(X, test_idx, 0)
y_train = np.delete(y, test_idx)
X_test = X[test_idx]
y_test = y[test_idx]

# Plots for Figures 5 a) and b)
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(131)
gamma_vals =  [0.1, 0.5, 1, 10, 50]
d = [1, 2, 3, 4]


for j, d_vals in enumerate(d):
    MCE = []
    for i in np.logspace(-1, 1, 20):
        clf = svm.SVC(kernel="poly", C= i, gamma=1, degree=d_vals)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        mce_train = (y_pred != y_train).mean()
        mce_test = (clf.predict(X_test) != y_test).mean()
        MCE.append(mce_train)
    ax1.plot(np.log10(np.logspace(-1, 1, 20)), MCE, label="d = " + str(d_vals))

ax1.set_title("Polynomial", fontsize=22)
ax1.set_xlabel("Parameter $log_{10}C$", fontsize=22)
ax1.set_ylabel("$MCE_{train}$", fontsize=22)
ax1.legend(fontsize=16)


ax2 = fig.add_subplot(132)
for j, G_vals in enumerate(gamma_vals):
    MCE = []
    for i in np.logspace(-1, 2, 20):
        clf = svm.SVC(kernel="rbf", C= i, gamma= G_vals)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        mce_train = (y_pred != y_train).mean()
        mce_test = (clf.predict(X_test) != y_test).mean()
        MCE.append(mce_train)
    ax2.plot(np.log10(np.logspace(-1, 2, 20)), MCE, label="$\gamma$ = " + str(G_vals))

ax2.set_title("RBF", fontsize=22)
ax2.set_xlabel("Parameter $log_{10}C$", fontsize=22)
ax2.set_ylabel("$MCE_{train}$", fontsize=22)
ax2.legend(fontsize=16)

ax3 = fig.add_subplot(133)
for j, G_vals in enumerate(gamma_vals):
    MCE = []
    for i in np.logspace(-1, 2, 20):
        clf = svm.SVC(kernel="sigmoid", C= i, gamma=G_vals)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        mce_train= (y_pred != y_train).mean()
        mce_test = (clf.predict(X_test) != y_test).mean()
        MCE.append(mce_train)
    ax3.plot(np.log10(np.logspace(-1, 2, 20)), MCE, label="$\gamma$ = " + str(G_vals))

ax3.set_title("Sigmoid", fontsize=22)
ax3.set_xlabel("Parameter $log_{10}C$", fontsize=22)
ax3.set_ylabel("$MCE_{train}$", fontsize=22)
ax3.legend(fontsize=16)
# Set the same y-axis scale for all subplots
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
ax3.set_ylim(y_min, y_max)
plt.show()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(131)
gamma_vals = [0.1, 0.5, 1, 10, 50]
d = [1, 2, 3, 4]

for j, d_vals in enumerate(d):
    MCE = []
    for i in np.logspace(-1, 1, 20):
        clf = svm.SVC(kernel="poly", C= i, gamma=1, degree=d_vals)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        mce_train = (y_pred != y_train).mean()
        mce_test = (clf.predict(X_test) != y_test).mean()
        MCE.append(mce_test)
    ax1.plot(np.log10(np.logspace(-1, 1, 20)), MCE, label="d = " + str(d_vals))

ax1.set_title("Polynomial", fontsize=22)
ax1.set_xlabel("Parameter $log_{10}C$", fontsize=22)
ax1.set_ylabel("$MCE_{test}$", fontsize=22)
ax1.legend(fontsize=16)

ax2 = fig.add_subplot(132)
for j, G_vals in enumerate(gamma_vals):
    MCE = []
    for i in np.logspace(-1, 2, 20):
        clf = svm.SVC(kernel="rbf", C= i, gamma= G_vals)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        mce_train = (y_pred != y_train).mean()
        mce_test = (clf.predict(X_test) != y_test).mean()
        MCE.append(mce_test)
    ax2.plot(np.log10(np.logspace(-1, 2, 20)), MCE, label="$\gamma$ = " + str(G_vals))

ax2.set_title("RBF", fontsize=22)
ax2.set_xlabel("Parameter $log_{10}C$", fontsize=22)
ax2.set_ylabel("$MCE_{test}$", fontsize=22)
ax2.legend(fontsize=16)

ax3 = fig.add_subplot(133)
for j, G_vals in enumerate(gamma_vals):
    MCE = []
    for i in np.logspace(-1, 2, 20):
        clf = svm.SVC(kernel="sigmoid", C= i, gamma=G_vals)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        mce_train= (y_pred != y_train).mean()
        mce_test = (clf.predict(X_test) != y_test).mean()
        MCE.append(mce_test)
    ax3.plot(np.log10(np.logspace(-1, 2, 20)), MCE, label="$\gamma$ = " + str(G_vals))

ax3.set_title("Sigmoid", fontsize=22)
ax3.set_xlabel("Parameter $log_{10}C$", fontsize=22)
ax3.set_ylabel("$MCE_{test}$", fontsize=22)
ax3.legend(fontsize=16)

# Set the same y-axis scale for all subplots
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
ax3.set_ylim(y_min, y_max)

plt.show()


# Plot Figure 6
poly_mce=[]
rbf_mce=[]
sig_mce=[]
for C in np.logspace(-2, 2, 30):
    poly_svc = svm.SVC(kernel='poly', C=C, gamma=1, degree=1)
    poly_svc.fit(X_train, y_train)
    poly_mce_test = (poly_svc.predict(X_test) != y_test).mean()
    poly_mce.append(poly_mce_test)

    rbf_svc = svm.SVC(kernel='rbf', C=C, gamma = 0.115)
    rbf_svc.fit(X_train, y_train)
    rbf_mce_test = (rbf_svc.predict(X_test) != y_test).mean()
    rbf_mce.append(rbf_mce_test)

    sigmoid_svc = svm.SVC(kernel='sigmoid', C=C, gamma = 0.0489, coef0=0.01)
    sigmoid_svc.fit(X_train, y_train)
    sig_mce_test = (sigmoid_svc.predict(X_test) != y_test).mean()
    sig_mce.append(sig_mce_test)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(np.logspace(-2, 2, 30), poly_mce, label="Poly")
ax.plot(np.logspace(-2, 2, 30), rbf_mce, label="RBF")
ax.plot(np.logspace(-2, 2, 30), sig_mce, label="Sigmoid")
ax.set_xscale("log")
ax.set_xlabel("Cost Parameter C", fontsize=24)
ax.set_ylabel("$MCE_{test}$", fontsize=24)
ax.legend(fontsize =20)

plt.show()
