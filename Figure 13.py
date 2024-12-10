import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import rdata 
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA



# code for figures 13 a) and b)

r_data = rdata.parser.parse_file("zipCodeAllDigits.RData")
as_dict = rdata.conversion.convert(r_data)
X_train = as_dict["train.X"]
y_train = as_dict["train.y"]
X_test = as_dict["test.X"]
y_test = as_dict["test.y"]

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)



fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(131)
gamma_vals =  [0.001, 0.01, 0.1, 1]
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

plt.show()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(131)
gamma_vals = [0.001, 0.01, 0.1, 1]
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
ax1.legend(fontsize=16, loc = 0, bbox_to_anchor=(0.5, 0., 0.5, 0.5))

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


plt.show()








