import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import rdata 
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import time

r_data = rdata.parser.parse_file("zipCodeAllDigits.RData")
as_dict = rdata.conversion.convert(r_data)
X_train = as_dict["train.X"]
y_train = as_dict["train.y"]
X_test = as_dict["test.X"]
y_test = as_dict["test.y"]

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Use the PCA reduced data for the SVM
# X_train = X_train_pca
# X_test = X_test_pca

# Plot the dimension reduced data on a scatter graph
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', alpha=0.7)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes", fontsize = 20)
ax.add_artist(legend1)
ax.set_xlabel("Principal Component 1", fontsize = 26)
ax.set_ylabel("Principal Component 2", fontsize = 26)
plt.show()

# Perform k-fold cross-validation to find optimal parameters for SVM with kernel of choice
param_grid = {
    'C': np.logspace(-1, 3, 30),
    'degree': [1, 2, 3, 4, 5, 6]
}
grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

print("Best parameters found: ", grid_search.best_params_)

# Plot the increase in fit time as C increases for the polynomial kernel for degrees [1, 2, 3]

C_values = np.logspace(-1, 1, 30)
fit_times = {degree: [] for degree in [1,2,3]}

for degree in [1,2,3]:
    for C in C_values:
        clf = svm.SVC(kernel='poly', C=C, gamma=1, degree=degree)
        start_time = time.time()
        clf.fit(X_train_pca, y_train)
        end_time = time.time()
        fit_times[degree].append(end_time - start_time)

fig, ax = plt.subplots(figsize=(10, 6))
for degree in [1,2,3]:
    ax.plot(C_values, fit_times[degree], label=f'Degree {degree}')

ax.set_xscale('log')
ax.set_xlabel('C value (log scale)', fontsize=26)
ax.set_ylabel('Fit time (seconds)', fontsize=26)
ax.legend(fontsize=20)
plt.show()

# Plot the decision boundaries for kernels of choice

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
gamma_vals = [1, 2, 3, 4]
for idx, gamma_val in enumerate(gamma_vals):
    clf = svm.SVC(kernel="poly", C= 41.8, degree=gamma_val)
    clf.fit(X_train_pca, y_train)
    
    # Plot the decision boundary
    DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, plot_method='pcolormesh', shading='auto', cmap=plt.cm.Paired, ax=axes[idx], alpha=0.5)
    
    # Plot also the training points
    axes[idx].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=30, cmap=plt.cm.Paired, edgecolors='k')
    axes[idx].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, s=30, cmap=plt.cm.Paired, edgecolors='k')
    axes[idx].set_title(f'd = {gamma_val}', fontsize = 18)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
gamma_vals = [0.01, 0.1, 1, 10]
for idx, gamma_vals in enumerate(gamma_vals):
    clf = svm.SVC(kernel="rbf", C= 30.4, gamma = gamma_vals)
    clf.fit(X_train_pca, y_train)
    
    # Plot the decision boundary
    DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, plot_method='pcolormesh', shading= 'auto',cmap=plt.cm.Paired, ax=axes[idx], alpha=0.5)
    
    # Plot also the training points
    axes[idx].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=30, cmap=plt.cm.Paired, edgecolors='k')
    axes[idx].scatter(X_test_pca[:,0], X_test_pca[:,1], c=y_test, s=30, cmap=plt.cm.Paired, edgecolors='k')
    axes[idx].set_title(f'gamma = {gamma_vals}', fontsize = 18)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
gamma_vals = [0.001, 0.01, 0.1, 1]
for idx, gamma_vals in enumerate(gamma_vals):
    clf = svm.SVC(kernel="sigmoid", C= 11.7, gamma = gamma_vals)
    clf.fit(X_train_pca, y_train)
    
    # Plot the decision boundary
    DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, plot_method='pcolormesh', shading= 'auto',cmap=plt.cm.Paired, ax=axes[idx], alpha=0.5)
    
    # Plot also the training points
    axes[idx].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=30, cmap=plt.cm.Paired, edgecolors='k')
    axes[idx].scatter(X_test_pca[:,0], X_test_pca[:,1], c=y_test, s=30, cmap=plt.cm.Paired, edgecolors='k')
    axes[idx].set_title(f'gamma = {gamma_vals}', fontsize = 18)

plt.tight_layout()
plt.show()