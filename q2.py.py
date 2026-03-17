# ---------------- IMPORTS ----------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

# ---------------- PREP DATA ----------------
X_train = x_train.reshape(x_train.shape[0], -1)
X_test  = x_test.reshape(x_test.shape[0], -1)

y_train_lbl = np.argmax(y_train, axis=1)
y_test_lbl  = np.argmax(y_test, axis=1)

#  SMALL SUBSET (FAST & ACCEPTABLE)
X_sub = X_train[:5000]
y_sub = y_train_lbl[:5000]

# ---------------- GRID SEARCH ----------------
param_grid = {"C": [0.1, 1, 10]}

grid = GridSearchCV(
    LogisticRegression(
        solver="saga",
        max_iter=300,
        n_jobs=-1
    ),
    param_grid,
    cv=2,
    scoring="accuracy"
)

grid.fit(X_sub, y_sub)
best_C = grid.best_params_["C"]

print("Best C:", best_C)

# ---------------- FINAL MODEL ----------------
model = LogisticRegression(
    C=best_C,
    solver="saga",
    max_iter=300,
    n_jobs=-1
)

model.fit(X_sub, y_sub)
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
print("Accuracy :", accuracy_score(y_test_lbl, y_pred))
print("Precision:", precision_score(y_test_lbl, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test_lbl, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test_lbl, y_pred, average="weighted"))

# ---------------- DECISION BOUNDARY (PCA) ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sub)

clf_2d = LogisticRegression(
    C=best_C,
    solver="saga",
    max_iter=300
)
clf_2d.fit(X_pca, y_sub)

xx, yy = np.meshgrid(
    np.linspace(X_pca[:,0].min()-1, X_pca[:,0].max()+1, 200),
    np.linspace(X_pca[:,1].min()-1, X_pca[:,1].max()+1, 200)
)

Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_sub, s=5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Logistic Regression Decision Boundary (PCA)")
plt.show()
