import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. UNSUPERVISED CLUSTERING & PCA
# ==========================================
print("--- 1. UNSUPERVISED LEARNING & DIMENSIONALITY REDUCTION ---")
# Synthetic dataset generation
X_cluster, _ = make_blobs(n_samples=500, centers=4, n_features=5, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster)
kmeans_silhouette = silhouette_score(X_cluster, kmeans_labels)

# Hierarchical Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_clustering.fit_predict(X_cluster)
agg_silhouette = silhouette_score(X_cluster, agg_labels)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)
explained_var_ratio = np.sum(pca.explained_variance_ratio_)

print(f"K-Means Silhouette Score:               {kmeans_silhouette:.4f}")
print(f"Hierarchical Clustering Silhouette Score: {agg_silhouette:.4f}")
print(f"PCA Total Explained Variance Ratio:     {explained_var_ratio:.4f}")

# ==========================================
# 2. MODEL EVALUATION & HYPERPARAMETER TUNING
# ==========================================
print("\n--- 2. MODEL EVALUATION & HYPERPARAMETER TUNING ---")
X_eval, y_eval = make_classification(n_samples=500, n_features=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_preds = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

# Cross-Validation Score
cv_scores = cross_val_score(best_model, X_eval, y_eval, cv=5, scoring='f1')

# Metrics Evaluation
cm = confusion_matrix(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
roc_auc = roc_auc_score(y_test, y_probs)

print(f"Best Hyperparameters:  {grid_search.best_params_}")
print(f"5-Fold CV Mean F1-Score: {np.mean(cv_scores):.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1-Score:             {f1:.4f}")
print(f"ROC-AUC Score:        {roc_auc:.4f}")

# Export clustering summary to CSV deliverable
summary_data = {
    "Algorithm": ["K-Means Clustering", "Hierarchical Clustering", "PCA (2 Components)"],
    "Primary Metric": ["Silhouette Score", "Silhouette Score", "Explained Variance Ratio"],
    "Score": [round(kmeans_silhouette, 4), round(agg_silhouette, 4), round(explained_var_ratio, 4)]
}
pd.DataFrame(summary_data).to_csv("clustering_evaluation_results.csv", index=False)