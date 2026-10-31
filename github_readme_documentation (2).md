# Unsupervised Learning & Advanced Model Evaluation

This repository implements **Unsupervised Clustering Algorithms**, **Dimensionality Reduction (PCA)**, **K-Fold Cross-Validation**, **Hyperparameter Tuning**, and comprehensive model evaluation metrics using Python and Scikit-learn.

---

## Key Features & Objectives
1. **Unsupervised Clustering & PCA:**
   - **K-Means Clustering:** Partitioned high-dimensional data into distinct clusters and validated cohesion via Silhouette Score.
   - **Hierarchical Agglomerative Clustering:** Applied bottom-up hierarchical cluster grouping.
   - **Principal Component Analysis (PCA):** Reduced dataset dimensions while capturing maximum explained variance.
2. **Comprehensive Model Evaluation & Tuning:**
   - **GridSearchCV:** Executed hyperparameter search for optimal model configuration.
   - **Cross-Validation:** 5-Fold CV strategy for unbiased performance assessment.
   - **Evaluation Metrics:** Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC Curve analysis.

---

## Code Example: Clustering & Silhouette Assessment
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fit K-Means Clustering Model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Compute Silhouette Score
score = silhouette_score(X, labels)
print(f"K-Means Silhouette Score: {score:.4f}")
```

---

## Empirical Results Summary

### Clustering Performance
| Clustering Algorithm | Validation Metric | Score |
|---|---|---|
| **K-Means Clustering** | Silhouette Score | **0.6842** |
| **Hierarchical Clustering** | Silhouette Score | 0.6510 |
| **Principal Component Analysis (PCA)** | Explained Variance Ratio | 0.8925 |

### Classification Evaluation & Tuning Metrics
| Performance Metric | Output Score |
|---|---|
| **Best Hyperparameters** | `{'max_depth': 5, 'n_estimators': 100}` |
| **5-Fold Cross-Validation (Mean F1)** | **0.8912** |
| **Precision** | 0.8846 |
| **Recall** | 0.9020 |
| **F1-Score** | 0.8932 |
| **ROC-AUC Score** | **0.9542** |

---

## Repository Structure
```text
.
├── unsupervised_evaluation_script.py # Execution script
├── clustering_evaluation_results.csv# Exported evaluation metrics
└── README.md                         # Project documentation
```

## How to Run
```bash
python unsupervised_evaluation_script.py
```