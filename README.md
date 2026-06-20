# 🌍 Earthquake Classification — Random Forest, KNN & Naive Bayes

Binary classification project predicting **tsunami occurrence** from earthquake features using multiple ML models with hyperparameter tuning — on 782 real earthquake records (2001–2023).

---

## 🎯 Problem Statement

Given earthquake sensor data, classify whether an earthquake will trigger a **tsunami** or not:
- **0** = No tsunami
- **1** = Tsunami

- **Dataset**: 782 earthquakes (2001–2023), 19 features
- **Target**: `tsunami` (binary classification)
- **Class balance**: 387 vs 387 (balanced via resampling)
- **Train/Test split**: 80/20

---

## 🔄 ML Pipeline

### 1. EDA
- Missing value analysis, correlation heatmap
- Feature distributions, outlier detection

### 2. Preprocessing
- Median imputation → IQR outlier capping → RobustScaler
- OneHotEncoding (`alert`, `magType`, `continent`)
- Class balancing (387 per class)

### 3. Model Comparison

| Model | Accuracy |
|-------|----------|
| GaussianNB | 78% |
| KNeighborsClassifier | 79% |
| **RandomForestClassifier (tuned)** | **91%** |

### 4. Hyperparameter Tuning

#### GridSearchCV — RandomForest
```
Best Parameters:
  class_weight  : balanced
  max_depth     : 20
  n_estimators  : 200
```

#### RandomizedSearchCV — RandomForest (50 candidates, 5-fold CV)
```
Best Parameters:
  n_estimators     : 100
  min_samples_split: 2
  min_samples_leaf : 1
  max_depth        : 15
  class_weight     : None
```

### 5. Best Model Results — RandomForestClassifier

Confusion Matrix:
```
[[81 10]
 [ 7 59]]
```

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| No Tsunami (0) | 0.93 | 0.87 | 0.90 |
| Tsunami (1) | 0.86 | 0.93 | 0.89 |
| **Accuracy** | | | **91%** |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn (RandomForest, KNN, GaussianNB) |
| Tuning | GridSearchCV, RandomizedSearchCV |
| Evaluation | Confusion Matrix, Classification Report |
| Preprocessing | RobustScaler, SimpleImputer, OneHotEncoder |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno ydata-profiling
```

Open `V2_Test_earthquake_classifier_week4_sub.ipynb` in Jupyter Notebook and run all cells.

---

## 📁 Project Structure

```
earthquake-classification/
│
├── V2_Test_earthquake_classifier_week4_sub.ipynb   # Main notebook
├── earthquake_data.csv                              # Dataset
└── README.md
```

---

## 🔗 Related Earthquake Projects

- [Earthquake Magnitude Regression](https://github.com/angelaadida/earthquake-magnitude-regression)
- [Earthquake KMeans Clustering](https://github.com/angelaadida/earthquake-kmeans-clustering)
- [Earthquake Hierarchical Clustering](https://github.com/angelaadida/earthquake-hierarchical-clustering)
- [Earthquake DBSCAN Clustering](https://github.com/angelaadida/earthquake-dbscan-clustering)

---

## 👩‍💻 Author

**Angela** — Data Scientist | AI • ML • GenAI • RAG  
📍 Kuala Lumpur, Malaysia  
🔗 [GitHub](https://github.com/angelaadida)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
