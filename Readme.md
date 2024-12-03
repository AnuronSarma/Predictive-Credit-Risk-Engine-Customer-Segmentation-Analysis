# Predictive-Credit-Risk-Engine-Customer-Segmentation-Analysis

This project focuses on analyzing and predicting loan approval statuses using supervised machine learning models. Additionally, it employs unsupervised clustering techniques to identify distinct patterns among businesses based on various financial and demographic features.

---

## 📋 **Table of Contents**

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Key Features and Insights](#key-features-and-insights)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Supervised Learning Models](#supervised-learning-models)
  - [Clustering Analysis](#clustering-analysis)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## 🔍 **Problem Statement**

The aim of this project is to:
1. Predict whether a loan application will be approved or denied using machine learning models.
2. Cluster businesses into meaningful groups based on their loan features, such as loan amount, job creation, and revolving line of credit usage, to derive actionable insights.

---

## 📊 **Dataset**

- **Source**: [Kaggle - SBA Loan Dataset](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied)
- **Size**: Over 900,000 records with 27 features.
- **Key Columns**:
  - `MIS_Status`: Loan status (Approved/Denied).
  - `Retained_Job`, `Urban_Rural`, `RevLineCr`, `Term`: Business and financial details.
  - `SBA_Guaranteed_Amount`: Loan details under the SBA guarantee.

---

## ✨ **Key Features and Insights**

- Businesses taking higher loans tend to create more jobs and have longer loan terms.
- Clustering revealed distinct groups:
  - One cluster characterized by fewer revolving lines of credit and higher urban exposure.
  - Another cluster differentiated by a high number of revolving credits.

---

## 🛠 **Methodology**

### **1. Data Preprocessing**
- Handled missing values and standardized numerical features.
- Encoded categorical variables using target encoding.
- Scaled features for clustering analysis.

### **2. Supervised Learning Models**
- **Algorithms**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Metrics**: Accuracy, Precision, Recall, F1-score.

### **3. Clustering Analysis**
- Applied **KMeans** and **DBSCAN** clustering algorithms.
- Used PCA for dimensionality reduction to visualize clusters in 2D space.
- Evaluated clusters using Silhouette Score and WCSS.

---

## 📈 **Evaluation Metrics**

- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Positive predictive value.
- **Recall**: Sensitivity or true positive rate.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **WCSS & Silhouette Score**: Metrics for clustering performance.

---

## 🏆 **Results**

- **Logistic Regression**: Accuracy - 97.78%
- **Random Forest**: Accuracy - 98.88%
- **XGBoost**: Accuracy - 98.81%
- Clustering revealed 3 distinct clusters with insights into loan approval trends and business behaviors.

---

## 📊 **Visualizations**

- Elbow plot and Silhouette score for optimal cluster selection.
- PCA scatter plots for cluster visualization.
- Word clouds and boxplots to understand categorical and numerical distributions.

---

## 🚀 **Usage**

### **Requirements**
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

---

🔧 Technologies Used

    Programming Language: Python
    Libraries: Scikit-learn, XGBoost, Matplotlib, Seaborn
    Visualization: WordClouds, PCA, Clustering Scatterplots

🌟 Future Improvements

    Incorporate additional features like industry-specific data.
    Experiment with advanced clustering algorithms like GMM.
    Automate the pipeline for real-time prediction.
