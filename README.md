# 📊 NayePankh Operations Optimization Pipeline
> **Supervised Predictive Framework comparing Support Vector Machines & XGBoost Ensembles**

## 💻 Project Overview
This project delivers a production-grade machine learning classification pipeline designed to analyze user behavior data and forecast active donation and volunteer conversion probabilities.

## 🛠️ Model Architecture & Comparison
The pipeline evaluates two distinct machine learning paradigms trained on balanced telemetry vectors:
- **Support Vector Machine (SVM):** Implemented using a Non-Linear Radial Basis Function (RBF) Kernel on standardized features.
- **XGBoost Classifier Engine:** Built via a tree-boosting ensemble framework (`max_depth=4`, `learning_rate=0.05`).

### 📈 Metrics Summary
- **SVM Validation Accuracy:** 96.00%
- **XGBoost Validation Accuracy:** 96.00%

## 🚀 How to Run the Pipeline
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python model_pipeline.py`
