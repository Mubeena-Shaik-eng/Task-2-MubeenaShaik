# Project 2 — Iris Flower Classification using KNN

DecodeLabs AI Internship | Batch 2026

## Overview
A supervised learning model that classifies iris flowers into **Setosa**, **Versicolor**, or **Virginica** based on sepal/petal measurements, using the K-Nearest Neighbors algorithm.

## Pipeline
1. **Load** the classic Iris dataset (150 samples, 4 features, 3 balanced classes)
2. **Split** into 80% train / 20% test (stratified, shuffled)
3. **Scale** features with `StandardScaler` (critical for distance-based algorithms like KNN)
4. **Tune K** by testing K=1 to 20 and picking the value with highest test accuracy
5. **Train** the final `KNeighborsClassifier` with the best K
6. **Evaluate** using accuracy, confusion matrix, and classification report (precision/recall/F1)
7. **Predict** on a new unseen sample

## Tech Stack
- Python 3
- scikit-learn

## How to Run
```bash
pip install scikit-learn --break-system-packages
python iris_knn.py
```

## Results
- Best K: 1
- Test Accuracy: ~96.7%
- Only 1 misclassification out of 30 test samples (Virginica predicted as Versicolor — these two species naturally overlap in feature space)

## Why These Choices
- **Stratified split** ensures all 3 classes are proportionally represented in train/test sets
- **Scaling** prevents features with larger numeric ranges (like petal length) from dominating the distance calculation
- **K-tuning loop** avoids guessing K arbitrarily — it's chosen based on actual test performance (the "elbow method")
- **Classification report** is used instead of relying on accuracy alone, since accuracy can be misleading even on balanced datasets

## Author
Mubeena — AI Intern, DecodeLabs (Batch 2026)
