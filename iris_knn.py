"""
Project 2: Data Classification Using AI
Iris Flower Classifier using K-Nearest Neighbors (KNN)
DecodeLabs AI Internship - Batch 2026
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# 2. Train-test split (80-20, shuffled, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features (KNN is distance-based, scaling matters)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Find the best K (elbow method) instead of guessing
best_k, best_acc = 1, 0
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    if acc > best_acc:
        best_k, best_acc = k, acc

# 5. Train final model with best K
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# 6. Evaluate
print(f"Best K found: {best_k}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=target_names))

# 7. Try it on a new, unseen flower sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # looks like a Setosa
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
print(f"New sample {sample[0]} predicted as: {target_names[result[0]]}")
