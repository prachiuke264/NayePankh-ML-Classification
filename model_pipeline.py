import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("⚙️ INITIALIZING NAYEPANKH DATA PIPELINE...")

# Simulate data
np.random.seed(42)
n_samples = 1000
user_age = np.random.randint(18, 60, n_samples)
past_donations = np.random.poisson(lam=1.5, size=n_samples)
interaction_time = np.random.exponential(scale=120, size=n_samples) + 30
active_days = np.random.randint(1, 365, n_samples)

success_prob = 0.1 + (past_donations * 0.2) + (interaction_time * 0.001) - (active_days * 0.0005)
y_target = np.where(success_prob > 0.5, 1, 0)

df_ngo = pd.DataFrame({
    'user_age': user_age,
    'past_donations_count': past_donations,
    'interaction_time_sec': interaction_time,
    'active_days': active_days
})

X_train, X_test, y_train, y_test = train_test_split(df_ngo, y_target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
model_svm = SVC(kernel='rbf', random_state=42)
model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)

model_svm.fit(X_train_scaled, y_train)
model_xgb.fit(X_train, y_train)

# Accuracy Check
acc_svm = accuracy_score(y_test, model_svm.predict(X_test_scaled))
acc_xgb = accuracy_score(y_test, model_xgb.predict(X_test))

print("\n--- PRODUCTION MODEL PERFORMANCE REPORT ---")
print(f"✅ Support Vector Machine (SVM) Accuracy: {acc_svm * 100:.2f}%")
print(f"✅ XGBoost Classifier Accuracy: {acc_xgb * 100:.2f}%")

print("\n--- DETAILED CLASSIFICATION REPORT (XGBOOST) ---")
print(classification_report(y_test, model_xgb.predict(X_test)))
