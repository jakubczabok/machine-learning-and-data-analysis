import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv('/content/znormalizowane.csv', sep=",")

for column in data.columns:
    if not pd.api.types.is_numeric_dtype(data[column]):
        non_numeric_rows = data[~data[column].apply(lambda x: isinstance(x, (int, float)))]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

assert np.issubdtype(X.dtype, np.number), "X contains non-numeric values."
assert np.issubdtype(y.dtype, np.number), "y contains non-numeric values."

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# base models definition (logistic regression and gradient boosting)
log_reg = LogisticRegression(C=1, max_iter=100, solver='liblinear')
gboost = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=50)

# training base models
log_reg.fit(X_train_scaled, y_train)
gboost.fit(X_train_scaled, y_train)

log_reg_train_preds = cross_val_predict(log_reg, X_train_scaled, y_train, cv=5, method='predict_proba')
gboost_train_preds = cross_val_predict(gboost, X_train_scaled, y_train, cv=5, method='predict_proba')

meta_X_train = np.hstack((log_reg_train_preds, gboost_train_preds))

# meta mode training
meta_model = RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=42)
meta_model.fit(meta_X_train, y_train)

log_reg_test_preds = log_reg.predict_proba(X_test_scaled)
gboost_test_preds = gboost.predict_proba(X_test_scaled)

meta_X_test = np.hstack((log_reg_test_preds, gboost_test_preds))

meta_test_preds = meta_model.predict(meta_X_test)
meta_test_probs = meta_model.predict_proba(meta_X_test)

accuracy = accuracy_score(y_test, meta_test_preds)

if len(np.unique(y)) == 2:
    auc = roc_auc_score(y_test, meta_test_probs[:, 1]) 
else:
    auc = roc_auc_score(y_test, meta_test_probs, multi_class='ovr')  
f1 = f1_score(y_test, meta_test_preds, average='weighted')

print(f"Accuracy: {accuracy:.5f}")
print(f"AUC: {auc:.5f}")
print(f"F1 Score: {f1:.5f}")

# roc curve
if len(np.unique(y)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, meta_test_probs[:, 1])
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


# K-Fold Cross-Validation
cv_scores = cross_val_score(meta_model, meta_X_train, y_train, cv=5, scoring='accuracy')
print(f"K-Fold Cross-Validation Accuracy: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")

# Stratified Shuffle Split Cross-Validation
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
sss_scores = cross_val_score(meta_model, meta_X_train, y_train, cv=sss, scoring='accuracy')
print(f"Stratified Shuffle Split Cross-Validation Accuracy: {np.mean(sss_scores):.5f} (+/- {np.std(sss_scores):.5f})")
