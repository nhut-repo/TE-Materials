import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# HistGradientBoosting model
hist_model = HistGradientBoostingClassifier(random_state=42)
hist_model.fit(X_train, y_train)

# RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Getting feature importances
hist_importances = hist_model.feature_importances_
rf_importances = rf_model.feature_importances_

# Displaying importance
print("Feature importances from HistGradientBoosting:", hist_importances)
print("Feature importances from RandomForest:", rf_importances)