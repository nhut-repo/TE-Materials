import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Generate some data
X, y = ... 

# Create the model
model = xgb.XGBClassifier()

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)

# Fit the model
grid_search.fit(X, y)

# Output the best parameters
print("Best parameters found: ", grid_search.best_params_)