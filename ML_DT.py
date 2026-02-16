import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
# Assuming df is your DataFrame containing the features and target variables.

def tune_decision_tree(df, target):
    # Split the data into features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Define the model
    dt = DecisionTreeClassifier()
    
    # Define hyperparameter search space
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define the grid search
    grid_search = GridSearchCV(estimator=dt,
                                 param_grid=param_grid,
                                 scoring='accuracy',
                                 cv=5,
                                 verbose=1,
                                 n_jobs=-1)
                                 
    # Fit the grid search
    grid_search.fit(X, y)

    # Print the best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)
    
    # Optionally evaluate on the test set
    y_pred = grid_search.predict(X)
    print(classification_report(y, y_pred))
    print('Accuracy:', accuracy_score(y, y_pred))

# Example usage for targets EC, SC, TC
# tune_decision_tree(your_data_frame, 'EC')
# tune_decision_tree(your_data_frame, 'SC')
# tune_decision_tree(your_data_frame, 'TC')