# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

# 1 Loading the precoessed data
df = pd.read_csv("../data/processed/train_models_sel.csv")
df.head()

# 2 Data Splitting
# Assuming df is your DataFrame
X = df.iloc[:, 1:]  # Features
y = df.iloc[:, 0]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)
X_train.head()
y_train.head()


# 3 Model Selection
def evaluate_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate different classification models and compare their performance.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    X_test (array-like): Testing features.
    y_test (array-like): Testing labels.

    Returns:
    pd.DataFrame: A DataFrame containing model names, training and testing performance metrics.
    """

    # Initialize models
    models = [
        ("Decision Tree", DecisionTreeRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("XGBRegressor", XGBRegressor(tree_method="gpu_hist", device="cuda")),
    ]

    # Initialize result DataFrame
    result = pd.DataFrame(
        columns=[
            "Model",
            "Train_MSE",
            "Train_MAE",
            "Train_R2",
            "Test_MSE",
            "Test_MAE",
            "Test_R2",
        ]
    )

    for model_name, model in models:
        if model_name == "XGBRegressor":
            # Train the model
            model.fit(X_train.values, y_train)

            # Cross-validation
            kfold = KFold(n_splits=3, shuffle=True, random_state=21)
            cross_val_MSE = -cross_val_score(
                model,
                X_train.values,
                y_train,
                cv=kfold,
                scoring="neg_mean_squared_error",
            ).mean()
            cross_val_MAE = -cross_val_score(
                model,
                X_train.values,
                y_train,
                cv=kfold,
                scoring="neg_mean_absolute_error",
            ).mean()
            cross_val_R2 = cross_val_score(
                model, X_train.values, y_train, cv=kfold, scoring="r2"
            ).mean()

            # Test the model
            predictions = model.predict(X_test.values)
            test_MSE = mean_squared_error(y_test, predictions)
            test_MAE = mean_absolute_error(y_test, predictions)
            test_R2 = r2_score(y_test, predictions)
        else:
            # Train the model
            model.fit(X_train, y_train)

            # Cross-validation
            kfold = KFold(n_splits=3, shuffle=True, random_state=21)
            cross_val_MSE = -cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error"
            ).mean()
            cross_val_MAE = -cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error"
            ).mean()
            cross_val_R2 = cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="r2"
            ).mean()

            # Test the model
            predictions = model.predict(X_test)
            test_MSE = mean_squared_error(y_test, predictions)
            test_MAE = mean_absolute_error(y_test, predictions)
            test_R2 = r2_score(y_test, predictions)

        # Store results
        result.loc[len(result)] = [
            model_name,
            cross_val_MSE,
            cross_val_MAE,
            cross_val_R2,
            test_MSE,
            test_MAE,
            test_R2,
        ]
    return result


evaluate_models(X_train, y_train, X_test, y_test)

# XGBoostRegressor it is!

# 4 Model Tuning (XGBoostRegressor)
# Initialize an XGBoostRegressor and use GPU
xgbr_model = XGBRegressor(
    tree_method="gpu_hist", device="cuda"
)  # GPU-based histogram algorithm

# Fine-tuning parameters
param_grid = {
    "max_depth": [3, 6, 9],  # More values for max_depth
    "subsample": [0.8, 1.0],  # More values for subsample
    "n_estimators": [100, 200, 300],  # More values for n_estimators
    "learning_rate": [0.1, 0.01, 0.001],  # More values for learning_rate
    "min_child_weight": [1, 5, 10],  # More values for min_child_weight
    "random_state": [42],  # Fixed random state for reproducibility
    "reg_alpha": [0, 0.1, 0.01],  # More values for reg_alpha
    "reg_lambda": [1, 0.5, 0.01],  # More values for reg_lambda
}

# Initialize Stratified K-Fold cross-validation
kfold = KFold(n_splits=3)

grid_search = GridSearchCV(
    estimator=xgbr_model,
    param_grid=param_grid,
    cv=kfold,
    scoring="neg_mean_absolute_error",
    verbose=2,
    n_jobs=-1,
)
grid_search.fit(X_train.values, y_train)

# Access the best estimator directly
best_estimator_params = grid_search.best_estimator_.get_params()
best_estimator_params

# Loading the whole precoessed data
df = pd.read_csv("../data/processed/train_fit.csv")
df.head()

# Splitting the new data
# Assuming df is your DataFrame
X = df.iloc[:, 1:]  # Features
y = df.iloc[:, 0]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)
X_train.head()
y_train.head()

# Now you can create a new XGBoostRegressor using the best parameters
best_xgbr_model = XGBRegressor(
    **best_estimator_params
)  # tree_method="gpu_hist", device="cuda" these are already included **best_estimator_params
best_xgbr_model.fit(X_train.values, y_train)

# Training set
kfold = KFold(n_splits=3)
cross_val_MSE = -cross_val_score(
    best_xgbr_model, X_train.values, y_train, cv=kfold, scoring="neg_mean_squared_error"
).mean()
cross_val_MAE = -cross_val_score(
    best_xgbr_model,
    X_train.values,
    y_train,
    cv=kfold,
    scoring="neg_mean_absolute_error",
).mean()
cross_val_R2 = cross_val_score(
    best_xgbr_model, X_train.values, y_train, cv=kfold, scoring="r2"
).mean()

# Test set
predictions = best_xgbr_model.predict(X_test.values)
test_MSE = mean_squared_error(y_test, predictions)
test_MAE = mean_absolute_error(y_test, predictions)
test_R2 = r2_score(y_test, predictions)


print(f"Train_MSE: {cross_val_MSE:.4f}")
print(f"Train_MAE: {cross_val_MAE:.4f}")
print(f"Train_R2: {cross_val_R2:.4f}")
print(f"Test_MSE: {test_MSE:.4f}")
print(f"Test_MAE: {test_MAE:.4f}")
print(f"Test_R2: {test_R2:.4f}")

# Retrain the model on full training data
best_xgbr_model.fit(X.values, y)

# 5 Feature Importance
# Create a subplot with desired aspect ratio
fig, ax = plt.subplots(figsize=(5, 10))  # Adjust the size here (width, height)

# Plot feature importance
importances = best_xgbr_model.feature_importances_
indices = np.argsort(importances)[::-1]
ax.barh(range(X.shape[1]), importances[indices], align="center")
ax.set_yticks(range(X.shape[1]))
ax.set_yticklabels([X.columns[i] for i in indices])
ax.invert_yaxis()  # Invert y-axis to have the most important feature at the top
ax.set_xlabel("Feature Importance")
ax.set_title("XGBoost Regressor Feature Importance")

plt.show()

# 7 Save Model
ref_cols = list(X.columns)
target = "fare_amount"
joblib.dump(value=[best_xgbr_model, ref_cols, target], filename="../models/model.pkl")
