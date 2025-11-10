import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the Dataset
housing = pd.read_csv("housing.csv")

# 2. Creating Stratified Test Set
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels = [1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis = 1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis = 1)

housing = strat_train_set.copy()     # We Will Work on copy of train set

# 3. Seperate labels & Features 

housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis = 1)

# 4. List Numerical & Categorical columns

num_attribs = housing.drop('ocean_proximity', axis = 1).columns.tolist()
cat_attribs = ['ocean_proximity']

# 5. Let's Make the Pipline

# For numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer (strategy = "median")),
    ("scaler", StandardScaler())
])

# For categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder (handle_unknown = "ignore"))
])

# Construct the Full Pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. Transform the Data

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 7. Train the model

#--------------------- Linear Regressior Model --------------------#

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Predictions on training data
lin_preds = lin_reg.predict(housing_prepared)

# Calculate RMSE manually
lin_rmse = np.sqrt(mean_squared_error(housing_labels, lin_preds))
print(f"\nLinear Regression (Training RMSE): {lin_rmse:.2f}")

# Cross-validation (10-fold)
lin_rmse_scores = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("\nLinear Regression (Cross-Validation RMSE Scores):")
print(pd.Series(lin_rmse_scores).describe())

#--------------------- Decision Tree Model ----------------------#

dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)

# Predictions on training data
dec_preds = dec_reg.predict(housing_prepared)

# Calculate RMSE manually
dec_rmse = np.sqrt(mean_squared_error(housing_labels, dec_preds))
print(f"\nDecision Tree Regression (Training RMSE): {dec_rmse:.2f}")

# Cross-validation
dec_rmse_scores = -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("\nDecision Tree Regression (Cross-Validation RMSE Scores):")
print(pd.Series(dec_rmse_scores).describe())

#--------------------- Random Forest  Model ---------------------#

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(housing_prepared, housing_labels)

# Predictions on training data
rf_preds = rf_reg.predict(housing_prepared)

# Calculate RMSE manually
rf_rmse = np.sqrt(mean_squared_error(housing_labels, rf_preds))
print(f"\nRandom Forest Regression (Training RMSE): {rf_rmse:.2f}")

# Cross-validation
rf_rmse_scores = -cross_val_score(
    rf_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("\nRandom Forest Regression (Cross-Validation RMSE Scores):")
print(pd.Series(rf_rmse_scores).describe())
