import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("C:/users/alikh/Documents/melb_data.csv")

# Select the subset of predictors
cols_to_use = ["Rooms", "Distance", "Landsize", "BuildingArea", "YearBuilt"]
X = data[cols_to_use]

y = data.Price

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# Set the number of models in the ensemble
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

# early_stopping_rounds - automatically find the ideal value for n_estimators
# eval_set - aside data for calculating the validation scores
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)],
             verbose=False)

# learning_rate - multiply the predictions from each model by a small number before adding them in
# Each tree we add to the ensemble helps us less, so we can set a higher value for n_estimators without overfitting
# Small learning rate and large number of estimators will yield more accurate XGBoost models, but it will take the model longer to train
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)],
             verbose=False)

# n_jobs - the number of cores on your computer
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)],
             verbose=False)
