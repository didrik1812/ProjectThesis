# %% [markdown]
# # Some exploration of methods

# %%
# Load libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import (
    SelectFromModel,
    mutual_info_regression,
    SelectPercentile,
    f_regression,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import pickle
from scipy.stats import pearsonr


# %%
# Prep data for XGBoost
print("LOADING DATA")
mass_residuals = pd.read_csv("../data/mass_residuals_data_full.csv")
print("DATA LOADED")
SNP_data = mass_residuals.iloc[:, 7:]
SNP_data.fillna(0)
Y = mass_residuals.ID
SNP_data = OneHotEncoder().fit_transform(SNP_data)
print("Original shape:", SNP_data.shape)
sel = SelectPercentile(f_regression, percentile=10).fit(SNP_data.toarray(), Y)
SNP_data = sel.transform(SNP_data)
print("Reduced shape:", SNP_data.shape)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    SNP_data, Y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# %%
# Elastic net for feature selection
# selector = SelectFromModel(ElasticNet(l1_ratio=0.5))
# selector.fit(X_train, y_train)

# X_train_val_red = selector.transform(X_train_val)
# X_train_red = selector.transform(X_train)
# X_val_red = selector.transform(X_val)
# X_test_red = selector.transform(X_test)

# print("Original shape",X_train.shape)
# print("Reduced shape",X_train_red.shape)

# %%
# hyperparameter tuning by baeysian optimization
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


space = {
    "depth": hp.choice("depth", range(4, 10)),
    "learning_rate": hp.loguniform("learning_rate", -3, -2),
    "random_strength": hp.uniform("random_strength", 0, 10),
    "l2_leaf_reg": hp.choice("l2_leaf_reg", range(2, 10)),
}


def objective(params):
    cat_r = CatBoostRegressor(**params)
    cat_r.fit(X_train, y_train)
    y_pred = cat_r.predict(X_val)
    score = mean_squared_error(y_val, y_pred, squared=False)
    return {"loss": score, "status": STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=75, trials=trials)
print("Best Hyperparamters:", best)

# %%
final_catboost = CatBoostRegressor(**best)  # type: ignore
final_catboost.fit(X_train_val, y_train_val)
y_pred = final_catboost.predict(X_test)

score = mean_squared_error(y_test, y_pred, squared=False)
corr = pearsonr(y_test, y_pred)

print("Pearson correlation coefficient:", corr[0])
print("p-value:", corr[1])
print("Final RMSE:", score)

# %%
pickle.dump(
    final_catboost, open("../models/catboost_bayesoptim_Fregselection_full.pkl", "wb")
)
