from testModel import testModel
import os
import subprocess
from catboost import CatBoostRegressor
from hyperopt import hp
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from shap import TreeExplainer


# Path: code/model_exploration.py

catboost_space = {
    "depth": hp.choice("depth", range(4, 10)),
    "learning_rate": hp.loguniform("learning_rate", -3, -2),
    "random_strength": hp.uniform("random_strength", 0, 10),
    "l2_leaf_reg": hp.choice("l2_leaf_reg", range(2, 10)),
}

xgboost_space = {
    "max_depth": hp.choice("max_depth", range(1, 20)),
    "eta": hp.loguniform("eta", -5, -2),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "n_estimators": hp.choice("n_estimators", range(20, 205, 5)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
}

def load_data():
    '''Load data from RDS file'''
    print("LOADING DATA")
    print("Running R script to load data")
    #res = subprocess.call("Rscript dataloader.R", shell=True)
    res = 0
    if res == 0:
        print("R script completed successfully")
        rds_path = "../data/d.dat.full.feather"
        mass_residuals = pd.read_feather(rds_path)
        print("DATA LOADED")
        return mass_residuals
    else:
        raise Exception("Could not load data")

def main():
    mass_residuals = load_data()
    SNP_data = mass_residuals.iloc[:, 7:]
    SNP_data.fillna(0, inplace=True)
    Y = mass_residuals.ID
    print("Starting model exploration")
    XGBcv = testModel(model=XGBRegressor(),
              X = SNP_data,
              Y = Y,
              search_space=xgboost_space,
              name="xgboostCV",
              ).cross_validate()

if __name__ == "__main__":
    main()