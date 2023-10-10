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
    "depth": hp.randint("depth", 4, 10),
    "learning_rate": hp.loguniform("learning_rate", -3, -2),
    "random_strength": hp.uniform("random_strength", 0, 10),
    "l2_leaf_reg": hp.choice("l2_leaf_reg", range(2, 10)),
}

xgboost_space = {
    "max_depth": hp.randint("max_depth", 1, 15),
    "alpha": hp.uniform("alpha", 0, 1000),
    "min_child_weight": hp.uniform("min_child_weight", 0, 120),
    "eta": hp.loguniform("eta", -5, -2),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
}


def load_data(phenotype = "mass"):
    """Load data from RDS file"""
    print("LOADING DATA")
    print("Running R script to load data")
    res = subprocess.call(f"Rscript --vanilla dataloader.R {phenotype}", shell=True)
    res = 0
    if res == 0:
        print("R script completed successfully")
        rds_path = "../data/d.dat.full.feather"
        mass_residuals = pd.read_feather(rds_path)
        print("DATA LOADED")
        return mass_residuals
    else:
        raise Exception("Could not load data")


def load_pickle_data():
    print("LOADING DATA")
    pickle_path = "../data/mass_resid_df.pkl"
    mass_residuals = pd.read_pickle(pickle_path)
    print("DATA LOADED")
    return mass_residuals


def main():
    phenotype = "mass"
    mass_residuals = load_pickle_data()
    SNP_data = mass_residuals.iloc[:, 7:]
    SNP_data.fillna(0, inplace=True)
    SNP_data = SNP_data.astype(int)
    #SNP_data = SNP_data.astype("category")
    Y = mass_residuals.ID
    print("Starting model exploration")
    XGBcv = testModel(
        model=XGBRegressor,
        X=SNP_data,
        Y=Y,
        search_space=xgboost_space,
        name="xgboostCorrNofill",
        num_trials=30,
        selection_method="corr",
        phenotype="mass",
    )
    XGBcv.cross_validate()


if __name__ == "__main__":
    main()
