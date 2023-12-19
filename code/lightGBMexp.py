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
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Path: code/model_exploration.py

catboost_space = {
    "depth": hp.randint("depth", 5, 40),
    "learning_rate": hp.loguniform("learning_rate", -3, -2),
    "random_strength": hp.uniform("random_strength", 0, 10),
    "l2_leaf_reg": hp.choice("l2_leaf_reg", range(2, 10)),
}

xgboost_space = {
    "max_depth": hp.randint("max_depth", 5, 40),
    "alpha": hp.uniform("alpha", 0, 1000),
    "lambda": hp.uniform("lambda", 0, 1000),
    "min_child_weight": hp.uniform("min_child_weight", 0, 120),
    "eta": hp.loguniform("eta", -5, -2),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
}

LGBM_space = {
    "learning_rate": hp.loguniform("learning_rate", -3, -2),
    "num_leaves": hp.randint("num_leaves", 10, 100),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1.0),
    "subsample": hp.uniform("subsample ", 0.1, 1.0),
    "subsample_freq": hp.uniform("subsample_freq", 0, 1),
    "max_depth": hp.randint("max_depth", 15, 100),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1000),
}

GBM_space = {
    "learning_rate": hp.loguniform("learning_rate", -3, -1),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "max_depth": hp.randint("max_depth", 5, 30),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.45),
}


catboost_space_paper = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "random_strength": hp.uniform("random_strength", 0, 20),
    "l2_leaf_reg": hp.loguniform("l2_leaf_reg", 1, 10),
    "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
    "leaf_estimation_iterations": hp.randint("leaf_estimation_iterations", 1, 10),
    "n_estimators": hp.randint("n_estimators", 20, 500),
}


xgboost_space_paper = {
    "max_depth": hp.randint("max_depth", 2, 10),
    "alpha": hp.loguniform("alpha", -8, 2),
    "lambda": hp.loguniform("lambda", -8, 2),
    "min_child_weight": hp.loguniform("min_child_weight", -8, 5),
    "eta": hp.loguniform("eta", -7, 0),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
    "gamma": hp.loguniform("gamma", -8, 2),
}


LGBM_space_paper = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "num_leaves": hp.randint("num_leaves", 10, 10000),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "subsample": hp.uniform("subsample ", 0.5, 1.0),
    "subsample_freq": hp.uniform("subsample_freq", 0, 1),
    "min_sum_hessian_in_leaf": hp.loguniform("min_sum_hessian_in_leaf", -15, 5),
    "min_data_in_leaf": hp.randint("min_data_in_leaf", 1, 5000),
    "lambda_l1": hp.loguniform("lambda_l1", -8, 2),
    "lambda_l2": hp.loguniform("lambda_l2", -8, 2),
}


def load_data(phenotype="mass"):
    """Load data from RDS file"""
    print("LOADING DATA")
    data_path = "/../../../../../../work/didrikls/ProjectThesis/data/"
    print("Running R script to load data")
    res = subprocess.call(f"Rscript --vanilla dataloader.R {phenotype}", shell=True)

    if res == 0:
        print("R script completed successfully")
        feather_path = data_path + phenotype + ".feather"
        mass_residuals = pd.read_feather(feather_path)
        print("DATA LOADED")
        return mass_residuals
    else:
        raise Exception("Could not load data")


def load_full_data(phenotype="mass"):
    """Load data from RDS file"""
    print("LOADING DATA")
    data_path = "/../../../../../../work/didrikls/ProjectThesis/data/"
    print("Running R script to load data")
    res = subprocess.call(
        f"Rscript --vanilla envGendataloader.R {phenotype}", shell=True
    )

    if res == 0:
        print("R script completed successfully")
        rds_path = data_path + "envGene_" + phenotype + ".feather"
        mass_residuals = pd.read_feather(rds_path)
        print("DATA LOADED")
        return mass_residuals
    else:
        raise Exception("Could not load data")


def main():
    phenotype = "tarsus"
    mass_residuals = load_full_data(phenotype=phenotype)
    # BV
    # SNP_data = mass_residuals.iloc[:, 8:]
    # SNP_data = SNP_data.fillna(0)
    # SNP_data = SNP_data.astype(int)
    # Y = mass_residuals.ID
    # mean_pheno = mass_residuals.mean_pheno

    # E-G
    SNP_data = mass_residuals.iloc[:, 2:]
    SNP_data.hatchyear = SNP_data.hatchyear.astype(int)
    SNP_data.island_current = SNP_data.island_current.astype(int)
    Y = mass_residuals.loc[:, phenotype]

    ringnrs = mass_residuals.ringnr
    print("Starting model exploration")
    XGBcv = testModel(
        model=CatBoostRegressor,
        X=SNP_data,
        Y=Y,
        search_space=catboost_space_paper,
        name="catboost_ET",
        num_trials=30,
        selection_method="corr",
        phenotype=phenotype,
        ringnrs=ringnrs,
    )
    XGBcv.cross_validate()


if __name__ == "__main__":
    main()
