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


def load_data(phenotype="mass"):
    """Load data from RDS file"""
    print("LOADING DATA")
    data_path = "/../../../../../../work/didrikls/ProjectThesis/data/"
    print("Running R script to load data")
    if not os.path.isfile(data_path + "envGene_" + phenotype + ".feather"):
        res = subprocess.call(
            f"Rscript --vanilla envGendataloader.R {phenotype}", shell=True
        )
    else:
        res = 0

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
    mass_residuals = load_data(phenotype=phenotype)
    SNP_data = mass_residuals.iloc[:, 2:]
    # SNP_data.fillna(0, inplace=True)
    # SNP_data = SNP_data.astype(int)
    # SNP_data = SNP_data.astype("category")
    SNP_data.hatchyear = SNP_data.hatchyear.astype(int)
    # SNP_data.island_current = SNP_data.island_current.astype(int)
    Y = mass_residuals.loc[:, phenotype]
    ringnrs = mass_residuals.ringnr
    print("Starting model exploration")
    XGBcv = testModel(
        model=XGBRegressor,
        X=SNP_data,
        Y=Y,
        search_space=xgboost_space,
        name="xgboostEGT_R",
        num_trials=30,
        selection_method="corr",
        phenotype=phenotype,
    )
    XGBcv.cross_validate()


if __name__ == "__main__":
    main()
