##############################################
#       Model exploration script             #
# Contains configuration for hyperparameter  #
# tuning and which model is to be used.      #
##############################################

# Importing modules
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

##############################################
# SET GLOBAL CONFIG VARIABLES, CHANGE AS NEEDED
##############################################
# Which phenotype to use
PHENOTYPE = "tarsus"  # Or "bodymass"
# Which model to use
MODEL = CatBoostRegressor  # Or XGBRegressor, RandomForestRegressor, LGBMRegressor, GradientBoostingRegressor, "INLA"
# Number of hyperparameter tuning iterations
NUM_TRIALS = 30
# Which feature selection method to use, corr is pearson correlation
FEATURE_SELECTION = "corr"  # Or "spearmanCorr", "kendallCorr", "elasticnet"
# Which strategy to use (two-step or one-step in thesis)
STRATEGY = "two-step"  # Or "one-step"
# Name of the model
MODEL_NAME = "catboost_GT"
##############################################
# Nothing below this line should be changed!
##############################################

# Hyperparameter search spaces
GBM_space = {
    "learning_rate": hp.loguniform("learning_rate", -3, -1),
    "n_estimators": hp.randint("n_estimators", 20, 205),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "max_depth": hp.randint("max_depth", 5, 30),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.45),
}


catboost_space = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "random_strength": hp.uniform("random_strength", 0, 20),
    "l2_leaf_reg": hp.loguniform("l2_leaf_reg", 1, 10),
    "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
    "leaf_estimation_iterations": hp.randint("leaf_estimation_iterations", 1, 10),
}


xgboost_space = {
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


LGBM_space = {
    "learning_rate": hp.loguniform("learning_rate", -7, 0),
    "num_leaves": hp.randint("num_leaves", 10, 10000),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "subsample": hp.uniform("subsample ", 0.5, 1.0),
    "min_sum_hessian_in_leaf": hp.loguniform("min_sum_hessian_in_leaf", -15, 5),
    "min_data_in_leaf": hp.randint("min_data_in_leaf", 1, 5000),
    "reg_alpha": hp.loguniform("reg_alpha", -8, 2),
    "reg_lambda": hp.loguniform("reg_lambda", -8, 2),
    "max_depth": hp.randint("max_depth", 15, 10000),
    "n_estimators": hp.randint("n_estimators", 20, 205),
}

space_dict = {
    CatBoostRegressor: catboost_space,
    XGBRegressor: xgboost_space,
    LGBMRegressor: LGBM_space,
    GradientBoostingRegressor: GBM_space,
    "INLA": None,
}


# Load data for two-step strategy
def load_data(phenotype="mass"):
    """Calls R script to extract data and then loads the data from feather file"""
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


# Load data for one-step strategy
def load_full_data(phenotype="mass"):
    """Calls R script to extract data and then loads the data from feather file"""
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
    """Main function that runs the model that is specified in the global variables"""
    search_space = space_dict[MODEL]
    if STRATEGY == "two-step":
        data = load_data(phenotype=PHENOTYPE)
        SNP_data = data.iloc[:, 8:]
        SNP_data = SNP_data.fillna(0)
        SNP_data = SNP_data.astype(int)
        Y = data.ID
        mean_pheno = data.mean_pheno
        ringnrs = data.ringnr

    # E-G
    else:  # STRATEGY == "one-step"
        data = load_full_data(phenotype=PHENOTYPE)
        SNP_data = data.iloc[:, 2:]
        SNP_data.hatchyear = SNP_data.hatchyear.astype(int)
        SNP_data.island_current = SNP_data.island_current.astype(int)
        Y = data.loc[:, PHENOTYPE]
        ringnrs = data.ringnr

    print("Starting model exploration")
    XGBcv = testModel(
        model=CatBoostRegressor,
        X=SNP_data,
        Y=Y,
        search_space=search_space,
        name=MODEL_NAME,
        num_trials=30,
        selection_method=FEATURE_SELECTION,
        phenotype=PHENOTYPE,
        ringnrs=ringnrs,
    )
    XGBcv.cross_validate()
    if MODEL == "INLA":
        res = subprocess.call(f"Rscript --vanilla INLAcv.R {PHENOTYPE}", shell=True)


if __name__ == "__main__":
    main()
