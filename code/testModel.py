##############################################
#       Model exploration class             #
# Performs feature selection, hyperparameter#
# tuning and cross validation for a given   #
# model and phenotype.                      #
##############################################
# Importing modules
from dataclasses import dataclass
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import pickle
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
import os
import subprocess
from pathlib import Path

# SET GLOBAL CONFIG VARIABLES, CHANGE AS NEEDED
# set path to where tmp data should be stored (for INLA)
# and where models and output should be saved
DATA_PATH = (
    Path.home().parent.parent.parent.parent
    / "work"
    / "didrikls"
    / "ProjectThesis"
    / "data"
)


@dataclass
class testModel:
    """Class for testing a generic model"""

    # Model params
    model: object
    search_space: dict
    name: str
    # Data input
    X: pd.DataFrame
    Y: pd.Series
    phenotype: str
    ringnrs: pd.Series = None
    mean_pheno: pd.Series = None
    # Other sepcifiers
    selection_method: str = "corr"  # Or "spearcorr", "kendallcorr", "elasticnet"

    # variables to be used later
    score: float = 0  # MSE
    corr: float = 0  # correlation with response
    best_params = {}  # best parameters
    CV_results = {}  # results from CV
    feature_percentile: float = 0.1  # how many features
    num_trials: int = 30  # max evaluations in hyperparamter tuning
    iterations: int = None  # How many boosting iterations

    data_path = DATA_PATH

    def cross_validate(self):
        """Do 10 fold cross validation with the given data and model"""

        print("Starting cross validation")
        np.random.seed(42)  # ensure same seed across all models
        # add feature percentile to search space
        self.search_space["feature_percentile"] = hp.uniform(
            "feature_percentile", 0.1, 0.9
        )
        # elasticnet needs l1ratio
        if self.selection_method == "elasticnet":
            self.search_space["l1ratio"] = hp.uniform("l1ratio", 0.009, 0.05)
        # Start cross validation with 10 folds, splits on ringnr
        kf = GroupKFold(n_splits=10)
        for fold, (train_val_index, test_index) in enumerate(
            kf.split(self.X, groups=self.ringnrs)
        ):
            # split data into train_val and test
            self.X_train_val, self.X_test = (
                self.X.iloc[train_val_index],
                self.X.iloc[test_index],
            )
            self.Y_train_val, self.Y_test = (
                self.Y.iloc[train_val_index],
                self.Y.iloc[test_index],
            )
            # Not always mean_pheno is given
            try:
                self.mean_pheno_test = self.mean_pheno.iloc[test_index]
            except Exception as e:
                pass
            print("Fold", fold + 1, "of", kf.get_n_splits(self.X))
            # If INLA is used a different approach is used (no hyperparmeter tuning),
            # CV indexes is sent to a seperate R script
            if isinstance(self.model, str):
                if self.model == "INLA":
                    self.run_INLA(train_val_index, test_index, fold)
            else:  # If not INLA, feature selection and hyperparameter tuning is done
                self.hyperparameter_tuning()  # Find best hyperparameters
                self.eval()  # Evaluate model on test set using best hyperparameters
            # Save results from fold
            self.CV_results[fold] = {
                "scores": self.score,
                "corrs": self.corr,
                "best_params": self.best_params,
                "sel": self.sel,
                "feature_percentile": self.feature_percentile,
            }
            print(f"Fold {fold+1} complete, score = {self.score}, corr = {self.corr}")
        # Find best performing fold and save model
        self.best_settings = max(
            self.CV_results, key=lambda x: self.CV_results[x]["corrs"]
        )
        if not isinstance(self.model, str):
            self.best_model = self.model(**self.CV_results[self.best_settings]["best_params"])  # type: ignore
        self.best_score = self.CV_results[self.best_settings]["scores"]
        self.best_corr = self.CV_results[self.best_settings]["corrs"]
        self.best_feat_perc = self.CV_results[self.best_settings]["feature_percentile"]
        print(
            f"""Preformance of {self.name}:\t MSE = {self.best_score}\t
            corr = {self.best_corr}\t feature_percentile = {self.best_feat_perc}"""
        )
        self.save()

    def feature_selection(self):
        """Perform feature selection with selected method"""
        if self.selection_method == "corr":
            # Calculate correlation between features and response
            sel_corr = abs(self.X_train_val.corrwith(self.Y_train_val))
            sel_corr.sort_values(ascending=False, inplace=True)
            # Select most correlated features based on percentile
            sel_corr = sel_corr[: round(len(sel_corr) * self.feature_percentile)]
            sorted_features = sel_corr.index.tolist()
            # Reduce data to only selected features
            self.X_train_val_red = self.X_train_val[sorted_features]
            self.X_test_red = self.X_test[sorted_features]
            self.sel = {
                "method": "correlation",
                "num_features": len(sorted_features),
                "features": sorted_features,
            }
        elif self.selection_method == "spearcorr":
            sel_corr = abs(
                self.X_train_val.corrwith(self.Y_train_val, method="spearman")
            )
            sel_corr.sort_values(ascending=False, inplace=True)
            sel_corr = sel_corr[: round(len(sel_corr) * self.feature_percentile)]
            sorted_features = sel_corr.index.tolist()
            self.X_train_val_red = self.X_train_val[sorted_features]
            self.X_test_red = self.X_test[sorted_features]
            self.sel = {
                "method": "spearcorrelation",
                "num_features": len(sorted_features),
                "features": sorted_features,
            }
        elif self.selection_method == "kendallcorr":
            sel_corr = abs(
                self.X_train_val.corrwith(self.Y_train_val, method="kendall")
            )
            sel_corr.sort_values(ascending=False, inplace=True)
            sel_corr = sel_corr[: round(len(sel_corr) * self.feature_percentile)]
            sorted_features = sel_corr.index.tolist()
            self.X_train_val_red = self.X_train_val[sorted_features]
            self.X_test_red = self.X_test[sorted_features]
            self.sel = {
                "method": "kendallcorrelation",
                "num_features": len(sorted_features),
                "features": sorted_features,
            }
        elif self.selection_method == "elasticnet":
            reg = ElasticNet(l1_ratio=self.l1ratio)
            reg.fit(self.X_train_val, self.Y_train_val)
            coefs = pd.Series(reg.coef_, index=self.X_train_val.columns)
            c = coefs[coefs != 0.0]
            self.X_train_val_red = self.X_train_val[c.index]
            self.X_test_red = self.X_test[c.index]
            sorted_features = c.index
            self.sel = {
                "method": "elasticnet",
                "num_features": len(sorted_features),
                "features": sorted_features,
            }

        self.choosen_features[self.feature_percentile] = sorted_features
        # If elasticnet is used, feature percentile is calculated based on number of non-zero coeficients
        if self.selection_method == "elasticnet":
            self.feature_percentile = round(
                len(c.index) / len(self.X_train_val.columns), 2
            )

    def objective(self, params):
        """
        Objective function for hyperparameter tuning,
        fits model on trainining set and returns MSE on validation set
        """
        # Some preprocessing is needed for some models
        self.feature_percentile = params.pop("feature_percentile")
        if self.selection_method == "elasticnet":
            self.l1ratio = params.pop("l1ratio")
        self.feature_selection()
        if self.model().__class__.__name__ == "GradientBoostingRegressor":
            params["n_iter_no_change"] = 50
        else:
            params["early_stopping_rounds"] = 50

        reg = self.model(**params)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X_train_val_red, self.Y_train_val, test_size=0.3, random_state=42
        )
        if self.iterations:
            reg = self.model(
                **params,
                iterations=self.iterations,
            )
        if self.model().__class__.__name__ == "GradientBoostingRegressor":
            reg = self.model(**params, validation_fraction=0.3, random_state=42)
            reg.fit(
                self.X_train_val_red,
                self.Y_train_val,
            )
        else:
            reg.fit(
                self.X_train,
                self.Y_train,
                eval_set=[(self.X_val, self.Y_val)],
                verbose=False,
            )
        self.y_pred = reg.predict(self.X_val)
        losses = mean_squared_error(self.Y_val, self.y_pred)
        return {"loss": losses, "status": STATUS_OK}

    def hyperparameter_tuning(self):
        """
        Optimize hyperparameters using bayesian optimization, 
        utilizing hyperopt and the objective function
        """
        self.choosen_features = {}
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self.search_space,
            algo=tpe.suggest,
            max_evals=self.num_trials,
            trials=trials,
        )
        self.best_params = best

    def eval(self):
        """Train model using best hyperparameters and evaluate it on test set"""
        # need some preprocessing for some models
        # feature percentile is used to select features and is not a hyperparameter to be sent in the models
        best_feat_perc = self.best_params.pop("feature_percentile")
        if self.selection_method == "elasticnet":
            self.l1ratio = self.best_params.pop("l1ratio")
        # initialize model with best hyperparameters
        reg = self.model(**self.best_params)
        # reduce data to only selected features
        self.X_train_val = self.X_train_val[self.choosen_features[best_feat_perc]]
        self.X_test = self.X_test[self.choosen_features[best_feat_perc]]

        if self.iterations:
            reg = self.model(**self.best_params, iterations=self.iterations)
        if self.model().__class__.__name__ == "GradientBoostingRegressor":
            reg.fit(self.X_train_val, self.Y_train_val)
        else:
            reg.fit(self.X_train_val, self.Y_train_val, verbose=False)  # type: ignore
        self.y_pred = reg.predict(self.X_test)  # type: ignore
        self.score = mean_squared_error(self.Y_test, self.y_pred)

        self.corr = pearsonr(self.Y_test, self.y_pred)[0]

    def save(self):
        """Save the best performing folds model to picklke file"""
        # create folders if they dont exist
        os.makedirs(f"models/{self.name}", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        # save CV results
        self.save_results()
        # save model objects
        with open(f"models/{self.name}/{self.name}.pkl", "wb") as f:
            pickle.dump(self.best_model, f)
        with open(f"models/{self.name}/{self.name}_best_sel.pkl", "wb") as f:
            pickle.dump(self.CV_results[self.best_settings]["sel"], f)

        with open("results/results.txt", "a") as f:
            f.write(
                f"""{self.name}:\tsocre = {self.best_score}, corr = {self.best_corr}\n"""
            )

    def save_results(self):
        """Save feature_percentile, correlations and MSE from each fold"""
        # some exeptions are needed for INLA
        if not isinstance(self.model, str):
            x = sorted(
                self.CV_results,
                key=lambda x: self.CV_results[x]["feature_percentile"],
                reverse=False,
            )
        else:
            x = self.CV_results.keys()
        # extract results from CV
        scores = [self.CV_results[key]["scores"] for key in x]
        corrs = [self.CV_results[key]["corrs"] for key in x]
        feature_percentiles = [self.CV_results[key]["feature_percentile"] for key in x]
        # save to pickled dataframes
        try:
            corrs_df = pd.read_pickle("results/corrs_df.pkl")
            MSE_df = pd.read_pickle("results/MSE_df.pkl")
            feat_perc_df = pd.read_pickle("results/feat_perc_df.pkl")

            corrs_df[self.name] = corrs
            MSE_df[self.name] = scores
            if not isinstance(self.model, str):
                feat_perc_df[self.name] = feature_percentiles

        except Exception as e:
            corrs_df = pd.DataFrame({self.name: corrs})
            MSE_df = pd.DataFrame({self.name: scores})
            feat_perc_df = pd.DataFrame({self.name: feature_percentiles})

        corrs_df.to_pickle("results/corrs_df.pkl")
        MSE_df.to_pickle("results/MSE_df.pkl")
        feat_perc_df.to_pickle("results/feat_perc_df.pkl")

    def run_INLA(self, train_val_index, test_index, fold):
        """Runs INLA by calling an R script. Indexes are sent by storing them temporaliy as feather files"""
        (self.data_path / Path("temp//ringnr_train.feather")).unlink(missing_ok=True)
        (self.data_path / Path("temp//ringnr_test.feather")).unlink(missing_ok=True)
        (self.data_path / Path("temp//INLA_result.feather")).unlink(missing_ok=True)

        train_ringnrs = (
            self.ringnrs.iloc[train_val_index]
            .to_frame()
            .reset_index()
            .to_feather(self.data_path / "temp" / f"ringnr_train_{fold}.feather")
        )
        test_ringnrs = (
            self.ringnrs.iloc[test_index]
            .to_frame()
            .reset_index()
            .to_feather(self.data_path / "temp" / f"ringnr_test_{fold}.feather")
        )
        return 0
