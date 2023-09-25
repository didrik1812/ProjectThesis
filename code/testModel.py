from dataclasses import dataclass
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

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
from sklearn.model_selection import KFold
import os
from sklearn.ensemble import RandomForestRegressor
from shap import TreeExplainer


@dataclass
class testModel:
    """Class for testing a generic model"""

    model: object
    X: pd.DataFrame
    Y: pd.Series
    search_space: dict
    score_func = mean_squared_error
    name: str
    score: float = 0
    corr: float = 0
    best_params = {}
    CV_results = {}
    selection_method: str = "shap"
    hyperCV:bool = False

    def cross_validate(self):
        print("Starting cross validation")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_val_index, test_index) in enumerate(kf.split(self.X)):
            self.X_train_val, self.X_test = (
                self.X.iloc[train_val_index],
                self.X.iloc[test_index],
            )
            self.Y_train_val, self.Y_test = (
                self.Y.iloc[train_val_index],
                self.Y.iloc[test_index],
            )
            print("Fold", fold + 1, "of", kf.get_n_splits(self.X))
            self.feature_selection()

            self.hyperparameter_tuning()
            self.eval()

            self.CV_results[fold] = {
                "scores": self.score,
                "corrs": self.corr,
                "best_params": self.best_params,
                "sel": self.sel,
            }
            print(f"Fold {fold+1} complete, score = {self.score}, corr = {self.corr}")

        self.best_settings = max(
            self.CV_results, key=lambda x: self.CV_results[x]["scores"]
        )
        self.model.set_params(**self.CV_results[self.best_settings]["best_params"])  # type: ignore
        self.best_score = self.CV_results[self.best_settings]["scores"]
        self.best_corr = self.CV_results[self.best_settings]["corrs"]

        self.save()

    def feature_selection(self):
        print("Starting feature selection")
        if self.selection_method == "shap":
            print("Training RF for feature selection")
            reg = RandomForestRegressor(n_estimators=40, max_depth=4, random_state=42)
            reg.fit(self.X_train_val, self.Y_train_val)
            print("Explaining feature importance with SHAP")
            explainer = TreeExplainer(reg)
            shap_values = explainer.shap_values(self.X_train_val)
            feature_importances = np.abs(shap_values).mean(axis=0)
            sorted_features = self.X_train_val.columns[
                np.argsort(feature_importances)[::-1]
            ]
            num_features = int(len(self.X.columns) * 0.10)
            self.X_train_val = self.X_train_val[sorted_features[:num_features]]
            self.X_test = self.X_test[sorted_features[:num_features]]

            self.sel = {"method": "shap", "features": sorted_features[:num_features]}

        elif self.selection_method == "f_reg":
            print("Selecting features using f_regression")
            sel = SelectPercentile(f_regression, percentile=10).fit(self.X_train_val, self.Y_train_val)  # type: ignore
            self.X_train_val = sel.transform(self.X_train_val)
            self.X_test = sel.transform(self.X_test)
            self.sel = {"method": "f_reg", "features": sel.get_support(indices=True)}

    def objective(self, params):
        """Objective function for hyperparameter tuning with cross validation"""
        self.model.set_params(**params)  # type: ignore
        if self.hyperCV:
            losses = [0] * 5
            kf_tune = KFold(n_splits=5, shuffle=True, random_state=42)
            for tunefold, (train_index, val_index) in enumerate(
                kf_tune.split(self.X_train_val)
            ):
                print(
                    f"Hyperparameter fold {tunefold+1} of {kf_tune.get_n_splits(self.X_train_val)}"
                )

                self.X_train, self.X_val = (
                    self.X_train_val.iloc[train_index],
                    self.X_train_val.iloc[val_index],
                )
                self.Y_train, self.Y_val = (
                    self.Y_train_val.iloc[train_index],
                    self.Y_train_val.iloc[val_index],
                )

                self.model.fit(self.X_train, self.Y_train)  # type: ignore
                self.y_pred = self.model.predict(self.X_val)  # type: ignore
                try:
                    losses[tunefold] = self.score_func(self.Y_val, self.y_pred)
                except:
                    print("score func not working")
                    losses[tunefold] = mean_squared_error(self.Y_val, self.y_pred)
                return {
                    "loss": np.mean(losses),
                    "loss_variance": np.var(losses),
                    "status": STATUS_OK,
                }
        else:
            self.model.fit(self.X_train_val, self.Y_train_val)
            self.y_pred = self.model.predict(self.X_test)
            try:
                losses = self.score_func(self.Y_test, self.y_pred)
            except:
                print("score func not working")
                losses = mean_squared_error(self.Y_test, self.y_pred)
            return {"loss": losses, "status": STATUS_OK}

    def hyperparameter_tuning(self):
        """Optimize hyperparameters using bayesian optimization"""
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self.search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )
        self.best_params = best

    def eval(self):
        """Train model using best hyperparameters"""
        self.model.set_params(**self.best_params)  # type: ignore
        self.model.fit(self.X_train_val, self.Y_train_val)  # type: ignore
        self.y_pred = self.model.predict(self.X_test)  # type: ignore
        try:
            self.score = self.score_func(self.Y_test, self.y_pred)
        except:
            print("score func not working")
            self.score = mean_squared_error(self.Y_test, self.y_pred)

        self.corr = pearsonr(self.Y_test, self.y_pred)[0]

    def save(self):
        """Save model to file"""

        os.makedirs(f"models/{self.name}", exist_ok=True)
        self.plot()
        with open(f"models/{self.name}/{self.name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(f"models/{self.name}/{self.name}_best_sel.pkl", "wb") as f:
            pickle.dump(self.CV_results[self.best_settings]["sel"], f)

        os.makedirs(f"results", exist_ok=True)
        with open(f"results/results.txt", "a") as f:
            f.write(
                f"""{self.name}:\tsocre = {self.best_score}, corr = {self.best_corr}\n"""
            )

    def plot(self):
        """Plot results, score and correlation"""
        plt.figure(1)
        x = self.CV_results.keys()
        scores = [self.CV_results[key]["scores"] for key in x]
        corrs = [self.CV_results[key]["corrs"] for key in x]
        plt.title(f"{self.score_func.__name__}")
        plt.plot(x, scores)
        plt.xlabel("Fold")
        plt.grid()
        plt.savefig(f"models/{self.name}/{self.name}_score.png")

        plt.figure(2)
        plt.title("Pearson correlation coefficient")
        plt.plot(x, corrs)
        plt.xlabel("Fold")
        plt.grid()
        plt.savefig(f"models/{self.name}/{self.name}_corr.png")
