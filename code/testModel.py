from dataclasses import dataclass
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectFromModel,
    mutual_info_regression,
    SelectPercentile,
    f_regression,
)
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import pickle
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import os
from sklearn.ensemble import RandomForestRegressor
from shap import TreeExplainer
import subprocess


@dataclass
class testModel:
    """Class for testing a generic model"""

    model: object
    X: pd.DataFrame
    Y: pd.Series
    search_space: dict
    phenotype: str
    score_func = mean_squared_error
    name: str
    score: float = 0
    corr: float = 0
    best_params = {}
    CV_results = {}
    selection_method: str = "shap"
    hyperCV: bool = False
    feature_percentile: float = 0.1
    num_trials: int = 20
    iterations: int = None
    ringnrs: pd.Series = None
    data_path = "~//..//..//..//..//work//didrikls//ProjectThesis//data//"

    def cross_validate(self):
        print("Starting cross validation")
        self.search_space["feature_percentile"] = hp.uniform(
            "feature_percentile", 0.1, 0.9
        )
        if self.selection_method == "elasticnet":
            self.search_space["l1ratio"] = hp.uniform("l1ratio", 0.009, 0.05)

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
            if isinstance(self.model, str):
                if self.model == "INLA":
                    self.run_INLA(train_val_index, test_index)
            else:
                self.hyperparameter_tuning()
                self.eval()

            self.CV_results[fold] = {
                "scores": self.score,
                "corrs": self.corr,
                "best_params": self.best_params,
                "sel": self.sel,
                "feature_percentile": self.feature_percentile,
            }
            print(f"Fold {fold+1} complete, score = {self.score}, corr = {self.corr}")

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
        if self.selection_method == "shap":
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(self.X_train_val, self.Y_train_val)
            explainer = TreeExplainer(reg)
            shap_values = explainer.shap_values(self.X_train_val)
            feature_importances = np.abs(shap_values).mean(axis=0)
            sorted_features = self.X_train_val.columns[
                np.argsort(feature_importances)[::-1]
            ]
            num_features = int(len(self.X.columns) * self.feature_percentile)
            sorted_features = sorted_features[:num_features]
            self.X_train_val_red = self.X_train_val[sorted_features]
            self.X_test_red = self.X_test[sorted_features]

            self.sel = {
                "method": "shap",
                "num_features": num_features,
                "features": sorted_features,
            }

        elif self.selection_method == "f_reg":
            sel = SelectPercentile(f_regression, percentile=10).fit(self.X_train_val, self.Y_train_val)  # type: ignore
            self.X_train_val = sel.transform(self.X_train_val)
            self.X_test = sel.transform(self.X_test)
            self.sel = {"method": "f_reg", "features": sel.get_support(indices=True)}

        elif self.selection_method == "corr":
            sel_corr = abs(self.X_train_val.corrwith(self.Y_train_val))
            sel_corr.sort_values(ascending=False, inplace=True)
            sel_corr = sel_corr[: round(len(sel_corr) * self.feature_percentile)]
            sorted_features = sel_corr.index.tolist()
            self.X_train_val_red = self.X_train_val[sorted_features]
            self.X_test_red = self.X_test[sorted_features]
            self.sel = {
                "method": "correlation",
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
        if self.selection_method == "elasticnet":
            self.feature_percentile = round(
                len(c.index) / len(self.X_train_val.columns), 2
            )

    def objective(self, params):
        """Objective function for hyperparameter tuning with cross validation"""
        self.feature_percentile = params.pop("feature_percentile")
        if self.selection_method == "elasticnet":
            self.l1ratio = params.pop("l1ratio")
        self.feature_selection()
        params["early_stopping_rounds"] = 50

        reg = self.model(**params)

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

                reg.fit(self.X_train, self.Y_train, verbose=False)  # type: ignore
                self.y_pred = reg.predict(self.X_val)  # type: ignore
                try:
                    losses[tunefold] = self.score_func(self.Y_val, self.y_pred)
                except:
                    losses[tunefold] = mean_squared_error(self.Y_val, self.y_pred)
            return {
                "loss": np.mean(losses),
                "loss_variance": np.var(losses),
                "status": STATUS_OK,
            }
        else:
            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
                self.X_train_val_red, self.Y_train_val, test_size=0.3, random_state=42
            )
            if self.iterations:
                reg = self.model(
                    **params,
                    iterations=self.iterations,
                )

            reg.fit(
                self.X_train,
                self.Y_train,
                eval_set=[(self.X_val, self.Y_val)],
                verbose=False,
            )
            self.y_pred = reg.predict(self.X_val)
            try:
                losses = self.score_func(self.Y_val, self.y_pred)
            except:
                losses = mean_squared_error(self.Y_val, self.y_pred)
            return {"loss": losses, "status": STATUS_OK}

    def hyperparameter_tuning(self):
        """Optimize hyperparameters using bayesian optimization"""
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
        """Train model using best hyperparameters"""
        best_feat_perc = self.best_params.pop("feature_percentile")
        if self.selection_method == "elasticnet":
            self.l1ratio = self.best_params.pop("l1ratio")

        reg = self.model(**self.best_params)

        self.X_train_val = self.X_train_val[self.choosen_features[best_feat_perc]]
        self.X_test = self.X_test[self.choosen_features[best_feat_perc]]

        if self.iterations:
            reg = self.model(**self.best_params, iterations=self.iterations)

        reg.fit(self.X_train_val, self.Y_train_val, verbose=False)  # type: ignore
        self.y_pred = reg.predict(self.X_test)  # type: ignore
        try:
            self.score = self.score_func(self.Y_test, self.y_pred)
        except:
            self.score = mean_squared_error(self.Y_test, self.y_pred)

        self.corr = pearsonr(self.Y_test, self.y_pred)[0]

    def save(self):
        """Save model to file"""

        os.makedirs(f"models/{self.name}", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        self.plot()
        with open(f"models/{self.name}/{self.name}.pkl", "wb") as f:
            pickle.dump(self.best_model, f)
        with open(f"models/{self.name}/{self.name}_best_sel.pkl", "wb") as f:
            pickle.dump(self.CV_results[self.best_settings]["sel"], f)

        with open("results/results.txt", "a") as f:
            f.write(
                f"""{self.name}:\tsocre = {self.best_score}, corr = {self.best_corr}\n"""
            )

    def plot(self):
        """Plot results, score and correlation"""
        plt.figure(1)
        if not isinstance(self.model, str):
            x = sorted(
                self.CV_results,
                key=lambda x: self.CV_results[x]["feature_percentile"],
                reverse=False,
            )
        else:
            x = self.CV_results.keys()
        scores = [self.CV_results[key]["scores"] for key in x]
        corrs = [self.CV_results[key]["corrs"] for key in x]
        feature_percentiles = [self.CV_results[key]["feature_percentile"] for key in x]

        try:
            corrs_df = pd.read_pickle("results/corrs_df.pkl")
            MSE_df = pd.read_pickle("results/MSE_df.pkl")
            feat_perc_df = pd.read_pickle("results/feat_perc_df.pkl")

            corrs_df[self.name] = corrs
            MSE_df[self.name] = scores
            if not isinstance(self.model, str):
                feat_perc_df[self.name] = feature_percentiles

        except:
            corrs_df = pd.DataFrame({self.name: corrs})
            MSE_df = pd.DataFrame({self.name: scores})
            feat_perc_df = pd.DataFrame({self.name: feature_percentiles})

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

        if not isinstance(self.model, str):
            plt.figure(3)
            plt.plot(feature_percentiles, scores)
            plt.title(f"{self.score_func.__name__} over feature percentile")
            plt.xlabel("Feature selection percentile")
            plt.grid()
            plt.savefig(f"models/{self.name}/{self.name}_featperc.png")

        corrs_df.to_pickle("results/corrs_df.pkl")
        MSE_df.to_pickle("results/MSE_df.pkl")
        feat_perc_df.to_pickle("results/feat_perc_df.pkl")

        self.box_plotter()

    def box_plotter(self):
        corrs_df = pd.read_pickle("results/corrs_df.pkl")
        MSE_df = pd.read_pickle("results/MSE_df.pkl")
        feat_perc_df = pd.read_pickle("results/feat_perc_df.pkl")

        plot_df = pd.read_pickle("results/plot_df.pkl")

        for i in range(corrs_df.shape[0]):
            plot_df.loc[len(plot_df)] = [
                self.name,
                corrs_df.iloc[i][self.name],
                MSE_df.iloc[i][self.name],
                feat_perc_df.iloc[i][self.name],
                self.phenotype,
            ]

        sns.set_style("darkgrid")
        sns.set_palette(palette="husl")

        plt.figure(4)
        plt.title("Overall MSE")
        sns.boxplot(
            data=plot_df, x="phenotype", y="MSE", hue="model", orient="v", width=0.5
        )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("results/MSE_boxplot.png", bbox_inches="tight")

        plt.figure(5)
        plt.title("Overall corr")
        sns.boxplot(
            data=plot_df, x="phenotype", y="corr", hue="model", orient="v", width=0.5
        )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("results/corr_boxplot.png", bbox_inches="tight")

        if not isinstance(self.model, str):
            plt.figure(6)
            plt.title("Overall Feature percentile")
            sns.boxplot(
                data=plot_df,
                x="phenotype",
                y="feat_perc",
                hue="model",
                orient="v",
                width=0.5,
            )
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig("results/feat_perc_boxplot.png", bbox_inches="tight")

        plot_df.to_pickle("results/plot_df.pkl")

    def run_INLA(self, train_val_index, test_index):
        train_ringnrs = (
            self.ringnrs.iloc[train_val_index]
            .to_frame()
            .reset_index()
            .to_feather(self.data_path + "temp//ringnr_train.feather")
        )
        test_ringnrs = (
            self.ringnrs.iloc[test_index]
            .to_frame()
            .reset_index()
            .to_feather(self.data_path + "temp//ringnr_test.feather")
        )
        res = subprocess.call(
            f"Rscript --vanilla runINLA.R {self.phenotype}", shell=True
        )
        if res == 0:
            results = pd.read_feather(self.data_path + "//temp//INLA_result.feather")

            self.score = float(results["score"].iloc[0])
            self.corr = float(results["corr"].iloc[0])
            self.best_params = None
            self.feature_percentile = None
            self.sel = None
        else:
            print(res)
            quit()
