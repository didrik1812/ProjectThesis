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



@dataclass
class testModel:
    '''Class for testing a generic model'''
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


    def cross_validate(self):
        kf = KFold(n_splits=10,random_state=42)
        for fold, (train_val_index, test_index) in enumerate(kf.split(self.X)):
            self.X_train_val, self.X_test = self.X.iloc[train_val_index], self.X.iloc[test_index]
            self.Y_train_val, self.Y_test = self.Y.iloc[train_val_index], self.Y.iloc[test_index]

            sel = SelectPercentile(f_regression, percentile=10).fit(self.X_train_val, self.Y_train_val) # type: ignore
            self.X_train_val = sel.transform(self.X_train_val)
            self.X_test = sel.transform(self.X_test)

            self.hyperparameter_tuning()
            self.eval()

            self.CV_results[fold] = {
                'scores': self.score,
                'corrs': self.corr,
                'best_params': self.best_params,
                'sel': sel
            }
            print(f"Fold {fold+1} complete")
        
        self.best_settings = max(self.CV_results, key=lambda x: self.CV_results[x]['scores'])
        self.model.set_params(**self.CV_results[self.best_settings]['best_params']) # type: ignore
        self.best_score = self.CV_results[self.best_settings]['scores']
        self.best_corr = self.CV_results[self.best_settings]['corrs']

        self.save()

    def objective(self,params):
        '''Objective function for hyperparameter tuning with cross validation'''
        self.model.set_params(**params) #type: ignore
        losses = [0]*5
        kf_tune = KFold(n_splits=5,random_state=42)
        for tunefold, (train_index,val_index ) in kf_tune.split(self.X_train_val):
                self.X_train, self.X_val = self.X_train_val[train_index], self.X_train_val[val_index]
                self.Y_train, self.Y_val = self.Y_train_val[train_index], self.Y_train_val[val_index]

                self.model.fit(self.X_train,self.Y_train) #type: ignore
                y_pred = self.model.predict(self.X_val) #type: ignore
                losses[tunefold] = self.score_func(self.Y_val,y_pred)

        return {'loss': np.mean(losses),'loss_variance': np.var(losses) ,'status': STATUS_OK}

    def hyperparameter_tuning(self):
        '''Optimize hyperparameters using bayesian optimization'''
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
        '''Train model using best hyperparameters'''
        self.model.set_params(**self.best_params) # type: ignore
        self.model.fit(self.X_train_val,self.Y_train_val) # type: ignore
        y_pred = self.model.predict(self.X_test) # type: ignore
        self.score = self.score_func(self.Y_test,y_pred)
        self.corr = pearsonr(self.Y_test,y_pred)[0]

    def save(self):
        '''Save model to file'''
        self.plot()
        os.makedirs(f'models/{self.name}',exist_ok=True)
        with open(f"models/{self.name}/{self.name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(f"models/{self.name}/{self.name}_best_sel.pkl", "wb") as f:
            pickle.dump(self.CV_results[self.best_settings]["sel"], f)

        os.makedirs(f'results',exist_ok=True)
        with open(f"results/results.txt", "a") as f:
            f.write(f'''{self.name}:\tsocre = {self.best_score}, corr = {self.best_corr}\n''')
            
    def plot(self):
        '''Plot results, score and correlation'''
        plt.figure(1)
        x = self.CV_results.keys()
        scores = [self.CV_results[key]['scores'] for key in x]
        corrs = [self.CV_results[key]['corrs'] for key in x]
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


