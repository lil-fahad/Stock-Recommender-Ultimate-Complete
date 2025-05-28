
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_ensemble(df, target='Close'):
    X = df.drop(columns=[target])
    y = df[target]
    rf = RandomForestRegressor(n_estimators=100)
    gb = GradientBoostingRegressor(n_estimators=100)
    rf.fit(X, y)
    gb.fit(X, y)
    return rf, gb

def predict_ensemble(models, X):
    rf, gb = models
    rf_pred = rf.predict(X)
    gb_pred = gb.predict(X)
    return (rf_pred + gb_pred) / 2  # average predictions
