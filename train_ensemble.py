
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_ensemble(df, target='Close'):
    X = df.drop(columns=[target])
    y = df[target]
    rf = RandomForestRegressor(n_estimators=100)
    gb = GradientBoostingRegressor(n_estimators=100)
    rf.fit(X, y)
    gb.fit(X, y)
    joblib.dump(rf, 'random_forest_model.pkl')
    joblib.dump(gb, 'gradient_boosting_model.pkl')
    print('✅ Ensemble models saved.')

if __name__ == '__main__':
    df = pd.read_csv('your_data.csv')  # replace with your data
    train_ensemble(df)
