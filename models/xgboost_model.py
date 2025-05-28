
import xgboost as xgb

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_input):
    return model.predict(X_input)
