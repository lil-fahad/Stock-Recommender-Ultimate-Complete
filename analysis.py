
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred, model_name='Model'):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f" - MSE: {mse:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f" - R^2 Score: {r2:.4f}")
    return {'mse': mse, 'mae': mae, 'r2': r2}

def plot_predictions(y_true, y_pred, title='Predictions vs Actual'):
    plt.figure(figsize=(10,6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
