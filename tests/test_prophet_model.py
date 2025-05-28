
import unittest
import pandas as pd
from models.prophet_model import train_prophet, predict_prophet

class TestProphetModel(unittest.TestCase):
    def test_train_and_predict(self):
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'Close': range(100)
        })
        model = train_prophet(df)
        forecast = predict_prophet(model, periods=10)
        self.assertIn('yhat', forecast.columns)

if __name__ == '__main__':
    unittest.main()
