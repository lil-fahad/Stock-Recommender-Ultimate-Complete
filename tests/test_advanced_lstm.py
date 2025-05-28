
import unittest
import numpy as np
from models.advanced_lstm import build_lstm_model, prepare_lstm_data
import pandas as pd

class TestAdvancedLSTM(unittest.TestCase):
    def test_build_lstm_model(self):
        model = build_lstm_model((60, 1))
        self.assertTrue(hasattr(model, 'fit'))

    def test_prepare_lstm_data(self):
        df = pd.DataFrame({'Close': np.random.rand(100)})
        X, y, scaler = prepare_lstm_data(df)
        self.assertEqual(X.shape[0], y.shape[0])

if __name__ == '__main__':
    unittest.main()
