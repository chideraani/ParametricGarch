import unittest
import numpy as np
import pandas as pd
from parametricGarch import Garch

class TestGarch(unittest.TestCase):

    def setUp(self):
        # Create a sample time series data
        self.data = pd.DataFrame({'returns': np.random.randn(100)})

    def test_garch_initialization(self):
        # Test Garch initialization with valid parameters
        model = Garch(self.data, p=1, q=1)
        self.assertIsInstance(model, Garch)

    def test_garch_invalid_data(self):
        # Test Garch initialization with invalid data
        with self.assertRaises(ValueError):
            Garch(None, p=1, q=1)

    def test_bootstrap(self):
        # Test bootstrap method
        model = Garch(self.data, p=1, q=1)
        self.assertTrue(model.bootstrap())

    def test_bootstrap_invalid_num_iterations(self):
        # Test bootstrap method with invalid num_iterations parameter
        model = Garch(self.data, p=1, q=1)
        with self.assertRaises(ValueError):
            model.bootstrap(num_iterations=-1)

    def test_estimate_risk(self):
        # Test estimate_risk method
        model = Garch(self.data, p=1, q=1)
        model.bootstrap()
        risk_estimates = model.estimate_risk()
        self.assertIsInstance(risk_estimates, dict)
        self.assertIn('Mean Volatility', risk_estimates)
        self.assertIn('Volatility Confidence Interval', risk_estimates)
        self.assertIn('Mean VaR', risk_estimates)
        self.assertIn('VaR Confidence Interval', risk_estimates)

    def test_estimate_risk_invalid_confidence_level(self):
        # Test estimate_risk method with invalid confidence_level parameter
        model = Garch(self.data, p=1, q=1)
        model.bootstrap()
        with self.assertRaises(ValueError):
            model.estimate_risk(confidence_level=2)

if __name__ == '__main__':
    unittest.main()
