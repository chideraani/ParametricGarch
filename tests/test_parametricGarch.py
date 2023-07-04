import unittest
import numpy as np
import pandas as pd
from parametricGarch import Garch

class TestGarch(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.data = pd.DataFrame({'returns': np.random.randn(100)})

    def test_bootstrap(self):
        # Create a Garch instance
        garch = Garch(self.data['returns'])

        # Perform bootstrap
        result = garch.bootstrap()

        # Check if the bootstrap was successful
        self.assertTrue(result)

        # Check if bootstrap samples are available
        bootstrap_samples = garch.bootstrap_samples
        self.assertIsNotNone(bootstrap_samples)
        self.assertIsInstance(bootstrap_samples, list)

    def test_estimate_risk(self):
        # Create a Garch instance
        garch = Garch(self.data)

        # Perform bootstrap
        garch.bootstrap()

        # Estimate risk
        risk_estimates = garch.estimate_risk()

        # Check if the risk estimates are of the correct type
        self.assertIsInstance(risk_estimates, dict)

        # Check if the required keys are present in the risk estimates dictionary
        required_keys = ['Mean Volatility', 'Volatility Confidence Interval', 'Mean VaR', 'VaR Confidence Interval']
        for key in required_keys:
            self.assertIn(key, risk_estimates)

if __name__ == '__main__':
    unittest.main()
