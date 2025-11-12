import pandas as pd
from .arima_model import ARIMAModel
from .bayesian_model import BayesianModel
from .mcmc_sampler import MCMCSampler

class CryptoPredictor:
    """
    Combines ARIMA + Bayesian + MCMC for BTC
    """
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, parse_dates=['Date']).sort_values('Date')
        self.arima = None
        self.bayes = None
        self.mcmc = None

    def run_arima(self, p=1, d=1, q=0, steps=5):
        self.arima = ARIMAModel(p, d, q)
        self.arima.fit(self.data['Close'])
        return self.arima.forecast(steps)

    def run_bayesian(self):
        returns = self.data['Close'].pct_change().dropna()
        self.bayes = BayesianModel()
        self.bayes.fit(returns)
        return self.bayes.summary()

    def run_mcmc(self):
        returns = self.data['Close'].pct_change().dropna()
        self.bayes = BayesianModel()
        self.bayes.fit(returns)
        self.mcmc = MCMCSampler(self.bayes.model)
        return self.mcmc.run()
