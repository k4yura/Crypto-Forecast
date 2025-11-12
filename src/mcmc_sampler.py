import pymc3 as pm

class MCMCSampler:
    """
    Run MCMC sampling for a given PyMC3 model
    """
    def __init__(self, model):
        self.model = model
        self.trace = None

    def run(self, samples=1000, tune=500):
        with self.model:
            self.trace = pm.sample(samples, tune=tune, cores=1, return_inferencedata=True)
        return self.trace
