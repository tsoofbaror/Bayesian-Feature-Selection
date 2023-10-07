import pandas as pd

from parameters import get_model_parameters
from spike_and_slab import RegressionSpikeAndSlab
import os


if __name__ == "__main__":
    params = get_model_parameters("deuterium.csv", os.path.abspath(__file__))
    params['data']['id_col'] = True
    params['priors']['a_alpha_prior'] = 3
    params['priors']['a_beta_prior'] = 2
    params['priors']['sigma_alpha_prior'] = 5
    params['priors']['sigma_beta_prior'] = 2
    params['priors']['tau_alpha_prior'] = 7
    params['priors']['tau_beta_prior'] = 3
    model = RegressionSpikeAndSlab(params)
    model.run()
