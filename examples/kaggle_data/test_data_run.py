import os

from spike_and_slab import RegressionSpikeAndSlab
from parameters import get_model_parameters


if __name__ == "__main__":
    params = get_model_parameters("test_data.csv", os.path.abspath(__file__))
    model = RegressionSpikeAndSlab(params)
    model.run()
