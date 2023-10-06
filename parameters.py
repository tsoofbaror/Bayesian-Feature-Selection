import os
def get_model_parameters(filename, script_path):
    script_dir = os.path.dirname(script_path)
    relative_path = os.path.relpath(script_dir)
    return {'data': {'input_path': f'{relative_path}/{filename}',
                     'id_col': False,
                     'output_path': f'{relative_path}/results',
                     'save_data': False, 'x': 0, 'y': 0, 'col_names': [],},

            'normalized_data': {'x_normalized': 0, 'y_normalized': 0, 'x_mean': 0, 'x_std': 0, 'y_mean': 0, 'y_std': 0},

            'run_metadata': {'throw': True, 'throw_ratio': 0.2, 'iterations': 3000, 'current_iteration': 0},

            'chain_samples': {'save_samples': True, 'a': {'a': [], 'alpha': [], 'beta': []},
                              'tau': {'tau': [], 'alpha': [], 'beta': []},
                              'sigma': {'sigma': [], 'alpha': [], 'beta': []}
                                ,'s': [], 'w': [], 'log_probs': [], 'mse': []},

            'priors': {'a_alpha_prior': 3, 'a_beta_prior': 2, 'tau_alpha_prior': 7, 'tau_beta_prior': 2,
                       'sigma_alpha_prior': 3, 'sigma_beta_prior': 2},

            'tests': {'geweke_test': False, 'true_w': None},
            }
