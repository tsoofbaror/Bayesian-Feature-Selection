import pickle
import random
import pandas as pd
import json
import tqdm
import torch
import numpy as np
import copy
from datetime import datetime
import os
from scipy.stats import multivariate_normal
from utils import load_and_split_data, normalize, get_xs, solve_with_cholesky
from visualizations import visualize_s, geweke_plot

# np.random.seed(0)
torch.set_printoptions(precision=4, sci_mode=False)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RegressionSpikeAndSlab:
    def __init__(self, metadata):
        self.metadata = metadata
        # init variables
        self.iter = self.metadata['run_metadata']['iterations']
        self.xs = None
        self.s_to_sample = 0

        self.log_prob = None
        self.mse = None

        # prepare data
        self.metadata['data']['x'], self.metadata['data']['y'], self.metadata['data'][
            'col_names'] = load_and_split_data(metadata['data']['input_path'])

        if metadata['data']['id_col']:
            self.metadata['data']['x'] = self.metadata['data']['x'][:, 1:]
        # normalize data
        (self.metadata['normalized_data']['x_normalized'], self.metadata['normalized_data']['x_mean'],
         self.metadata['normalized_data']['x_std']) = normalize(metadata['data']['x'])
        (self.metadata['normalized_data']['y_normalized'], self.metadata['normalized_data']['y_mean'],
         self.metadata['normalized_data']['y_std']) = normalize(metadata['data']['y'])

        # helper variables
        if self.metadata['tests']['geweke_test']:
            self.metadata['normalized_data']['y_normalized'] = np.random.normal(loc=0.0, scale=1, size=len(self.y))

        # init model latent variables and data
        self.x = self.metadata['normalized_data']['x_normalized'].astype(float)
        self.y = self.metadata['normalized_data']['y_normalized'].astype(float)

        self.a = 0
        self.tau = 0
        self.sigma = 0
        self.s = np.zeros((self.x.shape[1]))
        self.w = np.zeros((self.x.shape[1]))

        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.metadata['data']['output_path'] += "_" + self.current_time

        self.geweke = self.metadata['tests']['geweke_test']
        self.geweke_pairs = [[], [], [], [], []]
        if self.geweke:
            self.y_prior = None
            self.y = np.random.normal(loc=0.0, scale=1, size=len(self.y))

        # if parameters_from is not None:
        #     with open(
        #             f'./logs/{parameters_from}/parameters.pkl',
        #             'rb') as file:
        #         parameters = pickle.load(file)
        #         self.a = parameters['a']
        #         self.tau = parameters['tau']
        #         self.sigma = parameters['sigma']
        #         self.w = parameters['w']
        #         self.s = parameters['s']
        #         print("Loaded parameters from file")
        #         # print all parameters
        #         print("a: ", self.a)
        #         print("tau: ", self.tau)
        #         print("sigma: ", self.sigma)
        #         print("w: ", self.w)
        #         print("s: ", self.s)

    def sample_geweke(self):
        # y from posterior
        self.y = self.sigma * np.random.normal(0, 1, self.y.shape)
        if len(self.w) > 1 and self.w[0] != 0:
            self.y += self.xs @ self.w

        # y from prior
        sigma = np.random.gamma(self.metadata['priors']['sigma_alpha_prior'],
                                scale=1 / self.metadata['priors']['sigma_beta_prior'], size=1)[0] ** -0.5
        tau = np.random.gamma(self.metadata['priors']['tau_alpha_prior'],
                              scale=1 / self.metadata['priors']['tau_beta_prior'], size=1)[0] ** -0.5
        new_a = np.random.beta(self.metadata['priors']['a_alpha_prior'], self.metadata['priors']['a_beta_prior'], size=1)[0]
        s = np.random.binomial(1, new_a, size=self.x.shape[1])
        w = np.random.normal(loc=0.0, scale=tau, size=self.x.shape[1])
        w_s = s * w
        self.y_prior = self.x @ w_s + sigma * np.random.normal(0, 1, self.y.shape)
        y_pairs = (self.y, self.y_prior)
        sigma_pairs = (self.sigma, sigma)
        tau_pairs = (self.tau, tau)
        a_pairs = (self.a, new_a)
        s_pairs = (sum(self.s), sum(s))
        self.geweke_pairs[0].append(y_pairs)
        self.geweke_pairs[1].append(sigma_pairs)
        self.geweke_pairs[2].append(tau_pairs)
        self.geweke_pairs[3].append(a_pairs)
        self.geweke_pairs[4].append(s_pairs)

    def increment_s(self):
        self.current_size = sum(self.s)
        self.xs = get_xs(self.s, self.x)
        self.sample_w()
        if self.s_to_sample == self.x.shape[1] - 1:
            self.sample_a()
            self.sample_tau()
            self.sample_sigma()

            if self.geweke:
                self.sample_geweke()

            self.get_log_prob()
            self.calculate_mse()
            self.metadata['chain_samples']['s'].append(copy.deepcopy(self.s))
            self.add_s_to_heatmap()
            self.s_to_sample = 0
            return False

        else:
            self.s_to_sample += 1
            return True

    def get_log_prob(self):
        sum_s = sum(self.s)
        len_s = len(self.s)
        numerator = self.a ** sum_s * (1 - self.a) ** (len_s - sum_s)
        denominator = self.tau ** sum_s * (2 * np.pi * self.sigma ** 2) ** (len_s / 2)
        logs = np.log(numerator / denominator)
        xyt_sig_xty, sigma_inverse = self.get_values_for_s(self.s, get_xs(self.s, self.x))
        ytysig = (self.y.T @ self.y) * self.sigma ** 2
        log_prob = (logs + (xyt_sig_xty - ytysig) / (2 * self.sigma ** 4))
        self.metadata['chain_samples']['log_probs'].append(log_prob)

    def sample_a(self):

        alpha = self.current_size + self.metadata['priors']['a_alpha_prior']
        beta = len(self.s) - self.current_size + self.metadata['priors']['a_beta_prior']
        self.a = np.random.beta(alpha, beta, size=1)[0]

        self.metadata['chain_samples']['a']['a'].append(self.a)
        self.metadata['chain_samples']['a']['beta'].append(beta)
        self.metadata['chain_samples']['a']['alpha'].append(alpha)

    def sample_w(self, in_cycle=True):

        if self.current_size == 0:
            self.w = np.array([0])
            self.metadata['chain_samples']['w'].append(copy.deepcopy(self.w))
            return

        xty = self.xs.T @ self.y
        xtx = self.xs.T @ self.xs
        sigma_1 = (self.sigma ** 2) * np.eye(sum(self.s)) + (self.tau ** 2) * xtx
        sigma_2 = (self.tau ** (-2)) * np.eye(sum(self.s)) + (self.sigma ** (-2)) * xtx
        mu = (self.tau ** 2) * solve_with_cholesky(sigma_1, xty)
        # mu = (self.tau ** 2) * np.linalg.inv(sigma_1) @ xty

        # n = mu.shape[0]
        # u = np.random.randn(n)
        # L = np.linalg.cholesky(sigma_2)
        # self.w = mu + L @ u

        w = np.random.multivariate_normal(mu, np.linalg.inv(sigma_2))
        self.w = w
        self.metadata['chain_samples']['w'].append(copy.deepcopy(self.w))

    def sample_sigma(self):
        if self.current_size == 0:
            e = self.y
        else:
            xw = self.xs @ self.w
            e = self.y - xw

        alpha = (len(self.xs) / 2) + self.metadata['priors']['sigma_alpha_prior']
        beta = 1 / (0.5 * e.T @ e + self.metadata['priors']['sigma_beta_prior'])

        self.sigma = np.random.gamma(alpha, scale=beta, size=1)[0]
        self.sigma = self.sigma ** -0.5

        self.metadata['chain_samples']['sigma']['beta'].append(1 / beta)
        self.metadata['chain_samples']['sigma']['alpha'].append(alpha)
        self.metadata['chain_samples']['sigma']['sigma'].append(self.sigma)

        # if self.sigma < 0.0001:
        #     self.warning_values()

    def sample_tau(self):

        alpha = self.metadata['priors']['tau_alpha_prior'] + self.current_size / 2
        beta = 1 / (0.5 * (self.w.T @ self.w) + self.metadata['priors']['tau_beta_prior'])
        self.tau = np.random.gamma(alpha, scale=beta, size=1)[0] ** -0.5

        self.metadata['chain_samples']['tau']['alpha'].append(alpha)
        self.metadata['chain_samples']['tau']['beta'].append(1 / beta)
        self.metadata['chain_samples']['tau']['tau'].append(self.tau)

        # if self.tau < 0.0001:
        #     self.warning_values()

    def sample_s(self):
        s0 = copy.deepcopy(self.s)
        s0[self.s_to_sample] = 0
        s1 = copy.deepcopy(self.s)
        s1[self.s_to_sample] = 1

        # This part adds the new col to index 0, so the trick of determinant ratio will work
        xs_0 = get_xs(s0, self.x)
        col_k = self.x[:, self.s_to_sample][:, np.newaxis]
        xs_1 =  np.hstack((col_k, xs_0))

        ytx_sig_xty_0, sigma_inverse_0 = self.get_values_for_s(s0, xs_0)
        ytx_sig_xty_1, sigma_inverse_1 = self.get_values_for_s(s1, xs_1)

        exp_subtraction = (ytx_sig_xty_1 - ytx_sig_xty_0) / (2 * (self.sigma ** 4))
        s1_prob = 0
        if 30 > exp_subtraction > -30:
            det_ratio = self.dets_ratio(sigma_inverse_1, sigma_inverse_0, xs_0, xs_1, s0, s1)
            det_ratio = det_ratio ** (1 / 2)
            pre_exp_expression = det_ratio * self.a / ((1 - self.a) * self.tau)
            exp = np.exp(exp_subtraction)
            s1_prob = 1 - (1 / (1 + pre_exp_expression * exp))

        elif exp_subtraction > 30:
            s1_prob = 1
        elif exp_subtraction < -30:
            s1_prob = 0

        if random.random() < s1_prob:
            self.s[self.s_to_sample] = 1
        else:
            self.s[self.s_to_sample] = 0

        return self.increment_s()

    def get_values_for_s(self, s, xs):
        xtx = xs.T @ xs
        xty = xs.T @ self.y

        if self.tau == 0:
            sigma_inverse = (self.sigma ** -2) * xtx
        else:
            sigma_inverse = (self.tau ** -2) * np.eye(sum(s)) + (self.sigma ** -2) * xtx
        ytx_sig_xty = solve_with_cholesky(sigma_inverse, xty.T) @ xty

        return ytx_sig_xty, sigma_inverse

    def dets_ratio(self, sigma1, sigma0, xs_0, xs_1, s0, s1, use_fast=False):
        a11 = sigma1[0][0]
        v = sigma1[0][1:]
        L = np.linalg.cholesky(sigma0)
        y = np.linalg.solve(L, v.T)
        vtsig = y.T @ y
        denominator = a11 - vtsig
        new_ratio = 1 / denominator
        return new_ratio

    def sample_gweke(self):
        pass
        # # y from posterior
        # self.y = self.sigma * np.random.normal(0, 1, self.y.shape)
        # if len(self.w) > 1 and self.w[0] != 0:
        #     self.y += self.xs @ self.w
        #
        # # y from prior
        # sigma = np.random.gamma(self.sigma_alpha_prior, scale=1 / self.sigma_beta_prior, size=1)[0] ** -0.5
        # tau = np.random.gamma(self.tau_alpha_prior, scale=1 / self.tau_beta_prior, size=1)[0] ** -0.5
        # new_a = np.random.beta(self.a_alpha_prior, self.a_beta_prior, size=1)[0]
        # s = np.random.binomial(1, new_a, size=self.x.shape[1])
        # w = np.random.normal(loc=0.0, scale=tau, size=self.x.shape[1])
        # w_s = s * w
        # y_prior = self.x @ w_s + sigma * np.random.normal(0, 1, self.y.shape)
        # y_pairs = (self.y, y_prior)
        # sigma_pairs = (self.sigma, sigma)
        # tau_pairs = (self.tau, tau)
        # a_pairs = (self.a, new_a)
        # s_pairs = (sum(self.s), sum(s))
        # self.gweke_pairs[0].append(y_pairs)
        # self.gweke_pairs[1].append(sigma_pairs)
        # self.gweke_pairs[2].append(tau_pairs)
        # self.gweke_pairs[3].append(a_pairs)
        # self.gweke_pairs[4].append(s_pairs)

    def calculate_mse(self):
        if sum(self.s) == 0:
            y_hat = np.zeros(self.y.shape)
        else:
            y_hat = self.xs @ self.w
        self.metadata['chain_samples']['mse'].append(np.mean((self.y - y_hat) ** 2))

    def add_s_to_heatmap(self):
        for i in range(len(self.s)):
            for j in range(len(self.s)):
                self.heatmap[i][j] += self.s[i] * self.s[j]

    def gen_s(self):
        self.s = np.random.randint(low=0, high=2, size=self.x.shape[1])
        while sum(self.s) == 0:
            self.s = np.random.randint(low=0, high=2, size=self.x.shape[1])

    def init_vars(self):
        self.gen_s()
        self.metadata['chain_samples']['s'].append(copy.deepcopy(self.s))
        self.xs = get_xs(self.s, self.x)
        self.sigma = np.random.gamma(self.metadata['priors']['sigma_alpha_prior'],
                                     scale=1 / self.metadata['priors']['sigma_beta_prior'], size=1)[0] ** -0.5
        self.metadata['chain_samples']['sigma']['sigma'].append(self.sigma)
        self.tau = \
            np.random.gamma(self.metadata['priors']['tau_alpha_prior'],
                            scale=1 / self.metadata['priors']['tau_beta_prior'],
                            size=1)[0] ** -0.5
        self.metadata['chain_samples']['tau']['tau'].append(self.tau)
        self.a = \
            np.random.beta(self.metadata['priors']['a_alpha_prior'], self.metadata['priors']['a_beta_prior'], size=1)[0]
        self.metadata['chain_samples']['a']['a'].append(self.a)
        self.w = np.random.normal(loc=0, scale=self.tau, size=(1, sum(self.s)))
        self.metadata['chain_samples']['w'].append(self.w)
        self.get_log_prob()
        self.heatmap = np.zeros((len(self.s), len(self.s)))
        self.add_s_to_heatmap()

    def save(self):
        if not self.geweke:
            # Get the current script directory
            current_directory = os.path.dirname(os.path.abspath(__file__))

            # Construct the save path and create directory
            save_path = os.path.join(current_directory, self.metadata['data']['output_path'])
            os.makedirs(save_path)

            # save results
            file_path_pickle = os.path.join(save_path, 'data.pkl')
            file_path_txt = os.path.join(save_path, 'data.txt')
            with open(file_path_pickle, 'wb') as file:
                pickle.dump(self.metadata, file)

            # save txt for inspection
            keys_to_remove = ['chain_samples', 'normalized_data', 'data']
            reduced_metadata = {k: v for k, v in self.metadata.items() if k not in keys_to_remove}

            with open(file_path_txt, 'w') as file:
                file.write(json.dumps(reduced_metadata, indent=4))

            visualize_s(self.metadata)
        else:
            geweke_plot(self.geweke_pairs)

    def cycle(self):
        carry_on = True
        while carry_on:
            carry_on = self.sample_s()

    def run(self):
        self.init_vars()
        print("init a: " + str(self.a))
        print("init sigma: " + str(self.sigma))
        print("init tau: " + str(self.tau))
        print("init s: " + str(self.s))
        for self.current_iter in tqdm.tqdm(range(self.iter)):
            self.cycle()

        # self.s, self.s_ratios = f.binary_ratio(self.all_s, K=0.2, p=1, a_list=self.all_a)
        # self.xs = f.get_xs(self.s, self.x)
        # self.sample_w()
        # print("final a: " + str(np.mean(f.throw_ratio(self.all_a, 0.2))))
        # print("final sigma: " + str(np.mean(f.throw_ratio(self.all_sigma, 0.2))))
        # print("final tau: " + str(np.mean(f.throw_ratio(self.all_tau, 0.2))))
        # print("final W: " + str(self.w))
        # print("final s: " + str(self.s))
        # print("final s ratios: " + str(self.s_ratios))
        self.save()
        # self.animate()
        return self.s
