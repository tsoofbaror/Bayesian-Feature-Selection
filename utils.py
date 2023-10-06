import numpy as np
import os
import pandas as pd

def load_and_split_data(filepath):
    """
    Loads a csv file and splits it into x and y.
    """
    # Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File {filepath} not found!")

    # Check if the file is a csv
    if not filepath.endswith('.csv'):
        raise ValueError("Error: File is not a CSV!")

    # Load the CSV into a DataFrame
    df = pd.read_csv(filepath)

    # Check if 'target' column exists in the DataFrame
    if 'target' not in df.columns:
        raise ValueError("Error: 'target' column not found! Please rename the predicted column to 'target'")

    # Split data into X and y
    X = df.drop(columns=['target']).values
    y = df['target'].values
    col_names = df.columns.tolist()

    return np.array(X), np.array(y), col_names


def normalize(matrix, mean=None, std=None):

    if matrix.ndim == 1:
        col_means = np.array([matrix.mean()])
        col_stds = np.array([matrix.std()])
    else:
        col_means = matrix.astype(float).mean(axis=0)
        col_stds = matrix.astype(float).std(axis=0)

    if mean is not None:
        if len(mean) != len(col_means):
            raise ValueError(f"Provided mean has shape {mean.shape} but expected shape {(len(col_means),)}")
        col_means = mean

    if std is not None:
        if len(std) != len(col_stds):
            raise ValueError(f"Provided std has shape {std.shape} but expected shape {(len(col_stds),)}")
        col_stds = std

    matrix_centered = matrix - col_means
    epsilon = 1e-10  # or any small value
    col_stds_eps = np.where(col_stds == 0, epsilon, col_stds)
    matrix_normalized = matrix_centered / col_stds_eps
    return matrix_normalized, col_means, col_stds


def get_xs(s, x):
    """
    Returns the columns of x corresponding to the non-zero elements of s.
    """
    # find the indices of the non-zero elements of V
    indices = np.nonzero(s)[0]

    # select the columns of M with the non-zero indices
    xs = x[:, indices]

    return xs


def solve_with_cholesky(A, V):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, V)
    x = np.linalg.solve(L.T, y)
    return x


def logsumexp(arr):
    b = np.max(arr)
    return b + np.log(np.sum(np.exp(arr - b)))


def transform(s, w):
    w_index = 0
    res = []
    for val in s:
        if val == 1:
            res.append(w[w_index])
            w_index += 1
        else:
            res.append(0)
    return np.array(res)



# Data Preprocessing For Test Datasets


def import_chemical_data_regression():
    data = pd.read_excel('chemical_data.xlsx', header=1, sheet_name=1, skiprows=0)
    x_data = data.iloc[:, 1:18]
    y_data = data.iloc[:, 18]
    return x_data, y_data