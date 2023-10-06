import numpy as np
import os
import matplotlib.pyplot as plt


def visualize_s(metadata):
    title_size = 50
    label_size = 40

    s_values = metadata['chain_samples']['s']
    throw_ratio = metadata['run_metadata']['throw_ratio']
    log_probs = metadata['chain_samples']['log_probs']
    feature_names = metadata['data']['col_names']
    train_dir = metadata['data']['output_path']

    # Remove the first N values from S and L
    if metadata['run_metadata']['throw']:
        N = int(throw_ratio * len(metadata['chain_samples']['s']))
    else:
        N = len(s_values)
    S = np.array(s_values[N:])
    L = np.array(log_probs[N:])

    # Convert S and L to numpy arrays
    S = np.array(S)
    L = np.array(L)

    # Create a binary matrix of the binary vectors in S

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(55, 35))
    fig.subplots_adjust(hspace=0.5)
    ax1.yaxis.tick_left()
    if len(feature_names) == 0:
        ax1.set_yticks(np.arange(S.shape[1]), fontsize=label_size - 10)
        ax1.set_yticklabels(np.arange(1, S.shape[1] + 1), fontsize=label_size - 10)
    else:
        ax1.set_yticks(range(S.shape[1]))
        ax1.set_yticklabels(feature_names[:S.shape[1]], fontsize=label_size - 10)
    ax1.set_xlabel('Iteration', fontsize=label_size)
    ax1.set_ylabel('Variable', fontsize=label_size)
    ax1.set_title('S Vector Matrix', fontsize=title_size)
    ax1.tick_params(axis='x', labelsize=label_size)

    # Sort the binary vectors by log-likelihood and plot the sorted binary matrix
    sort_indices = np.argsort(L)[::-1]
    sorted_S = S[sort_indices]

    # Create a binary matrix of the sorted binary vectors
    ax2.yaxis.tick_left()
    if len(feature_names) == 0:
        ax2.set_yticks(np.arange(S.shape[1]), fontsize=label_size - 10)
        ax2.set_yticklabels(np.arange(1, S.shape[1] + 1), fontsize=label_size - 10)
    else:
        ax2.set_yticks(range(S.shape[1]))
        ax2.set_yticklabels(feature_names[:S.shape[1]], fontsize=label_size - 10)
    ax2.set_xlabel('Log-Likelihood', fontsize=label_size)
    ax2.set_ylabel('Variable', fontsize=label_size)
    ax2.set_title('S Vector Matrix (Sorted by Likelihood)', fontsize=title_size)
    sorted_L = np.round(L[sort_indices], 2)
    n = int(len(s_values) / 5)  # For example, set a tick every 5 values.
    indices = np.arange(0, len(sorted_L), n)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(sorted_L[indices], fontsize=label_size)

    # Display the resulting image
    ax1.imshow(S.T, cmap='viridis', aspect='auto')
    ax2.imshow(sorted_S.T, cmap='viridis', aspect='auto')

    # Get the current script directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the save path
    save_path = os.path.join(current_directory, metadata['data']['output_path'], 's_progress.png')

    # Ensure the directory exists
    save_directory = os.path.dirname(save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the file
    plt.savefig(save_path)
    plt.show()


def geweke_plot(geweke_pairs):
    # Create the Gweke plot
    names = ['Y', "Sigma", "Tau", "a", "Number of S's"]
    fig, axes = plt.subplots(nrows=len(names), ncols=2, figsize=(12, 20))
    throw_away = 0.3
    for i in range(len(names)):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]

        # Flatten the vectors
        data = geweke_pairs[i]
        if i == 0:
            x = [val for vec in data[int(len(data) * throw_away):] for val in vec[0]]
            y = [val for vec in data[int(len(data) * throw_away):] for val in vec[1]]
        else:
            x = [pair[0] for pair in data[int(len(data) * throw_away):]]
            y = [pair[1] for pair in data[int(len(data) * throw_away):]]

        # Define bin edges for the histograms
        if names[i] == 'Y':
            bins = np.arange(min(x), max(x) + 1, 1)
        elif names[i] == 'a':
            bins = np.arange(min(x), max(x) + 0.05, 0.05)
        else:
            bins = np.arange(min(x), max(x) + 0.1, 0.1)

        # Create the histograms
        hist1, bin_edges, _ = ax1.hist(x, bins=bins, alpha=0.5, density=True,
                                       label=f'{names[i]} From Chain (Mean: {str(round(np.mean(x), 4))}, STD: {str(round(np.std(x), 4))})')

        if names[i] == 'Y':
            bins = np.arange(min(y), max(y) + 1, 1)
        elif names[i] == 'a':
            bins = np.arange(min(y), max(y) + 0.05, 0.05)
        else:
            bins = np.arange(min(y), max(y) + 0.1, 0.1)

        hist2, bin_edges, _ = ax1.hist(y, bins=bins, alpha=0.5, density=True,
                                       label=f'{names[i]} From Priors (Mean: {str(round(np.mean(y), 4))}, STD: {str(round(np.std(y), 4))})')
        ax1.set_xlabel(f'Distribution of {names[i]}')
        ax1.set_title(f'Distribution Histograms - {names[i]}')
        ax1.legend()

        # Add Q-Q plot
        prior_quantiles = np.quantile(x, np.arange(0, 1, 0.01))
        posterior_quantiles = np.quantile(y, np.arange(0, 1, 0.01))

        ax2.scatter(x=prior_quantiles, y=posterior_quantiles, label='Actual fit')
        ax2.plot(prior_quantiles, prior_quantiles, color='red', label='Line of perfect fit')
        ax2.set_xlabel(f'Quantile of {names[i]} from Prior')
        ax2.set_ylabel(f'Quantile of {names[i]} from Chain')
        ax2.legend()
        ax2.set_title(f"QQ plot - {names[i]}");

    fig.subplots_adjust(hspace=0.5)
    # Show the plot
    plt.show()
