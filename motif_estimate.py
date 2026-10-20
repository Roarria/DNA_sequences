import json 
import numpy as np
import argparse
import pandas as pd 

 
# Parse command-line arguments

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False,
                        help='File with input data (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False,
                        help='File where the estimated parameters will be saved (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Should alpha be estimated or not? (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha


def log_prob(sequence, matrix):
    """
    Compute the logarithm of the probability of the sequence given the motif matrix (Theta).
    """
    probs = matrix[sequence, np.arange(len(sequence))]
    probs = np.maximum(probs, 1e-12)    # prevent numerical errors by avoiding log(0)
    return np.sum(np.log(probs))

def log_prob_bg(sequence, ThetaB):
    """
    Compute the log-probability of the sequence given the background distribution ThetaB.
    """
    probs = ThetaB[sequence]
    probs = np.maximum(probs, 1e-12)    # prevent numerical errors by avoiding log(0)
    return np.sum(np.log(probs))


def params_init(X):
    k, w = X.shape
    Theta = np.zeros((4, w))

    # Count nucleotide occurrences at each motif position (column)
    for j in range(w):
        counts = np.bincount(X[:, j], minlength=4)  # count occurrences of 0,1,2,3 at position j
        Theta[:, j] = counts + 1e-12  # avoiding zeros
    Theta /= np.sum(Theta, axis=0, keepdims=True)

    # Calculate global nucleotide frequencies for background
    ThetaB = np.bincount(X.flatten(), minlength=4) + 1e-12
    ThetaB /= np.sum(ThetaB)
    
    return Theta, ThetaB


def random_params(w, d):
    # Random initialization of Theta and ThetaB from Dirichlet distributions
    Theta = np.random.dirichlet([1]*d, w).T   # shape (4, w)
    ThetaB = np.random.dirichlet([1]*d)       # shape (4,)
    return Theta, ThetaB


def run_EM(X, alpha, max_iter=500, eps=1e-6, verbose=False, init_random=False):
    """
    EM algorithm to estimate motif matrix Theta and background distribution ThetaB.
    Inputs:
        X - numpy array (k x w) with integer values in [0, 1, 2, 3] representing nucleotides
        alpha - probability that a sequence contains the motif
    Outputs:
        Theta - estimated motif matrix (4 x w)
        ThetaB - estimated background nucleotide distribution (4,)
    """
    k, w = X.shape
    d = 4

    # Initialize parameters either randomly or from frequency counts
    if init_random:
        Theta, ThetaB = random_params(w, d)
    else:
        Theta, ThetaB = params_init(X)

    prev_log_likelihood = -np.inf

    for iter in range(max_iter):
        # E-step: compute posterior probabilities gamma_i = P(motif | sequence i)
        log_p_motif = np.array([log_prob(seq, Theta) for seq in X])
        log_p_bg = np.array([log_prob_bg(seq, ThetaB) for seq in X])

        log_gamma = np.log(alpha) + log_p_motif - np.logaddexp(
            np.log(alpha) + log_p_motif, np.log(1 - alpha) + log_p_bg
            )
        gamma = np.exp(log_gamma)

        # M-step: update Theta and ThetaB as weighted nucleotide counts
        Theta_new = np.zeros((d, w))
        ThetaB_new = np.zeros(d)

        for j in range(w):
            for b in range(d):
                Theta_new[b, j] = np.sum(gamma * (X[:, j] == b))

        for b in range(d):
            mask_b = (X == b)
            ThetaB_new[b] = np.sum((1 - gamma)[:, np.newaxis] * mask_b)

        # Normalize columns of Theta_new and ThetaB_new
        Theta_new /= np.sum(Theta_new, axis=0, keepdims=True)
        ThetaB_new /= np.sum(ThetaB_new)

        Theta = Theta_new
        ThetaB = ThetaB_new

        total_log_likelihood = np.sum(np.logaddexp(
            np.log(alpha) + log_p_motif,
            np.log(1 - alpha) + log_p_bg
        ))

        if verbose:
            print(f"[Iter {iter+1}] log-likelihood = {total_log_likelihood:.4f}, "
                  f"gamma.mean() = {np.mean(gamma):.4f}")

        if np.abs(total_log_likelihood - prev_log_likelihood) < eps:
            break

        prev_log_likelihood = total_log_likelihood

    if verbose:
        df_Theta = pd.DataFrame(Theta)
        df_ThetaB = pd.DataFrame(ThetaB)
        print(df_Theta.round(5))
        print(df_ThetaB.round(5))

    return Theta, ThetaB


if __name__ == "__main__":
    input_file, output_file, estimate_alpha = ParseArguments()
    
    with open(input_file, 'r') as inputfile:
        data = json.load(inputfile)
    

    alpha = float(data['alpha'])  # known value
    X = np.array(data['X'], dtype=int) - 1  # conversion: 1–4 → 0–3

    # TO DO: MAIN PART: Estimate Theta and ThetaB using EM and save to output_file
    Theta, ThetaB = run_EM(X, alpha, verbose=True, init_random=False)

    # BONUS TASK: if estimate_alpha == "yes", then
    # alpha must also be estimated (ignore the value provided in input_file)

    estimated_params = {
        "alpha" : alpha,            # we "copy" this alpha — it was not estimated
        "Theta" : Theta.tolist(),   # estimated
        "ThetaB" : ThetaB.tolist()  # estimated
        }


    with open(output_file, 'w') as outfile:
        json.dump(estimated_params, outfile)
        
