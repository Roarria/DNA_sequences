import json 
import numpy as np
import argparse 

 
# Read-in params:
def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False,
                        help='File with parameters (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False,
                        help='File to save generated data (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output

    
# Get file paths
param_file, output_file = ParseArguments()
 
# Load parameters (default: from params_set1.json)
with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)
 

# Extract parameters
w=params['w']
k=params['k']
alpha=params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB =np.asarray(params['ThetaB'])


# Generate sequences X = [x_1, ..., x_k], each x_i = (x_{i1}, ..., x_{iw})
# Each nucleotide is coded as 1 (A), 2 (C), 3 (G), or 4 (T)

X = np.zeros((k, w), dtype=int)

for i in range(k):
    is_motif = np.random.rand() < alpha
    if is_motif:
        # Sample from motif Θ, position-specific:
        for j in range(w):
            X[i, j] = np.random.choice([1, 2, 3, 4], p=Theta[:, j])
    else:
        # Sample from background ΘB, same at each position:
        X[i, :] = np.random.choice([1, 2, 3, 4], size=w, p=ThetaB)


gen_data = {    
    "alpha" : alpha,
    "X" : X.tolist()
    }



with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
 
