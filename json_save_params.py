import json 
import numpy as np
import argparse 

 
# Example: this was params will be stored:

w = 3

# Position weight matrix:
np.random.seed(23)
tmp = np.random.dirichlet([1]*4, w)   # shape (w, 4)
Theta = tmp.T
print(Theta)

# Background distribution
ThetaB = np.random.dirichlet([1]*4)       # shape (4,)
print(ThetaB)

params = {
    "w" : w,    # 3
    "alpha" : 0.5,
    "k" : 10,
    "Theta" : Theta.tolist(),
    "ThetaB" : ThetaB.tolist()
    }

#Note: matrices cannot be provided above — they are converted to lists here
#(later, after loading, just convert them back to a matrix using e.g. np.asarray(.))


with open('#params_set1.json', 'w') as outfile:  
    json.dump(params, outfile)

    
