import numpy as np
import pandas as pd
import json
from scipy import optimize


# Initialize Variables
with open("config/type_chart.json", "r") as outfile:
    type_chart = json.load(outfile)

type_names = list(type_chart.keys())
num_types = len(type_names)

# Construct the payoff matrix
payoff_mat = []
for p1 in type_names:
    payoff_vec = []
    for p2 in type_names:
        if type_chart[p2][p1] > type_chart[p1][p2]:
            payoff = 1
        elif type_chart[p1][p2] > type_chart[p2][p1]:
            payoff = -1
        else:
            payoff = 0
        payoff_vec.append(payoff)
    payoff_vec.append(-1)
    payoff_mat.append(payoff_vec)
payoff_mat_np = np.array(payoff_mat)

obj = np.array([0]*num_types + [1])
prob_sum = np.array([[1]*num_types + [0]])
lower_bound = np.array([0]*num_types + [-np.inf])
bounds = [(0, None) for i in range(num_types)]
bounds.append((None, None))
output = optimize.linprog(c=-obj, A_ub=-payoff_mat_np, b_ub=np.array([0]*num_types), A_eq=prob_sum, b_eq=1, bounds=bounds)

type_values_df = pd.DataFrame(
    {
        'Type': type_names,
        'Probability': [i for i in output.x[:-1]]
    }
)