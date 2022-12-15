import numpy as np
from numpy import linalg as la
import pandas as pd
import json


# Initialize Variables
with open("config/type_chart.json", "r") as outfile:
    type_chart = json.load(outfile)

type_names = list(type_chart.keys())
num_types = len(type_names)

# Compute probability transition matrix
transition_mat = [[(1/num_types) * (1 if type_chart[challenger][king] > type_chart[king][challenger] else (0.5 if type_chart[challenger][king] == type_chart[king][challenger] else 0)) for challenger in type_names] for king in type_names]
for ind in range(num_types):
    transition_mat[ind][ind] = 1 - sum(transition_mat[ind]) + transition_mat[ind][ind]
transition_mat_np = np.array(transition_mat)

# Generate stationary distribution by finding the largest eigenvector
w, v = la.eig(transition_mat_np.transpose())

limit_dist = [vec[0].real for vec in v]
limit_dist = limit_dist/sum(limit_dist)

type_values_df = pd.DataFrame(
    {
        'Type Name': type_names,
        'Probability': list(limit_dist),

    }
)
type_values_df = type_values_df.sort_values(by=['Probability'])

# Calculation of Dual Typings
dual_types = [{'type1': type_names[ind1], 'type2': type_names[ind2]} for ind1 in range(num_types) for ind2 in
              range(num_types) if ind1 > ind2]
num_dual_types = len(dual_types)

transition_mat = []
for king_dual_type in dual_types:
    king_type1, king_type2 = king_dual_type['type1'], king_dual_type['type2']
    king_prob_vec = []
    for challenger_dual_type in dual_types:
        challenger_type1, challenger_type2 = challenger_dual_type['type1'], challenger_dual_type['type2']
        king_atk = max(type_chart[king_type1][challenger_type1] * type_chart[king_type1][challenger_type2],
                       type_chart[king_type2][challenger_type1] * type_chart[king_type2][challenger_type2])
        challenger_atk = max(type_chart[challenger_type1][king_type1] * type_chart[challenger_type1][king_type2],
                             type_chart[challenger_type2][king_type1] * type_chart[challenger_type2][king_type2])
        if king_atk > challenger_atk:
            prob = 0
        elif king_atk == challenger_atk:
            prob = 1/num_dual_types*0.5
        else:
            prob = 1/num_dual_types
        king_prob_vec.append(prob)
    transition_mat.append(king_prob_vec)

for ind in range(num_dual_types):
    king_prob_vec_sum = sum(transition_mat[ind])
    transition_mat[ind][ind] = 1 - king_prob_vec_sum + transition_mat[ind][ind]
transition_mat_np = np.array(transition_mat)

# Generate stationary distribution by finding the largest eigenvector
w, v = la.eig(transition_mat_np.transpose())

limit_dist = [vec[0].real for vec in v]
limit_dist = limit_dist/sum(limit_dist)

dual_type_values_df = pd.DataFrame(
    {
        'Type 1 Name': [name['type1'] for name in dual_types],
        'Type 2 Name': [name['type2'] for name in dual_types],
        'Probability': list(limit_dist),

    }
)
dual_type_values_df = dual_type_values_df.sort_values(by=['Probability'])

