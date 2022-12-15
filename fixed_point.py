import numpy as np
import pandas as pd
import json

# Load Configurations and Type Chart
with open("config/type_chart.json", "r") as outfile:
    type_chart = json.load(outfile)

with open("config/calc_configs.json", "r") as outfile:
    calc_configs = json.load(outfile)

weight_calculator = calc_configs['weight_calculator']
meta_exponent = calc_configs['meta_importance_factor']
p = calc_configs['specialization']

# Initialize Variables
type_names = list(type_chart.keys())
num_types = len(type_names)

offensive_values = np.ones([num_types, 1])
defensive_values = np.ones([num_types, 1])
overall_values = np.ones([num_types, 1])

# Construct Step Matrix
off_mat = [[weight_calculator['offense'][str(float(type_chart[j][i]))] for i in type_names] for j in type_names]
off_mat_np = np.array(off_mat)

def_mat = [[weight_calculator['defense'][str(float(type_chart[i][j]))] for i in type_names] for j in type_names]
def_mat_np = np.array(def_mat)

# Iteratively Refine Values
tol = 1e-8
it = 0
change = 1

mean = 100
std = 15

while change > tol:
    # compute new offense value & normalize
    old_offensive_values = offensive_values
    offensive_values = np.matmul(off_mat_np, (defensive_values ** meta_exponent['offense']))
    offensive_values = (offensive_values - np.mean(offensive_values)) / np.std(offensive_values) * std + mean

    # compute new defense value & normalize
    defensive_values = np.matmul(def_mat_np, (old_offensive_values ** meta_exponent['defense']))
    defensive_values = (defensive_values - np.mean(defensive_values)) / np.std(defensive_values) * std + mean

    # compute new overall value
    old_overall_values = overall_values
    overall_values = (offensive_values ** p + defensive_values ** p) ** (1 / p)
    overall_values = (overall_values - np.mean(overall_values)) / np.std(overall_values) * std + mean

    # prep next iter
    change = np.linalg.norm(overall_values - old_overall_values)
    it = it + 1

# Process output
type_values_dict = {type_names[ind]: {'offense': float(offensive_values[ind]),
                                      'defense': float(defensive_values[ind]),
                                      'overall': float(overall_values[ind])}
                    for ind in range(num_types)}
type_values_df = pd.DataFrame(
    {
        'Type Name': type_names,
        'Offensive Value': list(offensive_values),
        'Defensive Value': list(defensive_values),
        'Overall Value': list(overall_values)
    }
)
type_values_df = type_values_df.sort_values(by=['Overall Value'])

# Calculation of Dual Typings
dual_types = [{'type1': type_names[ind1], 'type2': type_names[ind2]} for ind1 in range(num_types) for ind2 in
              range(num_types) if ind1 > ind2]

off_mat = [[weight_calculator['offense'][
                str(float(max(type_chart[dual_type['type1']][def_type], type_chart[dual_type['type2']][def_type])))]
            for def_type in type_names] for dual_type in dual_types]
off_mat_np = np.array(off_mat)

def_mat = [[weight_calculator['defense'][
                str(float(type_chart[def_type][dual_type['type1']] * type_chart[def_type][dual_type['type2']]))] for
            def_type in type_names] for dual_type in dual_types]
def_mat_np = np.array(def_mat)

dual_offensive_values = np.matmul(off_mat_np, (defensive_values ** meta_exponent['offense']))
dual_offensive_values = (dual_offensive_values - np.mean(dual_offensive_values)) / np.std(
    dual_offensive_values) * std + mean

dual_defensive_values = np.matmul(def_mat_np, (offensive_values ** meta_exponent['defense']))
dual_defensive_values = (dual_defensive_values - np.mean(dual_defensive_values)) / np.std(
    dual_defensive_values) * std + mean

dual_overall_values = (dual_offensive_values ** p + dual_defensive_values ** p) ** (1 / p)

# Producing similar output for the multitype cases
dual_type_values_dict = {
    (dual_types[ind]['type1'], dual_types[ind]['type2']): {'offense': float(dual_offensive_values[ind]),
                                                           'defense': float(dual_defensive_values[ind]),
                                                           'overall': float(dual_overall_values[ind])}
    for ind in range(len(dual_types))}
dual_type_values_df = pd.DataFrame(
    {
        'First Type': [dual_types[ind]['type1'] for ind in range(len(dual_types))],
        'Second Type': [dual_types[ind]['type2'] for ind in range(len(dual_types))],
        'Offensive Value': list(dual_offensive_values),
        'Defensive Value': list(dual_defensive_values),
        'Overall Value': list(dual_overall_values)
    }
)
dual_type_values_df = dual_type_values_df.sort_values(by=['Overall Value'])
