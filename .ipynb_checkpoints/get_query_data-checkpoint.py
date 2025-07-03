import numpy as np
import ast
import re

queries_file = "./queries.txt"
observations_file = "./observations.txt"


with open(queries_file, "r") as file:
    content = file.read()

matches = re.findall(r'array\(\[.*?\]\)', content, re.DOTALL)
inputs = []

for match in matches:
    try:
        inner_list = match[len("array("):-1]
        data = ast.literal_eval(inner_list)
        print(data)
        inputs.append(data)
    except Exception as e:
        print(f"Skipping due to error: {e}")


# for n in range(0,query_data.size, 8):
#     print(n)

def get_inputs(f_num):
    return [inputs[i] for i in range(f_num-1,len(inputs), 8)]



outputs_data = []
with open(observations_file, 'r') as f:
    for line in f:
        outputs_data.append(eval(line.strip()))

def get_outputs(f_num):
    return np.array([o[f_num-1] for o in outputs_data])