import numpy as np
import SALib
import SALib.sample
import SALib.sample.sobol
import pathlib
import io
import json
from typing import Tuple



def init_Ab_poisson_first_order_FD(system_size) -> Tuple[np.ndarray,np.ndarray,float]:
    """
    Initialize A and b of system Ax=b for the 1D Poisson equation discretized using first order finite-differences.
    """
    A = np.zeros((system_size,system_size))
    tmp = [-1, 2, -1]
    A[0,0:2] = [2, -1]
    A[-1,-2:] = [-1, 2]
    for i in range(1,system_size-1):
        A[i,i-1:i+2] = tmp
    b = np.ones(system_size)
    return (A, b)

def bin_for_discrete_samples(samples, params_to_investigate):
    """
    Convert continuous samples to discrete samples based on the specified parameter bounds.
    """
    discrete_samples = np.zeros_like(samples)
    for i, param in enumerate(params_to_investigate):
        if params_to_investigate[param]['discrete']:
            # Discretize the sample
            discrete_samples[:, i] = np.round(samples[:, i], decimals=0)
        else:
            # Keep the sample as is
            discrete_samples[:, i] = samples[:, i]
    return discrete_samples

def find_duplicates(samples):
    """
    Find duplicate samples in the array.
    """
    unique_samples, indices_unique, indices_inverse = np.unique(samples, axis=0, return_index=True, return_inverse=True)
    return indices_unique, indices_inverse

def npndarray_to_strbytes(ar, ar_name:str)->str:
    """
    Convert a numpy array to bytes.
    """
    outfile = io.BytesIO()
    if ar_name is None:
        raise ValueError("ar_name must be provided")
    else:
        np.savez_compressed(outfile, **{ar_name:ar})
    AA_save = str(outfile.getvalue())
    
    return AA_save

system_size = 4
A, b = init_Ab_poisson_first_order_FD(system_size=system_size)

params_to_investigate = {
     'c_reg_size': {'bounds': (1, 5), 'discrete': True},
     't_hamiltonian': {'bounds': (0.01, 2*np.pi), 'discrete': False},
     'r_hamiltonian': {'bounds': (1, 10), 'discrete': True},
}

# Define the SALib parameter space
SALib_problem = {
    'num_vars': len(params_to_investigate),
    'names': list(params_to_investigate.keys()),
    'bounds': [list(params_to_investigate[param]['bounds']) for param in params_to_investigate]
}

#num_samples = N(2D+2), D=SALib_problem['num_vars']<=2, N is parameter for sample
#num_samples = N( D+2), D=SALib_problem['num_vars'] >2, N is parameter for sample
#  N \ D |  1  |  2  |  3  |  4  |  5  |
#     4  |  12 |  16 |  32 |  40 |  48 |
#    512 | 1536| 2048| 4096| 5120| 6144|
N = 2
seed = 42
print(f"Number of variables: {SALib_problem['num_vars']}")
if SALib_problem['num_vars'] <= 2:
    num_sample_expected = N * (SALib_problem['num_vars'] + 2)
    print(f"Expected number of samples: {num_sample_expected}")
    samples = SALib.sample.sobol.sample(problem=SALib_problem, N=N, calc_second_order=False, seed=seed)
elif SALib_problem['num_vars'] > 2:
    num_sample_expected = N * (2 * SALib_problem['num_vars'] + 2)
    print(f"Expected number of samples: {num_sample_expected}")
    samples = SALib.sample.sobol.sample(problem=SALib_problem, N=N, calc_second_order=True, seed=seed)



#print(samples)
print(samples.shape)
print(type(samples))



discrete_samples = bin_for_discrete_samples(samples, params_to_investigate)
indices_unique_samples, indices_inverse_samples = find_duplicates(samples)
indices_unique_samples_discrete, indices_inverse_samples_discrete = find_duplicates(discrete_samples)
print(f"Number of unique samples:          {len(indices_unique_samples)}")
print(f"Number of unique samples discrete: {len(indices_unique_samples_discrete)}")
print(f"Number of unique samples discrete: {indices_inverse_samples}")
print(f"Number of unique samples discrete: {indices_inverse_samples_discrete}")
for i, key in enumerate(params_to_investigate.keys()):
    val = params_to_investigate[key]
    if val['discrete']:
        print(f"Parameter {key} is discrete, with bounds {val['bounds']} and has unique values\n\
              {np.unique(discrete_samples[:, i])}")
    else:
        print(f"Parameter {key} is continuous, with bounds {val['bounds']}")

samples_w_id = np.hstack((np.arange(samples.shape[0]).reshape(-1,1), samples))
discrete_samples_w_id = np.hstack((np.arange(samples.shape[0]).reshape(-1,1), discrete_samples))

dict_to_json_prepost = {
    'system_type': 'Poisson',
    'system_size': system_size,
    'params_to_investigate': params_to_investigate,
    'num_samples': samples.shape[0],
    'num_unique_samples': len(indices_unique_samples),
    'num_unique_samples_discrete': len(indices_unique_samples_discrete),
    'indices_unique_samples': npndarray_to_strbytes(indices_unique_samples, ar_name='indices_unique_samples'),
    'indices_inverse_samples': npndarray_to_strbytes(indices_inverse_samples, ar_name='indices_inverse_samples'),
    'indices_unique_samples_discrete': npndarray_to_strbytes(indices_unique_samples_discrete, ar_name='indices_unique_samples_discrete'),
    'indices_inverse_samples_discrete': npndarray_to_strbytes(indices_inverse_samples_discrete, ar_name='indices_inverse_samples_discrete'),
    'samples': npndarray_to_strbytes(samples_w_id, ar_name='samples'),
    'samples_discrete': npndarray_to_strbytes(discrete_samples_w_id, ar_name='samples_discrete'),
    'A': npndarray_to_strbytes(A, ar_name='A'),
    'b': npndarray_to_strbytes(b, ar_name='b'),
    'shots': int(1e6)
    }

#print(samples_w_id.size*samples_w_id.itemsize)

# Save the dictionary as a JSON file
json_file = 'dict_params_prepost.json'
with open(json_file, 'w') as f:
    json.dump(dict_to_json_prepost, f, indent=4)

dirname = pathlib.Path(__file__).parent.joinpath('input_files') if __file__ else pathlib.Path('input_files')
dirname.mkdir(parents=True, exist_ok=True)
print('indices_unique_samples_discrete', indices_unique_samples_discrete)
print(len(indices_unique_samples_discrete))
for i, id in enumerate(indices_unique_samples_discrete):
    dict_to_json_run = {
        'col_names': ['id'] + list(params_to_investigate.keys()),
        'samples_discrete': npndarray_to_strbytes(discrete_samples_w_id[id], ar_name='samples_discrete'),
        'A': npndarray_to_strbytes(A, ar_name='A'),
        'b': npndarray_to_strbytes(b, ar_name='b'),
        'shots': int(1e6)
    }
    filename = f'input_files/dict_params_run_{id}.json'
    with open(filename, 'w') as f:
        json.dump(dict_to_json_run, f, indent=4)


