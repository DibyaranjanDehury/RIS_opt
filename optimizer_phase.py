# optimizer_phase.py
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import physics_engine as phys

def run_discrete_phase_optimization(bit_depth, ris_params, p_bs, p_ue):
    """
    Optimizes the PHASE of each element given a bit-depth constraint.
    bit_depth = 1 -> Phases can only be [0, pi]
    bit_depth = 2 -> Phases can only be [0, pi/2, pi, 3pi/2]
    """
    print(f"--- STARTING {bit_depth}-BIT PHASE OPTIMIZATION ---")
    
    # Get Physics
    # We need the complex channels h and g separately
    # (Re-calculating briefly here for clarity)
    Nv, Nh = ris_params['Nv'], ris_params['Nh']
    Dv, Dh = ris_params['Dv'], ris_params['Dh']
    f_c = ris_params['f_c']
    c, lambda_c = 3e8, 3e8 / f_c
    N = Nv * Nh
    
    # Coordinates
    y_coords = np.linspace(0, (Nh - 1) * Dh, Nh)
    z_coords = np.linspace(0, (Nv - 1) * Dv, Nv)
    Y, Z = np.meshgrid(y_coords, z_coords)
    p_ris = np.stack([np.zeros(N), Y.ravel(), Z.ravel()], axis=1)
    
    d_t = np.linalg.norm(p_ris - p_bs, axis=1)
    d_r = np.linalg.norm(p_ris - p_ue, axis=1)
    
    # Complex Channels
    h = (1/d_t) * np.exp(-1j * 2 * np.pi * d_t / lambda_c)
    g = (1/d_r) * np.exp(-1j * 2 * np.pi * d_r / lambda_c)
    cascaded_channel = h * g # vector of size N
    
    # Discrete Phase Levels
    num_levels = 2**bit_depth
    possible_phases = np.linspace(0, 2*np.pi, num_levels, endpoint=False)
    
    N_opt = 64 # Optimization variables
    
    def objective_func(x):
        # x contains integers [0, 1, ... num_levels-1]
        indices = x.astype(int)
        phases = possible_phases[indices]
        phase_shifts = np.exp(1j * phases)
        
        signal = np.sum(cascaded_channel[:N_opt] * phase_shifts)
        
        return -(np.abs(signal)**2)

    varbound = np.array([[0, num_levels - 1]] * N_opt)
    
    algorithm_param = {
        'max_num_iteration': 200,
        'population_size': 40,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }

    model = ga(function=objective_func, dimension=N_opt, 
               variable_type='int', 
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)
    
    model.run()
    
    # Return best power (positive)
    return -model.output_dict['function']