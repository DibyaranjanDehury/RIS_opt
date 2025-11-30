# optimizer_spacing.py
import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
import physics_engine as phys
from matplotlib.patches import Circle, Rectangle

# --- HELPER: CHECK IF POINT IS BLOCKED ---
def is_blocked(y, z, obstacle, grid_params):
    """
    Checks if a continuous point (y, z) is inside the obstacle.
    """
    if obstacle is None:
        return False
 
    # Scale: Meters -> Pixels
    h_idx = y / grid_params['Dh']
    v_idx = z / grid_params['Dv']
    
    if obstacle['type'] == 'circle':
        dist = np.sqrt((h_idx - obstacle['c_h'])**2 + (v_idx - obstacle['c_v'])**2)
        return dist <= obstacle['r']
        
    elif obstacle['type'] == 'cross':
        # Check Horizontal Bar
        in_h_bar = (h_idx >= obstacle['h_min']) and (h_idx <= obstacle['h_max']) and \
                   (v_idx >= obstacle['v_cross_min']) and (v_idx <= obstacle['v_cross_max'])
        # Check Vertical Bar
        in_v_bar = (v_idx >= obstacle['v_min']) and (v_idx <= obstacle['v_max']) and \
                   (h_idx >= obstacle['h_cross_min']) and (h_idx <= obstacle['h_cross_max'])
        return in_h_bar or in_v_bar
        
    return False

def calculate_aperiodic_power_with_obstacle(yz_flat, p_bs, p_ue, obstacle, grid_params):
    c = 3e8
    f_c = 28e9
    lambda_c = c / f_c
    
    yz = yz_flat.reshape(-1, 2)
    N_elem = yz.shape[0]
    
    p_ris = np.zeros((N_elem, 3))
    p_ris[:, 1] = yz[:, 0]
    p_ris[:, 2] = yz[:, 1]
    
    # 1. Physics
    d_t = np.linalg.norm(p_ris - p_bs, axis=1)
    d_r = np.linalg.norm(p_ris - p_ue, axis=1)
    h = (1/d_t) * np.exp(-1j * 2 * np.pi * d_t / lambda_c)
    g = (1/d_r) * np.exp(-1j * 2 * np.pi * d_r / lambda_c)
    
    # 2. Obstacle Check
    valid_mask = []
    for i in range(N_elem):
        if is_blocked(yz[i,0], yz[i,1], obstacle, grid_params):
            valid_mask.append(0) # Element is dead/blocked
        else:
            valid_mask.append(1) # Element is active
    valid_mask = np.array(valid_mask)
    
    # 3. Coherent Sum (Only valid elements contribute)
    # We add a massive penalty if too many elements are blocked to force them out
    combined_signal = np.sum(np.abs(h * g) * valid_mask)
    
    penalty = 0
    if np.sum(valid_mask) < N_elem:
        # Penalize for every blocked element to push GA away from the zone
        penalty = 100 * (N_elem - np.sum(valid_mask))
        
    return -(combined_signal**2) + penalty

def run_spacing_optimization_ga(obstacle=None):
    print("--- STARTING SPACING OPTIMIZATION (With Obstacle Support) ---")
    
    ris_params = phys.get_ris_config("standard")
    ris_params['Nh'] = 1280 
    Board_H = ris_params['Nh'] * ris_params['Dh'] # 6.4m
    Board_V = ris_params['Nv'] * ris_params['Dv'] # 0.32m
    
    p_bs = np.array([2, 1.5, 0.5]) 
    p_ue = np.array([5, 0.32, 0.16])
    
    N_elements = 32
    
    varbound = []
    for _ in range(N_elements):
        varbound.append([0, Board_H]) 
        varbound.append([0, Board_V]) 
    varbound = np.array(varbound)
    
    algorithm_param = {
        'max_num_iteration': 50,
        'population_size': 50,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }
    
    # Run GA
    model = ga(function=lambda x: calculate_aperiodic_power_with_obstacle(x, p_bs, p_ue, obstacle, ris_params), 
               dimension=N_elements*2, 
               variable_type='real', 
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)
    model.run()
    
    return model.output_dict['variable'], p_bs, p_ue, Board_H, Board_V, ris_params