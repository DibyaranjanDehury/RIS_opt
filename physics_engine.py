import numpy as np

def get_ris_config(setup_type="standard"):
    """
    Returns dictionary of RIS parameters
    """
    # Standard setup from Paper Table I
    if setup_type == "standard":
        return {'Nv': 64, 'Nh': 128, 'Dv': 0.005, 'Dh': 0.005, 'f_c': 28e9}
    
    # "Smart Wall" setup for Spacing Optimization (Wide board)
    elif setup_type == "smart_wall":
        return {'Nv': 64, 'Nh': 1280, 'Dv': 0.005, 'Dh': 0.005, 'f_c': 28e9}


def get_sensitivity_map(p_bs, p_ue, ris_params, obstacle=None):
    """
    Calculates Potential Map.
    Supports 'circle', 'rectangle', and 'cross' obstacles.
    """
    Nv, Nh = ris_params['Nv'], ris_params['Nh']
    Dv, Dh = ris_params['Dv'], ris_params['Dh']
    N = Nv * Nh
    
    # 1. Coordinates
    y_coords = np.linspace(0, (Nh - 1) * Dh, Nh)
    z_coords = np.linspace(0, (Nv - 1) * Dv, Nv)
    Y, Z = np.meshgrid(y_coords, z_coords)
    p_ris = np.stack([np.zeros(N), Y.ravel(), Z.ravel()], axis=1)
    
    # 2. Distances & Physics
    d_t = np.linalg.norm(p_ris - p_bs, axis=1)
    d_r = np.linalg.norm(p_ris - p_ue, axis=1)
    h_mag = (1 / d_t) 
    g_mag = (1 / d_r)
    potential_map = h_mag * g_mag
    
    # Obsatcle Handling
    if obstacle is not None:
        pot_2d = potential_map.reshape(Nv, Nh)
        Grid_H, Grid_V = np.meshgrid(np.arange(Nh), np.arange(Nv))
        mask = np.ones((Nv, Nh), dtype=bool)

        # SHAPE 1: CIRCLE (The original)
        if obstacle['type'] == 'circle':
            dist = np.sqrt((Grid_H - obstacle['c_h'])**2 + (Grid_V - obstacle['c_v'])**2)
            mask = dist > obstacle['r']

        # SHAPE 2: RECTANGLE (A Wall)
        elif obstacle['type'] == 'rectangle':
            # Block a region defined by [h_min, h_max] and [v_min, v_max]
            h_mask = (Grid_H >= obstacle['h_min']) & (Grid_H <= obstacle['h_max'])
            v_mask = (Grid_V >= obstacle['v_min']) & (Grid_V <= obstacle['v_max'])
            mask = ~(h_mask & v_mask) # Inverse because we want the valid region

        # SHAPE 3: CROSS 
        elif obstacle['type'] == 'cross':
            # Horizontal Bar
            h_bar = (Grid_H >= obstacle['h_min']) & (Grid_H <= obstacle['h_max']) & \
                    (Grid_V >= obstacle['v_cross_min']) & (Grid_V <= obstacle['v_cross_max'])
            
            # Vertical Bar
            v_bar = (Grid_V >= obstacle['v_min']) & (Grid_V <= obstacle['v_max']) & \
                    (Grid_H >= obstacle['h_cross_min']) & (Grid_H <= obstacle['h_cross_max'])
            
            mask = ~(h_bar | v_bar)

        pot_2d = pot_2d * mask
        potential_map = pot_2d.flatten()
    # -------------------------------

    potential_map = potential_map / np.max(potential_map)
    return potential_map, N, (Nv, Nh)

def calculate_aperiodic_power(yz_flat, p_bs, p_ue, f_c=28e9):
    """
    Calculates power for arbitrary floating elements (Aperiodic).
    Used for Spacing GA optimization.
    """
    c = 3e8
    lambda_c = c / f_c
    
    # Reshape input vector to (N, 2)
    yz = yz_flat.reshape(-1, 2)
    N_elem = yz.shape[0]
    
    # Construct 3D positions (x=0)
    p_ris = np.zeros((N_elem, 3))
    p_ris[:, 1] = yz[:, 0] # y
    p_ris[:, 2] = yz[:, 1] # z
    
    # Physics
    d_t = np.linalg.norm(p_ris - p_bs, axis=1)
    d_r = np.linalg.norm(p_ris - p_ue, axis=1)
    
    h = (1/d_t) * np.exp(-1j * 2 * np.pi * d_t / lambda_c)
    g = (1/d_r) * np.exp(-1j * 2 * np.pi * d_r / lambda_c)
    
    # Coherent Sum
    total_mag = np.sum(np.abs(h * g))
    
    return -(total_mag**2) 