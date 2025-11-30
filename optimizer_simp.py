# optimizer_simp.py
import numpy as np
from scipy.signal import convolve2d
import physics_engine as phys

def run_simp(rmin, ris_params, p_bs, p_ue, max_loop=50, return_history=False, obstacle=None):
    #99-line function implementing SIMP optimization for RIS topology
    # Get Physics Map
    potential_base, N, shape = phys.get_sensitivity_map(p_bs, p_ue, ris_params, obstacle=obstacle)
    
    # SIMP Parameters
    volfrac = 0.25
    penal = 3.0
    
    # Initialize
    x = np.ones(N) * volfrac
    
    # Create Filter Kernel
    kernel_rad = int(np.ceil(rmin))
    k_size = kernel_rad * 2 + 1
    kernel = np.zeros((k_size, k_size))
    center = k_size // 2
    for i in range(k_size):
        for j in range(k_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist <= rmin:
                kernel[i, j] = rmin - dist
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)

    history = []

    # Optimization Loop
    for loop in range(max_loop):
        # Record history if requested
        if return_history:
            history.append(x.copy().reshape(shape))

        # Sensitivity
        dc = penal * (x ** (penal - 1)) * potential_base
        
        # Filtering
        dc_2d = dc.reshape(shape)
        dc_filtered = convolve2d(dc_2d, kernel, mode='same', boundary='symm')
        dc = dc_filtered.flatten()
        
        # OC Update (Bisection)
        l1, l2 = 0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-3:
            l_mid = 0.5 * (l2 + l1)
            B_k = np.sqrt(dc / (l_mid + 1e-10))
            x_new = np.maximum(0.0, np.maximum(x - 0.2, 
                              np.minimum(1.0, np.minimum(x + 0.2, x * B_k))))
            if np.sum(x_new) > volfrac * N: l1 = l_mid
            else: l2 = l_mid
        x = x_new
    
    # Return result based on mode
    if return_history:
        history.append(x.reshape(shape)) # Add final frame
        return history
    else:
        return x.reshape(shape)