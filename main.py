import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import physics_engine as phys
import optimizer_simp as simp_opt
import optimizer_spacing as spacing_opt
import optimizer_phase as phase_opt
import optimizer_comparison as algo_comp 

# --- EXPERIMENT 1: Shape (SIMP) ---
def experiment_1_shape():
    print("\n=== EXP 1: Shape Optimization (SIMP) ===")
    ris_params = phys.get_ris_config("standard")
    p_bs = np.array([50, 25, 2])
    p_ue = np.array([5, 0.32, 0.16])
    
    res_micro = simp_opt.run_simp(1.0, ris_params, p_bs, p_ue)
    res_macro = simp_opt.run_simp(8.0, ris_params, p_bs, p_ue)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(res_micro, cmap='gray_r'); ax1.set_title("Micro-Structure (r=1.0)")
    ax2.imshow(res_macro, cmap='gray_r'); ax2.set_title("Macro-Shape (r=8.0)")
    plt.savefig("results/Exp1_Shape.png")
    plt.show()

# --- EXPERIMENT 2: Spacing (GA) ---
def experiment_2_spacing():
    print("\n=== EXP 2: Spacing Optimization (GA) ===")
    # The plotting logic is inside the module
    spacing_opt.run_spacing_optimization_ga()

# --- EXPERIMENT 3: Phase Quantization ---
def experiment_3_quantization():
    print("\n=== EXP 3: Phase Quantization Impact ===")
    ris_params = phys.get_ris_config("standard")
    p_bs = np.array([50, 25, 2])
    p_ue = np.array([5, 0.32, 0.16])
    
    potentials, _, _ = phys.get_sensitivity_map(p_bs, p_ue, ris_params)
    top_64_mag = np.sort(potentials.flatten())[-64:]
    power_continuous = (np.sum(top_64_mag))**2 * 100 
    
    power_1bit = phase_opt.run_discrete_phase_optimization(1, ris_params, p_bs, p_ue)
    power_2bit = phase_opt.run_discrete_phase_optimization(2, ris_params, p_bs, p_ue)
    
    labels = ['1-Bit', '2-Bit', 'Continuous (Ideal)']
    values = [power_1bit, power_2bit, power_continuous]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['red', 'orange', 'green'])
    plt.ylabel("Received Power")
    plt.title("Impact of Hardware Quantization")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1e}", va='bottom', ha='center')
    plt.savefig("results/Exp3_Quantization.png")
    plt.show()

# --- EXPERIMENT 4: Element Sizing ---
def experiment_4_element_sizing():
    print("\n=== EXP 4: Element Sizing Sweep ===")
    sizes = np.linspace(0.002, 0.010, 10)
    powers = []
    f_c = 28e9
    c = 3e8
    lambda_c = c/f_c
    
    for s in sizes:
        loss_factor = 1.0
        if s > lambda_c / 2:
            loss_factor = (lambda_c / (2*s)) 
        powers.append(loss_factor)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes*1000, powers, 'o-', linewidth=2)
    plt.axvline(x=(lambda_c/2)*1000, color='r', linestyle='--', label='Lambda/2 Limit')
    plt.xlabel("Element Size (mm)")
    plt.ylabel("Efficiency")
    plt.title("Impact of Element Sizing")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/Exp4_Sizing.png")
    plt.show()

# --- EXPERIMENT 5: ALGORITHM COMPARISON (NEW) ---
def experiment_5_comparison():
    print("\n=== EXP 5: Algorithm Comparison (SIMP vs GA vs PSO) ===")
    # This runs the comparison script and generates the convergence plot
    algo_comp.run_comparison()

# --- ANIMATION GENERATOR ---
def create_topology_animation():
    print("\n=== GENERATING ANIMATION ===")
    ris_params = phys.get_ris_config("standard")
    p_bs = np.array([50, 25, 2])
    p_ue = np.array([5, 0.32, 0.16])
    
    hist_micro = simp_opt.run_simp(1.0, ris_params, p_bs, p_ue, return_history=True)
    hist_macro = simp_opt.run_simp(8.0, ris_params, p_bs, p_ue, return_history=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle("Topology Evolution", fontsize=14)
    
    im1 = ax1.imshow(hist_micro[0], cmap='gray_r', vmin=0, vmax=1)
    ax1.set_title("Micro (r=1.0)")
    im2 = ax2.imshow(hist_macro[0], cmap='gray_r', vmin=0, vmax=1)
    ax2.set_title("Macro (r=8.0)")
    
    def update(frame_idx):
        im1.set_data(hist_micro[frame_idx])
        im2.set_data(hist_macro[frame_idx])
        return im1, im2

    num_frames = min(len(hist_micro), len(hist_macro))
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    
    save_path = "results/topology_evolution.gif"
    try:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    except:
        print("Could not save GIF (missing pillow?). Displaying instead.")
    plt.show()

# In main_experiments.py

def experiment_6_obstacle_bypass():
    print("\n=== EXP 6: Obstacle Bypass (The 'Smart' Detour) ===")
    
    # Setup
    ris_params = phys.get_ris_config("standard")
    p_bs = np.array([50, 25, 2])
    p_ue = np.array([5, 0.32, 0.16])
    
    blocker = {
        'type': 'circle',   # <--- Added 'type'
        'c_h': 100,         # Renamed from 'center_h'
        'c_v': 32,          # Renamed from 'center_v'
        'r': 20             # Renamed from 'radius'
    }
    # --------------------------------------
    
    print("Running SIMP with a Blocker...")
    # Run SIMP with the obstacle
    res_blocked = simp_opt.run_simp(rmin=3.0, ris_params=ris_params, 
                                    p_bs=p_bs, p_ue=p_ue, obstacle=blocker)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: The 'Shadow'
    pot_map, _, shape = phys.get_sensitivity_map(p_bs, p_ue, ris_params, obstacle=blocker)
    ax1.imshow(pot_map.reshape(shape), cmap='jet')
    ax1.set_title("Input Physics: The 'Dead Zone'\n(Blue Circle = Obstacle)")
    ax1.set_xlabel("The physics map is zeroed out here")
    
    # Plot 2: The Optimized Result
    ax2.imshow(res_blocked, cmap='gray_r')
    ax2.set_title("Optimized Topology: The Detour")
    ax2.set_xlabel("Algorithm learns to form a ring/donut\nto bypass the blockage.")
    
    plt.tight_layout()
    plt.savefig("results/Exp6_Obstacle.png")
    plt.show()

def experiment_7_spacing_obstacle():
    print("\n=== EXP 7: Swarm Obstacle Avoidance (GA) ===")
    
    blocker = {
        'type': 'circle',
        'c_h': 300,     # Center Horizontal Index (at 1.5m)
        'c_v': 32,      # Center Vertical Index (Middle of board)
        'r': 40         # Radius (pixels) -> 40*0.005 = 0.2m radius physical
    }
    
    # Run GA
    res_flat, p_bs, p_ue, BH, BV, r_params = spacing_opt.run_spacing_optimization_ga(obstacle=blocker)
    yz_opt = res_flat.reshape(-1, 2)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # 1. Draw the Obstacle (Visual only)
    # Convert pixels to meters for plotting
    obs_y = blocker['c_h'] * r_params['Dh']
    obs_z = blocker['c_v'] * r_params['Dv']
    obs_r = blocker['r'] * r_params['Dh'] # Assuming Dh=Dv for radius scale
    
    circle = plt.Circle((obs_y, obs_z), obs_r, color='red', alpha=0.3, label='Blocked Zone')
    ax.add_patch(circle)
    
    # 2. Draw Elements
    ax.scatter(yz_opt[:,0], yz_opt[:,1], c='blue', s=50, label='Optimized Elements', zorder=10)
    
    # Styling
    ax.set_title("Swarm Intelligence: Elements Fleeing the Dead Zone")
    ax.set_xlim(0, BH)
    ax.set_ylim(0, BV)
    ax.set_xlabel("Y Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.legend()
    ax.grid(True, linestyle='--')
    
    # Zoom in on the hotspot for better view (Optional)
    # ax.set_xlim(0, 3.0) 
    
    plt.tight_layout()
    plt.savefig("results/Exp7_Spacing_Obstacle.png")
    plt.show()


def create_obstacle_animation():
    print("\n=== GENERATING CROSS-OBSTACLE ANIMATION ===")
    
    # 1. Setup
    ris_params = phys.get_ris_config("standard")
    p_bs = np.array([50, 25, 2])
    p_ue = np.array([5, 0.32, 0.16])
    
    obstacle_cross = {
        'type': 'cross',
        # Overall bounds
        'h_min': 80, 'h_max': 120,      # Horizontal span (Cols)
        'v_min': 12, 'v_max': 52,       # Vertical span (Rows)
        
        # Thickness of the bars
        'v_cross_min': 28, 'v_cross_max': 36, # The horizontal bar's vertical thickness
        'h_cross_min': 96, 'h_cross_max': 104 # The vertical bar's horizontal thickness
    }
    
    # 3. Get Input Map
    pot_map, _, shape = phys.get_sensitivity_map(p_bs, p_ue, ris_params, obstacle=obstacle_cross)
    
    # 4. Run Optimization
    print("Running SIMP with Cross Obstacle...")
    hist_blocked = simp_opt.run_simp(rmin=3.0, ris_params=ris_params, 
                                     p_bs=p_bs, p_ue=p_ue, 
                                     max_loop=70,  # Needs more time to navigate corners
                                     return_history=True, 
                                     obstacle=obstacle_cross)
    
    # 5. Animate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle("Topology Optimization: Navigating a Cross Obstacle", fontsize=14)
    
    # Plot Input Physics
    ax1.imshow(pot_map.reshape(shape), cmap='jet')
    ax1.set_title("Input Physics (Dead Zone)")
    
    # Plot Dynamic Topology
    im_topo = ax2.imshow(hist_blocked[0], cmap='gray_r', vmin=0, vmax=1)
    ax2.set_title("Optimized Topology Evolution")
    
    def update(frame_idx):
        im_topo.set_data(hist_blocked[frame_idx])
        ax2.set_xlabel(f"Iteration: {frame_idx}")
        return [im_topo]

    anim = animation.FuncAnimation(fig, update, frames=len(hist_blocked), interval=100)
    
    anim.save("results/cross_obstacle.gif", writer='pillow', fps=10)
    print("Animation saved to results/cross_obstacle.gif")
    plt.show()

# --- MAIN RUNNER ---
if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')

    experiment_1_shape()
    experiment_2_spacing()
    experiment_3_quantization()
    experiment_4_element_sizing()
    experiment_5_comparison()  
    experiment_6_obstacle_bypass() 
    create_topology_animation()
    create_obstacle_animation()
    experiment_7_spacing_obstacle()