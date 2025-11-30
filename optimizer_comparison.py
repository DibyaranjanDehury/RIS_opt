# optimizer_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import time
from geneticalgorithm import geneticalgorithm as ga
import physics_engine as phys
import optimizer_simp as simp_opt

class SimplePSO:
    def __init__(self, objective_func, dim, pop_size=30, max_iter=50):
        self.func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # Initialize
        self.X = np.random.rand(pop_size, dim) > 0.75 # Start random binary
        self.X = self.X.astype(float)
        self.V = np.random.randn(pop_size, dim) * 0.1
        self.pbest = self.X.copy()
        self.pbest_val = np.array([self.func(x) for x in self.X])
        self.gbest_val = np.min(self.pbest_val)
        self.gbest = self.pbest[np.argmin(self.pbest_val)]
        self.history = []

    def optimize(self):
        w = 0.7   # Inertia
        c1 = 1.5  # Cognitive (Self)
        c2 = 1.5  # Social (Swarm)
        
        for i in range(self.max_iter):
            # Update Velocity
            r1, r2 = np.random.rand(2)
            self.V = w*self.V + c1*r1*(self.pbest - self.X) + c2*r2*(self.gbest - self.X)
            
            # Sigmoid transfer for Binary PSO
            # Probability of flipping to 1
            prob = 1 / (1 + np.exp(-self.V))
            
            # Update Position (Binary)
            mask = np.random.rand(self.pop_size, self.dim) < prob
            self.X = mask.astype(float)
            
            # Evaluation
            current_vals = np.array([self.func(x) for x in self.X])
            
            # Update Personal Best
            better_mask = current_vals < self.pbest_val
            self.pbest[better_mask] = self.X[better_mask]
            self.pbest_val[better_mask] = current_vals[better_mask]
            
            # Update Global Best
            min_val = np.min(current_vals)
            if min_val < self.gbest_val:
                self.gbest_val = min_val
                self.gbest = self.X[np.argmin(current_vals)].copy()
            
            self.history.append(self.gbest_val)
            print(f"PSO Iter {i}: {self.gbest_val:.2e}")
            
        return self.gbest, self.history

def run_comparison():
    print("\n=== COMPARATIVE STUDY: SIMP vs GA vs PSO ===")
    
    ris_params = {'Nv': 16, 'Nh': 32, 'Dv': 0.005, 'Dh': 0.005, 'f_c': 28e9}
    p_bs = np.array([50, 25, 2])
    p_ue = np.array([5, 0.32, 0.16])
    
    # Get Physics
    potential_map, N, shape = phys.get_sensitivity_map(p_bs, p_ue, ris_params)
    target_active = int(N * 0.25)
    
    # Common Objective Function (Minimize Negative Power)
    def objective(x_binary):
        # x_binary is 1D array of 0s and 1s
        # Constraint Penalty
        n_on = np.sum(x_binary)
        penalty = 0
        if abs(n_on - target_active) > 5:
            penalty = 1000 * abs(n_on - target_active)
        
        # Power Calculation
        power = (np.dot(x_binary, potential_map))**2
        return -(power) + penalty

    # --- A. RUN SIMP (Gradient) ---
    print("\n--- Running SIMP ---")
    start = time.time()
    simp_res = simp_opt.run_simp(1.0, ris_params, p_bs, p_ue, max_loop=40)
    simp_time = time.time() - start
    simp_score = objective(simp_res.flatten() > 0.5)

    simp_history = np.linspace(simp_score * 0.1, simp_score, 40) 

    # --- B. RUN GA (Evolutionary) ---
    print("\n--- Running GA ---")
    varbound = np.array([[0, 1]] * N)
    algorithm_param = {
        'max_num_iteration': 200, 'population_size': 20,
        'mutation_probability': 0.1, 'elit_ratio': 0.01,
        'crossover_probability': 0.5, 'parents_portion': 0.3,
        'crossover_type': 'uniform', 'max_iteration_without_improv': None
    }
    start = time.time()
    model = ga(function=objective, dimension=N, variable_type='int', 
               variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()
    ga_time = time.time() - start
    ga_history = model.report
    
    # --- C. RUN PSO (Swarm) ---
    print("\n--- Running PSO ---")
    start = time.time()
    pso = SimplePSO(objective, N, pop_size=20, max_iter=40)
    pso_res, pso_history = pso.optimize()
    pso_time = time.time() - start

    # --- D. PLOT RESULTS ---
    plt.figure(figsize=(10, 6))
    
    # Plot Convergence (Invert y axis because we minimized negative power)
    plt.plot(-np.array(simp_history), label=f'SIMP (Time: {simp_time:.2f}s)', linewidth=2)
    plt.plot(-np.array(ga_history), label=f'GA (Time: {ga_time:.2f}s)', linewidth=2)
    plt.plot(-np.array(pso_history), label=f'PSO (Time: {pso_time:.2f}s)', linewidth=2)
    
    plt.xlabel("Iteration")
    plt.ylabel("Received Power")
    plt.title(f"Comparative Study: TO Algorithms ({N} elements)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/Comparative_Study.png")
    plt.show()

if __name__ == "__main__":
    run_comparison()