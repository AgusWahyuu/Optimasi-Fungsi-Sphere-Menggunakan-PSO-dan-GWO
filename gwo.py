import numpy as np

class GWO:

    def __init__(self, n_wolves, n_dimensions, search_range):
        self.n_wolves = n_wolves
        self.n_dimensions = n_dimensions
        self.search_range = search_range
        
        self.positions = None
        self.alpha_pos = np.zeros(n_dimensions)
        self.alpha_score = np.inf
        self.beta_pos = np.zeros(n_dimensions)
        self.beta_score = np.inf
        self.delta_pos = np.zeros(n_dimensions)
        self.delta_score = np.inf

    def sphere_function(self, x):
        return np.sum(x**2)

    def initialize_population_manual(self, initial_positions):
        self.positions = np.array(initial_positions)

    def optimize(self, max_iter, fixed_r_values):
        history = []

        for t in range(max_iter):
            fitness = np.apply_along_axis(self.sphere_function, 1, self.positions)

            for i in range(self.n_wolves):
                if fitness[i] < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness[i]
                    self.alpha_pos = self.positions[i].copy()
                elif fitness[i] > self.alpha_score and fitness[i] < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness[i]
                    self.beta_pos = self.positions[i].copy()
                elif fitness[i] > self.beta_score and fitness[i] < self.delta_score:
                    self.delta_score = fitness[i]
                    self.delta_pos = self.positions[i].copy()
            
            iter_log = {
                "iteration": t + 1,
                "positions_before": self.positions.copy(),
                "fitness_before": fitness.copy(),
                "alpha_pos": self.alpha_pos.copy(),
                "beta_pos": self.beta_pos.copy(),
                "delta_pos": self.delta_pos.copy(),
                "best_fitness": self.alpha_score
            }

            a = 2 - t * (2 / max_iter)
            r1, r2 = fixed_r_values[t]
            new_positions = np.zeros_like(self.positions)
            for i in range(self.n_wolves):
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                X1 = self.alpha_pos - A1 * D_alpha

                A2 = 2 * a * r1 - a 
                C2 = 2 * r2
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                X2 = self.beta_pos - A2 * D_beta

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i])
                X3 = self.delta_pos - A3 * D_delta

                new_positions[i] = (X1 + X2 + X3) / 3
            
            self.positions = new_positions
            self.positions = np.clip(self.positions, self.search_range[0], self.search_range[1])
            
            iter_log["a"] = a
            iter_log["r1"] = r1
            iter_log["r2"] = r2
            iter_log["positions_after"] = self.positions.copy()
            history.append(iter_log)

        return self.alpha_pos, self.alpha_score, history
