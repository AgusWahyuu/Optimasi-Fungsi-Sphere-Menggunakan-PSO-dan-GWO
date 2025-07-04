import numpy as np

class PSO:
    """
    Implementasi Particle Swarm Optimization (PSO).
    """
    def __init__(self, n_particles, n_dimensions, search_range, w, c1, c2):
        """
        Inisialisasi algoritma PSO.

        Args:
            n_particles (int): Jumlah partikel.
            n_dimensions (int): Jumlah dimensi (variabel).
            search_range (tuple): Rentang pencarian (min, max).
            w (float): Bobot inersia.
            c1 (float): Koefisien kognitif.
            c2 (float): Koefisien sosial.
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.search_range = search_range
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_values = np.full(n_particles, np.inf)
        self.gbest_position = None
        self.gbest_value = np.inf

    def sphere_function(self, x):
        """Fungsi Sphere: f(x) = sum(x_i^2)."""
        return np.sum(x**2)

    def initialize_population_manual(self, initial_positions, initial_velocities):
        """
        Inisialisasi populasi dengan nilai yang telah ditentukan.
        """
        self.positions = np.array(initial_positions)
        self.velocities = np.array(initial_velocities)
        self.pbest_positions = np.copy(self.positions)
        
        for i in range(self.n_particles):
            fitness = self.sphere_function(self.positions[i])
            self.pbest_values[i] = fitness
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = self.positions[i]

    def optimize(self, max_iter, fixed_r_values):
        """
        Menjalankan proses optimasi PSO.

        Args:
            max_iter (int): Jumlah maksimum iterasi.
            fixed_r_values (list of tuples): Nilai r1 dan r2 yang telah ditentukan untuk setiap iterasi.

        Returns:
            tuple: Posisi terbaik global, nilai terbaik global, dan riwayat proses.
        """
        history = []

        for t in range(max_iter):
            iter_log = {
                "iteration": t + 1,
                "positions_before": np.copy(self.positions),
                "velocities_before": np.copy(self.velocities),
                "pbest_positions_before": np.copy(self.pbest_positions),
                "gbest_position_before": np.copy(self.gbest_position),
                "gbest_value_before": self.gbest_value
            }

            r1, r2 = fixed_r_values[t]

            for i in range(self.n_particles):
                cognitive_component = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.search_range[0], self.search_range[1])

                current_fitness = self.sphere_function(self.positions[i])
                if current_fitness < self.pbest_values[i]:
                    self.pbest_values[i] = current_fitness
                    self.pbest_positions[i] = self.positions[i]

                if current_fitness < self.gbest_value:
                    self.gbest_value = current_fitness
                    self.gbest_position = self.positions[i]

            iter_log["r1"] = r1
            iter_log["r2"] = r2
            iter_log["velocities_after"] = np.copy(self.velocities)
            iter_log["positions_after"] = np.copy(self.positions)
            iter_log["pbest_values_after"] = np.copy(self.pbest_values)
            iter_log["gbest_value_after"] = self.gbest_value
            iter_log["gbest_position_after"] = np.copy(self.gbest_position)
            history.append(iter_log)

        return self.gbest_position, self.gbest_value, history
