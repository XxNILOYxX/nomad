import random
import numpy as np
import logging
import threading
from typing import List, Dict, Optional, Tuple
from .utils import fitness_function


class MultiSwarmPSO:
    """
    Manages multiple sub-swarms for enhanced global optimization.

    This class orchestrates several PSO sub-swarms, which can operate with
    different parameters (e.g., exploration vs. exploitation focus). It handles
    the initialization of these swarms and the periodic migration of the best
    particles between them to share information and prevent premature
    convergence on local optima.
    """

    def __init__(self, pso_instance: 'ParticleSwarmOptimizer'):
        """
        Initializes the multi-swarm handler.

        Args:
            pso_instance (ParticleSwarmOptimizer): The main PSO instance that
                this handler will manage.
        """
        self.main_pso = pso_instance
        self.config = pso_instance.pso_config
        self.sub_swarms = []
        self.sub_swarm_bests = []

        if self.config.get('enable_multi_swarm', False):
            self.num_sub_swarms = self.config.get('num_sub_swarms', 3)
            self.migration_frequency = self.config.get('migration_frequency', 100)
            self.migration_rate = self.config.get('migration_rate', 0.1)
            self._initialize_sub_swarms()

    def get_state(self) -> Dict:
        """
        Serializes the internal state of the multi-swarm handler.

        Captures the complete state of all sub-swarms, including positions,
        velocities, and personal bests, converting numpy arrays to lists
        for JSON compatibility.

        Returns:
            Dict: A dictionary containing the state of all sub-swarms.
        """
        sub_swarms_state = []
        for swarm in self.sub_swarms:
            state = swarm.copy()
            state['velocities'] = [v.tolist() for v in swarm['velocities']]
            state['best_position'] = swarm['best_position'][:] if swarm.get('best_position') else None
            state['best_fitness'] = swarm.get('best_fitness', -float('inf'))
            state['personal_bests'] = [p[:] for p in swarm['personal_bests']]
            sub_swarms_state.append(state)
        return {"sub_swarms": sub_swarms_state}

    def load_state(self, state: Dict):
        """
        Loads the multi-swarm handler's state from a dictionary.

        Reconstructs the sub-swarms from a serialized state, converting
        velocity lists back into numpy arrays. After loading, it synchronizes
        the main PSO instance with the loaded sub-swarm data.

        Args:
            state (Dict): The serialized state dictionary to load.
        """
        sub_swarms_state = state.get("sub_swarms", [])
        self.sub_swarms = []
        for s_state in sub_swarms_state:
            loaded_swarm = s_state.copy()
            loaded_swarm['velocities'] = [np.array(v) for v in s_state['velocities']]
            self.sub_swarms.append(loaded_swarm)

        if self.sub_swarms:
            logging.info(f"Multi-swarm state loaded with {len(self.sub_swarms)} sub-swarms.")


    def _initialize_sub_swarms(self):
        """
        Initializes multiple sub-swarms with potentially different characteristics.

        Divides the total particle count among the sub-swarms and assigns
        different cognitive and social coefficients to promote a balance
        between exploration and exploitation across the entire population.
        """
        total_particles = self.config['swarm_size']
        particles_per_swarm = total_particles // self.num_sub_swarms
        remainder = total_particles % self.num_sub_swarms

        self.sub_swarms = []
        particle_start_idx = 0
        for i in range(self.num_sub_swarms):
            sub_swarm_config = self.config.copy()

            if i == 0:  # Exploration-focused swarm
                sub_swarm_config['cognitive_coeff'] = 2.5
                sub_swarm_config['social_coeff'] = 1.0
            elif i == 1:  # Exploitation-focused swarm
                sub_swarm_config['cognitive_coeff'] = 1.0
                sub_swarm_config['social_coeff'] = 2.5
            else:  # Balanced swarm
                sub_swarm_config['cognitive_coeff'] = 1.5
                sub_swarm_config['social_coeff'] = 1.5

            num_particles_in_swarm = particles_per_swarm + (1 if i < remainder else 0)

            sub_swarm = {
                'config': sub_swarm_config, 'positions': [], 'velocities': [],
                'personal_bests': [], 'personal_best_fitnesses': [],
                'best_position': None, 'best_fitness': -float('inf'),
                'particle_indices': list(range(particle_start_idx, particle_start_idx + num_particles_in_swarm))
            }

            for _ in range(num_particles_in_swarm):
                position = self.main_pso._create_random_particle()
                max_v = sub_swarm_config.get('max_change_probability', 0.5)
                velocity = np.random.uniform(0, max_v, len(position))
                sub_swarm['positions'].append(position)
                sub_swarm['velocities'].append(velocity)
                sub_swarm['personal_bests'].append(position[:])
                sub_swarm['personal_best_fitnesses'].append(-float('inf'))

            self.sub_swarms.append(sub_swarm)
            particle_start_idx += num_particles_in_swarm

    def _migrate_particles(self):
        """
        Migrates the best particles from each swarm to another.

        This process facilitates information sharing across sub-swarms. The best
        particles from a source swarm replace the worst particles in a target
        swarm in a round-robin fashion.
        """
        if not self.sub_swarms or len(self.sub_swarms) < 2:
            return

        logging.info("Performing multi-swarm migration.")
        for i, source_swarm in enumerate(self.sub_swarms):
            target_swarm_idx = (i + 1) % len(self.sub_swarms)
            target_swarm = self.sub_swarms[target_swarm_idx]

            num_migrate = min(
                max(1, int(len(source_swarm['positions']) * self.migration_rate)),
                len(target_swarm['positions']) // 2
            )

            if num_migrate == 0: continue

            source_fitnesses = source_swarm['personal_best_fitnesses']
            best_source_indices = np.argsort(source_fitnesses)[-num_migrate:]

            target_fitnesses = target_swarm['personal_best_fitnesses']
            worst_target_indices = np.argsort(target_fitnesses)[:num_migrate]

            for source_idx, target_idx in zip(best_source_indices, worst_target_indices):
                target_swarm['positions'][target_idx] = source_swarm['personal_bests'][source_idx][:]
                target_swarm['velocities'][target_idx] = source_swarm['velocities'][source_idx].copy()
                target_swarm['personal_bests'][target_idx] = source_swarm['personal_bests'][source_idx][:]
                target_swarm['personal_best_fitnesses'][target_idx] = source_swarm['personal_best_fitnesses'][source_idx]

    def run_multi_swarm_iteration(self, inertia_weight: float):
        """
        Runs one full iteration of the multi-swarm PSO algorithm.

        This involves updating each sub-swarm according to its own parameters,
        performing migration if the iteration criteria are met, and finally
        synchronizing the results back to the main PSO instance.

        Args:
            inertia_weight (float): The current inertia weight for the PSO update.
        """
        if not self.sub_swarms:
            self.main_pso._update_swarm_discrete(inertia_weight)
            return

        for swarm in self.sub_swarms:
            self._update_sub_swarm(swarm, inertia_weight)

        if self.main_pso._current_iteration > 0 and self.main_pso._current_iteration % self.migration_frequency == 0:
            self._migrate_particles()
            self.main_pso.adaptive_event_log.append(f"Iter {self.main_pso._current_iteration}: Multi-swarm migration performed")

        self._synchronize_main_swarm()

    def _update_sub_swarm(self, swarm: Dict, inertia_weight: float):
        """
        Updates the velocities and positions of all particles in a single sub-swarm.

        Uses the specific configuration (e.g., c1, c2) of the given sub-swarm
        for the update calculations. Each particle determines its neighborhood
        best from the main PSO's global list of bests.

        Args:
            swarm (Dict): The sub-swarm to update.
            inertia_weight (float): The current inertia weight.
        """
        c1 = swarm['config']['cognitive_coeff']
        c2 = swarm['config']['social_coeff']
        max_v = swarm['config']['max_change_probability']

        for i, global_idx in enumerate(swarm['particle_indices']):
            nbest_position = self.main_pso._get_neighborhood_best(global_idx)
            new_pos, new_vel = self.main_pso._update_single_particle(
                global_idx, inertia_weight, c1, c2, max_v, nbest_position
            )
            swarm['positions'][i] = new_pos
            swarm['velocities'][i] = new_vel

    def _synchronize_main_swarm(self):
        """
        Synchronizes the main swarm state from all sub-swarms.

        Aggregates the positions, velocities, and personal bests from all
        sub-swarms into the main PSO handler's attributes. This is necessary
        for global evaluation, state saving, and neighborhood calculations.
        """
        if not self.sub_swarms: return

        all_positions, all_velocities, all_p_bests, all_p_fitnesses = [], [], [], []

        for swarm in self.sub_swarms:
            all_positions.extend(swarm['positions'])
            all_velocities.extend(swarm['velocities'])
            all_p_bests.extend(swarm['personal_bests'])
            all_p_fitnesses.extend(swarm['personal_best_fitnesses'])

        self.main_pso.swarm_positions = all_positions
        self.main_pso.swarm_velocities = all_velocities
        self.main_pso.personal_best_positions = all_p_bests
        self.main_pso.personal_best_fitnesses = all_p_fitnesses

    def update_sub_swarm_bests(self):
        """
        Updates personal and swarm-level bests for each sub-swarm.

        After the main swarm is evaluated globally, this method propagates the
        updated fitness values back to the individual sub-swarms, allowing them
        to update their internal personal and swarm-best records.
        """
        if not self.sub_swarms: return

        for swarm in self.sub_swarms:
            for i, global_idx in enumerate(swarm['particle_indices']):
                if self.main_pso.personal_best_fitnesses[global_idx] > swarm['personal_best_fitnesses'][i]:
                    swarm['personal_best_fitnesses'][i] = self.main_pso.personal_best_fitnesses[global_idx]
                    swarm['personal_bests'][i] = self.main_pso.personal_best_positions[global_idx][:]

            if swarm['personal_best_fitnesses']:
                best_local_idx = np.argmax(swarm['personal_best_fitnesses'])
                current_best_fitness = swarm['personal_best_fitnesses'][best_local_idx]
                if current_best_fitness > swarm['best_fitness']:
                    swarm['best_fitness'] = current_best_fitness
                    swarm['best_position'] = swarm['personal_bests'][best_local_idx][:]


class ParticleSwarmOptimizer:
    """
    Implements a robust Discrete Particle Swarm Optimization (DPSO) engine.

    This class provides a comprehensive DPSO implementation with many advanced
    features, including:
    - Multiple neighborhood topologies (global, ring, random, fitness-based).
    - Adaptive velocity bounds and inertia weight.
    - Diversity maintenance to prevent premature convergence.
    - Periodic local search to refine the global best solution.
    - Multi-swarm management via the `MultiSwarmPSO` class.
    - State serialization for checkpointing and resuming optimizations.
    """
    def __init__(self, config: Dict, keff_interpolator, ppf_interpolator):
        """
        Initializes the Particle Swarm Optimizer.

        Args:
            config (Dict): A dictionary containing configuration for the PSO,
                simulation, enrichment, and fitness function.
            keff_interpolator: An initialized predictor object for k-effective.
                Must have `predict` and `predict_batch` methods.
            ppf_interpolator: An initialized predictor object for power peaking factor.
                Must have `predict` and `predict_batch` methods.
        """
        self.pso_config = config['pso']
        self.sim_config = config['simulation']
        self.enrich_config = config['enrichment']
        self.tuning_config = config['fitness']

        self._validate_pso_config()

        self.keff_interpolator = keff_interpolator
        self.ppf_interpolator = ppf_interpolator

        self.swarm_positions, self.swarm_velocities = [], []
        self.personal_best_positions, self.personal_best_fitnesses = [], []
        self.global_best_position, self.global_best_fitness = None, -float('inf')

        self.central_vals = sorted(self.enrich_config['central_values'])
        self.outer_vals = sorted(self.enrich_config['outer_values'])

        self._current_iteration = 0  
        self.stagnation_counter = 0
        self.neighborhoods = {}
        self.adaptive_event_log = []
        self.performance_metrics = {'diversity_history': [], 'velocity_stats': []}

        self._rng = random.Random()  # Per-instance random generator
        self._np_rng = np.random.RandomState()  # Per-instance numpy RNG

        self._diversity_boost_counter = 0
        self.multi_swarm_handler = None
        if self.pso_config.get('enable_multi_swarm', False):
            self.multi_swarm_handler = MultiSwarmPSO(self)

    def get_state(self) -> Dict:
        """Returns serializable state with proper type conversion."""
        state = {
            "swarm_positions": self.swarm_positions,
            "swarm_velocities": [v.tolist() if isinstance(v, np.ndarray) else v for v in self.swarm_velocities],
            "personal_best_positions": self.personal_best_positions,
            "personal_best_fitnesses": self.personal_best_fitnesses,
            "global_best_position": self.global_best_position,
            "global_best_fitness": self.global_best_fitness,
            "stagnation_counter": self.stagnation_counter,
            "adaptive_event_log": self.adaptive_event_log[-1000:],  # Limit size
            "performance_metrics": {
                k: v[-1000:] if isinstance(v, list) else v
                for k, v in self.performance_metrics.items()
            },
            "_diversity_boost_counter": self._diversity_boost_counter,
            "multi_swarm_handler_state": self.multi_swarm_handler.get_state() if self.multi_swarm_handler else None,
        }
        return state

    def load_state(self, state: Dict):
        """Enhanced state loading with validation."""
        loaded_positions = state.get("swarm_positions", [])
        expected_size = self.pso_config['swarm_size']

        if expected_size != len(loaded_positions):
            logging.error(f"Swarm size mismatch! Expected {expected_size}, got {len(loaded_positions)}")
            self._full_initialization()
            return

        try:
            # Validate and load main swarm state
            self.swarm_positions = loaded_positions
            loaded_velocities = state.get("swarm_velocities", [])

            if len(loaded_velocities) != len(loaded_positions):
                logging.warning("Velocity array size mismatch, reinitializing velocities")
                initial_max_v = self.pso_config.get('max_change_probability', 0.5)
                self.swarm_velocities = [
                    np.array(self._np_rng.uniform(0, initial_max_v, len(pos)), dtype=np.float32)
                    for pos in self.swarm_positions
                ]
            else:
                self.swarm_velocities = [
                    np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray)
                    else v.astype(np.float32) for v in loaded_velocities
                ]

            # Load other state safely
            self.personal_best_positions = state.get("personal_best_positions", [])
            self.personal_best_fitnesses = state.get("personal_best_fitnesses", [])
            self.global_best_position = state.get("global_best_position")
            self.global_best_fitness = state.get("global_best_fitness", -float('inf'))
            self.stagnation_counter = state.get("stagnation_counter", 0)

            # Validate loaded data consistency
            if not self._validate_loaded_state():
                logging.error("Loaded state validation failed, reinitializing")
                self._full_initialization()
                return

            # Load multi-swarm state last
            multi_swarm_state = state.get("multi_swarm_handler_state")
            if self.multi_swarm_handler and multi_swarm_state:
                self.multi_swarm_handler.load_state(multi_swarm_state)

            logging.info(f"PSO state loaded successfully. Global best: {self.global_best_fitness:.6f}")

        except Exception as e:
            logging.error(f"State loading failed: {e}. Reinitializing swarm.")
            self._full_initialization()

    def _validate_loaded_state(self) -> bool:
        """Validates consistency of loaded state."""
        expected_size = self.pso_config['swarm_size']

        checks = [
            len(self.swarm_positions) == expected_size,
            len(self.swarm_velocities) == expected_size,
            len(self.personal_best_positions) == expected_size,
            len(self.personal_best_fitnesses) == expected_size,
            all(isinstance(v, np.ndarray) for v in self.swarm_velocities),
            all(len(pos) == len(self.swarm_positions[0]) for pos in self.swarm_positions)
        ]

        return all(checks)

    def initialize_with_population(self, population: List[List[float]]):
        """
        Seeds the swarm with a given population (e.g., from a genetic algorithm).

        Replaces the current swarm with individuals from the provided population.
        If the population is smaller than the configured swarm size, the remaining
        spots are filled with new random particles. All personal and global bests
        are reset.

        Args:
            population (List[List[float]]): A list of individuals to seed the swarm with.
        """
        logging.info(f"Seeding PSO swarm with {len(population)} individuals from GA.")

        num_to_seed = min(len(population), self.pso_config['swarm_size'])
        new_population = [p[:] for p in population[:num_to_seed]]

        while len(new_population) < self.pso_config['swarm_size']:
            new_population.append(self._create_random_particle())

        self.swarm_positions = new_population
        self.personal_best_positions = [p[:] for p in self.swarm_positions]
        self.personal_best_fitnesses = [-float('inf')] * len(self.swarm_positions)
        self.global_best_position, self.global_best_fitness = None, -float('inf')

        initial_max_v = self.pso_config.get('max_change_probability', 0.5)
        self.swarm_velocities = [
            np.array(self._np_rng.uniform(0, initial_max_v, len(pos)), dtype=np.float32) for pos in self.swarm_positions
        ]
        
        if self.multi_swarm_handler:
            logging.info("Distributing seeded population among sub-swarms.")
            self.multi_swarm_handler._initialize_sub_swarms() 
            
            particle_offset = 0
            for swarm in self.multi_swarm_handler.sub_swarms:
                num_particles = len(swarm['particle_indices'])
                end_offset = particle_offset + num_particles
                
                swarm['positions'] = self.swarm_positions[particle_offset:end_offset]
                swarm['velocities'] = self.swarm_velocities[particle_offset:end_offset]
                swarm['personal_bests'] = [p[:] for p in swarm['positions']]
                swarm['personal_best_fitnesses'] = [-float('inf')] * num_particles
                
                particle_offset = end_offset

            self.multi_swarm_handler._synchronize_main_swarm()

        self._evaluate_and_update_bests()
        logging.info(f"PSO swarm successfully seeded. New global best fitness: {self.global_best_fitness:.6f}")

    def run_pso_algorithm(self, seed_individual: Optional[List[float]] = None) -> List[float]:
        """
        Runs the main PSO algorithm for a configured number of iterations.

        This is the primary entry point for running an optimization cycle. It
        manages the main loop, applies all enhancements (like local search and
        diversity control), and handles termination conditions.

        Args:
            seed_individual (Optional[List[float]], optional): An external solution
                to inject into the swarm at the start. Defaults to None.

        Returns:
            List[float]: The best solution (particle position) found during the run.
        """
        if not self.swarm_positions:
            logging.info("PSO swarm is not initialized. Creating a new swarm.")
            self._full_initialization()

        if seed_individual:
            self._seed_global_best(seed_individual, replace_worst=True)

        max_iterations = self.pso_config['iterations_per_openmc_cycle']
        convergence_threshold = self.pso_config.get('pso_convergence_threshold', 200)
        self.stagnation_counter = 0

        enable_ls = self.pso_config.get('enable_local_search', False)
        local_search_freq = self.pso_config.get('local_search_frequency', 25)

        for iteration in range(max_iterations):
            self._current_iteration = iteration
            previous_gbest_fitness = self.global_best_fitness

            w_start, w_end = self.pso_config.get('inertia_weight_start', 0.9), self.pso_config.get('inertia_weight_end', 0.4)
            current_inertia = w_start - (w_start - w_end) * (iteration / max_iterations)

            if (iteration + 1) % self.pso_config.get('neighborhood_rebuild_frequency', 100) == 0:
                self._update_dynamic_neighborhoods()

            if self.multi_swarm_handler:
                self.multi_swarm_handler.run_multi_swarm_iteration(current_inertia)
            else:
                self._update_swarm_discrete(current_inertia)

            self._evaluate_and_update_bests()

            if self.global_best_fitness > previous_gbest_fitness:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            if enable_ls and (iteration + 1) % local_search_freq == 0:
                self._apply_periodic_local_search()

            if self.stagnation_counter > convergence_threshold:
                logging.info(f"PSO convergence after {self.stagnation_counter} stagnant iterations... Exiting.")
                break

            if iteration > 0 and iteration % 1000 == 0:
                self._limit_history_size(max_entries=1000)

            self._track_and_log_performance(iteration)

        self._limit_history_size()
        return self.global_best_position

    def _full_initialization(self):
        """
        Initializes or re-initializes the entire swarm state from scratch.
        
        This includes creating particles, setting initial velocities, defining
        neighborhoods, and performing the first fitness evaluation.
        """
        if self.multi_swarm_handler:
            self.multi_swarm_handler._initialize_sub_swarms()
            self.multi_swarm_handler._synchronize_main_swarm()
        else:
            self._initialize_swarm()

        self._initialize_neighborhoods()
        self._evaluate_and_update_bests()

    def _evaluate_and_update_bests(self):
        """
        Evaluates the entire swarm's fitness and updates personal and global bests.
        """
        if not self.swarm_positions: return

        fitnesses = self._evaluate_swarm(self.swarm_positions)
        if not self.personal_best_fitnesses:
            self.personal_best_fitnesses = [-float('inf')] * len(self.swarm_positions)
            self.personal_best_positions = [pos[:] for pos in self.swarm_positions]

        self._update_personal_bests(fitnesses)
        self._update_global_best(fitnesses, self.swarm_positions)
        if self.multi_swarm_handler:
            self.multi_swarm_handler.update_sub_swarm_bests()

    def _seed_global_best(self, seed_individual: List[float], replace_worst: bool = False):
        """
        Updates the global best with an external seed solution.

        Optionally, it can also replace the worst-performing particle in the
        swarm with this new seed individual.

        Args:
            seed_individual (List[float]): The external solution to inject.
            replace_worst (bool, optional): If True, replaces the worst particle.
                Defaults to False.
        """
        seed_keff, seed_ppf = self.keff_interpolator.predict(seed_individual), self.ppf_interpolator.predict(seed_individual)
        seed_fitness = fitness_function(seed_keff, seed_ppf, self.sim_config, self.tuning_config)

        if seed_fitness > self.global_best_fitness:
            logging.info(f"PSO global best updated by external seed. Fitness: {self.global_best_fitness:.6f} -> {seed_fitness:.6f}")
            self.global_best_fitness = seed_fitness
            self.global_best_position = seed_individual[:]

            if replace_worst and self.personal_best_fitnesses:
                worst_idx = np.argmin(self.personal_best_fitnesses)
                max_v = self.pso_config.get('max_change_probability', 0.5)

                self.swarm_positions[worst_idx] = seed_individual[:]
                self.swarm_velocities[worst_idx] = self._np_rng.uniform(0, max_v, len(seed_individual))
                self.personal_best_positions[worst_idx] = seed_individual[:]
                self.personal_best_fitnesses[worst_idx] = seed_fitness

                if self.multi_swarm_handler:
                    for swarm in self.multi_swarm_handler.sub_swarms:
                        if worst_idx in swarm['particle_indices']:
                            local_idx = swarm['particle_indices'].index(worst_idx)
                            swarm['positions'][local_idx] = seed_individual[:]
                            swarm['velocities'][local_idx] = self.swarm_velocities[worst_idx]
                            swarm['personal_bests'][local_idx] = seed_individual[:]
                            swarm['personal_best_fitnesses'][local_idx] = seed_fitness
                            break

    def _apply_periodic_local_search(self):
        """
        Applies a local search heuristic to the current global best solution.
        
        If the local search finds a better solution, the global best is updated.
        """
        if not self.global_best_position: return

        original_pos = self.global_best_position[:]
        improved_pos, new_fitness = self._implement_local_search(original_pos, self.global_best_fitness)

        if new_fitness > self.global_best_fitness:
            self.global_best_position = improved_pos
            self.global_best_fitness = new_fitness
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Periodic local search found improvement.")

    def _calculate_swarm_diversity(self) -> float:
        """
        Calculates the swarm's diversity using average Hamming distance.

        A sample of particles is taken, and the average pairwise Hamming
        distance is computed to estimate the diversity of the entire swarm.

        Returns:
            float: The calculated diversity metric.
        """
        if len(self.swarm_positions) < 2: return 0.0

        sample_size = min(len(self.swarm_positions), 100)
        sample_indices = self._rng.sample(range(len(self.swarm_positions)), sample_size)
        sample_positions = [self.swarm_positions[i] for i in sample_indices]
        
        total_distance, comparisons = 0.0, 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                pos1 = sample_positions[i]
                pos2 = sample_positions[j]
                distance = sum(1 for a, b in zip(pos1, pos2) if a != b)
                total_distance += distance
                comparisons += 1
        return total_distance / comparisons if comparisons > 0 else 0.0

    def _apply_diversity_maintenance(self):
        """
        Applies strategies if swarm diversity drops below a threshold.

        If diversity is too low, a portion of the worst-performing particles
        are re-initialized to random positions to inject new genetic material
        into the population. A temporary velocity boost is also activated.
        """
        current_diversity = self.performance_metrics['diversity_history'][-1]
        min_diversity = self.pso_config.get('min_diversity_threshold', 0.1)

        if current_diversity < min_diversity:
            num_to_reinit = max(1, self.pso_config['swarm_size'] // 10)
            worst_indices = np.argsort(self.personal_best_fitnesses)[:num_to_reinit]

            max_v = self.pso_config.get('max_change_probability', 0.5)
            for idx in worst_indices:
                new_particle = self._create_random_particle()
                self.swarm_positions[idx] = new_particle
                self.swarm_velocities[idx] = self._np_rng.uniform(0, max_v, len(new_particle))
                self.personal_best_positions[idx] = new_particle[:]
                self.personal_best_fitnesses[idx] = -float('inf')

            if self.multi_swarm_handler:
                for idx in worst_indices:
                    for swarm in self.multi_swarm_handler.sub_swarms:
                        if idx in swarm['particle_indices']:
                            local_idx = swarm['particle_indices'].index(idx)
                            swarm['positions'][local_idx] = self.swarm_positions[idx]
                            swarm['velocities'][local_idx] = self.swarm_velocities[idx]
                            swarm['personal_bests'][local_idx] = self.personal_best_positions[idx][:]
                            swarm['personal_best_fitnesses'][local_idx] = self.personal_best_fitnesses[idx]
                            break

            self._diversity_boost_counter = 20
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Low diversity ({current_diversity:.3f}) - Reinitialized {num_to_reinit} particles")

    def _get_enhanced_adaptive_velocity_bounds(self) -> float:
        """
        Calculates adaptive velocity, including a temporary boost for low diversity.

        Returns:
            float: The calculated maximum velocity (change probability).
        """
        base_velocity = self._get_adaptive_velocity_bounds()
        if self._diversity_boost_counter > 0:
            self._diversity_boost_counter -= 1
            return min(base_velocity + 0.3, 1.0)
        return base_velocity

    def _validate_and_fix_particle(self, position: List[float]) -> List[float]:
        """Enhanced particle validation with error handling."""
        if not position:
            logging.error("Empty position provided to validation")
            return self._create_random_particle()

        try:
            fixed_position = []
            num_central = self.sim_config['num_central_assemblies']

            for i, value in enumerate(position):
                is_central = i < num_central
                valid_values = self.central_vals if is_central else self.outer_vals

                if not valid_values:
                    logging.error(f"No valid values for position {i}")
                    fixed_position.append(self.central_vals[0] if is_central else self.outer_vals[0])
                    continue

                # If value is not in valid set, find closest valid value
                if value not in valid_values:
                    closest_value = min(valid_values, key=lambda x: abs(x - value))
                    fixed_position.append(closest_value)
                else:
                    fixed_position.append(value)

            return fixed_position

        except Exception as e:
            logging.error(f"Particle validation failed: {e}")
            return self._create_random_particle()

    def _implement_local_search(self, position: List[float], initial_fitness: float) -> Tuple[List[float], float]:
        """Applies hill-climbing local search with safe indexing."""
        best_pos_so_far = position[:]
        best_fitness_so_far = initial_fitness

        for i in range(len(position)):
            original_value = best_pos_so_far[i]
            is_central = i < self.sim_config['num_central_assemblies']
            valid_values = self.central_vals if is_central else self.outer_vals

            try:
                current_idx = valid_values.index(original_value)
            except ValueError:
                # Find closest value if not present
                distances = [abs(original_value - v) for v in valid_values]
                current_idx = distances.index(min(distances))

            neighbor_indices = []
            if current_idx > 0:
                neighbor_indices.append(current_idx - 1)
            if current_idx < len(valid_values) - 1:
                neighbor_indices.append(current_idx + 1)
            
            for idx in neighbor_indices:
                if 0 <= idx < len(valid_values):
                    new_value = valid_values[idx]
                    test_position = best_pos_so_far[:]
                    test_position[i] = new_value
                    
                    k_test = self.keff_interpolator.predict(test_position)
                    p_test = self.ppf_interpolator.predict(test_position)
                    test_fitness = fitness_function(k_test, p_test, self.sim_config, self.tuning_config)
                    
                    if test_fitness > best_fitness_so_far:
                        best_fitness_so_far = test_fitness
                        best_pos_so_far = test_position[:]
        
        return best_pos_so_far, best_fitness_so_far

    def _update_swarm_discrete(self, inertia_weight: float):
        """
        Updates particle velocities and positions for a single-swarm configuration.

        Args:
            inertia_weight (float): The current inertia weight for the update.
        """
        if self._current_iteration > 0 and self._current_iteration % 50 == 0:
            self._apply_diversity_maintenance()

        max_v = self._get_enhanced_adaptive_velocity_bounds()
        c1, c2 = self.pso_config['cognitive_coeff'], self.pso_config.get('social_coeff', 1.5)

        for i in range(self.pso_config['swarm_size']):
            nbest_position = self._get_neighborhood_best(i)
            new_pos, new_vel = self._update_single_particle(i, inertia_weight, c1, c2, max_v, nbest_position)
            self.swarm_positions[i] = new_pos
            self.swarm_velocities[i] = new_vel

    def _update_single_particle(
        self, particle_idx: int, inertia_weight: float, c1: float, c2: float,
        max_v: float, nbest_position: List[float]
    ) -> Tuple[List[float], np.ndarray]:
        """Enhanced particle update with error handling."""

        if not (0 <= particle_idx < len(self.swarm_positions)):
            logging.error(f"Invalid particle index: {particle_idx}")
            return self._create_random_particle(), np.zeros(len(self.swarm_positions[0]), dtype=np.float32)

        try:
            position = self.swarm_positions[particle_idx]
            velocity = self.swarm_velocities[particle_idx]
            pbest_position = self.personal_best_positions[particle_idx]

            if not isinstance(velocity, np.ndarray):
                velocity = np.array(velocity, dtype=np.float32)

            if len(position) != len(nbest_position) or len(position) != len(pbest_position):
                logging.warning(f"Dimension mismatch for particle {particle_idx}, using random position")
                return self._create_random_particle(), self._np_rng.uniform(0, max_v, len(position)).astype(np.float32)

            new_vel = np.zeros_like(velocity, dtype=np.float32)
            new_pos = position[:]

            r1, r2 = self._rng.random(), self._rng.random()

            # Calculate new velocity vector
            for j in range(len(position)):
                cognitive_attraction = 1.0 if pbest_position[j] != position[j] else 0.0
                social_attraction = 1.0 if nbest_position[j] != position[j] else 0.0

                inertia_comp = inertia_weight * velocity[j]
                cognitive_comp = c1 * r1 * cognitive_attraction
                social_comp = c2 * r2 * social_attraction

                new_vel_j = inertia_comp + cognitive_comp + social_comp
                new_vel[j] = np.clip(new_vel_j, 0, max_v)

            # Position update logic remains the same but with safer random access
            mutation_prob = self.pso_config.get('mutation_probability', 0.05)
            if self._diversity_boost_counter > 0:
                mutation_prob = min(mutation_prob * 3, 1.0)

            for j in range(len(position)):
                if self._rng.random() < mutation_prob:
                    is_central = j < self.sim_config['num_central_assemblies']
                    valid_values = self.central_vals if is_central else self.outer_vals
                    new_pos[j] = self._rng.choice(valid_values)
                    new_vel[j] = self._rng.uniform(0, max_v)
                    continue

                if self._rng.random() < new_vel[j]:
                    choices = [position[j]]
                    weights = [inertia_weight]

                    if pbest_position[j] != position[j]:
                        choices.append(pbest_position[j])
                        weights.append(c1 * r1)

                    if nbest_position[j] != position[j]:
                        choices.append(nbest_position[j])
                        weights.append(c2 * r2)

                    total_weight = sum(weights)
                    if total_weight > 0:
                        normalized_weights = [w / total_weight for w in weights]
                        new_pos[j] = self._rng.choices(choices, weights=normalized_weights, k=1)[0]

            # Validate final position
            new_pos = self._validate_and_fix_particle(new_pos)

            return new_pos, new_vel

        except Exception as e:
            logging.error(f"Error updating particle {particle_idx}: {e}")
            # Return safe fallback
            return self._create_random_particle(), np.zeros(len(self.swarm_positions[0]), dtype=np.float32)

    def _validate_pso_config(self):
        """Enhanced configuration validation."""
        required_params = ['swarm_size', 'iterations_per_openmc_cycle', 'cognitive_coeff']

        for param in required_params:
            if param not in self.pso_config:
                raise ValueError(f"Missing required PSO parameter: {param}")

        if self.pso_config['swarm_size'] <= 0:
            raise ValueError("Swarm size must be positive")

        if self.pso_config['iterations_per_openmc_cycle'] <= 0:
            raise ValueError("Iterations must be positive")

        if self.pso_config['cognitive_coeff'] < 0:
            raise ValueError("Cognitive coefficient must be non-negative")

        if not self.enrich_config.get('central_values') or not self.enrich_config.get('outer_values'):
            raise ValueError("Enrichment values cannot be empty")

        if not self.sim_config.get('num_assemblies') or not self.sim_config.get('num_central_assemblies'):
            raise ValueError("Assembly configuration incomplete")

        if self.sim_config['num_central_assemblies'] > self.sim_config['num_assemblies']:
            raise ValueError("Central assemblies cannot exceed total assemblies")

    def _track_and_log_performance(self, iteration: int):
        """
        Tracks performance metrics and logs periodic updates.

        Args:
            iteration (int): The current iteration number.
        """
        self._track_performance_metrics()

        log_freq = self.pso_config.get('log_frequency', 50)
        if (iteration + 1) % log_freq == 0:
            last_diversity = self.performance_metrics['diversity_history'][-1]
            logging.info(
                f"PSO Iteration {iteration+1:<4d}: "
                f"Swarm Diversity = {last_diversity:.4f}, "
                f"Global Best Fitness={self.global_best_fitness:.6f}, "
                f"Stagnation={self.stagnation_counter}"
            )

    def _track_performance_metrics(self):
        """
        Tracks swarm diversity and velocity statistics for analysis.
        """
        diversity = self._calculate_swarm_diversity()
        self.performance_metrics['diversity_history'].append(diversity)
        if self.swarm_velocities:
            all_velocities = np.concatenate([v for v in self.swarm_velocities if v is not None])
            if all_velocities.size > 0:
                stats = {'mean': np.mean(all_velocities), 'std': np.std(all_velocities), 'max': np.max(all_velocities)}
                self.performance_metrics['velocity_stats'].append(stats)

    def _limit_history_size(self, max_entries: int = 1000):
        """Enhanced memory management with safety checks."""
        try:

            for key, history_list in self.performance_metrics.items():
                if isinstance(history_list, list) and len(history_list) > max_entries:
                    self.performance_metrics[key] = history_list[-max_entries:]

            if hasattr(self, 'adaptive_event_log') and self.adaptive_event_log:
                if len(self.adaptive_event_log) > max_entries:
                    self.adaptive_event_log = self.adaptive_event_log[-max_entries:]

        except Exception as e:
            logging.warning(f"History limiting failed: {e}")

    def _initialize_swarm(self):
        """Creates the initial swarm with consistent velocity arrays."""
        self.swarm_positions = []
        self.swarm_velocities = []
        initial_max_v = self.pso_config.get('max_change_probability', 0.5)
        for _ in range(self.pso_config['swarm_size']):
            position = self._create_random_particle()
            velocity = np.array(
                self._np_rng.uniform(0, initial_max_v, len(position)),
                dtype=np.float32
            )
            self.swarm_positions.append(position)
            self.swarm_velocities.append(velocity)

    def _create_random_particle(self) -> List[float]:
        """Thread-safe random particle creation."""
        num_central = self.sim_config['num_central_assemblies']
        num_total = self.sim_config['num_assemblies']
        central = [self._rng.choice(self.central_vals) for _ in range(num_central)]
        outer = [self._rng.choice(self.outer_vals) for _ in range(num_total - num_central)]
        return central + outer

    def _initialize_neighborhoods(self):
        """
        Initializes particle neighborhoods based on the chosen topology.
        """
        topology = self.pso_config.get('topology', 'global')
        logging.info(f"Initializing PSO with '{topology}' neighborhood topology.")
        if topology == 'random': self._build_random_neighborhoods()
        elif topology == 'ring': self._build_ring_neighborhoods()
        elif topology == 'fitness_based': self._update_fitness_based_neighborhoods()
        else: self.neighborhoods = {}

    def _build_ring_neighborhoods(self):
        """
        Constructs a ring topology for particle neighborhoods.
        """
        k = self.pso_config.get('neighborhood_size', 4)
        swarm_size = self.pso_config['swarm_size']
        self.neighborhoods = {}
        for i in range(swarm_size):
            neighbors = [(i + j) % swarm_size for j in range(-(k//2), k//2 + 1)]
            self.neighborhoods[i] = list(set(neighbors) - {i})

    def _update_dynamic_neighborhoods(self):
        """
        Periodically rebuilds neighborhoods for dynamic topologies (random, fitness-based).
        """
        topology = self.pso_config.get('topology', 'global')
        if topology == 'random':
            self._build_random_neighborhoods()
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Rebuilt random neighborhoods")
        elif topology == 'fitness_based':
            self._update_fitness_based_neighborhoods()
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Rebuilt fitness-based neighborhoods")

    def _build_random_neighborhoods(self):
        """
        Constructs a random topology for particle neighborhoods.
        """
        swarm_size, k = self.pso_config['swarm_size'], self.pso_config.get('neighborhood_size', 4)
        self.neighborhoods = {}
        for i in range(swarm_size):
            others = [p for p in range(swarm_size) if p != i]
            self.neighborhoods[i] = self._rng.sample(others, min(k, len(others)))

    def _update_fitness_based_neighborhoods(self):
        """
        Constructs neighborhoods based on particle fitness rankings.
        
        Particles are connected to other particles that have similar fitness ranks.
        """
        if not self.personal_best_fitnesses or any(f == -float('inf') for f in self.personal_best_fitnesses):
            self._build_random_neighborhoods()
            return

        swarm_size, k = self.pso_config.get('swarm_size'), self.pso_config.get('neighborhood_size', 4)
        sorted_indices = np.argsort(self.personal_best_fitnesses)
        self.neighborhoods = {}
        for i in range(swarm_size):
            rank_list = np.where(sorted_indices == i)[0]
            if len(rank_list) == 0: continue
            rank = rank_list[0]
            
            neighbors = []
            for j in range(1, k // 2 + 1):
                neighbors.append(sorted_indices[(rank - j + swarm_size) % swarm_size])
                neighbors.append(sorted_indices[(rank + j) % swarm_size])
            self.neighborhoods[i] = list(set(neighbors))

    def _get_neighborhood_best(self, particle_idx: int) -> List[float]:
        """Safe neighborhood best with bounds checking."""
        topology = self.pso_config.get('topology', 'global')

        if topology == 'global':
            if self.global_best_position is not None:
                return self.global_best_position

            if 0 <= particle_idx < len(self.personal_best_positions):
                return self.personal_best_positions[particle_idx]
            return self._create_random_particle()  # Fallback

        neighbor_indices = self.neighborhoods.get(particle_idx, []) + [particle_idx]
        best_fitness = -float('inf')
        best_pos = None


        for neighbor_idx in neighbor_indices:
            if (0 <= neighbor_idx < len(self.personal_best_fitnesses) and
                    0 <= neighbor_idx < len(self.personal_best_positions)):
                if self.personal_best_fitnesses[neighbor_idx] > best_fitness:
                    best_fitness = self.personal_best_fitnesses[neighbor_idx]
                    best_pos = self.personal_best_positions[neighbor_idx]


        if best_pos is None:
            if 0 <= particle_idx < len(self.personal_best_positions):
                return self.personal_best_positions[particle_idx]
            return self._create_random_particle()

        return best_pos

    def _evaluate_swarm(self, positions: List[List[float]]) -> List[float]:
        """Enhanced swarm evaluation with error handling."""
        if not positions:
            return []

        try:

            valid_positions = []
            for pos in positions:
                if pos and len(pos) == len(positions[0]):
                    valid_positions.append(pos)
                else:
                    logging.warning("Invalid position found, using random replacement")
                    valid_positions.append(self._create_random_particle())

            keff_preds = self.keff_interpolator.predict_batch(valid_positions)
            ppf_preds = self.ppf_interpolator.predict_batch(valid_positions)


            if len(keff_preds) != len(valid_positions) or len(ppf_preds) != len(valid_positions):
                logging.error("Prediction length mismatch")
                return [-float('inf')] * len(valid_positions)

            fitnesses = []
            for k, p in zip(keff_preds, ppf_preds):
                try:
                    fitness = fitness_function(k, p, self.sim_config, self.tuning_config)

                    if not isinstance(fitness, (int, float)) or np.isnan(fitness) or np.isinf(fitness):
                        fitness = -float('inf')
                    fitnesses.append(fitness)
                except Exception as e:
                    logging.warning(f"Fitness evaluation failed: {e}")
                    fitnesses.append(-float('inf'))

            return fitnesses

        except Exception as e:
            logging.error(f"Swarm evaluation failed: {e}")
            return [-float('inf')] * len(positions)

    def _get_adaptive_velocity_bounds(self) -> float:
        """
        Dynamically adjusts the maximum velocity (change probability).

        The maximum velocity decreases as the optimization progresses and gets a
        boost when the swarm is stagnating.

        Returns:
            float: The adapted maximum velocity.
        """
        if not self.pso_config.get('adaptive_velocity', False): 
            return self.pso_config.get('max_change_probability', 0.5)

        base_prob = self.pso_config.get('base_change_probability', 0.3)
        max_prob = self.pso_config.get('max_change_probability', 0.8)
        progress = self._current_iteration / self.pso_config['iterations_per_openmc_cycle']

        adaptive_max = max_prob - (max_prob - base_prob) * progress
        if self.stagnation_counter > 20:
            return min(adaptive_max + 0.2, 1.0)
        return adaptive_max

    def _update_personal_bests(self, fitnesses: List[float]):
        """
        Updates each particle's personal best position if its new fitness is higher.

        Args:
            fitnesses (List[float]): The list of new fitness values for the swarm.
        """
        for i, fitness in enumerate(fitnesses):
            if fitness > self.personal_best_fitnesses[i]:
                self.personal_best_fitnesses[i] = fitness
                self.personal_best_positions[i] = self.swarm_positions[i][:]

    def _update_global_best(self, fitnesses: List[float], positions: List[List[float]]):
        """
        Updates the global best position if a better particle is found in the swarm.

        Args:
            fitnesses (List[float]): The list of new fitness values for the swarm.
            positions (List[List[float]]): The current positions of the swarm.
        """
        if not fitnesses: return
        max_fitness_in_gen = max(fitnesses)
        if max_fitness_in_gen > self.global_best_fitness:
            best_idx = fitnesses.index(max_fitness_in_gen)
            self.global_best_fitness = max_fitness_in_gen
            self.global_best_position = positions[best_idx][:]