import random
import numpy as np
import logging
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
        
        if self.config.get('enable_multi_swarm', False):
            self.num_sub_swarms = self.config.get('num_sub_swarms', 3)
            self.migration_frequency = self.config.get('migration_frequency', 100)
            self.migration_rate = self.config.get('migration_rate', 0.1)
            self._initialize_sub_swarms()

    def get_state(self) -> Dict:
        """
        Serializes the internal state of the multi-swarm handler.
        """
        sub_swarms_state = []
        for swarm in self.sub_swarms:
            state = swarm.copy()
            # Ensure all numpy arrays are converted to lists for JSON serialization
            state['velocities'] = [v.tolist() for v in swarm['velocities']]
            state['best_position'] = swarm['best_position'][:] if swarm.get('best_position') else None
            state['personal_bests'] = [p[:] for p in swarm['personal_bests']]
            sub_swarms_state.append(state)
        return {"sub_swarms": sub_swarms_state}

    def load_state(self, state: Dict):
        """
        Loads the multi-swarm handler's state from a dictionary.
        """
        sub_swarms_state = state.get("sub_swarms", [])
        self.sub_swarms = []
        for s_state in sub_swarms_state:
            loaded_swarm = s_state.copy()
            # Ensure velocities are loaded back as numpy arrays
            loaded_swarm['velocities'] = [np.array(v, dtype=np.float32) for v in s_state['velocities']]
            self.sub_swarms.append(loaded_swarm)

        if self.sub_swarms:
            logging.info(f"Multi-swarm state loaded with {len(self.sub_swarms)} sub-swarms.")
            # Immediately synchronize the main swarm after loading sub-swarm state
            self._synchronize_main_swarm_from_subs()


    def _initialize_sub_swarms(self):
        """
        Initializes multiple sub-swarms with potentially different characteristics.
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
            
            # (Thread Safety): Use the main PSO's RNGs to ensure consistency
            for _ in range(num_particles_in_swarm):
                position = self.main_pso._create_random_particle()
                max_v = sub_swarm_config.get('max_change_probability', 0.5)
                # Use main_pso._np_rng for numpy random operations
                velocity = self.main_pso._np_rng.uniform(0, max_v, len(position)).astype(np.float32)
                sub_swarm['positions'].append(position)
                sub_swarm['velocities'].append(velocity)
                sub_swarm['personal_bests'].append(position[:])
                sub_swarm['personal_best_fitnesses'].append(-float('inf'))

            self.sub_swarms.append(sub_swarm)
            particle_start_idx += num_particles_in_swarm

    def _migrate_particles(self):
        """
        Migrates the best particles from each swarm to another.
        """
        if not self.sub_swarms or len(self.sub_swarms) < 2:
            return

        logging.info("Performing multi-swarm migration.")
        try:
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
                    # Deep copy to prevent reference issues
                    migrating_position = source_swarm['personal_bests'][source_idx][:]
                    migrating_velocity = source_swarm['velocities'][source_idx].copy()
                    migrating_fitness = source_swarm['personal_best_fitnesses'][source_idx]
                    
                    target_swarm['positions'][target_idx] = migrating_position
                    target_swarm['velocities'][target_idx] = migrating_velocity
                    target_swarm['personal_bests'][target_idx] = migrating_position
                    target_swarm['personal_best_fitnesses'][target_idx] = migrating_fitness
                    
                    global_target_idx = target_swarm['particle_indices'][target_idx]
                    self.main_pso.swarm_positions[global_target_idx] = migrating_position
                    self.main_pso.swarm_velocities[global_target_idx] = migrating_velocity
                    self.main_pso.personal_best_positions[global_target_idx] = migrating_position
                    self.main_pso.personal_best_fitnesses[global_target_idx] = migrating_fitness

        except (IndexError, ValueError) as e:
            logging.error(f"Error during particle migration: {e}. Skipping migration for this iteration.")


    def run_multi_swarm_iteration(self, inertia_weight: float, neighborhood_bests: List[List[float]], current_keffs: List[Optional[float]]):
        """
        Runs one full iteration of the multi-swarm PSO algorithm.

        Args:
            inertia_weight (float): The current inertia weight for the PSO update.
            neighborhood_bests (List[List[float]]): Pre-calculated neighborhood bests for all particles.
            current_keffs (List[Optional[float]]): Pre-calculated keffs for smart mutation.
        """
        if not self.sub_swarms:
            self.main_pso._update_swarm_discrete(inertia_weight, neighborhood_bests, current_keffs)
            return

        # First, update all sub-swarms based on their own logic
        for swarm in self.sub_swarms:
            self._update_sub_swarm(swarm, inertia_weight, neighborhood_bests, current_keffs)

        # Then, synchronize the main swarm with the new positions/velocities from sub-swarms
        self._synchronize_main_swarm_from_subs()

        # Perform migration if it's time
        if self.main_pso._current_iteration > 0 and self.main_pso._current_iteration % self.migration_frequency == 0:
            self._migrate_particles()
            self.main_pso.adaptive_event_log.append(f"Iter {self.main_pso._current_iteration}: Multi-swarm migration performed")

    def _update_sub_swarm(self, swarm: Dict, inertia_weight: float, neighborhood_bests: List[List[float]], current_keffs: List[Optional[float]]):
        """
        Updates the velocities and positions of all particles in a single sub-swarm.
        
        Args:
            swarm (Dict): The sub-swarm to update.
            inertia_weight (float): The current inertia weight.
            neighborhood_bests (List[List[float]]): The pre-calculated neighborhood best positions for ALL particles in the main swarm.
            current_keffs (List[Optional[float]]): The pre-calculated keffs for ALL particles in the main swarm.
        """
        c1 = swarm['config']['cognitive_coeff']
        c2 = swarm['config']['social_coeff']
        max_v = swarm['config']['max_change_probability']

        for i, global_idx in enumerate(swarm['particle_indices']):
            # (Race Condition): Use the pre-calculated neighborhood best for the current global index.
            # This ensures all particles in one iteration use the same consistent state.
            nbest_position = neighborhood_bests[global_idx]
            
            new_pos, new_vel = self.main_pso._update_single_particle(
                global_idx, inertia_weight, c1, c2, max_v, nbest_position, current_keffs[global_idx]
            )
            swarm['positions'][i] = new_pos
            swarm['velocities'][i] = new_vel
    
    # Renamed for clarity. This method now correctly syncs sub-swarm data *to* the main swarm
    # before the main evaluation step.
    def _synchronize_main_swarm_from_subs(self):
        """
        Aggregates positions and velocities from all sub-swarms into the main PSO handler's attributes.
        This is done BEFORE global evaluation to ensure the main swarm has the latest particle states.
        """
        if not self.sub_swarms: return

        all_positions, all_velocities = [], []
        for swarm in self.sub_swarms:
            all_positions.extend(swarm['positions'])
            all_velocities.extend(swarm['velocities'])
            
        self.main_pso.swarm_positions = all_positions
        self.main_pso.swarm_velocities = all_velocities

    # This method now propagates updated fitness values FROM the main swarm back to sub-swarms
    # AFTER the main evaluation step.
    def update_sub_swarm_bests_from_main(self):
        """
        Updates personal and swarm-level bests for each sub-swarm using the globally evaluated fitnesses
        from the main PSO instance.
        """
        if not self.sub_swarms: return

        for swarm in self.sub_swarms:
            for i, global_idx in enumerate(swarm['particle_indices']):
                # Check if the main swarm's pbest is better
                if self.main_pso.personal_best_fitnesses[global_idx] > swarm['personal_best_fitnesses'][i]:
                    swarm['personal_best_fitnesses'][i] = self.main_pso.personal_best_fitnesses[global_idx]
                    swarm['personal_bests'][i] = self.main_pso.personal_best_positions[global_idx][:]

            # Update the sub-swarm's own global best
            if swarm['personal_best_fitnesses']:
                best_local_idx = np.argmax(swarm['personal_best_fitnesses'])
                current_best_fitness = swarm['personal_best_fitnesses'][best_local_idx]
                if current_best_fitness > swarm['best_fitness']:
                    swarm['best_fitness'] = current_best_fitness
                    swarm['best_position'] = swarm['personal_bests'][best_local_idx][:]

class ParticleSwarmOptimizer:
    """
    Implements a robust Discrete Particle Swarm Optimization (DPSO) engine.
    """
    def __init__(self, config: Dict, keff_interpolator, ppf_interpolator):
        """
        Initializes the Particle Swarm Optimizer.
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

        # Per-instance RNGs for thread safety.
        # All random calls in this class will use these.
        self._rng = random.Random()
        self._np_rng = np.random.RandomState()

        self._diversity_boost_counter = 0
        self.multi_swarm_handler = None
        if self.pso_config.get('enable_multi_swarm', False):
            self.multi_swarm_handler = MultiSwarmPSO(self)

    def get_state(self) -> Dict:
        """Returns serializable state with proper type conversion."""
        state = {
            "swarm_positions": self.swarm_positions,
            "swarm_velocities": [v.tolist() for v in self.swarm_velocities],
            "personal_best_positions": self.personal_best_positions,
            "personal_best_fitnesses": self.personal_best_fitnesses,
            "global_best_position": self.global_best_position,
            "global_best_fitness": self.global_best_fitness,
            "stagnation_counter": self.stagnation_counter,
            "adaptive_event_log": self.adaptive_event_log[-1000:],
            "performance_metrics": {
                k: v[-1000:] for k, v in self.performance_metrics.items()
            },
            "_diversity_boost_counter": self._diversity_boost_counter,
            "multi_swarm_handler_state": self.multi_swarm_handler.get_state() if self.multi_swarm_handler else None,
        }
        return state

    def load_state(self, state: Dict):
        """Enhanced state loading with validation."""
        try:
            expected_size = self.pso_config['swarm_size']
            loaded_positions = state["swarm_positions"]

            if expected_size != len(loaded_positions):
                logging.error(f"Swarm size mismatch! Expected {expected_size}, got {len(loaded_positions)}. Reinitializing.")
                self._full_initialization()
                return

            self.swarm_positions = loaded_positions
            self.swarm_velocities = [np.array(v, dtype=np.float32) for v in state["swarm_velocities"]]
            self.personal_best_positions = state["personal_best_positions"]
            self.personal_best_fitnesses = state["personal_best_fitnesses"]
            self.global_best_position = state["global_best_position"]
            self.global_best_fitness = state.get("global_best_fitness", -float('inf'))
            self.stagnation_counter = state.get("stagnation_counter", 0)

            if not self._validate_loaded_state():
                logging.error("Loaded state validation failed. Reinitializing.")
                self._full_initialization()
                return

            multi_swarm_state = state.get("multi_swarm_handler_state")
            if self.multi_swarm_handler and multi_swarm_state:
                self.multi_swarm_handler.load_state(multi_swarm_state)

            logging.info(f"PSO state loaded successfully. Global best: {self.global_best_fitness:.6f}")

        except (KeyError, TypeError, ValueError) as e:
            logging.error(f"State loading failed due to invalid data: {e}. Reinitializing swarm.")
            self._full_initialization()
    
    def initialize_with_population(self, individuals: List[List[float]]):
        """
        Re-initializes the swarm with a given population, typically from the GA.
        This method seeds the PSO with promising solutions.
        """
        logging.info(f"Seeding PSO swarm with {len(individuals)} individuals from GA.")
        swarm_size = self.pso_config['swarm_size']
        seeded_population = [ind[:] for ind in individuals]

        # If not enough individuals are provided, fill the rest of the swarm randomly
        while len(seeded_population) < swarm_size:
            seeded_population.append(self._create_random_particle())
        
        # Trim if too many individuals were provided
        self.swarm_positions = seeded_population[:swarm_size]
        
        # Reset velocities for the new swarm
        self.swarm_velocities = []
        initial_max_v = self.pso_config.get('max_change_probability', 0.5)
        num_dims = self.sim_config['num_assemblies']
        for _ in range(len(self.swarm_positions)):
            velocity = self._np_rng.uniform(0, initial_max_v, num_dims).astype(np.float32)
            self.swarm_velocities.append(velocity)

        # Reset personal and global bests to force re-evaluation
        self.personal_best_positions = []
        self.personal_best_fitnesses = []
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        
        # Perform an initial evaluation of the new (seeded) swarm
        self._evaluate_and_update_bests()
        logging.info(f"PSO swarm successfully seeded. New global best fitness: {self.global_best_fitness:.6f}")


    def _validate_loaded_state(self) -> bool:
        """Validates consistency of loaded state."""
        expected_size = self.pso_config['swarm_size']
        if not all(isinstance(v, np.ndarray) for v in self.swarm_velocities): return False
        
        return all(len(arr) == expected_size for arr in [
            self.swarm_positions, self.swarm_velocities,
            self.personal_best_positions, self.personal_best_fitnesses
        ])
    
    def run_pso_algorithm(self, seed_individual: Optional[List[float]] = None) -> List[float]:
        """
        Runs the main PSO algorithm for a configured number of iterations.
        """
        if not self.swarm_positions:
            logging.info("PSO swarm is not initialized. Creating a new swarm.")
            self._full_initialization()

        if seed_individual:
            self._seed_global_best(seed_individual, replace_worst=True)

        max_iterations = self.pso_config['iterations_per_openmc_cycle']
        convergence_threshold = self.pso_config.get('pso_convergence_threshold', 200)
        self.stagnation_counter = 0

        for iteration in range(max_iterations):
            self._current_iteration = iteration
            previous_gbest_fitness = self.global_best_fitness

            w_start, w_end = self.pso_config.get('inertia_weight_start', 0.9), self.pso_config.get('inertia_weight_end', 0.4)
            current_inertia = w_start - (w_start - w_end) * (iteration / max_iterations)

            # Rebuild dynamic neighborhoods if needed
            if (iteration + 1) % self.pso_config.get('neighborhood_rebuild_frequency', 100) == 0:
                self._update_dynamic_neighborhoods()
            
            # Pre-calculate neighborhood bests for all particles to ensure consistency
            neighborhood_bests = [self._get_neighborhood_best(i) for i in range(self.pso_config['swarm_size'])]

            # Pre-calculate keffs for smart mutation, if enabled
            if self.pso_config.get('enable_smart_mutation', False):
                current_keffs = self.keff_interpolator.predict_batch(self.swarm_positions)
            else:
                current_keffs = [None] * self.pso_config['swarm_size']

            # Run the appropriate update logic
            if self.multi_swarm_handler:
                self.multi_swarm_handler.run_multi_swarm_iteration(current_inertia, neighborhood_bests, current_keffs)
            else:
                self._update_swarm_discrete(current_inertia, neighborhood_bests, current_keffs)

            # Evaluate the entire swarm and update personal/global bests
            self._evaluate_and_update_bests()

            # Stagnation and convergence check
            if self.global_best_fitness > previous_gbest_fitness:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            if self.stagnation_counter > convergence_threshold:
                logging.info(f"PSO convergence after {self.stagnation_counter} stagnant iterations... Exiting.")
                break

            # Apply periodic enhancements
            if self.pso_config.get('enable_local_search', False) and (iteration + 1) % self.pso_config.get('local_search_frequency', 25) == 0:
                self._apply_periodic_local_search()
            
            self._track_and_log_performance(iteration)

        self._limit_history_size()
        return self.global_best_position

    def _full_initialization(self):
        """Initializes the entire swarm state from scratch."""
        # Enforce correct initialization order. Swarm is created first.
        if self.multi_swarm_handler:
            self.multi_swarm_handler._initialize_sub_swarms()
            self.multi_swarm_handler._synchronize_main_swarm_from_subs()
        else:
            self._initialize_swarm()
        
        # Neighborhoods are initialized only after the swarm and p-bests exist.
        self._initialize_neighborhoods()
        self._evaluate_and_update_bests()

    def _evaluate_and_update_bests(self):
        """Evaluates the entire swarm's fitness and updates personal and global bests."""
        if not self.swarm_positions: return

        fitnesses = self._evaluate_swarm(self.swarm_positions)
        
        if not self.personal_best_fitnesses:
            self.personal_best_fitnesses = [-float('inf')] * len(self.swarm_positions)
            self.personal_best_positions = [pos[:] for pos in self.swarm_positions]

        self._update_personal_bests(fitnesses)
        self._update_global_best(fitnesses, self.swarm_positions)
        
        # After main evaluation, propagate updated fitnesses back to sub-swarms.
        if self.multi_swarm_handler:
            self.multi_swarm_handler.update_sub_swarm_bests_from_main()

    def _seed_global_best(self, seed_individual: List[float], replace_worst: bool = False):
        """Updates the global best with an external seed solution."""
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

                # Also update the sub-swarm if applicable
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
        """Applies a local search heuristic to the current global best solution."""
        if not self.global_best_position: return

        original_pos = self.global_best_position[:]
        improved_pos, new_fitness = self._implement_local_search(original_pos, self.global_best_fitness)

        if new_fitness > self.global_best_fitness:
            self.global_best_position = improved_pos
            self.global_best_fitness = new_fitness
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Periodic local search found improvement.")
    
    def activate_post_switch_boost(self):
        """
        Activates a temporary exploration boost, typically after a switch from another
        algorithm, to encourage the swarm to spread out.
        """
        boost_duration = self.pso_config.get('post_switch_boost_duration', 30)
        self._diversity_boost_counter = boost_duration
        self.adaptive_event_log.append(f"Post-switch exploration boost activated for {boost_duration} iterations.")
        logging.info(f"PSO post-switch exploration boost activated for {boost_duration} iterations.")

    def _implement_local_search(self, position: List[float], current_fitness: float) -> Tuple[List[float], float]:
        """
        Implements a simple hill-climbing local search.
        It creates a few neighbors by mutating the given position and returns the best one found.
        """
        best_neighbor_pos = position[:]
        best_neighbor_fitness = current_fitness
        num_neighbors = 5  # Number of neighbors to explore
        
        for _ in range(num_neighbors):
            neighbor_pos = position[:]
            
            # Mutate one or two random genes
            for _ in range(self._rng.randint(1, 3)):
                idx_to_mutate = self._rng.randint(0, len(neighbor_pos) - 1)
                
                is_central = idx_to_mutate < self.sim_config['num_central_assemblies']
                valid_values = self.central_vals if is_central else self.outer_vals
                
                # Select a new value that is different from the current one
                current_val = neighbor_pos[idx_to_mutate]
                possible_new_vals = [v for v in valid_values if v != current_val]
                if possible_new_vals:
                    neighbor_pos[idx_to_mutate] = self._rng.choice(possible_new_vals)

            # Evaluate the new neighbor
            neighbor_keff = self.keff_interpolator.predict(neighbor_pos)
            neighbor_ppf = self.ppf_interpolator.predict(neighbor_pos)
            neighbor_fitness = fitness_function(neighbor_keff, neighbor_ppf, self.sim_config, self.tuning_config)

            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor_fitness = neighbor_fitness
                best_neighbor_pos = neighbor_pos
                
        return best_neighbor_pos, best_neighbor_fitness

    def _calculate_swarm_diversity(self) -> float:
        """Calculates the swarm's diversity using average Hamming distance."""
        if len(self.swarm_positions) < 2: return 0.0

        sample_size = min(len(self.swarm_positions), 100)
        # Use instance-specific RNG
        sample_indices = self._rng.sample(range(len(self.swarm_positions)), sample_size)
        sample_positions = [self.swarm_positions[i] for i in sample_indices]
        
        total_distance, comparisons = 0.0, 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                distance = sum(1 for a, b in zip(sample_positions[i], sample_positions[j]) if a != b)
                total_distance += distance
                comparisons += 1
        return total_distance / comparisons if comparisons > 0 else 0.0

    def _apply_diversity_maintenance(self):
        """Applies strategies if swarm diversity drops below a threshold."""
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

            # Also update the sub-swarm if applicable
            if self.multi_swarm_handler:
                self._update_subswarms_after_reinitialization(worst_indices)

            self._diversity_boost_counter = 20
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Low diversity ({current_diversity:.3f}) - Reinitialized {num_to_reinit} particles")

    def _update_subswarms_after_reinitialization(self, indices: List[int]):
        """Helper to sync sub-swarms after diversity maintenance."""
        for idx in indices:
            for swarm in self.multi_swarm_handler.sub_swarms:
                if idx in swarm['particle_indices']:
                    local_idx = swarm['particle_indices'].index(idx)
                    swarm['positions'][local_idx] = self.swarm_positions[idx]
                    swarm['velocities'][local_idx] = self.swarm_velocities[idx]
                    swarm['personal_bests'][local_idx] = self.personal_best_positions[idx][:]
                    swarm['personal_best_fitnesses'][local_idx] = self.personal_best_fitnesses[idx]
                    break
    
    # Removed potential for infinite recursion.
    # The method now ensures that the fallback does not re-trigger validation.
    def _validate_and_fix_particle(self, position: List[float], recursion_depth=0) -> List[float]:
        """Enhanced particle validation with recursion guard."""
        if recursion_depth > 1:
            logging.error("Recursive validation failed. Generating a new random particle.")
            return self._create_random_particle()

        if not position:
            logging.error("Empty position provided to validation")
            return self._validate_and_fix_particle(self._create_random_particle(), recursion_depth + 1)

        try:
            fixed_position = []
            num_central = self.sim_config['num_central_assemblies']

            for i, value in enumerate(position):
                is_central = i < num_central
                valid_values = self.central_vals if is_central else self.outer_vals

                if not valid_values:
                    logging.error(f"No valid values for position index {i}. Using fallback.")
                    fallback_value = self.central_vals[0] if is_central and self.central_vals else (self.outer_vals[0] if self.outer_vals else 0.0)
                    fixed_position.append(fallback_value)
                    continue

                if value not in valid_values:
                    closest_value = min(valid_values, key=lambda x: abs(x - value))
                    fixed_position.append(closest_value)
                else:
                    fixed_position.append(value)
            return fixed_position

        except Exception as e:
            logging.error(f"Particle validation failed: {e}. Attempting to create a new one.")
            return self._validate_and_fix_particle(self._create_random_particle(), recursion_depth + 1)
    
    def _update_swarm_discrete(self, inertia_weight: float, neighborhood_bests: List[List[float]], current_keffs: List[Optional[float]]):
        """
        Updates particle velocities and positions for a single-swarm configuration.
        """
        if self._current_iteration > 0 and self._current_iteration % 50 == 0:
            self._apply_diversity_maintenance()

        max_v = self.pso_config.get('max_change_probability', 0.5) # Max velocity is used as clamp
        c1, c2 = self.pso_config['cognitive_coeff'], self.pso_config.get('social_coeff', 1.5)

        for i in range(self.pso_config['swarm_size']):
            nbest_position = neighborhood_bests[i]
            new_pos, new_vel = self._update_single_particle(i, inertia_weight, c1, c2, max_v, nbest_position, current_keffs[i])
            self.swarm_positions[i] = new_pos
            self.swarm_velocities[i] = new_vel

    # Complete overhaul of the discrete update logic.
    # This now uses a standard and mathematically sound approach for discrete PSO.
    # Velocity now represents the probability of change towards pbest/gbest.
    def _update_single_particle(
        self, particle_idx: int, inertia_weight: float, c1: float, c2: float,
        max_v: float, nbest_position: List[float], current_keff: Optional[float]
    ) -> Tuple[List[float], np.ndarray]:
        """
        Updates a single particle's velocity and position using a standard discrete PSO model.
        """
        try:
            position = self.swarm_positions[particle_idx]
            velocity = self.swarm_velocities[particle_idx]
            pbest_position = self.personal_best_positions[particle_idx]

            r1, r2 = self._rng.random(), self._rng.random()

            # Cognitive component: attraction to personal best
            cognitive_attraction = c1 * r1 * (np.array(pbest_position) != np.array(position))
            
            # Social component: attraction to neighborhood best
            social_attraction = c2 * r2 * (np.array(nbest_position) != np.array(position))
            
            # Update velocity: v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            new_vel = inertia_weight * velocity + cognitive_attraction + social_attraction
            new_vel = np.clip(new_vel, 0, max_v) # Clamp velocity

            new_pos = position[:]
            
            # Update position based on velocity (probability of change)
            for j in range(len(position)):
                # Use sigmoid function to map velocity to a probability [0, 1]
                prob_change = 1 / (1 + np.exp(-new_vel[j]))
                
                if self._rng.random() < prob_change:
                    # If we decide to change, choose between pbest and nbest
                    # This is a simplified but effective way to handle the discrete choice.
                    if self._rng.random() < 0.5: # 50% chance to move towards pbest
                        if pbest_position[j] != position[j]:
                           new_pos[j] = pbest_position[j]
                    else: # 50% chance to move towards nbest
                        if nbest_position[j] != position[j]:
                           new_pos[j] = nbest_position[j]

            # GA-Style Smart Mutation 
            if self.pso_config.get('enable_smart_mutation', False) and current_keff is not None:
                mut_rate = self.pso_config.get('mutation_probability', 0.08)
                prob_bias = self.pso_config.get('smart_mutation_bias', 0.75)
                keff_diff = current_keff - self.sim_config['target_keff']

                for j in range(len(new_pos)):
                    if self._rng.random() < mut_rate:
                        is_central = j < self.sim_config['num_central_assemblies']
                        values = self.central_vals if is_central else self.outer_vals
                        current_val = new_pos[j]
                        
                        available = []
                        if abs(keff_diff) > 0.0005: # Apply bias
                            if keff_diff < 0: # Need to increase reactivity
                                preferred_vals = [v for v in values if v > current_val]
                                if self._rng.random() < prob_bias and preferred_vals:
                                    available = preferred_vals
                                else: # Fallback to any other value
                                    available = [v for v in values if v != current_val]
                            else: # Need to decrease reactivity
                                preferred_vals = [v for v in values if v < current_val]
                                if self._rng.random() < prob_bias and preferred_vals:
                                    available = preferred_vals
                                else: # Fallback
                                    available = [v for v in values if v != current_val]
                        else: # Near target, purely random mutation
                            available = [v for v in values if v != current_val]
                        
                        if available:
                            new_pos[j] = self._rng.choice(available)

            # Final validation to ensure the particle is valid
            new_pos = self._validate_and_fix_particle(new_pos)

            return new_pos, new_vel

        except (IndexError, ValueError) as e:
            logging.error(f"Error updating particle {particle_idx}: {e}. Returning random particle.")
            fallback_pos = self._create_random_particle()
            fallback_vel = np.zeros(len(fallback_pos), dtype=np.float32)
            return fallback_pos, fallback_vel


    def _validate_pso_config(self):
        """Enhanced configuration validation."""
        required_pso = ['swarm_size', 'iterations_per_openmc_cycle', 'cognitive_coeff']
        for param in required_pso:
            if param not in self.pso_config: raise ValueError(f"Missing required PSO parameter: {param}")

        required_enrich = ['central_values', 'outer_values']
        for param in required_enrich:
            if param not in self.enrich_config: raise ValueError(f"Missing required enrichment parameter: {param}")

        required_sim = ['num_assemblies', 'num_central_assemblies']
        for param in required_sim:
            if param not in self.sim_config: raise ValueError(f"Missing required simulation parameter: {param}")
    
    def _track_and_log_performance(self, iteration: int):
        """
        Tracks performance metrics and logs periodic updates.
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
        """Tracks swarm diversity and velocity statistics for analysis."""
        diversity = self._calculate_swarm_diversity()
        self.performance_metrics['diversity_history'].append(diversity)
        if self.swarm_velocities:
            # Call history limiting immediately to prevent memory leaks.
            self._limit_history_size()

    def _limit_history_size(self, max_entries: int = 1000):
        """Enhanced memory management with safety checks."""
        try:
            for key, history_list in self.performance_metrics.items():
                if isinstance(history_list, list) and len(history_list) > max_entries:
                    self.performance_metrics[key] = history_list[-max_entries:]

            if hasattr(self, 'adaptive_event_log') and len(self.adaptive_event_log) > max_entries:
                self.adaptive_event_log = self.adaptive_event_log[-max_entries:]
        except Exception as e:
            logging.warning(f"History limiting failed: {e}")

    def _initialize_swarm(self):
        """Creates the initial swarm with consistent velocity arrays."""
        self.swarm_positions = []
        self.swarm_velocities = []
        initial_max_v = self.pso_config.get('max_change_probability', 0.5)
        num_dims = self.sim_config['num_assemblies']
        for _ in range(self.pso_config['swarm_size']):
            position = self._create_random_particle()
            velocity = self._np_rng.uniform(0, initial_max_v, num_dims).astype(np.float32)
            self.swarm_positions.append(position)
            self.swarm_velocities.append(velocity)

    def _create_random_particle(self) -> List[float]:
        """Thread-safe random particle creation."""
        num_central = self.sim_config['num_central_assemblies']
        num_total = self.sim_config['num_assemblies']
        # Instance-specific RNG
        central = [self._rng.choice(self.central_vals) for _ in range(num_central)]
        outer = [self._rng.choice(self.outer_vals) for _ in range(num_total - num_central)]
        return central + outer

    def _initialize_neighborhoods(self):
        """Initializes particle neighborhoods based on the chosen topology."""
        topology = self.pso_config.get('topology', 'global')
        logging.info(f"Initializing PSO with '{topology}' neighborhood topology.")
        if topology == 'random': self._build_random_neighborhoods()
        elif topology == 'ring': self._build_ring_neighborhoods()
        elif topology == 'fitness_based': self._update_fitness_based_neighborhoods()
        else: self.neighborhoods = {} # Global topology uses an empty dict

    def _build_ring_neighborhoods(self):
        """Constructs a ring topology for particle neighborhoods."""
        k = self.pso_config.get('neighborhood_size', 4)
        swarm_size = self.pso_config['swarm_size']
        self.neighborhoods = {}
        for i in range(swarm_size):
            neighbors = [(i + j) % swarm_size for j in range(-(k//2), k//2 + 1)]
            self.neighborhoods[i] = [n for n in neighbors if n != i]
    
    def _update_dynamic_neighborhoods(self):
        """Periodically rebuilds neighborhoods for dynamic topologies."""
        topology = self.pso_config.get('topology', 'global')
        if topology == 'random':
            self._build_random_neighborhoods()
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Rebuilt random neighborhoods")
        elif topology == 'fitness_based':
            self._update_fitness_based_neighborhoods()
            self.adaptive_event_log.append(f"Iter {self._current_iteration}: Rebuilt fitness-based neighborhoods")

    def _build_random_neighborhoods(self):
        """Constructs a random topology for particle neighborhoods."""
        swarm_size, k = self.pso_config['swarm_size'], self.pso_config.get('neighborhood_size', 4)
        self.neighborhoods = {}
        for i in range(swarm_size):
            others = [p for p in range(swarm_size) if p != i]
            # Use instance-specific RNG
            self.neighborhoods[i] = self._rng.sample(others, min(k, len(others)))
            
    def _get_neighborhood_best(self, particle_idx: int) -> List[float]:
        """Safe neighborhood best with bounds checking and fallback."""
        topology = self.pso_config.get('topology', 'global')

        # For global topology, the best is always the global best
        if topology == 'global':
            if self.global_best_position:
                return self.global_best_position
            # Fallback if gbest is not yet set
            return self.personal_best_positions[particle_idx] if self.personal_best_positions else self._create_random_particle()

        best_fitness = -float('inf')
        # Start with the particle's own pbest as the initial best in its neighborhood
        best_pos = self.personal_best_positions[particle_idx]

        # Add robust checks to prevent index errors
        neighbor_indices = self.neighborhoods.get(particle_idx, [])
        for neighbor_idx in neighbor_indices:
            if 0 <= neighbor_idx < len(self.personal_best_fitnesses):
                if self.personal_best_fitnesses[neighbor_idx] > best_fitness:
                    best_fitness = self.personal_best_fitnesses[neighbor_idx]
                    best_pos = self.personal_best_positions[neighbor_idx]
            else:
                logging.warning(f"Invalid neighbor index {neighbor_idx} for particle {particle_idx}. Skipping.")
        
        return best_pos

    def _evaluate_swarm(self, positions: List[List[float]]) -> List[float]:
        """Enhanced swarm evaluation with error handling."""
        if not positions:
            return []

        try:
            # Batch prediction is more efficient
            keff_preds = self.keff_interpolator.predict_batch(positions)
            ppf_preds = self.ppf_interpolator.predict_batch(positions)

            fitnesses = [
                fitness_function(k, p, self.sim_config, self.tuning_config)
                for k, p in zip(keff_preds, ppf_preds)
            ]
            
            # Ensure fitness values are valid numbers
            return [f if isinstance(f, (int, float)) and np.isfinite(f) else -float('inf') for f in fitnesses]

        except Exception as e:
            logging.error(f"Swarm evaluation failed: {e}. Returning worst fitness for all particles.")
            return [-float('inf')] * len(positions)

    def _update_personal_bests(self, fitnesses: List[float]):
        """Updates each particle's personal best position if its new fitness is higher."""
        for i, fitness in enumerate(fitnesses):
            if fitness > self.personal_best_fitnesses[i]:
                self.personal_best_fitnesses[i] = fitness
                self.personal_best_positions[i] = self.swarm_positions[i][:]

    def _update_global_best(self, fitnesses: List[float], positions: List[List[float]]):
        """Updates the global best position if a better particle is found in the swarm."""
        if not fitnesses: return
        
        max_fitness_in_gen = max(fitnesses)
        if max_fitness_in_gen > self.global_best_fitness:
            best_idx = fitnesses.index(max_fitness_in_gen)
            self.global_best_fitness = max_fitness_in_gen
            self.global_best_position = positions[best_idx][:]
