import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from .ga_engine import GeneticAlgorithm
from .pso_engine import ParticleSwarmOptimizer
from .interpolators import KeffInterpolator, PPFInterpolator
from .utils import calculate_diversity

class HybridEngine:
    """
    Implements an enhanced hybrid optimization strategy that combines a Genetic
    Algorithm (GA) for global exploration and a Particle Swarm
    Optimizer (PSO) for local exploitation, with support for oscillation and
    adaptive switching.
    """
    def __init__(self, config: Dict, keff_interpolator: KeffInterpolator, ppf_interpolator: PPFInterpolator):
        self.hybrid_config = config.get('hybrid', {})
        self.config = config
        
        # Load switching criteria from config
        self.switch_mode = self.hybrid_config.get('switch_mode', 'fixed_cycles')
        self.ga_phase_cycles = self.hybrid_config.get('ga_phase_cycles', 10)
        self.pso_phase_cycles = self.hybrid_config.get('pso_phase_cycles', 15)
        self.stagnation_threshold = self.hybrid_config.get('stagnation_threshold', 10)
        self.ga_min_diversity_for_switch = self.hybrid_config.get('ga_min_diversity_for_switch', 0.25)
        self.ga_seed_ratio = self.hybrid_config.get('ga_seed_ratio', 0.25)
        self.fitness_improvement_tolerance = self.hybrid_config.get('fitness_improvement_tolerance', 1e-6)
        self.adaptive_switching_threshold = self.hybrid_config.get('adaptive_switching_threshold', 1.2)
        self.min_adaptive_phase_duration = self.hybrid_config.get('min_adaptive_phase_duration', 5)
        self.seeding_diversity_threshold = self.hybrid_config.get('seeding_diversity_threshold', 0.1)

        # Validate required configuration
        self._validate_config()
        
        self.ga_engine = GeneticAlgorithm(config, keff_interpolator, ppf_interpolator)
        self.pso_engine = ParticleSwarmOptimizer(config, keff_interpolator, ppf_interpolator)

        # State for the hybrid approach
        self.current_phase = 'ga'
        self.cycles_in_current_phase = 0
        self.stagnation_counter = 0
        self.phase_history: List[Dict] = []

        # Performance Tracking for Adaptive Switching
        self.fitness_at_phase_start = -float('inf')
        self.performance_history: Dict[str, List[float]] = {'ga': [], 'pso': []}

        logging.info(f"Hybrid Engine initialized. Switch mode: '{self.switch_mode.upper()}'")

    def get_state(self) -> Dict:
        """
        Returns the current state of the Hybrid Engine and its sub-engines for checkpointing.
        """
        return {
            "current_phase": self.current_phase,
            "cycles_in_current_phase": self.cycles_in_current_phase,
            "stagnation_counter": self.stagnation_counter,
            "phase_history": self.phase_history,
            "fitness_at_phase_start": self.fitness_at_phase_start,
            "performance_history": self.performance_history,
            "ga_engine_state": self.ga_engine.get_state(),
            "pso_engine_state": self.pso_engine.get_state(),
        }

    def load_state(self, state: Dict):
        """
        Loads the state of the Hybrid Engine and its sub-engines from a checkpoint dictionary.
        """
        self.current_phase = state.get("current_phase", "ga")
        self.cycles_in_current_phase = state.get("cycles_in_current_phase", 0)
        self.stagnation_counter = state.get("stagnation_counter", 0)
        self.phase_history = state.get("phase_history", [])
        self.fitness_at_phase_start = state.get("fitness_at_phase_start", -float('inf'))
        self.performance_history = state.get("performance_history", {'ga': [], 'pso': []})
        
        if "ga_engine_state" in state and state["ga_engine_state"]:
            self.ga_engine.load_state(state["ga_engine_state"])
        if "pso_engine_state" in state and state["pso_engine_state"]:
            self.pso_engine.load_state(state["pso_engine_state"])
            
        logging.info(f"Hybrid Engine state loaded. Current phase: {self.current_phase.upper()}. Cycles in phase: {self.cycles_in_current_phase}.")

    def _validate_config(self):
        """Enhanced configuration validation."""
        required_params = {
            'stagnation': ['stagnation_threshold', 'ga_min_diversity_for_switch'],
            'fixed_cycles': ['ga_phase_cycles'],
            'oscillate': ['ga_phase_cycles', 'pso_phase_cycles', 'stagnation_threshold'],
            'adaptive': ['adaptive_switching_threshold', 'min_adaptive_phase_duration']
        }
        
        switch_mode = self.hybrid_config.get('switch_mode', 'fixed_cycles')
        if switch_mode in required_params:
            for param in required_params[switch_mode]:
                if param not in self.hybrid_config:
                    raise ValueError(f"Missing required hybrid parameter for '{switch_mode}' mode: {param}")
        
        if 'ga_seed_ratio' in self.hybrid_config:
            ratio = self.hybrid_config['ga_seed_ratio']
            if not (0.0 <= ratio <= 1.0):
                raise ValueError("Hybrid 'ga_seed_ratio' must be between 0.0 and 1.0.")

        if self.switch_mode == 'adaptive':
            if self.adaptive_switching_threshold < 1.0:
                raise ValueError("Adaptive switching threshold should be >= 1.0 to ensure switching to a better strategy.")
            
            min_history_for_trend = 3
            if self.min_adaptive_phase_duration < min_history_for_trend:
                logging.warning(f"Adaptive phase duration ({self.min_adaptive_phase_duration}) "
                                f"is less than {min_history_for_trend}, which may be too short for reliable trend-based decisions.")

    def run_hybrid_algorithm(self, cycle_number: int, best_individual_so_far: Optional[List[float]], 
                                last_cycle_fitness: float, best_fitness_overall: float) -> List[float]:
            """Runs one cycle of the hybrid optimization algorithm."""
            if cycle_number == 0:
                self.fitness_at_phase_start = best_fitness_overall if best_fitness_overall > -float('inf') else 0.0

            # Update master stagnation counter based on overall fitness improvement
            if best_fitness_overall > last_cycle_fitness + self.fitness_improvement_tolerance:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            just_switched = self._check_and_perform_switch(cycle_number, best_fitness_overall)
            self.cycles_in_current_phase += 1
            
            seed_for_run = None if just_switched else best_individual_so_far

            if self.current_phase == 'ga':
                logging.info(f"Running GA Phase (Cycle {self.cycles_in_current_phase})")
                # The GA's internal stagnation counter is reset within its run method
                return self.ga_engine.run_genetic_algorithm(seed_individual=seed_for_run)
            else:
                logging.info(f"Running PSO Phase (Cycle {self.cycles_in_current_phase})")
                # Pass the master stagnation counter to the PSO engine
                self.pso_engine.stagnation_counter = self.stagnation_counter
                return self.pso_engine.run_pso_algorithm(seed_individual=seed_for_run)

    def _check_and_perform_switch(self, cycle_number: int, best_fitness_overall: float) -> bool:
            """
            Checks if the conditions to switch phases are met and performs the switch.
            Returns True if a switch was performed, False otherwise.
            """
            if cycle_number == 0:
                return False
                
            strategy_map = {
                'fixed_cycles': self._should_switch_fixed_cycles,
                'stagnation': self._should_switch_stagnation,
                'oscillate': self._should_switch_oscillate,
                'adaptive': self._should_switch_adaptive
            }
            
            switching_strategy = strategy_map.get(self.switch_mode, lambda: (False, ""))
            should_switch, switch_reason = switching_strategy()

            if should_switch:
                old_phase = self.current_phase
                
                fitness_gain = best_fitness_overall - self.fitness_at_phase_start
                self.performance_history[old_phase].append(fitness_gain)
                
                self.cycles_in_current_phase = 0
                # Reset the master stagnation counter upon a phase switch to give the new phase a fresh start
                self.stagnation_counter = 0
                
                if self.current_phase == 'ga':
                    self.current_phase = 'pso'
                    logging.info(f"--- Switching GA -> PSO --- Reason: {switch_reason}")
                    top_individuals = self.ga_engine.get_top_individuals(n=self.config['pso']['swarm_size'])
                    if hasattr(self.pso_engine, 'initialize_with_population'):
                        self.pso_engine.initialize_with_population(top_individuals)
                    else:
                        logging.warning("PSO engine is missing 'initialize_with_population' method. Cannot seed from GA.")
                else:
                    self.current_phase = 'ga'
                    logging.info(f"--- Switching PSO -> GA --- Reason: {switch_reason}")
                    self._seed_ga_from_pso()
                
                self.phase_history.append({
                    'cycle': cycle_number, 'from_phase': old_phase, 'to_phase': self.current_phase,
                    'reason': switch_reason, 'fitness_gain': fitness_gain
                })
                self.fitness_at_phase_start = best_fitness_overall
                return True # Indicate that a switch happened
            
            return False # No switch occurred

    def _should_switch_fixed_cycles(self) -> Tuple[bool, str]:
        """Logic for 'fixed_cycles' switching mode."""
        if self.current_phase == 'ga' and self.cycles_in_current_phase >= self.ga_phase_cycles:
            return True, f"GA phase completed ({self.ga_phase_cycles} cycles)"
        return False, ""

    def _should_switch_stagnation(self) -> Tuple[bool, str]:
        """Logic for 'stagnation' switching mode, relying on sub-engine counters."""
        if self.current_phase == 'ga':
            # Check the GA's internal stagnation counter, which tracks generations without improvement
            ga_stagnant = self.ga_engine.stagnation_counter >= self.config['ga']['stagnation_threshold']
            diversity = calculate_diversity(self.ga_engine.population, self.config['ga']['diversity_sample_size'])
            diversity_stagnant = diversity < self.ga_min_diversity_for_switch
            
            if ga_stagnant and diversity_stagnant:
                reason = f"GA stagnation detected ({self.ga_engine.stagnation_counter} gens) with low diversity ({diversity:.3f})."
                return True, reason
        elif self.current_phase == 'pso':
            # Check the PSO's internal counter, which tracks iterations without gbest improvement
            if self.pso_engine.stagnation_counter >= self.stagnation_threshold:
                 return True, f"PSO stagnation detected ({self.pso_engine.stagnation_counter} iterations)."

        return False, ""

    def _should_switch_oscillate(self) -> Tuple[bool, str]:
        """Logic for 'oscillate' switching mode."""
        # Use the master stagnation counter for overall lack of progress
        if self.stagnation_counter >= self.stagnation_threshold:
            return True, f"Overall stagnation detected ({self.stagnation_counter} cycles)"
        if self.current_phase == 'ga' and self.cycles_in_current_phase >= self.ga_phase_cycles:
            return True, f"GA phase completed ({self.ga_phase_cycles} cycles)"
        if self.current_phase == 'pso' and self.cycles_in_current_phase >= self.pso_phase_cycles:
            return True, f"PSO phase completed ({self.pso_phase_cycles} cycles)"
        return False, ""

    def _get_adaptive_phase_duration(self) -> int:
        """Dynamically adjust phase duration based on performance."""
        base_duration = self.min_adaptive_phase_duration
        
        if self.current_phase == 'ga':
            diversity = calculate_diversity(self.ga_engine.population, 50)
            if diversity > 0.4 and self.ga_engine.stagnation_counter < self.config['ga']['stagnation_threshold'] / 2:
                logging.debug("Extending GA phase due to high diversity and good progress.")
                return base_duration + 3
        
        if self.current_phase == 'pso':
            if self.pso_engine.stagnation_counter < self.stagnation_threshold / 2:
                logging.debug("Extending PSO phase due to strong convergence.")
                return base_duration + 2
        
        return base_duration
        
    def _should_switch_adaptive(self) -> Tuple[bool, str]:
        """IMPROVED: Logic for 'adaptive' switching mode based on performance and trend."""
        other_phase = 'pso' if self.current_phase == 'ga' else 'ga'
        min_points_for_decision = 3

        # Fallback: If not enough history, switch on stagnation to gather data
        if len(self.performance_history[other_phase]) < min_points_for_decision:
            if self.stagnation_counter >= self.stagnation_threshold:
                return True, f"Stagnation detected ({self.stagnation_counter} cycles). Switching to gather {other_phase.upper()} data."

        min_duration = self._get_adaptive_phase_duration()
        if self.cycles_in_current_phase < min_duration:
            return False, ""

        current_perf_hist = self.performance_history[self.current_phase]
        other_perf_hist = self.performance_history[other_phase]
        
        if len(current_perf_hist) >= min_points_for_decision and len(other_perf_hist) >= min_points_for_decision:
            analysis_window = 5
            current_avg_gain = np.mean(current_perf_hist[-analysis_window:])
            other_avg_gain = np.mean(other_perf_hist[-analysis_window:])
            
            current_trend = self._calculate_trend(current_perf_hist[-analysis_window:])
            other_trend = self._calculate_trend(other_perf_hist[-analysis_window:])

            is_other_better_perf = other_avg_gain > current_avg_gain * self.adaptive_switching_threshold
            
            if is_other_better_perf:
                dampening_factor = self.hybrid_config.get('adaptive_trend_dampening_factor', 0.5)
                is_trend_compelling = (other_trend > 0) or (current_trend < 0 and other_trend > current_trend * dampening_factor)

                if is_trend_compelling:
                    reason = (f"Adaptive: {other_phase.upper()} is performing better (Gain: {other_avg_gain:.2f} vs {current_avg_gain:.2f}) "
                            f"and has a more promising trend (Slope: {other_trend:.3f} vs {current_trend:.3f})")
                    return True, reason

        return False, ""

    def _calculate_trend(self, history: List[float]) -> float:
        """Calculates the performance trend using the slope of a linear regression line."""
        if len(history) < 2:
            return 0.0
        return np.polyfit(range(len(history)), history, 1)[0]
    
    def _is_sufficiently_diverse(self, candidate: List[float], selected_seeds: List[List[float]]) -> bool:
        """Checks if a candidate is diverse enough from already selected seeds."""
        if not selected_seeds:
            return True
        
        candidate_arr = np.array(candidate)
        total_distance = 0
        for seed in selected_seeds:
            total_distance += np.mean(np.abs(candidate_arr - np.array(seed)))
        
        avg_distance = total_distance / len(selected_seeds)
        return avg_distance > self.seeding_diversity_threshold
        
    def _seed_ga_from_pso(self):
        """IMPROVED: Safely seeds the GA population from PSO results, ensuring target seed count."""
        try:
            if not self.pso_engine.personal_best_positions:
                raise ValueError("PSO personal best positions are not available.")

            sorted_indices = np.argsort(self.pso_engine.personal_best_fitnesses)[::-1]
            num_seeds_target = int(self.config['ga']['population_size'] * self.ga_seed_ratio)
            
            diverse_seeds = []
            other_good_seeds = []

            # First, iterate through all high-performing candidates from PSO
            for idx in sorted_indices:
                candidate = self.pso_engine.personal_best_positions[idx][:]
                # Try to add to the diverse list first
                if len(diverse_seeds) < num_seeds_target and self._is_sufficiently_diverse(candidate, diverse_seeds):
                    diverse_seeds.append(candidate)
                else:
                    other_good_seeds.append(candidate)
            
            # Combine the lists, prioritizing diverse seeds
            seeded_population = diverse_seeds
            
            # If we still need more seeds, take the best-of-the-rest (which are not diverse)
            remaining_needed = num_seeds_target - len(seeded_population)
            if remaining_needed > 0:
                seeded_population.extend(other_good_seeds[:remaining_needed])
            
            self.ga_engine.population = seeded_population
            
            # Fill the rest of the population with random individuals
            while len(self.ga_engine.population) < self.config['ga']['population_size']:
                self.ga_engine.population.append(self.ga_engine._create_random_individual())
            
            logging.info(f"Seeded GA with {len(seeded_population)} individuals from PSO "
                         f"({len(diverse_seeds)} diverse, {len(seeded_population) - len(diverse_seeds)} best-fitness).")
        
        except Exception as e:
            logging.error(f"Failed to seed GA from PSO: {e}. Falling back to random initialization.")
            self.ga_engine.population = [self.ga_engine._create_random_individual() for _ in range(self.config['ga']['population_size'])]       

    def get_phase_statistics(self) -> Dict[str, any]:
        """Returns statistics about phase transitions and performance."""
        stats = {
            'current_phase': self.current_phase,
            'cycles_in_current_phase': self.cycles_in_current_phase,
            'stagnation_counter': self.stagnation_counter,
            'total_phase_switches': len(self.phase_history),
            'avg_fitness_gain_ga': np.mean(self.performance_history['ga']) if self.performance_history['ga'] else 0.0,
            'avg_fitness_gain_pso': np.mean(self.performance_history['pso']) if self.performance_history['pso'] else 0.0,
            'phase_history': self.phase_history[-10:]
        }
        return stats