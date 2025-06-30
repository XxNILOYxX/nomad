import random
import numpy as np
import logging
from typing import List, Tuple, Dict
from .interpolators import KeffInterpolator, PPFInterpolator
from .utils import calculate_diversity

class GeneticAlgorithm:
    """
    Implements the core logic of the genetic algorithm, including
    adaptive parameters and advanced selection/mutation operators.
    """
    def __init__(self, config: Dict, keff_interpolator: KeffInterpolator, ppf_interpolator: PPFInterpolator):
        self.ga_config = config['ga']
        self.sim_config = config['simulation']
        self.enrich_config = config['enrichment']
        self.true_fitness_config = config['true_fitness']

        self.keff_interpolator = keff_interpolator
        self.ppf_interpolator = ppf_interpolator

        self.population = []
        self.fitnesses = []
        self.best_fitness_history = []
        self.stagnation_counter = 0

    def run_genetic_algorithm(self, seed_individual: List[float] = None) -> List[float]:
        """
        Runs one full GA cycle (multiple generations).

        Args:
            seed_individual: An optional individual to seed the initial population.

        Returns:
            The best individual found during the cycle based on interpolator predictions.
        """
        self._initialize_population(seed_individual)
        
        best_individual_cycle = self.population[0]
        best_fitness_cycle = -float('inf')
        
        current_mutation_rate = self.ga_config['mutation_rate']
        current_crossover_rate = self.ga_config['crossover_rate']
        
        for gen in range(self.ga_config['generations_per_openmc_cycle']):
            keff_preds, ppf_preds = self._evaluate_population()

            # Update best of cycle
            current_best_idx = np.argmax(self.fitnesses)
            if self.fitnesses[current_best_idx] > best_fitness_cycle:
                best_fitness_cycle = self.fitnesses[current_best_idx]
                best_individual_cycle = self.population[current_best_idx]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            if (gen + 1) % self.ga_config['log_frequency'] == 0:
                diversity = calculate_diversity(self.population, self.ga_config['diversity_sample_size'])
                current_mutation_rate, current_crossover_rate = self._get_adaptive_parameters(
                    diversity, current_mutation_rate, current_crossover_rate
                )
                logging.info(f"Gen {gen+1:4d}: Best Fitness={best_fitness_cycle:.6f}, Diversity={diversity:.4f}, Mut. Rate={current_mutation_rate:.3f}")

            self._generate_new_population(keff_preds, current_mutation_rate, current_crossover_rate)

        return best_individual_cycle

    def _initialize_population(self, seed_individual: List[float] = None):
        """Creates the initial population for the GA."""
        self.population = []
        if seed_individual:
            self.population.append(seed_individual)
            
        while len(self.population) < self.ga_config['population_size']:
            self.population.append(self._create_random_individual())
    
    def _create_random_individual(self) -> List[float]:
        """Generates a single random individual."""
        num_central = self.sim_config['num_central_assemblies']
        num_total = self.sim_config['num_assemblies']
        
        central_vals = self.enrich_config['central_values']
        outer_vals = self.enrich_config['outer_values']
        
        individual = [random.choice(central_vals) for _ in range(num_central)]
        individual.extend([random.choice(outer_vals) for _ in range(num_total - num_central)])
        return individual

    def _evaluate_population(self) -> Tuple[List[float], List[float]]:
        """Calculates fitness for the entire population using interpolators."""
        keff_preds = self.keff_interpolator.predict_batch(self.population)
        ppf_preds = self.ppf_interpolator.predict_batch(self.population)
        self.fitnesses = [self.fitness_function(k, p) for k, p in zip(keff_preds, ppf_preds)]
        return keff_preds, ppf_preds

    
    def fitness_function(self, keff: float, ppf: float) -> float:
        """Calculates a balanced fitness score based on keff and PPF. Used in the GA LOOP"""
        target_keff = self.sim_config['target_keff']
        keff_tolerance = self.sim_config['keff_tolerance']
        
        keff_diff = abs(keff - target_keff)
        
        # Invert PPF so higher is better
        ppf_score = 1.0 / ppf if ppf > 0 else 0

        # Keff score: 1 if within tolerance, exponentially decaying penalty otherwise
        if keff_diff <= keff_tolerance:
            keff_score = 1.0
        else:
            penalty_factor = self.ga_config['keff_penalty_factor']
            keff_score = np.exp(-penalty_factor * (keff_diff - keff_tolerance))

        high_thresh = self.ga_config['high_keff_diff_threshold']
        med_thresh = self.ga_config['med_keff_diff_threshold']
        
        # Dynamically adjust weights based on how far we are from the target keff
        if keff_diff > high_thresh:
            w_ppf, w_keff = (0.3, 0.7) # Prioritize getting keff right
        elif keff_diff > med_thresh:
            w_ppf, w_keff = (0.5, 0.5) # Balanced approach
        else:
            w_ppf, w_keff = (0.7, 0.3) # Prioritize minimizing PPF

        return w_ppf * ppf_score + w_keff * keff_score

    def _generate_new_population(self, keff_preds: List[float], mut_rate: float, cross_rate: float):
        """Creates the next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(self.fitnesses)[-self.ga_config['elitism_count']:]
        for idx in elite_indices:
            new_population.append(self.population[idx])

        # Generate the rest of the population
        while len(new_population) < self.ga_config['population_size']:
            parent1 = self._selection()
            parent2 = self.population[np.random.choice(len(self.population))] # One tournament, one random parent
            
            child1, child2 = self._crossover(parent1, parent2, cross_rate)
            
            # Get the predicted keff for the parent to guide mutation
            parent1_idx = self.population.index(parent1)
            parent2_idx = self.population.index(parent2)
            
            new_population.append(self._smart_mutate(child1, mut_rate, keff_preds[parent1_idx]))
            if len(new_population) < self.ga_config['population_size']:
                new_population.append(self._smart_mutate(child2, mut_rate, keff_preds[parent2_idx]))
        
        self.population = new_population

    def _selection(self) -> List[float]:
        """Tournament selection."""
        competitors = random.sample(list(enumerate(self.fitnesses)), self.ga_config['tournament_size'])
        winner_idx, _ = max(competitors, key=lambda item: item[1])
        return self.population[winner_idx]

    def _crossover(self, parent1: List[float], parent2: List[float], cross_rate: float) -> Tuple[List[float], List[float]]:
        """Single-point crossover."""
        if random.random() < cross_rate:
            point = random.randint(1, len(parent1) - 2)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1[:], parent2[:]

    def _smart_mutate(self, individual: List[float], mut_rate: float, current_keff: float) -> List[float]:
        """Mutates an individual with a bias towards improving k-effective."""
        mutated = list(individual)
        keff_diff = current_keff - self.sim_config['target_keff']
        
        increase_factor = self.ga_config['smart_mutate_increase_factor']
        decrease_factor = self.ga_config['smart_mutate_decrease_factor']
        threshold = self.ga_config['med_keff_diff_threshold']

        # Adjust mutation intensity based on how far keff is from target
        rate_multiplier = increase_factor if abs(keff_diff) > threshold else decrease_factor
        
        for i in range(len(mutated)):
            if random.random() < (mut_rate * rate_multiplier):
                values = self.enrich_config['central_values'] if i < self.sim_config['num_central_assemblies'] else self.enrich_config['outer_values']
                
                # Nudge mutation in the right direction
                # Using a hardcoded 0.01 here as it's a small nudge value, not a major threshold
                if keff_diff < -0.01: # Keff is too low, need higher enrichment
                    available = [v for v in values if v >= mutated[i]] or values
                elif keff_diff > 0.01: # Keff is too high, need lower enrichment
                    available = [v for v in values if v <= mutated[i]] or values
                else: # Keff is close, any change is fine
                    available = values
                
                mutated[i] = random.choice(available)
        return mutated

    def _get_adaptive_parameters(self, diversity: float, current_mut_rate: float, current_cross_rate: float) -> Tuple[float, float]:
        """Adjusts mutation and crossover rates based on stagnation and diversity."""
        # Start with the current rates, not the base rates from the config file.
        mut_rate = current_mut_rate
        cross_rate = current_cross_rate

        # Increase mutation if stagnated
        if self.stagnation_counter > self.ga_config['stagnation_threshold']:
            # Increment the rate by 0.01 instead of multiplying by a factor.
            mut_rate = min(mut_rate + 0.01, self.ga_config['max_mutation_rate'])

        # Increase mutation and crossover if diversity is low
        if diversity < self.ga_config['diversity_threshold']:
            # Increment the rate by 0.01 instead of multiplying by a factor.
            mut_rate = min(mut_rate + 0.01, self.ga_config['max_mutation_rate'])
            cross_rate = min(cross_rate + 0.01, self.ga_config['max_crossover_rate']) # Make crossover incremental too

        return mut_rate, cross_rate
