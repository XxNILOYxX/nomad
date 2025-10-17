import random
import numpy as np
import logging
import functools
from typing import List, Tuple, Dict, Callable

from .interpolators import KeffInterpolator, PPFInterpolator
from .utils import calculate_diversity, fitness_function, find_nearest

class GeneticAlgorithm:
    """
    Implements the core logic of the genetic algorithm, including
    adaptive parameters and advanced selection/mutation operators.
    """
    def __init__(self, config: Dict, keff_interpolator: KeffInterpolator, ppf_interpolator: PPFInterpolator):
        self.ga_config = config['ga']
        self.sim_config = config['simulation']
        self.enrich_config = config['enrichment']
        self.fitness_config = config['fitness']

        self.keff_interpolator = keff_interpolator
        self.ppf_interpolator = ppf_interpolator

        self.population = []
        self.fitnesses = []
        self.best_fitness_history = []
        self.stagnation_counter = 0

        # Dynamically select crossover function based on config
        crossover_methods = {
            'single_point': self._single_point_crossover,
            'zone_based': self._zone_based_crossover,
            'blend': self._blend_crossover
        }
        method_name = self.ga_config.get('crossover_method', 'single_point')
        base_crossover_func = crossover_methods.get(method_name)

        if not base_crossover_func:
            raise ValueError(f"Invalid crossover method '{method_name}' specified in config.ini")

        # If the chosen method is 'blend', create a new function with the alpha value already included.
        if method_name == 'blend':
            alpha = self.ga_config.get('blend_crossover_alpha', 0.5)
            self.crossover_method_func: Callable = functools.partial(base_crossover_func, alpha=alpha)
            logging.info(f"GA initialized with '{method_name}' crossover (alpha={alpha}).")
        else:
            self.crossover_method_func: Callable = base_crossover_func
            logging.info(f"GA initialized with '{method_name}' crossover method.")

        self._initialize_population()


    def get_state(self) -> Dict:
            """
            Returns the current internal state of the GA for checkpointing.
            """
            return {
                "population": self.population,
                "fitnesses": self.fitnesses,
                "best_fitness_history": self.best_fitness_history,
                "stagnation_counter": self.stagnation_counter,
            }

    def load_state(self, state: Dict):
        """
        Loads the internal state of the GA from a checkpoint dictionary.
        """
        self.population = state.get("population", [])
        self.fitnesses = state.get("fitnesses", [])
        self.best_fitness_history = state.get("best_fitness_history", [])
        self.stagnation_counter = state.get("stagnation_counter", 0)
        
        if not self.population:
            logging.warning("GA state from checkpoint was empty. Re-initializing a random population.")
            self._initialize_population()
        else:
            logging.info(f"GA state loaded. Population size: {len(self.population)}. Stagnation counter: {self.stagnation_counter}.")


    def get_top_individuals(self, n: int) -> List[List[float]]:
        """
        Returns the top N individuals from the current population based on fitness.

        Args:
            n (int): The number of top individuals to return.

        Returns:
            A list of the N best individuals.
        """
        if not self.population or not self.fitnesses:
            return []
        
        # Get indices of the N best individuals
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        top_n_indices = sorted_indices[:n]
        
        return [self.population[i] for i in top_n_indices]
    

    def run_genetic_algorithm(self, seed_individual: List[float] = None) -> List[float]:
        """
        Runs one full GA cycle on the existing, persistent population.
        """
        if seed_individual and seed_individual not in self.population:
            # If a new best individual from OpenMC is provided, inject it into the
            # population by replacing the worst individual. This preserves genetic diversity.
            if self.fitnesses:
                worst_idx = np.argmin(self.fitnesses)
                self.population[worst_idx] = seed_individual
                # The fitness for the new individual will be calculated in the next step,
                # so explicitly setting it to -inf is redundant.
            else:
                # This case handles the very first run if the population is empty
                self.population.append(seed_individual)

        
        # Full evaluation of the persistent population
        # This re-evaluates the entire population with the latest ML models.
        keff_preds, ppf_preds = self._evaluate_individuals(self.population)

        self.fitnesses = [fitness_function(k, p, self.sim_config, self.fitness_config) for k, p in zip(keff_preds, ppf_preds)]

        best_individual_cycle = self.population[0]
        best_fitness_cycle = -float('inf')
        
        convergence_counter = 0
        convergence_threshold = self.ga_config.get('convergence_threshold', 200)
        
        current_mutation_rate = self.ga_config['mutation_rate']
        current_crossover_rate = self.ga_config['crossover_rate']
        
        for gen in range(self.ga_config['generations_per_openmc_cycle']):
            # Check for improvement and log (using previous generation's complete fitness list)
            current_best_idx = np.argmax(self.fitnesses)
            if self.fitnesses[current_best_idx] > best_fitness_cycle:
                best_fitness_cycle = self.fitnesses[current_best_idx]
                best_individual_cycle = self.population[current_best_idx]
                self.stagnation_counter = 0
                convergence_counter = 0 
            else:
                self.stagnation_counter += 1
                convergence_counter += 1

            log_freq = self.ga_config['log_frequency']
            div_check_multiplier = self.ga_config.get('diversity_check_multiplier', 5)
            if (gen + 1) % log_freq == 0:
                if (gen + 1) % (log_freq * div_check_multiplier) == 0:
                    diversity = calculate_diversity(self.population, self.ga_config['diversity_sample_size'])
                    current_mutation_rate, current_crossover_rate = self._get_adaptive_parameters(
                        diversity, current_mutation_rate, current_crossover_rate
                    )
                    msg = f"Gen {gen+1:4d}: Best Fitness={best_fitness_cycle:.6f}, Mut. Rate={current_mutation_rate:.3f}, Diversity={diversity:.4f}"
                else:
                    msg = f"Gen {gen+1:4d}: Best Fitness={best_fitness_cycle:.6f}, Mut. Rate={current_mutation_rate:.3f}"
                
                logging.info(msg)

            if convergence_counter > convergence_threshold:
                logging.info(f"Convergence reached after {gen+1} generations. Best predicted fitness has not improved in {convergence_threshold} generations. Exiting GA cycle.")
                break

            # Create the next generation
            elite_count = self.ga_config['elitism_count']
            
            # 1. Preserve the elites from the previous generation
            elite_indices = np.argsort(self.fitnesses)[-elite_count:]
            
            next_gen_population = [self.population[i] for i in elite_indices]
            next_gen_keff_preds = [keff_preds[i] for i in elite_indices]
            next_gen_ppf_preds = [ppf_preds[i] for i in elite_indices]
            next_gen_fitnesses = [self.fitnesses[i] for i in elite_indices]

            # 2. Generate the offspring (the rest of the population)
            num_offspring = self.ga_config['population_size'] - elite_count
            offspring_population = self._create_offspring(num_offspring, keff_preds, current_mutation_rate, current_crossover_rate)
            
            # 3. Evaluate ONLY the new offspring
            if offspring_population:
                offspring_keff_preds, offspring_ppf_preds = self._evaluate_individuals(offspring_population)
                offspring_fitnesses = [fitness_function(k, p, self.sim_config, self.fitness_config) for k, p in zip(offspring_keff_preds, offspring_ppf_preds)]
                
                # 4. Combine elites and offspring into the new generation's complete lists
                next_gen_population.extend(offspring_population)
                next_gen_keff_preds.extend(offspring_keff_preds)
                next_gen_ppf_preds.extend(offspring_ppf_preds)
                next_gen_fitnesses.extend(offspring_fitnesses)

            # 5. Update the main population state for the next iteration
            self.population = next_gen_population
            self.fitnesses = next_gen_fitnesses
            keff_preds = next_gen_keff_preds
            ppf_preds = next_gen_ppf_preds

        return best_individual_cycle
    
    def _initialize_population(self, seed_individual: List[float] = None):
        """Creates the initial, random population for the GA."""
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

    def _evaluate_individuals(self, individuals: List[List[float]]) -> Tuple[List[float], List[float]]:
        """A helper to evaluate a list of individuals, returning predictions."""
        if not individuals:
            return [], []
        keff_preds = self.keff_interpolator.predict_batch(individuals)
        ppf_preds = self.ppf_interpolator.predict_batch(individuals)
        return keff_preds, ppf_preds

    def _create_offspring(self, num_offspring: int, parent_keff_preds: List[float], mut_rate: float, cross_rate: float) -> List[List[float]]:
        """Creates new individuals through selection, crossover, and mutation."""
        offspring = []
        while len(offspring) < num_offspring:
            parent1, p1_idx = self._selection()
            parent2, p2_idx = self._selection()

            child1, child2 = self.crossover_method_func(parent1, parent2, cross_rate)
            
            keff1 = parent_keff_preds[p1_idx]
            keff2 = parent_keff_preds[p2_idx]

            offspring.append(self._smart_mutate(child1, mut_rate, keff1))
            if len(offspring) < num_offspring:
                offspring.append(self._smart_mutate(child2, mut_rate, keff2))
        return offspring

    def _selection(self) -> Tuple[List[float], int]:
        """
        Tournament selection with diversity-based tie-breaking.
        """
        tournament_size = self.ga_config['tournament_size']
        if tournament_size > len(self.population):
            tournament_size = len(self.population)

        competitor_indices = random.sample(range(len(self.population)), tournament_size)
        
        max_fitness = -float('inf')
        for idx in competitor_indices:
            if self.fitnesses[idx] > max_fitness:
                max_fitness = self.fitnesses[idx]
                
        winners_indices = [idx for idx in competitor_indices if self.fitnesses[idx] == max_fitness]
        
        if len(winners_indices) == 1:
            winner_idx = winners_indices[0]
            return self.population[winner_idx], winner_idx
        else:
            # Tie-breaking: select the winner that is most diverse from the *other winners*.
            best_winner_idx = -1
            max_diversity_score = -1.0
            
            # Get the genomes of the tied winners
            winner_genomes = [np.array(self.population[i]) for i in winners_indices]

            for i, winner_idx in enumerate(winners_indices):
                current_winner_genome = winner_genomes[i]
                # Calculate distance only to other tied winners
                other_winner_genomes = winner_genomes[:i] + winner_genomes[i+1:]
                if not other_winner_genomes: # Should not happen if len > 1, but safe
                    best_winner_idx = winner_idx
                    break

                total_distance = sum(np.mean(np.abs(current_winner_genome - other_genome)) for other_genome in other_winner_genomes)
                
                if total_distance > max_diversity_score:
                    max_diversity_score = total_distance
                    best_winner_idx = winner_idx
            
            return self.population[best_winner_idx], best_winner_idx

    def _single_point_crossover(self, parent1: List[float], parent2: List[float], cross_rate: float) -> Tuple[List[float], List[float]]:
        """Single-point crossover."""
        if random.random() < cross_rate:
            point = random.randint(1, len(parent1) - 2)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1[:], parent2[:]

    def _zone_based_crossover(self, parent1: List[float], parent2: List[float], cross_rate: float) -> Tuple[List[float], List[float]]:
        """Performs crossover by swapping the entire central and outer enrichment zones."""
        if random.random() < cross_rate:
            num_central = self.sim_config['num_central_assemblies']
            child1 = parent1[:num_central] + parent2[num_central:]
            child2 = parent2[:num_central] + parent1[num_central:]
            return child1, child2
        return parent1[:], parent2[:]

    def _blend_crossover(self, parent1: List[float], parent2: List[float], cross_rate: float, alpha: float) -> Tuple[List[float], List[float]]:
        """Performs blend crossover (BLX-alpha) and snaps to the nearest valid enrichment."""
        if random.random() < cross_rate:
            child1, child2 = [], []
            num_central = self.sim_config['num_central_assemblies']
            central_vals = self.enrich_config['central_values']
            outer_vals = self.enrich_config['outer_values']

            for i in range(len(parent1)):
                gene1, gene2 = parent1[i], parent2[i]
                
                diff = abs(gene1 - gene2)
                min_gene = min(gene1, gene2) - alpha * diff
                max_gene = max(gene1, gene2) + alpha * diff
                
                new_gene1 = random.uniform(min_gene, max_gene)
                new_gene2 = random.uniform(min_gene, max_gene)
                
                valid_values = central_vals if i < num_central else outer_vals
                child1.append(find_nearest(valid_values, new_gene1))
                child2.append(find_nearest(valid_values, new_gene2))
                
            return child1, child2
        return parent1[:], parent2[:]

    def _smart_mutate(self, individual: List[float], mut_rate: float, current_keff: float) -> List[float]:
            """Enhanced mutation with better edge case handling and adaptive bias."""
            mutated = list(individual)
            keff_diff = current_keff - self.sim_config['target_keff']
            
            # Dynamic bias based on distance from target
            if abs(keff_diff) > self.fitness_config['high_keff_diff_threshold']:
                rate_multiplier = self.ga_config['smart_mutate_increase_factor']
            elif abs(keff_diff) > self.fitness_config['med_keff_diff_threshold']:
                rate_multiplier = 1.0
            else:
                rate_multiplier = self.ga_config['smart_mutate_decrease_factor']
            
            # Calculate the effective mutation rate for this individual, but cap it
            # to prevent excessively disruptive mutations.
            effective_mut_rate = min(mut_rate * rate_multiplier, self.ga_config['max_mutation_rate'])

            for i in range(len(mutated)):
                if random.random() < effective_mut_rate:
                    is_central = i < self.sim_config['num_central_assemblies']
                    values = self.enrich_config['central_values'] if is_central else self.enrich_config['outer_values']
                    current_val = mutated[i]
                    
                    # Determine mutation direction based on keff_diff
                    if abs(keff_diff) > 0.0005:  # Only apply bias if significantly off target
                        if keff_diff < 0:  # Need to increase reactivity
                            available = [v for v in values if v > current_val]
                            if not available:
                                continue # No higher values available, skip mutation
                                
                        else:  # keff_diff > 0, need to decrease reactivity
                            available = [v for v in values if v < current_val]
                            if not available:
                                continue # No lower values available, skip mutation
                    else:
                        # Near target, use uniform random mutation
                        available = [v for v in values if v != current_val]
                        if not available:
                            continue  # Skip if only one possible value
                    
                    # Weighted selection favoring smaller changes when close to target
                    if len(available) > 1 and abs(keff_diff) < self.fitness_config['med_keff_diff_threshold']:
                        # Calculate distances from current value
                        distances = [abs(v - current_val) for v in available]
                        max_dist = max(distances) if distances else 1
                        # Invert weights so smaller distances have higher probability
                        weights = [(max_dist - d + 0.1) for d in distances]
                        mutated[i] = random.choices(available, weights=weights, k=1)[0]
                    elif available:
                        mutated[i] = random.choice(available)
            
            return mutated

    def _get_adaptive_parameters(self, diversity: float, current_mut_rate: float, current_cross_rate: float) -> Tuple[float, float]:
        """Adjusts mutation and crossover rates based on stagnation and diversity."""
        mut_rate = current_mut_rate
        cross_rate = current_cross_rate

        if self.stagnation_counter > self.ga_config['stagnation_threshold']:
            mut_rate = min(mut_rate + 0.01, self.ga_config['max_mutation_rate'])

        if diversity < self.ga_config['diversity_threshold']:
            mut_rate = min(mut_rate + 0.01, self.ga_config['max_mutation_rate'])
            cross_rate = min(cross_rate + 0.01, self.ga_config['max_crossover_rate'])

        return mut_rate, cross_rate
