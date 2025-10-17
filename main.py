import logging
import time
import os
import numpy as np
import datetime 
from typing import Dict, Any, List

from scipy.stats import qmc

from core.config_loader import ConfigLoader
from core.utils import setup_logging, detect_hardware, show_splash_screen, find_nearest, fitness_function
from core.fuel_handler import FuelHandler
from core.openmc_runner import OpenMCRunner
from core.interpolators import KeffInterpolator, PPFInterpolator
from core.ga_engine import GeneticAlgorithm
from core.pso_engine import ParticleSwarmOptimizer
from core.hybrid_engine import HybridEngine
from core.checkpoint import Checkpoint

class MainOptimizer:
    """
    Orchestrates the entire optimization process for nuclear reactor fuel loading patterns,
    supporting multiple optimization techniques like GA and PSO.
    """
    def __init__(self, config_path: str = 'config/config.ini', fuel_setup_path: str = 'config/setup_fuel.ini'):
        """Initializes all components of the optimizer."""
        show_splash_screen() 
        setup_logging()

        try:
            self.config_loader = ConfigLoader(config_path, fuel_setup_path)
            self.config = self.config_loader.get_params()
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Failed to initialize configuration: {e}")
            raise SystemExit("Exiting due to configuration error.") from e

        use_gpu = detect_hardware(self.config['hardware'])
        self.fuel_handler = FuelHandler(self.config)
        self.openmc_runner = OpenMCRunner(self.config)
        self.keff_interpolator = KeffInterpolator(self.config, use_gpu)
        self.ppf_interpolator = PPFInterpolator(self.config, use_gpu)
        self.checkpoint = Checkpoint(self.config)

        self.optimizer_technique = self.config['optimizer']['technique']
        self.optimizer_engine = None
        logging.info(f"Using optimization technique: {self.optimizer_technique.upper()}")

        if self.optimizer_technique == 'ga':
            self.optimizer_engine = GeneticAlgorithm(self.config, self.keff_interpolator, self.ppf_interpolator)
        elif self.optimizer_technique == 'pso':
            self.optimizer_engine = ParticleSwarmOptimizer(self.config, self.keff_interpolator, self.ppf_interpolator)
        elif self.optimizer_technique == 'hybrid':
            self.optimizer_engine = HybridEngine(self.config, self.keff_interpolator, self.ppf_interpolator)
        else:
            raise ValueError(f"Unknown optimizer technique specified: {self.optimizer_technique}")

        self.state = self._load_or_initialize_state()
        

    def _load_or_initialize_state(self) -> Dict[str, Any]:
        """
        Loads state from checkpoint. If no valid state exists,
        it triggers the resumable initialization phase for the interpolators.
        """
        loaded_state = self.checkpoint.load()

        if loaded_state and loaded_state.get('cycle_number', 0) > 0:
            loaded_state.setdefault('cycle_durations', [])
            loaded_state.setdefault('estimated_remaining_time', 'Calculating...')
            # Load data for both interpolators
            keff_loaded = self.keff_interpolator.load_data()
            ppf_loaded = self.ppf_interpolator.load_data()
            
            optimizer_state_loaded = False
            if 'optimizer_state' in loaded_state and loaded_state['optimizer_state']:
                try:
                    if hasattr(self.optimizer_engine, 'load_state'):
                        self.optimizer_engine.load_state(loaded_state['optimizer_state'])
                        optimizer_state_loaded = True
                    else:
                         logging.error(f"Optimizer {self.optimizer_technique.upper()} does not support loading state. This is a critical error.")
                except Exception as e:
                    logging.error(f"Failed to load optimizer state from checkpoint: {e}. The search will restart from a random population.")

            if keff_loaded and ppf_loaded and optimizer_state_loaded:
                logging.info(f"Successfully resumed {self.optimizer_technique.upper()} from cycle {loaded_state['cycle_number']}.")
                return loaded_state
            else:
                logging.error("Checkpoint exists but interpolator data or optimizer state is missing/corrupt. Re-initializing.")
                if os.path.exists(self.checkpoint.filepath):
                    os.remove(self.checkpoint.filepath)
    
        logging.info("No valid checkpoint found or resume failed. Entering data initialization phase.")
        self._initialize_interpolator_data()
        
        logging.info("Initialization complete. Preparing for main optimization cycles.")
        return {
            "cycle_number": 0,
            "best_individual_overall": None,
            "best_true_fitness": -float('inf'),
            "best_true_keff": 0.0,
            "best_true_ppf": float('inf'),
            "history": [],
            "cycle_durations": [],
            "estimated_remaining_time": "Calculating...",
            "optimizer_state": None,
        }

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into a human-readable string like Xd Yh Zm Ws."""
        if seconds <= 0:
            return "N/A"
        td = datetime.timedelta(seconds=int(seconds))
        days = td.days
        hours, rem = divmod(td.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
            
        return " ".join(parts)
            
    def _initialize_interpolator_data(self):
        """
        Runs initial OpenMC simulations to build a baseline dataset.
        This process is resumable and provides time estimates.
        """
        self.keff_interpolator.load_data()
        self.ppf_interpolator.load_data()
        # For PPF, check against the live dataset for completion status
        num_completed_samples = len(self.ppf_interpolator.live_features)

        sample_configurations = self.config['enrichment'].get('initial_configs', [])

        if not sample_configurations:
            logging.info("'initial_configs' not found. Generating diverse samples using Latin Hypercube Sampling.")
            num_samples = self.config['enrichment']['initial_samples']
            if num_samples == 0:
                logging.warning("initial_samples is 0. Skipping data generation.")
                return

            central_vals = self.config['enrichment']['central_values']
            outer_vals = self.config['enrichment']['outer_values']
            
            l_bounds = [min(central_vals), min(outer_vals)]
            u_bounds = [max(central_vals), max(outer_vals)]
            
            sampler = qmc.LatinHypercube(d=2)
            unit_samples = sampler.random(n=num_samples)
            scaled_samples = qmc.scale(unit_samples, l_bounds, u_bounds)
            
            temp_configs = set()
            for sample in scaled_samples:
                central_enrich = find_nearest(central_vals, sample[0])
                outer_enrich = find_nearest(outer_vals, sample[1])
                temp_configs.add((central_enrich, outer_enrich))
            
            sample_configurations = list(temp_configs)
            
            while len(sample_configurations) < num_samples:
                c_val = np.random.choice(central_vals)
                o_val = np.random.choice(outer_vals)
                if (c_val, o_val) not in sample_configurations:
                    sample_configurations.append((c_val, o_val))

        num_total_samples = len(sample_configurations)
        if num_total_samples == 0:
            logging.warning("No initial samples to generate. Skipping initialization.")
            return

        if num_completed_samples >= num_total_samples:
            logging.info(f"Initial dataset is already complete with {num_completed_samples} samples. Skipping initialization.")
            return

        logging.info(f"Resuming initial data generation. Target: {num_total_samples} samples. Completed: {num_completed_samples}.")
        
        num_central = self.config['simulation']['num_central_assemblies']
        num_total_assemblies = self.config['simulation']['num_assemblies']
        
        initial_run_durations = []
        
        for i in range(num_completed_samples, num_total_samples):
            sample_start_time = time.time()
            
            c_val, o_val = sample_configurations[i]
            individual = [c_val] * num_central + [o_val] * (num_total_assemblies - num_central)
            
            logging.info(f"--- Running Initial Sample {i + 1}/{num_total_samples} ---")
            logging.info(f"Configuration: Central Enrichment={c_val:.3f}, Outer Enrichment={o_val:.3f}")
            
            if not self.fuel_handler.update_materials(individual):
                logging.error(f"Failed to update materials for sample {i + 1}. Aborting initialization.")
                return 

            if self.openmc_runner.run_simulation():
                results = self.openmc_runner.extract_results()
                if results:
                    keff, ppf = results
                    self.keff_interpolator.add_data_point(individual, keff)
                    self.ppf_interpolator.add_data_point(individual, ppf)
                    
                    self.keff_interpolator.save_data()
                    self.ppf_interpolator.save_data() # Saves both live and best datasets
                    logging.info(f"Successfully completed and saved sample {i + 1}.")

                    duration = time.time() - sample_start_time
                    initial_run_durations.append(duration)
                    avg_duration = np.mean(initial_run_durations)
                    remaining_samples = num_total_samples - (i + 1)
                    if remaining_samples > 0:
                        eta_seconds = avg_duration * remaining_samples
                        eta_str = self._format_time(eta_seconds)
                        logging.info(f"Estimated time remaining for initialization: {eta_str}")

                else:
                    logging.error(f"Failed to extract results for sample {i + 1}. Aborting initialization.")
                    return
            else:
                logging.error(f"OpenMC simulation failed for sample {i + 1}. Aborting initialization phase.")
                return
            
        logging.info("All initial samples have been successfully generated.")

    def run(self):
        """The main optimization loop."""
        num_cycles = self.config['simulation']['num_cycles']
        start_cycle = self.state['cycle_number']
        
        for i in range(start_cycle, num_cycles):
            self.state['cycle_number'] = i + 1
            logging.info(f"--- Starting {self.optimizer_technique.upper()} Cycle {i+1}/{num_cycles} ---")
            cycle_start_time = time.time()

            seed = self.state['best_individual_overall']
            
            best_individual_from_cycle = None
            
            if self.optimizer_technique == 'hybrid':
                last_fitness = self.state['history'][-1]['fitness'] if self.state['history'] else -float('inf')
                best_fitness = self.state['best_true_fitness']
                best_individual_from_cycle = self.optimizer_engine.run_hybrid_algorithm(i, seed, last_fitness, best_fitness)
            elif self.optimizer_technique == 'ga':
                best_individual_from_cycle = self.optimizer_engine.run_genetic_algorithm(seed_individual=seed)
            elif self.optimizer_technique == 'pso':
                best_individual_from_cycle = self.optimizer_engine.run_pso_algorithm(seed_individual=seed)

            predicted_keff = self.keff_interpolator.predict(best_individual_from_cycle)
            predicted_ppf = self.ppf_interpolator.predict(best_individual_from_cycle)
            
            if not self.fuel_handler.update_materials(best_individual_from_cycle):
                logging.critical(f"Cycle {i+1} failed: The proposed individual resulted in an invalid material composition. Skipping simulation.")
                continue
            
            if not self.openmc_runner.run_simulation():
                logging.error("OpenMC simulation failed. Skipping to next cycle.")
                continue

            results = self.openmc_runner.extract_results()
            if results is None:
                logging.error("Failed to extract results. Skipping to next cycle.")
                continue
            true_keff, true_ppf = results
            
            self.keff_interpolator.add_data_point(best_individual_from_cycle, true_keff)
            self.ppf_interpolator.add_data_point(best_individual_from_cycle, true_ppf)

            true_fitness = fitness_function(true_keff, true_ppf, self.config['simulation'], self.config['fitness'])
            
            keff_error_percent = (abs(predicted_keff - true_keff) / true_keff) * 100 if true_keff != 0 else 0.0
            ppf_error_percent = (abs(predicted_ppf - true_ppf) / true_ppf) * 100 if true_ppf != 0 else 0.0
            
            logging.info(f"Cycle {i+1} Result: True Keff={true_keff:.5f}, True PPF={true_ppf:.4f}, True Fitness={true_fitness:.6f}")
            logging.info(f"Interpolator Errors: Keff Error={keff_error_percent:.2f}%, PPF Error={ppf_error_percent:.2f}%")

            if true_fitness > self.state['best_true_fitness']:
                self.state['best_true_fitness'] = true_fitness
                self.state['best_individual_overall'] = best_individual_from_cycle
                self.state['best_true_keff'] = true_keff
                self.state['best_true_ppf'] = true_ppf
                logging.info(f"*** New best-ever individual found! Fitness: {true_fitness:.6f} ***")

            self.state['history'].append({
                'cycle': i + 1,
                'keff': true_keff,
                'ppf': true_ppf,
                'fitness': true_fitness,
                'keff_error_percent': keff_error_percent,
                'ppf_error_percent': ppf_error_percent,
                'individual': best_individual_from_cycle
            })

            cycle_duration = time.time() - cycle_start_time
            self.state['cycle_durations'].append(cycle_duration)
            
            last_n_durations = self.state['cycle_durations'][-5:]
            avg_duration = np.mean(last_n_durations)
            remaining_cycles = num_cycles - (i + 1)
            remaining_seconds = avg_duration * remaining_cycles
            self.state['estimated_remaining_time'] = self._format_time(remaining_seconds)
            
            logging.info(f"Estimated time remaining: {self.state['estimated_remaining_time']}")
            
            # Get the optimizer's internal state before saving the checkpoint
            if hasattr(self.optimizer_engine, 'get_state'):
                self.state['optimizer_state'] = self.optimizer_engine.get_state()
            
            self.checkpoint.save(self.state)
            self.keff_interpolator.save_data()
            self.ppf_interpolator.save_data()
            
            logging.info(f"--- Cycle {i+1} finished in {self._format_time(cycle_duration)} ---")

        logging.info("--- Optimization Finished ---")
        logging.info(f"Best overall individual: {self.state['best_individual_overall']}")
        logging.info(f"Best true Keff: {self.state['best_true_keff']:.5f}")
        logging.info(f"Best true PPF: {self.state['best_true_ppf']:.4f}")
        logging.info(f"Best true Fitness: {self.state['best_true_fitness']:.6f}")
