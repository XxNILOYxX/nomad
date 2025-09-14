import configparser
import numpy as np
import logging
import ast
from typing import Dict

class ConfigLoader:
    """
    Loads and validates configuration from INI files.
    """
    def __init__(self, config_path: str, fuel_setup_path: str):
        """
        Initializes the ConfigLoader with paths to the configuration files.

        Args:
            config_path: Path to the main configuration file (config.ini).
            fuel_setup_path: Path to the fuel setup file (setup_fuel.ini).
        """
        self.config = configparser.ConfigParser()
        self.fuel_setup = configparser.ConfigParser()

        self.config.optionxform = str
        self.fuel_setup.optionxform = str

        if not self.config.read(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not self.fuel_setup.read(fuel_setup_path):
            raise FileNotFoundError(f"Fuel setup file not found: {fuel_setup_path}")

        self.params = {}
        self._load_all_configs()
        self._validate_configs()

    def _load_all_configs(self):
        """
        Loads all sections from the INI files into the params dictionary.
        """
        # Load [optimizer] section
        self.params['optimizer'] = {
            'technique': self.config.get('optimizer', 'technique').lower()
        }

        # Load [hybrid] section
        if self.params['optimizer']['technique'] == 'hybrid':
            self.params['hybrid'] = {
                'switch_mode': self.config.get('hybrid', 'switch_mode', fallback='fixed_cycles').lower(),
                'ga_phase_cycles': self.config.getint('hybrid', 'ga_phase_cycles', fallback=10),
                'pso_phase_cycles': self.config.getint('hybrid', 'pso_phase_cycles', fallback=15),
                'stagnation_threshold': self.config.getint('hybrid', 'stagnation_threshold', fallback=10),
                'ga_min_diversity_for_switch': self.config.getfloat('hybrid', 'ga_min_diversity_for_switch', fallback=0.25),
                'pso_min_diversity_for_switch': self.config.getfloat('hybrid', 'pso_min_diversity_for_switch', fallback=0.15),
                'ga_seed_ratio': self.config.getfloat('hybrid', 'ga_seed_ratio', fallback=0.25),
                'pso_seed_ratio': self.config.getfloat('hybrid', 'pso_seed_ratio', fallback=0.5),
                'adaptive_switching_threshold': self.config.getfloat('hybrid', 'adaptive_switching_threshold', fallback=1.2),
                'min_adaptive_phase_duration': self.config.getint('hybrid', 'min_adaptive_phase_duration', fallback=5),
                'adaptive_trend_dampening_factor': self.config.getfloat('hybrid', 'adaptive_trend_dampening_factor', fallback=0.5)
            }

        # Load [ga] section
        self.params['ga'] = {
            'population_size': self.config.getint('ga', 'population_size'),
            'generations_per_openmc_cycle': self.config.getint('ga', 'generations_per_openmc_cycle'),
            'mutation_rate': self.config.getfloat('ga', 'mutation_rate'),
            'max_mutation_rate': self.config.getfloat('ga', 'max_mutation_rate'),
            'crossover_rate': self.config.getfloat('ga', 'crossover_rate'),
            'max_crossover_rate': self.config.getfloat('ga', 'max_crossover_rate'),
            'elitism_count': self.config.getint('ga', 'elitism_count'),
            'stagnation_threshold': self.config.getint('ga', 'stagnation_threshold'),
            'diversity_threshold': self.config.getfloat('ga', 'diversity_threshold'),
            'tournament_size': self.config.getint('ga', 'tournament_size'),
        }

        # Load [fitness_tuning] section
        self.params['fitness'] = {
            'keff_fitness_weight': self.config.getfloat('fitness_tuning', 'keff_fitness_weight'),
            'ppf_fitness_weight': self.config.getfloat('fitness_tuning', 'ppf_fitness_weight'),
            'keff_penalty_factor': self.config.getfloat('fitness_tuning', 'keff_penalty_factor'),
            'high_keff_diff_threshold': self.config.getfloat('fitness_tuning', 'high_keff_diff_threshold'),
            'med_keff_diff_threshold': self.config.getfloat('fitness_tuning', 'med_keff_diff_threshold'),
        }

        # Load [ga_tuning] section
        ga_tuning_params = {
            'log_frequency': self.config.getint('ga_tuning', 'log_frequency'),
            'diversity_sample_size': self.config.getint('ga_tuning', 'diversity_sample_size'),
            'smart_mutate_increase_factor': self.config.getfloat('ga_tuning', 'smart_mutate_increase_factor'),
            'smart_mutate_decrease_factor': self.config.getfloat('ga_tuning', 'smart_mutate_decrease_factor'),
            'convergence_threshold': self.config.getint('ga_tuning', 'convergence_threshold', fallback=200),
            'smart_mutation_bias': self.config.getfloat('ga_tuning', 'smart_mutation_bias', fallback=0.75),
            'diversity_check_multiplier': self.config.getint('ga_tuning', 'diversity_check_multiplier', fallback=5),
        }
        self.params['ga'].update(ga_tuning_params)

        # Load [pso] section
        self.params['pso'] = {
            'swarm_size': self.config.getint('pso', 'swarm_size'),
            'iterations_per_openmc_cycle': self.config.getint('pso', 'iterations_per_openmc_cycle'),
            'cognitive_coeff': self.config.getfloat('pso', 'cognitive_coeff'),
            'social_coeff': self.config.getfloat('pso', 'social_coeff'),
            'pso_convergence_threshold': self.config.getint('pso', 'pso_convergence_threshold', fallback=200),
            'inertia_weight_start': self.config.getfloat('pso', 'inertia_weight_start', fallback=0.9),
            'inertia_weight_end': self.config.getfloat('pso', 'inertia_weight_end', fallback=0.4),
            'topology': self.config.get('pso', 'topology', fallback='global').lower(),
            'neighborhood_size': self.config.getint('pso', 'neighborhood_size', fallback=4),
            'neighborhood_rebuild_frequency': self.config.getint('pso', 'neighborhood_rebuild_frequency', fallback=100),
            'adaptive_velocity': self.config.getboolean('pso', 'adaptive_velocity', fallback=True),
            'base_change_probability': self.config.getfloat('pso', 'base_change_probability', fallback=0.2),
            'max_change_probability': self.config.getfloat('pso', 'max_change_probability', fallback=0.8),
            'dynamic_exploration': self.config.getboolean('pso', 'dynamic_exploration', fallback=True),
            'base_exploration_factor': self.config.getfloat('pso', 'base_exploration_factor', fallback=0.05),
            'diversity_threshold': self.config.getfloat('pso', 'diversity_threshold', fallback=0.15),
            'max_velocity_fraction': self.config.getfloat('pso', 'max_velocity_fraction', fallback=0.2),
            'pso_exploration_factor': self.config.getfloat('pso', 'pso_exploration_factor', fallback=0.1),
            'min_diversity_threshold': self.config.getfloat('pso', 'min_diversity_threshold', fallback=0.12),
            'mutation_probability': self.config.getfloat('pso', 'mutation_probability', fallback=0.08),
            'enable_multi_swarm': self.config.getboolean('pso', 'enable_multi_swarm', fallback=True),
            'num_sub_swarms': self.config.getint('pso', 'num_sub_swarms', fallback=3),
            'migration_frequency': self.config.getint('pso', 'migration_frequency', fallback=100),
            'migration_rate': self.config.getfloat('pso', 'migration_rate', fallback=0.1),
            'enable_local_search': self.config.getboolean('pso', 'enable_local_search', fallback=True),
            'local_search_frequency': self.config.getint('pso', 'local_search_frequency', fallback=25),
            'enable_smart_mutation': self.config.getboolean('pso', 'enable_smart_mutation', fallback=True),
            'smart_mutation_bias': self.config.getfloat('pso', 'smart_mutation_bias', fallback=0.75),
        }


        # Load [enrichment] section
        central_range = list(map(float, self.config.get('enrichment', 'central_range').split(',')))
        outer_range = list(map(float, self.config.get('enrichment', 'outer_range').split(',')))
        
        num_central_steps = int(round((central_range[1] - central_range[0]) / central_range[2])) + 1
        num_outer_steps = int(round((outer_range[1] - outer_range[0]) / outer_range[2])) + 1

        self.params['enrichment'] = {
            'central_values': [round(x, 2) for x in np.linspace(central_range[0], central_range[1], num_central_steps)],
            'outer_values': [round(x, 2) for x in np.linspace(outer_range[0], outer_range[1], num_outer_steps)],
            'initial_samples': self.config.getint('enrichment', 'initial_samples'),
        }
        
        initial_configs_str = self.config.get('enrichment', 'initial_configs', fallback='').strip()
        if initial_configs_str:
            try:
                self.params['enrichment']['initial_configs'] = ast.literal_eval(initial_configs_str)
                logging.info(f"Loaded {len(self.params['enrichment']['initial_configs'])} specific initial configurations from 'initial_configs'.")
            except (ValueError, SyntaxError) as e:
                logging.error(f"Error parsing 'initial_configs' in config.ini: {e}. It must be a valid Python list of tuples.")
                self.params['enrichment']['initial_configs'] = []
        else:
            self.params['enrichment']['initial_configs'] = []

        # Load [simulation] section
        self.params['simulation'] = {
            'target_keff': self.config.getfloat('simulation', 'target_keff'),
            'num_cycles': self.config.getint('simulation', 'num_cycles'),
            'num_assemblies': self.config.getint('simulation', 'num_assemblies'),
            'num_central_assemblies': self.config.getint('simulation', 'num_central_assemblies'),
            'start_id': self.config.getint('simulation', 'start_id'),
            'materials_xml_path': self.config.get('simulation', 'materials_xml_path'),
            'fission_tally_name': self.config.get('simulation', 'fission_tally_name'),
            'ppf_interp_file': self.config.get('simulation', 'ppf_interp_file'),
            'ppf_interp_file_best': self.config.get('simulation', 'ppf_interp_file_best'),
            'keff_interp_file': self.config.get('simulation', 'keff_interp_file'),
            'checkpoint_file': self.config.get('simulation', 'checkpoint_file'),
            'statepoint_filename_pattern': self.config.get('simulation', 'statepoint_filename_pattern'),
            'openmc_retries': self.config.getint('simulation', 'openmc_retries'),
            'openmc_retry_delay': self.config.getint('simulation', 'openmc_retry_delay'),
        }
        
        # Load [hardware] section
        self.params['hardware'] = {
            'cpu': self.config.getint('hardware', 'cpu'),
            'gpu': self.config.getint('hardware', 'gpu'),
        }

        # Load [interpolator] section
        hidden_layers_str = self.config.get('interpolator', 'nn_hidden_layers', fallback='128, 64')
        try:
            nn_hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
            if not nn_hidden_layers:
                 raise ValueError("nn_hidden_layers cannot be empty.")
        except ValueError:
            logging.error("Invalid format for 'nn_hidden_layers' in config.ini. Must be a comma-separated list of positive integers. Using default [64, 64].")
            nn_hidden_layers = [64, 64]
        self.params['interpolator'] = {
            'max_keff_points': self.config.getint('interpolator', 'max_keff_points'),
            'max_ppf_points': self.config.getint('interpolator', 'max_ppf_points'),
            'min_interp_points': self.config.getint('interpolator', 'min_interp_points'),
            'min_validation_score': self.config.getfloat('interpolator', 'min_validation_score'),
            'regressor_type': self.config.get('interpolator', 'regressor_type'),
            'nn_hidden_layers': nn_hidden_layers,
            'nn_epochs': self.config.getint('interpolator', 'nn_epochs', fallback=100),
            'nn_batch_size': self.config.getint('interpolator', 'nn_batch_size', fallback=32),
            'nn_learning_rate': self.config.getfloat('interpolator', 'nn_learning_rate', fallback=0.001),
            'n_neighbors': self.config.getint('interpolator', 'n_neighbors'),
            'nn_dropout_rate': self.config.getfloat('interpolator', 'nn_dropout_rate', fallback=0.2),
            'nn_patience': self.config.getint('interpolator', 'nn_patience', fallback=10),
            'nn_random_seed': self.config.getint('interpolator', 'nn_random_seed', fallback=42),
        }

        # Load fuel setup
        self.params['fuel'] = {
            'slack_isotope': self.fuel_setup.get('general', 'slack_isotope'),
            'fissile_flags': {k: self.fuel_setup.getboolean('fissile', k) for k in self.fuel_setup['fissile']},
            'pu_dist': {k: self.fuel_setup.getfloat('plutonium_distribution', k) for k in self.fuel_setup['plutonium_distribution']},
        }
        
    def _validate_configs(self):
        """
        Performs validation checks on the loaded configuration, including logical relationships.
        """
        logging.info("Validating configuration...")
        
        # --- Fitness Tuning Validation ---
        fit_p = self.params['fitness']
        if not (0 <= fit_p['keff_fitness_weight'] <= 1 and 0 <= fit_p['ppf_fitness_weight'] <= 1):
            raise ValueError("Fitness weights in [fitness_tuning] must be between 0.0 and 1.0.")
        if not np.isclose(fit_p['keff_fitness_weight'] + fit_p['ppf_fitness_weight'], 1.0):
            raise ValueError("Fitness weights 'keff_fitness_weight' and 'ppf_fitness_weight' in [fitness_tuning] must sum to 1.0.")
        
        # --- Optimizer Technique Validation ---
        if self.params['optimizer']['technique'] not in ['ga', 'pso', 'hybrid']:
            raise ValueError("Optimizer 'technique' must be 'ga', 'pso', or 'hybrid' in config.ini.")
            
        # --- Hybrid Engine Validation ---
        if self.params['optimizer']['technique'] == 'hybrid':
            hybrid_p = self.params['hybrid']
            if hybrid_p['switch_mode'] not in ['fixed_cycles', 'stagnation', 'oscillate', 'adaptive']:
                raise ValueError("Hybrid 'switch_mode' must be 'fixed_cycles', 'stagnation', 'oscillate', or 'adaptive'.")

            min_phase_cycles = 3
            if hybrid_p['ga_phase_cycles'] < min_phase_cycles:
                logging.warning(f"Hybrid 'ga_phase_cycles' is set to {hybrid_p['ga_phase_cycles']}, which is very low and may lead to ineffective optimization. A value of {min_phase_cycles} or higher is recommended.")
            if hybrid_p['pso_phase_cycles'] < min_phase_cycles:
                logging.warning(f"Hybrid 'pso_phase_cycles' is set to {hybrid_p['pso_phase_cycles']}, which is very low. A value of {min_phase_cycles} or higher is recommended.")
            if hybrid_p['stagnation_threshold'] < 2:
                raise ValueError("Hybrid 'stagnation_threshold' must be at least 2 to be effective.")
            if not (0 < hybrid_p['ga_min_diversity_for_switch'] < 1):
                raise ValueError("Hybrid 'ga_min_diversity_for_switch' must be between 0 and 1.")
            if not (0 < hybrid_p['pso_min_diversity_for_switch'] < 1):
                raise ValueError("Hybrid 'pso_min_diversity_for_switch' must be between 0 and 1.")
            if not (0 < hybrid_p['adaptive_trend_dampening_factor'] < 1):
                raise ValueError("Hybrid 'adaptive_trend_dampening_factor' must be between 0 and 1.")
            if not (0.0 <= hybrid_p['pso_seed_ratio'] <= 1.0):
                raise ValueError("Hybrid 'pso_seed_ratio' must be between 0.0 and 1.0.")

        # --- Fuel Setup Validation ---
        fissile_flags = self.params['fuel']['fissile_flags']
        if sum(fissile_flags.values()) == 0:
            raise ValueError("Fuel setup error: At least one isotope must be selected as fissile (set to 1) in setup_fuel.ini.")

        is_pu_selected = any(flag for key, flag in fissile_flags.items() if key.lower().startswith('pu'))
        if is_pu_selected:
            pu_dist_sum = sum(self.params['fuel']['pu_dist'].values())
            if not np.isclose(pu_dist_sum, 1.0):
                raise ValueError(f"Plutonium distribution weights in setup_fuel.ini must sum to 1.0, but sum to {pu_dist_sum}")

        # --- GA Validation ---
        if self.params['optimizer']['technique'] in ['ga', 'hybrid']:
            ga_p = self.params['ga']
            if not (0 < ga_p['mutation_rate'] <= 1 and 0 < ga_p['crossover_rate'] <= 1):
                raise ValueError("Mutation and crossover rates must be between 0 and 1.")
            if not (0.5 <= ga_p['smart_mutation_bias'] <= 1.0):
                 raise ValueError("GA 'smart_mutation_bias' must be between 0.5 and 1.0.")
            if ga_p['elitism_count'] >= ga_p['population_size']:
                raise ValueError("GA 'elitism_count' must be less than 'population_size'.")
            if ga_p['tournament_size'] > ga_p['population_size']:
                raise ValueError("GA 'tournament_size' cannot be greater than 'population_size'.")
            if ga_p['tournament_size'] < 2:
                raise ValueError("GA 'tournament_size' must be at least 2 for selection to occur.")
            if ga_p['generations_per_openmc_cycle'] < 10:
                logging.warning(f"GA 'generations_per_openmc_cycle' is set to {ga_p['generations_per_openmc_cycle']}, which is very low for meaningful evolution within a cycle.")

        # --- PSO Validation ---
        if self.params['optimizer']['technique'] in ['pso', 'hybrid']:
            pso_p = self.params['pso']
            if pso_p['swarm_size'] <= 0: raise ValueError("PSO 'swarm_size' must be a positive integer.")
            if pso_p['pso_convergence_threshold'] <= 0: raise ValueError("PSO 'pso_convergence_threshold' must be a positive integer.")
            
            valid_topologies = ['global', 'ring', 'random', 'fitness_based']
            if pso_p['topology'] not in valid_topologies:
                raise ValueError(f"PSO 'topology' must be one of {valid_topologies}")
            if pso_p['neighborhood_size'] % 2 != 0 and pso_p['topology'] in ['ring', 'fitness_based']:
                 logging.warning("For 'ring' or 'fitness_based' topologies, it is recommended to use an even 'neighborhood_size'.")
            
            if not (0 < pso_p['diversity_threshold'] < 1): raise ValueError("PSO 'diversity_threshold' must be between 0 and 1.")
            if not (0 < pso_p['min_diversity_threshold'] < 1): raise ValueError("PSO 'min_diversity_threshold' must be between 0 and 1.")
            if not (0 <= pso_p['mutation_probability'] < 1): raise ValueError("PSO 'mutation_probability' must be between 0 and 1.")
            
            if pso_p['iterations_per_openmc_cycle'] < 10:
                logging.warning(f"PSO 'iterations_per_openmc_cycle' is set to {pso_p['iterations_per_openmc_cycle']}, which is very low for meaningful convergence within a cycle.")

            if pso_p['enable_multi_swarm']:
                if pso_p['num_sub_swarms'] < 2: raise ValueError("PSO 'num_sub_swarms' must be at least 2 for multi-swarm mode.")
                if not (0 < pso_p['migration_rate'] < 1): raise ValueError("PSO 'migration_rate' must be between 0 and 1.")
                if pso_p['swarm_size'] % pso_p['num_sub_swarms'] != 0: logging.warning("For best results, 'swarm_size' should be evenly divisible by 'num_sub_swarms'.")
                if pso_p['swarm_size'] < pso_p['num_sub_swarms'] * 2: raise ValueError("In multi-swarm mode, 'swarm_size' must be large enough to allocate at least 2 particles per sub-swarm.")

            if pso_p['enable_local_search'] and pso_p['local_search_frequency'] <= 0:
                raise ValueError("PSO 'local_search_frequency' must be a positive integer.")
                
        # --- Simulation Validation ---
        sim_p = self.params['simulation']
        if sim_p['num_central_assemblies'] >= sim_p['num_assemblies']:
            raise ValueError("Number of central assemblies must be less than total assemblies.")
        if sim_p['start_id'] < 1:
            raise ValueError("start_id in [simulation] must be a positive integer.")

        # --- Hardware Validation ---
        hw_p = self.params['hardware']
        if hw_p['cpu'] == 0 and hw_p['gpu'] == 0:
            raise ValueError("Configuration error: At least one of CPU or GPU must be enabled in [hardware].")
        
        # --- Interpolator Validation --- 
        interp_p = self.params['interpolator']
        if interp_p['regressor_type'] == 'dnn':
            if not all(isinstance(n, int) and n > 0 for n in interp_p['nn_hidden_layers']):
                raise ValueError("All values in 'nn_hidden_layers' must be positive integers.")       

        logging.info("Configuration validated successfully.")

    def get_params(self) -> dict:
        """
        Returns the loaded and validated parameters.
        """
        return self.params
