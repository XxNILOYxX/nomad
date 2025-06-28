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

        # Load [ga_tuning] 
        ga_tuning_params = {
            'log_frequency': self.config.getint('ga_tuning', 'log_frequency'),
            'diversity_sample_size': self.config.getint('ga_tuning', 'diversity_sample_size'),
            'keff_penalty_factor': self.config.getfloat('ga_tuning', 'keff_penalty_factor'),
            'high_keff_diff_threshold': self.config.getfloat('ga_tuning', 'high_keff_diff_threshold'),
            'med_keff_diff_threshold': self.config.getfloat('ga_tuning', 'med_keff_diff_threshold'),
            'smart_mutate_increase_factor': self.config.getfloat('ga_tuning', 'smart_mutate_increase_factor'),
            'smart_mutate_decrease_factor': self.config.getfloat('ga_tuning', 'smart_mutate_decrease_factor'),
        }
        self.params['ga'].update(ga_tuning_params)


        # Load [enrichment] section
        central_range = list(map(float, self.config.get('enrichment', 'central_range').split(',')))
        outer_range = list(map(float, self.config.get('enrichment', 'outer_range').split(',')))
        self.params['enrichment'] = {
            'central_values': np.arange(central_range[0], central_range[1] + central_range[2], central_range[2]).tolist(),
            'outer_values': np.arange(outer_range[0], outer_range[1] + outer_range[2], outer_range[2]).tolist(),
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
            'keff_tolerance': self.config.getfloat('simulation', 'keff_tolerance'),
            'num_cycles': self.config.getint('simulation', 'num_cycles'),
            'num_assemblies': self.config.getint('simulation', 'num_assemblies'),
            'num_central_assemblies': self.config.getint('simulation', 'num_central_assemblies'),
            'start_id': self.config.getint('simulation', 'start_id'),
            'materials_xml_path': self.config.get('simulation', 'materials_xml_path'),
            'fission_tally_name': self.config.get('simulation', 'fission_tally_name'),
            'ppf_interp_file': self.config.get('simulation', 'ppf_interp_file'),
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
        self.params['interpolator'] = {
            'max_keff_points': self.config.getint('interpolator', 'max_keff_points'),
            'max_ppf_points': self.config.getint('interpolator', 'max_ppf_points'),
            'min_interp_points': self.config.getint('interpolator', 'min_interp_points'),
            'min_validation_score': self.config.getfloat('interpolator', 'min_validation_score'),
            'regressor_type': self.config.get('interpolator', 'regressor_type'),
            'n_neighbors': self.config.getint('interpolator', 'n_neighbors'),
        }

        # Load fuel setup
        self.params['fuel'] = {
            'slack_isotope': self.fuel_setup.get('general', 'slack_isotope'),
            'fissile_flags': {k: self.fuel_setup.getboolean('fissile', k) for k in self.fuel_setup['fissile']},
            'pu_dist': {k: self.fuel_setup.getfloat('plutonium_distribution', k) for k in self.fuel_setup['plutonium_distribution']},
        }
        
    def _validate_configs(self):
        """
        Performs validation checks on the loaded configuration.
        """
        logging.info("Validating configuration...")
        
        fissile_flags = self.params['fuel']['fissile_flags']
        num_fissile_selected = sum(fissile_flags.values())
        if num_fissile_selected == 0:
            raise ValueError("Fuel setup error: At least one isotope must be selected as fissile (set to 1) in setup_fuel.ini.")

        is_pu_selected = any(flag for key, flag in fissile_flags.items() if key.lower().startswith('pu'))
        if is_pu_selected:
            pu_dist_sum = sum(self.params['fuel']['pu_dist'].values())
            if not np.isclose(pu_dist_sum, 1.0):
                raise ValueError(f"Plutonium distribution weights in setup_fuel.ini must sum to 1.0, but sum to {pu_dist_sum}")

        ga_p = self.params['ga']
        if not (0 < ga_p['mutation_rate'] <= 1 and 0 < ga_p['crossover_rate'] <= 1):
            raise ValueError("Mutation and crossover rates must be between 0 and 1.")

        sim_p = self.params['simulation']
        if sim_p['num_central_assemblies'] >= sim_p['num_assemblies']:
            raise ValueError("Number of central assemblies must be less than total assemblies.")
        
        if sim_p['start_id'] < 1:
            raise ValueError("start_id in [simulation] must be a positive integer.")

        hw_p = self.params['hardware']
        if hw_p['cpu'] == 0 and hw_p['gpu'] == 0:
            raise ValueError("Configuration error: At least one of CPU or GPU must be enabled in [hardware].")

        logging.info("Configuration validated successfully.")

    def get_params(self) -> dict:
        """
        Returns the loaded and validated parameters.
        """
        return self.params