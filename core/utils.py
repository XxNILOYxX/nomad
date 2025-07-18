import logging
import os
import random
import datetime
import sys
import numpy as np
from typing import List, Dict

def setup_logging(log_dir: str = 'log', log_file: str = 'ga.log'):
    """
    Configures logging to output to both the console and a file.
    Creates the log directory if it doesn't exist.
    """
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure logging with two handlers: one for the file and one for the stream (console)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Output will be sent to the console and to '{log_path}'.")

def detect_hardware(config: Dict) -> bool:
    """
    Detects if a GPU is available and configured for use.
    
    Args:
        config: The hardware configuration dictionary.

    Returns:
        True if GPU is available and enabled, False otherwise.
    """
    use_gpu = False
    if config['gpu'] == 1:
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                use_gpu = True
                logging.info("GPU detected and enabled in config. cuML will be used.")
            else:
                logging.warning("GPU enabled in config, but no compatible device found. Falling back to CPU.")
        except ImportError:
            logging.warning("GPU enabled in config, but cuML/cupy not installed. Falling back to CPU.")
    else:
        logging.info("GPU not enabled in config. Using CPU (scikit-learn).")

    if config['cpu'] == 0 and not use_gpu:
        raise ValueError("Configuration error: GPU is disabled or not found, and CPU is also disabled. At least one must be enabled.")

    return use_gpu

def calculate_diversity(population: List[List[float]], sample_size: int = 50) -> float:
    """
    Calculates the genetic diversity of a population sample.

    Args:
        population: The list of individuals.
        sample_size: The number of individuals to sample for the calculation.

    Returns:
        The average Hamming distance as a measure of diversity.
    """
    if len(population) < 2:
        return 0.0
    
    sample_size = min(sample_size, len(population))
    sample_indices = random.sample(range(len(population)), sample_size)
    sample_pop = [population[i] for i in sample_indices]
    
    distances = []
    for i in range(len(sample_pop)):
        for j in range(i + 1, len(sample_pop)):
            # Using mean absolute difference for floating point chromosomes
            dist = np.mean(np.abs(np.array(sample_pop[i]) - np.array(sample_pop[j])))
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0

def is_dropping(scores: List[float], window: int = 5) -> bool:
    """
    Checks if the last `window` scores are monotonically decreasing.

    Args:
        scores: A list of historical scores.
        window: The number of recent scores to check.

    Returns:
        True if the scores have been consistently dropping, False otherwise.
    """
    if len(scores) < window:
        return False
    
    recent_scores = scores[-window:]
    return all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores) - 1))
    
def find_nearest(array: List[float], value: float) -> float:
    """Finds the nearest value in a sorted array."""
    # Ensure array is a numpy array and sorted
    arr = np.asarray(array)
    idx = np.searchsorted(arr, value, side="left")
    if idx > 0 and (idx == len(arr) or abs(value - arr[idx-1]) < abs(value - arr[idx])):
        return arr[idx-1]
    else:
        return arr[idx]
    
def fitness_function(keff: float, ppf: float, sim_config: Dict, tuning_config: Dict) -> float:
        """
        Calculates a balanced fitness score based on keff and PPF.
        This function provides a continuous score for keff, always rewarding
        solutions closer to the target.
        """
        target_keff = sim_config['target_keff']
        keff_diff = abs(keff - target_keff)
        
        # Invert PPF so that a lower PPF results in a higher score.
        ppf_score = 1.0 / ppf if ppf > 0 else 0

        # Keff score is calculated using a continuous exponential function.
        # This ensures that any improvement in keff results in a better score
        penalty_factor = tuning_config['keff_penalty_factor']
        keff_score = np.exp(-penalty_factor * keff_diff)

        high_thresh = tuning_config['high_keff_diff_threshold']
        med_thresh = tuning_config['med_keff_diff_threshold']
        
        # Dynamically adjust weights based on how far we are from the target keff.
        if keff_diff > high_thresh:
            w_ppf, w_keff = (0.3, 0.7) # Prioritize getting keff right.
        elif keff_diff > med_thresh:
            w_ppf, w_keff = (0.5, 0.5) # Balanced approach.
        else:
            # When keff is close to the target, prioritize minimizing the power peaking factor (PPF).
            w_ppf, w_keff = (0.7, 0.3) 

        return w_ppf * ppf_score + w_keff * keff_score


def show_splash_screen():
    """
    Displays the NOMAD ASCII art logo and application details.
    """
    logo = r"""
                                                              
                              @@@@@@                          
                @@@@         @@@@@@@@         @@@@            
              @@@@@@@@@@  @@@@@@@@@@@@@@  @@@@@@@@@@          
            @@        @@@@@@@@@@@@@@@ @@@@       @@@         
            @@          @@@@       @@@@          @@@         
            @@             @@@   @@@@            @@@         
            @@@              @@@@@               @@          
              @@              @@@@@              @@@          
              @@@         @@@@@@@@@@@@@          @@           
            @ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @          
            @@@@@@@      @@@          @@@      @@@@@@         
        @@@@    @@    @@@  @@@@@@@@@@ @@    @@@   @@@@      
      @@@        @@  @@@  @@@      @@@  @@  @@@       @@@    
      @@           @@@@        @@@        @@@@          @@   
      @@           @@@@        @@@         @@@           @@@  
      @@@         @@@@@@   @@@@    @@@    @@@@@         @@@   
        @@@@     @@@   @@   @@@@@@@@@   @@@   @@      @@@@    
          @@@@@@@@@     @@@           @@@      @@ @@@@@       
              @@@@@@@@@   @@@        @@@  @@@@@@@@@@          
            @ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @          
              @@             @@@ @@@             @@           
            @@@               @@@@               @@          
            @@              @@@ @@@@             @@@         
            @@            @@@     @@@@           @@@         
            @@         @@@@@@@@@@@@@ @@@         @@@         
            @@@    @@@@@ @@@@@@@@@@@@@ @@@@@     @@          
              @@@@@@@       @@@@@@@@       @@@@@@@           
                              @@@@@@                          
                                                                                      
    """
    __version__ = "1.0.6.1" # Version updated for PSO addition
    __copyright__ = "2025, MD Hasebul Hasan Niloy"
    __license__ = "https://github.com/XxNILOYxX/nomad/blob/main/LICENSE"

    print(logo)
    print(f"{'':>15} | The NOMAD Fuel Optimizer")
    print(f"{'Copyright':>15} | {__copyright__}")
    print(f"{'License':>15} | {__license__}")
    print(f"{'Version':>15} | {__version__}")
    print(f"{'Date/Time':>15} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
