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
    __version__ = "1.0.5"
    __copyright__ = "2025, MD Hasebul Hasan Niloy"
    __license__ = "https://github.com/XxNILOYxX/nomad/blob/main/LICENSE"

    print(logo)
    print(f"{'':>15} | The NOMAD Fuel Optimizer")
    print(f"{'Copyright':>15} | {__copyright__}")
    print(f"{'License':>15} | {__license__}")
    print(f"{'Version':>15} | {__version__}")
    print(f"{'Date/Time':>15} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
