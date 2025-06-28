import openmc
import subprocess
import logging
import os
import glob
import time
import numpy as np
from typing import Tuple, Optional, Dict

class OpenMCRunner:
    """
    Handles the execution of OpenMC simulations and extraction of results.
    """
    def __init__(self, config: Dict):
        """
        Initializes the OpenMCRunner.

        Args:
            config: A dictionary containing simulation parameters.
        """
        self.sim_config = config['simulation']
        self.retries = self.sim_config['openmc_retries']
        self.statepoint_pattern = self.sim_config['statepoint_filename_pattern']
        self.fission_tally_name = self.sim_config['fission_tally_name']

    def run_simulation(self) -> bool:
        """
        Executes an OpenMC simulation using a subprocess, allowing its
        output to stream to the console in real-time.

        Returns:
            True if the simulation completed successfully, False otherwise.
        """
        for attempt in range(self.retries):
            logging.info(f"Running OpenMC simulation (Attempt {attempt + 1}/{self.retries})...")
            try:
                process = subprocess.Popen(
                    ['python', '-c', 'import openmc; openmc.run()'],
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                _, stderr = process.communicate()
                
                if process.returncode == 0:
                    logging.info("OpenMC simulation finished successfully.")
                    return True
                else:
                    logging.error(f"OpenMC process failed with return code {process.returncode}.")
                    if stderr:
                        logging.error(f"STDERR:\n{stderr}")

            except FileNotFoundError:
                logging.error("The 'python' command was not found. Make sure Python and OpenMC are correctly installed in your environment.")
                return False
            except Exception as e:
                logging.error(f"An unexpected error occurred while running OpenMC: {e}")

            if attempt < self.retries - 1:
                delay = self.sim_config['openmc_retry_delay']
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        logging.error(f"OpenMC simulation failed after {self.retries} attempts.")
        return False

    def extract_results(self) -> Optional[Tuple[float, float]]:
        """
        Extracts keff and power peaking factor (PPF) from the latest statepoint file.

        Returns:
            A tuple containing (keff, ppf), or None if results cannot be extracted.
        """
        try:
            # Find the latest statepoint file dynamically
            statepoint_files = sorted(glob.glob(self.statepoint_pattern))
            if not statepoint_files:
                logging.error(f"No statepoint file found matching pattern: {self.statepoint_pattern}")
                return None
            latest_statepoint = statepoint_files[-1]
            logging.info(f"Extracting results from '{latest_statepoint}'...")

            with openmc.StatePoint(latest_statepoint) as sp:
                # Extract keff
                keff = sp.keff.nominal_value

                # Extract PPF from the specified fission tally
                tally = sp.get_tally(name=self.fission_tally_name)
                fission_rates = tally.get_slice(scores=['fission']).mean.ravel()

                if np.mean(fission_rates) > 0:
                    ppf = np.max(fission_rates) / np.mean(fission_rates)
                else:
                    logging.error("Cannot calculate PPF, mean fission rate is zero or negative.")
                    return None
            
            logging.info(f"Extracted results: keff = {keff:.5f}, PPF = {ppf:.4f}")
            return keff, ppf

        except Exception as e:
            logging.error(f"Failed to extract results from statepoint file: {e}")
            return None