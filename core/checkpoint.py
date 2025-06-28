import json
import logging
import os
from typing import Dict, Any, Optional

class Checkpoint:
    """
    Handles saving and loading of the GA's state to a JSON file.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Checkpoint manager.

        Args:
            config: The main configuration dictionary.
        """
        self.filepath = config['simulation']['checkpoint_file']
        self.keff_interp_file = config['simulation']['keff_interp_file']
        self.ppf_interp_file = config['simulation']['ppf_interp_file']
        
        # Ensure data directory exists
        data_dir = os.path.dirname(self.filepath)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def save(self, state: Dict[str, Any]):
        """
        Saves the current state to the checkpoint file.

        Args:
            state: A dictionary containing the GA state.
        """
        try:
            # Handle non-JSON-serializable float values
            state_to_save = state.copy()
            if 'best_true_fitness' in state_to_save:
                if state_to_save['best_true_fitness'] == float('inf'):
                    state_to_save['best_true_fitness'] = "inf"
                elif state_to_save['best_true_fitness'] == -float('inf'):
                    state_to_save['best_true_fitness'] = "-inf"

            with open(self.filepath, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            logging.info(f"Checkpoint successfully saved to {self.filepath}")

        except (TypeError, IOError) as e:
            logging.error(f"Could not save checkpoint file to {self.filepath}: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Loads the state from the checkpoint file.

        Returns:
            A dictionary containing the loaded state, or None if no valid checkpoint exists.
        """
        if not os.path.exists(self.filepath):
            logging.info("No checkpoint file found. Starting a new run.")
            return None
        
        # A valid state requires both the checkpoint and the interpolator data.
        if not os.path.exists(self.keff_interp_file) or not os.path.exists(self.ppf_interp_file):
            logging.warning("Checkpoint file found, but interpolator data is missing. Starting a new run.")
            return None

        try:
            with open(self.filepath, 'r') as f:
                state = json.load(f)
            
            # Restore non-JSON-serializable float values
            if state.get('best_true_fitness') == "inf":
                state['best_true_fitness'] = float('inf')
            elif state.get('best_true_fitness') == "-inf":
                state['best_true_fitness'] = -float('inf')

            logging.info(f"Checkpoint successfully loaded from {self.filepath}")
            return state

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logging.error(f"Failed to load or parse checkpoint file {self.filepath}: {e}. Starting a new run.")
            return None
