import json
import logging
import os
from typing import Dict, Any, Optional
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Checkpoint:
    """
    Handles saving and loading of the application's state to a JSON file.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Checkpoint manager.

        Args:
            config: The main configuration dictionary.
        """
        self.filepath = config['simulation']['checkpoint_file']
        
        # Ensure data directory exists
        data_dir = os.path.dirname(self.filepath)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _convert_infinities_to_str(self, data: Any) -> Any:
        """
        Recursively traverses a data structure to convert float infinities to strings.
        This is necessary because JSON does not support infinity values.
        """
        if isinstance(data, dict):
            return {k: self._convert_infinities_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_infinities_to_str(i) for i in data]
        elif data == float('inf'):
            return "inf"
        elif data == -float('inf'):
            return "-inf"
        return data

    def _convert_str_to_infinities(self, data: Any) -> Any:
        """
        Recursively traverses a data structure to convert "inf" strings back to float infinities
        after loading from a JSON file.
        """
        if isinstance(data, dict):
            return {k: self._convert_str_to_infinities(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_str_to_infinities(i) for i in data]
        elif data == "inf":
            return float('inf')
        elif data == "-inf":
            return float('-inf')
        return data

    def save(self, state: Dict[str, Any]):
        """
        Saves the current state to the checkpoint file using an atomic write pattern.
        """
        temp_filepath = self.filepath + ".tmp"
        try:
            state_to_save = self._convert_infinities_to_str(state)
            with open(temp_filepath, 'w') as f:
                json.dump(state_to_save, f, indent=4, cls=NumpyEncoder)
            
            # This is the atomic operation.
            os.replace(temp_filepath, self.filepath)
            
            logging.info(f"Checkpoint successfully saved to {self.filepath}")

        except (TypeError, IOError) as e:
            logging.error(f"Could not save checkpoint file to {self.filepath}: {e}")
        finally:
            # Clean up the temporary file if it still exists after an error
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Loads the state from the checkpoint file and restores non-serializable values.

        Returns:
            A dictionary containing the loaded state, or None if no valid checkpoint exists.
        """
        if not os.path.exists(self.filepath):
            logging.info("No checkpoint file found. Starting a new run.")
            return None

        try:
            with open(self.filepath, 'r') as f:
                loaded_state = json.load(f)
            
            # Recursively restore all potential infinity values from their string representation
            state = self._convert_str_to_infinities(loaded_state)

            logging.info(f"Checkpoint successfully loaded from {self.filepath}")
            return state

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logging.error(f"Failed to load or parse checkpoint file {self.filepath}: {e}. Starting a new run.")
            return None
