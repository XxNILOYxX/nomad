import logging
import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from typing import List, Dict, Any

from .utils import is_dropping

# Attempt to import GPU libraries
try:
    import cupy
    from cuml.neighbors import KNeighborsRegressor as cuKNN
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.linear_model import Ridge as cuRidge
    GPU_AVAILABLE = True
    logging.info("cuML (GPU) libraries found.")
except ImportError:
    GPU_AVAILABLE = False
    # This warning is now informational; the final decision is made at runtime.
    logging.warning("cuML (GPU) libraries not found. GPU acceleration will be disabled.")

# Always import CPU libraries as a fallback and for validation
from sklearn.neighbors import KNeighborsRegressor as skKNN
from sklearn.ensemble import RandomForestRegressor as skRF
from sklearn.linear_model import Ridge as skRidge
logging.info("scikit-learn (CPU) libraries loaded.")


CPU_MODELS = {
    'knn': skKNN,
    'random_forest': skRF,
    'ridge': skRidge
}

# This dictionary is now correctly guarded by the availability flag.
if GPU_AVAILABLE:
    GPU_MODELS = {
        'knn': cuKNN,
        'random_forest': cuRF,
        'ridge': cuRidge
    }
else:
    GPU_MODELS = {}


class BaseInterpolator:
    """Base class for shared interpolator functionality."""
    def __init__(self, config: Dict, use_gpu: bool):
        self.interp_config = config['interpolator']
        self.sim_config = config['simulation']
        # The final decision to use the GPU depends on both availability AND user config.
        # This line correctly determines if GPU should be used for this instance.
        self.use_gpu = use_gpu and GPU_AVAILABLE #
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        self.targets = []
        self.last_retrain_size = 0
    
    def _get_features(self, enrichment_config: List[float]) -> np.ndarray:
        """Calculates features from an enrichment configuration."""
        arr = np.array(enrichment_config)
        num_central = self.sim_config['num_central_assemblies'] #
        
        central_avg = np.mean(arr[:num_central]) #
        outer_avg = np.mean(arr[num_central:]) #
        enrich_std = np.std(arr) #
        enrich_grad = central_avg - outer_avg #
        max_enrich = np.max(arr) #
        
        return np.array([central_avg, outer_avg, enrich_std, enrich_grad, max_enrich]) #
    
    def predict_batch(self, enrichment_configs: List[List[float]]) -> List[float]:
        """Predicts target values for a batch of enrichment configurations."""
        if not self.features or len(self.features) < self.interp_config['min_interp_points'] or self.model is None: #
            # Return a sensible default for the whole batch
            default_val = self.sim_config['target_keff'] if isinstance(self, KeffInterpolator) else 2.0 #
            return [default_val] * len(enrichment_configs) #

        # Calculate features for the entire batch
        feature_vectors = np.array([self._get_features(config) for config in enrichment_configs]) #
        X_scaled = self.scaler.transform(feature_vectors) #
        
        try:
            if self.use_gpu:
                X_scaled_gpu = cupy.asarray(X_scaled) #
                predictions = self.model.predict(X_scaled_gpu) #
                return cupy.asnumpy(predictions).tolist() #
            else:
                predictions = self.model.predict(X_scaled) #
                return predictions.tolist() #
        except Exception as e:
            logging.error(f"Batch prediction failed in {self.__class__.__name__}: {e}") #
            default_val = self.sim_config['target_keff'] if isinstance(self, KeffInterpolator) else 2.0 #
            return [default_val] * len(enrichment_configs) #

    def predict(self, enrichment_config: List[float]) -> float:
        """Predicts a target value for a single enrichment configuration."""
        return self.predict_batch([enrichment_config])[0] #


    def add_data_point(self, enrichment_config: List[float], target_value: float):
        """Adds a new data point and retrains the model if needed."""
        feature_vector = self._get_features(enrichment_config) #
        self.features.append(feature_vector) #
        self.targets.append(target_value) #
        
        if len(self.features) > self.last_retrain_size + 5: #
            self.retrain() #
            self.last_retrain_size = len(self.features) #

    def retrain(self):
        raise NotImplementedError #

    def save_data(self, file_path: str):
        """Saves interpolator features and targets to a JSON file."""
        data_dir = os.path.dirname(file_path) #
        if not os.path.exists(data_dir): #
            os.makedirs(data_dir) #
            
        data = {
            'features': [f.tolist() for f in self.features], #
            'targets': self.targets #
        }
        try:
            with open(file_path, 'w') as f: #
                json.dump(data, f, indent=4) #
        except IOError as e:
            logging.error(f"Error saving interpolator data to {file_path}: {e}") #

    def load_data(self, file_path: str) -> bool:
        """Loads interpolator features and targets from a JSON file."""
        if not os.path.exists(file_path): #
            return False #
        try:
            with open(file_path, 'r') as f: #
                data = json.load(f) #
            self.features = [np.array(f) for f in data['features']] #
            self.targets = data['targets'] #
            if self.features:
                # Retrain with loaded data
                self.retrain() #
                logging.info(f"Loaded {len(self.features)} data points for {self.__class__.__name__} from {file_path}") #
                return True #
        except (IOError, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error loading interpolator data from {file_path}: {e}") #
        return False #


class KeffInterpolator(BaseInterpolator):
    """Interpolator for predicting k-effective (keff)."""
    def __init__(self, config: Dict, use_gpu: bool):
        super().__init__(config, use_gpu)
        n_neighbors = self.interp_config['n_neighbors'] #
        model_class = GPU_MODELS.get('knn') if self.use_gpu else CPU_MODELS.get('knn') #
        self.model = model_class(n_neighbors=n_neighbors) #
        logging.info(f"KeffInterpolator initialized. Using GPU: {self.use_gpu}") #

    def retrain(self):
        """Retrains the k-eff interpolator model."""
        if len(self.features) < self.interp_config['n_neighbors']: #
            return
            
        X = np.array(self.features) #
        y = np.array(self.targets) #
        
        # Prune dataset if it exceeds max size
        max_points = self.interp_config['max_keff_points'] #
        if len(X) > max_points: #
            indices = np.random.choice(len(X), max_points, replace=False) #
            X, y = X[indices], y[indices] #
            self.features = [f for i, f in enumerate(self.features) if i in indices] #
            self.targets = [t for i, t in enumerate(self.targets) if i in indices] #

        X_scaled = self.scaler.fit_transform(X) #
        
        try:
            if self.use_gpu:
                self.model.fit(cupy.asarray(X_scaled), cupy.asarray(y)) #
            else:
                self.model.fit(X_scaled, y) #
            logging.info(f"KeffInterpolator retrained with {len(self.features)} points.") #
        except Exception as e:
            logging.error(f"Failed to retrain KeffInterpolator model: {e}") #

class PPFInterpolator(BaseInterpolator):
    """Interpolator for predicting the peak power factor (PPF)."""
    def __init__(self, config: Dict, use_gpu: bool):
        super().__init__(config, use_gpu)
        self.regressor_type = self.interp_config['regressor_type'] #
        self.validation_scores = [] #
        self._setup_model()
        logging.info(f"PPFInterpolator initialized. Type: {self.regressor_type}. GPU: {self.use_gpu}") 

    def _setup_model(self):
        """Initializes the correct regressor model based on config."""
        n_neighbors = self.interp_config['n_neighbors'] #
        
        if self.use_gpu:
            model_class = GPU_MODELS.get(self.regressor_type) #
        else:
            model_class = CPU_MODELS.get(self.regressor_type) #
        
        if not model_class:
            raise ValueError(f"Unknown regressor type: {self.regressor_type}") #
            
        if self.regressor_type == 'knn':
            self.model = model_class(n_neighbors=n_neighbors) #
        elif self.regressor_type == 'random_forest':
            self.model = model_class(n_estimators=100) #
        else: # ridge
            self.model = model_class() #

    def retrain(self):
        """Validates and retrains the PPF model."""
        if len(self.features) < self.interp_config['min_interp_points']: #
            return

        # Prune dataset
        self._prune_dataset()
        
        X = np.array(self.features) #
        y = np.array(self.targets) #
        
        if len(X) < self.interp_config['min_interp_points']: #
             logging.warning("Not enough data points to retrain PPF interpolator after pruning.") #
             return

        val_model_class = CPU_MODELS.get(self.regressor_type) #
        if self.regressor_type == 'knn':
            temp_model = val_model_class(n_neighbors=self.interp_config['n_neighbors']) #
        elif self.regressor_type == 'random_forest':
            temp_model = val_model_class(n_estimators=100) #
        else: # ridge
            temp_model = val_model_class() #
        
        X_scaled_val = StandardScaler().fit_transform(X) #
        
        try:
            scores = cross_val_score(temp_model, X_scaled_val, y, cv=5, scoring='r2') #
            score = np.mean(scores) #
        except ValueError: 
            score = -1.0 #
        
        self._log_validation(score)
        
        if score < self.interp_config['min_validation_score']: #
            logging.warning(f"PPF validation score (R^2 = {score:.4f}) is below threshold ({self.interp_config['min_validation_score']}). Aborting retrain.") #
            return

        X_full_scaled = self.scaler.fit_transform(X) #
        try:
            if self.use_gpu:
                self.model.fit(cupy.asarray(X_full_scaled), cupy.asarray(y)) #
            else:
                self.model.fit(X_full_scaled, y) #
            
            logging.info(f"PPFInterpolator retrained with {len(self.features)} points. Validation Score (R^2): {score:.4f}") #
        except Exception as e:
            logging.error(f"Failed to retrain PPFInterpolator model: {e}") #

    def _prune_dataset(self):
        """Prunes the dataset to maintain quality and size."""
        max_points = self.interp_config['max_ppf_points'] #
        if len(self.features) > max_points: #
            indices = np.random.choice(len(self.features), max_points, replace=False) #
            self.features = [self.features[i] for i in indices] #
            self.targets = [self.targets[i] for i in indices] #
            logging.info(f"PPF dataset pruned randomly to {max_points} points.")

    def _log_validation(self, score: float):
        """Logs the validation score."""
        self.validation_scores.append(score) #
        if is_dropping(self.validation_scores, window=5): #
            logging.warning(f"PPF validation score has been consistently dropping. Last 5 scores: {[f'{s:.3f}' for s in self.validation_scores[-5:]]}") 