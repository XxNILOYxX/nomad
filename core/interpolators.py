import logging
import numpy as np
import os
import json
import copy
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from typing import List, Dict, Any

from sklearn.metrics import r2_score

from .utils import is_dropping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

if GPU_AVAILABLE:
    GPU_MODELS = {
        'knn': cuKNN,
        'random_forest': cuRF,
        'ridge': cuRidge
    }
else:
    GPU_MODELS = {}

class DNNRegressor(nn.Module):
    """A simple Deep Neural Network for regression."""
    def __init__(self, input_size=5, output_size=1, dropout_rate=0.2):
        super(DNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BaseInterpolator:
    """Base class for shared interpolator functionality."""
    def __init__(self, config: Dict, use_gpu: bool):
        self.interp_config = config['interpolator']
        self.sim_config = config['simulation']
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        self.targets = []
        self.model_is_ready = False

    def _get_features(self, enrichment_config: List[float]) -> np.ndarray:
        """Calculates features from an enrichment configuration with validation."""
        num_assemblies = self.sim_config['num_assemblies']
        if len(enrichment_config) != num_assemblies:
            raise ValueError(f"Invalid enrichment_config length. Expected {num_assemblies}, got {len(enrichment_config)}.")
        
        arr = np.array(enrichment_config)
        num_central = self.sim_config['num_central_assemblies']
        
        central_configs = arr[:num_central]
        outer_configs = arr[num_central:]
        
        if central_configs.size == 0 or outer_configs.size == 0:
            raise ValueError("Feature calculation error: central or outer assembly group is empty.")

        central_avg = np.mean(central_configs)
        outer_avg = np.mean(outer_configs)
        enrich_std = np.std(arr)
        enrich_grad = central_avg - outer_avg
        max_enrich = np.max(arr)
        
        return np.array([central_avg, outer_avg, enrich_std, enrich_grad, max_enrich])
    
    def predict_batch(self, enrichment_configs: List[List[float]]) -> List[float]:
        """Predicts target values for a batch of enrichment configurations."""
        if not self.model_is_ready:
            if hasattr(self, 'live_features') and len(self.live_features) >= self.interp_config['min_interp_points']:
                logging.warning(f"{self.__class__.__name__} is not ready. Attempting best-effort prediction with live data.")
                return self._best_effort_prediction(enrichment_configs)
            else:
                default_val = self._get_fallback_value()
                logging.warning(f"{self.__class__.__name__} is not ready and has insufficient data. Returning default value: {default_val}")
                return [default_val] * len(enrichment_configs)

        try:
            if hasattr(self, 'regressor_type') and self.regressor_type == 'dnn':
                return self._predict_dnn(enrichment_configs)
            else:
                return self._predict_sklearn(enrichment_configs)
        except Exception as e:
            logging.error(f"An unexpected error occurred during batch prediction: {e}")
            return [self._get_fallback_value()] * len(enrichment_configs)

    def _get_fallback_value(self) -> float:
        """Returns a configurable or default fallback value."""
        if isinstance(self, KeffInterpolator):
            return self.sim_config.get('target_keff', 1.0)
        else:
            return self.interp_config.get('fallback_ppf_value', 2.0)

    def _predict_dnn(self, enrichment_configs):
        """Handles prediction logic specifically for the DNN model."""
        try:
            feature_vectors = np.array([self._get_features(config) for config in enrichment_configs])
            X_scaled = self.scaler.transform(feature_vectors)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten().tolist()
        except Exception as e:
            logging.error(f"An unexpected error occurred during DNN batch prediction: {e}")
            return [self._get_fallback_value()] * len(enrichment_configs)

    def _predict_sklearn(self, enrichment_configs):
        """Handles prediction logic for scikit-learn/cuML models."""
        X_scaled_gpu, predictions_gpu = None, None
        try:
            feature_vectors = np.array([self._get_features(config) for config in enrichment_configs])
            X_scaled = self.scaler.transform(feature_vectors)
            
            if self.use_gpu:
                X_scaled_gpu = cupy.asarray(X_scaled)
                predictions_gpu = self.model.predict(X_scaled_gpu)
                return cupy.asnumpy(predictions_gpu).tolist()
            else:
                return self.model.predict(X_scaled).tolist()
        finally:
            if X_scaled_gpu is not None: del X_scaled_gpu
            if predictions_gpu is not None: del predictions_gpu

    def predict(self, enrichment_config: List[float]) -> float:
        return self.predict_batch([enrichment_config])[0]

    def _save_dataset_to_file(self, features: List[np.ndarray], targets: List[float], file_path: str):
        data_dir = os.path.dirname(file_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        data = {'features': [f.tolist() for f in features], 'targets': targets}
        temp_filepath = file_path + ".tmp"
        
        try:
            with open(temp_filepath, 'w') as f:
                json.dump(data, f, indent=4)
            
            # This is the atomic operation
            os.replace(temp_filepath, file_path)
            
        except (IOError, TypeError) as e:
            logging.error(f"Error saving interpolator data to {file_path}: {e}")
        finally:
            # Clean up the temp file if it still exists after an error
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
    def _load_dataset_from_file(self, file_path: str) -> (List[np.ndarray], List[float]):
        if not os.path.exists(file_path):
            return [], []
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            features = [np.array(f) for f in data['features']]
            targets = data['targets']
            return features, targets
        except (IOError, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error loading interpolator data from {file_path}: {e}")
        return [], []

class KeffInterpolator(BaseInterpolator):
    """Interpolator for predicting k-effective (keff). Uses a single dataset."""
    def __init__(self, config: Dict, use_gpu: bool):
        super().__init__(config, use_gpu)
        self.file_path = self.sim_config['keff_interp_file']
        n_neighbors = self.interp_config['n_neighbors']
        model_class = GPU_MODELS.get('knn') if self.use_gpu else CPU_MODELS.get('knn')
        self.model = model_class(n_neighbors=n_neighbors)
        self.lock = threading.Lock()

    def add_data_point(self, enrichment_config: List[float], target_value: float):
        with self.lock:
            feature_vector = self._get_features(enrichment_config)
            self.features.append(feature_vector)
            self.targets.append(target_value)
            self.retrain()
    
    def load_data(self):
        with self.lock:
            self.features, self.targets = self._load_dataset_from_file(self.file_path)
            if self.features:
                self.retrain()
                return True
        return False

    def save_data(self):
        with self.lock:
            self._save_dataset_to_file(self.features, self.targets, self.file_path)

    def retrain(self):
        if len(self.features) < self.interp_config['n_neighbors']:
            self.model_is_ready = False
            return
        
        X = np.array(self.features)
        y = np.array(self.targets)
        max_points = self.interp_config['max_keff_points']
        if len(X) > max_points:
            np.random.seed(self.interp_config['nn_random_seed'] + 1) # Use derived seed
            indices = np.random.choice(len(X), max_points, replace=False)
            self.features = [self.features[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]
            X, y = np.array(self.features), np.array(self.targets)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_scaled_gpu, y_gpu = None, None
        try:
            if self.use_gpu:
                X_scaled_gpu = cupy.asarray(X_scaled)
                y_gpu = cupy.asarray(y)
                self.model.fit(X_scaled_gpu, y_gpu)
            else:
                self.model.fit(X_scaled, y)
            
            self.model_is_ready = True
            logging.info(f"KeffInterpolator retrained with {len(self.features)} points.")
        except Exception as e:
            self.model_is_ready = False
            logging.error(f"Failed to retrain KeffInterpolator model: {e}")
        finally:
            if X_scaled_gpu is not None: del X_scaled_gpu
            if y_gpu is not None: del y_gpu

class PPFInterpolator(BaseInterpolator):
    """Interpolator for PPF using a "Live" vs "Best" dataset strategy."""
    def __init__(self, config: Dict, use_gpu: bool):
        super().__init__(config, use_gpu)
        self.live_file_path = self.sim_config['ppf_interp_file']
        self.best_file_path = self.sim_config['ppf_interp_file_best']
        self.live_features, self.live_targets = [], []
        self.regressor_type = self.interp_config['regressor_type']
        self.validation_scores = []
        self.lock = threading.Lock() # For thread safety
        self._setup_model()

    def _setup_model(self):
        if self.regressor_type == 'dnn':
            torch.manual_seed(self.interp_config['nn_random_seed'])
            self.model = DNNRegressor(input_size=5, dropout_rate=self.interp_config['nn_dropout_rate'])
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.criterion = nn.MSELoss()
        else:
            n_neighbors = self.interp_config['n_neighbors']
            model_class = GPU_MODELS.get(self.regressor_type) if self.use_gpu else CPU_MODELS.get(self.regressor_type)
            self.model = model_class(n_neighbors=n_neighbors) if self.regressor_type == 'knn' else model_class()

    def add_data_point(self, enrichment_config: List[float], target_value: float):
        with self.lock:
            feature_vector = self._get_features(enrichment_config)
            self.live_features.append(feature_vector)
            self.live_targets.append(target_value)
            self.retrain()

    def load_data(self):
        with self.lock:
            self.live_features, self.live_targets = self._load_dataset_from_file(self.live_file_path)
            best_features, best_targets = self._load_dataset_from_file(self.best_file_path)
            
            if best_features:
                logging.info(f"Found 'best' dataset with {len(best_features)} points. Training model.")
                if self._train_on_best_data(best_features, best_targets):
                    self.features = best_features
                    self.targets = best_targets
                return True
            elif self.live_features:
                logging.info("No 'best' dataset found. Attempting to validate and promote 'live' dataset.")
                self.retrain()
                return True
        return False

    def save_data(self):
        with self.lock:
            self._save_dataset_to_file(self.live_features, self.live_targets, self.live_file_path)
            self._save_dataset_to_file(self.features, self.targets, self.best_file_path)

    def _best_effort_prediction(self, enrichment_configs: List[List[float]]) -> List[float]:
        temp_model = self._create_new_model_instance()
        temp_scaler = StandardScaler()
        X_live = np.array(self.live_features)
        y_live = np.array(self.live_targets)
        
        try:
            X_live_scaled = temp_scaler.fit_transform(X_live)
            self._train_temp_model(temp_model, X_live_scaled, y_live)
            
            feature_vectors = np.array([self._get_features(config) for config in enrichment_configs])
            X_scaled_pred = temp_scaler.transform(feature_vectors)
            
            if self.regressor_type == 'dnn':
                X_tensor = torch.tensor(X_scaled_pred, dtype=torch.float32).to(self.device)
                temp_model.eval()
                with torch.no_grad():
                    predictions = temp_model(X_tensor)
                return predictions.cpu().numpy().flatten().tolist()
            else:
                return temp_model.predict(X_scaled_pred).tolist()
        except Exception as e:
            logging.error(f"Best-effort prediction failed: {e}. Returning default value.")
            return [self._get_fallback_value()] * len(enrichment_configs)

    def _train_on_best_data(self, features, targets):
        if not features:
            self.model_is_ready = False
            return False

        temp_model = self._create_new_model_instance()
        temp_scaler = StandardScaler()
        X = np.array(features)
        y = np.array(targets)
        
        try:
            X_scaled = temp_scaler.fit_transform(X)
            self._train_temp_model(temp_model, X_scaled, y, use_early_stopping=True)
            
            self.model = temp_model
            self.scaler = temp_scaler
            self.model_is_ready = True
            logging.info(f"Successfully trained new PPF model on dataset with {len(features)} points.")
            return True
        except Exception as e:
            logging.error(f"Failed to train new PPF model: {e}. The old model (if any) will be kept.")
            return False

    def retrain(self):
        self._prune_live_dataset()
        min_points_for_cv = max(self.interp_config['min_interp_points'], 6)
        if len(self.live_features) < min_points_for_cv:
            return

        X_live = np.array(self.live_features)
        y_live = np.array(self.live_targets)
        
        score = -1.0
        try:
            if self.regressor_type == 'dnn':
                score = self._validate_dnn_with_cv(X_live, y_live)
            else:
                val_model_class = CPU_MODELS.get(self.regressor_type)
                temp_model = val_model_class(n_neighbors=self.interp_config['n_neighbors']) if self.regressor_type == 'knn' else val_model_class()
                X_scaled_val = StandardScaler().fit_transform(X_live)
                scores = cross_val_score(temp_model, X_scaled_val, y_live, cv=5, scoring='r2')
                score = np.mean(scores)
        except Exception as e:
            logging.error(f"Validation of live dataset failed unexpectedly: {e}")
        
        self._log_validation(score)
        
        if score >= self.interp_config['min_validation_score']:
            logging.info(f"Live dataset passed validation with R^2 = {score:.4f}. Attempting to promote and train.")
            
            if self._train_on_best_data(self.live_features, self.live_targets):
                self.features = copy.deepcopy(self.live_features)
                self.targets = copy.deepcopy(self.live_targets)
                logging.info("Promotion of live dataset to best dataset successful.")
            else:
                logging.error("Model training failed. Live dataset was not promoted.")
        else:
            logging.info(f"Live dataset failed validation with R^2 = {score:.4f}. Model will continue using the old 'best' dataset.")

    def _prune_live_dataset(self):
        max_points = self.interp_config['max_ppf_points']
        if len(self.live_features) > max_points:
            np.random.seed(self.interp_config['nn_random_seed'] + 2) # Use derived seed
            indices = np.random.choice(len(self.live_features), max_points, replace=False)
            self.live_features = [self.live_features[i] for i in indices]
            self.live_targets = [self.live_targets[i] for i in indices]

    def _log_validation(self, score: float):
        self.validation_scores.append(score)
        if is_dropping(self.validation_scores, window=5):
            logging.warning(f"PPF live data validation score has been consistently dropping. Last 5 scores: {[f'{s:.3f}' for s in self.validation_scores[-5:]]}")

    def _validate_dnn_with_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        kf = KFold(n_splits=5, shuffle=True, random_state=self.interp_config['nn_random_seed'])
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            validation_model, optimizer = None, None
            try:
                validation_model = self._create_new_model_instance()
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                fold_scaler = StandardScaler()
                X_train_scaled = fold_scaler.fit_transform(X_train)
                X_val_scaled = fold_scaler.transform(X_val)
                
                self._train_temp_model(validation_model, X_train_scaled, y_train, X_val_scaled, y_val, use_early_stopping=True)
                
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                validation_model.eval()
                with torch.no_grad():
                    final_preds = validation_model(X_val_tensor)
                score = r2_score(y_val, final_preds.cpu().numpy())
                fold_scores.append(score)
            except Exception as e:
                logging.error(f"Error in CV fold {fold+1}: {e}")
                fold_scores.append(-1.0)
            finally:
                if validation_model is not None and self.use_gpu and torch.cuda.is_available():
                    del validation_model
                    torch.cuda.empty_cache()
        return np.mean(fold_scores) if fold_scores else -1.0

    def _create_new_model_instance(self):
        """Helper to create a fresh model instance."""
        if self.regressor_type == 'dnn':
            return DNNRegressor(input_size=5, dropout_rate=self.interp_config['nn_dropout_rate']).to(self.device)
        else:
            n_neighbors = self.interp_config['n_neighbors']
            model_class = GPU_MODELS.get(self.regressor_type) if self.use_gpu else CPU_MODELS.get(self.regressor_type)
            return model_class(n_neighbors=n_neighbors) if self.regressor_type == 'knn' else model_class()

    def _train_temp_model(self, model, X_train, y_train, X_val=None, y_val=None, use_early_stopping=False):
        """A generic training loop for a temporary model (DNN or sklearn)."""
        if self.regressor_type == 'dnn':
            optimizer = optim.Adam(model.parameters(), lr=self.interp_config['nn_learning_rate'])
            criterion = nn.MSELoss()
            dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device))
            loader = DataLoader(dataset, batch_size=self.interp_config['nn_batch_size'], shuffle=True)
            
            epochs_no_improve, best_val_loss = 0, float('inf')
            
            for epoch in range(self.interp_config['nn_epochs']):
                model.train()
                for inputs, labels in loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                if use_early_stopping and X_val is not None:
                    model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(self.device)
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= self.interp_config['nn_patience']:
                        logging.debug(f"Early stopping at epoch {epoch+1}.")
                        break
        else:
            # Sklearn/cuML models
            X_train_gpu, y_train_gpu = None, None
            try:
                if self.use_gpu:
                    X_train_gpu = cupy.asarray(X_train)
                    y_train_gpu = cupy.asarray(y_train)
                    model.fit(X_train_gpu, y_train_gpu)
                else:
                    model.fit(X_train, y_train)
            finally:
                if X_train_gpu is not None: del X_train_gpu
                if y_train_gpu is not None: del y_train_gpu
