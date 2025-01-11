
import os

CONFIG = {
    'DATA_DIR': 'data',
    'MODELS_DIR': 'models/trained_models',
    'AVAILABLE_BRANDS': ['Hubco', 'Airlink', 'PPL', 'Searl', 'FABL', 'DGKC', 'ATRL', 'GAL', 'FFBL', 'PSO', 'FFL', 'SNGP', 'KOSM', 'BIFO', 'THCCL', 'EFERT', 'PAEL', 'OGDCL'],
    'PREDICTION_INTERVAL': 30,  # minutes
    'PREDICTION_POINTS': 48,    # 24 hours worth of 30-min intervals
}

os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
os.makedirs(CONFIG['MODELS_DIR'], exist_ok=True)