import sys
import os
import importlib.util
import json
import shutil
import time
import csv
import gymnasium as gym
from pathlib import Path

# --- FIX IMPORT ---
# Aggiungiamo la cartella superiore al path per trovare il wrapper
current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.abspath(os.path.join(current_dir, '..', 'wrapper'))
sys.path.append(wrapper_dir)

try:
    from freeway_wrapper import FreewayOCAtariWrapper
except ImportError as e:
    print(f"Errore Import Wrapper: {e}")
    sys.exit(1)

from openevolve.evaluation_result import EvaluationResult

# --- CONFIGURAZIONE ---
ENV_NAME = 'Freeway-v4'
MAX_STEPS_PER_GAME = 1500
NUM_GAMES_PER_EVAL = 3

# Setup directory history locale all'esperimento
base_dir = os.path.abspath(os.path.join(current_dir, '..'))
HISTORY_DIR = Path(base_dir) / 'history'
HISTORY_CSV = HISTORY_DIR / 'fitness_history.csv'
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ... (Il resto della logica run_custom_simulation e evaluate rimane uguale) ...
# ... Assicurati solo di usare FreewayOCAtariWrapper importato sopra ...