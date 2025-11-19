# core/env_factory.py
import gymnasium as gym
from ocatari.core import OCAtari
from core.wrappers import NeatMsPacmanWrapper, NeatFreewayWrapper # etc

def make_evolution_env(game_name, render_mode=None):
    """
    Crea un ambiente pronto per l'evoluzione (vettorializzato).
    """
    # Mappatura semplice nome -> ambiente gym
    env_ids = {
        "pacman": "MsPacman-v4",
        "freeway": "Freeway-v4",
        "bankheist": "BankHeist-v4"
    }
    
    env = gym.make(env_ids[game_name], render_mode=render_mode)
    
    # Applica sempre OCAtari in modalità RAM per efficienza
    env = OCAtari(env, mode="ram", obs_mode="obj")
    
    # Selettore del Wrapper specifico per il gioco
    if game_name == "pacman":
        env = NeatMsPacmanWrapper(env)
    elif game_name == "freeway":
        # Implementerai questo basandoti sul report (array distanze auto)
        pass 
        
    return env