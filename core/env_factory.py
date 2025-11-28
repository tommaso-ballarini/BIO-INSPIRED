import gymnasium as gym
from ocatari.core import OCAtari
from core.wrappers import PacmanHybridWrapper, FreewayOCAtariWrapper, SpaceInvadersOCAtariWrapper

def make_evolution_env(game_name, render_mode=None):
    """
    Crea un ambiente pronto per l'evoluzione.
    CORREZIONE: Istanzia OCAtari passando la stringa del gioco, NON l'oggetto gym.
    """
    # Mappatura nomi
    env_ids = {
        "pacman": "MsPacman-v4",
        "freeway": "Freeway-v4",
        "bankheist": "BankHeist-v4",
        "spaceinvaders": "SpaceInvaders-v4"
    }
    
    if game_name not in env_ids:
        # Fallback per ambienti standard non-OCAtari
        try:
            return gym.make(game_name, render_mode=render_mode)
        except Exception:
            raise ValueError(f"Gioco '{game_name}' non supportato.")

    # --- FIX CRITICO ---
    # Invece di creare l'env con gym.make e passarlo a OCAtari,
    # passiamo direttamente la stringa ID (es. "SpaceInvaders-v4") a OCAtari.
    env = OCAtari(env_ids[game_name], mode="ram", obs_mode="obj", render_mode=render_mode)
    
    # Applicazione Wrapper
    if game_name == "pacman":
        env = PacmanHybridWrapper(env)
    elif game_name == "freeway":
        env = FreewayOCAtariWrapper(env)
    elif game_name == "bankheist":
        pass 
    elif game_name == "spaceinvaders":
        env = SpaceInvadersOCAtariWrapper(env)
        
    return env