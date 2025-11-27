# openevolve_experiments/initial_agent.py
import numpy as np

def get_action(observation):
    """
    Initial Kamikaze Agent.
    It sees the data, but it ignores it and just runs forward.
    The Evolution must figure out how to use 'observation' to survive.
    """
    # observation è l'array di 11 float, ma noi lo ignoriamo per ora.
    
    # Restituisce sempre 1 (UP)
    # L'agente correrà e verrà investito ripetutamente.
    # Score atteso: Basso/Medio (basato solo sulla fortuna).
    return 1