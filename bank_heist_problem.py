# ======================================================================
# file: bank_heist_problem.py
# ======================================================================

import numpy as np
from inspyred.ec import Bounder
from functools import partial


# Importiamo le nostre funzioni core
from evaluator import run_game_simulation
from agent_policy import decide_move as policy_decide_move
from agent_policy import TOTAL_WEIGHTS # Importiamo la nostra dimensione (2304)

class BankHeistProblem:
    def __init__(self, dimensions):
        # 'dimensions' sarà 2304, passato da run_ga_bankheist.py
        self.dimensions = dimensions 
        self.maximize = True # Vogliamo massimizzare il punteggio!
        
        # Controlla che le dimensioni siano corrette
        if self.dimensions != TOTAL_WEIGHTS:
            print(f"ATTENZIONE: num_vars ({self.dimensions}) non corrisponde a TOTAL_WEIGHTS ({TOTAL_WEIGHTS})")
        
        # Limita ogni peso tra -1.0 e 1.0
        self.bounder = Bounder([-1.0] * self.dimensions, [1.0] * self.dimensions)

    def generator(self, random, args):
        """ Genera un singolo individuo (cromosoma) """
        # Un individuo è un array di 2304 float casuali tra -1 e 1
        return np.array([random.uniform(-1.0, 1.0) for _ in range(self.dimensions)])

    def evaluator(self, candidates, args):
        """ Valuta una LISTA di candidati """
        fitness_scores = []
        
        for chromosome in candidates:
            # 'chromosome' è la lista di 2304 pesi
            
            # 1. Crea la funzione-agente specifica per questo cromosoma
            specific_agent_func = partial(policy_decide_move, weights=chromosome)
            
            # 2. Esegui la simulazione e ottieni il punteggio
            fitness, metrics = run_game_simulation(agent_decision_function=specific_agent_func)
            
            fitness_scores.append(fitness)
            
        return fitness_scores