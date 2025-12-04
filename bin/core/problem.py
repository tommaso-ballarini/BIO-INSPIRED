# FILE: core/problem.py
import numpy as np
from inspyred.ec import Bounder
from functools import partial
import gymnasium as gym
import ale_py

# Importiamo le nostre funzioni core generalizzate
from core.evaluator import run_game_simulation
# NOTA: non importiamo più TOTAL_WEIGHTS da nessuna parte!

# Registra ambienti Atari
gym.register_envs(ale_py)

class GenericGymProblem:
    """
    Un problema inspyred generico per qualsiasi ambiente Gym
    e qualsiasi policy parametrica.
    """
    
    def __init__(self, env_name, policy_class, obs_type="ram", max_steps=1500):
        
        self.env_name = env_name
        self.obs_type = obs_type
        self.max_steps = max_steps
        self.maximize = True # Vogliamo massimizzare il punteggio!
        
        # --- Logica di scoperta automatica ---
        # Creiamo un ambiente "test" solo per scoprire le dimensioni
        try:
            if obs_type:
                test_env = gym.make(env_name, obs_type=obs_type)
            else:
                test_env = gym.make(env_name)
                
            obs_space = test_env.observation_space
            act_space = test_env.action_space
            
            # Per RAM/CartPole è (N,)
            num_features = obs_space.shape[0] 
            # Per azioni discrete
            num_actions = act_space.n 
            
            test_env.close()
            
            print(f"--- Problema Inizializzato ---")
            print(f"Ambiente: {env_name} ({obs_type})")
            print(f"Features: {num_features}, Azioni: {num_actions}")
            
        except Exception as e:
            print(f"Errore fatale nell'inizializzare {env_name}: {e}")
            raise

        # 1. Crea l'istanza della policy
        #    (es. LinearPolicy(num_features=128, num_actions=18))
        self.policy = policy_class(num_features, num_actions)
        
        # 2. Ottieni le dimensioni del genoma DALLA POLICY
        self.dimensions = self.policy.total_weights
        print(f"Dimensioni Genoma: {self.dimensions}")
        
        # 3. Imposta i limiti (bounder)
        self.bounder = Bounder([-1.0] * self.dimensions, [1.0] * self.dimensions)

    def generator(self, random, args):
        """ Genera un singolo individuo (cromosoma) """
        return np.array([random.uniform(-1.0, 1.0) for _ in range(self.dimensions)])

    def evaluator(self, candidates, args):
        """ Valuta una LISTA di candidati """
        fitness_scores = []
        
        for chromosome in candidates:
            # 1. Crea la funzione-agente specifica per questo cromosoma
            #    Usa il metodo .decide_move della *nostra istanza* di policy
            specific_agent_func = partial(self.policy.decide_move, weights=chromosome)
            
            # 2. Esegui la simulazione generica
            fitness, metrics = run_game_simulation(
                agent_decision_function=specific_agent_func,
                env_name=self.env_name,
                max_steps=self.max_steps,
                obs_type=self.obs_type
            )
            
            fitness_scores.append(fitness)
            
        return fitness_scores