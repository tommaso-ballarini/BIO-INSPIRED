# experiments_openevolve/initial_program_ea.py
import random
import numpy as np # L'LLM può usarlo se vuole

def evolve_neural_network(problem, max_generations=10, pop_size=20):
    """
    Evolve i pesi di una rete neurale per un dato problema.
    
    'problem' ha:
    - problem.evaluate(genome) -> {'fitness': ...}
    - problem.get_ind_size() -> int
    """
    
    # Algoritmo iniziale: Ricerca Casuale (molto stupido)
    best_genome = None
    best_fitness = -float('inf')
    
    num_evals = max_generations * pop_size
    genome_size = problem.dimensions # Chiedi la dimensione al problema
    
    for _ in range(num_evals):
        
        # Genera un genoma casuale
        genome = [random.uniform(-1.0, 1.0) for _ in range(genome_size)]
        
        # Valuta
        # Questo chiama il tuo GenericGymProblem, che a sua volta
        # usa il tuo core/evaluator.py per giocare la partita!
        result = problem.evaluate(genome) 
        fitness = result['fitness']
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome
            
    # Restituisce la fitness migliore trovata da questo ALGORITMO
    return {"best_fitness_achieved": best_fitness}