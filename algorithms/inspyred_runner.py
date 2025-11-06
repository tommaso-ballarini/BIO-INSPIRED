# ======================================================================
# 🏛️ FILE: algorithms/inspyred_runner.py (Versione aggiornata)
# ======================================================================
from pylab import *
from inspyred import ec
import functools

# Importa i plotter dalla nuova posizione
from utils.lab_plotting_utils import plot_observer, plot_results_1D, plot_results_2D

# 'generator' e 'initial_pop_observer' possono essere rimossi
# da qui se sono già gestiti altrove, ma per sicurezza
# teniamo 'initial_pop_observer' che serve per i plot 1D/2D.
# Il 'generator' non serve più qui, perché ora è definito
# DENTRO la classe GenericGymProblem.

def initial_pop_observer(population, num_generations, num_evaluations,
                         args):
    """ Salva la popolazione iniziale per i plot 1D/2D """
    if num_generations == 0 :
        # Assicurati che il dizionario esista
        if "initial_pop_storage" not in args:
             args["initial_pop_storage"] = {}
             
        args["initial_pop_storage"]["individuals"] = asarray([guy.candidate
                                                             for guy in population])
        args["initial_pop_storage"]["fitnesses"] = asarray([guy.fitness
                                                            for guy in population])

# --- MODIFICA CHIAVE ---
# La funzione ora riceve 'problem' (un'istanza)
# invece di 'problem_class' e 'num_vars'.
def run_ga(random, problem, display=False, **kwargs) :
    
    # 'problem' è GIÀ un'istanza (es. GenericGymProblem)
    # Otteniamo le info da lì
    num_vars = problem.dimensions
    
    # Questo dizionario serve per i plot 1D/2D
    initial_pop_storage = {}
    
    # Inizializza l'algoritmo
    algorithm = ec.EvolutionaryComputation(random)
    algorithm.terminator = ec.terminators.generation_termination
    algorithm.replacer = ec.replacers.generational_replacement
    algorithm.variator = [ec.variators.uniform_crossover, ec.variators.gaussian_mutation]
    algorithm.selector = ec.selectors.tournament_selection

    if display :
        algorithm.observer = [plot_observer, initial_pop_observer]
    else :
        algorithm.observer = initial_pop_observer

    # --- Logica Semplificata ---
    
    # Il generatore è ora definito nel problema
    kwargs["generator"] = problem.generator
    
    # Il bounder è ora definito nel problema
    if kwargs.get("use_bounder", True) and hasattr(problem, 'bounder'):
        kwargs["bounder"] = problem.bounder
    else:
        kwargs.pop("bounder", None) # Rimuovi se non usato

    # 'num_selected' è richiesto dal tournament_selection
    # Assicurati che sia presente, altrimenti usa pop_size
    if "num_selected" not in kwargs:
         kwargs["num_selected"]=kwargs.get("pop_size", 100) # Default ragionevole


    # --- Esecuzione ---
    final_pop = algorithm.evolve(
        evaluator=problem.evaluator,
        maximize=problem.maximize, # Preso dal problema
        initial_pop_storage=initial_pop_storage,
        # num_vars non serve più qui, è usato solo da kwargs
        **kwargs 
    )

    # --- Raccolta Risultati ---
    final_pop_fitnesses = asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = asarray([guy.candidate for guy in final_pop])

    # Ordina i risultati
    if problem.maximize:
        sort_indexes = sorted(range(len(final_pop_fitnesses)), key=final_pop_fitnesses.__getitem__, reverse=True)
    else:
        sort_indexes = sorted(range(len(final_pop_fitnesses)), key=final_pop_fitnesses.__getitem__)

    final_pop_fitnesses = final_pop_fitnesses[sort_indexes]
    final_pop_candidates = final_pop_candidates[sort_indexes]
    
    best_guy = final_pop_candidates[0]
    best_fitness = final_pop_fitnesses[0]

    # --- Plotting (per problemi 1D e 2D) ---
    if display and "initial_pop_storage" in kwargs:
        if num_vars == 1 :
            plot_results_1D(problem, initial_pop_storage["individuals"],
                            initial_pop_storage["fitnesses"],
                            final_pop_candidates, final_pop_fitnesses,
                            'Initial Population', 'Final Population', kwargs)

        elif num_vars == 2 :
            plot_results_2D(problem, initial_pop_storage["individuals"],
                            final_pop_candidates, 'Initial Population',
                            'Final Population', kwargs)
        # Per num_vars > 2 (come il nostro 2322), questi plot
        # verranno saltati, ma plot_observer (il trend) funzionerà.

    return best_guy, best_fitness, final_pop