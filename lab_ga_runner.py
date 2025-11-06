# ======================================================================
# file: lab_ga_runner.py
# ======================================================================
from pylab import *
from inspyred import ec, benchmarks
import functools

# Importa i plotter dal file che abbiamo appena creato
from lab_plotting_utils import plot_observer, plot_results_1D, plot_results_2D

# (Il resto della tua Cella 3 è qui, incluse 'generator' e 'initial_pop_observer')

def generator(random, args):
    return asarray([random.uniform(args["pop_init_range"][0],
                                    args["pop_init_range"][1])
                    for _ in range(args["num_vars"])])

def initial_pop_observer(population, num_generations, num_evaluations,
                         args):
    if num_generations == 0 :
        args["initial_pop_storage"]["individuals"] = asarray([guy.candidate
                                                            for guy in population])
        args["initial_pop_storage"]["fitnesses"] = asarray([guy.fitness
                                                          for guy in population])

def run_ga(random, display=False, num_vars=0, problem_class=benchmarks.Sphere,
           maximize=False, use_bounder=True, **kwargs) :

    initial_pop_storage = {}

    algorithm = ec.EvolutionaryComputation(random)
    algorithm.terminator = ec.terminators.generation_termination
    algorithm.replacer = ec.replacers.generational_replacement
    algorithm.variator = [ec.variators.uniform_crossover,ec.variators.gaussian_mutation]
    algorithm.selector = ec.selectors.tournament_selection

    if display :
        algorithm.observer = [plot_observer,initial_pop_observer]
    else :
        algorithm.observer = initial_pop_observer

    kwargs["num_selected"]=kwargs.get("pop_size", 10) # Aggiunto default

    problem = problem_class(num_vars)

    if use_bounder and hasattr(problem, 'bounder'):
        kwargs["bounder"]=problem.bounder
    else:
        # Rimuovi bounder se non esiste o non è richiesto
        kwargs.pop("bounder", None) 
        
    if "pop_init_range" in kwargs :
        kwargs["generator"]=generator
    elif hasattr(problem, 'generator'):
        kwargs["generator"]=problem.generator
    else:
        # Fallback se 'problem.generator' non esiste
        kwargs["pop_init_range"] = [-1.0, 1.0] # Un default ragionevole
        kwargs["generator"]=generator


    final_pop = algorithm.evolve(evaluator=problem.evaluator,
                                 maximize=problem.maximize,
                                 initial_pop_storage=initial_pop_storage,
                                 num_vars=num_vars,
                                 **kwargs)

    final_pop_fitnesses = asarray([guy.fitness for guy in final_pop])
    final_pop_candidates = asarray([guy.candidate for guy in final_pop])

    # Modifica per ordinare correttamente (massimizzazione vs minimizzazione)
    if problem.maximize:
        sort_indexes = sorted(range(len(final_pop_fitnesses)), key=final_pop_fitnesses.__getitem__, reverse=True)
    else:
        sort_indexes = sorted(range(len(final_pop_fitnesses)), key=final_pop_fitnesses.__getitem__)

    final_pop_fitnesses = final_pop_fitnesses[sort_indexes]
    final_pop_candidates = final_pop_candidates[sort_indexes]
    
    best_guy = final_pop_candidates[0]
    best_fitness = final_pop_fitnesses[0]

    if display :
        # Questi plot verranno eseguiti solo se num_vars è 1 o 2
        # Nel nostro caso (num_vars=20), verranno saltati,
        # ma il plot 'fitness trend' (da plot_observer) funzionerà.
        if num_vars == 1 :
            plot_results_1D(problem, initial_pop_storage["individuals"],
                            initial_pop_storage["fitnesses"],
                            final_pop_candidates, final_pop_fitnesses,
                            'Initial Population', 'Final Population', kwargs)

        elif num_vars == 2 :
            plot_results_2D(problem, initial_pop_storage["individuals"],
                            final_pop_candidates, 'Initial Population',
                            'Final Population', kwargs)

    return best_guy, best_fitness, final_pop