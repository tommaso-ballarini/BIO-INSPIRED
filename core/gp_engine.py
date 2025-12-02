import random
import operator
import numpy
import multiprocessing
from deap import base, creator, tools, gp
from core.gp_primitives import setup_primitives
from core.gp_evaluator import run_gp_simulation

def eval_individual(individual, pset):
    # Compila l'albero in una funzione Python
    func = gp.compile(expr=individual, pset=pset)
    # Esegue la simulazione
    return run_gp_simulation(func)

def run_gp_evolution(n_gen=50, pop_size=100, cx_prob=0.5, initial_mut_prob=0.6):
    """
    Esegue l'evoluzione con strategia dinamica (Linear Decay della mutazione).
    """
    # 1. Setup Primitive
    pset = setup_primitives()

    # 2. Setup Creator
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    # 3. Setup Toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", eval_individual, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Bloat Control (limite profondità albero)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # --- CONFIGURAZIONE PARALLELISMO ---
    pool_size = multiprocessing.cpu_count() - 1
    
    # Usiamo il context manager per il Pool
    with multiprocessing.Pool(processes=pool_size) as pool:
        toolbox.register("map", pool.map)

        # 4. Inizializzazione
        print(f"Start Dynamic GP: Pop={pop_size}, Gens={n_gen}, Cores={pool_size}")
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        # Statistiche
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'mut_rate'] + (stats.fields if stats else [])

        # --- VALUTAZIONE GENERAZIONE 0 ---
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), mut_rate=initial_mut_prob, **record)
        print(logbook.stream)

        # --- LOOP EVOLUTIVO MANUALE ---
        for gen in range(1, n_gen + 1):
            # 1. Calcolo Dinamico della Mutazione (Exploration -> Exploitation)
            # La mutazione scende linearmente da 0.6 a 0.1
            progress = gen / n_gen
            mut_prob = initial_mut_prob - (0.5 * progress) 
            mut_prob = max(0.05, mut_prob) # Sicurezza: mai scendere sotto il 5%
            
            # 2. Selezione
            # Selezioniamo l'intera popolazione per generare i figli
            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 3. Applicazione Operatori Genetici (VarAnd custom)
            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < cx_prob:
                    toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            # Mutazione (usando la probabilità dinamica)
            for i in range(len(offspring)):
                if random.random() < mut_prob:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # 4. Valutazione (solo per chi è cambiato)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 5. Sostituzione e Aggiornamento Hall of Fame
            pop[:] = offspring
            hof.update(pop)

            # 6. Logging
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), mut_rate=round(mut_prob, 2), **record)
            print(logbook.stream)

    return pop, logbook, hof, pset