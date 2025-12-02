import sys
import os
import pathlib
import pickle
import numpy
import multiprocessing
from deap import base, creator, tools, gp

# Setup percorsi
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from core.gp_primitives import setup_space_invaders_primitives
from openevolve_experiments.gp_evaluator_si import run_gp_si_simulation

# Configurazione Checkpoint
CHECKPOINT_FREQ = 5  # Salva ogni 5 generazioni
HISTORY_DIR = project_root / 'evolution_history'
CHECKPOINT_FILE = HISTORY_DIR / "si_checkpoint.pkl"

def eval_individual(individual, pset):
    func = gp.compile(expr=individual, pset=pset)
    # Media di 3 episodi per robustezza
    scores = [run_gp_si_simulation(func)[0] for _ in range(3)]
    return (sum(scores) / len(scores)),

def save_checkpoint(pop, log, hof, gen):
    """Salva lo stato completo dell'evoluzione"""
    cp = dict(population=pop, generation=gen, halloffame=hof, logbook=log, rndstate=numpy.random.get_state())
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(cp, f)
    print(f"💾 Checkpoint salvato alla gen {gen}")

def save_best_code(hof_individual, filename="best_si_agent.py"):
    """Salva il codice Python dell'agente migliore"""
    tree_code = str(hof_individual)
    content = f"""
import math
import operator

def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def neg(a): return -a
def lt(a, b): return 1.0 if a < b else 0.0
def gt(a, b): return 1.0 if a > b else 0.0
def protected_div(l, r): return 1 if abs(r) < 1e-6 else l/r
def if_then_else(i, o1, o2): return o1 if i else o2

def get_action(observation):
    # Inputs (8 features)
    player_x, alien_dx, alien_dy, bullet_dx, bullet_dy, density, shielded, aiming = observation
    
    try:
        val = {tree_code}
    except:
        val = 0
    
    if val < -0.5: return 3 # LEFT
    if val > 0.5: return 2  # RIGHT
    return 1                # FIRE
"""
    with open(HISTORY_DIR / filename, "w") as f:
        f.write(content)
    print(f"🏆 Miglior Agente salvato in: {filename}")

def run_robust():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # 1. Setup GP
    pset = setup_space_invaders_primitives()
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_individual, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 2. Ripristino Checkpoint (se esiste)
    if os.path.exists(CHECKPOINT_FILE):
        print("🔄 Trovato checkpoint! Riprendo l'evoluzione...")
        with open(CHECKPOINT_FILE, "rb") as f:
            cp = pickle.load(f)
        pop = cp["population"]
        start_gen = cp["generation"] + 1
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        numpy.random.set_state(cp["rndstate"])
    else:
        print("🌱 Nuova Evoluzione Iniziata.")
        pop = toolbox.population(n=300) # Popolazione aumentata come richiesto
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'avg', 'max']

    # Statistiche
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    # 3. Loop Evolutivo Manuale (Safe)
    N_GEN = 50
    pool_size = max(1, multiprocessing.cpu_count() - 1)

    try:
        with multiprocessing.Pool(processes=pool_size) as pool:
            toolbox.register("map", pool.map)
            
            # Valuta generazione iniziale se nuova
            if start_gen == 0:
                fitnesses = list(toolbox.map(toolbox.evaluate, pop))
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit
                hof.update(pop)
                record = stats.compile(pop)
                logbook.record(gen=0, nevals=len(pop), **record)
                print(logbook.stream)

            for gen in range(start_gen, N_GEN + 1):
                # Selezione e Cloni
                offspring = toolbox.select(pop, len(pop))
                offspring = [toolbox.clone(ind) for ind in offspring]

                # Operatori (Crossover & Mutazione)
                # Strategia Dinamica semplificata
                mut_prob = 0.5 if gen < 20 else 0.2
                
                for i in range(1, len(offspring), 2):
                    if numpy.random.random() < 0.5:
                        toolbox.mate(offspring[i - 1], offspring[i])
                        del offspring[i - 1].fitness.values
                        del offspring[i].fitness.values

                for i in range(len(offspring)):
                    if numpy.random.random() < mut_prob:
                        toolbox.mutate(offspring[i])
                        del offspring[i].fitness.values

                # Valutazione
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Aggiornamento Popolazione
                pop[:] = offspring
                hof.update(pop)

                # Log e Checkpoint
                record = stats.compile(pop)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                print(logbook.stream)

                if gen % CHECKPOINT_FREQ == 0:
                    save_checkpoint(pop, logbook, hof, gen)
                    save_best_code(hof[0]) # Salva anche il codice intermedio

    except KeyboardInterrupt:
        print("\n🛑 Interruzione Manuale rilevata! Salvataggio in corso...")
        save_checkpoint(pop, logbook, hof, gen)
        save_best_code(hof[0])
        print("✅ Salvataggio completato. Puoi spegnere.")
        sys.exit(0)

    # Salvataggio Finale
    save_best_code(hof[0])
    print("\n🏁 Evoluzione Completata!")

if __name__ == "__main__":
    run_robust()