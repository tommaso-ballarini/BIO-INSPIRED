# FILE: algorithms/neat_runner.py
import neat
import os

def run_neat(eval_function, config_file_path, num_generations=50, output_dir=".", timestamp_str=""):
    """
    Esegue l'algoritmo NEAT.
    
    Args:
        eval_function: La funzione (es. 'eval_genomes') che NEAT chiamerà 
                       per valutare la fitness di ogni genoma.
        config_file_path: Percorso al file di configurazione NEAT.
        num_generations: Per quante generazioni evolvere.
        output_dir: Cartella dove salvare i report.
        timestamp_str: Timestamp per i nomi dei file.
    
    Returns:
        winner (neat.Genome): Il miglior genoma trovato.
        stats (neat.StatisticsReporter): Oggetto con le statistiche.
    """
    
    # 1. Carica la configurazione
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"File di configurazione NEAT non trovato in: {config_file_path}")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file_path)

    # 2. Crea la popolazione
    p = neat.Population(config)

    # 3. Aggiungi i "Reporter" (per log e statistiche)
    stats_filename = os.path.join(output_dir, f"neat_stats_{timestamp_str}.csv")
    p.add_reporter(neat.StdOutReporter(True)) # Stampa su console
    stats = neat.StatisticsReporter()         # Oggetto per raccogliere dati
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(output_dir, f"chk_{timestamp_str}_"))) # Salva checkpoint
    
    print(f"--- 🚀 Avvio Evoluzione NEAT (Generazioni: {num_generations}) ---")
    print(f"Statistiche salvate in: {stats_filename}")
    
    # 4. Esegui l'evoluzione
    # 'eval_function' è la funzione che abbiamo passato (es. eval_genomes)
    winner = p.run(eval_function, num_generations)

    print(f"\n--- 🏆 Evoluzione NEAT Terminata ---")
    print(f"Miglior genoma:\n{winner}")

    return winner, config, stats