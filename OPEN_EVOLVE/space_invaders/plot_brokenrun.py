import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys
import os
import re

# --- CONFIGURAZIONE UTENTE ---
# Inserisci qui il percorso del tuo file history.csv
CSV_PATH = r"C:\Users\Sport Tech Student\PYTHON_DIRECTORY\BIO-INSPIRED\OPEN_EVOLVE\space_invaders\results\run_si_20260107_153440_best\history.csv"

# --- METADATI MANUALI ---
META_MODEL = "qwen2.5-coder:7b"
META_POP = 2500
META_MAX_GEN = 4000
META_DURATION = "8h 37m 24s"

# Configurazione output
SAVE_PLOT = True
OUTPUT_FILENAME = f"fitness_plot_{META_MODEL.replace(':', '-')}_recovered.png"

def plot_manual_stats(csv_path):
    path = pathlib.Path(csv_path)
    
    if not path.exists():
        print(f"❌ Error: File not found at: {path}")
        print("   Please check the CSV_PATH variable in the script.")
        return

    print(f"Reading file: {path}...")
    
    try:
        df = pd.read_csv(path)
        
        if df.empty:
            print("❌ Error: The CSV file is empty.")
            return

        # Aggiungi colonna progressiva (tentativi)
        df['attempt'] = range(1, len(df) + 1)
        
        # --- LOGICA DI PLOTTING ---
        plt.figure(figsize=(12, 8))
        
        # 1. Filtri Crash vs Validi
        # Soglia -9 come nello script originale
        error_runs = df[df['score'] <= -9]
        valid_runs = df[df['score'] > -9]

        # 2. Scatter plot degli errori (Red X)
        if not error_runs.empty:
            plt.scatter(error_runs['attempt'], error_runs['score'], 
                        c='red', label='Crash / Errors', 
                        alpha=0.3, s=15, marker='x')

        # 3. Scatter plot dei run validi (Blue dots)
        if not valid_runs.empty:
            plt.scatter(valid_runs['attempt'], valid_runs['score'], 
                        c='blue', label='Valid Runs', 
                        alpha=0.6, s=20)

        # 4. Best So Far
        df['best_so_far'] = df['score'].cummax()
        plt.plot(df['attempt'], df['best_so_far'], 
                 c='green', linewidth=2.5, label='Best So Far')

        # --- FORMATTAZIONE TITOLI E ASSI (IN INGLESE) ---
        plt.subplots_adjust(top=0.85)
        
        # Titolo principale
        plt.suptitle(f"Space Invaders Evolution Analysis", fontsize=16, fontweight='bold', y=0.95)
        
        # Sottotitolo con i dati che hai fornito
        subtitle = (f"Model: {META_MODEL} | Pop: {META_POP} | "
                    f"Max Gen: {META_MAX_GEN} | Duration: {META_DURATION}")
        
        plt.title(subtitle, fontsize=10, pad=10, backgroundcolor='#eeeeee')

        # Assi in Inglese
        plt.xlabel("Number of Evaluations")
        plt.ylabel("Fitness (Score)")
        
        # Legenda
        plt.legend(loc='lower right')
        
        # Griglia
        plt.grid(True, alpha=0.3)

        # --- SALVATAGGIO ---
        if SAVE_PLOT:
            output_path = path.parent / OUTPUT_FILENAME
            plt.savefig(output_path, dpi=150)
            print(f"✅ Plot successfully saved to:\n{output_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Fix colori terminale Windows
    if sys.platform == "win32":
        os.system('color')
        
    plot_manual_stats(CSV_PATH)