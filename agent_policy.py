# ======================================================================
# 🧬 FILE: agent_policy.py (La nostra Policy Parametrica)
# ======================================================================
import numpy as np

# Le dimensioni della nostra rete neurale (Policy)
NUM_FEATURES = 128  # 128 bytes dalla RAM
NUM_ACTIONS = 18    # 18 azioni possibili per BankHeist

# Dimensioni della matrice di pesi (W)
WEIGHT_MATRIX_SHAPE = (NUM_ACTIONS, NUM_FEATURES) # (18, 128)
WEIGHT_MATRIX_SIZE = NUM_ACTIONS * NUM_FEATURES # 18 * 128 = 2304
# Dimensioni del vettore di bias (b)
BIAS_SIZE = NUM_ACTIONS # 18

# Dimensione totale del cromosoma (genoma) = W + b
TOTAL_WEIGHTS = WEIGHT_MATRIX_SIZE + BIAS_SIZE # 2304 + 18 = 2322
def decide_move(game_state, weights):
    """
    Decide una mossa basandosi sui pesi (cromosoma) e sullo stato (RAM).
    
    Args:
        game_state (np.array): Lo stato del gioco (vettore RAM di 128 bytes)
        weights (list o np.array): Il "cromosoma" del nostro GA. 
                                   Deve avere 2304 pesi.
    """
    
    # 1. Normalizza le features (la RAM è 0-255, la portiamo a 0-1)
    #    Questo aiuta la rete a imparare meglio.
    features = game_state / 255.0 
    
    # 2. Logica della Policy
    #    I 'weights' sono una lista di 2304 numeri
    try:
        # Trasforma il genoma in una matrice [18x128]
        # (Righe = Azioni, Colonne = Features)
        weight_matrix = np.array(weights[:-NUM_ACTIONS]).reshape(NUM_ACTIONS, NUM_FEATURES)
        bias = np.array(weights[-NUM_ACTIONS:])
    except ValueError:
        print(f"Errore: i pesi (len {len(weights)}) non corrispondono a {NUM_ACTIONS}x{NUM_FEATURES}")
        return 0 # Mossa di default (NOOP)

    # 3. Calcola un punteggio per ogni mossa (prodotto scalare)
    #    [18x128] @ [128x1] -> [18x1] (un punteggio per ogni azione)
    logit_scores = np.dot(weight_matrix, features) + bias
    scores= np.tanh(logit_scores)
    
    # 4. Scegli la mossa migliore (quella con punteggio più alto)
    return np.argmax(scores) # Restituisce indice 0-17