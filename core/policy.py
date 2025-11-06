# FILE: core/policy.py
import numpy as np

class LinearPolicy:
    """
    Una policy parametrica lineare (Matrice W + Vettore b)
    con attivazione tanh.
    """
    
    def __init__(self, num_features, num_actions):
        """
        Inizializza la policy in base alle dimensioni dell'ambiente.
        
        Args:
            num_features (int): Dimensione dello spazio di osservazione (es. 4 per CartPole, 128 per RAM)
            num_actions (int): Numero di azioni discrete (es. 2 per CartPole, 18 per BankHeist)
        """
        self.num_features = num_features
        self.num_actions = num_actions
        
        # Dimensioni della matrice di pesi (W)
        self.weight_matrix_shape = (self.num_actions, self.num_features)
        self.weight_matrix_size = self.num_actions * self.num_features
        
        # Dimensioni del vettore di bias (b)
        self.bias_size = self.num_actions
        
        # Dimensione totale del cromosoma (genoma) = W + b
        self.total_weights = self.weight_matrix_size + self.bias_size

    def decide_move(self, game_state, weights):
        """
        Decide una mossa basandosi sui pesi (cromosoma) e sullo stato.
        """
        
        # 1. Normalizza le features
        # Nota: game_state per Atari RAM è 0-255, ma per CartPole è
        #       un range diverso (es. -4.8 a 4.8).
        #       Per ora, normalizzare 0-255 va bene per Atari.
        #       Per CartPole, la normalizzazione potrebbe non essere necessaria
        #       o essere diversa. Per semplicità la lasciamo.
        if np.max(game_state) > 1.0:
             features = game_state / 255.0
        else:
             features = game_state # Già normalizzato o in range (es. CartPole)

        
        # 2. Estrai W e b dal genoma
        try:
            w_flat = weights[:self.weight_matrix_size]
            b = weights[self.weight_matrix_size:]
            
            weight_matrix = np.array(w_flat).reshape(self.weight_matrix_shape)
            bias = np.array(b)
            
        except (ValueError, IndexError):
            print(f"Errore: i pesi (len {len(weights)}) non corrispondono a {self.total_weights}")
            return 0 # Mossa di default (NOOP)

        # 3. Calcola i punteggi
        logit_scores = np.dot(weight_matrix, features) + bias
        scores = np.tanh(logit_scores)
        
        # 4. Scegli la mossa migliore
        return np.argmax(scores)