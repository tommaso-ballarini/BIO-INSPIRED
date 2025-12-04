# core/fitness_pacman.py
"""
Fitness Shaping Multi-Componente per Pac-Man secondo il Report.
Implementa: Potential-Based Reward, Penalità Anti-Camping, Bonus Aggressività.
"""

import numpy as np
import math

class PacmanFitnessCalculator:
    """
    Calcola fitness usando la formula multi-componente:
    F = α·Score + β·T_alive + γ·Explore + δ·Ghost_Bonus - ε·Penalty
    
    Con Potential-Based Reward Shaping per il cibo.
    """
    
    def __init__(self, 
                 w_score=1.0,           # Peso score grezzo
                 w_survival=0.01,       # Peso sopravvivenza (logaritmico)
                 w_exploration=2.0,     # Peso esplorazione
                 w_ghost_bonus=10.0,    # Peso mangiare fantasmi
                 w_potential=0.5,       # Peso potential-based (avvicinamento cibo)
                 penalty_death=50.0,    # Penalità morte
                 penalty_camping=0.05,  # Penalità per stazionamento
                 camping_threshold=100  # Frame senza progresso = camping
                ):
        
        self.w_score = w_score
        self.w_survival = w_survival
        self.w_exploration = w_exploration
        self.w_ghost_bonus = w_ghost_bonus
        self.w_potential = w_potential
        self.penalty_death = penalty_death
        self.penalty_camping = penalty_camping
        self.camping_threshold = camping_threshold
        
        # State tracking (per episodio)
        self.reset_episode()
    
    def reset_episode(self):
        """Reset stato all'inizio di ogni episodio."""
        self.total_fitness = 0.0
        self.visited_sectors = set()
        self.frames_no_progress = 0
        self.last_significant_event_frame = 0
        self.ghosts_eaten = 0
        self.prev_nearest_pellet_dist = None
        self.current_frame = 0
    
    def _get_potential(self, player_pos, pellets):
        """
        Calcola il potenziale Φ(s) = -distanza_min_cibo.
        Usato per Potential-Based Reward Shaping:
        R_shaped = γ·Φ(s_{t+1}) - Φ(s_t)
        """
        if not pellets:
            return 0.0
        
        p_x, p_y = player_pos
        min_dist = min([math.sqrt((p.x - p_x)**2 + (p.y - p_y)**2) for p in pellets])
        
        # Potenziale negativo: più lontano = più negativo
        return -min_dist / 160.0  # Normalizzato
    
    def update(self, observation, reward, objects, done):
        """
        Aggiorna fitness ad ogni step.
        
        Args:
            observation: Vettore feature dal wrapper
            reward: Reward grezzo dall'ambiente
            objects: Lista oggetti OCAtari (per calcolare potenziale)
            done: Bool se episodio terminato
        
        Returns:
            step_fitness: Contributo di fitness di questo step
        """
        self.current_frame += 1
        step_fitness = 0.0
        
        # --- 1. SCORE COMPONENT ---
        if reward > 0:
            step_fitness += self.w_score * reward
            self.frames_no_progress = 0  # Reset timer camping
            
            # Bonus extra se ha mangiato un fantasma (reward >= 200)
            if reward >= 200:
                self.ghosts_eaten += 1
                step_fitness += self.w_ghost_bonus * 50.0  # Boost aggressività
        
        # --- 2. SURVIVAL COMPONENT (Logaritmico per rendimenti decrescenti) ---
        step_fitness += self.w_survival * math.log(1 + self.current_frame / 100.0)
        
        # --- 3. EXPLORATION COMPONENT ---
        # Estrai posizione player dal vettore (primi 2 valori)
        p_x_norm = observation[0]
        p_y_norm = observation[1]
        
        # Griglia virtuale 20x20 per tracking esplorazione
        sector_x = int(p_x_norm * 20)
        sector_y = int(p_y_norm * 20)
        current_sector = (sector_x, sector_y)
        
        if current_sector not in self.visited_sectors:
            self.visited_sectors.add(current_sector)
            step_fitness += self.w_exploration * 5.0  # Bonus esplorazione
            self.frames_no_progress = 0  # Reset timer camping
        
        # --- 4. POTENTIAL-BASED REWARD SHAPING ---
        # Calcola Φ(s_t+1) - Φ(s_t)
        pellets = [o for o in objects if "Pellet" in o.category or "Small" in o.category]
        player = next((o for o in objects if "Player" in o.category or "Pacman" in o.category), None)
        
        if player and pellets:
            current_potential = self._get_potential((player.x, player.y), pellets)
            
            if self.prev_nearest_pellet_dist is not None:
                # Δ Potenziale: se ci avviciniamo al cibo, reward positivo
                potential_diff = current_potential - self.prev_nearest_pellet_dist
                step_fitness += self.w_potential * potential_diff * 100.0  # Scala amplificata
            
            self.prev_nearest_pellet_dist = current_potential
        
        # --- 5. ANTI-CAMPING PENALTY ---
        if reward == 0 and current_sector in self.visited_sectors:
            self.frames_no_progress += 1
            
            # Se sta fermo troppo a lungo
            if self.frames_no_progress > self.camping_threshold:
                step_fitness -= self.penalty_camping * (self.frames_no_progress - self.camping_threshold)
        
        # --- 6. DEATH PENALTY ---
        if done:
            step_fitness -= self.penalty_death
        
        self.total_fitness += step_fitness
        return step_fitness
    
    def get_final_fitness(self):
        """Restituisce fitness finale dell'episodio."""
        # Bonus finale per esplorazione completa
        exploration_bonus = len(self.visited_sectors) * 2.0
        
        # Bonus finale per fantasmi mangiati
        ghost_bonus = self.ghosts_eaten * self.w_ghost_bonus * 100.0
        
        final = self.total_fitness + exploration_bonus + ghost_bonus
        return max(0.0, final)  # Non può essere negativo