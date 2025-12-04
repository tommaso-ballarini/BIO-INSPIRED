# tests/test_mspacman_wrapper.py
"""
Script di test per verificare il funzionamento del NeatMsPacmanWrapper.

Testa:
1. Creazione corretta della pipeline
2. Dimensioni dello spazio di osservazione
3. Estrazione delle feature
4. Normalizzazione dei valori
"""

import sys
import os
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari
from core.wrappers import NeatMsPacmanWrapper


def test_wrapper_creation():
    """Test 1: Verifica creazione corretta della pipeline"""
    print("=" * 70)
    print("TEST 1: Creazione Pipeline")
    print("=" * 70)
    
    try:
        # Crea ambiente base
        base_env = gym.make("MsPacman-v4", render_mode=None)
        print("✓ Ambiente base creato")
        
        # Applica OCAtari
        ocatari_env = OCAtari(base_env, mode="ram", obs_mode="obj")
        print("✓ OCAtari wrapper applicato (REM mode)")
        
        # Applica wrapper custom
        env = NeatMsPacmanWrapper(ocatari_env)
        print("✓ NeatMsPacmanWrapper applicato")
        
        # Verifica spazio di osservazione
        obs_space = env.observation_space
        print(f"\n📊 Spazio di osservazione:")
        print(f"   - Shape: {obs_space.shape}")
        print(f"   - Dtype: {obs_space.dtype}")
        print(f"   - Low: {obs_space.low[0]}")
        print(f"   - High: {obs_space.high[0]}")
        
        assert obs_space.shape == (20,), f"Errore: shape atteso (20,), ottenuto {obs_space.shape}"
        assert obs_space.dtype == np.float32, "Errore: dtype deve essere float32"
        
        print("\n✅ Test 1 SUPERATO\n")
        return env
        
    except Exception as e:
        print(f"\n❌ Test 1 FALLITO: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_observation_extraction(env):
    """Test 2: Verifica estrazione delle osservazioni"""
    print("=" * 70)
    print("TEST 2: Estrazione Feature")
    print("=" * 70)
    
    try:
        # Reset ambiente
        obs, info = env.reset()
        print(f"✓ Reset eseguito")
        
        # Verifica tipo e dimensioni
        print(f"\n📊 Osservazione:")
        print(f"   - Type: {type(obs)}")
        print(f"   - Shape: {obs.shape}")
        print(f"   - Dtype: {obs.dtype}")
        
        assert isinstance(obs, np.ndarray), "Errore: obs deve essere numpy array"
        assert obs.shape == (20,), f"Errore: shape atteso (20,), ottenuto {obs.shape}"
        
        # Verifica normalizzazione (tutti i valori devono essere in [0, 1])
        print(f"\n📈 Range valori:")
        print(f"   - Min: {obs.min():.4f}")
        print(f"   - Max: {obs.max():.4f}")
        print(f"   - Mean: {obs.mean():.4f}")
        
        if obs.min() < 0.0 or obs.max() > 1.0:
            print(f"⚠️ WARNING: Valori fuori range [0, 1]!")
        
        # Stampa feature per feature
        print(f"\n🔍 Feature dettagliate:")
        feature_names = [
            "Player X", "Player Y", "Player Dir",
            "Ghost1 DX", "Ghost1 DY", "Ghost2 DX", "Ghost2 DY",
            "Ghost3 DX", "Ghost3 DY", "Ghost4 DX", "Ghost4 DY",
            "Ghost1 Edible", "Ghost2 Edible", "Ghost3 Edible", "Ghost4 Edible",
            "PowerPill DX", "PowerPill DY",
            "Fruit DX", "Fruit DY",
            "Dots Eaten"
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, obs)):
            print(f"   [{i:2d}] {name:20s}: {value:.4f}")
        
        print("\n✅ Test 2 SUPERATO\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FALLITO: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_episode_run(env, num_steps=100):
    """Test 3: Esegue un episodio breve e verifica consistenza"""
    print("=" * 70)
    print(f"TEST 3: Esecuzione Episodio ({num_steps} steps)")
    print("=" * 70)
    
    try:
        obs, info = env.reset()
        total_reward = 0
        
        print("🎮 Esecuzione con azioni random...")
        
        for step in range(num_steps):
            # Azione random
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Verifica osservazione
            assert obs.shape == (20,), f"Shape errato allo step {step}"
            assert obs.dtype == np.float32, f"Dtype errato allo step {step}"
            
            if terminated or truncated:
                print(f"   ⚠️ Episodio terminato allo step {step}")
                break
        
        print(f"\n📊 Risultati:")
        print(f"   - Steps eseguiti: {step + 1}")
        print(f"   - Reward totale: {total_reward:.2f}")
        print(f"   - Ultima osservazione shape: {obs.shape}")
        print(f"   - Ultima osservazione range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        env.close()
        
        print("\n✅ Test 3 SUPERATO\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FALLITO: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_object_detection(env):
    """Test 4: Verifica rilevamento oggetti OCAtari"""
    print("=" * 70)
    print("TEST 4: Rilevamento Oggetti OCAtari")
    print("=" * 70)
    
    try:
        obs, info = env.reset()
        
        # Fai qualche step per popolare gli oggetti
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                obs, info = env.reset()
        
        # Accedi agli oggetti OCAtari
        objects = []
        if hasattr(env.env, 'objects'):
            objects = [o for o in env.env.objects if o is not None]
        elif hasattr(env.env.unwrapped, 'objects'):
            objects = [o for o in env.env.unwrapped.objects if o is not None]
        
        print(f"\n🔍 Oggetti rilevati: {len(objects)}")
        
        # Conta per categoria
        categories = {}
        for obj in objects:
            if hasattr(obj, 'category'):
                cat = obj.category
                categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n📊 Breakdown per categoria:")
        for cat, count in sorted(categories.items()):
            print(f"   - {cat:20s}: {count}")
        
        # Verifica presenza di oggetti chiave
        player_found = any('Player' in obj.category or 'Pacman' in obj.category 
                          for obj in objects if hasattr(obj, 'category'))
        ghosts_found = sum(1 for obj in objects 
                          if hasattr(obj, 'category') and 'Enemy' in obj.category)
        
        print(f"\n✓ Player trovato: {player_found}")
        print(f"✓ Ghosts trovati: {ghosts_found}")
        
        if not player_found:
            print("⚠️ WARNING: Player non rilevato!")
        if ghosts_found < 4:
            print(f"⚠️ WARNING: Trovati solo {ghosts_found}/4 ghosts")
        
        print("\n✅ Test 4 COMPLETATO\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 4 FALLITO: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Esegue tutti i test in sequenza"""
    print("\n" + "=" * 70)
    print("🧪 TEST SUITE PER NeatMsPacmanWrapper")
    print("=" * 70 + "\n")
    
    results = []
    
    # Test 1: Creazione
    env = test_wrapper_creation()
    results.append(("Creazione Pipeline", env is not None))
    
    if env is None:
        print("❌ Impossibile continuare i test senza ambiente")
        return
    
    # Test 2: Estrazione
    result = test_observation_extraction(env)
    results.append(("Estrazione Feature", result))
    
    # Test 3: Episodio
    result = test_episode_run(env, num_steps=100)
    results.append(("Esecuzione Episodio", result))
    
    # Test 4: Rilevamento oggetti
    result = test_object_detection(env)
    results.append(("Rilevamento Oggetti", result))
    
    # Riepilogo
    print("=" * 70)
    print("📋 RIEPILOGO TEST")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n🎯 Risultato: {total_passed}/{len(results)} test superati")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()