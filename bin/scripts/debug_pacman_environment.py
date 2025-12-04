#!/usr/bin/env python3
"""
Script di debug per verificare che l'ambiente Pacman funzioni correttamente.
"""

import sys
import os
import numpy as np
from itertools import chain

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Monkey patch OCAtari
import ocatari.core

def patched_ns_state(self):
    valid_objects = [o for o in self.objects if o is not None and hasattr(o, '_nsrepr')]
    return list(chain.from_iterable([o._nsrepr for o in valid_objects]))

ocatari.core.OCAtari.ns_state = patched_ns_state

from ocatari.core import OCAtari
from core.wrappers_pacman import PacmanFeatureWrapper


def test_raw_environment():
    """Test 1: Ambiente base OCAtari senza wrapper"""
    print("=" * 80)
    print("🧪 TEST 1: Ambiente Base OCAtari (senza wrapper)")
    print("=" * 80)
    
    env = OCAtari("Pacman", mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)
    
    if hasattr(env.unwrapped, 'ale'):
        env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    
    obs, info = env.reset()
    
    print(f"\n📊 Info dopo reset:")
    print(f"   Observation type: {type(obs)}")
    if isinstance(obs, np.ndarray):
        print(f"   Observation shape: {obs.shape}")
    
    print(f"\n🎮 Action Space: {env.action_space}")
    print(f"   Azioni disponibili: {env.action_space.n}")
    
    # Test con azioni casuali
    print(f"\n🎲 Test con 100 azioni CASUALI:")
    total_reward = 0
    steps = 0
    
    for i in range(100):
        action = env.action_space.sample()  # Azione casuale
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if reward > 0:
            print(f"   Step {steps}: Action={action}, Reward={reward} ✅")
        
        if terminated or truncated:
            print(f"   ⚠️  Episodio terminato al step {steps}")
            break
    
    print(f"\n📈 Risultati Test 1:")
    print(f"   Total Steps: {steps}")
    print(f"   Total Reward: {total_reward}")
    print(f"   Avg Reward/Step: {total_reward/steps if steps > 0 else 0:.3f}")
    
    env.close()
    return total_reward


def test_wrapped_environment():
    """Test 2: Ambiente con PacmanFeatureWrapper"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Ambiente con PacmanFeatureWrapper")
    print("=" * 80)
    
    env = OCAtari("Pacman", mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)
    
    if hasattr(env.unwrapped, 'ale'):
        env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    
    env = PacmanFeatureWrapper(env, grid_rows=10, grid_cols=10)
    
    obs, info = env.reset()
    
    print(f"\n📊 Observation dopo wrapper:")
    print(f"   Type: {type(obs)}")
    print(f"   Shape: {obs.shape}")
    print(f"   Expected: (146,)")
    print(f"   Min value: {obs.min():.3f}")
    print(f"   Max value: {obs.max():.3f}")
    print(f"   Contiene NaN? {np.isnan(obs).any()}")
    print(f"   Contiene Inf? {np.isinf(obs).any()}")
    
    # Controlla feature specifiche
    print(f"\n🔍 Feature estratte:")
    print(f"   Player pos: ({obs[0]:.3f}, {obs[1]:.3f})")
    print(f"   Pellets remaining (norm): {obs[32]:.3f}")
    print(f"   Lives (norm): {obs[33]:.3f}")
    print(f"   Ghost edibility: {obs[26:30]}")
    
    # Test con azioni casuali
    print(f"\n🎲 Test con 100 azioni CASUALI (wrapped):")
    total_reward = 0
    steps = 0
    rewards_received = []
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if reward > 0:
            rewards_received.append(reward)
            print(f"   Step {steps}: Action={action}, Reward={reward} ✅")
        
        # Verifica observation valida
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"   ⚠️  WARNING: Observation contiene NaN/Inf al step {steps}!")
        
        if terminated or truncated:
            print(f"   ⚠️  Episodio terminato al step {steps}")
            break
    
    print(f"\n📈 Risultati Test 2:")
    print(f"   Total Steps: {steps}")
    print(f"   Total Reward: {total_reward}")
    print(f"   Avg Reward/Step: {total_reward/steps if steps > 0 else 0:.3f}")
    print(f"   Rewards ricevuti: {len(rewards_received)} volte")
    if rewards_received:
        print(f"   Reward values: {set(rewards_received)}")
    
    env.close()
    return total_reward


def test_deterministic_policy():
    """Test 3: Policy deterministica (sempre RIGHT)"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Policy Deterministica (sempre RIGHT)")
    print("=" * 80)
    
    env = OCAtari("Pacman", mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)
    
    if hasattr(env.unwrapped, 'ale'):
        env.unwrapped.ale.setFloat('repeat_action_probability', 0.0)
    
    env = PacmanFeatureWrapper(env, grid_rows=10, grid_cols=10)
    
    obs, info = env.reset()
    
    # Azioni: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN (tipico Atari)
    ACTIONS_TO_TEST = [
        (2, "RIGHT"),
        (4, "DOWN"),
        (1, "UP"),
        (3, "LEFT")
    ]
    
    for action_id, action_name in ACTIONS_TO_TEST:
        print(f"\n🎯 Test con azione fissa: {action_name} (id={action_id})")
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        for i in range(50):
            obs, reward, terminated, truncated, info = env.step(action_id)
            total_reward += reward
            steps += 1
            
            if reward > 0:
                print(f"   Step {steps}: Reward={reward}")
            
            if terminated or truncated:
                break
        
        print(f"   Risultato: {steps} steps, reward totale={total_reward}")
    
    env.close()


def test_action_mapping():
    """Test 4: Verifica mapping azioni"""
    print("\n" + "=" * 80)
    print("🧪 TEST 4: Verifica Action Mapping")
    print("=" * 80)
    
    env = OCAtari("Pacman", mode="ram", obs_mode="obj", render_mode="rgb_array", hud=False)
    
    print(f"\n🎮 Action Space Info:")
    print(f"   Type: {type(env.action_space)}")
    print(f"   N actions: {env.action_space.n}")
    
    # Prova a ottenere i significati delle azioni
    if hasattr(env.unwrapped, 'ale'):
        ale = env.unwrapped.ale
        print(f"\n📋 Azioni disponibili (ALE):")
        
        # ALE standard actions
        action_meanings = {
            0: "NOOP",
            1: "UP", 
            2: "RIGHT",
            3: "LEFT",
            4: "DOWN"
        }
        
        for i in range(env.action_space.n):
            meaning = action_meanings.get(i, f"UNKNOWN_{i}")
            print(f"   {i}: {meaning}")
    
    env.close()


if __name__ == "__main__":
    print("\n🔬 PACMAN ENVIRONMENT DEBUG SUITE")
    print("=" * 80)
    
    # Esegui tutti i test
    try:
        reward_raw = test_raw_environment()
        reward_wrapped = test_wrapped_environment()
        test_deterministic_policy()
        test_action_mapping()
        
        print("\n" + "=" * 80)
        print("📊 RIEPILOGO RISULTATI")
        print("=" * 80)
        print(f"✅ Test 1 (Raw Env): Total Reward = {reward_raw}")
        print(f"✅ Test 2 (Wrapped Env): Total Reward = {reward_wrapped}")
        
        if reward_raw > 50 and reward_wrapped > 50:
            print("\n✅ AMBIENTE OK - L'agente casuale riceve reward!")
        elif reward_raw > 50 and reward_wrapped < 10:
            print("\n⚠️  PROBLEMA NEL WRAPPER - L'ambiente raw funziona ma il wrapper no!")
        else:
            print("\n⚠️  PROBLEMA NELL'AMBIENTE - Anche l'agente casuale non riceve reward!")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante i test: {e}")
        import traceback
        traceback.print_exc()