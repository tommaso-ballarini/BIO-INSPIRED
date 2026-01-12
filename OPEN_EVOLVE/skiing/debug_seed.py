import sys
import os
import time
# Aggiungi il path per trovare i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import seed_agent
    from evaluator import run_custom_simulation, EVAL_SEEDS
except ImportError as e:
    print(f"Errore import: {e}")
    sys.exit(1)

print(f"--- DIAGNOSTICA SEED AGENT ---")
print(f"Testando su {len(EVAL_SEEDS)} seed: {EVAL_SEEDS}")

total = 0
for i, seed in enumerate(EVAL_SEEDS):
    print(f"\n❄️  Running Game {i+1} (Seed {seed})...")
    try:
        # Usa visualization=True per vedere se fa cose stupide
        score = run_custom_simulation(seed_agent.get_action, game_idx=i, visualization=True)
        print(f"   -> Score: {score}")
        total += score
    except Exception as e:
        print(f"   -> CRASH: {e}")
        score = -10000

avg = total / len(EVAL_SEEDS)
print(f"\n" + "="*30)
print(f"MEDIA REALE: {avg}")
print(f"="*30)