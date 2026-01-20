import json
import numpy as np
import os
import sys

def analyze_benchmarks():
    script_dir = "human_benchmarks"  
    config = {
        "hb_freeway.json": [
            "raw_score", 
            "fit_aggressive", 
            "fit_balanced", 
            "fit_timepen", 
            "fit_simple"
        ],
        "hb_skiing.json": [
            "native", 
            "custom"
        ],
        "hb_spaceinvaders.json": [
            "raw_score", 
            "custom_fitness_ego"
        ]
    }

    results = {}

    print(f"\n Searching for files in: {script_dir}")
    print(f"{'GAME / METRIC':<35} | {'MEAN':<10} | {'STD':<10} | {'BEST SCORE':<12} | {'BEST RUN id'}")
    print("-" * 95)

    for filename, keys in config.items():
        file_path = os.path.join(script_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {filename}")
            
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

        for key in keys:
            values = [entry[key] for entry in data if key in entry]
            
            if not values:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)
            
            max_val = np.max(values) 
            
            best_run = next((entry for entry in data if entry.get(key) == max_val), {})
            id = best_run.get('id', 'N/A')

            label = f"{filename.split('.')[0].replace('hb_', '')} ({key})"
            print(f"{label:<35} | {mean_val:<10.2f} | {std_val:<10.2f} | {max_val:<12.2f} | {id}")

    print("-" * 95)

if __name__ == "__main__":
    analyze_benchmarks()