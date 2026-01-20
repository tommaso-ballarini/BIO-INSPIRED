# EvoAtari: Benchmarking Bio-Inspired Agents across Atari Games

**Bio-Inspired Artificial Intelligence - Final Project (2025/2026)**  
**University of Trento**

**Authors:**
- Tommaso Ballarini
- Chiara Belli
- Elisa Negrini

---

## Overview

This project investigates **neuroevolutionary approaches** to the Atari 2600 Arcade Learning Environment (ALE), focusing on **object-centric state representations** as an alternative to pixel-based deep reinforcement learning. We leverage **NEAT (NeuroEvolution of Augmenting Topologies)** combined with the **OCAtari library** and **OPENEVOLVE**, to evolve compact neural networks capable of playing Atari games through semantic understanding rather than raw visual processing. 

Our approach addresses three fundamental challenges in neuroevolution for complex games:
- **Sparse Rewards**: Custom fitness shaping wrappers provide dense, frame-by-frame guidance
- **Curse of Dimensionality**: RAM Extraction Method (REM) reduces high-dimensional pixel input to compact semantic vectors
- **Credit Assignment**: Object-centric features enable relational reasoning and temporal dependencies

### Evaluated Environments

We benchmark our approach on three distinct Atari games, each testing specific evolutionary capabilities:

1. **Skiing** - Sparse rewards and delayed credit assignment
2. **Freeway** - Synchronization and behavioral optimization
3. **Space Invaders** - Dynamic complexity with variable object counts

---

## Results Preview

### Skiing
The best FFNN Dynamic agent successfully navigates all gates with human-competitive completion times (~48 seconds).

<!-- Add your GIF/video here -->
![Skiing Best Run](assets/skiing_best_run.gif)

### Freeway
The best FFNN (wrapper + shaped fitness) agent learns rhythmic timing patterns to cross ten lanes of traffic while minimizing collisions.

<!-- Add your GIF/video here -->
![Freeway Agent Demo](assets/freeway_demo.gif)

### Space Invaders
The Egocentric RNN agent clears multiple waves, demonstrating emergent target prioritization and UFO sniping.

<!-- Add your GIF/video here -->
![Space Invaders Agent Demo](assets/space_invaders_demo.gif)

---
## Repository Structure
```

BIO-INSPIRED/
├── Neat/                                    # NEAT-based evolutionary experiments
│   ├── Skiing/
│   │   ├── wrapper/                         # Custom environment wrappers
│   │   │   ├── ...
│   │   ├── config/                          # NEAT configuration files
│   │   │   ├── ...
│   │   ├── run/                             # Training scripts
│   │   │   ├── ...
│   │   ├── visualize/                       # Visualization scripts
│   │   │   ├── ...
│   │   └── results/                         # Saved agents and plots
│   │       ├── ...
│   │
│   ├── Freeway/                             # Same structure as for Skiing
│   │   ├── ...
│   │
│   └── SpaceInvaders/                       # Same structure as for Skiing
│       ├── ...
│
├── OpenEvolve/                              # LLM-driven evolutionary experiments
│   ├── Skiing/
│   │   ├── wrapper/
│   │   ├── config/
│   │   ├── run/
│   │   ├── visualize/
│   │   └── results/
│   │
│   ├── Freeway/                             # Same structure as for Skiing in OPENEVOLVE
│   │   ├── ...
│   │
│   └── SpaceInvaders/                       # Same structure as for Skiing in OPENEVOLVE
│       ├── ...
│
├── human_benchmarks/                        # Scripts and results for human benchmark assessments
│   └── ...
├── report/                                  # IEEE format project report
│   └── EvoAtari_Report.pdf
│
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

The tree above shows the two main experiment tracks (`Neat/` and `OpenEvolve/`), each organized per game with the same subfolders: wrappers for feature extraction, configs for NEAT hyperparameters, run scripts for training, visualization utilities, and results for saved models and plots. The root also includes the paper report and a shared `requirements.txt` so the environment is consistent across both tracks.

---

## Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux/macOS (recommended), Windows (with WSL)
- **Hardware**: Multi-core CPU recommended for parallel evolution

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/elisa-negrini/BIO-INSPIRED.git
   cd BIO-INSPIRED
```

2. **Create a virtual environment** (recommended)
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

### Required Libraries

- `gymnasium[atari]` - Atari 2600 environments
- `ale-py` - Arcade Learning Environment
- `neat-python` - NEAT implementation
- `ocatari` - Object-centric Atari wrapper
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `opencv-python` - Video rendering (optional)


---

##  The Critical Role of Environment Wrappers

Directly utilizing the raw RAM of the Atari 2600 console (128 bytes) for training bio-inspired agents presents significant challenges. Although it encapsulates the entire game state, the raw information is often **inaccessible or misleading** for neural networks: many bytes remain unused, others dynamically shift function depending on game context, and decimal values lack linear correlation with physical quantities they represent.

To overcome these limitations, we adopt **environment-specific wrappers** built with the **OCAtari library**. These wrappers act as **semantic filters** that:

1. **Extract** high-level game concepts (gate centers, threat distances, velocities) using the RAM Extraction Method (REM)
2. **Normalize** features to standardized ranges ([-1, 1] or [0, 1]) for stable neural network inputs
3. **Reduce** input dimensionality drastically while preserving task-relevant information

> **Key Insight**: Without wrappers, evolutionary algorithms converge to degenerate local minima. The wrapper is not an optimization—it's a **fundamental architectural requirement** for bio-inspired learning in complex Atari environments.
> 
---

## Experiments

### Skiing

| Configuration | Wrapper | Fitness Shaping | Architecture | Description |
|---------------|---------|-----------------|--------------|-------------|
| **Baseline** | ❌ | ❌ | FFNN | Raw RAM input, native sparse rewards |
| **Wrapper (FFNN)** | ✅ | ✅ | FFNN | Object-centric state, dense rewards |
| **Wrapper (RNN)** | ✅ | ✅ | RNN | Same as FFNN with recurrent connections |
| **Wrapper (FFNN Dyn.)** | ✅ | ✅ + Adaptive | FFNN | Progressive time penalty scaling |

**Wrapper Type:** Object-Centric Slalom Wrapper (9 inputs)
- Player dynamics (3 inputs): Horizontal position, ski orientation, velocity (∆x)
- Target navigation (3 inputs): Lateral offset to gate center, vertical proximity, gate detection flag
- Obstacle awareness (3 inputs): Nearest threat (x, y) position, threat type (tree/flag)

**Fitness Function:**
```
F = Σ(R_gate + R_magnet - P_collision - P_boundary - P_time)
```

Where:
- **R_gate**: Sparse impulse reward for clearing gates (+500)
- **R_magnet**: Dense alignment gradient for horizontal positioning (+1.0 if aligned, -0.5×|∆x| otherwise)
- **P_collision**: Collision penalty with obstacles (-10.0)
- **P_boundary**: Track boundary violation penalty (-5.0)
- **P_time**: Constant time penalty to encourage efficiency (-0.2 per frame)

**Dynamic Variant**: Progressively increases time penalty once consistent gate completion is achieved, encouraging faster descent strategies.


### Freeway

| Configuration | Wrapper | Fitness Shaping | Architecture | Description |
|---------------|---------|-----------------|--------------|-------------|
| **Baseline** | ❌ | ❌ | FFNN | Raw RAM input, native sparse rewards |
| **FFNN (wrapper)** | ✅ | ❌ | FFNN | Object-centric state, native sparse rewards |
| **FFNN (wrapper + fitness)** | ✅ | ✅ | FFNN | Object-centric state, dense rewards |
| **RNN (wrapper)** | ✅ | ❌  | RNN | Same as FFNN (wrapper) with recurrent connections |
| **RNN (wrapper + fitness)** | ✅ | ✅ | RNN | Same as FFNN (wrapper + fitness) with recurrent connections |


**Wrapper Type:** Speed-Aware RAM Wrapper (22 inputs)
- Agent state (2 inputs): Vertical position + collision flag
- Traffic state (10 inputs): 10 lane horizontal positions
- Temporal dynamics (10 inputs): 10 velocity features (∆x per lane)


**Fitness Function:**
```
F = Σ(R_crossing + R_ymax - P_collision - P_time)
```



### Space Invaders

**Wrapper Type:** Egocentric Wrapper (19 inputs)
- 5 proximity sensors (vertical ray-casting)
- 5 temporal deltas (projectile velocity)
- 1 targeting feature (nearest alien offset)
- 4 alien density quadrants
- 3 game state features (alien fraction, UFO position, UFO availability)

**Architecture:** RNN (1000 population, 300 generations)

**Hierarchical Fitness:**
```
F = Σ(R_kill + S_aim - P_danger - P_spam)
```
where penalties/bonuses are conditioned on danger assessment.

---

## Experimental setup

All configurations are evaluated using a standardized protocol:
- **Training Seeds**: Random seeds excluding the first 100
- **Evaluation Seeds**: The first 100 seeds (held-out for testing)
- **Runs per Configuration**: 100 episodes per agent
- **Metrics**: Average score (+- standard deviation), best score

---
## Results

### Skiing ###

Since the native Atari score represents negative elapsed time, early termination can paradoxically yield higher values than slow completion. To ensure meaningful comparisons, the table below reports raw scores **conditioned on successful gate completion** (custom fitness > 9700), with the exception of the baseline (*), whose results reflect unconditional metrics. The 'Best Score' corresponds to the episode achieving the highest shaped fitness value, while 'n' indicates the number of successful runs (out of 100 evaluation episodes).

| Configuration | n | Best Score | Avg Score | Notes |
|---------------|---|------------|-----------|-------|
| Baseline (no wrapper)* | N/A | -7922.0* | -7646.01 (±1305.43)* | Straight descent, no gate-seeking behavior |
| Wrapper (FFNN) | 100 | -5248.0 | -5581.08 (±136.48) | Conservative strategy, perfect gate completion |
| Wrapper (RNN) | 36 | -7152.0 | -7844.17 (±522.82) | Slower completion, lower success rate |
| **Wrapper (FFNN Dynamic)** | **35** | **-4886.0** | **-5391.63 (±334.82)** | **Fastest descent, aggressive optimization** |
| *Human Benchmark* | — | *-3385.00* | *-3736.57 (±299.27)* | *Expert-level performance* |

---

### Freeway ###
| Configuration | Best Score | Avg Score | Notes |
|---------------|------------|-----------|-------|
| Baseline | 20.0 | 17.07 (±1.12) | Managed to complete some crossings, but stuck in collision-recovery cycles |
| FFNN (wrapper) | 21.0 | 19.51 (±0.94) | Slightly more stable crossings, but limited improvement over baseline |
| **FFNN (wrapper + fitness)** | **29.0** | **25.06 (±1.57)** | Consistent progress across lanes with fewer stalls |
| RNN (wrapper) | 28.0 | 24.90 (±1.52) | Comparable to FFNN with wrapper, with no clear temporal advantage |
| RNN (wrapper + fitness) | 28.0 | 23.92 (±1.73) | Similar to RNN (wrapper), fitness shaping does not add gains here |

### Space Invaders

| Metric | Value |
|--------|-------|
| Peak Fitness | ~2700 |
| Raw Game Score | 2380 points |
| Waves Cleared | 2+ (partial 3rd wave) |
| Emergent Behaviors | UFO sniping, threat prioritization |

---


## Methodology Highlights ????

### NEAT Configuration

**Common Parameters:**
- Population size: 100 (Skiing/Freeway), 1000 (Space Invaders)
- Generations: 150 (Skiing/Freeway), 300 (Space Invaders)
- Speciation threshold: 3.5
- Survival threshold: 20%
- Elitism: 2 best genomes per species

**Structural Mutations:**
- Node addition: 30%
- Node deletion: 20%
- Connection add/delete: 50% each

**Activation Functions:**
- Default: `tanh`
- Alternatives: `sigmoid`, `relu` (5% mutation rate)

---

## License

This project is developed for academic purposes as part of the Bio-Inspired Artificial Intelligence course at the University of Trento. 

---

## Contact

For questions or collaboration inquiries:
- Tommaso Ballarini: [email]
- Chiara Belli: [email]
- Elisa Negrini: [email]

---

## References

[1] Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.  
[2] Bellemare et al., "The arcade learning environment: An evaluation platform for general agents," *JAIR*, 2013.  
[3] Delfosse et al., "OCAtari: Object-centric atari 2600 reinforcement learning environments," *arXiv:2306.08649*, 2023.  
[4] Delfosse et al., "Interpretable concept bottlenecks to align reinforcement learning agents," *NeurIPS*, 2024.  
[5] Machado et al., "Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents," *JAIR*, 2018.
