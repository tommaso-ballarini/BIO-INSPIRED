# EvoAtari: Benchmarking Bio-Inspired Agents across Atari Games

**Bio-Inspired Artificial Intelligence - Final Project (2025/2026)**  
**University of Trento**

**Authors:**
- Tommaso Ballarini
- Chiara Belli
- Elisa Negrini

---

## 📋 Overview

This project investigates **neuroevolutionary approaches** to the Atari 2600 Arcade Learning Environment (ALE), focusing on **object-centric state representations** as an alternative to pixel-based deep reinforcement learning. We leverage **NEAT (NeuroEvolution of Augmenting Topologies)** combined with the **OCAtari library** to evolve compact neural networks capable of playing Atari games through semantic understanding rather than raw visual processing.

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

## 🎮 Results Preview

### Skiing
The best FFNN Dynamic agent successfully navigates all gates with human-competitive completion times (~48 seconds).

<!-- Add your GIF/video here -->
![Skiing Agent Demo](assets/skiing_demo.gif)

### Freeway
The agent learns rhythmic timing patterns to cross ten lanes of traffic while minimizing collisions.

<!-- Add your GIF/video here -->
![Freeway Agent Demo](assets/freeway_demo.gif)

### Space Invaders
The Egocentric RNN agent clears multiple waves, demonstrating emergent target prioritization and UFO sniping.

<!-- Add your GIF/video here -->
![Space Invaders Agent Demo](assets/space_invaders_demo.gif)

---
##  🏗️ Repository Structure
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
│   ├── Freeway/
│   │   ├── wrapper/
│   │   ├── config/
│   │   ├── run/
│   │   ├── visualize/
│   │   └── results/
│   │
│   └── SpaceInvaders/
│       ├── wrapper/
│       ├── config/
│       ├── run/
│       ├── visualize/
│       └── results/
│
├── report/                                  # IEEE format project report
│   └── EvoAtari_Report.pdf
│
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🚀 Setup Instructions

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

## 🎯 The Critical Role of Environment Wrappers

Environment wrappers serve as **semantic filters** that transform raw Atari data into meaningful, compact representations suitable for neuroevolution.

### Why Wrappers Are Essential

**The Raw RAM Problem:**

The Atari 2600's 128-byte RAM theoretically contains the complete game state, but is fundamentally incompatible with bio-inspired learning:
- Many bytes are unused or serve internal functions unrelated to gameplay
- Decimal values often lack linear correlation with physical quantities (positions, velocities)
- No explicit relational information (distances to targets, alignment with goals)
- 128-dimensional search space still hinders evolutionary convergence

**The Pixel Alternative:**

Raw pixel input (160×210×3 = 100,800 dimensions) is computationally infeasible for evolutionary algorithms, which lack the gradient-based optimization that enables deep learning to handle such high-dimensional spaces.

### What Wrappers Provide

Our custom wrappers implement a **two-stage preprocessing pipeline**:

1. **Semantic Feature Extraction**: Compute high-level game concepts (gate centers, threat distances, velocities) from raw object properties using OCAtari's RAM Extraction Method
2. **Normalization**: Scale all features to standardized ranges ([-1, 1] or [0, 1]) for stable neural network inputs

### Impact on Performance

**Skiing Example:**

| Representation | Dimensions | Agent Behavior | Best Score |
|----------------|------------|----------------|------------|
| Raw RAM | 128 | Straight descent, crashes | -6528 |
| **Custom Wrapper** | **9** | Gate-seeking, avoidance | **-4886** |

The wrapper achieves **14× dimensionality reduction** while **improving performance by 26%**, demonstrating that semantically meaningful representations enable both efficiency and effectiveness.

### Domain Knowledge Injection

Wrappers encode task-specific intelligence:
- **Skiing**: Gate center computation provides continuous alignment guidance
- **Freeway**: Explicit velocity features (∆x) allow FFNNs to perceive motion without recurrence
- **Space Invaders**: Egocentric ray-casting mimics human-like threat perception

> **Key Insight**: Without wrappers, evolutionary algorithms converge to degenerate local minima (as shown in baseline experiments). The wrapper is not an optimization—it's a **fundamental architectural requirement** for bio-inspired learning in complex environments.

---


## 🎯 Quick Start

### Training Agents (in case you want to train)

#### Skiing FFNN Dynamic (Best Performance)
```bash
python run_scripts/run_wrapper_ffnn_dynamic.py
```

#### Freeway
```bash
python run_scripts/run_freeway.py
```

#### Space Invaders
```bash
python run_scripts/run_space_invaders.py
```

Training progress will be displayed in the console, and results will be saved in `evolution_results/`.

### Visualizing Trained Agents

You can visualize the best agent (you can directly visualize without training as we provided the .pkl):
```bash
# Skiing
python visualization/visualize_skiing.py

# Freeway
python visualization/visualize_freeway.py 

# Space Invaders
python visualization/visualize_space_invaders.py 
```

### Evaluation Protocol

All configurations are evaluated using a standardized protocol:
- **Training Seeds**: Random seeds excluding the first 100
- **Evaluation Seeds**: The first 100 seeds (held-out for testing)
- **Runs per Configuration**: 100 episodes per agent
- **Metrics**: Average score, best score

---

## 🧪 Experimental Configurations

### Skiing

| Configuration | Wrapper | Fitness Shaping | Architecture | Description |
|---------------|---------|-----------------|--------------|-------------|
| **Baseline** | ❌ | ❌ | FFNN | Raw RAM input, native sparse rewards |
| **Wrapper (FFNN)** | ✅ | ✅ | FFNN | Object-centric state, dense rewards |
| **Wrapper (RNN)** | ✅ | ✅ | RNN | Same as FFNN with recurrent connections |
| **Wrapper (FFNN Dyn.)** | ✅ | ✅ + Adaptive | FFNN | Progressive time penalty scaling |

**Key Results:**
- Baseline: Failed to complete course (stuck in local minimum)
- FFNN Dynamic: **Best performance** (-4886 score, ~48s completion)
- Standard FFNN: Conservative strategy (-5248 score)
- RNN: Slowest among successful agents (-7152 score, ~72s)

### Freeway

**Wrapper Type:** Speed-Aware RAM Wrapper (22 inputs)
- Agent state: Vertical position + collision flag
- Traffic state: 10 lane horizontal positions
- Temporal dynamics: 10 velocity features (∆x per lane)

**Fitness Function:**
```
F = R_crossing + R_ymax - P_collision - P_time
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

## 📊 Key Findings

### Object-Centric Representations
- **50× faster** than vision-based extraction
- Compact state vectors (9-22 inputs) vs. raw pixel dimensions
- Facilitates relational reasoning without high-dimensional search

### Architecture Comparison
- **FFNN**: Sufficient when wrapper provides velocity features (Skiing, Freeway)
- **RNN**: Critical for temporal reasoning in dynamic environments (Space Invaders)
- **Recurrence Overhead**: Minimal benefit in Markovian tasks (Skiing RNN underperformed)

### Fitness Shaping Impact
- **Baseline (no wrapper)**: Converged to degenerate strategies
- **Dense rewards**: Enabled successful gate navigation in all wrapper-based Skiing configurations
- **Adaptive penalties**: Dynamic FFNN achieved human-competitive performance through progressive time scaling

---

## 🔬 Methodology Highlights

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

### State Preprocessing Pipeline

1. **Object Extraction**: OCAtari REM retrieves semantic objects from RAM
2. **Feature Computation**: High-level features (gate centers, nearest threats, velocities)
3. **Normalization**: Scale all features to [-1, 1] or [0, 1] ranges
4. **Network Input**: Feed compact vector to NEAT-evolved topology

---

## 📈 Performance Metrics

### Skiing (100 Evaluation Runs)

| Configuration | Avg Score | Best Score | Completion Rate |
|---------------|-----------|------------|-----------------|
| Baseline | -7646 | N/A | 0% |
| Wrapper (RNN) | -6685 | -7152 | 100% |
| Wrapper (FFNN) | -5581 | -5248 | 100% |
| **Wrapper (FFNN Dyn.)** | **-5134** | **-4886** | **100%** |

*Note: Scores are conditioned on gate completion (fitness > 9700) except baseline.*

### Space Invaders

| Metric | Value |
|--------|-------|
| Peak Fitness | ~2700 |
| Raw Game Score | 2380 points |
| Waves Cleared | 2+ (partial 3rd wave) |
| Emergent Behaviors | UFO sniping, threat prioritization |

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'ocatari'`
```bash
pip install ocatari
```

**Issue**: ALE ROMs not found
```bash
pip install gymnasium[atari,accept-rom-license]
```

**Issue**: CUDA out of memory (if using GPU rendering)
```bash
# Use CPU rendering by setting render_mode="rgb_array" or None in wrapper
```

**Issue**: Training crashes with multiprocessing errors
```bash
# Reduce NUM_WORKERS in run scripts
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 4)
```

---

## 📄 License

This project is developed for academic purposes as part of the Bio-Inspired Artificial Intelligence course at the University of Trento. 

---

## 📧 Contact

For questions or collaboration inquiries:
- Tommaso Ballarini: [email]
- Chiara Belli: [email]
- Elisa Negrini: [email]

---

## 🔗 References

[1] Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.  
[2] Bellemare et al., "The arcade learning environment: An evaluation platform for general agents," *JAIR*, 2013.  
[3] Delfosse et al., "OCAtari: Object-centric atari 2600 reinforcement learning environments," *arXiv:2306.08649*, 2023.  
[4] Delfosse et al., "Interpretable concept bottlenecks to align reinforcement learning agents," *NeurIPS*, 2024.  
[5] Machado et al., "Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents," *JAIR*, 2018.

