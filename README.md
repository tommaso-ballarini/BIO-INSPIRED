# EvoAtari: Benchmarking Bio-Inspired Agents across Atari Games

**Bio-Inspired Artificial Intelligence - Final Project (2025/2026)**  
**University of Trento**

**Authors:**
- Tommaso Ballarini
- Chiara Belli
- Elisa Negrini

---

## Overview

This project presents a benchmark comparing two distinct evolutionary paradigms applied to the Atari 2600 Arcade Learning Environment (ALE):

1.  **Structural Neuroevolution (NEAT):** Evolving neural network topologies to create efficient but opaque "Black-Box" agents.
2.  **LLM-Driven Code Evolution (OpenEvolve):** Leveraging Large Language Models (**Qwen2.5-Coder-7B**) to evolve executable, interpretable "White-Box" Python code.

Instead of relying on raw pixel inputs, which often lead to the "curse of dimensionality", our approach utilizes **Object-Centric State Representations** via the **OCAtari** library.By extracting semantic features (RAM Extraction Method) and applying custom **Fitness Shaping wrappers**, we enable bio-inspired agents to master complex tasks with significantly reduced computational overhead compared to Deep Reinforcement Learning (DRL).

###  Key Objectives
* **State Representation & Efficiency:** Demonstrate how semantic wrappers are fundamental architectural requirements when operating under deliberate computational constraints, enabling effective learning where raw-pixel approaches fail due to excessive dimensionality.
* **Interpretability vs. Performance:** Analyze the trade-off between the raw scores of neural networks and the transparency of code-based agents.

### Evaluated Environments
We test these approaches on three environments chosen for their distinct evolutionary challenges:
* **Skiing:** Sparse rewards and delayed credit assignment.
* **Freeway:** Synchronization, sparse reward and local optima avoidance.
* **Space Invaders:** Dynamic complexity and multi-object management

---

## Results Preview

### Skiing
The best FFNN Dynamic agent successfully navigates all gates with human-competitive completion times (~48 seconds), while the OpenEvolve agent finds an even more efficient path.

<table width="100%">
  <tr>
    <th width="50%">NEAT Agent (FFNN Dynamic)</th>
    <th width="50%">OpenEvolve Agent (Gen 1500)</th>
  </tr>
  <tr>
    <td><img src="assets/Skiing_NEAT.gif" alt="Skiing NEAT Run" width="100%"></td>
    <td><img src="assets/Skiing_Open_Evolve.gif" alt="Skiing OpenEvolve Run" width="100%"></td>
  </tr>
</table>

### Freeway
The best FFNN (wrapper + shaped fitness) agent learns rhythmic timing patterns to cross ten lanes. Comparison shows NEAT's frame-perfect reactivity vs OpenEvolve's conservative strategy.

<table width="100%">
  <tr>
    <th width="50%">NEAT Agent (FFNN Wrapper)</th>
    <th width="50%">OpenEvolve Agent (Conservative)</th>
  </tr>
  <tr>
    <td><img src="assets/Freeway_NEAT.gif" alt="Freeway NEAT Run" width="100%"></td>
    <td><img src="assets/Freeway_Open_Evolve.gif" alt="Freeway OpenEvolve Run" width="100%"></td>
  </tr>
</table>

### Space Invaders
The Egocentric RNN agent demonstrates emergent target prioritization (UFO sniping), while the OpenEvolve agent shows robust generalized behavior across random seeds.

<table width="100%">
  <tr>
    <th width="50%">NEAT Agent (Egocentric RNN)</th>
    <th width="50%">OpenEvolve Agent (Generalist)</th>
  </tr>
  <tr>
    <td><img src="assets/SI_NEAT.gif" alt="Space Invaders NEAT Run" width="100%"></td>
    <td><img src="assets/SI_Open_Evolve.gif" alt="Space Invaders OpenEvolve Run" width="100%"></td>
  </tr>
</table>


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

## Methodology

To overcome the computational limitations inherent to raw pixel processing,  this study adopts a **semantic state extraction**. The idea is that for bio-inspired algorithms to succeed in limited compute environments, the "curse of dimensionality" must be solved architecturally, not just computationally.

### 1. Object-Centric Wrappers (OCAtari)
We utilize the **OCAtari** library to perform RAM Extraction Methods (REM). This transforms the 128-byte Atari RAM into compact, normalized feature vectors.

### 2. Game-Specific Strategies

To bridge the gap between sparse native rewards and evolutionary search, we designed custom **Fitness Shaping Functions** for each environment.

────────────────────────────────────────────────────────
#### 🎿 Skiing (Sparse Rewards)
**State Representation (9 Inputs)**:
* **Player Dynamics (3 inputs):** The agent's normalized horizontal position, ski orientation, and horizontal velocity ($x - x_{prev}$).
* **Target Navigation (3 inputs):** Guidance towards the next gate via lateral offset to the gate center, vertical proximity, and a binary gate detection flag.
* **Obstacle Awareness (3 inputs):** Relative $(x, y)$ coordinates of the nearest threat (tree or flag) and a categorical indicator distinguishing the type of obstacle.

**Fitness Function:**
    $$F = \sum_{t=0}^{T} (R_{gate} + R_{magnet} - P_{collision} - P_{boundary} - P_{time})$$
  
Where:
  
  &nbsp;- **$R_{kill}$**: Weighted bonus for alien elimination 
  
  &nbsp;- **$S_{aim}$**: Dense alignment gradient rewarding horizontal synchronization with the nearest target 
  
  &nbsp;- **$P_{danger}$**: Penalty for projectile proximity that scales as threats approach the player 
  
  &nbsp;- **$P_{spam}$**: Penalty for firing cooldown abuse to discourage random shooting 

────────────────────────────────────────────────────────
#### 🐔 Freeway (Synchronization)
**State Representation (22 Inputs)**:
* **Agent State (2 inputs):** Normalized vertical position ($y_{norm}$) and a binary collision flag.
* **Traffic State (10 inputs):** Horizontal positions ($x$) of the cars in each of the ten lanes, normalized to screen width.
* **Temporal Dynamics (10 inputs):** Computed velocities ($\Delta x$) for each car. This allows the network to perceive motion and direction without requiring frame-stacking or recurrent memory.

**Fitness Function:**
    $$F = \sum_{t=0}^{T} (R_{crossing} + R_{ymax} - P_{collision} - P_{time})$$

  Where:
  
  
  &nbsp;- **$R_{crossing}$**: Reward for the number of successful crossings.
    
  &nbsp;- **$R_{ymax}$**: Dense reward for the maximum normalized vertical progress achieved in the current attempt (provided even if no crossing is completed).
    
  &nbsp;- **$P_{collision}$**: Penalty inferred on the total count of collisions.
    
  &nbsp;- **$P_{timepenalty}$**: Penalty given for the total number of elapsed frames.


────────────────────────────────────────────────────────
#### 👾 Space Invaders (Dynamic Complexity)
**State Representation (19 Inputs)**:
* **Player Position (1 input):** Recalibrated normalized horizontal coordinate ($P_x$) within the playable area.
* **Threat Sensors (10 inputs):** 5 vertical ray-casting sensors detecting the nearest projectile in each sector, plus 5 temporal deltas to infer projectile velocity.
* **Targeting & Strategy (5 inputs):** Normalized relative horizontal distance to the nearest bottom-row alien ($x_{alien} - P_x$) and the enemy count across four quadrants.
* **Game State (3 inputs):** Tracks global game progression (alien fraction), UFO position, and UFO availability.
  
**Fitness Function:**
    $$F = \sum_{t=0}^{T} (R_{kill} + S_{aim} - P_{danger} - P_{spam})$$
    
Where:

  &nbsp;- **$R_{kill}$**: Weighted bonus for alien elimination.
  
   &nbsp;- **$S_{aim}$**: Dense alignment gradient rewarding horizontal synchronization with the nearest target.
    
   &nbsp;- **$P_{danger}$**: Penalty for projectile proximity that scales as threats approach the player.
    
   &nbsp;- **$P_{spam}$**: Penalty for firing cooldown abuse to discourage random shooting.


---

##  Experimental Results: NEAT

The effectiveness of structural neuroevolution was evaluated across the three environments. The tables below compare the baseline (Raw RAM, standard fitness) against the Object-Centric (Wrapper) agents.

### 1. Skiing
*Challenge: Sparse rewards and delayed credit assignment.*

| Configuration | Best Score | Avg Score | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** (Raw RAM) | -6528.0 | -6401.64 (±718.51) | Failed. Agent descends straight down to minimize time (local minima). |
| **Wrapper + FFNN** | -5248.0 | -5581 (±136) | Successful gate completion but conservative speed. |
| **Wrapper + RNN** | -7152.0 | -7844 (±522) | High variance; memory didn't help with static obstacles. |
| **Wrapper + FFNN (Dyn)** | **-4886.0** | **-5391 (±335)** | **Best Agent.** Dynamic time penalty forced faster descent. |
| *Human Benchmark* | *-3385.0* | *-3736 (±299)* | *Expert-level path planning.* |

> **Key Insight:** The "Dynamic" fitness variant was crucial. By progressively increasing the time penalty only *after* the agent learned to hit gates, we prevented the "reckless skiing" local minimum.

────────────────────────────────────────────────────────
### 2. Freeway
*Challenge: Synchronization and local optima.*

| Configuration | Best Score | Avg Score | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** | 20.0 | 17.07 (±1.12) | Stuck in collision-recovery loops ("deadlocks"). |
| **Wrapper + FFNN** | 21.0 | 19.51 (±0.94) | Better stability but lacked global timing. |
| **Wrapper + FFNN + Fit** | **29.0** | **25.06 (±1.57)** | **Near Optimal.** Learned to exploit gaps frame-perfectly. |
| **Wrapper + RNN** | 28.0 | 24.90 (±1.52) | Effective, but recurrent memory proved unnecessary. |
| **Wrapper + RNN + Fit** | 28.0 | 23.92 (±1.73) | Fitness shaping did not improve the recurrent model. |
| *Human Benchmark* | *31.0* | *24.96 (±4.1)* | *Higher peak score, but lower consistency than best agent.* |

> **Key Insight:** The continuous fitness shaping ($R_{ymax}$) solved the deadlock issue. The FFNN outperformed the RNN, suggesting that with velocity inputs included in the wrapper, the task is Markovian (no memory needed).

────────────────────────────────────────────────────────────────────────────────────────────────────────────────
### 3. Space Invaders
*Challenge: Dynamic complexity and multi-agent tracking.*

| Configuration | Best Score | Avg Score | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** | 760.0 | 221 (±156) | Spamming Fire pattern |
| **Wr. Col + FFNN** | 760.0 | 233 (±154) | "Column" inputs were too high-dimensional. |
| **Wr. Col + RNN** | 830.0 | 249 (±162) | Memory helped slightly, but input dimension was too high. |
| **Wr. Ego + RNN** | 1050.0 | 304 (±149) | Egocentric view improved tracking immediately. |
| **Wr. Ego + RNN (ext)** | 1880.0 | 366 (±113) | Extended training budget (300 gens) yielded better results. |
| **Wr. Ego + RNN + Fit (ext)** | **2380.0** | **421 (±120)** | In the best seed learned to snipe the UFO. |
| *Human Benchmark* | *970.0* | *527.4 (±188)* | *Higher average score, but significanty lower peak.* |

> **Key Insight:** This was the only environment where **Recurrent Neural Networks (RNN)** were strictly necessary. The agent needed memory to track projectile trajectories that momentarily disappeared or moved between sensors.

---

##  OpenEvolve: LLM-Driven Code Evolution

To contrast with NEAT's "Black-Box" networks, **OpenEvolve** was applied, a framework that treats Large Language Models as mutation operators.

* **Model:** Qwen2.5-Coder-7B (via Ollama)
* **Method:** A "Meta-Genome" system prompt guides the LLM to write executable Python policies based on the same OCAtari wrappers and fitness formula used for NEAT.
* **Evolution:** Both standard and cumulative strategies (where the best code from Generation $N$ is fed back as a few-shot example for Generation $N+1$) were used.

### Comparative Analysis: NEAT vs. OpenEvolve

The best "Black-Box" (NEAT) agents was benchmarked against the best "White-Box" (OpenEvolve) code.

| Environment | Metric | NEAT (Best Config) | OpenEvolve | 
|:-----------:|:------:|:------------------:|:----------:|
| **Skiing** | *Best Score* <br> *Avg Score* | -4886 <br> -5391.63 (±335) | **-3856** 🏆 <br> **-4004.96 (±86)** 🏆 | 
| **Freeway** | *Best Score* <br> *Avg Score* | **29.0** 🏆 <br> **25.06 (±1.57)** 🏆 | 18.0 <br> 15.28 (±1.43) | 
| **Space Inv.**| *Best Score* <br> *Avg Score* | **2380** 🏆 <br> 421 (±120) | 1495 <br> **517 (±212)** 🏆 | 

###  Discussion & Takeaways

1.  **Code vs. Neurons (Skiing):** OpenEvolve outperforms NEAT in Skiing. The LLM generated logical rules (e.g., *"If gate is to the left, move left immediately"*) which proved more efficient than the approximate function mapping of the neural network.
2.  **Reflex vs. Logic (Freeway):** NEAT dominated Freeway. The game requires frame-perfect reaction times ($~16ms$ precision) to weave through traffic. Neural networks excel at these reactive mappings, whereas the LLM-generated code adopted a "safety-first" if-else logic that was too conservative.
3.  **Generalization (Space Invaders):** While NEAT achieved the highest single-episode score (probable lucky seed), OpenEvolve had a higher **Average Score**. This suggests the code-based policy was more robust and generalized better to unseen starting states than the neural network.

## Conclusion

This project demonstrates that **Semantic State Representation** is the single most critical factor in evolving bio-inspired agents for Atari. By removing the pixel-processing burden:
1.  **NEAT** can master complex tasks on a standard laptop CPU.
2.  **LLMs** can evolve interpretable, human-readable Python strategies that rival neural networks.

----

----

----

----

----

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

| Configuration | Wrapper | Fitness Shaping | Architecture | Description |
|---------------|---------|-----------------|--------------|-------------|
| **Baseline** | ❌ | ❌ | FFNN | Raw RAM input (128 bytes), native scoring |
| **Column FFNN** | ✅ (Col) | ❌ | FFNN | Fixed column-based features, no memory |
| **Column RNN** | ✅ (Col) | ❌ | RNN | Column features + recurrent memory for projectiles |
| **Column RNN Fit** | ✅ (Col) | ✅ | RNN | Column features + dense rewards |
| **Egocentric RNN** | ✅ (Ego) | ❌ | RNN | Relative coordinates (player-centric), native scoring |
| **Egocentric RNN Fit** | ✅ (Ego) | ✅ | RNN | Relative coordinates + Hierarchical Fitness |

**Wrapper Type:** Egocentric Wrapper (19 inputs)
- 1 recalibrated normalized horizontal player coordinate (Px) 
- 5 proximity sensors (vertical ray-casting)
- 5 temporal deltas (projectile velocity)
- 1 targeting feature (nearest alien offset)
- 4 alien density quadrants
- 3 game state features (alien fraction, UFO position, UFO availability)

**Architecture:** RNN (1000 population, 300 generations)

**Fitness Function:**
```
F = Σ(R_kill + S_aim - P_danger - P_spam)
```
Where:
- **R_kill**: Weighted bonus for alien elimination 
- **S_aim**: Dense alignment gradient rewarding horizontal synchronization with the nearest target 
- **P_danger**: Penalty for projectile proximity that scales as threats approach the player 
- **P_spam**: Penalty for firing cooldown abuse to discourage random shooting 

**Survival Conditioning**: The fitness explicitly weights penalties to override aggression when threats are imminent, teaching the agent to prioritize survival over immediate scoring.


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

Experiments on Space Invaders were conducted iteratively to identify the optimal combination of state representation and reward shaping. Initial attempts with Raw RAM and Column-based wrappers lead to premature convergence. To overcome the dimensionality bottleneck, we transitioned to the Egocentric wrapper, scaling the evolutionary budget to more extended runs (population: 500, generations: 300) to match the complexity of the task.

| Configuration | Avg Score | Best Score | Notes |
|---------------|-----------|------------|-------|
| Baseline (Raw RAM) | 221 (±156) | 760 | |
| Column FFNN | 233 (±154) | 760 | |
| Column RNN | 249 (±162) | 830 | |
| Column RNN Fit | 167 (±106) | 615 | |
| Egocentric RNN | 304 (±149) | 1050 | |
| Egocentric RNN (ext. run) | 366 (±113) | 1880 | |
| **Egocentric RNN Fit (ext. run)** | **421 (±120)** | **2380** | |
| *Human Benchmark* | *527.40 (±188.93)* | *970.00* | |
---

## OpenEvolve: LLM-Driven Code Evolution

While NEAT optimizes weights within a growing topology, **OpenEvolve** shifts the evolutionary search from abstract neural matrices to **executable, interpretable Python code**.
This framework leverages the semantic knowledge of Large Language Models (LLMs) to perform "intelligent mutations", effectively treating the LLM as a biological mutation operator.

### The Framework

* **Mutation Operator**: A local instance of **Qwen2.5-Coder-7B** (via Ollama) generates candidate solutions.
* **Population Management**: Solutions are managed through a **MAP-Elites** database, indexing programs by score and algorithmic complexity to preserve diversity and prevent premature convergence.
* **Setup**: To ensure a fair comparison, OpenEvolve agents interact with the exact same **OCAtari wrappers** and **custmized fitness** used in the NEAT experiments.

### Evolutionary Pipeline

The pipeline required specific adaptations to function effectively in the Atari context:

1. **The "Meta-Genome"**: We designed a specialized System Prompt that acts as the genetic blueprint, setting operational constraints (e.g., "no external libraries") and optimization priorities without enforcing a specific strategy.
2. **Full Rewrite Strategy**: The LLM generates valid Python code via a "Full Rewrite" approach, as the model proved unstable when attempting "Diff-Based" edits.
3. **Cumulative Evolution**: We implemented an iterative strategy where the highest-performing code from generation $N$ serves as a **few-shot example** for generation $N+1$.
   
    > *Hypothesis:* Providing the LLM with its previous best attempt allows for iterative refinement, enabling the system to "debug" logic and optimize strategies incrementally over generations.

### Comparative Analysis: NEAT vs. OpenEvolve

To evaluate the efficacy of code evolution against standard neuroevolution, we benchmarked the best OpenEvolve agents against the optimal NEAT architectures identified in the previous experiments.

| Environment | Metric | NEAT (Best Config) | OpenEvolve | 
|:-----------:|:------:|:------------------:|:----------:|
| **Skiing** | *Best Score* <br> *Avg Score* | -4886 <br> -5391.63 (±335) | **-3856** 🏆 <br> **-4004.96 (±86)** 🏆 | 
| **Freeway** | *Best Score* <br> *Avg Score* | **29.0** 🏆 <br> **25.06 (±1.57)** 🏆 | 18.0 <br> 15.28 (±1.43) | 
| **Space Inv.**| *Best Score* <br> *Avg Score* | **2380** 🏆 <br> 421 (±120) | 1495 <br> **517 (±212)** 🏆 | 

**Possible Interpretation:**
* **Skiing (Code Wins):** The LLM logic significantly outperformed neural approximations, finding a more efficient path and solving the sparse reward problem better.
* **Freeway (Network Wins):** NEAT exploited frame-perfect reactive timing. OpenEvolve agents adopted a "conservative" strategy, waiting for safe gaps rather than risking collisions, resulting in lower throughput.
* **Space Invaders (Trade-off):** While NEAT achieved a higher peak score by aggressively exploring "lucky" seeds (e.g., catching all the UFOs), OpenEvolve demonstrated superior **stability and generalization** (higher average score) across unseen environments.

## Methodology Highlights ????

### NEAT Configuration

**Common Parameters:**
- Population size: 100 
- Generations: 150 
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
