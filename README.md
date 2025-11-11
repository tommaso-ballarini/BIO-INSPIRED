## Project Structure

This project follows a modular, decoupled architecture to facilitate experimentation with different neuroevolution algorithms and environments.

* **`/core`**: Contains the abstract, reusable logic.
    * `policy.py`: Defines the agent's "brain" (e.g., a linear model, a neural network).
    * `evaluator.py`: Manages the environment simulation (the "arena") to run a single fitness evaluation.
    * `problem.py`: Acts as a bridge, connecting the agent's policy and the evaluator to the interface required by an evolutionary algorithm library (like `inspyred`).

* **`/algorithms`**: Holds the "engines" or main loops for different evolutionary strategies (e.g., `inspyred_runner.py` for a standard GA, `neat_runner.py` for NEAT).

* **`/experiments`**: Contains the executable scripts. Each file represents a single, configurable experiment, importing and "wiring together" components from `/core` and `/algorithms`.

* **`/configs`**: Stores parameter files required by specific algorithms (e.g., NEAT configuration).

* **`/utils`**: Provides non-essential helper scripts, such as plotting and visualization tools.

* **`/evolution_results`**: Serves as the default output directory for saved models, logs, and graphs.

---

## Current Structure

BIO-INSPIRED/

├── algorithms/

│   ├── inspyred_runner.py

│   └── neat_runner.py 	

├── configs/

│   └── neat_bankheist_config.txt

├── core/

│   ├── evaluator.py

│   └── policy.py

│   └── problem.py

├── experiments/

│   ├── run_bankheist_ga.py

│   └── run_bankheist_neat.py 

├── utils/

│   ├── lab_plotting_utils.py

│   └── neat_plotting_utils.py 

│   └── visualize_agent.py 

├── evolution_results/

└── requirements.txt




