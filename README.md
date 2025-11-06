# 🧬 Progetto Bank Heist - Neuroevoluzione

Questo progetto utilizza un Algoritmo Genetico (GA) per evolvere una semplice rete neurale (policy) in grado di giocare al gioco Atari "Bank Heist" utilizzando direttamente la RAM del gioco (128 byte) come input.

---

## 🛠️ Struttura del Progetto

Ecco una panoramica dei file principali e del loro ruolo nel sistema:

### 🧠 `agent_policy.py`: Il Cervello

* **Ruolo:** Definisce il "cervello" dell'agente, una rete neurale a singolo strato.
* **Logica:** Prende i **128 byte** della RAM (`game_state`) come input.
* **Azione:** Utilizza un "cromosoma" (un vettore di **2322 pesi**) per calcolare un punteggio per ognuna delle **18 azioni** possibili e sceglie la migliore.

### 🏛️ `evaluator.py`: L'Arena

* **Ruolo:** È il simulatore di gioco che esegue la *fitness function*.
* **Logica:** Avvia un'istanza di "Bank Heist" (`gymnasium`).
* **Azione:** Testa un singolo cervello (cromosoma) facendolo giocare una partita. Il `fitness_score` restituito è semplicemente il punteggio (`total_reward`) ottenuto in quella partita.

### 📋 `bank_heist_problem.py`: L'Adattatore

* **Ruolo:** Collega la logica specifica del nostro gioco al framework generico `inspyred`.
* **Logica:** Definisce tre componenti chiave per `inspyred`:
    1.  **`generator`**: Come creare un nuovo cervello (un array di 2322 float casuali).
    2.  **`evaluator`**: Come testare una popolazione (chiama `run_game_simulation` per ogni cervello).
    3.  **`bounder`**: Come mantenere i pesi (geni) entro i limiti (da -1.0 a 1.0).

### 🏃‍♂️ `run_ga_bankheist.py`: Il Pannello di Controllo

* **Ruolo:** È lo script principale che **avvia l'esperimento**.
* **Logica:** Configura i parametri dell'evoluzione (dimensione della popolazione, generazioni, tassi di mutazione/crossover).
* **Azione:** Chiama il motore GA, gestisce l'esecuzione e, al termine, salva i risultati (il file `.json` con i pesi migliori e il grafico `.png` della fitness) nella cartella `ga_results/`.

### ⚙️ `lab_ga_runner.py`: Il Motore Evolutivo

* **Ruolo:** Contiene la logica *generica* per eseguire l'algoritmo genetico.
* **Logica:** È un template riutilizzabile che orchestra il processo di evoluzione: selezione (`tournament_selection`), accoppiamento (`uniform_crossover`) e mutazione (`gaussian_mutation`).
* **Azione:** Esegue il ciclo `evolve` generazione dopo generazione.

### 📊 `lab_plotting_utils.py`: L'Osservatore

* **Ruolo:** Contiene il codice per visualizzare l'andamento dell'evoluzione.
* **Logica:** La funzione `plot_observer` viene chiamata ad ogni generazione.
* **Azione:** Aggiorna e disegna il grafico che mostra la fitness migliore, media e peggiore nel tempo.
