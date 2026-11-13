# dynamic-traffic-lights-
dynamic traffic lights is basically cameras on traffic lights that dynamically changes times and cycles depending on how congested a traffic light is lowering the time if its very crowded at that particular lane and giving priority to emergency vehicles like ambulances  I do know concepts like these already exist like SCATS, SCOOT, Noog etc    


Now here are the main Capabilities that I would like to add 

**Real-Time Queue Weighting:** Uses YOLOv8 to measure vehicle density per approach lane and allocates green-light intervals dynamically.
* **Emergency Vehicle Preemption:** Identifies approaching emergency vehicles (ambulances, fire engines) to automatically trigger signal overrides and clear emergency corridors.
* **Edge-Optimized Architecture:** Runs inference locally on low-latency edge hardware without relying on expensive centralized infrastructure (SCATS/SCOOT).
* **Simulation Benchmarking:** Integrated with a simulation environment to measure performance gains against static fixed-timer baselines.
* I would much perfer not to over use ai in this project but I will try

here are the programs I would be using 
## Tech Stack
* **Language:** Python 3.10+
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Deep Learning Framework:** PyTorch (CUDA Accelerated)
* **Simulation Engine:** SUMO / Pygame
* note this may change in the future if I have to

* and this is my ## Development Roadmap
- [ ] CUDA environment & YOLOv8 baseline setup
- [ ] Multi-lane detection zone & queue weighting algorithm
- [ ] Anti-starvation traffic signal state machine
- [ ] Simulation integration & benchmark data logging
- [ ] Local fine-tuning for specialized vehicle detection


this how you run it and achieve the results 

# AI-Driven Dynamic Traffic Signal Controller (SUMO)

An adaptive traffic signal control and emergency vehicle preemption system built in Python using the **SUMO (Simulation of Urban MObility)** microsimulation framework and TraCI API. 

Unlike traditional fixed-time controllers that waste green-light capacity on empty lanes, this algorithm dynamically evaluates real-time geometric lane queues, applies hysteresis and gap-out logic (`PRESSURE_THRESHOLD = 6`, `MIN_GREEN_STEPS = 15`), and executes immediate corridor preemption for approaching emergency vehicles.

---

## Multi-Scenario Benchmark Results

Evaluated across 7 distinct traffic demand profiles comparing a standard Fixed-Time Signal against the AI Dynamic Controller:

| Scenario | Vehicles | Fixed-Time Delay | AI Dynamic Delay | Net Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Scenario 1 — Balanced Flow** | 800 | 126.61 s | **28.62 s** | **+77.4%** |
| **Scenario 2 — North/South Heavy** | 800 | 192.31 s | **43.20 s** | **+77.5%** |
| **Scenario 3 — East/West Heavy** | 800 | 132.09 s | **34.19 s** | **+74.1%** |
| **Scenario 4 — Saturated Rush Hour** | 662 | 52,009.92 s | **50,594.88 s** | **+2.7%** (-1,415s total delay) |
| **Scenario 5 — Low Traffic / Off-Peak** | 250 | 102.60 s | **24.14 s** | **+76.5%** |
| **Scenario 6 — Uneven Traffic** | 800 | 199.23 s | **36.95 s** | **+81.5%** |
| **Scenario 7 — Emergency Vehicle Preemption** | 800 | 124.43 s | **28.61 s** | **+77.0%** |

---

## Key Features
* **Geometric Lane Classification:** Dynamically maps incoming lane geometries (`get_lane_directions()`) to prevent phase misclassification across multi-lane intersections.
* **Adaptive Gap-Out & Hysteresis:** Holds green phases while active vehicle platoons clear and terminates phases early once queue pressure drops below threshold.
* **Emergency Vehicle (EV) Priority Preemption:** Detects approaching emergency vehicles and overrides standard phase cycles to clear the corridor with minimal disruption to cross-traffic.

---

## Repository Structure
* `run_all_benchmarks.py` — Main TraCI control loop and automated multi-scenario benchmark runner.
* `intersection.net.xml` & `*.sumocfg` — SUMO network topology and simulation configurations.
* `routes_s*.rou.xml` — Vehicle demand profiles for Scenarios 1 through 7.
* `multi_scenario_benchmark_results.csv` — Raw exported performance metrics from the benchmark suite.

---

## How to Reproduce
1. Install [SUMO](https://eclipse.dev/sumo/) and ensure `SUMO_HOME` is added to your system environment variables.
2. Install the Python TraCI bindings:
   ```bash
   pip install traci sumolib
