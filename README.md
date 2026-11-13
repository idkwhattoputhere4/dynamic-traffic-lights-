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
