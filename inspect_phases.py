import sumolib
import traci

sumo_binary = sumolib.checkBinary("sumo")
traci.start([sumo_binary, "-c", "intersection.sumocfg", "--no-warnings", "true"])
tls_id = traci.trafficlight.getIDList()[0]
logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]

print(f"Traffic Light ID: {tls_id}")
for idx, phase in enumerate(logic.phases):
    print(f"Index {idx} | State: {phase.state} | Duration: {phase.duration}s")

traci.close()