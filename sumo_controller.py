import time
import traci
import sumolib

sumo_binary = sumolib.checkBinary("sumo-gui")
traci.start([sumo_binary, "-c", "intersection.sumocfg", "--start", "true"])

tls_ids = traci.trafficlight.getIDList()
if not tls_ids:
    print("[ERROR] No traffic light found!")
    exit()

tls_id = tls_ids[0]
print(f"[SUCCESS] Controlling Traffic Light: {tls_id}")

controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
controlled_edges = list(dict.fromkeys([traci.lane.getEdgeID(lane) for lane in controlled_lanes]))
print(f"[INFO] Active Controlled Edges: {controlled_edges}")

step = 0
min_green_time = 10
last_switch_step = 0

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            print("[INFO] Simulation ended.")
            break

        vehicle_counts = {edge: traci.edge.getLastStepVehicleNumber(edge) for edge in controlled_edges}

        if vehicle_counts:
            busiest_edge = max(vehicle_counts, key=vehicle_counts.get)
            max_vehicles = vehicle_counts[busiest_edge]

            if step - last_switch_step >= min_green_time:
                current_phase = traci.trafficlight.getPhase(tls_id)
                num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)

                if max_vehicles > 4:
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    last_switch_step = step
                    print(f"[SIGNAL SWITCH] Step {step}: Managing heavy traffic on '{busiest_edge}' ({max_vehicles} vehicles)")

        if step % 20 == 0:
            print(f"Step {step:03d} | Vehicle Counts: {vehicle_counts}")

        step += 1
        time.sleep(0.05)

finally:
    try:
        traci.close()
    except Exception:
        pass