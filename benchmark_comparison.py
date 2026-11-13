import traci
import sumolib
import csv

def run_simulation(mode="fixed"):
    sumo_binary = sumolib.checkBinary("sumo")  # Runs headless
    traci.start([
        sumo_binary, 
        "-c", "intersection.sumocfg", 
        "--no-warnings", "true",
        "--time-to-teleport", "-1"
    ])
    
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        print("[ERROR] No traffic light found!")
        traci.close()
        return 0, 0, 0, 0
    
    tls_id = tls_ids[0]
    total_waiting_time = 0
    total_vehicles_processed = 0
    step = 0
    phase_timer = 0
    current_phase = 0
    
    print(f"\n[BENCHMARK] Running Simulation in '{mode.upper()}' Mode...")
    
    # Loop until all vehicles complete their routes
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Accumulate vehicle departures each step
        total_vehicles_processed += traci.simulation.getDepartedNumber()
        
        # Gather metrics across controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        current_step_wait = sum([traci.lane.getWaitingTime(lane) for lane in set(controlled_lanes)])
        total_waiting_time += current_step_wait
        
        if mode == "fixed":
            # Fixed-Time Signal Logic
            phase_timer += 1
            if phase_timer >= 30:
                current_phase = (current_phase + 2) % 4
                traci.trafficlight.setPhase(tls_id, current_phase)
                phase_timer = 0
                
        elif mode == "dynamic":
            # AI Dynamic Signal Logic
            edges = list(set([traci.lane.getEdgeID(lane) for lane in controlled_lanes]))
            counts = {edge: traci.edge.getLastStepVehicleNumber(edge) for edge in edges}
            
            if counts:
                busiest_edge = max(counts, key=counts.get)
                if counts[busiest_edge] > 3 and step % 15 == 0:
                    current_phase = 0 if current_phase == 2 else 2
                    traci.trafficlight.setPhase(tls_id, current_phase)

        step += 1
        
    traci.close()
    
    avg_wait_per_step = total_waiting_time / step if step > 0 else 0
    avg_wait_per_vehicle = total_waiting_time / total_vehicles_processed if total_vehicles_processed > 0 else 0
    
    print(f"[RESULT] {mode.upper()} Mode Finished | Processed Steps: {step} | Total Vehicles: {total_vehicles_processed}")
    print(f"         Avg Wait/Step: {avg_wait_per_step:.2f} sec | Avg Wait/Vehicle: {avg_wait_per_vehicle:.2f} sec")
    
    return avg_wait_per_step, avg_wait_per_vehicle, total_vehicles_processed, total_waiting_time

# Execute Both Modes
fixed_step_wait, fixed_veh_wait, fixed_count, fixed_total_time = run_simulation("fixed")
dynamic_step_wait, dynamic_veh_wait, dynamic_count, dynamic_total_time = run_simulation("dynamic")

# Calculate Improvement Percentage
step_improvement = ((fixed_step_wait - dynamic_step_wait) / fixed_step_wait) * 100 if fixed_step_wait > 0 else 0
veh_improvement = ((fixed_veh_wait - dynamic_veh_wait) / fixed_veh_wait) * 100 if fixed_veh_wait > 0 else 0

print("\n" + "="*70)
print("                         PERFORMANCE SUMMARY                         ")
print("="*70)
print(f"  Fixed-Time Mode:   {fixed_count} vehicles | Total Delay: {fixed_total_time:.1f}s")
print(f"                     Avg Wait/Step: {fixed_step_wait:.2f}s | Avg Wait/Vehicle: {fixed_veh_wait:.2f}s")
print("-" * 70)
print(f"  AI Dynamic Mode:   {dynamic_count} vehicles | Total Delay: {dynamic_total_time:.1f}s")
print(f"                     Avg Wait/Step: {dynamic_step_wait:.2f}s | Avg Wait/Vehicle: {dynamic_veh_wait:.2f}s")
print("-" * 70)
print(f"  Performance Gain:  {veh_improvement:.1f}% Reduction in Average Delay per Vehicle!")
print("="*70)

# Save results to CSV
with open("traffic_performance_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Control_Type", "Total_Vehicles", "Total_Wait_Time", "Avg_Wait_Per_Step", "Avg_Wait_Per_Vehicle"])
    writer.writerow(["Fixed Signal", fixed_count, fixed_total_time, fixed_step_wait, fixed_veh_wait])
    writer.writerow(["AI Dynamic Signal", dynamic_count, dynamic_total_time, dynamic_step_wait, dynamic_veh_wait])