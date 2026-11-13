import traci
import sumolib
import csv
import os

SCENARIOS = [
    ("Scenario 1 — Balanced", "routes_s1_balanced.rou.xml", False),
    ("Scenario 2 — North/South Heavy", "routes_s2_ns_heavy.rou.xml", False),
    ("Scenario 3 — East/West Heavy", "routes_s3_ew_heavy.rou.xml", False),
    ("Scenario 4 — Rush Hour", "routes_s4_rush_hour.rou.xml", False),
    ("Scenario 5 — Low Traffic", "routes_s5_low_traffic.rou.xml", False),
    ("Scenario 6 — Uneven Traffic", "routes_s6_uneven.rou.xml", False),
    ("Scenario 7 — Emergency Vehicle", "routes_s7_emergency.rou.xml", True)
]

def get_lane_directions(controlled_lanes):
    """Categorize SUMO lanes into North/South or East/West by geometry shape."""
    ns_lanes = []
    ew_lanes = []
    
    for lane_id in controlled_lanes:
        shape = traci.lane.getShape(lane_id)
        if len(shape) >= 2:
            dx = abs(shape[-1][0] - shape[0][0])
            dy = abs(shape[-1][1] - shape[0][1])
            if dy > dx:
                ns_lanes.append(lane_id)
            else:
                ew_lanes.append(lane_id)
        else:
            ns_lanes.append(lane_id)
            
    return ns_lanes, ew_lanes

def run_simulation(scenario_name, route_file, is_ev_test, mode="fixed"):
    sumo_binary = sumolib.checkBinary("sumo")
    
    cmd = [
        sumo_binary, 
        "-c", "intersection.sumocfg", 
        "-r", route_file,
        "--no-warnings", "true",
        "--time-to-teleport", "300"
    ]

    traci.start(cmd)
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        traci.close()
        return None
    
    tls_id = tls_ids[0]
    total_waiting_time = 0
    total_vehicles_processed = 0
    step = 0
    
    phase_timer = 0
    MAX_STEPS = 3600
    
    # Pre-calculate lane direction mapping once SUMO initializes
    controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    ns_lanes, ew_lanes = get_lane_directions(controlled_lanes)

    while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
        traci.simulationStep()
        total_vehicles_processed += traci.simulation.getDepartedNumber()
        
        current_step_wait = sum([traci.lane.getWaitingTime(lane) for lane in controlled_lanes])
        total_waiting_time += current_step_wait
        
        curr_phase = traci.trafficlight.getPhase(tls_id)
        
        # Scenario 7: EV Detection
        ev_detected = False
        ev_lane = None
        if is_ev_test:
            for lane in controlled_lanes:
                veh_list = traci.lane.getLastStepVehicleIDs(lane)
                for v in veh_list:
                    if v == "ev_ambulance" or traci.vehicle.getTypeID(v).lower() == "emergency":
                        ev_detected = True
                        ev_lane = lane
                        break
                if ev_detected:
                    break

        phase_timer += 1

        if mode == "fixed":
            pass

        elif mode == "dynamic":
            # Only evaluate during Green Phases (0 = NS Green, 2 = EW Green)
            if curr_phase in [0, 2]:
                ns_waiting = sum([traci.lane.getLastStepHaltingNumber(l) for l in ns_lanes])
                ew_waiting = sum([traci.lane.getLastStepHaltingNumber(l) for l in ew_lanes])

                current_waiting = ns_waiting if curr_phase == 0 else ew_waiting
                opposing_waiting = ew_waiting if curr_phase == 0 else ns_waiting

                should_switch = False

                # 1. Emergency Vehicle Preemption
                if ev_detected and ev_lane:
                    ev_in_ns = ev_lane in ns_lanes
                    # Only switch if emergency vehicle is waiting on the RED approach
                    if (curr_phase == 0 and not ev_in_ns) or (curr_phase == 2 and ev_in_ns):
                        should_switch = True

                # 2. Adaptive Minimum Green (15 seconds)
                elif phase_timer >= 15:
                    # Hard cap at 60 seconds
                    if phase_timer >= 60:
                        should_switch = True
                    # Active lane fully cleared, opposing lane has waiting vehicles
                    elif current_waiting == 0 and opposing_waiting > 0:
                        should_switch = True
                    # Significant queue advantage (+6 vehicles) on opposing lane
                    elif opposing_waiting >= current_waiting + 6:
                        should_switch = True

                if should_switch:
                    yellow_phase = 1 if curr_phase == 0 else 3
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    phase_timer = 0

            elif curr_phase in [1, 3]:
                phase_timer = 0

        step += 1
        
    traci.close()
    
    avg_wait_per_vehicle = total_waiting_time / total_vehicles_processed if total_vehicles_processed > 0 else 0
    return {
        "steps": step,
        "total_vehicles": total_vehicles_processed,
        "total_delay": total_waiting_time,
        "avg_wait_vehicle": avg_wait_per_vehicle
    }

results = []

print("="*80)
print("             STARTING MULTI-SCENARIO TRAFFIC BENCHMARK SUITE             ")
print("="*80)

for scenario_name, route_file, is_ev in SCENARIOS:
    if not os.path.exists(route_file):
        print(f"[SKIP] {route_file} missing. Run 'python generate_scenarios.py' first.")
        continue
        
    print(f"\n[RUNNING] {scenario_name}...")
    fixed_res = run_simulation(scenario_name, route_file, is_ev, mode="fixed")
    dynamic_res = run_simulation(scenario_name, route_file, is_ev, mode="dynamic")
    
    if fixed_res and dynamic_res:
        veh_imp = ((fixed_res["avg_wait_vehicle"] - dynamic_res["avg_wait_vehicle"]) / fixed_res["avg_wait_vehicle"]) * 100 if fixed_res["avg_wait_vehicle"] > 0 else 0
        
        results.append({
            "Scenario": scenario_name,
            "Vehicles": fixed_res["total_vehicles"],
            "Fixed_Avg_Wait_Veh": round(fixed_res["avg_wait_vehicle"], 2),
            "Dynamic_Avg_Wait_Veh": round(dynamic_res["avg_wait_vehicle"], 2),
            "Delay_Reduction_%": round(veh_imp, 1),
            "Fixed_Steps": fixed_res["steps"],
            "Dynamic_Steps": dynamic_res["steps"]
        })
        
        print(f"  Fixed Signal:    {fixed_res['avg_wait_vehicle']:.2f}s avg delay/veh | Steps: {fixed_res['steps']}")
        print(f"  AI Dynamic:      {dynamic_res['avg_wait_vehicle']:.2f}s avg delay/veh | Steps: {dynamic_res['steps']}")
        print(f"  Result:          {veh_imp:.1f}% Reduction in Average Delay per Vehicle")

# Summary Table
print("\n" + "="*85)
print("                               BENCHMARK SUMMARY RESULTS                               ")
print("="*85)
print(f"{'Scenario':<32} | {'Vehicles':<8} | {'Fixed Delay':<11} | {'AI Delay':<10} | {'Improvement':<11}")
print("-" * 85)
for r in results:
    print(f"{r['Scenario']:<32} | {r['Vehicles']:<8} | {r['Fixed_Avg_Wait_Veh']:<11.2f}s | {r['Dynamic_Avg_Wait_Veh']:<10.2f}s | {r['Delay_Reduction_%']:<10.1f}%")
print("="*85)

with open("multi_scenario_benchmark_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Scenario", "Vehicles", "Fixed_Avg_Wait_Veh", "Dynamic_Avg_Wait_Veh", "Delay_Reduction_%", "Fixed_Steps", "Dynamic_Steps"])
    writer.writeheader()
    writer.writerows(results)

print("\n[SUCCESS] Benchmark suite complete! Summary saved to 'multi_scenario_benchmark_results.csv'.")