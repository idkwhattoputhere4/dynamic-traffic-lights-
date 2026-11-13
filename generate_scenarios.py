import xml.etree.ElementTree as ET
import sumolib
import random
import os

net_file = "intersection.net.xml"

if not os.path.exists(net_file):
    print(f"[ERROR] '{net_file}' not found.")
    exit(1)

net = sumolib.net.readNet(net_file)
all_edges = [e for e in net.getEdges() if not e.isSpecial()]

# Group edges by directional orientation
ns_edges = []
ew_edges = []

for e in all_edges:
    # Check shape/angle to classify direction
    shape = e.getShape()
    if len(shape) >= 2:
        dx = abs(shape[-1][0] - shape[0][0])
        dy = abs(shape[-1][1] - shape[0][1])
        if dy > dx:
            ns_edges.append(e)
        else:
            ew_edges.append(e)
    else:
        ns_edges.append(e)

if not ns_edges: ns_edges = all_edges
if not ew_edges: ew_edges = all_edges

def get_valid_pairs(from_list, to_list):
    pairs = []
    for f in from_list:
        for t in to_list:
            if f != t:
                path, _ = net.getShortestPath(f, t)
                if path is not None:
                    pairs.append((f.getID(), t.getID()))
    return pairs if pairs else [(f.getID(), t.getID()) for f in all_edges for t in all_edges if f != t]

ns_pairs = get_valid_pairs(ns_edges, ns_edges)
ew_pairs = get_valid_pairs(ew_edges, ew_edges)
all_pairs = get_valid_pairs(all_edges, all_edges)

def generate_scenario(filename, scenario_type, target_count, seed_val):
    scenario_root = ET.Element("routes")
    ET.SubElement(scenario_root, "vType", id="car", accel="2.6", decel="4.5", length="5.0", maxSpeed="13.89")
    ET.SubElement(scenario_root, "vType", id="emergency", accel="3.5", decel="5.0", length="6.5", maxSpeed="20.00", vClass="emergency", guiShape="emergency")

    random.seed(seed_val)
    current_time = 0.0

    for i in range(target_count):
        # Select pool based on scenario definition
        if scenario_type == "ns_heavy":
            pool = ns_pairs if random.random() < 0.8 else ew_pairs
        elif scenario_type == "ew_heavy":
            pool = ew_pairs if random.random() < 0.8 else ns_pairs
        elif scenario_type == "uneven":
            pool = ns_pairs if random.random() < 0.9 else ew_pairs
        else:
            pool = all_pairs

        from_e, to_e = random.choice(pool)

        if scenario_type == "emergency" and i == 150:
            trip_elem = ET.SubElement(scenario_root, "trip", id="ev_ambulance", type="emergency", depart=f"{current_time:.2f}", from_edge=from_e, to=to_e)
            trip_elem.attrib['from'] = trip_elem.attrib.pop('from_edge')
        else:
            trip_elem = ET.SubElement(scenario_root, "trip", id=f"veh_{i}", type="car", depart=f"{current_time:.2f}", from_edge=from_e, to=to_e)
            trip_elem.attrib['from'] = trip_elem.attrib.pop('from_edge')

        if scenario_type == "rush_hour":
            current_time += random.uniform(0.3, 0.7)
        elif scenario_type == "low_traffic":
            current_time += random.uniform(3.0, 6.0)
        else:
            current_time += random.uniform(1.0, 2.5)

    tree_out = ET.ElementTree(scenario_root)
    ET.indent(tree_out, space="    ")
    tree_out.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"[GENERATED] {filename} ({target_count} trips)")

SCENARIOS = [
    ("routes_s1_balanced.rou.xml", "balanced", 800, 101),
    ("routes_s2_ns_heavy.rou.xml", "ns_heavy", 800, 202),
    ("routes_s3_ew_heavy.rou.xml", "ew_heavy", 800, 303),
    ("routes_s4_rush_hour.rou.xml", "rush_hour", 1200, 404),
    ("routes_s5_low_traffic.rou.xml", "low_traffic", 250, 505),
    ("routes_s6_uneven.rou.xml", "uneven", 800, 606),
    ("routes_s7_emergency.rou.xml", "emergency", 800, 707)
]

if __name__ == "__main__":
    print("Generating Direction-Differentiated Scenario XML Files...\n")
    for file_name, s_type, v_count, seed_val in SCENARIOS:
        generate_scenario(file_name, s_type, v_count, seed_val)
    print("\nDistinct scenario route files generated successfully.")