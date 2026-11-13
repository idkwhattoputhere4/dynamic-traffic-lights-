import sumolib

net_file = "intersection.net.xml"
output_route = "intersection.rou.xml"

net = sumolib.net.readNet(net_file)
tls_nodes = [node for node in net.getNodes() if node.getType() == "traffic_light"]

if not tls_nodes:
    tls_nodes = [node for node in net.getNodes() if len(node.getIncoming()) > 2]

tls_node = tls_nodes[0]
incoming_edges = [edge.getID() for edge in tls_node.getIncoming() if not edge.getID().startswith(":")]

valid_routes = []
for inc_id in incoming_edges:
    inc_edge = net.getEdge(inc_id)
    for out_edge in inc_edge.getOutgoing():
        out_id = out_edge.getID()
        if out_id != inc_id and not out_id.startswith(":"):
            valid_routes.append((inc_id, out_id))
            break

routes_xml = ['<routes>', '    <vType id="car" accel="2.6" decel="4.5" length="5.0" minGap="2.5" maxSpeed="13.89"/>']

for i, (src, dst) in enumerate(valid_routes):
    routes_xml.append(f'    <route id="r_{i}" edges="{src} {dst}"/>')
    routes_xml.append(f'    <flow id="f_{i}" type="car" route="r_{i}" begin="0" end="2000" period="2.0"/>')

routes_xml.append('</routes>')

with open(output_route, "w") as f:
    f.write("\n".join(routes_xml))

print("[SUCCESS] Restored original working route configuration!")