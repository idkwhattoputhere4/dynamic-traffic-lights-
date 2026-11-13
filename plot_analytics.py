import csv
from datetime import datetime
import matplotlib.pyplot as plt

csv_file = "traffic_analytics.csv"

timestamps = []
lane1_queues = []
lane2_queues = []
serviced_vehicles = []
fps_list = []

try:
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse CSV rows safely using built-in types
            timestamps.append(datetime.strptime(row['Timestamp'], "%Y-%m-%d %H:%M:%S"))
            lane1_queues.append(int(row['Lane_1_Queue']))
            lane2_queues.append(int(row['Lane_2_Queue']))
            serviced_vehicles.append(int(row['Total_Serviced']))
            fps_list.append(float(row['FPS']))

    print(f"Successfully loaded {len(timestamps)} data points.")
except FileNotFoundError:
    print(f"Error: Could not find '{csv_file}'. Make sure your main script generated it!")
    exit()

# Setup Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("AI Dynamic Traffic Controller — Performance Telemetry", fontsize=14, fontweight='bold')

# --- Plot 1: Queue Lengths ---
ax1.plot(timestamps, lane1_queues, label='Lane 1 Queue', color='orange', linewidth=2)
ax1.plot(timestamps, lane2_queues, label='Lane 2 Queue', color='magenta', linewidth=2)
ax1.set_ylabel("Vehicle Count")
ax1.set_title("Queue Length Dynamics Across Lanes")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')

# --- Plot 2: Serviced Vehicles & FPS ---
ax2.plot(timestamps, serviced_vehicles, label='Total Serviced Vehicles', color='cyan', linewidth=2)
ax2_fps = ax2.twinx()
ax2_fps.plot(timestamps, fps_list, label='Inference FPS', color='green', linestyle=':', alpha=0.7)

ax2.set_xlabel("Time")
ax2.set_ylabel("Serviced Vehicles Count", color='cyan')
ax2_fps.set_ylabel("System FPS", color='green')
ax2.set_title("Throughput & Processing Speed")
ax2.grid(True, linestyle='--', alpha=0.6)

# Format & Export
plt.tight_layout()
output_image = "traffic_performance_summary.png"
plt.savefig(output_image, dpi=300)
print(f"Chart saved successfully as '{output_image}'. Displaying graph window...")

plt.show()