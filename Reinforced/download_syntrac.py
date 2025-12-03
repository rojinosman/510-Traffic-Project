import json
import os
import pandas as pd

"""
Build syntrac_merged.csv from the local SynTraC JSON, using lane-level counts.

We use:
  SynTraC/Core/Intersection_camera.json

For each time step and each arm (A, B, C, D), we extract:
  - timestamp
  - signal_phase (1 = green, 0 = not green)
  - queue_length  ≈ Lane1.total_vehicles + Lane2.total_vehicles
  - waiting_time  ≈ Lane1.stopped_vehicles + Lane2.stopped_vehicles
  - throughput    ≈ number of moving vehicles (total - stopped)
"""

JSON_PATH = os.path.join("SynTraC", "Core", "Intersection_camera.json")
CSV_OUT = "syntrac_merged.csv"

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(
        f"Could not find {JSON_PATH}. "
        "Make sure you're running this from the Reinforced folder."
    )

print(f"Loading data from {JSON_PATH} ...")
with open(JSON_PATH, "r") as f:
    data = json.load(f)

rows = []
arms = ["A", "B", "C", "D"]

for key in sorted(data.keys(), key=lambda k: int(k)):
    entry = data[key]
    for arm in arms:
        if arm not in entry:
            continue
        arm_data = entry[arm]

        # Basic fields
        tl_state = arm_data["traffic_light"]          # 'Red', 'Green', 'Yellow', etc.
        ts = float(arm_data["timestamp"])

        lane1 = arm_data.get("Lane1", {})
        lane2 = arm_data.get("Lane2", {})

        # Total and stopped vehicles per lane
        l1_total = float(lane1.get("total_vehicles", 0.0))
        l2_total = float(lane2.get("total_vehicles", 0.0))
        l1_stopped = float(lane1.get("stopped_vehicles", 0.0))
        l2_stopped = float(lane2.get("stopped_vehicles", 0.0))

        # Approximate traffic metrics
        queue_length = l1_total + l2_total
        waiting_time = l1_stopped + l2_stopped           # proxy for "cars waiting"
        moving_l1 = max(l1_total - l1_stopped, 0.0)
        moving_l2 = max(l2_total - l2_stopped, 0.0)
        throughput = moving_l1 + moving_l2              # proxy for cars that can go

        # Map traffic light state to binary phase: 1 = green, 0 = not green
        tl_lower = tl_state.lower()
        signal_phase = 1 if tl_lower == "green" else 0

        rows.append(
            {
                "timestamp": ts,
                "signal_phase": signal_phase,
                "queue_length": queue_length,
                "waiting_time": waiting_time,
                "throughput": throughput,
            }
        )

# Build DataFrame and sort by time
df = pd.DataFrame(rows)
df = df.sort_values("timestamp").reset_index(drop=True)

df.to_csv(CSV_OUT, index=False)
print(f"✅ Saved {len(df)} rows to {CSV_OUT}")
